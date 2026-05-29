from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean, pstdev
from typing import Any

from app.swim_analysis.phase1_schemas import RejectedOutlier, SwapCorrectionEvent, TrajectoryExport, TrajectoryPoint
from app.swim_analysis.pose.base import KEYPOINT_NAMES, Keypoint, PoseFrameResult, average_keypoint_confidence, bbox_from_keypoints
from app.swim_analysis.smoothing import smooth_pose_sequence


LEFT_RIGHT_PAIRS = [
    ("left_wrist", "right_wrist"),
    ("left_elbow", "right_elbow"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]

DISTAL_JOINTS = {"left_wrist", "right_wrist", "left_ankle", "right_ankle"}
MID_JOINTS = {"left_elbow", "right_elbow", "left_knee", "right_knee"}


@dataclass(frozen=True)
class JointPhysicsThreshold:
    max_displacement_px: float
    max_velocity_px_s: float
    max_acceleration_px_s2: float
    confidence_drop: float = 0.62


@dataclass(frozen=True)
class TemporalTrackingConfig:
    visible_confidence: float = 0.25
    low_confidence: float = 0.12
    max_gap_frames: int = 5
    enable_smoothing: bool = True
    smoothing_min_confidence: float = 0.12
    smoothing_min_cutoff: float = 1.0
    smoothing_beta: float = 0.035
    smoothing_d_cutoff: float = 1.0
    smoothing_ema_alpha_low: float = 0.32
    smoothing_ema_alpha_high: float = 0.88
    enable_biomechanical_constraints: bool = True
    max_body_scale_change_ratio: float = 2.2
    swap_margin_px: float = 18.0
    min_swap_improvement_px: float = 22.0
    core_threshold: JointPhysicsThreshold = JointPhysicsThreshold(130.0, 2600.0, 65000.0)
    mid_threshold: JointPhysicsThreshold = JointPhysicsThreshold(170.0, 3400.0, 90000.0)
    distal_threshold: JointPhysicsThreshold = JointPhysicsThreshold(230.0, 4700.0, 125000.0)


@dataclass
class TrackingResult:
    raw_frames: list[PoseFrameResult]
    corrected_frames: list[PoseFrameResult]
    filtered_frames: list[PoseFrameResult]
    interpolated_frames: list[PoseFrameResult]
    clean_frames: list[PoseFrameResult]
    trajectory_export: dict[str, Any]
    rejected_outliers: list[dict[str, Any]]
    swap_events: list[dict[str, Any]]
    visibility_percentage: dict[str, float]
    tracking_quality_timeline: list[dict[str, Any]]
    frame_quality_timeline: list[dict[str, Any]]
    joint_reliability: dict[str, Any]
    smoothing_diagnostics: dict[str, Any]
    warnings: list[str]


class TemporalJointTracker:
    """Confidence-aware single-swimmer trajectory tracker for offline swimming analysis.

    The tracker deliberately prefers missing data over fabricated joints. It marks low-confidence
    observations, corrects plausible left/right swaps from temporal continuity, rejects physically
    implausible jumps, interpolates only short gaps, and finally applies confidence-aware smoothing.
    """

    def __init__(self, config: TemporalTrackingConfig | None = None) -> None:
        self.config = config or TemporalTrackingConfig()

    def process(self, frames: list[PoseFrameResult], *, view_type: str) -> TrackingResult:
        structured = [_structure_frame(frame, self.config) for frame in frames]
        corrected, swap_events = self._correct_left_right_swaps(structured)
        filtered, outliers = self._reject_physics_outliers(corrected)
        biomechanical = self._apply_biomechanical_constraints(filtered) if self.config.enable_biomechanical_constraints else filtered
        interpolated = self._interpolate_short_joint_gaps(biomechanical)
        quality_frames = self._attach_tracking_quality(interpolated)
        if self.config.enable_smoothing:
            clean_result = smooth_pose_sequence(
                quality_frames,
                min_confidence=self.config.smoothing_min_confidence,
                return_diagnostics=True,
                min_cutoff=self.config.smoothing_min_cutoff,
                beta=self.config.smoothing_beta,
                d_cutoff=self.config.smoothing_d_cutoff,
                ema_alpha_low=self.config.smoothing_ema_alpha_low,
                ema_alpha_high=self.config.smoothing_ema_alpha_high,
            )
            clean, smoothing_diagnostics = clean_result
        else:
            clean = quality_frames
            smoothing_diagnostics = {
                "enabled": False,
                "method": "disabled",
                "processed_frames": len(quality_frames),
                "low_confidence_holds": 0,
                "mean_raw_to_smoothed_px": 0.0,
                "per_joint_mean_displacement_px": {},
                "frame_displacement_timeline": [],
            }
        clean = self._attach_tracking_quality(clean)
        export = build_trajectory_export(
            view_type=view_type,
            raw_frames=structured,
            corrected_frames=corrected,
            filtered_frames=filtered,
            interpolated_frames=quality_frames,
            clean_frames=clean,
            outliers=outliers,
            swap_events=swap_events,
            config=self.config,
            smoothing_diagnostics=smoothing_diagnostics,
        )
        warnings = _tracking_warnings(export)
        return TrackingResult(
            raw_frames=structured,
            corrected_frames=corrected,
            filtered_frames=filtered,
            interpolated_frames=quality_frames,
            clean_frames=clean,
            trajectory_export=export,
            rejected_outliers=export["rejected_outliers"],
            swap_events=export["swap_events"],
            visibility_percentage=export["visibility_percentage"],
            tracking_quality_timeline=export["tracking_quality_timeline"],
            frame_quality_timeline=export["frame_quality_timeline"],
            joint_reliability=export["joint_reliability"],
            smoothing_diagnostics=export["smoothing_diagnostics"],
            warnings=warnings,
        )

    def _correct_left_right_swaps(self, frames: list[PoseFrameResult]) -> tuple[list[PoseFrameResult], list[SwapCorrectionEvent]]:
        corrected: list[PoseFrameResult] = []
        previous: dict[str, Keypoint] = {}
        events: list[SwapCorrectionEvent] = []

        for frame in frames:
            points = {point.name: point for point in frame.keypoints if point.confidence >= self.config.low_confidence}
            swapped_names: set[str] = set()
            debug_events = list(frame.debug_events)
            for left_name, right_name in LEFT_RIGHT_PAIRS:
                left = points.get(left_name)
                right = points.get(right_name)
                prev_left = previous.get(left_name)
                prev_right = previous.get(right_name)
                if not left or not right or not prev_left or not prev_right:
                    continue
                direct = _distance(left, prev_left) + _distance(right, prev_right)
                swapped = _distance(left, prev_right) + _distance(right, prev_left)
                if swapped + self.config.swap_margin_px < direct and direct - swapped >= self.config.min_swap_improvement_px:
                    new_left = replace(right, name=left_name, quality_flags=[*right.quality_flags, "left_right_swap_corrected"])
                    new_right = replace(left, name=right_name, quality_flags=[*left.quality_flags, "left_right_swap_corrected"])
                    points[left_name] = new_left
                    points[right_name] = new_right
                    swapped_names.update({left_name, right_name})
                    event = SwapCorrectionEvent(
                        frame_index=frame.frame_index,
                        timestamp=round(frame.timestamp, 4),
                        pair=f"{left_name}/{right_name}",
                        direct_cost=round(direct, 3),
                        swapped_cost=round(swapped, 3),
                        reason="cross-assignment preserved temporal continuity",
                    )
                    events.append(event)
                    debug_events.append(f"left/right swap corrected for {left_name}/{right_name}")

            next_points = [points.get(point.name, point) for point in frame.keypoints]
            corrected_frame = replace(frame, keypoints=next_points, debug_events=debug_events)
            corrected.append(corrected_frame)
            for point in next_points:
                if point.confidence >= self.config.visible_confidence and point.name in KEYPOINT_NAMES:
                    previous[point.name] = point
            for name in swapped_names:
                if name in points:
                    previous[name] = points[name]
        return corrected, events

    def _reject_physics_outliers(self, frames: list[PoseFrameResult]) -> tuple[list[PoseFrameResult], list[RejectedOutlier]]:
        filtered: list[PoseFrameResult] = []
        last_good: dict[str, Keypoint] = {}
        last_velocity: dict[str, float] = {}
        last_time: dict[str, float] = {}
        rejected: list[RejectedOutlier] = []

        for frame in frames:
            next_points: list[Keypoint] = []
            rejected_points: list[Keypoint] = []
            for point in frame.keypoints:
                if point.confidence < self.config.low_confidence:
                    next_points.append(replace(point, visibility_state="low_confidence"))
                    continue
                threshold = self._threshold_for(point.name)
                previous = last_good.get(point.name)
                reason = None
                displacement = None
                velocity = None
                acceleration = None
                if previous is not None:
                    dt = max(1e-3, frame.timestamp - last_time.get(point.name, frame.timestamp))
                    displacement = _distance(point, previous)
                    velocity = displacement / dt
                    previous_velocity = last_velocity.get(point.name)
                    acceleration = abs(velocity - previous_velocity) / dt if previous_velocity is not None else 0.0
                    confidence_drop = max(0.0, previous.confidence - point.confidence)
                    if displacement > threshold.max_displacement_px:
                        reason = "max_pixel_displacement"
                    elif velocity > threshold.max_velocity_px_s:
                        reason = "max_velocity"
                    elif acceleration > threshold.max_acceleration_px_s2:
                        reason = "max_acceleration"
                    elif confidence_drop > threshold.confidence_drop and displacement > threshold.max_displacement_px * 0.45:
                        reason = "confidence_drop_with_jump"

                if reason:
                    rejected_point = replace(point, visibility_state="rejected_outlier", rejected_reason=reason, quality_flags=[*point.quality_flags, reason])
                    rejected_points.append(rejected_point)
                    rejected.append(
                        RejectedOutlier(
                            frame_index=frame.frame_index,
                            timestamp=round(frame.timestamp, 4),
                            joint=point.name,
                            x=round(float(point.x), 3),
                            y=round(float(point.y), 3),
                            confidence=round(float(point.confidence), 4),
                            reason=reason,
                            displacement_px=round(displacement, 3) if displacement is not None else None,
                            velocity_px_s=round(velocity, 3) if velocity is not None else None,
                            acceleration_px_s2=round(acceleration, 3) if acceleration is not None else None,
                        )
                    )
                    continue

                visible_state = "visible" if point.confidence >= self.config.visible_confidence else "low_confidence"
                accepted = replace(point, visibility_state=visible_state)
                next_points.append(accepted)
                if point.confidence >= self.config.visible_confidence:
                    if previous is not None and velocity is not None:
                        last_velocity[point.name] = velocity
                    last_good[point.name] = accepted
                    last_time[point.name] = frame.timestamp

            filtered.append(
                replace(
                    frame,
                    keypoints=next_points,
                    rejected_keypoints=[*frame.rejected_keypoints, *rejected_points],
                    confidence=average_keypoint_confidence(next_points),
                    bbox=bbox_from_keypoints(next_points) or frame.bbox,
                    quality_flags=[*frame.quality_flags, "outlier_rejected"] if rejected_points else frame.quality_flags,
                )
            )
        return filtered, rejected

    def _interpolate_short_joint_gaps(self, frames: list[PoseFrameResult]) -> list[PoseFrameResult]:
        if not frames:
            return []
        output = [replace(frame, keypoints=list(frame.keypoints)) for frame in frames]
        for joint in KEYPOINT_NAMES:
            index = 0
            while index < len(output):
                if _point_for(output[index], joint, self.config.visible_confidence) is not None:
                    index += 1
                    continue
                gap_start = index
                while index < len(output) and _point_for(output[index], joint, self.config.visible_confidence) is None:
                    index += 1
                gap_end = index - 1
                before = _point_for(output[gap_start - 1], joint, self.config.visible_confidence) if gap_start > 0 else None
                after = _point_for(output[index], joint, self.config.visible_confidence) if index < len(output) else None
                gap_len = gap_end - gap_start + 1
                if before is None or after is None or gap_len > self.config.max_gap_frames:
                    continue
                if any(_is_tracking_reset_frame(output[pos]) for pos in range(gap_start, gap_end + 1)):
                    continue
                for offset, frame_pos in enumerate(range(gap_start, gap_end + 1), start=1):
                    ratio = offset / (gap_len + 1)
                    interpolated = Keypoint(
                        name=joint,
                        x=before.x + (after.x - before.x) * ratio,
                        y=before.y + (after.y - before.y) * ratio,
                        z=before.z + (after.z - before.z) * ratio if before.z is not None and after.z is not None else None,
                        confidence=max(0.05, min(before.confidence, after.confidence, 0.55) * (1.0 - 0.04 * offset)),
                        visibility_state="interpolated",
                        source_backend="temporal_interpolation",
                        frame_index=output[frame_pos].frame_index,
                        quality_flags=["interpolated_short_joint_gap"],
                        interpolated=True,
                    )
                    output[frame_pos] = _replace_or_append_point(output[frame_pos], interpolated)
        return [
            replace(
                frame,
                confidence=average_keypoint_confidence(frame.keypoints),
                bbox=bbox_from_keypoints(frame.keypoints) or frame.bbox,
                interpolated=frame.interpolated or any(point.interpolated for point in frame.keypoints),
            )
            for frame in output
        ]

    def _attach_tracking_quality(self, frames: list[PoseFrameResult]) -> list[PoseFrameResult]:
        return [replace(frame, tracking_quality=_frame_tracking_quality(frame, self.config)) for frame in frames]

    def _threshold_for(self, name: str) -> JointPhysicsThreshold:
        if name in DISTAL_JOINTS:
            return self.config.distal_threshold
        if name in MID_JOINTS:
            return self.config.mid_threshold
        return self.config.core_threshold

    def _apply_biomechanical_constraints(self, frames: list[PoseFrameResult]) -> list[PoseFrameResult]:
        constrained: list[PoseFrameResult] = []
        last_scale: float | None = None
        for frame in frames:
            reference_scale = _body_reference_scale(frame, self.config) or _bbox_diagonal(frame.bbox)
            flags = list(frame.quality_flags)
            debug_events = list(frame.debug_events)
            if reference_scale and last_scale:
                ratio = max(reference_scale / max(last_scale, 1e-3), last_scale / max(reference_scale, 1e-3))
                if ratio > self.config.max_body_scale_change_ratio and frame.confidence < 0.55:
                    flags.append("body_scale_jump_temporal_recovery")
                    debug_events.append(f"body scale jump damped ({ratio:.2f}x)")
            if reference_scale:
                last_scale = reference_scale
            next_points = _downgrade_implausible_limb_geometry(frame, reference_scale or 80.0, self.config)
            constrained.append(
                replace(
                    frame,
                    keypoints=next_points,
                    confidence=average_keypoint_confidence(next_points),
                    bbox=bbox_from_keypoints(next_points) or frame.bbox,
                    quality_flags=flags,
                    debug_events=debug_events,
                )
            )
        return constrained


def build_trajectory_export(
    *,
    view_type: str,
    raw_frames: list[PoseFrameResult],
    corrected_frames: list[PoseFrameResult],
    filtered_frames: list[PoseFrameResult],
    interpolated_frames: list[PoseFrameResult],
    clean_frames: list[PoseFrameResult],
    outliers: list[RejectedOutlier],
    swap_events: list[SwapCorrectionEvent],
    config: TemporalTrackingConfig | None = None,
    smoothing_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or TemporalTrackingConfig()
    smoothing_diagnostics = smoothing_diagnostics or {"enabled": False, "method": "unknown"}
    joint_reliability = _joint_reliability_summary(clean_frames, config=config)
    payload = {
        "view_type": view_type,
        "raw_keypoints": _trajectory_map(raw_frames, config=config),
        "corrected_keypoints": _trajectory_map(corrected_frames, config=config),
        "filtered_keypoints": _trajectory_map(clean_frames, config=config),
        "interpolated_keypoints": _trajectory_map(interpolated_frames, config=config),
        "rejected_outliers": [event.model_dump(mode="json") for event in outliers],
        "swap_events": [event.model_dump(mode="json") for event in swap_events],
        "confidence_timeline": _confidence_timeline(clean_frames, config=config),
        "visibility_percentage": _visibility_percentage(clean_frames, config=config),
        "tracking_quality_timeline": [
            {"frame_index": frame.frame_index, "timestamp": round(frame.timestamp, 4), "quality": round(float(frame.tracking_quality or 0), 4)}
            for frame in clean_frames
        ],
        "frame_quality_timeline": [
            {"frame_index": frame.frame_index, "timestamp": round(frame.timestamp, 4), "quality": round(float(frame.frame_quality_score if frame.frame_quality_score is not None else 1.0), 4)}
            for frame in clean_frames
        ],
        "joint_reliability": joint_reliability,
        "smoothing_diagnostics": smoothing_diagnostics,
        "center_trajectory": _center_trajectory(clean_frames),
        "roi_history": _roi_history(clean_frames),
        "dropped_frame_diagnostics": _dropped_frame_diagnostics(clean_frames),
        "processing_warnings": [],
    }
    validated = TrajectoryExport.model_validate(payload)
    return validated.model_dump(mode="json")


def _body_reference_scale(frame: PoseFrameResult, config: TemporalTrackingConfig) -> float | None:
    shoulders = (_point_for(frame, "left_shoulder", config.low_confidence), _point_for(frame, "right_shoulder", config.low_confidence))
    hips = (_point_for(frame, "left_hip", config.low_confidence), _point_for(frame, "right_hip", config.low_confidence))
    shoulder_mid = _midpoint(*shoulders)
    hip_mid = _midpoint(*hips)
    distances = [
        _distance(shoulders[0], shoulders[1]) if shoulders[0] and shoulders[1] else None,
        _distance(hips[0], hips[1]) if hips[0] and hips[1] else None,
        _distance(shoulder_mid, hip_mid) if shoulder_mid and hip_mid else None,
    ]
    values = [float(value) for value in distances if value is not None and value > 4]
    if not values:
        return None
    return max(24.0, max(values))


def _downgrade_implausible_limb_geometry(frame: PoseFrameResult, reference_scale: float, config: TemporalTrackingConfig) -> list[Keypoint]:
    by_name = {point.name: point for point in frame.keypoints}
    checks = [
        ("left_elbow", "left_shoulder", 3.4),
        ("right_elbow", "right_shoulder", 3.4),
        ("left_wrist", "left_elbow", 3.6),
        ("right_wrist", "right_elbow", 3.6),
        ("left_knee", "left_hip", 3.8),
        ("right_knee", "right_hip", 3.8),
        ("left_ankle", "left_knee", 3.8),
        ("right_ankle", "right_knee", 3.8),
    ]
    downgrade: dict[str, str] = {}
    for joint_name, anchor_name, max_ratio in checks:
        joint = by_name.get(joint_name)
        anchor = by_name.get(anchor_name)
        if joint is None or anchor is None:
            continue
        distance = _distance(joint, anchor)
        if distance > reference_scale * max_ratio and joint.confidence < 0.8:
            downgrade[joint_name] = "biomechanical_limb_length_recovery"
    if not downgrade:
        return list(frame.keypoints)
    output: list[Keypoint] = []
    for point in frame.keypoints:
        reason = downgrade.get(point.name)
        if not reason:
            output.append(point)
            continue
        output.append(
            replace(
                point,
                confidence=min(point.confidence, config.low_confidence * 0.92),
                visibility_state="low_confidence",
                quality_flags=[*point.quality_flags, reason],
            )
        )
    return output


def _joint_reliability_summary(frames: list[PoseFrameResult], *, config: TemporalTrackingConfig) -> dict[str, Any]:
    total = len(frames)
    if not frames:
        return {
            joint: {
                "visibility_percent": 0.0,
                "average_confidence": 0.0,
                "temporal_stability": 0.0,
                "motion_smoothness": 0.0,
                "supporting_frames": 0,
                "unstable_frame_count": 0,
                "category": "hidden",
            }
            for joint in KEYPOINT_NAMES
        }
    output: dict[str, Any] = {}
    for joint in KEYPOINT_NAMES:
        timeline = [_point_for(frame, joint, min_confidence=0.0) for frame in frames]
        present = [point for point in timeline if point is not None and point.visibility_state not in {"missing", "rejected_outlier", "skipped"}]
        supporting = [
            point
            for point in present
            if point.confidence >= config.low_confidence and point.visibility_state in {"visible", "interpolated", "low_confidence"}
        ]
        visibility = len(supporting) / total * 100.0 if total else 0.0
        avg_conf = mean([point.confidence for point in present if point.confidence > 0]) if present else 0.0
        stability, smoothness, unstable_count = _joint_motion_scores(frames, joint)
        if visibility < 25.0 or avg_conf < config.low_confidence:
            category = "hidden"
        elif stability < 0.42 or smoothness < 0.42 or avg_conf < config.visible_confidence:
            category = "unstable"
        elif visibility >= 60.0 and avg_conf >= config.visible_confidence and min(stability, smoothness) >= 0.52:
            category = "reliable"
        else:
            category = "estimated"
        output[joint] = {
            "visibility_percent": round(visibility, 2),
            "average_confidence": round(float(avg_conf), 4),
            "temporal_stability": round(stability, 4),
            "motion_smoothness": round(smoothness, 4),
            "supporting_frames": len(supporting),
            "unstable_frame_count": unstable_count,
            "category": category,
        }
    return output


def _joint_motion_scores(frames: list[PoseFrameResult], joint: str) -> tuple[float, float, int]:
    samples = [
        (float(frame.timestamp), float(point.x), float(point.y))
        for frame in frames
        if (point := _point_for(frame, joint, min_confidence=0.0)) is not None
        and point.visibility_state not in {"missing", "rejected_outlier", "skipped"}
        and point.confidence > 0
    ]
    if len(samples) < 3:
        return (0.35 if samples else 0.0, 0.35 if samples else 0.0, 0)
    velocities: list[float] = []
    for first, second in zip(samples, samples[1:]):
        dt = max(1e-3, second[0] - first[0])
        velocities.append((((second[1] - first[1]) ** 2 + (second[2] - first[2]) ** 2) ** 0.5) / dt)
    if not velocities:
        return (0.35, 0.35, 0)
    avg_velocity = mean(velocities)
    velocity_std = pstdev(velocities) if len(velocities) > 1 else 0.0
    smoothness = 1.0 - min(1.0, velocity_std / (avg_velocity + 90.0))
    accelerations = [abs(second - first) for first, second in zip(velocities, velocities[1:])]
    avg_accel = mean(accelerations) if accelerations else 0.0
    accel_std = pstdev(accelerations) if len(accelerations) > 1 else 0.0
    stability = 1.0 - min(1.0, accel_std / (avg_accel + 2200.0))
    if len(velocities) > 2:
        jump_gate = avg_velocity + max(120.0, velocity_std * 2.4)
        unstable_count = sum(1 for velocity in velocities if velocity > jump_gate)
    else:
        unstable_count = 0
    return max(0.0, smoothness), max(0.0, stability), unstable_count


def _center_trajectory(frames: list[PoseFrameResult]) -> list[dict[str, Any]]:
    points = []
    for frame in frames:
        bbox = frame.bbox or bbox_from_keypoints(frame.keypoints, min_confidence=0.0)
        if not bbox:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        points.append(
            {
                "frame_index": frame.frame_index,
                "timestamp": round(float(frame.timestamp), 4),
                "x": round((x1 + x2) / 2.0, 3),
                "y": round((y1 + y2) / 2.0, 3),
                "width": round(max(0.0, x2 - x1), 3),
                "height": round(max(0.0, y2 - y1), 3),
                "confidence": round(float(frame.confidence), 4),
                "tracking_quality": round(float(frame.tracking_quality or 0.0), 4),
            }
        )
    return points


def _roi_history(frames: list[PoseFrameResult]) -> list[dict[str, Any]]:
    history = []
    for frame in frames:
        roi = frame.roi or {}
        decision = frame.debug_info.get("roi_decision") if isinstance(frame.debug_info, dict) else None
        tracking = frame.debug_info.get("roi_tracking") if isinstance(frame.debug_info, dict) else None
        if not isinstance(decision, dict):
            decision = {}
        if not isinstance(tracking, dict):
            tracking = {}
        history.append(
            {
                "frame_index": frame.frame_index,
                "timestamp": round(float(frame.timestamp), 4),
                "valid": bool(roi.get("valid")) if roi else False,
                "source": str(roi.get("source") or decision.get("selected_source") or "full_frame"),
                "selected": str(decision.get("selected") or ("roi" if roi.get("valid") else "full_frame")),
                "x": _round_optional(roi.get("x")),
                "y": _round_optional(roi.get("y")),
                "width": _round_optional(roi.get("width")),
                "height": _round_optional(roi.get("height")),
                "confidence": _round_optional(roi.get("rtmpose_confidence") or frame.confidence),
                "candidate_score": _round_optional(roi.get("candidate_score") or decision.get("selected_score")),
                "boundary_touch_ratio": _round_optional(roi.get("boundary_touch_ratio")),
                "tracking_state": str(tracking.get("tracking_state") or ""),
                "center_jump_ratio": _round_optional(tracking.get("center_jump_ratio")),
                "roi_speed_ratio_s": _round_optional(tracking.get("roi_speed_ratio_s")),
                "roi_acceleration_ratio_s2": _round_optional(tracking.get("roi_acceleration_ratio_s2")),
                "scale_change_ratio": _round_optional(tracking.get("scale_change_ratio")),
                "body_joint_confidence": _round_optional(tracking.get("body_joint_confidence")),
                "body_to_roi_offset_ratio": _round_optional(tracking.get("body_to_roi_offset_ratio")),
                "motion_physics_anomaly": bool(tracking.get("motion_physics_anomaly")),
                "smoothed_roi_bbox": tracking.get("smoothed_roi_bbox"),
                "reacquisition_triggered": bool(tracking.get("reacquisition_triggered")),
                "smoothing_reset": bool(tracking.get("smoothing_reset")),
                "drift_reasons": list(tracking.get("drift_reasons") or []),
            }
        )
    return history


def _dropped_frame_diagnostics(frames: list[PoseFrameResult]) -> dict[str, Any]:
    skipped = [frame.frame_index for frame in frames if frame.skipped]
    no_pose = [frame.frame_index for frame in frames if not frame.keypoints and not frame.skipped]
    low_quality = [frame.frame_index for frame in frames if frame.tracking_quality is not None and frame.tracking_quality < 0.25]
    recovered = [frame.frame_index for frame in frames if frame.interpolated or any(point.interpolated for point in frame.keypoints)]
    return {
        "skipped_frames": len(skipped),
        "no_pose_frames": len(no_pose),
        "low_tracking_quality_frames": len(low_quality),
        "temporal_recovery_frames": len(recovered),
        "sample_skipped_frame_indices": skipped[:20],
        "sample_no_pose_frame_indices": no_pose[:20],
        "sample_low_quality_frame_indices": low_quality[:20],
        "sample_temporal_recovery_frame_indices": recovered[:20],
    }


def _structure_frame(frame: PoseFrameResult, config: TemporalTrackingConfig) -> PoseFrameResult:
    next_points = []
    for point in frame.keypoints:
        if point.confidence >= config.visible_confidence:
            state = "visible"
        elif point.confidence > 0:
            state = "low_confidence"
        else:
            state = "missing"
        next_points.append(
            replace(
                point,
                visibility_state=state,
                source_backend=point.source_backend or frame.backend,
                frame_index=frame.frame_index,
            )
        )
    return replace(frame, keypoints=next_points, confidence=average_keypoint_confidence(next_points), bbox=bbox_from_keypoints(next_points) or frame.bbox)


def _trajectory_map(frames: list[PoseFrameResult], *, config: TemporalTrackingConfig) -> dict[str, list[dict[str, Any]]]:
    return {
        joint: [
            _trajectory_point(frame, joint, _point_for(frame, joint, min_confidence=0.0), config=config).model_dump(mode="json")
            for frame in frames
        ]
        for joint in KEYPOINT_NAMES
    }


def _trajectory_point(frame: PoseFrameResult, joint: str, point: Keypoint | None, *, config: TemporalTrackingConfig) -> TrajectoryPoint:
    if frame.skipped:
        return TrajectoryPoint(frame_index=frame.frame_index, timestamp=round(frame.timestamp, 4), joint=joint, visibility_state="skipped", source_backend=frame.backend, flags=frame.quality_flags)
    if point is None:
        return TrajectoryPoint(frame_index=frame.frame_index, timestamp=round(frame.timestamp, 4), joint=joint, visibility_state="missing", source_backend=frame.backend, flags=frame.quality_flags)
    state = point.visibility_state
    if state == "visible" and point.confidence < config.visible_confidence:
        state = "low_confidence" if point.confidence > 0 else "missing"
    return TrajectoryPoint(
        frame_index=frame.frame_index,
        timestamp=round(frame.timestamp, 4),
        joint=joint,
        x=round(float(point.x), 3),
        y=round(float(point.y), 3),
        confidence=round(float(point.confidence), 4),
        visibility_state=state,  # type: ignore[arg-type]
        source_backend=point.source_backend or frame.backend,
        flags=[*frame.quality_flags, *point.quality_flags],
    )


def _confidence_timeline(frames: list[PoseFrameResult], *, config: TemporalTrackingConfig) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for joint in KEYPOINT_NAMES:
        values = []
        for frame in frames:
            point = _point_for(frame, joint, min_confidence=0.0)
            state = "missing" if point is None else point.visibility_state
            confidence = 0.0 if point is None else point.confidence
            values.append(
                {
                    "frame_index": frame.frame_index,
                    "timestamp": round(frame.timestamp, 4),
                    "confidence": round(float(confidence), 4),
                    "visibility_state": state,
                }
            )
        output[joint] = values
    return output


def _visibility_percentage(frames: list[PoseFrameResult], *, config: TemporalTrackingConfig) -> dict[str, float]:
    if not frames:
        return {joint: 0.0 for joint in KEYPOINT_NAMES}
    output = {}
    for joint in KEYPOINT_NAMES:
        visible = 0
        for frame in frames:
            point = _point_for(frame, joint, min_confidence=0.0)
            if point is not None and point.visibility_state in {"visible", "interpolated"} and point.confidence >= config.low_confidence:
                visible += 1
        output[joint] = round(visible / len(frames) * 100.0, 2)
    return output


def _frame_tracking_quality(frame: PoseFrameResult, config: TemporalTrackingConfig) -> float:
    if frame.skipped:
        return 0.0
    expected = len(KEYPOINT_NAMES)
    visible_points = [point for point in frame.keypoints if point.visibility_state == "visible" and point.confidence >= config.visible_confidence]
    interpolated_points = [point for point in frame.keypoints if point.visibility_state == "interpolated"]
    low_points = [point for point in frame.keypoints if point.visibility_state == "low_confidence"]
    rejected = len(frame.rejected_keypoints)
    confidences = [point.confidence for point in frame.keypoints if point.confidence > 0]
    avg_confidence = mean(confidences) if confidences else 0.0
    support = (len(visible_points) + len(interpolated_points) * 0.65) / expected
    penalty = min(0.45, rejected / expected * 0.35 + len(low_points) / expected * 0.12)
    frame_quality = frame.frame_quality_score if frame.frame_quality_score is not None else 1.0
    quality = support * 0.48 + avg_confidence * 0.32 + frame_quality * 0.2 - penalty
    return round(max(0.0, min(1.0, quality)), 4)


def _tracking_warnings(export: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    visibility = export.get("visibility_percentage", {})
    if visibility:
        average_visibility = mean(float(value) for value in visibility.values())
        if average_visibility < 35:
            warnings.append("Joint visibility is low; biomechanics metrics should be treated as low confidence.")
    if len(export.get("rejected_outliers", [])) > 8:
        warnings.append("Many physically implausible joint jumps were rejected; check camera stability and occlusion.")
    if len(export.get("swap_events", [])) > 4:
        warnings.append("Frequent left/right corrections were needed; front/back view or splash may be confusing the pose backend.")
    reliability = export.get("joint_reliability", {})
    if isinstance(reliability, dict) and reliability:
        unstable = [joint for joint, item in reliability.items() if isinstance(item, dict) and item.get("category") == "unstable"]
        hidden = [joint for joint, item in reliability.items() if isinstance(item, dict) and item.get("category") == "hidden"]
        if len(unstable) >= 4:
            warnings.append("Several joints were temporally unstable; coaching metrics should use reliability gates.")
        if len(hidden) >= 5:
            warnings.append("Several joints were hidden for most sampled frames; avoid strong biomechanics conclusions.")
    smoothing = export.get("smoothing_diagnostics", {})
    if isinstance(smoothing, dict) and int(smoothing.get("low_confidence_holds", 0) or 0) > max(6, len(export.get("tracking_quality_timeline", [])) // 2):
        warnings.append("Temporal smoothing frequently had to hold low-confidence joints from previous frames.")
    return warnings


def _replace_or_append_point(frame: PoseFrameResult, point: Keypoint) -> PoseFrameResult:
    points = [candidate for candidate in frame.keypoints if candidate.name != point.name]
    points.append(point)
    return replace(
        frame,
        keypoints=points,
        interpolated=True,
        quality_flags=[*frame.quality_flags, "interpolated_short_joint_gap"],
        debug_events=[*frame.debug_events, f"interpolated {point.name}"],
    )


def _point_for(frame: PoseFrameResult, name: str, min_confidence: float) -> Keypoint | None:
    for point in frame.keypoints:
        if point.name == name and point.confidence >= min_confidence and point.visibility_state != "rejected_outlier":
            return point
    return None


def _is_tracking_reset_frame(frame: PoseFrameResult) -> bool:
    tracking = frame.debug_info.get("roi_tracking") if isinstance(frame.debug_info, dict) else None
    if isinstance(tracking, dict) and bool(tracking.get("smoothing_reset")):
        return True
    state = str(tracking.get("tracking_state") if isinstance(tracking, dict) else "")
    return state in {"LOST", "REACQUIRE", "REVIEW_ONLY"} or "smoothing_reset" in frame.quality_flags


def _midpoint(a: Keypoint | None, b: Keypoint | None) -> Keypoint | None:
    if a is None or b is None:
        return None
    return Keypoint(
        name="midpoint",
        x=(float(a.x) + float(b.x)) / 2.0,
        y=(float(a.y) + float(b.y)) / 2.0,
        confidence=min(float(a.confidence), float(b.confidence)),
    )


def _distance(first: Keypoint, second: Keypoint) -> float:
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5


def _bbox_diagonal(bbox: list[float] | None) -> float | None:
    if not bbox or len(bbox) < 4:
        return None
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    diagonal = (width**2 + height**2) ** 0.5
    return diagonal if diagonal > 0 else None


def _round_optional(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None
