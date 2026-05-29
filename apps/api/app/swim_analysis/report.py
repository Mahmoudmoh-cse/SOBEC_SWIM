from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.swim_analysis.pose.base import KEYPOINT_NAMES, Keypoint, PoseFrameResult
from app.swim_analysis.quality import confidence_value, data_quality_score


SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def build_analysis_report(
    *,
    analysis_id: str,
    stroke_type: str,
    duration_sec: float,
    video_quality: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    metadata: dict[str, Any],
    pose_backend: str,
    processing_errors: list[str] | None = None,
    calibration: dict[str, Any] | None = None,
    velocity: dict[str, Any] | None = None,
    phases: list[dict[str, Any]] | None = None,
    faults: list[dict[str, Any]] | None = None,
    coaching_report: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    reliability_score: float | None = None,
    pose_diagnostics: dict[str, Any] | None = None,
    trajectories: dict[str, Any] | None = None,
    research_figures: list[dict[str, Any]] | None = None,
    debug_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overall = _metric_value(metrics, "overall_technique_score", 0)
    confidence = _metric_value(metrics, "confidence_score", 0)
    quality = data_quality_score(video_quality)
    single_video = metadata.get("input_mode") == "single_video"
    findings = build_findings(metrics, faults or [], single_video=single_video)
    return {
        "analysis_id": analysis_id,
        "status": "completed",
        "summary": {
            "stroke_type": stroke_type,
            "duration_sec": round(duration_sec, 2),
            "overall_score": overall,
            "confidence_score": confidence,
            "reliability_score": reliability_score if reliability_score is not None else confidence,
            "data_quality_score": quality,
            "pose_backend": pose_backend,
        },
        "video_quality": video_quality,
        "video_metadata": metadata,
        "metrics": metrics,
        "calibration": calibration or {},
        "velocity": velocity or {},
        "phase_segments": phases or [],
        "faults": faults or [],
        "findings": findings,
        "recommendations": build_recommendations(metrics, findings, single_video=single_video),
        "coaching_report": coaching_report or {},
        "artifacts": artifacts,
        "warnings": warnings or [],
        "pose_diagnostics": pose_diagnostics or {},
        "trajectories": trajectories or {},
        "research_figures": research_figures or [],
        "debug_info": debug_info or {},
        "processing_errors": processing_errors or [],
    }


def build_findings(metrics: dict[str, Any], faults: list[dict[str, Any]] | None = None, *, single_video: bool = False) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for fault in faults or []:
        findings.append(
            {
                "type": str(fault.get("name", "technique_fault")),
                "severity": str(fault.get("severity", "medium")),
                "message": str(fault.get("coach_explanation", "Technique pattern detected for coach review.")),
                "confidence": round(float(fault.get("confidence", 0) or 0), 3),
                "evidence": {
                    "metrics": fault.get("evidence_metrics", {}),
                    "time_range_sec": fault.get("timestamp_range"),
                    "recommended_drill": fault.get("recommended_drill"),
                },
            }
        )
    checks = [
        ("body_alignment_score", 76, "body_alignment", "Hip position appears to drop or body line breaks during parts of the swim."),
        ("arm_symmetry_score", 78, "arm_symmetry", "Left and right arm motion do not look evenly balanced from the uploaded video." if single_video else "Left and right arm motion do not look evenly balanced from the front/back view."),
        ("head_stability_score", 75, "head_stability", "Head movement appears inconsistent across the uploaded video timeline." if single_video else "Head movement appears inconsistent across the side-view timeline."),
        ("kick_rhythm_score", 70, "kick_rhythm", "Kick timing appears uneven when ankle landmarks are visible."),
        ("centerline_deviation_score", 75, "centerline", "The swimmer drifts away from the body centerline in the uploaded video." if single_video else "The swimmer drifts away from the body centerline in the front/back view."),
    ]
    for metric_name, threshold, finding_type, message in checks:
        metric = metrics.get(metric_name, {})
        value = metric.get("value")
        confidence = confidence_value(metric.get("confidence", 0))
        if isinstance(value, (int, float)) and value < threshold and confidence >= 0.35:
            findings.append(
                {
                    "type": finding_type,
                    "severity": "high" if value < threshold - 15 else "medium",
                    "message": message,
                    "confidence": round(confidence, 3),
                    "evidence": metric.get("evidence", {}),
                }
            )
    hip_drop = metrics.get("hip_drop_indicator", {})
    if hip_drop.get("value") in {"medium", "high"} and confidence_value(hip_drop.get("confidence", 0)) >= 0.35:
        findings.append(
            {
                "type": "hip_drop",
                "severity": "high" if hip_drop.get("value") == "high" else "medium",
                "message": "Hip position appears lower than the shoulder line in visible uploaded video frames." if single_video else "Hip position appears lower than the shoulder line in visible side-view frames.",
                "confidence": round(confidence_value(hip_drop.get("confidence", 0)), 3),
                "evidence": hip_drop.get("evidence", {}),
            }
        )
    return findings[:8]


def build_recommendations(metrics: dict[str, Any], findings: list[dict[str, Any]], *, single_video: bool = False) -> list[dict[str, Any]]:
    templates = {
        "body_alignment": ("Focus on maintaining a longer body line before and during breathing cycles.", "body_alignment_score"),
        "hip_drop": ("Use side-kick and 6-kick switch work to keep the hips from dropping during rotation.", "hip_drop_indicator"),
        "arm_symmetry": ("Use single-arm freestyle and catch-up timing checks to even out left/right arm rhythm.", "arm_symmetry_score"),
        "head_stability": ("Keep the head quieter by rotating to breathe instead of lifting through the neck.", "head_stability_score"),
        "kick_rhythm": ("Add short fin kick sets and tempo changes to make the kick rhythm more repeatable.", "kick_rhythm_score"),
        "centerline": ("Use lane-line sighting and front-view feedback to keep hand entry from crossing the centerline.", "centerline_deviation_score"),
    }
    recommendations = []
    for priority, finding in enumerate(findings, start=1):
        message, metric = templates.get(str(finding.get("type")), ("Review this segment with a coach before changing technique.", "overall_technique_score"))
        recommendations.append({"priority": priority, "message": message, "linked_metric": metric})
    if not recommendations and confidence_value(metrics.get("confidence_score", {}).get("value", 0)) >= 0.45:
        recommendations.append(
            {
                "priority": 1,
                "message": "Keep collecting the same fixed-camera video so trends can be compared across sessions." if single_video else "Keep collecting the same two fixed camera views so trends can be compared across sessions.",
                "linked_metric": "confidence_score",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "priority": 1,
                "message": "Retake the video with the full swimmer visible before making technique conclusions." if single_video else "Retake both videos with the full swimmer visible before making technique conclusions.",
                "linked_metric": "data_quality_score",
            }
        )
    return recommendations[:5]


def write_json_artifact(path: str | Path, payload: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(target)


def generate_annotated_video(
    *,
    source_path: str | Path,
    target_path: str | Path,
    frames: list[PoseFrameResult],
    metrics: dict[str, Any],
    view_type: str,
    max_dimension: int = 1600,
    velocity: dict[str, Any] | None = None,
    phases: list[dict[str, Any]] | None = None,
    faults: list[dict[str, Any]] | None = None,
    debug_overlays: dict[str, Any] | None = None,
) -> str | None:
    try:
        import cv2  # type: ignore
        from app.swim_analysis.preprocessing import standardize_frame
    except ImportError:
        return None

    frame_map = {frame.frame_index: frame for frame in frames}
    source = Path(source_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        return None
    first_frame = standardize_frame(first_frame, max_dimension=max_dimension, enhance=False)
    height, width = first_frame.shape[:2]
    writer, output_path = _open_browser_video_writer(cv2, target, fps, (width, height))
    if writer is None:
        cap.release()
        return None

    known_indices = sorted(frame_map)
    current_pose: PoseFrameResult | None = None
    pose_history: list[PoseFrameResult] = []
    frame_index = 0
    try:
        pending_frame = first_frame
        while True:
            image = pending_frame
            pending_frame = None
            if frame_index in frame_map:
                current_pose = frame_map[frame_index]
            elif current_pose and known_indices:
                nearest = min(known_indices, key=lambda index: abs(index - frame_index))
                if abs(nearest - frame_index) < fps * 0.35:
                    current_pose = frame_map[nearest]
            if current_pose:
                timestamp = frame_index / fps
                if current_pose.frame_index == frame_index or not pose_history or pose_history[-1].frame_index != current_pose.frame_index:
                    pose_history.append(current_pose)
                pose_history = [pose for pose in pose_history if timestamp - float(pose.timestamp) <= 1.8]
                if (debug_overlays or {}).get("show_trails", True):
                    _draw_roi_history(cv2, image, pose_history)
                    _draw_joint_trails(cv2, image, pose_history)
                _draw_pose(cv2, image, current_pose, debug_overlays or {})
            timestamp = frame_index / fps
            _draw_path(cv2, image, velocity, timestamp)
            _draw_phase_and_faults(cv2, image, timestamp, phases or [], faults or [])
            _draw_metric_overlay(cv2, image, timestamp, metrics, view_type, velocity=velocity)
            writer.write(image)
            frame_index += 1
            ok, next_frame = cap.read()
            if not ok:
                break
            pending_frame = standardize_frame(next_frame, max_dimension=max_dimension, enhance=False)
    finally:
        cap.release()
        writer.release()

    return str(output_path) if output_path and output_path.exists() else None


def _open_browser_video_writer(cv2: Any, target: Path, fps: float, frame_size: tuple[int, int]) -> tuple[Any | None, Path | None]:
    suffix = target.suffix.lower()
    if suffix == ".webm":
        candidates = [(target, "VP80"), (target, "VP90")]
    elif suffix == ".mp4":
        candidates = [(target, "avc1"), (target, "H264"), (target.with_suffix(".webm"), "VP80"), (target.with_suffix(".webm"), "VP90")]
    else:
        candidates = [(target.with_suffix(".webm"), "VP80"), (target.with_suffix(".webm"), "VP90")]

    for path, fourcc in candidates:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, frame_size)
        if writer.isOpened():
            return writer, path
        writer.release()
        if path.exists() and path.stat().st_size == 0:
            path.unlink(missing_ok=True)
    return None, None


def _draw_pose(cv2: Any, image: Any, pose: PoseFrameResult, options: dict[str, Any] | None = None) -> None:
    options = options or {}
    if options.get("show_roi", True):
        _draw_roi(cv2, image, pose)
    if options.get("show_raw_pose") and pose.raw_keypoints:
        _draw_raw_pose(cv2, image, pose.raw_keypoints)

    by_name = {point.name: point for point in pose.keypoints if point.confidence >= 0.2 and point.visibility_state != "rejected_outlier"}
    for start, end in SKELETON_CONNECTIONS:
        first = by_name.get(start)
        second = by_name.get(end)
        if first and second:
            cv2.line(image, (int(first.x), int(first.y)), (int(second.x), int(second.y)), (0, 210, 255), 2, cv2.LINE_AA)
    for point in by_name.values():
        color = (80, 220, 255) if point.visibility_state == "interpolated" else (0, 230, 120) if point.confidence >= 0.55 else (0, 165, 255)
        if any(flag in point.quality_flags for flag in {"biomechanical_limb_length_recovery", "confidence_aware_temporal_hold"}):
            color = (0, 95, 255)
        cv2.circle(image, (int(point.x), int(point.y)), 4, color, -1, cv2.LINE_AA)
    if options.get("show_debug", True):
        _draw_low_confidence_points(cv2, image, pose)
        _draw_rejected_points(cv2, image, pose)
        _draw_tracking_quality(cv2, image, pose)


def _draw_metric_overlay(cv2: Any, image: Any, timestamp: float, metrics: dict[str, Any], view_type: str, *, velocity: dict[str, Any] | None = None) -> None:
    confidence = metrics.get("confidence_score", {}).get("value", 0)
    overall = _display_value(metrics.get("overall_technique_score", {}).get("value", "--"))
    current_velocity = _nearest_velocity(velocity or {}, timestamp)
    if view_type in {"side", "video"}:
        primary = f"Body {_display_value(metrics.get('body_alignment_score', {}).get('value'))} | Head {_display_value(metrics.get('head_stability_score', {}).get('value'))}"
    else:
        primary = f"Symmetry {_display_value(metrics.get('arm_symmetry_score', {}).get('value'))} | Centerline {_display_value(metrics.get('centerline_deviation_score', {}).get('value'))}"
    velocity_text = "--" if current_velocity is None else f"{current_velocity['velocity']} {current_velocity['velocity_unit']}"
    warning = "Low confidence: verify video quality" if confidence == "low" or (isinstance(confidence, (int, float)) and confidence < 0.45) else "Confidence-aware analysis"
    lines = [f"AquaIQ {view_type} {timestamp:05.2f}s", f"Overall: {overall}  Confidence: {confidence}  Velocity: {velocity_text}", primary, warning]
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (650, 126), (8, 20, 32), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, image)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (24, 35 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 250, 255), 2, cv2.LINE_AA)


def _draw_roi(cv2: Any, image: Any, pose: PoseFrameResult) -> None:
    roi = pose.roi or {}
    if not roi.get("valid"):
        return
    x = int(roi.get("x", 0) or 0)
    y = int(roi.get("y", 0) or 0)
    width = int(roi.get("width", 0) or 0)
    height = int(roi.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        return
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 170, 0), 2, cv2.LINE_AA)


def _draw_raw_pose(cv2: Any, image: Any, points: list[Keypoint]) -> None:
    by_name = {point.name: point for point in points if point.confidence >= 0.2}
    for start, end in SKELETON_CONNECTIONS:
        first = by_name.get(start)
        second = by_name.get(end)
        if first and second:
            cv2.line(image, (int(first.x), int(first.y)), (int(second.x), int(second.y)), (180, 180, 180), 1, cv2.LINE_AA)
    for point in by_name.values():
        cv2.circle(image, (int(point.x), int(point.y)), 3, (190, 190, 190), 1, cv2.LINE_AA)


def _draw_low_confidence_points(cv2: Any, image: Any, pose: PoseFrameResult) -> None:
    low_points = [point for point in pose.keypoints if 0 < point.confidence < 0.2 or point.visibility_state == "low_confidence"]
    for point in low_points:
        cv2.circle(image, (int(point.x), int(point.y)), 5, (0, 130, 255), 1, cv2.LINE_AA)
    missing_count = len([name for name in KEYPOINT_NAMES if name not in {point.name for point in pose.keypoints if point.confidence >= 0.12}])
    if missing_count:
        cv2.putText(image, f"Missing joints: {missing_count}", (24, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 2, cv2.LINE_AA)


def _draw_rejected_points(cv2: Any, image: Any, pose: PoseFrameResult) -> None:
    for point in pose.rejected_keypoints:
        x = int(point.x)
        y = int(point.y)
        cv2.line(image, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(image, (x - 6, y + 6), (x + 6, y - 6), (0, 0, 255), 2, cv2.LINE_AA)


def _draw_tracking_quality(cv2: Any, image: Any, pose: PoseFrameResult) -> None:
    if pose.tracking_quality is None:
        return
    quality = max(0.0, min(1.0, float(pose.tracking_quality)))
    x = 24
    y = 174
    width = 160
    cv2.rectangle(image, (x, y), (x + width, y + 10), (20, 30, 42), -1)
    color = (0, 220, 120) if quality >= 0.7 else (0, 190, 255) if quality >= 0.42 else (0, 95, 255)
    cv2.rectangle(image, (x, y), (x + int(width * quality), y + 10), color, -1)
    cv2.putText(image, f"Tracking {int(quality * 100)}%", (x + width + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 250, 255), 1, cv2.LINE_AA)
    tracking = pose.debug_info.get("roi_tracking") if isinstance(pose.debug_info, dict) else None
    if isinstance(tracking, dict):
        state = str(tracking.get("tracking_state") or "REVIEW_ONLY")
        state_color = _tracking_state_color(state)
        cv2.putText(image, state, (x, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, state_color, 2, cv2.LINE_AA)
        if tracking.get("drift_detected"):
            cv2.rectangle(image, (4, 4), (image.shape[1] - 5, image.shape[0] - 5), (0, 0, 255), 4, cv2.LINE_AA)
            reasons = tracking.get("drift_reasons") or []
            if reasons:
                cv2.putText(image, str(reasons[0]).replace("_", " ")[:42], (x, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 190, 255), 1, cv2.LINE_AA)


def _draw_joint_trails(cv2: Any, image: Any, history: list[PoseFrameResult]) -> None:
    if len(history) < 2:
        return
    joints = {
        "left_wrist": (255, 185, 40),
        "right_wrist": (35, 150, 255),
        "left_ankle": (70, 220, 120),
        "right_ankle": (210, 120, 255),
        "nose": (245, 245, 245),
    }
    for joint, color in joints.items():
        samples = []
        for pose in history[-45:]:
            point = pose.keypoint(joint, 0.18)
            if point and point.visibility_state != "rejected_outlier":
                samples.append((int(point.x), int(point.y)))
        for index, (first, second) in enumerate(zip(samples, samples[1:])):
            alpha_color = tuple(int(channel * (0.35 + 0.65 * (index + 1) / max(1, len(samples)))) for channel in color)
            cv2.line(image, first, second, alpha_color, 1, cv2.LINE_AA)


def _draw_roi_history(cv2: Any, image: Any, history: list[PoseFrameResult]) -> None:
    for pose in history[-18:]:
        roi = pose.roi or {}
        if not roi.get("valid"):
            continue
        x = int(roi.get("x", 0) or 0)
        y = int(roi.get("y", 0) or 0)
        width = int(roi.get("width", 0) or 0)
        height = int(roi.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            continue
        cv2.rectangle(image, (x, y), (x + width, y + height), (120, 120, 120), 1, cv2.LINE_AA)


def _draw_path(cv2: Any, image: Any, velocity: dict[str, Any] | None, timestamp: float) -> None:
    if not velocity:
        return
    points = [
        point
        for point in velocity.get("series", [])
        if float(point.get("timestamp", 0)) <= timestamp and point.get("centroid_px")
    ][-60:]
    if len(points) < 2:
        return
    for first, second in zip(points, points[1:]):
        a = first["centroid_px"]
        b = second["centroid_px"]
        cv2.line(image, (int(a["x"]), int(a["y"])), (int(b["x"]), int(b["y"])), (255, 170, 0), 2, cv2.LINE_AA)


def _draw_phase_and_faults(cv2: Any, image: Any, timestamp: float, phases: list[dict[str, Any]], faults: list[dict[str, Any]]) -> None:
    phase = next((item for item in phases if item.get("type") != "stroke_cycle" and float(item.get("start_sec", 0)) <= timestamp <= float(item.get("end_sec", 0))), None)
    active_faults = [
        fault
        for fault in faults
        if _range_contains(fault.get("timestamp_range"), timestamp)
    ][:2]
    y = image.shape[0] - 70
    if phase:
        color = _phase_color(str(phase.get("type") or ""))
        overlay = image.copy()
        cv2.rectangle(overlay, (0, image.shape[0] - 92), (image.shape[1], image.shape[0] - 52), color, -1)
        cv2.addWeighted(overlay, 0.22, image, 0.78, 0, image)
        cv2.putText(image, f"Phase: {str(phase.get('type')).replace('_', ' ')}", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
    for index, fault in enumerate(active_faults):
        cv2.putText(image, f"Fault: {str(fault.get('name')).replace('_', ' ')}", (24, y + 26 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 255), 2, cv2.LINE_AA)


def _nearest_velocity(velocity: dict[str, Any], timestamp: float) -> dict[str, Any] | None:
    series = velocity.get("series", [])
    if not series:
        return None
    return min(series, key=lambda point: abs(float(point.get("timestamp", 0)) - timestamp))


def _range_contains(value: Any, timestamp: float) -> bool:
    if not isinstance(value, list) or len(value) < 2 or value[0] is None or value[1] is None:
        return False
    return float(value[0]) <= timestamp <= float(value[1])


def _metric_value(metrics: dict[str, Any], name: str, default: Any) -> Any:
    return metrics.get(name, {}).get("value", default)


def _display_value(value: Any) -> Any:
    return "--" if value is None else value


def _phase_color(phase_type: str) -> tuple[int, int, int]:
    colors = {
        "start_push_off": (0, 210, 255),
        "underwater": (255, 170, 0),
        "breakout": (70, 220, 120),
        "free_swim": (255, 255, 255),
        "catch": (80, 220, 255),
        "pull": (0, 180, 255),
        "push": (0, 120, 255),
        "recovery": (170, 120, 255),
        "body_roll_timing": (255, 170, 120),
        "kick_rhythm": (120, 230, 120),
        "turn": (80, 120, 255),
        "finish": (245, 245, 245),
    }
    return colors.get(phase_type, (255, 255, 255))


def _tracking_state_color(state: str) -> tuple[int, int, int]:
    colors = {
        "LOCKED": (0, 220, 120),
        "SUSPECT": (0, 190, 255),
        "LOST": (0, 0, 255),
        "REACQUIRE": (255, 170, 0),
        "REVIEW_ONLY": (180, 180, 180),
    }
    return colors.get(state, (245, 245, 245))
