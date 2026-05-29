from __future__ import annotations

from statistics import mean, median
from typing import Any

from app.swim_analysis.calibration import centroid_for_frame
from app.swim_analysis.pose.base import PoseFrameResult
from app.swim_analysis.smoothing import moving_average


MAX_PLAUSIBLE_AVERAGE_MPS = 3.2
MAX_PLAUSIBLE_PEAK_MPS = 4.2
MIN_PLAUSIBLE_AVERAGE_MPS = 0.25
MAX_PLAUSIBLE_ACCELERATION_MPS2 = 9.0
BODY_CENTROID_NAMES = {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}
POSE_CENTROID_NAMES = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}


def compute_velocity_analysis(
    frames: list[PoseFrameResult],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    axis = str(calibration.get("motion_axis") or "x")
    pixel_to_meter = calibration.get("pixel_to_meter")
    calibration_conf = float(calibration.get("confidence", 0) or 0)
    centroids = [_motion_point_for_frame(frame) or centroid_for_frame(frame) for frame in frames]
    centroids = [point for point in centroids if point is not None]
    if len(centroids) < 3:
        return _empty_velocity(calibration, "Not enough visible swimmer centroids for velocity analysis.")

    centroid_only = all(int(point.get("visible_keypoints", 0) or 0) == 0 for point in centroids)
    duration_sec = max(0.0, float(centroids[-1]["timestamp"]) - float(centroids[0]["timestamp"]))
    origin = float(centroids[0][axis])
    direction = _dominant_direction(centroids, axis)
    raw_distances_px = [(float(point[axis]) - origin) * direction for point in centroids]
    distances_m = [value * pixel_to_meter if pixel_to_meter else None for value in raw_distances_px]

    raw_velocities = [0.0]
    raw_accelerations = [0.0]
    for index in range(1, len(centroids)):
        dt = max(1e-3, float(centroids[index]["timestamp"]) - float(centroids[index - 1]["timestamp"]))
        if pixel_to_meter:
            delta = (raw_distances_px[index] - raw_distances_px[index - 1]) * pixel_to_meter
            raw_velocities.append(max(0.0, delta / dt))
        else:
            delta = raw_distances_px[index] - raw_distances_px[index - 1]
            raw_velocities.append(max(0.0, delta / dt))
        raw_accelerations.append((raw_velocities[-1] - raw_velocities[-2]) / dt)

    velocities = moving_average(_median_filter(raw_velocities, window=5), window=5)
    accelerations = moving_average(_median_filter(raw_accelerations, window=5), window=5)
    unit = "m/s" if pixel_to_meter else "px/s"
    plausibility = _apply_velocity_plausibility_gates(
        velocities=velocities,
        accelerations=accelerations,
        raw_velocities=raw_velocities,
        raw_accelerations=raw_accelerations,
        distances_m=distances_m,
        centroids=centroids,
        raw_distances_px=raw_distances_px,
        calibration=calibration,
        duration_sec=duration_sec,
        centroid_only=centroid_only,
    )
    velocities = plausibility["velocities"]
    accelerations = plausibility["accelerations"]
    distances_m = plausibility["distances_m"]
    motion_anomalies = plausibility["motion_anomalies"]

    series = []
    for index, point in enumerate(centroids):
        anomaly_names = motion_anomalies[index]["anomalies"] if index < len(motion_anomalies) else []
        anomaly_penalty = min(0.42, len(anomaly_names) * 0.1)
        point_confidence = min(plausibility["point_confidence_cap"], float(point["confidence"]) * 0.65 + calibration_conf * 0.35)
        series.append(
            {
                "timestamp": round(float(point["timestamp"]), 3),
                "frame_index": int(point["frame_index"]),
                "centroid_px": {"x": round(float(point["x"]), 2), "y": round(float(point["y"]), 2)},
                "motion_source": str(point.get("source") or "unknown"),
                "visible_keypoints": int(point.get("visible_keypoints", 0) or 0),
                "distance_m": round(distances_m[index], 3) if distances_m[index] is not None else None,
                "distance_px": round(raw_distances_px[index], 3),
                "raw_velocity": round(raw_velocities[index], 3),
                "velocity": round(velocities[index], 3),
                "corrected_velocity": round(velocities[index], 3),
                "velocity_unit": unit,
                "raw_acceleration": round(raw_accelerations[index], 3),
                "acceleration": round(accelerations[index], 3),
                "acceleration_unit": "m/s^2" if pixel_to_meter else "px/s^2",
                "confidence": round(max(0.0, point_confidence - anomaly_penalty), 3),
                "tracking_state": str(point.get("tracking_state") or ""),
                "motion_anomalies": anomaly_names,
            }
        )

    velocity_values = [point["velocity"] for point in series]
    distance_values = [float(point["distance_m"]) for point in series if point.get("distance_m") is not None]
    measured_duration = max(0.0, float(series[-1]["timestamp"]) - float(series[0]["timestamp"])) if len(series) > 1 else 0.0
    avg_velocity = (
        (max(distance_values) - min(distance_values)) / measured_duration
        if unit == "m/s" and distance_values and measured_duration > 0
        else mean(velocity_values) if velocity_values else 0.0
    )
    base_confidence = round(min(1.0, calibration_conf * 0.55 + mean([point["confidence"] for point in series]) * 0.45), 3)
    summary_confidence = round(min(base_confidence, plausibility["summary_confidence_cap"]), 3)
    dead_spots = _dead_spots(series, avg_velocity, summary_confidence=summary_confidence, centroid_only=centroid_only)
    review_splits = _review_splits(series, calibration, summary_confidence=summary_confidence)
    summary = {
        "average_velocity": round(avg_velocity, 3),
        "max_velocity": round(max(velocity_values), 3) if velocity_values else 0.0,
        "min_velocity": round(min(velocity_values), 3) if velocity_values else 0.0,
        "velocity_unit": unit,
        "average_acceleration": round(mean([point["acceleration"] for point in series]), 3) if series else 0.0,
        "dead_spot_count": len(dead_spots),
        "confidence": summary_confidence,
        "evidence_mode": "centroid_only" if centroid_only else "pose_centroid",
        "distance_review_m": round(max(distance_values) - min(distance_values), 3) if distance_values else None,
        "motion_validation": plausibility["motion_validation"],
    }
    if plausibility["warnings"]:
        summary["warnings"] = plausibility["warnings"]
        summary["reason"] = plausibility["warnings"][0]
    return {
        "calibration": calibration,
        "summary": summary,
        "series": series,
        "dead_spots": dead_spots,
        "breathing_velocity_loss": [],
        "turn_velocity_loss": [],
        "phase_summary": [],
        "review_splits": review_splits,
        "chart_ready": True,
    }


def attach_phase_velocity_summary(velocity: dict[str, Any], phases: list[dict[str, Any]]) -> dict[str, Any]:
    series = velocity.get("series", [])
    summaries = []
    for phase in phases:
        values = [
            point
            for point in series
            if float(phase.get("start_sec", 0)) <= float(point.get("timestamp", 0)) <= float(phase.get("end_sec", 0))
        ]
        if not values:
            continue
        velocities = [float(point["velocity"]) for point in values]
        summaries.append(
            {
                "phase": phase.get("type"),
                "start_sec": phase.get("start_sec"),
                "end_sec": phase.get("end_sec"),
                "average_velocity": round(mean(velocities), 3),
                "max_velocity": round(max(velocities), 3),
                "min_velocity": round(min(velocities), 3),
                "velocity_unit": velocity.get("summary", {}).get("velocity_unit", "m/s"),
                "confidence": round(min(float(phase.get("confidence", 0)), float(velocity.get("summary", {}).get("confidence", 0))), 3),
            }
        )
    velocity["phase_summary"] = summaries
    velocity["breakout_speed"] = _phase_speed(summaries, "breakout")
    velocity["velocity_loss_before_after_turn"] = _turn_loss(summaries)
    return velocity


def velocity_loss_after_events(
    velocity: dict[str, Any],
    event_times: list[float],
    *,
    window_sec: float = 0.7,
    label: str = "breathing",
) -> list[dict[str, Any]]:
    series = velocity.get("series", [])
    losses = []
    for event_time in event_times:
        before = [point for point in series if event_time - window_sec <= float(point["timestamp"]) < event_time]
        after = [point for point in series if event_time <= float(point["timestamp"]) <= event_time + window_sec]
        if not before or not after:
            continue
        before_avg = mean([float(point["velocity"]) for point in before])
        after_avg = mean([float(point["velocity"]) for point in after])
        losses.append(
            {
                "type": label,
                "timestamp": round(event_time, 3),
                "before_velocity": round(before_avg, 3),
                "after_velocity": round(after_avg, 3),
                "loss": round(max(0.0, before_avg - after_avg), 3),
                "velocity_unit": velocity.get("summary", {}).get("velocity_unit", "m/s"),
                "confidence": round(float(velocity.get("summary", {}).get("confidence", 0)) * 0.8, 3),
            }
        )
    return losses


def _empty_velocity(calibration: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "calibration": calibration,
        "summary": {
            "average_velocity": None,
            "max_velocity": None,
            "min_velocity": None,
            "velocity_unit": "m/s" if calibration.get("pixel_to_meter") else "px/s",
            "average_acceleration": None,
            "dead_spot_count": 0,
            "confidence": 0.0,
            "reason": reason,
        },
        "series": [],
        "dead_spots": [],
        "breathing_velocity_loss": [],
        "turn_velocity_loss": [],
        "phase_summary": [],
        "chart_ready": False,
    }


def _motion_point_for_frame(frame: PoseFrameResult) -> dict[str, Any] | None:
    body = [
        point
        for point in frame.keypoints
        if point.name in BODY_CENTROID_NAMES and point.confidence >= 0.12 and point.visibility_state != "rejected_outlier"
    ]
    if len(body) >= 2:
        return _centroid_payload(frame, body, source="body_centroid")

    pose = [
        point
        for point in frame.keypoints
        if point.name in POSE_CENTROID_NAMES and point.confidence >= 0.12 and point.visibility_state != "rejected_outlier"
    ]
    if len(pose) >= 3:
        return _centroid_payload(frame, pose, source="pose_centroid")

    bbox = _frame_motion_bbox(frame)
    if bbox is None:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return {
        "x": (x1 + x2) / 2.0,
        "y": (y1 + y2) / 2.0,
        "confidence": max(0.0, min(0.42, float(frame.confidence))),
        "visible_keypoints": 0,
        "timestamp": frame.timestamp,
        "frame_index": frame.frame_index,
        "source": "roi_center" if frame.roi and frame.roi.get("valid") else "bbox_center",
        "tracking_state": _tracking_state(frame),
        "tracking_quality": frame.tracking_quality,
        "frame_quality_score": frame.frame_quality_score,
    }


def _centroid_payload(frame: PoseFrameResult, points: list[Any], *, source: str) -> dict[str, Any]:
    confidence = sum(float(point.confidence) for point in points) / len(points)
    return {
        "x": sum(float(point.x) for point in points) / len(points),
        "y": sum(float(point.y) for point in points) / len(points),
        "confidence": confidence,
        "visible_keypoints": len(points),
        "timestamp": frame.timestamp,
        "frame_index": frame.frame_index,
        "source": source,
        "tracking_state": _tracking_state(frame),
        "tracking_quality": frame.tracking_quality,
        "frame_quality_score": frame.frame_quality_score,
    }


def _frame_motion_bbox(frame: PoseFrameResult) -> list[float] | None:
    roi = frame.roi or {}
    if roi.get("valid") and roi.get("width") and roi.get("height"):
        x = float(roi.get("x", 0) or 0)
        y = float(roi.get("y", 0) or 0)
        return [x, y, x + float(roi.get("width", 0) or 0), y + float(roi.get("height", 0) or 0)]
    if frame.bbox:
        return [float(value) for value in frame.bbox[:4]]
    return None


def _tracking_state(frame: PoseFrameResult) -> str:
    tracking = frame.debug_info.get("roi_tracking") if isinstance(frame.debug_info, dict) else None
    if isinstance(tracking, dict):
        return str(tracking.get("tracking_state") or "")
    return ""


def _dominant_direction(centroids: list[dict[str, float | int]], axis: str) -> int:
    return 1 if float(centroids[-1][axis]) >= float(centroids[0][axis]) else -1


def _apply_velocity_plausibility_gates(
    *,
    velocities: list[float],
    accelerations: list[float],
    raw_velocities: list[float],
    raw_accelerations: list[float],
    distances_m: list[float | None],
    centroids: list[dict[str, float | int]],
    raw_distances_px: list[float],
    calibration: dict[str, Any],
    duration_sec: float,
    centroid_only: bool,
) -> dict[str, Any]:
    warnings: list[str] = []
    pixel_to_meter = calibration.get("pixel_to_meter")
    lane_length_m = calibration.get("lane_length_m")
    summary_confidence_cap = 1.0
    point_confidence_cap = 1.0
    motion_anomalies = _motion_anomaly_diagnostics(
        centroids=centroids,
        velocities=velocities,
        accelerations=accelerations,
        raw_velocities=raw_velocities,
        raw_accelerations=raw_accelerations,
        pixel_to_meter=bool(pixel_to_meter),
    )
    anomaly_names = _flatten_anomaly_names(motion_anomalies)
    if anomaly_names:
        warnings.append("Motion physics validation corrected unstable speed samples from ROI jitter, camera motion, or tracking loss.")
        summary_confidence_cap = min(summary_confidence_cap, _motion_confidence_cap(anomaly_names, len(centroids)))

    if centroid_only:
        summary_confidence_cap = 0.34
        point_confidence_cap = 0.38
        warnings.append("OpenCV fallback found a swimmer bbox but no skeleton landmarks; velocity is approximate and biomechanics are unavailable.")

    if not pixel_to_meter:
        if centroid_only:
            warnings.append("No meter calibration is available, so the speed curve is pixel-scale only.")
        positive_px = [float(value) for value in velocities if float(value) > 0]
        px_peak_cap = max(250.0, (median(positive_px) if positive_px else 0.0) * 3.0)
        velocities = _correct_motion_anomalies(velocities, motion_anomalies, peak_cap=px_peak_cap)
        velocities = [max(0.0, min(float(value), px_peak_cap)) for value in moving_average(_median_filter(velocities, window=5), window=5)]
        accelerations = _recompute_accelerations(velocities, centroids)
        motion_anomalies = _motion_anomaly_diagnostics(
            centroids=centroids,
            velocities=velocities,
            accelerations=accelerations,
            raw_velocities=raw_velocities,
            raw_accelerations=raw_accelerations,
            pixel_to_meter=False,
        )
        return {
            "velocities": velocities,
            "accelerations": accelerations,
            "distances_m": distances_m,
            "warnings": warnings,
            "summary_confidence_cap": summary_confidence_cap,
            "point_confidence_cap": point_confidence_cap,
            "motion_anomalies": motion_anomalies,
            "motion_validation": _motion_validation_summary(
                motion_anomalies,
                raw_velocities=raw_velocities,
                corrected_velocities=velocities,
                accelerations=accelerations,
                unit="px/s",
            ),
        }

    raw_average = mean(velocities) if velocities else 0.0
    raw_peak = max(velocities) if velocities else 0.0
    target_average = None
    target_distance = None
    if isinstance(lane_length_m, (int, float)) and lane_length_m > 0 and duration_sec > 0:
        target_distance = float(lane_length_m)
        target_average = float(lane_length_m) / duration_sec
        if MIN_PLAUSIBLE_AVERAGE_MPS <= target_average <= MAX_PLAUSIBLE_AVERAGE_MPS:
            velocities = (
                _centroid_review_velocity_profile(velocities, target_average)
                if centroid_only
                else _scale_series(velocities, accelerations, target_average)[0]
            )
            accelerations = _recompute_accelerations(velocities, centroids)
            distances_m = _integrated_distances(velocities, centroids)
            warnings.append("Velocity was normalized to manual lane length and clip duration to reduce centroid-jitter error.")
        elif target_average > MAX_PLAUSIBLE_AVERAGE_MPS:
            warnings.append("Manual lane length divided by clip duration implies an impossible swim speed; velocity was capped for plausibility.")
            target_distance = MAX_PLAUSIBLE_AVERAGE_MPS * duration_sec
            velocities = _centroid_review_velocity_profile(velocities, MAX_PLAUSIBLE_AVERAGE_MPS) if centroid_only else _scale_series(velocities, accelerations, MAX_PLAUSIBLE_AVERAGE_MPS)[0]
            accelerations = _recompute_accelerations(velocities, centroids)
            distances_m = _integrated_distances(velocities, centroids)
            summary_confidence_cap = min(summary_confidence_cap, 0.28)
        else:
            warnings.append("Manual lane length and clip duration imply a very slow or partial-lane capture; treat velocity as approximate.")
            summary_confidence_cap = min(summary_confidence_cap, 0.42)

    if raw_average > MAX_PLAUSIBLE_AVERAGE_MPS * 1.8 or raw_peak > MAX_PLAUSIBLE_PEAK_MPS * 2.0:
        warnings.append(
            f"Raw centroid speed was implausible before correction (avg {raw_average:.2f} m/s, peak {raw_peak:.2f} m/s), likely from camera motion or bbox jitter."
        )
        summary_confidence_cap = min(summary_confidence_cap, 0.38)

    peak_cap = MAX_PLAUSIBLE_PEAK_MPS
    if target_average and target_average > 0:
        peak_cap = min(MAX_PLAUSIBLE_PEAK_MPS, max(1.3, target_average * 2.2))
    velocities = _correct_motion_anomalies(velocities, motion_anomalies, peak_cap=peak_cap)
    velocities = _cap_acceleration(velocities, centroids, max_acceleration=MAX_PLAUSIBLE_ACCELERATION_MPS2)
    velocities = [max(0.0, min(float(value), peak_cap)) for value in moving_average(_median_filter(velocities, window=5), window=5)]
    accelerations = _recompute_accelerations(velocities, centroids)
    if target_distance and duration_sec > 0 and MIN_PLAUSIBLE_AVERAGE_MPS <= target_distance / duration_sec <= MAX_PLAUSIBLE_AVERAGE_MPS:
        velocities = _scale_velocities_to_distance(velocities, centroids, target_distance)
        velocities = _cap_acceleration(velocities, centroids, max_acceleration=MAX_PLAUSIBLE_ACCELERATION_MPS2)
        velocities = [max(0.0, min(float(value), peak_cap)) for value in velocities]
        accelerations = _recompute_accelerations(velocities, centroids)
    if distances_m and any(value is not None for value in distances_m):
        distances_m = _integrated_distances(velocities, centroids)
    motion_anomalies = _motion_anomaly_diagnostics(
        centroids=centroids,
        velocities=velocities,
        accelerations=accelerations,
        raw_velocities=raw_velocities,
        raw_accelerations=raw_accelerations,
        pixel_to_meter=True,
    )

    return {
        "velocities": velocities,
        "accelerations": accelerations,
        "distances_m": distances_m,
        "warnings": _dedupe_strings(warnings),
        "summary_confidence_cap": summary_confidence_cap,
        "point_confidence_cap": point_confidence_cap,
        "motion_anomalies": motion_anomalies,
        "motion_validation": _motion_validation_summary(
            motion_anomalies,
            raw_velocities=raw_velocities,
            corrected_velocities=velocities,
            accelerations=accelerations,
            unit="m/s",
        ),
    }


def _median_filter(values: list[float], *, window: int = 5) -> list[float]:
    if len(values) < 3:
        return list(values)
    radius = max(1, window // 2)
    output = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        output.append(float(median(values[start:end])))
    return output


def _motion_anomaly_diagnostics(
    *,
    centroids: list[dict[str, Any]],
    velocities: list[float],
    accelerations: list[float],
    raw_velocities: list[float],
    raw_accelerations: list[float],
    pixel_to_meter: bool,
) -> list[dict[str, Any]]:
    if not centroids:
        return []
    positive = [float(value) for value in raw_velocities if float(value) > 1e-6]
    positive_accel = [abs(float(value)) for value in raw_accelerations if abs(float(value)) > 1e-6]
    median_velocity = median(positive) if positive else 0.0
    median_accel = median(positive_accel) if positive_accel else 0.0
    speed_gate = MAX_PLAUSIBLE_PEAK_MPS if pixel_to_meter else max(900.0, median_velocity * 4.0)
    accel_gate = MAX_PLAUSIBLE_ACCELERATION_MPS2 if pixel_to_meter else max(9000.0, median_accel * 4.0)
    zero_gate = 0.08 if pixel_to_meter else max(10.0, median_velocity * 0.05)
    zero_like = [float(value) <= zero_gate for value in raw_velocities]
    zero_runs = _boolean_runs(zero_like, min_length=3)
    diagnostics: list[dict[str, Any]] = []

    for index, point in enumerate(centroids):
        anomalies: list[str] = []
        raw_velocity = float(raw_velocities[index]) if index < len(raw_velocities) else 0.0
        corrected_velocity = float(velocities[index]) if index < len(velocities) else raw_velocity
        raw_accel = float(raw_accelerations[index]) if index < len(raw_accelerations) else 0.0
        accel = float(accelerations[index]) if index < len(accelerations) else raw_accel
        source = str(point.get("source") or "unknown")
        tracking_state = str(point.get("tracking_state") or "")

        if raw_velocity > speed_gate:
            anomalies.append("unrealistic_speed_spike")
        if abs(raw_accel) > accel_gate:
            anomalies.append("unrealistic_acceleration_spike")
        if index in zero_runs and len(raw_velocities) > 3 and (source in {"roi_center", "bbox_center"} or tracking_state in {"SUSPECT", "LOST", "REACQUIRE", "REVIEW_ONLY"}):
            anomalies.append("frozen_roi_plateau")
        if _is_zero_velocity_collapse(raw_velocities, index, zero_gate):
            anomalies.append("sudden_zero_velocity_collapse")
        if source in {"roi_center", "bbox_center"} and raw_velocity > max(zero_gate * 5.0, speed_gate * 0.42):
            anomalies.append("roi_motion_dominates_body_evidence")
        if source != "body_centroid" and raw_velocity > max(zero_gate * 6.0, speed_gate * 0.6):
            anomalies.append("camera_motion_or_roi_jitter")
        if tracking_state in {"LOST", "REACQUIRE", "REVIEW_ONLY"} and raw_velocity > zero_gate * 2.0:
            anomalies.append("low_tracking_motion_not_reliable")

        diagnostics.append(
            {
                "frame_index": int(point.get("frame_index", index) or index),
                "timestamp": round(float(point.get("timestamp", 0) or 0), 4),
                "motion_source": source,
                "tracking_state": tracking_state,
                "raw_velocity": round(raw_velocity, 4),
                "corrected_velocity": round(corrected_velocity, 4),
                "raw_acceleration": round(raw_accel, 4),
                "acceleration": round(accel, 4),
                "anomalies": _dedupe_strings(anomalies),
            }
        )
    return diagnostics


def _boolean_runs(values: list[bool], *, min_length: int) -> set[int]:
    indices: set[int] = set()
    start: int | None = None
    for index, value in enumerate([*values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_length:
                indices.update(range(start, index))
            start = None
    return indices


def _is_zero_velocity_collapse(values: list[float], index: int, zero_gate: float) -> bool:
    if index <= 0 or index >= len(values) - 1:
        return False
    previous_value = float(values[index - 1])
    current = float(values[index])
    next_value = float(values[index + 1])
    neighbor_floor = max(zero_gate * 4.0, min(previous_value, next_value) * 0.28)
    return current <= zero_gate and previous_value >= neighbor_floor and next_value >= neighbor_floor


def _correct_motion_anomalies(velocities: list[float], diagnostics: list[dict[str, Any]], *, peak_cap: float) -> list[float]:
    if not velocities:
        return []
    output = [float(value) for value in velocities]
    severe = {"unrealistic_speed_spike", "unrealistic_acceleration_spike", "sudden_zero_velocity_collapse", "camera_motion_or_roi_jitter"}
    for index, item in enumerate(diagnostics):
        names = set(item.get("anomalies") or [])
        if not names.intersection(severe):
            output[index] = max(0.0, min(output[index], peak_cap))
            continue
        neighbors = [
            output[pos]
            for pos in range(max(0, index - 2), min(len(output), index + 3))
            if pos != index and not set(diagnostics[pos].get("anomalies") or []).intersection(severe)
        ]
        if neighbors:
            output[index] = float(median(neighbors))
        elif index > 0:
            output[index] = output[index - 1]
        output[index] = max(0.0, min(output[index], peak_cap))
    return output


def _cap_acceleration(velocities: list[float], centroids: list[dict[str, Any]], *, max_acceleration: float) -> list[float]:
    if len(velocities) < 2:
        return list(velocities)
    output = [max(0.0, float(velocities[0]))]
    for index in range(1, len(velocities)):
        dt = max(1e-3, float(centroids[index]["timestamp"]) - float(centroids[index - 1]["timestamp"]))
        max_delta = max_acceleration * dt
        previous = output[-1]
        value = max(0.0, float(velocities[index]))
        output.append(max(0.0, min(previous + max_delta, max(previous - max_delta, value))))
    return output


def _flatten_anomaly_names(diagnostics: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in diagnostics:
        names.extend(str(name) for name in item.get("anomalies", []) or [])
    return names


def _motion_confidence_cap(anomaly_names: list[str], sample_count: int) -> float:
    if not anomaly_names or sample_count <= 0:
        return 1.0
    rate = len(anomaly_names) / max(1, sample_count)
    severe = sum(1 for name in anomaly_names if name in {"unrealistic_speed_spike", "unrealistic_acceleration_spike", "camera_motion_or_roi_jitter"})
    cap = 0.58 - min(0.26, rate * 0.18) - min(0.18, severe / max(1, sample_count) * 0.2)
    return round(max(0.24, min(0.58, cap)), 3)


def _motion_validation_summary(
    diagnostics: list[dict[str, Any]],
    *,
    raw_velocities: list[float],
    corrected_velocities: list[float],
    accelerations: list[float],
    unit: str,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for item in diagnostics:
        source = str(item.get("motion_source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        for name in item.get("anomalies", []) or []:
            key = str(name)
            counts[key] = counts.get(key, 0) + 1
    return {
        "raw_peak_velocity": round(max(raw_velocities), 3) if raw_velocities else 0.0,
        "corrected_peak_velocity": round(max(corrected_velocities), 3) if corrected_velocities else 0.0,
        "max_abs_acceleration": round(max([abs(float(value)) for value in accelerations], default=0.0), 3),
        "velocity_unit": unit,
        "acceleration_unit": "m/s^2" if unit == "m/s" else "px/s^2",
        "anomaly_count": sum(counts.values()),
        "anomaly_counts": counts,
        "motion_source_counts": source_counts,
        "diagnostics": diagnostics,
    }


def _scale_series(velocities: list[float], accelerations: list[float], target_average: float) -> tuple[list[float], list[float]]:
    current_average = mean(velocities) if velocities else 0.0
    if current_average <= 0:
        return velocities, accelerations
    scale = target_average / current_average
    return [max(0.0, value * scale) for value in velocities], [value * scale for value in accelerations]


def _centroid_review_velocity_profile(velocities: list[float], target_average: float) -> list[float]:
    if not velocities:
        return []
    if max(velocities) <= 0:
        return [target_average for _ in velocities]
    smoothed = moving_average([max(0.0, value) for value in velocities], window=13)
    current_average = mean(smoothed) if smoothed else 0.0
    if current_average <= 0:
        return [target_average for _ in velocities]
    normalized = [value / current_average for value in smoothed]
    blended = [target_average * (0.72 + 0.28 * min(2.0, ratio)) for ratio in normalized]
    low = target_average * 0.45
    high = min(MAX_PLAUSIBLE_PEAK_MPS, max(1.3, target_average * 1.9))
    return [max(low, min(high, value)) for value in blended]


def _scale_velocities_to_distance(velocities: list[float], centroids: list[dict[str, float | int]], target_distance: float) -> list[float]:
    if not velocities or len(centroids) < 2 or target_distance <= 0:
        return velocities
    current_distance = _integrated_distances(velocities, centroids)[-1]
    if current_distance <= 0:
        duration = max(1e-3, float(centroids[-1]["timestamp"]) - float(centroids[0]["timestamp"]))
        return [target_distance / duration for _ in velocities]
    scale = target_distance / current_distance
    return [max(0.0, value * scale) for value in velocities]


def _integrated_distances(velocities: list[float], centroids: list[dict[str, float | int]]) -> list[float]:
    distances = [0.0]
    for index in range(1, len(centroids)):
        dt = max(1e-3, float(centroids[index]["timestamp"]) - float(centroids[index - 1]["timestamp"]))
        distances.append(distances[-1] + max(0.0, float(velocities[index])) * dt)
    return distances


def _recompute_accelerations(velocities: list[float], centroids: list[dict[str, float | int]]) -> list[float]:
    if not velocities:
        return []
    accelerations = [0.0]
    for index in range(1, len(velocities)):
        dt = max(1e-3, float(centroids[index]["timestamp"]) - float(centroids[index - 1]["timestamp"]))
        accelerations.append((float(velocities[index]) - float(velocities[index - 1])) / dt)
    return moving_average(accelerations, window=5)


def _dedupe_strings(values: list[str]) -> list[str]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _dead_spots(series: list[dict[str, Any]], average_velocity: float, *, summary_confidence: float, centroid_only: bool) -> list[dict[str, Any]]:
    if average_velocity <= 0 or summary_confidence < 0.45 or centroid_only:
        return []
    threshold = average_velocity * 0.55
    spots = []
    start = None
    last = None
    min_value = None
    for point in series:
        value = float(point["velocity"])
        if value < threshold:
            start = float(point["timestamp"]) if start is None else start
            last = float(point["timestamp"])
            min_value = value if min_value is None else min(min_value, value)
        elif start is not None and last is not None:
            if last - start >= 0.25:
                spots.append(
                    {
                        "start_sec": round(start, 3),
                        "end_sec": round(last, 3),
                        "min_velocity": round(float(min_value or 0), 3),
                        "threshold": round(threshold, 3),
                        "confidence": round(min(float(point.get("confidence", 0)), 0.8), 3),
                    }
                )
            start = None
            last = None
            min_value = None
    return spots[:12]


def _phase_speed(summaries: list[dict[str, Any]], phase_name: str) -> dict[str, Any] | None:
    for summary in summaries:
        if summary.get("phase") == phase_name:
            return {
                "value": summary.get("average_velocity"),
                "unit": summary.get("velocity_unit"),
                "confidence": summary.get("confidence"),
            }
    return None


def _turn_loss(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    losses = []
    for index, summary in enumerate(summaries):
        if summary.get("phase") != "turn":
            continue
        before = summaries[index - 1] if index > 0 else None
        after = summaries[index + 1] if index + 1 < len(summaries) else None
        if before and after:
            losses.append(
                {
                    "turn_start_sec": summary.get("start_sec"),
                    "turn_end_sec": summary.get("end_sec"),
                    "before_average_velocity": before.get("average_velocity"),
                    "after_average_velocity": after.get("average_velocity"),
                    "loss": round(max(0.0, float(before.get("average_velocity") or 0) - float(after.get("average_velocity") or 0)), 3),
                    "velocity_unit": summary.get("velocity_unit"),
                    "confidence": min(float(summary.get("confidence", 0)), float(before.get("confidence", 0)), float(after.get("confidence", 0))),
                }
            )
    return losses


def _review_splits(series: list[dict[str, Any]], calibration: dict[str, Any], *, summary_confidence: float) -> list[dict[str, Any]]:
    lane_length = calibration.get("lane_length_m")
    if not isinstance(lane_length, (int, float)) or lane_length <= 0:
        return []
    distance_values = [float(point["distance_m"]) for point in series if point.get("distance_m") is not None]
    if not distance_values:
        return []
    final_distance = max(distance_values) - min(distance_values)
    if final_distance < min(float(lane_length) * 0.35, 10.0):
        return []
    milestones = _distance_milestones(float(lane_length), final_distance)
    output = []
    previous_distance = 0.0
    previous_time = float(series[0]["timestamp"])
    for milestone in milestones:
        split_time = _time_at_distance(series, milestone)
        if split_time is None or split_time <= previous_time:
            continue
        distance = milestone - previous_distance
        duration = split_time - previous_time
        output.append(
            {
                "label": f"{previous_distance:g}-{milestone:g}m",
                "start_distance_m": round(previous_distance, 3),
                "end_distance_m": round(milestone, 3),
                "start_sec": round(previous_time, 3),
                "end_sec": round(split_time, 3),
                "split_time_sec": round(duration, 3),
                "distance_m": round(distance, 3),
                "average_velocity": round(distance / duration, 3) if duration > 0 else None,
                "velocity_unit": "m/s",
                "confidence": round(min(summary_confidence, float(calibration.get("confidence", 0) or 0), 0.42), 3),
                "reason": "Manual lane length plus centroid progression; suitable for review timing, not biomechanics.",
            }
        )
        previous_distance = milestone
        previous_time = split_time
    return output


def _distance_milestones(lane_length: float, final_distance: float) -> list[float]:
    candidates = [15.0, 25.0, 50.0, lane_length]
    if lane_length > 50:
        candidates.extend(range(25, int(lane_length) + 1, 25))
    milestones = sorted({float(value) for value in candidates if 0 < float(value) <= min(lane_length, final_distance) + 1e-6})
    if final_distance < lane_length and final_distance not in milestones:
        milestones.append(round(final_distance, 3))
    return milestones[:8]


def _time_at_distance(series: list[dict[str, Any]], target_distance: float) -> float | None:
    previous = None
    for point in series:
        distance = point.get("distance_m")
        if distance is None:
            continue
        current_distance = float(distance)
        current_time = float(point["timestamp"])
        if current_distance >= target_distance:
            if previous is None:
                return current_time
            prev_distance = float(previous["distance_m"])
            prev_time = float(previous["timestamp"])
            span = current_distance - prev_distance
            if span <= 1e-6:
                return current_time
            ratio = (target_distance - prev_distance) / span
            return prev_time + (current_time - prev_time) * max(0.0, min(1.0, ratio))
        previous = point
    return None
