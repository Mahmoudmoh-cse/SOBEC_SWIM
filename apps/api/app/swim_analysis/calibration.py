from __future__ import annotations

from statistics import median
from typing import Any

from app.swim_analysis.pose.base import Keypoint, PoseFrameResult


DEFAULT_SWIMMER_BODY_LENGTH_M = 1.8


def calibrate_pool_view(
    side_frames: list[PoseFrameResult],
    *,
    lane_length_m: float | None = None,
) -> dict[str, Any]:
    """Estimate pixel-to-meter conversion from manual lane length or visible body scale.

    Manual lane length is more useful than a guessed body scale, but still only approximate
    unless the camera sees the full lane in the direction of travel.
    """

    centroids = [centroid_for_frame(frame) for frame in side_frames if centroid_for_frame(frame) is not None]
    axis = dominant_motion_axis(centroids)
    span_px = _axis_span(centroids, axis)
    visibility = _pose_visibility(side_frames)
    landmark_visibility = _landmark_visibility(side_frames)

    if lane_length_m and lane_length_m > 0 and span_px > 40:
        confidence = min(0.82, 0.42 + visibility * 0.35 + min(0.2, span_px / 1200))
        if landmark_visibility < 0.2:
            confidence = min(confidence, 0.42)
        return {
            "method": "manual_lane_length_motion_span",
            "lane_length_m": lane_length_m,
            "pixel_to_meter": lane_length_m / span_px,
            "pixels_per_meter": span_px / lane_length_m,
            "motion_axis": axis,
            "motion_span_px": round(span_px, 3),
            "confidence": round(confidence, 3),
            "landmark_visibility": round(landmark_visibility, 3),
            "reason": (
                "Manual lane length divided by visible swimmer travel span in the side-view video."
                if landmark_visibility >= 0.2
                else "Manual lane length scaled a bbox-only swimmer track; use as an approximate speed trend, not biomechanics-grade calibration."
            ),
        }

    body_lengths = [_body_length_px(frame) for frame in side_frames]
    body_lengths = [value for value in body_lengths if value and value > 20]
    if body_lengths:
        body_px = median(body_lengths)
        pixel_to_meter = DEFAULT_SWIMMER_BODY_LENGTH_M / body_px
        return {
            "method": "estimated_body_length_scale",
            "lane_length_m": lane_length_m,
            "pixel_to_meter": pixel_to_meter,
            "pixels_per_meter": 1 / pixel_to_meter if pixel_to_meter else None,
            "motion_axis": axis,
            "motion_span_px": round(span_px, 3),
            "confidence": round(min(0.48, 0.18 + visibility * 0.25), 3),
            "reason": "Fallback scale estimated from visible shoulder-to-ankle body length; use manual calibration for velocity-grade results.",
        }

    return {
        "method": "uncalibrated_pixels_only",
        "lane_length_m": lane_length_m,
        "pixel_to_meter": None,
        "pixels_per_meter": None,
        "motion_axis": axis,
        "motion_span_px": round(span_px, 3),
        "confidence": 0.0,
        "reason": "No reliable lane-length or visible body scale was available.",
    }


def centroid_for_frame(frame: PoseFrameResult, *, min_confidence: float = 0.2) -> dict[str, float] | None:
    names = {
        "nose",
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
    points = [point for point in frame.keypoints if point.name in names and point.confidence >= min_confidence]
    if not points and frame.bbox:
        return {
            "x": (float(frame.bbox[0]) + float(frame.bbox[2])) / 2.0,
            "y": (float(frame.bbox[1]) + float(frame.bbox[3])) / 2.0,
            "confidence": max(0.0, min(0.45, frame.confidence)),
            "visible_keypoints": 0,
            "timestamp": frame.timestamp,
            "frame_index": frame.frame_index,
        }
    if not points:
        return None
    confidence = sum(point.confidence for point in points) / len(points)
    return {
        "x": sum(point.x for point in points) / len(points),
        "y": sum(point.y for point in points) / len(points),
        "confidence": confidence,
        "visible_keypoints": len(points),
        "timestamp": frame.timestamp,
        "frame_index": frame.frame_index,
    }


def dominant_motion_axis(centroids: list[dict[str, float | int]]) -> str:
    if len(centroids) < 2:
        return "x"
    x_span = max(float(point["x"]) for point in centroids) - min(float(point["x"]) for point in centroids)
    y_span = max(float(point["y"]) for point in centroids) - min(float(point["y"]) for point in centroids)
    return "x" if x_span >= y_span else "y"


def interpretation_for_calibration(calibration: dict[str, Any]) -> str:
    confidence = float(calibration.get("confidence", 0) or 0)
    if confidence >= 0.7:
        return "velocity estimates are suitable for coaching review"
    if confidence >= 0.35:
        return "velocity estimates are approximate; compare trends rather than exact speed"
    return "velocity is pixel-scale only; do not use exact meters-per-second values"


def _axis_span(centroids: list[dict[str, float | int]], axis: str) -> float:
    if len(centroids) < 2:
        return 0.0
    values = [float(point[axis]) for point in centroids]
    return max(values) - min(values)


def _body_length_px(frame: PoseFrameResult) -> float | None:
    shoulder = _midpoint(frame.keypoint("left_shoulder", 0.25), frame.keypoint("right_shoulder", 0.25))
    ankles = _midpoint(frame.keypoint("left_ankle", 0.2), frame.keypoint("right_ankle", 0.2))
    if shoulder is None or ankles is None:
        return None
    return ((shoulder.x - ankles.x) ** 2 + (shoulder.y - ankles.y) ** 2) ** 0.5


def _pose_visibility(frames: list[PoseFrameResult]) -> float:
    if not frames:
        return 0.0
    visible = sum(1 for frame in frames if frame.has_pose(min_confidence=0.2) or frame.bbox is not None)
    avg_conf = sum(frame.confidence for frame in frames) / len(frames)
    return max(0.0, min(1.0, visible / len(frames) * 0.65 + avg_conf * 0.35))


def _landmark_visibility(frames: list[PoseFrameResult]) -> float:
    if not frames:
        return 0.0
    visible = sum(1 for frame in frames if frame.has_pose(min_confidence=0.2))
    return visible / len(frames)


def _midpoint(a: Keypoint | None, b: Keypoint | None) -> Keypoint | None:
    if a is None or b is None:
        return None
    return Keypoint("midpoint", (a.x + b.x) / 2, (a.y + b.y) / 2, min(a.confidence, b.confidence))
