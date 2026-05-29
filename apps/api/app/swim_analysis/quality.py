from __future__ import annotations

from typing import Any

from app.swim_analysis.pose.base import PoseFrameResult


def confidence_label(confidence: float | str) -> str:
    if isinstance(confidence, str):
        return confidence
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def confidence_value(confidence: float | str) -> float:
    if isinstance(confidence, str):
        return 0.0 if confidence == "low" else 0.5
    return max(0.0, min(1.0, float(confidence)))


def visible_ratio(frames: list[PoseFrameResult], names: set[str], *, min_confidence: float = 0.25) -> float:
    if not frames:
        return 0.0
    visible = 0
    for frame in frames:
        found = {point.name for point in frame.keypoints if point.confidence >= min_confidence}
        if names.issubset(found):
            visible += 1
    return visible / len(frames)


def average_pose_confidence(frames: list[PoseFrameResult]) -> float:
    values = [frame.confidence for frame in frames if frame.confidence > 0]
    if not values:
        return 0.0
    return sum(values) / len(values)


def metric_confidence(frames: list[PoseFrameResult], names: set[str], *, extra: float = 1.0) -> float:
    visibility = visible_ratio(frames, names)
    pose_conf = average_pose_confidence(frames)
    return max(0.0, min(1.0, (visibility * 0.7 + pose_conf * 0.3) * extra))


def confidence_payload(confidence: float, evidence: dict[str, Any]) -> float | str:
    if confidence < 0.35:
        evidence["numeric_confidence"] = round(confidence, 3)
        return "low"
    return round(confidence, 3)


def data_quality_score(video_quality: dict[str, Any]) -> float:
    views = [view for view in video_quality.values() if isinstance(view, dict)]
    if not views:
        return 0.0
    values = []
    for view in views:
        values.append(
            0.3 * float(view.get("usable_frame_ratio", 0))
            + 0.25 * float(view.get("blur_score", 0))
            + 0.25 * float(view.get("lighting_score", 0))
            + 0.2 * float(view.get("stability_score", 0))
        )
    return round(sum(values) / len(values), 3)


def enrich_video_quality(
    base_quality: dict[str, Any],
    metadata: dict[str, Any],
    frames: list[PoseFrameResult],
) -> dict[str, Any]:
    output = dict(base_quality)
    resolution = metadata.get("resolution", {}) if isinstance(metadata, dict) else {}
    width = int(resolution.get("width", 0) or 0)
    height = int(resolution.get("height", 0) or 0)
    fps = float(metadata.get("fps", 0) or 0) if isinstance(metadata, dict) else 0.0
    resolution_score = min(1.0, (width * height) / (1280 * 720)) if width and height else 0.0
    fps_score = 1.0 if fps >= 30 else max(0.0, min(1.0, fps / 30.0))
    landmark_visibility = visible_ratio(frames, {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}, min_confidence=0.2)
    bbox_visibility = sum(1 for frame in frames if frame.bbox is not None) / len(frames) if frames else 0.0
    visibility_score = max(landmark_visibility, bbox_visibility * 0.55)
    output.update(
        {
            "resolution_score": round(resolution_score, 3),
            "fps_score": round(fps_score, 3),
            "swimmer_visibility_score": round(visibility_score, 3),
            "landmark_visibility_score": round(landmark_visibility, 3),
            "bbox_visibility_score": round(bbox_visibility, 3),
            "overall_quality_score": round(
                0.18 * resolution_score
                + 0.14 * fps_score
                + 0.18 * float(output.get("blur_score", 0) or 0)
                + 0.18 * float(output.get("lighting_score", 0) or 0)
                + 0.14 * float(output.get("stability_score", 0) or 0)
                + 0.18 * visibility_score,
                3,
            ),
        }
    )
    warnings = list(output.get("warnings", []) or [])
    if visibility_score < 0.45:
        warnings.append("Swimmer landmarks are visible in too few frames for high-confidence biomechanics.")
    if fps_score < 0.75:
        warnings.append("Frame rate is low; velocity and stroke-timing estimates may be coarse.")
    output["warnings"] = warnings
    return output
