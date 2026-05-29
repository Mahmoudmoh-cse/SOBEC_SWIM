from __future__ import annotations

from dataclasses import replace

from app.swim_analysis.pose.base import Keypoint, PoseFrameResult, average_keypoint_confidence, bbox_from_keypoints


def track_swimmer_identity(frames: list[PoseFrameResult], *, max_center_jump_px: float = 260.0) -> list[PoseFrameResult]:
    """SORT-style single-target guard for the main swimmer track.

    The backends already choose one person per frame. This layer keeps the selected swimmer stable,
    rejects impossible center jumps, and tags accepted frames with a stable track id.
    """

    tracked: list[PoseFrameResult] = []
    last_center: tuple[float, float] | None = None
    for frame in frames:
        bbox = frame.bbox or bbox_from_keypoints(frame.keypoints)
        if bbox is None:
            tracked.append(replace(frame, track_id=1))
            continue

        center = _bbox_center(bbox)
        if last_center is not None:
            jump = ((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2) ** 0.5
            if jump > max_center_jump_px:
                tracked.append(replace(frame, keypoints=[], confidence=0.0, bbox=None, track_id=1, quality_flags=[*frame.quality_flags, "tracking_jump_rejected"]))
                continue
        last_center = center
        tracked.append(replace(frame, bbox=bbox, track_id=1))
    return tracked


def interpolate_short_gaps(frames: list[PoseFrameResult], *, max_gap_frames: int = 5) -> list[PoseFrameResult]:
    if not frames:
        return []
    output = list(frames)
    index = 0
    while index < len(output):
        if output[index].has_pose():
            index += 1
            continue
        gap_start = index
        while index < len(output) and not output[index].has_pose():
            index += 1
        gap_end = index - 1
        before = output[gap_start - 1] if gap_start > 0 else None
        after = output[index] if index < len(output) else None
        gap_len = gap_end - gap_start + 1
        if before is None or after is None or gap_len > max_gap_frames:
            continue
        for offset, frame_index in enumerate(range(gap_start, gap_end + 1), start=1):
            ratio = offset / (gap_len + 1)
            output[frame_index] = _interpolate_frame(before, after, output[frame_index], ratio)
    return output


def _interpolate_frame(before: PoseFrameResult, after: PoseFrameResult, current: PoseFrameResult, ratio: float) -> PoseFrameResult:
    after_points = {point.name: point for point in after.keypoints}
    points = []
    for point in before.keypoints:
        other = after_points.get(point.name)
        if other is None:
            continue
        points.append(
            Keypoint(
                name=point.name,
                x=point.x + (other.x - point.x) * ratio,
                y=point.y + (other.y - point.y) * ratio,
                z=point.z + (other.z - point.z) * ratio if point.z is not None and other.z is not None else None,
                confidence=min(point.confidence, other.confidence, 0.45),
            )
        )
    return replace(
        current,
        keypoints=points,
        confidence=average_keypoint_confidence(points),
        bbox=bbox_from_keypoints(points),
        interpolated=bool(points),
        quality_flags=[*current.quality_flags, "interpolated_short_gap"] if points else current.quality_flags,
    )


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
