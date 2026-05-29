from __future__ import annotations

from dataclasses import replace

from app.swim_analysis.pose.base import PoseFrameResult


def estimate_sync_offset_audio_stub(*_: object, **__: object) -> dict[str, float | str]:
    return {"offset_sec": 0.0, "confidence": 0.0, "method": "audio_stub_unavailable_phase_1"}


def estimate_sync_offset_motion(
    side_frames: list[PoseFrameResult],
    front_frames: list[PoseFrameResult],
    *,
    max_offset_sec: float = 3.0,
) -> dict[str, float | str]:
    side_signal = _motion_signal(side_frames)
    front_signal = _motion_signal(front_frames)
    if len(side_signal) < 4 or len(front_signal) < 4:
        return {"offset_sec": 0.0, "confidence": 0.0, "method": "manual_zero_offset_insufficient_motion"}

    side_step = _median_dt(side_frames)
    front_step = _median_dt(front_frames)
    step = max(1e-3, (side_step + front_step) / 2.0)
    max_lag = max(1, round(max_offset_sec / step))
    best_lag = 0
    best_score = -1.0
    for lag in range(-max_lag, max_lag + 1):
        score = _correlation_at_lag(side_signal, front_signal, lag)
        if score > best_score:
            best_score = score
            best_lag = lag
    confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
    if confidence < 0.25:
        return {"offset_sec": 0.0, "confidence": round(confidence, 3), "method": "manual_zero_offset_low_motion_agreement"}
    return {"offset_sec": round(best_lag * step, 3), "confidence": round(confidence, 3), "method": "motion_cross_correlation"}


def align_timelines(
    side_frames: list[PoseFrameResult],
    front_frames: list[PoseFrameResult],
    *,
    manual_offset_sec: float = 0.0,
    motion_offset_sec: float | None = None,
) -> dict[str, list[PoseFrameResult] | float]:
    offset = manual_offset_sec if manual_offset_sec != 0 else float(motion_offset_sec or 0.0)
    aligned_front = [replace(frame, timestamp=frame.timestamp + offset) for frame in front_frames]
    return {"side": side_frames, "front": aligned_front, "offset_sec": offset}


def _motion_signal(frames: list[PoseFrameResult]) -> list[float]:
    centers = []
    for frame in frames:
        points = [point for point in frame.keypoints if point.name in {"left_wrist", "right_wrist", "left_hip", "right_hip"} and point.confidence >= 0.2]
        if not points:
            continue
        centers.append((frame.timestamp, sum(point.x for point in points) / len(points), sum(point.y for point in points) / len(points)))
    if len(centers) < 3:
        return []
    values = []
    for previous, current in zip(centers, centers[1:]):
        dt = max(1e-3, current[0] - previous[0])
        values.append((((current[1] - previous[1]) ** 2 + (current[2] - previous[2]) ** 2) ** 0.5) / dt)
    return _zscore(values)


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    avg = sum(values) / len(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    std = variance**0.5
    if std < 1e-6:
        return [0.0 for _ in values]
    return [(value - avg) / std for value in values]


def _correlation_at_lag(left: list[float], right: list[float], lag: int) -> float:
    pairs = []
    for index, value in enumerate(left):
        other_index = index + lag
        if 0 <= other_index < len(right):
            pairs.append((value, right[other_index]))
    if len(pairs) < 3:
        return -1.0
    return sum(a * b for a, b in pairs) / len(pairs)


def _median_dt(frames: list[PoseFrameResult]) -> float:
    deltas = [b.timestamp - a.timestamp for a, b in zip(frames, frames[1:]) if b.timestamp > a.timestamp]
    if not deltas:
        return 1.0 / 30.0
    ordered = sorted(deltas)
    return ordered[len(ordered) // 2]
