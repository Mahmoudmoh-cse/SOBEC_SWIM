from __future__ import annotations

from dataclasses import replace
from math import pi
from statistics import mean
from typing import Any, overload

from app.swim_analysis.pose.base import Keypoint, PoseFrameResult, average_keypoint_confidence


class LowPassFilter:
    def __init__(self) -> None:
        self.initialized = False
        self.previous_raw = 0.0
        self.previous_filtered = 0.0

    def filter(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.initialized = True
            self.previous_raw = value
            self.previous_filtered = value
            return value
        filtered = alpha * value + (1.0 - alpha) * self.previous_filtered
        self.previous_raw = value
        self.previous_filtered = filtered
        return filtered


class OneEuroFilter:
    def __init__(self, *, min_cutoff: float = 1.0, beta: float = 0.035, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: float | None = None

    def __call__(self, value: float, timestamp: float, *, confidence: float = 1.0) -> float:
        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filter.filter(value, 1.0)

        dt = max(1e-3, timestamp - self.last_time)
        self.last_time = timestamp
        dx = (value - self.x_filter.previous_raw) / dt
        edx = self.dx_filter.filter(dx, _alpha(dt, self.d_cutoff))
        confidence = max(0.0, min(1.0, confidence))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = _alpha(dt, cutoff) * (0.35 + 0.65 * confidence)
        return self.x_filter.filter(value, max(0.01, min(1.0, alpha)))


@overload
def smooth_pose_sequence(
    frames: list[PoseFrameResult],
    *,
    min_confidence: float = 0.15,
    moving_average_window: int = 3,
    return_diagnostics: bool = False,
    min_cutoff: float = 1.0,
    beta: float = 0.035,
    d_cutoff: float = 1.0,
    ema_alpha_low: float = 0.32,
    ema_alpha_high: float = 0.88,
) -> list[PoseFrameResult]:
    ...


@overload
def smooth_pose_sequence(
    frames: list[PoseFrameResult],
    *,
    min_confidence: float = 0.15,
    moving_average_window: int = 3,
    return_diagnostics: bool,
    min_cutoff: float = 1.0,
    beta: float = 0.035,
    d_cutoff: float = 1.0,
    ema_alpha_low: float = 0.32,
    ema_alpha_high: float = 0.88,
) -> tuple[list[PoseFrameResult], dict[str, Any]]:
    ...


def smooth_pose_sequence(
    frames: list[PoseFrameResult],
    *,
    min_confidence: float = 0.15,
    moving_average_window: int = 3,
    return_diagnostics: bool = False,
    min_cutoff: float = 1.0,
    beta: float = 0.035,
    d_cutoff: float = 1.0,
    ema_alpha_low: float = 0.32,
    ema_alpha_high: float = 0.88,
) -> list[PoseFrameResult] | tuple[list[PoseFrameResult], dict[str, Any]]:
    if not frames:
        diagnostics = _empty_diagnostics(enabled=True, min_confidence=min_confidence)
        return ([], diagnostics) if return_diagnostics else []

    filters: dict[tuple[str, str], OneEuroFilter] = {}
    ema_state: dict[tuple[str, str], float] = {}
    previous_points: dict[str, Keypoint] = {}
    previous_raw: dict[tuple[str, str], tuple[float, float]] = {}
    displacements_by_joint: dict[str, list[float]] = {}
    frame_displacements: list[dict[str, Any]] = []
    low_confidence_holds = 0
    smoothed: list[PoseFrameResult] = []
    for frame in frames:
        reset_smoothing = _should_reset_smoothing(frame)
        if reset_smoothing:
            filters.clear()
            ema_state.clear()
            previous_points.clear()
            previous_raw.clear()
        next_points: list[Keypoint] = []
        frame_distances: list[float] = []
        for point in frame.keypoints:
            raw_point = point
            if point.confidence < min_confidence:
                held = _temporal_hold(point, previous_points.get(point.name), frame_index=frame.frame_index)
                if held is not point:
                    low_confidence_holds += 1
                    distance = _point_distance(raw_point, held)
                    displacements_by_joint.setdefault(point.name, []).append(distance)
                    frame_distances.append(distance)
                    next_points.append(held)
                    previous_points[held.name] = held
                else:
                    next_points.append(point)
                continue
            x_filter = filters.setdefault((point.name, "x"), OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff))
            y_filter = filters.setdefault((point.name, "y"), OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff))
            velocity_hint = _velocity_hint(previous_raw, point, frame.timestamp)
            euro_x = x_filter(point.x, frame.timestamp, confidence=point.confidence)
            euro_y = y_filter(point.y, frame.timestamp, confidence=point.confidence)
            smoothed_x = _confidence_aware_ema(
                state=ema_state,
                key=(point.name, "x"),
                value=euro_x,
                confidence=point.confidence,
                velocity_hint=velocity_hint,
                alpha_low=ema_alpha_low,
                alpha_high=ema_alpha_high,
            )
            smoothed_y = _confidence_aware_ema(
                state=ema_state,
                key=(point.name, "y"),
                value=euro_y,
                confidence=point.confidence,
                velocity_hint=velocity_hint,
                alpha_low=ema_alpha_low,
                alpha_high=ema_alpha_high,
            )
            z_value = point.z
            if z_value is not None:
                z_filter = filters.setdefault((point.name, "z"), OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff))
                z_euro = z_filter(z_value, frame.timestamp, confidence=point.confidence)
                z_value = _confidence_aware_ema(
                    state=ema_state,
                    key=(point.name, "z"),
                    value=z_euro,
                    confidence=point.confidence,
                    velocity_hint=velocity_hint,
                    alpha_low=ema_alpha_low,
                    alpha_high=ema_alpha_high,
                )
            smoothed_point = replace(point, x=smoothed_x, y=smoothed_y, z=z_value)
            distance = _point_distance(raw_point, smoothed_point)
            displacements_by_joint.setdefault(point.name, []).append(distance)
            frame_distances.append(distance)
            next_points.append(smoothed_point)
            previous_points[smoothed_point.name] = smoothed_point
            previous_raw[(point.name, "x")] = (float(point.x), float(frame.timestamp))
            previous_raw[(point.name, "y")] = (float(point.y), float(frame.timestamp))
        if not next_points:
            next_points = _moving_average_fallback(smoothed, frame, moving_average_window)
            if next_points:
                low_confidence_holds += len(next_points)
        smoothed.append(
            replace(
                frame,
                keypoints=next_points,
                raw_keypoints=list(frame.keypoints),
                confidence=average_keypoint_confidence(next_points),
                debug_info={
                    **frame.debug_info,
                    "smoothing": {
                        "method": "one_euro_confidence_ema",
                        "mean_displacement_px": round(mean(frame_distances), 3) if frame_distances else 0.0,
                        "reset": reset_smoothing,
                    },
                },
            )
        )
        frame_displacements.append(
            {
                "frame_index": frame.frame_index,
                "timestamp": round(float(frame.timestamp), 4),
                "mean_displacement_px": round(mean(frame_distances), 3) if frame_distances else 0.0,
            }
        )
    diagnostics = {
        "enabled": True,
        "method": "one_euro_confidence_ema",
        "parameters": {
            "min_confidence": round(float(min_confidence), 4),
            "min_cutoff": round(float(min_cutoff), 4),
            "beta": round(float(beta), 4),
            "d_cutoff": round(float(d_cutoff), 4),
            "ema_alpha_low": round(float(ema_alpha_low), 4),
            "ema_alpha_high": round(float(ema_alpha_high), 4),
        },
        "processed_frames": len(frames),
        "low_confidence_holds": low_confidence_holds,
        "mean_raw_to_smoothed_px": round(mean([value for values in displacements_by_joint.values() for value in values]), 3)
        if displacements_by_joint
        else 0.0,
        "per_joint_mean_displacement_px": {
            joint: round(mean(values), 3)
            for joint, values in sorted(displacements_by_joint.items())
            if values
        },
        "frame_displacement_timeline": frame_displacements,
    }
    return (smoothed, diagnostics) if return_diagnostics else smoothed


def moving_average(values: list[float], window: int = 3) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return list(values)
    smoothed = []
    radius = max(1, window // 2)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def _moving_average_fallback(previous: list[PoseFrameResult], frame: PoseFrameResult, window: int) -> list[Keypoint]:
    if not previous:
        return []
    recent = [candidate for candidate in previous[-window:] if candidate.keypoints]
    if not recent:
        return []
    by_name: dict[str, list[Keypoint]] = {}
    for candidate in recent:
        for point in candidate.keypoints:
            by_name.setdefault(point.name, []).append(point)
    return [
        Keypoint(
            name=name,
            x=sum(point.x for point in points) / len(points),
            y=sum(point.y for point in points) / len(points),
            z=sum(point.z for point in points if point.z is not None) / len([point for point in points if point.z is not None])
            if any(point.z is not None for point in points)
            else None,
            confidence=min(0.35, sum(point.confidence for point in points) / len(points)),
            visibility_state="interpolated",
            source_backend="smoothing_fallback",
            frame_index=frame.frame_index,
            quality_flags=["moving_average_fallback"],
            interpolated=True,
        )
        for name, points in by_name.items()
    ]


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _confidence_aware_ema(
    *,
    state: dict[tuple[str, str], float],
    key: tuple[str, str],
    value: float,
    confidence: float,
    velocity_hint: float,
    alpha_low: float,
    alpha_high: float,
) -> float:
    confidence = max(0.0, min(1.0, float(confidence)))
    alpha_low = max(0.01, min(0.95, float(alpha_low)))
    alpha_high = max(alpha_low, min(0.98, float(alpha_high)))
    velocity_weight = 0.35 + 0.65 * confidence
    velocity_boost = min(0.38, max(0.0, velocity_hint - 0.4) * 0.12) * velocity_weight
    alpha = min(0.98, alpha_low + (alpha_high - alpha_low) * confidence + velocity_boost)
    previous = state.get(key)
    if previous is None:
        state[key] = float(value)
        return float(value)
    filtered = alpha * float(value) + (1.0 - alpha) * previous
    state[key] = filtered
    return filtered


def _velocity_hint(previous_raw: dict[tuple[str, str], tuple[float, float]], point: Keypoint, timestamp: float) -> float:
    previous = previous_raw.get((point.name, "x"))
    previous_y = previous_raw.get((point.name, "y"))
    if previous is None or previous_y is None:
        return 0.0
    dt = max(1e-3, float(timestamp) - previous[1])
    displacement = ((float(point.x) - previous[0]) ** 2 + (float(point.y) - previous_y[0]) ** 2) ** 0.5
    return min(4.0, displacement / dt / 500.0)


def _temporal_hold(point: Keypoint, previous: Keypoint | None, *, frame_index: int) -> Keypoint:
    if previous is None:
        return point
    return replace(
        point,
        x=previous.x * 0.82 + point.x * 0.18,
        y=previous.y * 0.82 + point.y * 0.18,
        z=previous.z if previous.z is not None else point.z,
        confidence=max(0.01, min(point.confidence, previous.confidence * 0.72)),
        visibility_state="low_confidence",
        source_backend=point.source_backend or previous.source_backend,
        frame_index=frame_index,
        quality_flags=[*point.quality_flags, "confidence_aware_temporal_hold"],
    )


def _point_distance(first: Keypoint, second: Keypoint) -> float:
    return ((float(first.x) - float(second.x)) ** 2 + (float(first.y) - float(second.y)) ** 2) ** 0.5


def _empty_diagnostics(*, enabled: bool, min_confidence: float) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "method": "one_euro_confidence_ema" if enabled else "disabled",
        "parameters": {"min_confidence": round(float(min_confidence), 4)},
        "processed_frames": 0,
        "low_confidence_holds": 0,
        "mean_raw_to_smoothed_px": 0.0,
        "per_joint_mean_displacement_px": {},
        "frame_displacement_timeline": [],
    }


def _should_reset_smoothing(frame: PoseFrameResult) -> bool:
    tracking = frame.debug_info.get("roi_tracking") if isinstance(frame.debug_info, dict) else None
    if isinstance(tracking, dict) and bool(tracking.get("smoothing_reset")):
        return True
    state = str(tracking.get("tracking_state") if isinstance(tracking, dict) else "")
    return state in {"LOST", "REACQUIRE", "REVIEW_ONLY"} or "smoothing_reset" in frame.quality_flags
