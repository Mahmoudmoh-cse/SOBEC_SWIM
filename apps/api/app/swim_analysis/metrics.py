from __future__ import annotations

from math import acos, atan2, degrees
from statistics import mean, median, pstdev
from typing import Any

from app.swim_analysis.pose.base import Keypoint, PoseFrameResult
from app.swim_analysis.quality import confidence_payload, confidence_value, data_quality_score, metric_confidence, visible_ratio


def compute_swimming_metrics(
    side_frames: list[PoseFrameResult],
    front_frames: list[PoseFrameResult],
    *,
    stroke_type: str,
    video_quality: dict[str, Any],
    sync_offset_sec: float = 0.0,
    calibration: dict[str, Any] | None = None,
    velocity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    side = compute_side_view_metrics(side_frames, stroke_type=stroke_type)
    front = compute_front_view_metrics(front_frames)
    combined = compute_combined_metrics(side, front, video_quality=video_quality, sync_offset_sec=sync_offset_sec)
    metrics = {**side, **front, **combined}
    metrics.update(compute_biomechanics_metrics(side_frames, front_frames, metrics, calibration=calibration, velocity=velocity))
    return normalize_metric_schema(metrics)


def compute_side_view_metrics(frames: list[PoseFrameResult], *, stroke_type: str = "freestyle") -> dict[str, Any]:
    duration = _duration(frames)
    wrist_peaks = _stroke_peaks(frames)
    wrist_samples = _keypoint_sample_count(frames, {"left_wrist", "right_wrist"})
    stroke_count = len(wrist_peaks) if wrist_samples else None
    stroke_rate = stroke_count / duration * 60.0 if isinstance(stroke_count, int) and duration > 0 else None
    cycle_intervals = [current - previous for previous, current in zip(wrist_peaks, wrist_peaks[1:])]
    cycle_consistency = _consistency_score(cycle_intervals) if len(cycle_intervals) >= 2 else None
    wrist_conf = metric_confidence(frames, {"left_wrist", "right_wrist"}, extra=min(1.0, stroke_count / 4 if stroke_count else 0.6))
    duration_evidence = {"duration_sec": round(duration, 2), "stroke_peak_times_sec": [round(value, 2) for value in wrist_peaks[:24]]}

    alignment = _body_alignment(frames)
    head = _head_stability(frames)
    hip_drop = _hip_drop(frames)
    breathing = _breathing_events(frames)
    kick = _kick_rhythm(frames)
    streamline_inputs = [
        (alignment["score"], confidence_value(alignment["confidence"])),
        (head["score"], confidence_value(head["confidence"])),
        (kick["score"], confidence_value(kick["confidence"]) * 0.5),
    ]
    streamline_value = _weighted_score(streamline_inputs) if _has_numeric_input(streamline_inputs) else None
    streamline_conf_values = [
        confidence
        for value, confidence in streamline_inputs[:2]
        if isinstance(value, (int, float)) and confidence > 0
    ]
    streamline_conf = mean(streamline_conf_values) if streamline_conf_values else 0.0

    return {
        "stroke_count": _metric(
            value=stroke_count,
            confidence=wrist_conf,
            evidence=duration_evidence,
            reason=(
                f"Wrist motion peaks detected from the {stroke_type} side-view timeline."
                if stroke_count is not None
                else "Wrist landmarks were not available, so stroke count was not estimated."
            ),
        ),
        "stroke_rate_spm": _metric(
            value=round(stroke_rate, 1) if stroke_rate is not None else None,
            confidence=wrist_conf,
            evidence=duration_evidence,
            reason=(
                f"{stroke_count} explainable wrist-motion peaks over {duration:.1f} seconds."
                if stroke_count is not None
                else "Wrist landmarks were not available, so stroke rate was not estimated."
            ),
        ),
        "stroke_cycle_consistency": _metric(
            value=cycle_consistency,
            confidence=wrist_conf * min(1.0, len(cycle_intervals) / 4 if cycle_intervals else 0.5),
            evidence={"cycle_intervals_sec": [round(value, 2) for value in cycle_intervals[:24]]},
            reason="Computed from variation between consecutive wrist-motion stroke peaks.",
        ),
        "body_alignment_score": _metric(
            value=alignment["score"],
            confidence=confidence_value(alignment["confidence"]),
            evidence=alignment["evidence"],
            reason="Shoulder, hip, and ankle vertical spread stayed closer to one line in higher-scoring frames.",
        ),
        "hip_drop_indicator": _metric(
            value=hip_drop["indicator"],
            confidence=confidence_value(hip_drop["confidence"]),
            evidence=hip_drop["evidence"],
            reason="Hip position was compared with the shoulder line, normalized by torso length.",
        ),
        "head_stability_score": _metric(
            value=head["score"],
            confidence=confidence_value(head["confidence"]),
            evidence=head["evidence"],
            reason="Nose movement was measured relative to the shoulder center across usable frames.",
        ),
        "breathing_event_count": _metric(
            value=breathing["count"],
            confidence=confidence_value(breathing["confidence"]),
            evidence=breathing["evidence"],
            reason="Breathing events are inferred from repeated head deviations relative to the shoulder line.",
        ),
        "breathing_side_estimate": _metric(
            value=breathing["side"],
            confidence=confidence_value(breathing["confidence"]),
            evidence=breathing["evidence"],
            reason="Estimated from the side of head displacement during inferred breathing events.",
        ),
        "kick_rhythm_score": _metric(
            value=kick["score"],
            confidence=confidence_value(kick["confidence"]),
            evidence=kick["evidence"],
            reason="Ankle motion peak timing is more consistent when the score is higher.",
        ),
        "streamline_score": _metric(
            value=round(streamline_value) if streamline_value is not None else None,
            confidence=streamline_conf,
            evidence={"body_alignment_score": alignment["score"], "head_stability_score": head["score"], "kick_rhythm_score": kick["score"]},
            reason="Composite of body alignment and head stability, with kick rhythm as supporting evidence.",
        ),
    }


def compute_front_view_metrics(frames: list[PoseFrameResult]) -> dict[str, Any]:
    symmetry = _arm_symmetry(frames)
    shoulder = _shoulder_balance(frames)
    centerline = _centerline_deviation(frames)
    timing = _left_right_timing(frames)
    hand_width = _hand_entry_width(frames)

    return {
        "arm_symmetry_score": _metric(
            value=symmetry["score"],
            confidence=confidence_value(symmetry["confidence"]),
            evidence=symmetry["evidence"],
            reason="Left and right wrist motion amplitudes and visibility were compared over time.",
        ),
        "shoulder_balance_score": _metric(
            value=shoulder["score"],
            confidence=confidence_value(shoulder["confidence"]),
            evidence=shoulder["evidence"],
            reason="Left and right shoulder height difference was normalized by shoulder width.",
        ),
        "centerline_deviation_score": _metric(
            value=centerline["score"],
            confidence=confidence_value(centerline["confidence"]),
            evidence=centerline["evidence"],
            reason="Head and hip center drift from the shoulder centerline was measured in visible frames.",
        ),
        "left_right_timing_difference": _metric(
            value=timing["difference_sec"],
            confidence=confidence_value(timing["confidence"]),
            evidence=timing["evidence"],
            reason="Timing uses nearest left-wrist and right-wrist motion peaks.",
        ),
        "hand_entry_width_estimate": _metric(
            value=hand_width["width_ratio"],
            confidence=confidence_value(hand_width["confidence"]),
            evidence=hand_width["evidence"],
            reason="Wrist separation was divided by shoulder width when both hands and shoulders were visible.",
        ),
    }


def compute_combined_metrics(
    side: dict[str, Any],
    front: dict[str, Any],
    *,
    video_quality: dict[str, Any],
    sync_offset_sec: float = 0.0,
) -> dict[str, Any]:
    data_quality = data_quality_score(video_quality)
    technique_inputs = [
        side["body_alignment_score"],
        side["head_stability_score"],
        side["stroke_cycle_consistency"],
        side["streamline_score"],
        front["arm_symmetry_score"],
        front["shoulder_balance_score"],
        front["centerline_deviation_score"],
    ]
    scored_inputs = [(float(metric["value"]), confidence_value(metric["confidence"])) for metric in technique_inputs if isinstance(metric["value"], (int, float))]
    overall = _weighted_score(scored_inputs)
    confidence_values = [confidence_value(metric["confidence"]) for metric in technique_inputs]
    confidence_score = round((mean(confidence_values) if confidence_values else 0.0) * 0.75 + data_quality * 0.25, 3)
    view_agreement = _view_agreement(side, front, sync_offset_sec)
    return {
        "overall_technique_score": _metric(
            value=round(overall),
            confidence=confidence_score,
            evidence={"component_count": len(scored_inputs), "data_quality_score": data_quality},
            reason="Weighted score from explainable side-view and front/back-view landmark metrics.",
        ),
        "confidence_score": _metric(
            value=confidence_score,
            confidence=confidence_score,
            evidence={"metric_confidences": [round(value, 3) for value in confidence_values]},
            reason="Aggregate of pose visibility, metric support, and video quality.",
        ),
        "data_quality_score": _metric(
            value=data_quality,
            confidence=0.95,
            evidence=video_quality,
            reason="Derived from blur, lighting, stability, and usable-frame ratio for both videos.",
        ),
        "view_agreement_score": _metric(
            value=view_agreement["score"],
            confidence=view_agreement["confidence"],
            evidence=view_agreement["evidence"],
            reason="Compares whether the two camera views tell a consistent timing and quality story.",
        ),
    }


def detect_stroke_cycles(frames: list[PoseFrameResult], keypoint_name: str = "left_wrist") -> list[float]:
    series = _series(frames, keypoint_name, axis="y")
    return _peak_times(series)


def _metric(*, value: Any, confidence: float, evidence: dict[str, Any], reason: str) -> dict[str, Any]:
    evidence = dict(evidence)
    payload_confidence = confidence_payload(confidence, evidence)
    return {
        "value": value,
        "confidence": payload_confidence,
        "evidence": evidence,
        "reason": reason,
        "supporting_frame_count": _supporting_frame_count(evidence),
        "reliability_category": _metric_reliability_category(value, confidence),
    }


def _supporting_frame_count(evidence: dict[str, Any]) -> int:
    candidates: list[int] = []
    for key, value in evidence.items():
        if key in {"visible_frames", "supporting_frames", "video_frames", "side_frames", "front_frames"} and isinstance(value, (int, float)):
            candidates.append(int(value))
        elif key == "frames" and isinstance(value, list):
            candidates.append(len(value))
        elif isinstance(value, dict):
            nested = _supporting_frame_count(value)
            if nested:
                candidates.append(nested)
    return max(candidates) if candidates else 0


def _metric_reliability_category(value: Any, confidence: float) -> str:
    if value is None or confidence < 0.35:
        return "insufficient_evidence"
    if confidence < 0.6:
        return "estimated"
    return "reliable"


def _stroke_peaks(frames: list[PoseFrameResult]) -> list[float]:
    left = _peak_times(_series(frames, "left_wrist", axis="y"))
    right = _peak_times(_series(frames, "right_wrist", axis="y"))
    merged = sorted([*left, *right])
    if not merged:
        return []
    deduped = [merged[0]]
    for value in merged[1:]:
        if value - deduped[-1] >= 0.18:
            deduped.append(value)
    return deduped


def _body_alignment(frames: list[PoseFrameResult]) -> dict[str, Any]:
    spreads = []
    frame_indices = []
    for frame in frames:
        shoulder = _midpoint(frame.keypoint("left_shoulder", 0.25), frame.keypoint("right_shoulder", 0.25))
        hip = _midpoint(frame.keypoint("left_hip", 0.25), frame.keypoint("right_hip", 0.25))
        ankle = _midpoint(frame.keypoint("left_ankle", 0.2), frame.keypoint("right_ankle", 0.2))
        if not shoulder or not hip or not ankle:
            continue
        torso = _distance(shoulder, hip) or 1.0
        spread = (abs(shoulder.y - hip.y) + abs(hip.y - ankle.y)) / (2 * torso)
        spreads.append(spread)
        frame_indices.append(frame.frame_index)
    score = None if not spreads else round(max(0.0, min(100.0, 100 - median(spreads) * 65)))
    confidence = metric_confidence(frames, {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}, extra=min(1.0, len(spreads) / max(1, len(frames) * 0.45)))
    return {
        "score": score,
        "confidence": confidence,
        "evidence": {"median_alignment_spread": round(median(spreads), 3) if spreads else None, "visible_frames": len(spreads), "frames": frame_indices[:12]},
    }


def _hip_drop(frames: list[PoseFrameResult]) -> dict[str, Any]:
    drops = []
    frames_seen = []
    for frame in frames:
        shoulder = _midpoint(frame.keypoint("left_shoulder", 0.25), frame.keypoint("right_shoulder", 0.25))
        hip = _midpoint(frame.keypoint("left_hip", 0.25), frame.keypoint("right_hip", 0.25))
        if not shoulder or not hip:
            continue
        torso = _distance(shoulder, hip) or 1.0
        drops.append((hip.y - shoulder.y) / torso)
        frames_seen.append(frame.frame_index)
    value = median(drops) if drops else None
    indicator = "unknown" if value is None else "high" if value > 0.95 else "medium" if value > 0.72 else "low"
    confidence = metric_confidence(frames, {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}, extra=min(1.0, len(drops) / max(1, len(frames) * 0.45)))
    return {
        "indicator": indicator,
        "confidence": confidence,
        "evidence": {"median_hip_drop_ratio": round(value, 3) if drops else None, "frames": frames_seen[:12]},
    }


def _head_stability(frames: list[PoseFrameResult]) -> dict[str, Any]:
    offsets = []
    visible_frames = []
    for frame in frames:
        nose = frame.keypoint("nose", 0.2)
        shoulder = _midpoint(frame.keypoint("left_shoulder", 0.25), frame.keypoint("right_shoulder", 0.25))
        if nose is None or shoulder is None:
            continue
        shoulder_width = _distance(frame.keypoint("left_shoulder", 0.25), frame.keypoint("right_shoulder", 0.25)) or 1.0
        offsets.append(abs(nose.y - shoulder.y) / shoulder_width)
        visible_frames.append(frame.frame_index)
    if not offsets:
        return {"score": None, "confidence": 0.0, "evidence": {"head_offset_std": None, "visible_frames": 0, "frames": []}}
    variability = pstdev(offsets) if len(offsets) > 1 else 0.0
    score = round(max(0.0, min(100.0, 100 - variability * 220)))
    confidence = metric_confidence(frames, {"nose", "left_shoulder", "right_shoulder"}, extra=min(1.0, len(offsets) / max(1, len(frames) * 0.45)))
    return {"score": score, "confidence": confidence, "evidence": {"head_offset_std": round(variability, 3), "visible_frames": len(offsets), "frames": visible_frames[:12]}}


def _breathing_events(frames: list[PoseFrameResult]) -> dict[str, Any]:
    series = []
    sides = []
    samples = 0
    for frame in frames:
        nose = frame.keypoint("nose", 0.2)
        shoulder = _midpoint(frame.keypoint("left_shoulder", 0.25), frame.keypoint("right_shoulder", 0.25))
        left = frame.keypoint("left_shoulder", 0.25)
        right = frame.keypoint("right_shoulder", 0.25)
        if not nose or not shoulder or not left or not right:
            continue
        samples += 1
        shoulder_width = _distance(left, right) or 1.0
        lateral = (nose.x - shoulder.x) / shoulder_width
        vertical = (nose.y - shoulder.y) / shoulder_width
        if abs(lateral) > 0.18 or vertical < -0.14:
            series.append(frame.timestamp)
            sides.append("left" if lateral < 0 else "right")
    events = _dedupe_times(series, min_distance=0.7)
    if sides:
        left_count = sides.count("left")
        right_count = sides.count("right")
        side = "mixed" if left_count and right_count and abs(left_count - right_count) <= 1 else "left" if left_count > right_count else "right"
    else:
        side = "unknown"
    confidence = metric_confidence(frames, {"nose", "left_shoulder", "right_shoulder"}, extra=min(1.0, len(events) / 3 if events else 0.5))
    return {
        "count": len(events) if samples else None,
        "side": side,
        "confidence": confidence if samples else 0.0,
        "evidence": {"event_times_sec": [round(value, 2) for value in events[:16]], "side_votes": sides[:16], "visible_frames": samples},
    }


def _kick_rhythm(frames: list[PoseFrameResult]) -> dict[str, Any]:
    left = _peak_times(_series(frames, "left_ankle", axis="y"))
    right = _peak_times(_series(frames, "right_ankle", axis="y"))
    peaks = sorted([*left, *right])
    intervals = [b - a for a, b in zip(peaks, peaks[1:])]
    score = _consistency_score(intervals) if peaks else None
    confidence = metric_confidence(frames, {"left_ankle", "right_ankle"}, extra=min(1.0, len(intervals) / 6 if intervals else 0.45))
    return {"score": score, "confidence": confidence, "evidence": {"kick_peak_times_sec": [round(value, 2) for value in peaks[:24]], "intervals_sec": [round(value, 2) for value in intervals[:24]]}}


def _arm_symmetry(frames: list[PoseFrameResult]) -> dict[str, Any]:
    left = [value for _, value in _series(frames, "left_wrist", axis="x")]
    right = [value for _, value in _series(frames, "right_wrist", axis="x")]
    if len(left) < 3 or len(right) < 3:
        score = None
        ratio = None
    else:
        left_amp = max(left) - min(left)
        right_amp = max(right) - min(right)
        ratio = min(left_amp, right_amp) / max(left_amp, right_amp, 1.0)
        score = round(max(0.0, min(100.0, ratio * 100)))
    confidence = metric_confidence(frames, {"left_wrist", "right_wrist"}, extra=min(1.0, min(len(left), len(right)) / max(1, len(frames) * 0.45)))
    return {"score": score, "confidence": confidence, "evidence": {"left_right_amplitude_ratio": round(ratio, 3) if ratio is not None else None, "visible_left": len(left), "visible_right": len(right)}}


def _shoulder_balance(frames: list[PoseFrameResult]) -> dict[str, Any]:
    ratios = []
    for frame in frames:
        left = frame.keypoint("left_shoulder", 0.25)
        right = frame.keypoint("right_shoulder", 0.25)
        width = _distance(left, right)
        if left and right and width:
            ratios.append(abs(left.y - right.y) / width)
    score = None if not ratios else round(max(0.0, min(100.0, 100 - median(ratios) * 180)))
    confidence = metric_confidence(frames, {"left_shoulder", "right_shoulder"}, extra=min(1.0, len(ratios) / max(1, len(frames) * 0.45)))
    return {"score": score, "confidence": confidence, "evidence": {"median_shoulder_tilt_ratio": round(median(ratios), 3) if ratios else None, "visible_frames": len(ratios)}}


def _centerline_deviation(frames: list[PoseFrameResult]) -> dict[str, Any]:
    deviations = []
    for frame in frames:
        shoulder = _midpoint(frame.keypoint("left_shoulder", 0.25), frame.keypoint("right_shoulder", 0.25))
        hip = _midpoint(frame.keypoint("left_hip", 0.25), frame.keypoint("right_hip", 0.25))
        nose = frame.keypoint("nose", 0.2)
        left = frame.keypoint("left_shoulder", 0.25)
        right = frame.keypoint("right_shoulder", 0.25)
        width = _distance(left, right)
        if not shoulder or not width:
            continue
        values = []
        if hip:
            values.append(abs(hip.x - shoulder.x) / width)
        if nose:
            values.append(abs(nose.x - shoulder.x) / width)
        if values:
            deviations.append(mean(values))
    score = None if not deviations else round(max(0.0, min(100.0, 100 - median(deviations) * 140)))
    confidence = metric_confidence(frames, {"left_shoulder", "right_shoulder"}, extra=min(1.0, len(deviations) / max(1, len(frames) * 0.45)))
    return {"score": score, "confidence": confidence, "evidence": {"median_centerline_deviation": round(median(deviations), 3) if deviations else None, "visible_frames": len(deviations)}}


def _left_right_timing(frames: list[PoseFrameResult]) -> dict[str, Any]:
    left_peaks = _peak_times(_series(frames, "left_wrist", axis="y"))
    right_peaks = _peak_times(_series(frames, "right_wrist", axis="y"))
    differences = []
    for left in left_peaks:
        if right_peaks:
            differences.append(min(abs(left - right) for right in right_peaks))
    diff = median(differences) if differences else None
    confidence = metric_confidence(frames, {"left_wrist", "right_wrist"}, extra=min(1.0, len(differences) / 3 if differences else 0.45))
    return {
        "difference_sec": round(diff, 3) if diff is not None else None,
        "confidence": confidence if diff is not None else 0.0,
        "evidence": {"left_peaks_sec": [round(value, 2) for value in left_peaks[:12]], "right_peaks_sec": [round(value, 2) for value in right_peaks[:12]]},
    }


def _hand_entry_width(frames: list[PoseFrameResult]) -> dict[str, Any]:
    ratios = []
    for frame in frames:
        left_wrist = frame.keypoint("left_wrist", 0.25)
        right_wrist = frame.keypoint("right_wrist", 0.25)
        left_shoulder = frame.keypoint("left_shoulder", 0.25)
        right_shoulder = frame.keypoint("right_shoulder", 0.25)
        shoulder_width = _distance(left_shoulder, right_shoulder)
        if left_wrist and right_wrist and shoulder_width:
            ratios.append(abs(left_wrist.x - right_wrist.x) / shoulder_width)
    confidence = metric_confidence(frames, {"left_wrist", "right_wrist", "left_shoulder", "right_shoulder"}, extra=min(1.0, len(ratios) / max(1, len(frames) * 0.35)))
    return {"width_ratio": round(median(ratios), 2) if ratios else None, "confidence": confidence, "evidence": {"median_wrist_to_shoulder_width_ratio": round(median(ratios), 3) if ratios else None, "visible_frames": len(ratios)}}


def _view_agreement(side: dict[str, Any], front: dict[str, Any], sync_offset_sec: float) -> dict[str, Any]:
    side_conf = confidence_value(side["stroke_rate_spm"]["confidence"])
    front_conf = confidence_value(front["arm_symmetry_score"]["confidence"])
    sync_penalty = min(0.35, abs(sync_offset_sec) / 6.0)
    score = round(max(0.0, min(100.0, (side_conf * 0.5 + front_conf * 0.5 - sync_penalty) * 100)))
    return {"score": score, "confidence": round((side_conf + front_conf) / 2, 3), "evidence": {"side_confidence": round(side_conf, 3), "front_confidence": round(front_conf, 3), "sync_offset_sec": sync_offset_sec}}


def _series(frames: list[PoseFrameResult], keypoint_name: str, *, axis: str) -> list[tuple[float, float]]:
    values = []
    for frame in frames:
        point = frame.keypoint(keypoint_name, 0.2)
        if point is None:
            continue
        values.append((frame.timestamp, point.x if axis == "x" else point.y))
    return values


def _peak_times(series: list[tuple[float, float]]) -> list[float]:
    if len(series) < 5:
        return []
    values = [value for _, value in series]
    span = max(values) - min(values)
    if span < 4:
        return []
    threshold = min(values) + span * 0.58
    peaks = []
    for index in range(1, len(series) - 1):
        previous = values[index - 1]
        current = values[index]
        following = values[index + 1]
        if current >= previous and current > following and current >= threshold:
            peaks.append(series[index][0])
    return _dedupe_times(peaks, min_distance=0.28)


def _dedupe_times(times: list[float], *, min_distance: float) -> list[float]:
    if not times:
        return []
    output = [times[0]]
    for value in times[1:]:
        if value - output[-1] >= min_distance:
            output.append(value)
    return output


def _consistency_score(intervals: list[float]) -> int:
    if len(intervals) < 2:
        return 0
    avg = mean(intervals)
    if avg <= 0:
        return 0
    coefficient = pstdev(intervals) / avg
    return round(max(0.0, min(100.0, 100 - coefficient * 140)))


def _duration(frames: list[PoseFrameResult]) -> float:
    if len(frames) < 2:
        return 0.0
    return max(0.0, frames[-1].timestamp - frames[0].timestamp)


def _keypoint_sample_count(frames: list[PoseFrameResult], names: set[str], *, min_confidence: float = 0.2) -> int:
    return sum(1 for frame in frames for point in frame.keypoints if point.name in names and point.confidence >= min_confidence)


def _has_numeric_input(values: list[tuple[Any, float]]) -> bool:
    return any(isinstance(value, (int, float)) and confidence > 0 for value, confidence in values)


def _weighted_score(values: list[tuple[Any, float]]) -> float:
    weighted = [
        (float(value), max(0.0, min(1.0, confidence)))
        for value, confidence in values
        if isinstance(value, (int, float)) and confidence > 0
    ]
    if not weighted:
        return 0.0
    denom = sum(confidence for _, confidence in weighted)
    if denom == 0:
        return 0.0
    return sum(value * confidence for value, confidence in weighted) / denom


def _midpoint(a: Keypoint | None, b: Keypoint | None) -> Keypoint | None:
    if a is None or b is None:
        return None
    return Keypoint(name="midpoint", x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0, z=None, confidence=min(a.confidence, b.confidence))


def _distance(a: Keypoint | None, b: Keypoint | None) -> float | None:
    if a is None or b is None:
        return None
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def body_line_angle(frame: PoseFrameResult) -> float | None:
    shoulder = _midpoint(frame.keypoint("left_shoulder", 0.25), frame.keypoint("right_shoulder", 0.25))
    hip = _midpoint(frame.keypoint("left_hip", 0.25), frame.keypoint("right_hip", 0.25))
    if not shoulder or not hip:
        return None
    return degrees(atan2(hip.y - shoulder.y, hip.x - shoulder.x))


def compute_biomechanics_metrics(
    side_frames: list[PoseFrameResult],
    front_frames: list[PoseFrameResult],
    existing: dict[str, Any],
    *,
    calibration: dict[str, Any] | None = None,
    velocity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stroke_count = existing.get("stroke_count", {}).get("value", 0)
    landmark_samples = _keypoint_sample_count(side_frames + front_frames, {"left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_hip", "right_hip"})
    total_distance_m = _total_distance_m(velocity)
    calibration_conf = float((calibration or {}).get("confidence", 0) or 0)
    stroke_conf = confidence_value(existing.get("stroke_count", {}).get("confidence", 0))
    distance_conf = min(calibration_conf, stroke_conf)
    dps = total_distance_m / stroke_count if total_distance_m and isinstance(stroke_count, int) and stroke_count > 0 else None
    cycles = max(1, int(stroke_count / 2)) if isinstance(stroke_count, int) and stroke_count else 0
    dpc = total_distance_m / cycles if total_distance_m and cycles else None

    elbow = _elbow_biomechanics(side_frames)
    shoulder_roll = _body_roll(side_frames)
    kick = _kick_biomechanics(side_frames)
    breathing_timing = _breathing_timing(side_frames)
    streamline = _numeric_metric(existing, "streamline_score")
    body_alignment = _numeric_metric(existing, "body_alignment_score")
    arm_symmetry = _numeric_metric(existing, "arm_symmetry_score")
    asymmetry = 100 - arm_symmetry if arm_symmetry is not None else None
    stroke_consistency = _numeric_metric(existing, "stroke_cycle_consistency")
    velocity_conf = float((velocity or {}).get("summary", {}).get("confidence", 0) or 0)
    avg_velocity = (velocity or {}).get("summary", {}).get("average_velocity")

    propulsion_inputs: list[tuple[Any, float]] = [
        (min(100, dps * 45), distance_conf) if dps is not None else (None, 0.0),
        (stroke_consistency, confidence_value(existing.get("stroke_cycle_consistency", {}).get("confidence", 0))),
        (min(100, float(avg_velocity) * 55), velocity_conf) if isinstance(avg_velocity, (int, float)) and velocity_conf >= 0.45 else (None, 0.0),
    ]
    propulsion = _weighted_score(propulsion_inputs) if _has_numeric_input(propulsion_inputs) and landmark_samples else None
    propulsion_conf_values = [confidence for value, confidence in propulsion_inputs if isinstance(value, (int, float)) and confidence > 0]
    propulsion_conf = mean(propulsion_conf_values) if propulsion is not None and propulsion_conf_values else 0.0

    drag_inputs = [
        (streamline, confidence_value(existing.get("streamline_score", {}).get("confidence", 0))),
        (body_alignment, confidence_value(existing.get("body_alignment_score", {}).get("confidence", 0))),
    ]
    drag_risk = max(0, min(100, 100 - _weighted_score(drag_inputs))) if _has_numeric_input(drag_inputs) and landmark_samples else None
    drag_conf_values = [confidence for value, confidence in drag_inputs if isinstance(value, (int, float)) and confidence > 0]
    drag_conf = mean(drag_conf_values) if drag_risk is not None and drag_conf_values else 0.0

    return {
        "stroke_length": _metric(
            value=round(dps, 3) if dps is not None else None,
            confidence=distance_conf,
            evidence={"total_distance_m": total_distance_m, "stroke_count": stroke_count, "calibration": calibration or {}},
            reason="Estimated total calibrated swimmer travel divided by detected stroke count.",
        ),
        "distance_per_stroke": _metric(
            value=round(dps, 3) if dps is not None else None,
            confidence=distance_conf,
            evidence={"total_distance_m": total_distance_m, "stroke_count": stroke_count, "calibration": calibration or {}},
            reason="Same as stroke length; included for coaching terminology consistency.",
        ),
        "distance_per_cycle": _metric(
            value=round(dpc, 3) if dpc is not None else None,
            confidence=distance_conf,
            evidence={"total_distance_m": total_distance_m, "cycle_count": cycles, "calibration": calibration or {}},
            reason="Estimated calibrated travel divided by full left/right stroke cycles.",
        ),
        "catch_angle": _metric(
            value=elbow["catch_angle"],
            confidence=elbow["confidence"],
            evidence=elbow["evidence"],
            reason="Phase 1 proxy from shoulder-elbow-wrist angle during visible catch-like arm positions.",
        ),
        "elbow_angle": _metric(
            value=elbow["elbow_angle"],
            confidence=elbow["confidence"],
            evidence=elbow["evidence"],
            reason="Average visible elbow angle from shoulder-elbow-wrist landmarks.",
        ),
        "shoulder_rotation_body_roll": _metric(
            value=shoulder_roll["value"],
            confidence=shoulder_roll["confidence"],
            evidence=shoulder_roll["evidence"],
            reason="Approximate body roll from shoulder-to-hip line angle variation in the side view.",
        ),
        "hip_alignment": _metric(
            value=body_alignment,
            confidence=confidence_value(existing.get("body_alignment_score", {}).get("confidence", 0)),
            evidence=existing.get("body_alignment_score", {}).get("evidence", {}),
            reason="Mapped from body alignment because hip line stability is a main contributor.",
        ),
        "kick_frequency": _metric(
            value=kick["frequency"],
            confidence=kick["confidence"],
            evidence=kick["evidence"],
            reason="Ankle motion peaks per minute from visible side-view ankle trajectories.",
        ),
        "kick_symmetry": _metric(
            value=kick["symmetry"],
            confidence=kick["confidence"],
            evidence=kick["evidence"],
            reason="Left/right ankle motion amplitude agreement.",
        ),
        "breathing_timing": _metric(
            value=breathing_timing["value"],
            confidence=breathing_timing["confidence"],
            evidence=breathing_timing["evidence"],
            reason="Breath-like head deviations are compared with nearby stroke-cycle peaks.",
        ),
        "left_right_asymmetry_score": _metric(
            value=round(asymmetry) if asymmetry is not None else None,
            confidence=confidence_value(existing.get("arm_symmetry_score", {}).get("confidence", 0)),
            evidence={"arm_symmetry_score": existing.get("arm_symmetry_score", {})},
            reason="Inverse of arm symmetry score: higher means more asymmetry risk.",
        ),
        "propulsion_efficiency_score": _metric(
            value=round(propulsion) if propulsion is not None else None,
            confidence=propulsion_conf,
            evidence={"distance_per_stroke": dps, "stroke_cycle_consistency": stroke_consistency, "average_velocity": avg_velocity, "landmark_samples": landmark_samples},
            reason="Composite of calibrated distance per stroke, rhythm consistency, and speed trend.",
        ),
        "drag_risk_score": _metric(
            value=round(drag_risk) if drag_risk is not None else None,
            confidence=drag_conf,
            evidence={"streamline_score": streamline, "body_alignment_score": body_alignment, "landmark_samples": landmark_samples},
            reason="Higher score means greater drag risk from body-line and streamline evidence.",
        ),
    }


def normalize_metric_schema(metrics: dict[str, Any]) -> dict[str, Any]:
    for name, metric in metrics.items():
        if not isinstance(metric, dict):
            continue
        metric.setdefault("unit", _unit_for_metric(name, metric.get("value")))
        numeric_conf = confidence_value(metric.get("confidence", 0))
        metric.setdefault("interpretation", _interpret_metric(name, metric.get("value"), numeric_conf))
        metric.setdefault("coaching_meaning", _coaching_meaning(name, metric.get("value"), numeric_conf))
    return metrics


def _elbow_biomechanics(frames: list[PoseFrameResult]) -> dict[str, Any]:
    angles = []
    frame_indices = []
    for frame in frames:
        left = _joint_angle(frame, "left_shoulder", "left_elbow", "left_wrist")
        right = _joint_angle(frame, "right_shoulder", "right_elbow", "right_wrist")
        for value in [left, right]:
            if value is not None:
                angles.append(value)
                frame_indices.append(frame.frame_index)
    confidence = metric_confidence(frames, {"left_shoulder", "left_elbow", "left_wrist"}, extra=0.85)
    avg = mean(angles) if angles else None
    return {
        "catch_angle": round(avg, 1) if avg is not None else None,
        "elbow_angle": round(avg, 1) if avg is not None else None,
        "confidence": confidence,
        "evidence": {"sample_count": len(angles), "frames": frame_indices[:12]},
    }


def _body_roll(frames: list[PoseFrameResult]) -> dict[str, Any]:
    angles = [body_line_angle(frame) for frame in frames]
    angles = [value for value in angles if value is not None]
    if not angles:
        return {"value": None, "confidence": 0.0, "evidence": {"sample_count": 0}}
    roll_range = max(angles) - min(angles)
    confidence = metric_confidence(frames, {"left_shoulder", "right_shoulder", "left_hip", "right_hip"})
    return {"value": round(abs(roll_range), 1), "confidence": confidence, "evidence": {"angle_range_deg": round(roll_range, 2), "sample_count": len(angles)}}


def _kick_biomechanics(frames: list[PoseFrameResult]) -> dict[str, Any]:
    left_series = _series(frames, "left_ankle", axis="y")
    right_series = _series(frames, "right_ankle", axis="y")
    left_peaks = _peak_times(left_series)
    right_peaks = _peak_times(right_series)
    duration = _duration(frames)
    total_peaks = len(left_peaks) + len(right_peaks)
    frequency = total_peaks / duration * 60 if total_peaks > 0 and duration > 0 else None
    left_values = [value for _, value in left_series]
    right_values = [value for _, value in right_series]
    if left_values and right_values:
        left_amp = max(left_values) - min(left_values)
        right_amp = max(right_values) - min(right_values)
        symmetry = min(left_amp, right_amp) / max(left_amp, right_amp, 1.0) * 100
    else:
        symmetry = None
    confidence = metric_confidence(frames, {"left_ankle", "right_ankle"}, extra=min(1.0, (len(left_peaks) + len(right_peaks)) / 6 if duration else 0.5))
    return {
        "frequency": round(frequency, 1) if frequency is not None else None,
        "symmetry": round(symmetry) if symmetry is not None else None,
        "confidence": confidence,
        "evidence": {"left_kick_peaks": left_peaks[:12], "right_kick_peaks": right_peaks[:12]},
    }


def _breathing_timing(frames: list[PoseFrameResult]) -> dict[str, Any]:
    breaths = _breathing_events(frames)
    stroke_times = _stroke_peaks(frames)
    event_times = breaths["evidence"].get("event_times_sec", [])
    if not event_times or not stroke_times:
        return {"value": "unknown", "confidence": 0.0, "evidence": {"event_times_sec": event_times, "stroke_peak_times_sec": stroke_times[:12]}}
    offsets = []
    for event in event_times:
        nearest = min(stroke_times, key=lambda value: abs(value - event))
        offsets.append(event - nearest)
    late_ratio = sum(1 for value in offsets if value > 0.22) / len(offsets)
    value = "late" if late_ratio >= 0.5 else "on_time"
    confidence = confidence_value(breaths["confidence"]) * min(1.0, len(offsets) / 3)
    return {
        "value": value,
        "confidence": confidence,
        "evidence": {
            "event_times_sec": event_times,
            "stroke_peak_times_sec": [round(value, 2) for value in stroke_times[:12]],
            "offsets_sec": [round(value, 3) for value in offsets[:12]],
            "time_range_sec": [min(event_times), max(event_times)],
        },
    }


def _joint_angle(frame: PoseFrameResult, a: str, b: str, c: str) -> float | None:
    first = frame.keypoint(a, 0.2)
    middle = frame.keypoint(b, 0.2)
    last = frame.keypoint(c, 0.2)
    if first is None or middle is None or last is None:
        return None
    return _angle((first.x, first.y), (middle.x, middle.y), (last.x, last.y))


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denom = ((ba[0] ** 2 + ba[1] ** 2) ** 0.5) * ((bc[0] ** 2 + bc[1] ** 2) ** 0.5)
    if denom == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / denom))
    return degrees(acos(cosine))


def _total_distance_m(velocity: dict[str, Any] | None) -> float | None:
    if not velocity:
        return None
    distances = [point.get("distance_m") for point in velocity.get("series", []) if point.get("distance_m") is not None]
    if not distances:
        return None
    return max(distances) - min(distances)


def _numeric_metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name, {}).get("value")
    return float(value) if isinstance(value, (int, float)) else None


def _unit_for_metric(name: str, value: Any) -> str | None:
    if name in {"stroke_rate_spm", "kick_frequency"}:
        return "cycles/min"
    if name in {"stroke_length", "distance_per_stroke", "distance_per_cycle"}:
        return "m"
    if name in {"catch_angle", "elbow_angle", "shoulder_rotation_body_roll"}:
        return "deg"
    if name.endswith("_score") or name in {"body_alignment_score", "streamline_score", "kick_symmetry", "hip_alignment"}:
        return "score_0_100"
    if name in {"breathing_event_count", "stroke_count"}:
        return "count"
    if name == "left_right_timing_difference":
        return "sec"
    if isinstance(value, str):
        return "category"
    return None


def _interpret_metric(name: str, value: Any, confidence: float) -> str:
    if confidence < 0.35:
        return "low_confidence"
    if not isinstance(value, (int, float)):
        return str(value) if value is not None else "unavailable"
    if name in {"drag_risk_score", "left_right_asymmetry_score"}:
        return "high" if value >= 55 else "normal" if value >= 30 else "low"
    if name in {"catch_angle", "elbow_angle"}:
        return "high" if value > 155 else "normal" if value >= 115 else "low"
    if name in {"stroke_length", "distance_per_stroke", "distance_per_cycle"}:
        return "low" if value < 1.2 else "normal" if value < 2.4 else "high"
    if name.endswith("_score") or name in {"kick_symmetry", "hip_alignment"}:
        return "high" if value >= 82 else "normal" if value >= 68 else "low"
    return "normal"


def _coaching_meaning(name: str, value: Any, confidence: float) -> str:
    if confidence < 0.35:
        return "Not enough reliable evidence to coach this metric from this clip."
    meanings = {
        "stroke_length": "How much water the swimmer covers per detected stroke.",
        "distance_per_stroke": "A practical distance-per-stroke proxy; low values can indicate short reach, low propulsion, or high drag.",
        "distance_per_cycle": "Travel per full left/right stroke cycle.",
        "catch_angle": "A catch proxy; very open angles can indicate a dropped elbow or late forearm setup.",
        "elbow_angle": "Arm-shape evidence through the pull/catch windows.",
        "shoulder_rotation_body_roll": "Approximate rotation range; extremes can disrupt balance.",
        "kick_frequency": "How often ankle peaks appear; useful only when both ankles are visible.",
        "kick_symmetry": "Whether the left and right kick amplitudes look balanced.",
        "breathing_timing": "Whether breath-like head motion aligns with stroke rhythm.",
        "propulsion_efficiency_score": "Composite estimate of how rhythm, distance per stroke, and speed trend work together.",
        "drag_risk_score": "Higher values suggest body-line or streamline patterns likely adding resistance.",
    }
    return meanings.get(name, "Confidence-aware swimming metric computed from landmark time-series.")
