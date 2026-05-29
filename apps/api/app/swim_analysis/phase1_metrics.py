from __future__ import annotations

from statistics import mean, median, pstdev
from typing import Any

from app.swim_analysis.pose.base import Keypoint, PoseFrameResult
from app.swim_analysis.quality import confidence_payload, confidence_value


def compute_phase1_research_metrics(
    side_frames: list[PoseFrameResult],
    front_frames: list[PoseFrameResult],
    base_metrics: dict[str, Any],
    *,
    side_tracking: dict[str, Any],
    front_tracking: dict[str, Any],
    video_quality: dict[str, Any],
    single_video_mode: bool = False,
) -> dict[str, Any]:
    """Compute conservative Phase 1 metrics from cleaned trajectories and tracking metadata."""

    hip_stability = _hip_stability(side_frames)
    joint_visibility = _joint_visibility_score(side_tracking, front_tracking, single_video_mode=single_video_mode)
    tracking_quality = _tracking_quality_score(side_tracking, front_tracking, single_video_mode=single_video_mode)
    base_conf = confidence_value(base_metrics.get("confidence_score", {}).get("confidence", 0))
    quality_score = _video_quality_value(video_quality)
    reliability = _weighted_average(
        [
            (joint_visibility["value"], joint_visibility["confidence"]),
            (tracking_quality["value"], tracking_quality["confidence"]),
            (base_conf * 100.0, base_conf),
            (quality_score * 100.0, 0.85),
        ]
    )
    stroke_rhythm = _score_from_existing(base_metrics, "stroke_cycle_consistency", fallback="stroke_rate_spm")
    kick_rhythm = _score_from_existing(base_metrics, "kick_rhythm_score", fallback="kick_frequency")
    symmetry = _left_right_symmetry(base_metrics)

    return {
        "hip_stability_score": _metric(
            value=hip_stability["score"],
            confidence=hip_stability["confidence"],
            evidence=hip_stability["evidence"],
            reason="Hip center motion variability from cleaned uploaded video trajectories." if single_video_mode else "Hip center motion variability from cleaned side-view trajectories.",
            unit="score_0_100",
        ),
        "stroke_rhythm_estimate": _metric(
            value=stroke_rhythm["value"],
            confidence=stroke_rhythm["confidence"],
            evidence=stroke_rhythm["evidence"],
            reason="Estimated from cleaned wrist motion peaks and stroke-cycle timing consistency.",
            unit="score_0_100",
        ),
        "kick_rhythm_estimate": _metric(
            value=kick_rhythm["value"],
            confidence=kick_rhythm["confidence"],
            evidence=kick_rhythm["evidence"],
            reason="Estimated from cleaned ankle motion peaks; unavailable when ankles are not visible.",
            unit="score_0_100",
        ),
        "left_right_symmetry_estimate": _metric(
            value=symmetry["value"],
            confidence=symmetry["confidence"],
            evidence=symmetry["evidence"],
            reason="Phase 1 left/right balance estimate from arm and kick symmetry metrics.",
            unit="score_0_100",
        ),
        "joint_visibility_score": _metric(
            value=round(joint_visibility["value"]) if joint_visibility["value"] is not None else None,
            confidence=joint_visibility["confidence"],
            evidence=joint_visibility["evidence"],
            reason="Average percentage of expected joints visible or safely interpolated in the uploaded video." if single_video_mode else "Average percentage of expected joints visible or safely interpolated across both views.",
            unit="score_0_100",
        ),
        "tracking_quality_score": _metric(
            value=round(tracking_quality["value"]) if tracking_quality["value"] is not None else None,
            confidence=tracking_quality["confidence"],
            evidence=tracking_quality["evidence"],
            reason="Frame-level score combining visibility, confidence, rejected outliers, interpolation, and frame quality.",
            unit="score_0_100",
        ),
        "analysis_reliability_score": _metric(
            value=round(reliability) if reliability is not None else None,
            confidence=min(1.0, max(joint_visibility["confidence"], tracking_quality["confidence"]) * 0.55 + quality_score * 0.45),
            evidence={
                "joint_visibility_score": joint_visibility["value"],
                "tracking_quality_score": tracking_quality["value"],
                "base_confidence_score": round(base_conf * 100, 2),
                "video_quality_score": round(quality_score * 100, 2),
            },
            reason="Composite reliability gate for whether this run should support coaching conclusions.",
            unit="score_0_100",
        ),
    }


def reliability_score_from_metrics(metrics: dict[str, Any]) -> float:
    value = metrics.get("analysis_reliability_score", {}).get("value")
    return round(float(value) / 100.0, 3) if isinstance(value, (int, float)) else 0.0


def _hip_stability(frames: list[PoseFrameResult]) -> dict[str, Any]:
    offsets = []
    frame_indices = []
    for frame in frames:
        hip = _midpoint(frame.keypoint("left_hip", 0.25), frame.keypoint("right_hip", 0.25))
        shoulder = _midpoint(frame.keypoint("left_shoulder", 0.25), frame.keypoint("right_shoulder", 0.25))
        if hip is None or shoulder is None:
            continue
        torso = _distance(hip, shoulder) or 1.0
        offsets.append(abs(hip.y - shoulder.y) / torso)
        frame_indices.append(frame.frame_index)
    if not offsets:
        return {"score": None, "confidence": 0.0, "evidence": {"visible_frames": 0, "hip_offset_std": None}}
    variability = pstdev(offsets) if len(offsets) > 1 else 0.0
    baseline = median(offsets)
    score = round(max(0.0, min(100.0, 100.0 - variability * 185.0 - max(0.0, baseline - 1.05) * 35.0)))
    confidence = min(1.0, len(offsets) / max(1, len(frames)) * 0.75 + 0.2)
    return {
        "score": score,
        "confidence": confidence,
        "evidence": {
            "visible_frames": len(offsets),
            "median_hip_offset": round(baseline, 3),
            "hip_offset_std": round(variability, 3),
            "frames": frame_indices[:16],
        },
    }


def _joint_visibility_score(side_tracking: dict[str, Any], front_tracking: dict[str, Any], *, single_video_mode: bool = False) -> dict[str, Any]:
    side = side_tracking.get("visibility_percentage", {})
    if single_video_mode:
        values = [float(value) for value in side.values() if isinstance(value, (int, float))]
        if not values:
            return {"value": None, "confidence": 0.0, "evidence": {"video": side}}
        score = mean(values)
        confidence = min(1.0, score / 100.0)
        return {"value": score, "confidence": confidence, "evidence": {"video": side}}
    front = front_tracking.get("visibility_percentage", {})
    values = [float(value) for value in [*side.values(), *front.values()] if isinstance(value, (int, float))]
    if not values:
        return {"value": None, "confidence": 0.0, "evidence": {"side": side, "front": front}}
    score = mean(values)
    confidence = min(1.0, score / 100.0)
    return {"value": score, "confidence": confidence, "evidence": {"side": side, "front": front}}


def _tracking_quality_score(side_tracking: dict[str, Any], front_tracking: dict[str, Any], *, single_video_mode: bool = False) -> dict[str, Any]:
    side = [float(point.get("quality", 0) or 0) for point in side_tracking.get("tracking_quality_timeline", [])]
    if single_video_mode:
        if not side:
            return {"value": None, "confidence": 0.0, "evidence": {"video_frames": 0}}
        avg = mean(side)
        return {
            "value": avg * 100.0,
            "confidence": min(1.0, avg * 1.1),
            "evidence": {
                "video_mean_quality": round(mean(side), 3),
                "video_frames": len(side),
            },
        }
    front = [float(point.get("quality", 0) or 0) for point in front_tracking.get("tracking_quality_timeline", [])]
    values = side + front
    if not values:
        return {"value": None, "confidence": 0.0, "evidence": {"side_frames": 0, "front_frames": 0}}
    avg = mean(values)
    return {
        "value": avg * 100.0,
        "confidence": min(1.0, avg * 1.1),
        "evidence": {
            "side_mean_quality": round(mean(side), 3) if side else None,
            "front_mean_quality": round(mean(front), 3) if front else None,
            "side_frames": len(side),
            "front_frames": len(front),
        },
    }


def _score_from_existing(metrics: dict[str, Any], name: str, *, fallback: str) -> dict[str, Any]:
    primary = metrics.get(name, {})
    value = primary.get("value")
    confidence = confidence_value(primary.get("confidence", 0))
    evidence = dict(primary.get("evidence", {}))
    if isinstance(value, (int, float)):
        return {"value": round(float(value)), "confidence": confidence, "evidence": evidence}
    secondary = metrics.get(fallback, {})
    secondary_value = secondary.get("value")
    secondary_conf = confidence_value(secondary.get("confidence", 0))
    if isinstance(secondary_value, (int, float)) and fallback.endswith("_frequency"):
        proxy_score = max(0.0, min(100.0, 100.0 - abs(float(secondary_value) - 90.0) * 0.45))
        return {"value": round(proxy_score), "confidence": secondary_conf * 0.6, "evidence": dict(secondary.get("evidence", {}))}
    return {"value": None, "confidence": 0.0, "evidence": evidence or dict(secondary.get("evidence", {}))}


def _left_right_symmetry(metrics: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    evidence: dict[str, Any] = {}
    for name in ["arm_symmetry_score", "kick_symmetry"]:
        metric = metrics.get(name, {})
        value = metric.get("value")
        confidence = confidence_value(metric.get("confidence", 0))
        evidence[name] = {"value": value, "confidence": confidence}
        if isinstance(value, (int, float)) and confidence > 0:
            candidates.append((float(value), confidence))
    value = _weighted_average(candidates)
    confidence = mean([conf for _, conf in candidates]) if candidates else 0.0
    return {"value": round(value) if value is not None else None, "confidence": confidence, "evidence": evidence}


def _video_quality_value(video_quality: dict[str, Any]) -> float:
    values = []
    for view in video_quality.values():
        if not isinstance(view, dict):
            continue
        if "overall_quality_score" in view:
            values.append(float(view.get("overall_quality_score") or 0))
        else:
            values.append(
                0.3 * float(view.get("usable_frame_ratio", 0) or 0)
                + 0.25 * float(view.get("blur_score", 0) or 0)
                + 0.25 * float(view.get("lighting_score", 0) or 0)
                + 0.2 * float(view.get("stability_score", 0) or 0)
            )
    return mean(values) if values else 0.0


def _metric(*, value: Any, confidence: float, evidence: dict[str, Any], reason: str, unit: str | None = None) -> dict[str, Any]:
    evidence = dict(evidence)
    output = {
        "value": value,
        "confidence": confidence_payload(confidence, evidence),
        "evidence": evidence,
        "reason": reason,
        "unit": unit,
        "supporting_frame_count": _supporting_frame_count(evidence),
        "reliability_category": _metric_reliability_category(value, confidence),
    }
    output["interpretation"] = _interpret_score(value, confidence)
    output["coaching_meaning"] = "Phase 1 reliability-aware metric computed from cleaned landmark trajectories."
    return output


def _interpret_score(value: Any, confidence: float) -> str:
    if confidence < 0.35:
        return "low_confidence"
    if not isinstance(value, (int, float)):
        return "unavailable"
    if value >= 82:
        return "high"
    if value >= 66:
        return "normal"
    return "low"


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


def _weighted_average(values: list[tuple[Any, float]]) -> float | None:
    weighted = [(float(value), max(0.0, min(1.0, confidence))) for value, confidence in values if isinstance(value, (int, float)) and confidence > 0]
    if not weighted:
        return None
    denom = sum(confidence for _, confidence in weighted)
    if denom <= 0:
        return None
    return sum(value * confidence for value, confidence in weighted) / denom


def _midpoint(a: Keypoint | None, b: Keypoint | None) -> Keypoint | None:
    if a is None or b is None:
        return None
    return Keypoint("midpoint", (a.x + b.x) / 2.0, (a.y + b.y) / 2.0, min(a.confidence, b.confidence))


def _distance(a: Keypoint | None, b: Keypoint | None) -> float | None:
    if a is None or b is None:
        return None
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
