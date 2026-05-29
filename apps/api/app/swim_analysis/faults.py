from __future__ import annotations

from typing import Any


def detect_swimming_faults(
    metrics: dict[str, Any],
    velocity: dict[str, Any],
    phases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    faults = []
    _maybe_add_score_fault(
        faults,
        metrics,
        "catch_angle",
        threshold=135,
        comparison="high",
        name="dropped_elbow_catch",
        drill="Scull 1 + dog paddle catch progression",
        explanation="The catch angle suggests the forearm is not setting early enough to press water backward.",
    )
    _maybe_add_score_fault(
        faults,
        metrics,
        "head_stability_score",
        threshold=72,
        comparison="low",
        name="excessive_head_lift",
        drill="One-goggle breathing with snorkel alternation",
        explanation="Head movement is costing line stability; rotate to breathe instead of lifting.",
    )
    _maybe_add_score_fault(
        faults,
        metrics,
        "streamline_score",
        threshold=74,
        comparison="low",
        name="poor_streamline",
        drill="Push-off streamline holds into 6-beat breakout",
        explanation="The body line is creating extra drag during glide or early swim phases.",
    )
    _maybe_add_score_fault(
        faults,
        metrics,
        "kick_symmetry",
        threshold=74,
        comparison="low",
        name="asymmetric_kick",
        drill="Vertical kick and side-kick symmetry checks",
        explanation="Left/right ankle rhythm is uneven enough to affect balance and propulsion.",
    )
    _maybe_add_score_fault(
        faults,
        metrics,
        "distance_per_stroke",
        threshold=1.2,
        comparison="low",
        name="short_stroke_length",
        drill="Catch-up 50s with stroke-count target",
        explanation="Distance per stroke is short for the available calibration, suggesting lost length or excess drag.",
    )
    _maybe_add_score_fault(
        faults,
        metrics,
        "drag_risk_score",
        threshold=45,
        comparison="high",
        name="high_drag_body_position",
        drill="Side-kick line control and snorkel body-balance work",
        explanation="Alignment and velocity evidence suggest the swimmer is carrying unnecessary drag.",
    )
    _maybe_add_score_fault(
        faults,
        metrics,
        "stroke_cycle_consistency",
        threshold=70,
        comparison="low",
        name="inconsistent_stroke_rhythm",
        drill="Tempo trainer 25s at stable cadence",
        explanation="Stroke-cycle timing varies enough to disturb rhythm and speed continuity.",
    )

    velocity_confidence = float(velocity.get("summary", {}).get("confidence", 0) or 0)
    centroid_only_velocity = velocity.get("summary", {}).get("evidence_mode") == "centroid_only"
    for dead_spot in velocity.get("dead_spots", [])[:4]:
        dead_spot_confidence = float(dead_spot.get("confidence", 0) or 0)
        if centroid_only_velocity or min(velocity_confidence, dead_spot_confidence) < 0.45:
            continue
        faults.append(
            {
                "name": "velocity_dead_spot",
                "severity": "high" if float(dead_spot.get("min_velocity", 0)) < float(dead_spot.get("threshold", 0)) * 0.6 else "medium",
                "evidence_metrics": {"dead_spot": dead_spot},
                "timestamp_range": [dead_spot.get("start_sec"), dead_spot.get("end_sec")],
                "confidence": min(velocity_confidence, dead_spot_confidence),
                "recommended_drill": "Build speed through the transition with 6 strokes fast after breakout or turn.",
                "coach_explanation": "The swimmer loses too much speed over this window; look for timing, breath, or body-line interruption.",
            }
        )

    breakout = _phase(phases, "breakout")
    breakout_speed = velocity.get("breakout_speed")
    avg = velocity.get("summary", {}).get("average_velocity")
    if (
        not centroid_only_velocity
        and velocity_confidence >= 0.45
        and breakout
        and breakout_speed
        and avg
        and breakout_speed.get("value")
        and float(breakout_speed["value"]) < float(avg) * 0.85
    ):
        faults.append(
            {
                "name": "weak_breakout",
                "severity": "medium",
                "evidence_metrics": {"breakout_speed": breakout_speed, "average_velocity": avg},
                "timestamp_range": [breakout["start_sec"], breakout["end_sec"]],
                "confidence": min(float(breakout.get("confidence", 0)), float(breakout_speed.get("confidence", 0))),
                "recommended_drill": "Underwater 6-kick breakout into 4 fast strokes.",
                "coach_explanation": "Breakout speed is lower than the swimmer's average pace, so speed is leaking when transitioning to free swim.",
            }
        )

    breathing = metrics.get("breathing_timing", {})
    if breathing.get("interpretation") == "late" and _confidence(breathing) >= 0.35:
        faults.append(
            {
                "name": "late_breathing",
                "severity": "medium",
                "evidence_metrics": {"breathing_timing": breathing},
                "timestamp_range": breathing.get("evidence", {}).get("time_range_sec", [None, None]),
                "confidence": _confidence(breathing),
                "recommended_drill": "3-3-3 breathing timing drill",
                "coach_explanation": "Breathing appears to happen late in the stroke cycle, which can interrupt rotation and velocity.",
            }
        )

    return _dedupe_faults(faults)


def _maybe_add_score_fault(
    faults: list[dict[str, Any]],
    metrics: dict[str, Any],
    metric_name: str,
    *,
    threshold: float,
    comparison: str,
    name: str,
    drill: str,
    explanation: str,
) -> None:
    metric = metrics.get(metric_name)
    if not metric:
        return
    value = metric.get("value")
    confidence = _confidence(metric)
    if not isinstance(value, (int, float)) or confidence < 0.35:
        return
    triggered = value < threshold if comparison == "low" else value > threshold
    if not triggered:
        return
    severity = "high" if (comparison == "low" and value < threshold * 0.85) or (comparison == "high" and value > threshold * 1.12) else "medium"
    faults.append(
        {
            "name": name,
            "severity": severity,
            "evidence_metrics": {metric_name: metric},
            "timestamp_range": metric.get("evidence", {}).get("time_range_sec", metric.get("evidence", {}).get("event_times_sec", [None, None])),
            "confidence": confidence,
            "recommended_drill": drill,
            "coach_explanation": explanation,
        }
    )


def _phase(phases: list[dict[str, Any]], phase_name: str) -> dict[str, Any] | None:
    for phase in phases:
        if phase.get("type") == phase_name:
            return phase
    return None


def _confidence(metric: dict[str, Any]) -> float:
    confidence = metric.get("confidence", 0)
    if confidence == "low":
        return float(metric.get("evidence", {}).get("numeric_confidence", 0) or 0)
    return float(confidence or 0)


def _dedupe_faults(faults: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    severity_order = {"high": 0, "medium": 1, "low": 2}
    for fault in sorted(faults, key=lambda item: (severity_order.get(str(item.get("severity")), 3), -float(item.get("confidence", 0) or 0))):
        name = fault.get("name")
        if name in seen:
            continue
        seen.add(name)
        output.append(fault)
    return output[:10]
