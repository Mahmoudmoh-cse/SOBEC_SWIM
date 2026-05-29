from __future__ import annotations

from typing import Any


def generate_coaching_report(
    *,
    metrics: dict[str, Any],
    faults: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    velocity: dict[str, Any],
    video_quality: dict[str, Any],
    input_mode: str = "two_view",
) -> dict[str, Any]:
    top_faults = faults[:3]
    reliability = _metric_reliability_summary(metrics)
    actionable_faults = [fault for fault in top_faults if float(fault.get("confidence", 0) or 0) >= 0.45 and reliability["insufficient_evidence"] <= reliability["reliable"] + reliability["estimated"] + 2]
    strengths = _strengths(metrics)
    single_video = input_mode == "single_video"
    return {
        "technical_summary": _technical_summary(metrics, velocity, actionable_faults, reliability),
        "top_issues": [
            {
                "name": fault["name"],
                "severity": fault["severity"],
                "confidence": fault["confidence"],
                "explanation": fault["coach_explanation"],
                "evidence_level": "actionable" if fault in actionable_faults else "review_only",
            }
            for fault in top_faults
        ],
        "top_strengths": strengths[:3],
        "phase_by_phase_analysis": _phase_analysis(phases, velocity),
        "recommended_drills": _drills(actionable_faults, reliability=reliability),
        "weekly_correction_plan": _weekly_plan(actionable_faults, reliability=reliability, single_video=single_video),
        "race_improvement_suggestions": _race_suggestions(metrics, velocity, single_video=single_video),
        "metric_reliability": reliability,
        "safety_note": "AquaIQ is coaching decision support. It does not provide medical diagnosis or injury claims.",
        "quality_note": _quality_note(video_quality, single_video=single_video),
    }


def _technical_summary(metrics: dict[str, Any], velocity: dict[str, Any], faults: list[dict[str, Any]], reliability: dict[str, int]) -> str:
    overall = metrics.get("overall_technique_score", {}).get("value", 0)
    avg_velocity = velocity.get("summary", {}).get("average_velocity")
    unit = velocity.get("summary", {}).get("velocity_unit", "m/s")
    evidence_mode = velocity.get("summary", {}).get("evidence_mode")
    velocity_conf = float(velocity.get("summary", {}).get("confidence", 0) or 0)
    if evidence_mode == "centroid_only":
        return (
            f"OpenCV fallback produced centroid-only motion review at approximately {avg_velocity} {unit}. "
            "Skeleton landmarks were unavailable, so AquaIQ did not score biomechanics or technique faults from this run."
        )
    if reliability["reliable"] == 0 and reliability["estimated"] <= 2:
        return f"Overall technique score is {overall}. Metric evidence is insufficient for strong coaching advice; use this run mainly to inspect capture, ROI, and tracking diagnostics."
    if faults:
        focus = faults[0]["name"].replace("_", " ")
        return f"Overall technique score is {overall}. The highest-priority correction is {focus}. Average velocity is {avg_velocity} {unit} where calibration supports it."
    if velocity.get("chart_ready") and overall == 0:
        return f"Pose landmarks were unavailable, so AquaIQ produced centroid-only velocity review at {avg_velocity} {unit}. Do not draw biomechanics conclusions from this run."
    if velocity_conf < 0.45:
        return f"Overall technique score is {overall}. Velocity and phase evidence are low confidence, so use this run mainly to check capture quality."
    return f"Overall technique score is {overall}. No high-confidence major fault crossed the current Phase 1.5 thresholds."


def _strengths(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for name in ["body_alignment_score", "head_stability_score", "arm_symmetry_score", "streamline_score", "stroke_cycle_consistency"]:
        metric = metrics.get(name)
        if not metric or not isinstance(metric.get("value"), (int, float)):
            continue
        if metric.get("reliability_category") == "insufficient_evidence":
            continue
        confidence = metric.get("confidence")
        confidence_value = 0 if confidence == "low" else float(confidence or 0)
        if float(metric["value"]) >= 78 and confidence_value >= 0.35:
            candidates.append(
                {
                    "name": name,
                    "message": f"{name.replace('_', ' ')} is a usable strength in this clip.",
                    "value": metric["value"],
                    "confidence": confidence_value,
                }
            )
    return sorted(candidates, key=lambda item: item["value"], reverse=True)


def _phase_analysis(phases: list[dict[str, Any]], velocity: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = velocity.get("phase_summary", [])
    output = []
    for phase in phases:
        if phase.get("type") == "stroke_cycle":
            continue
        match = next((item for item in summaries if item.get("phase") == phase.get("type")), None)
        output.append(
            {
                "phase": phase.get("type"),
                "time_range_sec": [phase.get("start_sec"), phase.get("end_sec")],
                "confidence": phase.get("confidence"),
                "velocity": match,
                "coach_note": phase.get("reason"),
            }
        )
    return output


def _drills(faults: list[dict[str, Any]], *, reliability: dict[str, int] | None = None) -> list[dict[str, Any]]:
    if not faults:
        reason = "No major high-confidence fault crossed threshold from reliable skeleton evidence."
        if reliability and reliability.get("insufficient_evidence", 0) > reliability.get("reliable", 0) + reliability.get("estimated", 0):
            reason = "Metric evidence is weak, so AquaIQ is avoiding strong technique correction from this run."
        return [{"priority": 1, "drill": "Recapture or enable RTMPose before technique correction", "why": reason}]
    return [
        {
            "priority": index,
            "drill": fault["recommended_drill"],
            "why": fault["coach_explanation"],
            "linked_fault": fault["name"],
        }
        for index, fault in enumerate(faults, start=1)
    ]


def _weekly_plan(faults: list[dict[str, Any]], *, reliability: dict[str, int] | None = None, single_video: bool = False) -> list[dict[str, Any]]:
    drills = _drills(faults, reliability=reliability)
    retest = "Re-record one fixed-camera video and compare velocity dead spots and confidence." if single_video else "Re-record side and front/back views and compare velocity dead spots and confidence."
    return [
        {"week": 1, "focus": "Awareness and clean video recapture", "main_work": drills[0]["drill"] if drills else "Technique baseline"},
        {"week": 2, "focus": "Controlled repetition", "main_work": drills[min(1, len(drills) - 1)]["drill"] if drills else "Tempo consistency"},
        {"week": 3, "focus": "Race-pace integration", "main_work": "Blend the correction into 12.5m and 25m race-pace repeats."},
        {"week": 4, "focus": "Retest", "main_work": retest},
    ]


def _race_suggestions(metrics: dict[str, Any], velocity: dict[str, Any], *, single_video: bool = False) -> list[str]:
    suggestions = []
    velocity_conf = float(velocity.get("summary", {}).get("confidence", 0) or 0)
    centroid_only = velocity.get("summary", {}).get("evidence_mode") == "centroid_only"
    if velocity.get("summary", {}).get("dead_spot_count", 0) and velocity_conf >= 0.45 and not centroid_only:
        suggestions.append("Reduce speed interruptions by checking breath timing and breakout transitions.")
    stroke_consistency = _numeric_metric_value(metrics, "stroke_cycle_consistency", require_reliable=True)
    streamline = _numeric_metric_value(metrics, "streamline_score", require_reliable=True)
    if stroke_consistency is not None and stroke_consistency < 75:
        suggestions.append("Use tempo control before race day so stroke rhythm survives fatigue.")
    if streamline is not None and streamline < 76:
        suggestions.append("Prioritize push-off line and breakout speed, especially after starts and turns.")
    fallback = "Keep the same single-video camera protocol and compare speed curve changes across sessions." if single_video else "Keep the same camera protocol and compare speed curve changes across sessions."
    return suggestions or [fallback]


def _quality_note(video_quality: dict[str, Any], *, single_video: bool = False) -> str:
    values = []
    for view in video_quality.values():
        if isinstance(view, dict):
            values.append(float(view.get("overall_quality_score", view.get("usable_frame_ratio", 0)) or 0))
    if values and min(values) < 0.55:
        return "The uploaded video has weak quality; interpret low-confidence findings as prompts for coach review." if single_video else "At least one video has weak quality; interpret low-confidence findings as prompts for coach review."
    return "Video quality is sufficient for Phase 1.5 coaching review when metric confidence is medium or high."


def _metric_reliability_summary(metrics: dict[str, Any]) -> dict[str, int]:
    summary = {"reliable": 0, "estimated": 0, "insufficient_evidence": 0}
    for metric in metrics.values():
        if not isinstance(metric, dict):
            continue
        category = str(metric.get("reliability_category") or _fallback_reliability_category(metric))
        if category not in summary:
            category = "estimated"
        summary[category] += 1
    return summary


def _fallback_reliability_category(metric: dict[str, Any]) -> str:
    confidence = metric.get("confidence")
    numeric = 0.0 if confidence == "low" else float(confidence or 0) if isinstance(confidence, (int, float)) else 0.0
    if metric.get("value") is None or numeric < 0.35:
        return "insufficient_evidence"
    if numeric < 0.6:
        return "estimated"
    return "reliable"


def _numeric_metric_value(metrics: dict[str, Any], name: str, *, require_reliable: bool = False) -> float | None:
    metric = metrics.get(name, {})
    if require_reliable and isinstance(metric, dict) and metric.get("reliability_category") == "insufficient_evidence":
        return None
    value = metric.get("value") if isinstance(metric, dict) else None
    return float(value) if isinstance(value, (int, float)) else None
