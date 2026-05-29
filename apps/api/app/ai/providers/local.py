from __future__ import annotations

import json
import time
from typing import Any

from app.ai.providers.base import AIProvider
from app.ai.schemas import AIProviderRequest, AIProviderResult, empty_parsed_output


class LocalAIProvider(AIProvider):
    name = "local"
    model = "local-deterministic-v1"

    def generate(self, request: AIProviderRequest) -> AIProviderResult:
        start = time.perf_counter()
        parsed = _local_output(request.output_type, request.context)
        return AIProviderResult(
            provider=self.name,
            model=self.model,
            raw_response=json.dumps(parsed),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def _local_output(output_type: str, context: dict[str, Any]) -> dict[str, Any]:
    if output_type == "technique_summary":
        swimmer_name = str(context.get("swimmer_name", "This swimmer"))
        score = context.get("technique_score", "n/a")
        top_fault = str(context.get("top_fault", "the highest-priority technique limiter"))
        time_gain = context.get("time_gain_possible", 0)
        return {
            "summary": (
                f"{swimmer_name} scored {score}/100. The main focus is {top_fault}, "
                f"with about {time_gain}s of potential improvement flagged by AquaIQ."
            ),
            "key_findings": [f"Primary limiter: {top_fault}", f"Technique score: {score}/100"],
            "recommendations": [str(context.get("primary_drill", "Use the prescribed drill set before the main set."))],
            "risk_flags": _risk_flags(context),
            "next_steps": ["Review the video quality and repeat the same view next session."],
        }

    if output_type == "training_rationale":
        weeks = context.get("weeks_total", "the current")
        phase = context.get("current_phase", "base")
        event = context.get("race_event", "target event")
        return {
            "summary": f"Generated a {weeks}-week {event} plan starting in the {phase} phase.",
            "key_findings": [f"Current phase: {phase}", f"Target event: {event}"],
            "recommendations": ["Follow the weekly load progression and adjust only when recovery trends drop."],
            "risk_flags": [],
            "next_steps": ["Log each session so the next plan update can use real compliance data."],
        }

    if output_type == "race_debrief":
        strategy = context.get("strategy_type", "custom")
        score = context.get("strategy_score", "n/a")
        delta = context.get("time_vs_pb_seconds")
        delta_text = "No PB comparison available." if delta is None else f"PB delta: {delta}s."
        return {
            "summary": f"Race execution classified as {strategy} with a strategy score of {score}/100. {delta_text}",
            "key_findings": [f"Strategy: {strategy}", f"Execution score: {score}/100"],
            "recommendations": ["Use split feedback in the next race-pace set."],
            "risk_flags": ["Second-half fade risk"] if strategy == "positive" else [],
            "next_steps": ["Compare the largest split delta against the race plan."],
        }

    if output_type == "mental_routine":
        composite = context.get("composite_score", "n/a")
        low_dimensions = context.get("low_dimensions", [])
        return {
            "summary": f"Mental check-in composite is {composite}. Use a short routine focused on the lowest readiness signals.",
            "key_findings": [f"Lowest dimensions: {', '.join(low_dimensions) or 'none flagged'}"],
            "recommendations": ["Keep the routine simple: breathing, one cue word, one visualized race detail."],
            "risk_flags": [f"Low {item}" for item in low_dimensions],
            "next_steps": ["Re-check confidence and calm before the next race or hard set."],
        }

    return empty_parsed_output(summary="AquaIQ generated a deterministic local coaching output.")


def _risk_flags(context: dict[str, Any]) -> list[str]:
    flags = []
    if str(context.get("analysis_status", "")) in {"low_confidence", "no_pose_detected", "failed"}:
        flags.append(f"Analysis quality: {context.get('analysis_status')}")
    if context.get("rpe", 0) and int(context["rpe"]) >= 8:
        flags.append("High RPE session")
    return flags

