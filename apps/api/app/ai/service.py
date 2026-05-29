from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session as DbSession

from app.ai.parser import parse_ai_response
from app.ai.providers import AIProvider, AIProviderConfigError, AIProviderError, AnthropicAIProvider, LocalAIProvider
from app.ai.schemas import AIProviderRequest
from app.core.config import Settings, get_settings
from app.models import AIOutput, MentalCheckin, RaceAnalysis, Session, Swimmer, TechniqueReport, TrainingPlan


PROMPT_VERSION = "v1"
PROMPT_DIR = Path(__file__).parent / "prompts"


def select_ai_provider(settings: Settings | None = None) -> AIProvider:
    settings = settings or get_settings()
    mode = settings.ai_provider.lower()
    if mode == "local":
        return LocalAIProvider()
    if mode == "anthropic":
        return AnthropicAIProvider(
            api_key=settings.anthropic_api_key,
            heavy_model=settings.claude_model_heavy,
            fast_model=settings.claude_model_fast,
        )
    if mode == "hybrid":
        if settings.anthropic_api_key:
            return AnthropicAIProvider(
                api_key=settings.anthropic_api_key,
                heavy_model=settings.claude_model_heavy,
                fast_model=settings.claude_model_fast,
            )
        return LocalAIProvider()
    raise AIProviderConfigError("AI_PROVIDER must be one of: local, anthropic, hybrid.")


class AICoachingService:
    def __init__(self, db: DbSession, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()

    def technique_summary(self, swimmer: Swimmer, session: Session, report: TechniqueReport) -> AIOutput:
        faults = report.faults or []
        drills = report.drill_prescriptions or []
        context = {
            "swimmer_name": swimmer.name,
            "stroke": report.stroke,
            "session_type": session.session_type,
            "rpe": session.rpe,
            "technique_score": report.overall_score,
            "top_fault": faults[0].get("title") if faults else "Stable body line",
            "time_gain_possible": (report.keypoint_data or {}).get("time_gain_possible", 0),
            "primary_drill": _drill_label(drills[0]) if drills else "Maintain current technical cues.",
            "analysis_status": report.analysis_status,
            "confidence_label": report.confidence_label,
            "pose_detection_rate": report.pose_detection_rate,
        }
        return self.generate_and_record(
            output_type="technique_summary",
            template_name="technique_summary.j2",
            context=context,
            swimmer_id=swimmer.id,
            session_id=session.id,
            technique_report_id=report.id,
            model_tier="fast",
        )

    def training_rationale(self, swimmer: Swimmer, plan: TrainingPlan) -> AIOutput:
        context = {
            "swimmer_name": swimmer.name,
            "primary_event": swimmer.primary_event,
            "race_event": plan.race_event,
            "race_date": plan.race_date.isoformat(),
            "target_time_seconds": plan.target_time_seconds,
            "weeks_total": plan.weeks_total,
            "current_phase": plan.current_phase,
            "phase_config": plan.phase_config,
            "weekly_plan_count": len(plan.weekly_plans or []),
        }
        return self.generate_and_record(
            output_type="training_rationale",
            template_name="training_rationale.j2",
            context=context,
            swimmer_id=swimmer.id,
            training_plan_id=plan.id,
            model_tier="heavy",
        )

    def race_debrief(self, swimmer: Swimmer, analysis: RaceAnalysis) -> AIOutput:
        context = {
            "swimmer_name": swimmer.name,
            "event": analysis.event,
            "official_time_seconds": analysis.official_time_seconds,
            "splits_actual": analysis.splits_actual,
            "splits_predicted": analysis.splits_predicted,
            "strategy_type": analysis.strategy_type,
            "strategy_score": analysis.strategy_score,
            "time_vs_pb_seconds": analysis.time_vs_pb_seconds,
            "phase_analysis": analysis.phase_analysis,
        }
        return self.generate_and_record(
            output_type="race_debrief",
            template_name="race_debrief.j2",
            context=context,
            swimmer_id=swimmer.id,
            race_analysis_id=analysis.id,
            model_tier="heavy",
        )

    def mental_routine(self, swimmer: Swimmer, checkin: MentalCheckin) -> AIOutput:
        scores = {
            "focus": checkin.mood_focus,
            "confidence": checkin.mood_confidence,
            "energy": checkin.mood_energy,
            "calm": checkin.mood_calm,
            "recovery": checkin.mood_recovery,
            "motivation": checkin.mood_motivation,
        }
        context = {
            "swimmer_name": swimmer.name,
            "checkin_type": checkin.checkin_type,
            "composite_score": checkin.composite_score,
            "scores": scores,
            "low_dimensions": [key for key, value in scores.items() if value <= 5],
            "routine_generated": checkin.routine_generated,
        }
        return self.generate_and_record(
            output_type="mental_routine",
            template_name="mental_routine.j2",
            context=context,
            swimmer_id=swimmer.id,
            mental_checkin_id=checkin.id,
            model_tier="fast",
        )

    def generate_and_record(
        self,
        *,
        output_type: str,
        template_name: str,
        context: dict[str, Any],
        swimmer_id: str,
        session_id: str | None = None,
        technique_report_id: str | None = None,
        training_plan_id: str | None = None,
        race_analysis_id: str | None = None,
        mental_checkin_id: str | None = None,
        model_tier: str = "fast",
    ) -> AIOutput:
        prompt_text = render_prompt(template_name, context)
        provider = select_ai_provider(self.settings)
        status = "completed"
        provider_error: str | None = None

        try:
            result = provider.generate(AIProviderRequest(output_type, prompt_text, context, model_tier))
        except AIProviderError as exc:
            if self.settings.ai_provider.lower() != "hybrid":
                raise
            provider_error = str(exc)
            provider = LocalAIProvider()
            result = provider.generate(AIProviderRequest(output_type, prompt_text, context, model_tier))
            status = "fallback"

        parsed = parse_ai_response(result.raw_response)
        output = AIOutput(
            swimmer_id=swimmer_id,
            session_id=session_id,
            technique_report_id=technique_report_id,
            training_plan_id=training_plan_id,
            race_analysis_id=race_analysis_id,
            mental_checkin_id=mental_checkin_id,
            output_type=output_type,
            provider=result.provider,
            model=result.model,
            prompt_version=PROMPT_VERSION,
            prompt_text=prompt_text,
            raw_response=result.raw_response,
            parsed_json=parsed,
            status=status,
            error=provider_error,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        self.db.add(output)
        self.db.flush()
        return output


def render_prompt(template_name: str, context: dict[str, Any]) -> str:
    template = (PROMPT_DIR / template_name).read_text(encoding="utf-8")
    context_json = json.dumps(context, indent=2, default=str)
    return template.replace("{{ context_json }}", context_json)


def _drill_label(drill: dict[str, Any]) -> str:
    name = drill.get("name", "Prescribed drill")
    focus = drill.get("focus")
    volume = drill.get("volume") or drill.get("sets")
    parts = [str(name)]
    if focus:
        parts.append(f"focus: {focus}")
    if volume:
        parts.append(f"volume: {volume}")
    return ", ".join(parts)

