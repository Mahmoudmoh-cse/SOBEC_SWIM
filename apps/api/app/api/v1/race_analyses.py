from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.service import AICoachingService
from app.api.deps import get_current_user
from app.api.v1.utils import get_owned_swimmer
from app.db.session import get_db
from app.models import RaceAnalysis, TrainingPlan, User
from app.schemas.common import RaceAnalysisCreate, RaceAnalysisRead
from app.services.race import (
    classify_strategy,
    generate_race_insights,
    phase_analysis,
    predicted_splits,
    strategy_score,
)

router = APIRouter(prefix="/race-analyses", tags=["race analyses"])


@router.get("", response_model=list[RaceAnalysisRead])
def list_race_analyses(
    swimmer_id: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[RaceAnalysis]:
    statement = select(RaceAnalysis).join(RaceAnalysis.swimmer).where(RaceAnalysis.swimmer.has(coach_id=current_user.id))
    if swimmer_id:
        get_owned_swimmer(db, swimmer_id, current_user)
        statement = statement.where(RaceAnalysis.swimmer_id == swimmer_id)
    return list(db.scalars(statement.order_by(RaceAnalysis.race_date.desc(), RaceAnalysis.created_at.desc())))


@router.post("", response_model=RaceAnalysisRead, status_code=201)
def create_race_analysis(
    payload: RaceAnalysisCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RaceAnalysis:
    swimmer = get_owned_swimmer(db, payload.swimmer_id, current_user)
    active_plan = db.scalar(
        select(TrainingPlan)
        .where(TrainingPlan.swimmer_id == swimmer.id, TrainingPlan.is_active.is_(True))
        .order_by(TrainingPlan.created_at.desc())
    )
    predicted_total = active_plan.target_time_seconds if active_plan else payload.official_time_seconds
    predicted = predicted_splits(predicted_total, len(payload.splits_actual))
    strategy = classify_strategy(payload.splits_actual)
    score = strategy_score(payload.splits_actual, predicted)
    pb = swimmer.personal_bests.get(payload.event) or swimmer.personal_bests.get(swimmer.primary_event)
    time_vs_pb = round(payload.official_time_seconds - pb, 2) if pb else None

    analysis = RaceAnalysis(
        swimmer_id=swimmer.id,
        race_date=payload.race_date,
        event=payload.event,
        official_time_seconds=payload.official_time_seconds,
        splits_actual=payload.splits_actual,
        splits_predicted=predicted,
        strategy_type=strategy,
        strategy_score=score,
        reaction_time_ms=payload.reaction_time_ms,
        turn_times=payload.turn_times,
        phase_analysis=phase_analysis(payload.splits_actual, predicted),
        time_vs_pb_seconds=time_vs_pb,
        ai_insights=generate_race_insights(strategy, score, time_vs_pb),
    )
    db.add(analysis)
    db.flush()
    ai_output = AICoachingService(db).race_debrief(swimmer, analysis)
    ai_insights = _ai_insights_from(ai_output.parsed_json)
    if ai_insights:
        analysis.ai_insights = ai_insights
    db.commit()
    db.refresh(analysis)
    return analysis


def _ai_insights_from(parsed: dict) -> list[str]:
    insights = []
    for key in ("summary", "key_findings", "recommendations", "next_steps", "risk_flags"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            insights.append(value.strip())
        elif isinstance(value, list):
            insights.extend(str(item).strip() for item in value if str(item).strip())
    return insights
