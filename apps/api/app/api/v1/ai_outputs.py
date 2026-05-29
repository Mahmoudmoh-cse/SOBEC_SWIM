from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.ai.service import AICoachingService
from app.api.deps import get_current_user
from app.api.v1.utils import get_owned_swimmer
from app.db.session import get_db
from app.models import AIOutput, MentalCheckin, RaceAnalysis, Session, Swimmer, TechniqueReport, TrainingPlan, User
from app.schemas.common import AIOutputRead

router = APIRouter(prefix="/ai-outputs", tags=["ai outputs"])


@router.get("", response_model=list[AIOutputRead])
def list_ai_outputs(
    swimmer_id: str | None = None,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[AIOutput]:
    statement = select(AIOutput).join(AIOutput.swimmer).where(AIOutput.swimmer.has(coach_id=current_user.id))
    if swimmer_id:
        get_owned_swimmer(db, swimmer_id, current_user)
        statement = statement.where(AIOutput.swimmer_id == swimmer_id)
    return list(db.scalars(statement.order_by(AIOutput.created_at.desc())))


@router.post("/backfill/{swimmer_id}", response_model=list[AIOutputRead])
def backfill_ai_outputs(
    swimmer_id: str,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[AIOutput]:
    swimmer = get_owned_swimmer(db, swimmer_id, current_user)
    service = AICoachingService(db)
    created: list[AIOutput] = []

    existing = list(db.scalars(select(AIOutput).where(AIOutput.swimmer_id == swimmer.id)))
    existing_report_ids = {item.technique_report_id for item in existing if item.output_type == "technique_summary"}
    existing_plan_ids = {item.training_plan_id for item in existing if item.output_type == "training_rationale"}
    existing_race_ids = {item.race_analysis_id for item in existing if item.output_type == "race_debrief"}
    existing_checkin_ids = {item.mental_checkin_id for item in existing if item.output_type == "mental_routine"}

    reports = db.scalars(select(TechniqueReport).where(TechniqueReport.swimmer_id == swimmer.id)).all()
    for report in reports:
        if report.id in existing_report_ids:
            continue
        session = db.get(Session, report.session_id)
        if session:
            created.append(service.technique_summary(swimmer, session, report))

    plans = db.scalars(select(TrainingPlan).where(TrainingPlan.swimmer_id == swimmer.id)).all()
    for plan in plans:
        if plan.id not in existing_plan_ids:
            created.append(service.training_rationale(swimmer, plan))

    races = db.scalars(select(RaceAnalysis).where(RaceAnalysis.swimmer_id == swimmer.id)).all()
    for race in races:
        if race.id not in existing_race_ids:
            created.append(service.race_debrief(swimmer, race))

    checkins = db.scalars(select(MentalCheckin).where(MentalCheckin.swimmer_id == swimmer.id)).all()
    for checkin in checkins:
        if checkin.id not in existing_checkin_ids:
            created.append(service.mental_routine(swimmer, checkin))

    db.commit()
    for output in created:
        db.refresh(output)
    return created

