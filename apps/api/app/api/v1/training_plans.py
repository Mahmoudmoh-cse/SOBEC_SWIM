from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.service import AICoachingService
from app.api.deps import get_current_user
from app.api.v1.utils import get_owned_swimmer
from app.db.session import get_db
from app.models import TrainingPlan, User
from app.schemas.common import TrainingPlanGenerate, TrainingPlanRead
from app.services.training import generate_training_plan

router = APIRouter(prefix="/training-plans", tags=["training plans"])


@router.post("/generate", response_model=TrainingPlanRead, status_code=201)
def generate_plan(
    payload: TrainingPlanGenerate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrainingPlan:
    swimmer = get_owned_swimmer(db, payload.swimmer_id, current_user)
    active_plans = db.scalars(
        select(TrainingPlan).where(
            TrainingPlan.swimmer_id == payload.swimmer_id,
            TrainingPlan.is_active.is_(True),
        )
    )
    for plan in active_plans:
        plan.is_active = False

    plan_data = generate_training_plan(
        race_date=payload.race_date,
        race_event=payload.race_event,
        target_time_seconds=payload.target_time_seconds,
    )
    plan = TrainingPlan(
        swimmer_id=payload.swimmer_id,
        race_date=payload.race_date,
        race_event=payload.race_event,
        target_time_seconds=payload.target_time_seconds,
        **plan_data,
    )
    db.add(plan)
    db.flush()
    ai_output = AICoachingService(db).training_rationale(swimmer, plan)
    if ai_output.parsed_json.get("summary"):
        plan.adaptation_log = [
            *(plan.adaptation_log or []),
            {"event": "ai_rationale", "message": ai_output.parsed_json["summary"]},
        ]
    db.commit()
    db.refresh(plan)
    return plan


@router.get("/active/{swimmer_id}", response_model=TrainingPlanRead)
def get_active_plan(
    swimmer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrainingPlan:
    get_owned_swimmer(db, swimmer_id, current_user)
    plan = db.scalar(
        select(TrainingPlan)
        .where(TrainingPlan.swimmer_id == swimmer_id, TrainingPlan.is_active.is_(True))
        .order_by(TrainingPlan.created_at.desc())
    )
    if plan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active plan not found")
    return plan
