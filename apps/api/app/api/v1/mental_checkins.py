from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.service import AICoachingService
from app.api.deps import get_current_user
from app.api.v1.utils import get_owned_swimmer
from app.db.session import get_db
from app.models import MentalCheckin, User
from app.schemas.common import MentalCheckinCreate, MentalCheckinRead
from app.services.calculations import MOOD_FIELDS, calculate_mental_composite
from app.services.mental import generate_routine

router = APIRouter(prefix="/mental-checkins", tags=["mental checkins"])


@router.get("", response_model=list[MentalCheckinRead])
def list_mental_checkins(
    swimmer_id: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[MentalCheckin]:
    statement = select(MentalCheckin).join(MentalCheckin.swimmer).where(MentalCheckin.swimmer.has(coach_id=current_user.id))
    if swimmer_id:
        get_owned_swimmer(db, swimmer_id, current_user)
        statement = statement.where(MentalCheckin.swimmer_id == swimmer_id)
    return list(db.scalars(statement.order_by(MentalCheckin.checkin_date.desc())))


@router.post("", response_model=MentalCheckinRead, status_code=201)
def create_mental_checkin(
    payload: MentalCheckinCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MentalCheckin:
    swimmer = get_owned_swimmer(db, payload.swimmer_id, current_user)
    values = {field: getattr(payload, field) for field in MOOD_FIELDS}
    checkin = MentalCheckin(
        **payload.model_dump(exclude={"checkin_date"}),
        checkin_date=payload.checkin_date or datetime.now(timezone.utc),
        composite_score=calculate_mental_composite(values),
        routine_generated=generate_routine(values),
    )
    db.add(checkin)
    db.flush()
    ai_output = AICoachingService(db).mental_routine(swimmer, checkin)
    if ai_output.parsed_json.get("summary"):
        checkin.routine_generated = {
            **(checkin.routine_generated or {}),
            "ai_summary": ai_output.parsed_json["summary"],
            "ai_next_steps": "; ".join(ai_output.parsed_json.get("next_steps", [])),
        }
    db.commit()
    db.refresh(checkin)
    return checkin
