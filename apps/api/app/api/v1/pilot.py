from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.ai.service import AICoachingService
from app.api.deps import get_current_user
from app.api.v1.utils import get_owned_swimmer
from app.db.session import get_db
from app.models import MentalCheckin, RaceAnalysis, Session, Swimmer, TechniqueReport, User
from app.schemas.common import PilotBoardCard, PilotBoardResponse, PilotSessionEntryCreate, PilotSessionEntryResponse
from app.services.calculations import MOOD_FIELDS, calculate_mental_composite, calculate_session_load
from app.services.mental import generate_routine

router = APIRouter(prefix="/pilot", tags=["pilot"])


@router.get("/board", response_model=PilotBoardResponse)
def pilot_board(
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PilotBoardResponse:
    swimmers = list(db.scalars(select(Swimmer).where(Swimmer.coach_id == current_user.id).order_by(Swimmer.name)))
    cards = [_build_card(db, swimmer) for swimmer in swimmers]
    totals = {
        "swimmers": len(cards),
        "needs_session": sum(1 for card in cards if "log_session" in card.missing_flags),
        "needs_video": sum(1 for card in cards if "upload_video" in card.missing_flags),
        "needs_checkin": sum(1 for card in cards if "mental_checkin" in card.missing_flags),
        "at_risk": sum(1 for card in cards if card.status == "watch"),
    }
    return PilotBoardResponse(generated_at=datetime.now(timezone.utc), swimmers=cards, totals=totals)


@router.post("/session-entry", response_model=PilotSessionEntryResponse, status_code=201)
def create_pilot_session_entry(
    payload: PilotSessionEntryCreate,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PilotSessionEntryResponse:
    swimmer = get_owned_swimmer(db, payload.swimmer_id, current_user)
    duplicate = db.scalar(
        select(Session)
        .where(
            Session.swimmer_id == swimmer.id,
            Session.session_date == payload.session_date,
            Session.session_type == payload.session_type,
        )
        .limit(1)
    )
    duplicate_warning = (
        f"{swimmer.name} already has a {payload.session_type} session on {payload.session_date.isoformat()}."
        if duplicate
        else None
    )

    session_fields = payload.model_dump(
        exclude={"create_mental_checkin", "checkin_type", "checkin_notes"}
    )
    session = Session(
        **session_fields,
        load_score=calculate_session_load(payload.distance_m, payload.rpe),
    )
    db.add(session)
    db.flush()

    checkin = None
    if payload.create_mental_checkin:
        values = {field: getattr(payload, field) for field in MOOD_FIELDS}
        checkin = MentalCheckin(
            swimmer_id=swimmer.id,
            checkin_type=payload.checkin_type,
            mood_focus=payload.mood_focus,
            mood_confidence=payload.mood_confidence,
            mood_energy=payload.mood_energy,
            mood_calm=payload.mood_calm,
            mood_recovery=payload.mood_recovery,
            mood_motivation=payload.mood_motivation,
            composite_score=calculate_mental_composite(values),
            routine_generated=generate_routine(values),
            notes=payload.checkin_notes or payload.notes,
        )
        db.add(checkin)

    db.flush()
    if checkin:
        ai_output = AICoachingService(db).mental_routine(swimmer, checkin)
        if ai_output.parsed_json.get("summary"):
            checkin.routine_generated = {
                **(checkin.routine_generated or {}),
                "ai_summary": ai_output.parsed_json["summary"],
                "ai_next_steps": "; ".join(ai_output.parsed_json.get("next_steps", [])),
            }

    db.commit()
    db.refresh(session)
    if checkin:
        db.refresh(checkin)

    return PilotSessionEntryResponse(
        session=session,
        mental_checkin=checkin,
        duplicate_warning=duplicate_warning,
        entry_summary=(
            f"Saved {payload.distance_m}m {payload.session_type} session for {swimmer.name} "
            f"with load {session.load_score}."
        ),
    )


def _build_card(db: DbSession, swimmer: Swimmer) -> PilotBoardCard:
    latest_session = db.scalar(
        select(Session)
        .where(Session.swimmer_id == swimmer.id)
        .order_by(Session.session_date.desc(), Session.created_at.desc())
        .limit(1)
    )
    latest_report = db.scalar(
        select(TechniqueReport)
        .where(TechniqueReport.swimmer_id == swimmer.id)
        .order_by(TechniqueReport.created_at.desc())
        .limit(1)
    )
    latest_checkin = db.scalar(
        select(MentalCheckin)
        .where(MentalCheckin.swimmer_id == swimmer.id)
        .order_by(MentalCheckin.checkin_date.desc())
        .limit(1)
    )
    latest_race = db.scalar(
        select(RaceAnalysis)
        .where(RaceAnalysis.swimmer_id == swimmer.id)
        .order_by(RaceAnalysis.race_date.desc(), RaceAnalysis.created_at.desc())
        .limit(1)
    )

    missing_flags = []
    if latest_session is None:
        missing_flags.append("log_session")
    if latest_session and latest_session.video_url is None:
        missing_flags.append("upload_video")
    if latest_checkin is None:
        missing_flags.append("mental_checkin")
    if latest_report is None:
        missing_flags.append("mock_analysis")

    status = "ready"
    if latest_session is None:
        status = "needs_data"
    elif latest_session.rpe >= 8 or latest_session.mood_recovery <= 5:
        status = "watch"
    elif missing_flags:
        status = "incomplete"

    next_action = _next_action(missing_flags, latest_session)
    return PilotBoardCard(
        swimmer=swimmer,
        latest_session=latest_session,
        latest_report=latest_report,
        latest_checkin=latest_checkin,
        latest_race=latest_race,
        missing_flags=missing_flags,
        next_action=next_action,
        status=status,
    )


def _next_action(missing_flags: list[str], latest_session: Session | None) -> str:
    if "log_session" in missing_flags:
        return "Log first session"
    if latest_session and (latest_session.rpe >= 8 or latest_session.mood_recovery <= 5):
        return "Review recovery"
    if "upload_video" in missing_flags:
        return "Upload technique video"
    if "mental_checkin" in missing_flags:
        return "Capture mental check-in"
    if "mock_analysis" in missing_flags:
        return "Run mock analysis"
    return "Ready for next session"
