from sqlalchemy import func, select
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import Session as SwimSession
from app.models import Swimmer, TrainingPlan, User
from app.schemas.common import DashboardOverview

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/overview", response_model=DashboardOverview)
def overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DashboardOverview:
    swimmers = list(db.scalars(select(Swimmer).where(Swimmer.coach_id == current_user.id).order_by(Swimmer.name)))
    swimmer_ids = [swimmer.id for swimmer in swimmers]

    if swimmer_ids:
        sessions = list(
            db.scalars(
                select(SwimSession)
                .where(SwimSession.swimmer_id.in_(swimmer_ids))
                .order_by(SwimSession.session_date.desc(), SwimSession.created_at.desc())
                .limit(8)
            )
        )
        sessions_count = db.scalar(select(func.count(SwimSession.id)).where(SwimSession.swimmer_id.in_(swimmer_ids))) or 0
        average_load = db.scalar(select(func.avg(SwimSession.load_score)).where(SwimSession.swimmer_id.in_(swimmer_ids))) or 0
        active_plans = list(
            db.scalars(
                select(TrainingPlan)
                .where(TrainingPlan.swimmer_id.in_(swimmer_ids), TrainingPlan.is_active.is_(True))
                .order_by(TrainingPlan.created_at.desc())
            )
        )
    else:
        sessions = []
        sessions_count = 0
        average_load = 0
        active_plans = []

    alerts = []
    for session in sessions:
        swimmer = next((item for item in swimmers if item.id == session.swimmer_id), None)
        if session.rpe >= 8 and session.mood_recovery <= 5:
            alerts.append(
                {
                    "type": "overtraining_risk",
                    "swimmer_id": session.swimmer_id,
                    "swimmer_name": swimmer.name if swimmer else "Unknown swimmer",
                    "message": "High RPE paired with low recovery on the latest session.",
                }
            )
        if session.video_url is None:
            alerts.append(
                {
                    "type": "video_missing",
                    "swimmer_id": session.swimmer_id,
                    "swimmer_name": swimmer.name if swimmer else "Unknown swimmer",
                    "message": "No technique video attached to this recent session.",
                }
            )
        if len(alerts) >= 5:
            break

    return DashboardOverview(
        swimmers_count=len(swimmers),
        sessions_count=sessions_count,
        average_load=round(float(average_load), 2),
        high_risk_count=sum(1 for alert in alerts if alert["type"] == "overtraining_risk"),
        swimmers=swimmers,
        recent_sessions=sessions,
        active_plans=active_plans,
        alerts=alerts,
    )
