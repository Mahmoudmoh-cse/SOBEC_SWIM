from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.api.v1.utils import get_owned_swimmer
from app.db.session import get_db
from app.models import TechniqueReport, User
from app.schemas.common import TechniqueReportRead

router = APIRouter(prefix="/technique-reports", tags=["technique reports"])


@router.get("", response_model=list[TechniqueReportRead])
def list_technique_reports(
    swimmer_id: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[TechniqueReport]:
    statement = select(TechniqueReport).join(TechniqueReport.swimmer).where(
        TechniqueReport.swimmer.has(coach_id=current_user.id)
    )
    if swimmer_id:
        get_owned_swimmer(db, swimmer_id, current_user)
        statement = statement.where(TechniqueReport.swimmer_id == swimmer_id)
    return list(db.scalars(statement.order_by(TechniqueReport.created_at.desc())))


@router.get("/{report_id}", response_model=TechniqueReportRead)
def get_technique_report(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TechniqueReport:
    report = db.get(TechniqueReport, report_id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Technique report not found")
    get_owned_swimmer(db, report.swimmer_id, current_user)
    return report
