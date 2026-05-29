from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import User
from app.swim_analysis.schemas import (
    SwimAnalysisAnnotatedVideoResponse,
    SwimAnalysisMetricsResponse,
    SwimAnalysisReportResponse,
    SwimAnalysisStatusResponse,
    SwimAnalysisUploadResponse,
    SwimAnalysisVelocityResponse,
)
from app.swim_analysis.service import SwimAnalysisService, process_swim_analysis_job

router = APIRouter(prefix="/swim-analysis", tags=["swim-analysis"])


@router.post("/upload", response_model=SwimAnalysisUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_swim_analysis(
    background_tasks: BackgroundTasks,
    video: UploadFile | None = File(default=None),
    side_video: UploadFile | None = File(default=None),
    front_video: UploadFile | None = File(default=None),
    swimmer_id: str | None = Form(default=None),
    stroke_type: str = Form(default="freestyle"),
    target_fps: int = Form(default=6),
    quality_mode: str = Form(default="balanced"),
    lane_length_m: float | None = Form(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SwimAnalysisUploadResponse:
    analysis = await SwimAnalysisService(db).create_analysis(
        video=video,
        side_video=side_video,
        front_video=front_video,
        swimmer_id=swimmer_id,
        stroke_type=stroke_type,
        current_user=current_user,
    )
    background_tasks.add_task(
        process_swim_analysis_job,
        analysis.id,
        target_fps=target_fps,
        quality_mode=quality_mode,
        lane_length_m=lane_length_m,
    )
    return SwimAnalysisUploadResponse(analysis_id=analysis.id, status="queued")


@router.get("/{analysis_id}", response_model=SwimAnalysisStatusResponse)
def get_swim_analysis_status(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    service = SwimAnalysisService(db)
    analysis = service.get_analysis_for_user(analysis_id, current_user)
    return service.status_payload(analysis)


@router.get("/{analysis_id}/metrics", response_model=SwimAnalysisMetricsResponse)
def get_swim_analysis_metrics(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    service = SwimAnalysisService(db)
    analysis = service.get_analysis_for_user(analysis_id, current_user)
    return service.metrics_payload(analysis)


@router.get("/{analysis_id}/velocity", response_model=SwimAnalysisVelocityResponse)
def get_swim_analysis_velocity(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    service = SwimAnalysisService(db)
    analysis = service.get_analysis_for_user(analysis_id, current_user)
    return service.velocity_payload(analysis)


@router.get("/{analysis_id}/report", response_model=SwimAnalysisReportResponse)
def get_swim_analysis_report(
    analysis_id: str,
    debug: bool = Query(default=False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    service = SwimAnalysisService(db)
    analysis = service.get_analysis_for_user(analysis_id, current_user)
    return service.report_payload(analysis, debug=debug)


@router.get("/{analysis_id}/annotated-video", response_model=SwimAnalysisAnnotatedVideoResponse)
def get_swim_analysis_annotated_video(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    service = SwimAnalysisService(db)
    analysis = service.get_analysis_for_user(analysis_id, current_user)
    return service.annotated_video_payload(analysis)
