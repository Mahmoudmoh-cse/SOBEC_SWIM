from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.ai.service import AICoachingService
from app.api.deps import get_current_user
from app.api.v1.utils import get_owned_swimmer
from app.db.session import get_db
from app.models import AnalysisJob, Session, TechniqueReport, UploadedFile, User
from app.providers.storage import get_storage_provider
from app.providers.technique import PROCESSING_ERROR_MESSAGE, failed_technique_report_payload, get_technique_analyzer
from app.schemas.common import SessionCreate, SessionRead, VideoUploadResponse
from app.services.calculations import calculate_session_load

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionRead])
def list_sessions(
    swimmer_id: str | None = None,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Session]:
    statement = select(Session).join(Session.swimmer).where(Session.swimmer.has(coach_id=current_user.id))
    if swimmer_id:
        get_owned_swimmer(db, swimmer_id, current_user)
        statement = statement.where(Session.swimmer_id == swimmer_id)
    return list(db.scalars(statement.order_by(Session.session_date.desc(), Session.created_at.desc())))


@router.post("", response_model=SessionRead, status_code=201)
def create_session(
    payload: SessionCreate,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Session:
    get_owned_swimmer(db, payload.swimmer_id, current_user)
    session = Session(
        **payload.model_dump(),
        load_score=calculate_session_load(payload.distance_m, payload.rpe),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/{session_id}", response_model=SessionRead)
def get_session(
    session_id: str,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Session:
    session = db.get(Session, session_id)
    if session is None:
        get_owned_swimmer(db, "missing", current_user)
    get_owned_swimmer(db, session.swimmer_id, current_user)
    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    get_owned_swimmer(db, session.swimmer_id, current_user)
    db.delete(session)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{session_id}/video", response_model=VideoUploadResponse)
async def upload_video(
    session_id: str,
    request: Request,
    db: DbSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VideoUploadResponse:
    session = db.get(Session, session_id)
    if session is None:
        get_owned_swimmer(db, "missing", current_user)
    swimmer = get_owned_swimmer(db, session.swimmer_id, current_user)

    storage = get_storage_provider()
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form(max_files=1, max_fields=4)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not parse the upload form. Use the web uploader again or send the video as a raw file body.",
            ) from exc
        file = form.get("file")
        if not file or not hasattr(file, "filename") or not hasattr(file, "read"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload must include a video file field named 'file'.")
        stored = await storage.save_video(file, swimmer.id, session.id)
    else:
        data = await request.body()
        if not data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload body is empty.")
        filename = unquote(request.headers.get("x-filename") or "video.mp4")
        stored = await storage.save_video_bytes(data, filename, content_type or None, swimmer.id, session.id)

    uploaded_file = UploadedFile(
        session_id=session.id,
        swimmer_id=swimmer.id,
        original_filename=stored.original_filename,
        stored_path=stored.stored_path,
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
    )
    db.add(uploaded_file)
    db.flush()

    job = AnalysisJob(
        session_id=session.id,
        swimmer_id=swimmer.id,
        uploaded_file_id=uploaded_file.id,
        status="processing",
    )
    db.add(job)
    db.flush()

    try:
        report_payload = get_technique_analyzer().analyze(swimmer, session, uploaded_file)
    except Exception:
        report_payload = failed_technique_report_payload(swimmer, uploaded_file, PROCESSING_ERROR_MESSAGE)
    report = TechniqueReport(
        session_id=session.id,
        swimmer_id=swimmer.id,
        **report_payload,
    )
    db.add(report)
    db.flush()
    ai_output = AICoachingService(db).technique_summary(swimmer, session, report)
    if ai_output.parsed_json.get("summary"):
        report.coaching_summary = ai_output.parsed_json["summary"]
    session.video_url = stored.stored_path
    job.status = "failed" if report_payload["analysis_status"] == "failed" else "complete"
    job.error_message = report_payload.get("analysis_error")
    db.commit()
    db.refresh(uploaded_file)
    db.refresh(job)
    db.refresh(report)
    return VideoUploadResponse(file=uploaded_file, job=job, report=report)
