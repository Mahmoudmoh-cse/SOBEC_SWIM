from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models import CalibrationData, PhaseSegment, SwimAnalysis, SwimFault, SwimMetric, User, VelocitySeries
from app.models.entities import new_id, utc_now
from app.swim_analysis.models import SwimAnalysisStatus
from app.swim_analysis.preprocessing import ALLOWED_VIDEO_SUFFIXES, validate_video_file
from app.swim_analysis.processor import SwimAnalysisProcessor


class SwimAnalysisService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()

    async def create_analysis(
        self,
        *,
        video: UploadFile | None = None,
        side_video: UploadFile | None = None,
        front_video: UploadFile | None = None,
        swimmer_id: str | None,
        stroke_type: str,
        current_user: User,
    ) -> SwimAnalysis:
        if swimmer_id:
            self._assert_swimmer_owned(swimmer_id, current_user)

        primary_video = video or side_video
        if primary_video is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload one video file.")

        primary_field = "video" if video is not None else "side_video"
        self._validate_upload(primary_video, field_name=primary_field)
        if front_video is not None:
            self._validate_upload(front_video, field_name="front_video")

        analysis_id = new_id()
        target_dir = Path(self.settings.upload_dir) / "swim_analysis" / analysis_id / "original"
        target_dir.mkdir(parents=True, exist_ok=True)
        side_path = await self._save_upload(primary_video, target_dir, "video")
        front_path = await self._save_upload(front_video, target_dir, "front") if front_video is not None else side_path

        try:
            validate_video_file(side_path)
            if front_path != side_path:
                validate_video_file(front_path)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        analysis = SwimAnalysis(
            id=analysis_id,
            swimmer_id=swimmer_id,
            stroke_type=stroke_type or "freestyle",
            status=SwimAnalysisStatus.QUEUED,
            progress=0,
            pose_backend=self.settings.pose_backend,
            side_video_path=str(side_path),
            front_video_path=str(front_path),
            report_json={},
        )
        self.db.add(analysis)
        self.db.commit()
        self.db.refresh(analysis)
        return analysis

    def get_analysis_for_user(self, analysis_id: str, current_user: User) -> SwimAnalysis:
        analysis = self.db.get(SwimAnalysis, analysis_id)
        if analysis is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Swim analysis not found")
        if analysis.swimmer_id:
            self._assert_swimmer_owned(analysis.swimmer_id, current_user)
        return analysis

    def status_payload(self, analysis: SwimAnalysis) -> dict:
        self.mark_stale_analysis_if_needed(analysis)
        report = analysis.report_json or {}
        return {
            "id": analysis.id,
            "status": analysis.status,
            "progress": analysis.progress,
            "stroke_type": analysis.stroke_type,
            "pose_backend": analysis.pose_backend,
            "swimmer_id": analysis.swimmer_id,
            "video_metadata": report.get("video_metadata", {}),
            "processing_errors": report.get("processing_errors", []),
            "error_message": analysis.error_message,
            "created_at": analysis.created_at,
            "updated_at": analysis.updated_at,
        }

    def report_payload(self, analysis: SwimAnalysis, *, debug: bool = False) -> dict:
        self.mark_stale_analysis_if_needed(analysis)
        if not analysis.report_json or analysis.status not in {SwimAnalysisStatus.COMPLETED, SwimAnalysisStatus.FAILED}:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Swim analysis report is not ready yet.")
        payload = dict(analysis.report_json)
        if not debug:
            payload["debug_info"] = {}
        return payload

    def mark_stale_analysis_if_needed(self, analysis: SwimAnalysis) -> None:
        if analysis.status not in {SwimAnalysisStatus.QUEUED, SwimAnalysisStatus.PROCESSING}:
            return
        updated_at = analysis.updated_at
        if updated_at.tzinfo is None:
            now = utc_now().replace(tzinfo=None)
        else:
            now = utc_now()
        if now - updated_at < timedelta(minutes=8):
            return

        message = "Analysis was interrupted before completion. Restart the analysis after confirming the API server is not reloading."
        analysis.status = SwimAnalysisStatus.FAILED
        analysis.progress = 100
        analysis.error_message = message
        analysis.report_json = {
            "analysis_id": analysis.id,
            "status": SwimAnalysisStatus.FAILED,
            "summary": {
                "stroke_type": analysis.stroke_type,
                "duration_sec": 0,
                "overall_score": 0,
                "confidence_score": 0,
                "data_quality_score": 0,
                "pose_backend": analysis.pose_backend,
            },
            "video_quality": {},
            "metrics": {},
            "findings": [],
            "recommendations": [
                {
                    "priority": 1,
                    "message": "Restart this analysis. The previous worker was interrupted, so no technique conclusion was produced.",
                    "linked_metric": "confidence_score",
                }
            ],
            "artifacts": {},
            "processing_errors": [message],
        }
        analysis.updated_at = utc_now()
        self.db.commit()

    def _assert_swimmer_owned(self, swimmer_id: str, current_user: User) -> None:
        from app.models import Swimmer

        swimmer = self.db.get(Swimmer, swimmer_id)
        if swimmer is None or swimmer.coach_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Swimmer not found")

    def _validate_upload(self, file: UploadFile, *, field_name: str) -> None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must be MP4, MOV, M4V, WEBM, or AVI.",
            )
        if file.content_type and not (file.content_type.startswith("video/") or file.content_type == "application/octet-stream"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must be uploaded as a video file.",
            )

    async def _save_upload(self, file: UploadFile, target_dir: Path, view_name: str) -> Path:
        suffix = Path(file.filename or f"{view_name}.mp4").suffix.lower() or ".mp4"
        target_path = target_dir / f"{view_name}_{uuid4()}{suffix}"
        size = 0
        with target_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                handle.write(chunk)
        if size == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{view_name} video is empty.")
        return target_path


    def metrics_payload(self, analysis: SwimAnalysis) -> dict:
        self.mark_stale_analysis_if_needed(analysis)
        report = analysis.report_json or {}
        return {
            "analysis_id": analysis.id,
            "status": analysis.status,
            "metrics": report.get("metrics", {}),
            "faults": report.get("faults", []),
            "calibration": report.get("calibration", {}),
        }

    def velocity_payload(self, analysis: SwimAnalysis) -> dict:
        self.mark_stale_analysis_if_needed(analysis)
        report = analysis.report_json or {}
        return {
            "analysis_id": analysis.id,
            "status": analysis.status,
            "velocity": report.get("velocity", {}),
            "phase_segments": report.get("phase_segments", []),
            "calibration": report.get("calibration", {}),
        }

    def annotated_video_payload(self, analysis: SwimAnalysis) -> dict:
        self.mark_stale_analysis_if_needed(analysis)
        report = analysis.report_json or {}
        artifacts = report.get("artifacts", {})
        metadata = report.get("video_metadata", {})
        single_video = isinstance(metadata, dict) and metadata.get("input_mode") == "single_video"
        videos = {
            "side": artifacts.get("side_annotated_video_url"),
            "front": artifacts.get("front_annotated_video_url"),
        }
        if single_video:
            videos["video"] = artifacts.get("video_annotated_video_url") or artifacts.get("side_annotated_video_url")
        return {
            "analysis_id": analysis.id,
            "status": analysis.status,
            "videos": videos,
            "confidence_warning": _confidence_warning(report),
        }


def process_swim_analysis_job(
    analysis_id: str,
    *,
    target_fps: int = 6,
    quality_mode: str = "balanced",
    lane_length_m: float | None = None,
) -> None:
    db = SessionLocal()
    try:
        analysis = db.get(SwimAnalysis, analysis_id)
        if analysis is None:
            return
        analysis.status = SwimAnalysisStatus.PROCESSING
        analysis.progress = 5
        analysis.updated_at = utc_now()
        db.commit()

        def update_progress(value: int) -> None:
            fresh = db.get(SwimAnalysis, analysis_id)
            if fresh is None:
                return
            fresh.progress = max(fresh.progress, min(99, value))
            fresh.updated_at = utc_now()
            db.commit()

        processor = SwimAnalysisProcessor(target_fps=target_fps, quality_mode=quality_mode, lane_length_m=lane_length_m)
        result = processor.process(analysis, progress_callback=update_progress)
        analysis = db.get(SwimAnalysis, analysis_id)
        if analysis is None:
            return
        analysis.status = SwimAnalysisStatus.COMPLETED
        analysis.progress = 100
        analysis.pose_backend = result["pose_backend"]
        analysis.side_annotated_path = result["side_annotated_path"]
        analysis.front_annotated_path = result["front_annotated_path"]
        analysis.raw_landmarks_path = result["raw_landmarks_path"]
        analysis.smoothed_landmarks_path = result["smoothed_landmarks_path"]
        analysis.report_json = result["report"]
        analysis.error_message = None
        analysis.updated_at = utc_now()
        _replace_structured_outputs(db, analysis, result)
        db.commit()
    except Exception as exc:
        analysis = db.get(SwimAnalysis, analysis_id)
        if analysis is not None:
            message = str(exc)
            analysis.status = SwimAnalysisStatus.FAILED
            analysis.progress = 100
            analysis.error_message = message
            analysis.report_json = {
                "analysis_id": analysis.id,
                "status": SwimAnalysisStatus.FAILED,
                "summary": {
                    "stroke_type": analysis.stroke_type,
                    "duration_sec": 0,
                    "overall_score": 0,
                    "confidence_score": 0,
                    "data_quality_score": 0,
                    "pose_backend": analysis.pose_backend,
                },
                "video_quality": {},
                "metrics": {},
                "findings": [],
                "recommendations": [
                    {
                        "priority": 1,
                        "message": "Resolve the processing error and rerun the analysis before drawing technique conclusions.",
                        "linked_metric": "confidence_score",
                    }
                ],
                "artifacts": {},
                "processing_errors": [message],
            }
            analysis.updated_at = utc_now()
            db.commit()
    finally:
        db.close()


def _replace_structured_outputs(db: Session, analysis: SwimAnalysis, result: dict) -> None:
    for collection in [analysis.metrics, analysis.faults, analysis.velocity_series, analysis.phase_segments]:
        collection.clear()
    if analysis.calibration_data is not None:
        db.delete(analysis.calibration_data)
        db.flush()

    for name, metric in result.get("metrics", {}).items():
        if not isinstance(metric, dict):
            continue
        db.add(
            SwimMetric(
                analysis_id=analysis.id,
                metric_name=name,
                value_json=metric,
                unit=metric.get("unit"),
                confidence=_numeric_confidence(metric.get("confidence"), metric.get("evidence", {})),
                interpretation=metric.get("interpretation"),
            )
        )

    for fault in result.get("faults", []):
        db.add(
            SwimFault(
                analysis_id=analysis.id,
                fault_name=str(fault.get("name", "technique_fault")),
                severity=str(fault.get("severity", "medium")),
                confidence=float(fault.get("confidence", 0) or 0),
                timestamp_range=fault.get("timestamp_range", []),
                evidence_json=fault.get("evidence_metrics", {}),
                recommended_drill=fault.get("recommended_drill"),
                coach_explanation=fault.get("coach_explanation"),
            )
        )

    velocity = result.get("velocity", {})
    db.add(
        VelocitySeries(
            analysis_id=analysis.id,
            series_json=velocity,
            summary_json=velocity.get("summary", {}) if isinstance(velocity, dict) else {},
        )
    )

    for phase in result.get("phases", []):
        db.add(
            PhaseSegment(
                analysis_id=analysis.id,
                segment_type=str(phase.get("type", "phase")),
                start_sec=float(phase.get("start_sec", 0) or 0),
                end_sec=float(phase.get("end_sec", 0) or 0),
                confidence=float(phase.get("confidence", 0) or 0),
                reason=phase.get("reason"),
                segment_json=phase,
            )
        )

    calibration = result.get("calibration", {})
    db.add(
        CalibrationData(
            analysis_id=analysis.id,
            method=str(calibration.get("method", "unknown")),
            lane_length_m=calibration.get("lane_length_m"),
            pixel_to_meter=calibration.get("pixel_to_meter"),
            confidence=float(calibration.get("confidence", 0) or 0),
            calibration_json=calibration,
        )
    )


def _numeric_confidence(value: object, evidence: dict | None = None) -> float:
    if value == "low":
        return float((evidence or {}).get("numeric_confidence", 0) or 0)
    return float(value or 0)


def _confidence_warning(report: dict) -> str | None:
    summary = report.get("summary", {})
    confidence = float(summary.get("confidence_score", 0) or 0)
    if confidence < 0.45:
        return "Annotated video is low confidence; use it for review, not final conclusions."
    return None
