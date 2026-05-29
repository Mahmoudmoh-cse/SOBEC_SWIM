from io import BytesIO

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import SwimAnalysis


def test_swim_analysis_upload_rejects_invalid_file_type(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        "/api/swim-analysis/upload",
        headers=auth_headers,
        files={
            "side_video": ("side.txt", BytesIO(b"not a video"), "text/plain"),
            "front_video": ("front.mp4", BytesIO(b"fake mp4"), "video/mp4"),
        },
        data={"stroke_type": "freestyle", "target_fps": "30", "quality_mode": "high_accuracy"},
    )

    assert response.status_code == 400
    assert "side_video" in response.json()["detail"]


def test_swim_analysis_status_requires_existing_analysis(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.get("/api/swim-analysis/missing-analysis", headers=auth_headers)

    assert response.status_code == 404


def test_swim_analysis_new_engine_endpoints_return_stable_json(client: TestClient, db: Session, auth_headers: dict[str, str]) -> None:
    analysis = SwimAnalysis(
        id="analysis-engine-1",
        stroke_type="freestyle",
        status="completed",
        progress=100,
        pose_backend="synthetic",
        side_video_path="side.mp4",
        front_video_path="front.mp4",
        report_json={
            "analysis_id": "analysis-engine-1",
            "status": "completed",
            "summary": {"overall_score": 80, "confidence_score": 0.7, "data_quality_score": 0.8},
            "video_quality": {},
            "metrics": {"stroke_rate_spm": {"value": 40, "confidence": 0.7, "unit": "cycles/min", "evidence": {}, "reason": "test"}},
            "calibration": {"confidence": 0.7},
            "velocity": {"summary": {"average_velocity": 1.4}, "series": []},
            "phase_segments": [{"type": "free_swim", "start_sec": 1, "end_sec": 5, "confidence": 0.7, "reason": "test"}],
            "faults": [],
            "findings": [],
            "recommendations": [],
            "coaching_report": {},
            "artifacts": {"side_annotated_video_url": "/uploads/side.mp4", "front_annotated_video_url": None},
        },
    )
    db.add(analysis)
    db.commit()

    metrics = client.get("/api/swim-analysis/analysis-engine-1/metrics", headers=auth_headers)
    velocity = client.get("/api/swim-analysis/analysis-engine-1/velocity", headers=auth_headers)
    annotated = client.get("/api/swim-analysis/analysis-engine-1/annotated-video", headers=auth_headers)

    assert metrics.status_code == 200
    assert metrics.json()["metrics"]["stroke_rate_spm"]["value"] == 40
    assert velocity.status_code == 200
    assert velocity.json()["phase_segments"][0]["type"] == "free_swim"
    assert annotated.status_code == 200
    assert annotated.json()["videos"]["side"] == "/uploads/side.mp4"


def test_annotated_video_payload_exposes_single_video_alias(client: TestClient, db: Session, auth_headers: dict[str, str]) -> None:
    analysis = SwimAnalysis(
        id="analysis-single-video",
        stroke_type="freestyle",
        status="completed",
        progress=100,
        pose_backend="synthetic",
        side_video_path="video.mp4",
        front_video_path="video.mp4",
        report_json={
            "analysis_id": "analysis-single-video",
            "status": "completed",
            "summary": {"overall_score": 80, "confidence_score": 0.7, "data_quality_score": 0.8},
            "video_metadata": {"input_mode": "single_video"},
            "video_quality": {"video": {}},
            "metrics": {},
            "findings": [],
            "recommendations": [],
            "artifacts": {"video_annotated_video_url": "/uploads/video.mp4", "side_annotated_video_url": "/uploads/video.mp4", "front_annotated_video_url": None},
        },
    )
    db.add(analysis)
    db.commit()

    annotated = client.get("/api/swim-analysis/analysis-single-video/annotated-video", headers=auth_headers)

    assert annotated.status_code == 200
    assert annotated.json()["videos"]["video"] == "/uploads/video.mp4"
    assert annotated.json()["videos"]["front"] is None
