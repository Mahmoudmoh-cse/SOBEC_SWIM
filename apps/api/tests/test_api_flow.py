from datetime import date, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.v1 import sessions as sessions_api
from app.models import AIOutput
from app.providers.technique import MockTechniqueAnalyzer


def test_primary_api_flow(client: TestClient, auth_headers: dict[str, str], db) -> None:
    swimmer_response = client.post(
        "/api/v1/swimmers",
        headers=auth_headers,
        json={
            "name": "Maya Hassan",
            "date_of_birth": "2009-04-12",
            "primary_stroke": "freestyle",
            "primary_event": "100m freestyle",
            "level": "age_group",
            "personal_bests": {"100m freestyle": 61.84},
        },
    )
    assert swimmer_response.status_code == 201
    swimmer_id = swimmer_response.json()["id"]

    session_response = client.post(
        "/api/v1/sessions",
        headers=auth_headers,
        json={
            "swimmer_id": swimmer_id,
            "session_date": date.today().isoformat(),
            "session_type": "technique",
            "distance_m": 3400,
            "duration_min": 82,
            "rpe": 7,
            "mood_focus": 8,
            "mood_confidence": 7,
            "mood_energy": 7,
            "mood_calm": 8,
            "mood_recovery": 6,
            "mood_motivation": 8,
            "sleep_hours": 7.5,
            "notes": "Strong aerobic rhythm.",
        },
    )
    assert session_response.status_code == 201
    session = session_response.json()
    assert session["load_score"] == 23.8

    upload_response = client.post(
        f"/api/v1/sessions/{session['id']}/video",
        headers=auth_headers,
        files={"file": ("stroke.mp4", b"fake-video", "video/mp4")},
    )
    assert upload_response.status_code == 200
    upload_payload = upload_response.json()
    assert upload_payload["job"]["status"] == "complete"
    assert upload_payload["report"]["processing_status"] == "complete"
    assert upload_payload["report"]["analysis_status"] == "completed"
    assert upload_payload["report"]["frames_analyzed"] >= 0
    assert upload_payload["report"]["pose_detected_frames"] >= 0
    assert upload_payload["report"]["pose_detection_rate"] >= 0
    assert upload_payload["report"]["confidence_label"] in {"high", "medium", "low"}
    assert upload_payload["report"]["analysis_overlay_video_url"] is None
    assert isinstance(upload_payload["report"]["analysis_frame_urls"], list)
    assert upload_payload["report"]["analysis_events"]

    plan_response = client.post(
        "/api/v1/training-plans/generate",
        headers=auth_headers,
        json={
            "swimmer_id": swimmer_id,
            "race_date": (date.today() + timedelta(days=56)).isoformat(),
            "race_event": "100m freestyle",
            "target_time_seconds": 61.2,
        },
    )
    assert plan_response.status_code == 201
    assert plan_response.json()["current_phase"] == "base"

    race_response = client.post(
        "/api/v1/race-analyses",
        headers=auth_headers,
        json={
            "swimmer_id": swimmer_id,
            "race_date": date.today().isoformat(),
            "event": "100m freestyle",
            "official_time_seconds": 61.5,
            "splits_actual": [15.1, 15.5, 15.7, 15.2],
            "reaction_time_ms": 720,
            "turn_times": [{"position_m": 50, "contact_time_ms": 310}],
        },
    )
    assert race_response.status_code == 201
    assert race_response.json()["strategy_type"] in {"even", "negative", "positive", "custom"}

    mental_response = client.post(
        "/api/v1/mental-checkins",
        headers=auth_headers,
        json={
            "swimmer_id": swimmer_id,
            "checkin_type": "pre_race",
            "mood_focus": 8,
            "mood_confidence": 7,
            "mood_energy": 7,
            "mood_calm": 6,
            "mood_recovery": 7,
            "mood_motivation": 9,
        },
    )
    assert mental_response.status_code == 201
    assert mental_response.json()["composite_score"] > 0

    dashboard_response = client.get("/api/v1/dashboard/overview", headers=auth_headers)
    assert dashboard_response.status_code == 200
    assert dashboard_response.json()["swimmers_count"] == 1

    ai_outputs = list(db.scalars(select(AIOutput)))
    output_types = {output.output_type for output in ai_outputs}
    assert {"technique_summary", "training_rationale", "race_debrief", "mental_routine"}.issubset(output_types)
    assert all(output.parsed_json["summary"] for output in ai_outputs)

    ai_output_response = client.get(f"/api/v1/ai-outputs?swimmer_id={swimmer_id}", headers=auth_headers)
    assert ai_output_response.status_code == 200
    assert {item["output_type"] for item in ai_output_response.json()} == output_types

    backfill_response = client.post(f"/api/v1/ai-outputs/backfill/{swimmer_id}", headers=auth_headers)
    assert backfill_response.status_code == 200
    assert backfill_response.json() == []


def test_failed_video_processing_returns_trust_fields(client: TestClient, auth_headers: dict[str, str], monkeypatch) -> None:
    class BrokenAnalyzer:
        def analyze(self, *_args):
            raise RuntimeError("boom")

    swimmer_response = client.post(
        "/api/v1/swimmers",
        headers=auth_headers,
        json={
            "name": "Omar Nabil",
            "date_of_birth": "2010-01-10",
            "primary_stroke": "freestyle",
            "primary_event": "50m freestyle",
            "level": "age_group",
            "personal_bests": {"50m freestyle": 28.2},
        },
    )
    swimmer_id = swimmer_response.json()["id"]
    session_response = client.post(
        "/api/v1/sessions",
        headers=auth_headers,
        json={
            "swimmer_id": swimmer_id,
            "session_date": date.today().isoformat(),
            "session_type": "technique",
            "distance_m": 1800,
            "duration_min": 45,
            "rpe": 6,
            "mood_focus": 7,
            "mood_confidence": 7,
            "mood_energy": 7,
            "mood_calm": 7,
            "mood_recovery": 7,
            "mood_motivation": 7,
        },
    )
    session_id = session_response.json()["id"]
    monkeypatch.setattr(sessions_api, "get_technique_analyzer", lambda: BrokenAnalyzer())

    upload_response = client.post(
        f"/api/v1/sessions/{session_id}/video",
        headers={**auth_headers, "content-type": "video/mp4", "x-filename": "broken.mp4"},
        content=b"not-a-real-video",
    )

    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["job"]["status"] == "failed"
    assert payload["report"]["processing_status"] == "failed"
    assert payload["report"]["analysis_status"] == "failed"
    assert payload["report"]["confidence_label"] == "low"
    assert payload["report"]["analysis_error"]
    assert payload["report"]["analysis_overlay_video_url"] is None
    assert payload["report"]["analysis_frame_urls"] == []
    assert payload["report"]["analysis_events"][0]["type"] == "processing_error"


def test_video_upload_response_includes_explainability_assets(client: TestClient, auth_headers: dict[str, str], monkeypatch) -> None:
    class ExplainableAnalyzer:
        def analyze(self, swimmer, session, uploaded_file):
            payload = MockTechniqueAnalyzer().analyze(swimmer, session, uploaded_file)
            payload.update(
                {
                    "analysis_overlay_video_url": "test_uploads/processed_videos/file-1_analysis_overlay.mp4",
                    "analysis_frame_urls": ["test_uploads/analysis_frames/file-1_frame_10.jpg"],
                    "analysis_events": [
                        {
                            "timestamp_s": 0.0,
                            "type": "pose_detected",
                            "label": "Pose detected",
                            "message": "MediaPipe found a swimmer pose.",
                        },
                        {
                            "timestamp_s": 3.2,
                            "type": "low_confidence",
                            "label": "Low confidence",
                            "message": "Pose visibility dropped.",
                        },
                    ],
                }
            )
            return payload

    swimmer_response = client.post(
        "/api/v1/swimmers",
        headers=auth_headers,
        json={
            "name": "Lina Samir",
            "date_of_birth": "2011-06-05",
            "primary_stroke": "backstroke",
            "primary_event": "100m backstroke",
            "level": "age_group",
            "personal_bests": {"100m backstroke": 69.4},
        },
    )
    swimmer_id = swimmer_response.json()["id"]
    session_response = client.post(
        "/api/v1/sessions",
        headers=auth_headers,
        json={
            "swimmer_id": swimmer_id,
            "session_date": date.today().isoformat(),
            "session_type": "technique",
            "distance_m": 2200,
            "duration_min": 55,
            "rpe": 6,
            "mood_focus": 7,
            "mood_confidence": 7,
            "mood_energy": 7,
            "mood_calm": 7,
            "mood_recovery": 7,
            "mood_motivation": 7,
        },
    )
    monkeypatch.setattr(sessions_api, "get_technique_analyzer", lambda: ExplainableAnalyzer())

    upload_response = client.post(
        f"/api/v1/sessions/{session_response.json()['id']}/video",
        headers=auth_headers,
        files={"file": ("stroke.mp4", b"fake-video", "video/mp4")},
    )

    assert upload_response.status_code == 200
    report = upload_response.json()["report"]
    assert report["analysis_overlay_video_url"].endswith("_analysis_overlay.mp4")
    assert report["analysis_frame_urls"] == ["test_uploads/analysis_frames/file-1_frame_10.jpg"]
    assert [event["type"] for event in report["analysis_events"]] == ["pose_detected", "low_confidence"]
