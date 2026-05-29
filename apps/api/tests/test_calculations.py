from datetime import date, timedelta
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.models import Session, Swimmer, UploadedFile
from app.providers.technique import PROCESSING_ERROR_MESSAGE, MediaPipeTechniqueAnalyzer, MockTechniqueAnalyzer, build_analysis_trust
from app.services.calculations import calculate_mental_composite, calculate_session_load
from app.services.race import classify_strategy, predicted_splits, strategy_score
from app.services.training import generate_training_plan


def test_session_load_calculation() -> None:
    assert calculate_session_load(3600, 7) == 25.2


def test_mental_composite() -> None:
    assert calculate_mental_composite(
        {
            "mood_focus": 8,
            "mood_confidence": 7,
            "mood_energy": 6,
            "mood_calm": 7,
            "mood_recovery": 6,
            "mood_motivation": 8,
        }
    ) == 7


def test_training_plan_generation() -> None:
    plan = generate_training_plan(
        race_date=date(2026, 7, 15),
        race_event="100m freestyle",
        target_time_seconds=60.1,
        today=date(2026, 5, 20),
    )
    assert plan["weeks_total"] == 8
    assert len(plan["weekly_plans"]) == 8
    assert set(plan["phase_config"]) == {"base", "build", "peak", "taper"}


def test_race_strategy_helpers() -> None:
    splits = [15.2, 15.4, 15.1, 14.9]
    predicted = predicted_splits(60.4, 4)
    assert classify_strategy(splits) == "negative"
    assert strategy_score(splits, predicted) >= 90


def test_mock_technique_analyzer_generates_report() -> None:
    swimmer = Swimmer(
        id="swimmer-1",
        coach_id="coach-1",
        name="Maya",
        date_of_birth=date(2009, 1, 1),
        primary_stroke="freestyle",
        primary_event="100m freestyle",
        level="age_group",
        personal_bests={"100m freestyle": 61.2},
        technique_profile={},
        mental_profile={},
    )
    session = Session(
        id="session-1",
        swimmer_id="swimmer-1",
        session_date=date.today() - timedelta(days=1),
        session_type="technique",
        distance_m=3200,
        duration_min=80,
        rpe=7,
        load_score=22.4,
        mood_focus=8,
        mood_confidence=7,
        mood_energy=7,
        mood_calm=8,
        mood_recovery=6,
        mood_motivation=8,
    )
    uploaded_file = UploadedFile(
        id="file-1",
        session_id="session-1",
        swimmer_id="swimmer-1",
        original_filename="stroke.mp4",
        stored_path="uploads/stroke.mp4",
        content_type="video/mp4",
        size_bytes=128,
    )

    report = MockTechniqueAnalyzer().analyze(swimmer, session, uploaded_file)

    assert report["processing_status"] == "complete"
    assert report["analysis_status"] == "completed"
    assert report["confidence_label"] == "high"
    assert report["pose_detection_rate"] == 1
    assert report["analysis_overlay_video_url"] is None
    assert report["analysis_frame_urls"] == []
    assert report["analysis_events"][0]["type"] == "mock_analysis"
    assert report["overall_score"] > 0
    assert report["faults"][0]["title"] == "Dropped elbow on catch"


def test_analysis_trust_detects_no_pose() -> None:
    trust = build_analysis_trust(frames_total=120, frames_analyzed=30, pose_detected_frames=0)

    assert trust["analysis_status"] == "no_pose_detected"
    assert trust["confidence_label"] == "low"
    assert trust["pose_detection_rate"] == 0
    assert "No clear swimmer pose" in trust["analysis_warning"]


def test_analysis_trust_detects_low_confidence() -> None:
    trust = build_analysis_trust(frames_total=120, frames_analyzed=30, pose_detected_frames=8, average_visibility=0.35)

    assert trust["analysis_status"] == "low_confidence"
    assert trust["confidence_label"] == "low"
    assert trust["pose_detection_rate"] == 0.27
    assert trust["analysis_warning"]


def test_analysis_trust_detects_failed_processing() -> None:
    trust = build_analysis_trust(
        frames_total=0,
        frames_analyzed=0,
        pose_detected_frames=0,
        processing_error=PROCESSING_ERROR_MESSAGE,
    )

    assert trust["analysis_status"] == "failed"
    assert trust["analysis_error"] == PROCESSING_ERROR_MESSAGE


def test_mediapipe_explainability_asset_helpers_create_dirs(tmp_path) -> None:
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")

    settings = get_settings()
    previous_upload_dir = settings.upload_dir
    settings.upload_dir = tmp_path
    try:
        analyzer = MediaPipeTechniqueAnalyzer()
        frame = analyzer.np.zeros((120, 160, 3), dtype=analyzer.np.uint8)

        overlay_path = analyzer._analysis_asset_path("processed_videos", "file-1_analysis_overlay.mp4")
        assert overlay_path.parent.exists()

        snapshots = analyzer._write_snapshot_frames("file-1", [(10, frame), (20, frame)])
        assert len(snapshots) == 2
        assert (tmp_path / "analysis_frames").exists()
        assert all((tmp_path / "analysis_frames" / Path(path).name).exists() for path in snapshots)
    finally:
        settings.upload_dir = previous_upload_dir
