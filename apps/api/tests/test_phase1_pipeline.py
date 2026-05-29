import pytest

from app.swim_analysis.phase1_metrics import compute_phase1_research_metrics
from app.swim_analysis.phase1_schemas import TrajectoryExport
from app.swim_analysis.pose.base import Keypoint, PoseFrameResult
from app.swim_analysis.roi import ROIConfig, centered_roi_from_bbox, project_pose_to_original
from app.swim_analysis.temporal_tracking import JointPhysicsThreshold, TemporalJointTracker, TemporalTrackingConfig


def test_roi_coordinate_mapping_projects_pose_back_to_original_frame() -> None:
    np = pytest.importorskip("numpy")
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    roi = centered_roi_from_bbox(frame, [50, 40, 90, 80], confidence=0.7, config=ROIConfig(scale=1.0, min_size_px=40))
    pose = PoseFrameResult(
        timestamp=0.0,
        frame_index=3,
        keypoints=[Keypoint("left_wrist", 10, 12, 0.9)],
        confidence=0.9,
        bbox=[8, 9, 22, 24],
        backend="synthetic",
    )

    projected = project_pose_to_original(pose, roi)

    assert projected.roi is not None
    assert projected.keypoints[0].x == pytest.approx(roi.x + 10)
    assert projected.keypoints[0].y == pytest.approx(roi.y + 12)
    assert projected.bbox == pytest.approx([roi.x + 8, roi.y + 9, roi.x + 22, roi.y + 24])


def test_missing_keypoint_handling_keeps_missing_as_missing() -> None:
    frames = [
        PoseFrameResult(0.0, 0, [Keypoint("left_wrist", 10, 10, 0.9), Keypoint("right_wrist", 30, 10, 0.05)], 0.5, backend="synthetic"),
        PoseFrameResult(0.1, 1, [Keypoint("left_wrist", 11, 10, 0.9)], 0.9, backend="synthetic"),
    ]

    result = TemporalJointTracker(TemporalTrackingConfig(enable_smoothing=False)).process(frames, view_type="side")
    raw = result.trajectory_export["raw_keypoints"]

    assert raw["right_wrist"][0]["visibility_state"] == "low_confidence"
    assert raw["right_wrist"][1]["visibility_state"] == "missing"
    assert raw["right_wrist"][1]["x"] is None


def test_physics_outlier_rejection_marks_impossible_joint_jump() -> None:
    config = TemporalTrackingConfig(
        enable_smoothing=False,
        core_threshold=JointPhysicsThreshold(20, 200, 1000),
        mid_threshold=JointPhysicsThreshold(20, 200, 1000),
        distal_threshold=JointPhysicsThreshold(20, 200, 1000),
    )
    frames = [
        PoseFrameResult(0.0, 0, [Keypoint("left_wrist", 10, 10, 0.9)], 0.9, backend="synthetic"),
        PoseFrameResult(0.1, 1, [Keypoint("left_wrist", 300, 300, 0.9)], 0.9, backend="synthetic"),
        PoseFrameResult(0.2, 2, [Keypoint("left_wrist", 12, 10, 0.9)], 0.9, backend="synthetic"),
    ]

    result = TemporalJointTracker(config).process(frames, view_type="side")

    assert result.rejected_outliers
    assert result.rejected_outliers[0]["joint"] == "left_wrist"
    assert result.rejected_outliers[0]["reason"] in {"max_pixel_displacement", "max_velocity"}
    assert result.trajectory_export["filtered_keypoints"]["left_wrist"][1]["visibility_state"] in {"missing", "interpolated"}


def test_short_missing_joint_gap_is_interpolated_and_marked() -> None:
    frames = [
        PoseFrameResult(0.0, 0, [Keypoint("left_wrist", 10, 10, 0.9)], 0.9, backend="synthetic"),
        PoseFrameResult(0.1, 1, [], 0.0, backend="synthetic"),
        PoseFrameResult(0.2, 2, [Keypoint("left_wrist", 20, 20, 0.9)], 0.9, backend="synthetic"),
    ]

    result = TemporalJointTracker(TemporalTrackingConfig(max_gap_frames=2, enable_smoothing=False)).process(frames, view_type="side")
    point = result.trajectory_export["interpolated_keypoints"]["left_wrist"][1]

    assert point["visibility_state"] == "interpolated"
    assert point["x"] == pytest.approx(15)
    assert point["confidence"] < 0.9


def test_left_right_swap_correction_uses_temporal_continuity() -> None:
    frames = [
        PoseFrameResult(0.0, 0, [Keypoint("left_wrist", 10, 10, 0.9), Keypoint("right_wrist", 100, 10, 0.9)], 0.9, backend="synthetic"),
        PoseFrameResult(0.1, 1, [Keypoint("left_wrist", 101, 10, 0.9), Keypoint("right_wrist", 11, 10, 0.9)], 0.9, backend="synthetic"),
    ]

    result = TemporalJointTracker(TemporalTrackingConfig(enable_smoothing=False)).process(frames, view_type="front")
    corrected = result.trajectory_export["corrected_keypoints"]

    assert result.swap_events
    assert corrected["left_wrist"][1]["x"] == pytest.approx(11)
    assert corrected["right_wrist"][1]["x"] == pytest.approx(101)


def test_temporal_export_contains_smoothing_and_joint_reliability() -> None:
    frames = [
        PoseFrameResult(
            index / 30,
            index,
            [
                Keypoint("nose", 100 + index * 0.2, 40, 0.9),
                Keypoint("left_wrist", 130 + ((-1) ** index) * 14, 90, 0.82),
            ],
            0.86,
            backend="synthetic",
            roi={"valid": True, "source": "previous_roi", "x": 40, "y": 20, "width": 120, "height": 80},
        )
        for index in range(10)
    ]

    result = TemporalJointTracker(TemporalTrackingConfig(enable_smoothing=True)).process(frames, view_type="video")
    export = result.trajectory_export

    assert export["smoothing_diagnostics"]["enabled"] is True
    assert export["smoothing_diagnostics"]["mean_raw_to_smoothed_px"] >= 0
    assert export["joint_reliability"]["nose"]["category"] in {"reliable", "estimated"}
    assert export["joint_reliability"]["right_ankle"]["category"] == "hidden"
    assert export["center_trajectory"]
    assert export["roi_history"][0]["source"] == "previous_roi"
    assert "dropped_frame_diagnostics" in export


def test_joint_reliability_marks_noisy_joint_unstable() -> None:
    config = TemporalTrackingConfig(
        enable_smoothing=False,
        core_threshold=JointPhysicsThreshold(1000, 20000, 500000),
        mid_threshold=JointPhysicsThreshold(1000, 20000, 500000),
        distal_threshold=JointPhysicsThreshold(1000, 20000, 500000),
    )
    x_values = [100, 120, 380, 390, 80, 410, 90]
    frames = [
        PoseFrameResult(
            index / 10,
            index,
            [
                Keypoint("nose", 50 + index, 40, 0.9),
                Keypoint("left_wrist", x, 90, 0.9),
            ],
            0.9,
            backend="synthetic",
        )
        for index, x in enumerate(x_values)
    ]

    result = TemporalJointTracker(config).process(frames, view_type="side")

    assert result.joint_reliability["left_wrist"]["category"] == "unstable"
    assert result.joint_reliability["nose"]["category"] in {"reliable", "estimated"}


def test_phase1_metrics_output_schema_is_valid() -> None:
    frames = _metric_frames()
    tracking = TemporalJointTracker(TemporalTrackingConfig(enable_smoothing=False)).process(frames, view_type="side")
    base_metrics = {
        "confidence_score": {"value": 0.8, "confidence": 0.8, "evidence": {}},
        "stroke_cycle_consistency": {"value": 82, "confidence": 0.7, "evidence": {}},
        "kick_rhythm_score": {"value": 76, "confidence": 0.65, "evidence": {}},
        "arm_symmetry_score": {"value": 88, "confidence": 0.75, "evidence": {}},
    }

    metrics = compute_phase1_research_metrics(
        frames,
        frames,
        base_metrics,
        side_tracking=tracking.trajectory_export,
        front_tracking=tracking.trajectory_export,
        video_quality={"side": {"overall_quality_score": 0.8}, "front": {"overall_quality_score": 0.8}},
    )

    TrajectoryExport.model_validate(tracking.trajectory_export)
    assert set(metrics) >= {
        "hip_stability_score",
        "stroke_rhythm_estimate",
        "kick_rhythm_estimate",
        "left_right_symmetry_estimate",
        "joint_visibility_score",
        "tracking_quality_score",
        "analysis_reliability_score",
    }
    assert metrics["analysis_reliability_score"]["value"] is not None


def test_phase1_metrics_single_video_uses_video_evidence_label() -> None:
    frames = _metric_frames()
    tracking = TemporalJointTracker(TemporalTrackingConfig(enable_smoothing=False)).process(frames, view_type="side")

    metrics = compute_phase1_research_metrics(
        frames,
        frames,
        {"confidence_score": {"value": 0.8, "confidence": 0.8, "evidence": {}}},
        side_tracking=tracking.trajectory_export,
        front_tracking=tracking.trajectory_export,
        video_quality={"side": {"overall_quality_score": 0.8}, "front": {"overall_quality_score": 0.8}},
        single_video_mode=True,
    )

    assert "video" in metrics["joint_visibility_score"]["evidence"]
    assert "front" not in metrics["joint_visibility_score"]["evidence"]
    assert metrics["tracking_quality_score"]["evidence"]["video_frames"] > 0


def _metric_frames() -> list[PoseFrameResult]:
    frames = []
    for index in range(8):
        x = 100 + index
        frames.append(
            PoseFrameResult(
                index / 30,
                index,
                [
                    Keypoint("left_shoulder", x, 80, 0.9),
                    Keypoint("right_shoulder", x + 40, 80, 0.9),
                    Keypoint("left_hip", x + 4, 130, 0.9),
                    Keypoint("right_hip", x + 36, 130, 0.9),
                    Keypoint("left_wrist", x - 20, 90 + index, 0.9),
                    Keypoint("right_wrist", x + 60, 90 - index, 0.9),
                    Keypoint("left_ankle", x + 8, 190, 0.8),
                    Keypoint("right_ankle", x + 32, 190, 0.8),
                ],
                confidence=0.86,
                view_type="side",
                backend="synthetic",
            )
        )
    return frames
