from dataclasses import replace
from pathlib import Path

import pytest

from app.swim_analysis.calibration import calibrate_pool_view
from app.swim_analysis.coaching_report import generate_coaching_report
from app.swim_analysis.faults import detect_swimming_faults
from app.swim_analysis.metrics import compute_swimming_metrics
from app.swim_analysis.pose.base import Keypoint, PoseBackendUnavailable, PoseFrameResult
from app.swim_analysis.pose.opencv_motion_backend import OpenCVMotionBackend
from app.swim_analysis.pose.rtmpose_backend import KEYPOINT_NAMES, RTMPoseBackend, raw_keypoint_debug, resolve_rtmpose_paths
from app.swim_analysis.preprocessing import assess_video_quality
from app.swim_analysis.processor import ROITrackingState, _candidate_rejection_summary, _infer_best_roi_candidate, _select_pose_candidate
from app.swim_analysis.report import generate_annotated_video
from app.swim_analysis.roi import ROIConfig, ROIResult, SwimmerLocalizer, centered_roi_from_bbox, project_pose_to_original
from app.swim_analysis.schemas import SwimAnalysisMetricsResponse, SwimAnalysisReportResponse, SwimAnalysisVelocityResponse
from app.swim_analysis.segmentation import segment_swim_phases
from app.swim_analysis.velocity import attach_phase_velocity_summary, compute_velocity_analysis


def test_calibration_math_with_manual_lane_length() -> None:
    frames = _moving_frames(count=30, start_x=100, step_x=10)

    calibration = calibrate_pool_view(frames, lane_length_m=25)

    assert calibration["method"] == "manual_lane_length_motion_span"
    assert calibration["pixel_to_meter"] == pytest.approx(25 / 290)
    assert calibration["confidence"] > 0.5


def test_velocity_calculation_returns_chart_ready_series() -> None:
    frames = _moving_frames(count=30, start_x=100, step_x=5)
    calibration = calibrate_pool_view(frames, lane_length_m=25)

    velocity = compute_velocity_analysis(frames, calibration)

    assert velocity["chart_ready"] is True
    assert len(velocity["series"]) == 30
    assert velocity["summary"]["average_velocity"] > 0
    assert velocity["summary"]["velocity_unit"] == "m/s"
    assert velocity["series"][0]["motion_source"] == "body_centroid"
    assert "raw_velocity" in velocity["series"][0]


def test_velocity_physics_validation_corrects_implausible_spikes() -> None:
    frames = _spiky_motion_frames()
    calibration = calibrate_pool_view(frames, lane_length_m=25)

    velocity = compute_velocity_analysis(frames, calibration)

    validation = velocity["summary"]["motion_validation"]
    assert validation["raw_peak_velocity"] > 4.2
    assert velocity["summary"]["max_velocity"] <= 4.2
    assert validation["anomaly_counts"]["unrealistic_speed_spike"] >= 1
    assert validation["anomaly_counts"]["unrealistic_acceleration_spike"] >= 1
    assert velocity["summary"]["confidence"] < 0.6


def test_centroid_only_velocity_is_plausibility_gated() -> None:
    frames = _bbox_frames(count=120, duration=31.9, start_x=100, end_x=1500)
    calibration = calibrate_pool_view(frames, lane_length_m=50)

    velocity = compute_velocity_analysis(frames, calibration)

    assert velocity["chart_ready"] is True
    assert velocity["summary"]["evidence_mode"] == "centroid_only"
    assert velocity["summary"]["average_velocity"] == pytest.approx(50 / 31.9, abs=0.18)
    assert velocity["summary"]["distance_review_m"] == pytest.approx(50, abs=0.5)
    assert velocity["summary"]["max_velocity"] <= 4.2
    assert velocity["summary"]["confidence"] <= 0.34
    assert velocity["summary"]["dead_spot_count"] == 0
    assert velocity["dead_spots"] == []
    assert velocity["summary"]["warnings"]
    assert [split["end_distance_m"] for split in velocity["review_splits"]] == [15.0, 25.0, 50.0]


def test_stroke_segmentation_outputs_required_phase_types() -> None:
    frames = _moving_frames(count=90, start_x=100, step_x=3)
    velocity = compute_velocity_analysis(frames, calibrate_pool_view(frames, lane_length_m=25))

    phases = segment_swim_phases(frames, velocity)
    velocity = attach_phase_velocity_summary(velocity, phases)
    phase_types = {phase["type"] for phase in phases}

    assert "start_push_off" in phase_types
    assert "breakout" in phase_types
    assert "free_swim" in phase_types
    assert "finish" in phase_types
    assert velocity["phase_summary"]


def test_stroke_segmentation_adds_motion_based_subphases() -> None:
    frames = _moving_frames(count=120, start_x=100, step_x=2)
    velocity = compute_velocity_analysis(frames, calibrate_pool_view(frames, lane_length_m=25))

    phases = segment_swim_phases(frames, velocity)
    phase_types = {phase["type"] for phase in phases}

    assert {"catch", "pull", "push", "recovery"}.issubset(phase_types)
    assert any(phase.get("category") == "temporal_pattern" for phase in phases)
    assert all(phase.get("confidence", 0) <= 1 for phase in phases)


def test_centroid_only_segmentation_does_not_invent_turns() -> None:
    frames = _bbox_frames(count=120, duration=31.9, start_x=100, end_x=1500)
    velocity = compute_velocity_analysis(frames, calibrate_pool_view(frames, lane_length_m=50))

    phases = segment_swim_phases(frames, velocity)
    phase_types = [phase["type"] for phase in phases]

    assert "turn" not in phase_types
    assert "stroke_cycle" not in phase_types
    assert phase_types == ["start_push_off", "free_swim", "finish"]


def test_fault_detection_uses_metric_confidence() -> None:
    metrics = {
        "drag_risk_score": {"value": 68, "confidence": 0.8, "evidence": {}},
        "streamline_score": {"value": 60, "confidence": 0.8, "evidence": {}},
        "kick_symmetry": {"value": 55, "confidence": 0.7, "evidence": {}},
        "catch_angle": {"value": 162, "confidence": "low", "evidence": {"numeric_confidence": 0.2}},
    }
    velocity = {
        "dead_spots": [{"start_sec": 1.0, "end_sec": 1.5, "min_velocity": 0.3, "threshold": 0.8, "confidence": 0.7}],
        "summary": {"confidence": 0.7, "evidence_mode": "pose_centroid"},
    }

    faults = detect_swimming_faults(metrics, velocity, [])

    names = {fault["name"] for fault in faults}
    assert "high_drag_body_position" in names
    assert "asymmetric_kick" in names
    assert "velocity_dead_spot" in names
    assert "dropped_elbow_catch" not in names


def test_centroid_only_dead_spots_do_not_become_technique_faults() -> None:
    velocity = {
        "dead_spots": [{"start_sec": 5.0, "end_sec": 5.8, "min_velocity": 0.2, "threshold": 0.9, "confidence": 0.7}],
        "summary": {"confidence": 0.7, "evidence_mode": "centroid_only"},
    }

    faults = detect_swimming_faults({}, velocity, [])

    assert faults == []


def test_no_skeleton_metrics_are_unavailable_not_fake_scores() -> None:
    frames = _bbox_frames(count=90, duration=30.0, start_x=100, end_x=1200)
    calibration = calibrate_pool_view(frames, lane_length_m=50)
    velocity = compute_velocity_analysis(frames, calibration)

    metrics = compute_swimming_metrics(
        frames,
        frames,
        stroke_type="freestyle",
        video_quality={"side": {"usable_frame_ratio": 1}, "front": {"usable_frame_ratio": 1}},
        calibration=calibration,
        velocity=velocity,
    )

    assert metrics["stroke_rate_spm"]["value"] is None
    assert metrics["body_alignment_score"]["value"] is None
    assert metrics["head_stability_score"]["value"] is None
    assert metrics["arm_symmetry_score"]["value"] is None
    assert metrics["kick_frequency"]["value"] is None
    assert metrics["propulsion_efficiency_score"]["value"] is None
    assert metrics["drag_risk_score"]["value"] is None
    assert metrics["confidence_score"]["value"] < 0.35


def test_centroid_only_coaching_report_handles_unavailable_metrics() -> None:
    frames = _bbox_frames(count=90, duration=30.0, start_x=100, end_x=1200)
    calibration = calibrate_pool_view(frames, lane_length_m=50)
    velocity = compute_velocity_analysis(frames, calibration)
    phases = segment_swim_phases(frames, velocity)
    velocity = attach_phase_velocity_summary(velocity, phases)
    metrics = compute_swimming_metrics(
        frames,
        frames,
        stroke_type="freestyle",
        video_quality={"side": {"usable_frame_ratio": 1}, "front": {"usable_frame_ratio": 1}},
        calibration=calibration,
        velocity=velocity,
    )

    coaching = generate_coaching_report(
        metrics=metrics,
        faults=[],
        phases=phases,
        velocity=velocity,
        video_quality={"side": {"usable_frame_ratio": 1}, "front": {"usable_frame_ratio": 1}},
    )

    assert "centroid-only" in coaching["technical_summary"]
    assert coaching["race_improvement_suggestions"]
    assert coaching["recommended_drills"][0]["drill"] == "Recapture or enable RTMPose before technique correction"


def test_video_quality_scoring_with_synthetic_video(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    cv2.setNumThreads(1)
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 90))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable in this environment.")
    for index in range(12):
        frame = np.full((90, 160, 3), 80 + index * 3, dtype=np.uint8)
        cv2.line(frame, (20 + index * 5, 45), (90 + index * 5, 45), (240, 240, 240), 3)
        writer.write(frame)
    writer.release()

    quality = assess_video_quality(path, target_samples=6)

    assert quality.sampled_frames > 0
    assert 0 <= quality.blur_score <= 1
    assert 0 <= quality.lighting_score <= 1


def test_annotated_video_uses_browser_playable_webm(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    cv2.setNumThreads(1)
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 90))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is unavailable in this environment.")
    for index in range(12):
        frame = np.full((90, 160, 3), 70, dtype=np.uint8)
        cv2.circle(frame, (25 + index * 5, 45), 10, (240, 240, 240), -1)
        writer.write(frame)
    writer.release()

    output = generate_annotated_video(
        source_path=source,
        target_path=tmp_path / "annotated.webm",
        frames=_moving_frames(count=12, start_x=25, step_x=5),
        metrics={},
        view_type="video",
        max_dimension=320,
    )
    if output is None:
        pytest.skip("Browser-playable WebM writer is unavailable in this environment.")

    cap = cv2.VideoCapture(output)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    assert Path(output).suffix == ".webm"
    assert codec in {"VP80", "VP90"}
    assert frame_count > 0


def test_api_response_schemas_accept_engine_payloads() -> None:
    report = SwimAnalysisReportResponse(
        analysis_id="a1",
        status="completed",
        summary={"overall_score": 80},
        video_quality={},
        video_metadata={"input_mode": "single_video"},
        metrics={},
        pose_diagnostics={"requested_backend": "rtmpose", "actual_backend": "rtmpose"},
        calibration={"confidence": 0.7},
        velocity={"series": []},
        phase_segments=[],
        faults=[],
        findings=[],
        recommendations=[],
        coaching_report={"safety_note": "avoid medical claims"},
        artifacts={},
    )
    metrics = SwimAnalysisMetricsResponse(analysis_id="a1", status="completed", metrics={}, faults=[], calibration={})
    velocity = SwimAnalysisVelocityResponse(analysis_id="a1", status="completed", velocity={}, phase_segments=[], calibration={})

    assert report.analysis_id == "a1"
    assert report.video_metadata["input_mode"] == "single_video"
    assert report.pose_diagnostics["actual_backend"] == "rtmpose"
    assert metrics.status == "completed"
    assert velocity.phase_segments == []


def test_rtmpose_missing_config_fails_gracefully() -> None:
    backend = RTMPoseBackend(config_path=None, checkpoint_path=None, device="cpu")

    with pytest.raises(PoseBackendUnavailable):
        backend.load()


def test_rtmpose_checkpoint_path_resolution(tmp_path: Path) -> None:
    config = tmp_path / "config.py"
    checkpoint = tmp_path / "checkpoint.pth"
    config.write_text("model = dict(head=dict(out_channels=17))", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")

    resolved_config, resolved_checkpoint = resolve_rtmpose_paths(str(config), str(checkpoint))

    assert resolved_config == config.resolve()
    assert resolved_checkpoint == checkpoint.resolve()


def test_rtmpose_keypoint_debug_uses_coco17_order() -> None:
    keypoints = [[float(index), float(index + 1)] for index in range(17)]
    scores = [0.5 for _ in range(17)]

    debug = raw_keypoint_debug(keypoints, scores)

    assert [item["mapped_joint"] for item in debug] == KEYPOINT_NAMES
    assert debug[5]["mapped_joint"] == "left_shoulder"
    assert debug[16]["mapped_joint"] == "right_ankle"


def test_rtmpose_prediction_parser_accepts_nested_score_vector() -> None:
    backend = RTMPoseBackend(config_path="config.py", checkpoint_path="checkpoint.pth", device="cpu", confidence_threshold=0.05)
    keypoints = [[[float(index), float(index + 1)] for index in range(17)]]
    scores = [[0.5 + index * 0.01 for index in range(17)]]

    class Prediction:
        pass

    pred_instances = Prediction()
    pred_instances.keypoints = keypoints
    pred_instances.keypoint_scores = scores
    pred_instances.bboxes = [[0, 0, 100, 100]]
    prediction = Prediction()
    prediction.pred_instances = pred_instances

    result = backend._prediction_to_result(
        [prediction],
        frame_index=3,
        timestamp=0.25,
        view_type="side",
        quality_flags=[],
    )

    assert len(result.keypoints) == 17
    assert len(result.raw_keypoints or []) == 17
    assert result.keypoints[0].confidence == pytest.approx(0.5)
    assert result.keypoints[-1].confidence == pytest.approx(0.66)
    assert result.debug_info["raw_keypoints_by_index"][5]["mapped_joint"] == "left_shoulder"


def test_rtmpose_threshold_filtering_keeps_raw_landmarks() -> None:
    backend = RTMPoseBackend(config_path="config.py", checkpoint_path="checkpoint.pth", device="cpu", confidence_threshold=0.4)
    keypoints = [[[float(index), float(index + 1)] for index in range(17)]]
    scores = [[0.1 if index < 10 else 0.8 for index in range(17)]]

    class Prediction:
        pass

    pred_instances = Prediction()
    pred_instances.keypoints = keypoints
    pred_instances.keypoint_scores = scores
    pred_instances.bboxes = [[0, 0, 100, 100]]
    prediction = Prediction()
    prediction.pred_instances = pred_instances

    result = backend._prediction_to_result(
        [prediction],
        frame_index=3,
        timestamp=0.25,
        view_type="side",
        quality_flags=[],
    )

    assert len(result.raw_keypoints or []) == 17
    assert len(result.keypoints) == 7
    assert result.debug_info["removed_by_threshold"] == 10


def test_roi_projection_remaps_raw_and_filtered_coordinates() -> None:
    import numpy as np

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    roi = centered_roi_from_bbox(frame, [50, 20, 90, 60], confidence=0.9, config=ROIConfig(scale=1.0, min_size_px=10))
    pose = PoseFrameResult(
        timestamp=0,
        frame_index=1,
        keypoints=[Keypoint("left_shoulder", 5, 6, 0.8)],
        raw_keypoints=[Keypoint("left_shoulder", 5, 6, 0.8)],
        backend="rtmpose",
        debug_info={"raw_keypoints_by_index": [{"raw_index": 5, "mapped_joint": "left_shoulder", "x": 5, "y": 6, "score": 0.8}]},
    )

    projected = project_pose_to_original(pose, roi)

    assert projected.keypoints[0].x == pytest.approx(roi.x + 5)
    assert projected.raw_keypoints[0].y == pytest.approx(roi.y + 6)
    assert projected.debug_info["raw_keypoints_by_index"][0]["x"] == pytest.approx(roi.x + 5)


def test_roi_compare_selects_full_frame_when_roi_pose_is_worse() -> None:
    roi_pose = PoseFrameResult(timestamp=0, frame_index=1, keypoints=[Keypoint("nose", 1, 1, 0.2)], raw_keypoints=[Keypoint("nose", 1, 1, 0.2)], backend="rtmpose")
    full_pose = PoseFrameResult(timestamp=0, frame_index=1, keypoints=_moving_frames(1, 10, 1)[0].keypoints, raw_keypoints=_moving_frames(1, 10, 1)[0].keypoints, backend="rtmpose", confidence=0.8)

    selected = _select_pose_candidate(roi_pose, full_pose)

    assert "roi_compare_selected_full_frame" in selected.quality_flags


def test_roi_rejects_full_width_lane_line_boundary() -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    frame = np.zeros((240, 420, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 20), (420, 31), (255, 255, 255), -1)
    localizer = SwimmerLocalizer(ROIConfig(max_candidates=4))

    candidates = localizer.candidate_rois(frame, frame_index=1, timestamp=0.0, view_type="video")

    assert all(candidate.width < 0.82 * frame.shape[1] for candidate in candidates)
    assert all(candidate.boundary_touch_ratio <= 0.4 for candidate in candidates)


def test_roi_candidates_include_center_before_full_frame_fallback() -> None:
    pytest.importorskip("cv2")
    import numpy as np

    frame = np.zeros((240, 420, 3), dtype=np.uint8)
    localizer = SwimmerLocalizer(ROIConfig(max_candidates=4, center_crop_width_ratio=0.3, center_crop_height_ratio=0.25))

    candidates = localizer.candidate_rois(frame, frame_index=1, timestamp=0.0, view_type="video")

    assert candidates
    assert any(candidate.source == "center_debug_crop" for candidate in candidates)
    assert all(candidate.source != "full_frame" for candidate in candidates)


def test_roi_temporal_continuity_penalizes_far_static_candidate() -> None:
    import numpy as np

    frame = np.zeros((240, 420, 3), dtype=np.uint8)
    localizer = SwimmerLocalizer(ROIConfig())
    previous = centered_roi_from_bbox(frame, [120, 90, 180, 130], confidence=0.7, config=ROIConfig(scale=1.0), source="motion")
    localizer.remember_roi(replace(previous, frame_index=0, timestamp=0.0))
    near = centered_roi_from_bbox(frame, [130, 92, 190, 132], confidence=0.65, config=ROIConfig(scale=1.0), source="edge_contour")
    far = centered_roi_from_bbox(frame, [320, 92, 380, 132], confidence=0.65, config=ROIConfig(scale=1.0), source="edge_contour")

    scored = localizer._apply_temporal_continuity([near, far], frame_index=1, timestamp=0.33)

    assert scored[0].candidate_score > scored[1].candidate_score
    assert scored[0].metrics["temporal_jump_ratio"] < scored[1].metrics["temporal_jump_ratio"]
    assert scored[0].timestamp == pytest.approx(0.33)


def test_roi_tracking_state_flags_drift_and_smoothing_reset() -> None:
    tracker = ROITrackingState()
    good_roi = ROIResult(True, None, x=80, y=90, width=100, height=60, frame_width=420, frame_height=240, source="motion", candidate_score=0.62)
    good_pose = PoseFrameResult(
        0.0,
        0,
        [
            Keypoint("left_shoulder", 100, 100, 0.45),
            Keypoint("right_shoulder", 140, 100, 0.45),
            Keypoint("left_hip", 108, 140, 0.42),
            Keypoint("right_hip", 132, 140, 0.42),
            Keypoint("left_wrist", 90, 125, 0.38),
        ],
        0.42,
        backend="rtmpose",
    )
    good_diag = tracker.diagnose(good_pose, good_roi, {"decision": {"selected_score": 0.5}}, reattempt=False)
    locked = tracker.commit(good_pose, good_roi, {"decision": {"selected_score": 0.5}}, diagnosis=good_diag)

    bad_roi = ROIResult(True, None, x=350, y=5, width=18, height=210, frame_width=420, frame_height=240, source="previous_roi", candidate_score=0.5, boundary_touch_ratio=0.3)
    bad_pose = PoseFrameResult(
        0.33,
        10,
        [Keypoint("nose", 358, 20, 0.42), Keypoint("left_eye", 356, 18, 0.38)],
        0.4,
        backend="rtmpose",
    )
    bad_diag = tracker.diagnose(bad_pose, bad_roi, {"decision": {"selected_score": 0.2}}, reattempt=False)
    suspect = tracker.commit(bad_pose, bad_roi, {"decision": {"selected_score": 0.2}}, diagnosis=bad_diag)

    assert locked["tracking_state"] == "LOCKED"
    assert suspect["tracking_state"] == "SUSPECT"
    assert suspect["drift_detected"] is True
    assert suspect["reacquisition_triggered"] is True
    assert suspect["smoothing_reset"] is True
    assert "face_only_or_background_pose" in suspect["drift_reasons"]
    assert "teleporting_bbox_center" in suspect["drift_reasons"]
    assert suspect["motion_physics_anomaly"] is True


def test_roi_identity_rejects_far_false_positive_reacquire() -> None:
    tracker = ROITrackingState()
    good_roi = ROIResult(True, None, x=80, y=90, width=120, height=70, frame_width=500, frame_height=260, source="motion", candidate_score=0.7)
    good_pose = PoseFrameResult(
        0.0,
        0,
        [
            Keypoint("nose", 124, 88, 0.32),
            Keypoint("left_eye", 121, 86, 0.25),
            Keypoint("right_eye", 127, 86, 0.25),
            Keypoint("left_shoulder", 105, 110, 0.55),
            Keypoint("right_shoulder", 150, 110, 0.55),
            Keypoint("left_hip", 112, 150, 0.52),
            Keypoint("right_hip", 144, 150, 0.52),
            Keypoint("left_knee", 116, 190, 0.45),
            Keypoint("right_knee", 140, 190, 0.45),
        ],
        0.45,
        backend="rtmpose",
    )
    good_identity = tracker.candidate_identity_diagnostics(good_pose, good_roi)
    good_diag = tracker.diagnose(good_pose, good_roi, {"selected_identity": good_identity, "decision": {"selected_score": 0.7}}, reattempt=False)
    tracker.commit(good_pose, good_roi, {"selected_identity": good_identity, "decision": {"selected_score": 0.7}}, diagnosis=good_diag)

    tracker.state = "LOST"
    tracker.bad_frames = 2
    false_roi = ROIResult(True, None, x=390, y=20, width=70, height=180, frame_width=500, frame_height=260, source="motion", candidate_score=0.76)
    false_pose = PoseFrameResult(
        0.5,
        15,
        [
            Keypoint("left_shoulder", 405, 50, 0.62),
            Keypoint("right_shoulder", 435, 52, 0.62),
            Keypoint("left_elbow", 395, 80, 0.6),
            Keypoint("right_elbow", 448, 82, 0.6),
            Keypoint("left_hip", 410, 120, 0.58),
            Keypoint("right_hip", 432, 120, 0.58),
            Keypoint("left_knee", 412, 165, 0.55),
            Keypoint("right_knee", 430, 165, 0.55),
        ],
        0.59,
        backend="rtmpose",
    )
    false_identity = tracker.candidate_identity_diagnostics(false_pose, false_roi)
    false_diag = tracker.diagnose(false_pose, false_roi, {"selected_identity": false_identity, "decision": {"selected_score": 0.65}}, reattempt=True)
    final = tracker.commit(false_pose, false_roi, {"selected_identity": false_identity, "decision": {"selected_score": 0.65}}, diagnosis=false_diag)

    assert final["tracking_state"] == "REVIEW_ONLY"
    assert final["identity_hard_reject"] is True
    assert "identity_spatial_jump" in final["drift_reasons"]
    assert "head_body_relationship_low" in final["drift_reasons"]
    assert final["good_lock"] is False


def test_roi_recovery_candidates_before_full_frame_from_last_swimmer() -> None:
    import numpy as np

    tracker = ROITrackingState()
    frame = np.zeros((260, 500, 3), dtype=np.uint8)
    good_roi = ROIResult(True, frame[90:160, 80:200], x=80, y=90, width=120, height=70, frame_width=500, frame_height=260, source="motion", candidate_score=0.7)
    good_pose = PoseFrameResult(
        0.0,
        0,
        [
            Keypoint("nose", 124, 88, 0.32),
            Keypoint("left_shoulder", 105, 110, 0.55),
            Keypoint("right_shoulder", 150, 110, 0.55),
            Keypoint("left_hip", 112, 150, 0.52),
            Keypoint("right_hip", 144, 150, 0.52),
            Keypoint("left_knee", 116, 190, 0.45),
            Keypoint("right_knee", 140, 190, 0.45),
        ],
        0.45,
        backend="rtmpose",
    )
    identity = tracker.candidate_identity_diagnostics(good_pose, good_roi)
    diag = tracker.diagnose(good_pose, good_roi, {"selected_identity": identity, "decision": {"selected_score": 0.7}}, reattempt=False)
    tracker.commit(good_pose, good_roi, {"selected_identity": identity, "decision": {"selected_score": 0.7}}, diagnosis=diag)
    tracker.state = "LOST"
    tracker.bad_frames = 2

    candidates = tracker.recovery_candidates(frame, ROIConfig(scale=1.0), frame_index=15, timestamp=0.5)
    sources = {candidate.source for candidate in candidates}

    assert "last_good_roi_expanded" in sources
    assert sources & {"predicted_trajectory_crop", "center_on_last_swimmer"}
    assert "full_frame" not in sources
    assert all(candidate.valid and candidate.crop is not None for candidate in candidates)


def test_roi_identity_relaxes_for_plausible_high_confidence_recovery() -> None:
    tracker = ROITrackingState()
    good_roi = ROIResult(True, None, x=80, y=90, width=120, height=70, frame_width=500, frame_height=260, source="motion", candidate_score=0.7)
    good_pose = PoseFrameResult(
        0.0,
        0,
        [
            Keypoint("nose", 124, 88, 0.32),
            Keypoint("left_eye", 121, 86, 0.25),
            Keypoint("right_eye", 127, 86, 0.25),
            Keypoint("left_shoulder", 105, 110, 0.55),
            Keypoint("right_shoulder", 150, 110, 0.55),
            Keypoint("left_hip", 112, 150, 0.52),
            Keypoint("right_hip", 144, 150, 0.52),
            Keypoint("left_knee", 116, 190, 0.45),
            Keypoint("right_knee", 140, 190, 0.45),
        ],
        0.45,
        backend="rtmpose",
    )
    good_identity = tracker.candidate_identity_diagnostics(good_pose, good_roi)
    good_diag = tracker.diagnose(good_pose, good_roi, {"selected_identity": good_identity, "decision": {"selected_score": 0.7}}, reattempt=False)
    tracker.commit(good_pose, good_roi, {"selected_identity": good_identity, "decision": {"selected_score": 0.7}}, diagnosis=good_diag)
    tracker.state = "LOST"
    tracker.bad_frames = 2

    recovery_roi = ROIResult(True, None, x=98, y=92, width=135, height=82, frame_width=500, frame_height=260, source="predicted_trajectory_crop", candidate_score=0.68, confidence=0.65)
    recovery_pose = PoseFrameResult(
        0.5,
        15,
        [
            Keypoint("nose", 150, 90, 0.32),
            Keypoint("left_eye", 147, 88, 0.24),
            Keypoint("right_eye", 153, 88, 0.24),
            Keypoint("left_shoulder", 126, 112, 0.48),
            Keypoint("right_shoulder", 174, 112, 0.48),
            Keypoint("left_elbow", 116, 136, 0.42),
            Keypoint("right_elbow", 184, 136, 0.42),
            Keypoint("left_hip", 132, 154, 0.46),
            Keypoint("right_hip", 168, 154, 0.46),
            Keypoint("left_knee", 136, 192, 0.38),
            Keypoint("right_knee", 164, 192, 0.38),
        ],
        0.42,
        backend="rtmpose",
    )

    identity = tracker.candidate_identity_diagnostics(recovery_pose, recovery_roi)

    assert identity["high_confidence_plausible_roi"] is True
    assert identity["identity_hard_reject"] is False
    assert identity["identity_relock_ready"] is True


def test_candidate_rejection_summary_exports_all_candidate_reasons() -> None:
    summary = _candidate_rejection_summary(
        [
            {
                "source": "motion",
                "rejected": True,
                "rejection_reasons": ["identity_spatial_jump"],
                "rtmpose_selection_score": 0.12,
                "identity": {"trajectory_distance_ratio": 2.4, "human_shape_score": 0.7},
            },
            {
                "source": "last_good_roi_expanded",
                "rejected": False,
                "rtmpose_selection_score": 0.38,
                "identity": {"soft_reasons": ["identity_spatial_distance_high"], "high_confidence_plausible_roi": True},
            },
        ]
    )

    assert summary[0]["status"] == "rejected"
    assert summary[0]["rejection_reasons"] == ["identity_spatial_jump"]
    assert summary[1]["status"] == "accepted"
    assert summary[1]["soft_reasons"] == ["identity_spatial_distance_high"]
    assert summary[1]["high_confidence_plausible_roi"] is True


def test_roi_selection_uses_recovery_candidate_before_full_frame() -> None:
    import numpy as np

    class FakeEstimator:
        name = "rtmpose"

        def infer_frame(self, frame, *, frame_index, timestamp, view_type, quality_flags):
            source = next((flag.replace("roi_source_", "") for flag in quality_flags if flag.startswith("roi_source_")), "full_frame")
            if source == "full_frame":
                return PoseFrameResult(timestamp, frame_index, [Keypoint("nose", 1, 1, 0.18)], 0.18, backend="rtmpose", view_type=view_type, quality_flags=quality_flags)
            if source == "motion":
                points = [
                    Keypoint("left_shoulder", 8, 12, 0.55),
                    Keypoint("right_shoulder", 38, 12, 0.55),
                    Keypoint("left_hip", 12, 70, 0.52),
                    Keypoint("right_hip", 35, 70, 0.52),
                    Keypoint("left_knee", 12, 115, 0.48),
                    Keypoint("right_knee", 35, 115, 0.48),
                ]
                return PoseFrameResult(timestamp, frame_index, points, 0.52, backend="rtmpose", view_type=view_type, quality_flags=quality_flags)
            points = [
                Keypoint("nose", 54, 16, 0.32),
                Keypoint("left_eye", 51, 14, 0.22),
                Keypoint("right_eye", 57, 14, 0.22),
                Keypoint("left_shoulder", 32, 38, 0.46),
                Keypoint("right_shoulder", 78, 38, 0.46),
                Keypoint("left_elbow", 22, 62, 0.4),
                Keypoint("right_elbow", 88, 62, 0.4),
                Keypoint("left_hip", 38, 80, 0.44),
                Keypoint("right_hip", 72, 80, 0.44),
                Keypoint("left_knee", 40, 118, 0.36),
                Keypoint("right_knee", 70, 118, 0.36),
            ]
            return PoseFrameResult(timestamp, frame_index, points, 0.4, backend="rtmpose", view_type=view_type, quality_flags=quality_flags)

    frame = np.zeros((260, 500, 3), dtype=np.uint8)
    tracker = ROITrackingState()
    good_roi = ROIResult(True, frame[90:160, 80:200], x=80, y=90, width=120, height=70, frame_width=500, frame_height=260, source="motion", candidate_score=0.7)
    good_pose = PoseFrameResult(
        0.0,
        0,
        [
            Keypoint("nose", 124, 88, 0.32),
            Keypoint("left_eye", 121, 86, 0.25),
            Keypoint("right_eye", 127, 86, 0.25),
            Keypoint("left_shoulder", 105, 110, 0.55),
            Keypoint("right_shoulder", 150, 110, 0.55),
            Keypoint("left_hip", 112, 150, 0.52),
            Keypoint("right_hip", 144, 150, 0.52),
            Keypoint("left_knee", 116, 190, 0.45),
            Keypoint("right_knee", 140, 190, 0.45),
        ],
        0.45,
        backend="rtmpose",
    )
    identity = tracker.candidate_identity_diagnostics(good_pose, good_roi)
    diag = tracker.diagnose(good_pose, good_roi, {"selected_identity": identity, "decision": {"selected_score": 0.7}}, reattempt=False)
    tracker.commit(good_pose, good_roi, {"selected_identity": identity, "decision": {"selected_score": 0.7}}, diagnosis=diag)
    tracker.state = "LOST"
    tracker.bad_frames = 2
    false_far = ROIResult(True, frame[20:200, 390:460], x=390, y=20, width=70, height=180, frame_width=500, frame_height=260, source="motion", candidate_score=0.76, confidence=0.76)
    recovery = ROIResult(True, frame[90:190, 95:235], x=95, y=90, width=140, height=100, frame_width=500, frame_height=260, source="last_good_roi_expanded", candidate_score=0.6, confidence=0.56)

    pose, debug = _infer_best_roi_candidate(
        FakeEstimator(),
        frame,
        [false_far, recovery],
        frame_index=15,
        timestamp=0.5,
        view_type="side",
        quality_flags=[],
        compare_full_frame=True,
        fallback_full_frame=True,
        identity_tracker=tracker,
    )

    assert pose.roi["source"] == "last_good_roi_expanded"
    assert debug["decision"]["selected"] == "roi"
    assert debug["decision"]["selected_source"] == "last_good_roi_expanded"
    assert any(candidate["source"] == "motion" and candidate["rejected"] for candidate in debug["decision"]["candidates"])


def test_roi_assisted_scoring_prefers_manual_crop_confidence() -> None:
    from app.swim_analysis.processor import _roi_assisted_pose_score

    import numpy as np

    frame = np.zeros((300, 600, 3), dtype=np.uint8)
    good_roi = centered_roi_from_bbox(frame, [180, 120, 420, 210], confidence=0.7, config=ROIConfig(scale=1.0), source="manual_debug")
    full_roi = centered_roi_from_bbox(frame, [0, 0, 600, 300], confidence=0.2, config=ROIConfig(scale=1.0, max_boundary_clip_ratio=1.0), source="full_frame_reference")
    body_points = [
        Keypoint("left_shoulder", 200, 140, 0.45),
        Keypoint("right_shoulder", 260, 140, 0.44),
        Keypoint("left_elbow", 180, 160, 0.43),
        Keypoint("right_elbow", 280, 160, 0.43),
        Keypoint("left_hip", 210, 190, 0.42),
        Keypoint("right_hip", 250, 190, 0.42),
    ]
    weak_points = [replace(point, confidence=0.14) for point in body_points]

    good_pose = PoseFrameResult(timestamp=0, frame_index=1, keypoints=body_points, raw_keypoints=body_points, backend="rtmpose", confidence=0.43)
    weak_pose = PoseFrameResult(timestamp=0, frame_index=1, keypoints=weak_points, raw_keypoints=weak_points, backend="rtmpose", confidence=0.14)

    assert _roi_assisted_pose_score(good_pose, good_roi) > _roi_assisted_pose_score(weak_pose, full_roi)


def test_opencv_motion_backend_returns_bbox_without_fake_skeleton() -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    cv2.rectangle(frame, (45, 60), (170, 100), (255, 255, 255), -1)
    backend = OpenCVMotionBackend()
    backend.load()

    result = backend.infer_frame(frame, frame_index=1, timestamp=0.1)

    assert result.backend == "opencv_motion"
    assert result.bbox is not None
    assert result.keypoints == []
    assert "centroid_only_no_skeleton" in result.quality_flags


def _moving_frames(count: int, start_x: float, step_x: float) -> list[PoseFrameResult]:
    frames = []
    for index in range(count):
        x = start_x + index * step_x
        wave = 25 if index % 12 in {3, 4, 5} else 0
        points = [
            Keypoint("nose", x + 10, 80, 0.9),
            Keypoint("left_shoulder", x, 110, 0.9),
            Keypoint("right_shoulder", x + 30, 110, 0.9),
            Keypoint("left_elbow", x - 12, 125, 0.86),
            Keypoint("right_elbow", x + 42, 125, 0.86),
            Keypoint("left_wrist", x - 25, 145 + wave, 0.86),
            Keypoint("right_wrist", x + 55, 145 + (25 - wave), 0.86),
            Keypoint("left_hip", x + 5, 175, 0.9),
            Keypoint("right_hip", x + 25, 175, 0.9),
            Keypoint("left_knee", x + 8, 225, 0.85),
            Keypoint("right_knee", x + 22, 225, 0.85),
            Keypoint("left_ankle", x + 10, 285 + wave, 0.84),
            Keypoint("right_ankle", x + 20, 285 + (25 - wave), 0.84),
        ]
        frames.append(PoseFrameResult(index / 30, index, points, confidence=0.88, view_type="side", backend="synthetic"))
    return frames


def _spiky_motion_frames() -> list[PoseFrameResult]:
    frames = []
    xs = [100, 145, 190, 235, 280, 6000, 370, 415, 460, 505, 550, 595, 640, 685, 730, 775, 820, 865, 910, 955]
    for index, x in enumerate(xs):
        points = [
            Keypoint("left_shoulder", x, 110, 0.85),
            Keypoint("right_shoulder", x + 30, 110, 0.85),
            Keypoint("left_hip", x + 5, 175, 0.82),
            Keypoint("right_hip", x + 25, 175, 0.82),
            Keypoint("left_wrist", x - 25, 145, 0.72),
            Keypoint("right_wrist", x + 55, 145, 0.72),
        ]
        frames.append(PoseFrameResult(index * 0.5, index, points, confidence=0.82, view_type="side", backend="synthetic"))
    return frames


def _bbox_frames(count: int, duration: float, start_x: float, end_x: float) -> list[PoseFrameResult]:
    frames = []
    for index in range(count):
        progress = index / max(1, count - 1)
        x = start_x + (end_x - start_x) * progress
        timestamp = duration * progress
        frames.append(
            PoseFrameResult(
                timestamp=timestamp,
                frame_index=index,
                keypoints=[],
                confidence=0.42,
                bbox=[x - 35, 120, x + 35, 220],
                view_type="side",
                quality_flags=["centroid_only_no_skeleton"],
                backend="opencv_motion",
            )
        )
    return frames
