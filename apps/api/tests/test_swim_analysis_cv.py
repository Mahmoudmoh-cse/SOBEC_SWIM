from app.swim_analysis.metrics import compute_front_view_metrics, compute_side_view_metrics, compute_swimming_metrics
from app.swim_analysis.pose.base import Keypoint, PoseEstimator, PoseFrameResult
from app.swim_analysis.report import build_analysis_report
from app.swim_analysis.smoothing import moving_average, smooth_pose_sequence
from app.swim_analysis.sync import align_timelines, estimate_sync_offset_audio_stub, estimate_sync_offset_motion


class DummyPoseEstimator(PoseEstimator):
    name = "dummy"

    def load(self) -> None:
        self.loaded = True

    def infer_frame(self, frame, *, frame_index=0, timestamp=0.0, view_type="side", quality_flags=None) -> PoseFrameResult:
        return PoseFrameResult(
            timestamp=timestamp,
            frame_index=frame_index,
            keypoints=[Keypoint("nose", 10, 20, 0.9)],
            confidence=0.9,
            view_type=view_type,
            quality_flags=quality_flags or [],
            backend=self.name,
        )


def test_pose_backend_interface_infers_video_frames() -> None:
    estimator = DummyPoseEstimator()
    estimator.load()
    results = estimator.infer_video([(0, 0.0, object(), [], "side"), (1, 0.1, object(), ["dark"], "front")])

    assert estimator.loaded is True
    assert len(results) == 2
    assert results[1].quality_flags == ["dark"]
    assert results[0].keypoints[0].name == "nose"


def test_smoothing_functions_reduce_simple_series_noise() -> None:
    assert moving_average([0, 9, 0], window=3) == [4.5, 3.0, 4.5]
    frames = [
        PoseFrameResult(i / 30, i, [Keypoint("left_wrist", 100 + ((-1) ** i) * 8, 50, 0.9)], 0.9)
        for i in range(8)
    ]

    smoothed = smooth_pose_sequence(frames)

    raw_span = max(frame.keypoints[0].x for frame in frames) - min(frame.keypoints[0].x for frame in frames)
    smooth_span = max(frame.keypoints[0].x for frame in smoothed) - min(frame.keypoints[0].x for frame in smoothed)
    assert smooth_span < raw_span
    assert smoothed[0].raw_keypoints is not None


def test_smoothing_reset_does_not_hold_stale_wrong_joint() -> None:
    frames = [
        PoseFrameResult(0.0, 0, [Keypoint("left_wrist", 10, 50, 0.9)], 0.9),
        PoseFrameResult(
            0.1,
            1,
            [Keypoint("left_wrist", 280, 50, 0.05)],
            0.05,
            quality_flags=["smoothing_reset"],
            debug_info={"roi_tracking": {"tracking_state": "REVIEW_ONLY", "smoothing_reset": True}},
        ),
    ]

    smoothed = smooth_pose_sequence(frames, min_confidence=0.12)

    assert smoothed[1].keypoints[0].x == 280
    assert smoothed[1].debug_info["smoothing"]["reset"] is True


def test_sync_functions_are_deterministic() -> None:
    side = _synthetic_side_frames(count=40)
    front = _synthetic_front_frames(count=40)

    audio = estimate_sync_offset_audio_stub()
    motion = estimate_sync_offset_motion(side, front)
    aligned = align_timelines(side, front, manual_offset_sec=0.25)

    assert audio["offset_sec"] == 0.0
    assert "method" in motion
    assert aligned["offset_sec"] == 0.25
    assert aligned["front"][0].timestamp == 0.25


def test_metrics_use_synthetic_landmark_time_series() -> None:
    side = _synthetic_side_frames(count=90)
    front = _synthetic_front_frames(count=90)
    quality = {
        "side": {"usable_frame_ratio": 0.92, "blur_score": 0.82, "lighting_score": 0.9, "stability_score": 0.86},
        "front": {"usable_frame_ratio": 0.9, "blur_score": 0.8, "lighting_score": 0.88, "stability_score": 0.84},
    }

    side_metrics = compute_side_view_metrics(side)
    front_metrics = compute_front_view_metrics(front)
    combined = compute_swimming_metrics(side, front, stroke_type="freestyle", video_quality=quality)

    assert side_metrics["stroke_count"]["value"] > 4
    assert side_metrics["body_alignment_score"]["value"] >= 80
    assert front_metrics["arm_symmetry_score"]["value"] >= 80
    assert combined["overall_technique_score"]["value"] > 70


def test_low_confidence_metrics_are_marked_low() -> None:
    empty_frames = [PoseFrameResult(i / 30, i, [], 0.0) for i in range(12)]
    metrics = compute_side_view_metrics(empty_frames)

    assert metrics["stroke_rate_spm"]["confidence"] == "low"
    assert metrics["stroke_rate_spm"]["evidence"]["numeric_confidence"] == 0.0


def test_report_schema_contains_confidence_aware_sections() -> None:
    metrics = compute_swimming_metrics(
        _synthetic_side_frames(count=60),
        _synthetic_front_frames(count=60),
        stroke_type="freestyle",
        video_quality={
            "side": {"usable_frame_ratio": 0.8, "blur_score": 0.8, "lighting_score": 0.8, "stability_score": 0.8},
            "front": {"usable_frame_ratio": 0.8, "blur_score": 0.8, "lighting_score": 0.8, "stability_score": 0.8},
        },
    )
    report = build_analysis_report(
        analysis_id="analysis-1",
        stroke_type="freestyle",
        duration_sec=2.0,
        video_quality={},
        metrics=metrics,
        artifacts={"debug_landmarks_json_url": "/uploads/a.json"},
        metadata={},
        pose_backend="rtmpose",
    )

    assert report["analysis_id"] == "analysis-1"
    assert report["status"] == "completed"
    assert "confidence_score" in report["summary"]
    assert "metrics" in report
    assert isinstance(report["recommendations"], list)


def _synthetic_side_frames(count: int) -> list[PoseFrameResult]:
    frames = []
    for i in range(count):
        wave = _triangle_wave(i, period=15, amplitude=30)
        opposite = _triangle_wave(i + 7, period=15, amplitude=30)
        keypoints = [
            Keypoint("nose", 145, 95 + _triangle_wave(i, 30, 2), 0.92),
            Keypoint("left_shoulder", 170, 100, 0.95),
            Keypoint("right_shoulder", 190, 100, 0.95),
            Keypoint("left_elbow", 155, 105, 0.9),
            Keypoint("right_elbow", 205, 104, 0.9),
            Keypoint("left_wrist", 140, 95 + wave, 0.92),
            Keypoint("right_wrist", 220, 95 + opposite, 0.92),
            Keypoint("left_hip", 245, 104, 0.95),
            Keypoint("right_hip", 265, 104, 0.95),
            Keypoint("left_knee", 310, 106, 0.88),
            Keypoint("right_knee", 330, 106, 0.88),
            Keypoint("left_ankle", 375, 104 + _triangle_wave(i, 8, 12), 0.86),
            Keypoint("right_ankle", 395, 104 + _triangle_wave(i + 4, 8, 12), 0.86),
        ]
        frames.append(PoseFrameResult(i / 30, i, keypoints, 0.9, view_type="side", backend="synthetic"))
    return frames


def _synthetic_front_frames(count: int) -> list[PoseFrameResult]:
    frames = []
    for i in range(count):
        left_wave = _triangle_wave(i, period=15, amplitude=35)
        right_wave = _triangle_wave(i + 7, period=15, amplitude=35)
        keypoints = [
            Keypoint("nose", 220, 70, 0.92),
            Keypoint("left_shoulder", 180, 110, 0.95),
            Keypoint("right_shoulder", 260, 110, 0.95),
            Keypoint("left_elbow", 165, 155, 0.9),
            Keypoint("right_elbow", 275, 155, 0.9),
            Keypoint("left_wrist", 160 + left_wave, 210 + left_wave * 0.15, 0.92),
            Keypoint("right_wrist", 280 - right_wave, 210 + right_wave * 0.15, 0.92),
            Keypoint("left_hip", 195, 210, 0.95),
            Keypoint("right_hip", 245, 210, 0.95),
            Keypoint("left_knee", 198, 285, 0.9),
            Keypoint("right_knee", 242, 285, 0.9),
            Keypoint("left_ankle", 200, 350, 0.88),
            Keypoint("right_ankle", 240, 350, 0.88),
        ]
        frames.append(PoseFrameResult(i / 30, i, keypoints, 0.9, view_type="front", backend="synthetic"))
    return frames


def _triangle_wave(index: int, period: int, amplitude: float) -> float:
    phase = (index % period) / period
    return amplitude * (1 - abs(phase * 2 - 1))
