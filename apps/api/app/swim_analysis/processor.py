from __future__ import annotations

import logging
import csv
from dataclasses import replace
from math import ceil
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.models import SwimAnalysis
from app.swim_analysis.calibration import calibrate_pool_view
from app.swim_analysis.coaching_report import generate_coaching_report
from app.swim_analysis.faults import detect_swimming_faults
from app.swim_analysis.metrics import compute_swimming_metrics
from app.swim_analysis.phase1_metrics import compute_phase1_research_metrics, reliability_score_from_metrics
from app.swim_analysis.pose.base import Keypoint, PoseBackendUnavailable, PoseEstimator, PoseFrameResult
from app.swim_analysis.pose.mediapipe_backend import MediaPipeBackend
from app.swim_analysis.pose.opencv_motion_backend import OpenCVMotionBackend
from app.swim_analysis.pose.rtmpose_backend import RTMPoseBackend
from app.swim_analysis.pose.yolo_pose_backend import YOLOPoseBackend
from app.swim_analysis.preprocessing import assess_video_quality, iter_sample_video_frames, read_video_metadata
from app.swim_analysis.quality import enrich_video_quality
from app.swim_analysis.report import build_analysis_report, generate_annotated_video, write_json_artifact
from app.swim_analysis.roi import ROIConfig, ROIResult, SwimmerLocalizer, centered_roi_from_bbox, project_pose_to_original, skipped_pose_from_roi, stamp_pose_metadata
from app.swim_analysis.segmentation import segment_swim_phases
from app.swim_analysis.sync import align_timelines, estimate_sync_offset_motion
from app.swim_analysis.temporal_tracking import TemporalJointTracker, TemporalTrackingConfig
from app.swim_analysis.tracking import track_swimmer_identity
from app.swim_analysis.velocity import attach_phase_velocity_summary, compute_velocity_analysis, velocity_loss_after_events
from app.swim_analysis.visualizations import generate_research_figures


logger = logging.getLogger(__name__)


class SwimAnalysisProcessor:
    def __init__(self, *, target_fps: int = 6, quality_mode: str = "balanced", lane_length_m: float | None = None) -> None:
        self.target_fps = max(1, min(60, int(target_fps or 6)))
        self.quality_mode = quality_mode
        self.lane_length_m = lane_length_m
        self.settings = get_settings()

    def process(self, analysis: SwimAnalysis, *, progress_callback: Any | None = None) -> dict[str, Any]:
        root = Path(self.settings.upload_dir) / "swim_analysis" / analysis.id
        artifacts_dir = root / "artifacts"
        processed_dir = root / "processed"
        landmarks_dir = root / "landmarks"
        processing_errors: list[str] = []

        _progress(progress_callback, 8)
        side_metadata = read_video_metadata(analysis.side_video_path)
        single_video_mode = _same_video_path(analysis.side_video_path, analysis.front_video_path)
        front_metadata = side_metadata if single_video_mode else read_video_metadata(analysis.front_video_path)
        if not side_metadata.valid or not front_metadata.valid:
            raise ValueError(f"Invalid videos. side={side_metadata.error}; front={front_metadata.error}")

        side_quality = assess_video_quality(analysis.side_video_path)
        front_quality = side_quality if single_video_mode else assess_video_quality(analysis.front_video_path)
        metadata = {
            "side": side_metadata.to_dict(),
            "front": front_metadata.to_dict(),
            "input_mode": "single_video" if single_video_mode else "two_view",
        }
        if single_video_mode:
            metadata["video"] = side_metadata.to_dict()

        _progress(progress_callback, 18)
        max_dimension = _max_dimension_for_quality(self.quality_mode, self.settings.pose_device)
        side_frames = iter_sample_video_frames(
            analysis.side_video_path,
            target_fps=self.target_fps,
            view_type="side",
            max_dimension=max_dimension,
            enhance=True,
        )
        front_frames = None
        if not single_video_mode:
            front_frames = iter_sample_video_frames(
                analysis.front_video_path,
                target_fps=self.target_fps,
                view_type="front",
                max_dimension=max_dimension,
                enhance=True,
            )

        _progress(progress_callback, 30)
        estimator, backend_errors = load_pose_estimator()
        processing_errors.extend(backend_errors)
        analysis.pose_backend = estimator.name
        roi_config = _roi_config_from_settings(self.settings)
        roi_compare_full_frame = bool(self.settings.swim_roi_compare_full_frame and estimator.name == "rtmpose")
        roi_fallback_full_frame = bool(self.settings.swim_roi_fallback_full_frame)
        logger.info(
            "swim_analysis_backend analysis=%s requested=%s actual=%s fallback_errors=%s roi_enabled=%s roi_compare_full_frame=%s threshold=%s",
            analysis.id,
            self.settings.pose_backend,
            estimator.name,
            backend_errors,
            roi_config.enabled,
            roi_compare_full_frame,
            getattr(estimator, "confidence_threshold", None),
        )
        raw_side = _infer_pose_stream(
            estimator,
            side_frames,
            roi_config=roi_config,
            compare_full_frame=roi_compare_full_frame,
            fallback_full_frame=roi_fallback_full_frame,
            roi_debug_dir=artifacts_dir / "roi_debug" / ("video" if single_video_mode else "side"),
            progress_callback=progress_callback,
            start_progress=32,
            end_progress=48,
            expected_frames=_expected_sampled_frames(side_metadata.frame_count, side_metadata.fps, self.target_fps),
        )
        _progress(progress_callback, 48)
        if single_video_mode:
            raw_front = _clone_frames_for_view(raw_side, "front")
            _progress(progress_callback, 62)
        else:
            raw_front = _infer_pose_stream(
                estimator,
                front_frames,
                roi_config=roi_config,
                compare_full_frame=roi_compare_full_frame,
                fallback_full_frame=roi_fallback_full_frame,
                roi_debug_dir=artifacts_dir / "roi_debug" / "front",
                progress_callback=progress_callback,
                start_progress=50,
                end_progress=62,
                expected_frames=_expected_sampled_frames(front_metadata.frame_count, front_metadata.fps, self.target_fps),
            )
        _progress(progress_callback, 62)

        temporal_tracker = TemporalJointTracker(_temporal_config_from_settings(self.settings))
        side_identity = track_swimmer_identity(raw_side)
        front_identity = track_swimmer_identity(raw_front)
        side_tracking = temporal_tracker.process(side_identity, view_type="side")
        front_tracking = temporal_tracker.process(front_identity, view_type="front")
        side_smoothed = side_tracking.clean_frames
        front_smoothed = front_tracking.clean_frames

        sync = estimate_sync_offset_motion(side_smoothed, front_smoothed)
        aligned = align_timelines(
            side_smoothed,
            front_smoothed,
            manual_offset_sec=0.0,
            motion_offset_sec=float(sync.get("offset_sec", 0.0)),
        )
        aligned_side = aligned["side"]
        aligned_front = aligned["front"]
        assert isinstance(aligned_side, list)
        assert isinstance(aligned_front, list)

        _progress(progress_callback, 68)
        calibration = calibrate_pool_view(aligned_side, lane_length_m=self.lane_length_m)
        velocity = compute_velocity_analysis(aligned_side, calibration)
        phases = segment_swim_phases(aligned_side, velocity)
        velocity = attach_phase_velocity_summary(velocity, phases)

        side_quality_enriched = enrich_video_quality(side_quality.to_dict(), metadata["side"], aligned_side)
        front_quality_enriched = enrich_video_quality(front_quality.to_dict(), metadata["front"], aligned_front)
        metric_video_quality = {"side": side_quality_enriched, "front": front_quality_enriched}
        report_video_quality = {"video": side_quality_enriched} if single_video_mode else metric_video_quality

        _progress(progress_callback, 72)
        metrics = compute_swimming_metrics(
            aligned_side,
            aligned_front,
            stroke_type=analysis.stroke_type,
            video_quality=metric_video_quality,
            sync_offset_sec=float(aligned["offset_sec"]),
            calibration=calibration,
            velocity=velocity,
        )
        metrics.update(
            compute_phase1_research_metrics(
                aligned_side,
                aligned_front,
                metrics,
                side_tracking=side_tracking.trajectory_export,
                front_tracking=front_tracking.trajectory_export,
                video_quality=metric_video_quality,
                single_video_mode=single_video_mode,
            )
        )
        if single_video_mode:
            metrics = _adapt_metrics_for_single_video(metrics)
        breathing_times = metrics.get("breathing_timing", {}).get("evidence", {}).get("event_times_sec", [])
        if isinstance(breathing_times, list):
            velocity["breathing_velocity_loss"] = velocity_loss_after_events(velocity, [float(value) for value in breathing_times], label="breathing")
        faults = detect_swimming_faults(metrics, velocity, phases)
        coaching_report = generate_coaching_report(
            metrics=metrics,
            faults=faults,
            phases=phases,
            velocity=velocity,
            video_quality=report_video_quality,
            input_mode="single_video" if single_video_mode else "two_view",
        )

        video_trajectory_export = _trajectory_export_for_view(side_tracking.trajectory_export, "video")
        pose_diagnostics = _pose_diagnostics(
            requested_backend=self.settings.pose_backend,
            actual_backend=estimator.name,
            backend_errors=backend_errors,
            estimator=estimator,
            frames=raw_side if single_video_mode else [*raw_side, *raw_front],
            roi_config=roi_config,
            tracking_exports={"video": video_trajectory_export} if single_video_mode else {"side": side_tracking.trajectory_export, "front": front_tracking.trajectory_export},
            phases=phases,
        )
        raw_rtmpose_debug_path = write_json_artifact(landmarks_dir / "rtmpose_raw_keypoints_debug.json", _raw_pose_debug_payload(raw_side if single_video_mode else [*raw_side, *raw_front]))
        roi_debug_manifest_path = write_json_artifact(artifacts_dir / "roi_debug_manifest.json", _roi_debug_manifest(raw_side if single_video_mode else [*raw_side, *raw_front]))

        raw_landmarks_payload = {
            "video": [frame.to_dict() for frame in raw_side],
            "backend": estimator.name,
            "input_mode": "single_video",
        } if single_video_mode else {
            "side": [frame.to_dict() for frame in raw_side],
            "front": [frame.to_dict() for frame in raw_front],
            "backend": estimator.name,
        }
        raw_landmarks_path = write_json_artifact(
            landmarks_dir / "raw_landmarks.json",
            raw_landmarks_payload,
        )
        smoothed_landmarks_payload = {
            "video": [frame.to_dict() for frame in aligned_side],
            "sync": sync,
            "calibration": calibration,
            "input_mode": "single_video",
        } if single_video_mode else {
            "side": [frame.to_dict() for frame in aligned_side],
            "front": [frame.to_dict() for frame in aligned_front],
            "sync": sync,
            "calibration": calibration,
        }
        smoothed_landmarks_path = write_json_artifact(
            landmarks_dir / "smoothed_landmarks.json",
            smoothed_landmarks_payload,
        )
        trajectories_payload = {
            "video": video_trajectory_export,
            "backend": estimator.name,
            "sync": sync,
            "input_mode": "single_video",
        } if single_video_mode else {
            "side": side_tracking.trajectory_export,
            "front": front_tracking.trajectory_export,
            "backend": estimator.name,
            "sync": sync,
        }
        trajectories_path = write_json_artifact(landmarks_dir / "phase1_trajectories.json", trajectories_payload)

        _progress(progress_callback, 82)
        debug_overlays = {
            "show_raw_pose": bool(self.settings.swim_annotation_show_raw_pose),
            "show_debug": bool(self.settings.swim_annotation_show_debug),
            "show_roi": True,
        }
        side_annotated_path = generate_annotated_video(
            source_path=analysis.side_video_path,
            target_path=processed_dir / ("video_annotated.webm" if single_video_mode else "side_annotated.webm"),
            frames=aligned_side,
            metrics=metrics,
            view_type="video" if single_video_mode else "side",
            max_dimension=max_dimension,
            velocity=velocity,
            phases=phases,
            faults=faults,
            debug_overlays=debug_overlays,
        )
        front_annotated_path = None
        if not single_video_mode:
            front_annotated_path = generate_annotated_video(
                source_path=analysis.front_video_path,
                target_path=processed_dir / "front_annotated.webm",
                frames=aligned_front,
                metrics=metrics,
                view_type="front",
                max_dimension=max_dimension,
                velocity=velocity,
                phases=phases,
                faults=faults,
                debug_overlays=debug_overlays,
            )

        figures, figure_warnings = generate_research_figures(
            artifacts_dir / "figures",
            {"video": video_trajectory_export} if single_video_mode else {"side": side_tracking.trajectory_export, "front": front_tracking.trajectory_export},
            velocity=velocity,
        )
        research_figures = [_figure_with_upload_url(item) for item in figures]
        figures_manifest_path = write_json_artifact(artifacts_dir / "research_figures.json", research_figures)

        tracking_warnings = [*side_tracking.warnings] if single_video_mode else [*side_tracking.warnings, *front_tracking.warnings]
        side_quality_warnings = list(side_quality_enriched.get("warnings") or [])
        front_quality_warnings = list(front_quality_enriched.get("warnings") or [])
        quality_warnings = side_quality_warnings if single_video_mode else [*side_quality_warnings, *front_quality_warnings]
        velocity_warnings = list(velocity.get("summary", {}).get("warnings") or []) if isinstance(velocity, dict) else []
        pipeline_warnings = _dedupe_strings(
            [
                *tracking_warnings,
                *figure_warnings,
                *quality_warnings,
                *velocity_warnings,
            ]
        )
        if single_video_mode:
            pipeline_warnings.append("Single-video mode: all charts and review metrics are based on the uploaded video only, so camera-specific conclusions are lower confidence.")
        reliability_score = reliability_score_from_metrics(metrics)

        artifacts = {
            "video_annotated_video_url": _upload_url(side_annotated_path) if single_video_mode else None,
            "side_annotated_video_url": _upload_url(side_annotated_path),
            "front_annotated_video_url": _upload_url(front_annotated_path),
            "debug_landmarks_json_url": _upload_url(smoothed_landmarks_path),
            "raw_landmarks_json_url": _upload_url(raw_landmarks_path),
            "raw_pose_debug_json_url": _upload_url(raw_rtmpose_debug_path),
            "roi_debug_manifest_url": _upload_url(roi_debug_manifest_path),
            "roi_tracking_timeline_url": _upload_url(artifacts_dir / "roi_debug" / ("video" if single_video_mode else "side") / "roi_tracking_timeline.json"),
            "trajectory_data_url": _upload_url(trajectories_path),
            "research_figures_manifest_url": _upload_url(figures_manifest_path),
        }
        duration = min(side_metadata.duration_sec, front_metadata.duration_sec)
        report_trajectories = {
            "trajectory_data_url": _upload_url(trajectories_path),
            "video_visibility_percentage": side_tracking.visibility_percentage,
            "video_tracking_quality": side_tracking.tracking_quality_timeline,
            "video_joint_reliability": video_trajectory_export.get("joint_reliability", {}),
            "video_roi_history": video_trajectory_export.get("roi_history", []),
        } if single_video_mode else {
            "trajectory_data_url": _upload_url(trajectories_path),
            "side_visibility_percentage": side_tracking.visibility_percentage,
            "front_visibility_percentage": front_tracking.visibility_percentage,
            "side_tracking_quality": side_tracking.tracking_quality_timeline,
            "front_tracking_quality": front_tracking.tracking_quality_timeline,
            "side_joint_reliability": side_tracking.joint_reliability,
            "front_joint_reliability": front_tracking.joint_reliability,
            "side_roi_history": side_tracking.trajectory_export.get("roi_history", []),
            "front_roi_history": front_tracking.trajectory_export.get("roi_history", []),
        }
        debug_info = {
            "input_mode": "single_video",
            "roi_config": roi_config.__dict__,
            "temporal_config": _temporal_config_summary(_temporal_config_from_settings(self.settings)),
            "video_rejected_outliers": side_tracking.rejected_outliers[:50],
            "video_swap_events": side_tracking.swap_events[:50],
        } if single_video_mode else {
            "input_mode": "two_view",
            "roi_config": roi_config.__dict__,
            "temporal_config": _temporal_config_summary(_temporal_config_from_settings(self.settings)),
            "side_rejected_outliers": side_tracking.rejected_outliers[:50],
            "front_rejected_outliers": front_tracking.rejected_outliers[:50],
            "side_swap_events": side_tracking.swap_events[:50],
            "front_swap_events": front_tracking.swap_events[:50],
        }
        report = build_analysis_report(
            analysis_id=analysis.id,
            stroke_type=analysis.stroke_type,
            duration_sec=duration,
            video_quality=report_video_quality,
            metrics=metrics,
            artifacts=artifacts,
            metadata=metadata,
            pose_backend=estimator.name,
            processing_errors=processing_errors,
            calibration=calibration,
            velocity=velocity,
            phases=phases,
            faults=faults,
            coaching_report=coaching_report,
            warnings=pipeline_warnings,
            reliability_score=reliability_score,
            pose_diagnostics=pose_diagnostics,
            trajectories=report_trajectories,
            research_figures=research_figures,
            debug_info=debug_info,
        )
        report_path = write_json_artifact(artifacts_dir / "report.json", report)
        report["artifacts"]["report_json_url"] = _upload_url(report_path)

        _close_estimator(estimator)
        _progress(progress_callback, 94)
        return {
            "report": report,
            "side_annotated_path": side_annotated_path,
            "front_annotated_path": front_annotated_path,
            "raw_landmarks_path": raw_landmarks_path,
            "smoothed_landmarks_path": smoothed_landmarks_path,
            "trajectory_path": trajectories_path,
            "pose_backend": estimator.name,
            "metrics": metrics,
            "velocity": velocity,
            "phases": phases,
            "faults": faults,
            "calibration": calibration,
        }


def load_pose_estimator() -> tuple[PoseEstimator, list[str]]:
    settings = get_settings()
    errors: list[str] = []
    backend_names = [settings.pose_backend]
    fallback = settings.fallback_pose_backend if settings.pose_allow_fallback else None
    if fallback and fallback not in backend_names:
        backend_names.append(fallback)
    if "opencv_motion" not in backend_names:
        backend_names.append("opencv_motion")

    for index, name in enumerate(backend_names):
        estimator = _make_backend(name)
        try:
            estimator.load()
            if errors:
                logger.warning("pose_backend_fallback actual=%s errors=%s", estimator.name, errors)
            return estimator, errors
        except PoseBackendUnavailable as exc:
            message = f"{estimator.name}: {exc}"
            errors.append(message)
            logger.warning("pose_backend_load_failed requested=%s error=%s", estimator.name, exc)
            if index == 0 and not settings.pose_allow_fallback:
                raise PoseBackendUnavailable(message) from exc
    raise PoseBackendUnavailable("; ".join(errors) or "No pose backend configured.")


def _make_backend(name: str | None) -> PoseEstimator:
    normalized = (name or "rtmpose").strip().lower()
    if normalized in {"rtmpose", "rtmw", "mmpose"}:
        return RTMPoseBackend()
    if normalized in {"yolo", "yolo_pose", "ultralytics"}:
        return YOLOPoseBackend()
    if normalized in {"mediapipe", "mp"}:
        return MediaPipeBackend()
    if normalized in {"opencv", "opencv_motion", "centroid"}:
        return OpenCVMotionBackend()
    raise PoseBackendUnavailable(f"Unknown pose backend '{name}'.")


def _close_estimator(estimator: PoseEstimator) -> None:
    close = getattr(estimator, "close", None)
    if callable(close):
        close()


def _progress(callback: Any | None, value: int) -> None:
    if callback:
        callback(value)


def _infer_pose_stream(
    estimator: PoseEstimator,
    frames: Any,
    *,
    roi_config: ROIConfig,
    compare_full_frame: bool = False,
    fallback_full_frame: bool = True,
    roi_debug_dir: Path | None = None,
    progress_callback: Any | None,
    start_progress: int,
    end_progress: int,
    expected_frames: int,
) -> list[PoseFrameResult]:
    results: list[PoseFrameResult] = []
    expected_frames = max(1, expected_frames)
    span = max(0, end_progress - start_progress)
    last_progress = start_progress
    _progress(progress_callback, start_progress)
    localizer = SwimmerLocalizer(roi_config)
    roi_tracker = ROITrackingState()
    roi_timeline: list[dict[str, Any]] = []
    for count, (frame_index, timestamp, frame, quality_flags, view_type) in enumerate(frames, start=1):
        if roi_config.enabled:
            allow_previous = roi_tracker.allow_previous_roi()
            candidates = localizer.candidate_rois(frame, frame_index=frame_index, timestamp=timestamp, view_type=view_type, quality_flags=quality_flags, allow_previous=allow_previous)
            recovery_candidates = roi_tracker.recovery_candidates(frame, roi_config, frame_index=frame_index, timestamp=timestamp)
            if recovery_candidates:
                candidates = _merge_roi_candidates(candidates, recovery_candidates)
            if candidates:
                selected, debug_payload = _infer_best_roi_candidate(
                    estimator,
                    frame,
                    candidates,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    view_type=view_type,
                    quality_flags=quality_flags,
                    compare_full_frame=compare_full_frame,
                    fallback_full_frame=fallback_full_frame,
                    identity_tracker=roi_tracker,
                )
                selected_roi = debug_payload.get("selected_roi")
                primary_diag = roi_tracker.diagnose(selected, selected_roi if isinstance(selected_roi, ROIResult) else None, debug_payload, reattempt=False)
                non_previous = [candidate for candidate in candidates if candidate.source != "previous_roi"]
                if primary_diag["reacquisition_triggered"] and non_previous and any(candidate.source == "previous_roi" for candidate in candidates):
                    reacquired, reacquire_debug = _infer_best_roi_candidate(
                        estimator,
                        frame,
                        non_previous,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        view_type=view_type,
                        quality_flags=[*quality_flags, "roi_reacquire_search"],
                        compare_full_frame=compare_full_frame,
                        fallback_full_frame=fallback_full_frame,
                        identity_tracker=roi_tracker,
                    )
                    reacquired_roi = reacquire_debug.get("selected_roi")
                    reacquire_diag = roi_tracker.diagnose(reacquired, reacquired_roi if isinstance(reacquired_roi, ROIResult) else None, reacquire_debug, reattempt=True)
                    if _prefer_reacquired_pose(primary_diag, reacquire_diag):
                        selected = reacquired
                        selected_roi = reacquired_roi
                        debug_payload = reacquire_debug
                        primary_diag = reacquire_diag
                final_diag = roi_tracker.commit(selected, selected_roi if isinstance(selected_roi, ROIResult) else None, debug_payload, diagnosis=primary_diag)
                selected = _attach_roi_tracking_diagnostics(selected, final_diag)
                if isinstance(selected_roi, ROIResult) and final_diag["tracking_state"] in {"LOCKED", "SUSPECT"}:
                    localizer.remember_roi(roi_tracker.roi_for_memory or selected_roi)
                if final_diag["tracking_state"] in {"LOST", "REACQUIRE", "REVIEW_ONLY"}:
                    localizer.reset_tracking(keep_image_history=True)
                debug_payload["selected"] = selected
                _write_roi_debug_frame(roi_debug_dir, frame, frame_index=frame_index, debug_payload=debug_payload)
                roi_timeline.append(final_diag)
                results.append(selected)
            else:
                invalid_roi = ROIResult(False, None, frame_width=frame.shape[1], frame_height=frame.shape[0], reason="roi_no_candidates", source="none")
                if fallback_full_frame:
                    fallback_flags = [*(quality_flags or []), "roi_full_frame_fallback", "roi_no_candidates"]
                    pose = estimator.infer_frame(
                        frame,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        view_type=view_type,
                        quality_flags=fallback_flags,
                    )
                    stamped = stamp_pose_metadata(pose)
                    selected = replace(stamped, roi=invalid_roi.to_dict(), debug_events=[*stamped.debug_events, "ROI full-frame fallback: no candidates"], debug_info={**stamped.debug_info, "roi_decision": {"selected": "full_frame", "reason": "roi_no_candidates", "roi_valid": False}})
                    final_diag = roi_tracker.commit(selected, None, {"decision": {"selected": "full_frame", "fallback_reason": "roi_no_candidates"}}, diagnosis=roi_tracker.diagnose(selected, None, {}, reattempt=True))
                    selected = _attach_roi_tracking_diagnostics(selected, final_diag)
                    localizer.reset_tracking(keep_image_history=True)
                    _write_roi_debug_frame(roi_debug_dir, frame, frame_index=frame_index, debug_payload={"full_frame_pose": selected, "candidates": [], "selected": selected})
                    roi_timeline.append(final_diag)
                    results.append(selected)
                else:
                    selected = skipped_pose_from_roi(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        view_type=view_type,
                        quality_flags=quality_flags,
                        backend=estimator.name,
                        roi=invalid_roi,
                    )
                    final_diag = roi_tracker.commit(selected, None, {}, diagnosis=roi_tracker.diagnose(selected, None, {}, reattempt=True))
                    selected = _attach_roi_tracking_diagnostics(selected, final_diag)
                    roi_timeline.append(final_diag)
                    results.append(selected)
        else:
            selected = stamp_pose_metadata(
                estimator.infer_frame(
                    frame,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    view_type=view_type,
                    quality_flags=quality_flags,
                )
            )
            final_diag = roi_tracker.commit(selected, None, {"decision": {"selected": "full_frame", "fallback_reason": "roi_disabled"}}, diagnosis=roi_tracker.diagnose(selected, None, {}, reattempt=False))
            selected = _attach_roi_tracking_diagnostics(selected, final_diag)
            _write_roi_debug_frame(roi_debug_dir, frame, frame_index=frame_index, debug_payload={"full_frame_pose": selected, "candidates": [], "selected": selected})
            roi_timeline.append(final_diag)
            results.append(selected)
        frame_result = results[-1]
        logger.info(
            "pose_frame_result backend=%s frame=%s time=%.3f keypoints=%s raw_keypoints=%s confidence=%.3f skipped=%s flags=%s",
            frame_result.backend,
            frame_result.frame_index,
            frame_result.timestamp,
            len(frame_result.keypoints),
            len(frame_result.raw_keypoints or []),
            frame_result.confidence,
            frame_result.skipped,
            ",".join(frame_result.quality_flags[:6]),
        )
        next_progress = min(end_progress, start_progress + round(span * count / expected_frames))
        if next_progress > last_progress and (count == 1 or count % 8 == 0 or next_progress == end_progress):
            _progress(progress_callback, next_progress)
            last_progress = next_progress
    _progress(progress_callback, end_progress)
    _write_roi_tracking_timeline(roi_timeline, roi_debug_dir)
    return results


BODY_JOINT_NAMES = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}
FACE_JOINT_NAMES = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}


class ROITrackingState:
    def __init__(self) -> None:
        self.state = "REACQUIRE"
        self.bad_frames = 0
        self.previous_good_roi: ROIResult | None = None
        self.previous_good_pose_confidence: float = 0.0
        self.previous_good_body_confidence: float = 0.0
        self.previous_roi_speed_ratio_s: float | None = None
        self.previous_roi_timestamp: float | None = None
        self.frozen_roi_streak = 0
        self.roi_for_memory: ROIResult | None = None
        self.last_swimmer_roi: ROIResult | None = None
        self.previous_identity: dict[str, Any] | None = None
        self.last_swimmer_identity: dict[str, Any] | None = None
        self.identity_velocity_px_s: tuple[float, float] | None = None
        self.identity_timestamp: float | None = None
        self.pending_reacquire_identity: dict[str, Any] | None = None
        self.pending_reacquire_frames = 0

    def allow_previous_roi(self) -> bool:
        return self.state in {"LOCKED", "SUSPECT"} and self.bad_frames < 2

    def recovery_candidates(self, frame: Any, config: ROIConfig, *, frame_index: int, timestamp: float) -> list[ROIResult]:
        if self.state not in {"SUSPECT", "LOST", "REACQUIRE", "REVIEW_ONLY"} and self.bad_frames == 0:
            return []
        height, width = frame.shape[:2]
        temp_config = replace(config, scale=1.0, max_boundary_clip_ratio=max(float(config.max_boundary_clip_ratio), 0.58))
        candidates: list[ROIResult] = []
        last_roi = self.last_swimmer_roi
        if last_roi is not None and last_roi.valid:
            cx, cy = _box_center(_roi_box(last_roi, None))
            bbox = _centered_bbox(cx, cy, float(last_roi.width) * 1.55, float(last_roi.height) * 1.55, width=width, height=height)
            candidates.append(_recovery_roi(frame, bbox, temp_config, source="last_good_roi_expanded", confidence=0.56, score=0.54, frame_index=frame_index, timestamp=timestamp))
        identity = self.previous_identity or self.last_swimmer_identity
        predicted = self._predicted_identity_center(timestamp)
        if predicted and identity:
            scale = max(float(identity.get("scale", 0) or 0), float(config.min_size_px))
            bbox = _centered_bbox(float(predicted["x"]), float(predicted["y"]), scale * 1.28, scale * 0.92, width=width, height=height)
            candidates.append(_recovery_roi(frame, bbox, temp_config, source="predicted_trajectory_crop", confidence=0.6, score=0.58, frame_index=frame_index, timestamp=timestamp))
        if identity and isinstance(identity.get("center"), dict):
            center = identity["center"]
            scale = max(float(identity.get("scale", 0) or 0), float(config.min_size_px))
            bbox = _centered_bbox(float(center["x"]), float(center["y"]), scale * 1.36, scale, width=width, height=height)
            candidates.append(_recovery_roi(frame, bbox, temp_config, source="center_on_last_swimmer", confidence=0.52, score=0.5, frame_index=frame_index, timestamp=timestamp))
        output: list[ROIResult] = []
        for candidate in candidates:
            box = _roi_box(candidate, None)
            if not candidate.valid or candidate.crop is None or box is None:
                continue
            if any((existing_box := _roi_box(existing, None)) is not None and _box_iou(box, existing_box) > 0.82 for existing in output):
                continue
            output.append(candidate)
        return output[:3]

    def candidate_identity_diagnostics(self, pose: PoseFrameResult, roi: ROIResult | None) -> dict[str, Any]:
        features = _swimmer_identity_features(pose, roi)
        previous = self.previous_identity
        timestamp = float(pose.timestamp)
        predicted = self._predicted_identity_center(timestamp)
        reasons: list[str] = []
        soft_reasons: list[str] = []
        trajectory_distance_ratio = 0.0
        scale_change_ratio = 1.0
        direction_score = 0.5
        temporal_consistency = 0.5
        confidence_consistency = 0.5
        fully_lost = self._previous_swimmer_fully_lost()
        high_conf_plausible = False

        if previous and features.get("center"):
            center = features["center"]
            reference = max(24.0, float(previous.get("scale", 0) or 0), float(features.get("scale", 0) or 0))
            base_center = predicted or previous.get("center")
            if isinstance(base_center, dict):
                distance = ((float(center["x"]) - float(base_center["x"])) ** 2 + (float(center["y"]) - float(base_center["y"])) ** 2) ** 0.5
                trajectory_distance_ratio = distance / reference
            previous_area = max(1.0, float(previous.get("area", 0) or 0))
            area = max(1.0, float(features.get("area", 0) or 0))
            scale_change_ratio = max(area / previous_area, previous_area / area)
            direction_score = self._motion_direction_score(center, previous)
            temporal_consistency = max(0.0, min(1.0, 1.0 - trajectory_distance_ratio / (2.4 if fully_lost else 1.35)))
            high_conf_plausible = _high_confidence_plausible_roi(features, roi, trajectory_distance_ratio, direction_score)

            spatial_jump_limit = 2.35 if fully_lost else 1.55 if high_conf_plausible else 1.15
            if trajectory_distance_ratio > spatial_jump_limit:
                reasons.append("identity_spatial_jump")
            elif trajectory_distance_ratio > (1.05 if high_conf_plausible else 0.8):
                soft_reasons.append("identity_spatial_distance_high")
            if scale_change_ratio > (4.4 if high_conf_plausible else 3.3 if fully_lost else 2.35):
                reasons.append("identity_scale_mismatch")
            if direction_score < (0.12 if high_conf_plausible else 0.22) and trajectory_distance_ratio > 0.42 and not fully_lost:
                reasons.append("identity_motion_direction_conflict")

        pending_streak = self._pending_reacquire_streak(features)
        if self.pending_reacquire_identity:
            pending_conf = float(self.pending_reacquire_identity.get("pose_confidence", 0) or 0)
            confidence_delta = abs(float(features.get("pose_confidence", 0) or 0) - pending_conf)
            confidence_consistency = max(0.0, min(1.0, 1.0 - confidence_delta / 0.35))
        delayed_relock_required = bool(previous and self.state in {"LOST", "REACQUIRE", "REVIEW_ONLY"}) or (
            bool(previous) and trajectory_distance_ratio > 1.15
        )
        identity_relock_ready = not delayed_relock_required or (
            pending_streak >= 2
            and (fully_lost or trajectory_distance_ratio <= 1.55)
            and confidence_consistency >= 0.36
            and float(features.get("human_shape_score", 0) or 0) >= 0.42
        ) or (
            high_conf_plausible
            and trajectory_distance_ratio <= 0.95
            and float(features.get("body_joint_confidence", 0) or 0) >= 0.18
            and float(features.get("human_shape_score", 0) or 0) >= 0.6
        )
        if delayed_relock_required and not identity_relock_ready:
            reasons.append("delayed_relock_pending")

        if float(features.get("human_shape_score", 0) or 0) < 0.32:
            reasons.append("human_shape_untrusted")
        elif float(features.get("human_shape_score", 0) or 0) < 0.46:
            soft_reasons.append("human_shape_weak")
        if (
            float(features.get("head_body_ratio", 0) or 0) < 0.18
            and float(features.get("body_joint_confidence", 0) or 0) >= 0.18
            and bool(previous)
            and (delayed_relock_required or trajectory_distance_ratio > 0.7 or self.state in {"LOST", "REACQUIRE", "REVIEW_ONLY"})
        ):
            reasons.append("head_body_relationship_low")
        if float(features.get("body_joint_confidence", 0) or 0) >= 0.28 and float(features.get("core_structure_score", 0) or 0) < 0.36:
            reasons.append("splash_reflection_like_body_pose")

        hard_reasons = {
            "identity_spatial_jump",
            "identity_scale_mismatch",
            "identity_motion_direction_conflict",
            "human_shape_untrusted",
            "head_body_relationship_low",
            "splash_reflection_like_body_pose",
        }
        hard_reject = any(reason in hard_reasons for reason in reasons)
        identity_score = (
            float(features.get("human_shape_score", 0) or 0) * 0.38
            + temporal_consistency * 0.26
            + direction_score * 0.16
            + confidence_consistency * 0.1
            + min(1.0, float(features.get("body_joint_count", 0) or 0) / 8.0) * 0.1
        )
        identity_score -= min(0.45, len(reasons) * 0.12 + len(soft_reasons) * 0.04)
        return {
            "identity_score": round(max(0.0, min(1.0, identity_score)), 4),
            "identity_hard_reject": bool(hard_reject),
            "identity_rejection_reasons": _dedupe_strings(reasons),
            "identity_soft_reasons": _dedupe_strings(soft_reasons),
            "trajectory_distance_ratio": round(float(trajectory_distance_ratio), 4),
            "scale_change_ratio_identity": round(float(scale_change_ratio), 4),
            "motion_consistency_score": round(float(direction_score), 4),
            "temporal_consistency_score": round(float(temporal_consistency), 4),
            "confidence_consistency_score": round(float(confidence_consistency), 4),
            "delayed_relock_required": bool(delayed_relock_required),
            "pending_relock_streak": int(pending_streak),
            "identity_relock_ready": bool(identity_relock_ready),
            "high_confidence_plausible_roi": bool(high_conf_plausible),
            "previous_swimmer_fully_lost": bool(fully_lost),
            "predicted_center": predicted,
            "features": features,
        }

    def diagnose(self, pose: PoseFrameResult, roi: ROIResult | None, debug_payload: dict[str, Any], *, reattempt: bool) -> dict[str, Any]:
        metrics = _roi_tracking_metrics(pose, roi, self.previous_good_roi)
        metrics.update(self._motion_physics_metrics(metrics, roi))
        identity = _selected_identity_diagnostics(debug_payload)
        if identity:
            metrics.update(_identity_metrics_for_timeline(identity))
        reasons = _roi_drift_reasons(
            metrics,
            previous_pose_confidence=self.previous_good_pose_confidence,
            previous_body_confidence=self.previous_good_body_confidence,
        )
        body_confidence = float(metrics["body_joint_confidence"])
        pose_confidence = float(metrics["pose_confidence"])
        identity_ok = (
            float(metrics.get("identity_score", 1.0) or 0.0) >= 0.42
            and not bool(metrics.get("identity_hard_reject"))
            and bool(metrics.get("identity_relock_ready", True))
        )
        good_lock = body_confidence >= 0.16 and pose_confidence >= 0.12 and identity_ok and not _has_severe_drift(reasons)
        reacquisition_triggered = (
            reattempt
            or self.state in {"LOST", "REACQUIRE", "REVIEW_ONLY"}
            or _has_severe_drift(reasons)
            or (self.state == "SUSPECT" and bool(reasons))
            or body_confidence < 0.08
            or bool(metrics.get("identity_hard_reject"))
        )
        decision = debug_payload.get("decision") if isinstance(debug_payload.get("decision"), dict) else {}
        return {
            **metrics,
            "previous_tracking_state": self.state,
            "tracking_state": self.state,
            "drift_detected": bool(reasons),
            "drift_reasons": reasons,
            "reacquisition_triggered": bool(reacquisition_triggered),
            "smoothing_reset": False,
            "reacquire_attempt": bool(reattempt),
            "good_lock": bool(good_lock),
            "selected": decision.get("selected"),
            "fallback_reason": decision.get("fallback_reason"),
        }

    def commit(
        self,
        pose: PoseFrameResult,
        roi: ROIResult | None,
        debug_payload: dict[str, Any],
        *,
        diagnosis: dict[str, Any],
    ) -> dict[str, Any]:
        previous_state = self.state
        good_lock = bool(diagnosis.get("good_lock"))
        if good_lock:
            self.bad_frames = 0
            self.state = "LOCKED"
        else:
            self.bad_frames += 1
            if bool(diagnosis.get("reacquire_attempt")):
                self.state = "REVIEW_ONLY"
            elif self.bad_frames == 1:
                self.state = "SUSPECT"
            elif self.bad_frames >= 2:
                self.state = "LOST"

        reacquisition_triggered = bool(diagnosis.get("reacquisition_triggered")) or self.state in {"LOST", "REACQUIRE"}
        smoothing_reset = self.state in {"LOST", "REACQUIRE", "REVIEW_ONLY"} or previous_state in {"LOST", "REACQUIRE"} or reacquisition_triggered
        decision = debug_payload.get("decision") if isinstance(debug_payload.get("decision"), dict) else {}
        output = {
            **diagnosis,
            "previous_tracking_state": previous_state,
            "tracking_state": self.state,
            "bad_frame_streak": self.bad_frames,
            "reacquisition_triggered": reacquisition_triggered,
            "smoothing_reset": smoothing_reset,
            "selected_score": decision.get("selected_score", diagnosis.get("roi_score")),
            "best_roi_score": decision.get("best_roi_score"),
            "full_frame_score": decision.get("full_frame_score"),
            "candidate_rejections": _candidate_rejection_summary(decision.get("candidates", [])),
            "candidate_rejection_count": sum(1 for item in decision.get("candidates", []) if isinstance(item, dict) and item.get("rejected")),
        }
        self._update_motion_memory(output, roi)
        return output

    def _motion_physics_metrics(self, metrics: dict[str, Any], roi: ROIResult | None) -> dict[str, Any]:
        timestamp = float(metrics.get("timestamp", 0) or 0)
        center_jump = float(metrics.get("center_jump_ratio", 0) or 0)
        dt = max(1e-3, timestamp - self.previous_roi_timestamp) if self.previous_roi_timestamp is not None else 0.0
        speed_ratio_s = center_jump / dt if dt > 0 else 0.0
        acceleration_ratio_s2 = 0.0
        if dt > 0 and self.previous_roi_speed_ratio_s is not None:
            acceleration_ratio_s2 = abs(speed_ratio_s - self.previous_roi_speed_ratio_s) / dt
        nearly_frozen = center_jump < 0.012 and metrics.get("roi_source") == "previous_roi"
        projected_frozen_streak = self.frozen_roi_streak + 1 if nearly_frozen else 0
        body_offset = _body_to_roi_offset_ratio(metrics)
        return {
            "roi_speed_ratio_s": round(speed_ratio_s, 4),
            "roi_acceleration_ratio_s2": round(acceleration_ratio_s2, 4),
            "frozen_roi_streak": projected_frozen_streak,
            "body_to_roi_offset_ratio": round(body_offset, 4) if body_offset is not None else None,
            "motion_physics_anomaly": bool(
                speed_ratio_s > 2.9
                or acceleration_ratio_s2 > 8.5
                or projected_frozen_streak >= 3
                or (body_offset is not None and body_offset > 1.15)
            ),
        }

    def _update_motion_memory(self, diagnostic: dict[str, Any], roi: ROIResult | None) -> None:
        timestamp = float(diagnostic.get("timestamp", 0) or 0)
        if bool(diagnostic.get("good_lock")) and roi is not None and roi.valid:
            stable_roi = self._smoothed_roi_for_memory(roi, diagnostic)
            self.roi_for_memory = stable_roi
            self.previous_good_roi = stable_roi
            self.last_swimmer_roi = stable_roi
            self.previous_good_pose_confidence = float(diagnostic.get("pose_confidence", 0.0) or 0.0)
            self.previous_good_body_confidence = float(diagnostic.get("body_joint_confidence", 0.0) or 0.0)
            self.previous_roi_speed_ratio_s = float(diagnostic.get("roi_speed_ratio_s", 0.0) or 0.0)
            self.previous_roi_timestamp = timestamp
            self.frozen_roi_streak = int(diagnostic.get("frozen_roi_streak", 0) or 0)
            diagnostic["smoothed_roi_bbox"] = _roi_box(stable_roi, None)
            diagnostic["bbox_center_smoothing_applied"] = stable_roi is not roi
        elif self.state in {"LOST", "REVIEW_ONLY", "REACQUIRE"}:
            self.roi_for_memory = None
            self.previous_good_roi = None
            self.previous_good_pose_confidence = 0.0
            self.previous_good_body_confidence = 0.0
            self.previous_roi_speed_ratio_s = None
            self.previous_roi_timestamp = None
            self.frozen_roi_streak = 0
        else:
            self.roi_for_memory = roi if roi is not None and roi.valid else None
        self._update_identity_memory(diagnostic)

    def _smoothed_roi_for_memory(self, roi: ROIResult, diagnostic: dict[str, Any]) -> ROIResult:
        previous = self.previous_good_roi
        if previous is None or not previous.valid:
            return roi
        dt = max(1e-3, float(diagnostic.get("timestamp", 0) or 0) - float(previous.timestamp or diagnostic.get("timestamp", 0) or 0))
        previous_box = _roi_box(previous, None)
        current_box = _roi_box(roi, None)
        if previous_box is None or current_box is None:
            return roi
        prev_center = _box_center(previous_box)
        cur_center = _box_center(current_box)
        prev_size = _box_size(previous_box)
        cur_size = _box_size(current_box)
        alpha = 0.62 if float(diagnostic.get("body_joint_confidence", 0) or 0) >= 0.24 else 0.42
        max_move = max(prev_size) * max(0.08, min(0.75, 2.2 * dt))
        raw_dx = cur_center[0] - prev_center[0]
        raw_dy = cur_center[1] - prev_center[1]
        raw_dist = (raw_dx**2 + raw_dy**2) ** 0.5
        if raw_dist > max_move and raw_dist > 0:
            scale = max_move / raw_dist
            cur_center = (prev_center[0] + raw_dx * scale, prev_center[1] + raw_dy * scale)
        smoothed_center = (prev_center[0] * (1 - alpha) + cur_center[0] * alpha, prev_center[1] * (1 - alpha) + cur_center[1] * alpha)
        size_alpha = min(alpha, 0.48)
        smoothed_size = (
            max(24.0, prev_size[0] * (1 - size_alpha) + cur_size[0] * size_alpha),
            max(24.0, prev_size[1] * (1 - size_alpha) + cur_size[1] * size_alpha),
        )
        x = int(round(max(0.0, min(float(roi.frame_width) - smoothed_size[0], smoothed_center[0] - smoothed_size[0] / 2.0))))
        y = int(round(max(0.0, min(float(roi.frame_height) - smoothed_size[1], smoothed_center[1] - smoothed_size[1] / 2.0))))
        width = int(round(min(float(roi.frame_width) - x, smoothed_size[0])))
        height = int(round(min(float(roi.frame_height) - y, smoothed_size[1])))
        if width <= 0 or height <= 0:
            return roi
        metrics = dict(roi.metrics or {})
        metrics.update({"center_smoothed": True, "size_smoothed": True, "raw_bbox": current_box})
        return replace(roi, x=x, y=y, width=width, height=height, metrics=metrics)

    def _predicted_identity_center(self, timestamp: float) -> dict[str, float] | None:
        if not self.previous_identity or not self.previous_identity.get("center"):
            return None
        center = self.previous_identity["center"]
        if not self.identity_velocity_px_s or self.identity_timestamp is None:
            return {"x": round(float(center["x"]), 3), "y": round(float(center["y"]), 3)}
        dt = max(0.0, min(1.4, timestamp - float(self.identity_timestamp)))
        return {
            "x": round(float(center["x"]) + self.identity_velocity_px_s[0] * dt, 3),
            "y": round(float(center["y"]) + self.identity_velocity_px_s[1] * dt, 3),
        }

    def _motion_direction_score(self, center: dict[str, float], previous: dict[str, Any]) -> float:
        velocity = self.identity_velocity_px_s
        previous_center = previous.get("center") if isinstance(previous.get("center"), dict) else None
        if not velocity or not previous_center:
            return 0.5
        vx, vy = velocity
        speed = (vx**2 + vy**2) ** 0.5
        dx = float(center["x"]) - float(previous_center["x"])
        dy = float(center["y"]) - float(previous_center["y"])
        distance = (dx**2 + dy**2) ** 0.5
        if speed < 12 or distance < 6:
            return 0.62
        cosine = (dx * vx + dy * vy) / max(1e-6, distance * speed)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def _pending_reacquire_streak(self, features: dict[str, Any]) -> int:
        if not self.pending_reacquire_identity or not features.get("center"):
            return 1
        pending_center = self.pending_reacquire_identity.get("center")
        if not isinstance(pending_center, dict):
            return 1
        center = features["center"]
        reference = max(24.0, float(self.pending_reacquire_identity.get("scale", 0) or 0), float(features.get("scale", 0) or 0))
        distance = ((float(center["x"]) - float(pending_center["x"])) ** 2 + (float(center["y"]) - float(pending_center["y"])) ** 2) ** 0.5
        previous_area = max(1.0, float(self.pending_reacquire_identity.get("area", 0) or 0))
        area = max(1.0, float(features.get("area", 0) or 0))
        scale_ratio = max(area / previous_area, previous_area / area)
        if distance / reference <= 0.62 and scale_ratio <= 1.85:
            return self.pending_reacquire_frames + 1
        return 1

    def _previous_swimmer_fully_lost(self) -> bool:
        return self.state in {"LOST", "REACQUIRE", "REVIEW_ONLY"} and self.bad_frames >= 3

    def _update_identity_memory(self, diagnostic: dict[str, Any]) -> None:
        features = diagnostic.get("identity_features") if isinstance(diagnostic.get("identity_features"), dict) else None
        if not features or not features.get("center"):
            return
        timestamp = float(diagnostic.get("timestamp", 0) or 0)
        if bool(diagnostic.get("good_lock")) and not bool(diagnostic.get("identity_hard_reject")):
            if self.previous_identity and self.previous_identity.get("center") and self.identity_timestamp is not None:
                previous_center = self.previous_identity["center"]
                dt = max(1e-3, timestamp - self.identity_timestamp)
                self.identity_velocity_px_s = (
                    (float(features["center"]["x"]) - float(previous_center["x"])) / dt,
                    (float(features["center"]["y"]) - float(previous_center["y"])) / dt,
                )
            self.previous_identity = _identity_memory_from_features(features, timestamp)
            self.last_swimmer_identity = self.previous_identity
            self.identity_timestamp = timestamp
            self.pending_reacquire_identity = None
            self.pending_reacquire_frames = 0
            return
        if bool(diagnostic.get("delayed_relock_required")) and not bool(diagnostic.get("identity_hard_reject")):
            self.pending_reacquire_frames = int(diagnostic.get("pending_relock_streak", 1) or 1)
            self.pending_reacquire_identity = _identity_memory_from_features(features, timestamp)
        elif bool(diagnostic.get("identity_hard_reject")):
            self.pending_reacquire_identity = None
            self.pending_reacquire_frames = 0


def _prefer_reacquired_pose(primary: dict[str, Any], reacquired: dict[str, Any]) -> bool:
    if reacquired.get("identity_hard_reject"):
        return False
    if reacquired.get("good_lock"):
        return True
    if primary.get("roi_source") == "previous_roi" and primary.get("drift_detected"):
        return True
    return float(reacquired.get("body_joint_confidence", 0) or 0) >= float(primary.get("body_joint_confidence", 0) or 0) + 0.02


def _attach_roi_tracking_diagnostics(pose: PoseFrameResult, diagnostic: dict[str, Any]) -> PoseFrameResult:
    flags = [*pose.quality_flags, f"tracking_state_{diagnostic['tracking_state'].lower()}"]
    if diagnostic.get("drift_detected"):
        flags.append("roi_drift_detected")
    if diagnostic.get("reacquisition_triggered"):
        flags.append("roi_reacquisition_triggered")
    if diagnostic.get("motion_physics_anomaly"):
        flags.append("motion_physics_anomaly")
    if diagnostic.get("smoothing_reset"):
        flags.append("smoothing_reset")
    return replace(
        pose,
        quality_flags=flags,
        debug_events=[
            *pose.debug_events,
            f"Tracking state: {diagnostic['tracking_state']}",
            *[f"ROI drift: {reason}" for reason in diagnostic.get("drift_reasons", [])[:4]],
        ],
        debug_info={**pose.debug_info, "roi_tracking": diagnostic},
    )


def _roi_tracking_metrics(pose: PoseFrameResult, roi: ROIResult | None, previous: ROIResult | None) -> dict[str, Any]:
    raw = pose.raw_keypoints or pose.keypoints
    body = [point for point in raw if point.name in BODY_JOINT_NAMES]
    face = [point for point in raw if point.name in FACE_JOINT_NAMES]
    body_visible = [point for point in body if point.confidence >= 0.05]
    face_visible = [point for point in face if point.confidence >= 0.05]
    body_confidence = sum(point.confidence for point in body) / max(1, len(body))
    face_confidence = sum(point.confidence for point in face) / max(1, len(face))
    box = _roi_box(roi, pose)
    center = _box_center(box)
    size = _box_size(box)
    body_center = _body_centroid(body_visible)
    previous_box = _roi_box(previous, None)
    center_jump_ratio = 0.0
    scale_change_ratio = 1.0
    iou = None
    if box and previous_box:
        distance = ((center[0] - _box_center(previous_box)[0]) ** 2 + (center[1] - _box_center(previous_box)[1]) ** 2) ** 0.5
        normalizer = max(1.0, max(_box_size(previous_box)))
        center_jump_ratio = distance / normalizer
        area = max(1.0, size[0] * size[1])
        previous_area = max(1.0, _box_size(previous_box)[0] * _box_size(previous_box)[1])
        scale_change_ratio = max(area / previous_area, previous_area / area)
        iou = _box_iou(box, previous_box)
    roi_metrics = roi.metrics if roi is not None else {}
    roi_score = float(roi.candidate_score) if roi is not None else 0.0
    roi_source = str(roi.source) if roi is not None else "full_frame"
    boundary = float(roi.boundary_touch_ratio) if roi is not None else 0.0
    aspect = size[0] / max(1.0, size[1]) if size[0] and size[1] else 0.0
    return {
        "frame_index": pose.frame_index,
        "timestamp": round(float(pose.timestamp), 4),
        "roi_bbox": [round(value, 3) for value in box] if box else None,
        "roi_source": roi_source,
        "roi_score": round(roi_score, 4),
        "pose_confidence": round(float(pose.confidence), 4),
        "body_joint_confidence": round(float(body_confidence), 4),
        "face_joint_confidence": round(float(face_confidence), 4),
        "body_joint_count": len(body_visible),
        "face_joint_count": len(face_visible),
        "bbox_center": {"x": round(center[0], 3), "y": round(center[1], 3)} if box else None,
        "bbox_size": {"width": round(size[0], 3), "height": round(size[1], 3)} if box else None,
        "body_centroid": {"x": round(body_center[0], 3), "y": round(body_center[1], 3)} if body_center else None,
        "center_jump_ratio": round(float(center_jump_ratio), 4),
        "scale_change_ratio": round(float(scale_change_ratio), 4),
        "iou_with_previous_roi": round(float(iou), 4) if iou is not None else None,
        "boundary_touch_ratio": round(boundary, 4),
        "aspect_ratio": round(float(aspect), 4),
        "temporal_motion_supported": bool(roi_metrics.get("temporal_motion_supported")) if isinstance(roi_metrics, dict) else False,
        "temporal_jump_ratio": float(roi_metrics.get("temporal_jump_ratio", center_jump_ratio)) if isinstance(roi_metrics, dict) else center_jump_ratio,
    }


def _roi_drift_reasons(metrics: dict[str, Any], *, previous_pose_confidence: float, previous_body_confidence: float) -> list[str]:
    reasons: list[str] = []
    reasons.extend(str(reason) for reason in metrics.get("identity_rejection_reasons", []) or [])
    motion_supported = bool(metrics.get("temporal_motion_supported"))
    if float(metrics.get("center_jump_ratio", 0) or 0) > (2.35 if motion_supported else 1.15):
        reasons.append("center_jump_without_motion_support" if not motion_supported else "large_motion_supported_center_jump")
    if float(metrics.get("roi_speed_ratio_s", 0) or 0) > 2.9:
        reasons.append("teleporting_bbox_center")
    if float(metrics.get("roi_acceleration_ratio_s2", 0) or 0) > 8.5:
        reasons.append("unrealistic_roi_acceleration_spike")
    if float(metrics.get("scale_change_ratio", 1) or 1) > 2.4:
        reasons.append("roi_scale_change")
    if int(metrics.get("frozen_roi_streak", 0) or 0) >= 3 and float(metrics.get("body_joint_confidence", 0) or 0) < 0.18:
        reasons.append("frozen_roi_plateau")
    body_offset = metrics.get("body_to_roi_offset_ratio")
    if body_offset is not None and float(body_offset) > 1.15 and float(metrics.get("body_joint_confidence", 0) or 0) >= 0.12:
        reasons.append("roi_body_motion_disagreement")
    iou = metrics.get("iou_with_previous_roi")
    if iou is not None and float(iou) < 0.06 and not motion_supported and metrics.get("roi_source") == "previous_roi":
        reasons.append("previous_roi_overlap_lost")
    pose_confidence = float(metrics.get("pose_confidence", 0) or 0)
    body_confidence = float(metrics.get("body_joint_confidence", 0) or 0)
    face_confidence = float(metrics.get("face_joint_confidence", 0) or 0)
    if previous_pose_confidence > 0 and pose_confidence < previous_pose_confidence * 0.55 and body_confidence < max(0.12, previous_body_confidence * 0.65):
        reasons.append("pose_confidence_drop")
    if body_confidence < 0.09 and face_confidence > body_confidence + 0.12:
        reasons.append("face_only_or_background_pose")
    if float(metrics.get("aspect_ratio", 0) or 0) > 7.2 or 0 < float(metrics.get("aspect_ratio", 0) or 0) < 0.35:
        reasons.append("lane_line_or_thin_roi_shape")
    if float(metrics.get("boundary_touch_ratio", 0) or 0) > 0.24:
        reasons.append("roi_border_touch")
    if int(metrics.get("body_joint_count", 0) or 0) < 4 and body_confidence < 0.12:
        reasons.append("body_joints_missing")
    return _dedupe_strings(reasons)


def _has_severe_drift(reasons: list[str]) -> bool:
    severe = {
        "center_jump_without_motion_support",
        "teleporting_bbox_center",
        "unrealistic_roi_acceleration_spike",
        "roi_scale_change",
        "frozen_roi_plateau",
        "roi_body_motion_disagreement",
        "identity_spatial_jump",
        "identity_scale_mismatch",
        "identity_motion_direction_conflict",
        "human_shape_untrusted",
        "head_body_relationship_low",
        "splash_reflection_like_body_pose",
        "delayed_relock_pending",
        "face_only_or_background_pose",
        "lane_line_or_thin_roi_shape",
        "body_joints_missing",
    }
    return any(reason in severe for reason in reasons)


def _roi_box(roi: ROIResult | None, pose: PoseFrameResult | None) -> list[float] | None:
    if roi is not None and roi.valid and roi.width > 0 and roi.height > 0:
        return [float(roi.x), float(roi.y), float(roi.x + roi.width), float(roi.y + roi.height)]
    if pose is not None and pose.bbox:
        return [float(value) for value in pose.bbox[:4]]
    return None


def _box_center(box: list[float] | None) -> tuple[float, float]:
    if not box:
        return (0.0, 0.0)
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_size(box: list[float] | None) -> tuple[float, float]:
    if not box:
        return (0.0, 0.0)
    return (max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1]))


def _box_iou(first: list[float], second: list[float]) -> float:
    inter_w = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    inter_h = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = inter_w * inter_h
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / max(1.0, first_area + second_area - intersection)


def _body_centroid(points: list[Any]) -> tuple[float, float] | None:
    if not points:
        return None
    return (
        sum(float(point.x) for point in points) / len(points),
        sum(float(point.y) for point in points) / len(points),
    )


def _body_to_roi_offset_ratio(metrics: dict[str, Any]) -> float | None:
    center = metrics.get("bbox_center") if isinstance(metrics.get("bbox_center"), dict) else None
    body = metrics.get("body_centroid") if isinstance(metrics.get("body_centroid"), dict) else None
    size = metrics.get("bbox_size") if isinstance(metrics.get("bbox_size"), dict) else None
    if not center or not body or not size:
        return None
    normalizer = max(1.0, float(size.get("width", 0) or 0), float(size.get("height", 0) or 0))
    distance = ((float(center.get("x", 0) or 0) - float(body.get("x", 0) or 0)) ** 2 + (float(center.get("y", 0) or 0) - float(body.get("y", 0) or 0)) ** 2) ** 0.5
    return distance / normalizer


def _high_confidence_plausible_roi(features: dict[str, Any], roi: ROIResult | None, trajectory_distance_ratio: float, direction_score: float) -> bool:
    if roi is None or not roi.valid:
        return False
    source = str(roi.source or "")
    if source == "full_frame":
        return False
    body_confidence = float(features.get("body_joint_confidence", 0) or 0)
    pose_confidence = float(features.get("pose_confidence", 0) or 0)
    human_shape = float(features.get("human_shape_score", 0) or 0)
    core_structure = float(features.get("core_structure_score", 0) or 0)
    head_body_ratio = float(features.get("head_body_ratio", 0) or 0)
    body_count = int(features.get("body_joint_count", 0) or 0)
    roi_score = float(roi.candidate_score or 0)
    roi_confidence = float(roi.confidence or 0)
    boundary = float(roi.boundary_touch_ratio or 0)
    recovery_source = source in {"last_good_roi_expanded", "predicted_trajectory_crop", "center_on_last_swimmer"}
    enough_pose = body_confidence >= 0.16 or pose_confidence >= 0.26 or roi_confidence >= 0.54
    enough_shape = human_shape >= 0.5 and core_structure >= 0.25 and body_count >= 4
    enough_roi = roi_score >= 0.34 and boundary <= 0.24
    continuity_ok = trajectory_distance_ratio <= (1.75 if recovery_source else 1.35) and direction_score >= (0.16 if recovery_source else 0.24)
    head_ok = head_body_ratio >= 0.16 or trajectory_distance_ratio <= 0.65 or recovery_source
    return bool(enough_pose and enough_shape and enough_roi and continuity_ok and head_ok)


def _candidate_rejection_summary(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        identity = candidate.get("identity") if isinstance(candidate.get("identity"), dict) else {}
        rejection_reasons = list(candidate.get("rejection_reasons") or identity.get("rejection_reasons") or [])
        soft_reasons = list(identity.get("soft_reasons") or [])
        status = "rejected" if bool(candidate.get("rejected")) else "accepted"
        output.append(
            {
                "rank": index + 1,
                "source": candidate.get("source"),
                "status": status,
                "rejected": bool(candidate.get("rejected")),
                "rejection_reasons": rejection_reasons,
                "soft_reasons": soft_reasons,
                "rtmpose_base_score": candidate.get("rtmpose_base_score"),
                "rtmpose_selection_score": candidate.get("rtmpose_selection_score"),
                "roi_confidence": candidate.get("roi_confidence"),
                "candidate_score": candidate.get("candidate_score"),
                "trajectory_distance_ratio": identity.get("trajectory_distance_ratio"),
                "motion_consistency_score": identity.get("motion_consistency_score"),
                "human_shape_score": identity.get("human_shape_score"),
                "head_body_ratio": identity.get("head_body_ratio"),
                "high_confidence_plausible_roi": identity.get("high_confidence_plausible_roi"),
            }
        )
    return output


def _merge_roi_candidates(primary: list[ROIResult], recovery: list[ROIResult]) -> list[ROIResult]:
    output: list[ROIResult] = []
    for candidate in [*recovery, *primary]:
        box = _roi_box(candidate, None)
        if box is None:
            continue
        if any((existing_box := _roi_box(existing, None)) is not None and _box_iou(box, existing_box) > 0.82 for existing in output):
            continue
        output.append(candidate)
    return output


def _centered_bbox(cx: float, cy: float, box_width: float, box_height: float, *, width: int, height: int) -> list[float]:
    frame_width = float(width)
    frame_height = float(height)
    box_w = max(24.0, min(float(box_width), frame_width))
    box_h = max(24.0, min(float(box_height), frame_height))
    x1 = max(0.0, min(frame_width - box_w, float(cx) - box_w / 2.0))
    y1 = max(0.0, min(frame_height - box_h, float(cy) - box_h / 2.0))
    return [x1, y1, x1 + box_w, y1 + box_h]


def _recovery_roi(frame: Any, bbox: list[float], config: ROIConfig, *, source: str, confidence: float, score: float, frame_index: int, timestamp: float) -> ROIResult:
    roi = centered_roi_from_bbox(frame, bbox, confidence=confidence, config=config, source=source, reason=source, candidate_score=score)
    metrics = dict(roi.metrics or {})
    metrics.update({"recovery_fallback": True, "recovery_source": source})
    return replace(roi, metrics=metrics, frame_index=frame_index, timestamp=timestamp)


def _write_roi_tracking_timeline(timeline: list[dict[str, Any]], roi_debug_dir: Path | None) -> None:
    if not timeline:
        return
    targets: list[Path] = []
    if roi_debug_dir is not None:
        targets.append(Path(roi_debug_dir))
    root_debug = Path(__file__).resolve().parents[4] / "debug_outputs"
    targets.append(root_debug)
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        json_path = target / "roi_tracking_timeline.json"
        csv_path = target / "roi_tracking_timeline.csv"
        write_json_artifact(json_path, {"frames": timeline})
        _write_roi_tracking_csv(csv_path, timeline)


def _write_roi_tracking_csv(path: Path, timeline: list[dict[str, Any]]) -> None:
    fieldnames = [
        "frame_index",
        "timestamp",
        "tracking_state",
        "roi_source",
        "roi_score",
        "pose_confidence",
        "body_joint_confidence",
        "face_joint_confidence",
        "body_joint_count",
        "center_x",
        "center_y",
        "width",
        "height",
        "center_jump_ratio",
        "roi_speed_ratio_s",
        "roi_acceleration_ratio_s2",
        "scale_change_ratio",
        "iou_with_previous_roi",
        "boundary_touch_ratio",
        "aspect_ratio",
        "frozen_roi_streak",
        "body_to_roi_offset_ratio",
        "motion_physics_anomaly",
        "smoothed_roi_bbox",
        "identity_score",
        "human_shape_score",
        "head_body_ratio",
        "trajectory_distance_ratio",
        "motion_consistency_score",
        "temporal_consistency_score",
        "pending_relock_streak",
        "identity_hard_reject",
        "identity_relock_ready",
        "high_confidence_plausible_roi",
        "candidate_rejection_count",
        "candidate_rejections",
        "drift_detected",
        "drift_reasons",
        "reacquisition_triggered",
        "smoothing_reset",
        "bad_frame_streak",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in timeline:
            center = item.get("bbox_center") if isinstance(item.get("bbox_center"), dict) else {}
            size = item.get("bbox_size") if isinstance(item.get("bbox_size"), dict) else {}
            writer.writerow(
                {
                    "frame_index": item.get("frame_index"),
                    "timestamp": item.get("timestamp"),
                    "tracking_state": item.get("tracking_state"),
                    "roi_source": item.get("roi_source"),
                    "roi_score": item.get("roi_score"),
                    "pose_confidence": item.get("pose_confidence"),
                    "body_joint_confidence": item.get("body_joint_confidence"),
                    "face_joint_confidence": item.get("face_joint_confidence"),
                    "body_joint_count": item.get("body_joint_count"),
                    "center_x": center.get("x"),
                    "center_y": center.get("y"),
                    "width": size.get("width"),
                    "height": size.get("height"),
                    "center_jump_ratio": item.get("center_jump_ratio"),
                    "roi_speed_ratio_s": item.get("roi_speed_ratio_s"),
                    "roi_acceleration_ratio_s2": item.get("roi_acceleration_ratio_s2"),
                    "scale_change_ratio": item.get("scale_change_ratio"),
                    "iou_with_previous_roi": item.get("iou_with_previous_roi"),
                    "boundary_touch_ratio": item.get("boundary_touch_ratio"),
                    "aspect_ratio": item.get("aspect_ratio"),
                    "frozen_roi_streak": item.get("frozen_roi_streak"),
                    "body_to_roi_offset_ratio": item.get("body_to_roi_offset_ratio"),
                    "motion_physics_anomaly": item.get("motion_physics_anomaly"),
                    "smoothed_roi_bbox": item.get("smoothed_roi_bbox"),
                    "identity_score": item.get("identity_score"),
                    "human_shape_score": item.get("human_shape_score"),
                    "head_body_ratio": item.get("head_body_ratio"),
                    "trajectory_distance_ratio": item.get("trajectory_distance_ratio"),
                    "motion_consistency_score": item.get("motion_consistency_score"),
                    "temporal_consistency_score": item.get("temporal_consistency_score"),
                    "pending_relock_streak": item.get("pending_relock_streak"),
                    "identity_hard_reject": item.get("identity_hard_reject"),
                    "identity_relock_ready": item.get("identity_relock_ready"),
                    "high_confidence_plausible_roi": item.get("high_confidence_plausible_roi"),
                    "candidate_rejection_count": item.get("candidate_rejection_count"),
                    "candidate_rejections": ";".join(
                        f"{entry.get('source')}:{','.join(entry.get('rejection_reasons') or entry.get('soft_reasons') or ['accepted'])}"
                        for entry in item.get("candidate_rejections", [])
                        if isinstance(entry, dict)
                    ),
                    "drift_detected": item.get("drift_detected"),
                    "drift_reasons": ";".join(item.get("drift_reasons", [])),
                    "reacquisition_triggered": item.get("reacquisition_triggered"),
                    "smoothing_reset": item.get("smoothing_reset"),
                    "bad_frame_streak": item.get("bad_frame_streak"),
                }
            )


def _infer_best_roi_candidate(
    estimator: PoseEstimator,
    frame: Any,
    candidates: list[ROIResult],
    *,
    frame_index: int,
    timestamp: float,
    view_type: str,
    quality_flags: list[str],
    compare_full_frame: bool,
    fallback_full_frame: bool,
    identity_tracker: ROITrackingState | None = None,
) -> tuple[PoseFrameResult, dict[str, Any]]:
    candidate_payloads: list[dict[str, Any]] = []
    scored_candidates: list[dict[str, Any]] = []
    best_roi_pose: PoseFrameResult | None = None
    best_roi: ROIResult | None = None
    best_score = -1.0
    best_identity: dict[str, Any] | None = None
    for candidate in candidates:
        pose = estimator.infer_frame(
            candidate.crop,
            frame_index=frame_index,
            timestamp=timestamp,
            view_type=view_type,
            quality_flags=[*(quality_flags or []), "roi_candidate", f"roi_source_{candidate.source}"],
        )
        projected = project_pose_to_original(pose, candidate)
        base_score = _roi_assisted_pose_score(projected, candidate)
        identity = identity_tracker.candidate_identity_diagnostics(projected, candidate) if identity_tracker is not None else _neutral_identity_diagnostics(projected, candidate)
        score = _identity_adjusted_roi_score(base_score, identity)
        roi_payload = _pose_candidate_summary(projected)
        roi_payload.update(
            {
                "source": candidate.source,
                "roi_confidence": round(float(candidate.confidence), 4),
                "candidate_score": round(float(candidate.candidate_score), 4),
                "boundary_touch_ratio": round(float(candidate.boundary_touch_ratio), 4),
                "rtmpose_base_score": round(base_score, 4),
                "rtmpose_selection_score": round(score, 4),
                "identity": _public_identity_diagnostics(identity),
                "rejected": bool(identity.get("identity_hard_reject")),
                "rejection_reasons": identity.get("identity_rejection_reasons", []),
                "roi": candidate.to_dict(),
            }
        )
        candidate_payloads.append(roi_payload)
        scored_candidates.append({"roi": candidate, "pose": projected, "score": score, "base_score": base_score, "identity": identity, "payload": roi_payload})
        if score > best_score:
            best_score = score
            best_roi_pose = projected
            best_roi = candidate
            best_identity = identity

    accepted_candidates = [record for record in scored_candidates if not bool(record.get("identity", {}).get("identity_hard_reject"))]
    if accepted_candidates:
        best_record = max(accepted_candidates, key=lambda item: float(item.get("score", 0.0)))
        best_score = float(best_record.get("score", 0.0))
        best_roi_pose = best_record.get("pose") if isinstance(best_record.get("pose"), PoseFrameResult) else best_roi_pose
        best_roi = best_record.get("roi") if isinstance(best_record.get("roi"), ROIResult) else best_roi
        best_identity = best_record.get("identity") if isinstance(best_record.get("identity"), dict) else best_identity

    full_pose = stamp_pose_metadata(
        estimator.infer_frame(
            frame,
            frame_index=frame_index,
            timestamp=timestamp,
            view_type=view_type,
            quality_flags=[*(quality_flags or []), "roi_full_frame_reference"],
        )
    )
    full_score = _full_frame_reference_score(full_pose)
    selected = best_roi_pose
    selected_roi = best_roi
    selected_kind = "roi"
    fallback_reason = None
    all_swimmer_centered_fallbacks_failed = not accepted_candidates and (
        selected is None
        or best_score < 0.08
        or bool(best_identity and best_identity.get("identity_hard_reject"))
    )
    if selected is None:
        selected = full_pose
        selected_roi = None
        selected_kind = "full_frame"
        fallback_reason = "roi_candidates_failed"
    elif (
        fallback_full_frame
        and compare_full_frame
        and all_swimmer_centered_fallbacks_failed
        and full_score > best_score + 0.08
        and full_score >= 0.2
        and not (best_identity and best_identity.get("delayed_relock_required") and not best_identity.get("identity_hard_reject"))
    ):
        selected = full_pose
        selected_roi = None
        selected_kind = "full_frame"
        fallback_reason = "full_frame_reference_scored_higher_than_weak_roi"

    decision = {
        "selected": selected_kind,
        "selected_source": selected_roi.source if selected_roi else "full_frame",
        "selected_score": round(_roi_assisted_pose_score(selected, selected_roi) if selected_roi else full_score, 4),
        "best_roi_score": round(best_score, 4),
        "full_frame_score": round(full_score, 4),
        "fallback_reason": fallback_reason,
        "candidates": sorted(candidate_payloads, key=lambda item: float(item.get("rtmpose_selection_score", 0.0)), reverse=True),
        "full_frame": _pose_candidate_summary(full_pose),
        "selected_identity": _public_identity_diagnostics(best_identity or {}) if selected_roi else {},
    }
    if selected_roi is not None:
        decision["selected_score"] = round(float(best_score), 4)
    display_records = sorted(scored_candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    display_candidates: list[ROIResult] = []
    if selected_roi is not None:
        display_candidates.append(selected_roi)
    for record in display_records:
        roi = record.get("roi")
        roi_box = _roi_box(roi, None) if isinstance(roi, ROIResult) else None
        if (
            isinstance(roi, ROIResult)
            and roi is not selected_roi
            and roi_box is not None
            and not any((existing_box := _roi_box(existing, None)) is not None and _box_iou(roi_box, existing_box) > 0.78 for existing in display_candidates)
        ):
            display_candidates.append(roi)
        if len(display_candidates) >= 2:
            break
    if selected_roi is not None:
        roi_dict = selected_roi.to_dict()
        roi_dict["rtmpose_confidence"] = round(float(selected.confidence), 4)
        roi_dict["rtmpose_selection_score"] = decision["selected_score"]
        roi_dict["identity"] = decision["selected_identity"]
        selected = replace(
            selected,
            roi=roi_dict,
            quality_flags=[*selected.quality_flags, "roi_compare_selected_roi", f"roi_selected_{selected_roi.source}"],
            debug_events=[*selected.debug_events, f"ROI selected: {selected_roi.source}"],
            debug_info={**selected.debug_info, "roi_decision": decision},
        )
    else:
        selected = replace(
            selected,
            quality_flags=[*selected.quality_flags, "roi_full_frame_fallback"],
            debug_events=[*selected.debug_events, f"ROI full-frame fallback: {fallback_reason or 'no selected ROI'}"],
            debug_info={**selected.debug_info, "roi_decision": decision},
        )
    return selected, {
        "selected": selected,
        "selected_roi": selected_roi,
        "full_frame_pose": full_pose,
        "roi_pose": best_roi_pose,
        "candidates": candidates,
        "display_candidates": display_candidates,
        "candidate_payloads": candidate_payloads,
        "selected_identity": best_identity or {} if selected_roi is not None else {},
        "decision": decision,
    }


def _roi_assisted_pose_score(pose: PoseFrameResult, roi: ROIResult | None) -> float:
    raw = pose.raw_keypoints or pose.keypoints
    body = [point for point in raw if point.name not in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}]
    face = [point for point in raw if point.name in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}]
    body_avg = sum(point.confidence for point in body) / max(1, len(body))
    face_avg = sum(point.confidence for point in face) / max(1, len(face))
    all_avg = sum(point.confidence for point in raw) / max(1, len(raw))
    body_support = sum(1 for point in body if point.confidence >= 0.05) / 12.0
    face_only_penalty = 0.18 if face_avg > body_avg + 0.12 and body_support < 0.45 else 0.0
    roi_score = float(roi.candidate_score) if roi is not None else 0.0
    boundary_penalty = float(roi.boundary_touch_ratio) * 0.35 if roi is not None else 0.08
    shape_penalty = 0.0
    previous_low_conf_penalty = 0.0
    if roi is not None:
        aspect = float((roi.metrics or {}).get("aspect_ratio", 0) or 0)
        if aspect > 7.2 or 0 < aspect < 0.35:
            shape_penalty = 0.18
        if roi.source == "previous_roi" and (body_avg < 0.12 or body_support < 0.25):
            previous_low_conf_penalty = 0.22
    score = pose.confidence * 0.5 + body_avg * 0.28 + body_support * 0.14 + roi_score * 0.08 - face_only_penalty - boundary_penalty - shape_penalty - previous_low_conf_penalty
    return max(0.0, min(1.0, score))


def _identity_adjusted_roi_score(base_score: float, identity: dict[str, Any]) -> float:
    identity_score = float(identity.get("identity_score", 0.5) or 0.0)
    score = float(base_score) * 0.62 + identity_score * 0.38
    if identity.get("identity_hard_reject"):
        score -= 0.42
    score -= min(0.18, len(identity.get("identity_rejection_reasons", []) or []) * 0.045)
    score -= min(0.08, len(identity.get("identity_soft_reasons", []) or []) * 0.025)
    return max(0.0, min(1.0, score))


def _neutral_identity_diagnostics(pose: PoseFrameResult, roi: ROIResult | None) -> dict[str, Any]:
    features = _swimmer_identity_features(pose, roi)
    human_shape = float(features.get("human_shape_score", 0.5) or 0.0)
    return {
        "identity_score": round(max(0.0, min(1.0, human_shape * 0.75 + 0.18)), 4),
        "identity_hard_reject": False,
        "identity_rejection_reasons": [],
        "identity_soft_reasons": [],
        "trajectory_distance_ratio": 0.0,
        "scale_change_ratio_identity": 1.0,
        "motion_consistency_score": 0.5,
        "temporal_consistency_score": 0.5,
        "confidence_consistency_score": 0.5,
        "delayed_relock_required": False,
        "pending_relock_streak": 0,
        "identity_relock_ready": True,
        "high_confidence_plausible_roi": False,
        "previous_swimmer_fully_lost": False,
        "predicted_center": None,
        "features": features,
    }


def _public_identity_diagnostics(identity: dict[str, Any]) -> dict[str, Any]:
    features = identity.get("features") if isinstance(identity.get("features"), dict) else {}
    return {
        "identity_score": identity.get("identity_score"),
        "identity_hard_reject": identity.get("identity_hard_reject"),
        "rejection_reasons": list(identity.get("identity_rejection_reasons") or []),
        "soft_reasons": list(identity.get("identity_soft_reasons") or []),
        "trajectory_distance_ratio": identity.get("trajectory_distance_ratio"),
        "scale_change_ratio": identity.get("scale_change_ratio_identity"),
        "motion_consistency_score": identity.get("motion_consistency_score"),
        "temporal_consistency_score": identity.get("temporal_consistency_score"),
        "confidence_consistency_score": identity.get("confidence_consistency_score"),
        "delayed_relock_required": identity.get("delayed_relock_required"),
        "pending_relock_streak": identity.get("pending_relock_streak"),
        "identity_relock_ready": identity.get("identity_relock_ready"),
        "high_confidence_plausible_roi": identity.get("high_confidence_plausible_roi"),
        "predicted_center": identity.get("predicted_center"),
        "human_shape_score": features.get("human_shape_score"),
        "core_structure_score": features.get("core_structure_score"),
        "head_body_ratio": features.get("head_body_ratio"),
        "body_orientation_deg": features.get("body_orientation_deg"),
        "center_source": features.get("center_source"),
    }


def _selected_identity_diagnostics(debug_payload: dict[str, Any]) -> dict[str, Any]:
    identity = debug_payload.get("selected_identity") if isinstance(debug_payload, dict) else None
    return identity if isinstance(identity, dict) else {}


def _identity_metrics_for_timeline(identity: dict[str, Any]) -> dict[str, Any]:
    features = identity.get("features") if isinstance(identity.get("features"), dict) else {}
    return {
        "identity_score": float(identity.get("identity_score", 0) or 0),
        "identity_hard_reject": bool(identity.get("identity_hard_reject")),
        "identity_rejection_reasons": list(identity.get("identity_rejection_reasons") or []),
        "identity_soft_reasons": list(identity.get("identity_soft_reasons") or []),
        "trajectory_distance_ratio": float(identity.get("trajectory_distance_ratio", 0) or 0),
        "scale_change_ratio_identity": float(identity.get("scale_change_ratio_identity", 1) or 1),
        "motion_consistency_score": float(identity.get("motion_consistency_score", 0) or 0),
        "temporal_consistency_score": float(identity.get("temporal_consistency_score", 0) or 0),
        "confidence_consistency_score": float(identity.get("confidence_consistency_score", 0) or 0),
        "delayed_relock_required": bool(identity.get("delayed_relock_required")),
        "pending_relock_streak": int(identity.get("pending_relock_streak", 0) or 0),
        "identity_relock_ready": bool(identity.get("identity_relock_ready", True)),
        "high_confidence_plausible_roi": bool(identity.get("high_confidence_plausible_roi")),
        "human_shape_score": float(features.get("human_shape_score", 0) or 0),
        "core_structure_score": float(features.get("core_structure_score", 0) or 0),
        "head_body_ratio": float(features.get("head_body_ratio", 0) or 0),
        "body_orientation_deg": features.get("body_orientation_deg"),
        "identity_features": features,
    }


def _swimmer_identity_features(pose: PoseFrameResult, roi: ROIResult | None) -> dict[str, Any]:
    raw = pose.raw_keypoints or pose.keypoints
    by_name = {point.name: point for point in raw}
    body = [point for point in raw if point.name in BODY_JOINT_NAMES]
    face = [point for point in raw if point.name in FACE_JOINT_NAMES]
    body_visible = [point for point in body if point.confidence >= 0.05]
    face_visible = [point for point in face if point.confidence >= 0.05]
    core_names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    core_points = [_identity_point(by_name, name, 0.05) for name in core_names]
    core_count = sum(1 for point in core_points if point is not None)
    shoulders = (_identity_point(by_name, "left_shoulder", 0.05), _identity_point(by_name, "right_shoulder", 0.05))
    hips = (_identity_point(by_name, "left_hip", 0.05), _identity_point(by_name, "right_hip", 0.05))
    shoulder_mid = _midpoint(*shoulders)
    hip_mid = _midpoint(*hips)
    shoulder_width = _distance(shoulders[0], shoulders[1]) if shoulders[0] and shoulders[1] else None
    hip_width = _distance(hips[0], hips[1]) if hips[0] and hips[1] else None
    torso_length = _distance(shoulder_mid, hip_mid) if shoulder_mid and hip_mid else None
    orientation = _orientation_degrees(shoulder_mid, hip_mid) if shoulder_mid and hip_mid else None
    body_confidence = sum(float(point.confidence) for point in body) / max(1, len(body))
    head_confidence = sum(float(point.confidence) for point in face) / max(1, len(face))
    head_body_ratio = head_confidence / max(0.01, body_confidence)
    center = _identity_body_center(body_visible) or _identity_roi_center(roi) or _identity_pose_bbox_center(pose)
    center_source = "body_centroid" if _identity_body_center(body_visible) else "roi_center" if _identity_roi_center(roi) else "pose_bbox"
    spread = _identity_point_spread(body_visible)
    roi_size = _box_size(_roi_box(roi, pose))
    scale = max(24.0, roi_size[0], roi_size[1], spread.get("width", 0), spread.get("height", 0))
    area = max(1.0, roi_size[0] * roi_size[1], spread.get("width", 0) * spread.get("height", 0))
    core_structure_score = core_count / 4.0
    torso_score = 0.35
    if torso_length is not None and 0.035 * scale <= torso_length <= 1.18 * scale:
        torso_score = 1.0
    elif torso_length is not None and torso_length > 2:
        torso_score = 0.62
    width_score = 0.45
    widths = [value for value in [shoulder_width, hip_width] if value is not None]
    if widths:
        width_score = max(0.2, min(1.0, sum(widths) / len(widths) / max(18.0, scale * 0.34)))
    limb_spread_score = max(0.0, min(1.0, (spread.get("width", 0) + spread.get("height", 0)) / max(1.0, scale * 1.35)))
    head_score = 1.0 if head_confidence >= 0.1 else 0.6 if face_visible else 0.24
    human_shape_score = (
        core_structure_score * 0.34
        + torso_score * 0.18
        + width_score * 0.16
        + limb_spread_score * 0.14
        + head_score * 0.18
    )
    return {
        "center": {"x": round(center[0], 3), "y": round(center[1], 3)} if center else None,
        "center_source": center_source,
        "scale": round(float(scale), 3),
        "area": round(float(area), 3),
        "body_joint_confidence": round(float(body_confidence), 4),
        "head_confidence": round(float(head_confidence), 4),
        "head_body_ratio": round(float(head_body_ratio), 4),
        "body_joint_count": len(body_visible),
        "head_joint_count": len(face_visible),
        "core_structure_score": round(float(core_structure_score), 4),
        "human_shape_score": round(max(0.0, min(1.0, human_shape_score)), 4),
        "body_orientation_deg": round(float(orientation), 3) if orientation is not None else None,
        "shoulder_width_px": round(float(shoulder_width), 3) if shoulder_width is not None else None,
        "hip_width_px": round(float(hip_width), 3) if hip_width is not None else None,
        "torso_length_px": round(float(torso_length), 3) if torso_length is not None else None,
        "limb_spread_score": round(float(limb_spread_score), 4),
        "pose_confidence": round(float(pose.confidence), 4),
    }


def _identity_memory_from_features(features: dict[str, Any], timestamp: float) -> dict[str, Any]:
    return {
        "center": features.get("center"),
        "scale": features.get("scale"),
        "area": features.get("area"),
        "body_orientation_deg": features.get("body_orientation_deg"),
        "body_joint_confidence": features.get("body_joint_confidence"),
        "head_confidence": features.get("head_confidence"),
        "human_shape_score": features.get("human_shape_score"),
        "pose_confidence": features.get("pose_confidence"),
        "timestamp": round(float(timestamp), 4),
    }


def _identity_point(points: dict[str, Any], name: str, min_confidence: float) -> Any | None:
    point = points.get(name)
    if point is None or float(point.confidence) < min_confidence:
        return None
    return point


def _identity_body_center(points: list[Any]) -> tuple[float, float] | None:
    core = [point for point in points if point.name in {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}]
    samples = core if len(core) >= 2 else points
    if not samples:
        return None
    return (sum(float(point.x) for point in samples) / len(samples), sum(float(point.y) for point in samples) / len(samples))


def _identity_roi_center(roi: ROIResult | None) -> tuple[float, float] | None:
    if roi is None or not roi.valid:
        return None
    return (float(roi.x) + float(roi.width) / 2.0, float(roi.y) + float(roi.height) / 2.0)


def _identity_pose_bbox_center(pose: PoseFrameResult) -> tuple[float, float] | None:
    if not pose.bbox:
        return None
    box = [float(value) for value in pose.bbox[:4]]
    return _box_center(box)


def _identity_point_spread(points: list[Any]) -> dict[str, float]:
    if not points:
        return {"width": 0.0, "height": 0.0}
    xs = [float(point.x) for point in points]
    ys = [float(point.y) for point in points]
    return {"width": max(xs) - min(xs), "height": max(ys) - min(ys)}


def _orientation_degrees(first: Any | None, second: Any | None) -> float | None:
    if first is None or second is None:
        return None
    from math import atan2, degrees

    return degrees(atan2(float(second.y) - float(first.y), float(second.x) - float(first.x)))


def _midpoint(a: Any | None, b: Any | None) -> Keypoint | None:
    if a is None or b is None:
        return None
    return Keypoint("midpoint", (float(a.x) + float(b.x)) / 2.0, (float(a.y) + float(b.y)) / 2.0, min(float(a.confidence), float(b.confidence)))


def _distance(first: Any | None, second: Any | None) -> float:
    if first is None or second is None:
        return 0.0
    return ((float(first.x) - float(second.x)) ** 2 + (float(first.y) - float(second.y)) ** 2) ** 0.5


def _full_frame_reference_score(pose: PoseFrameResult) -> float:
    return max(0.0, _roi_assisted_pose_score(pose, None) - 0.08)


def _expected_sampled_frames(frame_count: int, fps: float, target_fps: int) -> int:
    source_fps = fps or 30.0
    sample_every = max(1, ceil(source_fps / max(1, target_fps)))
    return max(1, (max(0, frame_count) + sample_every - 1) // sample_every)


def _use_full_frame_for_boundary_clipped_roi(roi: Any) -> bool:
    reason = str(getattr(roi, "reason", "") or "")
    return reason.startswith("roi_boundary_clip_ratio_")


def _select_pose_candidate(roi_pose: PoseFrameResult, full_pose: PoseFrameResult) -> PoseFrameResult:
    roi_score = _pose_selection_score(roi_pose)
    full_score = _pose_selection_score(full_pose)
    roi_debug = _pose_candidate_summary(roi_pose)
    full_debug = _pose_candidate_summary(full_pose)
    decision = {
        "roi": roi_debug,
        "full_frame": full_debug,
        "roi_score": round(roi_score, 4),
        "full_frame_score": round(full_score, 4),
    }
    if full_score > roi_score + 0.08:
        return replace(
            full_pose,
            quality_flags=[*full_pose.quality_flags, "roi_compare_selected_full_frame"],
            debug_events=[*full_pose.debug_events, "ROI comparison selected full-frame pose"],
            debug_info={**full_pose.debug_info, "roi_decision": {**decision, "selected": "full_frame"}},
        )
    return replace(
        roi_pose,
        quality_flags=[*roi_pose.quality_flags, "roi_compare_selected_roi"],
        debug_events=[*roi_pose.debug_events, "ROI comparison selected ROI-remapped pose"],
        debug_info={**roi_pose.debug_info, "roi_decision": {**decision, "selected": "roi"}},
    )


def _pose_selection_score(pose: PoseFrameResult) -> float:
    if pose.skipped:
        return 0.0
    raw = pose.raw_keypoints or pose.keypoints
    if not raw:
        return 0.0
    body = [point for point in raw if point.name not in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"} and point.confidence >= 0.05]
    visible = [point for point in pose.keypoints if point.confidence >= 0.05]
    avg = sum(point.confidence for point in raw) / max(1, len(raw))
    body_support = len(body) / 12.0
    visible_support = len(visible) / 17.0
    return max(0.0, min(1.0, body_support * 0.48 + visible_support * 0.28 + avg * 0.24))


def _pose_candidate_summary(pose: PoseFrameResult) -> dict[str, Any]:
    raw = pose.raw_keypoints or pose.keypoints
    visible = [point for point in pose.keypoints if point.confidence >= 0.05]
    body = [point for point in visible if point.name not in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}]
    return {
        "backend": pose.backend,
        "keypoints": len(pose.keypoints),
        "raw_keypoints": len(raw),
        "visible_at_005": len(visible),
        "body_visible_at_005": len(body),
        "avg_confidence": round(float(pose.confidence), 4),
        "bbox": pose.bbox,
        "coordinate_validity": _coordinate_validity(pose),
        "flags": pose.quality_flags[:8],
    }


def _coordinate_validity(pose: PoseFrameResult) -> dict[str, Any]:
    roi = pose.roi or {}
    width = float(roi.get("frame_width") or 0)
    height = float(roi.get("frame_height") or 0)
    if width <= 0 or height <= 0:
        return {"checked": False}
    raw = pose.raw_keypoints or pose.keypoints
    valid = [point for point in raw if 0 <= float(point.x) <= width and 0 <= float(point.y) <= height]
    return {"checked": True, "valid": len(valid), "total": len(raw), "frame_width": width, "frame_height": height}


def _write_roi_debug_frame(roi_debug_dir: Path | None, frame: Any, *, frame_index: int, debug_payload: dict[str, Any]) -> None:
    if roi_debug_dir is None:
        return
    try:
        import cv2  # type: ignore
    except ImportError:
        return
    roi_debug_dir.mkdir(parents=True, exist_ok=True)
    original = frame.copy()
    candidates = debug_payload.get("display_candidates") or debug_payload.get("candidates") or []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, ROIResult):
            continue
        color = (80, 220, 80) if candidate is debug_payload.get("selected_roi") else (0, 170, 255)
        cv2.rectangle(original, (int(candidate.x), int(candidate.y)), (int(candidate.x + candidate.width), int(candidate.y + candidate.height)), color, 2)
        label = f"{index}:{candidate.source} {candidate.candidate_score:.2f}"
        cv2.putText(original, label[:40], (int(candidate.x), max(18, int(candidate.y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    decision = debug_payload.get("decision")
    if isinstance(decision, dict):
        text = f"selected={decision.get('selected_source', decision.get('selected'))} score={decision.get('selected_score')}"
        cv2.putText(original, text[:80], (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(roi_debug_dir / f"frame_{frame_index:06d}_original_roi.jpg"), original)

    selected_roi = debug_payload.get("selected_roi")
    if isinstance(selected_roi, ROIResult) and selected_roi.crop is not None:
        cv2.imwrite(str(roi_debug_dir / f"frame_{frame_index:06d}_selected_crop_{selected_roi.source}.jpg"), selected_roi.crop)
    roi_pose = debug_payload.get("roi_pose")
    if isinstance(roi_pose, PoseFrameResult):
        cv2.imwrite(str(roi_debug_dir / f"frame_{frame_index:06d}_roi_remapped_skeleton.jpg"), _draw_pose_debug(frame.copy(), roi_pose, cv2))
    full_pose = debug_payload.get("full_frame_pose")
    if isinstance(full_pose, PoseFrameResult):
        cv2.imwrite(str(roi_debug_dir / f"frame_{frame_index:06d}_full_frame_skeleton.jpg"), _draw_pose_debug(frame.copy(), full_pose, cv2))


def _draw_pose_debug(image: Any, pose: PoseFrameResult, cv2: Any) -> Any:
    links = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    points = {point.name: point for point in pose.keypoints}
    for start, end in links:
        first = points.get(start)
        second = points.get(end)
        if first and second:
            cv2.line(image, (int(first.x), int(first.y)), (int(second.x), int(second.y)), (0, 220, 255), 2)
    for point in pose.raw_keypoints or []:
        cv2.circle(image, (int(point.x), int(point.y)), 4, (80, 80, 255), 1)
    for point in pose.keypoints:
        cv2.circle(image, (int(point.x), int(point.y)), 5, (0, 255, 120), -1)
    return image


def _same_video_path(first: str | Path | None, second: str | Path | None) -> bool:
    if not first or not second:
        return False
    try:
        return Path(first).resolve() == Path(second).resolve()
    except OSError:
        return str(first) == str(second)


def _clone_frames_for_view(frames: list[PoseFrameResult], view_type: str) -> list[PoseFrameResult]:
    output: list[PoseFrameResult] = []
    for frame in frames:
        raw_keypoints = [replace(point) for point in frame.raw_keypoints] if frame.raw_keypoints is not None else None
        output.append(
            replace(
                frame,
                view_type=view_type,
                keypoints=[replace(point) for point in frame.keypoints],
                raw_keypoints=raw_keypoints,
                rejected_keypoints=[replace(point) for point in frame.rejected_keypoints],
                quality_flags=list(frame.quality_flags),
                debug_events=list(frame.debug_events),
                debug_info=dict(frame.debug_info),
            )
        )
    return output


def _trajectory_export_for_view(export: dict[str, Any], view_type: str) -> dict[str, Any]:
    output = dict(export)
    output["view_type"] = view_type
    return output


def _pose_diagnostics(
    *,
    requested_backend: str | None,
    actual_backend: str,
    backend_errors: list[str],
    estimator: PoseEstimator,
    frames: list[PoseFrameResult],
    roi_config: ROIConfig,
    tracking_exports: dict[str, dict[str, Any]] | None = None,
    phases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    total = len(frames)
    skipped = sum(1 for frame in frames if frame.skipped)
    usable = sum(1 for frame in frames if _pose_selection_score(frame) >= 0.28)
    raw_counts = [len(frame.raw_keypoints or frame.keypoints) for frame in frames]
    filtered_counts = [len(frame.keypoints) for frame in frames]
    avg_conf = sum(frame.confidence for frame in frames) / total if total else 0.0
    backend_counts: dict[str, int] = {}
    threshold_removed = 0
    roi_selected_full = 0
    roi_selected_roi = 0
    roi_invalid_full_fallback = 0
    roi_boundary_clip = 0
    roi_source_counts: dict[str, int] = {}
    roi_confidence_by_source: dict[str, list[float]] = {}
    roi_candidate_scores: list[float] = []
    roi_temporal_jumps: list[float] = []
    roi_temporal_high_jumps = 0
    body_removed_many = 0
    for frame in frames:
        backend_counts[frame.backend or actual_backend] = backend_counts.get(frame.backend or actual_backend, 0) + 1
        info = frame.debug_info or {}
        threshold_removed += int(info.get("removed_by_threshold", 0) or 0)
        if "rtmpose_body_joints_low_confidence" in frame.quality_flags:
            body_removed_many += 1
        if "roi_full_frame_fallback" in frame.quality_flags:
            roi_invalid_full_fallback += 1
        if any(str(flag).startswith("roi_boundary_clip_ratio_") for flag in frame.quality_flags):
            roi_boundary_clip += 1
        selected = ((info.get("roi_decision") or {}) if isinstance(info.get("roi_decision"), dict) else {}).get("selected")
        selected_source = str(((info.get("roi_decision") or {}) if isinstance(info.get("roi_decision"), dict) else {}).get("selected_source") or "unknown")
        if selected == "full_frame":
            roi_selected_full += 1
        elif selected == "roi":
            roi_selected_roi += 1
        if frame.roi:
            source = str(frame.roi.get("source") or selected_source)
            roi_source_counts[source] = roi_source_counts.get(source, 0) + 1
            roi_confidence_by_source.setdefault(source, []).append(float(frame.confidence or 0.0))
            roi_candidate_scores.append(float(frame.roi.get("candidate_score") or 0.0))
            metrics = frame.roi.get("metrics") if isinstance(frame.roi.get("metrics"), dict) else {}
            if isinstance(metrics, dict) and metrics.get("temporal_jump_ratio") is not None:
                jump = float(metrics.get("temporal_jump_ratio") or 0.0)
                roi_temporal_jumps.append(jump)
                if jump > 2.1:
                    roi_temporal_high_jumps += 1
            if float(frame.roi.get("boundary_touch_ratio") or 0.0) > 0.2:
                roi_boundary_clip += 1
    backend_diag = dict(getattr(estimator, "diagnostics", {}) or {})
    warnings = list(backend_diag.get("warnings") or [])
    if total and usable / total < 0.35:
        warnings.append("Pose usable-frame ratio is low; model, ROI, or threshold diagnostics should be reviewed before trusting metrics.")
    if total and body_removed_many / total > 0.4:
        warnings.append("Low-confidence filtering removed most body joints in many sampled frames.")
    if total and roi_invalid_full_fallback / total > 0.4:
        warnings.append("ROI localization failed on many sampled frames; RTMPose used full-frame fallback for those frames.")
    if total and roi_boundary_clip / total > 0.4:
        warnings.append("ROI localization repeatedly touched frame boundaries, which usually means the crop is following pool edges/reflections instead of the swimmer.")
    if total and _avg(raw_counts) >= 16 and avg_conf < 0.2:
        warnings.append("RTMPose returned a full 17-joint tensor but with low confidence; this points to bbox/ROI or crop-scale mismatch more than keypoint-order mismatch.")
    if backend_errors:
        warnings.append("Pose backend fallback/errors occurred; see fallback_reason.")
    probable_cause = _probable_pose_cause(
        total=total,
        avg_raw_count=_avg(raw_counts),
        avg_confidence=avg_conf,
        roi_invalid_full_fallback=roi_invalid_full_fallback,
        roi_boundary_clip=roi_boundary_clip,
        backend_errors=backend_errors,
        actual_backend=actual_backend,
    )
    tracking_diag = _tracking_diagnostics(tracking_exports or {}, phases or [])
    if float(tracking_diag.get("average_tracking_quality", 0) or 0) < 0.35:
        warnings.append("Temporal tracking quality is low; use the joint reliability diagnostics before trusting stroke metrics.")
    if len(tracking_diag.get("unstable_joints", []) or []) >= 4:
        warnings.append("Multiple joints were unstable across time, so strong coaching advice is gated.")
    return {
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "probable_cause": probable_cause,
        "fallback_reason": "; ".join(backend_errors) if backend_errors else None,
        "backend_counts": backend_counts,
        "frames_sampled": total,
        "frames_skipped": skipped,
        "usable_pose_frames": usable,
        "average_pose_confidence": round(avg_conf, 4),
        "average_raw_joint_count": round(sum(raw_counts) / total, 3) if total else 0.0,
        "average_filtered_joint_count": round(sum(filtered_counts) / total, 3) if total else 0.0,
        "threshold_removed_joint_count": threshold_removed,
        "temporal_visible_confidence": settings.swim_temporal_visible_confidence,
        "roi": {
            "enabled": roi_config.enabled,
            "compare_full_frame": bool(settings.swim_roi_compare_full_frame),
            "fallback_full_frame": bool(settings.swim_roi_fallback_full_frame),
            "selected_full_frame_frames": roi_selected_full,
            "selected_roi_frames": roi_selected_roi,
            "invalid_full_frame_fallback_frames": roi_invalid_full_fallback,
            "full_frame_fallback_percentage": round((roi_selected_full / total) * 100.0, 2) if total else 0.0,
            "boundary_clip_frames": roi_boundary_clip,
            "source_counts": roi_source_counts,
            "average_selected_candidate_score": round(_avg(roi_candidate_scores), 4) if roi_candidate_scores else 0.0,
            "average_temporal_jump_ratio": round(_avg(roi_temporal_jumps), 4) if roi_temporal_jumps else 0.0,
            "high_temporal_jump_frames": roi_temporal_high_jumps,
            "rtmpose_confidence_by_source": {source: round(_avg(values), 4) for source, values in roi_confidence_by_source.items()},
            "config": roi_config.__dict__,
        },
        "backend": backend_diag,
        "tracking": tracking_diag,
        "warnings": _dedupe_strings(warnings),
    }


def _tracking_diagnostics(tracking_exports: dict[str, dict[str, Any]], phases: list[dict[str, Any]]) -> dict[str, Any]:
    if not tracking_exports:
        return {
            "average_tracking_quality": 0.0,
            "temporal_stability": 0.0,
            "motion_smoothness": 0.0,
            "smoothing_status": "unavailable",
            "reliable_joints": [],
            "estimated_joints": [],
            "unstable_joints": [],
            "hidden_joints": [],
            "phase_detection_confidence": round(_avg([float(item.get("confidence", 0) or 0) for item in phases]), 4) if phases else 0.0,
        }
    quality_values: list[float] = []
    stability_values: list[float] = []
    smoothness_values: list[float] = []
    categories: dict[str, set[str]] = {"reliable": set(), "estimated": set(), "unstable": set(), "hidden": set()}
    smoothing_methods: set[str] = set()
    smoothing_enabled = False
    smoothing_displacements: list[float] = []
    roi_timeline: list[dict[str, Any]] = []
    dropped = {"skipped_frames": 0, "no_pose_frames": 0, "low_tracking_quality_frames": 0, "temporal_recovery_frames": 0}
    for view, export in tracking_exports.items():
        for point in export.get("tracking_quality_timeline", []) or []:
            quality_values.append(float(point.get("quality", 0) or 0))
        reliability = export.get("joint_reliability", {})
        if isinstance(reliability, dict):
            for joint, item in reliability.items():
                if not isinstance(item, dict):
                    continue
                stability_values.append(float(item.get("temporal_stability", 0) or 0))
                smoothness_values.append(float(item.get("motion_smoothness", 0) or 0))
                category = str(item.get("category") or "hidden")
                categories.setdefault(category, set()).add(str(joint))
        smoothing = export.get("smoothing_diagnostics", {})
        if isinstance(smoothing, dict):
            smoothing_enabled = smoothing_enabled or bool(smoothing.get("enabled"))
            smoothing_methods.add(str(smoothing.get("method") or "unknown"))
            smoothing_displacements.append(float(smoothing.get("mean_raw_to_smoothed_px", 0) or 0))
        for entry in (export.get("roi_history", []) or [])[:80]:
            if isinstance(entry, dict):
                roi_timeline.append({"view": view, **entry})
        dropped_info = export.get("dropped_frame_diagnostics", {})
        if isinstance(dropped_info, dict):
            for key in dropped:
                dropped[key] += int(dropped_info.get(key, 0) or 0)
    return {
        "average_tracking_quality": round(_avg(quality_values), 4),
        "temporal_stability": round(_avg(stability_values), 4),
        "motion_smoothness": round(_avg(smoothness_values), 4),
        "smoothing_status": "enabled" if smoothing_enabled else "disabled",
        "smoothing_method": ", ".join(sorted(smoothing_methods)) if smoothing_methods else "unknown",
        "mean_smoothing_displacement_px": round(_avg(smoothing_displacements), 3),
        "reliable_joints": sorted(categories.get("reliable", set())),
        "estimated_joints": sorted(categories.get("estimated", set())),
        "unstable_joints": sorted(categories.get("unstable", set())),
        "hidden_joints": sorted(categories.get("hidden", set())),
        "phase_detection_confidence": round(_avg([float(item.get("confidence", 0) or 0) for item in phases]), 4) if phases else 0.0,
        "roi_source_timeline": roi_timeline[:120],
        "dropped_frame_diagnostics": dropped,
    }


def _raw_pose_debug_payload(frames: list[PoseFrameResult]) -> dict[str, Any]:
    return {
        "frames": [
            {
                "frame_index": frame.frame_index,
                "timestamp": round(float(frame.timestamp), 4),
                "backend": frame.backend,
                "quality_flags": frame.quality_flags,
                "debug_events": frame.debug_events,
                "raw_keypoints_by_index": (frame.debug_info or {}).get("raw_keypoints_by_index", []),
                "raw_joint_count": (frame.debug_info or {}).get("raw_joint_count", len(frame.raw_keypoints or [])),
                "filtered_joint_count": (frame.debug_info or {}).get("filtered_joint_count", len(frame.keypoints)),
                "removed_by_threshold": (frame.debug_info or {}).get("removed_by_threshold", 0),
                "roi_decision": (frame.debug_info or {}).get("roi_decision"),
                "roi": frame.roi,
            }
            for frame in frames
        ]
    }


def _roi_debug_manifest(frames: list[PoseFrameResult]) -> dict[str, Any]:
    return {
        "frames": [
            {
                "frame_index": frame.frame_index,
                "timestamp": round(float(frame.timestamp), 4),
                "roi": frame.roi,
                "decision": (frame.debug_info or {}).get("roi_decision"),
                "confidence": round(float(frame.confidence), 4),
                "quality_flags": frame.quality_flags,
            }
            for frame in frames
        ]
    }


def _avg(values: list[float] | list[int]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _probable_pose_cause(
    *,
    total: int,
    avg_raw_count: float,
    avg_confidence: float,
    roi_invalid_full_fallback: int,
    roi_boundary_clip: int,
    backend_errors: list[str],
    actual_backend: str,
) -> str:
    if backend_errors or actual_backend != "rtmpose":
        return "backend_fallback"
    if total and roi_boundary_clip / total > 0.4:
        return "roi_boundary_or_reflection_crop"
    if total and roi_invalid_full_fallback / total > 0.4:
        return "roi_localization_failed_full_frame_fallback"
    if avg_raw_count >= 16 and avg_confidence < 0.2:
        return "low_confidence_crop_or_scale_mismatch"
    if avg_raw_count < 8:
        return "missing_pose_keypoints"
    return "not_identified"


def _adapt_metrics_for_single_video(metrics: dict[str, Any]) -> dict[str, Any]:
    return _replace_report_text(
        metrics,
        [
            ("front/back view", "uploaded video"),
            ("front/back-view", "uploaded video"),
            ("front view", "uploaded video"),
            ("side-view", "uploaded video"),
            ("side view", "uploaded video"),
            ("both views", "the uploaded video"),
            ("both videos", "the uploaded video"),
        ],
    )


def _replace_report_text(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_report_text(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_report_text(item, replacements) for item in value]
    if not isinstance(value, str):
        return value
    text = value
    for old, new in replacements:
        text = text.replace(old, new)
        text = text.replace(old.capitalize(), new.capitalize())
    return text


def _max_dimension_for_quality(quality_mode: str, pose_device: str | None) -> int:
    cpu_mode = (pose_device or "cpu").strip().lower() == "cpu"
    if quality_mode == "high_accuracy":
        return 1280 if cpu_mode else 1600
    return 960 if cpu_mode else 1280


def _upload_url(path: str | Path | None) -> str | None:
    if path is None:
        return None
    settings = get_settings()
    try:
        relative = Path(path).resolve().relative_to(Path(settings.upload_dir).resolve())
    except ValueError:
        relative = Path(path)
    return "/uploads/" + str(relative).replace("\\", "/")


def _roi_config_from_settings(settings: Any) -> ROIConfig:
    return ROIConfig(
        enabled=bool(settings.swim_roi_enabled and not settings.swim_roi_disable),
        scale=float(settings.swim_roi_scale),
        min_size_px=int(settings.swim_roi_min_size_px),
        max_boundary_clip_ratio=float(settings.swim_roi_max_boundary_clip_ratio),
        max_candidates=int(settings.swim_roi_max_candidates),
    )


def _temporal_config_from_settings(settings: Any) -> TemporalTrackingConfig:
    return TemporalTrackingConfig(
        visible_confidence=float(settings.swim_temporal_visible_confidence),
        low_confidence=float(settings.swim_temporal_low_confidence),
        max_gap_frames=int(settings.swim_temporal_max_gap_frames),
        smoothing_min_confidence=float(settings.swim_temporal_smoothing_min_confidence),
        smoothing_min_cutoff=float(settings.swim_temporal_smoothing_min_cutoff),
        smoothing_beta=float(settings.swim_temporal_smoothing_beta),
        smoothing_d_cutoff=float(settings.swim_temporal_smoothing_d_cutoff),
        smoothing_ema_alpha_low=float(settings.swim_temporal_smoothing_ema_alpha_low),
        smoothing_ema_alpha_high=float(settings.swim_temporal_smoothing_ema_alpha_high),
        enable_biomechanical_constraints=bool(settings.swim_temporal_biomechanical_constraints),
    )


def _temporal_config_summary(config: TemporalTrackingConfig) -> dict[str, Any]:
    return {
        "visible_confidence": config.visible_confidence,
        "low_confidence": config.low_confidence,
        "max_gap_frames": config.max_gap_frames,
        "enable_smoothing": config.enable_smoothing,
        "smoothing_min_confidence": config.smoothing_min_confidence,
        "smoothing_min_cutoff": config.smoothing_min_cutoff,
        "smoothing_beta": config.smoothing_beta,
        "smoothing_d_cutoff": config.smoothing_d_cutoff,
        "smoothing_ema_alpha_low": config.smoothing_ema_alpha_low,
        "smoothing_ema_alpha_high": config.smoothing_ema_alpha_high,
        "enable_biomechanical_constraints": config.enable_biomechanical_constraints,
        "max_body_scale_change_ratio": config.max_body_scale_change_ratio,
        "swap_margin_px": config.swap_margin_px,
        "min_swap_improvement_px": config.min_swap_improvement_px,
        "core_threshold": config.core_threshold.__dict__,
        "mid_threshold": config.mid_threshold.__dict__,
        "distal_threshold": config.distal_threshold.__dict__,
    }


def _figure_with_upload_url(figure: dict[str, Any]) -> dict[str, Any]:
    output = dict(figure)
    output["file_path"] = _upload_url(output.get("file_path")) or str(output.get("file_path") or "")
    return output


def _dedupe_strings(values: list[Any]) -> list[str]:
    output: list[str] = []
    for value in values:
        if not value:
            continue
        text = str(value)
        if text not in output:
            output.append(text)
    return output
