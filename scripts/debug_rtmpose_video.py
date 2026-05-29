from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.swim_analysis.pose.base import KEYPOINT_NAMES, PoseFrameResult  # noqa: E402
from app.swim_analysis.pose.rtmpose_backend import RTMPoseBackend  # noqa: E402
from app.swim_analysis.preprocessing import iter_sample_video_frames  # noqa: E402
from app.swim_analysis.processor import ROITrackingState, _attach_roi_tracking_diagnostics, _infer_best_roi_candidate, _merge_roi_candidates, _write_roi_debug_frame, _write_roi_tracking_timeline  # noqa: E402
from app.swim_analysis.roi import ROIConfig, ROIResult, SwimmerLocalizer, stamp_pose_metadata  # noqa: E402
from app.swim_analysis.temporal_tracking import TemporalJointTracker, TemporalTrackingConfig  # noqa: E402


SKELETON = [
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RTMPose video sanity diagnostics.")
    parser.add_argument("--video", required=True, help="Path to video.")
    parser.add_argument("--config", default=None, help="Path to RTMPose config. Defaults to env RTMPOSE_CONFIG_PATH.")
    parser.add_argument("--checkpoint", default=None, help="Path to RTMPose checkpoint. Defaults to env RTMPOSE_CHECKPOINT_PATH.")
    parser.add_argument("--device", default="cpu", help="Inference device, for example cpu or cuda.")
    parser.add_argument("--target-fps", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--disable-roi", action="store_true")
    parser.add_argument("--compare-roi", action="store_true", help="Run ROI-remapped and full-frame inference on the same frames, then compare them.")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--visible-confidence", type=float, default=None, help="Visibility threshold for the exported tracking summary. Defaults to --threshold.")
    parser.add_argument("--output-dir", default="debug_outputs/rtmpose_video")
    args = parser.parse_args()

    import cv2  # type: ignore

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "annotated_frames"
    roi_debug_dir = output_dir / "roi_debug_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    roi_debug_dir.mkdir(parents=True, exist_ok=True)
    backend_kwargs = {"device": args.device, "confidence_threshold": args.threshold}
    if args.config:
        backend_kwargs["config_path"] = args.config
    if args.checkpoint:
        backend_kwargs["checkpoint_path"] = args.checkpoint
    backend = RTMPoseBackend(**backend_kwargs)
    backend.load()

    raw_results: list[PoseFrameResult] = []
    frame_rows = []
    localizer = SwimmerLocalizer(ROIConfig(enabled=not args.disable_roi))
    roi_tracker = ROITrackingState()
    roi_timeline = []
    for sample_index, (frame_index, timestamp, frame, quality_flags, view_type) in enumerate(
        iter_sample_video_frames(args.video, target_fps=args.target_fps, view_type="video", max_dimension=1280, enhance=True),
        start=1,
    ):
        if sample_index > args.max_frames:
            break
        pose, roi_debug = _infer_frame(
            backend,
            localizer,
            frame,
            frame_index,
            timestamp,
            quality_flags,
            roi_tracker=roi_tracker,
            disable_roi=args.disable_roi,
            compare_roi=args.compare_roi,
        )
        selected_roi = roi_debug.get("selected_roi") if roi_debug else None
        diagnosis = roi_tracker.diagnose(pose, selected_roi if isinstance(selected_roi, ROIResult) else None, roi_debug, reattempt=False)
        final_diag = roi_tracker.commit(pose, selected_roi if isinstance(selected_roi, ROIResult) else None, roi_debug, diagnosis=diagnosis)
        pose = _attach_roi_tracking_diagnostics(pose, final_diag)
        if isinstance(selected_roi, ROIResult) and final_diag["tracking_state"] in {"LOCKED", "SUSPECT"}:
            localizer.remember_roi(roi_tracker.roi_for_memory or selected_roi)
        elif final_diag["tracking_state"] in {"LOST", "REACQUIRE", "REVIEW_ONLY"}:
            localizer.reset_tracking(keep_image_history=True)
        raw_results.append(pose)
        roi_timeline.append(final_diag)
        row = _frame_summary(pose)
        if roi_debug:
            row["roi_comparison"] = roi_debug.get("comparison")
        frame_rows.append(row)
        if sample_index <= 12:
            cv2.imwrite(str(frames_dir / f"frame_{frame_index:06d}.jpg"), _draw_pose(frame.copy(), pose, cv2))
            _write_roi_debug_frame(roi_debug_dir, frame, frame_index=frame_index, debug_payload=roi_debug)

    visible_confidence = float(args.visible_confidence if args.visible_confidence is not None else args.threshold)
    tracker = TemporalJointTracker(TemporalTrackingConfig(visible_confidence=visible_confidence, low_confidence=args.threshold)).process(raw_results, view_type="video")
    output = {
        "backend": backend.diagnostics,
        "video": str(Path(args.video).resolve()),
        "target_fps": args.target_fps,
        "max_frames": args.max_frames,
        "disable_roi": args.disable_roi,
        "compare_roi": args.compare_roi,
        "threshold": args.threshold,
        "visible_confidence": visible_confidence,
        "frames": frame_rows,
        "joint_visibility_summary": tracker.visibility_percentage,
        "joint_reliability": tracker.joint_reliability,
        "smoothing_diagnostics": tracker.smoothing_diagnostics,
        "tracking_quality_estimate": _tracking_quality_summary(tracker.tracking_quality_timeline),
        "center_trajectory": tracker.trajectory_export.get("center_trajectory", []),
        "roi_history": tracker.trajectory_export.get("roi_history", []),
        "dropped_frame_diagnostics": tracker.trajectory_export.get("dropped_frame_diagnostics", {}),
        "roi_tracking_timeline": roi_timeline,
        "raw_landmarks": [frame.to_dict() for frame in raw_results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rtmpose_video_debug.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (output_dir / "raw_landmarks.json").write_text(json.dumps([frame.to_dict() for frame in raw_results], indent=2), encoding="utf-8")
    (output_dir / "trajectory_debug_export.json").write_text(json.dumps(tracker.trajectory_export, indent=2), encoding="utf-8")
    _write_roi_tracking_timeline(roi_timeline, output_dir)
    print(json.dumps({key: output[key] for key in ["backend", "joint_visibility_summary", "tracking_quality_estimate", "smoothing_diagnostics"]}, indent=2))
    return 0


def _infer_frame(
    backend: RTMPoseBackend,
    localizer: SwimmerLocalizer,
    frame: object,
    frame_index: int,
    timestamp: float,
    quality_flags: list[str],
    *,
    roi_tracker: ROITrackingState,
    disable_roi: bool,
    compare_roi: bool,
) -> tuple[PoseFrameResult, dict[str, object]]:
    if disable_roi:
        pose = stamp_pose_metadata(backend.infer_frame(frame, frame_index=frame_index, timestamp=timestamp, view_type="video", quality_flags=[*quality_flags, "debug_disable_roi"]))
        return pose, {"full_frame_pose": pose, "comparison": {"selected": "full_frame", "reason": "roi_disabled"}}
    candidates = localizer.candidate_rois(frame, frame_index=frame_index, timestamp=timestamp, view_type="video", quality_flags=quality_flags, allow_previous=roi_tracker.allow_previous_roi())
    recovery_candidates = roi_tracker.recovery_candidates(frame, localizer.config, frame_index=frame_index, timestamp=timestamp)
    if recovery_candidates:
        candidates = _merge_roi_candidates(candidates, recovery_candidates)
    if not candidates:
        pose = stamp_pose_metadata(backend.infer_frame(frame, frame_index=frame_index, timestamp=timestamp, view_type="video", quality_flags=[*quality_flags, "debug_roi_invalid", "roi_no_candidates"]))
        pose = replace(
            pose,
            debug_events=[*pose.debug_events, "ROI invalid, full-frame fallback: no candidates"],
            debug_info={**pose.debug_info, "roi_decision": {"selected": "full_frame", "reason": "roi_no_candidates", "roi_valid": False}},
        )
        return pose, {"full_frame_pose": pose, "comparison": {"selected": "full_frame", "reason": "roi_no_candidates", "roi_valid": False}}
    pose, debug_payload = _infer_best_roi_candidate(
        backend,
        frame,
        candidates,
        frame_index=frame_index,
        timestamp=timestamp,
        view_type="video",
        quality_flags=[*quality_flags, "debug_roi_assisted"],
        compare_full_frame=compare_roi,
        fallback_full_frame=True,
        identity_tracker=roi_tracker,
    )
    debug_payload["comparison"] = debug_payload.get("decision", {})
    return pose, debug_payload


def _frame_summary(pose: PoseFrameResult) -> dict[str, object]:
    raw = pose.raw_keypoints or pose.keypoints
    body = [point for point in pose.keypoints if point.name not in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}]
    return {
        "frame_index": pose.frame_index,
        "timestamp": round(float(pose.timestamp), 4),
        "backend": pose.backend,
        "raw_joint_count": len(raw),
        "filtered_joint_count": len(pose.keypoints),
        "body_joint_count": len(body),
        "average_confidence": round(float(pose.confidence), 4),
        "roi": pose.roi,
        "quality_flags": pose.quality_flags,
        "debug_events": pose.debug_events,
    }


def _tracking_quality_summary(timeline: list[dict[str, object]]) -> dict[str, object]:
    values = [float(point.get("quality", 0) or 0) for point in timeline]
    return {
        "frames": len(values),
        "average": round(sum(values) / len(values), 4) if values else 0.0,
        "min": round(min(values), 4) if values else 0.0,
        "max": round(max(values), 4) if values else 0.0,
    }


def _compare_candidates(roi_pose: PoseFrameResult, full_pose: PoseFrameResult) -> dict[str, object]:
    roi_summary = _candidate_summary(roi_pose)
    full_summary = _candidate_summary(full_pose)
    selected = "full_frame" if float(full_summary["score"]) > float(roi_summary["score"]) + 0.08 else "roi"
    return {
        "selected": selected,
        "roi_valid": True,
        "roi": roi_summary,
        "full_frame": full_summary,
    }


def _candidate_summary(pose: PoseFrameResult) -> dict[str, object]:
    raw = pose.raw_keypoints or pose.keypoints
    body = [point for point in pose.keypoints if point.name not in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}]
    avg_raw = sum(float(point.confidence) for point in raw) / max(1, len(raw))
    score = (len(body) / 12.0) * 0.48 + (len(pose.keypoints) / 17.0) * 0.28 + avg_raw * 0.24
    return {
        "raw_joint_count": len(raw),
        "filtered_joint_count": len(pose.keypoints),
        "body_joint_count": len(body),
        "average_confidence": round(float(pose.confidence), 4),
        "average_raw_confidence": round(avg_raw, 4),
        "score": round(max(0.0, min(1.0, score)), 4),
        "bbox": pose.bbox,
        "flags": pose.quality_flags[:8],
    }


def _draw_pose(image: object, pose: PoseFrameResult, cv2: object) -> object:
    points = {point.name: point for point in pose.keypoints}
    for start, end in SKELETON:
        first = points.get(start)
        second = points.get(end)
        if first and second:
            cv2.line(image, (int(first.x), int(first.y)), (int(second.x), int(second.y)), (0, 210, 255), 2)
    for point in pose.raw_keypoints or []:
        cv2.circle(image, (int(point.x), int(point.y)), 4, (80, 80, 255), 1)
    for point in pose.keypoints:
        cv2.circle(image, (int(point.x), int(point.y)), 5, (0, 255, 120), -1)
    return image


def _write_roi_debug_images(output_dir: Path, frame_index: int, frame: object, roi_debug: dict[str, object], cv2: object) -> None:
    if not roi_debug:
        return
    original = frame.copy()
    roi = roi_debug.get("roi")
    if isinstance(roi, dict):
        x = int(roi.get("x") or 0)
        y = int(roi.get("y") or 0)
        width = int(roi.get("width") or 0)
        height = int(roi.get("height") or 0)
        color = (0, 220, 0) if roi.get("valid") else (0, 0, 255)
        if width > 0 and height > 0:
            cv2.rectangle(original, (x, y), (x + width, y + height), color, 2)
        reason = str(roi.get("reason") or "")
        if reason:
            cv2.putText(original, reason[:60], (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.imwrite(str(output_dir / f"frame_{frame_index:06d}_original.jpg"), original)

    roi_crop = roi_debug.get("roi_crop")
    if roi_crop is not None:
        cv2.imwrite(str(output_dir / f"frame_{frame_index:06d}_roi_crop.jpg"), roi_crop)
    roi_pose = roi_debug.get("roi_pose")
    if isinstance(roi_pose, PoseFrameResult):
        cv2.imwrite(str(output_dir / f"frame_{frame_index:06d}_roi_remapped_skeleton.jpg"), _draw_pose(frame.copy(), roi_pose, cv2))
    full_pose = roi_debug.get("full_frame_pose")
    if isinstance(full_pose, PoseFrameResult):
        cv2.imwrite(str(output_dir / f"frame_{frame_index:06d}_full_frame_skeleton.jpg"), _draw_pose(frame.copy(), full_pose, cv2))


if __name__ == "__main__":
    raise SystemExit(main())
