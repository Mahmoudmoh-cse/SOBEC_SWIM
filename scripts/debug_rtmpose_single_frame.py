from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.swim_analysis.pose.base import KEYPOINT_NAMES, PoseFrameResult  # noqa: E402
from app.swim_analysis.pose.rtmpose_backend import RTMPoseBackend  # noqa: E402
from app.swim_analysis.preprocessing import standardize_frame  # noqa: E402
from app.swim_analysis.processor import _infer_best_roi_candidate, _write_roi_debug_frame  # noqa: E402
from app.swim_analysis.roi import ROIConfig, SwimmerLocalizer, stamp_pose_metadata  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Run one-frame RTMPose inference with raw/mapped debug outputs.")
    parser.add_argument("--image", required=True, help="Path to an image frame.")
    parser.add_argument("--checkpoint", required=True, help="Path to RTMPose checkpoint.")
    parser.add_argument("--config", required=True, help="Path to RTMPose config.")
    parser.add_argument("--device", default="cpu", help="Inference device, for example cpu or cuda.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Pose confidence threshold for filtered keypoints.")
    parser.add_argument("--disable-roi", action="store_true", help="Run RTMPose on the full frame.")
    parser.add_argument("--output-dir", default="debug_outputs/rtmpose_single_frame", help="Directory for JSON and visualization outputs.")
    args = parser.parse_args()

    import cv2  # type: ignore

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")
    image = standardize_frame(image, max_dimension=1280, enhance=True)

    backend = RTMPoseBackend(config_path=args.config, checkpoint_path=args.checkpoint, device=args.device, confidence_threshold=args.threshold)
    backend.load()
    pose = _infer_image(backend, image, disable_roi=args.disable_roi, output_dir=output_dir)

    raw_debug = (pose.debug_info or {}).get("raw_keypoints_by_index", [])
    (output_dir / "raw_keypoints.json").write_text(json.dumps(raw_debug, indent=2), encoding="utf-8")
    (output_dir / "mapped_keypoints.json").write_text(json.dumps(pose.to_dict(), indent=2), encoding="utf-8")
    summary = _summary(pose, backend.diagnostics)
    (output_dir / "confidence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cv2.imwrite(str(output_dir / "skeleton_visualization.jpg"), _draw_pose(image.copy(), pose, cv2))

    print(json.dumps(summary, indent=2))
    return 0


def _infer_image(backend: RTMPoseBackend, image: object, *, disable_roi: bool, output_dir: Path) -> PoseFrameResult:
    if disable_roi:
        return stamp_pose_metadata(backend.infer_frame(image, frame_index=0, timestamp=0.0, view_type="video", quality_flags=["debug_disable_roi"]))
    localizer = SwimmerLocalizer(ROIConfig(enabled=True))
    candidates = localizer.candidate_rois(image, frame_index=0, timestamp=0.0, view_type="video")
    if not candidates:
        return stamp_pose_metadata(backend.infer_frame(image, frame_index=0, timestamp=0.0, view_type="video", quality_flags=["debug_roi_no_candidates"]))
    pose, debug_payload = _infer_best_roi_candidate(
        backend,
        image,
        candidates,
        frame_index=0,
        timestamp=0.0,
        view_type="video",
        quality_flags=["debug_roi_assisted"],
        compare_full_frame=True,
        fallback_full_frame=True,
    )
    _write_roi_debug_frame(output_dir / "roi_debug_frames", image, frame_index=0, debug_payload=debug_payload)
    return pose


def _summary(pose: PoseFrameResult, backend_diagnostics: dict[str, object]) -> dict[str, object]:
    raw = pose.raw_keypoints or pose.keypoints
    filtered_names = {point.name for point in pose.keypoints}
    body = [point for point in pose.keypoints if point.name not in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}]
    return {
        "backend": pose.backend,
        "backend_diagnostics": backend_diagnostics,
        "raw_joint_count": len(raw),
        "filtered_joint_count": len(pose.keypoints),
        "body_joint_count": len(body),
        "average_confidence": round(float(pose.confidence), 4),
        "missing_after_filter": [name for name in KEYPOINT_NAMES if name not in filtered_names],
        "quality_flags": pose.quality_flags,
        "debug_events": pose.debug_events,
        "roi": pose.roi,
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
        cv2.putText(image, point.name, (int(point.x) + 5, int(point.y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return image


if __name__ == "__main__":
    raise SystemExit(main())
