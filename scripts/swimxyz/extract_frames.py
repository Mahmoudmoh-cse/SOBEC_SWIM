from __future__ import annotations

import argparse
import json
from pathlib import Path

from swimxyz_common import ensure_dir, find_video_files, infer_stroke_view_from_path, safe_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a controlled subset of SwimXYZ frames for AquaIQ preprocessing.")
    parser.add_argument("--videos-root", default="data/raw/swimxyz/videos", help="Root containing SwimXYZ videos.")
    parser.add_argument("--output-root", default="data/interim/frames", help="Where extracted frames and metadata are saved.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Deprecated compatibility option. Extraction now writes every source frame; subsample later in convert_to_coco.py.",
    )
    parser.add_argument("--max-videos", type=int, default=10, help="Safety cap on processed videos. Use a small number locally.")
    parser.add_argument("--stroke", default="freestyle", help="Stroke filter inferred from path/metadata.")
    parser.add_argument("--view", default="side", help="View filter inferred from path/metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_videos is None or args.max_videos <= 0:
        raise SystemExit("--max-videos must be a positive safety cap. Refusing to process an unbounded dataset.")

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Install opencv-python-headless.") from exc

    videos_root = Path(args.videos_root)
    output_root = ensure_dir(args.output_root)
    videos = _filter_videos(find_video_files(videos_root), videos_root=videos_root, stroke=args.stroke, view=args.view)
    selected = videos[: args.max_videos]
    if not selected:
        raise SystemExit(f"No videos matched stroke={args.stroke!r}, view={args.view!r} under {videos_root}.")

    manifest_frames = []
    manifest_videos = []
    if args.fps:
        print("[extract] --fps is ignored during extraction; use convert_to_coco.py --subsample-stride after exact matching.")
    print(f"Found {len(videos)} matching videos. Extracting all frames from {len(selected)} videos.")
    for video_number, video_path in enumerate(selected, start=1):
        relative_video = video_path.relative_to(videos_root) if video_path.is_relative_to(videos_root) else video_path.name
        target_dir = ensure_dir(output_root / safe_stem(relative_video))
        video_metadata = _extract_video(cv2, video_path, target_dir, relative_video=str(relative_video))
        manifest_videos.append(video_metadata)
        manifest_frames.extend(video_metadata["frames"])
        print(
            f"[{video_number}/{len(selected)}] {relative_video}: "
            f"{len(video_metadata['frames'])}/{video_metadata['source_frame_count']} frames"
        )

    manifest = {
        "videos_root": str(videos_root),
        "output_root": str(output_root),
        "extraction_mode": "all_frames_exact_indices",
        "deprecated_requested_fps": args.fps,
        "stroke": args.stroke,
        "view": args.view,
        "video_count": len(manifest_videos),
        "frame_count": len(manifest_frames),
        "videos": manifest_videos,
        "frames": manifest_frames,
    }
    (output_root / "frame_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved frame manifest: {output_root / 'frame_manifest.json'}")


def _filter_videos(videos: list[Path], *, videos_root: Path, stroke: str, view: str) -> list[Path]:
    output = []
    for path in videos:
        relative = path.relative_to(videos_root) if path.is_relative_to(videos_root) else path
        inferred_stroke, inferred_view = infer_stroke_view_from_path(relative)
        stroke_ok = not stroke or inferred_stroke is None or inferred_stroke == stroke.lower()
        view_ok = not view or inferred_view is None or inferred_view == view.lower()
        if stroke_ok and view_ok:
            output.append(path)
    return output


def _extract_video(cv2, video_path: Path, target_dir: Path, *, relative_video: str) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = []
    frame_index = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            file_name = f"frame_{frame_index:06d}.jpg"
            image_path = target_dir / file_name
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Could not write frame: {image_path}")
            frames.append(
                {
                    "source_video": str(video_path),
                    "relative_video": relative_video.replace("\\", "/"),
                    "video_key": safe_stem(relative_video),
                    "frame_index": frame_index,
                    "original_frame_index": frame_index,
                    "annotation_frame_index": frame_index,
                    "saved_index": frame_index,
                    "frame_file_name": file_name,
                    "timestamp_sec": round(frame_index / source_fps, 6),
                    "image_path": str(image_path),
                    "relative_image_path": str(image_path.relative_to(target_dir.parent)).replace("\\", "/"),
                    "width": width,
                    "height": height,
                }
            )
            frame_index += 1
    finally:
        cap.release()

    metadata = {
        "source_video": str(video_path),
        "relative_video": relative_video.replace("\\", "/"),
        "video_key": safe_stem(relative_video),
        "source_fps": source_fps,
        "extraction_mode": "all_frames_exact_indices",
        "sample_every": 1,
        "source_frame_count": frame_count,
        "decoded_frame_count": len(frames),
        "resolution": {"width": width, "height": height},
        "frames": frames,
    }
    (target_dir / "_frame_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    main()
