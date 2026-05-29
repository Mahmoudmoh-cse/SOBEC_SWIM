from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from swimxyz_common import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    bbox_from_coco_keypoints,
    ensure_dir,
    extract_pose_records,
    find_annotation_files,
    load_annotation_file,
    load_frame_manifest,
    map_to_coco_keypoints,
    normalized_key,
    safe_stem,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SwimXYZ 2D joints into a COCO-style swimming pose dataset.")
    parser.add_argument("--data-root", default="data/raw/swimxyz", help="SwimXYZ raw root containing annotations.")
    parser.add_argument("--frames-root", default="data/interim/frames", help="Root created by extract_frames.py.")
    parser.add_argument("--output-root", default="data/processed/swimxyz_coco", help="COCO dataset output root.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic train/val split seed.")
    parser.add_argument(
        "--subsample-stride",
        type=int,
        default=1,
        help="Keep every Nth matched frame after exact annotation matching. Use 1 to keep all matched frames.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap applied after exact matching/subsampling.")
    parser.add_argument("--debug-log", default="outputs/reports/coco_matching_debug.jsonl", help="JSONL frame matching debug log.")
    parser.add_argument("--validation-samples", type=int, default=10, help="Random matched samples to validate in the report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val-ratio must be in [0, 1).")
    if args.subsample_stride <= 0:
        raise SystemExit("--subsample-stride must be a positive integer.")
    if args.validation_samples < 0:
        raise SystemExit("--validation-samples must be non-negative.")

    data_root = Path(args.data_root)
    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)
    images_train = ensure_dir(output_root / "images" / "train")
    images_val = ensure_dir(output_root / "images" / "val")
    annotations_dir = ensure_dir(output_root / "annotations")

    frames = load_frame_manifest(frames_root)
    if not frames:
        raise SystemExit(f"No extracted frame metadata found under {frames_root}. Run extract_frames.py first.")

    frame_keys = _frame_match_keys(frames)
    records = _load_records(data_root, frame_keys)
    if not records:
        raise SystemExit(f"No 2D pose records were parsed from annotations under {data_root}. Run inspect_swimxyz.py and adjust parser mapping.")

    record_index = _index_records(records)
    matched = []
    unmatched_frames = []
    match_events: list[dict[str, Any]] = []
    for frame in frames:
        record, match_event = _match_record(frame, record_index)
        match_events.append(match_event)
        if record is None:
            unmatched_frames.append(frame)
            continue
        matched.append((frame, record))

    if not matched:
        raise SystemExit("No extracted frames matched parsed annotations. Check video names/frame indices and inspect report.")

    warnings: list[str] = []
    validation_lines, validation_warnings = _validate_matches(matched, seed=args.seed, sample_size=args.validation_samples)
    warnings.extend(validation_warnings)

    matched_before_subsample = len(matched)
    matched = _subsample_matched(matched, stride=args.subsample_stride)
    if args.max_frames:
        matched = matched[: args.max_frames]

    shuffled = list(matched)
    random.Random(args.seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * args.val_ratio))
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]

    train = _build_split(train_items, images_train, split="train", warnings=warnings)
    val = _build_split(val_items, images_val, split="val", warnings=warnings)

    (annotations_dir / "train.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
    (annotations_dir / "val.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    debug_log_path = _write_matching_debug(args.debug_log, match_events)

    report_path = Path("outputs/reports/coco_conversion_report.md")
    write_markdown(
        report_path,
        "SwimXYZ COCO Conversion Report",
        [
            ("Inputs", f"- Data root: `{data_root}`\n- Frames root: `{frames_root}`\n- Output root: `{output_root}`"),
            (
                "Counts",
                "\n".join(
                    [
                        f"- Parsed pose records: {len(records)}",
                        f"- Extracted frames: {len(frames)}",
                        f"- Matched frames before subsampling: {matched_before_subsample}",
                        f"- Subsample stride: {args.subsample_stride}",
                        f"- Matched frames after subsampling/cap: {len(matched)}",
                        f"- Unmatched frames: {len(unmatched_frames)}",
                        f"- Train images: {len(train['images'])}",
                        f"- Val images: {len(val['images'])}",
                    ]
                ),
            ),
            ("Frame Matching Validation", "\n".join(validation_lines) or "Validation disabled."),
            ("Warnings", "\n".join(f"- {warning}" for warning in warnings[:80]) or "No parser warnings."),
            (
                "Output",
                "\n".join(
                    [
                        f"- Train JSON: `{annotations_dir / 'train.json'}`",
                        f"- Val JSON: `{annotations_dir / 'val.json'}`",
                        f"- Images: `{output_root / 'images'}`",
                        f"- Matching debug log: `{debug_log_path}`",
                    ]
                ),
            ),
        ],
    )

    print(f"Parsed records: {len(records)}")
    print(f"Matched frames before subsampling: {matched_before_subsample} / {len(frames)}")
    print(f"Matched frames after subsampling/cap: {len(matched)}")
    if unmatched_frames:
        print(f"Unmatched frames: {len(unmatched_frames)}")
    for warning in validation_warnings[:10]:
        print(f"[validation warning] {warning}")
    print(f"Train images: {len(train['images'])}")
    print(f"Val images: {len(val['images'])}")
    print(f"Saved COCO annotations: {annotations_dir}")
    print(f"Saved matching debug log: {debug_log_path}")
    print(f"Saved report: {report_path}")


def _load_records(data_root: Path, frame_keys: set[str] | None = None):
    output = []
    for annotation_path in find_annotation_files(data_root):
        annotation_key = normalized_key(annotation_path)
        if frame_keys and not any(key and (key in annotation_key or annotation_key in key) for key in frame_keys):
            continue
        try:
            payload = load_annotation_file(annotation_path)
            output.extend(extract_pose_records(payload, annotation_path))
        except Exception as exc:
            print(f"Warning: skipped annotation {annotation_path}: {exc}")
    return output


def _index_records(records) -> dict[str, dict[int, Any]]:
    index: dict[str, dict[int, Any]] = defaultdict(dict)
    for record in records:
        keys = {record.video_key, normalized_key(record.video_key), normalized_key(record.annotation_file)}
        for key in keys:
            if key:
                index[key][int(record.frame_index)] = record
    return index


def _match_record(frame: dict[str, Any], index: dict[str, dict[int, Any]]):
    frame_index = int(frame.get("original_frame_index", frame.get("frame_index", frame.get("saved_index", 0))))
    keys = sorted(_frame_keys(frame, include_leaf=False))
    event: dict[str, Any] = {
        "status": "unmatched",
        "relative_video": frame.get("relative_video"),
        "image_path": frame.get("image_path"),
        "frame_index": frame_index,
        "match_keys": keys,
        "candidate_key_count": sum(1 for key in keys if key in index),
    }
    for key in keys:
        records = index.get(key)
        if records and frame_index in records:
            record = records[frame_index]
            event.update(
                {
                    "status": "matched",
                    "matched_key": key,
                    "annotation_frame_index": int(record.frame_index),
                    "annotation_file": record.annotation_file,
                    "record_video_key": record.video_key,
                    "frame_index_delta": frame_index - int(record.frame_index),
                }
            )
            return record, event
        if records:
            event.setdefault("nearest_available_frames", {})[key] = _nearest_frames(records.keys(), frame_index)
    return None, event


def _frame_match_keys(frames: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for frame in frames:
        keys.update(_frame_keys(frame, include_leaf=False))
    return {key for key in keys if key}


def _frame_keys(frame: dict[str, Any], *, include_leaf: bool = True) -> set[str]:
    relative_video = str(frame.get("relative_video", ""))
    keys = {
        normalized_key(frame.get("video_key")),
        normalized_key(relative_video),
    }
    if include_leaf:
        keys.add(normalized_key(Path(relative_video).stem))
    return keys


def _build_split(items: list[tuple[dict[str, Any], Any]], images_dir: Path, *, split: str, warnings: list[str]) -> dict[str, Any]:
    images = []
    annotations = []
    for item_id, (frame, record) in enumerate(items, start=1):
        source_path = Path(frame["image_path"])
        if not source_path.exists():
            warnings.append(f"missing extracted frame: {source_path}")
            continue
        width = int(frame.get("width") or 0)
        height = int(frame.get("height") or 0)
        keypoints, num_keypoints, mapping_warnings = map_to_coco_keypoints(record.keypoints2d, record.joint_names)
        warnings.extend(f"{record.annotation_file}: {warning}" for warning in mapping_warnings)
        keypoints, transformed = _apply_coordinate_transform(keypoints, record, image_height=height)
        keypoints, out_of_bounds = _mark_out_of_bounds_invisible(keypoints, width=width, height=height)
        if out_of_bounds:
            warnings.append(f"{source_path}: {out_of_bounds} out-of-frame keypoints marked invisible")
        num_keypoints = _count_visible_keypoints(keypoints)
        bbox = None if transformed else _normalize_bbox(record.bbox)
        bbox = bbox or bbox_from_coco_keypoints(keypoints)
        if bbox is None:
            warnings.append(f"no visible keypoints for {source_path}; annotation skipped")
            continue
        original_frame_index = int(frame.get("original_frame_index", frame.get("frame_index", item_id)))
        annotation_frame_index = int(record.frame_index)
        if original_frame_index != annotation_frame_index:
            warnings.append(
                f"frame index drift for {source_path}: image={original_frame_index}, annotation={annotation_frame_index}"
            )
        file_name = f"{safe_stem(frame.get('relative_video', frame.get('video_key', 'video')))}__frame_{annotation_frame_index:06d}.jpg"
        target_path = images_dir / file_name
        shutil.copy2(source_path, target_path)
        area = max(1.0, float(bbox[2]) * float(bbox[3]))
        image_id = item_id
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "frame_index": original_frame_index,
                "original_frame_index": original_frame_index,
                "annotation_frame_index": annotation_frame_index,
                "source_video": frame.get("relative_video"),
                "timestamp_sec": frame.get("timestamp_sec"),
                "split": split,
            }
        )
        annotations.append(
            {
                "id": item_id,
                "image_id": image_id,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": int(num_keypoints),
                "bbox": bbox,
                "area": round(area, 3),
                "iscrowd": 0,
                "metadata": {
                    "annotation_file": record.annotation_file,
                    "video_key": record.video_key,
                    "source_frame_index": record.frame_index,
                    "original_frame_index": original_frame_index,
                    "frame_index_delta": original_frame_index - annotation_frame_index,
                },
            }
        )

    return {
        "info": {
            "description": "AquaIQ SwimXYZ COCO keypoint dataset",
            "version": "phase1-dataset-engineering",
            "split": split,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "swimmer",
                "supercategory": "person",
                "keypoints": COCO_KEYPOINT_NAMES,
                "skeleton": COCO_SKELETON,
            }
        ],
    }


def _normalize_bbox(value: Any) -> list[float] | None:
    if not value or len(value) < 4:
        return None
    x1, y1, third, fourth = [float(v) for v in value[:4]]
    if third > x1 and fourth > y1:
        return [round(x1, 3), round(y1, 3), round(third - x1, 3), round(fourth - y1, 3)]
    return [round(x1, 3), round(y1, 3), round(max(1.0, third), 3), round(max(1.0, fourth), 3)]


def _apply_coordinate_transform(keypoints: list[float], record: Any, *, image_height: int) -> tuple[list[float], bool]:
    metadata = getattr(record, "metadata", {}) or {}
    if metadata.get("coordinate_origin") != "bottom_left" or image_height <= 0:
        return keypoints, False
    transformed = list(keypoints)
    for index in range(0, len(transformed), 3):
        if transformed[index + 2] <= 0:
            continue
        transformed[index + 1] = round(float(image_height) - float(transformed[index + 1]), 3)
    return transformed, True


def _mark_out_of_bounds_invisible(keypoints: list[float], *, width: int, height: int) -> tuple[list[float], int]:
    if width <= 0 or height <= 0:
        return keypoints, 0
    cleaned = list(keypoints)
    out_of_bounds = 0
    for index in range(0, len(cleaned), 3):
        if cleaned[index + 2] <= 0:
            continue
        x = float(cleaned[index])
        y = float(cleaned[index + 1])
        if x < 0 or y < 0 or x >= width or y >= height:
            cleaned[index] = 0.0
            cleaned[index + 1] = 0.0
            cleaned[index + 2] = 0.0
            out_of_bounds += 1
    return cleaned, out_of_bounds


def _count_visible_keypoints(keypoints: list[float]) -> int:
    return sum(1 for index in range(2, len(keypoints), 3) if keypoints[index] > 0)


def _subsample_matched(items: list[tuple[dict[str, Any], Any]], *, stride: int) -> list[tuple[dict[str, Any], Any]]:
    if stride <= 1:
        return items
    return [item for index, item in enumerate(items) if index % stride == 0]


def _validate_matches(
    items: list[tuple[dict[str, Any], Any]],
    *,
    seed: int,
    sample_size: int,
) -> tuple[list[str], list[str]]:
    if sample_size <= 0 or not items:
        return [], []
    rng = random.Random(seed)
    selected = rng.sample(items, min(sample_size, len(items)))
    lines: list[str] = []
    warnings: list[str] = []
    for frame, record in selected:
        image_frame_index = int(frame.get("original_frame_index", frame.get("frame_index", -1)))
        annotation_frame_index = int(record.frame_index)
        file_frame_index = _frame_index_from_file_name(str(frame.get("frame_file_name") or Path(str(frame.get("image_path", ""))).name))
        frame_video_key = normalized_key(frame.get("relative_video"))
        annotation_video_key = normalized_key(record.video_key)
        drift = image_frame_index - annotation_frame_index
        ok = drift == 0 and file_frame_index in {None, image_frame_index} and frame_video_key == annotation_video_key
        status = "OK" if ok else "WARN"
        lines.append(
            " - "
            f"{status}: video=`{frame.get('relative_video')}`, "
            f"image_frame={image_frame_index}, annotation_frame={annotation_frame_index}, "
            f"file_frame={file_frame_index}, drift={drift}"
        )
        if drift != 0:
            warnings.append(
                f"frame drift detected for {frame.get('relative_video')}: image={image_frame_index}, annotation={annotation_frame_index}"
            )
        if file_frame_index is not None and file_frame_index != image_frame_index:
            warnings.append(
                f"image filename frame mismatch for {frame.get('image_path')}: filename={file_frame_index}, metadata={image_frame_index}"
            )
        if frame_video_key != annotation_video_key:
            warnings.append(
                f"video key mismatch for frame {image_frame_index}: frame={frame_video_key}, annotation={annotation_video_key}"
            )
    return lines, warnings


def _frame_index_from_file_name(file_name: str) -> int | None:
    stem = Path(file_name).stem
    marker = "frame_"
    if marker not in stem:
        return None
    suffix = stem.rsplit(marker, 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _nearest_frames(values, target: int) -> list[int]:
    frames = sorted(int(value) for value in values)
    if not frames:
        return []
    return sorted(frames, key=lambda value: abs(value - target))[:5]


def _write_matching_debug(path: str | Path, events: list[dict[str, Any]]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, default=str) + "\n")
    return target


if __name__ == "__main__":
    main()
