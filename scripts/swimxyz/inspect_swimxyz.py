from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from swimxyz_common import (
    ANNOTATION_EXTENSIONS,
    VIDEO_EXTENSIONS,
    describe_schema,
    detect_joint_shapes,
    ensure_dir,
    extract_pose_records,
    find_annotation_files,
    find_video_files,
    infer_stroke_view_from_path,
    load_annotation_file,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SwimXYZ raw videos and annotations for AquaIQ dataset engineering.")
    parser.add_argument("--data-root", default="data/raw/swimxyz", help="Root containing SwimXYZ videos and annotations.")
    parser.add_argument("--report", default="outputs/reports/swimxyz_inspection.md", help="Markdown report path.")
    parser.add_argument("--max-files", type=int, default=8, help="Maximum annotation files to deeply inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    report_path = Path(args.report)
    ensure_dir(report_path.parent)

    annotation_files = find_annotation_files(data_root)
    video_files = find_video_files(data_root)
    stroke_counter: Counter[str] = Counter()
    view_counter: Counter[str] = Counter()
    for path in [*annotation_files, *video_files]:
        stroke, view = infer_stroke_view_from_path(path)
        if stroke:
            stroke_counter[stroke] += 1
        if view:
            view_counter[view] += 1

    inspected = []
    records_total = 0
    conversion_needed = True
    for annotation_path in annotation_files[: max(0, args.max_files)]:
        try:
            payload = load_annotation_file(annotation_path)
        except Exception as exc:
            inspected.append({"path": annotation_path, "error": str(exc)})
            continue
        records = extract_pose_records(payload, annotation_path)
        records_total += len(records)
        joints2d, joints3d = detect_joint_shapes(payload)
        conversion_needed = conversion_needed and not _looks_like_coco(payload)
        inspected.append(
            {
                "path": annotation_path,
                "schema": describe_schema(payload),
                "joints2d": joints2d[:6],
                "joints3d": joints3d[:6],
                "records": len(records),
                "sample_metadata": records[0].metadata if records else _sample_metadata(payload),
            }
        )

    sections = [
        (
            "Dataset Location",
            f"Data root: `{data_root}`\n\nExists: `{data_root.exists()}`",
        ),
        (
            "Structure",
            "\n".join(
                [
                    f"- Annotation files: {len(annotation_files)} ({', '.join(sorted(ANNOTATION_EXTENSIONS))})",
                    f"- Video files: {len(video_files)} ({', '.join(sorted(VIDEO_EXTENSIONS))})",
                    f"- Example annotation files: {_format_paths(annotation_files[:10])}",
                    f"- Example video files: {_format_paths(video_files[:10])}",
                ]
            ),
        ),
        (
            "Inferred Strokes And Views",
            "\n".join(
                [
                    f"- Strokes: `{dict(stroke_counter)}`",
                    f"- Views: `{dict(view_counter)}`",
                    "- These are inferred from file names and metadata only; validate against SwimXYZ docs before training.",
                ]
            ),
        ),
        (
            "Sequences And Records",
            "\n".join(
                [
                    f"- Parsed pose records from inspected files: {records_total}",
                    f"- Estimated sequences/videos: {max(len(video_files), len(annotation_files))}",
                ]
            ),
        ),
        (
            "Annotation Schema Samples",
            "\n\n".join(_format_inspected(item) for item in inspected) or "No annotations were loaded.",
        ),
        (
            "COCO Conversion Needed",
            "Yes. Convert SwimXYZ 2D swimming joints into a COCO-style keypoint dataset for RTMPose/ViTPose fine-tuning."
            if conversion_needed
            else "Input appears to already contain COCO sections. Still run conversion to normalize paths and filtering.",
        ),
    ]
    write_markdown(report_path, "SwimXYZ Inspection Report", sections)

    print(f"SwimXYZ data root: {data_root}")
    print(f"Annotation files: {len(annotation_files)}")
    print(f"Video files: {len(video_files)}")
    print(f"Inferred strokes: {dict(stroke_counter)}")
    print(f"Inferred views: {dict(view_counter)}")
    print(f"Parsed records from inspected files: {records_total}")
    print(f"Saved report: {report_path}")


def _format_paths(paths: list[Path]) -> str:
    if not paths:
        return "none"
    return ", ".join(f"`{path}`" for path in paths)


def _format_inspected(item: dict[str, Any]) -> str:
    if "error" in item:
        return f"### `{item['path']}`\n\nError: `{item['error']}`"
    return "\n".join(
        [
            f"### `{item['path']}`",
            "",
            f"- Parsed records: {item['records']}",
            f"- 2D joint shapes: `{item['joints2d']}`",
            f"- 3D joint shapes: `{item['joints3d']}`",
            f"- Sample metadata: `{json.dumps(item['sample_metadata'], default=str)[:600]}`",
            "",
            "```text",
            item["schema"],
            "```",
        ]
    )


def _looks_like_coco(value: Any) -> bool:
    return isinstance(value, dict) and {"images", "annotations", "categories"}.issubset(value.keys())


def _sample_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): str(type(child).__name__) for key, child in list(value.items())[:12]}
    return {"root_type": type(value).__name__}


if __name__ == "__main__":
    main()
