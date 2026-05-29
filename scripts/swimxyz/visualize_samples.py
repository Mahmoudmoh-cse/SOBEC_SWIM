"""Draw COCO keypoints from processed SwimXYZ samples for parser validation."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from swimxyz_common import COCO_KEYPOINT_NAMES, COCO_SKELETON, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", default="data/processed/swimxyz_coco/annotations/train.json")
    parser.add_argument("--images-root", default="data/processed/swimxyz_coco/images/train")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-root", default="outputs/debug_visualizations")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_coco(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation.get("image_id", -1))].append(annotation)
    return list(payload.get("images", [])), annotations_by_image


def _draw_sample(
    image: np.ndarray,
    annotation: dict[str, Any],
    image_info: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    height, width = image.shape[:2]
    keypoints = np.asarray(annotation.get("keypoints", []), dtype=np.float32).reshape(-1, 3)
    if keypoints.shape[0] < len(COCO_KEYPOINT_NAMES):
        warnings.append(f"image_id={image_info.get('id')} has only {keypoints.shape[0]} keypoints")

    bbox = annotation.get("bbox", [])
    if len(bbox) == 4:
        x, y, w, h = [int(round(float(value))) for value in bbox]
        cv2.rectangle(image, (x, y), (x + w, y + h), (30, 180, 255), 2)

    for left_idx, right_idx in COCO_SKELETON:
        left = left_idx - 1
        right = right_idx - 1
        if left >= keypoints.shape[0] or right >= keypoints.shape[0]:
            continue
        p1 = keypoints[left]
        p2 = keypoints[right]
        if p1[2] > 0 and p2[2] > 0:
            cv2.line(
                image,
                (int(round(p1[0])), int(round(p1[1]))),
                (int(round(p2[0])), int(round(p2[1]))),
                (80, 220, 120),
                2,
                cv2.LINE_AA,
            )

    for idx, point in enumerate(keypoints):
        if point[2] <= 0:
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or y < 0 or x >= width or y >= height:
            name = COCO_KEYPOINT_NAMES[idx] if idx < len(COCO_KEYPOINT_NAMES) else f"joint_{idx}"
            warnings.append(f"{image_info.get('file_name')}: {name} outside image bounds ({x}, {y})")
            color = (0, 0, 255)
        else:
            color = (255, 80, 80) if idx % 2 else (255, 220, 60)
        cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 6, (20, 20, 20), 1, cv2.LINE_AA)

    label = f"id={image_info.get('id')} visible={annotation.get('num_keypoints', 0)}"
    cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return warnings


def main() -> None:
    args = parse_args()
    coco_json = Path(args.coco_json)
    images_root = Path(args.images_root)
    output_root = ensure_dir(args.output_root)

    if not coco_json.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    images, annotations_by_image = _load_coco(coco_json)
    candidates = [image for image in images if int(image.get("id", -1)) in annotations_by_image]
    if not candidates:
        raise ValueError(f"No annotated images found in {coco_json}")

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, min(args.num_samples, len(candidates)))
    all_warnings: list[str] = []

    for rank, image_info in enumerate(selected, start=1):
        image_path = images_root / str(image_info.get("file_name"))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            all_warnings.append(f"Could not read image: {image_path}")
            continue
        annotations = annotations_by_image[int(image_info["id"])]
        annotation = max(annotations, key=lambda item: float(item.get("area", 0.0)))
        all_warnings.extend(_draw_sample(image, annotation, image_info))
        output_path = output_root / f"swimxyz_sample_{rank:03d}_{Path(str(image_info.get('file_name'))).stem}.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"[visualize] wrote {output_path}")

    if all_warnings:
        warning_path = output_root / "swimxyz_visualization_warnings.txt"
        warning_path.write_text("\n".join(all_warnings) + "\n", encoding="utf-8")
        print(f"[visualize] warnings saved to {warning_path}")
        for warning in all_warnings[:20]:
            print(f"[warning] {warning}")

    print(f"[visualize] completed {len(selected)} sample visualizations")


if __name__ == "__main__":
    main()

