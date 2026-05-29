"""CPU smoke test for the AquaIQ SwimXYZ COCO PyTorch dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = PROJECT_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.data_pipeline.swimxyz_dataset import SwimXYZCocoDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", default="data/processed/swimxyz_coco/annotations/train.json")
    parser.add_argument("--images-root", default="data/processed/swimxyz_coco/images/train")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "keypoints": torch.stack([sample["keypoints"] for sample in samples], dim=0),
        "bbox": torch.stack([sample["bbox"] for sample in samples], dim=0),
        "image_id": torch.stack([sample["image_id"] for sample in samples], dim=0),
        "metadata": [sample["metadata"] for sample in samples],
    }


def main() -> None:
    args = parse_args()
    dataset = SwimXYZCocoDataset(
        coco_json=args.coco_json,
        images_root=args.images_root,
        train=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    batch = next(iter(loader))

    print(f"Dataset samples: {len(dataset)}")
    print(f"Image batch shape: {tuple(batch['image'].shape)}")
    print(f"Keypoint batch shape: {tuple(batch['keypoints'].shape)}")
    print(f"BBox shape: {tuple(batch['bbox'].shape)}")
    print(f"Image IDs: {batch['image_id'].tolist()}")
    print("Sample metadata:")
    for item in batch["metadata"]:
        print(f"  - {item}")


if __name__ == "__main__":
    main()

