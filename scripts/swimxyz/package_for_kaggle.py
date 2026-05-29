"""Package processed SwimXYZ COCO data for Kaggle training workflows."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from swimxyz_common import ensure_dir


README_TEXT = """# AquaIQ SwimXYZ COCO Dataset

This archive contains the processed COCO-style SwimXYZ subset prepared by the
AquaIQ local dataset engineering pipeline. It is intended for later Kaggle GPU
fine-tuning experiments with RTMPose, ViTPose, or similar keypoint models.

## Structure

```text
swimxyz_coco/
  images/
    train/
    val/
  annotations/
    train.json
    val.json
```

Each annotation uses COCO keypoint fields: `keypoints`, `num_keypoints`, `bbox`,
`area`, `iscrowd`, `image_id`, and `category_id`.

## Kaggle Workflow

1. Upload this zip as a Kaggle Dataset.
2. Attach the dataset to a GPU notebook.
3. Install or use a Kaggle image that includes the pose framework you plan to
   fine-tune, such as MMPose for RTMPose or a ViTPose training implementation.
4. Point the framework's train/val annotation paths at:
   - `/kaggle/input/<dataset-name>/swimxyz_coco/annotations/train.json`
   - `/kaggle/input/<dataset-name>/swimxyz_coco/annotations/val.json`
5. Point image roots at:
   - `/kaggle/input/<dataset-name>/swimxyz_coco/images/train`
   - `/kaggle/input/<dataset-name>/swimxyz_coco/images/val`

## Important Notes

- This archive does not include raw SwimXYZ videos.
- This archive does not include YouTube videos.
- Local AquaIQ preprocessing is CPU-only and performs no model training.
- Validate sample visualizations before training; incorrect parser assumptions
  will produce poor fine-tuning data.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", default="data/processed/swimxyz_coco")
    parser.add_argument("--output", default="outputs/swimxyz_coco_kaggle_ready.zip")
    parser.add_argument("--readme", default="outputs/README_KAGGLE.md")
    return parser.parse_args()


def _validate_processed_root(processed_root: Path) -> None:
    required = [
        processed_root / "images" / "train",
        processed_root / "images" / "val",
        processed_root / "annotations" / "train.json",
        processed_root / "annotations" / "val.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Processed COCO dataset is incomplete. Missing:\n"
            f"{formatted}\nRun convert_to_coco.py before packaging."
        )


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    output_path = Path(args.output)
    readme_path = Path(args.readme)

    _validate_processed_root(processed_root)
    ensure_dir(output_path.parent)
    ensure_dir(readme_path.parent)
    readme_path.write_text(README_TEXT, encoding="utf-8")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(processed_root.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, Path("swimxyz_coco") / path.relative_to(processed_root))
        archive.write(readme_path, "README_KAGGLE.md")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[package] wrote {output_path} ({size_mb:.2f} MB)")
    print("[package] raw videos, interim frames, debug images, and reports were not included")


if __name__ == "__main__":
    main()

