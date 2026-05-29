# SwimXYZ Data Pipeline

AquaIQ uses this local pipeline to convert a small, selected SwimXYZ subset into
a clean COCO-style pose dataset. This phase is dataset engineering only: no
local model training, no YouTube training data, and no automatic full-dataset
processing.

## What To Download First

Start with the SwimXYZ files that include side-view freestyle sequences. Part 2
is preferred when available because it best matches the first training target.
Place files under:

```text
data/raw/swimxyz/
  Side_above_water/
  Side_underwater/
  Side_water_level/
  annotations/
    Freestyle/
  smpl/
```

The annotation files are mandatory. Videos alone are not enough for supervised
pose fine-tuning because RTMPose/ViTPose need per-frame joint labels. The
pipeline can inspect 3D joints and SMPL metadata when present, but the first
COCO export uses 2D joints.

For the local AquaIQ copy, extract the archives like this:

```powershell
Expand-Archive -Path "D:\learning\AquaIQ\Freestyle_part2.zip" -DestinationPath data\raw\swimxyz -Force
Expand-Archive -Path "D:\learning\AquaIQ\Freestyle_labels.zip" -DestinationPath data\raw\swimxyz\annotations -Force
Expand-Archive -Path "D:\learning\AquaIQ\smpl_swimming_motions.zip" -DestinationPath data\raw\swimxyz\smpl -Force
```

The converter currently uses the `COCO\2D_cam.txt` label files for 2D pose
training data. SMPL `.npz` files are kept locally for future 3D/SMPL work and
are not used for Phase 1 COCO conversion.

## Why Freestyle Side View First

Freestyle side-view clips give the most direct signal for body alignment, head
stability, hip position, kick rhythm, and stroke timing. Starting with one stroke
and one view keeps the parser, skeleton mapping, and quality checks auditable
before expanding to additional views and strokes.

## Local Workflow

Run commands from the AquaIQ project root.

```powershell
python scripts/swimxyz/inspect_swimxyz.py --data-root data/raw/swimxyz

python scripts/swimxyz/extract_frames.py `
  --videos-root data/raw/swimxyz `
  --output-root data/interim/frames `
  --max-videos 10 `
  --stroke freestyle `
  --view side

python scripts/swimxyz/convert_to_coco.py `
  --data-root data/raw/swimxyz `
  --frames-root data/interim/frames `
  --output-root data/processed/swimxyz_coco `
  --val-ratio 0.2 `
  --subsample-stride 12

python scripts/swimxyz/visualize_samples.py `
  --coco-json data/processed/swimxyz_coco/annotations/train.json `
  --images-root data/processed/swimxyz_coco/images/train `
  --num-samples 20

python scripts/swimxyz/test_dataloader.py `
  --coco-json data/processed/swimxyz_coco/annotations/train.json `
  --images-root data/processed/swimxyz_coco/images/train

python scripts/swimxyz/package_for_kaggle.py `
  --processed-root data/processed/swimxyz_coco `
  --output outputs/swimxyz_coco_kaggle_ready.zip
```

The frame extractor requires `--max-videos` so a large SwimXYZ download is not
processed accidentally. It writes every decoded source frame and preserves exact
original frame indices. Use `convert_to_coco.py --subsample-stride N` only after
annotation matching if you want a smaller training subset.

SwimXYZ `COCO\2D_cam.txt` labels use a bottom-left y-axis origin. The converter
flips y into normal image coordinates, validates exact video/frame matches, and
writes `outputs/reports/coco_matching_debug.jsonl` for audit.

## Outputs

Inspection report:

```text
outputs/reports/swimxyz_inspection.md
```

Processed COCO dataset:

```text
data/processed/swimxyz_coco/
  images/
    train/
    val/
  annotations/
    train.json
    val.json
```

Visual sanity checks:

```text
outputs/debug_visualizations/
```

Kaggle package:

```text
outputs/swimxyz_coco_kaggle_ready.zip
outputs/README_KAGGLE.md
```

## Kaggle Workflow

Upload only `outputs/swimxyz_coco_kaggle_ready.zip` as a Kaggle Dataset. Attach
it to a GPU notebook and point the training framework to the COCO annotation
files and image roots inside `swimxyz_coco/`.

Use Kaggle for model training and fine-tuning. Local AquaIQ preprocessing should
stay CPU-only and focused on inspection, conversion, visualization, and packaging.

## What Not To Upload

Do not upload raw videos, interim frame folders, YouTube clips, debug
visualizations, local database files, or analysis outputs unless a future
experiment explicitly requires them. YouTube videos are reserved for later
testing and demos, not training.

## Known Limitations

SwimXYZ is synthetic, so models fine-tuned on it may not transfer cleanly to real
pool footage without domain adaptation or mixed real labels.

YouTube swimming videos are noisy, compressed, and often have moving cameras or
partial swimmers. They should be treated as evaluation/demo material until
proper labels and licensing are handled.

Visual validation is critical. Always inspect the generated sample overlays
before training; if joints are flipped, shifted, or outside the swimmer, fix the
parser or schema mapping before packaging for Kaggle.
