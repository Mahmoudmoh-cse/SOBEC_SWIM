# AquaIQ Phase 1

AquaIQ is a coach-first swimming performance platform. Phase 1 now includes the original coach dashboard plus a professional-shaped two-view swimming motion analysis pipeline for smartphone video: one side-view clip and one front/back-view clip. The system is designed for explainable, confidence-aware feedback. It does not claim medical, Olympic-grade, or lab-grade biomechanics accuracy.

## Prerequisites

- Node.js 20 or newer
- Python 3.11 or newer
- Docker Desktop, required for the local PostgreSQL database

Docker is not currently available on this machine, so install Docker Desktop before running the database commands.

## Project Structure

```
text
apps/
  api/   FastAPI, SQLAlchemy, Alembic, provider interfaces, tests
  web/   Next.js App Router, Tailwind, Recharts, Zustand
docs/    Product blueprint and implementation notes
scripts/swimxyz/  SwimXYZ inspection, frame extraction, COCO conversion, validation, packaging
data/    Raw/interim/processed local datasets, kept out of production runtime assumptions
outputs/ Debug visualizations, reports, and Kaggle-ready archives
```

Professional two-view analysis lives under:

```
text
apps/api/app/swim_analysis/
  router.py          /api/swim-analysis endpoints
  service.py         upload storage, DB job lifecycle
  processor.py       two-view CV pipeline orchestration
  preprocessing.py   OpenCV metadata, quality scoring, frame sampling
  pose/              RTMPose primary backend plus YOLO/MediaPipe fallbacks
  roi.py             swimmer localization, centered ROI crops, coordinate remapping
  tracking.py        single-swimmer identity guard
  temporal_tracking.py confidence-aware trajectories, outlier rejection, swaps, interpolation
  smoothing.py       OneEuroFilter and moving-average fallback
  sync.py            deterministic zero/manual sync and motion rough sync
  phase1_metrics.py  reliability-first Phase 1 biomechanics metrics
  visualizations.py  matplotlib research graphs and figure metadata
  metrics.py         landmark time-series metrics only
  report.py          structured report and annotated videos
  quality.py         metric confidence helpers
```

SwimXYZ dataset engineering lives under:

```text
apps/api/app/data_pipeline/
  swimxyz_dataset.py  CPU PyTorch loader for processed COCO exports
scripts/swimxyz/
  inspect_swimxyz.py      inspect available SwimXYZ videos/annotations
  extract_frames.py       safely extract all frames for a capped video subset
  convert_to_coco.py      exact frame matching, post-match subsampling, COCO export
  visualize_samples.py    draw keypoint overlays for parser validation
  test_dataloader.py      CPU-only DataLoader smoke test
  package_for_kaggle.py   zip processed data for later Kaggle GPU training
docs/swimxyz_data_pipeline.md
```

## Backend Setup

```powershell
cd apps/api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
cd ..\..
docker compose up -d postgres
cd apps/api
alembic upgrade head
python -m app.scripts.seed
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

Seeded coach login:

- Email: `coach@aquaiq.local`
- Password: `AquaIQ123!`

## Backend Setup Without Docker

Use this path if Docker Desktop is not installed yet. It runs the same API against a local SQLite database file.

```
powershell
cd D:\swimming_Project\apps\api
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="sqlite:///./aquaiq.db"
alembic upgrade head
python -m app.scripts.seed
uvicorn app.main:app --reload --port 8001
```

The legacy single-video session uploader still uses `TECHNIQUE_ANALYZER=auto` by default. When MediaPipe dependencies from `requirements.txt` are installed, that legacy path processes uploaded session videos with MediaPipe pose extraction; otherwise it falls back to deterministic mock analysis and labels it in the UI. To force real analysis while testing the legacy uploader:

```powershell
$env:TECHNIQUE_ANALYZER="mediapipe"
uvicorn app.main:app --reload --port 8001
```

Technique reports include analysis quality fields so coaches can trust the result: `analysis_status`, `frames_total`, `frames_analyzed`, `pose_detected_frames`, `pose_detection_rate`, `confidence_score`, `confidence_label`, `analysis_warning`, and `analysis_error`. Low-confidence or no-pose reports tell the coach what to fix, such as using a clearer side-view video with the full body visible.

MediaPipe reports also generate Phase 1 explainability assets:

- The original upload remains unchanged under `uploads/<swimmer_id>/<session_id>/`.
- The AI skeleton/keypoint overlay video is stored under `uploads/processed_videos/`.
- Snapshot fallbacks are stored under `uploads/analysis_frames/` if overlay generation fails.
- The report API returns `analysis_overlay_video_url`, `analysis_frame_urls`, and `analysis_events` so the web app can show the original video, AI overlay, skeleton snapshots, and a readable analysis timeline.

This is an explainability layer for local pilot validation, not final production-grade biomechanics.

## Professional Two-View Motion Analysis

The new Phase 1 CV path is available at:

- `POST /api/swim-analysis/upload`
- `GET /api/swim-analysis/{analysis_id}`
- `GET /api/swim-analysis/{analysis_id}/metrics`
- `GET /api/swim-analysis/{analysis_id}/velocity`
- `GET /api/swim-analysis/{analysis_id}/report`
- `GET /api/swim-analysis/{analysis_id}/annotated-video`

The web UI is integrated into the coach console at `http://localhost:3000/dashboard/swim-analysis` after login. The original `/swim-analysis` route still works as a direct shortcut.

Upload form fields:

- `side_video`: side-view smartphone video
- `front_video`: front/back-view smartphone video
- `swimmer_id`: optional existing swimmer id
- `stroke_type`: default `freestyle`
- `target_fps`: default `30`
- `quality_mode`: default `high_accuracy`
- `lane_length_m`: optional manual lane calibration, usually `25` or `50`

The pipeline saves originals, raw landmarks, cleaned trajectories, research figures, annotated videos, and a report JSON under `uploads/swim_analysis/<analysis_id>/`.

Current Phase 1 pipeline:

```text
video upload
-> frame extraction and quality flags
-> swimmer localization
-> centered ROI crop
-> pose estimation on ROI
-> coordinate remapping to output video coordinates
-> confidence-aware keypoint structuring
-> single-swimmer tracking
-> temporal left/right correction
-> physics-aware outlier rejection
-> short-gap interpolation only
-> confidence-aware smoothing
-> trajectory JSON export
-> biomechanics/reliability metrics
-> research visualization PNGs
-> annotated videos
-> API response and dashboard
```

The performance engine now includes:

- video quality scoring for resolution, FPS, blur, lighting, stability, and swimmer visibility
- lane calibration with manual lane length and fallback body/motion estimates
- centroid velocity, acceleration, dead-spot detection, breakout speed, and phase-level speed summaries
- start/push-off, underwater, breakout, free-swim, turn, finish, and stroke-cycle segmentation
- biomechanics metrics with units, confidence, interpretation, and coaching meaning
- rule-based technique faults with timestamp ranges and drills
- deterministic local coaching report with strengths, issues, weekly correction plan, and safety note
- ROI boxes, low-confidence joints, missing-joint counts, rejected outliers, and tracking quality indicators in annotated videos
- research-style figures for before/after trajectories, swap correction, missing target percentage, outlier impact, drift/noise proxies, stroke-cycle curves, and reliability timelines

Generated Phase 1 artifacts include:

- `landmarks/raw_landmarks.json`
- `landmarks/smoothed_landmarks.json`
- `landmarks/phase1_trajectories.json`
- `artifacts/research_figures.json`
- `artifacts/figures/*.png`
- `processed/side_annotated.mp4`
- `processed/front_annotated.mp4`

The report response includes `summary.reliability_score`, `warnings`, `trajectories.trajectory_data_url`, `research_figures`, annotated video URLs, metrics, and optional `debug_info` when `GET /api/swim-analysis/{analysis_id}/report?debug=true` is used.

### Pose Backend Configuration

RTMPose through MMPose is the primary backend. It is intentionally optional because MMPose/MMCV installation differs by CUDA, PyTorch, and operating system.

Environment variables:

```powershell
$env:POSE_BACKEND="rtmpose"
$env:FALLBACK_POSE_BACKEND="mediapipe"   # use "yolo_pose", "mediapipe", "opencv_motion", or leave empty
$env:POSE_DEVICE="cuda"                  # use "cpu" when CUDA is unavailable
$env:RTMPOSE_CONFIG_PATH="D:\swimming_Project\aquaiq_rtmpose_s_swimxyz_smoke.py"
$env:RTMPOSE_CHECKPOINT_PATH="D:\swimming_Project\models\aquaiq_swimpose_rtmpose_v1\aquaiq_swimming_rtmpose_v1.pth"
$env:YOLO_POSE_MODEL="yolo11x-pose.pt"
```

The current local trained checkpoint was unpacked from:

```text
D:\swimming_Project\AquaIQ SwimPose RTMPose v1.zip
```

into:

```text
D:\swimming_Project\models\aquaiq_swimpose_rtmpose_v1\aquaiq_swimming_rtmpose_v1.pth
```

The local `apps/api/.env` is already pointed at this checkpoint with `POSE_DEVICE=cpu` for compatibility. Change it to `POSE_DEVICE=cuda` after installing a CUDA-compatible PyTorch/MMPose stack.

Recommended high-accuracy setup:

1. Install PyTorch for your CUDA version from the official PyTorch selector.
2. Install MMPose following the official MMPose install guide for your platform.
3. Use the AquaIQ SwimPose checkpoint above, or download another RTMPose/RTMW high-accuracy config/checkpoint from the MMPose model zoo.
4. Set `RTMPOSE_CONFIG_PATH`, `RTMPOSE_CHECKPOINT_PATH`, and `POSE_DEVICE`.
5. Keep `FALLBACK_POSE_BACKEND=mediapipe` or `yolo_pose` during pilots so a missing RTMPose checkpoint produces a clear warning instead of a silent fake result.

The backend records pose-backend errors in `processing_errors` and never fabricates high-confidence metrics when pose data is missing. If RTMPose, YOLO, and MediaPipe are unavailable, AquaIQ falls back to `opencv_motion`, a centroid/bbox-only OpenCV tracker. This allows video quality, rough calibration, phase timing, and velocity trends to complete, but skeleton and biomechanics metrics remain low-confidence or unavailable. Annotated videos draw the swimmer path and labels but do not fake a skeleton.

### Metrics and Confidence

Metrics are computed only from landmark time-series:

- Side view: stroke count/rate, cycle consistency, body alignment, hip drop indicator, head stability, breathing events, kick rhythm, streamline score.
- Front/back view: arm symmetry, shoulder balance, centerline deviation, left/right timing difference, hand-entry width estimate.
- Combined: overall technique score, confidence score, data quality score, view agreement score.
- Phase 1 reliability layer: hip stability, stroke rhythm estimate, kick rhythm estimate, left/right symmetry estimate, joint visibility score, tracking quality score, analysis reliability score.

Every metric includes:

- `value`
- `confidence`
- `evidence`
- `reason`

Low-support metrics are marked with `"confidence": "low"` and include the numeric confidence inside evidence for debugging.

Force, thrust, and propulsion-style values are not treated as measured physical outputs in Phase 1. Any efficiency-style score is a trajectory-derived coaching proxy and should be interpreted through its confidence and evidence fields.

### Research Visualizations

AquaIQ generates matplotlib PNG figures for each completed two-view run. The dashboard shows them under **Research Visualizations** with tabs for trajectories, tracking quality, error/noise analysis, and stroke cycles.

Figure metadata returned by the API:

- `figure_id`
- `title`
- `type`
- `file_path`
- `description`
- `metric_source`

When no ground-truth labels are supplied, all error-like figures are labeled as `metric_source: "proxy"`. The current proxy charts include raw-to-filtered displacement, trajectory noise/stability, and horizontal/vertical standard deviation. They are useful for coach review and pipeline debugging, but they are not true localization error.

### Recording Setup

- Use two fixed smartphone videos of the same swim.
- Side view: deck-level, swimmer full body visible for multiple strokes.
- Front/back view: centered on the lane, avoid severe cropping.
- Record at 30 fps or higher when possible.
- Avoid backlighting, dark pool corners, heavy splash occlusion, and shaky handheld footage.
- Keep clips at least 5-10 seconds for more stable stroke-cycle timing.

### Known Limitations

- Phase 1 is two-dimensional and camera-angle sensitive.
- There is no underwater camera, IMU, or live stream fusion yet.
- Motion sync is deterministic rough sync from landmark movement peaks; audio clap sync is a future upgrade.
- Swimming-specific RTMPose fine-tuning is not included yet.
- No new model training happens in Phase 1; the upgrade improves preprocessing, ROI handling, temporal tracking, filtering, correction, and reporting around existing pretrained models.
- Occlusion, lane reflections, and crowded pools can lower confidence.
- Long missing joint segments stay missing. Short gaps are interpolated only when reliable endpoints exist and are clearly marked as interpolated.
- Left/right corrections are temporal heuristics, not proof of anatomical identity; swap events are logged for audit.
- Error/noise figures are proxy diagnostics unless ground-truth annotations are added.
- Reports are coaching decision support, not clinical diagnosis or certified biomechanics measurement.
- OpenCV centroid fallback is useful for testing and speed-curve review, but not for catch/elbow/body-roll biomechanics.

### Future Phase 2

- Fine-tune RTMPose/RTMW on swimming-specific above-water and underwater data.
- Add ground-truth annotation imports so mean error and standard deviation figures can report true localization error rather than proxies.
- Add audio clap sync and explicit manual offset controls.
- Add 3D pose reconstruction from calibrated multi-view video.
- Add optional smartwatch/IMU fusion for tempo and kick rhythm validation.
- Add live capture and coach review tools once offline accuracy is stable.

If port `8000` is already busy, use `8001` as shown above and set the web API URL to match:

```powershell
cd D:\swimming_Project\apps\web
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8001/api/v1"
npm run dev
```

## Frontend Setup

```powershell
cd apps/web
npm install
Copy-Item .env.example .env.local
npm run dev
```

The web app will be available at `http://localhost:3000`.

If npm hangs on this Windows environment with certificate revocation errors, retry the install with:

```powershell
npm install --strict-ssl=false --no-audit --no-fund
```

## Generate AI Engineering Guide

The internal AI Engineering Guide is generated locally and kept out of source control.

```powershell
cd D:\swimming_Project
npm install
npm run docs:ai-guide
```

The generated document is written to `docs\output\AquaIQ_AI_Guide.docx`.

## AI Coaching Layer

AquaIQ includes a backend AI provider layer that records coaching outputs in the `ai_outputs` audit table. Local deterministic coaching is the default so the app works without paid services.

```powershell
# Local deterministic provider
$env:AI_PROVIDER="local"

# Hybrid mode: use Claude only when an API key exists, otherwise local fallback
$env:AI_PROVIDER="hybrid"
$env:ANTHROPIC_API_KEY=""

# Anthropic-only mode: requires a real key and errors clearly if missing
$env:AI_PROVIDER="anthropic"
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:CLAUDE_MODEL_HEAVY="claude-sonnet-4-5"
$env:CLAUDE_MODEL_FAST="claude-3-5-haiku-latest"
```

The first AI-assisted outputs are technique summaries, training plan rationales, race debriefs, and mental routine explanations. Each request keeps the existing API response stable while saving provider, model, prompt, raw response, parsed JSON, token metadata when available, and status for later review.

## Useful Commands

```powershell
# Backend tests
cd apps/api
pytest

# Frontend build
cd apps/web
npm run build

# Stop database
docker compose down
```

## Phase 1 Acceptance Flow

1. Start PostgreSQL, run migrations, and seed data.
2. Start the API and web app.
3. Log in as the seeded coach.
4. Create or open a swimmer profile.
5. Log a training session.
6. Upload a video against a session and review the mock technique report.
7. Generate a training plan.
8. Add a race analysis and mental check-in.
9. Confirm the dashboard updates with swimmers, alerts, sessions, and active plans.
