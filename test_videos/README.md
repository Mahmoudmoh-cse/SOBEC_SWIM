# AquaIQ Test Videos

This folder contains local video fixtures for testing AquaIQ's MediaPipe analysis and analysis-quality trust layer.

The downloaded videos are intentionally ignored by git because they are binary test assets. The folder structure, this README, and `manifest.json` are source-controlled so the fixture set can be recreated or audited later.

## Folders

- `freestyle_good/`: clear freestyle clip for expected successful pose detection.
- `freestyle_low_light/`: generated low-light version to stress confidence scoring.
- `freestyle_bad_angle/`: generated cropped/tilted version to stress poor-angle detection.
- `freestyle_no_pose/`: generated pool-water clip with no swimmer for no-pose fallback tests.
- `butterfly/`: real butterfly stroke clip.
- `breaststroke/`: real breaststroke race clip.
- `underwater/`: real underwater swimming clip.
- `crowded_pool/`: real race clip with multiple swimmers and pool activity.
- `shaky_camera/`: generated shaky-camera version to stress stability.

## Notes

- Real source videos are from Wikimedia Commons and use Creative Commons-compatible licenses listed in `manifest.json`.
- Generated variants are derived locally from the freestyle source clip for test purposes.
- Keep new binary videos out of git. Add metadata to `manifest.json` when adding or replacing fixtures.

## Current MediaPipe Baseline

Using the current AquaIQ analyzer, the fixtures produced these results:

- `freestyle_good`: completed, high confidence, 100% pose rate.
- `freestyle_low_light`: low confidence, low confidence label, 12% pose rate.
- `freestyle_bad_angle`: low confidence, low confidence label, 15% pose rate.
- `freestyle_no_pose`: no pose detected, 0% pose rate.
- `butterfly`: completed, medium confidence, 72% pose rate.
- `breaststroke`: completed, high confidence, 100% pose rate.
- `underwater`: completed, medium confidence, 69% pose rate.
- `crowded_pool`: completed, medium confidence, 100% pose rate.
- `shaky_camera`: completed, high confidence, 100% pose rate.
