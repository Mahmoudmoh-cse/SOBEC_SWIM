from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.swim_analysis.pose.base import (
    KEYPOINT_NAMES,
    Keypoint,
    PoseBackendUnavailable,
    PoseEstimator,
    PoseFrameResult,
    average_keypoint_confidence,
    bbox_from_keypoints,
    empty_pose_result,
)


MEDIAPIPE_TO_COCO = {
    0: "nose",
    2: "left_eye",
    5: "right_eye",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


class MediaPipeBackend(PoseEstimator):
    name = "mediapipe"

    def __init__(self) -> None:
        self.mp: Any | None = None
        self.cv2: Any | None = None
        self.pose: Any | None = None

    def load(self) -> None:
        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise PoseBackendUnavailable(f"MediaPipe fallback is not installed: {exc}") from exc

        settings = get_settings()
        self.cv2 = cv2
        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=max(0, min(2, int(settings.mediapipe_model_complexity))),
            min_detection_confidence=max(0.1, min(0.95, float(settings.mediapipe_min_detection_confidence))),
            min_tracking_confidence=max(0.1, min(0.95, float(settings.mediapipe_min_tracking_confidence))),
        )

    def infer_frame(
        self,
        frame: Any,
        *,
        frame_index: int = 0,
        timestamp: float = 0.0,
        view_type: str = "side",
        quality_flags: list[str] | None = None,
    ) -> PoseFrameResult:
        if self.pose is None or self.cv2 is None:
            self.load()

        height, width = frame.shape[:2]
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return empty_pose_result(
                frame_index=frame_index,
                timestamp=timestamp,
                view_type=view_type,
                quality_flags=quality_flags,
                backend=self.name,
            )

        landmarks = result.pose_landmarks.landmark
        by_name: dict[str, Keypoint] = {}
        for mp_index, name in MEDIAPIPE_TO_COCO.items():
            landmark = landmarks[mp_index]
            by_name[name] = Keypoint(
                name=name,
                x=float(landmark.x) * width,
                y=float(landmark.y) * height,
                z=float(landmark.z),
                confidence=float(landmark.visibility),
            )
        keypoints = [by_name[name] for name in KEYPOINT_NAMES if name in by_name]
        return PoseFrameResult(
            timestamp=timestamp,
            frame_index=frame_index,
            keypoints=keypoints,
            confidence=average_keypoint_confidence(keypoints),
            bbox=bbox_from_keypoints(keypoints),
            view_type=view_type,
            quality_flags=quality_flags or [],
            backend=self.name,
        )

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()
            self.pose = None
