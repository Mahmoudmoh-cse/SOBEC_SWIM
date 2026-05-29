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


class YOLOPoseBackend(PoseEstimator):
    name = "yolo_pose"

    def __init__(self, model_path: str | None = None, device: str | None = None) -> None:
        settings = get_settings()
        self.model_path = model_path or settings.yolo_pose_model
        self.device = device or settings.pose_device
        self.model: Any | None = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise PoseBackendUnavailable(f"Ultralytics YOLO is not installed: {exc}") from exc

        try:
            self.model = YOLO(self.model_path)
        except Exception as exc:
            raise PoseBackendUnavailable(f"Could not load YOLO pose model {self.model_path}: {exc}") from exc

    def infer_frame(
        self,
        frame: Any,
        *,
        frame_index: int = 0,
        timestamp: float = 0.0,
        view_type: str = "side",
        quality_flags: list[str] | None = None,
    ) -> PoseFrameResult:
        if self.model is None:
            self.load()
        try:
            results = self.model.predict(frame, verbose=False, device=self.device)
        except Exception:
            return empty_pose_result(
                frame_index=frame_index,
                timestamp=timestamp,
                view_type=view_type,
                quality_flags=quality_flags,
                backend=self.name,
            )

        if not results:
            return empty_pose_result(
                frame_index=frame_index,
                timestamp=timestamp,
                view_type=view_type,
                quality_flags=quality_flags,
                backend=self.name,
            )
        return self._result_to_pose(results[0], frame_index, timestamp, view_type, quality_flags or [])

    def _result_to_pose(
        self,
        result: Any,
        frame_index: int,
        timestamp: float,
        view_type: str,
        quality_flags: list[str],
    ) -> PoseFrameResult:
        boxes = getattr(result, "boxes", None)
        keypoints_obj = getattr(result, "keypoints", None)
        if boxes is None or keypoints_obj is None or len(boxes) == 0:
            return empty_pose_result(
                frame_index=frame_index,
                timestamp=timestamp,
                view_type=view_type,
                quality_flags=quality_flags,
                backend=self.name,
            )

        box_values = _tensor_to_list(boxes.xyxy)
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in box_values]
        person_index = max(range(len(areas)), key=lambda index: areas[index])
        xy_values = _tensor_to_list(keypoints_obj.xy)[person_index]
        conf_values = _tensor_to_list(keypoints_obj.conf)[person_index]
        keypoints = [
            Keypoint(name=KEYPOINT_NAMES[index], x=float(point[0]), y=float(point[1]), confidence=float(conf_values[index]))
            for index, point in enumerate(xy_values[: len(KEYPOINT_NAMES)])
        ]
        bbox = [float(value) for value in box_values[person_index][:4]] if box_values else bbox_from_keypoints(keypoints)
        return PoseFrameResult(
            timestamp=timestamp,
            frame_index=frame_index,
            keypoints=keypoints,
            confidence=average_keypoint_confidence(keypoints),
            bbox=bbox,
            view_type=view_type,
            quality_flags=quality_flags,
            backend=self.name,
        )


def _tensor_to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
