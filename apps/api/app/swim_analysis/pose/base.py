from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class PoseBackendUnavailable(RuntimeError):
    """Raised when an optional pose stack is not installed or not configured."""


@dataclass
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float
    z: float | None = None
    visibility_state: str = "visible"
    source_backend: str | None = None
    frame_index: int | None = None
    quality_flags: list[str] = field(default_factory=list)
    rejected_reason: str | None = None
    interpolated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PoseFrameResult:
    timestamp: float
    frame_index: int
    keypoints: list[Keypoint] = field(default_factory=list)
    confidence: float = 0.0
    bbox: list[float] | None = None
    view_type: str = "side"
    quality_flags: list[str] = field(default_factory=list)
    backend: str = ""
    track_id: int | None = None
    raw_keypoints: list[Keypoint] | None = None
    interpolated: bool = False
    roi: dict[str, Any] | None = None
    skipped: bool = False
    rejected_keypoints: list[Keypoint] = field(default_factory=list)
    tracking_quality: float | None = None
    frame_quality_score: float | None = None
    debug_events: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)

    def has_pose(self, min_confidence: float = 0.2) -> bool:
        return any(point.confidence >= min_confidence and point.visibility_state != "rejected_outlier" for point in self.keypoints)

    def keypoint(self, name: str, min_confidence: float = 0.0) -> Keypoint | None:
        for point in self.keypoints:
            if point.name == name and point.confidence >= min_confidence and point.visibility_state != "rejected_outlier":
                return point
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "timestamp": round(float(self.timestamp), 4),
            "frame_index": int(self.frame_index),
            "keypoints": [point.to_dict() for point in self.keypoints],
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox,
            "view_type": self.view_type,
            "quality_flags": list(self.quality_flags),
            "backend": self.backend,
            "track_id": self.track_id,
            "interpolated": self.interpolated,
            "skipped": self.skipped,
        }
        if self.raw_keypoints is not None:
            payload["raw_keypoints"] = [point.to_dict() for point in self.raw_keypoints]
        if self.roi is not None:
            payload["roi"] = self.roi
        if self.rejected_keypoints:
            payload["rejected_keypoints"] = [point.to_dict() for point in self.rejected_keypoints]
        if self.tracking_quality is not None:
            payload["tracking_quality"] = round(float(self.tracking_quality), 4)
        if self.frame_quality_score is not None:
            payload["frame_quality_score"] = round(float(self.frame_quality_score), 4)
        if self.debug_events:
            payload["debug_events"] = list(self.debug_events)
        if self.debug_info:
            payload["debug_info"] = self.debug_info
        return payload


class PoseEstimator(ABC):
    name = "base"

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer_frame(
        self,
        frame: Any,
        *,
        frame_index: int = 0,
        timestamp: float = 0.0,
        view_type: str = "side",
        quality_flags: list[str] | None = None,
    ) -> PoseFrameResult:
        raise NotImplementedError

    def infer_video(self, frames: Iterable[tuple[int, float, Any, list[str], str]]) -> list[PoseFrameResult]:
        return [
            self.infer_frame(
                frame,
                frame_index=frame_index,
                timestamp=timestamp,
                view_type=view_type,
                quality_flags=quality_flags,
            )
            for frame_index, timestamp, frame, quality_flags, view_type in frames
        ]


def empty_pose_result(
    *,
    frame_index: int,
    timestamp: float,
    view_type: str,
    quality_flags: list[str] | None = None,
    backend: str = "",
) -> PoseFrameResult:
    return PoseFrameResult(
        timestamp=timestamp,
        frame_index=frame_index,
        keypoints=[],
        confidence=0.0,
        bbox=None,
        view_type=view_type,
        quality_flags=quality_flags or [],
        backend=backend,
    )


def average_keypoint_confidence(keypoints: list[Keypoint]) -> float:
    if not keypoints:
        return 0.0
    return sum(max(0.0, min(1.0, point.confidence)) for point in keypoints) / len(keypoints)


def bbox_from_keypoints(keypoints: list[Keypoint], min_confidence: float = 0.2) -> list[float] | None:
    visible = [point for point in keypoints if point.confidence >= min_confidence and point.visibility_state != "rejected_outlier"]
    if not visible:
        return None
    xs = [point.x for point in visible]
    ys = [point.y for point in visible]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
