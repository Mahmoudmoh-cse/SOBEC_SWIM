from __future__ import annotations

from typing import Any

from app.swim_analysis.pose.base import PoseBackendUnavailable, PoseEstimator, PoseFrameResult


class OpenCVMotionBackend(PoseEstimator):
    """Centroid/bbox-only fallback for quality and velocity analysis.

    This backend intentionally returns no skeleton keypoints. It keeps the pipeline alive
    when RTMPose/YOLO/MediaPipe are unavailable, while preventing fake biomechanics.
    """

    name = "opencv_motion"

    def __init__(self) -> None:
        self.cv2: Any | None = None

    def load(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise PoseBackendUnavailable(f"OpenCV fallback is not installed: {exc}") from exc
        self.cv2 = cv2

    def infer_frame(
        self,
        frame: Any,
        *,
        frame_index: int = 0,
        timestamp: float = 0.0,
        view_type: str = "side",
        quality_flags: list[str] | None = None,
    ) -> PoseFrameResult:
        if self.cv2 is None:
            self.load()
        cv2 = self.cv2
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(gray, 55, 145)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return PoseFrameResult(
                timestamp=timestamp,
                frame_index=frame_index,
                keypoints=[],
                confidence=0.0,
                bbox=None,
                view_type=view_type,
                quality_flags=[*(quality_flags or []), "opencv_motion_no_contour"],
                backend=self.name,
            )

        min_area = max(120.0, width * height * 0.002)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if 0.05 <= aspect <= 8.0:
                candidates.append((area, x, y, w, h))
        if not candidates:
            return PoseFrameResult(
                timestamp=timestamp,
                frame_index=frame_index,
                keypoints=[],
                confidence=0.0,
                bbox=None,
                view_type=view_type,
                quality_flags=[*(quality_flags or []), "opencv_motion_low_visibility"],
                backend=self.name,
            )
        area, x, y, w, h = max(candidates, key=lambda item: item[0])
        area_ratio = area / max(1.0, width * height)
        confidence = max(0.12, min(0.42, 0.18 + area_ratio * 10))
        return PoseFrameResult(
            timestamp=timestamp,
            frame_index=frame_index,
            keypoints=[],
            confidence=confidence,
            bbox=[float(x), float(y), float(x + w), float(y + h)],
            view_type=view_type,
            quality_flags=[*(quality_flags or []), "centroid_only_no_skeleton"],
            backend=self.name,
        )
