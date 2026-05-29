from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from app.swim_analysis.pose.base import Keypoint, PoseFrameResult, average_keypoint_confidence, bbox_from_keypoints, empty_pose_result


@dataclass(frozen=True)
class ROIConfig:
    enabled: bool = True
    scale: float = 1.85
    min_size_px: int = 96
    max_boundary_clip_ratio: float = 0.35
    min_bbox_area_ratio: float = 0.0012
    max_aspect_ratio: float = 10.0
    max_candidates: int = 5
    center_crop_width_ratio: float = 0.32
    center_crop_height_ratio: float = 0.23
    previous_roi_expand: float = 1.18
    temporal_continuity_weight: float = 0.16
    max_temporal_jump_ratio: float = 2.1
    motion_supported_jump_ratio: float = 3.2


@dataclass
class ROIResult:
    valid: bool
    crop: Any | None
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    frame_width: int = 0
    frame_height: int = 0
    localization_bbox: list[float] | None = None
    confidence: float = 0.0
    reason: str | None = None
    source: str = "unknown"
    candidate_score: float = 0.0
    boundary_touch_ratio: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    frame_index: int | None = None
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "localization_bbox": self.localization_bbox,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "source": self.source,
            "candidate_score": round(float(self.candidate_score), 4),
            "boundary_touch_ratio": round(float(self.boundary_touch_ratio), 4),
            "metrics": self.metrics,
            "candidates": self.candidates,
            "frame_index": self.frame_index,
            "timestamp": round(float(self.timestamp), 4) if self.timestamp is not None else None,
        }


class SwimmerLocalizer:
    """Find a swimmer-centered crop using the OpenCV motion/edge detector already used by the fallback stack."""

    def __init__(self, config: ROIConfig | None = None) -> None:
        self.config = config or ROIConfig()
        self.cv2: Any | None = None
        self._previous_gray: Any | None = None
        self._previous_roi: ROIResult | None = None
        self._roi_velocity: tuple[float, float] | None = None

    def localize(
        self,
        frame: Any,
        *,
        frame_index: int,
        timestamp: float,
        view_type: str,
        quality_flags: list[str] | None = None,
    ) -> ROIResult:
        if not self.config.enabled:
            height, width = frame.shape[:2]
            return ROIResult(True, frame, 0, 0, width, height, width, height, [0.0, 0.0, float(width), float(height)], 1.0, source="full_frame_disabled")

        candidates = self.candidate_rois(frame, frame_index=frame_index, timestamp=timestamp, view_type=view_type, quality_flags=quality_flags)
        if candidates:
            selected = candidates[0]
            self.remember_roi(selected)
            return replace(selected, candidates=[candidate.to_dict() for candidate in candidates])
        height, width = frame.shape[:2]
        return ROIResult(False, None, frame_width=width, frame_height=height, confidence=0.0, reason="swimmer_not_localized", source="none")

    def candidate_rois(
        self,
        frame: Any,
        *,
        frame_index: int,
        timestamp: float,
        view_type: str,
        quality_flags: list[str] | None = None,
        allow_previous: bool = True,
    ) -> list[ROIResult]:
        if not self.config.enabled:
            height, width = frame.shape[:2]
            return [ROIResult(True, frame, 0, 0, width, height, width, height, [0.0, 0.0, float(width), float(height)], 1.0, source="full_frame_disabled")]

        cv2 = self._cv2()
        height, width = frame.shape[:2]
        candidates: list[ROIResult] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if allow_previous:
            candidates.extend(self._previous_roi_candidates(frame))
        candidates.extend(self._motion_candidates(cv2, frame, gray))
        candidates.extend(self._edge_candidates(cv2, frame))
        candidates.extend(self._color_water_suppression_candidates(cv2, frame))
        candidates.extend(self._center_candidates(frame))
        self._previous_gray = gray

        valid = _dedupe_rois([candidate for candidate in candidates if candidate.valid and candidate.crop is not None])
        valid = self._apply_temporal_continuity(valid, frame_index=frame_index, timestamp=timestamp)
        valid.sort(key=lambda item: item.candidate_score, reverse=True)
        if not valid:
            return []
        output: list[ROIResult] = []
        for candidate in valid:
            if not any(_roi_overlap_ratio(candidate, existing) > 0.78 for existing in output):
                output.append(candidate)
            if len(output) >= max(1, int(self.config.max_candidates)):
                break
        return output

    def reset_tracking(self, *, keep_image_history: bool = True) -> None:
        self._previous_roi = None
        self._roi_velocity = None
        if not keep_image_history:
            self._previous_gray = None

    def remember_roi(self, roi: ROIResult | None) -> None:
        if roi is not None and roi.valid and roi.width > 0 and roi.height > 0:
            if self._previous_roi is not None and self._previous_roi.timestamp is not None and roi.timestamp is not None:
                dt = max(1e-3, float(roi.timestamp) - float(self._previous_roi.timestamp))
                previous_cx, previous_cy = _roi_center(self._previous_roi)
                current_cx, current_cy = _roi_center(roi)
                self._roi_velocity = ((current_cx - previous_cx) / dt, (current_cy - previous_cy) / dt)
            self._previous_roi = roi

    def _previous_roi_candidates(self, frame: Any) -> list[ROIResult]:
        if self._previous_roi is None or not self._previous_roi.valid:
            return []
        roi = self._previous_roi
        cx = roi.x + roi.width / 2.0
        cy = roi.y + roi.height / 2.0
        if self._roi_velocity is not None:
            # Candidate generation does not know the current timestamp; the temporal scoring pass
            # below performs the exact prediction. This small nudge keeps the previous crop useful
            # when the swimmer is moving steadily between sampled frames.
            cx += self._roi_velocity[0] * 0.12
            cy += self._roi_velocity[1] * 0.12
        width = max(self.config.min_size_px, roi.width * self.config.previous_roi_expand)
        height = max(self.config.min_size_px, roi.height * self.config.previous_roi_expand)
        bbox = [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]
        return [_roi_from_bbox(frame, bbox, source="previous_roi", confidence=0.62, score_bonus=0.22, config=self.config, scale=1.0)]

    def _apply_temporal_continuity(self, candidates: list[ROIResult], *, frame_index: int, timestamp: float) -> list[ROIResult]:
        if not candidates:
            return []
        previous = self._previous_roi
        if previous is None or not previous.valid:
            return [replace(candidate, frame_index=frame_index, timestamp=timestamp) for candidate in candidates]

        previous_cx, previous_cy = _roi_center(previous)
        predicted_x = previous_cx
        predicted_y = previous_cy
        if self._roi_velocity is not None and previous.timestamp is not None:
            dt = max(0.0, min(1.2, float(timestamp) - float(previous.timestamp)))
            predicted_x += self._roi_velocity[0] * dt
            predicted_y += self._roi_velocity[1] * dt
        normalizer = max(1.0, (_roi_diagonal(previous) + _frame_diagonal(previous)) * 0.5)
        output: list[ROIResult] = []
        for candidate in candidates:
            cx, cy = _roi_center(candidate)
            distance = ((cx - predicted_x) ** 2 + (cy - predicted_y) ** 2) ** 0.5
            jump_ratio = distance / normalizer
            overlap = _roi_overlap_ratio(candidate, previous)
            motion_supported = candidate.source in {"motion", "water_suppressed_color"} and float(candidate.confidence) >= 0.24
            max_jump = self.config.motion_supported_jump_ratio if motion_supported else self.config.max_temporal_jump_ratio
            continuity_bonus = max(0.0, 1.0 - min(1.0, jump_ratio)) * self.config.temporal_continuity_weight
            overlap_bonus = min(0.08, overlap * 0.08)
            jump_penalty = min(0.34, max(0.0, jump_ratio - 0.75) * (0.08 if motion_supported else 0.15))
            if jump_ratio > max_jump:
                jump_penalty += min(0.2, (jump_ratio - max_jump) * 0.08)
            metrics = dict(candidate.metrics)
            metrics.update(
                {
                    "temporal_jump_ratio": round(float(jump_ratio), 4),
                    "temporal_distance_px": round(float(distance), 3),
                    "temporal_overlap_previous": round(float(overlap), 4),
                    "temporal_motion_supported": motion_supported,
                    "temporal_score_delta": round(float(continuity_bonus + overlap_bonus - jump_penalty), 4),
                    "predicted_center_x": round(float(predicted_x), 3),
                    "predicted_center_y": round(float(predicted_y), 3),
                }
            )
            output.append(
                replace(
                    candidate,
                    candidate_score=max(0.0, min(1.0, candidate.candidate_score + continuity_bonus + overlap_bonus - jump_penalty)),
                    metrics=metrics,
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )
        return output

    def _motion_candidates(self, cv2: Any, frame: Any, gray: Any) -> list[ROIResult]:
        if self._previous_gray is None:
            return []
        diff = cv2.absdiff(self._previous_gray, gray)
        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        height, width = mask.shape[:2]
        mask[: int(height * 0.18), :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 9)), iterations=1)
        return self._candidates_from_mask(
            cv2,
            frame,
            mask,
            source="motion",
            scale_options=(1.15, 1.45),
            max_area_ratio=0.09,
            score_bonus=0.08,
        )

    def _edge_candidates(self, cv2: Any, frame: Any) -> list[ROIResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 80, 180)
        mask = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        height = mask.shape[0]
        mask[: int(height * 0.18), :] = 0
        return self._candidates_from_mask(
            cv2,
            frame,
            mask,
            source="edge_contour",
            scale_options=(0.95, 1.2),
            max_area_ratio=0.08,
            score_bonus=0.03,
        )

    def _color_water_suppression_candidates(self, cv2: Any, frame: Any) -> list[ROIResult]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        water = ((hue >= 70) & (hue <= 112) & (saturation >= 25) & (value >= 35)).astype("uint8") * 255
        height, width = water.shape[:2]
        pool_top = int(height * 0.2)
        rows = (water > 0).mean(axis=1)
        for row in range(int(height * 0.18), height):
            if rows[row] > 0.45:
                pool_top = row
                break
        non_water_body = (((hue < 70) | (hue > 125)) & (saturation >= 30) & (value >= 25) & (value <= 210)).astype("uint8") * 255
        dark_body = ((value <= 115) & (saturation >= 20)).astype("uint8") * 255
        mask = cv2.bitwise_or(non_water_body, dark_body)
        mask[: max(0, pool_top - 5), :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 9)), iterations=1)
        return self._candidates_from_mask(
            cv2,
            frame,
            mask,
            source="water_suppressed_color",
            scale_options=(1.25, 1.55),
            max_area_ratio=0.06,
            score_bonus=0.04,
        )

    def _center_candidates(self, frame: Any) -> list[ROIResult]:
        height, width = frame.shape[:2]
        crop_w = max(float(self.config.min_size_px), width * self.config.center_crop_width_ratio)
        crop_h = max(float(self.config.min_size_px), height * self.config.center_crop_height_ratio)
        center_x = width * 0.5
        center_y = height * 0.49
        bbox = [center_x - crop_w / 2.0, center_y - crop_h / 2.0, center_x + crop_w / 2.0, center_y + crop_h / 2.0]
        return [_roi_from_bbox(frame, bbox, source="center_debug_crop", confidence=0.34, score_bonus=-0.04, config=self.config, scale=1.0)]

    def _candidates_from_mask(
        self,
        cv2: Any,
        frame: Any,
        mask: Any,
        *,
        source: str,
        scale_options: tuple[float, ...],
        max_area_ratio: float,
        score_bonus: float,
    ) -> list[ROIResult]:
        height, width = frame.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = max(1.0, float(width * height))
        min_area = max(120.0, frame_area * self.config.min_bbox_area_ratio)
        rois: list[ROIResult] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > frame_area * max_area_ratio:
                continue
            x, y, box_w, box_h = cv2.boundingRect(contour)
            bbox = [float(x), float(y), float(x + box_w), float(y + box_h)]
            if _reject_swimmer_bbox(bbox, width=width, height=height):
                continue
            for scale in scale_options:
                rois.append(_roi_from_bbox(frame, bbox, source=source, confidence=_heuristic_confidence(bbox, width, height, area), score_bonus=score_bonus, config=self.config, scale=scale))
        return rois

    def _detect_bbox(self, cv2: Any, frame: Any) -> tuple[list[float] | None, float, str | None]:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(gray, 50, 145)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(edges, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0, "roi_no_contour"

        frame_area = max(1.0, float(width * height))
        min_area = max(120.0, frame_area * self.config.min_bbox_area_ratio)
        candidates: list[tuple[float, float, int, int, int, int]] = []
        center_x = width / 2.0
        center_y = height / 2.0
        max_center_distance = (center_x**2 + center_y**2) ** 0.5 or 1.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if aspect <= 0 or aspect > self.config.max_aspect_ratio:
                continue
            candidate_center_x = x + w / 2.0
            candidate_center_y = y + h / 2.0
            center_penalty = ((candidate_center_x - center_x) ** 2 + (candidate_center_y - center_y) ** 2) ** 0.5 / max_center_distance
            score = area * (1.0 - min(0.45, center_penalty * 0.35))
            candidates.append((score, area, x, y, w, h))

        if not candidates:
            return None, 0.0, "roi_low_visibility"
        score, area, x, y, w, h = max(candidates, key=lambda item: item[0])
        area_ratio = area / frame_area
        confidence = max(0.12, min(0.76, 0.2 + area_ratio * 18.0))
        return [float(x), float(y), float(x + w), float(y + h)], confidence, None

    def _cv2(self) -> Any:
        if self.cv2 is not None:
            return self.cv2
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for ROI-centered swim analysis.") from exc
        self.cv2 = cv2
        return cv2


def centered_roi_from_bbox(
    frame: Any,
    bbox: list[float],
    *,
    confidence: float,
    config: ROIConfig,
    reason: str | None = None,
    source: str = "bbox",
    candidate_score: float | None = None,
) -> ROIResult:
    """Create a swimmer-centered crop and retain the offset needed to project pose results back."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if x2 <= x1 or y2 <= y1:
        return ROIResult(False, None, frame_width=width, frame_height=height, localization_bbox=bbox, reason="roi_invalid_bbox", source=source)
    if x2 < 0 or y2 < 0 or x1 > width or y1 > height:
        return ROIResult(False, None, frame_width=width, frame_height=height, localization_bbox=bbox, reason="roi_bbox_outside_frame", source=source)

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    crop_w = max(float(config.min_size_px), bbox_w * config.scale)
    crop_h = max(float(config.min_size_px), bbox_h * config.scale)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    raw_left = center_x - crop_w / 2.0
    raw_top = center_y - crop_h / 2.0
    raw_right = center_x + crop_w / 2.0
    raw_bottom = center_y + crop_h / 2.0

    left = int(max(0, round(raw_left)))
    top = int(max(0, round(raw_top)))
    right = int(min(width, round(raw_right)))
    bottom = int(min(height, round(raw_bottom)))
    requested_area = max(1.0, (raw_right - raw_left) * (raw_bottom - raw_top))
    clipped_area = max(0.0, (raw_right - raw_left) * (raw_bottom - raw_top) - max(0, right - left) * max(0, bottom - top))
    clip_ratio = clipped_area / requested_area
    boundary_ratio = _boundary_touch_ratio(left, top, right, bottom, width=width, height=height)
    score = _candidate_score([left, top, right, bottom], confidence=confidence, width=width, height=height) if candidate_score is None else candidate_score
    metrics = _bbox_metrics([left, top, right, bottom], width=width, height=height)
    metrics["clip_ratio"] = round(float(clip_ratio), 4)
    if right <= left or bottom <= top:
        return ROIResult(False, None, frame_width=width, frame_height=height, localization_bbox=bbox, reason="roi_empty_after_clamp", source=source, boundary_touch_ratio=boundary_ratio, metrics=metrics)
    if clip_ratio > config.max_boundary_clip_ratio:
        return ROIResult(
            False,
            None,
            left,
            top,
            right - left,
            bottom - top,
            width,
            height,
            bbox,
            confidence,
            f"roi_boundary_clip_ratio_{clip_ratio:.2f}",
            source,
            score,
            boundary_ratio,
            metrics,
        )

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return ROIResult(False, None, frame_width=width, frame_height=height, localization_bbox=bbox, reason="roi_empty_crop", source=source, candidate_score=score, boundary_touch_ratio=boundary_ratio, metrics=metrics)
    return ROIResult(True, crop, left, top, right - left, bottom - top, width, height, bbox, confidence, reason, source, score, boundary_ratio, metrics)


def _roi_from_bbox(
    frame: Any,
    bbox: list[float],
    *,
    source: str,
    confidence: float,
    score_bonus: float,
    config: ROIConfig,
    scale: float,
) -> ROIResult:
    temp_config = replace(config, scale=scale)
    height, width = frame.shape[:2]
    base_score = _candidate_score(bbox, confidence=confidence, width=width, height=height) + score_bonus
    roi = centered_roi_from_bbox(
        frame,
        bbox,
        confidence=confidence,
        config=temp_config,
        reason=source,
        source=source,
        candidate_score=max(0.0, min(1.0, base_score)),
    )
    return roi


def _reject_swimmer_bbox(bbox: list[float], *, width: int, height: int) -> bool:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if box_w <= 0 or box_h <= 0:
        return True
    aspect = box_w / max(1.0, box_h)
    area_ratio = (box_w * box_h) / max(1.0, float(width * height))
    if aspect > 9.0 or aspect < 0.45:
        return True
    if box_h < height * 0.025 or box_w < width * 0.025:
        return True
    if box_w > width * 0.82 or box_h > height * 0.62:
        return True
    if area_ratio > 0.1:
        return True
    if _boundary_touch_ratio(int(x1), int(y1), int(x2), int(y2), width=width, height=height) > 0.4:
        return True
    return False


def _heuristic_confidence(bbox: list[float], width: int, height: int, contour_area: float) -> float:
    metrics = _bbox_metrics(bbox, width=width, height=height)
    area_ratio = float(contour_area) / max(1.0, float(width * height))
    human_aspect_bonus = max(0.0, 1.0 - abs(float(metrics["aspect_ratio"]) - 2.6) / 4.8)
    size_bonus = max(0.0, 1.0 - abs(float(metrics["area_ratio"]) - 0.035) / 0.08)
    return max(0.12, min(0.78, 0.18 + area_ratio * 8.0 + human_aspect_bonus * 0.18 + size_bonus * 0.16))


def _candidate_score(bbox: list[float], *, confidence: float, width: int, height: int) -> float:
    metrics = _bbox_metrics(bbox, width=width, height=height)
    aspect = float(metrics["aspect_ratio"])
    area_ratio = float(metrics["area_ratio"])
    boundary = float(metrics["boundary_touch_ratio"])
    human_aspect = max(0.0, 1.0 - abs(aspect - 2.6) / 5.0)
    useful_size = max(0.0, 1.0 - abs(area_ratio - 0.055) / 0.11)
    horizontal = 1.0 if aspect >= 1.0 else 0.35
    lane_line_penalty = 0.36 if aspect > 7.5 or area_ratio < 0.002 else 0.0
    pool_boundary_penalty = 0.48 if boundary > 0.2 or area_ratio > 0.18 else 0.0
    score = confidence * 0.32 + human_aspect * 0.26 + useful_size * 0.24 + horizontal * 0.18 - lane_line_penalty - pool_boundary_penalty
    return max(0.0, min(1.0, score))


def _bbox_metrics(bbox: list[float], *, width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    return {
        "aspect_ratio": round(box_w / max(1.0, box_h), 4),
        "area_ratio": round((box_w * box_h) / max(1.0, float(width * height)), 6),
        "boundary_touch_ratio": round(_boundary_touch_ratio(int(x1), int(y1), int(x2), int(y2), width=width, height=height), 4),
        "center_x_ratio": round(((x1 + x2) / 2.0) / max(1.0, float(width)), 4),
        "center_y_ratio": round(((y1 + y2) / 2.0) / max(1.0, float(height)), 4),
    }


def _boundary_touch_ratio(left: int, top: int, right: int, bottom: int, *, width: int, height: int) -> float:
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    perimeter = 2.0 * (box_w + box_h)
    touched = 0.0
    tolerance = max(2, int(min(width, height) * 0.01))
    if left <= tolerance:
        touched += box_h
    if right >= width - tolerance:
        touched += box_h
    if top <= tolerance:
        touched += box_w
    if bottom >= height - tolerance:
        touched += box_w
    return max(0.0, min(1.0, touched / max(1.0, perimeter)))


def _dedupe_rois(rois: list[ROIResult]) -> list[ROIResult]:
    output: list[ROIResult] = []
    for roi in sorted(rois, key=lambda item: item.candidate_score, reverse=True):
        if not any(_roi_overlap_ratio(roi, existing) > 0.78 for existing in output):
            output.append(roi)
    return output


def _roi_overlap_ratio(first: ROIResult, second: ROIResult) -> float:
    ax1, ay1, ax2, ay2 = first.x, first.y, first.x + first.width, first.y + first.height
    bx1, by1, bx2, by2 = second.x, second.y, second.x + second.width, second.y + second.height
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    first_area = max(1, first.width * first.height)
    second_area = max(1, second.width * second.height)
    return intersection / max(1, min(first_area, second_area))


def _roi_center(roi: ROIResult) -> tuple[float, float]:
    return (float(roi.x) + float(roi.width) / 2.0, float(roi.y) + float(roi.height) / 2.0)


def _roi_diagonal(roi: ROIResult) -> float:
    return (float(roi.width) ** 2 + float(roi.height) ** 2) ** 0.5


def _frame_diagonal(roi: ROIResult) -> float:
    return (float(roi.frame_width) ** 2 + float(roi.frame_height) ** 2) ** 0.5


def skipped_pose_from_roi(
    *,
    frame_index: int,
    timestamp: float,
    view_type: str,
    quality_flags: list[str] | None,
    backend: str,
    roi: ROIResult,
) -> PoseFrameResult:
    flags = [*(quality_flags or []), "roi_skipped"]
    if roi.reason:
        flags.append(str(roi.reason))
    result = empty_pose_result(frame_index=frame_index, timestamp=timestamp, view_type=view_type, quality_flags=flags, backend=backend)
    return replace(result, skipped=True, roi=roi.to_dict(), frame_quality_score=_frame_quality_score(flags), debug_events=[f"ROI skipped: {roi.reason}"])


def project_pose_to_original(pose: PoseFrameResult, roi: ROIResult) -> PoseFrameResult:
    if not roi.valid:
        return replace(pose, skipped=True, roi=roi.to_dict())

    projected_points = [
        replace(
            point,
            x=float(point.x) + roi.x,
            y=float(point.y) + roi.y,
            source_backend=point.source_backend or pose.backend,
            frame_index=pose.frame_index,
            visibility_state=_visibility_for_confidence(point.confidence),
        )
        for point in pose.keypoints
    ]
    projected_raw = None
    if pose.raw_keypoints is not None:
        projected_raw = [
            replace(point, x=float(point.x) + roi.x, y=float(point.y) + roi.y, source_backend=point.source_backend or pose.backend, frame_index=pose.frame_index)
            for point in pose.raw_keypoints
        ]

    bbox = _project_bbox(pose.bbox, roi) or roi.localization_bbox or bbox_from_keypoints(projected_points)
    flags = list(pose.quality_flags)
    if roi.reason:
        flags.append(str(roi.reason))
    debug_info = dict(pose.debug_info)
    raw_by_index = debug_info.get("raw_keypoints_by_index")
    if isinstance(raw_by_index, list):
        projected_debug = []
        for item in raw_by_index:
            if not isinstance(item, dict):
                projected_debug.append(item)
                continue
            next_item = dict(item)
            if next_item.get("x") is not None:
                next_item["x"] = float(next_item["x"]) + roi.x
            if next_item.get("y") is not None:
                next_item["y"] = float(next_item["y"]) + roi.y
            projected_debug.append(next_item)
        debug_info["raw_keypoints_by_index"] = projected_debug
    return replace(
        pose,
        keypoints=projected_points,
        raw_keypoints=projected_raw,
        bbox=bbox,
        confidence=average_keypoint_confidence(projected_points) if projected_points else pose.confidence,
        roi=roi.to_dict(),
        frame_quality_score=_frame_quality_score(flags),
        quality_flags=flags,
        debug_info=debug_info,
    )


def stamp_pose_metadata(frame: PoseFrameResult) -> PoseFrameResult:
    points = [
        replace(
            point,
            source_backend=point.source_backend or frame.backend,
            frame_index=frame.frame_index,
            visibility_state=_visibility_for_confidence(point.confidence),
        )
        for point in frame.keypoints
    ]
    return replace(frame, keypoints=points, frame_quality_score=_frame_quality_score(frame.quality_flags))


def _project_bbox(bbox: list[float] | None, roi: ROIResult) -> list[float] | None:
    if bbox is None:
        return None
    return [float(bbox[0]) + roi.x, float(bbox[1]) + roi.y, float(bbox[2]) + roi.x, float(bbox[3]) + roi.y]


def _visibility_for_confidence(confidence: float) -> str:
    if confidence >= 0.25:
        return "visible"
    if confidence > 0:
        return "low_confidence"
    return "missing"


def _frame_quality_score(flags: list[str] | None) -> float:
    if not flags:
        return 1.0
    penalties = {
        "dark": 0.22,
        "blurry": 0.2,
        "overexposed": 0.18,
        "low_visibility": 0.18,
        "roi_skipped": 0.45,
        "tracking_jump_rejected": 0.35,
    }
    score = 1.0
    for flag in flags:
        score -= next((value for key, value in penalties.items() if key in flag), 0.05)
    return max(0.0, min(1.0, score))
