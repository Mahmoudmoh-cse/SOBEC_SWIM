from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from statistics import mean
from typing import Any, Iterator


ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".avi"}


@dataclass
class VideoMetadata:
    path: str
    fps: float
    duration_sec: float
    resolution: dict[str, int]
    frame_count: int
    valid: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "fps": round(self.fps, 3),
            "duration_sec": round(self.duration_sec, 3),
            "resolution": self.resolution,
            "frame_count": self.frame_count,
            "valid": self.valid,
            "error": self.error,
        }


@dataclass
class VideoQualityReport:
    blur_score: float
    lighting_score: float
    stability_score: float
    usable_frame_ratio: float
    dark_frame_ratio: float
    blurry_frame_ratio: float
    low_visibility_frame_ratio: float
    sampled_frames: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blur_score": round(self.blur_score, 3),
            "lighting_score": round(self.lighting_score, 3),
            "stability_score": round(self.stability_score, 3),
            "usable_frame_ratio": round(self.usable_frame_ratio, 3),
            "dark_frame_ratio": round(self.dark_frame_ratio, 3),
            "blurry_frame_ratio": round(self.blurry_frame_ratio, 3),
            "low_visibility_frame_ratio": round(self.low_visibility_frame_ratio, 3),
            "sampled_frames": self.sampled_frames,
            "warnings": self.warnings,
        }


def validate_video_file(path: str | Path) -> None:
    video_path = Path(path)
    if not video_path.exists() or not video_path.is_file():
        raise ValueError("Video file was not saved correctly.")
    if video_path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise ValueError("Upload must be a phone video file: MP4, MOV, M4V, WEBM, or AVI.")
    if video_path.stat().st_size <= 0:
        raise ValueError("Uploaded video is empty.")


def read_video_metadata(path: str | Path) -> VideoMetadata:
    cv2 = _cv2()
    video_path = Path(path)
    try:
        validate_video_file(video_path)
    except ValueError as exc:
        return VideoMetadata(str(video_path), 0, 0, {"width": 0, "height": 0}, 0, False, str(exc))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoMetadata(str(video_path), 0, 0, {"width": 0, "height": 0}, 0, False, "OpenCV could not open the video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 and frame_count else 0
    return VideoMetadata(
        path=str(video_path),
        fps=fps or 30.0,
        duration_sec=duration,
        resolution={"width": width, "height": height},
        frame_count=frame_count,
        valid=width > 0 and height > 0 and frame_count > 0,
        error=None if width > 0 and height > 0 and frame_count > 0 else "Video contains no readable frames.",
    )


def sample_video_frames(
    path: str | Path,
    *,
    target_fps: int = 30,
    view_type: str,
    max_dimension: int = 1280,
    enhance: bool = True,
) -> list[tuple[int, float, Any, list[str], str]]:
    return list(
        iter_sample_video_frames(
            path,
            target_fps=target_fps,
            view_type=view_type,
            max_dimension=max_dimension,
            enhance=enhance,
        )
    )


def iter_sample_video_frames(
    path: str | Path,
    *,
    target_fps: int = 30,
    view_type: str,
    max_dimension: int = 1280,
    enhance: bool = True,
) -> Iterator[tuple[int, float, Any, list[str], str]]:
    cv2 = _cv2()
    metadata = read_video_metadata(path)
    if not metadata.valid:
        raise ValueError(metadata.error or "Invalid video file.")
    cap = cv2.VideoCapture(str(path))
    source_fps = metadata.fps or 30.0
    sample_every = max(1, ceil(source_fps / max(1, target_fps)))
    frame_index = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % sample_every == 0:
                processed = standardize_frame(frame, max_dimension=max_dimension, enhance=enhance)
                flags = frame_quality_flags(processed)
                yield frame_index, frame_index / source_fps, processed, flags, view_type
            frame_index += 1
    finally:
        cap.release()


def iter_video_frames(path: str | Path) -> Iterator[tuple[int, float, Any]]:
    cv2 = _cv2()
    metadata = read_video_metadata(path)
    if not metadata.valid:
        return
    cap = cv2.VideoCapture(str(path))
    fps = metadata.fps or 30.0
    frame_index = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_index, frame_index / fps, frame
            frame_index += 1
    finally:
        cap.release()


def standardize_frame(frame: Any, *, max_dimension: int = 1280, enhance: bool = True) -> Any:
    cv2 = _cv2()
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    if enhance:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        frame = cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    return frame


def assess_video_quality(path: str | Path, *, target_samples: int = 90) -> VideoQualityReport:
    cv2 = _cv2()
    metadata = read_video_metadata(path)
    if not metadata.valid:
        return VideoQualityReport(0, 0, 0, 0, 1, 1, 1, 0, [metadata.error or "Invalid video file."])

    cap = cv2.VideoCapture(str(path))
    sample_every = max(1, metadata.frame_count // max(1, target_samples))
    frame_index = 0
    blur_values: list[float] = []
    lighting_values: list[float] = []
    contrast_values: list[float] = []
    edge_centers: list[tuple[float, float]] = []
    bad_counts = {"dark": 0, "blurry": 0, "low_visibility": 0}

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            quality_frame = _resize_for_quality(frame)
            gray = cv2.cvtColor(quality_frame, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            brightness = float(gray.mean())
            contrast = float(gray.std())
            blur_values.append(_normalize(blur, 35, 260))
            lighting_values.append(_lighting_score(brightness))
            contrast_values.append(_normalize(contrast, 18, 68))
            flags = frame_quality_flags(frame)
            for flag in bad_counts:
                if flag in flags:
                    bad_counts[flag] += 1
            edge_centers.append(_edge_center(gray))
        frame_index += 1
    cap.release()

    sampled = len(blur_values)
    if sampled == 0:
        return VideoQualityReport(0, 0, 0, 0, 1, 1, 1, 0, ["No readable frames were sampled."])

    bad_frames = sum(1 for idx in range(sampled) if _frame_bad(idx, blur_values, lighting_values, contrast_values))
    stability = _stability_score(edge_centers)
    warnings = []
    if mean(blur_values) < 0.45:
        warnings.append("Many frames are soft or blurry; mount the phone or increase shutter speed if possible.")
    if mean(lighting_values) < 0.55:
        warnings.append("Lighting is weak; avoid backlit lanes and dark pool corners.")
    if stability < 0.55:
        warnings.append("Camera movement is high; use a fixed deck-level position.")

    return VideoQualityReport(
        blur_score=mean(blur_values),
        lighting_score=mean(lighting_values),
        stability_score=stability,
        usable_frame_ratio=max(0.0, 1.0 - bad_frames / sampled),
        dark_frame_ratio=bad_counts["dark"] / sampled,
        blurry_frame_ratio=bad_counts["blurry"] / sampled,
        low_visibility_frame_ratio=bad_counts["low_visibility"] / sampled,
        sampled_frames=sampled,
        warnings=warnings,
    )


def frame_quality_flags(frame: Any) -> list[str]:
    cv2 = _cv2()
    quality_frame = _resize_for_quality(frame)
    gray = cv2.cvtColor(quality_frame, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    flags = []
    if blur < 35:
        flags.append("blurry")
    if brightness < 45:
        flags.append("dark")
    if brightness > 225:
        flags.append("overexposed")
    if contrast < 18:
        flags.append("low_visibility")
    return flags


def _frame_bad(index: int, blur_values: list[float], lighting_values: list[float], contrast_values: list[float]) -> bool:
    return blur_values[index] < 0.25 or lighting_values[index] < 0.25 or contrast_values[index] < 0.25


def _resize_for_quality(frame: Any, *, max_dimension: int = 640) -> Any:
    cv2 = _cv2()
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def _edge_center(gray: Any) -> tuple[float, float]:
    cv2 = _cv2()
    moments = cv2.moments(cv2.Canny(gray, 70, 160))
    if moments["m00"] == 0:
        height, width = gray.shape[:2]
        return width / 2, height / 2
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _stability_score(centers: list[tuple[float, float]]) -> float:
    if len(centers) < 3:
        return 1.0
    xs = [point[0] for point in centers]
    ys = [point[1] for point in centers]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return max(0.0, min(1.0, 1.0 - span / 160.0))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _lighting_score(brightness: float) -> float:
    if 70 <= brightness <= 180:
        return 1.0
    if brightness < 70:
        return _normalize(brightness, 25, 70)
    return max(0.0, 1.0 - (brightness - 180) / 70)


def _cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video preprocessing. Install opencv-python or opencv-python-headless.") from exc
    return cv2
