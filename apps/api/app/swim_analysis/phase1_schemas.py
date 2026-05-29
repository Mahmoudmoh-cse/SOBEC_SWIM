from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


VisibilityState = Literal["visible", "low_confidence", "missing", "interpolated", "rejected_outlier", "skipped"]


class TrajectoryPoint(BaseModel):
    """Single confidence-aware joint sample in original-frame coordinates."""

    frame_index: int
    timestamp: float
    joint: str
    x: float | None = None
    y: float | None = None
    confidence: float = 0.0
    visibility_state: VisibilityState = "missing"
    source_backend: str = ""
    flags: list[str] = Field(default_factory=list)


class RejectedOutlier(BaseModel):
    frame_index: int
    timestamp: float
    joint: str
    x: float | None = None
    y: float | None = None
    confidence: float = 0.0
    reason: str
    displacement_px: float | None = None
    velocity_px_s: float | None = None
    acceleration_px_s2: float | None = None


class SwapCorrectionEvent(BaseModel):
    frame_index: int
    timestamp: float
    pair: str
    direct_cost: float
    swapped_cost: float
    reason: str


class ResearchFigureMetadata(BaseModel):
    figure_id: str
    title: str
    type: str
    file_path: str
    description: str
    metric_source: Literal["ground_truth", "proxy"] = "proxy"


class TrajectoryExport(BaseModel):
    view_type: str
    raw_keypoints: dict[str, list[TrajectoryPoint]]
    corrected_keypoints: dict[str, list[TrajectoryPoint]] = Field(default_factory=dict)
    filtered_keypoints: dict[str, list[TrajectoryPoint]]
    interpolated_keypoints: dict[str, list[TrajectoryPoint]]
    rejected_outliers: list[RejectedOutlier] = Field(default_factory=list)
    swap_events: list[SwapCorrectionEvent] = Field(default_factory=list)
    confidence_timeline: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    visibility_percentage: dict[str, float] = Field(default_factory=dict)
    tracking_quality_timeline: list[dict[str, Any]] = Field(default_factory=list)
    frame_quality_timeline: list[dict[str, Any]] = Field(default_factory=list)
    joint_reliability: dict[str, dict[str, Any]] = Field(default_factory=dict)
    smoothing_diagnostics: dict[str, Any] = Field(default_factory=dict)
    center_trajectory: list[dict[str, Any]] = Field(default_factory=list)
    roi_history: list[dict[str, Any]] = Field(default_factory=list)
    dropped_frame_diagnostics: dict[str, Any] = Field(default_factory=dict)
    processing_warnings: list[str] = Field(default_factory=list)


class Phase1ArtifactSummary(BaseModel):
    trajectory_data_url: str | None = None
    research_figures: list[ResearchFigureMetadata] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    reliability_score: float = 0.0
