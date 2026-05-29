from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SwimAnalysisUploadResponse(BaseModel):
    analysis_id: str
    status: str = "queued"


class SwimAnalysisStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(serialization_alias="analysis_id")
    status: str
    progress: int
    stroke_type: str
    pose_backend: str
    swimmer_id: str | None
    video_metadata: dict[str, Any] = Field(default_factory=dict)
    processing_errors: list[str] = Field(default_factory=list)
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class SwimAnalysisReportResponse(BaseModel):
    analysis_id: str
    status: str
    summary: dict[str, Any]
    video_quality: dict[str, Any]
    video_metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any]
    calibration: dict[str, Any] = Field(default_factory=dict)
    velocity: dict[str, Any] = Field(default_factory=dict)
    phase_segments: list[dict[str, Any]] = Field(default_factory=list)
    faults: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    coaching_report: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    pose_diagnostics: dict[str, Any] = Field(default_factory=dict)
    trajectories: dict[str, Any] = Field(default_factory=dict)
    research_figures: list[dict[str, Any]] = Field(default_factory=list)
    debug_info: dict[str, Any] = Field(default_factory=dict)
    processing_errors: list[str] = Field(default_factory=list)


class SwimAnalysisMetricsResponse(BaseModel):
    analysis_id: str
    status: str
    metrics: dict[str, Any]
    faults: list[dict[str, Any]] = Field(default_factory=list)
    calibration: dict[str, Any] = Field(default_factory=dict)


class SwimAnalysisVelocityResponse(BaseModel):
    analysis_id: str
    status: str
    velocity: dict[str, Any]
    phase_segments: list[dict[str, Any]] = Field(default_factory=list)
    calibration: dict[str, Any] = Field(default_factory=dict)


class SwimAnalysisAnnotatedVideoResponse(BaseModel):
    analysis_id: str
    status: str
    videos: dict[str, str | None]
    confidence_warning: str | None = None
