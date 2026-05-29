from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserRead(ORMModel):
    id: str
    email: str
    full_name: str
    role: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class LoginRequest(BaseModel):
    email: str
    password: str


class SwimmerBase(BaseModel):
    name: str = Field(min_length=2, max_length=255)
    date_of_birth: date | None = None
    primary_stroke: str
    primary_event: str
    level: str = "age_group"
    personal_bests: dict[str, float] = Field(default_factory=dict)


class SwimmerCreate(SwimmerBase):
    pass


class SwimmerUpdate(BaseModel):
    name: str | None = None
    date_of_birth: date | None = None
    primary_stroke: str | None = None
    primary_event: str | None = None
    level: str | None = None
    personal_bests: dict[str, float] | None = None
    technique_profile: dict[str, Any] | None = None
    mental_profile: dict[str, Any] | None = None


class SwimmerRead(SwimmerBase, ORMModel):
    id: str
    coach_id: str
    technique_profile: dict[str, Any]
    mental_profile: dict[str, Any]
    created_at: datetime


class SessionBase(BaseModel):
    swimmer_id: str
    session_date: date
    session_type: str
    distance_m: int = Field(gt=0)
    duration_min: int = Field(gt=0)
    rpe: int = Field(ge=1, le=10)
    mood_focus: int = Field(ge=1, le=10)
    mood_confidence: int = Field(ge=1, le=10)
    mood_energy: int = Field(ge=1, le=10)
    mood_calm: int = Field(ge=1, le=10)
    mood_recovery: int = Field(ge=1, le=10)
    mood_motivation: int = Field(ge=1, le=10)
    sleep_hours: float | None = Field(default=None, ge=0, le=16)
    hrv_morning: int | None = Field(default=None, ge=20, le=220)
    notes: str | None = None


class SessionCreate(SessionBase):
    pass


class SessionRead(SessionBase, ORMModel):
    id: str
    load_score: float
    video_url: str | None
    created_at: datetime


class UploadedFileRead(ORMModel):
    id: str
    session_id: str
    swimmer_id: str
    original_filename: str
    stored_path: str
    content_type: str | None
    size_bytes: int
    created_at: datetime


class AnalysisJobRead(ORMModel):
    id: str
    session_id: str
    swimmer_id: str
    uploaded_file_id: str
    job_type: str
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class TechniqueReportRead(ORMModel):
    id: str
    session_id: str
    swimmer_id: str
    stroke: str
    overall_score: int
    dps_meters: float
    stroke_rate: int
    faults: list[dict[str, Any]]
    drill_prescriptions: list[dict[str, Any]]
    keypoint_data: dict[str, Any]
    processing_status: str
    analysis_status: str
    frames_total: int
    frames_analyzed: int
    pose_detected_frames: int
    pose_detection_rate: float
    confidence_score: float
    confidence_label: str
    analysis_warning: str | None
    analysis_error: str | None
    analysis_overlay_video_url: str | None = None
    analysis_frame_urls: list[str] = Field(default_factory=list)
    analysis_events: list[dict[str, Any]] = Field(default_factory=list)
    coaching_summary: str
    created_at: datetime


class VideoUploadResponse(BaseModel):
    file: UploadedFileRead
    job: AnalysisJobRead
    report: TechniqueReportRead


class TrainingPlanGenerate(BaseModel):
    swimmer_id: str
    race_date: date
    race_event: str
    target_time_seconds: float = Field(gt=0)


class TrainingPlanRead(ORMModel):
    id: str
    swimmer_id: str
    race_date: date
    race_event: str
    target_time_seconds: float
    weeks_total: int
    current_phase: str
    current_week: int
    phase_config: dict[str, Any]
    weekly_plans: list[dict[str, Any]]
    adaptation_log: list[dict[str, Any]]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class RaceAnalysisCreate(BaseModel):
    swimmer_id: str
    race_date: date
    event: str
    official_time_seconds: float = Field(gt=0)
    splits_actual: list[float] = Field(min_length=1)
    reaction_time_ms: int | None = Field(default=None, ge=0)
    turn_times: list[dict[str, Any]] = Field(default_factory=list)


class RaceAnalysisRead(ORMModel):
    id: str
    swimmer_id: str
    race_date: date
    event: str
    official_time_seconds: float
    splits_actual: list[float]
    splits_predicted: list[float]
    strategy_type: str
    strategy_score: int
    reaction_time_ms: int | None
    turn_times: list[dict[str, Any]]
    phase_analysis: dict[str, Any]
    time_vs_pb_seconds: float | None
    ai_insights: list[str]
    created_at: datetime


class MentalCheckinCreate(BaseModel):
    swimmer_id: str
    checkin_type: str
    checkin_date: datetime | None = None
    mood_focus: int = Field(ge=1, le=10)
    mood_confidence: int = Field(ge=1, le=10)
    mood_energy: int = Field(ge=1, le=10)
    mood_calm: int = Field(ge=1, le=10)
    mood_recovery: int = Field(ge=1, le=10)
    mood_motivation: int = Field(ge=1, le=10)
    race_result_id: str | None = None
    performance_delta: float | None = None
    notes: str | None = None


class MentalCheckinRead(ORMModel):
    id: str
    swimmer_id: str
    checkin_type: str
    checkin_date: datetime
    mood_focus: int
    mood_confidence: int
    mood_energy: int
    mood_calm: int
    mood_recovery: int
    mood_motivation: int
    composite_score: float
    routine_generated: dict[str, Any]
    race_result_id: str | None
    performance_delta: float | None
    notes: str | None


class AIOutputRead(ORMModel):
    id: str
    swimmer_id: str
    session_id: str | None
    technique_report_id: str | None
    training_plan_id: str | None
    race_analysis_id: str | None
    mental_checkin_id: str | None
    output_type: str
    provider: str
    model: str
    prompt_version: str
    parsed_json: dict[str, Any]
    status: str
    error: str | None
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    created_at: datetime


class DashboardOverview(BaseModel):
    swimmers_count: int
    sessions_count: int
    average_load: float
    high_risk_count: int
    swimmers: list[SwimmerRead]
    recent_sessions: list[SessionRead]
    active_plans: list[TrainingPlanRead]
    alerts: list[dict[str, Any]]


class PilotBoardCard(BaseModel):
    swimmer: SwimmerRead
    latest_session: SessionRead | None
    latest_report: TechniqueReportRead | None
    latest_checkin: MentalCheckinRead | None
    latest_race: RaceAnalysisRead | None
    missing_flags: list[str]
    next_action: str
    status: str


class PilotBoardResponse(BaseModel):
    generated_at: datetime
    swimmers: list[PilotBoardCard]
    totals: dict[str, int]


class PilotSessionEntryCreate(BaseModel):
    swimmer_id: str
    session_date: date
    session_type: str
    distance_m: int = Field(gt=0)
    duration_min: int = Field(gt=0)
    rpe: int = Field(ge=1, le=10)
    mood_focus: int = Field(default=7, ge=1, le=10)
    mood_confidence: int = Field(default=7, ge=1, le=10)
    mood_energy: int = Field(default=7, ge=1, le=10)
    mood_calm: int = Field(default=7, ge=1, le=10)
    mood_recovery: int = Field(default=7, ge=1, le=10)
    mood_motivation: int = Field(default=7, ge=1, le=10)
    sleep_hours: float | None = Field(default=None, ge=0, le=16)
    hrv_morning: int | None = Field(default=None, ge=20, le=220)
    notes: str | None = None
    create_mental_checkin: bool = False
    checkin_type: str = "post_session"
    checkin_notes: str | None = None


class PilotSessionEntryResponse(BaseModel):
    session: SessionRead
    mental_checkin: MentalCheckinRead | None
    duplicate_warning: str | None
    entry_summary: str
