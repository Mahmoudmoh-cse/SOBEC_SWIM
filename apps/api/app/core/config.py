from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AquaIQ API"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://aquaiq:aquaiq@localhost:5432/aquaiq"
    secret_key: str = "change-this-before-production"
    access_token_expire_minutes: int = 60 * 24
    upload_dir: Path = Path("uploads")
    ai_provider: str = "local"
    technique_analyzer: str = "auto"
    pose_backend: str = "rtmpose"
    fallback_pose_backend: str | None = "mediapipe"
    pose_allow_fallback: bool = False
    pose_device: str = "cpu"
    pose_confidence_threshold: float = 0.05
    rtmpose_config_path: str | None = None
    rtmpose_checkpoint_path: str | None = None
    rtmpose_keypoint_names: str | None = None
    yolo_pose_model: str = "yolo11x-pose.pt"
    swim_roi_enabled: bool = True
    swim_roi_disable: bool = False
    swim_roi_compare_full_frame: bool = True
    swim_roi_fallback_full_frame: bool = True
    swim_roi_rtmpose_assisted: bool = True
    swim_roi_max_candidates: int = 5
    swim_roi_scale: float = 1.85
    swim_roi_min_size_px: int = 96
    swim_roi_max_boundary_clip_ratio: float = 0.35
    swim_temporal_max_gap_frames: int = 5
    swim_temporal_visible_confidence: float = 0.25
    swim_temporal_low_confidence: float = 0.12
    swim_temporal_smoothing_min_confidence: float = 0.12
    swim_temporal_smoothing_min_cutoff: float = 1.0
    swim_temporal_smoothing_beta: float = 0.035
    swim_temporal_smoothing_d_cutoff: float = 1.0
    swim_temporal_smoothing_ema_alpha_low: float = 0.32
    swim_temporal_smoothing_ema_alpha_high: float = 0.88
    swim_temporal_biomechanical_constraints: bool = True
    swim_annotation_show_raw_pose: bool = False
    swim_annotation_show_debug: bool = True
    mediapipe_model_complexity: int = 1
    mediapipe_min_detection_confidence: float = 0.5
    mediapipe_min_tracking_confidence: float = 0.5
    anthropic_api_key: str | None = None
    claude_model_heavy: str = "claude-sonnet-4-5"
    claude_model_fast: str = "claude-3-5-haiku-latest"
    backend_cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
