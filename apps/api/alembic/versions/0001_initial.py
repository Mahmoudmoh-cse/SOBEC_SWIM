"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-20
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "swimmers",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("coach_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("primary_stroke", sa.String(length=50), nullable=False),
        sa.Column("primary_event", sa.String(length=100), nullable=False),
        sa.Column("level", sa.String(length=50), nullable=False),
        sa.Column("personal_bests", sa.JSON(), nullable=False),
        sa.Column("technique_profile", sa.JSON(), nullable=False),
        sa.Column("mental_profile", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["coach_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_swimmers_coach_id"), "swimmers", ["coach_id"], unique=False)

    op.create_table(
        "race_analyses",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("swimmer_id", sa.String(length=36), nullable=False),
        sa.Column("race_date", sa.Date(), nullable=False),
        sa.Column("event", sa.String(length=100), nullable=False),
        sa.Column("official_time_seconds", sa.Float(), nullable=False),
        sa.Column("splits_actual", sa.JSON(), nullable=False),
        sa.Column("splits_predicted", sa.JSON(), nullable=False),
        sa.Column("strategy_type", sa.String(length=50), nullable=False),
        sa.Column("strategy_score", sa.Integer(), nullable=False),
        sa.Column("reaction_time_ms", sa.Integer(), nullable=True),
        sa.Column("turn_times", sa.JSON(), nullable=False),
        sa.Column("phase_analysis", sa.JSON(), nullable=False),
        sa.Column("time_vs_pb_seconds", sa.Float(), nullable=True),
        sa.Column("ai_insights", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["swimmer_id"], ["swimmers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_race_analyses_swimmer_id"), "race_analyses", ["swimmer_id"], unique=False)

    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("swimmer_id", sa.String(length=36), nullable=False),
        sa.Column("session_date", sa.Date(), nullable=False),
        sa.Column("session_type", sa.String(length=50), nullable=False),
        sa.Column("distance_m", sa.Integer(), nullable=False),
        sa.Column("duration_min", sa.Integer(), nullable=False),
        sa.Column("rpe", sa.Integer(), nullable=False),
        sa.Column("load_score", sa.Float(), nullable=False),
        sa.Column("mood_focus", sa.Integer(), nullable=False),
        sa.Column("mood_confidence", sa.Integer(), nullable=False),
        sa.Column("mood_energy", sa.Integer(), nullable=False),
        sa.Column("mood_calm", sa.Integer(), nullable=False),
        sa.Column("mood_recovery", sa.Integer(), nullable=False),
        sa.Column("mood_motivation", sa.Integer(), nullable=False),
        sa.Column("sleep_hours", sa.Float(), nullable=True),
        sa.Column("hrv_morning", sa.Integer(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("video_url", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["swimmer_id"], ["swimmers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_sessions_swimmer_id"), "sessions", ["swimmer_id"], unique=False)

    op.create_table(
        "training_plans",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("swimmer_id", sa.String(length=36), nullable=False),
        sa.Column("race_date", sa.Date(), nullable=False),
        sa.Column("race_event", sa.String(length=100), nullable=False),
        sa.Column("target_time_seconds", sa.Float(), nullable=False),
        sa.Column("weeks_total", sa.Integer(), nullable=False),
        sa.Column("current_phase", sa.String(length=50), nullable=False),
        sa.Column("current_week", sa.Integer(), nullable=False),
        sa.Column("phase_config", sa.JSON(), nullable=False),
        sa.Column("weekly_plans", sa.JSON(), nullable=False),
        sa.Column("adaptation_log", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["swimmer_id"], ["swimmers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_training_plans_swimmer_id"), "training_plans", ["swimmer_id"], unique=False)

    op.create_table(
        "mental_checkins",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("swimmer_id", sa.String(length=36), nullable=False),
        sa.Column("checkin_type", sa.String(length=50), nullable=False),
        sa.Column("checkin_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mood_focus", sa.Integer(), nullable=False),
        sa.Column("mood_confidence", sa.Integer(), nullable=False),
        sa.Column("mood_energy", sa.Integer(), nullable=False),
        sa.Column("mood_calm", sa.Integer(), nullable=False),
        sa.Column("mood_recovery", sa.Integer(), nullable=False),
        sa.Column("mood_motivation", sa.Integer(), nullable=False),
        sa.Column("composite_score", sa.Float(), nullable=False),
        sa.Column("routine_generated", sa.JSON(), nullable=False),
        sa.Column("race_result_id", sa.String(length=36), nullable=True),
        sa.Column("performance_delta", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["race_result_id"], ["race_analyses.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["swimmer_id"], ["swimmers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_mental_checkins_swimmer_id"), "mental_checkins", ["swimmer_id"], unique=False)

    op.create_table(
        "uploaded_files",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("swimmer_id", sa.String(length=36), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("stored_path", sa.String(length=500), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["swimmer_id"], ["swimmers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_uploaded_files_session_id"), "uploaded_files", ["session_id"], unique=False)
    op.create_index(op.f("ix_uploaded_files_swimmer_id"), "uploaded_files", ["swimmer_id"], unique=False)

    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("swimmer_id", sa.String(length=36), nullable=False),
        sa.Column("uploaded_file_id", sa.String(length=36), nullable=False),
        sa.Column("job_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["swimmer_id"], ["swimmers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["uploaded_file_id"], ["uploaded_files.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_analysis_jobs_session_id"), "analysis_jobs", ["session_id"], unique=False)
    op.create_index(op.f("ix_analysis_jobs_swimmer_id"), "analysis_jobs", ["swimmer_id"], unique=False)

    op.create_table(
        "technique_reports",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("swimmer_id", sa.String(length=36), nullable=False),
        sa.Column("stroke", sa.String(length=50), nullable=False),
        sa.Column("overall_score", sa.Integer(), nullable=False),
        sa.Column("dps_meters", sa.Float(), nullable=False),
        sa.Column("stroke_rate", sa.Integer(), nullable=False),
        sa.Column("faults", sa.JSON(), nullable=False),
        sa.Column("drill_prescriptions", sa.JSON(), nullable=False),
        sa.Column("keypoint_data", sa.JSON(), nullable=False),
        sa.Column("processing_status", sa.String(length=50), nullable=False),
        sa.Column("coaching_summary", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["swimmer_id"], ["swimmers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_technique_reports_session_id"), "technique_reports", ["session_id"], unique=False)
    op.create_index(op.f("ix_technique_reports_swimmer_id"), "technique_reports", ["swimmer_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_technique_reports_swimmer_id"), table_name="technique_reports")
    op.drop_index(op.f("ix_technique_reports_session_id"), table_name="technique_reports")
    op.drop_table("technique_reports")
    op.drop_index(op.f("ix_analysis_jobs_swimmer_id"), table_name="analysis_jobs")
    op.drop_index(op.f("ix_analysis_jobs_session_id"), table_name="analysis_jobs")
    op.drop_table("analysis_jobs")
    op.drop_index(op.f("ix_uploaded_files_swimmer_id"), table_name="uploaded_files")
    op.drop_index(op.f("ix_uploaded_files_session_id"), table_name="uploaded_files")
    op.drop_table("uploaded_files")
    op.drop_index(op.f("ix_mental_checkins_swimmer_id"), table_name="mental_checkins")
    op.drop_table("mental_checkins")
    op.drop_index(op.f("ix_training_plans_swimmer_id"), table_name="training_plans")
    op.drop_table("training_plans")
    op.drop_index(op.f("ix_sessions_swimmer_id"), table_name="sessions")
    op.drop_table("sessions")
    op.drop_index(op.f("ix_race_analyses_swimmer_id"), table_name="race_analyses")
    op.drop_table("race_analyses")
    op.drop_index(op.f("ix_swimmers_coach_id"), table_name="swimmers")
    op.drop_table("swimmers")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
