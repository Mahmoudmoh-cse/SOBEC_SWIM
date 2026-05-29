"""add swim performance engine outputs

Revision ID: 0006_swim_performance_engine
Revises: 0005_swim_analyses
Create Date: 2026-05-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006_swim_performance_engine"
down_revision: Union[str, None] = "0005_swim_analyses"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "swim_metrics",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("metric_name", sa.String(length=100), nullable=False),
        sa.Column("value_json", sa.JSON(), nullable=False),
        sa.Column("unit", sa.String(length=50), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("interpretation", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["analysis_id"], ["swim_analyses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_swim_metrics_analysis_id"), "swim_metrics", ["analysis_id"], unique=False)
    op.create_index(op.f("ix_swim_metrics_metric_name"), "swim_metrics", ["metric_name"], unique=False)

    op.create_table(
        "swim_faults",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("fault_name", sa.String(length=100), nullable=False),
        sa.Column("severity", sa.String(length=50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("timestamp_range", sa.JSON(), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("recommended_drill", sa.String(length=255), nullable=True),
        sa.Column("coach_explanation", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["analysis_id"], ["swim_analyses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_swim_faults_analysis_id"), "swim_faults", ["analysis_id"], unique=False)
    op.create_index(op.f("ix_swim_faults_fault_name"), "swim_faults", ["fault_name"], unique=False)

    op.create_table(
        "velocity_series",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("series_json", sa.JSON(), nullable=False),
        sa.Column("summary_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["analysis_id"], ["swim_analyses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_velocity_series_analysis_id"), "velocity_series", ["analysis_id"], unique=False)

    op.create_table(
        "phase_segments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("segment_type", sa.String(length=100), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("segment_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["analysis_id"], ["swim_analyses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_phase_segments_analysis_id"), "phase_segments", ["analysis_id"], unique=False)
    op.create_index(op.f("ix_phase_segments_segment_type"), "phase_segments", ["segment_type"], unique=False)

    op.create_table(
        "calibration_data",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("method", sa.String(length=100), nullable=False),
        sa.Column("lane_length_m", sa.Float(), nullable=True),
        sa.Column("pixel_to_meter", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("calibration_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["analysis_id"], ["swim_analyses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("analysis_id"),
    )
    op.create_index(op.f("ix_calibration_data_analysis_id"), "calibration_data", ["analysis_id"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_calibration_data_analysis_id"), table_name="calibration_data")
    op.drop_table("calibration_data")
    op.drop_index(op.f("ix_phase_segments_segment_type"), table_name="phase_segments")
    op.drop_index(op.f("ix_phase_segments_analysis_id"), table_name="phase_segments")
    op.drop_table("phase_segments")
    op.drop_index(op.f("ix_velocity_series_analysis_id"), table_name="velocity_series")
    op.drop_table("velocity_series")
    op.drop_index(op.f("ix_swim_faults_fault_name"), table_name="swim_faults")
    op.drop_index(op.f("ix_swim_faults_analysis_id"), table_name="swim_faults")
    op.drop_table("swim_faults")
    op.drop_index(op.f("ix_swim_metrics_metric_name"), table_name="swim_metrics")
    op.drop_index(op.f("ix_swim_metrics_analysis_id"), table_name="swim_metrics")
    op.drop_table("swim_metrics")
