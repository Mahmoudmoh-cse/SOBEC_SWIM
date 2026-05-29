"""add technique analysis trust fields

Revision ID: 0002_analysis_trust_fields
Revises: 0001_initial
Create Date: 2026-05-21
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_analysis_trust_fields"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("technique_reports", sa.Column("analysis_status", sa.String(length=50), server_default="completed", nullable=False))
    op.add_column("technique_reports", sa.Column("frames_total", sa.Integer(), server_default="0", nullable=False))
    op.add_column("technique_reports", sa.Column("frames_analyzed", sa.Integer(), server_default="0", nullable=False))
    op.add_column("technique_reports", sa.Column("pose_detected_frames", sa.Integer(), server_default="0", nullable=False))
    op.add_column("technique_reports", sa.Column("pose_detection_rate", sa.Float(), server_default="0", nullable=False))
    op.add_column("technique_reports", sa.Column("confidence_score", sa.Float(), server_default="0", nullable=False))
    op.add_column("technique_reports", sa.Column("confidence_label", sa.String(length=50), server_default="low", nullable=False))
    op.add_column("technique_reports", sa.Column("analysis_warning", sa.Text(), nullable=True))
    op.add_column("technique_reports", sa.Column("analysis_error", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("technique_reports", "analysis_error")
    op.drop_column("technique_reports", "analysis_warning")
    op.drop_column("technique_reports", "confidence_label")
    op.drop_column("technique_reports", "confidence_score")
    op.drop_column("technique_reports", "pose_detection_rate")
    op.drop_column("technique_reports", "pose_detected_frames")
    op.drop_column("technique_reports", "frames_analyzed")
    op.drop_column("technique_reports", "frames_total")
    op.drop_column("technique_reports", "analysis_status")
