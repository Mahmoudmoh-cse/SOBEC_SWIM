"""add professional swim analysis jobs

Revision ID: 0005_swim_analyses
Revises: 0004_ai_outputs
Create Date: 2026-05-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005_swim_analyses"
down_revision: Union[str, None] = "0004_ai_outputs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "swim_analyses",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("swimmer_id", sa.String(length=36), nullable=True),
        sa.Column("stroke_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("pose_backend", sa.String(length=100), nullable=False),
        sa.Column("side_video_path", sa.String(length=500), nullable=False),
        sa.Column("front_video_path", sa.String(length=500), nullable=False),
        sa.Column("side_annotated_path", sa.String(length=500), nullable=True),
        sa.Column("front_annotated_path", sa.String(length=500), nullable=True),
        sa.Column("raw_landmarks_path", sa.String(length=500), nullable=True),
        sa.Column("smoothed_landmarks_path", sa.String(length=500), nullable=True),
        sa.Column("report_json", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["swimmer_id"], ["swimmers.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_swim_analyses_swimmer_id"), "swim_analyses", ["swimmer_id"], unique=False)
    op.create_index(op.f("ix_swim_analyses_status"), "swim_analyses", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_swim_analyses_status"), table_name="swim_analyses")
    op.drop_index(op.f("ix_swim_analyses_swimmer_id"), table_name="swim_analyses")
    op.drop_table("swim_analyses")
