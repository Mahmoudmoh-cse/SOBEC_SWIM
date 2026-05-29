"""add ai outputs audit table

Revision ID: 0004_ai_outputs
Revises: 0003_analysis_explainability_assets
Create Date: 2026-05-21
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_ai_outputs"
down_revision: Union[str, None] = "0003_analysis_explainability_assets"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ai_outputs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("swimmer_id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=True),
        sa.Column("technique_report_id", sa.String(length=36), nullable=True),
        sa.Column("training_plan_id", sa.String(length=36), nullable=True),
        sa.Column("race_analysis_id", sa.String(length=36), nullable=True),
        sa.Column("mental_checkin_id", sa.String(length=36), nullable=True),
        sa.Column("output_type", sa.String(length=100), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("prompt_version", sa.String(length=50), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("raw_response", sa.Text(), nullable=False),
        sa.Column("parsed_json", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["mental_checkin_id"], ["mental_checkins.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["race_analysis_id"], ["race_analyses.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["swimmer_id"], ["swimmers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["technique_report_id"], ["technique_reports.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["training_plan_id"], ["training_plans.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_ai_outputs_swimmer_id"), "ai_outputs", ["swimmer_id"], unique=False)
    op.create_index(op.f("ix_ai_outputs_output_type"), "ai_outputs", ["output_type"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_ai_outputs_output_type"), table_name="ai_outputs")
    op.drop_index(op.f("ix_ai_outputs_swimmer_id"), table_name="ai_outputs")
    op.drop_table("ai_outputs")

