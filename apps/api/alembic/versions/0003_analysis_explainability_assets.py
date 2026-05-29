"""add technique analysis explainability assets

Revision ID: 0003_analysis_explainability_assets
Revises: 0002_analysis_trust_fields
Create Date: 2026-05-21
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_analysis_explainability_assets"
down_revision: Union[str, None] = "0002_analysis_trust_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("technique_reports", sa.Column("analysis_overlay_video_url", sa.String(length=500), nullable=True))
    op.add_column("technique_reports", sa.Column("analysis_frame_urls", sa.JSON(), server_default="[]", nullable=False))
    op.add_column("technique_reports", sa.Column("analysis_events", sa.JSON(), server_default="[]", nullable=False))


def downgrade() -> None:
    op.drop_column("technique_reports", "analysis_events")
    op.drop_column("technique_reports", "analysis_frame_urls")
    op.drop_column("technique_reports", "analysis_overlay_video_url")
