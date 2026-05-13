"""add sentence_upload_transcripts table

Revision ID: b2e8f1a4c9d0
Revises: 7a4c2d1b9e6f
Create Date: 2026-05-12 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "b2e8f1a4c9d0"
down_revision: Union[str, Sequence[str], None] = "7a4c2d1b9e6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sentence_upload_transcripts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transcript", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="sentence_upload_transcripts_pkey"),
    )
    op.create_index(
        "idx_sentence_upload_transcripts_created_at",
        "sentence_upload_transcripts",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_sentence_upload_transcripts_created_at", table_name="sentence_upload_transcripts")
    op.drop_table("sentence_upload_transcripts")
