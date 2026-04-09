"""add clip processing fields

Revision ID: 7a4c2d1b9e6f
Revises: c3d9a7f4e2b1
Create Date: 2026-04-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7a4c2d1b9e6f"
down_revision: Union[str, Sequence[str], None] = "c3d9a7f4e2b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(bind: sa.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(table_name)}


def _index_names(bind: sa.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {idx["name"] for idx in inspector.get_indexes(table_name) if idx.get("name")}


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    columns = _column_names(bind, "clips")

    additions: list[tuple[str, sa.Column]] = [
        ("local_key", sa.Column("local_key", sa.Text(), nullable=True)),
        ("processing_status", sa.Column("processing_status", sa.Text(), nullable=False, server_default="created")),
        ("processing_error", sa.Column("processing_error", sa.Text(), nullable=True)),
        ("provider_job_id", sa.Column("provider_job_id", sa.Text(), nullable=True)),
        ("processing_started_at", sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True)),
        ("processing_completed_at", sa.Column("processing_completed_at", sa.DateTime(timezone=True), nullable=True)),
        ("processing_cancelled_at", sa.Column("processing_cancelled_at", sa.DateTime(timezone=True), nullable=True)),
    ]

    for name, column in additions:
        if name not in columns:
            op.add_column("clips", column)

    op.execute("UPDATE clips SET processing_status = COALESCE(processing_status, 'ready')")
    op.alter_column("clips", "processing_status", server_default=None)

    index_names = _index_names(bind, "clips")
    if "idx_clips_session_local_key" not in index_names:
        op.create_index(
            "idx_clips_session_local_key",
            "clips",
            ["session_id", "local_key"],
            unique=True,
        )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    columns = _column_names(bind, "clips")
    index_names = _index_names(bind, "clips")

    if "idx_clips_session_local_key" in index_names:
        op.drop_index("idx_clips_session_local_key", table_name="clips")

    for name in [
        "processing_cancelled_at",
        "processing_completed_at",
        "processing_started_at",
        "provider_job_id",
        "processing_error",
        "processing_status",
        "local_key",
    ]:
        if name in columns:
            op.drop_column("clips", name)
