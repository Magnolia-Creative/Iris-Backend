"""add clips.session_id

Revision ID: 9c8f3f2a1b7c
Revises:
Create Date: 2026-03-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9c8f3f2a1b7c"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(bind: sa.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(table_name)}


def _foreign_key_names(bind: sa.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {fk["name"] for fk in inspector.get_foreign_keys(table_name) if fk.get("name")}


def _index_names(bind: sa.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {idx["name"] for idx in inspector.get_indexes(table_name) if idx.get("name")}


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    columns = _column_names(bind, "clips")

    if "session_id" not in columns:
        op.add_column("clips", sa.Column("session_id", sa.BigInteger(), nullable=True))

    fk_names = _foreign_key_names(bind, "clips")
    if "fk_clips_session_id_sessions" not in fk_names:
        op.create_foreign_key(
            "fk_clips_session_id_sessions",
            "clips",
            "sessions",
            ["session_id"],
            ["id"],
            ondelete="CASCADE",
        )

    index_names = _index_names(bind, "clips")
    if "idx_clips_session_id" not in index_names:
        op.create_index("idx_clips_session_id", "clips", ["session_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()

    index_names = _index_names(bind, "clips")
    if "idx_clips_session_id" in index_names:
        op.drop_index("idx_clips_session_id", table_name="clips")

    fk_names = _foreign_key_names(bind, "clips")
    if "fk_clips_session_id_sessions" in fk_names:
        op.drop_constraint("fk_clips_session_id_sessions", "clips", type_="foreignkey")

    columns = _column_names(bind, "clips")
    if "session_id" in columns:
        op.drop_column("clips", "session_id")
