"""add Clerk owner ids

Revision ID: f6a7b8c9d0e1
Revises: e3f4a5b6c7d8
Create Date: 2026-05-16 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, Sequence[str], None] = "e3f4a5b6c7d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(bind: sa.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(table_name)}


def _index_names(bind: sa.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {idx["name"] for idx in inspector.get_indexes(table_name) if idx.get("name")}


def upgrade() -> None:
    bind = op.get_bind()

    project_columns = _column_names(bind, "projects")
    if "clerk_user_id" not in project_columns:
        op.add_column("projects", sa.Column("clerk_user_id", sa.Text(), nullable=True))

    session_columns = _column_names(bind, "sessions")
    if "clerk_user_id" not in session_columns:
        op.add_column("sessions", sa.Column("clerk_user_id", sa.Text(), nullable=True))

    project_indexes = _index_names(bind, "projects")
    if "idx_projects_clerk_user_id" not in project_indexes:
        op.create_index("idx_projects_clerk_user_id", "projects", ["clerk_user_id"])

    session_indexes = _index_names(bind, "sessions")
    if "idx_sessions_clerk_user_id" not in session_indexes:
        op.create_index("idx_sessions_clerk_user_id", "sessions", ["clerk_user_id"])


def downgrade() -> None:
    bind = op.get_bind()

    session_indexes = _index_names(bind, "sessions")
    if "idx_sessions_clerk_user_id" in session_indexes:
        op.drop_index("idx_sessions_clerk_user_id", table_name="sessions")

    project_indexes = _index_names(bind, "projects")
    if "idx_projects_clerk_user_id" in project_indexes:
        op.drop_index("idx_projects_clerk_user_id", table_name="projects")

    session_columns = _column_names(bind, "sessions")
    if "clerk_user_id" in session_columns:
        op.drop_column("sessions", "clerk_user_id")

    project_columns = _column_names(bind, "projects")
    if "clerk_user_id" in project_columns:
        op.drop_column("projects", "clerk_user_id")
