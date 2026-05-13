"""project-scoped clips, sessions.project_id, nullable embedding session_id

Revision ID: e3f4a5b6c7d8
Revises: d1a2b3c4d5e6
Create Date: 2026-05-14 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


revision: str = "e3f4a5b6c7d8"
down_revision: Union[str, Sequence[str], None] = "d1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE sessions
        ADD COLUMN IF NOT EXISTS project_id BIGINT REFERENCES projects(id) ON DELETE SET NULL
        """
    )
    op.execute(
        """
        UPDATE sessions s
        SET project_id = (
            SELECT c.project_id FROM clips c WHERE c.session_id = s.id LIMIT 1
        )
        WHERE EXISTS (SELECT 1 FROM clips c WHERE c.session_id = s.id)
          AND s.project_id IS NULL
        """
    )

    op.execute("DROP INDEX IF EXISTS idx_clips_session_local_key")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_clips_project_local_key
        ON clips (project_id, local_key)
        WHERE local_key IS NOT NULL
        """
    )

    op.execute(
        """
        ALTER TABLE clip_chunk_embeddings
        ALTER COLUMN session_id DROP NOT NULL
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DELETE FROM clip_chunk_embeddings WHERE session_id IS NULL
        """
    )
    op.execute(
        """
        ALTER TABLE clip_chunk_embeddings
        ALTER COLUMN session_id SET NOT NULL
        """
    )

    op.execute("DROP INDEX IF EXISTS idx_clips_project_local_key")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_clips_session_local_key
        ON clips (session_id, local_key)
        """
    )

    op.execute("ALTER TABLE sessions DROP COLUMN IF EXISTS project_id")
