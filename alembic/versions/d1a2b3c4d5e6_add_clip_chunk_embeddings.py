"""add clip_chunk_embeddings with pgvector halfvec

Revision ID: d1a2b3c4d5e6
Revises: b2e8f1a4c9d0
Create Date: 2026-05-13 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


revision: str = "d1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "b2e8f1a4c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS clip_chunk_embeddings (
            id BIGSERIAL PRIMARY KEY,
            project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            session_id BIGINT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            clip_id BIGINT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
            local_key TEXT NOT NULL,
            modality TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_time_seconds DOUBLE PRECISION NOT NULL,
            end_time_seconds DOUBLE PRECISION NOT NULL,
            center_time_seconds DOUBLE PRECISION NOT NULL,
            embedding halfvec(3072) NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_clip_chunk_embeddings_clip_mod_idx UNIQUE (clip_id, modality, chunk_index)
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_clip_chunk_embeddings_project ON clip_chunk_embeddings (project_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_clip_chunk_embeddings_clip ON clip_chunk_embeddings (clip_id)")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_clip_chunk_embeddings_hnsw
        ON clip_chunk_embeddings
        USING hnsw (embedding halfvec_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS clip_chunk_embeddings")
