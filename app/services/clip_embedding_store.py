"""Persist and query clip chunk embeddings in Postgres (pgvector halfvec)."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


def _halfvec_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in values) + "]"


async def delete_embeddings_for_clip(db: AsyncSession, *, clip_id: int) -> None:
    result = await db.execute(
        text("DELETE FROM clip_chunk_embeddings WHERE clip_id = :clip_id"),
        {"clip_id": clip_id},
    )
    await db.flush()
    rc = getattr(result, "rowcount", None)
    logger.info("[embeddings] cleared existing rows clip_id=%s rows_deleted=%s", clip_id, rc)


async def insert_embedding_row(
    db: AsyncSession,
    *,
    project_id: int,
    session_id: int | None,
    clip_id: int,
    local_key: str,
    modality: str,
    chunk_index: int,
    start_time_seconds: float,
    end_time_seconds: float,
    center_time_seconds: float,
    embedding: list[float],
    model_name: str,
) -> None:
    lit = _halfvec_literal(embedding)
    await db.execute(
        text(
            """
            INSERT INTO clip_chunk_embeddings (
                project_id, session_id, clip_id, local_key, modality, chunk_index,
                start_time_seconds, end_time_seconds, center_time_seconds,
                embedding, model_name
            ) VALUES (
                :project_id, :session_id, :clip_id, :local_key, :modality, :chunk_index,
                :start_time_seconds, :end_time_seconds, :center_time_seconds,
                CAST(:embedding AS halfvec(3072)), :model_name
            )
            ON CONFLICT (clip_id, modality, chunk_index) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                start_time_seconds = EXCLUDED.start_time_seconds,
                end_time_seconds = EXCLUDED.end_time_seconds,
                center_time_seconds = EXCLUDED.center_time_seconds,
                model_name = EXCLUDED.model_name,
                local_key = EXCLUDED.local_key,
                session_id = EXCLUDED.session_id,
                project_id = EXCLUDED.project_id
            """
        ),
        {
            "project_id": project_id,
            "session_id": session_id,
            "clip_id": clip_id,
            "local_key": local_key,
            "modality": modality,
            "chunk_index": chunk_index,
            "start_time_seconds": start_time_seconds,
            "end_time_seconds": end_time_seconds,
            "center_time_seconds": center_time_seconds,
            "embedding": lit,
            "model_name": model_name,
        },
    )
    await db.flush()


async def search_project_nearest(
    db: AsyncSession,
    *,
    project_id: int,
    query_embedding: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    lit = _halfvec_literal(query_embedding)
    result = await db.execute(
        text(
            """
            SELECT
                e.id,
                e.clip_id,
                e.local_key,
                c.file_name AS file_name,
                e.modality,
                e.chunk_index,
                e.start_time_seconds,
                e.end_time_seconds,
                e.center_time_seconds,
                e.embedding <=> CAST(:q AS halfvec(3072)) AS distance
            FROM clip_chunk_embeddings e
            INNER JOIN clips c ON c.id = e.clip_id
            WHERE e.project_id = :project_id
            ORDER BY e.embedding <=> CAST(:q AS halfvec(3072))
            LIMIT :lim
            """
        ),
        {"q": lit, "project_id": project_id, "lim": limit},
    )
    rows = result.mappings().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        dist = float(row["distance"])
        score = 1.0 - dist
        out.append(
            {
                "id": int(row["id"]),
                "clip_id": int(row["clip_id"]),
                "local_key": row["local_key"],
                "file_name": row["file_name"],
                "modality": row["modality"],
                "chunk_index": int(row["chunk_index"]),
                "start_time_seconds": float(row["start_time_seconds"]),
                "end_time_seconds": float(row["end_time_seconds"]),
                "center_time_seconds": float(row["center_time_seconds"]),
                "score": score,
            }
        )
    logger.info(
        "[embeddings] search nearest project_id=%s limit=%s returned_rows=%s",
        project_id,
        limit,
        len(out),
    )
    return out


async def count_embeddings_for_project(db: AsyncSession, *, project_id: int) -> int:
    result = await db.execute(
        text("SELECT COUNT(*) FROM clip_chunk_embeddings WHERE project_id = :pid"),
        {"pid": project_id},
    )
    return int(result.scalar_one() or 0)
