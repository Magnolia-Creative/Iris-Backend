"""Project-scoped semantic search over clip_chunk_embeddings."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services import clip_embedding_store, gemini_embedding
from app.services.semantic_constants import CHUNK_TOP_K, RESULTS_LIMIT
from app.services.semantic_range_merge import ChunkHit, merge_chunk_hits

logger = logging.getLogger(__name__)


async def search_project_semantic(
    db: AsyncSession,
    *,
    project_id: int,
    query: str,
    limit: int | None = None,
) -> dict[str, Any]:
    lim = limit if limit is not None and limit > 0 else RESULTS_LIMIT
    logger.info(
        "[semantic_search] request project_id=%s query_len=%s limit=%s",
        project_id,
        len(query.strip()),
        lim,
    )

    if not query.strip():
        logger.info("[semantic_search] empty query project_id=%s", project_id)
        return {"matches": [], "query": query}

    if not gemini_embedding.gemini_configured():
        logger.info("[semantic_search] disabled project_id=%s reason=no_gemini_key", project_id)
        return {"matches": [], "query": query, "disabled_reason": "GEMINI_API_KEY not set"}

    count = await clip_embedding_store.count_embeddings_for_project(db, project_id=project_id)
    logger.info("[semantic_search] project_id=%s embedding_row_count=%s", project_id, count)
    if count == 0:
        logger.info("[semantic_search] no index rows project_id=%s", project_id)
        return {"matches": [], "query": query}

    try:
        qemb = await gemini_embedding.embed_text(query.strip())
    except Exception as exc:
        logger.warning("[semantic_search] query embed failed project_id=%s: %s", project_id, exc)
        return {"matches": [], "query": query, "error": str(exc)}

    logger.info(
        "[semantic_search] query embedded project_id=%s vector_dim=%s",
        project_id,
        len(qemb),
    )

    fetch_k = max(CHUNK_TOP_K, (limit or RESULTS_LIMIT) * 4)
    rows = await clip_embedding_store.search_project_nearest(
        db, project_id=project_id, query_embedding=qemb, limit=fetch_k
    )

    hits = [
        ChunkHit(
            clip_id=int(r["clip_id"]),
            local_key=str(r["local_key"]),
            file_name=str(r["file_name"]) if r.get("file_name") is not None else None,
            modality=str(r["modality"]),
            chunk_index=int(r["chunk_index"]),
            start_time_seconds=float(r["start_time_seconds"]),
            end_time_seconds=float(r["end_time_seconds"]),
            center_time_seconds=float(r["center_time_seconds"]),
            score=float(r["score"]),
        )
        for r in rows
    ]

    merged = merge_chunk_hits(hits)
    top = merged[:lim]

    matches: list[dict[str, Any]] = []
    for m in top:
        matches.append(
            {
                "clip_id": m.clip_id,
                "local_key": m.local_key,
                "file_name": m.file_name,
                "start_time_seconds": m.start_time_seconds,
                "end_time_seconds": m.end_time_seconds,
                "confidence": m.confidence,
                "source": m.source,
            }
        )

    logger.info(
        "[semantic_search] project_id=%s raw_nn_hits=%s merged_ranges=%s returned_matches=%s",
        project_id,
        len(rows),
        len(merged),
        len(matches),
    )
    if matches:
        first = matches[0]
        logger.info(
            "[semantic_search] top_match project_id=%s clip_id=%s local_key=%s "
            "confidence=%.4f source=%s time=%.2f-%.2f",
            project_id,
            first["clip_id"],
            first["local_key"],
            float(first["confidence"]),
            first["source"],
            float(first["start_time_seconds"]),
            float(first["end_time_seconds"]),
        )

    return {"matches": matches, "query": query}
