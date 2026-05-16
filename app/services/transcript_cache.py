import json
import logging
import ssl
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import certifi
from redis.asyncio import Redis

from app.config import settings


logger = logging.getLogger(__name__)
_redis_client: Redis | None = None


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def get_transcript_cache_key(session_id: str, clip_id: str) -> str:
    return f"session:{session_id}:transcript:{clip_id}"


def get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        redis_url = settings.redis_url
        parsed = urlparse(redis_url)
        logger.info(
            "[transcript_cache] Initializing Redis client scheme=%s host=%s port=%s",
            parsed.scheme,
            parsed.hostname,
            parsed.port,
        )
        tls_kwargs = (
            {"ssl_ca_certs": certifi.where(), "ssl_cert_reqs": ssl.CERT_REQUIRED}
            if parsed.scheme == "rediss"
            else {}
        )
        _redis_client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            **tls_kwargs,
        )
    return _redis_client


async def cache_transcript(
    session_id: str,
    clip_id: str,
    transcript_payload: dict[str, Any],
    ttl_seconds: int | None = None,
) -> str:
    cache_key = get_transcript_cache_key(session_id, clip_id)
    redis = get_redis_client()
    ttl = ttl_seconds or settings.transcript_cache_ttl_seconds
    started_at = perf_counter()
    logger.info(
        "[transcript_cache] set starting key=%s ttl=%s payload_bytes=%s",
        cache_key,
        ttl,
        len(json.dumps(transcript_payload)),
    )
    await redis.set(cache_key, json.dumps(transcript_payload), ex=ttl)
    logger.info("[transcript_cache] set completed key=%s elapsed_ms=%s", cache_key, _elapsed_ms(started_at))
    return cache_key


async def get_cached_transcript(cache_key: str) -> dict[str, Any] | None:
    redis = get_redis_client()
    started_at = perf_counter()
    logger.info("[transcript_cache] get starting key=%s", cache_key)
    payload = await redis.get(cache_key)
    if payload is None:
        logger.info(
            "[transcript_cache] get miss key=%s elapsed_ms=%s",
            cache_key,
            _elapsed_ms(started_at),
        )
        return None
    logger.info(
        "[transcript_cache] get hit key=%s payload_bytes=%s elapsed_ms=%s",
        cache_key,
        len(payload),
        _elapsed_ms(started_at),
    )
    return json.loads(payload)


async def is_transcript_cached(cache_key: str) -> bool:
    redis = get_redis_client()
    started_at = perf_counter()
    logger.info("[transcript_cache] exists starting key=%s", cache_key)
    exists = bool(await redis.exists(cache_key))
    logger.info(
        "[transcript_cache] exists completed key=%s exists=%s elapsed_ms=%s",
        cache_key,
        exists,
        _elapsed_ms(started_at),
    )
    return exists


async def delete_cached_transcript(session_id: str, clip_id: str) -> None:
    redis = get_redis_client()
    cache_key = get_transcript_cache_key(session_id, clip_id)
    started_at = perf_counter()
    logger.info("[transcript_cache] delete starting key=%s", cache_key)
    await redis.delete(cache_key)
    logger.info(
        "[transcript_cache] delete completed key=%s elapsed_ms=%s",
        cache_key,
        _elapsed_ms(started_at),
    )
