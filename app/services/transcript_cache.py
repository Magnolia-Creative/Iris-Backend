import json
import ssl
from typing import Any
from urllib.parse import urlparse

import certifi
from redis.asyncio import Redis

from app.config import settings


_redis_client: Redis | None = None


def get_transcript_cache_key(session_id: str, clip_id: str) -> str:
    return f"session:{session_id}:transcript:{clip_id}"


def get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        redis_url = settings.redis_url
        tls_kwargs = (
            {"ssl_ca_certs": certifi.where(), "ssl_cert_reqs": ssl.CERT_REQUIRED}
            if urlparse(redis_url).scheme == "rediss"
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
    await redis.set(cache_key, json.dumps(transcript_payload), ex=ttl)
    return cache_key


async def get_cached_transcript(cache_key: str) -> dict[str, Any] | None:
    redis = get_redis_client()
    payload = await redis.get(cache_key)
    if payload is None:
        return None
    return json.loads(payload)


async def is_transcript_cached(cache_key: str) -> bool:
    redis = get_redis_client()
    return bool(await redis.exists(cache_key))


async def delete_cached_transcript(session_id: str, clip_id: str) -> None:
    redis = get_redis_client()
    await redis.delete(get_transcript_cache_key(session_id, clip_id))
