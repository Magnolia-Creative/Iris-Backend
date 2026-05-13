"""Gemini Embedding 2 client for multimodal clip indexing."""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-embedding-2"
OUTPUT_DIMENSIONALITY = 3072


def _client():
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _format_retrieval_query(text: str) -> str:
    return f"task: search result | query: {text}"


def embed_text_sync(query: str) -> list[float]:
    from google.genai import types

    client = _client()
    result = client.models.embed_content(
        model=MODEL_ID,
        contents=_format_retrieval_query(query.strip()),
        config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIMENSIONALITY),
    )
    embeddings = getattr(result, "embeddings", None) or []
    if not embeddings:
        raise RuntimeError("Gemini embed_content returned no embeddings for text.")
    values = getattr(embeddings[0], "values", None)
    if values is None:
        raise RuntimeError("Gemini embedding missing values.")
    return [float(x) for x in values]


def embed_image_bytes_sync(*, image_bytes: bytes, mime_type: str, title: str | None = None) -> list[float]:
    from google.genai import types

    client = _client()
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/jpeg")
    result = client.models.embed_content(
        model=MODEL_ID,
        contents=[part],
        config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIMENSIONALITY),
    )
    embeddings = getattr(result, "embeddings", None) or []
    if not embeddings:
        raise RuntimeError("Gemini embed_content returned no embeddings for image.")
    values = getattr(embeddings[0], "values", None)
    if values is None:
        raise RuntimeError("Gemini embedding missing values.")
    return [float(x) for x in values]


def embed_audio_bytes_sync(*, audio_bytes: bytes, mime_type: str) -> list[float]:
    from google.genai import types

    client = _client()
    part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type or "audio/wav")
    result = client.models.embed_content(
        model=MODEL_ID,
        contents=[part],
        config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIMENSIONALITY),
    )
    embeddings = getattr(result, "embeddings", None) or []
    if not embeddings:
        raise RuntimeError("Gemini embed_content returned no embeddings for audio.")
    values = getattr(embeddings[0], "values", None)
    if values is None:
        raise RuntimeError("Gemini embedding missing values.")
    return [float(x) for x in values]


async def embed_text(query: str) -> list[float]:
    return await asyncio.to_thread(embed_text_sync, query)


async def embed_image_bytes(*, image_bytes: bytes, mime_type: str, title: str | None = None) -> list[float]:
    return await asyncio.to_thread(embed_image_bytes_sync, image_bytes=image_bytes, mime_type=mime_type, title=title)


async def embed_audio_bytes(*, audio_bytes: bytes, mime_type: str) -> list[float]:
    return await asyncio.to_thread(embed_audio_bytes_sync, audio_bytes=audio_bytes, mime_type=mime_type)


def gemini_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))
