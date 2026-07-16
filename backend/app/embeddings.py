"""Embedding generation via Google's gemini-embedding-2 multimodal model.

Images and text queries are mapped into the *same* vector space, so a text
query can retrieve matching garment photos directly. All vectors are
L2-normalized (recommended for reduced Matryoshka dimensions and makes cosine
similarity == dot product).

If no API key is configured, a deterministic pseudo-embedding is used instead so
the app remains runnable end-to-end for UI/plumbing work (not semantically
meaningful).
"""
import hashlib
import io
import logging

import numpy as np
from PIL import Image

from .config import get_settings

logger = logging.getLogger("wardrobe.embeddings")
settings = get_settings()

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _normalize(values) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def _fake_embedding(seed: bytes) -> list[float]:
    """Deterministic unit vector derived from the input — DEMO mode only."""
    digest = hashlib.sha256(seed).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return _normalize(rng.standard_normal(settings.embedding_dim))


def _downscale_for_embedding(image_bytes: bytes) -> tuple[bytes, str]:
    """Shrink to a sane size + re-encode as JPEG to cut latency and cost."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        img.thumbnail((settings.max_image_edge, settings.max_image_edge))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"
    except Exception as exc:  # corrupt/unsupported image — let caller's error surface
        logger.warning("Could not preprocess image, sending original bytes: %s", exc)
        return image_bytes, "image/jpeg"


def embed_image(image_bytes: bytes) -> list[float]:
    if not settings.use_real_embeddings:
        return _fake_embedding(image_bytes)

    from google.genai import types

    payload, mime = _downscale_for_embedding(image_bytes)
    result = _get_client().models.embed_content(
        model=settings.embedding_model,
        contents=[types.Part.from_bytes(data=payload, mime_type=mime)],
        config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dim),
    )
    return _normalize(result.embeddings[0].values)


# gemini-embedding-2 has no task_type parameter; Google recommends encoding the
# retrieval task as a prompt prefix on the query text. This lifts the query↔image
# cosine similarity substantially (measured ~0.32 -> ~0.56 on our own items),
# while preserving ranking. Images (the "documents") are embedded as-is.
_QUERY_TASK_PREFIX = "task: search result | query: "


def embed_query(text: str) -> list[float]:
    if not settings.use_real_embeddings:
        return _fake_embedding(text.strip().lower().encode("utf-8"))

    from google.genai import types

    result = _get_client().models.embed_content(
        model=settings.embedding_model,
        contents=f"{_QUERY_TASK_PREFIX}{text}",
        config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dim),
    )
    return _normalize(result.embeddings[0].values)
