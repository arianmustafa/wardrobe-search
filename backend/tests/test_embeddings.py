"""Embedding behavior: demo-mode determinism, and what is sent to Gemini.

Real-mode tests use the `fake_gemini` fixture — the third-party client is the
one thing mocked; the app's own image handling and normalization run for real.
"""
import io

import numpy as np
import pytest
from PIL import Image

from app import embeddings

from .conftest import image_bytes


def norm(vec: list[float]) -> float:
    return float(np.linalg.norm(vec))


# --- demo mode (no API key): deterministic stand-in embeddings ---


def test_demo_image_embedding_is_deterministic():
    data = image_bytes()
    assert embeddings.embed_image(data) == embeddings.embed_image(data)


def test_demo_embeddings_differ_between_images():
    a = embeddings.embed_image(image_bytes(size=(20, 20)))
    b = embeddings.embed_image(image_bytes(size=(30, 30)))
    assert a != b


def test_demo_embeddings_have_configured_dimension_and_unit_length():
    vec = embeddings.embed_query("red shirt")
    assert len(vec) == embeddings.settings.embedding_dim
    assert norm(vec) == pytest.approx(1.0)


def test_demo_query_embedding_ignores_case_and_whitespace():
    assert embeddings.embed_query("  Red Shirt ") == embeddings.embed_query("red shirt")


# --- real mode (Gemini mocked): payload shape and vector normalization ---


def sent_image(fake) -> tuple[bytes, str]:
    part = fake.calls[0]["contents"][0]
    return part.inline_data.data, part.inline_data.mime_type


def test_image_sent_to_gemini_is_downscaled_jpeg(fake_gemini):
    embeddings.embed_image(image_bytes(size=(2048, 512)))
    payload, mime = sent_image(fake_gemini)
    assert mime == "image/jpeg"
    img = Image.open(io.BytesIO(payload))
    assert img.format == "JPEG"
    assert max(img.size) <= embeddings.settings.max_image_edge


def test_small_images_are_not_upscaled(fake_gemini):
    embeddings.embed_image(image_bytes(size=(64, 48)))
    payload, _ = sent_image(fake_gemini)
    assert Image.open(io.BytesIO(payload)).size == (64, 48)


def test_gemini_vectors_are_renormalized_to_unit_length(fake_gemini):
    dim = embeddings.settings.embedding_dim
    fake_gemini.queue = [[3.0, 4.0] + [0.0] * (dim - 2)]
    vec = embeddings.embed_image(image_bytes())
    assert norm(vec) == pytest.approx(1.0)
    assert vec[0] == pytest.approx(0.6)  # 3/5 — direction preserved
