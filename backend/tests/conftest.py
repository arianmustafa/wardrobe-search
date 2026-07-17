"""Shared fixtures.

Tests run fully offline: the API key env vars are blanked so the app uses its
deterministic demo-mode embeddings, and the data directories are redirected to
a temp dir. Both must happen *before* the app modules are imported, because
store.py opens its ChromaDB client and main.py mounts the images directory at
import time.

The only thing mocked is the third-party Gemini client (`fake_gemini`); all
domain code runs for real.
"""
import io
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

os.environ["GEMINI_API_KEY"] = ""
os.environ["GEMINI_API_KEY_FILE"] = ""

from app import config  # noqa: E402

_tmp = Path(tempfile.mkdtemp(prefix="wardrobe-tests-"))
config.DATA_DIR = _tmp
config.IMAGES_DIR = _tmp / "images"
config.CHROMA_DIR = _tmp / "chroma"
config.get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app import embeddings, main, store  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def _clean_store():
    yield
    for meta in store.list_items():
        store.delete_item(meta["id"])
    for path in config.IMAGES_DIR.iterdir():
        path.unlink()


class FakeGemini:
    """Stand-in for the google-genai client — the third-party edge we mock.

    Records every embed_content call in `calls`. Returns vectors popped from
    `queue` (falling back to a fixed unit vector), or raises `error` if set.
    """

    def __init__(self, dim: int):
        self.models = self  # real client exposes embed_content under .models
        self.calls: list[dict] = []
        self.queue: list[list[float]] = []
        self.error: Exception | None = None
        self._dim = dim

    def embed_content(self, *, model, contents, config):
        if self.error is not None:
            raise self.error
        self.calls.append({"model": model, "contents": contents, "config": config})
        values = self.queue.pop(0) if self.queue else [1.0] + [0.0] * (self._dim - 1)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


@pytest.fixture
def fake_gemini(monkeypatch) -> FakeGemini:
    """Switch the app into real-embeddings mode, backed by the fake client."""
    fake = FakeGemini(dim=embeddings.settings.embedding_dim)
    monkeypatch.setattr(embeddings, "_client", fake)
    monkeypatch.setattr(embeddings.settings, "gemini_api_key", "unit-test-key")
    return fake


def image_bytes(fmt: str = "PNG", size: tuple[int, int] = (32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (180, 40, 40)).save(buf, format=fmt)
    return buf.getvalue()
