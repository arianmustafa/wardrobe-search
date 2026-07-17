"""API endpoints via TestClient, running against a temp store in demo mode.

Tests taking the `fake_gemini` fixture exercise the real-embeddings code path
with the third-party Gemini client mocked out — vectors are chosen per test so
search scores are exact cosines.
"""
import io
import math
import struct
import zlib

import pytest
from PIL import Image

from app import config, main, store

from .conftest import image_bytes


def upload(client, data: bytes | None = None, *, name="photo.png", content_type="image/png", title=None):
    form = {"title": title} if title is not None else {}
    return client.post(
        "/api/items",
        files={"file": (name, data if data is not None else image_bytes(), content_type)},
        data=form,
    )


def unit_vector(cosine: float) -> list[float]:
    """Unit vector whose cosine similarity to unit_vector(1.0) is `cosine`."""
    vec = [0.0] * main.settings.embedding_dim
    vec[0] = cosine
    vec[1] = math.sqrt(1.0 - cosine * cosine)
    return vec


def one_bit_png(size: tuple[int, int]) -> bytes:
    """Real PNG whose pixel count is huge but whose bytes stay small (1-bit)."""
    buf = io.BytesIO()
    Image.new("1", size).save(buf, format="PNG")
    return buf.getvalue()


def png_claiming_size(width: int, height: int) -> bytes:
    """Minimal PNG header declaring absurd dimensions, without pixel data."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b""))
        + chunk(b"IEND", b"")
    )


# --- health ---


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["real_embeddings"] is False
    assert body["items"] == 0


# --- upload ---


def test_upload_returns_item(client):
    res = upload(client, title="Red tee")
    assert res.status_code == 201
    item = res.json()
    assert item["title"] == "Red tee"
    assert item["original_name"] == "photo.png"
    assert item["filename"].endswith(".png")
    assert item["image_url"] == f"/images/{item['filename']}"
    assert (config.IMAGES_DIR / item["filename"]).is_file()


def test_upload_title_falls_back_to_original_name(client):
    assert upload(client).json()["title"] == "photo.png"


def test_upload_extension_comes_from_bytes_not_header(client):
    # JPEG bytes declared as image/png must be stored as .jpg
    res = upload(client, image_bytes(fmt="JPEG"), content_type="image/png")
    assert res.status_code == 201
    assert res.json()["filename"].endswith(".jpg")


@pytest.mark.parametrize(("fmt", "ext"), [("WEBP", ".webp"), ("GIF", ".gif")])
def test_upload_supports_webp_and_gif(client, fmt, ext):
    res = upload(client, image_bytes(fmt=fmt), name=f"photo{ext}", content_type=f"image/{fmt.lower()}")
    assert res.status_code == 201
    assert res.json()["filename"].endswith(ext)


def test_upload_accepts_missing_content_type(client):
    res = upload(client, content_type="application/octet-stream")
    assert res.status_code == 201


def test_upload_rejects_disallowed_content_type(client):
    res = upload(client, content_type="text/plain")
    assert res.status_code == 400
    assert "Unsupported file type" in res.json()["detail"]


def test_upload_rejects_non_image_bytes(client):
    res = upload(client, b"definitely not an image")
    assert res.status_code == 400
    assert "not a valid image" in res.json()["detail"]


def test_upload_rejects_empty_file(client):
    assert upload(client, b"").status_code == 400


def test_upload_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(main.settings, "max_upload_mb", 1)
    res = upload(client, b"\0" * (1024 * 1024 + 1))
    assert res.status_code == 413


def test_upload_rejects_excessive_pixel_count(client):
    res = upload(client, one_bit_png((8000, 8000)))  # 64M pixels > 50M app limit
    assert res.status_code == 400
    assert "pixel limit" in res.json()["detail"]


def test_upload_rejects_decompression_bomb_header(client):
    # Dimensions huge enough that Pillow itself refuses at open()
    res = upload(client, png_claiming_size(60_000, 60_000))
    assert res.status_code == 400
    assert "pixel limit" in res.json()["detail"]


def test_failed_upload_leaves_no_files_behind(client):
    upload(client, b"garbage")
    assert list(config.IMAGES_DIR.iterdir()) == []


def test_embedding_failure_returns_502_and_rolls_back_file(client, fake_gemini):
    fake_gemini.error = RuntimeError("quota exceeded")
    res = upload(client)
    assert res.status_code == 502
    assert "Embedding failed" in res.json()["detail"]
    assert list(config.IMAGES_DIR.iterdir()) == []


def test_store_failure_returns_500_and_rolls_back_file(client, monkeypatch):
    def broken_add(*args, **kwargs):
        raise RuntimeError("chroma unavailable")

    monkeypatch.setattr(store._collection, "add", broken_add)
    res = upload(client)
    assert res.status_code == 500
    assert list(config.IMAGES_DIR.iterdir()) == []


# --- list ---


def test_list_items_newest_first(client):
    first = upload(client, title="older").json()
    second = upload(client, title="newer").json()
    body = client.get("/api/items").json()
    assert body["count"] == 2
    assert [i["id"] for i in body["items"]] == [second["id"], first["id"]]


# --- search ---


def test_search_empty_store(client):
    body = client.get("/api/search", params={"q": "red shirt"}).json()
    assert body == {"query": "red shirt", "results": []}


def test_search_rejects_empty_query(client):
    assert client.get("/api/search", params={"q": ""}).status_code == 422
    assert client.get("/api/search", params={"q": " "}).status_code == 400


def test_search_rejects_out_of_range_n(client):
    assert client.get("/api/search", params={"q": "shirt", "n": 0}).status_code == 422
    assert client.get("/api/search", params={"q": "shirt", "n": 101}).status_code == 422


def test_search_returns_scored_results(client):
    upload(client, image_bytes(size=(20, 20)), title="a")
    upload(client, image_bytes(size=(30, 30)), title="b")
    results = client.get("/api/search", params={"q": "red shirt"}).json()["results"]
    assert len(results) == 2
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    for r in results:
        assert 0.0 <= r["relevance"] <= 1.0


def test_search_respects_n(client):
    for size in (20, 24, 28):
        upload(client, image_bytes(size=(size, size)))
    results = client.get("/api/search", params={"q": "shirt", "n": 2}).json()["results"]
    assert len(results) == 2


def test_search_n_larger_than_store_is_clamped(client):
    upload(client)
    results = client.get("/api/search", params={"q": "shirt", "n": 100}).json()["results"]
    assert len(results) == 1


# --- relevance calibration (through the API, scores controlled via fake Gemini) ---


@pytest.mark.parametrize(
    ("cosine", "expected_relevance"),
    [
        (0.30, 0.0),  # floor -> 0%
        (0.65, 1.0),  # ceiling -> 100%
        (0.475, 0.5),  # midpoint of the band
        (0.10, 0.0),  # below the floor clamps to 0
        (0.90, 1.0),  # above the ceiling clamps to 1
    ],
)
def test_search_maps_cosine_band_to_relevance(client, fake_gemini, cosine, expected_relevance):
    fake_gemini.queue = [unit_vector(1.0), unit_vector(cosine)]
    upload(client)
    result = client.get("/api/search", params={"q": "navy shirt"}).json()["results"][0]
    assert result["score"] == pytest.approx(cosine, abs=1e-3)
    assert result["relevance"] == pytest.approx(expected_relevance, abs=2e-3)


def test_degenerate_relevance_band_clamps_raw_score(client, fake_gemini, monkeypatch):
    monkeypatch.setattr(main.settings, "relevance_floor", 0.5)
    monkeypatch.setattr(main.settings, "relevance_ceiling", 0.5)
    fake_gemini.queue = [unit_vector(1.0), unit_vector(0.7)]
    upload(client)
    result = client.get("/api/search", params={"q": "navy shirt"}).json()["results"][0]
    assert result["relevance"] == pytest.approx(0.7, abs=2e-3)


# --- delete ---


def test_delete_removes_item_and_image(client):
    item = upload(client).json()
    image_path = config.IMAGES_DIR / item["filename"]
    assert client.delete(f"/api/items/{item['id']}").status_code == 204
    assert not image_path.exists()
    assert client.get("/api/items").json()["count"] == 0


def test_delete_unknown_item_404(client):
    assert client.delete("/api/items/does-not-exist").status_code == 404
