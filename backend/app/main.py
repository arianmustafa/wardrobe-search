"""FastAPI application: upload garment images, embed them, and search by text."""
import io
import logging
import mimetypes
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import embeddings, store
from .config import IMAGES_DIR, get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wardrobe")

# Ensure stored .avif files are served with the correct Content-Type regardless
# of the host's mimetypes database.
mimetypes.add_type("image/avif", ".avif")

settings = get_settings()
app = FastAPI(title="Wardrobe Semantic Search", version="0.1.0")

# Dev convenience: the Vite dev server proxies /api and /images, but allow direct
# cross-origin calls too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images statically at /images/<filename>.
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/avif"}
# Extension by the format PIL actually detects — the client's Content-Type header
# is only a cheap pre-check, the stored extension comes from the real bytes.
EXT_BY_FORMAT = {
    "jpeg": ".jpg",
    "png": ".png",
    "webp": ".webp",
    "gif": ".gif",
    "avif": ".avif",
}


# Reject absurd pixel counts before decoding — a small compressed file can
# expand into gigabytes of raster memory (decompression bomb).
MAX_IMAGE_PIXELS = 50_000_000


def _sniff_image_ext(data: bytes) -> str:
    """Verify the bytes decode as a supported image; return the true extension."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").lower()
            width, height = img.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    400, f"Image too large: {width}x{height} exceeds pixel limit"
                )
            img.verify()
    except HTTPException:
        raise
    except Image.DecompressionBombError:
        # Pillow refuses extreme pixel counts at open(), before our own check runs.
        raise HTTPException(400, "Image too large: exceeds pixel limit")
    except Exception:
        raise HTTPException(400, "File is not a valid image")
    ext = EXT_BY_FORMAT.get(fmt)
    if not ext:
        raise HTTPException(400, f"Unsupported image format: {fmt or 'unknown'}")
    return ext


def _to_item(meta: dict) -> dict:
    return {
        "id": meta["id"],
        "filename": meta["filename"],
        "original_name": meta.get("original_name") or None,
        "title": meta.get("title") or meta.get("original_name") or "Untitled",
        "uploaded_at": meta.get("uploaded_at"),
        "image_url": f"/images/{meta['filename']}",
    }


def _calibrated_relevance(score: float) -> float:
    """Map raw cross-modal cosine [floor, ceiling] onto 0–1 for display.

    Monotonic, so it never reorders results — it only rescales the compressed
    similarity band into an intuitive range.
    """
    lo, hi = settings.relevance_floor, settings.relevance_ceiling
    if hi <= lo:
        return max(0.0, min(1.0, score))
    return max(0.0, min(1.0, (score - lo) / (hi - lo)))


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "items": store.count(),
        "model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "real_embeddings": settings.use_real_embeddings,
    }


@app.get("/api/items")
def list_items() -> dict:
    items = [_to_item(m) for m in store.list_items()]
    items.sort(key=lambda x: x["uploaded_at"] or "", reverse=True)
    return {"items": items, "count": len(items)}


@app.post("/api/items", status_code=201)
async def create_item(
    file: UploadFile = File(...),
    title: str | None = Form(None),
) -> dict:
    # Cheap pre-check on the declared type only when one is given — non-browser
    # clients (and some browsers) send no/generic Content-Type, and the bytes
    # are validated by PIL below regardless.
    content_type = (file.content_type or "").lower()
    if content_type not in ("", "application/octet-stream") and content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {content_type}")

    # Enforce the size limit here rather than relying on a reverse proxy —
    # the backend is also reachable directly (dev proxy, localhost).
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > max_bytes:
        raise HTTPException(413, f"File too large (max {settings.max_upload_mb} MB)")

    ext = _sniff_image_ext(data)

    item_id = uuid.uuid4().hex
    filename = f"{item_id}{ext}"
    image_path = IMAGES_DIR / filename
    image_path.write_bytes(data)

    try:
        # The Gemini call is blocking; run it off the event loop so concurrent
        # requests aren't stalled while an upload embeds.
        vector = await run_in_threadpool(embeddings.embed_image, data)
    except Exception as exc:  # roll back the saved file if embedding fails
        image_path.unlink(missing_ok=True)
        logger.exception("Embedding failed for upload")
        raise HTTPException(502, f"Embedding failed: {exc}") from exc

    meta = {
        "id": item_id,
        "filename": filename,
        "original_name": file.filename or "",
        "title": (title or "").strip(),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        store.add_item(item_id, vector, meta)
    except Exception as exc:  # roll back the saved file if the store insert fails
        image_path.unlink(missing_ok=True)
        logger.exception("Storing item failed")
        raise HTTPException(500, "Failed to store item") from exc
    return _to_item(meta)


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    n: int | None = Query(None, ge=1, le=100),
) -> dict:
    q = q.strip()
    if not q:
        raise HTTPException(400, "Query must not be empty")
    top_n = n or settings.default_top_n

    if store.count() == 0:
        return {"query": q, "results": []}

    try:
        vector = embeddings.embed_query(q)
    except Exception as exc:
        logger.exception("Embedding failed for query")
        raise HTTPException(502, f"Embedding failed: {exc}") from exc

    results = []
    for raw in store.search(vector, top_n):
        item = _to_item(raw)
        score = float(raw["score"])
        item["score"] = round(score, 4)  # raw cosine similarity
        item["relevance"] = round(_calibrated_relevance(score), 4)  # 0–1 for display
        results.append(item)
    return {"query": q, "results": results}


@app.delete("/api/items/{item_id}", status_code=204)
def delete_item(item_id: str) -> None:
    meta = store.get_item(item_id)
    if not meta:
        raise HTTPException(404, "Item not found")
    store.delete_item(item_id)
    (IMAGES_DIR / meta["filename"]).unlink(missing_ok=True)
