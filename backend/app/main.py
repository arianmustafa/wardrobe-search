"""FastAPI application: upload garment images, embed them, and search by text."""
import logging
import mimetypes
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
EXT_BY_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
}


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
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {content_type or 'unknown'}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    item_id = uuid.uuid4().hex
    filename = f"{item_id}{EXT_BY_TYPE.get(content_type, '.jpg')}"
    image_path = IMAGES_DIR / filename
    image_path.write_bytes(data)

    try:
        vector = embeddings.embed_image(data)
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
    store.add_item(item_id, vector, meta)
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
