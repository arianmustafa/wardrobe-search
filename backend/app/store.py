"""Local persistent vector store backed by ChromaDB.

Stores one record per wardrobe item: its embedding vector plus metadata
(filename, original name, title, upload timestamp). Uses cosine space; vectors
are pre-normalized upstream so cosine distance is a clean 1 - similarity.
"""
import chromadb

from .config import CHROMA_DIR

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(
    name="wardrobe",
    metadata={"hnsw:space": "cosine"},
)


def add_item(item_id: str, embedding: list[float], metadata: dict) -> None:
    _collection.add(ids=[item_id], embeddings=[embedding], metadatas=[metadata])


def delete_item(item_id: str) -> None:
    _collection.delete(ids=[item_id])


def get_item(item_id: str) -> dict | None:
    res = _collection.get(ids=[item_id], include=["metadatas"])
    metas = res.get("metadatas") or []
    return metas[0] if metas else None


def list_items() -> list[dict]:
    res = _collection.get(include=["metadatas"])
    return list(res.get("metadatas") or [])


def search(embedding: list[float], n: int) -> list[dict]:
    """Return up to `n` nearest items, each metadata dict augmented with `score`."""
    total = count()
    if total == 0:
        return []
    n = max(1, min(n, total))
    res = _collection.query(
        query_embeddings=[embedding],
        n_results=n,
        include=["metadatas", "distances"],
    )
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    items: list[dict] = []
    for meta, dist in zip(metas, dists):
        item = dict(meta)
        item["score"] = 1.0 - float(dist)  # cosine distance -> similarity
        items.append(item)
    return items


def count() -> int:
    return _collection.count()
