"""Application configuration and shared filesystem paths."""
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ directory (parent of the app/ package)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
CHROMA_DIR = DATA_DIR / "chroma"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Auth. Provide the key EITHER directly via GEMINI_API_KEY (simple, for local
    # dev) OR — preferred — via GEMINI_API_KEY_FILE pointing at a file to read it
    # from (e.g. a Docker secret at /run/secrets/gemini_api_key). The file path
    # wins nothing over an explicit key: an inline GEMINI_API_KEY takes priority.
    # When neither is set, the app falls back to deterministic fake embeddings so
    # the whole stack still runs end-to-end (useful for local UI development).
    gemini_api_key: str | None = None
    gemini_api_key_file: str | None = None

    # Embedding model + output dimensionality (Matryoshka: 128-3072).
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = 1536

    # Search + image handling.
    default_top_n: int = 12
    max_image_edge: int = 1024  # longest edge (px) of the image sent for embedding

    # Display calibration: text↔image cosine lives in a compressed band, so a
    # strong match reads as a mediocre "50%". These map the raw cosine range
    # [floor, ceiling] onto 0–100% "relevance" for display. Monotonic — it does
    # NOT change result ranking. Tune to your own data if needed.
    relevance_floor: float = 0.30
    relevance_ceiling: float = 0.65

    @model_validator(mode="after")
    def _read_key_from_file(self) -> "Settings":
        if not (self.gemini_api_key and self.gemini_api_key.strip()) and self.gemini_api_key_file:
            path = Path(self.gemini_api_key_file)
            if path.is_file():
                self.gemini_api_key = path.read_text(encoding="utf-8").strip()
        return self

    @property
    def use_real_embeddings(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Settings()
