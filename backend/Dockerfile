# syntax=docker/dockerfile:1

# --- Stage 1: build the virtual environment with uv ---
FROM python:3.13-slim-bookworm AS builder

# Grab just the uv binary — no need for the full uv base image.
COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev \
    # chromadb pulls these in only for its default (local) embedding function
    # and server mode — neither of which we use, since we pass our own Gemini
    # embeddings. Removing them trims ~130 MB from the venv.
    && uv pip uninstall --python /app/.venv/bin/python \
        onnxruntime kubernetes tokenizers hf-xet huggingface-hub

# --- Stage 2: minimal runtime (no uv, no build cache) ---
FROM python:3.13-slim-bookworm

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Create the user first so COPY can set ownership inline (avoids a second,
# duplicate layer for the venv from a recursive chown).
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser app ./app

# data/ holds uploaded images + the Chroma DB (mount a volume here).
RUN mkdir -p /app/data && chown appuser:appuser /app/data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
