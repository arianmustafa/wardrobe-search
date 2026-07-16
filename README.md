# Wardrobe — Semantic Clothing Search

Upload photos of your clothes, embed them with Google's multimodal
`gemini-embedding-2` model, and search your wardrobe with natural language
(e.g. _"navy linen shirt for summer"_). Images and text queries live in the same
vector space, so a text query retrieves matching garment photos directly.

- **Backend:** FastAPI + [ChromaDB](https://www.trychroma.com/) (local, on-disk vector store)
- **Embeddings:** `gemini-embedding-2` via the `google-genai` SDK (Google AI Studio API key)
- **Frontend:** React + Vite + TypeScript single-page app

```
wardrobe/
├── docker-compose.yml         build + run both services with one command
├── docker-compose.secrets.yml optional overlay: read the key from a file secret
├── secrets/                   holds the gitignored key file (opt-in hardening)
├── backend/                   FastAPI app, embeddings, vector store
│   ├── app/
│   ├── Dockerfile
│   └── pyproject.toml
└── frontend/                  React + Vite + TypeScript SPA
    ├── Dockerfile             multi-stage build → nginx (also proxies /api + /images)
    └── nginx.conf
```

First, get a free Google AI Studio API key at <https://aistudio.google.com/apikey>.
Without one the app runs in **demo mode** using placeholder embeddings — the UI
works, but search results are not semantically meaningful.

## Quick start with Docker (recommended)

```bash
cp .env.example .env          # then set GEMINI_API_KEY=... in .env
docker compose up --build
```

- App: <http://localhost:8080>
- API + interactive docs: <http://localhost:8000/docs>

Uploaded images and the vector DB persist in the `wardrobe-data` Docker volume.
Stop with `docker compose down` (add `-v` to also wipe the stored wardrobe). If
port 8080 or 8000 is already in use, set `FRONTEND_PORT` / `BACKEND_PORT` in `.env`.
The backend port is published on `127.0.0.1` only, so the raw API/docs are not
exposed publicly — the app is served through the frontend on the web port.

## API key & secrets

The key is read from **either** an env var **or** a file, in that order:

- **Default — env var (`.env`).** `GEMINI_API_KEY` in `.env` is injected into the
  backend container. Simplest, and the usual self-hosting convention.
- **Opt-in — file secret (more hardened).** Keeps the key out of the environment
  and out of `docker inspect`. Overlay `docker-compose.secrets.yml`:

  ```bash
  printf '%s' 'YOUR_KEY' > secrets/gemini_api_key.txt
  chmod 700 secrets && chmod 444 secrets/gemini_api_key.txt
  docker compose -f docker-compose.yml -f docker-compose.secrets.yml up -d --build
  ```

Whichever you use, **restrict the key at Google** (Cloud console → API key →
restrict to the *Generative Language API*), set a **quota + billing alert**, and
rotate it. A restricted, capped key limits the damage if it ever leaks. Never
commit `.env` or `secrets/*` (both are gitignored).

## Run locally without Docker

Requires [uv](https://docs.astral.sh/uv/) and [pnpm](https://pnpm.io/).

**Backend** (terminal 1):

```bash
cd backend
uv sync
cp .env.example .env          # set GEMINI_API_KEY=...
uv run uvicorn app.main:app --reload --port 8000
```

**Frontend** (terminal 2):

```bash
cd frontend
pnpm install
pnpm dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api` and `/images`
to the backend, so no CORS setup is needed.

## Deploying to a VPS

The same `docker compose up -d --build` works on a server. Notes:

- Put `GEMINI_API_KEY` in `.env` (or use the file-secret overlay above).
- Only the frontend port is public; the backend API/docs bind to `127.0.0.1`.
- Set `FRONTEND_PORT` as needed and front it with a TLS-terminating reverse proxy
  (e.g. Caddy or Traefik with Let's Encrypt) for HTTPS.

> ⚠️ **No authentication yet.** The app has no login — anyone who can reach the
> frontend can view, upload, and delete items. Before exposing it publicly,
> restrict access: keep it on a private network (Tailscale/WireGuard/SSH tunnel),
> or put basic-auth on the reverse proxy. Adding app-level auth is a planned
> follow-up.

## API reference

| Method   | Path                | Purpose                                   |
| -------- | ------------------- | ----------------------------------------- |
| `POST`   | `/api/items`        | Upload an image (`file`, optional `title`); embeds + stores it |
| `GET`    | `/api/items`        | List all wardrobe items                   |
| `GET`    | `/api/search?q=&n=` | Semantic search, returns top-`n` matches with scores |
| `DELETE` | `/api/items/{id}`   | Remove an item and its image              |
| `GET`    | `/api/health`       | Status, item count, active embedding model |

## Configuration

Env vars, from the root `.env` (Docker) or `backend/.env` (local dev):

| Variable             | Default              | Notes                                          |
| -------------------- | -------------------- | ---------------------------------------------- |
| `GEMINI_API_KEY`     | _(none → demo mode)_ | Google AI Studio key                           |
| `GEMINI_API_KEY_FILE`| _(none)_             | Path to read the key from (used by the secret overlay); ignored if `GEMINI_API_KEY` is set |
| `EMBEDDING_MODEL`    | `gemini-embedding-2` | Multimodal embedding model                     |
| `EMBEDDING_DIM`      | `1536`               | Output dimensionality (128–3072, Matryoshka)   |
| `DEFAULT_TOP_N`      | `12`                 | Default number of search results               |
| `MAX_IMAGE_EDGE`     | `1024`               | Longest edge (px) of the image sent to embed   |
| `RELEVANCE_FLOOR`    | `0.30`               | Cosine mapped to 0% relevance (display only)   |
| `RELEVANCE_CEILING`  | `0.65`               | Cosine mapped to 100% relevance (display only) |

Vectors are L2-normalized and stored in ChromaDB using cosine distance. The
search response returns both the raw `score` (cosine) and a calibrated
`relevance` (the `[floor, ceiling]` band mapped to 0–1); the UI shows the latter.
The mapping is monotonic, so it never changes result ranking.

## License

MIT — see [LICENSE](LICENSE).
