# Wardrobe — notes for AI agents

Semantic clothing search: FastAPI + ChromaDB backend (`backend/`), React + Vite
frontend (`frontend/`). See README.md for architecture and configuration.

## Commands

- Backend tests: `cd backend && uv run pytest` — fully offline (demo-mode
  embeddings, temp data dir), no API key needed
- Frontend check: `cd frontend && pnpm build` (tsc + vite)
- Run the stack: `docker compose up --build`
- CI (`.github/workflows/ci.yml`) runs both jobs on every push to `main` and every PR

## Rules

Topic-specific rules live in `.claude/rules/` (auto-loaded):

- `testing.md` — behavior-first testing rules and how `backend/tests/` implements them
