# syntax=docker/dockerfile:1

# --- Stage 1: build the static SPA (type-check + Vite bundle) ---
FROM node:22-bookworm-slim AS build
WORKDIR /app
RUN corepack enable
# pnpm version comes from the "packageManager" field in package.json.
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm build

# --- Stage 2: serve with nginx + reverse-proxy /api and /images to the backend ---
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
