import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies API + image requests to the FastAPI backend on :8000,
// so the frontend can use same-origin relative URLs (no CORS in dev).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/images': 'http://localhost:8000',
    },
  },
})
