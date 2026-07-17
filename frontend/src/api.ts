// Thin, typed fetch wrapper around the FastAPI backend.

export interface Item {
  id: string
  filename: string
  original_name: string | null
  title: string
  uploaded_at: string | null
  image_url: string
}

export interface SearchResultItem extends Item {
  score: number // raw cosine similarity
  relevance: number // calibrated 0–1 for display
}

export interface ItemsResponse {
  items: Item[]
  count: number
}

export interface SearchResponse {
  query: string
  results: SearchResultItem[]
}

export interface HealthResponse {
  status: string
  items: number
  model: string
  embedding_dim: number
  real_embeddings: boolean
}

async function request<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = (await res.json()) as { detail?: string }
      detail = body.detail ?? detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return (res.status === 204 ? null : await res.json()) as T
}

export function listItems(): Promise<ItemsResponse> {
  return fetch('/api/items', { cache: 'no-store' }).then((r) => request<ItemsResponse>(r))
}

export function search(q: string, n?: number): Promise<SearchResponse> {
  // Only send `n` when the caller asks for it, so the backend's configured
  // default (DEFAULT_TOP_N) applies otherwise.
  const params = new URLSearchParams({ q })
  if (n !== undefined) params.set('n', String(n))
  return fetch(`/api/search?${params}`, { cache: 'no-store' }).then((r) =>
    request<SearchResponse>(r)
  )
}

export function uploadItem(file: File, title?: string): Promise<Item> {
  const fd = new FormData()
  fd.append('file', file)
  if (title) fd.append('title', title)
  return fetch('/api/items', { method: 'POST', body: fd }).then((r) => request<Item>(r))
}

export function deleteItem(id: string): Promise<void> {
  return fetch(`/api/items/${id}`, { method: 'DELETE' }).then((r) => request<void>(r))
}

export function health(): Promise<HealthResponse> {
  return fetch('/api/health').then((r) => request<HealthResponse>(r))
}
