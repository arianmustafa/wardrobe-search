import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, FormEvent } from 'react'
import { deleteItem, health, listItems, search as apiSearch, uploadItem } from './api'
import type { HealthResponse, Item, SearchResultItem } from './api'

type DisplayItem = Item | SearchResultItem

const SUGGESTIONS = [
  'navy linen shirt for summer',
  'cozy knit for cold evenings',
  'office-ready blazer',
  'something for a beach day',
]

const IMAGE_EXT = /\.(avif|bmp|gif|heic|jpe?g|png|tiff?|webp)$/i

// Items uploaded without a title fall back to their filename on the backend;
// turn "navy-linen-shirt.jpg" into "navy linen shirt" and hash-named files
// into a graceful placeholder instead of showing raw hex in the caption.
function displayTitle(title: string): { text: string; untitled: boolean } {
  const trimmed = title.trim()
  if (!trimmed) return { text: 'Untitled', untitled: true }
  if (!IMAGE_EXT.test(trimmed)) return { text: trimmed, untitled: false }
  const stem = trimmed.replace(IMAGE_EXT, '')
  if (/^[0-9a-f-]{12,}$/i.test(stem)) return { text: 'Untitled', untitled: true }
  return { text: stem.replace(/[-_]+/g, ' '), untitled: false }
}

export default function App() {
  const [items, setItems] = useState<Item[]>([])
  const [results, setResults] = useState<SearchResultItem[] | null>(null) // null = browsing
  const [query, setQuery] = useState('')
  const [activeQuery, setActiveQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [searching, setSearching] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState('')
  const [status, setStatus] = useState<HealthResponse | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const refresh = useCallback(async () => {
    try {
      const data = await listItems()
      setItems(data.items)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  useEffect(() => {
    refresh().finally(() => setLoading(false))
    health().then(setStatus).catch(() => {})
  }, [refresh])

  const runSearch = useCallback(async (q: string) => {
    setSearching(true)
    setError('')
    try {
      const data = await apiSearch(q)
      setResults(data.results)
      setActiveQuery(q)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSearching(false)
    }
  }, [])

  function onSearch(e: FormEvent) {
    e.preventDefault()
    const q = query.trim()
    if (!q) {
      clearSearch()
      return
    }
    runSearch(q)
  }

  function onSuggestion(s: string) {
    setQuery(s)
    runSearch(s)
  }

  function clearSearch() {
    setQuery('')
    setActiveQuery('')
    setResults(null)
    setError('')
  }

  async function handleFiles(fileList: FileList | null) {
    if (uploading || !fileList) return
    const files = Array.from(fileList).filter((f) => f.type.startsWith('image/'))
    if (!files.length) return
    setUploading(true)
    setError('')
    setProgress({ done: 0, total: files.length })
    let uploaded = 0
    try {
      for (let i = 0; i < files.length; i++) {
        await uploadItem(files[i])
        uploaded += 1
        setProgress({ done: i + 1, total: files.length })
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      if (uploaded > 0) {
        // Return to the wardrobe view so freshly-added items are always visible
        // — even if the upload happened while search results were showing, and
        // even if a later file in the batch failed after earlier ones succeeded.
        setResults(null)
        setQuery('')
        setActiveQuery('')
        await refresh()
      }
      setUploading(false)
      setProgress(null)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  async function onDelete(id: string) {
    try {
      await deleteItem(id)
      setItems((prev) => prev.filter((i) => i.id !== id))
      setResults((prev) => (prev ? prev.filter((i) => i.id !== id) : prev))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragOver(false)
    handleFiles(e.dataTransfer.files)
  }

  const grid: DisplayItem[] = results !== null ? results : items
  const browsing = results === null

  const ticker = [
    'Semantic closet archive',
    `${items.length} piece${items.length === 1 ? '' : 's'} indexed`,
    status ? (status.real_embeddings ? status.model : 'demo mode — placeholder vectors') : null,
    status ? `${status.embedding_dim}-d cosine space` : null,
    'text ⇄ image retrieval',
  ]
    .filter(Boolean)
    .join(' ✦ ')

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Wardrobe<sup>®</sup>
        </div>
        {status && (
          <span
            className={`status ${status.real_embeddings ? 'status-live' : 'status-demo'}`}
            title={
              status.real_embeddings
                ? `Embedding with ${status.model}`
                : 'No GEMINI_API_KEY set — using placeholder embeddings'
            }
          >
            {status.real_embeddings ? `● live — ${status.model}` : '○ demo mode'}
          </span>
        )}
      </header>

      <div className="ticker" aria-hidden="true">
        <div className="ticker-track">
          <span>{ticker}&ensp;✦&ensp;</span>
          <span>{ticker}&ensp;✦&ensp;</span>
        </div>
      </div>

      <section className="hero">
        <h1>
          <span className="hero-line">Describe it.</span>
          <span className="hero-line hero-outline">Wear it.</span>
        </h1>
        <p className="hero-note">
          Photograph each piece once. Then pull it up the way you actually think about it —
          “warm wool for rainy days”, “something for a beach day”.
        </p>
      </section>

      <form className="searchbar" onSubmit={onSearch}>
        <input
          type="search"
          placeholder="warm wool for rainy days"
          value={query}
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
            const value = e.target.value
            setQuery(value)
            // Emptying the field (incl. the native ✕) returns to browsing, like Clear.
            if (!value.trim()) clearSearch()
          }}
        />
        {!browsing && (
          <button type="button" className="ghost" onClick={clearSearch}>
            Clear
          </button>
        )}
        <button type="submit" disabled={searching}>
          {searching ? 'Searching…' : 'Search →'}
        </button>
      </form>

      <div className="chips">
        <span className="chips-label">Try:</span>
        {SUGGESTIONS.map((s) => (
          <button
            type="button"
            className="chip"
            key={s}
            disabled={searching}
            onClick={() => onSuggestion(s)}
          >
            {s}
          </button>
        ))}
      </div>

      <div
        className={`dropzone ${dragOver ? 'dropzone-active' : ''} ${uploading ? 'dropzone-busy' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          if (!uploading) setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => {
          if (!uploading) fileRef.current?.click()
        }}
      >
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          disabled={uploading}
          onChange={(e: ChangeEvent<HTMLInputElement>) => handleFiles(e.target.files)}
        />
        {uploading ? (
          <div className="upload-progress">
            <span>
              {progress && progress.total > 1
                ? `Uploading & embedding… ${progress.done} / ${progress.total}`
                : 'Uploading & embedding…'}
            </span>
            <div className="progressbar">
              <div
                className="progressbar-fill"
                style={{ width: progress ? `${(progress.done / progress.total) * 100}%` : '0%' }}
              />
              <div className="progressbar-stripe" />
            </div>
          </div>
        ) : (
          <>
            <span className="dz-plus">+</span>
            <span className="dz-label">Add pieces</span>
            <span className="dz-sub">drop images or click to browse</span>
          </>
        )}
      </div>

      {error && <div className="error">Error: {error}</div>}

      <div className="section-head">
        {browsing ? (
          <h2>
            Index <span className="count">({loading ? '–' : items.length})</span>
          </h2>
        ) : (
          <h2>
            Results: “{activeQuery}” <span className="count">({grid.length})</span>
          </h2>
        )}
        {!browsing && <span className="head-meta">sorted by relevance</span>}
      </div>

      {loading && browsing ? (
        <div className="grid" aria-hidden="true">
          {Array.from({ length: 8 }, (_, i) => (
            <div className="card skeleton" key={i} style={{ animationDelay: `${i * 40}ms` }}>
              <div className="card-img" />
              <div className="skeleton-caption" />
            </div>
          ))}
        </div>
      ) : grid.length === 0 ? (
        <div className="empty">
          {browsing ? (
            <>
              <h3>Nothing indexed.</h3>
              <p>Upload a few photos of your clothes and they become searchable in plain words.</p>
            </>
          ) : (
            <>
              <h3>No matches.</h3>
              <p>Try describing the piece differently — fabric, colour, occasion.</p>
              <button type="button" className="ghost" onClick={clearSearch}>
                Show everything
              </button>
            </>
          )}
        </div>
      ) : (
        <div
          className={`grid ${searching ? 'grid-busy' : ''}`}
          key={browsing ? 'browse' : activeQuery}
        >
          {grid.map((item, i) => {
            const result = 'relevance' in item ? item : null
            const title = displayTitle(item.title)
            return (
              <figure
                className="card"
                key={item.id}
                style={{ animationDelay: `${Math.min(i, 11) * 30}ms` }}
              >
                <div className="card-img">
                  <img src={item.image_url} alt={title.text} loading="lazy" />
                  {result && i === 0 && <span className="tag-best">Best match</span>}
                  <button
                    className="delete"
                    title="Remove"
                    aria-label={`Remove ${title.text}`}
                    onClick={() => onDelete(item.id)}
                  >
                    ✕
                  </button>
                </div>
                <figcaption>
                  <span className="idx">{String(i + 1).padStart(2, '0')}</span>
                  <span className={`ttl ${title.untitled ? 'untitled' : ''}`}>{title.text}</span>
                  {result && (
                    <span
                      className="pct"
                      title={`relevance ${(result.relevance * 100).toFixed(0)}% · cosine ${result.score.toFixed(3)}`}
                    >
                      {(result.relevance * 100).toFixed(0)}%
                    </span>
                  )}
                </figcaption>
              </figure>
            )
          })}
        </div>
      )}

      <footer className="footer">
        <span>Wardrobe® — semantic closet archive</span>
        {status && (
          <span>
            {items.length} piece{items.length === 1 ? '' : 's'} ·{' '}
            {status.real_embeddings ? status.model : 'placeholder vectors (demo)'}
          </span>
        )}
      </footer>
    </div>
  )
}
