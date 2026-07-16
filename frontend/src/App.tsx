import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, FormEvent } from 'react'
import { deleteItem, health, listItems, search as apiSearch, uploadItem } from './api'
import type { HealthResponse, Item, SearchResultItem } from './api'

type DisplayItem = Item | SearchResultItem

export default function App() {
  const [items, setItems] = useState<Item[]>([])
  const [results, setResults] = useState<SearchResultItem[] | null>(null) // null = browsing
  const [query, setQuery] = useState('')
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
    refresh()
    health().then(setStatus).catch(() => {})
  }, [refresh])

  async function onSearch(e: FormEvent) {
    e.preventDefault()
    const q = query.trim()
    if (!q) {
      setResults(null)
      return
    }
    setSearching(true)
    setError('')
    try {
      const data = await apiSearch(q)
      setResults(data.results)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSearching(false)
    }
  }

  function clearSearch() {
    setQuery('')
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

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Wardrobe</h1>
          <p className="tagline">Semantic search over your clothes</p>
        </div>
        {status && (
          <span
            className={`badge ${status.real_embeddings ? 'badge-live' : 'badge-demo'}`}
            title={
              status.real_embeddings
                ? `Embedding with ${status.model}`
                : 'No GEMINI_API_KEY set — using placeholder embeddings'
            }
          >
            {status.real_embeddings ? status.model : 'demo mode'}
          </span>
        )}
      </header>

      <form className="searchbar" onSubmit={onSearch}>
        <input
          type="search"
          placeholder="Search your wardrobe — e.g. “navy linen shirt for summer”"
          value={query}
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
            const value = e.target.value
            setQuery(value)
            // Emptying the field (incl. the native ✕) returns to browsing, like Clear.
            if (!value.trim()) clearSearch()
          }}
        />
        <button type="submit" disabled={searching}>
          {searching ? 'Searching…' : 'Search'}
        </button>
        {!browsing && (
          <button type="button" className="ghost" onClick={clearSearch}>
            Clear
          </button>
        )}
      </form>

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
            <span className="dropzone-icon">＋</span>
            <span>Drop clothing photos here, or click to upload</span>
          </>
        )}
      </div>

      {error && <div className="error">{error}</div>}

      <div className="section-title">
        {browsing ? (
          <span>
            Your wardrobe <span className="muted">· {items.length} items</span>
          </span>
        ) : (
          <span>
            Results for “{query}” <span className="muted">· {grid.length} matches</span>
          </span>
        )}
      </div>

      {grid.length === 0 ? (
        <div className="empty">
          {browsing
            ? 'No items yet — upload some photos of your clothes to get started.'
            : 'No matches found.'}
        </div>
      ) : (
        <div className="grid">
          {grid.map((item) => {
            const result = 'relevance' in item ? item : null
            return (
              <figure className="card" key={item.id}>
                <div className="card-img">
                  <img src={item.image_url} alt={item.title} loading="lazy" />
                  {result && (
                    <span
                      className="score"
                      title={`relevance ${(result.relevance * 100).toFixed(0)}% · cosine ${result.score.toFixed(3)}`}
                    >
                      {(result.relevance * 100).toFixed(0)}%
                    </span>
                  )}
                  <button className="delete" title="Remove" onClick={() => onDelete(item.id)}>
                    ×
                  </button>
                </div>
                <figcaption>{item.title}</figcaption>
              </figure>
            )
          })}
        </div>
      )}
    </div>
  )
}
