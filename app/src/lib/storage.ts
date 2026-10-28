/**
 * Browser-storage plumbing for the theme preference.
 *
 * Keys are namespaced per app slug (`ivf:<slug>:theme`) so that
 * several apps served from one origin never read each other's values. Every storage access
 * is guarded: private windows or blocked site data must degrade to the in-memory fallback,
 * never throw.
 */

export interface StorageLike {
  getItem(k: string): string | null
  setItem(k: string, v: string): void
  removeItem(k: string): void
}

/**
 * In-memory fallback used when localStorage is missing or blocked (private windows,
 * storage-blocking browsers, Node's method-less experimental stub). The theme then
 * still works for the session — it just doesn't persist across reloads.
 */
const memMap = new Map<string, string>()
export const memoryFallback: StorageLike = {
  getItem: (k) => (memMap.has(k) ? (memMap.get(k) as string) : null),
  setItem: (k, v) => void memMap.set(k, v),
  removeItem: (k) => void memMap.delete(k),
}
export function resetMemoryFallback(): void {
  memMap.clear()
}

export function safeStorage(): StorageLike {
  try {
    const s = window.localStorage
    const probe = '__ivf_probe__'
    s.setItem(probe, '1')
    s.removeItem(probe)
    return s
  } catch {
    return memoryFallback
  }
}
