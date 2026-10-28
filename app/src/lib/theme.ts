/**
 * Two-theme support: each app ships its native identity plus a second rendering —
 * 'dark' (desk) and 'corporate' (light, client-meeting-safe). The active theme is
 * stamped on <html data-theme> so the app stylesheet switches token blocks, and the
 * choice persists per app alongside settings.
 */
import { safeStorage, type StorageLike } from './storage'

export type ThemeName = 'dark' | 'corporate'

const themeKey = (slug: string) => `ivf:${slug}:theme`

export function getStoredTheme(slug: string, storage: StorageLike | null = safeStorage()): ThemeName | null {
  try {
    const v = storage?.getItem(themeKey(slug))
    return v === 'dark' || v === 'corporate' ? v : null
  } catch {
    return null
  }
}

export function setStoredTheme(slug: string, theme: ThemeName, storage: StorageLike | null = safeStorage()): void {
  try {
    storage?.setItem(themeKey(slug), theme)
  } catch {
    /* guarded */
  }
}

export function applyTheme(theme: ThemeName, doc: Document = document): void {
  doc.documentElement.setAttribute('data-theme', theme)
}
