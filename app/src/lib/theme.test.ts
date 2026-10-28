import { describe, it, expect } from 'vitest'
import { getStoredTheme, setStoredTheme, applyTheme } from './theme'
import type { StorageLike } from './storage'

function memStorage(): StorageLike {
  const map = new Map<string, string>()
  return {
    getItem: (k) => (map.has(k) ? (map.get(k) as string) : null),
    setItem: (k, v) => void map.set(k, v),
    removeItem: (k) => void map.delete(k),
  }
}

describe('theme storage', () => {
  it('round-trips and namespaces by slug', () => {
    const st = memStorage()
    expect(getStoredTheme('a', st)).toBeNull()
    setStoredTheme('a', 'corporate', st)
    expect(getStoredTheme('a', st)).toBe('corporate')
    expect(getStoredTheme('b', st)).toBeNull()
  })
  it('rejects corrupt values and survives unavailable storage', () => {
    const st = memStorage()
    st.setItem('ivf:a:theme', 'neon')
    expect(getStoredTheme('a', st)).toBeNull()
    expect(getStoredTheme('a', null)).toBeNull()
    expect(() => setStoredTheme('a', 'dark', null)).not.toThrow()
  })
  it('applyTheme stamps the document element', () => {
    applyTheme('corporate')
    expect(document.documentElement.getAttribute('data-theme')).toBe('corporate')
    applyTheme('dark')
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
  })
})
