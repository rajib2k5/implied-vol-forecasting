import { afterEach, describe, expect, it } from 'vitest'
import { memoryFallback, resetMemoryFallback, safeStorage } from './storage'

afterEach(() => {
  resetMemoryFallback()
})

describe('memoryFallback', () => {
  it('stores, reads and removes values; a missing key reads as null', () => {
    expect(memoryFallback.getItem('k')).toBeNull()
    memoryFallback.setItem('k', 'v1')
    expect(memoryFallback.getItem('k')).toBe('v1')
    memoryFallback.setItem('k', 'v2')
    expect(memoryFallback.getItem('k')).toBe('v2')
    memoryFallback.removeItem('k')
    expect(memoryFallback.getItem('k')).toBeNull()
  })
  it('resetMemoryFallback clears every key', () => {
    memoryFallback.setItem('a', '1')
    memoryFallback.setItem('b', '2')
    resetMemoryFallback()
    expect(memoryFallback.getItem('a')).toBeNull()
    expect(memoryFallback.getItem('b')).toBeNull()
  })
})

describe('safeStorage', () => {
  it('returns the real localStorage when it works, and leaves no probe key behind', () => {
    const s = safeStorage()
    expect(s).toBe(window.localStorage)
    expect(s.getItem('__ivf_probe__')).toBeNull()
  })
  it('falls back to the in-memory store when localStorage access throws (blocked site data)', () => {
    const original = Object.getOwnPropertyDescriptor(window, 'localStorage')
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      get() {
        throw new Error('SecurityError')
      },
    })
    try {
      expect(safeStorage()).toBe(memoryFallback)
    } finally {
      if (original) Object.defineProperty(window, 'localStorage', original)
      else delete (window as unknown as Record<string, unknown>).localStorage
    }
  })
  it('falls back when setItem throws (quota or private-mode storage)', () => {
    const spy = Storage.prototype.setItem
    Storage.prototype.setItem = () => {
      throw new Error('QuotaExceededError')
    }
    try {
      expect(safeStorage()).toBe(memoryFallback)
    } finally {
      Storage.prototype.setItem = spy
    }
  })
})
