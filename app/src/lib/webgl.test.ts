import { describe, expect, it } from 'vitest'
import { detectWebGL } from './webgl'

/** A fake document whose canvas answers getContext from a table of context-type -> result. */
function docWith(contexts: Record<string, unknown>, calls: string[] = []) {
  return {
    createElement: () =>
      ({
        getContext: (type: string) => {
          calls.push(type)
          return contexts[type] ?? null
        },
      }) as unknown as HTMLCanvasElement,
  }
}

describe('detectWebGL', () => {
  it('is true when WebGL 2 is available, without needing to ask for WebGL 1', () => {
    const calls: string[] = []
    expect(detectWebGL(docWith({ webgl2: {} }, calls))).toBe(true)
    expect(calls).toEqual(['webgl2'])
  })
  it('falls back to WebGL 1', () => {
    const calls: string[] = []
    expect(detectWebGL(docWith({ webgl: {} }, calls))).toBe(true)
    expect(calls).toEqual(['webgl2', 'webgl'])
  })
  it('is false when neither context can be created', () => {
    expect(detectWebGL(docWith({}))).toBe(false)
  })
  it('is false when getContext returns undefined rather than null', () => {
    const doc = { createElement: () => ({ getContext: () => undefined }) as unknown as HTMLCanvasElement }
    expect(detectWebGL(doc)).toBe(false)
  })
  it('is false, not an exception, when the browser throws while probing', () => {
    const throwingContext = { createElement: () => ({ getContext: () => { throw new Error('blocked') } }) as unknown as HTMLCanvasElement }
    expect(detectWebGL(throwingContext)).toBe(false)
    const throwingElement = { createElement: () => { throw new Error('no canvas') } }
    expect(detectWebGL(throwingElement)).toBe(false)
  })
})
