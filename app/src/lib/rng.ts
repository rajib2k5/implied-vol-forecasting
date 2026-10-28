/**
 * Seeded pseudo-randomness. Nothing in this app may call Math.random(): every number a test
 * or a chart depends on has to be reproducible from a seed.
 */

/** mulberry32 — 32-bit state PRNG, uniform on [0, 1). */
export function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Box–Muller over a uniform source, returning standard normals.
 * The transform produces two normals per pair of uniforms; the spare is cached
 * so the stream stays deterministic regardless of how many draws are taken.
 */
export function gaussianFrom(rand: () => number): () => number {
  let spare: number | null = null
  return () => {
    if (spare !== null) {
      const s = spare
      spare = null
      return s
    }
    // log(0) is -Infinity: resample the (astronomically rare) exact zero.
    let u = rand()
    while (u === 0) u = rand()
    const v = rand()
    const radius = Math.sqrt(-2 * Math.log(u))
    const theta = 2 * Math.PI * v
    spare = radius * Math.sin(theta)
    return radius * Math.cos(theta)
  }
}
