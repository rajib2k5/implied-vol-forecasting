import { describe, it, expect } from 'vitest'
import { mulberry32, gaussianFrom } from './rng'

describe('mulberry32', () => {
  it('matches an independent 32-bit implementation (Python, written separately from this file)', () => {
    // Oracle: the same algorithm re-implemented in Python with explicit 32-bit masking
    // (imul as (a*b) & 0xFFFFFFFF), outputs printed with repr(). Not derived from this module.
    const one = mulberry32(1)
    expect([one(), one(), one()]).toEqual([0.6270739405881613, 0.002735721180215478, 0.5274470399599522])
    const answer = mulberry32(42)
    expect([answer(), answer(), answer()]).toEqual([0.6011037519201636, 0.44829055899754167, 0.8524657934904099])
  })
  it('is reproducible from a seed and differs between seeds', () => {
    const a = mulberry32(42)
    const b = mulberry32(42)
    expect([a(), a(), a()]).toEqual([b(), b(), b()])
    expect(mulberry32(1)()).not.toBe(mulberry32(2)())
  })
  it('stays inside [0, 1)', () => {
    const rand = mulberry32(7)
    for (let i = 0; i < 5000; i++) {
      const v = rand()
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThan(1)
    }
  })
  it('treats the seed as a 32-bit unsigned value', () => {
    // 2^32 wraps to 0 under >>> 0, so both seeds start the same stream.
    expect(mulberry32(0)()).toBe(mulberry32(2 ** 32)())
  })
})

describe('gaussianFrom', () => {
  it('is reproducible and returns both halves of each Box-Muller pair', () => {
    const a = gaussianFrom(mulberry32(9))
    const b = gaussianFrom(mulberry32(9))
    // Four draws forces the cached-spare path twice.
    expect([a(), a(), a(), a()]).toEqual([b(), b(), b(), b()])
  })
  it('hand-worked pair: u=0.5, v=0.25 gives radius sqrt(2 ln 2) and theta = pi/2', () => {
    // radius = sqrt(-2 ln 0.5) = sqrt(2 * 0.693147) = sqrt(1.386294) = 1.177410
    // theta  = 2 pi * 0.25 = pi/2  ->  cos = 0, sin = 1
    // first draw  = radius * cos(theta) = 0 ; the cached spare = radius * sin(theta) = 1.177410
    const queue = [0.5, 0.25]
    const draw = gaussianFrom(() => queue.shift() ?? 0.5)
    expect(draw()).toBeCloseTo(0, 12)
    expect(draw()).toBeCloseTo(1.17741, 5)
  })
  it('has approximately zero mean and unit variance over a large sample', () => {
    const draw = gaussianFrom(mulberry32(20260830))
    const n = 200_000
    let sum = 0
    let sumSq = 0
    for (let i = 0; i < n; i++) {
      const z = draw()
      expect(Number.isFinite(z)).toBe(true)
      sum += z
      sumSq += z * z
    }
    const mean = sum / n
    const variance = sumSq / n - mean * mean
    // 3 standard errors of the mean for n = 200k is ~0.0067.
    expect(Math.abs(mean)).toBeLessThan(0.01)
    expect(variance).toBeGreaterThan(0.97)
    expect(variance).toBeLessThan(1.03)
  })
  it('resamples a uniform of exactly 0 instead of returning -Infinity', () => {
    const queue = [0, 0.5, 0.125]
    const draw = gaussianFrom(() => queue.shift() ?? 0.5)
    const z = draw()
    expect(Number.isFinite(z)).toBe(true)
    // u = 0 is skipped, so the pair is built from (u, v) = (0.5, 0.125):
    // radius = sqrt(2 ln 2) = 1.177410, theta = 2 pi * 0.125 = pi/4, cos = 0.707107
    // first draw = 1.177410 * 0.707107 = 0.832555
    expect(z).toBeCloseTo(0.832555, 5)
  })
})
