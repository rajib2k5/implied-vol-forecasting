/**
 * Term-structure data for the 3D ribbon: types, a seeded synthetic generator, a validating
 * parser, and the pure geometry math that the React Three Fiber component only renders.
 *
 * Why the maths lives here and not in the component: jsdom (the unit-test DOM) cannot run
 * WebGL, so anything that decides what the ribbon looks like (vertex positions, triangle
 * indices, contango / backwardation colours) is a plain function of the data that can be
 * tested with hand-worked numbers. The component just hands the arrays to three.js.
 *
 * The data here is SYNTHETIC. CBOE index history is not redistributed with this project
 * (personal, non-commercial terms), so the app ships a generated fixture instead and says so.
 */
import { gaussianFrom, mulberry32 } from './rng.ts' // explicit .ts: scripts/genFixture.ts imports this file under plain Node

/** Version of the JSON payload shape. Provisional until the Python exporter fixes the contract. */
export const SCHEMA_VERSION = 1

/** Maturity buckets of the CBOE volatility indices (9-day ... 1-year), shortest first. */
export const TENORS = ['9D', '30D', '3M', '6M', '1Y'] as const
/** Calendar days to expiry behind each tenor label; only the curve's shape depends on these. */
export const TENOR_DAYS = [9, 30, 91, 182, 365] as const

/**
 * Where the data came from, spelled as in the Python exporter's envelope (ADR 0001 D2). The deploy
 * guard (scripts/guardDist.mjs) refuses to ship any JSON data file whose top-level `data_mode` is not
 * "synthetic", so a payload without this field can never reach a build.
 */
export type DataMode = 'synthetic' | 'real'

export interface TermStructureData {
  schema_version: number
  data_mode: DataMode
  /** True for generated data. A real export would say false. The UI labels the data by this flag. */
  synthetic: boolean
  description: string
  generator?: { name: string; seed: number }
  tenors: string[]
  /** ISO dates (yyyy-mm-dd), strictly ascending. */
  dates: string[]
  /** levels[dateIndex][tenorIndex]: implied volatility in annualised percentage points. */
  levels: number[][]
}

/* ---------------------------------------------------------------- dates */

function isIsoDate(s: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) return false
  const d = new Date(`${s}T00:00:00Z`)
  // Round-trip catches impossible dates such as 2024-02-30, which Date silently rolls over.
  return !Number.isNaN(d.getTime()) && d.toISOString().slice(0, 10) === s
}

/**
 * `count` consecutive weekdays starting on `startISO` (or the next weekday if it is a weekend).
 * Holidays are ignored: this only labels synthetic data. Uses UTC so the result never depends
 * on the machine's time zone or clock.
 */
export function businessDates(startISO: string, count: number): string[] {
  if (!isIsoDate(startISO)) throw new Error(`invalid start date: ${startISO}`)
  if (!Number.isInteger(count) || count < 0) throw new Error(`invalid count: ${count}`)
  const out: string[] = []
  const d = new Date(`${startISO}T00:00:00Z`)
  while (out.length < count) {
    const dow = d.getUTCDay()
    if (dow !== 0 && dow !== 6) out.push(d.toISOString().slice(0, 10))
    d.setUTCDate(d.getUTCDate() + 1)
  }
  return out
}

/* ------------------------------------------------------------ generator */

export interface GeneratorOptions {
  seed: number
  nDates: number
  startDate: string
  /** Level (vol points) the whole curve reverts to far out. */
  longRun: number
  /** e-folding horizon in calendar days: how fast the curve relaxes from spot to longRun. */
  reversionDays: number
  /** Calm-state spot vol (vol points); log-spot is pulled towards ln(spotMean). */
  spotMean: number
  /** Daily pull of log-spot back towards ln(spotMean), in [0, 1]. */
  kappa: number
  /** Daily standard deviation of the log-spot shock. */
  sigma: number
  /** Index of the date with a scripted volatility spike, or null for none. */
  shockDay: number | null
  /** Spot vol (vol points) the market is pinned to on shockDay. */
  shockLevel: number
  /** Standard deviation (vol points) of independent per-tenor noise. */
  noiseSd: number
}

export const FIXTURE_SEED = 20261027

export const DEFAULT_GENERATOR: GeneratorOptions = {
  seed: FIXTURE_SEED,
  nDates: 60,
  startDate: '2024-01-02',
  longRun: 20,
  reversionDays: 60,
  spotMean: 15,
  kappa: 0.1,
  sigma: 0.05,
  shockDay: 30,
  shockLevel: 30,
  noiseSd: 0.15,
}

/**
 * Implied vol at `tauDays` ahead when the instantaneous ("spot") vol is `spot`:
 * the curve relaxes exponentially from spot (tau = 0) towards `longRun` (tau -> infinity),
 *   level(tau) = longRun + (spot - longRun) * exp(-tau / reversionDays).
 * Spot below longRun gives an upward-sloping curve (contango); spot above gives backwardation.
 */
export function curveLevel(spot: number, longRun: number, tauDays: number, reversionDays: number): number {
  if (!(reversionDays > 0)) throw new Error('reversionDays must be positive')
  return longRun + (spot - longRun) * Math.exp(-tauDays / reversionDays)
}

const round2 = (v: number) => Math.round(v * 100) / 100

/**
 * Deterministic synthetic term structure: log-spot mean-reverts (Ornstein-Uhlenbeck, daily step)
 * with one scripted spike, and each tenor reads the curve above plus a little noise.
 *
 * Why the spike is pinned to a level instead of being a random jump: with defaults, day 0 starts
 * at spot 15 < longRun 20 (contango) and shockDay pins spot to 30 > 20 (backwardation), so every
 * seed yields both regimes and the colouring always has something to show. A random jump on top
 * of a random state could occasionally fail to push spot above the long-run level.
 */
export function generateTermStructure(overrides: Partial<GeneratorOptions> = {}): TermStructureData {
  const o = { ...DEFAULT_GENERATOR, ...overrides }
  if (!Number.isInteger(o.nDates) || o.nDates < 1) throw new Error('nDates must be a positive integer')
  if (!(o.longRun > 0) || !(o.spotMean > 0)) throw new Error('longRun and spotMean must be positive')
  if (!(o.kappa >= 0 && o.kappa <= 1)) throw new Error('kappa must be in [0, 1]')
  if (!(o.sigma >= 0) || !(o.noiseSd >= 0)) throw new Error('sigma and noiseSd must be non-negative')
  if (o.shockDay !== null && (!Number.isInteger(o.shockDay) || o.shockDay < 0 || o.shockDay >= o.nDates)) {
    throw new Error('shockDay must be null or an integer in [0, nDates)')
  }
  if (!(o.shockLevel > 0)) throw new Error('shockLevel must be positive')

  const normal = gaussianFrom(mulberry32(o.seed))
  const dates = businessDates(o.startDate, o.nDates)
  const logMean = Math.log(o.spotMean)
  let x = logMean // log of spot vol
  const levels: number[][] = []
  for (let d = 0; d < o.nDates; d++) {
    if (d > 0) x += o.kappa * (logMean - x) + o.sigma * normal()
    if (d === o.shockDay) x = Math.log(o.shockLevel)
    const spot = Math.exp(x)
    levels.push(TENOR_DAYS.map((tau) => round2(curveLevel(spot, o.longRun, tau, o.reversionDays) + o.noiseSd * normal())))
  }
  return {
    schema_version: SCHEMA_VERSION,
    data_mode: 'synthetic',
    synthetic: true,
    description:
      'Synthetic term structure: log-spot vol mean-reverts with one scripted spike; each tenor reads ' +
      'level = longRun + (spot - longRun) * exp(-days / reversionDays) plus noise. Not CBOE data.',
    generator: { name: 'ivf-synthetic-term-structure', seed: o.seed },
    tenors: [...TENORS],
    dates,
    levels,
  }
}

/* --------------------------------------------------------------- parser */

/**
 * Validate an untrusted payload (a JSON file) and return it typed. The app reads static JSON
 * produced elsewhere, so every structural assumption the rest of the code makes is checked here
 * once, with a message that says which part of the file is wrong.
 */
export function parseTermStructure(raw: unknown): TermStructureData {
  if (typeof raw !== 'object' || raw === null) throw new Error('term structure: expected an object')
  const r = raw as Record<string, unknown>
  if (r.schema_version !== SCHEMA_VERSION) {
    throw new Error(`term structure: schema_version ${SCHEMA_VERSION} required, got ${String(r.schema_version)}`)
  }
  if (r.data_mode !== 'synthetic' && r.data_mode !== 'real') {
    throw new Error('term structure: data_mode must be "synthetic" or "real"')
  }
  if (typeof r.synthetic !== 'boolean') throw new Error('term structure: synthetic must be a boolean')
  // Two flags for one fact must never disagree: the UI badge reads `synthetic`, the guard reads data_mode.
  if (r.synthetic !== (r.data_mode === 'synthetic')) throw new Error('term structure: synthetic contradicts data_mode')
  const { tenors, dates, levels } = r
  if (!Array.isArray(tenors) || tenors.length === 0 || !tenors.every((t) => typeof t === 'string')) {
    throw new Error('term structure: tenors must be a non-empty array of strings')
  }
  if (!Array.isArray(dates) || dates.length === 0 || !dates.every((d) => typeof d === 'string' && isIsoDate(d))) {
    throw new Error('term structure: dates must be a non-empty array of ISO dates (yyyy-mm-dd)')
  }
  for (let i = 1; i < dates.length; i++) {
    // ISO dates sort lexicographically, so string comparison is date comparison.
    if (!(dates[i - 1] < dates[i])) throw new Error('term structure: dates must be strictly ascending')
  }
  if (!Array.isArray(levels) || levels.length !== dates.length) {
    throw new Error('term structure: levels must have one row per date')
  }
  levels.forEach((row, i) => {
    if (!Array.isArray(row) || row.length !== tenors.length) {
      throw new Error(`term structure: levels[${i}] must have one value per tenor`)
    }
    row.forEach((v, j) => {
      if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) {
        throw new Error(`term structure: levels[${i}][${j}] must be a finite positive number`)
      }
    })
  })
  const gen = r.generator as { name?: unknown; seed?: unknown } | undefined
  return {
    schema_version: SCHEMA_VERSION,
    data_mode: r.data_mode,
    synthetic: r.synthetic,
    description: typeof r.description === 'string' ? r.description : '',
    ...(gen && typeof gen.name === 'string' && typeof gen.seed === 'number'
      ? { generator: { name: gen.name, seed: gen.seed } }
      : {}),
    tenors: tenors as string[],
    dates: dates as string[],
    levels: levels as number[][],
  }
}

/* ------------------------------------------------ contango / backwardation */

/** Curves flatter than this many vol points (end to end) count as flat, not as contango. */
export const FLAT_TOLERANCE = 0.05

export type CurveShape = 'contango' | 'backwardation' | 'flat'

/** Long end minus short end, in vol points: positive = upward-sloping (contango). */
export function termSlope(row: readonly number[]): number {
  if (row.length < 2) throw new Error('a curve needs at least two tenors')
  return row[row.length - 1] - row[0]
}

export function classifyCurve(row: readonly number[], tolerance = FLAT_TOLERANCE): CurveShape {
  const slope = termSlope(row)
  if (slope > tolerance) return 'contango'
  if (slope < -tolerance) return 'backwardation'
  return 'flat'
}

export function regimeCounts(data: TermStructureData): Record<CurveShape, number> {
  const counts: Record<CurveShape, number> = { contango: 0, backwardation: 0, flat: 0 }
  for (const row of data.levels) counts[classifyCurve(row)]++
  return counts
}

export function regimeSummary(data: TermStructureData): string {
  const c = regimeCounts(data)
  return `${data.dates.length} dates: ${c.contango} in contango, ${c.backwardation} in backwardation, ${c.flat} flat`
}

/* -------------------------------------------------------------- colours */

export type RGB = readonly [number, number, number]
/** Blue: upward-sloping curve. The pair is blue/orange so it survives red-green colour blindness. */
export const CONTANGO_RGB: RGB = [0.31, 0.56, 0.85] // ~#4f8fd9
/** Orange: inverted curve. */
export const BACKWARDATION_RGB: RGB = [0.88, 0.5, 0.23] // ~#e0803b
/** Light grey: flat curve. */
export const NEUTRAL_RGB: RGB = [0.78, 0.78, 0.78]

/** `#rrggbb` for DOM legends, so the legend swatches come from the same constants as the mesh. */
export function rgbToHex(rgb: RGB): string {
  return `#${rgb.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, '0')).join('')}`
}

/**
 * Diverging colour for a curve slope: grey at 0, fading to blue for slope >= +scale and to
 * orange for slope <= -scale, linearly in between.
 */
export function slopeColour(slope: number, scale: number): RGB {
  if (!(scale > 0)) throw new Error('scale must be positive')
  const s = Math.max(-1, Math.min(1, slope / scale))
  const target = s >= 0 ? CONTANGO_RGB : BACKWARDATION_RGB
  const w = Math.abs(s)
  return [
    NEUTRAL_RGB[0] + (target[0] - NEUTRAL_RGB[0]) * w,
    NEUTRAL_RGB[1] + (target[1] - NEUTRAL_RGB[1]) * w,
    NEUTRAL_RGB[2] + (target[2] - NEUTRAL_RGB[2]) * w,
  ]
}

/* ------------------------------------------------------------- geometry */

export interface RibbonOptions {
  /** Extent along x (tenor axis). */
  width: number
  /** Extent along z (time axis). */
  depth: number
  /** Extent along y (level axis). */
  height: number
  /** Slope (vol points) at which the colour is fully saturated. */
  slopeScale: number
}

export const DEFAULT_RIBBON: RibbonOptions = { width: 8, depth: 6, height: 3, slopeScale: 5 }

export interface RibbonGeometry {
  /** x, y, z per vertex. Vertex number = dateIndex * nTenors + tenorIndex. */
  positions: Float32Array
  /** r, g, b per vertex (0..1), one colour per date: the sign and size of that day's slope. */
  colors: Float32Array
  /** Three vertex numbers per triangle, counter-clockwise seen from above (+y). */
  indices: Uint32Array
  vertexCount: number
  triangleCount: number
  /** Long minus short end per date, vol points. */
  slopes: number[]
  levelRange: { min: number; max: number }
}

/**
 * Turn the date x tenor grid into a triangle mesh.
 *  - x runs over tenors (shortest at -width/2), z over dates (oldest at -depth/2, newest nearest
 *    the default camera), y is the level scaled into [0, height] between the grid's min and max.
 *  - Each grid cell (d, t) becomes two triangles: (a, c, b) and (b, c, e) where
 *    a = (d,t), b = (d,t+1), c = (d+1,t), e = (d+1,t+1).
 *  - All tenors of one date share a colour from that date's slope, so the eye reads
 *    "when was the curve inverted" as bands across time.
 */
export function buildRibbonGeometry(data: TermStructureData, opts: Partial<RibbonOptions> = {}): RibbonGeometry {
  const o = { ...DEFAULT_RIBBON, ...opts }
  if (![o.width, o.depth, o.height, o.slopeScale].every((v) => Number.isFinite(v) && v > 0)) {
    throw new Error('ribbon options must be finite and positive')
  }
  const nD = data.dates.length
  const nT = data.tenors.length
  if (nD < 2 || nT < 2) throw new Error('a ribbon needs at least 2 dates and 2 tenors')

  let min = Infinity
  let max = -Infinity
  for (const row of data.levels) {
    for (const v of row) {
      if (v < min) min = v
      if (v > max) max = v
    }
  }
  const range = max - min

  const positions = new Float32Array(nD * nT * 3)
  const colors = new Float32Array(nD * nT * 3)
  const slopes: number[] = []
  for (let d = 0; d < nD; d++) {
    const slope = termSlope(data.levels[d])
    slopes.push(slope)
    const [r, g, b] = slopeColour(slope, o.slopeScale)
    for (let t = 0; t < nT; t++) {
      const k = (d * nT + t) * 3
      positions[k] = -o.width / 2 + (o.width * t) / (nT - 1)
      // A perfectly flat grid has range 0: pin it to the floor instead of dividing by zero.
      positions[k + 1] = range === 0 ? 0 : (o.height * (data.levels[d][t] - min)) / range
      positions[k + 2] = -o.depth / 2 + (o.depth * d) / (nD - 1)
      colors[k] = r
      colors[k + 1] = g
      colors[k + 2] = b
    }
  }

  const triangleCount = (nD - 1) * (nT - 1) * 2
  const indices = new Uint32Array(triangleCount * 3)
  let n = 0
  for (let d = 0; d < nD - 1; d++) {
    for (let t = 0; t < nT - 1; t++) {
      const a = d * nT + t
      const b = a + 1
      const c = a + nT
      const e = c + 1
      indices.set([a, c, b, b, c, e], n)
      n += 6
    }
  }
  return { positions, colors, indices, vertexCount: nD * nT, triangleCount, slopes, levelRange: { min, max } }
}
