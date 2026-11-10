/**
 * The error surface of the Evaluation view: evaluation.json's per-regime losses (schema 1.2.0) laid
 * out as a model x horizon x regime grid, and the pure geometry of its 3D bars.
 *
 * Nothing here computes a result. Every value is a ratio to the random walk as the JSON holds it;
 * the grid only arranges the rows, and the geometry only places a box for each. The 3D component
 * (`ErrorSurfaceCanvas` in pages/RibbonCanvas.tsx) hands the boxes to three.js as they are, which is
 * why the decisions live here: jsdom cannot run WebGL, but it can test these functions.
 */
import { REGIMES, type Evaluation, type LossRow } from './results'

export type Measure = 'rmse' | 'qlike'
const RATIO: Record<Measure, 'rmseRatioRw' | 'qlikeRatioRw'> = { rmse: 'rmseRatioRw', qlike: 'qlikeRatioRw' }

export interface SurfaceRegime { scheme: string; regime: string; label: string }

export interface SurfaceGrid {
  targetId: string
  measure: Measure
  /** File order, which is the registry order the exporter writes. */
  models: string[]
  horizons: number[]
  /** REGIMES order, only the regimes the file has for this target; unknown ids after them. */
  regimes: SurfaceRegime[]
  /** cells[model][horizon][regime]: the ratio to the random walk, or null (suppressed cell, or no row). */
  cells: (number | null)[][][]
  /** whole[model][horizon]: the whole-period ratio from `losses`, or null. */
  whole: (number | null)[][]
}

/** The regime rows of one target as a dense model x horizon x regime grid of one ratio. */
export function surfaceGrid(ev: Evaluation, targetId: string, measure: Measure): SurfaceGrid {
  const key = RATIO[measure]
  const rows = ev.regimeLosses.filter((r) => r.targetId === targetId)
  const models = [...new Set(rows.map((r) => r.modelId))]
  const horizons = [...new Set(rows.map((r) => r.horizon))].sort((a, b) => a - b)
  const present = (scheme: string, regime: string) => rows.some((r) => r.scheme === scheme && r.regime === regime)
  const known: SurfaceRegime[] = REGIMES.filter((r) => present(r.scheme, r.regime)).map(({ scheme, regime, label }) => ({ scheme, regime, label }))
  const unknown: SurfaceRegime[] = []
  for (const r of rows) {
    if (!REGIMES.some((k) => k.scheme === r.scheme && k.regime === r.regime) && !unknown.some((u) => u.scheme === r.scheme && u.regime === r.regime)) {
      unknown.push({ scheme: r.scheme, regime: r.regime, label: r.regime })
    }
  }
  const regimes = [...known, ...unknown]
  const ratio = (row: LossRow | undefined) => row?.[key] ?? null
  return {
    targetId,
    measure,
    models,
    horizons,
    regimes,
    cells: models.map((m) =>
      horizons.map((h) => regimes.map((g) => ratio(rows.find((r) => r.modelId === m && r.horizon === h && r.scheme === g.scheme && r.regime === g.regime)))),
    ),
    whole: models.map((m) => horizons.map((h) => ratio(ev.losses.find((r) => r.targetId === targetId && r.modelId === m && r.horizon === h)))),
  }
}

/** Index of the lowest finite value, or -1 when there is none (ties: the first). */
export function lowestIndex(values: (number | null)[]): number {
  let best = -1
  values.forEach((v, i) => {
    if (v !== null && Number.isFinite(v) && (best < 0 || v < (values[best] as number))) best = i
  })
  return best
}

export interface SurfaceOptions {
  /** Largest extent along x (horizon blocks of model columns) and z (regime rows), world units. */
  width: number
  depth: number
  /** Length of the longest bar. */
  height: number
  /** Empty columns between two horizon blocks, and empty rows where the regime scheme changes. */
  gap: number
  /** Bar footprint as a share of the cell. */
  footprint: number
  /** A ratio of exactly 1 still gets a tile this thick, so the cell is visibly there. */
  minHeight: number
}

export const DEFAULT_SURFACE: SurfaceOptions = { width: 10, depth: 5, height: 2.5, gap: 1, footprint: 0.7, minHeight: 0.03 }

export type Vec3 = [number, number, number]

export interface SurfaceBar {
  modelId: string
  horizon: number
  regime: SurfaceRegime
  /** The ratio to the random walk, as the JSON holds it. */
  value: number
  /** Colour slot of the model (`--series-N`), or null for the ink colour. */
  slot: number | null
  /** Centre of the box. */
  position: Vec3
  /** Extent of the box along x, y, z. */
  size: Vec3
}

export interface SurfaceLabel { text: string; position: Vec3 }

export interface SurfaceLayout {
  bars: SurfaceBar[]
  /** One per model column, at the front edge. */
  modelLabels: SurfaceLabel[]
  /** One per regime row, at the left edge (the text ends at the point). */
  regimeLabels: SurfaceLabel[]
  /** One per horizon block, at the back edge. */
  horizonLabels: SurfaceLabel[]
  /** The plane at ratio 1 (the random walk): its extent along x and z, centred on the origin. */
  plane: { width: number; depth: number }
  /** The largest |ln ratio| among the bars drawn; the bar of that ratio is `height` long. 0 when all are 1. */
  scale: number
  /** The smallest and largest ratio drawn, or null when no bar is drawn. */
  range: { min: number; max: number } | null
}

/**
 * Place one box per (shown model, horizon, regime) cell that has a ratio.
 *
 *  - x runs over the horizon blocks (shortest first, at -x), each holding one column per shown model
 *    in grid order, with `gap` empty columns between blocks; so the models of one horizon and regime
 *    stand side by side. z runs over the regimes in grid order (the first at -z, the back), with
 *    `gap` empty rows wherever the scheme changes: the VIX-level and the term-structure regimes slice
 *    the same dates, they are not one partition. Cells are square: the side is the largest that fits
 *    both `width` and `depth`.
 *  - The plane y = 0 is the random walk (ratio 1). A bar rises above it by ln(ratio) when the model
 *    loses to the random walk and hangs below it when the model wins, scaled so that the largest
 *    |ln ratio| drawn is `height` long. A log scale, so a halving and a doubling are the same length.
 *  - A null or non-positive ratio gets no box: the cell is missing, not zero.
 */
export function surfaceLayout(grid: SurfaceGrid, shown: string[], slotOf: (modelId: string) => number | null, opts: Partial<SurfaceOptions> = {}): SurfaceLayout {
  const o = { ...DEFAULT_SURFACE, ...opts }
  if (![o.width, o.depth, o.height, o.footprint, o.minHeight].every((v) => Number.isFinite(v) && v > 0) || !(o.gap >= 0)) {
    throw new Error('surface options must be finite and positive (gap non-negative)')
  }
  const models = grid.models.filter((m) => shown.includes(m))
  const nM = models.length
  const nH = grid.horizons.length
  const nR = grid.regimes.length
  const empty = { bars: [], modelLabels: [], regimeLabels: [], horizonLabels: [], plane: { width: 0, depth: 0 }, scale: 0, range: null }
  if (nM === 0 || nH === 0 || nR === 0) return empty
  const nCols = nH * nM + (nH - 1) * o.gap
  const rowOf: number[] = []
  grid.regimes.forEach((g, r) => rowOf.push(r === 0 ? 0 : rowOf[r - 1] + 1 + (g.scheme === grid.regimes[r - 1].scheme ? 0 : o.gap)))
  const nRows = rowOf[nR - 1] + 1
  const cell = Math.min(o.width / nCols, o.depth / nRows)
  const xOf = (h: number, mi: number) => (h * (nM + o.gap) + mi - (nCols - 1) / 2) * cell
  const zOf = (r: number) => (rowOf[r] - (nRows - 1) / 2) * cell

  const cells: { m: string; mi: number; h: number; r: number; v: number }[] = []
  models.forEach((m, mi) => {
    const gi = grid.models.indexOf(m)
    grid.horizons.forEach((_, h) =>
      grid.regimes.forEach((_, r) => {
        const v = grid.cells[gi][h][r]
        if (v !== null && Number.isFinite(v) && v > 0) cells.push({ m, mi, h, r, v })
      }),
    )
  })
  const scale = Math.max(0, ...cells.map((c) => Math.abs(Math.log(c.v))))
  const side = cell * o.footprint
  const bars = cells.map(({ m, mi, h, r, v }): SurfaceBar => {
    const length = scale === 0 ? 0 : (Math.log(v) / scale) * o.height
    const tall = Math.abs(length) >= o.minHeight
    return {
      modelId: m,
      horizon: grid.horizons[h],
      regime: grid.regimes[r],
      value: v,
      slot: slotOf(m),
      position: [xOf(h, mi), tall ? length / 2 : 0, zOf(r)],
      size: [side, tall ? Math.abs(length) : o.minHeight, side],
    }
  })
  const width = nCols * cell
  const depth = nRows * cell
  return {
    bars,
    modelLabels: grid.horizons.flatMap((_, h) => models.map((m, mi): SurfaceLabel => ({ text: m, position: [xOf(h, mi), 0, depth / 2 + cell * 0.4] }))),
    regimeLabels: grid.regimes.map((g, r) => ({ text: g.label, position: [-width / 2 - cell * 0.4, 0, zOf(r)] })),
    horizonLabels: grid.horizons.map((hz, h) => ({ text: `h = ${hz}`, position: [xOf(h, (nM - 1) / 2), 0, -depth / 2 - cell * 0.6] })),
    plane: { width, depth },
    scale,
    range: cells.length === 0 ? null : { min: Math.min(...cells.map((c) => c.v)), max: Math.max(...cells.map((c) => c.v)) },
  }
}
