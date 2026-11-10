import { describe, expect, it } from 'vitest'
import { DEFAULT_SURFACE, lowestIndex, surfaceGrid, surfaceLayout } from './errorSurface'
import type { Evaluation, RegimeLossRow } from './results'

const row = (modelId: string, horizon: number, regime: string, rmseRatioRw: number | null, qlikeRatioRw = rmseRatioRw, targetId = 't'): RegimeLossRow => ({
  targetId,
  horizon,
  modelId,
  scheme: 'vix_level',
  regime,
  n: 30,
  rmse: 1,
  qlike: 0.1,
  rmseRatioRw,
  qlikeRatioRw,
})

// A hand-made file. The regime rows are written out of display order (h = 5 first, high before low)
// and hold a term-structure regime, an id the app does not know ('storm') and a row of another target ('u').
const EV: Evaluation = {
  sourceRunId: 'abcdef012345',
  dmMethod: 'fixed_b',
  dmFixedB: 0.1,
  mzMethod: 'hansen_hodrick',
  losses: [
    // Another target's row first, so a lookup that ignores the target would find it.
    { targetId: 'u', horizon: 1, modelId: 'har', n: 90, rmse: 1, qlike: 0.1, rmseRatioRw: 7, qlikeRatioRw: 7 },
    { targetId: 't', horizon: 1, modelId: 'rw', n: 90, rmse: 1, qlike: 0.1, rmseRatioRw: 1, qlikeRatioRw: 1 },
    { targetId: 't', horizon: 1, modelId: 'har', n: 90, rmse: 1, qlike: 0.1, rmseRatioRw: 0.9, qlikeRatioRw: 0.7 },
    { targetId: 't', horizon: 5, modelId: 'har', n: 90, rmse: 1, qlike: 0.1, rmseRatioRw: 0.8, qlikeRatioRw: 0.6 },
  ],
  regimeLosses: [
    row('rw', 5, 'high', 1),
    row('har', 5, 'high', 2, 4),
    row('rw', 1, 'low', 1),
    row('har', 1, 'low', 0.5, 0.25),
    row('rw', 1, 'high', 1),
    row('har', 1, 'high', null),
    row('rw', 1, 'storm', 1),
    { ...row('har', 5, 'backwardation', 1.5, 3), scheme: 'term_structure' },
    row('har', 1, 'low', 9, 9, 'u'),
  ],
  dm: [],
  mz: [],
  calibration: { cells: 0, helps: 0 },
}

describe('surfaceGrid', () => {
  it('lays one target out as model x horizon x regime, regimes in display order and unknown ids last', () => {
    const g = surfaceGrid(EV, 't', 'rmse')
    expect(g.models).toEqual(['rw', 'har']) // file order
    expect(g.horizons).toEqual([1, 5]) // ascending, whatever the file order
    expect(g.regimes.map((r) => r.label)).toEqual(['low VIX', 'high VIX', 'backwardation', 'storm'])
    // cells[model][horizon][regime]; a missing row and a suppressed (null) cell are both null.
    expect(g.cells).toEqual([
      [
        [1, 1, null, 1],
        [null, 1, null, null],
      ],
      [
        [0.5, null, null, null],
        [null, 2, 1.5, null],
      ],
    ])
    expect(g.whole).toEqual([
      [1, null],
      [0.9, 0.8],
    ]) // target t only: u's 7 is not read
  })
  it('reads the QLIKE ratio for the QLIKE measure', () => {
    const g = surfaceGrid(EV, 't', 'qlike')
    expect(g.cells[1][0][0]).toBe(0.25)
    expect(g.cells[1][1][1]).toBe(4)
    expect(g.cells[1][1][2]).toBe(3)
    expect(g.whole[1]).toEqual([0.7, 0.6])
  })
  it('is empty for a target without regime rows', () => {
    const g = surfaceGrid(EV, 'nope', 'rmse')
    expect([g.models, g.horizons, g.regimes, g.cells]).toEqual([[], [], [], []])
  })
})

describe('lowestIndex', () => {
  it('finds the first lowest finite value', () => {
    expect(lowestIndex([null, 2, 1, 1])).toBe(2)
    expect(lowestIndex([3])).toBe(0)
    expect(lowestIndex([])).toBe(-1)
    expect(lowestIndex([null, NaN])).toBe(-1)
  })
})

describe('surfaceLayout', () => {
  const opts = { width: 8, depth: 8, height: 2, gap: 1, footprint: 0.5, minHeight: 0.1 }
  const slotOf = (id: string) => (id === 'har' ? 2 : null)
  // Columns: 2 horizon blocks x 2 models + 1 gap column = 5. Rows: low 0, high 1, (scheme changes: gap)
  // backwardation 3, (changes back: gap) storm 5, so 6 rows. cell = min(8 / 5, 8 / 6) = 4/3.
  const cell = 4 / 3
  const layout = surfaceLayout(surfaceGrid(EV, 't', 'rmse'), ['rw', 'har'], slotOf, opts)
  const bar = (m: string, h: number, regime: string) => layout.bars.find((b) => b.modelId === m && b.horizon === h && b.regime.regime === regime)

  it('draws one box per cell with a ratio, none for a null', () => {
    // rw: h1 low, high, storm; h5 high. har: h1 low; h5 high, backwardation. (har h1 high is null.)
    expect(layout.bars).toHaveLength(7)
    expect(bar('har', 1, 'high')).toBeUndefined()
  })
  it('hangs a winning bar below the plane and raises a losing one, on a log scale', () => {
    // The longest |ln ratio| is ln 2 (both 0.5 and 2), so each of those bars is height = 2 long.
    expect(layout.scale).toBeCloseTo(Math.log(2), 12)
    const win = bar('har', 1, 'low')!
    expect(win.value).toBe(0.5)
    expect(win.position[1]).toBeCloseTo(-1, 12) // ln 0.5 / ln 2 * 2 = -2: centre at -1
    expect(win.size[1]).toBeCloseTo(2, 12)
    const lose = bar('har', 5, 'high')!
    expect(lose.position[1]).toBeCloseTo(1, 12)
    expect(lose.size[1]).toBeCloseTo(2, 12)
    // ln 1.5 / ln 2 * 2 = 1.169925...: shorter than the doubling, as the log scale says.
    expect(bar('har', 5, 'backwardation')!.size[1]).toBeCloseTo(1.1699250014423124, 12)
    // A ratio of exactly 1 is a thin tile on the plane.
    expect(bar('rw', 1, 'low')!.position[1]).toBe(0)
    expect(bar('rw', 1, 'low')!.size[1]).toBe(0.1)
  })
  it('stands the models of a horizon side by side along x, and the regimes in rows along z', () => {
    // x = (block * (2 models + 1 gap) + model - (5 - 1) / 2) * cell.
    expect(bar('rw', 1, 'low')!.position[0]).toBeCloseTo(-2 * cell, 12)
    expect(bar('har', 1, 'low')!.position[0]).toBeCloseTo(-1 * cell, 12)
    expect(bar('rw', 5, 'high')!.position[0]).toBeCloseTo(1 * cell, 12)
    expect(bar('har', 5, 'high')!.position[0]).toBeCloseTo(2 * cell, 12)
    // z = (row - (6 - 1) / 2) * cell, with a gap row at each change of scheme.
    expect(bar('rw', 1, 'low')!.position[2]).toBeCloseTo(-2.5 * cell, 12)
    expect(bar('rw', 1, 'high')!.position[2]).toBeCloseTo(-1.5 * cell, 12)
    expect(bar('har', 5, 'backwardation')!.position[2]).toBeCloseTo(0.5 * cell, 12)
    expect(bar('rw', 1, 'storm')!.position[2]).toBeCloseTo(2.5 * cell, 12)
    const win = bar('har', 1, 'low')!
    expect(win.size[0]).toBeCloseTo(cell * 0.5, 12)
    expect(win.size[2]).toBeCloseTo(cell * 0.5, 12)
    expect(layout.plane.width).toBeCloseTo(5 * cell, 12)
    expect(layout.plane.depth).toBeCloseTo(8, 12) // 6 rows x 4/3
  })
  it('without a gap, packs the blocks and the schemes together', () => {
    // 4 columns, 4 rows: cell = 2; x = (block * 2 + model - 1.5) * 2, z = (row - 1.5) * 2.
    const tight = surfaceLayout(surfaceGrid(EV, 't', 'rmse'), ['rw', 'har'], slotOf, { ...opts, gap: 0 })
    const at = (m: string, h: number, regime: string) => tight.bars.find((b) => b.modelId === m && b.horizon === h && b.regime.regime === regime)!.position
    expect(at('rw', 5, 'high')[0]).toBeCloseTo(1, 12)
    expect(at('har', 5, 'backwardation')[2]).toBeCloseTo(1, 12)
    expect(at('rw', 1, 'storm')[2]).toBeCloseTo(3, 12)
    expect(tight.plane).toEqual({ width: 8, depth: 8 })
  })
  it('colours each bar by its model slot and reports the range drawn', () => {
    expect(bar('har', 1, 'low')!.slot).toBe(2)
    expect(bar('rw', 1, 'low')!.slot).toBeNull()
    expect(layout.range).toEqual({ min: 0.5, max: 2 })
  })
  it('labels the model columns at the front, the regimes on the left and the horizons at the back', () => {
    expect(layout.modelLabels.map((l) => l.text)).toEqual(['rw', 'har', 'rw', 'har'])
    expect(layout.modelLabels[3].position[0]).toBeCloseTo(2 * cell, 12)
    expect(layout.modelLabels[0].position[2]).toBeCloseTo(4 + 0.4 * cell, 12)
    expect(layout.regimeLabels.map((l) => l.text)).toEqual(['low VIX', 'high VIX', 'backwardation', 'storm'])
    expect(layout.regimeLabels[2].position).toEqual([expect.closeTo(-2.5 * cell - 0.4 * cell, 12), 0, expect.closeTo(0.5 * cell, 12)])
    expect(layout.horizonLabels.map((l) => l.text)).toEqual(['h = 1', 'h = 5'])
    // The middle of each block of two: x = (block * 3 + 0.5 - 2) * cell.
    expect(layout.horizonLabels[0].position).toEqual([expect.closeTo(-1.5 * cell, 12), 0, expect.closeTo(-4 - 0.6 * cell, 12)])
    expect(layout.horizonLabels[1].position[0]).toBeCloseTo(1.5 * cell, 12)
  })
  it('draws only the models shown, in grid order', () => {
    // One model: 3 columns (h1, gap, h5); cell = min(8 / 3, 8 / 6) = 4/3, x = (block * 2 - 1) * cell.
    const one = surfaceLayout(surfaceGrid(EV, 't', 'rmse'), ['har', 'lstm'], slotOf, opts)
    expect(one.bars.map((b) => b.modelId)).toEqual(['har', 'har', 'har'])
    expect(one.bars.map((b) => b.position[0])).toEqual([expect.closeTo(-cell, 12), expect.closeTo(cell, 12), expect.closeTo(cell, 12)])
    expect(one.modelLabels.map((l) => l.text)).toEqual(['har', 'har'])
    expect(surfaceLayout(surfaceGrid(EV, 't', 'rmse'), [], slotOf, opts)).toMatchObject({ bars: [], range: null, scale: 0 })
  })
  it('handles all-ones, no ratio at all, and no rows', () => {
    const flat = surfaceLayout(surfaceGrid(EV, 't', 'rmse'), ['rw'], slotOf, opts)
    expect(flat.scale).toBe(0)
    expect(flat.bars.every((b) => b.size[1] === 0.1 && b.position[1] === 0)).toBe(true)
    const nulls = { ...EV, regimeLosses: [row('rw', 1, 'low', null), row('rw', 1, 'high', 0)] }
    expect(surfaceLayout(surfaceGrid(nulls, 't', 'rmse'), ['rw'], slotOf, opts)).toMatchObject({ bars: [], range: null }) // null and 0 are no bar
    expect(surfaceLayout(surfaceGrid(EV, 'nope', 'rmse'), ['rw'], slotOf).bars).toEqual([])
  })
  it('refuses options that are not finite and positive', () => {
    const grid = surfaceGrid(EV, 't', 'rmse')
    expect(() => surfaceLayout(grid, ['rw'], slotOf, { height: 0 })).toThrow('finite and positive')
    expect(() => surfaceLayout(grid, ['rw'], slotOf, { width: NaN })).toThrow('finite and positive')
    expect(() => surfaceLayout(grid, ['rw'], slotOf, { gap: -1 })).toThrow('gap non-negative')
    expect(DEFAULT_SURFACE.gap).toBeGreaterThanOrEqual(0)
  })
})
