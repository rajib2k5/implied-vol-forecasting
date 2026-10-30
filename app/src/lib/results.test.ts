import { describe, expect, it } from 'vitest'
import { segments, spreadLabels, strokeFor } from './chartGeom'
import {
  DatasetError,
  colorSlot,
  dmMatrix,
  linearTicks,
  logTicks,
  nearestIndex,
  parseEda,
  parseEnvelope,
  parseEquityCurves,
  parseEvaluation,
  parseModelRace,
  show,
  slotOrder,
  targetDates,
  type Evaluation,
} from './results'

const env = (dataset: string, payload: unknown, extra: Record<string, unknown> = {}) => ({
  schema_version: '1.1.0',
  generated_at: '2026-10-29T00:00:00Z',
  source_commit: '0'.repeat(40),
  source_dirty: true,
  data_mode: 'synthetic',
  dataset,
  payload,
  ...extra,
})

// A hand-made race: calendar of 6 dates, one panel h = 2 with 3 origins from index 1:
// row i is scored at calendar[1 + i + 2] = d3, d4, d5.
const RACE = {
  calendar: ['d0', 'd1', 'd2', 'd3', 'd4', 'd5'],
  holdout_start_idx: 5,
  panels: [
    {
      target_id: 'vix_level',
      horizon: 2,
      origin_idx0: 1,
      n: 3,
      actual: [20, null, 22],
      series: [{ model_id: 'rw', yhat: [19, 20, 21] }],
      mz: [{ model_id: 'rw', period: 'dev', n: 3, alpha: 0.1, beta: 0.9, wald_p: 0.5 }],
    },
  ],
}

describe('parseEnvelope', () => {
  it('accepts MAJOR 1 of the expected dataset and reports the data mode', () => {
    const e = parseEnvelope(env('eda', {}), 'eda')
    expect(e.dataMode).toBe('synthetic')
    expect(e.schemaVersion).toBe('1.1.0')
  })
  it.each([
    [null, 'not a JSON object'],
    [env('eda', {}, { schema_version: '2.0.0' }), 'is not MAJOR 1'],
    [env('model_race', {}), 'expected dataset eda'],
    [env('eda', {}, { data_mode: 'demo' }), 'unknown data_mode'],
    [env('eda', [1]), 'payload must be an object'],
  ])('refuses %j', (raw, message) => {
    expect(() => parseEnvelope(raw, 'eda')).toThrow(DatasetError)
    expect(() => parseEnvelope(raw, 'eda')).toThrow(message)
  })
})

describe('parseModelRace', () => {
  it('reads the panels and maps each row to its target date', () => {
    const race = parseModelRace(env('model_race', RACE)).payload
    expect(race.panels[0].series[0]).toEqual({ modelId: 'rw', yhat: [19, 20, 21] })
    expect(race.panels[0].mz[0]).toEqual({ modelId: 'rw', n: 3, alpha: 0.1, beta: 0.9, waldP: 0.5 })
    expect(targetDates(race, race.panels[0])).toEqual(['d3', 'd4', 'd5'])
  })
  it('refuses arrays of the wrong length and rows past the calendar', () => {
    const short = structuredClone(RACE)
    short.panels[0].actual = [20, 21]
    expect(() => parseModelRace(env('model_race', short))).toThrow('arrays must have n = 3 values')
    const late = structuredClone(RACE)
    late.panels[0].origin_idx0 = 2 // last row: 2 + 2 + 2 = 6 > 5
    expect(() => parseModelRace(env('model_race', late))).toThrow('rows run past the calendar')
    const text = structuredClone(RACE) as unknown as { panels: { series: { yhat: unknown[] }[] }[] }
    text.panels[0].series[0].yhat = ['19', 20, 21]
    expect(() => parseModelRace(env('model_race', text))).toThrow('yhat[0] must be a number or null')
  })
})

const EV_PAYLOAD = {
  period: 'dev',
  source_run_id: 'abcdef012345',
  dm_method: 'fixed_b',
  dm_fixed_b: 0.1,
  mz_method: 'hansen_hodrick',
  losses: ['rw', 'har', 'lstm'].map((m) => ({ target_id: 't', horizon: 1, model_id: m, n: 9, rmse: 1, mae: 1, bias: 0, mape: 0.1, qlike: 0.1, rmse_ratio_rw: 1, qlike_ratio_rw: 1 })),
  dm: [
    { target_id: 't', horizon: 1, loss: 'qlike', model_a: 'har', model_b: 'rw', n: 9, dm_stat: -3, p_value: 0.001, p_holm: 0.003, lower_loss: 'har' },
    { target_id: 't', horizon: 1, loss: 'qlike', model_a: 'lstm', model_b: 'rw', n: 9, dm_stat: 1, p_value: 0.3, p_holm: 0.6, lower_loss: 'rw' },
    { target_id: 't', horizon: 1, loss: 'sq_error', model_a: 'har', model_b: 'rw', n: 9, dm_stat: -3, p_value: 0.001, p_holm: 0.003, lower_loss: 'rw' },
  ],
  mz: [{ target_id: 't', horizon: 1, model_id: 'rw', n: 9, alpha: 0, alpha_se: 0.1, beta: 1, beta_se: 0.1, r2: 0.9, wald_p: 0.7 }],
  calibration: { cells: 28, helps: 0 },
}

describe('parseEvaluation and dmMatrix', () => {
  const ev: Evaluation = parseEvaluation(env('evaluation', EV_PAYLOAD)).payload
  it('reads every table', () => {
    expect(ev.losses.map((l) => l.modelId)).toEqual(['rw', 'har', 'lstm'])
    expect(ev.dm).toHaveLength(3)
    expect(ev.mz[0].waldP).toBe(0.7)
    expect(ev.calibration).toEqual({ cells: 28, helps: 0 })
  })
  it('lays the pairs of one loss out as a grid: row wins, column wins, neither, self, untested', () => {
    const { models, cells } = dmMatrix(ev, 't', 1, 'qlike')
    expect(models).toEqual(['rw', 'har', 'lstm'])
    // rw vs har: har has the lower loss, Holm 0.003 < 0.05, har is the COLUMN here.
    expect(cells[0][1]).toEqual({ verdict: 'col', pHolm: 0.003, lower: 'har' })
    // har vs rw: the same pair seen from har's row.
    expect(cells[1][0]).toEqual({ verdict: 'row', pHolm: 0.003, lower: 'har' })
    // rw vs lstm: Holm 0.6 is not below 0.05.
    expect(cells[0][2].verdict).toBe('none')
    expect(cells[1][1].verdict).toBe('self')
    expect(cells[1][2]).toEqual({ verdict: 'missing', pHolm: null, lower: null }) // har vs lstm was not in the file
  })
  it('keeps the losses apart', () => {
    expect(dmMatrix(ev, 't', 1, 'sq_error').cells[1][0].verdict).toBe('col') // rw has the lower squared error
  })
  it('a p exactly at alpha is not significant', () => {
    expect(dmMatrix(ev, 't', 1, 'qlike', 0.003).cells[1][0].verdict).toBe('none')
  })
})

describe('parseEquityCurves', () => {
  const payload = {
    dates: ['a', 'b'],
    holdout_start_idx: null,
    cost_bps: 5,
    strategies: [{ strategy_id: 'p', label: 'P', kind: 'forecast', model_id: 'garch', horizon: 21, equity: [1, 1.1], drawdown: [0, 0] }],
    regime_segments: [],
    summary: [{ strategy_id: 'p', period: 'dev', n_days: 2, ann_return: 0.1, ann_vol: 0.2, sharpe_ann: 0.5, psr0: 0.6, dsr: null, n_trials: 5, max_drawdown: 0, turnover_ann: 1 }],
    caveats: ['c'],
  }
  it('reads a 1.0 file without the optional cost table', () => {
    const eq = parseEquityCurves(env('equity_curves', payload)).payload
    expect(eq.strategies[0].equity).toEqual([1, 1.1])
    expect(eq.summary[0]).toMatchObject({ strategyId: 'p', costBps: null, dsr: null, nTrials: 5 })
    expect(eq.costSensitivity).toEqual([])
  })
  it('reads the cost table when present, and refuses a non-positive equity', () => {
    const withCosts = { ...payload, cost_sensitivity: [{ ...payload.summary[0], cost_bps: 10 }] }
    expect(parseEquityCurves(env('equity_curves', withCosts)).payload.costSensitivity[0].costBps).toBe(10)
    const bad = structuredClone(payload)
    bad.strategies[0].equity = [1, 0]
    expect(() => parseEquityCurves(env('equity_curves', bad))).toThrow('equity[1] must be a positive number')
    const short = structuredClone(payload)
    short.strategies[0].equity = [1]
    expect(() => parseEquityCurves(env('equity_curves', short))).toThrow('one equity value per date')
  })
})

describe('parseEda', () => {
  it('reads the ACF series and the half-lives, keeping an unbounded end as null', () => {
    const eda = parseEda(
      env('eda', {
        period: 'pre_holdout',
        source_run_id: 'abcdef012345',
        first: '1990-01-02',
        last: '2024-06-28',
        holdout_start: '2024-07-01',
        nlags: 2,
        acf: [{ series_id: 'abs_r', label: '|r|', n: 100, acf: [0.2, null], band95: 0.196, ljung_box_p_last: null }],
        half_life: [{ sample: 'dev', phi: 0.98, phi_ci95: [0.97, 0.99], half_life_sessions: 34.31, half_life_ci95: [22.76, null], n: 100 }],
      }),
    ).payload
    expect(eda.acf[0]).toEqual({ seriesId: 'abs_r', label: '|r|', n: 100, acf: [0.2, null], band95: 0.196, ljungBoxPLast: null })
    expect(eda.halfLife[0].halfLifeCi95).toEqual([22.76, null])
  })
})

describe('chart helpers', () => {
  it('linear ticks are round steps covering the range', () => {
    expect(linearTicks(0, 10, 6)).toEqual([0, 2, 4, 6, 8, 10]) // raw step 2 -> 2
    expect(linearTicks(-0.03, 0.17, 5)).toEqual([0, 0.05, 0.1, 0.15]) // raw 0.05 -> 0.05; -0.05 is below the range
    expect(linearTicks(3, 3)).toEqual([3])
    expect(linearTicks(NaN, 1)).toEqual([])
  })
  it('log ticks use 1-2-5 per decade and fall back to powers of ten', () => {
    expect(logTicks(0.9, 16)).toEqual([1, 2, 5, 10])
    expect(logTicks(0.0015, 17)).toEqual([0.01, 0.1, 1, 10]) // 1-2-5 would give 12 ticks
    expect(logTicks(0, 1)).toEqual([])
  })
  it('nearest index rounds and clamps', () => {
    expect(nearestIndex(5, 0.49)).toBe(2) // 0.49 * 4 = 1.96
    expect(nearestIndex(5, -1)).toBe(0)
    expect(nearestIndex(5, 2)).toBe(4)
    expect(nearestIndex(0, 0.5)).toBe(-1)
  })
  it('a colour slot follows the entity, not its rank, and runs out after the palette', () => {
    const registry = ['rw', 'har', 'garch']
    expect(colorSlot(registry, 'garch')).toBe(3) // whatever else is shown
    expect(colorSlot(registry, 'lstm')).toBeNull()
    expect(colorSlot(['a', 'b', 'c'], 'c', 2)).toBeNull()
    expect(strokeFor(3)).toBe('var(--series-3)')
    expect(strokeFor(null)).toBe('var(--ink)')
  })
  it('the default entities take the first slots, the rest keep their order', () => {
    expect(slotOrder(['rw', 'har_rv', 'vix_implied', 'garch', 'gjr'], ['rw', 'har_log', 'har_rv', 'garch'])).toEqual(['rw', 'har_rv', 'garch', 'vix_implied', 'gjr'])
    expect(slotOrder(['a', 'b'], [])).toEqual(['a', 'b'])
  })
  it('segments break at nulls and non-finite values', () => {
    expect(segments([1, 2, null, 4, NaN, 6, 7])).toEqual([
      { start: 0, points: [1, 2] },
      { start: 3, points: [4] },
      { start: 5, points: [6, 7] },
    ])
    expect(segments([])).toEqual([])
  })
  it('labels are pushed apart in their own order', () => {
    expect(spreadLabels([100, 95, 200], 13)).toEqual([108, 95, 200]) // 95 stays; 100 -> 95 + 13
  })
  it('show prints the JSON value or a dash', () => {
    expect(show(0.1235)).toBe('0.1235')
    expect(show(null)).toBe('–')
    expect(show(undefined)).toBe('–')
  })
})
