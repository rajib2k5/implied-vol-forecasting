/**
 * Readers for the result views' JSON (ADR 0001 D1 tolerant reader) and the small pure UI
 * transforms the views need. Nothing here computes a result: every number on screen is a value
 * from the JSON, re-arranged (a matrix layout, an axis scale, a nearest point under the cursor).
 *
 * The reader checks what the views use: the envelope's MAJOR version (1), the dataset name, the
 * data mode, and the presence and type of the fields drawn. Unknown fields are ignored, so a
 * file from a newer MINOR still renders.
 */

export type DataMode = 'synthetic' | 'real'

export interface Envelope<P> {
  dataMode: DataMode
  schemaVersion: string
  payload: P
}

export class DatasetError extends Error {}

type Obj = Record<string, unknown>
const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v)

function need<T>(ok: boolean, what: string, value: T): T {
  if (!ok) throw new DatasetError(what)
  return value
}

const arr = (o: Obj, key: string, where: string): unknown[] => need(Array.isArray(o[key]), `${where}.${key} must be an array`, o[key] as unknown[])
const str = (o: Obj, key: string, where: string): string => need(typeof o[key] === 'string', `${where}.${key} must be a string`, o[key] as string)
const num = (o: Obj, key: string, where: string): number =>
  need(typeof o[key] === 'number' && Number.isFinite(o[key]), `${where}.${key} must be a number`, o[key] as number)
const numOrNull = (o: Obj, key: string, where: string): number | null =>
  need(o[key] === null || (typeof o[key] === 'number' && Number.isFinite(o[key])), `${where}.${key} must be a number or null`, o[key] as number | null)
const numsOrNull = (o: Obj, key: string, where: string): (number | null)[] =>
  arr(o, key, where).map((v, i) => need(v === null || (typeof v === 'number' && Number.isFinite(v)), `${where}.${key}[${i}] must be a number or null`, v as number | null))

/** The envelope: MAJOR 1, the expected dataset, a known data mode, an object payload. */
export function parseEnvelope(raw: unknown, dataset: string): Envelope<Obj> {
  if (!isObj(raw)) throw new DatasetError(`${dataset}: not a JSON object`)
  const version = str(raw, 'schema_version', dataset)
  if (!/^1\.\d+\.\d+$/.test(version)) throw new DatasetError(`${dataset}: schema_version ${version} is not MAJOR 1`)
  if (raw.dataset !== dataset) throw new DatasetError(`expected dataset ${dataset}, got ${String(raw.dataset)}`)
  const mode = raw.data_mode
  if (mode !== 'synthetic' && mode !== 'real') throw new DatasetError(`${dataset}: unknown data_mode ${String(mode)}`)
  if (!isObj(raw.payload)) throw new DatasetError(`${dataset}: payload must be an object`)
  return { dataMode: mode, schemaVersion: version, payload: raw.payload }
}

// --- model_race --------------------------------------------------------------

export interface RaceSeries { modelId: string; yhat: (number | null)[] }
export interface RaceMz { modelId: string; n: number; alpha: number | null; beta: number | null; waldP: number | null }
export interface RacePanel { targetId: string; horizon: number; originIdx0: number; n: number; actual: (number | null)[]; series: RaceSeries[]; mz: RaceMz[] }
export interface ModelRace { calendar: string[]; holdoutStartIdx: number; panels: RacePanel[] }

export function parseModelRace(raw: unknown): Envelope<ModelRace> {
  const env = parseEnvelope(raw, 'model_race')
  const p = env.payload
  const calendar = arr(p, 'calendar', 'model_race').map(String)
  const panels = arr(p, 'panels', 'model_race').map((x, i): RacePanel => {
    const where = `model_race.panels[${i}]`
    const o = need(isObj(x), `${where} must be an object`, x as Obj)
    const n = num(o, 'n', where)
    const panel: RacePanel = {
      targetId: str(o, 'target_id', where),
      horizon: num(o, 'horizon', where),
      originIdx0: num(o, 'origin_idx0', where),
      n,
      actual: numsOrNull(o, 'actual', where),
      series: arr(o, 'series', where).map((s, j) => {
        const so = need(isObj(s), `${where}.series[${j}] must be an object`, s as Obj)
        return { modelId: str(so, 'model_id', `${where}.series[${j}]`), yhat: numsOrNull(so, 'yhat', `${where}.series[${j}]`) }
      }),
      mz: arr(o, 'mz', where).map((m, j) => {
        const mo = need(isObj(m), `${where}.mz[${j}] must be an object`, m as Obj)
        const w = `${where}.mz[${j}]`
        return { modelId: str(mo, 'model_id', w), n: num(mo, 'n', w), alpha: numOrNull(mo, 'alpha', w), beta: numOrNull(mo, 'beta', w), waldP: numOrNull(mo, 'wald_p', w) }
      }),
    }
    need(panel.actual.length === n && panel.series.every((s) => s.yhat.length === n), `${where}: arrays must have n = ${n} values`, null)
    need(panel.originIdx0 + n - 1 + panel.horizon < calendar.length, `${where}: rows run past the calendar`, null)
    return panel
  })
  return { ...env, payload: { calendar, holdoutStartIdx: num(p, 'holdout_start_idx', 'model_race'), panels } }
}

/** The date each row of a panel is scored at: calendar[origin_idx0 + i + horizon] (data-model.md 5.3). */
export function targetDates(race: ModelRace, panel: RacePanel): string[] {
  return Array.from({ length: panel.n }, (_, i) => race.calendar[panel.originIdx0 + i + panel.horizon])
}

// --- evaluation --------------------------------------------------------------

export interface LossRow { targetId: string; horizon: number; modelId: string; n: number; rmse: number | null; qlike: number | null; rmseRatioRw: number | null; qlikeRatioRw: number | null }
export interface DmRow { targetId: string; horizon: number; loss: string; modelA: string; modelB: string; n: number; dmStat: number | null; pValue: number | null; pHolm: number | null; lowerLoss: string }
export interface MzRow { targetId: string; horizon: number; modelId: string; n: number; alpha: number | null; beta: number | null; r2: number | null; waldP: number | null }
export interface Evaluation { sourceRunId: string; dmMethod: string; dmFixedB: number; mzMethod: string; losses: LossRow[]; dm: DmRow[]; mz: MzRow[]; calibration: { cells: number; helps: number } }

export function parseEvaluation(raw: unknown): Envelope<Evaluation> {
  const env = parseEnvelope(raw, 'evaluation')
  const p = env.payload
  const rows = (key: string) => arr(p, key, 'evaluation').map((x, i) => need(isObj(x), `evaluation.${key}[${i}] must be an object`, x as Obj))
  const cal = need(isObj(p.calibration), 'evaluation.calibration must be an object', p.calibration as Obj)
  return {
    ...env,
    payload: {
      sourceRunId: str(p, 'source_run_id', 'evaluation'),
      dmMethod: str(p, 'dm_method', 'evaluation'),
      dmFixedB: num(p, 'dm_fixed_b', 'evaluation'),
      mzMethod: str(p, 'mz_method', 'evaluation'),
      losses: rows('losses').map((o, i) => {
        const w = `evaluation.losses[${i}]`
        return { targetId: str(o, 'target_id', w), horizon: num(o, 'horizon', w), modelId: str(o, 'model_id', w), n: num(o, 'n', w), rmse: numOrNull(o, 'rmse', w), qlike: numOrNull(o, 'qlike', w), rmseRatioRw: numOrNull(o, 'rmse_ratio_rw', w), qlikeRatioRw: numOrNull(o, 'qlike_ratio_rw', w) }
      }),
      dm: rows('dm').map((o, i) => {
        const w = `evaluation.dm[${i}]`
        return { targetId: str(o, 'target_id', w), horizon: num(o, 'horizon', w), loss: str(o, 'loss', w), modelA: str(o, 'model_a', w), modelB: str(o, 'model_b', w), n: num(o, 'n', w), dmStat: numOrNull(o, 'dm_stat', w), pValue: numOrNull(o, 'p_value', w), pHolm: numOrNull(o, 'p_holm', w), lowerLoss: str(o, 'lower_loss', w) }
      }),
      mz: rows('mz').map((o, i) => {
        const w = `evaluation.mz[${i}]`
        return { targetId: str(o, 'target_id', w), horizon: num(o, 'horizon', w), modelId: str(o, 'model_id', w), n: num(o, 'n', w), alpha: numOrNull(o, 'alpha', w), beta: numOrNull(o, 'beta', w), r2: numOrNull(o, 'r2', w), waldP: numOrNull(o, 'wald_p', w) }
      }),
      calibration: { cells: num(cal, 'cells', 'evaluation.calibration'), helps: num(cal, 'helps', 'evaluation.calibration') },
    },
  }
}

export type DmVerdict = 'row' | 'col' | 'none' | 'self' | 'missing'
export interface DmCell { verdict: DmVerdict; pHolm: number | null; lower: string | null }

/**
 * The DM pairs of one (target, horizon, loss) laid out as a models x models grid. Cell (i, j) says
 * whether the ROW model i or the COLUMN model j has the lower loss with Holm p < alpha ('row' /
 * 'col'), or neither ('none'). The diagonal is 'self'; a pair absent from the file is 'missing'.
 */
export function dmMatrix(ev: Evaluation, targetId: string, horizon: number, loss: string, alpha = 0.05): { models: string[]; cells: DmCell[][] } {
  const models = ev.losses.filter((l) => l.targetId === targetId && l.horizon === horizon).map((l) => l.modelId)
  const pairs = ev.dm.filter((d) => d.targetId === targetId && d.horizon === horizon && d.loss === loss)
  const cells = models.map((a) =>
    models.map((b): DmCell => {
      if (a === b) return { verdict: 'self', pHolm: null, lower: null }
      const d = pairs.find((x) => (x.modelA === a && x.modelB === b) || (x.modelA === b && x.modelB === a))
      if (!d) return { verdict: 'missing', pHolm: null, lower: null }
      const significant = d.pHolm !== null && d.pHolm < alpha
      return { verdict: significant ? (d.lowerLoss === a ? 'row' : 'col') : 'none', pHolm: d.pHolm, lower: d.lowerLoss }
    }),
  )
  return { models, cells }
}

// --- equity_curves -------------------------------------------------------------

export interface Strategy { strategyId: string; label: string; kind: string; modelId: string | null; equity: number[] }
export interface SummaryRow { strategyId: string; costBps: number | null; nDays: number; annReturn: number | null; sharpe: number | null; psr0: number | null; dsr: number | null; nTrials: number; maxDrawdown: number | null }
export interface EquityCurves { dates: string[]; costBps: number; strategies: Strategy[]; summary: SummaryRow[]; costSensitivity: SummaryRow[]; caveats: string[] }

function summaryRow(o: Obj, w: string): SummaryRow {
  return { strategyId: str(o, 'strategy_id', w), costBps: o.cost_bps === undefined ? null : num(o, 'cost_bps', w), nDays: num(o, 'n_days', w), annReturn: numOrNull(o, 'ann_return', w), sharpe: numOrNull(o, 'sharpe_ann', w), psr0: numOrNull(o, 'psr0', w), dsr: numOrNull(o, 'dsr', w), nTrials: num(o, 'n_trials', w), maxDrawdown: numOrNull(o, 'max_drawdown', w) }
}

export function parseEquityCurves(raw: unknown): Envelope<EquityCurves> {
  const env = parseEnvelope(raw, 'equity_curves')
  const p = env.payload
  const dates = arr(p, 'dates', 'equity_curves').map(String)
  const objs = (key: string) => (p[key] === undefined ? [] : arr(p, key, 'equity_curves')).map((x, i) => need(isObj(x), `equity_curves.${key}[${i}] must be an object`, x as Obj))
  const strategies = objs('strategies').map((o, i) => {
    const w = `equity_curves.strategies[${i}]`
    const equity = arr(o, 'equity', w).map((v, j) => need(typeof v === 'number' && v > 0, `${w}.equity[${j}] must be a positive number`, v as number))
    need(equity.length === dates.length, `${w}: one equity value per date is required`, null)
    return { strategyId: str(o, 'strategy_id', w), label: str(o, 'label', w), kind: str(o, 'kind', w), modelId: o.model_id === null ? null : str(o, 'model_id', w), equity }
  })
  return {
    ...env,
    payload: {
      dates,
      costBps: num(p, 'cost_bps', 'equity_curves'),
      strategies,
      summary: objs('summary').map((o, i) => summaryRow(o, `equity_curves.summary[${i}]`)),
      // Optional since schema 1.1.0: a 1.0.x file has none, and the view says so.
      costSensitivity: objs('cost_sensitivity').map((o, i) => summaryRow(o, `equity_curves.cost_sensitivity[${i}]`)),
      caveats: arr(p, 'caveats', 'equity_curves').map(String),
    },
  }
}

// --- eda -----------------------------------------------------------------------

export interface AcfSeries { seriesId: string; label: string; n: number; acf: (number | null)[]; band95: number; ljungBoxPLast: number | null }
export interface HalfLife { sample: string; phi: number | null; phiCi95: (number | null)[]; halfLife: number | null; halfLifeCi95: (number | null)[]; n: number }
export interface Eda { first: string; last: string; holdoutStart: string; acf: AcfSeries[]; halfLife: HalfLife[] }

export function parseEda(raw: unknown): Envelope<Eda> {
  const env = parseEnvelope(raw, 'eda')
  const p = env.payload
  const objs = (key: string) => arr(p, key, 'eda').map((x, i) => need(isObj(x), `eda.${key}[${i}] must be an object`, x as Obj))
  return {
    ...env,
    payload: {
      first: str(p, 'first', 'eda'),
      last: str(p, 'last', 'eda'),
      holdoutStart: str(p, 'holdout_start', 'eda'),
      acf: objs('acf').map((o, i) => {
        const w = `eda.acf[${i}]`
        return { seriesId: str(o, 'series_id', w), label: str(o, 'label', w), n: num(o, 'n', w), acf: numsOrNull(o, 'acf', w), band95: num(o, 'band95', w), ljungBoxPLast: numOrNull(o, 'ljung_box_p_last', w) }
      }),
      halfLife: objs('half_life').map((o, i) => {
        const w = `eda.half_life[${i}]`
        return { sample: str(o, 'sample', w), phi: numOrNull(o, 'phi', w), phiCi95: numsOrNull(o, 'phi_ci95', w), halfLife: numOrNull(o, 'half_life_sessions', w), halfLifeCi95: numsOrNull(o, 'half_life_ci95', w), n: num(o, 'n', w) }
      }),
    },
  }
}

// --- chart helpers (pure) ----------------------------------------------------------

/** Round-number ticks covering [min, max]: steps of 1, 2 or 5 x 10^k, about `count` of them. */
export function linearTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return []
  if (min === max) return [min]
  const raw = (max - min) / Math.max(count - 1, 1)
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag
  const out: number[] = []
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)))
  return out
}

/** Ticks for a log axis: 1, 2 and 5 x 10^k inside [min, max] (min > 0); only powers of ten if that is too many. */
export function logTicks(min: number, max: number): number[] {
  if (!(min > 0) || !(max >= min)) return []
  const out: number[] = []
  for (let k = Math.floor(Math.log10(min)); k <= Math.ceil(Math.log10(max)); k++) {
    for (const m of [1, 2, 5]) {
      const v = Number((m * 10 ** k).toPrecision(12))
      if (v >= min * (1 - 1e-12) && v <= max * (1 + 1e-12)) out.push(v)
    }
  }
  return out.length > 9 ? out.filter((v) => Number.isInteger(Math.log10(v))) : out
}

/** Index of the point nearest to fraction `f` (0..1) of an n-point axis. */
export function nearestIndex(n: number, f: number): number {
  if (n <= 0) return -1
  return Math.min(n - 1, Math.max(0, Math.round(f * (n - 1))))
}

/**
 * Colour slot (1-based) of each entity: its position in `registry`, so a filter that hides other
 * entities never repaints this one (dataviz rule: colour follows the entity, never its rank).
 * Entities beyond the palette get null and must be drawn in the neutral ink.
 */
export function colorSlot(registry: string[], id: string, slots = 8): number | null {
  const i = registry.indexOf(id)
  return i >= 0 && i < slots ? i + 1 : null
}

/**
 * The colour-slot order of a view: the entities it shows by default first (in `preferred` order),
 * then the rest in their own order. The defaults then take slots 1..k, and the first three slots are
 * the ones validated for every pair (`--pairs all`), so the default view never puts two hard-to-tell
 * hues side by side (polish review: rw, har_rv and garch had landed on slots 1, 2 and 4).
 */
export function slotOrder(ids: string[], preferred: string[]): string[] {
  const first = preferred.filter((id) => ids.includes(id))
  return [...first, ...ids.filter((id) => !first.includes(id))]
}

/** A number as the JSON holds it, or an en dash for null. */
export function show(v: number | null | undefined): string {
  return v === null || v === undefined ? '–' : String(v)
}
