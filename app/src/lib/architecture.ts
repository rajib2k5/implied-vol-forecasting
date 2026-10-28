/**
 * Content and invariants for the Architecture page: the end-to-end pipeline graph, one
 * walk-forward fold as a sequence, and the per-milestone process as an activity flow.
 *
 * The diagram components only draw what they are given, so mistakes (an arrow to a box that
 * does not exist, a label that overflows its box, a scaler fitted after the forecast) would be
 * silent. Each section therefore has a `validate*` function returning readable problems; the
 * tests run them on the real data and, separately, prove they flag deliberately broken data.
 */
import type { ActEdge, ActNode, ArchLayer, SeqStep } from './diagrams'

/* ------------------------------------------------------------- pipeline */

/**
 * What data a node holds, for the licence rule in ADR-0001 D2. `ROOT` below is
 * `$VOL_DATA_DIR/implied-vol-forecasting`, a directory outside the repository.
 * - `synthetic`: generated from a seed, contains no vendor data.
 * - `real-series`: Cboe, FRED or Yahoo data, or anything a daily series can be rebuilt from.
 * - `real-aggregate`: real-data-derived numbers from the closed publication set P of ADR-0001 D2:
 *   unitless and rounded, per whole period, with no regime cells. These are the only real-derived
 *   numbers that may be committed. (The identifier predates the rule; it now means "P".)
 */
export type DataKind = 'synthetic' | 'real-series' | 'real-aggregate'

export interface PipelineNode {
  id: string
  label: string
  /** Name of the layer (row of the diagram) the node sits in. */
  layer: string
  /** Set on nodes that hold data files; left out for code. */
  data?: DataKind
  /** True if the files this node holds are committed to the repository. Only valid with `data`. */
  committed?: boolean
}

export interface PipelineEdge {
  from: string
  to: string
}

/**
 * Layers top to bottom; data only ever flows from a layer to a later one. The module names come
 * from ADR-0001 (D1, D3) and the project plan; the two model rows are both `ivf.models`, split by
 * milestone because a row's name has to fit the 110px gutter (see MAX_LAYER_CHARS).
 */
export const PIPELINE_LAYERS: readonly string[] = [
  'Sources',
  'Load',
  'Raw',
  'Interim',
  'Features',
  'Processed',
  'Splits',
  'ivf.models M2',
  'ivf.models M3+',
  'Forecasts',
  'Evaluate',
  'Backtest',
  'Outputs',
  'App JSON',
  'App',
]

export const PIPELINE_NODES: readonly PipelineNode[] = [
  { id: 'cboe', label: 'CBOE CSV CDN', layer: 'Sources' },
  { id: 'fred', label: 'FRED CLS series', layer: 'Sources' },
  { id: 'yf', label: 'yfinance', layer: 'Sources' },
  { id: 'fetch', label: 'ivf.data.fetch (hashed, logged)', layer: 'Load' },
  { id: 'synth', label: 'ivf.data.synthetic (seeded)', layer: 'Load' },
  { id: 'raw', label: 'ROOT/raw/{source}/{symbol}/{sha256}.csv', layer: 'Raw', data: 'real-series', committed: false },
  { id: 'synraw', label: 'ROOT/synthetic/raw/ (generated CSVs)', layer: 'Raw', data: 'synthetic', committed: false },
  { id: 'interim', label: 'ivf.data.build_interim -> ROOT/interim/{table}.parquet', layer: 'Interim', data: 'real-series', committed: false },
  { id: 'features', label: 'ivf.features (log-IV, realised vol, HAR lags)', layer: 'Features' },
  { id: 'processed', label: 'ROOT/processed/run_id=<id>/{table}.parquet', layer: 'Processed', data: 'real-series', committed: false },
  { id: 'splits', label: 'ivf.evaluation.splits (walk-forward folds)', layer: 'Splits' },
  { id: 'rw', label: 'random walk', layer: 'ivf.models M2' },
  { id: 'mean', label: 'historical mean', layer: 'ivf.models M2' },
  { id: 'arima', label: 'ARIMA (log-IV)', layer: 'ivf.models M2' },
  { id: 'har', label: 'HAR', layer: 'ivf.models M2' },
  { id: 'garch', label: 'GARCH / GJR / EGARCH', layer: 'ivf.models M3+' },
  { id: 'arimagarch', label: 'ARIMA-GARCH', layer: 'ivf.models M3+' },
  { id: 'lstm', label: 'LSTM', layer: 'ivf.models M3+' },
  { id: 'forecasts', label: 'forecast table (model x origin x horizon)', layer: 'Forecasts', data: 'real-series', committed: false },
  { id: 'eval', label: 'ivf.evaluation: RMSE MAE MAPE QLIKE, DM, MZ, regimes', layer: 'Evaluate' },
  { id: 'backtest', label: 'ivf.backtest: VXX / SVXY, costs, PSR / DSR', layer: 'Backtest' },
  { id: 'reports', label: 'ivf.reports -> reports/*.csv (set P only)', layer: 'Outputs', data: 'real-aggregate', committed: true },
  { id: 'export', label: 'tools/export_app_data.py', layer: 'Outputs' },
  { id: 'synthjson', label: 'app/public/data/synthetic/ (committed)', layer: 'App JSON', data: 'synthetic', committed: true },
  { id: 'realjson', label: 'ROOT/app-data/real/ (outside the repo)', layer: 'App JSON', data: 'real-series', committed: false },
  { id: 'app', label: 'this app: React 19 + three.js', layer: 'App' },
]

const MODEL_IDS = ['rw', 'mean', 'arima', 'har', 'garch', 'arimagarch', 'lstm']

/**
 * Data dependencies. Some skip layers: the models read the processed tables as well as the splits,
 * and the exporter reads the interim, processed, forecast and result tables directly (the race
 * and the ribbon do not go through the evaluation step).
 */
export const PIPELINE_EDGES: readonly PipelineEdge[] = [
  { from: 'cboe', to: 'fetch' },
  { from: 'fred', to: 'fetch' },
  { from: 'yf', to: 'fetch' },
  { from: 'fetch', to: 'raw' },
  { from: 'synth', to: 'synraw' },
  { from: 'raw', to: 'interim' },
  { from: 'synraw', to: 'interim' },
  { from: 'interim', to: 'features' },
  { from: 'features', to: 'processed' },
  { from: 'processed', to: 'splits' },
  ...MODEL_IDS.map((id) => ({ from: 'splits', to: id })),
  ...MODEL_IDS.map((id) => ({ from: 'processed', to: id })),
  ...MODEL_IDS.map((id) => ({ from: id, to: 'forecasts' })),
  { from: 'forecasts', to: 'eval' },
  { from: 'processed', to: 'eval' },
  { from: 'forecasts', to: 'backtest' },
  { from: 'interim', to: 'backtest' },
  { from: 'eval', to: 'reports' },
  { from: 'backtest', to: 'reports' },
  { from: 'eval', to: 'export' },
  { from: 'backtest', to: 'export' },
  { from: 'forecasts', to: 'export' },
  { from: 'processed', to: 'export' },
  { from: 'interim', to: 'export' },
  { from: 'export', to: 'synthjson' },
  { from: 'export', to: 'realjson' },
  { from: 'synthjson', to: 'app' },
  { from: 'realjson', to: 'app' },
]

/**
 * Longest label (in characters) that fits a box when `boxCount` boxes share one ArchDiagram row.
 * Mirrors the diagram's geometry (row width 588 of the 720-wide drawing, 10 gap between boxes,
 * 11px monospace at about 6.6px a character, 8px of padding), so an overflow shows up in a test
 * rather than as clipped text in the browser.
 */
export function maxLabelChars(boxCount: number): number {
  const rowWidth = 588
  const gap = 10
  const boxWidth = (rowWidth - (boxCount - 1) * gap) / boxCount
  return Math.floor((boxWidth - 8) / 6.6)
}

/**
 * Longest layer name (in characters) that fits the label gutter left of the boxes. Mirrors the
 * diagram's geometry: the name starts at x = 8 and the boxes at x = 118, and the 10px upper-case
 * monospace text takes about 6px a glyph plus 1px of letter spacing: floor((118 - 8) / 7) = 15.
 */
export const MAX_LAYER_CHARS = 15

/** Problems with the pipeline graph; empty means valid. */
export function validatePipeline(
  nodes: readonly PipelineNode[],
  edges: readonly PipelineEdge[],
  layers: readonly string[],
): string[] {
  const problems: string[] = []
  const byId = new Map<string, PipelineNode>()
  for (const n of nodes) {
    if (byId.has(n.id)) problems.push(`duplicate node id: ${n.id}`)
    byId.set(n.id, n)
    if (!layers.includes(n.layer)) problems.push(`node ${n.id} is in unknown layer "${n.layer}"`)
    // ADR-0001 D2: nothing reconstructable from Cboe, FRED or Yahoo data may be committed.
    if (n.committed && n.data === undefined) problems.push(`node ${n.id} is marked committed but does not say what data it holds`)
    if (n.committed && n.data === 'real-series') problems.push(`node ${n.id} would commit real daily series to the repository`)
  }
  for (const layer of layers) {
    if (layer.length > MAX_LAYER_CHARS) problems.push(`layer name "${layer}" (${layer.length} chars) is too long for the label gutter`)
    const inLayer = nodes.filter((n) => n.layer === layer)
    if (inLayer.length === 0) problems.push(`layer "${layer}" has no nodes`)
    const labels = new Set<string>()
    for (const n of inLayer) {
      if (labels.has(n.label)) problems.push(`layer "${layer}" repeats the label "${n.label}"`)
      labels.add(n.label)
      if (n.label.length > maxLabelChars(inLayer.length)) {
        problems.push(`label "${n.label}" (${n.label.length} chars) is too long for a row of ${inLayer.length} boxes`)
      }
    }
  }
  const touched = new Set<string>()
  const seenEdges = new Set<string>()
  const rank = (id: string) => layers.indexOf(byId.get(id)?.layer ?? '')
  for (const e of edges) {
    const key = `${e.from}->${e.to}`
    if (seenEdges.has(key)) problems.push(`duplicate edge: ${key}`)
    seenEdges.add(key)
    if (!byId.has(e.from)) problems.push(`edge ${key} starts at unknown node "${e.from}"`)
    if (!byId.has(e.to)) problems.push(`edge ${key} ends at unknown node "${e.to}"`)
    if (e.from === e.to) problems.push(`edge ${key} is a self-loop`)
    touched.add(e.from)
    touched.add(e.to)
    if (byId.has(e.from) && byId.has(e.to) && rank(e.to) <= rank(e.from)) {
      problems.push(`edge ${key} does not flow to a later layer`)
    }
  }
  for (const n of nodes) if (!touched.has(n.id)) problems.push(`node ${n.id} is an orphan (no edges)`)
  return problems
}

/** Group nodes into the rows ArchDiagram draws, in layer order. Throws if the data is invalid. */
export function toLayers(nodes: readonly PipelineNode[], layers: readonly string[]): ArchLayer[] {
  for (const n of nodes) {
    if (!layers.includes(n.layer)) throw new Error(`node ${n.id} is in unknown layer "${n.layer}"`)
  }
  return layers.map((label) => ({ label, boxes: nodes.filter((n) => n.layer === label).map((n) => n.label) }))
}

/** The edges as readable "from -> to" lines, the text alternative to the diagram's arrows. */
export function describeEdges(nodes: readonly PipelineNode[], edges: readonly PipelineEdge[]): string[] {
  const label = new Map(nodes.map((n) => [n.id, n.label]))
  return edges.map((e) => `${label.get(e.from) ?? e.from} -> ${label.get(e.to) ?? e.to}`)
}

/* ---------------------------------------------------- one walk-forward fold */

export type FoldStepKind = 'features' | 'fit-scaler' | 'fit-model' | 'forecast' | 'predict' | 'actual' | 'score' | 'advance'

export interface FoldStep extends SeqStep {
  kind: FoldStepKind
}

/** The roles in ADR-0001 D4: ivf.evaluation.splits, ivf.features, the scaler, ivf.models, ivf.evaluation. */
export const FOLD_ACTORS = ['Splitter', 'Features', 'Scaler', 'Model', 'Evaluator']

/**
 * One fold. The cutoff is `c_k = idx(holdout_start) + 63k`. Parameters are fitted from data dated
 * up to the cutoff (L1), on pairs whose target is also dated up to it (L2), with the scaler fitted
 * on those training rows only (L3). They then stay frozen while the daily origins of the fold run.
 */
export const FOLD_STEPS: readonly FoldStep[] = [
  { kind: 'features', from: 0, to: 1, label: 'build(data <= cutoff)' },
  { kind: 'fit-scaler', from: 0, to: 2, label: 'fit(train only, L3)' },
  { kind: 'fit-model', from: 2, to: 3, label: 'fit(target <= cutoff)' },
  { kind: 'forecast', from: 0, to: 3, label: 'forecast(origin t, h), params frozen' },
  { kind: 'predict', from: 3, to: 4, label: 'y_hat(t + h)' },
  { kind: 'actual', from: 0, to: 4, label: 'y(t + h), the realised value' },
  { kind: 'score', from: 4, to: 0, label: 'loss rows: RMSE MAE MAPE QLIKE' },
  { kind: 'advance', from: 0, to: 0, label: 'next fold: cutoff += 63 sessions' },
]

/** The order in which the kinds must first occur: features and fitting before forecasting before scoring. */
export const FOLD_ORDER: readonly FoldStepKind[] = ['features', 'fit-scaler', 'fit-model', 'forecast', 'predict', 'actual', 'score', 'advance']

/** Problems with the fold sequence; empty means valid. */
export function validateFold(actors: readonly string[], steps: readonly FoldStep[]): string[] {
  const problems: string[] = []
  steps.forEach((s, i) => {
    for (const end of [s.from, s.to]) {
      if (!Number.isInteger(end) || end < 0 || end >= actors.length) problems.push(`step ${i} refers to missing actor ${end}`)
    }
    if (s.label.trim() === '') problems.push(`step ${i} has an empty label`)
  })
  const first = new Map<FoldStepKind, number>()
  steps.forEach((s, i) => {
    if (!first.has(s.kind)) first.set(s.kind, i)
  })
  let previous = -1
  for (const kind of FOLD_ORDER) {
    const at = first.get(kind)
    if (at === undefined) {
      problems.push(`the fold has no "${kind}" step`)
      continue
    }
    if (at < previous) problems.push(`"${kind}" happens before an earlier stage of the fold; no look-ahead requires ${FOLD_ORDER.join(' -> ')}`)
    previous = Math.max(previous, at)
  }
  return problems
}

/* ---------------------------------------------------- milestone process */

/**
 * The milestone process, in the same stages as docs/architecture/specs/process.yaml. Nodes are drawn
 * top to bottom in this order. The spec's "Fix" step is folded into the label of the loop back to
 * the review, to keep the diagram short.
 */
export const PROCESS_NODES: readonly ActNode[] = [
  { id: 'start', label: 'Milestone begins', type: 'start' },
  { id: 'build', label: 'Build the milestone', type: 'action' },
  { id: 'tests', label: 'pytest + mutation check', type: 'action' },
  { id: 'notebooks', label: 'notebooks --check', type: 'action' },
  { id: 'appverify', label: 'app: npm run verify', type: 'action' },
  { id: 'walk', label: 'Walkthrough + concepts', type: 'action' },
  { id: 'predict', label: 'Predict + explain back', type: 'action' },
  { id: 'review', label: 'Adversarial review', type: 'action' },
  { id: 'verdict', label: 'Any claim refuted?', type: 'decision' },
  { id: 'remeasure', label: 'Verifier re-measures', type: 'action' },
  { id: 'firstpush', label: 'First push?', type: 'decision' },
  { id: 'security', label: 'Security review', type: 'action' },
  { id: 'push', label: 'Commit and push', type: 'action' },
  { id: 'pause', label: 'Pause for the learner', type: 'end' },
]

export const PROCESS_EDGES: readonly ActEdge[] = [
  { from: 'start', to: 'build' },
  { from: 'build', to: 'tests' },
  { from: 'tests', to: 'notebooks' },
  { from: 'notebooks', to: 'appverify' },
  { from: 'appverify', to: 'walk' },
  { from: 'walk', to: 'predict' },
  { from: 'predict', to: 'review' },
  { from: 'review', to: 'verdict' },
  { from: 'verdict', to: 'review', label: 'yes: fix, re-review' },
  { from: 'verdict', to: 'remeasure', label: 'no' },
  { from: 'remeasure', to: 'firstpush' },
  { from: 'firstpush', to: 'security', label: 'yes' },
  { from: 'firstpush', to: 'push', label: 'no' },
  { from: 'security', to: 'push' },
  { from: 'push', to: 'pause' },
]

/** Longest label that fits an action box (180px wide, 11px monospace at ~6.6px a glyph, 8px padding each side). */
export const MAX_ACTION_CHARS = 24

/** Problems with the process flow; empty means valid. */
export function validateProcess(nodes: readonly ActNode[], edges: readonly ActEdge[]): string[] {
  const problems: string[] = []
  const ids = new Set<string>()
  for (const n of nodes) {
    if (ids.has(n.id)) problems.push(`duplicate node id: ${n.id}`)
    ids.add(n.id)
    if ((n.type ?? 'action') === 'action' && n.label.length > MAX_ACTION_CHARS) {
      problems.push(`action "${n.label}" is longer than ${MAX_ACTION_CHARS} characters and will overflow its box`)
    }
  }
  const starts = nodes.filter((n) => n.type === 'start')
  if (starts.length !== 1) problems.push(`exactly one start node is required, found ${starts.length}`)
  if (!nodes.some((n) => n.type === 'end')) problems.push('at least one end node is required')

  const out = new Map<string, ActEdge[]>()
  for (const e of edges) {
    if (!ids.has(e.from)) problems.push(`edge from unknown node "${e.from}"`)
    if (!ids.has(e.to)) problems.push(`edge to unknown node "${e.to}"`)
    out.set(e.from, [...(out.get(e.from) ?? []), e])
  }
  for (const n of nodes) {
    const leaving = out.get(n.id) ?? []
    if (n.type !== 'end' && leaving.length === 0) problems.push(`node ${n.id} is a dead end (no outgoing edge)`)
    if (n.type === 'decision') {
      if (leaving.length < 2) problems.push(`decision ${n.id} needs at least two outgoing edges`)
      if (leaving.some((e) => !e.label)) problems.push(`every edge leaving decision ${n.id} needs a label`)
    }
  }
  if (starts.length === 1) {
    const seen = new Set<string>([starts[0].id])
    const queue = [starts[0].id]
    while (queue.length > 0) {
      for (const e of out.get(queue.shift() as string) ?? []) {
        if (ids.has(e.to) && !seen.has(e.to)) {
          seen.add(e.to)
          queue.push(e.to)
        }
      }
    }
    for (const n of nodes) if (!seen.has(n.id)) problems.push(`node ${n.id} is unreachable from the start`)
  }
  return problems
}
