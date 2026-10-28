import { describe, expect, it } from 'vitest'
import {
  FOLD_ACTORS,
  FOLD_ORDER,
  FOLD_STEPS,
  MAX_ACTION_CHARS,
  MAX_LAYER_CHARS,
  PIPELINE_EDGES,
  PIPELINE_LAYERS,
  PIPELINE_NODES,
  PROCESS_EDGES,
  PROCESS_NODES,
  describeEdges,
  maxLabelChars,
  toLayers,
  validateFold,
  validatePipeline,
  validateProcess,
  type FoldStep,
  type PipelineEdge,
  type PipelineNode,
} from './architecture'
import type { ActEdge, ActNode } from './diagrams'

/* ------------------------------------------------------------ helpers */

/** Nodes reachable by following edges forward from `start`. */
function reachableFrom(start: string, edges: readonly { from: string; to: string }[]): Set<string> {
  const seen = new Set([start])
  const queue = [start]
  while (queue.length > 0) {
    const at = queue.shift() as string
    for (const e of edges) {
      if (e.from === at && !seen.has(e.to)) {
        seen.add(e.to)
        queue.push(e.to)
      }
    }
  }
  return seen
}

/* ------------------------------------------------------------ pipeline */

describe('maxLabelChars', () => {
  it('hand-worked: floor(((588 - 10 (n-1)) / n - 8) / 6.6)', () => {
    expect(maxLabelChars(1)).toBe(87) // (588 - 8) / 6.6 = 87.9
    expect(maxLabelChars(2)).toBe(42) // ((588 - 10) / 2 - 8) / 6.6 = 281 / 6.6 = 42.6
    expect(maxLabelChars(3)).toBe(27) // ((588 - 20) / 3 - 8) / 6.6 = 181.33 / 6.6 = 27.5
    expect(maxLabelChars(4)).toBe(19) // ((588 - 30) / 4 - 8) / 6.6 = 131.5 / 6.6 = 19.9
    expect(maxLabelChars(7)).toBe(10) // ((588 - 60) / 7 - 8) / 6.6 = 67.43 / 6.6 = 10.2
  })
})

describe('MAX_LAYER_CHARS', () => {
  it('hand-worked: the gutter runs from x = 8 to x = 118 at 7px a glyph, floor(110 / 7) = floor(15.7) = 15', () => {
    expect(MAX_LAYER_CHARS).toBe(15)
  })
  it('lets every real layer name fit, including the longest', () => {
    expect(Math.max(...PIPELINE_LAYERS.map((l) => l.length))).toBeLessThanOrEqual(MAX_LAYER_CHARS)
    expect('ivf.models M3+').toHaveLength(14) // 'ivf.models' is 10, ' M3+' is 4
  })
})

describe('validatePipeline on the real pipeline', () => {
  it('has no problems', () => {
    expect(validatePipeline(PIPELINE_NODES, PIPELINE_EDGES, PIPELINE_LAYERS)).toEqual([])
  })
  it('puts every node in a known layer and fills every layer', () => {
    for (const n of PIPELINE_NODES) expect(PIPELINE_LAYERS).toContain(n.layer)
    for (const layer of PIPELINE_LAYERS) expect(PIPELINE_NODES.some((n) => n.layer === layer)).toBe(true)
  })
  it('lets every source reach the app, and no data flow out of the app or into a source', () => {
    const sources = PIPELINE_NODES.filter((n) => n.layer === PIPELINE_LAYERS[0]).map((n) => n.id)
    expect(sources.length).toBeGreaterThan(0)
    for (const s of sources) expect(reachableFrom(s, PIPELINE_EDGES).has('app'), s).toBe(true)
    expect(PIPELINE_EDGES.filter((e) => e.from === 'app')).toEqual([])
    expect(PIPELINE_EDGES.filter((e) => sources.includes(e.to))).toEqual([])
  })
  it('routes every model through the shared splits into the forecasts table', () => {
    const models = PIPELINE_NODES.filter((n) => n.layer === 'ivf.models M2' || n.layer === 'ivf.models M3+').map((n) => n.id)
    expect(models).toHaveLength(7) // RW, hist mean, ARIMA, HAR, GARCH family, ARIMA-GARCH, LSTM
    for (const m of models) {
      expect(PIPELINE_EDGES, `${m} reads the splits`).toContainEqual({ from: 'splits', to: m })
      expect(PIPELINE_EDGES, `${m} reads the processed features`).toContainEqual({ from: 'processed', to: m })
      expect(PIPELINE_EDGES, `${m} writes forecasts`).toContainEqual({ from: m, to: 'forecasts' })
    }
  })
  it('exports after the evaluation, the backtest and the forecasts, and only into the two JSON locations', () => {
    expect(PIPELINE_EDGES).toContainEqual({ from: 'eval', to: 'export' })
    expect(PIPELINE_EDGES).toContainEqual({ from: 'backtest', to: 'export' })
    expect(PIPELINE_EDGES).toContainEqual({ from: 'forecasts', to: 'export' })
    expect(PIPELINE_EDGES.filter((e) => e.from === 'export').map((e) => e.to)).toEqual(['synthjson', 'realjson'])
    expect(PIPELINE_EDGES.filter((e) => e.to === 'app')).toEqual([
      { from: 'synthjson', to: 'app' },
      { from: 'realjson', to: 'app' },
    ])
  })
  it('hand-worked: 10 edges above the models, 21 for the models, 4 into evaluation and backtest, 11 below = 46', () => {
    // sources to fetch 3; fetch->raw, synth->synraw 2; raw/synraw->interim 2; interim->features 1;
    // features->processed 1; processed->splits 1  => 10
    // models: splits->m, processed->m, m->forecasts for each of 7 models => 21
    // forecasts->eval, processed->eval, forecasts->backtest, interim->backtest => 4
    // eval/backtest->reports 2; eval/backtest/forecasts/processed/interim->export 5; export->2 JSON 2; 2 JSON->app 2 => 11
    expect(PIPELINE_EDGES).toHaveLength(10 + 21 + 4 + 11) // 46
  })
  it('names the modules and paths ADR-0001 uses', () => {
    const labels = PIPELINE_NODES.map((n) => n.label).join(' | ')
    for (const name of [
      'ivf.data.fetch',
      'ivf.data.synthetic',
      'ivf.data.build_interim',
      'ROOT/raw/{source}/{symbol}/{sha256}.csv',
      'ROOT/interim/{table}.parquet',
      'ROOT/processed/run_id=<id>/{table}.parquet',
      'ivf.features',
      'ivf.evaluation.splits',
      'ivf.evaluation:',
      'ivf.backtest:',
      'ivf.reports',
      'tools/export_app_data.py',
    ]) {
      expect(labels, name).toContain(name)
    }
  })
})

describe('the synthetic / real data split (ADR-0001 D2)', () => {
  const node = (id: string) => PIPELINE_NODES.find((n) => n.id === id) as PipelineNode
  it('commits the synthetic JSON fixture under app/public/data/synthetic', () => {
    expect(node('synthjson')).toMatchObject({ data: 'synthetic', committed: true })
    expect(node('synthjson').label).toContain('app/public/data/synthetic/')
  })
  it('keeps the real JSON outside the repository, not under app/', () => {
    expect(node('realjson')).toMatchObject({ data: 'real-series', committed: false })
    expect(node('realjson').label).toContain('ROOT/app-data/real/')
    expect(PIPELINE_NODES.filter((n) => n.label.includes('app/public/data/real'))).toEqual([])
    expect(node('realjson').label.startsWith('app/')).toBe(false)
  })
  it('commits only the synthetic fixture and the aggregate reports', () => {
    expect(PIPELINE_NODES.filter((n) => n.committed).map((n) => n.id)).toEqual(['reports', 'synthjson'])
    expect(node('reports').data).toBe('real-aggregate')
  })
  it('says the committed reports hold the publication set P only (ADR-0001 D2)', () => {
    expect(node('reports').label).toContain('set P only')
    expect(node('reports').label).not.toContain('aggregates') // the old per-cell rule's wording
  })
  it('never marks a node that holds real daily series as committed', () => {
    for (const n of PIPELINE_NODES.filter((x) => x.data === 'real-series')) expect(n.committed, n.id).toBe(false)
  })
  it('runs the synthetic fixture through the same interim layer as the real data', () => {
    expect(PIPELINE_EDGES).toContainEqual({ from: 'synraw', to: 'interim' })
    expect(PIPELINE_EDGES).toContainEqual({ from: 'raw', to: 'interim' })
  })
})

describe('validatePipeline flags broken graphs', () => {
  const layers = ['A', 'B']
  const nodes: PipelineNode[] = [
    { id: 'a', label: 'alpha', layer: 'A' },
    { id: 'b', label: 'beta', layer: 'B' },
  ]
  const edges: PipelineEdge[] = [{ from: 'a', to: 'b' }]
  const problems = (n: PipelineNode[], e: PipelineEdge[], l = layers) => validatePipeline(n, e, l).join(' | ')

  it('control: the minimal graph is valid', () => {
    expect(validatePipeline(nodes, edges, layers)).toEqual([])
  })
  it('flags an edge to a node that does not exist', () => {
    expect(problems(nodes, [...edges, { from: 'a', to: 'ghost' }])).toContain('ends at unknown node "ghost"')
    expect(problems(nodes, [...edges, { from: 'ghost', to: 'b' }])).toContain('starts at unknown node "ghost"')
  })
  it('flags an orphan node', () => {
    expect(problems([...nodes, { id: 'c', label: 'gamma', layer: 'B' }], edges)).toContain('node c is an orphan')
  })
  it('flags backward, same-layer and self-loop edges', () => {
    expect(problems(nodes, [{ from: 'b', to: 'a' }])).toContain('does not flow to a later layer')
    const sameLayer = [...nodes, { id: 'c', label: 'gamma', layer: 'A' }]
    expect(problems(sameLayer, [...edges, { from: 'a', to: 'c' }, { from: 'c', to: 'b' }])).toContain('edge a->c does not flow to a later layer')
    expect(problems(nodes, [...edges, { from: 'a', to: 'a' }])).toContain('self-loop')
  })
  it('flags duplicate ids, duplicate edges and duplicate labels in one layer', () => {
    expect(problems([...nodes, { id: 'a', label: 'again', layer: 'A' }], edges)).toContain('duplicate node id: a')
    expect(problems(nodes, [...edges, { from: 'a', to: 'b' }])).toContain('duplicate edge: a->b')
    const twin = [...nodes, { id: 'b2', label: 'beta', layer: 'B' }]
    expect(problems(twin, [...edges, { from: 'a', to: 'b2' }])).toContain('repeats the label "beta"')
  })
  it('flags an unknown layer and an empty layer', () => {
    expect(problems([...nodes, { id: 'c', label: 'gamma', layer: 'Z' }], [...edges, { from: 'a', to: 'c' }])).toContain('unknown layer "Z"')
    expect(problems(nodes, edges, ['A', 'B', 'C'])).toContain('layer "C" has no nodes')
  })
  it('flags real daily series marked as committed, but not synthetic or aggregate data', () => {
    const withData = (data: PipelineNode['data'], committed: boolean | undefined): PipelineNode[] => [{ ...nodes[0], data, committed }, nodes[1]]
    expect(problems(withData('real-series', true), edges)).toContain('node a would commit real daily series')
    expect(validatePipeline(withData('real-series', false), edges, layers)).toEqual([])
    expect(validatePipeline(withData('synthetic', true), edges, layers)).toEqual([])
    expect(validatePipeline(withData('real-aggregate', true), edges, layers)).toEqual([])
  })
  it('flags a committed node that does not say what data it holds', () => {
    expect(problems([{ ...nodes[0], committed: true }, nodes[1]], edges)).toContain('node a is marked committed but does not say what data it holds')
  })
  it('flags a layer name too long for the gutter, but not one exactly at the limit', () => {
    const long = 'x'.repeat(MAX_LAYER_CHARS + 1)
    expect(problems([{ id: 'a', label: 'alpha', layer: long }, nodes[1]], edges, [long, 'B'])).toContain(`layer name "${long}" (16 chars) is too long`)
    const exact = 'x'.repeat(MAX_LAYER_CHARS)
    expect(validatePipeline([{ id: 'a', label: 'alpha', layer: exact }, nodes[1]], edges, [exact, 'B'])).toEqual([])
  })
  it('flags a label too long for its row, using the row width rule', () => {
    const long = 'x'.repeat(maxLabelChars(1) + 1) // 88 chars in a one-box row
    expect(problems([{ id: 'a', label: long, layer: 'A' }, nodes[1]], edges)).toContain('too long for a row of 1 boxes')
    const fits = 'x'.repeat(maxLabelChars(1))
    expect(validatePipeline([{ id: 'a', label: fits, layer: 'A' }, nodes[1]], edges, layers)).toEqual([])
  })
})

describe('toLayers and describeEdges', () => {
  it('groups labels into rows in layer order, keeping node order inside a row', () => {
    const rows = toLayers(PIPELINE_NODES, PIPELINE_LAYERS)
    expect(rows.map((r) => r.label)).toEqual([...PIPELINE_LAYERS])
    expect(rows.find((r) => r.label === 'ivf.models M2')?.boxes).toEqual(['random walk', 'historical mean', 'ARIMA (log-IV)', 'HAR'])
    expect(rows.flatMap((r) => r.boxes)).toHaveLength(PIPELINE_NODES.length)
  })
  it('refuses a node whose layer is not listed', () => {
    expect(() => toLayers([{ id: 'a', label: 'x', layer: 'Nope' }], ['A'])).toThrow('unknown layer "Nope"')
  })
  it('renders one readable line per edge, falling back to the id for unknown nodes', () => {
    const nodes: PipelineNode[] = [
      { id: 'a', label: 'alpha', layer: 'A' },
      { id: 'b', label: 'beta', layer: 'B' },
    ]
    expect(describeEdges(nodes, [{ from: 'a', to: 'b' }, { from: 'b', to: 'zzz' }])).toEqual(['alpha -> beta', 'beta -> zzz'])
    expect(describeEdges(PIPELINE_NODES, PIPELINE_EDGES)).toHaveLength(PIPELINE_EDGES.length)
  })
})

/* ------------------------------------------------------ walk-forward fold */

describe('the walk-forward fold sequence', () => {
  it('has no problems', () => {
    expect(validateFold(FOLD_ACTORS, FOLD_STEPS)).toEqual([])
  })
  it('builds the features and fits the scaler on the training window before anything is forecast or scored', () => {
    const first = (kind: FoldStep['kind']) => FOLD_STEPS.findIndex((s) => s.kind === kind)
    expect(first('features')).toBe(0)
    expect(first('fit-scaler')).toBe(1)
    expect(FOLD_STEPS[first('fit-scaler')].label).toContain('train only')
    for (let i = 1; i < FOLD_ORDER.length; i++) {
      expect(first(FOLD_ORDER[i]), FOLD_ORDER[i]).toBeGreaterThan(first(FOLD_ORDER[i - 1]))
    }
  })
  it('states the ADR-0001 D4 leak guards: label availability, frozen parameters, a 63-session fold', () => {
    const label = (kind: FoldStep['kind']) => FOLD_STEPS.find((s) => s.kind === kind)?.label ?? ''
    expect(label('fit-model')).toContain('target <= cutoff') // L2: a pair is usable only if its target is dated up to the cutoff
    expect(label('forecast')).toContain('params frozen') // parameters do not change inside a fold
    expect(label('advance')).toContain('63 sessions') // fold_len
  })
  it('names the five roles of ADR-0001 D4 in the order the data passes through them', () => {
    expect(FOLD_ACTORS).toEqual(['Splitter', 'Features', 'Scaler', 'Model', 'Evaluator'])
  })
  it('only talks between actors that exist', () => {
    for (const s of FOLD_STEPS) {
      expect(s.from).toBeLessThan(FOLD_ACTORS.length)
      expect(s.to).toBeLessThan(FOLD_ACTORS.length)
    }
  })
  describe('flags broken folds', () => {
    const swap = (i: number, j: number) => {
      const copy = [...FOLD_STEPS]
      ;[copy[i], copy[j]] = [copy[j], copy[i]]
      return copy
    }
    it('flags a forecast that happens before the scaler is fitted (look-ahead)', () => {
      const scalerAt = FOLD_STEPS.findIndex((s) => s.kind === 'fit-scaler')
      const forecastAt = FOLD_STEPS.findIndex((s) => s.kind === 'forecast')
      expect(validateFold(FOLD_ACTORS, swap(scalerAt, forecastAt)).join(' | ')).toContain('look-ahead')
    })
    it('flags a forecast that happens before the features are built', () => {
      const featuresAt = FOLD_STEPS.findIndex((s) => s.kind === 'features')
      const forecastAt = FOLD_STEPS.findIndex((s) => s.kind === 'forecast')
      expect(validateFold(FOLD_ACTORS, swap(featuresAt, forecastAt)).join(' | ')).toContain('look-ahead')
    })
    it('flags scoring before predicting', () => {
      const predictAt = FOLD_STEPS.findIndex((s) => s.kind === 'predict')
      const scoreAt = FOLD_STEPS.findIndex((s) => s.kind === 'score')
      expect(validateFold(FOLD_ACTORS, swap(predictAt, scoreAt)).join(' | ')).toContain('look-ahead')
    })
    it('flags a missing stage', () => {
      expect(validateFold(FOLD_ACTORS, FOLD_STEPS.filter((s) => s.kind !== 'fit-scaler')).join(' | ')).toContain('no "fit-scaler" step')
    })
    it('flags a step that points at a missing actor and an empty label', () => {
      const scalerAt = FOLD_STEPS.findIndex((s) => s.kind === 'fit-scaler')
      const bad: FoldStep[] = FOLD_STEPS.map((s, i) => (i === scalerAt ? { kind: 'fit-scaler', from: 0, to: 9, label: '  ' } : s))
      const out = validateFold(FOLD_ACTORS, bad).join(' | ')
      expect(out).toContain('missing actor 9')
      expect(out).toContain('empty label')
    })
  })
})

/* ------------------------------------------------------ milestone process */

describe('the milestone process flow', () => {
  it('has no problems', () => {
    expect(validateProcess(PROCESS_NODES, PROCESS_EDGES)).toEqual([])
  })
  it('cannot reach "push" without passing through the adversarial review', () => {
    // Remove the review node and its edges: push must become unreachable from the start.
    const withoutReview = PROCESS_EDGES.filter((e) => e.from !== 'review' && e.to !== 'review')
    expect(reachableFrom('start', PROCESS_EDGES).has('push')).toBe(true)
    expect(reachableFrom('start', withoutReview).has('push')).toBe(false)
  })
  it('cannot reach "push" without the verifier re-measuring the numbers', () => {
    const withoutVerifier = PROCESS_EDGES.filter((e) => e.from !== 'remeasure' && e.to !== 'remeasure')
    expect(reachableFrom('start', withoutVerifier).has('push')).toBe(false)
  })
  it('loops back to the review when a claim is refuted, and moves on to the verifier when none is', () => {
    expect(PROCESS_EDGES).toContainEqual({ from: 'verdict', to: 'review', label: 'yes: fix, re-review' })
    expect(PROCESS_EDGES).toContainEqual({ from: 'verdict', to: 'remeasure', label: 'no' })
  })
  it('runs the security review on the first push only: it sits on one branch, and the other goes straight to the push', () => {
    expect(PROCESS_EDGES).toContainEqual({ from: 'firstpush', to: 'security', label: 'yes' })
    expect(PROCESS_EDGES).toContainEqual({ from: 'firstpush', to: 'push', label: 'no' })
    expect(PROCESS_EDGES).toContainEqual({ from: 'security', to: 'push' })
    const withoutSecurity = PROCESS_EDGES.filter((e) => e.from !== 'security' && e.to !== 'security')
    expect(reachableFrom('start', withoutSecurity).has('push')).toBe(true) // the "no" branch still pushes
  })
  it('ends with a pause for the learner after the push', () => {
    expect(PROCESS_EDGES).toContainEqual({ from: 'push', to: 'pause' })
    expect(PROCESS_NODES.find((n) => n.id === 'pause')?.type).toBe('end')
  })
  it('runs the checks in the order build, tests, notebooks, app verify, before the walkthrough', () => {
    const order = ['build', 'tests', 'notebooks', 'appverify', 'walk', 'predict', 'review']
    for (let i = 1; i < order.length; i++) {
      expect(PROCESS_EDGES, `${order[i - 1]} -> ${order[i]}`).toContainEqual({ from: order[i - 1], to: order[i] })
    }
  })
  it('hand-worked: an action box fits floor((180 - 2 * 8) / 6.6) = floor(24.85) = 24 characters', () => {
    expect(MAX_ACTION_CHARS).toBe(24)
  })
  it('keeps every action label inside its box', () => {
    for (const n of PROCESS_NODES.filter((x) => (x.type ?? 'action') === 'action')) {
      expect(n.label.length, n.label).toBeLessThanOrEqual(MAX_ACTION_CHARS)
    }
  })

  describe('flags broken flows', () => {
    const nodes: ActNode[] = [
      { id: 's', label: 'go', type: 'start' },
      { id: 'a', label: 'work', type: 'action' },
      { id: 'e', label: 'done', type: 'end' },
    ]
    const edges: ActEdge[] = [
      { from: 's', to: 'a' },
      { from: 'a', to: 'e' },
    ]
    const problems = (n: ActNode[], e: ActEdge[]) => validateProcess(n, e).join(' | ')

    it('control: the minimal flow is valid', () => {
      expect(validateProcess(nodes, edges)).toEqual([])
    })
    it('flags missing or repeated start nodes and a missing end', () => {
      expect(problems(nodes.filter((n) => n.id !== 's'), [{ from: 'a', to: 'e' }])).toContain('exactly one start node')
      expect(problems([...nodes, { id: 's2', label: 'again', type: 'start' }], [...edges, { from: 's2', to: 'a' }])).toContain('exactly one start node')
      expect(problems(nodes.filter((n) => n.id !== 'e'), [{ from: 's', to: 'a' }])).toContain('at least one end node')
    })
    it('flags edges to nodes that do not exist and duplicate ids', () => {
      expect(problems(nodes, [...edges, { from: 'a', to: 'ghost' }])).toContain('edge to unknown node "ghost"')
      expect(problems(nodes, [...edges, { from: 'ghost', to: 'a' }])).toContain('edge from unknown node "ghost"')
      expect(problems([...nodes, { id: 'a', label: 'dup', type: 'action' }], edges)).toContain('duplicate node id: a')
    })
    it('flags a dead end and an unreachable node', () => {
      expect(problems(nodes, [{ from: 's', to: 'a' }])).toContain('node a is a dead end')
      const island: ActNode[] = [...nodes, { id: 'x', label: 'island', type: 'action' }]
      const out = problems(island, [...edges, { from: 'x', to: 'e' }])
      expect(out).toContain('node x is unreachable from the start')
    })
    it('flags a decision with one exit or an unlabelled exit', () => {
      const dec: ActNode[] = [{ id: 's', label: 'go', type: 'start' }, { id: 'd', label: 'ok?', type: 'decision' }, { id: 'e', label: 'done', type: 'end' }]
      expect(problems(dec, [{ from: 's', to: 'd' }, { from: 'd', to: 'e', label: 'yes' }])).toContain('decision d needs at least two outgoing edges')
      expect(problems(dec, [{ from: 's', to: 'd' }, { from: 'd', to: 'e' }, { from: 'd', to: 's' }])).toContain('every edge leaving decision d needs a label')
      expect(validateProcess(dec, [{ from: 's', to: 'd' }, { from: 'd', to: 'e', label: 'yes' }, { from: 'd', to: 's', label: 'no' }])).toEqual([])
    })
    it('flags an action label that would overflow its box, but not one exactly at the limit', () => {
      const long = 'x'.repeat(MAX_ACTION_CHARS + 1)
      expect(problems([nodes[0], { id: 'a', label: long, type: 'action' }, nodes[2]], edges)).toContain('will overflow')
      const exact = 'x'.repeat(MAX_ACTION_CHARS)
      expect(validateProcess([nodes[0], { id: 'a', label: exact, type: 'action' }, nodes[2]], edges)).toEqual([])
    })
    it('treats a node without a type as an action when checking label length', () => {
      const long = 'x'.repeat(MAX_ACTION_CHARS + 1)
      expect(problems([nodes[0], { id: 'a', label: long }, nodes[2]], edges)).toContain('will overflow')
    })
  })
})
