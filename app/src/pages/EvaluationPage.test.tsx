import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, within } from '@testing-library/react'
import evaluation from '../../public/data/synthetic/evaluation.json'
import { serve } from '../test/committedData'
import EvaluationPage from './EvaluationPage'

// jsdom has no WebGL, so the three.js module is replaced by a stand-in that lists the bars it was
// given (one button each, which reports a hover); `behaviour.throws` simulates a renderer that dies.
type Bar = { modelId: string; horizon: number; value: number; slot: number | null; regime: { regime: string } }
const behaviour = vi.hoisted(() => ({ throws: false }))
vi.mock('./RibbonCanvas', () => ({
  default: () => null,
  ErrorSurfaceCanvas: ({ layout, onHover }: { layout: { bars: Bar[] }; onHover: (i: number, on: boolean) => void }) => {
    if (behaviour.throws) throw new Error('WebGL context creation failed')
    return (
      <div data-testid="surface-canvas">
        {layout.bars.map((b, i) => (
          <button key={i} type="button" data-bar={`${b.modelId}|${b.horizon}|${b.regime.regime}|${b.value}|${b.slot}`} onMouseEnter={() => onHover(i, true)} onMouseLeave={() => onHover(i, false)}>
            bar
          </button>
        ))}
      </div>
    )
  },
}))

/** Make detectWebGL() see (or not see) a context on every canvas jsdom creates. */
function stubWebGL(available: boolean) {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation((() => (available ? {} : null)) as never)
}

beforeEach(() => {
  behaviour.throws = false
  stubWebGL(false)
})
afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

const ev = evaluation.payload
/** One cell's expected verdict, worked out from the file's own row for that pair. */
function expected(target: string, h: number, loss: string, row: string, col: string) {
  const d = ev.dm.find((x) => x.target_id === target && x.horizon === h && x.loss === loss && [x.model_a, x.model_b].sort().join() === [row, col].sort().join())!
  return { p: d.p_holm, verdict: d.p_holm < 0.05 ? (d.lower_loss === row ? 'row' : 'col') : 'none' }
}
function cell(row: string, col: string) {
  const table = screen.getByRole('table', { name: '' }) as HTMLTableElement
  const cols = Array.from(table.tHead!.rows[0].cells).map((c) => c.textContent)
  const tr = Array.from(table.tBodies[0].rows).find((r) => r.cells[0].textContent === row)!
  return tr.cells[cols.indexOf(col)]
}

describe('EvaluationPage on the committed synthetic file', () => {
  it('draws a models x models DM grid for the chosen target, horizon and loss', async () => {
    render((serve(), <EvaluationPage />))
    await screen.findByText('synthetic data')
    const models = ev.losses.filter((l) => l.target_id === 'vix_level' && l.horizon === 1).map((l) => l.model_id)
    const table = document.querySelector('table.dm-matrix') as HTMLTableElement
    expect(table.tBodies[0].rows).toHaveLength(models.length)
    expect(table.querySelectorAll('td')).toHaveLength(models.length * models.length)
    const e = expected('vix_level', 1, 'sq_error', 'har_log', 'rw')
    expect(cell('har_log', 'rw')).toHaveAttribute('data-verdict', e.verdict)
    expect(cell('har_log', 'rw')).toHaveTextContent(String(e.p))
    expect(cell('rw', 'har_log')).toHaveAttribute('data-verdict', e.verdict === 'row' ? 'col' : e.verdict === 'col' ? 'row' : 'none')
    expect(cell('rw', 'rw')).toHaveAttribute('data-verdict', 'self')
  })
  it('switches target and loss', async () => {
    render((serve(), <EvaluationPage />))
    await screen.findByText('synthetic data')
    fireEvent.click(screen.getByRole('button', { name: 'rv21 h = 21' }))
    fireEvent.click(screen.getByRole('button', { name: 'QLIKE' }))
    const e = expected('rv21', 21, 'qlike', 'garch', 'rw')
    expect(cell('garch', 'rw')).toHaveAttribute('data-verdict', e.verdict)
    expect(cell('garch', 'rw')).toHaveTextContent(String(e.p))
  })
  it('lists the Mincer-Zarnowitz rows of the panel and the calibration count', async () => {
    render((serve(), <EvaluationPage />))
    const table = await screen.findByRole('table', { name: 'Mincer-Zarnowitz regressions' })
    const mz = ev.mz.filter((m) => m.target_id === 'vix_level' && m.horizon === 1)
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows.map((r) => r.querySelector('th')!.textContent)).toEqual(mz.map((m) => m.model_id))
    expect(rows[0].querySelectorAll('td')[4]).toHaveTextContent(String(mz[0].wald_p))
    expect(screen.getByText(new RegExp(`helped in ${ev.calibration.helps} of ${ev.calibration.cells} cells`))).toBeInTheDocument()
  })
  it('greys out when the file is missing', async () => {
    render((serve({}), <EvaluationPage />))
    expect(await screen.findByRole('alert')).toHaveTextContent('evaluation.json: HTTP 404')
  })
})

// --- the error surface (schema 1.2.0) ------------------------------------------------

const DEFAULTS = ['rw', 'har_log', 'har_rv', 'arima_garch_log', 'garch'] // the model race's defaults, slots 1-3 per target
type Cell = (typeof ev.regime_losses)[number]
const LABEL: Record<string, string> = { low: 'low VIX', mid: 'mid VIX', high: 'high VIX', contango: 'contango', backwardation: 'backwardation' }
/** The file's regime rows of one target and model set that have a ratio: what the 3D view must draw. */
const drawn = (target: string, models: string[], key: 'rmse_ratio_rw' | 'qlike_ratio_rw' = 'rmse_ratio_rw') =>
  ev.regime_losses.filter((r: Cell) => r.target_id === target && models.includes(r.model_id) && r[key] !== null)
const bars = () => screen.getAllByRole('button', { name: 'bar' }).map((b) => b.getAttribute('data-bar')!.split('|'))

describe('EvaluationPage error surface', () => {
  it('draws one bar per regime cell of the default models, coloured by their slots', async () => {
    stubWebGL(true)
    render((serve(), <EvaluationPage />))
    await screen.findByTestId('surface-canvas')
    const want = drawn('vix_level', DEFAULTS)
    expect(want.length).toBeGreaterThan(0)
    expect(bars().map(([m, h, g, v]) => `${m}|${h}|${g}|${v}`).sort()).toEqual(want.map((r) => `${r.model_id}|${r.horizon}|${r.regime}|${r.rmse_ratio_rw}`).sort())
    // vix_level's defaults are rw, har_log and arima_garch_log: slots 1, 2 and 3, as in the model race.
    const slotOf = Object.fromEntries(bars().map(([m, , , , slot]) => [m, slot]))
    expect(slotOf).toEqual({ rw: '1', har_log: '2', arima_garch_log: '3' })
    const legend = within(screen.getByRole('figure', { name: /Error surface/ })).getByRole('list', { name: 'Legend' })
    expect(within(legend).getAllByRole('listitem').map((li) => li.textContent)).toEqual(['rw', 'har_log', 'arima_garch_log'])
  })
  it('switches target and loss, and adds a model when it is ticked', async () => {
    stubWebGL(true)
    render((serve(), <EvaluationPage />))
    await screen.findByTestId('surface-canvas')
    fireEvent.click(screen.getByRole('button', { name: '21-session realised volatility' }))
    fireEvent.click(screen.getByRole('button', { name: 'QLIKE / RW' }))
    const want = drawn('rv21', DEFAULTS, 'qlike_ratio_rw')
    expect(bars().map(([m, h, g, v]) => `${m}|${h}|${g}|${v}`).sort()).toEqual(want.map((r) => `${r.model_id}|${r.horizon}|${r.regime}|${r.qlike_ratio_rw}`).sort())
    fireEvent.click(screen.getByRole('checkbox', { name: 'lstm_log' }))
    expect(bars()).toHaveLength(drawn('rv21', [...DEFAULTS, 'lstm_log'], 'qlike_ratio_rw').length)
    fireEvent.click(screen.getByRole('checkbox', { name: 'rw' }))
    expect(bars().some(([m]) => m === 'rw')).toBe(false)
  })
  it('reads out the bar under the pointer', async () => {
    stubWebGL(true)
    render((serve(), <EvaluationPage />))
    await screen.findByTestId('surface-canvas')
    const target = screen.getAllByRole('button', { name: 'bar' }).find((b) => b.getAttribute('data-bar')!.startsWith('har_log|21|high|'))!
    const row = ev.regime_losses.find((r: Cell) => r.target_id === 'vix_level' && r.model_id === 'har_log' && r.horizon === 21 && r.regime === 'high')!
    expect(screen.getByText('Point at a bar to read its value.')).toBeInTheDocument()
    fireEvent.mouseEnter(target)
    expect(screen.getByText(`har_log, h = 21, ${LABEL.high}: RMSE / RW ${row.rmse_ratio_rw}`)).toBeInTheDocument()
    fireEvent.mouseLeave(target)
    expect(screen.getByText('Point at a bar to read its value.')).toBeInTheDocument()
  })
  it('has a table view: every model per horizon, the lowest ratio in each regime column in bold', async () => {
    stubWebGL(true)
    render((serve(), <EvaluationPage />))
    await screen.findByTestId('surface-canvas')
    fireEvent.click(screen.getByRole('button', { name: 'Show table' }))
    expect(screen.queryByTestId('surface-canvas')).toBeNull()
    const table = screen.getByRole('table', { name: 'Loss ratio to the random walk by regime' })
    const header = within(table).getAllByRole('columnheader').map((c) => c.textContent)
    expect(header).toEqual(['h', 'Model', 'whole period', 'low VIX', 'mid VIX', 'high VIX', 'contango', 'backwardation'])
    const models = [...new Set(ev.regime_losses.filter((r: Cell) => r.target_id === 'vix_level').map((r: Cell) => r.model_id))]
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(3 * models.length)
    // h = 1, high VIX: the bold cell is the model with the file's lowest ratio there.
    const cells = ev.regime_losses.filter((r: Cell) => r.target_id === 'vix_level' && r.horizon === 1 && r.regime === 'high')
    const best = cells.reduce((a: Cell, b: Cell) => (b.rmse_ratio_rw < a.rmse_ratio_rw ? b : a))
    const col = header.indexOf('high VIX')
    const bold = rows.slice(0, models.length).filter((r) => r.querySelectorAll('td,th')[col].classList.contains('best'))
    expect(bold.map((r) => r.querySelector('th')!.textContent)).toEqual([best.model_id])
    expect(bold[0].querySelectorAll('td,th')[col]).toHaveTextContent(String(best.rmse_ratio_rw))
    fireEvent.click(screen.getByRole('button', { name: 'Show 3D' }))
    expect(await screen.findByTestId('surface-canvas')).toBeInTheDocument()
  })
  it('without WebGL shows the table and says why', async () => {
    render((serve(), <EvaluationPage />))
    expect(await screen.findByText(/The 3D view is unavailable \(this browser has no WebGL\)/)).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Loss ratio to the random walk by regime' })).toBeInTheDocument()
    expect(screen.queryByTestId('surface-canvas')).toBeNull()
  })
  it('falls back to the table when the renderer fails', async () => {
    stubWebGL(true)
    behaviour.throws = true
    vi.spyOn(console, 'error').mockImplementation(() => {}) // React logs the caught render error
    render((serve(), <EvaluationPage />))
    expect(await screen.findByText(/the 3D renderer failed to start/)).toBeInTheDocument()
    expect(screen.getByRole('table', { name: 'Loss ratio to the random walk by regime' })).toBeInTheDocument()
  })
  it('says so for a file without regime losses', async () => {
    const old = structuredClone(evaluation) as { payload: Record<string, unknown> }
    delete old.payload.regime_losses
    render((serve({ evaluation: old }), <EvaluationPage />))
    expect(await screen.findByText(/no losses by regime/)).toBeInTheDocument()
    expect(screen.queryByRole('table', { name: 'Loss ratio to the random walk by regime' })).toBeNull()
  })
})
