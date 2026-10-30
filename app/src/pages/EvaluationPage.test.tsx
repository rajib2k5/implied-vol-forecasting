import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, within } from '@testing-library/react'
import evaluation from '../../public/data/synthetic/evaluation.json'
import { serve } from '../test/committedData'
import EvaluationPage from './EvaluationPage'

afterEach(() => vi.unstubAllGlobals())

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
