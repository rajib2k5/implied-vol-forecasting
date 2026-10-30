import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, within } from '@testing-library/react'
import race from '../../public/data/synthetic/model_race.json'
import evaluation from '../../public/data/synthetic/evaluation.json'
import { COMMITTED, serve } from '../test/committedData'
import ModelRacePage from './ModelRacePage'

afterEach(() => vi.unstubAllGlobals())

const panel = (target: string, h: number) => race.payload.panels.find((p) => p.target_id === target && p.horizon === h)!
const registry = (target: string, h: number) => panel(target, h).series.map((s) => s.model_id)
/** The page's defaults (rw, the panel's HAR, its structural model) take slots 1, 2, 3 in that order. */
const DEFAULT_SLOTS: Record<string, number> = { rw: 1, har_log: 2, arima_garch_log: 3, har_rv: 2, garch: 3 }
const drawn = (container: HTMLElement) => [...new Set(Array.from(container.querySelectorAll('polyline')).map((l) => l.getAttribute('data-series')))]

describe('ModelRacePage on the committed synthetic files', () => {
  it('fetches the synthetic race and labels it synthetic', async () => {
    const fetchMock = serve()
    render(<ModelRacePage />)
    await screen.findByRole('img')
    expect(fetchMock).toHaveBeenCalledWith('./data/synthetic/model_race.json')
    expect(screen.getByText('synthetic data')).toBeInTheDocument()
  })
  it('draws the actual and the default models of the first panel, each in its registry slot', async () => {
    const { container } = render((serve(), <ModelRacePage />))
    await screen.findByRole('img')
    // First panel in the file: vix_level h = 1. Defaults present there: rw, har_log, arima_garch_log.
    expect(race.payload.panels[0]).toMatchObject({ target_id: 'vix_level', horizon: 1 })
    expect(drawn(container)).toEqual(['actual', 'rw', 'har_log', 'arima_garch_log'])
    for (const id of ['rw', 'har_log', 'arima_garch_log'])
      expect(container.querySelector(`polyline[data-series="${id}"]`)).toHaveAttribute('stroke', `var(--series-${DEFAULT_SLOTS[id]})`)
    const slot = DEFAULT_SLOTS.har_log
    // Hiding rw must not repaint har_log (colour follows the entity, not its rank).
    fireEvent.click(screen.getByRole('checkbox', { name: 'rw' }))
    expect(drawn(container)).toEqual(['actual', 'har_log', 'arima_garch_log'])
    expect(container.querySelector('polyline[data-series="har_log"]')).toHaveAttribute('stroke', `var(--series-${slot})`)
    // The chart's points are the file's numbers: one point per origin of the panel.
    expect(container.querySelector('polyline[data-series="har_log"]')!.getAttribute('points')!.split(' ')).toHaveLength(panel('vix_level', 1).n)
  })
  it('switches to the rv21 panel and its own models', async () => {
    const { container } = render((serve(), <ModelRacePage />))
    await screen.findByRole('img')
    fireEvent.click(screen.getByRole('button', { name: /21-session realised volatility, h = 21/ }))
    expect(drawn(container)).toEqual(['actual', 'rw', 'har_rv', 'garch'])
    for (const id of ['rw', 'har_rv', 'garch'])
      expect(container.querySelector(`polyline[data-series="${id}"]`)).toHaveAttribute('stroke', `var(--series-${DEFAULT_SLOTS[id]})`)
    // The defaults first (they hold slots 1-3), then the other models in the file's registry order.
    const defaults = ['rw', 'har_rv', 'garch']
    expect(screen.getAllByRole('checkbox').map((c) => c.parentElement!.textContent!.trim())).toEqual([...defaults, ...registry('rv21', 21).filter((id) => !defaults.includes(id))])
  })
  it('shows the whole-period losses from evaluation.json, the lowest in bold', async () => {
    render((serve(), <ModelRacePage />))
    const table = await screen.findByRole('table', { name: 'Losses by model' })
    const losses = evaluation.payload.losses.filter((l) => l.target_id === 'vix_level' && l.horizon === 1)
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows.map((r) => r.querySelector('th')!.textContent)).toEqual(losses.map((l) => l.model_id))
    const best = losses.reduce((a, b) => (b.rmse_ratio_rw < a.rmse_ratio_rw ? b : a))
    expect(rows.filter((r) => r.querySelectorAll('td')[1].classList.contains('best'))).toHaveLength(1) // a tie at 4 decimals is broken
    const bestRow = rows.find((r) => r.querySelector('th')!.textContent === best.model_id)!
    expect(bestRow.querySelectorAll('td')[1]).toHaveClass('best')
    expect(bestRow.querySelectorAll('td')[1]).toHaveTextContent(String(best.rmse))
    const mz = panel('vix_level', 1).mz.find((m) => m.model_id === 'rw')!
    expect(rows.find((r) => r.querySelector('th')!.textContent === 'rw')!.querySelectorAll('td')[5]).toHaveTextContent(String(mz.beta))
  })
  it('greys out without its file, and keeps the chart without evaluation.json', async () => {
    render((serve(COMMITTED, ['evaluation']), <ModelRacePage />))
    expect(await screen.findByText(/needs evaluation.json/)).toBeInTheDocument()
    vi.unstubAllGlobals()
    render((serve(COMMITTED, ['model_race']), <ModelRacePage />))
    expect(await screen.findByRole('alert')).toHaveTextContent('model_race.json: HTTP 404')
  })
})
