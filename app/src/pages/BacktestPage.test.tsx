import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, within } from '@testing-library/react'
import equity from '../../public/data/synthetic/equity_curves.json'
import { serve } from '../test/committedData'
import BacktestPage from './BacktestPage'

afterEach(() => vi.unstubAllGlobals())

const eq = equity.payload

describe('BacktestPage on the committed synthetic file', () => {
  it('draws one equity line per strategy on a log axis, each in its own slot', async () => {
    const { container } = render((serve(), <BacktestPage />))
    await screen.findByRole('img')
    expect(screen.getByText('synthetic data')).toBeInTheDocument()
    const lines = Array.from(container.querySelectorAll('polyline'))
    expect(lines.map((l) => l.getAttribute('data-series'))).toEqual(eq.strategies.map((s) => s.strategy_id))
    // The three traded series take slots 1-3 in file order; cash is the dashed ink reference line.
    expect(eq.strategies.map((s) => s.strategy_id)).toEqual(['fvix_rv21_b2', 'bench_long_vxx', 'bench_long_svxy', 'bench_cash'])
    expect(lines.map((l) => l.getAttribute('stroke'))).toEqual(['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--ink)'])
    expect(lines[3]).toHaveAttribute('stroke-dasharray')
    expect(lines[0].getAttribute('points')!.split(' ')).toHaveLength(eq.dates.length)
    expect(screen.getByRole('img', { name: /growth of 1 \(log scale\)/ })).toBeInTheDocument()
  })
  it('shows PSR and DSR per strategy at the default cost', async () => {
    render((serve(), <BacktestPage />))
    const table = await screen.findByRole('table', { name: 'Backtest summary' })
    const rows = within(table).getAllByRole('row').slice(1)
    const label = (id: string) => eq.strategies.find((s) => s.strategy_id === id)!.label
    expect(rows.map((r) => r.querySelector('th')!.textContent)).toEqual(eq.summary.map((r) => label(r.strategy_id)))
    const cells = rows[0].querySelectorAll('td')
    expect([cells[1].textContent, cells[2].textContent, cells[3].textContent]).toEqual([String(eq.summary[0].sharpe_ann), String(eq.summary[0].psr0), String(eq.summary[0].dsr)])
  })
  it('lays out the cost sensitivity as strategy x cost level', async () => {
    render((serve(), <BacktestPage />))
    const table = await screen.findByRole('table', { name: 'Cost sensitivity' })
    const costs = [...new Set(eq.cost_sensitivity.map((r) => r.cost_bps))]
    expect(within(table).getAllByRole('columnheader').slice(1).map((c) => c.textContent)).toEqual(costs.map((c) => `${c} bps`))
    const primary = eq.strategies[0].strategy_id
    const at20 = eq.cost_sensitivity.find((r) => r.strategy_id === primary && r.cost_bps === 20)!
    expect(within(table).getAllByRole('row')[1].querySelectorAll('td')[costs.indexOf(20)]).toHaveTextContent(String(at20.sharpe_ann))
    for (const c of eq.caveats) expect(screen.getByText(c)).toBeInTheDocument()
  })
  it('says so for a file without the optional cost table', async () => {
    const old = structuredClone(equity) as { payload: Record<string, unknown> }
    delete old.payload.cost_sensitivity
    render((serve({ equity_curves: old }), <BacktestPage />))
    expect(await screen.findByText(/no cost-sensitivity table/)).toBeInTheDocument()
  })
})

// --- regime ribbons (schema 1.2.0) ---------------------------------------------------

type Segment = (typeof eq.regime_segments)[number]
const LABEL: Record<string, string> = { low: 'low VIX', mid: 'mid VIX', high: 'high VIX', contango: 'contango', backwardation: 'backwardation' }
const segmentsOf = (scheme: string) => eq.regime_segments.filter((s: Segment) => s.scheme === scheme)

describe('BacktestPage regime ribbons', () => {
  it('shades the VIX-level regimes behind the equity lines by default, one band per segment', async () => {
    const { container } = render((serve(), <BacktestPage />))
    await screen.findByRole('img')
    const want = segmentsOf('vix_level')
    expect(want.length).toBeGreaterThan(0)
    const rects = Array.from(container.querySelectorAll('rect[data-band]'))
    expect(rects.map((r) => r.getAttribute('data-band'))).toEqual(want.map((s: Segment) => LABEL[s.regime_id]))
    // Darker for more stress: low, mid and high VIX take the three steps of the one-hue ramp.
    const shade = (label: string) => rects.find((r) => r.getAttribute('data-band') === label)!.getAttribute('class')
    expect([shade('low VIX'), shade('mid VIX'), shade('high VIX')]).toEqual(['viz-shade-fill viz-shade-0', 'viz-shade-fill viz-shade-1', 'viz-shade-fill viz-shade-2'])
    const legend = screen.getByRole('list', { name: 'Regime (VIX level)' })
    expect(within(legend).getAllByRole('listitem').map((li) => li.textContent)).toEqual(['low VIX', 'mid VIX', 'high VIX'])
    expect(screen.getByRole('button', { name: 'VIX level' })).toHaveAttribute('aria-pressed', 'true')
  })
  it('switches to the term structure and turns the ribbons off', async () => {
    const { container } = render((serve(), <BacktestPage />))
    await screen.findByRole('img')
    fireEvent.click(screen.getByRole('button', { name: 'term structure' }))
    const want = segmentsOf('term_structure')
    expect(Array.from(container.querySelectorAll('rect[data-band]')).map((r) => r.getAttribute('data-band'))).toEqual(want.map((s: Segment) => LABEL[s.regime_id]))
    expect(within(screen.getByRole('list', { name: 'Regime (term structure)' })).getAllByRole('listitem').map((li) => li.textContent)).toEqual(['contango', 'backwardation'])
    fireEvent.click(screen.getByRole('button', { name: 'off' }))
    expect(container.querySelectorAll('rect[data-band]')).toHaveLength(0)
    expect(screen.queryByRole('list', { name: /^Regime/ })).toBeNull()
    // The lines do not move when the ribbons change.
    expect(container.querySelectorAll('polyline')).toHaveLength(eq.strategies.length)
  })
  it('says so for a file without ribbons, and offers no ribbon control', async () => {
    const old = structuredClone(equity) as { payload: Record<string, unknown> }
    old.payload.regime_segments = []
    const { container } = render((serve({ equity_curves: old }), <BacktestPage />))
    expect(await screen.findByText(/no regime ribbons/)).toBeInTheDocument()
    expect(screen.queryByRole('group', { name: 'Regime ribbons' })).toBeNull()
    expect(container.querySelectorAll('rect[data-band]')).toHaveLength(0)
  })
})
