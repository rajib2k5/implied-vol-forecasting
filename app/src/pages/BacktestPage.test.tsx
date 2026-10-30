import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
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
