import { describe, expect, it } from 'vitest'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { BarChart } from './BarChart'
import { XYChart, type XYSeries } from './XYChart'

const X = ['2024-01-01', '2024-01-02', '2024-01-03']
const series: XYSeries[] = [
  { id: 'actual', label: 'actual', values: [10, 20, null], slot: null },
  { id: 'rw', label: 'rw', values: [11, 19, 21], slot: 1 },
]

/** jsdom has no layout: give the hit area a 300px-wide box at x = 0. */
function stubBox(el: Element) {
  el.getBoundingClientRect = () => ({ left: 0, width: 300, top: 0, height: 100, right: 300, bottom: 100, x: 0, y: 0, toJSON: () => ({}) })
}

describe('XYChart', () => {
  it('draws one line per run of values, in the slot colour, and breaks at a null', () => {
    const { container } = render(<XYChart title="T" x={X} series={series} yLabel="vol" />)
    const lines = container.querySelectorAll('polyline')
    expect(lines).toHaveLength(2) // actual has one run (two points), rw one run of three
    expect(container.querySelector('polyline[data-series="rw"]')).toHaveAttribute('stroke', 'var(--series-1)')
    expect(container.querySelector('polyline[data-series="actual"]')).toHaveAttribute('stroke', 'var(--ink)')
    expect(container.querySelector('polyline[data-series="actual"]')!.getAttribute('points')!.split(' ')).toHaveLength(2)
  })
  it('has a legend and direct labels for two to four series, and no legend for one', () => {
    const { container, rerender } = render(<XYChart title="T" x={X} series={series} yLabel="vol" />)
    expect(within(screen.getByRole('list', { name: 'Legend' })).getAllByRole('listitem').map((li) => li.textContent)).toEqual(['actual', 'rw'])
    expect(container.querySelectorAll('.viz-direct')).toHaveLength(2)
    rerender(<XYChart title="T" x={X} series={[series[1]]} yLabel="vol" />)
    expect(screen.queryByRole('list', { name: 'Legend' })).toBeNull()
    const five = [1, 2, 3, 4, 5].map((k) => ({ id: `m${k}`, label: `m${k}`, values: [k, k, k], slot: k }))
    rerender(<XYChart title="T" x={X} series={five} yLabel="vol" />)
    expect(container.querySelectorAll('.viz-direct')).toHaveLength(0) // more than four: legend only
  })
  it('shows a crosshair and a tooltip with every value at the hovered date', () => {
    render(<XYChart title="T" x={X} series={series} yLabel="vol" />)
    const hit = screen.getByTestId('viz-hit')
    stubBox(hit)
    fireEvent.mouseMove(hit, { clientX: 290 }) // 290 / 300 * 2 = 1.93 -> index 2
    const tip = screen.getByRole('tooltip')
    expect(tip).toHaveTextContent('2024-01-03')
    expect(tip).toHaveTextContent('rw: 21')
    expect(tip).toHaveTextContent('actual: –')
    fireEvent.mouseLeave(hit)
    expect(screen.queryByRole('tooltip')).toBeNull()
  })
  it('moves the crosshair with the arrow keys', () => {
    render(<XYChart title="T" x={X} series={series} yLabel="vol" />)
    const svg = screen.getByRole('img')
    fireEvent.keyDown(svg, { key: 'ArrowRight' })
    expect(screen.getByRole('tooltip')).toHaveTextContent('2024-01-01')
    fireEvent.keyDown(svg, { key: 'ArrowRight' })
    expect(screen.getByRole('tooltip')).toHaveTextContent('rw: 19')
    fireEvent.keyDown(svg, { key: 'ArrowLeft' })
    expect(screen.getByRole('tooltip')).toHaveTextContent('rw: 11')
    fireEvent.keyDown(svg, { key: 'Escape' })
    expect(screen.queryByRole('tooltip')).toBeNull()
  })
  it('has a table view of the same numbers', () => {
    render(<XYChart title="T" x={X} series={series} yLabel="vol" />)
    fireEvent.click(screen.getByRole('button', { name: 'Show table' }))
    const rows = screen.getAllByRole('row').map((r) => Array.from(r.querySelectorAll('td,th')).map((c) => c.textContent))
    expect(rows).toEqual([['Date', 'actual', 'rw'], ['2024-01-01', '10', '11'], ['2024-01-02', '20', '19'], ['2024-01-03', '–', '21']])
    fireEvent.click(screen.getByRole('button', { name: 'Show chart' }))
    expect(screen.getByRole('img')).toBeInTheDocument()
  })
  it('on a log axis, equal ratios are equal heights', () => {
    // values 1, 10, 100: on a log scale the middle point sits exactly half-way between the others.
    const { container } = render(<XYChart title="T" x={X} series={[{ id: 'e', label: 'e', values: [1, 10, 100], slot: 2 }]} yLabel="g" log />)
    const ys = container.querySelector('polyline')!.getAttribute('points')!.split(' ').map((p) => Number(p.split(',')[1]))
    expect(ys[1] - ys[0]).toBeCloseTo(ys[2] - ys[1], 9)
    expect(screen.getAllByText('10').length).toBeGreaterThan(0) // a 1-2-5 tick
  })
  it('extends a linear axis to the round ticks around the data, and uses the short name for direct labels', () => {
    // 14..36: raw step (36 - 14) / 4 = 5.5 -> 10, so the inner ticks are 20 and 30 and the axis runs 10 to 40.
    const { container } = render(<XYChart title="T" x={X} series={[{ id: 'e', label: 'a long legend name', short: 'short', values: [14, 25, 36], slot: 1 }]} yLabel="g" />)
    expect(Array.from(container.querySelectorAll('.viz-tick')).map((t) => t.textContent).slice(0, 4)).toEqual(['10', '20', '30', '40'])
    expect(container.querySelector('.viz-direct-label')).toHaveTextContent('short')
  })
  it('says so when there is nothing to draw', () => {
    render(<XYChart title="T" x={X} series={[{ id: 'e', label: 'e', values: [null, null, null], slot: 1 }]} yLabel="g" />)
    expect(screen.getByText('No data to draw.')).toBeInTheDocument()
  })
})

describe('BarChart', () => {
  it('draws one bar per finite value from the zero baseline, with the band', () => {
    render(<BarChart title="ACF of |r|" labels={['1', '2', '3']} values={[0.2, -0.05, null]} band={0.02} slot={1} xLabel="lag" />)
    const bars = screen.getAllByTestId('viz-bar')
    expect(bars).toHaveLength(2)
    expect(bars[0]).toHaveAttribute('fill', 'var(--series-1)')
    // The positive bar ends above the baseline, the negative one below: their y and heights differ in sign.
    const y0 = Number(bars[0].getAttribute('y')) + Number(bars[0].getAttribute('height'))
    expect(Number(bars[1].getAttribute('y'))).toBeCloseTo(y0, 6) // the negative bar starts at the baseline
    expect(screen.getByTestId('viz-band').querySelectorAll('line')).toHaveLength(2) // +band and -band
    expect(screen.getByRole('figure', { name: 'ACF of |r|' })).toHaveTextContent('dashed: ±0.02 (95% white-noise band)')
  })
  it('shows a tooltip on hover and on focus, and a table view', () => {
    render(<BarChart title="ACF" labels={['1', '2']} values={[0.2, 0.1]} slot={1} xLabel="lag" />)
    const [first, second] = screen.getAllByTestId('viz-bar')
    fireEvent.mouseEnter(first)
    expect(screen.getByRole('tooltip')).toHaveTextContent('lag 1: 0.2')
    fireEvent.mouseLeave(first)
    fireEvent.focus(second)
    expect(screen.getByRole('tooltip')).toHaveTextContent('lag 2: 0.1')
    fireEvent.blur(second)
    expect(screen.queryByRole('tooltip')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Table' }))
    expect(screen.getAllByRole('row').map((r) => r.textContent)).toEqual(['lagValue', '10.2', '20.1'])
  })
  it('draws small multiples on a shared domain when one is given', () => {
    const { container } = render(<BarChart title="ACF" labels={['1']} values={[0.1]} slot={1} xLabel="lag" domain={[-1, 1]} />)
    const ticks = Array.from(container.querySelectorAll('.viz-tick')).map((t) => t.textContent)
    expect(ticks).toContain('-1')
    expect(ticks).toContain('1')
  })
  it('says so when every value is missing', () => {
    render(<BarChart title="ACF" labels={['1']} values={[null]} slot={1} xLabel="lag" />)
    expect(screen.getByText('No data to draw.')).toBeInTheDocument()
  })
})
