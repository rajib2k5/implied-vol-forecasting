import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import fixture from '../data/term-structure.synthetic.json'
import { fmt } from '../lib/format'
import { buildRibbonGeometry, parseTermStructure, regimeCounts, regimeSummary } from '../lib/termStructure'
import TermStructurePage from './TermStructurePage'

// jsdom has no WebGL, so the real three.js canvas can never run here. The lazy module is replaced
// by a stand-in that reports what it was given; `behaviour.throws` simulates a renderer that dies.
const behaviour = vi.hoisted(() => ({ throws: false }))
vi.mock('./RibbonCanvas', () => ({
  default: ({ geometry }: { geometry: { vertexCount: number; triangleCount: number } }) => {
    if (behaviour.throws) throw new Error('WebGL context creation failed')
    return <div data-testid="ribbon-canvas" data-vertices={geometry.vertexCount} data-triangles={geometry.triangleCount} />
  },
}))

const data = parseTermStructure(fixture)

/** Make detectWebGL() see (or not see) a context on every canvas jsdom creates. */
function stubWebGL(available: boolean) {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation((() => (available ? {} : null)) as never)
}

beforeEach(() => {
  behaviour.throws = false
})
afterEach(() => {
  vi.restoreAllMocks()
})

describe('TermStructurePage with WebGL', () => {
  it('lazy-loads the canvas and hands it the full ribbon geometry', async () => {
    stubWebGL(true)
    render(<TermStructurePage />)
    const canvas = await screen.findByTestId('ribbon-canvas')
    // 60 dates x 5 tenors = 300 vertices; (60-1) x (5-1) cells x 2 triangles = 472 triangles.
    expect(canvas).toHaveAttribute('data-vertices', String(data.dates.length * data.tenors.length))
    expect(canvas).toHaveAttribute('data-triangles', String((data.dates.length - 1) * (data.tenors.length - 1) * 2))
    expect(screen.queryByRole('status')).toBeNull()
  })
  it('labels the data as synthetic and names the seed, never implying it is CBOE data', async () => {
    stubWebGL(true)
    render(<TermStructurePage />)
    await screen.findByTestId('ribbon-canvas')
    expect(screen.getByText('synthetic data')).toBeInTheDocument()
    expect(screen.getByText(new RegExp(`seed ${data.generator?.seed}`))).toBeInTheDocument()
    expect(screen.getByText(/not from CBOE data/)).toBeInTheDocument()
  })
  it('shows the regime counts and level range computed from the data', async () => {
    stubWebGL(true)
    render(<TermStructurePage />)
    await screen.findByTestId('ribbon-canvas')
    const counts = regimeCounts(data)
    const g = buildRibbonGeometry(data)
    const stat = (label: string) => screen.getByText(label).nextElementSibling as HTMLElement
    expect(stat('Contango dates').textContent).toBe(String(counts.contango))
    expect(stat('Backwardation dates').textContent).toBe(String(counts.backwardation))
    expect(stat('Level range (vol pts)').textContent).toBe(`${fmt(g.levelRange.min)} to ${fmt(g.levelRange.max)}`)
    expect(screen.getByRole('figure', { name: new RegExp(regimeSummary(data)) })).toBeInTheDocument()
  })
  it('degrades to the flat heatmap, with an explanation, if the renderer throws', async () => {
    stubWebGL(true)
    behaviour.throws = true
    vi.spyOn(console, 'error').mockImplementation(() => {}) // React logs the caught render error
    render(<TermStructurePage />)
    expect(await screen.findByRole('status')).toHaveTextContent('the 3D renderer failed to start')
    expect(screen.getByRole('img', { name: 'Heatmap' })).toBeInTheDocument()
    expect(screen.queryByTestId('ribbon-canvas')).toBeNull()
  })
})

describe('TermStructurePage without WebGL', () => {
  it('does not try to load the 3D code and shows the heatmap fallback instead', () => {
    stubWebGL(false)
    render(<TermStructurePage />)
    expect(screen.getByRole('status')).toHaveTextContent('this browser has no WebGL')
    expect(screen.queryByTestId('ribbon-canvas')).toBeNull()
    // One heat cell per date x tenor: the fallback shows every number the ribbon would.
    expect(screen.getAllByTestId('ivf-heat-cell')).toHaveLength(data.dates.length * data.tenors.length)
  })
  it('lays the heatmap out with tenors as rows and dates as columns, brightest where the level is highest', () => {
    stubWebGL(false)
    render(<TermStructurePage />)
    const cells = screen.getAllByTestId('ivf-heat-cell')
    // Cells are drawn row by row, so cell number = tenorIndex * nDates + dateIndex.
    expect(new Set(cells.map((c) => c.getAttribute('y'))).size).toBe(data.tenors.length)
    expect(new Set(cells.map((c) => c.getAttribute('x'))).size).toBe(data.dates.length)
    let best = { t: 0, d: 0, v: -Infinity }
    data.levels.forEach((row, d) => row.forEach((v, t) => (v > best.v ? (best = { t, d, v }) : null)))
    // Opacity = 0.15 + 0.85 * normalised level, so the single highest level is fully opaque (1.0).
    expect(Number(cells[best.t * data.dates.length + best.d].getAttribute('opacity'))).toBeCloseTo(1, 10)
  })
  it('still shows the synthetic label and the legend', () => {
    stubWebGL(false)
    render(<TermStructurePage />)
    expect(screen.getByText('synthetic data')).toBeInTheDocument()
    expect(screen.getByText(/contango: long end above short end/)).toBeInTheDocument()
    expect(screen.getByText(/backwardation: curve inverted/)).toBeInTheDocument()
  })
})
