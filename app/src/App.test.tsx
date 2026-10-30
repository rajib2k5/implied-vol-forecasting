import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import App from './App'
import { APP_SLUG, MILESTONES, PROJECT_PITCH, PROJECT_TITLE } from './lib/project'
import { resetMemoryFallback, safeStorage } from './lib/storage'

const STORE = safeStorage()

beforeEach(() => {
  resetMemoryFallback()
  STORE.removeItem(`ivf:${APP_SLUG}:theme`)
  window.location.hash = ''
  // jsdom has no WebGL; say so quietly (its default getContext logs a "not implemented" error).
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation((() => null) as never)
})
afterEach(() => {
  vi.restoreAllMocks()
})

// Integration: shell + pages + pure lib modules, end to end, through the nav links.
describe('App (integration)', () => {
  it('opens on the About page with the project title and pitch in the header', () => {
    render(<App />)
    expect(screen.getByRole('heading', { level: 1, name: PROJECT_TITLE })).toBeInTheDocument()
    expect(screen.getByText(PROJECT_PITCH)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'About' })).toHaveAttribute('aria-current', 'page')
    expect(screen.getByRole('list', { name: 'Milestones' }).querySelectorAll('li')).toHaveLength(MILESTONES.length)
  })
  it('offers About, Architecture, the EDA, the 3D page and the three result views in the primary nav, with no Metrics or Admin page', () => {
    render(<App />)
    const nav = screen.getByRole('navigation', { name: 'Primary' })
    const labels = Array.from(nav.querySelectorAll('a')).map((a) => a.textContent)
    expect(labels).toEqual(['About', 'Architecture', 'EDA', 'Term structure 3D', 'Model race', 'Evaluation', 'Backtest'])
  })
  it('navigates to Architecture and shows the diagrams', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('link', { name: 'Architecture' }))
    expect(window.location.hash).toBe('#/architecture')
    expect(screen.getByRole('img', { name: 'Architecture diagram' })).toBeInTheDocument()
  })
  it('navigates to the 3D page and, without WebGL, falls back to the labelled 2D view', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('link', { name: 'Term structure 3D' }))
    expect(window.location.hash).toBe('#/term-structure')
    expect(screen.getByText('synthetic data')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('no WebGL')
  })
})
