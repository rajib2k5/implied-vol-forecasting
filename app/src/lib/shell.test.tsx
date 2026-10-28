import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { AppShell } from './shell'
import { resetMemoryFallback, safeStorage } from './storage'

const PAGES = [
  { id: 'dashboard', label: 'Dashboard', render: () => <p>dash-content</p> },
  { id: 'analysis', label: 'Analysis', render: () => <p>analysis-content</p> },
]

const STORE = safeStorage()

function shell(storage = STORE) {
  return (
    <AppShell
      slug="test-app"
      title="Test App"
      subtitle="A test"
      pages={PAGES}
      storage={storage}
    />
  )
}

beforeEach(() => {
  resetMemoryFallback()
  STORE.removeItem('ivf:test-app:theme')
  window.location.hash = ''
  document.documentElement.removeAttribute('data-theme')
})

describe('AppShell navigation', () => {
  it('renders the nav with exactly the app pages, dashboard active', () => {
    render(shell())
    const nav = screen.getByRole('navigation', { name: 'Primary' })
    expect(Array.from(nav.querySelectorAll('a')).map((a) => a.textContent)).toEqual(['Dashboard', 'Analysis'])
    expect(screen.getByText('dash-content')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Dashboard' })).toHaveAttribute('aria-current', 'page')
  })
  it('switches pages on nav click and updates the hash', () => {
    render(shell())
    fireEvent.click(screen.getByRole('link', { name: 'Analysis' }))
    expect(screen.getByText('analysis-content')).toBeInTheDocument()
    expect(screen.queryByText('dash-content')).not.toBeInTheDocument()
    expect(window.location.hash).toBe('#/analysis')
  })
})

describe('AppShell routing edge cases', () => {
  it('falls back to the first page for an unknown route', () => {
    window.location.hash = '#/no-such-page'
    render(shell())
    expect(screen.getByText('dash-content')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Dashboard' })).toHaveAttribute('aria-current', 'page')
  })
  it('reacts to the browser changing the hash (back / forward)', () => {
    render(shell())
    act(() => {
      window.location.hash = '#/analysis'
      window.dispatchEvent(new HashChangeEvent('hashchange'))
    })
    expect(screen.getByText('analysis-content')).toBeInTheDocument()
  })
  it('renders the header with an empty nav and an empty main area when the app has no pages', () => {
    render(<AppShell slug="test-app" title="Empty" pages={[]} storage={STORE} />)
    expect(screen.getByRole('heading', { level: 1, name: 'Empty' })).toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Primary' }).querySelectorAll('a')).toHaveLength(0)
    expect(screen.getByRole('main')).toBeEmptyDOMElement()
  })
})

describe('Theme toggle', () => {
  it('renders top-right, toggles corporate/dark on the document element, and persists', () => {
    render(shell())
    const btn = screen.getByRole('button', { name: 'Switch to corporate theme' })
    fireEvent.click(btn)
    expect(document.documentElement.getAttribute('data-theme')).toBe('corporate')
    expect(screen.getByRole('button', { name: 'Switch to dark theme' })).toBeInTheDocument()
    expect(STORE.getItem('ivf:test-app:theme')).toBe('corporate')
    fireEvent.click(screen.getByRole('button', { name: 'Switch to dark theme' }))
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark')
  })
  it('honors a stored theme preference on mount', () => {
    STORE.setItem('ivf:test-app:theme', 'corporate')
    render(shell())
    expect(document.documentElement.getAttribute('data-theme')).toBe('corporate')
    expect(screen.getByRole('button', { name: 'Switch to dark theme' })).toBeInTheDocument()
  })
})
