import { cleanup, render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { DiagramsPage, type DiagramsContent } from './diagramsPage'

const base: DiagramsContent = {
  intro: ['An app.'],
  architecture: { layers: [] },
  sequence: { actors: [], steps: [] },
  activity: { nodes: [], edges: [] },
}

describe('DiagramsPage use cases', () => {
  it('renders every use case with who, how and the page that delivers it', () => {
    const useCases = [
      { title: 'Size a trade', who: 'A PM', how: 'Set the risk budget and read the size.', page: 'Sizing' },
      { title: 'Audit a model', who: 'Model risk', how: 'Replay the seed and compare.', page: 'Registry' },
    ]
    render(<DiagramsPage content={{ ...base, useCases }} />)
    const items = within(screen.getByRole('list', { name: 'Use cases' })).getAllByRole('listitem')
    expect(items).toHaveLength(2)
    expect(items[1].textContent).toContain('Audit a model')
    expect(items[1].textContent).toContain('Model risk')
    expect(items[1].textContent).toContain('Registry')
  })

  it('omits the section entirely when there are no use cases', () => {
    render(<DiagramsPage content={base} />)
    expect(screen.queryByRole('heading', { name: 'Use cases' })).toBeNull()
    expect(screen.queryByRole('list', { name: 'Use cases' })).toBeNull()
  })
})

describe('DiagramsPage headings', () => {
  it('uses generic defaults when no headings are given', () => {
    render(<DiagramsPage content={base} />)
    expect(screen.getByRole('heading', { name: 'How this system works' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Architecture' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Sequence — primary flow' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Activity — decision flow' })).toBeInTheDocument()
  })
  it('lets each heading be overridden independently', () => {
    const headings = { title: 'My title', architecture: 'My pipeline', sequence: 'My sequence', activity: 'My process' }
    render(<DiagramsPage content={{ ...base, headings }} />)
    for (const name of Object.values(headings)) expect(screen.getByRole('heading', { name })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Architecture' })).toBeNull()
    cleanup()
    render(<DiagramsPage content={{ ...base, headings: { sequence: 'Only this one' } }} />)
    expect(screen.getByRole('heading', { name: 'Only this one' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Architecture' })).toBeInTheDocument()
  })
})
