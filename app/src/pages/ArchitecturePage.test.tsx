import { describe, expect, it } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { PIPELINE_EDGES, PIPELINE_LAYERS, PIPELINE_NODES, PROCESS_NODES, FOLD_ACTORS, describeEdges } from '../lib/architecture'
import ArchitecturePage from './ArchitecturePage'

describe('ArchitecturePage', () => {
  it('draws the three diagrams', () => {
    render(<ArchitecturePage />)
    expect(screen.getByRole('img', { name: 'Architecture diagram' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Sequence diagram' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Activity diagram' })).toBeInTheDocument()
  })
  it('explains the publication set P that the Outputs box refers to', () => {
    render(<ArchitecturePage />)
    expect(screen.getByText(/closed publication set P/)).toBeInTheDocument()
    expect(screen.getByText(/ratio to the random walk/)).toBeInTheDocument()
  })
  it('shows every pipeline layer and every box from the architecture data', () => {
    render(<ArchitecturePage />)
    const svg = screen.getByRole('img', { name: 'Architecture diagram' })
    const texts = Array.from(svg.querySelectorAll('text')).map((t) => t.textContent)
    // ArchDiagram upper-cases layer names; box labels are drawn as given.
    for (const layer of PIPELINE_LAYERS) expect(texts).toContain(layer.toUpperCase())
    for (const n of PIPELINE_NODES) expect(texts).toContain(n.label)
  })
  it('lists every data dependency as text, as an alternative to the diagram', () => {
    render(<ArchitecturePage />)
    const items = within(screen.getByLabelText('Pipeline edges', { selector: 'ul' })).getAllByRole('listitem')
    expect(items.map((li) => li.textContent)).toEqual(describeEdges(PIPELINE_NODES, PIPELINE_EDGES))
    expect(items).toHaveLength(PIPELINE_EDGES.length)
  })
  it('draws the fold sequence between the named actors and the process with one decision', () => {
    render(<ArchitecturePage />)
    const seq = screen.getByRole('img', { name: 'Sequence diagram' })
    const seqTexts = Array.from(seq.querySelectorAll('text')).map((t) => t.textContent)
    for (const actor of FOLD_ACTORS) expect(seqTexts).toContain(actor)
    expect(screen.getAllByTestId('ivf-act-decision')).toHaveLength(PROCESS_NODES.filter((n) => n.type === 'decision').length)
  })
})
