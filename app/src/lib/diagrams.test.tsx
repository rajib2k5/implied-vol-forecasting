import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
import { ArchDiagram, SequenceDiagram, ActivityDiagram } from './diagrams'
import { DiagramsPage } from './diagramsPage'

afterEach(() => {
  vi.useRealTimers()
})

describe('ArchDiagram', () => {
  it('renders every layer label and box, with flowing connector arrows between layers', () => {
    const { container } = render(
      <ArchDiagram
        layers={[
          { label: 'Data', boxes: ['synthetic.json', 'env key'] },
          { label: 'Core', boxes: ['engine'] },
          { label: 'UI', boxes: ['Dashboard', 'Admin', 'Metrics'] },
        ]}
      />,
    )
    for (const t of ['DATA', 'CORE', 'UI', 'synthetic.json', 'engine', 'Metrics']) expect(screen.getByText(t)).toBeInTheDocument()
    expect(container.querySelectorAll('rect')).toHaveLength(6)
    // continuous dashed-flow animation class on both inter-layer connectors
    expect(container.querySelectorAll('line.ivf-arch-flow')).toHaveLength(2)
  })
})

describe('SequenceDiagram (looping stepper)', () => {
  const actors = ['User', 'Engine', 'Store']
  const steps = [
    { from: 0, to: 1, label: 'run()' },
    { from: 1, to: 1, label: 'validate params' },
    { from: 1, to: 2, label: 'read as-of' },
    { from: 2, to: 1, label: 'rows' },
  ]
  it('starts with step 0 running, others pending, and a traveling dot on the active arrow', () => {
    vi.useFakeTimers()
    const { container } = render(<SequenceDiagram actors={actors} steps={steps} />)
    expect(container.querySelectorAll('.ivf-seq-line.run')).toHaveLength(1)
    expect(container.querySelectorAll('.ivf-seq-line.pending')).toHaveLength(2) // self-step is a note
    expect(screen.getByTestId('ivf-seq-dot')).toBeInTheDocument()
    expect(screen.getByText('↺ validate params')).toBeInTheDocument()
  })
  it('advances on the timer, completes all arrows, then loops back to the start', () => {
    vi.useFakeTimers()
    const { container } = render(<SequenceDiagram actors={actors} steps={steps} />)
    act(() => {
      vi.advanceTimersByTime(850 * 3) // phases 1,2,3 -> last arrow running
    })
    expect(container.querySelectorAll('.ivf-seq-line.done')).toHaveLength(2)
    expect(container.querySelectorAll('.ivf-seq-line.run')).toHaveLength(1)
    act(() => {
      vi.advanceTimersByTime(850) // phase 4: hold — everything drawn
    })
    expect(container.querySelectorAll('.ivf-seq-line.done')).toHaveLength(3)
    expect(container.querySelector('.ivf-seq-line.run')).toBeNull()
    act(() => {
      vi.advanceTimersByTime(850 * 3) // remaining holds -> wraps to phase 0
    })
    expect(container.querySelectorAll('.ivf-seq-line.pending')).toHaveLength(2)
    expect(container.querySelectorAll('.ivf-seq-line.run')).toHaveLength(1)
  })
  it('replay resets to the first step and fires the callback', () => {
    vi.useFakeTimers()
    const onReplay = vi.fn()
    const { container } = render(<SequenceDiagram actors={actors} steps={steps} onReplay={onReplay} />)
    act(() => {
      vi.advanceTimersByTime(850 * 4)
    })
    expect(container.querySelector('.ivf-seq-line.run')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Replay sequence' }))
    expect(onReplay).toHaveBeenCalledTimes(1)
    expect(container.querySelectorAll('.ivf-seq-line.run')).toHaveLength(1)
    expect(container.querySelectorAll('.ivf-seq-line.done')).toHaveLength(0)
  })
})

describe('ActivityDiagram (walking token)', () => {
  const nodes = [
    { id: 's', label: 'load', type: 'start' as const },
    { id: 'k', label: 'API key?', type: 'decision' as const },
    { id: 'live', label: 'fetch live' },
    { id: 'syn', label: 'read synthetic' },
    { id: 'e', label: 'render', type: 'end' as const },
  ]
  const edges = [
    { from: 's', to: 'k' },
    { from: 'k', to: 'live', label: 'yes' },
    { from: 'k', to: 'syn', label: 'no' },
    { from: 'live', to: 'syn' },
    { from: 'syn', to: 'e' },
  ]
  it('renders shapes, decision diamond, side-routed edges, and ignores unknown edge ids', () => {
    vi.useFakeTimers()
    const { container } = render(<ActivityDiagram nodes={nodes} edges={[...edges, { from: 'ghost', to: 'e' }]} />)
    expect(screen.getByText('API key?')).toBeInTheDocument()
    expect(container.querySelectorAll('[data-testid="ivf-act-decision"]')).toHaveLength(1)
    expect(container.querySelectorAll('[data-testid="ivf-act-side"]').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('yes')).toBeInTheDocument()
  })
  it('walks the token down the flow and loops', () => {
    vi.useFakeTimers()
    render(<ActivityDiagram nodes={nodes} edges={edges} />)
    const at = () => screen.getByTestId('ivf-act-token').style.transform
    expect(at()).toBe('translateY(0px)')
    act(() => {
      vi.advanceTimersByTime(800)
    })
    expect(at()).toBe('translateY(74px)')
    act(() => {
      vi.advanceTimersByTime(800 * 3) // reach last node
    })
    expect(at()).toBe(`translateY(${74 * 4}px)`)
    act(() => {
      vi.advanceTimersByTime(800) // hold phase: token hidden
    })
    expect(screen.queryByTestId('ivf-act-token')).toBeNull()
    act(() => {
      vi.advanceTimersByTime(800) // loops back to node 0
    })
    expect(at()).toBe('translateY(0px)')
  })
})

describe('DiagramsPage', () => {
  it('renders intro and all three diagrams', () => {
    render(
      <DiagramsPage
        content={{
          intro: ['What this system does.'],
          architecture: { layers: [{ label: 'Core', boxes: ['engine'] }] },
          sequence: { actors: ['A', 'B'], steps: [{ from: 0, to: 1, label: 'msg' }] },
          activity: {
            nodes: [
              { id: 's', label: 'go', type: 'start' },
              { id: 'e', label: 'done', type: 'end' },
            ],
            edges: [{ from: 's', to: 'e' }],
          },
        }}
      />,
    )
    expect(screen.getByText('What this system does.')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Architecture diagram' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Sequence diagram' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Activity diagram' })).toBeInTheDocument()
  })
})
