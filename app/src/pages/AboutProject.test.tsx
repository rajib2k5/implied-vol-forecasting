import { describe, expect, it } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { DATA_SOURCES, DISCLAIMER, MILESTONES, PROJECT_INTRO, REQUIREMENT, TEACHING_POINTS } from '../lib/project'
import AboutProject from './AboutProject'

describe('AboutProject', () => {
  it('states the project and the original requirement', () => {
    render(<AboutProject />)
    expect(screen.getByText(PROJECT_INTRO)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'The original requirement' })).toBeInTheDocument()
    expect(screen.getByText(REQUIREMENT)).toBeInTheDocument()
  })
  it('lists every teaching point with its explanation', () => {
    render(<AboutProject />)
    const items = within(screen.getByRole('list', { name: 'Teaching points' })).getAllByRole('listitem')
    expect(items).toHaveLength(TEACHING_POINTS.length)
    for (const t of TEACHING_POINTS) expect(screen.getByRole('heading', { name: t.title })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /GARCH belongs on returns/ })).toBeInTheDocument()
  })
  it('shows the whole roadmap and marks exactly the milestone the data says is current', () => {
    render(<AboutProject />)
    const items = within(screen.getByRole('list', { name: 'Milestones' })).getAllByRole('listitem')
    expect(items).toHaveLength(MILESTONES.length)
    const current = MILESTONES.filter((m) => m.status === 'current')
    expect(current).toHaveLength(1)
    const marked = items.filter((li) => li.getAttribute('aria-current') === 'step')
    expect(marked).toHaveLength(1)
    expect(marked[0]).toHaveTextContent(`M${current[0].id}`)
    expect(marked[0]).toHaveTextContent(current[0].title)
    expect(within(marked[0]).getByText('current')).toBeInTheDocument()
    // Planned milestones carry no status badge.
    const planned = items.filter((li) => li.className.includes('planned'))
    expect(planned.length).toBe(MILESTONES.filter((m) => m.status === 'planned').length)
    for (const li of planned) expect(within(li).queryByText('current')).toBeNull()
  })
  it('explains the data sources and that CBOE data is not redistributed', () => {
    render(<AboutProject />)
    const items = within(screen.getByRole('list', { name: 'Data sources' })).getAllByRole('listitem')
    expect(items).toHaveLength(DATA_SOURCES.length)
    expect(screen.getAllByText(/CBOE data is not redistributed/).length).toBeGreaterThan(0)
    expect(screen.getByText(/synthetic fixture/i, { selector: 'p' })).toBeInTheDocument()
  })
  it('carries the not-financial-advice disclaimer under its own heading', () => {
    render(<AboutProject />)
    expect(screen.getByRole('heading', { name: 'Not financial advice' })).toBeInTheDocument()
    expect(screen.getByText(DISCLAIMER)).toBeInTheDocument()
  })
  it('marks a done milestone with a done badge once the roadmap has progressed', () => {
    const progressed = MILESTONES.map((m, i) => ({ ...m, status: i === 0 ? ('done' as const) : i === 1 ? ('current' as const) : ('planned' as const) }))
    render(<AboutProject milestones={progressed} />)
    const items = within(screen.getByRole('list', { name: 'Milestones' })).getAllByRole('listitem')
    expect(within(items[0]).getByText('done')).toBeInTheDocument()
    expect(items[0]).not.toHaveAttribute('aria-current')
    expect(items[1]).toHaveAttribute('aria-current', 'step')
  })
})
