import { describe, expect, it } from 'vitest'
import pkg from '../../package.json'
import {
  APP_SLUG,
  DATA_SOURCES,
  DISCLAIMER,
  MILESTONES,
  PROJECT_PITCH,
  PROJECT_TITLE,
  REQUIREMENT,
  TEACHING_POINTS,
  validateDataSources,
  validateMilestones,
  type DataSource,
  type Milestone,
} from './project'

const ms = (statuses: Milestone['status'][]): Milestone[] =>
  statuses.map((status, id) => ({ id, title: `T${id}`, summary: `S${id}`, status }))

describe('project facts', () => {
  it('uses the package name as the storage slug, so the theme key matches the project', () => {
    expect(APP_SLUG).toBe(pkg.name)
  })
  it('has a title, a pitch and the original requirement', () => {
    expect(PROJECT_TITLE.length).toBeGreaterThan(0)
    expect(PROJECT_PITCH.length).toBeGreaterThan(0)
    expect(REQUIREMENT).toContain('ARIMA')
    expect(REQUIREMENT).toContain('GARCH')
    expect(REQUIREMENT).toContain('LSTM')
    expect(REQUIREMENT).toContain('RMSE')
    expect(REQUIREMENT).toContain('backtest')
  })
  it('says plainly that this is not financial advice', () => {
    expect(DISCLAIMER).toMatch(/not financial advice/i)
  })
})

describe('teaching points', () => {
  it('covers the points the project exists to teach, each with a non-empty explanation', () => {
    const ids = TEACHING_POINTS.map((t) => t.id)
    for (const id of ['har', 'garch-category', 'qlike', 'dm-mz', 'splits', 'holdout']) expect(ids).toContain(id)
    expect(new Set(ids).size).toBe(ids.length)
    for (const t of TEACHING_POINTS) {
      expect(t.title.trim().length, t.id).toBeGreaterThan(0)
      expect(t.body.trim().length, t.id).toBeGreaterThan(0)
    }
  })
})

describe('milestone roadmap', () => {
  it('is valid and runs 0..6', () => {
    expect(validateMilestones(MILESTONES)).toEqual([])
    expect(MILESTONES.map((m) => m.id)).toEqual([0, 1, 2, 3, 4, 5, 6])
  })
  it('accepts a roadmap with some milestones done', () => {
    expect(validateMilestones(ms(['done', 'done', 'current', 'planned']))).toEqual([])
  })
  it('flags ids that are not 0, 1, 2, ...', () => {
    const bad = ms(['current', 'planned', 'planned'])
    bad[2].id = 5
    expect(validateMilestones(bad).join(' | ')).toContain('milestone at position 2 has id 5')
  })
  it('flags zero or several current milestones', () => {
    expect(validateMilestones(ms(['planned', 'planned'])).join(' | ')).toContain('exactly one milestone must be current, found 0')
    expect(validateMilestones(ms(['current', 'current'])).join(' | ')).toContain('exactly one milestone must be current, found 2')
  })
  it('flags a done milestone after the current one and a planned one before it', () => {
    expect(validateMilestones(ms(['current', 'done'])).join(' | ')).toContain('milestone 1 comes after the current one and must be planned')
    expect(validateMilestones(ms(['planned', 'current'])).join(' | ')).toContain('milestone 0 comes before the current one and must be done')
  })
  it('flags an empty title or summary', () => {
    const bad = ms(['current', 'planned'])
    bad[1].title = '  '
    expect(validateMilestones(bad).join(' | ')).toContain('milestone 1 needs a title and a summary')
    const bad2 = ms(['current', 'planned'])
    bad2[0].summary = ''
    expect(validateMilestones(bad2).join(' | ')).toContain('milestone 0 needs a title and a summary')
  })
})

describe('data sources', () => {
  it('are valid and include the three vendors plus the synthetic fixture', () => {
    expect(validateDataSources(DATA_SOURCES)).toEqual([])
    expect(DATA_SOURCES.map((s) => s.id).sort()).toEqual(['cboe', 'fred', 'synthetic', 'yfinance'])
  })
  it('never commit third-party data, and ship only the synthetic fixture', () => {
    expect(DATA_SOURCES.filter((s) => s.committedToRepo).map((s) => s.id)).toEqual(['synthetic'])
  })
  it('state the CBOE restriction', () => {
    const cboe = DATA_SOURCES.find((s) => s.id === 'cboe') as DataSource
    expect(cboe.terms).toMatch(/non-commercial/)
    expect(cboe.terms).toMatch(/not redistributed/)
  })
  describe('flag policy violations', () => {
    const ok: DataSource[] = [
      { id: 'cboe', name: 'n', provides: 'p', terms: 't', committedToRepo: false },
      { id: 'synthetic', name: 'n', provides: 'p', terms: 't', committedToRepo: true },
    ]
    it('control: the minimal list is valid', () => {
      expect(validateDataSources(ok)).toEqual([])
    })
    it('flags committed vendor data', () => {
      const bad = [{ ...ok[0], committedToRepo: true }, ok[1]]
      expect(validateDataSources(bad).join(' | ')).toContain('cboe: third-party data must not be committed')
    })
    it('flags a synthetic fixture that is not shipped', () => {
      expect(validateDataSources([ok[0], { ...ok[1], committedToRepo: false }]).join(' | ')).toContain('expected to ship')
    })
    it('flags duplicate ids and missing terms', () => {
      expect(validateDataSources([...ok, ok[0]]).join(' | ')).toContain('duplicate data source id: cboe')
      expect(validateDataSources([{ ...ok[0], terms: ' ' }, ok[1]]).join(' | ')).toContain('cboe: terms must be stated')
    })
  })
})
