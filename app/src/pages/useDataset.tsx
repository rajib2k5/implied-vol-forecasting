/* eslint-disable react-refresh/only-export-components -- the loading hook and its two small render helpers live together on purpose */
/**
 * Fetches one result file of the app-data contract and parses it (ADR 0001 D1: Python computes,
 * versioned JSON carries, the app renders). A file that is missing or does not parse leaves the
 * view greyed out with the reason, never a crash (data-model.md 5.1).
 *
 * Under the current data policy the app ships and reads synthetic data only: `DATA_ROOT` is
 * `data/synthetic/`. Real-data JSON is written outside the repository
 * (`tools/export_app_data.py --mode real`); serving it to a local dev server is the documented hook,
 * not wired while the owner's policy change is pending.
 */
import { useEffect, useState, type ReactNode } from 'react'
import type { DataMode, Envelope } from '../lib/results'

export const DATA_ROOT = './data/synthetic/'

export type Loaded<T> = { state: 'loading' } | { state: 'error'; message: string } | { state: 'ready'; value: Envelope<T> }

export function useDataset<T>(name: string, parse: (raw: unknown) => Envelope<T>): Loaded<T> {
  const [loaded, setLoaded] = useState<Loaded<T>>({ state: 'loading' })
  useEffect(() => {
    let live = true
    fetch(`${DATA_ROOT}${name}.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`${name}.json: HTTP ${r.status}`)
        return r.json()
      })
      .then((raw) => {
        if (live) setLoaded({ state: 'ready', value: parse(raw) })
      })
      .catch((e: unknown) => {
        if (live) setLoaded({ state: 'error', message: e instanceof Error ? e.message : String(e) })
      })
    return () => {
      live = false
    }
  }, [name, parse])
  return loaded
}

/** Loading and error states, then the view. */
export function Gate<T>({ loaded, children }: { loaded: Loaded<T>; children: (value: Envelope<T>) => ReactNode }) {
  if (loaded.state === 'loading') return <p className="sub">Loading...</p>
  if (loaded.state === 'error')
    return (
      <p role="alert" className="sub">
        This view is unavailable: {loaded.message}
      </p>
    )
  return <>{children(loaded.value)}</>
}

/** A synthetic file always carries the badge (app/CLAUDE.md: synthetic data is always labelled). */
export function ModeBadge({ mode }: { mode: DataMode }) {
  return mode === 'synthetic' ? <span className="badge synthetic">synthetic data</span> : <span className="badge">real data (local only)</span>
}
