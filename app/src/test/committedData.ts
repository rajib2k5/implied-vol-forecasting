/**
 * The committed synthetic result files (written by tools/export_app_data.py --mode synthetic), and a
 * fetch stand-in that serves them the way the built app is served: `./data/synthetic/<name>.json`.
 * Page tests draw from these files and take their expected values from them, not from the UI.
 */
import { vi } from 'vitest'
import eda from '../../public/data/synthetic/eda.json'
import equityCurves from '../../public/data/synthetic/equity_curves.json'
import evaluation from '../../public/data/synthetic/evaluation.json'
import modelRace from '../../public/data/synthetic/model_race.json'

export const COMMITTED: Record<string, unknown> = { eda, equity_curves: equityCurves, evaluation, model_race: modelRace }

/** Serve `files` by name; a name in `missing` answers 404. Returns the mock to inspect the URLs asked for. */
export function serve(files: Record<string, unknown> = COMMITTED, missing: string[] = []) {
  const fetchMock = vi.fn(async (url: string) => {
    const name = /data\/synthetic\/([a-z_]+)\.json$/.exec(url)?.[1] ?? ''
    if (missing.includes(name) || !(name in files)) return { ok: false, status: 404, json: async () => ({}) }
    return { ok: true, status: 200, json: async () => structuredClone(files[name]) }
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}
