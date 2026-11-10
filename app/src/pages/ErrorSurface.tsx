import { lazy, Suspense, useMemo, useState } from 'react'
import { strokeFor } from '../lib/chartGeom'
import { lowestIndex, surfaceGrid, surfaceLayout, type Measure, type SurfaceGrid } from '../lib/errorSurface'
import { DEFAULT_MODELS, colorSlot, show, slotOrder, type Evaluation } from '../lib/results'
import { detectWebGL } from '../lib/webgl'
import { RenderBoundary } from './RenderBoundary'

// Lazy, from the one three.js module: the 3D chunk is fetched only when this view draws in 3D.
const ErrorSurfaceCanvas = lazy(() => import('./RibbonCanvas').then((m) => ({ default: m.ErrorSurfaceCanvas })))

const TARGETS: [string, string][] = [
  ['vix_level', 'VIX close'],
  ['rv21', '21-session realised volatility'],
]
const MEASURES: [Measure, string][] = [
  ['rmse', 'RMSE / RW'],
  ['qlike', 'QLIKE / RW'],
]

/** The same numbers as the 3D bars: per horizon, one row per model, one column per regime. */
function SurfaceTable({ grid, measureLabel }: { grid: SurfaceGrid; measureLabel: string }) {
  return (
    <div className="scroll-x">
      <table aria-label="Loss ratio to the random walk by regime">
        <caption className="sub">
          {measureLabel} by horizon, model and regime (regime at the forecast origin). Bold: the lowest in its column within the horizon. A dash is a
          suppressed cell (too few origins).
        </caption>
        <thead>
          <tr>
            <th scope="col">h</th>
            <th scope="col">Model</th>
            <th scope="col">whole period</th>
            {grid.regimes.map((g) => (
              <th key={`${g.scheme}-${g.regime}`} scope="col">{g.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {grid.horizons.flatMap((h, hi) => {
            const bestWhole = lowestIndex(grid.models.map((_, mi) => grid.whole[mi][hi]))
            const best = grid.regimes.map((_, ri) => lowestIndex(grid.models.map((_, mi) => grid.cells[mi][hi][ri])))
            return grid.models.map((m, mi) => (
              <tr key={`${h}-${m}`}>
                <td className="num">{h}</td>
                <th scope="row">{m}</th>
                <td className={`num${mi === bestWhole ? ' best' : ''}`}>{show(grid.whole[mi][hi])}</td>
                {grid.regimes.map((g, ri) => (
                  <td key={`${g.scheme}-${g.regime}`} className={`num${mi === best[ri] ? ' best' : ''}`}>{show(grid.cells[mi][hi][ri])}</td>
                ))}
              </tr>
            ))
          })}
        </tbody>
      </table>
    </div>
  )
}

/**
 * The 3D error surface of the Evaluation view: model x horizon x regime, bar length = the loss ratio
 * to the random walk on a log scale about the plane at 1. A target and a loss toggle, a model picker
 * (colour follows the model: the defaults take palette slots 1 to 3, as in the model race), a table
 * view, and the table again whenever WebGL is missing or the renderer fails.
 */
export function ErrorSurface({ ev }: { ev: Evaluation }) {
  const [targetId, setTargetId] = useState('vix_level')
  const [measure, setMeasure] = useState<Measure>('rmse')
  const [chosen, setChosen] = useState<string[]>(DEFAULT_MODELS)
  const [table, setTable] = useState(false)
  const [hover, setHover] = useState<number | null>(null)
  const [webgl] = useState(() => detectWebGL())
  const grid = useMemo(() => surfaceGrid(ev, targetId, measure), [ev, targetId, measure])
  const registry = useMemo(() => slotOrder(grid.models, DEFAULT_MODELS), [grid])
  const slotOf = (id: string) => colorSlot(registry, id)
  const layout = useMemo(() => surfaceLayout(grid, chosen, (id) => colorSlot(registry, id)), [grid, chosen, registry])
  const measureLabel = MEASURES.find(([id]) => id === measure)![1]

  if (ev.regimeLosses.length === 0) {
    return <p className="sub">This file has no losses by regime (a file from before schema 1.2.0, or a real-data export, which keeps regime cells on the local machine).</p>
  }
  const pick = (f: () => void) => () => {
    setHover(null)
    f()
  }
  const toggle = (id: string) => pick(() => setChosen((c) => (c.includes(id) ? c.filter((m) => m !== id) : [...c, id])))
  const bar = hover === null ? undefined : layout.bars[hover]
  const shown = registry.filter((m) => chosen.includes(m))
  const fallback = (reason: string) => (
    <div className="card">
      <p role="status" className="sub">
        The 3D view is unavailable ({reason}). Here are the same numbers as a table.
      </p>
      <SurfaceTable grid={grid} measureLabel={measureLabel} />
    </div>
  )

  return (
    <>
      <div className="controls" role="group" aria-label="Error surface target and loss">
        {TARGETS.map(([id, label]) => (
          <button key={id} type="button" aria-pressed={id === targetId} className={id === targetId ? 'primary' : ''} onClick={pick(() => setTargetId(id))}>
            {label}
          </button>
        ))}
        {MEASURES.map(([id, label]) => (
          <button key={id} type="button" aria-pressed={id === measure} className={id === measure ? 'primary' : ''} onClick={pick(() => setMeasure(id))}>
            {label}
          </button>
        ))}
      </div>
      <fieldset className="controls model-picks">
        <legend className="sub">Models drawn (colour stays with the model)</legend>
        {registry.map((id) => (
          <label key={id}>
            <input type="checkbox" checked={chosen.includes(id)} onChange={toggle(id)} /> {id}
          </label>
        ))}
      </fieldset>
      {!webgl ? (
        fallback('this browser has no WebGL')
      ) : (
        <figure className="viz" aria-label={`Error surface: ${measureLabel} by model, horizon and regime`}>
          <div className="viz-head">
            <ul className="viz-legend" aria-label="Legend">
              {shown.map((m) => (
                <li key={m}>
                  <i className="swatch" style={{ background: strokeFor(slotOf(m)) }} aria-hidden="true" />
                  {m}
                </li>
              ))}
            </ul>
            <button type="button" className="viz-toggle" aria-pressed={table} onClick={() => setTable((v) => !v)}>
              {table ? 'Show 3D' : 'Show table'}
            </button>
          </div>
          {table ? (
            <SurfaceTable grid={grid} measureLabel={measureLabel} />
          ) : (
            <RenderBoundary fallback={fallback('the 3D renderer failed to start')}>
              <Suspense fallback={<p className="sub">Loading the 3D renderer...</p>}>
                <ErrorSurfaceCanvas layout={layout} onHover={(i, on) => setHover((h) => (on ? i : h === i ? null : h))} />
              </Suspense>
              <p className="surface-readout" role="status" aria-live="polite">
                {bar ? `${bar.modelId}, h = ${bar.horizon}, ${bar.regime.label}: ${measureLabel} ${bar.value}` : 'Point at a bar to read its value.'}
              </p>
            </RenderBoundary>
          )}
          <figcaption className="legend">
            <span>
              The grey plane is the random walk (ratio 1). A bar rising above it loses to the random walk; a bar hanging below it beats it. Bar length is
              the log of the ratio, so a halving and a doubling are the same length.
            </span>
            {layout.range && (
              <span className="sub">
                Ratios drawn: {layout.range.min} to {layout.range.max}.
              </span>
            )}
            {layout.bars.length === 0 && <span className="sub">Pick at least one model to draw.</span>}
            <span className="sub">Drag to rotate, scroll or pinch to zoom, right-drag to pan.</span>
          </figcaption>
        </figure>
      )}
    </>
  )
}
