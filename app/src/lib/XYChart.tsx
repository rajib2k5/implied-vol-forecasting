/**
 * Line chart for the result views, built to the dataviz skill's rules:
 * - one y axis (linear or log), recessive grid, 2px lines, no dual axis;
 * - colour from a fixed slot per entity (`--series-N`, validated for both themes), never by rank;
 *   a null slot draws in the ink colour (used for the actual series);
 * - a legend for two or more series, plus direct labels at the line ends for four or fewer;
 *   all text in text tokens, never the series colour;
 * - a crosshair and tooltip on hover (mouse) and on arrow keys (keyboard focus);
 * - a table view of the same numbers, because some light-theme slots sit below 3:1 contrast.
 * UI component (a documented exception in app/CLAUDE.md): it uses the DOM, never computes a result.
 */
import { useState, type KeyboardEvent, type MouseEvent } from 'react'
import { segments, spreadLabels, strokeFor } from './chartGeom'
import { linearTicks, logTicks, nearestIndex, show } from './results'

export interface XYSeries {
  id: string
  label: string
  values: (number | null)[]
  slot: number | null
  dashed?: boolean
  /** Shorter name for the direct label at the line end; the legend and table use `label`. */
  short?: string
}

const W = 720
const PAD = { l: 56, t: 12, b: 26 }
const LABEL_GAP = 13

export function XYChart({
  title,
  x,
  series,
  yLabel,
  log = false,
  height = 260,
}: {
  title: string
  x: string[]
  series: XYSeries[]
  yLabel: string
  log?: boolean
  height?: number
}) {
  const [hover, setHover] = useState<number | null>(null)
  const [table, setTable] = useState(false)
  const finite = series.flatMap((s) => s.values).filter((v): v is number => v !== null && Number.isFinite(v) && (!log || v > 0))
  if (finite.length === 0 || x.length === 0) return <p className="sub">No data to draw.</p>
  const direct = series.length <= 4
  const padR = direct ? 132 : 12
  const lo = Math.min(...finite)
  const hi = Math.max(...finite)
  const inner = log ? logTicks(lo, hi) : linearTicks(lo, hi, 5)
  // A linear axis runs from the round tick at or below the data to the one at or above it.
  const stepY = inner.length > 1 ? inner[1] - inner[0] : 0
  const ticks = log || stepY === 0 ? inner : [...(inner[0] > lo ? [inner[0] - stepY] : []), ...inner, ...(inner[inner.length - 1] < hi ? [inner[inner.length - 1] + stepY] : [])].map((v) => Number(v.toPrecision(12)))
  const yMin = log ? lo : Math.min(lo, ticks[0] ?? lo)
  const yMax = log ? hi : Math.max(hi, ticks[ticks.length - 1] ?? hi)
  const t = (v: number) => (log ? Math.log(v) : v)
  const span = t(yMax) - t(yMin) || 1
  const plotW = W - PAD.l - padR
  const px = (i: number) => PAD.l + (x.length === 1 ? plotW / 2 : (i / (x.length - 1)) * plotW)
  const py = (v: number) => PAD.t + (1 - (t(v) - t(yMin)) / span) * (height - PAD.t - PAD.b)
  const ends = series.map((s) => {
    const segs = segments(s.values)
    const last = segs[segs.length - 1]
    return last ? py(last.points[last.points.length - 1]) : PAD.t
  })
  const labelYs = spreadLabels(ends, LABEL_GAP)
  const xTicks = [...new Set([0, Math.floor((x.length - 1) / 2), x.length - 1])]

  const move = (e: MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    setHover(nearestIndex(x.length, box.width > 0 ? (e.clientX - box.left) / box.width : 0))
  }
  const key = (e: KeyboardEvent<SVGSVGElement>) => {
    if (e.key === 'ArrowRight') setHover((h) => Math.min(x.length - 1, (h ?? -1) + 1))
    else if (e.key === 'ArrowLeft') setHover((h) => Math.max(0, (h ?? x.length) - 1))
    else if (e.key === 'Escape') setHover(null)
  }

  return (
    <figure className="viz" aria-label={title}>
      <div className="viz-head">
        {series.length > 1 && (
          <ul className="viz-legend" aria-label="Legend">
            {series.map((s) => (
              <li key={s.id}>
                <svg width="18" height="8" aria-hidden="true">
                  <line x1="1" x2="17" y1="4" y2="4" stroke={strokeFor(s.slot)} strokeWidth="2" strokeDasharray={s.dashed ? '4 3' : undefined} />
                </svg>
                {s.label}
              </li>
            ))}
          </ul>
        )}
        <button type="button" className="viz-toggle" aria-pressed={table} onClick={() => setTable((v) => !v)}>
          {table ? 'Show chart' : 'Show table'}
        </button>
      </div>
      {table ? (
        <div className="viz-table scroll-x">
          <table>
            <caption className="sub">{title}</caption>
            <thead>
              <tr>
                <th scope="col">Date</th>
                {series.map((s) => (
                  <th key={s.id} scope="col">{s.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {x.map((d, i) => (
                <tr key={d + i}>
                  <td className="num">{d}</td>
                  {series.map((s) => (
                    <td key={s.id} className="num">{show(s.values[i])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="viz-stage">
          <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`${title}. ${yLabel}.`} tabIndex={0} onKeyDown={key} onBlur={() => setHover(null)}>
            <text x={4} y={10} className="viz-axis-label">{yLabel}</text>
            {ticks.map((v) => (
              <g key={v}>
                <line className="viz-grid" x1={PAD.l} x2={PAD.l + plotW} y1={py(v)} y2={py(v)} />
                <text className="viz-tick" x={PAD.l - 6} y={py(v) + 4} textAnchor="end">{v}</text>
              </g>
            ))}
            {xTicks.map((i) => (
              <text key={i} className="viz-tick" x={px(i)} y={height - 8} textAnchor={i === 0 ? 'start' : i === x.length - 1 ? 'end' : 'middle'}>
                {x[i]}
              </text>
            ))}
            {series.map((s) =>
              segments(s.values).map((seg) => (
                <polyline
                  key={`${s.id}-${seg.start}`}
                  data-series={s.id}
                  className="viz-line"
                  fill="none"
                  stroke={strokeFor(s.slot)}
                  strokeDasharray={s.dashed ? '5 4' : undefined}
                  points={seg.points.map((v, k) => `${px(seg.start + k)},${py(v)}`).join(' ')}
                />
              )),
            )}
            {direct &&
              series.map((s, k) => (
                <g key={s.id} className="viz-direct">
                  <circle cx={PAD.l + plotW + 8} cy={labelYs[k] - 3} r="3" fill={strokeFor(s.slot)} />
                  <text x={PAD.l + plotW + 15} y={labelYs[k]} className="viz-direct-label">{s.short ?? s.label}</text>
                </g>
              ))}
            {hover !== null && (
              <g data-testid="viz-crosshair" pointerEvents="none">
                <line className="viz-crosshair" x1={px(hover)} x2={px(hover)} y1={PAD.t} y2={height - PAD.b} />
                {series.map((s) => {
                  const v = s.values[hover]
                  return v === null || v === undefined || (log && v <= 0) ? null : (
                    <circle key={s.id} cx={px(hover)} cy={py(v)} r="4" fill={strokeFor(s.slot)} className="viz-dot" />
                  )
                })}
              </g>
            )}
            <rect
              data-testid="viz-hit"
              x={PAD.l}
              y={PAD.t}
              width={plotW}
              height={height - PAD.t - PAD.b}
              fill="transparent"
              onMouseMove={move}
              onMouseLeave={() => setHover(null)}
            />
          </svg>
          {hover !== null && (
            <div role="tooltip" className="viz-tooltip" style={{ left: `${(px(hover) / W) * 100}%` }}>
              <div className="viz-tooltip-date">{x[hover]}</div>
              {series.map((s) => (
                <div key={s.id}>
                  <i className="swatch" style={{ background: strokeFor(s.slot) }} />
                  {s.label}: <span className="num">{show(s.values[hover])}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </figure>
  )
}
