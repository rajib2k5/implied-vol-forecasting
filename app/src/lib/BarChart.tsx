/**
 * Vertical bar chart from a zero baseline, for one series (an ACF by lag). Dataviz rules: thin bars
 * with a 2px surface gap, one hue (the series slot), a dashed reference band (the 95% white-noise
 * band of the ACF) labelled in text, a per-bar hover tooltip (mouse and keyboard focus), and a table
 * view. A single series needs no legend: the title names it.
 * UI component (a documented exception in app/CLAUDE.md): it uses the DOM, never computes a result.
 */
import { useState } from 'react'
import { strokeFor } from './chartGeom'
import { linearTicks, show } from './results'

const W = 340
const H = 180
const PAD = { l: 40, r: 8, t: 10, b: 22 }

export function BarChart({
  title,
  labels,
  values,
  band,
  slot,
  xLabel,
  domain,
}: {
  title: string
  labels: string[]
  values: (number | null)[]
  band?: number
  slot: number | null
  xLabel: string
  /** A shared [min, max] for small multiples that must be compared on one scale. */
  domain?: [number, number]
}) {
  const [hover, setHover] = useState<number | null>(null)
  const [table, setTable] = useState(false)
  const finite = values.filter((v): v is number => v !== null && Number.isFinite(v))
  if (finite.length === 0) return <p className="sub">No data to draw.</p>
  const lo = Math.min(0, ...finite, band !== undefined ? -band : 0, domain?.[0] ?? 0)
  const hi = Math.max(0, ...finite, band ?? 0, domain?.[1] ?? 0)
  const ticks = linearTicks(lo, hi, 4)
  const yMin = Math.min(lo, ticks[0])
  const yMax = Math.max(hi, ticks[ticks.length - 1])
  const py = (v: number) => PAD.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - PAD.t - PAD.b)
  const step = (W - PAD.l - PAD.r) / values.length
  const bw = Math.max(step - 2, 1) // 2px surface gap between adjacent bars

  return (
    <figure className="viz viz-small" aria-label={title}>
      <div className="viz-head">
        <figcaption className="viz-title">
          {title}
          {band !== undefined && <span className="viz-band-label"> · dashed: ±{band} (95% white-noise band)</span>}
        </figcaption>
        <button type="button" className="viz-toggle" aria-pressed={table} onClick={() => setTable((v) => !v)}>
          {table ? 'Chart' : 'Table'}
        </button>
      </div>
      {table ? (
        <table>
          <thead>
            <tr>
              <th scope="col">{xLabel}</th>
              <th scope="col">Value</th>
            </tr>
          </thead>
          <tbody>
            {labels.map((l, i) => (
              <tr key={l}>
                <td className="num">{l}</td>
                <td className="num">{show(values[i])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="viz-stage">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}, by ${xLabel}`}>
            {ticks.map((v) => (
              <g key={v}>
                <line className="viz-grid" x1={PAD.l} x2={W - PAD.r} y1={py(v)} y2={py(v)} />
                <text className="viz-tick" x={PAD.l - 5} y={py(v) + 4} textAnchor="end">{v}</text>
              </g>
            ))}
            {values.map((v, i) => {
              if (v === null || !Number.isFinite(v)) return null
              const y0 = py(0)
              const y1 = py(v)
              return (
                <rect
                  key={labels[i]}
                  data-testid="viz-bar"
                  tabIndex={0}
                  aria-label={`${xLabel} ${labels[i]}: ${v}`}
                  x={PAD.l + i * step + 1}
                  y={Math.min(y0, y1)}
                  width={bw}
                  height={Math.max(Math.abs(y1 - y0), 1)}
                  rx={1.5}
                  fill={strokeFor(slot)}
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover(null)}
                  onFocus={() => setHover(i)}
                  onBlur={() => setHover(null)}
                />
              )
            })}
            <line className="viz-baseline" x1={PAD.l} x2={W - PAD.r} y1={py(0)} y2={py(0)} />
            {band !== undefined && (
              <g data-testid="viz-band">
                <line className="viz-band" x1={PAD.l} x2={W - PAD.r} y1={py(band)} y2={py(band)} />
                <line className="viz-band" x1={PAD.l} x2={W - PAD.r} y1={py(-band)} y2={py(-band)} />
              </g>
            )}
            {[0, labels.length - 1].map((i) => (
              <text key={i} className="viz-tick" x={PAD.l + i * step + step / 2} y={H - 6} textAnchor="middle">
                {labels[i]}
              </text>
            ))}
          </svg>
          {hover !== null && (
            <div role="tooltip" className="viz-tooltip" style={{ left: `${((PAD.l + hover * step + step / 2) / W) * 100}%` }}>
              {xLabel} {labels[hover]}: <span className="num">{show(values[hover])}</span>
            </div>
          )}
        </div>
      )}
    </figure>
  )
}
