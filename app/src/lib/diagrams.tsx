/**
 * Dependency-free ANIMATED SVG diagram primitives for the Architecture page.
 *
 * Animation model (v3 — continuous, not one-shot):
 *  - SequenceDiagram: a JS stepper draws messages one by one with a traveling dot,
 *    holds, then loops; Replay restarts immediately.
 *  - ActivityDiagram: a token walks the flow node-to-node forever; the active node
 *    is highlighted.
 *  - ArchDiagram: layers rise in once, then dashed connectors flow continuously.
 * Theme-aware via CSS custom properties. prefers-reduced-motion keeps the step/fade
 * cues (non-motion) but disables travel, flow, and entrance movement.
 * Timers here are UI-only animation; they never feed a computed result.
 */
import { useEffect, useState, type CSSProperties } from 'react'

const STYLE = `
@keyframes ivfDraw { to { stroke-dashoffset: 0; } }
@keyframes ivfRise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes ivfFlow { to { stroke-dashoffset: -20; } }
@keyframes ivfTravel { from { transform: translateX(0); } to { transform: translateX(var(--dx)); } }
.ivf-arch-layer { opacity: 0; animation: ivfRise .5s ease forwards; animation-delay: var(--d); }
.ivf-arch-flow { stroke-dasharray: 4 6; animation: ivfFlow 1.1s linear infinite; }
.ivf-seq-line.pending { opacity: .18; }
.ivf-seq-line.run { animation: ivfDraw .6s ease forwards; }
.ivf-seq-line.done { stroke-dashoffset: 0; }
.ivf-seq-label.pending { opacity: .25; }
.ivf-seq-label.run, .ivf-seq-label.done { opacity: 1; transition: opacity .3s ease; }
.ivf-seq-dot { animation: ivfTravel .8s ease forwards; }
.ivf-act-token { transition: transform .45s ease; }
.ivf-act-active-shape { stroke: var(--accent, #888) !important; stroke-width: 2; }
.ivf-diagram text { font-family: var(--mono, ui-monospace, Menlo, monospace); }
@media (prefers-reduced-motion: reduce) {
  .ivf-arch-layer { animation: none; opacity: 1; }
  .ivf-arch-flow { animation: none; stroke-dasharray: none; }
  .ivf-seq-line.run { animation: none; stroke-dashoffset: 0; }
  .ivf-seq-dot { display: none; }
  .ivf-act-token { transition: none; }
}
`

function Defs({ id }: { id: string }) {
  return (
    <defs>
      <marker id={id} markerWidth="9" markerHeight="8" refX="8" refY="4" orient="auto-start-reverse">
        <path d="M0,0 L9,4 L0,8 z" fill="var(--accent, #888)" />
      </marker>
    </defs>
  )
}

/** Shared looping stepper: cycles 0..(count-1) then `holdPhases` rest phases. */
function useLoopStep(count: number, holdPhases: number, ms: number): [number, () => void] {
  const [phase, setPhase] = useState(0)
  const total = count + holdPhases
  useEffect(() => {
    if (total <= 0) return
    const t = setInterval(() => setPhase((p) => (p + 1) % total), ms)
    return () => clearInterval(t)
  }, [total, ms])
  return [phase, () => setPhase(0)]
}

/* ---------------- Architecture ---------------- */

export interface ArchLayer {
  label: string
  boxes: string[]
}

export function ArchDiagram({ layers }: { layers: ArchLayer[] }) {
  const W = 720
  const boxH = 40
  const gap = 52
  const top = 14
  const H = top + layers.length * boxH + (layers.length - 1) * gap + 14
  const x0 = 118
  const avail = W - x0 - 14
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ivf-diagram" role="img" aria-label="Architecture diagram" style={{ width: '100%', height: 'auto' }}>
      <style>{STYLE}</style>
      <Defs id="ivf-arch-arrow" />
      {layers.map((layer, i) => {
        const y = top + i * (boxH + gap)
        const n = layer.boxes.length
        const bw = (avail - (n - 1) * 10) / n
        return (
          <g key={layer.label} className="ivf-arch-layer" style={{ '--d': `${i * 0.16}s` } as CSSProperties}>
            <text x={8} y={y + boxH / 2 + 4} fontSize="10" fill="var(--muted, #888)" letterSpacing="1">
              {layer.label.toUpperCase()}
            </text>
            {layer.boxes.map((b, j) => (
              <g key={b}>
                <rect x={x0 + j * (bw + 10)} y={y} width={bw} height={boxH} rx="2" fill="var(--surface, #222)" stroke="var(--line, #555)" />
                <text x={x0 + j * (bw + 10) + bw / 2} y={y + boxH / 2 + 4} fontSize="11" textAnchor="middle" fill="var(--ink, #ddd)">
                  {b}
                </text>
              </g>
            ))}
            {i < layers.length - 1 && (
              <line
                className="ivf-arch-flow"
                x1={x0 + avail / 2}
                y1={y + boxH + 4}
                x2={x0 + avail / 2}
                y2={y + boxH + gap - 6}
                stroke="var(--accent, #888)"
                strokeWidth="1.4"
                markerEnd="url(#ivf-arch-arrow)"
              />
            )}
          </g>
        )
      })}
    </svg>
  )
}

/* ---------------- Sequence ---------------- */

export interface SeqStep {
  from: number
  to: number
  label: string
}

export function SequenceDiagram({ actors, steps, onReplay }: { actors: string[]; steps: SeqStep[]; onReplay?: () => void }) {
  // steps.length drawing phases + 3 hold phases at the end, looping forever
  const [phase, restart] = useLoopStep(steps.length, 3, 850)
  const W = 720
  const xFor = (i: number) => 80 + (i * (W - 160)) / Math.max(actors.length - 1, 1)
  const y0 = 74
  const dy = 42
  const H = y0 + steps.length * dy + 18
  const stateOf = (i: number) => (i < phase ? 'done' : i === phase ? 'run' : 'pending')
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="ivf-diagram" role="img" aria-label="Sequence diagram" style={{ width: '100%', height: 'auto' }}>
        <style>{STYLE}</style>
        <Defs id="ivf-seq-arrow" />
        {actors.map((a, i) => (
          <g key={a}>
            <rect x={xFor(i) - 52} y={10} width={104} height={26} rx="2" fill="var(--surface, #222)" stroke="var(--line, #555)" />
            <text x={xFor(i)} y={27} fontSize="11" textAnchor="middle" fill="var(--ink, #ddd)">
              {a}
            </text>
            <line x1={xFor(i)} y1={38} x2={xFor(i)} y2={H - 8} stroke="var(--line, #555)" strokeWidth="1" strokeDasharray="3 4" />
          </g>
        ))}
        {steps.map((s, i) => {
          const y = y0 + i * dy
          const st = stateOf(i)
          if (s.from === s.to) {
            return (
              <text key={i} className={`ivf-seq-label ${st}`} x={xFor(s.from) + 8} y={y + 4} fontSize="10" fill="var(--muted, #888)">
                ↺ {s.label}
              </text>
            )
          }
          const x1 = xFor(s.from)
          const x2 = xFor(s.to)
          const dir = x2 > x1 ? 1 : -1
          const len = Math.abs(x2 - x1) - 10
          return (
            <g key={i}>
              <line
                className={`ivf-seq-line ${st}`}
                x1={x1}
                y1={y}
                x2={x2 - dir * 8}
                y2={y}
                stroke="var(--accent, #888)"
                strokeWidth="1.4"
                markerEnd="url(#ivf-seq-arrow)"
                strokeDasharray={len}
                strokeDashoffset={st === 'done' ? 0 : len}
              />
              <text className={`ivf-seq-label ${st}`} x={(x1 + x2) / 2} y={y - 7} fontSize="10" textAnchor="middle" fill="var(--muted, #888)">
                {s.label}
              </text>
              {st === 'run' && (
                <circle
                  key={`dot-${phase}`}
                  className="ivf-seq-dot"
                  data-testid="ivf-seq-dot"
                  cx={x1}
                  cy={y}
                  r="3.4"
                  fill="var(--accent, #888)"
                  style={{ '--dx': `${x2 - dir * 8 - x1}px` } as CSSProperties}
                />
              )}
            </g>
          )
        })}
      </svg>
      <div className="controls">
        <button
          onClick={() => {
            restart()
            onReplay?.()
          }}
        >
          Replay sequence
        </button>
      </div>
    </div>
  )
}

/* ---------------- Activity ---------------- */

export interface ActNode {
  id: string
  label: string
  type?: 'start' | 'action' | 'decision' | 'end'
}

export interface ActEdge {
  from: string
  to: string
  label?: string
}

export function ActivityDiagram({ nodes, edges }: { nodes: ActNode[]; edges: ActEdge[] }) {
  // token walks node 0..n-1, holds one phase, loops
  const [phase] = useLoopStep(nodes.length, 1, 800)
  const active = phase < nodes.length ? phase : -1
  const W = 720
  const cx = 300
  const dy = 74
  const yFor = (i: number) => 34 + i * dy
  const H = yFor(nodes.length - 1) + 40
  const idx = new Map(nodes.map((n, i) => [n.id, i]))
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ivf-diagram" role="img" aria-label="Activity diagram" style={{ width: '100%', height: 'auto' }}>
      <style>{STYLE}</style>
      <Defs id="ivf-act-arrow" />
      {edges.map((e, k) => {
        const a = idx.get(e.from)
        const b = idx.get(e.to)
        if (a === undefined || b === undefined) return null
        const yA = yFor(a)
        const yB = yFor(b)
        if (b === a + 1) {
          return (
            <g key={k}>
              <line x1={cx} y1={yA + 20} x2={cx} y2={yB - 24} stroke="var(--line, #666)" strokeWidth="1.3" markerEnd="url(#ivf-act-arrow)" />
              {e.label && (
                <text x={cx + 10} y={(yA + yB) / 2 + 2} fontSize="10" fill="var(--muted, #888)">
                  {e.label}
                </text>
              )}
            </g>
          )
        }
        const side = 465 + Math.abs(b - a) * 22
        return (
          <g key={k} data-testid="ivf-act-side">
            <path
              d={`M ${cx + 90} ${yA} C ${side} ${yA}, ${side} ${yB}, ${cx + 94} ${yB}`}
              fill="none"
              stroke="var(--line, #666)"
              strokeWidth="1.3"
              markerEnd="url(#ivf-act-arrow)"
            />
            {e.label && (
              <text x={side} y={(yA + yB) / 2 + 2} fontSize="10" textAnchor="middle" fill="var(--muted, #888)">
                {e.label}
              </text>
            )}
          </g>
        )
      })}
      {nodes.map((n, i) => {
        const y = yFor(i)
        const t = n.type ?? 'action'
        const activeCls = i === active ? ' ivf-act-active-shape' : ''
        return (
          <g key={n.id} data-active={i === active || undefined}>
            {t === 'start' && <circle className={activeCls} cx={cx} cy={y} r="9" fill="var(--accent, #888)" />}
            {t === 'end' && (
              <g>
                <circle className={activeCls} cx={cx} cy={y} r="11" fill="none" stroke="var(--accent, #888)" strokeWidth="1.4" />
                <circle cx={cx} cy={y} r="6" fill="var(--accent, #888)" />
              </g>
            )}
            {t === 'action' && <rect className={activeCls} x={cx - 90} y={y - 17} width="180" height="34" rx="3" fill="var(--surface, #222)" stroke="var(--line, #555)" />}
            {t === 'decision' && (
              <polygon
                className={activeCls}
                points={`${cx},${y - 21} ${cx + 95},${y} ${cx},${y + 21} ${cx - 95},${y}`}
                fill="var(--surface, #222)"
                stroke="var(--accent, #888)"
                strokeWidth="1.2"
                data-testid="ivf-act-decision"
              />
            )}
            {(t === 'action' || t === 'decision') && (
              <text x={cx} y={y + 4} fontSize="11" textAnchor="middle" fill="var(--ink, #ddd)">
                {n.label}
              </text>
            )}
            {(t === 'start' || t === 'end') && (
              <text x={cx + 104} y={y + 4} fontSize="10" fill="var(--muted, #888)">
                {n.label}
              </text>
            )}
          </g>
        )
      })}
      {active >= 0 && (
        <circle
          className="ivf-act-token"
          data-testid="ivf-act-token"
          cx={cx - 130}
          cy={yFor(0)}
          r="5"
          fill="var(--accent, #888)"
          style={{ transform: `translateY(${yFor(active) - yFor(0)}px)` }}
        />
      )}
    </svg>
  )
}
