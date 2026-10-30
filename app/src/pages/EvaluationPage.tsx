import { useState } from 'react'
import { dmMatrix, parseEvaluation, show, type DmVerdict, type Envelope, type Evaluation } from '../lib/results'
import { Gate, ModeBadge, useDataset } from './useDataset'

const PANELS: [string, number][] = [['vix_level', 1], ['vix_level', 5], ['vix_level', 21], ['rv21', 21]]
const LOSSES: [string, string][] = [['sq_error', 'squared error'], ['qlike', 'QLIKE']]
const GLYPH: Record<DmVerdict, string> = { row: '←', col: '↑', none: '', self: '', missing: 'n/a' }
const MEANING: Record<DmVerdict, string> = {
  row: 'the row model has the lower loss',
  col: 'the column model has the lower loss',
  none: 'no difference at Holm p < 0.05',
  self: 'same model',
  missing: 'pair not tested',
}

function Matrix({ ev, targetId, horizon, loss }: { ev: Evaluation; targetId: string; horizon: number; loss: string }) {
  const { models, cells } = dmMatrix(ev, targetId, horizon, loss)
  return (
    <figure className="viz" aria-label="Diebold-Mariano matrix">
      <div className="scroll-x">
        <table className="dm-matrix">
          <thead>
            <tr>
              <th scope="col">row vs column</th>
              {models.map((m) => (
                <th key={m} scope="col">{m}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((a, i) => (
              <tr key={a}>
                <th scope="row">{a}</th>
                {cells[i].map((c, j) => (
                  <td key={models[j]} className={`num dm-${c.verdict}`} data-verdict={c.verdict} title={`${a} vs ${models[j]}: ${MEANING[c.verdict]}${c.pHolm === null ? '' : `, Holm p ${c.pHolm}`}`}>
                    {c.verdict === 'self' ? '' : `${GLYPH[c.verdict]} ${show(c.pHolm)}`.trim()}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption className="legend">
        <span><i className="swatch dm-row" /> ← the row model has the lower loss (Holm p &lt; 0.05)</span>
        <span><i className="swatch dm-col" /> ↑ the column model has the lower loss</span>
        <span><i className="swatch dm-none" /> no significant difference</span>
        <span className="sub">Each cell shows the Holm-adjusted p-value.</span>
      </figcaption>
    </figure>
  )
}

function View({ env }: { env: Envelope<Evaluation> }) {
  const [panel, setPanel] = useState(0)
  const [loss, setLoss] = useState('sq_error')
  const ev = env.payload
  const [targetId, horizon] = PANELS[panel]
  const mz = ev.mz.filter((m) => m.targetId === targetId && m.horizon === horizon)
  return (
    <>
      <div>
        <ModeBadge mode={env.dataMode} />
        <span className="badge">development period</span>
      </div>
      <p className="sub">
        Diebold-Mariano tests of equal predictive accuracy for every pair of models, with the {ev.dmMethod.replace('_', '-')} long-run variance (b ={' '}
        {ev.dmFixedB}) chosen by a size simulation, and Holm's correction across the pairs. The holdout has not been looked at.
      </p>
      <div className="controls" role="group" aria-label="Target and horizon">
        {PANELS.map(([t, h], i) => (
          <button key={`${t}-${h}`} type="button" aria-pressed={i === panel} className={i === panel ? 'primary' : ''} onClick={() => setPanel(i)}>
            {t} h = {h}
          </button>
        ))}
        {LOSSES.map(([id, label]) => (
          <button key={id} type="button" aria-pressed={id === loss} className={id === loss ? 'primary' : ''} onClick={() => setLoss(id)}>
            {label}
          </button>
        ))}
      </div>
      <Matrix ev={ev} targetId={targetId} horizon={horizon} loss={loss} />
      <h2>Mincer-Zarnowitz calibration</h2>
      <p className="sub">
        actual = alpha + beta x forecast; a calibrated forecast has alpha = 0 and beta = 1. Wald p tests both at once ({ev.mzMethod.replace('_', '-')}{' '}
        errors). Walk-forward bias calibration helped in {ev.calibration.helps} of {ev.calibration.cells} cells.
      </p>
      <div className="scroll-x">
        <table aria-label="Mincer-Zarnowitz regressions">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">n</th>
              <th scope="col">alpha</th>
              <th scope="col">beta</th>
              <th scope="col">R²</th>
              <th scope="col">Wald p</th>
            </tr>
          </thead>
          <tbody>
            {mz.map((m) => (
              <tr key={m.modelId}>
                <th scope="row">{m.modelId}</th>
                <td className="num">{m.n}</td>
                <td className="num">{show(m.alpha)}</td>
                <td className="num">{show(m.beta)}</td>
                <td className="num">{show(m.r2)}</td>
                <td className="num">{show(m.waldP)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

export default function EvaluationPage() {
  const loaded = useDataset('evaluation', parseEvaluation)
  return (
    <section className="results">
      <h2>Evaluation: is the difference real?</h2>
      <Gate loaded={loaded}>{(env) => <View env={env} />}</Gate>
    </section>
  )
}
