import { useState } from 'react'
import { XYChart, type XYSeries } from '../lib/XYChart'
import { colorSlot, parseEvaluation, parseModelRace, show, slotOrder, targetDates, type Envelope, type Evaluation, type ModelRace } from '../lib/results'
import { Gate, ModeBadge, useDataset } from './useDataset'

/** Shown first: the random walk, the HAR baseline and the best-known structural model of each target. */
const DEFAULT_MODELS = ['rw', 'har_log', 'har_rv', 'arima_garch_log', 'garch']
const TARGET_LABEL: Record<string, string> = { vix_level: 'VIX close', rv21: '21-session realised volatility' }

const panelName = (targetId: string, horizon: number) => `${TARGET_LABEL[targetId] ?? targetId}, h = ${horizon}`

function Race({ race, evaluation }: { race: Envelope<ModelRace>; evaluation: Envelope<Evaluation> | null }) {
  const [panelIdx, setPanelIdx] = useState(0)
  const [chosen, setChosen] = useState<string[]>(DEFAULT_MODELS)
  const { panels } = race.payload
  const panel = panels[Math.min(panelIdx, panels.length - 1)]
  const registry = slotOrder(
    panel.series.map((s) => s.modelId),
    DEFAULT_MODELS,
  )
  const dates = targetDates(race.payload, panel)
  const series: XYSeries[] = [
    { id: 'actual', label: 'actual', values: panel.actual, slot: null },
    ...panel.series
      .filter((s) => chosen.includes(s.modelId))
      .map((s) => ({ id: s.modelId, label: s.modelId, values: s.yhat, slot: colorSlot(registry, s.modelId) })),
  ]
  const losses = evaluation?.payload.losses.filter((l) => l.targetId === panel.targetId && l.horizon === panel.horizon) ?? []
  // The lowest by the ratio to the random walk, which keeps 6 decimals: two RMSEs that tie at 4 decimals
  // (1.1798 for ARIMA and ARIMA-GARCH at h = 1) are told apart by it.
  const best = (key: 'rmseRatioRw' | 'qlikeRatioRw') => Math.min(...losses.map((l) => l[key] ?? Infinity))
  const toggle = (id: string) => setChosen((c) => (c.includes(id) ? c.filter((m) => m !== id) : [...c, id]))

  return (
    <>
      <div className="controls" role="group" aria-label="Target and horizon">
        {panels.map((p, i) => (
          <button key={`${p.targetId}-${p.horizon}`} type="button" aria-pressed={p === panel} className={p === panel ? 'primary' : ''} onClick={() => setPanelIdx(i)}>
            {panelName(p.targetId, p.horizon)}
          </button>
        ))}
      </div>
      <fieldset className="controls model-picks">
        <legend className="sub">Models drawn (colour stays with the model)</legend>
        {registry.map((id) => (
          <label key={id}>
            <input type="checkbox" checked={chosen.includes(id)} onChange={() => toggle(id)} /> {id}
          </label>
        ))}
      </fieldset>
      <XYChart
        title={`Forecast vs actual: ${panelName(panel.targetId, panel.horizon)}, last ${panel.n} development origins`}
        x={dates}
        series={series}
        yLabel="vol points, at the target date"
      />
      <h2>Losses over the whole development period</h2>
      {evaluation === null ? (
        <p className="sub">The loss table needs evaluation.json, which did not load.</p>
      ) : (
        <div className="scroll-x">
          <table aria-label="Losses by model">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">n</th>
                <th scope="col">RMSE</th>
                <th scope="col">RMSE / RW</th>
                <th scope="col">QLIKE</th>
                <th scope="col">QLIKE / RW</th>
                <th scope="col">MZ beta</th>
                <th scope="col">MZ Wald p</th>
              </tr>
            </thead>
            <tbody>
              {losses.map((l) => {
                const mz = panel.mz.find((m) => m.modelId === l.modelId)
                return (
                  <tr key={l.modelId}>
                    <th scope="row">{l.modelId}</th>
                    <td className="num">{l.n}</td>
                    <td className={`num${l.rmseRatioRw === best('rmseRatioRw') ? ' best' : ''}`}>{show(l.rmse)}</td>
                    <td className="num">{show(l.rmseRatioRw)}</td>
                    <td className={`num${l.qlikeRatioRw === best('qlikeRatioRw') ? ' best' : ''}`}>{show(l.qlike)}</td>
                    <td className="num">{show(l.qlikeRatioRw)}</td>
                    <td className="num">{show(mz?.beta)}</td>
                    <td className="num">{show(mz?.waldP)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="sub">
        Bold: the lowest loss in the column (ties at the shown decimals are broken by the ratio to the random walk). RMSE and QLIKE are the primary losses; a ratio below 1 beats the random walk. MZ: Mincer-Zarnowitz
        regression of the actual on the forecast, Wald test of alpha = 0 and beta = 1 (Hansen-Hodrick errors).
      </p>
    </>
  )
}

export default function ModelRacePage() {
  const race = useDataset('model_race', parseModelRace)
  const evaluation = useDataset('evaluation', parseEvaluation)
  return (
    <section className="results">
      <h2>Model race: forecast vs actual</h2>
      <Gate loaded={race}>
        {(r) => (
          <>
            <div>
              <ModeBadge mode={r.dataMode} />
              <span className="badge">development period</span>
            </div>
            <p className="sub">
              Each line is a model's forecast, drawn at the date it is scored against. Milestones 2 to 4: random walk, historical means, ARIMA,
              HAR, the GARCH family and the LSTM, on identical walk-forward folds. The holdout has not been looked at.
            </p>
            <Race race={r} evaluation={evaluation.state === 'ready' ? evaluation.value : null} />
          </>
        )}
      </Gate>
    </section>
  )
}
