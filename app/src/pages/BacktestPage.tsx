import { XYChart } from '../lib/XYChart'
import { colorSlot, parseEquityCurves, show, type Envelope, type EquityCurves } from '../lib/results'
import { Gate, ModeBadge, useDataset } from './useDataset'

function View({ env }: { env: Envelope<EquityCurves> }) {
  const eq = env.payload
  const ids = eq.strategies.map((s) => s.strategyId)
  // Cash is a flat reference line at 1, drawn dashed in the ink colour; the three traded series take
  // slots 1 to 3, the slots validated for every pair in both themes.
  const CASH = 'bench_cash'
  const traded = ids.filter((id) => id !== CASH)
  const label = (id: string) => eq.strategies.find((s) => s.strategyId === id)?.label ?? id
  const costs = [...new Set(eq.costSensitivity.map((r) => r.costBps as number))]
  return (
    <>
      <div>
        <ModeBadge mode={env.dataMode} />
        <span className="badge">development period</span>
        <span className="badge">{eq.costBps} bps costs</span>
      </div>
      <p className="sub">
        The pre-registered rule trades the 21-day forecast against the VIX: long VXX when the forecast is above it by more than the band, long
        SVXY when below, cash otherwise. Growth of 1 after costs, on a log scale so that a halving and a doubling look the same size.
      </p>
      <XYChart
        title={`Equity, ${eq.dates[0]} to ${eq.dates[eq.dates.length - 1]}`}
        x={eq.dates}
        series={eq.strategies.map((s) => ({
          id: s.strategyId,
          label: s.label,
          short: s.kind === 'forecast' ? 'primary rule' : s.label.replace('always ', ''),
          values: s.equity,
          slot: s.strategyId === CASH ? null : colorSlot(traded, s.strategyId),
          dashed: s.strategyId === CASH,
        }))}
        yLabel="growth of 1 (log scale)"
        log
        height={300}
      />
      <h2>Is the Sharpe ratio luck?</h2>
      <div className="scroll-x">
        <table aria-label="Backtest summary">
          <thead>
            <tr>
              <th scope="col">Strategy</th>
              <th scope="col">Ann. return</th>
              <th scope="col">Sharpe (annualised)</th>
              <th scope="col">PSR</th>
              <th scope="col">DSR</th>
              <th scope="col">Max drawdown</th>
              <th scope="col">Trials</th>
            </tr>
          </thead>
          <tbody>
            {eq.summary.map((r) => (
              <tr key={r.strategyId}>
                <th scope="row">{label(r.strategyId)}</th>
                <td className="num">{show(r.annReturn)}</td>
                <td className="num">{show(r.sharpe)}</td>
                <td className="num">{show(r.psr0)}</td>
                <td className="num">{show(r.dsr)}</td>
                <td className="num">{show(r.maxDrawdown)}</td>
                <td className="num">{r.nTrials}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="sub">Sharpe: daily Sharpe x √252. PSR: probability that the true Sharpe ratio is above 0. DSR: the same against the best Sharpe ratio expected from the number of strategies tried.</p>
      <h2>Cost sensitivity (annualised Sharpe ratio)</h2>
      {costs.length === 0 ? (
        <p className="sub">This file has no cost-sensitivity table (schema before 1.1.0).</p>
      ) : (
        <div className="scroll-x">
          <table aria-label="Cost sensitivity">
            <thead>
              <tr>
                <th scope="col">Strategy</th>
                {costs.map((c) => (
                  <th key={c} scope="col">{c} bps</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ids.map((id) => (
                <tr key={id}>
                  <th scope="row">{label(id)}</th>
                  {costs.map((c) => (
                    <td key={c} className="num">{show(eq.costSensitivity.find((r) => r.strategyId === id && r.costBps === c)?.sharpe)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <h2>Caveats</h2>
      <ul className="caveats">
        {eq.caveats.map((c) => (
          <li key={c}>{c}</li>
        ))}
      </ul>
    </>
  )
}

export default function BacktestPage() {
  const loaded = useDataset('equity_curves', parseEquityCurves)
  return (
    <section className="results">
      <h2>Backtest: VXX and SVXY on the forecast</h2>
      <Gate loaded={loaded}>{(env) => <View env={env} />}</Gate>
      <p className="disclaimer">Hypothetical results on generated prices. Not financial advice.</p>
    </section>
  )
}
