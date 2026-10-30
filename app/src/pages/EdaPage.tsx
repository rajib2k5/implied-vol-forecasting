import { BarChart } from '../lib/BarChart'
import { parseEda, show, type Eda, type Envelope } from '../lib/results'
import { Gate, ModeBadge, useDataset } from './useDataset'

const SAMPLE_LABEL: Record<string, string> = { pre_holdout: 'all sessions before the holdout', burn_in: 'burn-in (before dev)', dev: 'development period' }

/** The return series share one y scale, so |r| and r² can be compared with r by eye; ln VIX has its own. */
const RETURNS = ['r', 'abs_r', 'r2']

function View({ env }: { env: Envelope<Eda> }) {
  const eda = env.payload
  const shared = eda.acf.filter((s) => RETURNS.includes(s.seriesId)).flatMap((s) => [...s.acf, s.band95, -s.band95]).filter((v): v is number => v !== null)
  const domain: [number, number] = [Math.min(0, ...shared), Math.max(0, ...shared)]
  return (
    <>
      <div>
        <ModeBadge mode={env.dataMode} />
        <span className="badge">
          {eda.first} to {eda.last}
        </span>
      </div>
      <p className="sub">
        Returns barely autocorrelate, but their size does: |r| and r² stay correlated for weeks. That is volatility clustering, the reason a
        volatility forecast can beat a constant. The dashed lines are the 95% band for white noise.
      </p>
      <div className="grid small-multiples">
        {eda.acf.map((s) => (
          <BarChart
            key={s.seriesId}
            title={`ACF of ${s.label}${RETURNS.includes(s.seriesId) ? '' : ' (own scale)'}`}
            labels={s.acf.map((_, i) => String(i + 1))}
            values={s.acf}
            band={s.band95}
            slot={1}
            xLabel="lag"
            domain={RETURNS.includes(s.seriesId) ? domain : undefined}
          />
        ))}
      </div>
      <h2>How fast does a VIX shock fade?</h2>
      <p className="sub">AR(1) on ln VIX: half-life = ln(0.5) / ln(phi) sessions, with its 95% confidence interval.</p>
      <div className="grid">
        {eda.halfLife.map((h) => (
          <div key={h.sample} className="stat">
            <div className="k">{SAMPLE_LABEL[h.sample] ?? h.sample}</div>
            <div className="v">{show(h.halfLife)} sessions</div>
            <div className="sub ci">
              95% CI {show(h.halfLifeCi95[0])} to {h.halfLifeCi95[1] === null ? 'unbounded' : show(h.halfLifeCi95[1])}; phi {show(h.phi)}; n {h.n}
            </div>
          </div>
        ))}
      </div>
    </>
  )
}

export default function EdaPage() {
  const loaded = useDataset('eda', parseEda)
  return (
    <section className="results">
      <h2>Exploratory analysis: volatility clustering</h2>
      <Gate loaded={loaded}>{(env) => <View env={env} />}</Gate>
    </section>
  )
}
