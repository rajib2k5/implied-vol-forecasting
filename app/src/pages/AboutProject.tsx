import { DATA_SOURCES, DISCLAIMER, MILESTONES, PROJECT_INTRO, REQUIREMENT, TEACHING_POINTS, type Milestone } from '../lib/project'

export default function AboutProject({ milestones = MILESTONES }: { milestones?: readonly Milestone[] }) {
  return (
    <section className="about">
      <h2>About this project</h2>
      <p className="sub">{PROJECT_INTRO}</p>

      <h2>The original requirement</h2>
      <blockquote className="card requirement">{REQUIREMENT}</blockquote>

      <h2>What it teaches</h2>
      <ul className="grid teach" aria-label="Teaching points">
        {TEACHING_POINTS.map((t) => (
          <li className="card" key={t.id}>
            <h3>{t.title}</h3>
            <p>{t.body}</p>
          </li>
        ))}
      </ul>

      <h2>Roadmap</h2>
      <ol className="roadmap" aria-label="Milestones">
        {milestones.map((m) => (
          <li key={m.id} className={`card ${m.status}`} aria-current={m.status === 'current' ? 'step' : undefined}>
            <div>
              <span className="badge">M{m.id}</span>
              <strong>{m.title}</strong>
              {m.status !== 'planned' && <span className={`badge ${m.status === 'current' ? 'current' : 'ok-badge'}`}>{m.status}</span>}
            </div>
            <p>{m.summary}</p>
          </li>
        ))}
      </ol>

      <h2>Data sources and licensing</h2>
      <p className="sub">
        CBOE data is not redistributed with this project. The app ships a synthetic fixture instead, and anything drawn from it is labelled synthetic.
      </p>
      <ul className="sources" aria-label="Data sources">
        {DATA_SOURCES.map((s) => (
          <li className="card" key={s.id}>
            <h3>{s.name}</h3>
            <p>{s.provides}</p>
            <p className="sub">{s.terms}</p>
          </li>
        ))}
      </ul>

      <h2>Not financial advice</h2>
      <p className="disclaimer">{DISCLAIMER}</p>
    </section>
  )
}
