/**
 * Diagrams page: overview copy plus three animated diagrams
 * (architecture, sequence, activity). Content is app-specific and passed in.
 */
import { ArchDiagram, SequenceDiagram, ActivityDiagram, type ArchLayer, type SeqStep, type ActNode, type ActEdge } from './diagrams'

/**
 * One concrete way someone could use this app. `page` names the nav tab that delivers it, so a
 * reader can go and do it; a use case with no page behind it is a claim, not a use case.
 */
export interface UseCase {
  title: string
  who: string
  how: string
  page: string
}

// Self-contained so every app gets a usable layout without touching its own stylesheet; the
// colours come from the app's theme tokens, with fallbacks for apps that name them differently.
const USECASE_STYLE = `
.usecases { list-style: none; margin: 0 0 1rem; padding: 0; display: grid; gap: .8rem; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
.usecase { margin: 0; display: flex; flex-direction: column; gap: .3rem; }
.usecase-title { font-weight: 600; color: var(--ink, inherit); }
.usecase-who { font-size: .8rem; color: var(--accent, currentColor); }
.usecase-how { margin: 0; font-size: .88rem; color: var(--muted, inherit); flex: 1; }
.usecase-page { align-self: flex-start; margin-top: .2rem; }
`

/** Section headings; every one is optional and falls back to a generic default. */
export interface DiagramsHeadings {
  title?: string
  architecture?: string
  sequence?: string
  activity?: string
}

export interface DiagramsContent {
  headings?: DiagramsHeadings
  intro: string[]
  useCases?: UseCase[]
  architecture: { layers: ArchLayer[] }
  sequence: { actors: string[]; steps: SeqStep[] }
  activity: { nodes: ActNode[]; edges: ActEdge[] }
}

export function DiagramsPage({ content }: { content: DiagramsContent }) {
  const h = content.headings ?? {}
  return (
    <section className="diagrams">
      <h2>{h.title ?? 'How this system works'}</h2>
      {content.intro.map((p, i) => (
        <p className="sub" key={i}>
          {p}
        </p>
      ))}
      {content.useCases && content.useCases.length > 0 && (
        <>
          <style>{USECASE_STYLE}</style>
          <h2>Use cases</h2>
          <ul className="usecases" aria-label="Use cases">
            {content.useCases.map((u) => (
              <li className="card usecase" key={u.title}>
                <div className="usecase-title">{u.title}</div>
                <div className="usecase-who">{u.who}</div>
                <p className="usecase-how">{u.how}</p>
                <span className="badge usecase-page">{u.page}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <h2>{h.architecture ?? 'Architecture'}</h2>
      <div className="card">
        <ArchDiagram layers={content.architecture.layers} />
      </div>
      <h2>{h.sequence ?? 'Sequence — primary flow'}</h2>
      <div className="card">
        <SequenceDiagram actors={content.sequence.actors} steps={content.sequence.steps} />
      </div>
      <h2>{h.activity ?? 'Activity — decision flow'}</h2>
      <div className="card">
        <ActivityDiagram nodes={content.activity.nodes} edges={content.activity.edges} />
      </div>
    </section>
  )
}
