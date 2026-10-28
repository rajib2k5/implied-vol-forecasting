import { useMemo } from 'react'
import { DiagramsPage } from '../lib/diagramsPage'
import {
  FOLD_ACTORS,
  FOLD_STEPS,
  PIPELINE_EDGES,
  PIPELINE_LAYERS,
  PIPELINE_NODES,
  PROCESS_EDGES,
  PROCESS_NODES,
  describeEdges,
  toLayers,
} from '../lib/architecture'

export default function ArchitecturePage() {
  const layers = useMemo(() => toLayers(PIPELINE_NODES, PIPELINE_LAYERS), [])
  const edgeLines = useMemo(() => describeEdges(PIPELINE_NODES, PIPELINE_EDGES), [])
  return (
    <>
      <DiagramsPage
        content={{
          headings: {
            title: 'Architecture: from raw index files to this app',
            architecture: 'Pipeline',
            sequence: 'Sequence: one walk-forward fold',
            activity: 'Activity: one milestone',
          },
          intro: [
            'The Python package does all the computation. The app never computes a result; it only draws the versioned JSON the pipeline exports, plus small tested transforms such as the ribbon geometry.',
            'Arrows in the pipeline diagram show layer order. The exact data dependencies, which skip layers in places, are listed under it.',
            'From real data the repository only ever holds a closed publication set P (ADR 0001 D2): unitless, rounded results per whole period, such as a model\'s RMSE as a ratio to the random walk\'s. Daily series, per-date errors and equity curves stay on the author\'s machine.',
          ],
          architecture: { layers },
          sequence: { actors: FOLD_ACTORS, steps: [...FOLD_STEPS] },
          activity: { nodes: [...PROCESS_NODES], edges: [...PROCESS_EDGES] },
        }}
      />
      <details className="card">
        <summary>Exact data dependencies (text version of the pipeline)</summary>
        <ul aria-label="Pipeline edges">
          {edgeLines.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </details>
    </>
  )
}
