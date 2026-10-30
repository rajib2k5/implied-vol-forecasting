import { AppShell } from './lib/shell'
import { APP_SLUG, PROJECT_PITCH, PROJECT_TITLE } from './lib/project'
import AboutProject from './pages/AboutProject'
import ArchitecturePage from './pages/ArchitecturePage'
import BacktestPage from './pages/BacktestPage'
import EdaPage from './pages/EdaPage'
import EvaluationPage from './pages/EvaluationPage'
import ModelRacePage from './pages/ModelRacePage'
import TermStructurePage from './pages/TermStructurePage'

export default function App() {
  return (
    <AppShell
      slug={APP_SLUG}
      title={PROJECT_TITLE}
      subtitle={PROJECT_PITCH}
      pages={[
        { id: 'about', label: 'About', render: () => <AboutProject /> },
        { id: 'architecture', label: 'Architecture', render: () => <ArchitecturePage /> },
        { id: 'eda', label: 'EDA', render: () => <EdaPage /> },
        { id: 'term-structure', label: 'Term structure 3D', render: () => <TermStructurePage /> },
        { id: 'model-race', label: 'Model race', render: () => <ModelRacePage /> },
        { id: 'evaluation', label: 'Evaluation', render: () => <EvaluationPage /> },
        { id: 'backtest', label: 'Backtest', render: () => <BacktestPage /> },
      ]}
    />
  )
}
