import { AppShell } from './lib/shell'
import { APP_SLUG, PROJECT_PITCH, PROJECT_TITLE } from './lib/project'
import AboutProject from './pages/AboutProject'
import ArchitecturePage from './pages/ArchitecturePage'
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
        { id: 'term-structure', label: 'Term structure 3D', render: () => <TermStructurePage /> },
      ]}
    />
  )
}
