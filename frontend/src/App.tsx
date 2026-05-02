import { useState } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import PredictPage from './pages/PredictPage';
import MetricsPage from './pages/MetricsPage';
import BatchUploadPage from './pages/BatchUploadPage';
import LiveMonitorPage from './pages/LiveMonitorPage';

type Page = 'dashboard' | 'live' | 'predict' | 'batch' | 'metrics';

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('live');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'live':
        return <LiveMonitorPage />;
      case 'predict':
        return <PredictPage />;
      case 'batch':
        return <BatchUploadPage />;
      case 'metrics':
        return <MetricsPage />;
      default:
        return <LiveMonitorPage />;
    }
  };

  return (
    <div className="flex h-screen bg-dark-950 text-gray-100 overflow-hidden">
      <Sidebar
        currentPage={currentPage}
        setCurrentPage={setCurrentPage}
        isOpen={sidebarOpen}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar onMenuClick={() => setSidebarOpen(!sidebarOpen)} />

        <main className="flex-1 overflow-auto">
          <div className="p-6">
            {renderPage()}
          </div>
        </main>
      </div>
    </div>
  );
}
