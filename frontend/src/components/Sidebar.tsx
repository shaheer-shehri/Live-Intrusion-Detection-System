import { LayoutDashboard, Zap, Upload, BarChart3, ChevronRight, Radio } from 'lucide-react';

interface SidebarProps {
  currentPage: string;
  setCurrentPage: (page: any) => void;
  isOpen: boolean;
}

const menuItems = [
  { id: 'live', label: 'Live Monitor', icon: Radio },
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'predict', label: 'Predict Flow', icon: Zap },
  { id: 'batch', label: 'Batch Upload', icon: Upload },
  { id: 'metrics', label: 'Metrics', icon: BarChart3 },
];

export default function Sidebar({ currentPage, setCurrentPage, isOpen }: SidebarProps) {
  return (
    <aside
      className={`${
        isOpen ? 'w-64' : 'w-20'
      } bg-dark-900 border-r border-gray-700 transition-all duration-300 flex flex-col shadow-lg`}
    >
      {/* Logo */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center justify-center h-10 bg-gradient-to-r from-cyber-500 to-cyber-600 rounded-lg">
          {isOpen && <span className="text-white font-bold text-sm">IDS</span>}
          {!isOpen && <span className="text-white font-bold">🔒</span>}
        </div>
      </div>

      {/* Menu Items */}
      <nav className="flex-1 px-4 py-6 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 group ${
                isActive
                  ? 'bg-cyber-600/20 border-l-2 border-cyber-500 text-cyber-400'
                  : 'text-gray-400 hover:bg-dark-800 hover:text-gray-300'
              }`}
            >
              <Icon size={20} className={isActive ? 'text-cyber-400' : 'group-hover:text-cyber-400'} />
              {isOpen && (
                <>
                  <span className="flex-1 text-left text-sm font-medium">{item.label}</span>
                  {isActive && <ChevronRight size={16} />}
                </>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-6 border-t border-gray-700">
        {isOpen && (
          <div className="text-xs text-gray-500 space-y-1">
            <p className="font-semibold">Status: 🟢 Online</p>
            <p>Model: XGBoost</p>
            <p>Accuracy: 94.43%</p>
          </div>
        )}
      </div>
    </aside>
  );
}
