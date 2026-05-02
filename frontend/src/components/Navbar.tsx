import { Menu, Bell, Settings, LogOut } from 'lucide-react';

interface NavbarProps {
  onMenuClick: () => void;
}

export default function Navbar({ onMenuClick }: NavbarProps) {
  return (
    <nav className="bg-dark-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between shadow-lg">
      <div className="flex items-center gap-4">
        <button
          onClick={onMenuClick}
          className="p-2 hover:bg-dark-700 rounded-lg transition"
        >
          <Menu size={20} className="text-cyber-400" />
        </button>
        
        <div>
          <h1 className="text-2xl font-bold glow-text">🛡️ IDS Dashboard</h1>
          <p className="text-xs text-gray-500">Intrusion Detection System</p>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <button className="p-2 hover:bg-dark-700 rounded-lg relative">
          <Bell size={20} className="text-gray-400 hover:text-cyber-400" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse" />
        </button>

        <button className="p-2 hover:bg-dark-700 rounded-lg">
          <Settings size={20} className="text-gray-400 hover:text-cyber-400" />
        </button>

        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyber-500 to-cyber-700 flex items-center justify-center text-sm font-bold cursor-pointer hover:shadow-glow">
          AD
        </div>

        <button className="p-2 hover:bg-dark-700 rounded-lg">
          <LogOut size={18} className="text-gray-400 hover:text-red-400" />
        </button>
      </div>
    </nav>
  );
}
