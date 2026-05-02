import { useEffect, useRef, useState } from 'react';
import { Activity, AlertTriangle, Shield, Pause, Play } from 'lucide-react';
import { api, MonitorFlow, MonitorStats } from '../services/api';

const MAX_FLOWS = 200;

export default function LiveMonitorPage() {
  const [flows, setFlows] = useState<MonitorFlow[]>([]);
  const [stats, setStats] = useState<MonitorStats | null>(null);
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const pausedRef = useRef(paused);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    const es = new EventSource(api.monitorLiveURL());
    esRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (event) => {
      if (pausedRef.current) return;
      try {
        const data = JSON.parse(event.data) as { flows: MonitorFlow[]; stats: MonitorStats };
        if (data.stats) setStats(data.stats);
        if (data.flows && data.flows.length > 0) {
          setFlows((prev) => {
            const merged = [...prev, ...data.flows];
            return merged.slice(-MAX_FLOWS);
          });
        }
      } catch {
        /* ignore malformed event */
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, []);

  const attackActive = stats?.current_state && stats.current_state !== 'normal';

  const recent = [...flows].reverse();

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-3xl font-bold glow-text">Live Traffic Monitor</h2>
          <p className="text-gray-400 mt-2">
            Real-time classification stream. Visiting one of the listed sites injects a matching attack burst.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`px-3 py-1 rounded text-xs font-semibold ${
            connected ? 'bg-green-900/30 text-green-400 border border-green-700' : 'bg-red-900/30 text-red-400 border border-red-700'
          }`}>
            {connected ? '● LIVE' : '● DISCONNECTED'}
          </span>
          <button
            onClick={() => setPaused(p => !p)}
            className="btn-primary flex items-center gap-2"
          >
            {paused ? <Play size={16} /> : <Pause size={16} />}
            {paused ? 'Resume' : 'Pause'}
          </button>
        </div>
      </div>

      {attackActive && (
        <div className="card p-4 border-l-4 border-red-500 bg-red-900/10">
          <div className="flex items-center gap-3">
            <AlertTriangle className="text-red-400" size={24} />
            <div className="flex-1">
              <p className="font-bold text-red-300">
                ATTACK DETECTED — {String(stats?.current_state).toUpperCase()}
              </p>
              <p className="text-sm text-gray-400">
                Reverts to normal in {stats?.attack_expires_in_sec}s
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-4 gap-4">
        <StatCard title="Total Flows"   value={(stats?.total_flows  ?? 0).toLocaleString()} icon={<Activity size={24} />} color="cyber" />
        <StatCard title="Normal Flows"  value={(stats?.normal_flows ?? 0).toLocaleString()} icon={<Shield size={24} />}   color="green" />
        <StatCard title="Attack Flows"  value={(stats?.attack_flows ?? 0).toLocaleString()} icon={<AlertTriangle size={24} />} color="red" />
        <StatCard title="Attack Rate"   value={`${(stats?.attack_rate_pct ?? 0).toFixed(1)}%`} icon={<Activity size={24} />} color="yellow" />
      </div>

      <div className="card overflow-hidden">
        <div className="p-6 border-b border-gray-700 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-200">Live Flows ({flows.length})</h3>
          <span className="text-xs text-gray-500">Newest first · max {MAX_FLOWS}</span>
        </div>
        <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-dark-900 z-10">
              <tr className="border-b border-gray-700">
                <th className="px-3 py-2 text-left font-semibold text-gray-400">Time</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-400">Source</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-400">Dest</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-400">Proto</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-400">Service</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-400">Prediction</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-400">Conf</th>
                <th className="px-3 py-2 text-left font-semibold text-gray-400">Domain</th>
              </tr>
            </thead>
            <tbody>
              {recent.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-3 py-8 text-center text-gray-500">
                    Waiting for traffic...
                  </td>
                </tr>
              )}
              {recent.map((f, i) => (
                <tr
                  key={`${f.ts}-${i}`}
                  className={`border-b border-gray-800 ${
                    f.is_attack ? 'bg-red-900/10' : 'hover:bg-dark-900/50'
                  }`}
                >
                  <td className="px-3 py-2 text-gray-400 text-xs">{f.time}</td>
                  <td className="px-3 py-2 text-gray-300 text-xs">{f.source_ip}:{f.src_port}</td>
                  <td className="px-3 py-2 text-gray-300 text-xs">{f.dest_ip}:{f.dst_port}</td>
                  <td className="px-3 py-2 text-gray-300">{f.protocol}</td>
                  <td className="px-3 py-2 text-gray-300">{f.service}</td>
                  <td className="px-3 py-2"><PredictionBadge prediction={f.prediction} /></td>
                  <td className="px-3 py-2 text-gray-400">{(f.confidence * 100).toFixed(1)}%</td>
                  <td className="px-3 py-2 text-xs text-gray-400">{f.source_domain ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function PredictionBadge({ prediction }: { prediction: string }) {
  const colors: { [key: string]: string } = {
    Normal:         'bg-green-900/30 text-green-400 border-green-700',
    DoS:            'bg-red-900/30 text-red-400 border-red-700',
    Exploits:       'bg-orange-900/30 text-orange-400 border-orange-700',
    Reconnaissance: 'bg-blue-900/30 text-blue-400 border-blue-700',
    Fuzzers:        'bg-purple-900/30 text-purple-400 border-purple-700',
    Generic:        'bg-pink-900/30 text-pink-400 border-pink-700',
  };
  const cls = colors[prediction] ?? colors.Generic;
  return <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${cls}`}>{prediction}</span>;
}

function StatCard({
  title, value, icon, color,
}: {
  title: string;
  value: string;
  icon: React.ReactNode;
  color: 'cyber' | 'green' | 'red' | 'yellow';
}) {
  const colorMap = {
    cyber:  'bg-cyber-600/20 text-cyber-400',
    green:  'bg-green-900/20 text-green-400',
    red:    'bg-red-900/20 text-red-400',
    yellow: 'bg-yellow-900/20 text-yellow-400',
  };
  return (
    <div className="card p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="metric-label">{title}</p>
          <p className="metric-value">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${colorMap[color]}`}>{icon}</div>
      </div>
    </div>
  );
}
