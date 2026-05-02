import { useState, useEffect } from 'react';
import { Activity, Zap, AlertTriangle } from 'lucide-react';
import { api } from '../services/api';

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await api.getMetrics();
        setMetrics(data);
        setHistory(prev => [data, ...prev].slice(0, 20));
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!metrics) return <div className="text-center text-gray-400">Loading...</div>;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold glow-text">Metrics & Performance</h2>
        <p className="text-gray-400 mt-2">Real-time system performance indicators</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="Total Requests"
          value={(metrics.requests?.total || 0).toLocaleString()}
          icon={<Activity size={24} />}
          color="cyber"
        />
        <MetricCard
          title="Success Rate"
          value={`${(((metrics.requests?.successful || 0) / (metrics.requests?.total || 1)) * 100).toFixed(1)}%`}
          icon={<Zap size={24} />}
          color="green"
        />
        <MetricCard
          title="Failed"
          value={(metrics.requests?.failed || 0).toString()}
          icon={<AlertTriangle size={24} />}
          color="red"
        />
        <MetricCard
          title="Rate Limited"
          value={(metrics.requests?.rate_limited || 0).toString()}
          icon={<AlertTriangle size={24} />}
          color="yellow"
        />
      </div>

      {/* Latency Metrics */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-200">API Latency (ms)</h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="p-4 bg-dark-900 rounded border border-gray-700">
            <p className="metric-label">Average</p>
            <p className="metric-value">{(metrics.latency_ms?.average || 0).toFixed(1)}</p>
          </div>
          <div className="p-4 bg-dark-900 rounded border border-gray-700">
            <p className="metric-label">P95</p>
            <p className="metric-value">{(metrics.latency_ms?.p95 || 0).toFixed(1)}</p>
          </div>
          <div className="p-4 bg-dark-900 rounded border border-gray-700">
            <p className="metric-label">P99</p>
            <p className="metric-value">{(metrics.latency_ms?.p99 || 0).toFixed(1)}</p>
          </div>
        </div>
      </div>

      {/* Throughput */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-200">Throughput</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-dark-900 rounded border border-gray-700">
            <p className="metric-label">Requests/Second</p>
            <p className="metric-value">{(metrics.throughput?.requests_per_second || 0).toFixed(2)}</p>
          </div>
          <div className="p-4 bg-dark-900 rounded border border-gray-700">
            <p className="metric-label">Active Requests</p>
            <p className="metric-value">{metrics.throughput?.active_requests || 0}</p>
          </div>
        </div>
      </div>

      {/* Circuit Breaker Status */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-200">Circuit Breaker Status</h3>
        <div className={`p-4 rounded border-l-4 ${
          metrics.circuit_breaker?.state === 'closed'
            ? 'border-green-500 bg-green-900/10'
            : 'border-red-500 bg-red-900/10'
        }`}>
          <p className="text-sm font-semibold mb-2">Current State</p>
          <p className="text-2xl font-bold">
            {metrics.circuit_breaker?.state?.toUpperCase() || 'CLOSED'}
          </p>
          <div className={`mt-3 inline-block px-3 py-1 rounded text-sm font-semibold ${
            metrics.circuit_breaker?.state === 'closed'
              ? 'bg-green-900/30 text-green-400 border border-green-700'
              : 'bg-red-900/30 text-red-400 border border-red-700'
          }`}>
            {metrics.circuit_breaker?.state === 'closed' ? '✓ Operational' : '✗ Alert'}
          </div>
        </div>
      </div>

      {/* Metrics History */}
      {history.length > 1 && (
        <div className="card overflow-hidden">
          <div className="p-6 border-b border-gray-700">
            <h3 className="text-lg font-semibold text-gray-200">Historical Metrics</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700 bg-dark-900">
                  <th className="px-4 py-3 text-left font-semibold text-gray-400">Timestamp</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-400">Total Req</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-400">Avg Latency</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-400">Req/Sec</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 10).map((m, i) => (
                  <tr key={i} className="border-b border-gray-700 hover:bg-dark-900/50">
                    <td className="px-4 py-3 text-gray-300 text-xs">
                      {new Date().toLocaleTimeString()}
                    </td>
                    <td className="px-4 py-3 text-gray-300">{m.requests?.total || 0}</td>
                    <td className="px-4 py-3 text-gray-300">
                      {(m.latency_ms?.average || 0).toFixed(1)}ms
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      {(m.throughput?.requests_per_second || 0).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({
  title,
  value,
  icon,
  color,
}: {
  title: string;
  value: string;
  icon: React.ReactNode;
  color: 'cyber' | 'green' | 'red' | 'yellow';
}) {
  const colorMap = {
    cyber: 'bg-cyber-600/20 text-cyber-400',
    green: 'bg-green-900/20 text-green-400',
    red: 'bg-red-900/20 text-red-400',
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
