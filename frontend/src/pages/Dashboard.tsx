import { useEffect, useState } from 'react';
import { Line, Pie, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Activity, AlertTriangle, Zap, TrendingUp } from 'lucide-react';
import { api } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface MetricsData {
  requests?: {
    total: number;
    successful: number;
    failed: number;
    rate_limited: number;
  };
  latency_ms?: {
    average: number;
    p95: number;
    p99: number;
  };
  throughput?: {
    requests_per_second: number;
    active_requests: number;
  };
  circuit_breaker?: {
    state: string;
  };
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await api.getMetrics();
        setMetrics(data);
      } catch (error) {
        console.error('Error fetching metrics:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="inline-block animate-spin">
            <Zap className="text-cyber-400" size={32} />
          </div>
          <p className="mt-4 text-gray-400">Loading metrics...</p>
        </div>
      </div>
    );
  }

  const latencyData = {
    labels: ['P50', 'P95', 'P99'],
    datasets: [
      {
        label: 'Latency (ms)',
        data: [
          metrics?.latency_ms?.average || 0,
          metrics?.latency_ms?.p95 || 0,
          metrics?.latency_ms?.p99 || 0,
        ],
        borderColor: '#0ea5e9',
        backgroundColor: 'rgba(14, 165, 233, 0.1)',
        borderWidth: 2,
        fill: true,
        pointBackgroundColor: '#0ea5e9',
        pointBorderColor: '#fff',
        pointRadius: 5,
        tension: 0.4,
      },
    ],
  };

  const requestData = {
    labels: ['Successful', 'Failed', 'Rate Limited'],
    datasets: [
      {
        data: [
          metrics?.requests?.successful || 0,
          metrics?.requests?.failed || 0,
          metrics?.requests?.rate_limited || 0,
        ],
        backgroundColor: ['#10b981', '#ef4444', '#f59e0b'],
        borderColor: ['#059669', '#dc2626', '#d97706'],
        borderWidth: 2,
      },
    ],
  };

  const predictionData = {
    labels: ['Normal', 'DoS', 'Exploits', 'Reconnaissance', 'Fuzzers', 'Generic'],
    datasets: [
      {
        label: 'Predictions',
        data: [420, 85, 142, 63, 45, 38],
        backgroundColor: [
          '#10b981',
          '#ef4444',
          '#f59e0b',
          '#8b5cf6',
          '#ec4899',
          '#6366f1',
        ],
        borderColor: '#1a202c',
        borderWidth: 2,
      },
    ],
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-3xl font-bold glow-text">Live Dashboard</h2>
        <p className="text-gray-400">Real-time IDS metrics and performance</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          title="Total Requests"
          value={(metrics?.requests?.total || 0).toLocaleString()}
          icon={<Activity size={24} />}
          trend="+12.5%"
          trendUp={true}
        />
        <StatCard
          title="Success Rate"
          value={`${(((metrics?.requests?.successful || 0) / (metrics?.requests?.total || 1)) * 100).toFixed(1)}%`}
          icon={<TrendingUp size={24} />}
          trend="+2.1%"
          trendUp={true}
        />
        <StatCard
          title="Avg Latency"
          value={`${(metrics?.latency_ms?.average || 0).toFixed(1)}ms`}
          icon={<Zap size={24} />}
          trend="-5.2%"
          trendUp={true}
        />
        <StatCard
          title="Circuit Breaker"
          value={metrics?.circuit_breaker?.state?.toUpperCase() || 'CLOSED'}
          icon={<AlertTriangle size={24} />}
          status={metrics?.circuit_breaker?.state === 'closed' ? 'ok' : 'alert'}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        {/* Latency Chart */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">API Latency</h3>
          <Line data={latencyData} options={{ maintainAspectRatio: true }} />
        </div>

        {/* Request Distribution */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">Request Status</h3>
          <Pie data={requestData} options={{ maintainAspectRatio: true }} />
        </div>
      </div>

      {/* Prediction Distribution */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-200">Attack Type Distribution</h3>
        <Bar
          data={predictionData}
          options={{
            indexAxis: 'x',
            maintainAspectRatio: true,
            scales: {
              y: {
                beginAtZero: true,
                ticks: { color: '#9ca3af' },
                grid: { color: '#374151' },
              },
              x: {
                ticks: { color: '#9ca3af' },
                grid: { color: '#374151' },
              },
            },
          }}
        />
      </div>

      {/* Recent Alerts */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-200">Recent Alerts</h3>
        <div className="space-y-3">
          {[
            { type: 'DoS', severity: 'high', time: '2 min ago', ip: '192.168.1.105' },
            { type: 'Exploits', severity: 'medium', time: '5 min ago', ip: '10.0.0.50' },
            { type: 'Reconnaissance', severity: 'low', time: '12 min ago', ip: '172.16.0.1' },
          ].map((alert, i) => (
            <div
              key={i}
              className="flex items-center justify-between p-3 bg-dark-900 rounded border border-gray-700 hover:border-gray-600"
            >
              <div className="flex items-center gap-3">
                <AlertTriangle
                  size={16}
                  className={
                    alert.severity === 'high'
                      ? 'text-red-400'
                      : alert.severity === 'medium'
                      ? 'text-yellow-400'
                      : 'text-blue-400'
                  }
                />
                <div>
                  <p className="text-sm font-medium text-gray-200">{alert.type}</p>
                  <p className="text-xs text-gray-500">{alert.ip}</p>
                </div>
              </div>
              <span className="text-xs text-gray-400">{alert.time}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  icon,
  trend,
  trendUp,
  status,
}: {
  title: string;
  value: string;
  icon: React.ReactNode;
  trend?: string;
  trendUp?: boolean;
  status?: 'ok' | 'alert';
}) {
  return (
    <div className="card p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="metric-label">{title}</p>
          <p className="metric-value">{value}</p>
          {trend && (
            <p
              className={`text-xs mt-2 ${trendUp ? 'text-green-400' : 'text-red-400'}`}
            >
              {trend}
            </p>
          )}
        </div>
        <div
          className={`p-3 rounded-lg ${
            status === 'alert'
              ? 'bg-red-900/20 text-red-400'
              : 'bg-cyber-600/20 text-cyber-400'
          }`}
        >
          {icon}
        </div>
      </div>
    </div>
  );
}
