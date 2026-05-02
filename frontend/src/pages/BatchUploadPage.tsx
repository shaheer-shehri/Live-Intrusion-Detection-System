import { useState } from 'react';
import { Upload, AlertTriangle, CheckCircle } from 'lucide-react';
import { api } from '../services/api';

export default function BatchUploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [classCounts, setClassCounts] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setPredictions([]);
      setClassCounts({});
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a CSV file');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await api.predictBatch(file);
      setPredictions(result.rows || []);
      setClassCounts(result.class_counts || {});
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message || 'Batch prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold glow-text">Batch Upload</h2>
        <p className="text-gray-400 mt-2">Upload CSV file with network flows for bulk prediction</p>
      </div>

      <div className="card p-8">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex items-center justify-center w-full">
            <label className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-gray-700 rounded-lg cursor-pointer hover:border-cyber-500 transition">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <Upload className="w-10 h-10 text-cyber-400 mb-2" />
                <p className="mb-2 text-sm text-gray-300">
                  <span className="font-semibold">Click to upload</span> or drag and drop
                </p>
                <p className="text-xs text-gray-500">CSV file with network flows</p>
              </div>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
          </div>

          {file && (
            <div className="p-3 bg-green-900/20 border border-green-700 rounded flex items-center gap-2">
              <CheckCircle size={18} className="text-green-400" />
              <span className="text-sm text-green-300">{file.name} selected ({(file.size / 1024).toFixed(1)} KB)</span>
            </div>
          )}

          {error && (
            <div className="p-3 bg-red-900/20 border border-red-700 rounded flex items-center gap-2">
              <AlertTriangle size={18} className="text-red-400" />
              <span className="text-sm text-red-300">{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !file}
            className="w-full btn-primary flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <Upload size={18} />
            {loading ? 'Processing...' : 'Predict All Flows'}
          </button>
        </form>
      </div>

      {Object.keys(classCounts).length > 0 && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">Class Distribution</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(classCounts).map(([cls, cnt]) => (
              <div key={cls} className="p-4 bg-dark-900 rounded border border-gray-700">
                <p className="metric-label">{cls}</p>
                <p className="metric-value">{cnt}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {predictions.length > 0 && (
        <div className="card overflow-hidden">
          <div className="p-6 border-b border-gray-700">
            <h3 className="text-lg font-semibold text-gray-200">
              Predictions ({predictions.length})
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700 bg-dark-900">
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">#</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">Protocol</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">Service</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">State</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">SrcPort</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">DstPort</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-400">Prediction</th>
                </tr>
              </thead>
              <tbody>
                {predictions.slice(0, 50).map((pred, i) => (
                  <tr key={i} className="border-b border-gray-700 hover:bg-dark-900/50">
                    <td className="px-4 py-3 text-sm text-gray-400">{i + 1}</td>
                    <td className="px-4 py-3 text-sm text-gray-300">{pred.proto ?? '-'}</td>
                    <td className="px-4 py-3 text-sm text-gray-300">{pred.service ?? '-'}</td>
                    <td className="px-4 py-3 text-sm text-gray-300">{pred.state ?? '-'}</td>
                    <td className="px-4 py-3 text-sm text-gray-300">{pred.sport ?? '-'}</td>
                    <td className="px-4 py-3 text-sm text-gray-300">{pred.dsport ?? '-'}</td>
                    <td className="px-4 py-3 text-sm">
                      <PredictionBadge prediction={pred.prediction} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {predictions.length > 50 && (
            <div className="p-4 text-center text-sm text-gray-500 border-t border-gray-700">
              Showing 50 of {predictions.length} results
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PredictionBadge({ prediction }: { prediction: string }) {
  const colors: { [key: string]: string } = {
    'Normal': 'badge-normal',
    'DoS': 'badge-dos',
    'Exploits': 'badge-exploits',
    'Reconnaissance': 'badge-attack',
    'Fuzzers': 'badge-attack',
    'Generic': 'badge-attack',
  };
  return <span className={colors[prediction] || colors['Generic']}>{prediction}</span>;
}
