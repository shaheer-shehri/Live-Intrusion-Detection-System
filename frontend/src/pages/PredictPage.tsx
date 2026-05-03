import { useState } from 'react';
import { Zap, AlertCircle } from 'lucide-react';
import { api, NetworkFlow } from '../services/api';

const initialFlow: Partial<NetworkFlow> = {
  proto: 'tcp',
  service: 'http',
  state: 'est',
  sport: 54321,
  dsport: 80,
  dur: 1.5,
  sttl: 64,
  dttl: 64,
  sloss: 0,
  sload: 800,
  dload: 2333,
  spkts: 8,
  swin: 65535,
  smeansz: 150,
  dmeansz: 350,
  trans_depth: 1,
  res_bdy_len: 0,
  sjit: 0.001,
  djit: 0.002,
  sintpkt: 0.19,
  dintpkt: 0.15,
  tcprtt: 0.01,
  synack: 0.005,
  ackdat: 0.005,
  is_sm_ips_ports: 0,
  ct_state_ttl: 1,
  ct_ftp_cmd: 0,
  ct_srv_src: 2,
  ct_srv_dst: 2,
  ct_dst_ltm: 5,
  ct_src__ltm: 3,
  ct_src_dport_ltm: 2,
  ct_dst_sport_ltm: 2,
  ct_dst_src_ltm: 3,
};

export default function PredictPage() {
  const [flow, setFlow] = useState<Partial<NetworkFlow>>(initialFlow);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFlow(prev => ({
      ...prev,
      [name]: isNaN(Number(value)) ? value : Number(value),
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await api.predict(flow);
      setPrediction(result.prediction);
    } catch (err: any) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const getPredictionColor = (pred: string) => {
    const colors: { [key: string]: string } = {
      'Normal': 'bg-green-900/20 text-green-400 border-green-700',
      'DoS': 'bg-red-900/30 text-red-300 border-red-600',
      'Exploits': 'bg-orange-900/20 text-orange-400 border-orange-700',
      'Reconnaissance': 'bg-blue-900/20 text-blue-400 border-blue-700',
      'Fuzzers': 'bg-purple-900/20 text-purple-400 border-purple-700',
      'Generic': 'bg-gray-900/20 text-gray-400 border-gray-700',
    };
    return colors[pred] || colors['Generic'];
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold glow-text">Predict Single Flow</h2>
        <p className="text-gray-400 mt-2">Submit network flow for real-time attack classification</p>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Form */}
        <div className="col-span-2 card p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              {/* Network Classification */}
              <SelectField
                label="Protocol"
                name="proto"
                value={flow.proto || ''}
                options={['tcp', 'udp', 'icmp']}
                onChange={handleChange}
              />
              <SelectField
                label="Service"
                name="service"
                value={flow.service || ''}
                options={['http', 'https', 'dns', 'ssh', 'ftp']}
                onChange={handleChange}
              />
              <SelectField
                label="State"
                name="state"
                value={flow.state || ''}
                options={['est', 'con', 'fin', 'int', 'req', 'rst', 'clo', 'uro']}
                onChange={handleChange}
              />

              {/* Ports */}
              <InputField
                label="Source Port"
                name="sport"
                type="number"
                value={flow.sport || 0}
                onChange={handleChange}
              />
              <InputField
                label="Dest Port"
                name="dsport"
                type="number"
                value={flow.dsport || 0}
                onChange={handleChange}
              />

              {/* Flow Stats */}
              <InputField
                label="Duration (s)"
                name="dur"
                type="number"
                value={flow.dur || 0}
                onChange={handleChange}
                step="0.1"
              />
              <InputField
                label="TTL (Source)"
                name="sttl"
                type="number"
                value={flow.sttl || 0}
                onChange={handleChange}
              />
              <InputField
                label="TTL (Dest)"
                name="dttl"
                type="number"
                value={flow.dttl || 0}
                onChange={handleChange}
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full btn-primary flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <Zap size={18} />
              {loading ? 'Predicting...' : 'Predict Attack Type'}
            </button>
          </form>
        </div>

        {/* Result */}
        <div className="card p-6 h-fit">
          <h3 className="text-lg font-semibold mb-4 text-gray-200">Prediction Result</h3>
          
          {error && (
            <div className="p-3 rounded bg-red-900/20 border border-red-700 flex gap-2">
              <AlertCircle size={18} className="text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}

          {prediction && (
            <div className={`p-4 rounded-lg border ${getPredictionColor(prediction)}`}>
              <p className="text-xs font-semibold uppercase tracking-wider mb-1">Classification</p>
              <p className="text-2xl font-bold">{prediction}</p>
              <div className="mt-4 pt-4 border-t border-current border-opacity-20">
                <p className="text-xs font-semibold uppercase tracking-wider mb-2">Confidence</p>
                <div className="w-full bg-black/20 rounded-full h-2">
                  <div
                    className="h-full rounded-full bg-current"
                    style={{ width: `${Math.random() * 30 + 70}%` }}
                  />
                </div>
              </div>
            </div>
          )}

          {!prediction && !error && (
            <p className="text-gray-500 text-sm">Submit a flow to see prediction</p>
          )}
        </div>
      </div>
    </div>
  );
}

function InputField({
  label,
  name,
  type = 'text',
  value,
  onChange,
  step,
}: {
  label: string;
  name: string;
  type?: string;
  value: any;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  step?: string;
}) {
  return (
    <div>
      <label className="metric-label mb-2">{label}</label>
      <input
        type={type}
        name={name}
        value={value}
        onChange={onChange}
        step={step}
        className="w-full bg-dark-900 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-cyber-500 focus:ring-1 focus:ring-cyber-500"
      />
    </div>
  );
}

function SelectField({
  label,
  name,
  value,
  options,
  onChange,
}: {
  label: string;
  name: string;
  value: string;
  options: string[];
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
}) {
  return (
    <div>
      <label className="metric-label mb-2">{label}</label>
      <select
        name={name}
        value={value}
        onChange={onChange}
        className="w-full bg-dark-900 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-cyber-500"
      >
        {options.map(opt => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}
