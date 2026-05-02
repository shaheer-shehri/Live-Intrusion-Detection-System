import axios from 'axios';

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? 'http://localhost:8000';

export interface NetworkFlow {
  srcip?: string;
  dstip?: string;
  proto: string;
  service: string;
  state: string;
  sport: number;
  dsport: number;
  dur: number;
  sbytes: number;
  dbytes: number;
  sttl: number;
  dttl: number;
  sloss: number;
  dloss: number;
  sload: number;
  dload: number;
  spkts: number;
  dpkts: number;
  swin: number;
  dwin: number;
  stcpb: number;
  dtcpb: number;
  smeansz: number;
  dmeansz: number;
  trans_depth: number;
  res_bdy_len: number;
  sjit: number;
  djit: number;
  stime: number;
  ltime: number;
  sintpkt: number;
  dintpkt: number;
  tcprtt: number;
  synack: number;
  ackdat: number;
  is_sm_ips_ports: number;
  ct_state_ttl: number;
  ct_flw_http_mthd: number;
  is_ftp_login: number;
  ct_ftp_cmd: number;
  ct_srv_src: number;
  ct_srv_dst: number;
  ct_dst_ltm: number;
  ct_src__ltm: number;
  ct_src_dport_ltm: number;
  ct_dst_sport_ltm: number;
  ct_dst_src_ltm: number;
}

export interface MonitorFlow {
  ts: number;
  time: string;
  prediction: string;
  confidence: number;
  source_ip: string;
  dest_ip: string;
  protocol: string;
  service: string;
  src_port: number;
  dst_port: number;
  is_attack: boolean;
  source_domain: string | null;
  description: string | null;
  cve_hint: string | null;
}

export interface MonitorStats {
  total_flows: number;
  normal_flows: number;
  attack_flows: number;
  attack_rate_pct: number;
  class_counts: Record<string, number>;
  current_state: string;
  active_scenario: any;
  attack_expires_in_sec: number;
  session_duration_sec: number;
}

export interface MonitorEvent {
  flows: MonitorFlow[];
  stats: MonitorStats;
}

export const API_BASE_URL = API_BASE;

export const api = {
  async predict(flow: Partial<NetworkFlow>) {
    const response = await axios.post(`${API_BASE}/predict`, flow);
    return response.data;
  },

  async predictBatch(file: File) {
    const form = new FormData();
    form.append('file', file);
    const response = await axios.post(`${API_BASE}/predict-batch`, form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  async assessDrift(flows: Partial<NetworkFlow>[], topN = 10) {
    const response = await axios.post(`${API_BASE}/drift/assess`, {
      flows,
      top_n: topN,
      store_history: true,
    });
    return response.data;
  },

  async getMetrics() {
    const response = await axios.get(`${API_BASE}/metrics`);
    return response.data;
  },

  async getHealth() {
    const response = await axios.get(`${API_BASE}/health`);
    return response.data;
  },

  async getMonitorSnapshot(n = 50): Promise<MonitorEvent> {
    const response = await axios.get(`${API_BASE}/monitor`, { params: { n } });
    return response.data;
  },

  monitorLiveURL(): string {
    return `${API_BASE}/monitor/live`;
  },
};
