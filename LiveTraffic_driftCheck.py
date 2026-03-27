"""Capture live traffic, build flow-based records, and align with UNSW-NB15 schema.

Run with admin rights so Scapy can sniff. Example:
	python LiveTraffic_driftCheck.py --iface Ethernet --seconds 60 --output live_flows.csv --predict
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scapy.all import IP, TCP, UDP, Raw, sniff

# Columns expected by the trained pipeline (from preprocessing_pipeline_full.joblib)
TARGET_COLUMNS: List[str] = [
	"srcip",
	"sport",
	"dstip",
	"dsport",
	"proto",
	"state",
	"dur",
	"sbytes",
	"dbytes",
	"sttl",
	"dttl",
	"sloss",
	"dloss",
	"service",
	"sload",
	"dload",
	"spkts",
	"dpkts",
	"swin",
	"dwin",
	"stcpb",
	"dtcpb",
	"smeansz",
	"dmeansz",
	"trans_depth",
	"res_bdy_len",
	"sjit",
	"djit",
	"stime",
	"ltime",
	"sintpkt",
	"dintpkt",
	"tcprtt",
	"synack",
	"ackdat",
	"is_sm_ips_ports",
	"ct_state_ttl",
	"ct_flw_http_mthd",
	"is_ftp_login",
	"ct_ftp_cmd",
	"ct_srv_src",
	"ct_srv_dst",
	"ct_dst_ltm",
	"ct_src__ltm",
	"ct_src_dport_ltm",
	"ct_dst_sport_ltm",
	"ct_dst_src_ltm",
]

# Minimal port-to-service mapping to approximate UNSW service labels
PORT_SERVICE_MAP = {
	20: "ftp-data",
	21: "ftp",
	22: "ssh",
	23: "telnet",
	25: "smtp",
	53: "dns",
	80: "http",
	110: "pop3",
	123: "ntp",
	143: "imap",
	443: "https",
	445: "smb",
	465: "smtp",
	587: "smtp",
	993: "imap",
	995: "pop3",
	3306: "mysql",
	3389: "rdp",
	8080: "http",
}


def _service_from_ports(port: int, proto: str) -> str:
	if port in PORT_SERVICE_MAP:
		return PORT_SERVICE_MAP[port]
	if proto == "udp" and port == 67:
		return "dhcp"
	return "other"


def _state_from_flags(pkt: TCP) -> str:
	flags = pkt.flags
	if flags & 0x04:
		return "RST"
	if flags & 0x01:
		return "FIN"
	if flags & 0x12:  # SYN+ACK
		return "ACC"
	if flags & 0x02:
		return "CON"
	if flags & 0x10:
		return "EST"
	return "INT"


def _has_http_method(payload: bytes) -> bool:
	http_methods = (b"GET ", b"POST ", b"PUT ", b"DELETE ", b"HEAD ", b"OPTIONS ", b"PATCH ")
	return any(payload.startswith(m) for m in http_methods)


def _is_ftp_login(payload: bytes) -> bool:
	upper = payload.upper()
	return upper.startswith(b"USER") or upper.startswith(b"PASS")


@dataclass
class FlowStats:
	srcip: str
	sport: int
	dstip: str
	dsport: int
	proto: str
	start_ts: float
	end_ts: float
	sbytes: int = 0
	dbytes: int = 0
	spkts: int = 0
	dpkts: int = 0
	sttl: int = 0
	dttl: int = 0
	swin: int = 0
	dwin: int = 0
	stcpb: int = 0
	dtcpb: int = 0
	sjit_samples: List[float] = field(default_factory=list)
	djit_samples: List[float] = field(default_factory=list)
	s_times: List[float] = field(default_factory=list)
	d_times: List[float] = field(default_factory=list)
	state: str = "INT"
	trans_depth: int = 0
	res_bdy_len: int = 0
	syn_ts: Optional[float] = None
	synack: float = 0.0
	ackdat: float = 0.0
	has_http_method: bool = False
	ftp_cmds: int = 0
	ftp_login: bool = False

	def update(self, pkt, direction: str) -> None:
		now = float(pkt.time)
		self.end_ts = now

		if direction == "fwd":
			self.sbytes += int(len(pkt))
			self.spkts += 1
			self.s_times.append(now)
			if IP in pkt:
				self.sttl = int(pkt[IP].ttl)
			if TCP in pkt:
				self.swin = int(pkt[TCP].window)
				if self.stcpb == 0:
					self.stcpb = int(pkt[TCP].seq)
		else:
			self.dbytes += int(len(pkt))
			self.dpkts += 1
			self.d_times.append(now)
			if IP in pkt:
				self.dttl = int(pkt[IP].ttl)
			if TCP in pkt:
				self.dwin = int(pkt[TCP].window)
				if self.dtcpb == 0:
					self.dtcpb = int(pkt[TCP].seq)

		if TCP in pkt:
			self.state = _state_from_flags(pkt[TCP])
			if direction == "fwd" and pkt[TCP].flags & 0x02 and not (pkt[TCP].flags & 0x10):
				self.syn_ts = now
			if direction == "rev" and pkt[TCP].flags & 0x12 and self.syn_ts:
				self.synack = max(now - self.syn_ts, 0.0)
			if direction == "fwd" and pkt[TCP].flags & 0x10 and self.synack > 0 and self.ackdat == 0:
				self.ackdat = max(now - (self.syn_ts or now), 0.0)

		if Raw in pkt:
			payload = bytes(pkt[Raw].load)
			if _has_http_method(payload):
				self.has_http_method = True
			if _is_ftp_login(payload):
				self.ftp_login = True
			if b"FTP" in payload.upper():
				self.ftp_cmds += 1


def _inter_arrival_stats(times: List[float]) -> Tuple[float, float]:
	if len(times) < 2:
		return 0.0, 0.0
	diffs = np.diff(sorted(times))
	return float(np.mean(diffs)), float(np.std(diffs))


def _build_flow_features(flows: Dict[Tuple[str, int, str, int, str], FlowStats]) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []

	# Pre-compute aggregates used by ct_* features
	by_state_ttl: Dict[Tuple[str, int], int] = {}
	by_srv_src: Dict[Tuple[str, str], int] = {}
	by_srv_dst: Dict[Tuple[str, str], int] = {}
	by_dst: Dict[str, int] = {}
	by_src: Dict[str, int] = {}
	by_src_dport: Dict[Tuple[str, int], int] = {}
	by_dst_sport: Dict[Tuple[str, int], int] = {}
	by_pair: Dict[Tuple[str, str], int] = {}

	# First pass to set service and aggregate counts
	flow_list: List[Tuple[Tuple[str, int, str, int, str], FlowStats, str]] = []
	for key, flow in flows.items():
		service = _service_from_ports(flow.dsport, flow.proto)
		flow_list.append((key, flow, service))
		by_state_ttl[(flow.state, flow.sttl)] = by_state_ttl.get((flow.state, flow.sttl), 0) + 1
		by_srv_src[(flow.srcip, service)] = by_srv_src.get((flow.srcip, service), 0) + 1
		by_srv_dst[(flow.dstip, service)] = by_srv_dst.get((flow.dstip, service), 0) + 1
		by_dst[flow.dstip] = by_dst.get(flow.dstip, 0) + 1
		by_src[flow.srcip] = by_src.get(flow.srcip, 0) + 1
		by_src_dport[(flow.srcip, flow.dsport)] = by_src_dport.get((flow.srcip, flow.dsport), 0) + 1
		by_dst_sport[(flow.dstip, flow.sport)] = by_dst_sport.get((flow.dstip, flow.sport), 0) + 1
		by_pair[(flow.srcip, flow.dstip)] = by_pair.get((flow.srcip, flow.dstip), 0) + 1

	for _, flow, service in flow_list:
		dur = max(flow.end_ts - flow.start_ts, 1e-6)
		sintpkt, sjit = _inter_arrival_stats(flow.s_times)
		dintpkt, djit = _inter_arrival_stats(flow.d_times)

		row = {
			"srcip": flow.srcip,
			"sport": flow.sport,
			"dstip": flow.dstip,
			"dsport": flow.dsport,
			"proto": flow.proto,
			"state": flow.state,
			"dur": dur,
			"sbytes": flow.sbytes,
			"dbytes": flow.dbytes,
			"sttl": flow.sttl,
			"dttl": flow.dttl,
			"sloss": 0,
			"dloss": 0,
			"service": service,
			"sload": flow.sbytes / dur if dur > 0 else 0,
			"dload": flow.dbytes / dur if dur > 0 else 0,
			"spkts": flow.spkts,
			"dpkts": flow.dpkts,
			"swin": flow.swin,
			"dwin": flow.dwin,
			"stcpb": flow.stcpb,
			"dtcpb": flow.dtcpb,
			"smeansz": flow.sbytes / flow.spkts if flow.spkts else 0,
			"dmeansz": flow.dbytes / flow.dpkts if flow.dpkts else 0,
			"trans_depth": flow.trans_depth,
			"res_bdy_len": flow.res_bdy_len,
			"sjit": sjit,
			"djit": djit,
			"stime": flow.start_ts,
			"ltime": flow.end_ts,
			"sintpkt": sintpkt,
			"dintpkt": dintpkt,
			"tcprtt": flow.ackdat if flow.ackdat else 0,
			"synack": flow.synack,
			"ackdat": flow.ackdat,
			"is_sm_ips_ports": int(flow.srcip == flow.dstip or flow.sport == flow.dsport),
			"ct_state_ttl": by_state_ttl.get((flow.state, flow.sttl), 1),
			"ct_flw_http_mthd": int(flow.has_http_method),
			"is_ftp_login": int(flow.ftp_login),
			"ct_ftp_cmd": flow.ftp_cmds,
			"ct_srv_src": by_srv_src.get((flow.srcip, service), 1),
			"ct_srv_dst": by_srv_dst.get((flow.dstip, service), 1),
			"ct_dst_ltm": by_dst.get(flow.dstip, 1),
			"ct_src__ltm": by_src.get(flow.srcip, 1),
			"ct_src_dport_ltm": by_src_dport.get((flow.srcip, flow.dsport), 1),
			"ct_dst_sport_ltm": by_dst_sport.get((flow.dstip, flow.sport), 1),
			"ct_dst_src_ltm": by_pair.get((flow.srcip, flow.dstip), 1),
		}
		rows.append(row)

	df = pd.DataFrame(rows)
	if df.empty:
		return df

	# Ensure all expected columns are present
	for col in TARGET_COLUMNS:
		if col not in df.columns:
			df[col] = 0

	df = df[TARGET_COLUMNS]
	# Coerce numeric fields to numeric; leave proto/state/service as strings
	categorical = {"proto", "state", "service"}
	for col in df.columns:
		if col not in categorical:
			df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
	df["proto"] = df["proto"].astype(str)
	df["state"] = df["state"].astype(str)
	df["service"] = df["service"].astype(str)
	return df


def capture_flows(iface: Optional[str], seconds: int, max_packets: int) -> pd.DataFrame:
	flows: Dict[Tuple[str, int, str, int, str], FlowStats] = {}

	def _handle(pkt) -> None:
		if IP not in pkt:
			return
		if TCP in pkt:
			proto = "tcp"
			sport = int(pkt[TCP].sport)
			dsport = int(pkt[TCP].dport)
		elif UDP in pkt:
			proto = "udp"
			sport = int(pkt[UDP].sport)
			dsport = int(pkt[UDP].dport)
		else:
			return

		srcip = pkt[IP].src
		dstip = pkt[IP].dst
		key = (srcip, sport, dstip, dsport, proto)
		rev_key = (dstip, dsport, srcip, sport, proto)

		direction = "fwd"
		if key in flows:
			flow = flows[key]
		elif rev_key in flows:
			flow = flows[rev_key]
			direction = "rev"
		else:
			flow = FlowStats(srcip, sport, dstip, dsport, proto, start_ts=float(pkt.time), end_ts=float(pkt.time))
			flows[key] = flow

		flow.update(pkt, direction)

	sniff(iface=iface, prn=_handle, store=False, count=max_packets if max_packets > 0 else 0, timeout=seconds)
	return _build_flow_features(flows)


def run_prediction(df: pd.DataFrame, pipeline_path: Path, model_path: Path) -> pd.DataFrame:
	pipeline = joblib.load(pipeline_path)
	model_blob = joblib.load(model_path)
	if isinstance(model_blob, dict) and "model" in model_blob:
		model = model_blob["model"]
	else:
		model = model_blob

	X_proc = pipeline.transform(df)
	preds = model.predict(X_proc)
	df_out = df.copy()
	df_out["prediction"] = preds
	return df_out


def main() -> None:
	parser = argparse.ArgumentParser(description="Live traffic capture -> UNSW-NB15 style flows")
	parser.add_argument("--iface", help="Interface name (omit for Scapy default)")
	parser.add_argument("--seconds", type=int, default=60, help="Capture duration")
	parser.add_argument("--max-packets", type=int, default=0, help="Optional hard packet cap")
	parser.add_argument("--output", type=Path, default=Path("processed_data_full/live_flows.csv"))
	parser.add_argument("--predict", action="store_true", help="Run pipeline+model prediction")
	parser.add_argument(
		"--pipeline-path",
		type=Path,
		default=Path("processed_data_full/preprocessing_pipeline_full.joblib"),
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=Path("models/saved/random_forest_full_pipeline.joblib"),
	)

	args = parser.parse_args()

	print(f"Capturing iface={args.iface or 'default'} for {args.seconds}s (max_packets={args.max_packets or 'unbounded'})")
	df = capture_flows(args.iface, args.seconds, args.max_packets)
	if df.empty:
		print("No flows captured")
		return

	args.output.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(args.output, index=False)
	print(f"Captured {len(df)} flows -> {args.output}")

	if args.predict:
		df_pred = run_prediction(df, args.pipeline_path, args.model_path)
		pred_path = args.output.with_name(args.output.stem + "_pred.csv")
		df_pred.to_csv(pred_path, index=False)
		counts = df_pred["prediction"].value_counts().to_dict()
		print(f"Predictions saved to {pred_path}; class counts: {counts}")


if __name__ == "__main__":
	main()
