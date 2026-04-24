"""Capture live traffic, build flow-based records, and align with UNSW-NB15 schema.

Run with admin rights so Scapy can sniff. Example:
	python LiveTraffic_driftCheck.py --iface Ethernet --seconds 60 --output live_flows.csv --predict --drift
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monitoring import drift as drift_monitoring

try:
	from scapy.all import IP, TCP, UDP, Raw, sniff
except ModuleNotFoundError:
	IP = TCP = UDP = Raw = sniff = None

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

DRIFT_CATEGORICAL_COLUMNS: List[str] = ["proto", "state", "service"]

DEFAULT_BASELINE_PATH = drift_monitoring.DEFAULT_BASELINE_PATH
DEFAULT_PIPELINE_PATH = drift_monitoring.DEFAULT_PIPELINE_PATH
DEFAULT_MODEL_PATH = drift_monitoring.DEFAULT_MODEL_PATH
DEFAULT_HISTORY_PATH = drift_monitoring.DEFAULT_ENGINEERED_DRIFT_HISTORY_PATH

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


def _state_from_flags(pkt) -> str:
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


def _load_model_blob(model_path: Path):
	blob = joblib.load(model_path)
	if isinstance(blob, dict) and "model" in blob:
		return blob["model"]
	return blob


def _clean_series(series: pd.Series) -> pd.Series:
	values = pd.to_numeric(series, errors="coerce")
	values = values.replace([np.inf, -np.inf], np.nan).dropna()
	return values.astype(float)


def _psi_from_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
	expected = _clean_series(expected)
	actual = _clean_series(actual)
	if expected.empty or actual.empty:
		return 0.0
	if expected.nunique(dropna=True) <= 1:
		return 0.0 if actual.nunique(dropna=True) <= 1 else 1.0

	quantiles = np.unique(np.quantile(expected.to_numpy(), np.linspace(0, 1, bins + 1)))
	if len(quantiles) < 2:
		return 0.0

	if len(quantiles) == 2:
		quantiles = np.array([quantiles[0], quantiles[1]])

	expected_bins = pd.cut(expected, bins=quantiles, include_lowest=True, duplicates="drop")
	actual_bins = pd.cut(actual, bins=quantiles, include_lowest=True, duplicates="drop")
	expected_counts = expected_bins.value_counts(normalize=True, sort=False)
	actual_counts = actual_bins.value_counts(normalize=True, sort=False)
	all_bins = expected_counts.index.union(actual_counts.index)
	expected_counts = expected_counts.reindex(all_bins, fill_value=0.0)
	actual_counts = actual_counts.reindex(all_bins, fill_value=0.0)

	epsilon = 1e-6
	psi = ((actual_counts + epsilon) - (expected_counts + epsilon)) * np.log((actual_counts + epsilon) / (expected_counts + epsilon))
	return float(psi.sum())


def _psi_from_categorical(expected: pd.Series, actual: pd.Series) -> float:
	expected = expected.astype(str).replace({"nan": np.nan}).dropna()
	actual = actual.astype(str).replace({"nan": np.nan}).dropna()
	if expected.empty or actual.empty:
		return 0.0

	categories = sorted(set(expected.unique()).union(set(actual.unique())))
	if not categories:
		return 0.0

	expected_counts = expected.value_counts(normalize=True).reindex(categories, fill_value=0.0)
	actual_counts = actual.value_counts(normalize=True).reindex(categories, fill_value=0.0)
	epsilon = 1e-6
	psi = ((actual_counts + epsilon) - (expected_counts + epsilon)) * np.log((actual_counts + epsilon) / (expected_counts + epsilon))
	return float(psi.sum())


def compute_drift_report(
	live_df: pd.DataFrame,
	baseline_df: pd.DataFrame,
	bins: int = 10,
) -> pd.DataFrame:
	assessment = drift_monitoring.build_drift_summary(live_df, baseline_df)
	rows = [result.to_dict() for result in assessment.features]
	for row in rows:
		row.setdefault("psi", row.get("psi", 0.0))
		row["status"] = "stable" if row.get("severity", "stable") == "stable" else "drift"
		row.setdefault("mean_delta", None)
	return pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)


def save_drift_plot(report: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
	if report.empty:
		return

	features: List[drift_monitoring.FeatureDriftResult] = []
	for row in report.to_dict(orient="records"):
		features.append(
			drift_monitoring.FeatureDriftResult(
				feature=str(row.get("feature", "unknown")),
				feature_type=str(row.get("feature_type", "numeric")),
				drift_score=float(row.get("drift_score", row.get("risk_score", 0.0)) or 0.0),
				risk_score=float(row.get("risk_score", row.get("drift_score", 0.0)) or 0.0),
				severity=str(row.get("severity", row.get("status", "stable"))),
				importance=float(row.get("importance", 0.0) or 0.0),
				ks_stat=float(row.get("ks_stat", 0.0) or 0.0),
				ks_pvalue=float(row.get("ks_pvalue", 1.0) or 1.0),
				js_divergence=float(row.get("js_divergence", 0.0) or 0.0),
				chi2_stat=float(row.get("chi2_stat", 0.0) or 0.0),
				chi2_pvalue=float(row.get("chi2_pvalue", 1.0) or 1.0),
				psi=float(row.get("psi", 0.0) or 0.0),
				unseen_rate=float(row.get("unseen_rate", 0.0) or 0.0),
				baseline_mean=row.get("baseline_mean"),
				live_mean=row.get("live_mean"),
			)
		)
	assessment = drift_monitoring.DriftAssessment(
		timestamp=drift_monitoring._utc_now(),
		batch_size=int(report.shape[0]),
		feature_count=int(report.shape[0]),
		drift_count=int((report.get("severity", report.get("status")) != "stable").sum()) if not report.empty else 0,
		severe_count=int((report.get("severity", report.get("status")) == "severe").sum()) if not report.empty else 0,
		drift_fraction=float((report.get("severity", report.get("status")) != "stable").mean()) if not report.empty else 0.0,
		severe_fraction=float((report.get("severity", report.get("status")) == "severe").mean()) if not report.empty else 0.0,
		overall_score=float(report.get("risk_score", pd.Series([0.0])).mean()),
		overall_severity=str(report.get("severity", pd.Series(["stable"])).iloc[0]),
		alert=bool((report.get("severity", report.get("status")) == "severe").any()),
		alert_level="warning",
		alert_reason="compatibility plot generation",
		thresholds={"low": 0.2, "moderate": 0.4, "severe": 0.6},
		trend="stable",
		trend_delta=0.0,
		trend_slope=0.0,
		history_count=0,
		features=features,
	)
	drift_monitoring.save_drift_plot(assessment, output_path, top_n=top_n)


def run_drift_check(df: pd.DataFrame, baseline_path: Path, report_path: Path, figure_path: Path, bins: int = 10) -> pd.DataFrame:
	baseline = pd.read_csv(baseline_path, low_memory=False)
	assessment = drift_monitoring.build_drift_summary(df, baseline)
	report = pd.DataFrame([result.to_dict() for result in assessment.features]).sort_values("risk_score", ascending=False).reset_index(drop=True)
	if not report.empty:
		report["status"] = np.where(report["severity"] == "stable", "stable", "drift")
	report_path.parent.mkdir(parents=True, exist_ok=True)
	report.to_csv(report_path, index=False)
	drift_monitoring.save_drift_plot(assessment, figure_path)
	return report


def capture_flows(iface: Optional[str], seconds: int, max_packets: int) -> pd.DataFrame:
	if sniff is None or IP is None or TCP is None or UDP is None or Raw is None:
		raise RuntimeError("Scapy is required for live packet capture. Install the project dependencies before using capture mode.")

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
	model = _load_model_blob(model_path)

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
		"--drift",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Run drift detection against the saved training distribution",
	)
	parser.add_argument(
		"--pipeline-path",
		type=Path,
		default=DEFAULT_PIPELINE_PATH,
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=DEFAULT_MODEL_PATH,
	)
	parser.add_argument(
		"--baseline-path",
		type=Path,
		default=DEFAULT_BASELINE_PATH,
		help="Baseline raw training split used for live drift comparison",
	)
	parser.add_argument(
		"--drift-report",
		type=Path,
		default=Path("processed_data_full/live_drift_report.csv"),
		help="CSV file containing per-feature drift scores",
	)
	parser.add_argument(
		"--drift-figure",
		type=Path,
		default=Path("processed_data_full/live_drift_report.png"),
		help="PNG summary of the drift scores",
	)
	parser.add_argument(
		"--dashboard-json",
		type=Path,
		default=Path("processed_data_full/live_drift_dashboard.json"),
		help="Dashboard-ready JSON summary",
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

	if args.drift:
		if not args.baseline_path.exists():
			print(f"Drift check skipped: baseline not found at {args.baseline_path}")
		else:
			model_blob = joblib.load(args.model_path)
			model_for_drift = model_blob["model"] if isinstance(model_blob, dict) and "model" in model_blob else model_blob
			pipeline_for_drift = joblib.load(args.pipeline_path)
			assessment = drift_monitoring.build_model_input_drift_summary(
				live_df=df,
				baseline_df=args.baseline_path,
				pipeline=pipeline_for_drift,
				model=model_for_drift,
				history_path=DEFAULT_HISTORY_PATH,
				top_n=12,
				baseline_sample_size=2500,
				store_history=True,
			)
			report = pd.DataFrame([result.to_dict() for result in assessment.features]).sort_values("risk_score", ascending=False).reset_index(drop=True)
			report.to_csv(args.drift_report, index=False)
			drift_monitoring.save_drift_plot(assessment, args.drift_figure)
			args.dashboard_json.parent.mkdir(parents=True, exist_ok=True)
			args.dashboard_json.write_text(json.dumps(assessment.dashboard, indent=2), encoding="utf-8")
			drift_count = assessment.drift_count
			watch_count = int(sum(result.severity == "moderate" for result in assessment.features))
			overall_psi = assessment.overall_score
			print(
				f"Drift report saved to {args.drift_report}; figure saved to {args.drift_figure}; "
				f"overall score={overall_psi:.4f}, drift features={drift_count}, watch features={watch_count}, dashboard={args.dashboard_json}"
			)


if __name__ == "__main__":
	main()
