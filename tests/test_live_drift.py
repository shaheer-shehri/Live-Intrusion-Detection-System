import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

import LiveTraffic_driftCheck as live_drift


def _make_frame(rows: int, service: str, sbytes_start: int, sttl: int) -> pd.DataFrame:
	data = {
		"srcip": ["10.0.0.1"] * rows,
		"sport": list(range(40000, 40000 + rows)),
		"dstip": ["10.0.0.2"] * rows,
		"dsport": [80] * rows,
		"proto": ["tcp"] * rows,
		"state": ["EST"] * rows,
		"dur": [1.0 + (idx % 5) * 0.1 for idx in range(rows)],
		"sbytes": [sbytes_start + idx for idx in range(rows)],
		"dbytes": [200 + idx for idx in range(rows)],
		"sttl": [sttl] * rows,
		"dttl": [64] * rows,
		"sloss": [0] * rows,
		"dloss": [0] * rows,
		"service": [service] * rows,
		"sload": [10.0 + idx for idx in range(rows)],
		"dload": [20.0 + idx for idx in range(rows)],
		"spkts": [5 + (idx % 3) for idx in range(rows)],
		"dpkts": [5 + (idx % 2) for idx in range(rows)],
		"swin": [2000] * rows,
		"dwin": [3000] * rows,
		"stcpb": [1000 + idx for idx in range(rows)],
		"dtcpb": [2000 + idx for idx in range(rows)],
		"smeansz": [50.0 + idx * 0.1 for idx in range(rows)],
		"dmeansz": [60.0 + idx * 0.1 for idx in range(rows)],
		"trans_depth": [1] * rows,
		"res_bdy_len": [0] * rows,
		"sjit": [0.1 + idx * 0.01 for idx in range(rows)],
		"djit": [0.2 + idx * 0.01 for idx in range(rows)],
		"stime": [1000.0 + idx for idx in range(rows)],
		"ltime": [1001.0 + idx for idx in range(rows)],
		"sintpkt": [0.05 + idx * 0.001 for idx in range(rows)],
		"dintpkt": [0.06 + idx * 0.001 for idx in range(rows)],
		"tcprtt": [0.01] * rows,
		"synack": [0.02] * rows,
		"ackdat": [0.03] * rows,
		"is_sm_ips_ports": [0] * rows,
		"ct_state_ttl": [1] * rows,
		"ct_flw_http_mthd": [0] * rows,
		"is_ftp_login": [0] * rows,
		"ct_ftp_cmd": [0] * rows,
		"ct_srv_src": [1] * rows,
		"ct_srv_dst": [1] * rows,
		"ct_dst_ltm": [1] * rows,
		"ct_src__ltm": [1] * rows,
		"ct_src_dport_ltm": [1] * rows,
		"ct_dst_sport_ltm": [1] * rows,
		"ct_dst_src_ltm": [1] * rows,
	}
	return pd.DataFrame(data)


def test_compute_drift_report_flags_shifted_features(tmp_path):
	baseline = _make_frame(50, "http", 100, 64)
	live = _make_frame(50, "ssh", 500, 32)

	report = live_drift.compute_drift_report(live, baseline)

	assert not report.empty
	assert "psi" in report.columns
	assert (report["status"] == "drift").any()
	assert report.iloc[0]["feature"] in {"sbytes", "service", "sttl"}

	figure_path = tmp_path / "drift.png"
	live_drift.save_drift_plot(report, figure_path)
	assert figure_path.exists()


def test_run_drift_check_writes_report_and_figure(tmp_path):
	baseline = _make_frame(30, "http", 100, 64)
	live = _make_frame(30, "dns", 900, 16)
	baseline_path = tmp_path / "baseline.csv"
	report_path = tmp_path / "report.csv"
	figure_path = tmp_path / "report.png"
	baseline.to_csv(baseline_path, index=False)

	report = live_drift.run_drift_check(live, baseline_path, report_path, figure_path)

	assert not report.empty
	assert report_path.exists()
	assert figure_path.exists()
	loaded = pd.read_csv(report_path)
	assert len(loaded) == len(report)