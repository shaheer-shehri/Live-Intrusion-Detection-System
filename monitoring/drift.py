from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

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

DRIFT_FEATURE_COLUMNS: List[str] = TARGET_COLUMNS.copy()
DRIFT_CATEGORICAL_COLUMNS = {"proto", "state", "service"}
RAW_NUMERIC_COLUMNS = [col for col in TARGET_COLUMNS if col not in DRIFT_CATEGORICAL_COLUMNS]
DEFAULT_BASELINE_PATH = Path("processed_data_mc/data_splits/X_train_raw.csv")
DEFAULT_PIPELINE_PATH = Path("processed_data_mc/combined_preprocessing_pipeline.joblib")
DEFAULT_MODEL_PATH = Path("models/saved/improved_xgboost_mc.joblib")
DEFAULT_DRIFT_HISTORY_PATH = Path("processed_data_full/drift_history.jsonl")
DEFAULT_ENGINEERED_DRIFT_HISTORY_PATH = Path("processed_data_full/model_input_drift_history.jsonl")

DERIVED_FEATURE_SOURCES: Dict[str, List[str]] = {
	"duration": ["stime", "ltime"],
	"packet_rate": ["dpkts", "stime", "ltime"],
	"byte_ratio": ["sbytes", "dbytes"],
}


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _as_frame(data: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
	if isinstance(data, pd.DataFrame):
		return data.copy()
	return pd.DataFrame(list(data))


def _prepare_numeric_frame(data: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
	frame = _as_frame(data)
	frame.columns = frame.columns.astype(str).str.strip().str.lower()
	for column in frame.columns:
		frame[column] = pd.to_numeric(frame[column], errors="coerce")
	return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_text(value: Any) -> str:
	if value is None or (isinstance(value, float) and math.isnan(value)):
		return ""
	return str(value).strip().lower()


def _clean_series(series: pd.Series) -> pd.Series:
	cleaned = pd.to_numeric(series, errors="coerce")
	cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
	return cleaned.astype(float)


def _safe_probabilities(values: np.ndarray) -> np.ndarray:
	values = np.asarray(values, dtype=float)
	values = np.clip(values, 0.0, None)
	total = float(values.sum())
	if total <= 0:
		return np.full(len(values), 1.0 / max(len(values), 1), dtype=float)
	return values / total


def _adaptive_bin_edges(expected: pd.Series, actual: pd.Series, max_bins: int = 15) -> np.ndarray:
	combined = pd.concat([_clean_series(expected), _clean_series(actual)], ignore_index=True)
	combined = combined.dropna()
	if combined.empty:
		return np.asarray([0.0, 1.0], dtype=float)

	unique_values = np.unique(combined.to_numpy())
	if len(unique_values) <= 2:
		left = float(unique_values.min()) - 0.5 if len(unique_values) else 0.0
		right = float(unique_values.max()) + 0.5 if len(unique_values) else 1.0
		return np.asarray([left, right], dtype=float)

	try:
		edges = np.histogram_bin_edges(combined.to_numpy(), bins="fd")
	except ValueError:
		edges = np.histogram_bin_edges(combined.to_numpy(), bins="sturges")

	if len(edges) < 3:
		quantiles = np.unique(np.quantile(combined.to_numpy(), np.linspace(0.0, 1.0, min(max_bins, len(unique_values)) + 1)))
		if len(quantiles) < 2:
			return np.asarray([float(unique_values.min()) - 0.5, float(unique_values.max()) + 0.5], dtype=float)
		return quantiles

	if len(edges) > max_bins + 1:
		quantiles = np.unique(np.quantile(combined.to_numpy(), np.linspace(0.0, 1.0, max_bins + 1)))
		if len(quantiles) >= 2:
			return quantiles

	return edges


def _histogram_distribution(series: pd.Series, edges: np.ndarray) -> np.ndarray:
	cleaned = _clean_series(series)
	if cleaned.empty:
		return np.asarray([1.0], dtype=float)
	counts, _ = np.histogram(cleaned.to_numpy(), bins=edges)
	return _safe_probabilities(counts)


def _js_divergence(expected: np.ndarray, actual: np.ndarray) -> float:
	return float(jensenshannon(expected, actual, base=2.0) ** 2)


def _psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-6) -> float:
	return float(np.sum((actual - expected) * np.log((actual + epsilon) / (expected + epsilon))))


def _chi_square_stat(expected_counts: pd.Series, actual_counts: pd.Series) -> Tuple[float, float]:
	frame = pd.DataFrame({"baseline": expected_counts, "live": actual_counts}).fillna(0.0)
	if frame.shape[0] < 2 or frame.to_numpy().sum() <= 0:
		return 0.0, 1.0
	try:
		statistic, pvalue, _, _ = chi2_contingency(frame.to_numpy().T, correction=False)
		return float(statistic), float(pvalue)
	except ValueError:
		return 0.0, 1.0


def _ks_stat(expected: pd.Series, actual: pd.Series) -> Tuple[float, float]:
	base = _clean_series(expected)
	live = _clean_series(actual)
	if base.empty or live.empty:
		return 0.0, 1.0
	try:
		result = ks_2samp(base.to_numpy(), live.to_numpy(), alternative="two-sided", mode="auto")
		return float(result.statistic), float(result.pvalue)
	except ValueError:
		return 0.0, 1.0


def _feature_type(feature: str) -> str:
	return "categorical" if feature in DRIFT_CATEGORICAL_COLUMNS else "numeric"


def _categorical_prepare(series: pd.Series) -> pd.Series:
	return series.map(_normalize_text)


def _categorical_with_unseen(baseline: pd.Series, live: pd.Series) -> Tuple[pd.Series, pd.Series, float]:
	base = _categorical_prepare(baseline)
	live_norm = _categorical_prepare(live)
	baseline_categories = set(base.dropna().tolist())
	unseen_mask = ~live_norm.isin(baseline_categories)
	if unseen_mask.any():
		live_norm = live_norm.where(~unseen_mask, other="__unseen__")
	unseen_rate = float(unseen_mask.mean()) if len(live_norm) else 0.0
	return base, live_norm, unseen_rate


def _categorical_distribution(baseline: pd.Series, live: pd.Series) -> Tuple[pd.Series, pd.Series, float, float, float]:
	base, live_norm, unseen_rate = _categorical_with_unseen(baseline, live)
	categories = sorted(set(base.dropna().unique()).union(set(live_norm.dropna().unique())))
	if not categories:
		return pd.Series(dtype=float), pd.Series(dtype=float), unseen_rate, 0.0, 1.0
	base_counts = base.value_counts(normalize=True).reindex(categories, fill_value=0.0)
	live_counts = live_norm.value_counts(normalize=True).reindex(categories, fill_value=0.0)
	chi2_stat, chi2_pvalue = _chi_square_stat(base.value_counts(), live_norm.value_counts())
	return base_counts, live_counts, unseen_rate, chi2_stat, chi2_pvalue


def _severity_from_score(score: float, thresholds: Mapping[str, float]) -> str:
	if score >= thresholds["severe"]:
		return "severe"
	if score >= thresholds["moderate"]:
		return "moderate"
	if score >= thresholds["low"]:
		return "low"
	return "stable"


def _severity_rank(severity: str) -> int:
	order = {"stable": 0, "low": 1, "moderate": 2, "severe": 3}
	return order.get(severity, 0)


def _blend_thresholds(current: Sequence[float], history: Sequence[float], defaults: Tuple[float, float, float]) -> Dict[str, float]:
	current_values = np.asarray(list(current), dtype=float)
	history_values = np.asarray(list(history), dtype=float)

	if current_values.size == 0:
		return {"low": defaults[0], "moderate": defaults[1], "severe": defaults[2]}

	if history_values.size:
		hist = np.quantile(history_values, [0.60, 0.85, 0.95])
		blended = 0.85 * np.asarray(defaults, dtype=float) + 0.15 * hist
	else:
		blended = np.asarray(defaults, dtype=float)

	low = float(max(defaults[0], blended[0]))
	moderate = float(max(low + 0.05, blended[1]))
	severe = float(max(moderate + 0.05, blended[2]))
	return {"low": low, "moderate": moderate, "severe": severe}


def _trend_label(history: Sequence[Mapping[str, Any]], current_score: float) -> Tuple[str, float, float]:
	if not history:
		return "new", 0.0, 0.0

	previous_scores = [float(item.get("overall_score", 0.0)) for item in history if item.get("overall_score") is not None]
	if not previous_scores:
		return "new", 0.0, 0.0

	last_score = previous_scores[-1]
	delta = current_score - last_score
	recent = previous_scores[-4:] + [current_score]
	slope = 0.0
	if len(recent) >= 2:
		x = np.arange(len(recent), dtype=float)
		slope = float(np.polyfit(x, np.asarray(recent, dtype=float), 1)[0])

	if delta >= 0.18 and current_score >= 0.35:
		return "sudden", delta, slope
	if slope >= 0.04 and current_score >= 0.25:
		return "gradual", delta, slope
	return "stable", delta, slope


@dataclass
class FeatureDriftResult:
	feature: str
	feature_type: str
	drift_score: float
	risk_score: float
	severity: str
	importance: float
	ks_stat: float = 0.0
	ks_pvalue: float = 1.0
	js_divergence: float = 0.0
	chi2_stat: float = 0.0
	chi2_pvalue: float = 1.0
	psi: float = 0.0
	unseen_rate: float = 0.0
	baseline_mean: Optional[float] = None
	live_mean: Optional[float] = None
	baseline_unique: int = 0
	live_unique: int = 0
	alert: bool = False
	note: str = ""

	def to_dict(self) -> Dict[str, Any]:
		data = asdict(self)
		data["status"] = self.severity
		for key, value in list(data.items()):
			if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
				data[key] = None
		return data


@dataclass
class DriftAssessment:
	timestamp: str
	batch_size: int
	feature_count: int
	drift_count: int
	severe_count: int
	drift_fraction: float
	severe_fraction: float
	overall_score: float
	overall_severity: str
	alert: bool
	alert_level: str
	alert_reason: str
	thresholds: Dict[str, float]
	trend: str
	trend_delta: float
	trend_slope: float
	history_count: int
	features: List[FeatureDriftResult] = field(default_factory=list)
	history: List[Dict[str, Any]] = field(default_factory=list)
	dashboard: Dict[str, Any] = field(default_factory=dict)

	def top_features(self, limit: int = 10) -> List[Dict[str, Any]]:
		ordered = sorted(self.features, key=lambda item: item.risk_score, reverse=True)
		return [item.to_dict() for item in ordered[:limit]]

	def to_dict(self, top_n: int = 10) -> Dict[str, Any]:
		return {
			"timestamp": self.timestamp,
			"batch_size": self.batch_size,
			"feature_count": self.feature_count,
			"drift_count": self.drift_count,
			"severe_count": self.severe_count,
			"drift_fraction": self.drift_fraction,
			"severe_fraction": self.severe_fraction,
			"overall_score": self.overall_score,
			"overall_severity": self.overall_severity,
			"alert": self.alert,
			"alert_level": self.alert_level,
			"alert_reason": self.alert_reason,
			"thresholds": self.thresholds,
			"trend": self.trend,
			"trend_delta": self.trend_delta,
			"trend_slope": self.trend_slope,
			"history_count": self.history_count,
			"top_features": self.top_features(limit=top_n),
			"history": self.history,
			"dashboard": self.dashboard,
		}


class DriftHistoryStore:
	def __init__(self, path: Path):
		self.path = Path(path)

	def load(self, limit: int = 20) -> List[Dict[str, Any]]:
		if not self.path.exists():
			return []
		entries: List[Dict[str, Any]] = []
		with self.path.open("r", encoding="utf-8") as handle:
			for line in handle:
				line = line.strip()
				if not line:
					continue
				try:
					entries.append(json.loads(line))
				except json.JSONDecodeError:
					continue
		return entries[-limit:]

	def append(self, record: Mapping[str, Any]) -> None:
		self.path.parent.mkdir(parents=True, exist_ok=True)
		with self.path.open("a", encoding="utf-8") as handle:
			handle.write(json.dumps(record, sort_keys=True) + "\n")

	def last(self) -> Optional[Dict[str, Any]]:
		entries = self.load(limit=1)
		return entries[-1] if entries else None


class DriftMonitor:
	def __init__(
		self,
		model: Any,
		pipeline: Any,
		baseline_frame: pd.DataFrame,
		history_path: Path = DEFAULT_DRIFT_HISTORY_PATH,
		baseline_sample_size: int = 2500,
		importance_map: Optional[Dict[str, float]] = None,
	):
		self.model = model
		self.pipeline = pipeline
		self.baseline_frame = self.prepare_frame(baseline_frame)
		self.history = DriftHistoryStore(history_path)
		self.baseline_sample_size = baseline_sample_size
		self.importance_map = importance_map or self._build_importance_map()

	@classmethod
	def from_artifacts(
		cls,
		model: Any,
		pipeline: Any,
		baseline_path: Path = DEFAULT_BASELINE_PATH,
		history_path: Path = DEFAULT_DRIFT_HISTORY_PATH,
		baseline_sample_size: int = 2500,
		importance_map: Optional[Dict[str, float]] = None,
	) -> "DriftMonitor":
		baseline = pd.read_csv(baseline_path, low_memory=False)
		return cls(
			model=model,
			pipeline=pipeline,
			baseline_frame=baseline,
			history_path=history_path,
			baseline_sample_size=baseline_sample_size,
			importance_map=importance_map,
		)

	def prepare_frame(self, frame: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
		prepared = _as_frame(frame)
		prepared.columns = prepared.columns.str.strip().str.lower()
		if "target" in prepared.columns:
			prepared = prepared.drop(columns=["target"])
		for column in DRIFT_CATEGORICAL_COLUMNS:
			if column in prepared.columns:
				prepared[column] = prepared[column].map(_normalize_text)
		for column in TARGET_COLUMNS:
			if column in prepared.columns and column not in DRIFT_CATEGORICAL_COLUMNS:
				prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
		return prepared[[column for column in TARGET_COLUMNS if column in prepared.columns]].copy()

	def _transform_for_importance(self) -> pd.DataFrame:
		sample = self.baseline_frame.head(self.baseline_sample_size).copy()
		if sample.empty:
			return pd.DataFrame()
		transformed = self.pipeline.transform(sample)
		if isinstance(transformed, pd.DataFrame):
			return transformed
		feature_names = getattr(self.pipeline, "get_feature_names_out", None)
		if callable(feature_names):
			columns = list(feature_names())
		else:
			columns = [f"feature_{idx}" for idx in range(np.asarray(transformed).shape[1])]
		return pd.DataFrame(np.asarray(transformed), columns=columns, index=sample.index)

	def _source_features_for_transformed_name(self, name: str) -> List[str]:
		encoder = getattr(self.pipeline, "encoder", None)
		if encoder is not None:
			for column in getattr(encoder, "onehot_columns_", []):
				if name.startswith(f"{column}_"):
					return [column]
		if name in DERIVED_FEATURE_SOURCES:
			return DERIVED_FEATURE_SOURCES[name]
		if name in TARGET_COLUMNS:
			return [name]
		if "__" in name:
			prefix = name.split("__", 1)[0]
			if prefix in TARGET_COLUMNS:
				return [prefix]
		if "_" in name:
			prefix = name.split("_", 1)[0]
			if prefix in TARGET_COLUMNS:
				return [prefix]
		return []

	def _build_importance_map(self) -> Dict[str, float]:
		if not hasattr(self.model, "feature_importances_") or not hasattr(self.pipeline, "transform"):
			return {column: 1.0 for column in TARGET_COLUMNS if column in self.baseline_frame.columns}

		transformed = self._transform_for_importance()
		if transformed.empty:
			return {column: 1.0 for column in TARGET_COLUMNS if column in self.baseline_frame.columns}

		importances = np.asarray(getattr(self.model, "feature_importances_"), dtype=float)
		if len(importances) != transformed.shape[1]:
			return {column: 1.0 for column in TARGET_COLUMNS if column in self.baseline_frame.columns}

		feature_scores = pd.Series(importances, index=list(transformed.columns), dtype=float)
		collapsed: Dict[str, float] = {column: 0.0 for column in TARGET_COLUMNS}
		for transformed_name, score in feature_scores.items():
			sources = self._source_features_for_transformed_name(str(transformed_name))
			if not sources:
				continue
			share = float(score) / max(len(sources), 1)
			for source in sources:
				if source in collapsed:
					collapsed[source] += share

		total = sum(collapsed.values())
		if total <= 0:
			return {column: 1.0 for column in TARGET_COLUMNS if column in self.baseline_frame.columns}
		return {column: value / total for column, value in collapsed.items() if column in self.baseline_frame.columns}

	def _feature_importance(self, feature: str) -> float:
		return float(self.importance_map.get(feature, 0.0))

	def _numeric_result(self, feature: str, baseline: pd.Series, live: pd.Series) -> FeatureDriftResult:
		base = _clean_series(baseline)
		live_clean = _clean_series(live)
		if base.empty or live_clean.empty:
			return FeatureDriftResult(feature=feature, feature_type="numeric", drift_score=0.0, risk_score=0.0, severity="stable", importance=self._feature_importance(feature))

		ks_stat, ks_pvalue = _ks_stat(base, live_clean)
		edges = _adaptive_bin_edges(base, live_clean)
		base_dist = _histogram_distribution(base, edges)
		live_dist = _histogram_distribution(live_clean, edges)
		js_div = _js_divergence(base_dist, live_dist)
		base_hist, _ = np.histogram(base, bins=edges)
		live_hist, _ = np.histogram(live_clean, bins=edges)
		psi_value = _psi(_safe_probabilities(base_hist), _safe_probabilities(live_hist))
		psi_norm = min(abs(psi_value) / 0.25, 1.0)
		mean_delta = abs(float(live_clean.mean()) - float(base.mean()))
		std_base = float(base.std(ddof=0))
		mean_shift = 0.0 if std_base <= 1e-9 else min(mean_delta / (std_base + 1e-9), 3.0) / 3.0
		drift_score = float(np.clip(0.35 * ks_stat + 0.25 * psi_norm + 0.25 * js_div + 0.15 * mean_shift, 0.0, 1.0))
		return FeatureDriftResult(
			feature=feature,
			feature_type="numeric",
			drift_score=drift_score,
			risk_score=drift_score,
			severity="stable",
			importance=self._feature_importance(feature),
			ks_stat=ks_stat,
			ks_pvalue=ks_pvalue,
			js_divergence=js_div,
			psi=psi_value,
			baseline_mean=float(base.mean()),
			live_mean=float(live_clean.mean()),
			baseline_unique=int(base.nunique()),
			live_unique=int(live_clean.nunique()),
		)

	def _categorical_result(self, feature: str, baseline: pd.Series, live: pd.Series) -> FeatureDriftResult:
		base_dist, live_dist, unseen_rate, chi2_stat, chi2_pvalue = _categorical_distribution(baseline, live)
		if base_dist.empty or live_dist.empty:
			return FeatureDriftResult(feature=feature, feature_type="categorical", drift_score=0.0, risk_score=0.0, severity="stable", importance=self._feature_importance(feature))

		js_div = _js_divergence(base_dist.to_numpy(), live_dist.to_numpy())
		chi_strength = 1.0 - min(chi2_pvalue, 1.0)
		drift_score = float(np.clip(0.45 * js_div + 0.35 * chi_strength + 0.20 * unseen_rate, 0.0, 1.0))
		baseline_mean = None
		live_mean = None
		return FeatureDriftResult(
			feature=feature,
			feature_type="categorical",
			drift_score=drift_score,
			risk_score=drift_score,
			severity="stable",
			importance=self._feature_importance(feature),
			js_divergence=js_div,
			chi2_stat=chi2_stat,
			chi2_pvalue=chi2_pvalue,
			unseen_rate=unseen_rate,
			baseline_mean=baseline_mean,
			live_mean=live_mean,
			baseline_unique=int(_categorical_prepare(baseline).nunique()),
			live_unique=int(_categorical_prepare(live).nunique()),
			note="Unseen categories are tracked explicitly in the `__unseen__` bucket.",
		)

	def compare_features(self, live_frame: pd.DataFrame | Sequence[Mapping[str, Any]]) -> List[FeatureDriftResult]:
		live = self.prepare_frame(live_frame)
		results: List[FeatureDriftResult] = []
		for feature in [column for column in TARGET_COLUMNS if column in live.columns and column in self.baseline_frame.columns]:
			baseline = self.baseline_frame[feature]
			live_series = live[feature]
			if feature in DRIFT_CATEGORICAL_COLUMNS:
				result = self._categorical_result(feature, baseline, live_series)
			else:
				result = self._numeric_result(feature, baseline, live_series)
			results.append(result)
		return results

	def compare_numeric_features(
		self,
		live_frame: pd.DataFrame | Sequence[Mapping[str, Any]],
		baseline_frame: Optional[pd.DataFrame | Sequence[Mapping[str, Any]]] = None,
	) -> List[FeatureDriftResult]:
		baseline = _prepare_numeric_frame(self.baseline_frame if baseline_frame is None else baseline_frame)
		live = _prepare_numeric_frame(live_frame)
		results: List[FeatureDriftResult] = []
		for feature in [column for column in baseline.columns if column in live.columns]:
			result = self._numeric_result(feature, baseline[feature], live[feature])
			results.append(result)
		return results

	def _severity_thresholds(self, scores: Sequence[float], history_scores: Sequence[float]) -> Dict[str, float]:
		defaults = (0.20, 0.40, 0.60)
		return _blend_thresholds(scores, history_scores, defaults)

	def assess(
		self,
		live_frame: pd.DataFrame | Sequence[Mapping[str, Any]],
		store_history: bool = True,
		top_n: int = 10,
		alert_feature_fraction: float = 0.30,
	) -> DriftAssessment:
		return self._assess_results(
			self.compare_features(live_frame),
			live_frame=live_frame,
			store_history=store_history,
			top_n=top_n,
			alert_feature_fraction=alert_feature_fraction,
		)

	def assess_numeric_features(
		self,
		live_frame: pd.DataFrame | Sequence[Mapping[str, Any]],
		baseline_frame: Optional[pd.DataFrame | Sequence[Mapping[str, Any]]] = None,
		store_history: bool = True,
		top_n: int = 10,
		alert_feature_fraction: float = 0.30,
	) -> DriftAssessment:
		return self._assess_results(
			self.compare_numeric_features(live_frame, baseline_frame=baseline_frame),
			live_frame=live_frame,
			store_history=store_history,
			top_n=top_n,
			alert_feature_fraction=alert_feature_fraction,
		)

	def _assess_results(
		self,
		results: List[FeatureDriftResult],
		live_frame: pd.DataFrame | Sequence[Mapping[str, Any]],
		store_history: bool = True,
		top_n: int = 10,
		alert_feature_fraction: float = 0.30,
	) -> DriftAssessment:
		history = self.history.load(limit=20)
		history_overall_scores = [float(item.get("overall_score", 0.0)) for item in history if item.get("overall_score") is not None]
		feature_scores = [result.drift_score for result in results]
		thresholds = self._severity_thresholds(feature_scores, history_overall_scores)

		for result in results:
			base_risk = result.drift_score
			importance_weight = 0.5 + 0.5 * float(np.clip(result.importance, 0.0, 1.0))
			result.risk_score = float(np.clip(base_risk * importance_weight, 0.0, 1.0))
			result.severity = _severity_from_score(result.drift_score, thresholds)
			result.alert = result.severity == "severe"

		ordered = sorted(results, key=lambda item: item.risk_score, reverse=True)
		drift_count = sum(item.severity in {"low", "moderate", "severe"} for item in ordered)
		severe_count = sum(item.severity == "severe" for item in ordered)
		drift_fraction = drift_count / max(len(ordered), 1)
		severe_fraction = severe_count / max(len(ordered), 1)
		overall_score = float(np.average([item.risk_score for item in ordered], weights=[max(item.importance, 0.05) for item in ordered]) if ordered else 0.0)
		overall_thresholds = self._severity_thresholds([overall_score], history_overall_scores)
		overall_severity = _severity_from_score(overall_score, overall_thresholds)
		if severe_fraction >= 0.10:
			overall_severity = "severe"
		elif drift_fraction >= alert_feature_fraction and _severity_rank(overall_severity) < _severity_rank("moderate"):
			overall_severity = "moderate"
		trend, trend_delta, trend_slope = _trend_label(history, overall_score)
		alert_reason_parts = []
		if severe_fraction >= 0.10:
			alert_reason_parts.append(f"{severe_fraction:.0%} of features are severe")
		if drift_fraction >= alert_feature_fraction:
			alert_reason_parts.append(f"{drift_fraction:.0%} of features drifted")
		if overall_severity == "severe":
			alert_reason_parts.append("overall drift severity is severe")
		if trend == "sudden":
			alert_reason_parts.append("sudden upward drift detected")
		alert = bool(alert_reason_parts)
		alert_level = "critical" if severe_fraction >= 0.2 or overall_severity == "severe" else "warning" if alert else "info"
		alert_reason = "; ".join(alert_reason_parts) if alert_reason_parts else "no alert conditions met"
		timestamp = _utc_now()
		assessment = DriftAssessment(
			timestamp=timestamp,
			batch_size=int(len(_as_frame(live_frame))),
			feature_count=len(ordered),
			drift_count=drift_count,
			severe_count=severe_count,
			drift_fraction=drift_fraction,
			severe_fraction=severe_fraction,
			overall_score=overall_score,
			overall_severity=overall_severity,
			alert=alert,
			alert_level=alert_level,
			alert_reason=alert_reason,
			thresholds=thresholds,
			trend=trend,
			trend_delta=trend_delta,
			trend_slope=trend_slope,
			history_count=len(history),
			features=ordered,
			history=history,
		)
		assessment.dashboard = build_dashboard_json(assessment, top_n=top_n)
		if store_history:
			self.history.append(assessment.to_dict(top_n=top_n))
		return assessment


def build_dashboard_json(assessment: DriftAssessment, top_n: int = 10) -> Dict[str, Any]:
	return {
		"timestamp": assessment.timestamp,
		"overall": {
			"score": assessment.overall_score,
			"severity": assessment.overall_severity,
			"alert": assessment.alert,
			"level": assessment.alert_level,
			"reason": assessment.alert_reason,
		},
		"counts": {
			"features": assessment.feature_count,
			"drifted": assessment.drift_count,
			"severe": assessment.severe_count,
		},
		"fractions": {
			"drifted": assessment.drift_fraction,
			"severe": assessment.severe_fraction,
		},
		"thresholds": assessment.thresholds,
		"trend": {
			"label": assessment.trend,
			"delta": assessment.trend_delta,
			"slope": assessment.trend_slope,
		},
		"top_features": assessment.top_features(limit=top_n),
		"history": [
			{
				"timestamp": item.get("timestamp"),
				"overall_score": item.get("overall_score"),
				"severity": item.get("overall_severity"),
				"trend": item.get("trend"),
			}
			for item in assessment.history[-10:]
		],
	}


def save_drift_plot(assessment: DriftAssessment, output_path: Path, top_n: int = 12) -> None:
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if not assessment.features:
		return

	ordered = sorted(assessment.features, key=lambda item: item.risk_score, reverse=True)[:top_n]
	ordered = list(reversed(ordered))
	severity_palette = {
		"stable": "#2a9d8f",
		"low": "#8ecae6",
		"moderate": "#f4a261",
		"severe": "#e76f51",
	}
	colors = [severity_palette.get(item.severity, "#457b9d") for item in ordered]

	if assessment.history:
		fig, (bar_ax, trend_ax) = plt.subplots(1, 2, figsize=(15, max(5, 0.45 * len(ordered) + 2)), gridspec_kw={"width_ratios": [2.2, 1]})
	else:
		fig, bar_ax = plt.subplots(1, 1, figsize=(12, max(5, 0.45 * len(ordered) + 2)))
		trend_ax = None

	bar_ax.barh([item.feature for item in ordered], [item.risk_score for item in ordered], color=colors)
	bar_ax.axvline(assessment.thresholds["low"], color="#8ecae6", linestyle="--", linewidth=1, label="low threshold")
	bar_ax.axvline(assessment.thresholds["moderate"], color="#f4a261", linestyle="--", linewidth=1, label="moderate threshold")
	bar_ax.axvline(assessment.thresholds["severe"], color="#e76f51", linestyle="--", linewidth=1, label="severe threshold")
	bar_ax.set_xlabel("Risk score")
	bar_ax.set_ylabel("Feature")
	bar_ax.set_title("Live Drift Risk Ranking")
	bar_ax.legend(loc="lower right")

	if trend_ax is not None:
		history_scores = [float(item.get("overall_score", 0.0)) for item in assessment.history[-20:]]
		x = list(range(len(history_scores)))
		trend_ax.plot(x, history_scores, marker="o", linewidth=2, color="#264653")
		trend_ax.axhline(assessment.overall_score, color="#e76f51", linestyle="--", linewidth=1, label="current")
		trend_ax.set_title("Overall Drift Trend")
		trend_ax.set_xlabel("Batch")
		trend_ax.set_ylabel("Overall score")
		trend_ax.legend(loc="lower right")

	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def compute_drift_report(live_df: pd.DataFrame | Sequence[Mapping[str, Any]], baseline_df: pd.DataFrame | Sequence[Mapping[str, Any]], bins: int = 10) -> pd.DataFrame:
	monitor = DriftMonitor(
		model=object(),
		pipeline=object(),
		baseline_frame=_as_frame(baseline_df),
		importance_map={column: 1.0 for column in TARGET_COLUMNS if column in _as_frame(baseline_df).columns},
	)
	results = monitor.compare_features(live_df)
	return pd.DataFrame([result.to_dict() for result in results]).sort_values("risk_score", ascending=False).reset_index(drop=True)


def build_drift_summary(
	live_df: pd.DataFrame | Sequence[Mapping[str, Any]],
	baseline_df: pd.DataFrame | Sequence[Mapping[str, Any]],
	history_path: Path = DEFAULT_DRIFT_HISTORY_PATH,
	top_n: int = 10,
) -> DriftAssessment:
	prepared_baseline = _as_frame(baseline_df)
	monitor = DriftMonitor(
		model=object(),
		pipeline=object(),
		baseline_frame=prepared_baseline,
		history_path=history_path,
		importance_map={column: 1.0 for column in TARGET_COLUMNS if column in prepared_baseline.columns},
	)
	return monitor.assess(live_df, store_history=False, top_n=top_n)


def _clean_preprocessing_input(frame: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
	prepared = _as_frame(frame)
	prepared = prepared.copy()
	prepared.columns = prepared.columns.astype(str).str.strip().str.lower()
	prepared = prepared.drop(columns=[column for column in ["label", "attack_cat", "target"] if column in prepared.columns], errors="ignore")
	return prepared


def _load_baseline_source(source: Any, sample_size: int) -> pd.DataFrame:
	if isinstance(source, (str, Path)):
		return pd.read_csv(source, low_memory=False, nrows=sample_size)
	return _as_frame(source)


def build_model_input_drift_summary(
	live_df: pd.DataFrame | Sequence[Mapping[str, Any]],
	baseline_df: pd.DataFrame | Sequence[Mapping[str, Any]] | str | Path,
	pipeline: Any,
	model: Any = None,
	history_path: Path = DEFAULT_ENGINEERED_DRIFT_HISTORY_PATH,
	top_n: int = 10,
	baseline_sample_size: int = 2500,
	store_history: bool = True,
) -> DriftAssessment:
	clean_live = _clean_preprocessing_input(live_df)
	clean_baseline = _clean_preprocessing_input(_load_baseline_source(baseline_df, baseline_sample_size))
	engineered_live = pipeline.transform(clean_live)
	engineered_baseline = pipeline.transform(clean_baseline)

	feature_names = []
	if hasattr(pipeline, "get_feature_names_out") and callable(getattr(pipeline, "get_feature_names_out")):
		try:
			feature_names = list(pipeline.get_feature_names_out())
		except Exception:
			feature_names = list(engineered_baseline.columns)
	elif hasattr(pipeline, "get_feature_names") and callable(getattr(pipeline, "get_feature_names")):
		try:
			feature_names = list(pipeline.get_feature_names())
		except Exception:
			feature_names = list(engineered_baseline.columns)
	else:
		feature_names = list(engineered_baseline.columns)

	if len(feature_names) == engineered_baseline.shape[1]:
		engineered_baseline = pd.DataFrame(engineered_baseline.to_numpy(), columns=feature_names, index=engineered_baseline.index)
		engineered_live = pd.DataFrame(engineered_live.to_numpy(), columns=feature_names, index=engineered_live.index)

	importance_map = {column: 1.0 for column in engineered_baseline.columns}
	if model is not None and hasattr(model, "feature_importances_"):
		importances = np.asarray(getattr(model, "feature_importances_"), dtype=float)
		if len(importances) == engineered_baseline.shape[1]:
			importance_map = {column: float(score) for column, score in zip(engineered_baseline.columns, importances)}

	monitor = DriftMonitor(
		model=model or object(),
		pipeline=pipeline,
		baseline_frame=engineered_baseline,
		history_path=history_path,
		importance_map=importance_map,
	)
	return monitor.assess_numeric_features(engineered_live, baseline_frame=engineered_baseline, store_history=store_history, top_n=top_n)


def load_default_monitor(model: Any, pipeline: Any, baseline_path: Path = DEFAULT_BASELINE_PATH, history_path: Path = DEFAULT_DRIFT_HISTORY_PATH) -> DriftMonitor:
	return DriftMonitor.from_artifacts(model=model, pipeline=pipeline, baseline_path=baseline_path, history_path=history_path)


def load_artifacts(model_path: Path = DEFAULT_MODEL_PATH, pipeline_path: Path = DEFAULT_PIPELINE_PATH) -> Tuple[Any, Any]:
	model_blob = joblib.load(model_path)
	if isinstance(model_blob, dict) and "model" in model_blob:
		model = model_blob["model"]
	else:
		model = model_blob
	pipeline = joblib.load(pipeline_path)
	return model, pipeline
