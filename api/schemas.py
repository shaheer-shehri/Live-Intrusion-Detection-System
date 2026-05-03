from typing import Any, Dict, List

from pydantic import BaseModel, Field

class NetworkInput(BaseModel):
    # Flow identifiers (IPs optional but accepted)
    srcip: str | None = None
    dstip: str | None = None
    proto: str
    service: str
    state: str
    sport: float
    dsport: float

    # Core totals and timing (sbytes, dbytes, stime, ltime removed - used for engineered features)
    dur: float
    sttl: float
    dttl: float
    sloss: float
    sload: float
    dload: float
    spkts: float
    swin: float
    smeansz: float
    dmeansz: float
    trans_depth: float
    res_bdy_len: float
    sjit: float
    djit: float
    sintpkt: float
    dintpkt: float
    tcprtt: float
    synack: float
    ackdat: float

    # Engineered features (created during preprocessing)
    duration: float
    packet_rate: float
    byte_ratio: float

    # Flags / binary indicators (ct_flw_http_mthd, is_ftp_login removed - high missing values)
    is_sm_ips_ports: float
    ct_state_ttl: float
    ct_ftp_cmd: float

    # Count-based features
    ct_srv_src: float
    ct_srv_dst: float
    ct_dst_ltm: float
    ct_src__ltm: float
    ct_src_dport_ltm: float
    ct_dst_sport_ltm: float
    ct_dst_src_ltm: float


class DriftAssessmentRequest(BaseModel):
    flows: List[NetworkInput]
    top_n: int = 10
    store_history: bool = True


class DriftFeatureScore(BaseModel):
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
    status: str = "stable"


class DriftThresholds(BaseModel):
    low: float
    moderate: float
    severe: float


class DriftTrend(BaseModel):
    label: str
    delta: float
    slope: float


class DriftOverallSummary(BaseModel):
    score: float
    severity: str
    alert: bool
    level: str
    reason: str


class DriftAssessmentResponse(BaseModel):
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
    thresholds: DriftThresholds
    trend: DriftTrend
    history_count: int
    overall: DriftOverallSummary | None = None
    top_features: List[DriftFeatureScore]
    history: List[Dict[str, Any]] = Field(default_factory=list)
    dashboard: Dict[str, Any]