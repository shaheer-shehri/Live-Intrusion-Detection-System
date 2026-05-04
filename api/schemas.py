from typing import Any, Dict, List

from pydantic import BaseModel, Field

class NetworkInput(BaseModel):
    # String categorical fields
    proto: str = "tcp"
    service: str = "http"
    state: str = "est"

    # Numeric port and timing fields
    sport: float = 0.0
    dsport: float = 0.0
    dur: float = 0.0
    sttl: float = 64.0
    dttl: float = 64.0
    sloss: float = 0.0
    sload: float = 0.0
    dload: float = 0.0
    spkts: float = 0.0
    swin: float = 0.0
    smeansz: float = 0.0
    dmeansz: float = 0.0
    trans_depth: float = 0.0
    res_bdy_len: float = 0.0
    sjit: float = 0.0
    djit: float = 0.0
    sintpkt: float = 0.0
    dintpkt: float = 0.0
    tcprtt: float = 0.0
    synack: float = 0.0
    ackdat: float = 0.0

    # Fields required by the model that are not user-visible in the simple form
    sbytes: float = 500.0
    dbytes: float = 1000.0
    dwin: float = 8192.0
    stcpb: float = 0.0
    dtcpb: float = 0.0
    stime: float = 0.0

    # Binary and count-based features
    is_sm_ips_ports: float = 0.0
    ct_state_ttl: float = 0.0
    ct_ftp_cmd: float = 0.0
    ct_srv_src: float = 0.0
    ct_srv_dst: float = 0.0
    ct_dst_ltm: float = 0.0
    ct_src__ltm: float = 0.0
    ct_src_dport_ltm: float = 0.0
    ct_dst_sport_ltm: float = 0.0
    ct_dst_src_ltm: float = 0.0


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