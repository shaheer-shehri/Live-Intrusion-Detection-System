from pydantic import BaseModel

class NetworkInput(BaseModel):
    # Flow identifiers (IPs optional but accepted)
    srcip: str | None = None
    dstip: str | None = None
    proto: str
    service: str
    state: str
    sport: float
    dsport: float

    # Core totals and timing
    dur: float
    sbytes: float
    dbytes: float
    sttl: float
    dttl: float
    sloss: float
    dloss: float
    sload: float
    dload: float
    spkts: float
    dpkts: float
    swin: float
    dwin: float
    stcpb: float
    dtcpb: float
    smeansz: float
    dmeansz: float
    trans_depth: float
    res_bdy_len: float
    sjit: float
    djit: float
    stime: float
    ltime: float
    sintpkt: float
    dintpkt: float
    tcprtt: float
    synack: float
    ackdat: float

    # Flags / binary indicators
    is_sm_ips_ports: float
    ct_state_ttl: float
    ct_flw_http_mthd: float
    is_ftp_login: float
    ct_ftp_cmd: float

    # Count-based features
    ct_srv_src: float
    ct_srv_dst: float
    ct_dst_ltm: float
    ct_src__ltm: float
    ct_src_dport_ltm: float
    ct_dst_sport_ltm: float
    ct_dst_src_ltm: float