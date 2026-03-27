import math
from typing import Dict, List

import numpy as np

from fastapi import FastAPI, HTTPException
import pandas as pd
from api.schemas import NetworkInput
from api.model_loader import class_labels, model, pipeline

app = FastAPI(title="Intrusion Detection API")

@app.get("/")
def root():
    return {"message": "IDS API running"}

def _validate_input(data: NetworkInput) -> Dict[str, object]:
    ordered_fields: List[str] = [
        "srcip",
        "dstip",
        "proto",
        "service",
        "state",
        "sport",
        "dsport",
        "dur",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
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

    non_negative: set[str] = {
        "sport",
        "dsport",
        "dur",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
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
    }

    values: Dict[str, object] = {name: getattr(data, name) for name in ordered_fields}

    for name, value in values.items():
        if name in {"srcip", "dstip"}:
            # Optional; allow None/empty. Pipeline will drop these.
            if value is None:
                values[name] = ""
            continue

        if value is None:
            raise HTTPException(status_code=400, detail=f"Missing field: {name}")

        if name in {"proto", "service", "state"}:
            if not str(value).strip():
                raise HTTPException(status_code=400, detail=f"Field '{name}' cannot be empty")
            continue

        if not math.isfinite(float(value)):
            raise HTTPException(status_code=400, detail=f"Field '{name}' must be a finite number")
        if name in non_negative and float(value) < 0:
            raise HTTPException(status_code=400, detail=f"Field '{name}' must be non-negative")

    return values


@app.post("/predict")
def predict(data: NetworkInput):
    try:
        values = _validate_input(data)
        df = pd.DataFrame([values])
        X_proc = pipeline.transform(df)
        if hasattr(X_proc, "to_numpy"):
            X_proc = X_proc.to_numpy()
        else:
            X_proc = np.asarray(X_proc)
        raw_pred = model.predict(X_proc)[0]

        if class_labels and int(raw_pred) < len(class_labels):
            label = class_labels[int(raw_pred)]
        else:
            label = str(raw_pred)

        return {"prediction": label}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))