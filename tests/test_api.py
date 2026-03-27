import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on path for `api` imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import main
from api import model_loader


@pytest.fixture()
def client(monkeypatch):
    class DummyModel:
        def predict(self, X):
            # Always predict class index 1
            return np.array([1])

    class DummyPipeline:
        def transform(self, df):
            # Pass through as a numpy array to match main.predict expectations
            return df.to_numpy()

    dummy_model = DummyModel()
    dummy_pipeline = DummyPipeline()
    dummy_labels = ["Normal", "Generic"]

    # Patch both model_loader and main module references
    monkeypatch.setattr(model_loader, "model", dummy_model)
    monkeypatch.setattr(model_loader, "pipeline", dummy_pipeline)
    monkeypatch.setattr(model_loader, "class_labels", dummy_labels)
    monkeypatch.setattr(main, "model", dummy_model)
    monkeypatch.setattr(main, "pipeline", dummy_pipeline)
    monkeypatch.setattr(main, "class_labels", dummy_labels)

    return TestClient(main.app)


def _sample_payload():
    return {
        "srcip": "",
        "dstip": "",
        "proto": "tcp",
        "service": "http",
        "state": "est",
        "sport": 12345,
        "dsport": 80,
        "dur": 1.0,
        "sbytes": 10,
        "dbytes": 20,
        "sttl": 60,
        "dttl": 60,
        "sloss": 0,
        "dloss": 0,
        "sload": 1,
        "dload": 2,
        "spkts": 2,
        "dpkts": 3,
        "swin": 1000,
        "dwin": 1000,
        "stcpb": 0,
        "dtcpb": 0,
        "smeansz": 5,
        "dmeansz": 7,
        "trans_depth": 0,
        "res_bdy_len": 0,
        "sjit": 0,
        "djit": 0,
        "stime": 0,
        "ltime": 1,
        "sintpkt": 0.1,
        "dintpkt": 0.1,
        "tcprtt": 0.01,
        "synack": 0.01,
        "ackdat": 0.01,
        "is_sm_ips_ports": 0,
        "ct_state_ttl": 1,
        "ct_flw_http_mthd": 0,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_srv_src": 1,
        "ct_srv_dst": 1,
        "ct_dst_ltm": 1,
        "ct_src__ltm": 1,
        "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 1,
    }


def test_root_health(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json().get("message") == "IDS API running"


def test_predict_ok(client):
    resp = client.post("/predict", json=_sample_payload())
    assert resp.status_code == 200
    body = resp.json()
    # Temporary visibility for debugging
    print({"prediction": body.get("prediction")})
    assert body.get("prediction") == "Generic"
