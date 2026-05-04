import json
import math
import os
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd

from api.schemas import NetworkInput
from api.schemas import (
    DriftAssessmentRequest,
    DriftAssessmentResponse,
)
from api.model_loader import class_labels, model, pipeline
from monitoring import drift as drift_monitoring
from simulator import TrafficSimulator
from domain_watcher import DomainWatcher
from api.stress_handling import (
    StressHandlingMiddleware,
    StressHandler,
    RateLimitConfig,
    CircuitBreakerConfig,
    stress_handler,
)

# OpenAPI Response Schemas for Documentation
class RootResponse(BaseModel):
    """Root endpoint response"""
    message: str

class HealthCheckResponse(BaseModel):
    """Health check endpoint response"""
    status: str
    circuit_breaker: str
    model_loaded: bool
    simulator_active: bool
    simulator_error: Optional[str]
    active_requests: int
    error_rate: float

class PredictionResponse(BaseModel):
    """Prediction endpoint response"""
    prediction: str

class LatencyMetrics(BaseModel):
    """Latency metrics in milliseconds"""
    average: float
    p95: float
    p99: float

class ThroughputMetrics(BaseModel):
    """Throughput metrics"""
    requests_per_second: float
    active_requests: int

class RequestMetrics(BaseModel):
    """Request count metrics"""
    total: int
    successful: int
    failed: int
    rate_limited: int

class CircuitBreakerMetrics(BaseModel):
    """Circuit breaker state"""
    state: str

class MetricsResponse(BaseModel):
    """Metrics endpoint response"""
    requests: RequestMetrics
    latency_ms: LatencyMetrics
    throughput: ThroughputMetrics
    circuit_breaker: CircuitBreakerMetrics

class RateLimitConfig_Schema(BaseModel):
    """Rate limit configuration"""
    requests_per_window: int
    window_seconds: float

class CircuitBreakerConfig_Schema(BaseModel):
    """Circuit breaker configuration"""
    failure_threshold: int
    recovery_timeout: float
    current_state: str

class StressConfigResponse(BaseModel):
    """Stress configuration endpoint response"""
    rate_limit: RateLimitConfig_Schema
    circuit_breaker: CircuitBreakerConfig_Schema

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str

# Batch Processing Setup

MAX_BATCH_SIZE = 10
MAX_WAIT_TIME = 0.020  # 20 milliseconds window for micro-batching
DEFAULT_DRIFT_HISTORY_PATH = drift_monitoring.DEFAULT_DRIFT_HISTORY_PATH
DEFAULT_BASELINE_PATH = drift_monitoring.DEFAULT_BASELINE_PATH

batch_queue = []
queue_lock = asyncio.Lock()
batch_ready_event = asyncio.Event()
drift_monitor = drift_monitoring.load_default_monitor(
    model=model,
    pipeline=pipeline,
    baseline_path=DEFAULT_BASELINE_PATH,
    history_path=DEFAULT_DRIFT_HISTORY_PATH,
)

_simulator = TrafficSimulator(buffer_size=200)
_watcher   = DomainWatcher(trigger_fn=_simulator.trigger)
_DISABLE_LOCAL_WATCHER = os.environ.get("IDS_DISABLE_LOCAL_WATCHER", "").lower() in {"1", "true", "yes"}


def _run_ml_batch(data_list: List[Dict]) -> List[str]:
    """Runs the whole batch through the ML model synchronously."""
    df = pd.DataFrame(data_list)
    X_proc = pipeline.transform(df)
    # Pass the DataFrame directly so XGBoost 3.x feature-name validation passes.
    # Converting to numpy would strip column names and raise a ValueError.
    raw_preds = model.predict(X_proc)
    
    results = []
    for raw_pred in raw_preds:
        if class_labels and int(raw_pred) < len(class_labels):
            results.append(class_labels[int(raw_pred)])
        else:
            results.append(str(raw_pred))
    return results

async def predict_task():
    """Background task that repeatedly pulls data from the queue and processes batches."""
    while True:
        await batch_ready_event.wait()
        # Wait short time for more requests to arrive for optimal batch size
        await asyncio.sleep(MAX_WAIT_TIME)
        
        async with queue_lock:
            current_batch = batch_queue[:MAX_BATCH_SIZE]
            del batch_queue[:MAX_BATCH_SIZE]
            if not batch_queue:
                batch_ready_event.clear()
        
        if not current_batch:
            continue
            
        try:
            # Offload heavy pandas/numpy work to a background thread
            preds = await asyncio.to_thread(_run_ml_batch, [req['data'] for req in current_batch])
            for req, pred in zip(current_batch, preds):
                req['future'].set_result(pred)
        except Exception as e:
            for req in current_batch:
                if not req['future'].done():
                    req['future'].set_exception(e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the startup and shutdown of the background batch task."""
    _simulator.start()
    if not _DISABLE_LOCAL_WATCHER:
        _watcher.start()
    task = asyncio.create_task(predict_task())
    yield
    task.cancel()
    _simulator.stop()
    if not _DISABLE_LOCAL_WATCHER:
        _watcher.stop()


app = FastAPI(
    title="Intrusion Detection System (IDS) API",
    lifespan=lifespan,
    description=""" API for trafic monitoring""",
    version="2.0.0",
    contact={
        "name": "IDS Support",
        "url": "http://localhost:8000/docs",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://live-intrusion-detection-system.vercel.app",
    ],
    allow_origin_regex=r"https://live-intrusion-detection-system.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add stress handling middleware
app.add_middleware(
    StressHandlingMiddleware,
    handler=stress_handler,
    exclude_paths=[
        "/", "/health", "/metrics",
        "/drift/assess", "/drift/latest",
        "/monitor", "/monitor/live",
        "/predict-batch",
        "/docs", "/openapi.json",
    ],
)


@app.get(
    "/",
    response_model=RootResponse,
    tags=["Health"],
    summary="Root endpoint",
    description="Basic API status check. Not rate limited.",
)
def root():
    """
    health check endpoint
    """
    return {"message": "IDS API running"}


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Detailed health check",
    description="API health status",
)
def health_check():
    metrics = stress_handler.get_metrics()

    # Determine health status
    status = "healthy"
    if metrics.circuit_state == "open":
        status = "degraded"
    elif metrics.failed_requests > metrics.successful_requests * 0.1:  # >10% error rate
        status = "degraded"

    return {
        "status": status,
        "circuit_breaker": metrics.circuit_state,
        "model_loaded": model is not None,
        "simulator_active": not _simulator._disabled,
        "simulator_error": _simulator._disabled_reason if _simulator._disabled else None,
        "active_requests": metrics.active_requests,
        "error_rate": (
            metrics.failed_requests / metrics.total_requests
            if metrics.total_requests > 0
            else 0.0
        ),
    }


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Monitoring"],
    summary="Performance metrics",
    description="real-time performance metrics",
)
def get_metrics():
    metrics = stress_handler.get_metrics()

    return {
        "requests": {
            "total": metrics.total_requests,
            "successful": metrics.successful_requests,
            "failed": metrics.failed_requests,
            "rate_limited": metrics.rate_limited_requests,
        },
        "latency_ms": {
            "average": round(metrics.avg_latency_ms, 2),
            "p95": round(metrics.p95_latency_ms, 2),
            "p99": round(metrics.p99_latency_ms, 2),
        },
        "throughput": {
            "requests_per_second": round(metrics.requests_per_second, 2),
            "active_requests": metrics.active_requests,
        },
        "circuit_breaker": {
            "state": metrics.circuit_state,
        },
    }


@app.get(
    "/stress/config",
    response_model=StressConfigResponse,
    tags=["Stress Handling"],
    summary="Get stress handling configuration",
    description="to view current rate limiting and circuit breaker configuration",
)
def get_stress_config():
    return {
        "rate_limit": {
            "requests_per_window": stress_handler.rate_limiter.config.requests_per_window,
            "window_seconds": stress_handler.rate_limiter.config.window_seconds,
        },
        "circuit_breaker": {
            "failure_threshold": stress_handler.circuit_breaker.config.failure_threshold,
            "recovery_timeout": stress_handler.circuit_breaker.config.recovery_timeout,
            "current_state": stress_handler.circuit_breaker.state.value,
        },
    }

@app.post(
    "/stress/reset",
    response_model=MessageResponse,
    tags=["Stress Handling"],
    summary="Reset stress handling state",
    description="Reset all stress handling counters and circuit breaker state",
)
def reset_stress_handler():
    stress_handler.reset_all()
    return {"message": "Stress handler reset successfully"}

def _validate_input(data: NetworkInput) -> Dict[str, object]:
    ordered_fields: List[str] = [
        "proto",
        "service",
        "state",
        "sport",
        "dsport",
        "dur",
        "sttl",
        "dttl",
        "sloss",
        "sload",
        "dload",
        "spkts",
        "swin",
        "smeansz",
        "dmeansz",
        "trans_depth",
        "res_bdy_len",
        "sjit",
        "djit",
        "sintpkt",
        "dintpkt",
        "tcprtt",
        "synack",
        "ackdat",
        "is_sm_ips_ports",
        "ct_state_ttl",
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
        "sttl",
        "dttl",
        "sloss",
        "sload",
        "dload",
        "spkts",
        "swin",
        "smeansz",
        "dmeansz",
        "trans_depth",
        "res_bdy_len",
        "sjit",
        "djit",
        "sintpkt",
        "dintpkt",
        "tcprtt",
        "synack",
        "ackdat",
        "is_sm_ips_ports",
        "ct_state_ttl",
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


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Classify network traffic",
    description="""Predict if network traffic""",
)
async def predict(data: NetworkInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check server logs for the load error")
    try:
        values = _validate_input(data)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        async with queue_lock:
            batch_queue.append({'data': values, 'future': future})
            batch_ready_event.set()

        label = await future
        return {"prediction": label}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict-batch",
    tags=["Prediction"],
    summary="Bulk predict from a CSV upload",
    description="Upload a CSV file of network flows. Returns one prediction per row.",
)
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    try:
        raw = await file.read()
        from io import BytesIO
        df = pd.read_csv(BytesIO(raw), low_memory=False)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")

        drop_cols = [c for c in ["target", "attack_cat", "label"] if c in df.columns]
        feat_df   = df.drop(columns=drop_cols, errors="ignore").fillna(0)

        X_proc = await asyncio.to_thread(pipeline.transform, feat_df)
        raw_preds = await asyncio.to_thread(model.predict, X_proc)
        labels = [_label_from_raw(p) for p in raw_preds]

        from collections import Counter
        counts = dict(Counter(labels))

        rows = []
        for i, lbl in enumerate(labels):
            row = df.iloc[i].to_dict()
            row["prediction"] = lbl
            rows.append(row)

        return {
            "filename":     file.filename,
            "row_count":    len(labels),
            "predictions":  labels,
            "class_counts": counts,
            "rows":         rows,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")


def _label_from_raw(raw_pred) -> str:
    if class_labels and int(raw_pred) < len(class_labels):
        return class_labels[int(raw_pred)]
    return str(raw_pred)


@app.post(
    "/drift/assess",
    response_model=DriftAssessmentResponse,
    tags=["Monitoring"],
    summary="Assess drift for a live batch",
    description="Compute drift metrics for a batch of live network flows and return a structured summary.",
)
async def assess_drift(payload: DriftAssessmentRequest):
    if not payload.flows:
        raise HTTPException(status_code=400, detail="At least one flow is required for drift assessment")

    try:
        frame = pd.DataFrame([item.model_dump(exclude_none=True) for item in payload.flows])
        assessment = drift_monitor.assess(frame, store_history=payload.store_history, top_n=payload.top_n)
        return assessment.to_dict(top_n=payload.top_n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/drift/latest",
    response_model=DriftAssessmentResponse,
    tags=["Monitoring"],
    summary="Get the latest stored drift assessment",
    description="Returns the most recent drift summary captured by the monitor.",
)
def latest_drift():
    last = drift_monitor.history.last()
    if not last:
        raise HTTPException(status_code=404, detail="No drift history has been recorded yet")
    return last


@app.get(
    "/monitor",
    tags=["Monitoring"],
    summary="Snapshot — last N classified flows",
    description="One-shot JSON snapshot of the last *n* flows (default 50). Use /monitor/live for a continuous stream.",
)
def monitor(n: int = 50):
    n = max(1, min(n, 200))
    return {
        "flows": _simulator.get_recent(n),
        "stats": _simulator.get_stats(),
    }


@app.get(
    "/monitor/live",
    tags=["Monitoring"],
    summary="Live SSE stream — continuous traffic feed",
    description=(
        "Server-Sent Events stream. Keeps the connection open and pushes a JSON event "
        "every second containing only **new** flows since the last push, plus current stats.\n\n"
        "Connect with: `curl -N http://localhost:8000/monitor/live`\n\n"
        "Or open in a browser — each `data:` line is a JSON object with `flows` and `stats`.\n\n"
        "Shows **Normal** traffic continuously. Automatically injects attack-class flows "
        "when any of the following domains is resolved in the browser:\n\n"
        "- **testphp.vulnweb.com** → Exploits (45 s)\n"
        "- **ddostest.me** → DoS (30 s)\n"
        "- **scanme.nmap.org** → Reconnaissance (40 s)\n"
        "- **hackthissite.org** → Generic (35 s)\n"
        "- **www.webscantest.com** → Fuzzers (30 s)"
    ),
)
async def monitor_live(request: Request):
    async def event_stream():
        # Seed: send the last 20 flows immediately so the client has something to show
        last_ts: float = 0.0
        seed = _simulator.get_recent(20)
        if seed:
            last_ts = seed[-1]["ts"]
            yield f"data: {json.dumps({'flows': seed, 'stats': _simulator.get_stats()})}\n\n"

        tick = 0
        while True:
            if await request.is_disconnected():
                break
            new_flows = _simulator.get_since(last_ts)
            if new_flows:
                last_ts = new_flows[-1]["ts"]
            payload = json.dumps({"flows": new_flows, "stats": _simulator.get_stats()})
            yield f"data: {payload}\n\n"
            tick += 1
            # SSE comment keeps proxies (Cloudflare, nginx) from closing an idle stream
            if tick % 15 == 0:
                yield ": keepalive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if proxied
            "Connection":       "keep-alive",
        },
    )


# ── Remote trigger (used by local_agent.py when backend is in the cloud) ──────

_TRIGGER_TOKEN = os.environ.get("IDS_TRIGGER_TOKEN", "").strip()


@app.post(
    "/trigger/{scenario}",
    tags=["Simulator"],
    summary="Force-trigger a scenario (used by local agent)",
    description=(
        "Switch the simulator into the named attack scenario for its configured duration, "
        "or pass `normal` to revert immediately. If env var `IDS_TRIGGER_TOKEN` is set, "
        "requests must include `Authorization: Bearer <token>`."
    ),
)
def trigger_scenario(scenario: str, request: Request):
    if _TRIGGER_TOKEN:
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth[7:].strip() != _TRIGGER_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")
    scenario = scenario.lower().strip()
    if scenario == "normal":
        _simulator.reset_normal()
        return {"message": "reset to normal"}
    result = _simulator.trigger(scenario)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result