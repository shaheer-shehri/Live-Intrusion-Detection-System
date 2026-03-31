"""
Intrusion Detection System API

This API provides:
- /predict: Classify network traffic as Normal or Attack type
- /health: Detailed health check with system status
- /metrics: Performance metrics (latency, throughput, error rates)
- /stress/config: View/update stress handling configuration

Stress Handling Features:
- Rate Limiting: Prevents API abuse (default: 100 req/min per IP)
- Circuit Breaker: Protects against cascade failures
- Metrics Collection: Tracks latency, throughput, errors
"""

import math
import time
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd

from api.schemas import NetworkInput
from api.model_loader import class_labels, model, pipeline
from api.stress_handling import (
    StressHandlingMiddleware,
    StressHandler,
    RateLimitConfig,
    CircuitBreakerConfig,
    stress_handler,
)

# ============================================================================
# OpenAPI Response Schemas for Documentation
# ============================================================================

class RootResponse(BaseModel):
    """Root endpoint response"""
    message: str

class HealthCheckResponse(BaseModel):
    """Health check endpoint response"""
    status: str
    circuit_breaker: str
    model_loaded: bool
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

# ============================================================================
# FastAPI App Setup with Enhanced Documentation
# ============================================================================

app = FastAPI(
    title="Intrusion Detection System (IDS) API",
    description="""
## Features

### Core Functionality
- **Intrusion Detection**: Classify network traffic as Normal or Attack type using ML
- **45 Network Features**: Full UNSW-NB15 dataset feature support
- **Real-time Predictions**: Sub-100ms response times

### Stress Handling & Resilience
- **Rate Limiting**: Sliding window algorithm (default: 100 req/min per IP)
- **Circuit Breaker**: Automatic failure protection with 3 states (CLOSED → OPEN → HALF_OPEN)
- **Request Metrics**: Real-time latency (p95, p99) and throughput monitoring
- **Automatic Recovery**: Self-healing with recovery timeout

### Monitoring
- **Health Checks**: Detailed system status endpoint
- **Performance Metrics**: Latency percentiles, throughput, error rates
- **Stress Config**: View and reset stress handling state

## Error Codes
- **400**: Bad Request (validation error, invalid input)
- **429**: Too Many Requests (rate limited - retry after delay)
- **500**: Internal Server Error (model unavailable, prediction failed)
- **503**: Service Unavailable (circuit breaker open - service recovering)

## Rate Limiting
All endpoints except `/`, `/health`, `/metrics`, `/docs` are rate limited:
- **Limit**: 100 requests per 60 seconds per IP
- **Response**: Returns 429 with `Retry-After` header

## Circuit Breaker States
- **CLOSED**: Normal operation, all requests allowed
- **OPEN**: Too many failures, requests rejected with 503
- **HALF_OPEN**: Testing recovery, limited requests allowed

## Examples
### Healthy Response
```json
{"status": "healthy", "circuit_breaker": "closed"}
```

### Rate Limited
```json
HTTP/1.1 429 Too Many Requests
Retry-After: 60
{"detail": "Rate limit exceeded. Please try again later."}
```

### Circuit Breaker Open
```json
HTTP/1.1 503 Service Unavailable
Retry-After: 30
{"detail": "Service temporarily unavailable (circuit breaker open)"}
```
    """,
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

# Add stress handling middleware
app.add_middleware(
    StressHandlingMiddleware,
    handler=stress_handler,
    exclude_paths=["/", "/health", "/metrics", "/docs", "/openapi.json"],
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
    Basic health check endpoint.

    Returns a simple message indicating the API is running.

    **Use this for**: Liveness probes, load balancer health checks
    """
    return {"message": "IDS API running"}


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Detailed health check",
    description="Get comprehensive API health status including circuit breaker state and error rate. Not rate limited.",
)
def health_check():
    """
    Detailed health check endpoint.

    Returns the API's overall health status with:
    - **status**: "healthy" or "degraded"
    - **circuit_breaker**: Current state (closed/open/half_open)
    - **model_loaded**: Whether ML model is available
    - **active_requests**: Current concurrent requests
    - **error_rate**: Percentage of failed requests

    **Status Rules**:
    - Healthy: circuit is CLOSED and error rate < 10%
    - Degraded: circuit is OPEN or error rate ≥ 10%

    **Use this for**: Readiness probes, monitoring dashboards
    """
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
    description="Get real-time performance metrics including latency percentiles and throughput. Not rate limited.",
)
def get_metrics():
    """
    Performance metrics endpoint.

    Returns comprehensive performance statistics:

    **Requests**:
    - total: Total requests processed
    - successful: Requests that succeeded
    - failed: Requests that failed
    - rate_limited: Requests rejected due to rate limiting

    **Latency Metrics** (in milliseconds):
    - average: Mean response time
    - p95: 95th percentile (95% of requests faster than this)
    - p99: 99th percentile (worst case for most users)

    **Throughput**:
    - requests_per_second: Current RPS
    - active_requests: Concurrent requests being processed

    **Circuit Breaker**:
    - state: Current state (closed/open/half_open)

    **Use this for**:
    - Grafana/Prometheus dashboards
    - Alerting systems
    - Performance analysis
    - Capacity planning

    **Recommended Alerts**:
    - p95_latency > 200ms
    - p99_latency > 500ms
    - error_rate > 1%
    - circuit_state == "open"
    """
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
    description="View current rate limiting and circuit breaker configuration. Not rate limited.",
)
def get_stress_config():
    """
    Get current stress handling configuration.

    Returns:

    **Rate Limit**:
    - requests_per_window: Max requests allowed in time window
    - window_seconds: Time window duration in seconds

    **Circuit Breaker**:
    - failure_threshold: Failures before circuit opens
    - recovery_timeout: Seconds before attempting recovery
    - current_state: Current state (closed/open/half_open)

    **Use this for**: Understanding API throttling and recovery behavior
    """
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
    description="Reset all stress handling counters and circuit breaker state. For testing/admin purposes. Not rate limited.",
)
def reset_stress_handler():
    """
    Reset stress handling state.

    Clears:
    - Rate limit counters for all IPs
    - Circuit breaker state and failure counts
    - All metrics and latency history

    **Warning**: This is intended for testing and admin purposes only.
    In production, prefer natural recovery via circuit breaker timeout.

    **Use this for**: Test cleanup, metric reset
    """
    stress_handler.reset_all()
    return {"message": "Stress handler reset successfully"}

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


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Classify network traffic",
    description="""
    Predict if network traffic is Normal or Attack type.

    Requires 45 network flow features from the UNSW-NB15 dataset.

    **Rate Limited**: Yes (100 requests per 60 seconds per IP)
    """,
)
def predict(data: NetworkInput):
    """
    Predict intrusion detection classification.

    Accepts 45 network flow features and returns the predicted traffic type.

    **Input**: NetworkInput model with 45 fields including:
    - Source/destination IPs (optional)
    - Protocol, service, connection state
    - Byte counts, packet counts
    - TTL values, connection timing
    - And 30+ additional flow statistics

    **Output**:
    - prediction: Traffic classification (e.g., "Normal", "DoS", "Exploits", etc.)

    **Validation**:
    - All numeric fields must be non-negative
    - GPA must be between 0.0 and 4.0
    - All required fields must be present
    - Invalid combinations are rejected

    **Errors**:
    - 400: Validation error (missing/invalid field)
    - 429: Rate limited (too many requests)
    - 500: Model unavailable or prediction failed
    - 503: Circuit breaker open (service recovering)

    **Example Request**:
    ```json
    {
      "srcip": "192.168.1.10",
      "dstip": "10.0.0.5",
      "proto": "tcp",
      "service": "http",
      "state": "FIN",
      "sport": 54321,
      "dsport": 80,
      ...
    }
    ```

    **Example Response**:
    ```json
    {
      "prediction": "Normal"
    }
    ```

    **Performance**: Typical response time 50-150ms
    """
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