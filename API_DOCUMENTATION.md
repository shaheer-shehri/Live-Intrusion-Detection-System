# Intrusion Detection System API - Documentation Guide

## Quick Start

### 1. Access Interactive API Documentation

When the API is running, visit: **http://localhost:8000/docs**

This opens **Swagger UI** with:
- All endpoints with descriptions
- Request/response schema visualization
- Try-it-out functionality (test endpoints directly)
- Example requests and responses
- Full API schema download

Alternative: **http://localhost:8000/redoc** (ReDoc - alternative UI)

### 2. OpenAPI Schema

Raw machine-readable schema: **http://localhost:8000/openapi.json**

---

## API Endpoints Overview

### Health & Status (Not Rate Limited)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Basic API status |
| `/health` | GET | Detailed health with circuit breaker state |
| `/metrics` | GET | Performance metrics (latency, throughput) |
| `/stress/config` | GET | View stress handling configuration |
| `/stress/reset` | POST | Reset stress handling state |

### Prediction (Rate Limited: 100 req/min per IP)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Classify network traffic |

---

## Response Schema Documentation

### 1. Root Response `/`

```json
{
  "message": "IDS API running"
}
```

**Use for**: Liveness probes, load balancer checks

---

### 2. Health Response `/health`

```json
{
  "status": "healthy",  // or "degraded"
  "circuit_breaker": "closed",  // closed | open | half_open
  "model_loaded": true,
  "active_requests": 3,
  "error_rate": 0.0  // 0.0 to 1.0
}
```

**Status Interpretation**:
- `"healthy"`: Circuit is CLOSED and error rate < 10%
- `"degraded"`: Circuit is OPEN or error rate ≥ 10%

**Use for**: Readiness probes, Kubernetes health checks

---

### 3. Metrics Response `/metrics`

```json
{
  "requests": {
    "total": 1523,
    "successful": 1510,
    "failed": 13,
    "rate_limited": 0
  },
  "latency_ms": {
    "average": 87.5,
    "p95": 145.2,
    "p99": 312.8
  },
  "throughput": {
    "requests_per_second": 25.3,
    "active_requests": 2
  },
  "circuit_breaker": {
    "state": "closed"
  }
}
```

**Latency Interpretation**:
- `average`: Mean response time
- `p95`: 95% of requests are faster (good target: < 200ms)
- `p99`: Worst case for most users (good target: < 500ms)

**Use for**: Grafana dashboards, Prometheus scraping, alerting

---

### 4. Stress Config Response `/stress/config`

```json
{
  "rate_limit": {
    "requests_per_window": 100,
    "window_seconds": 60.0
  },
  "circuit_breaker": {
    "failure_threshold": 10,
    "recovery_timeout": 30.0,
    "current_state": "closed"
  }
}
```

**Meaning**:
- Max 100 requests per 60 seconds per IP
- Circuit opens after 10 failures in a window
- After 30 seconds, circuit attempts recovery (HALF_OPEN)

---

### 5. Prediction Response `/predict`

```json
{
  "prediction": "Normal"  // or "DoS", "Exploits", "Generic", etc.
}
```

---

## Error Responses

### 400 Bad Request

**Cause**: Validation error (invalid input)

```json
{
  "detail": "Field 'proto' cannot be empty"
}
```

**Common causes**:
- Missing required field
- Invalid data type
- Out of range value (e.g., GPA > 4.0)
- Negative value for non-negative field

### 429 Too Many Requests

**Cause**: Rate limit exceeded

```json
HTTP/1.1 429
Retry-After: 60

{
  "detail": "Rate limit exceeded. Please try again later."
}
```

**How to handle**:
1. Read `Retry-After` header
2. Wait that many seconds before retrying
3. Implement exponential backoff for robustness

### 503 Service Unavailable

**Cause**: Circuit breaker is OPEN (too many failures)

```json
HTTP/1.1 503
Retry-After: 30

{
  "detail": "Service temporarily unavailable (circuit breaker open)"
}
```

**What it means**:
- API is experiencing issues
- Too many recent failures detected
- System is in recovery mode
- Wait 30 seconds and retry

### 500 Internal Server Error

**Cause**: Unexpected error during prediction

```json
{
  "detail": "Prediction failed: [error details]"
}
```

---

## Rate Limiting Details

### How It Works

Uses **sliding window algorithm**:

```
Time: 0s    30s    60s    90s    120s
      |-----|------|------|------|
IP1:  50req 30req  40req  → Window moves, old requests expire
```

### Response Headers

Every response includes:

```
X-Response-Time: 87.45ms
X-RateLimit-Remaining: 62
```

**X-RateLimit-Remaining** shows remaining requests in current window.

### What Counts?

Only these endpoints count toward rate limit:
- `/predict`

These are **exempt**:
- `/` (root)
- `/health`
- `/metrics`
- `/stress/config`
- `/stress/reset`
- `/docs`, `/openapi.json`

### Retry Strategy

```python
import time
import requests

for attempt in range(5):
    try:
        response = requests.post("http://localhost:8000/predict", json=data)
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        break
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        time.sleep(2 ** attempt)  # Exponential backoff
```

---

## Circuit Breaker Behavior

### States

| State | Meaning | Behavior |
|-------|---------|----------|
| **CLOSED** | Normal | Accept all requests |
| **OPEN** | Too many failures | Reject requests (503) |
| **HALF_OPEN** | Testing recovery | Allow limited test requests |

### State Transitions

```
        ┌─────────────────────┐
        │    Requests OK      │
        │  Failures Reset     │
        │                     ▼
     ┌──────┐  failures   ┌──────┐  timeout  ┌──────────┐
     │CLOSED├────────────→│ OPEN │──────────→│HALF_OPEN │
     └──────┘             └──────┘           └──────────┘
        ▲                                          │
        │                  success (3+ requests)  │
        └──────────────────────────────────────────┘
```

### Thresholds

- **Failures to Open**: 10 consecutive failures
- **Recovery Timeout**: 30 seconds
- **Half-Open Test Calls**: 3 successful calls to close circuit

### Monitoring

Check circuit state via `/health` or `/metrics`:

```python
response = requests.get("http://localhost:8000/health")
data = response.json()
if data["circuit_breaker"] == "open":
    print("Circuit is OPEN - service is recovering")
```

---

## Performance Targets & Alerts

### Latency Targets

- **p95 < 200ms**: Excellent
- **p95 < 500ms**: Acceptable
- **p95 > 500ms**: Investigate

- **p99 < 500ms**: Excellent
- **p99 < 1000ms**: Acceptable
- **p99 > 1000ms**: Investigate

### Throughput Targets

- **Error Rate < 1%**: Excellent
- **Error Rate 1-5%**: Warning
- **Error Rate > 5%**: Critical

### Alerting Rules

```yaml
# Prometheus alerting rules

- alert: HighLatency
  expr: ids_latency_p95_ms > 200
  for: 5m

- alert: CircuitBreakerOpen
  expr: ids_circuit_breaker_state == "open"
  for: 1m

- alert: HighErrorRate
  expr: (ids_failed_requests / ids_total_requests) > 0.05
  for: 5m

- alert: Overload
  expr: ids_requests_per_second > 1000
  for: 1m
```

---

## Example Use Cases

### 1. Continuous Monitoring (Prometheus)

```python
# Scrape metrics every 15 seconds
GET http://localhost:8000/metrics

# Parse and export to Prometheus format
```

### 2. Health Check Loop

```python
import requests
import time

while True:
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        data = response.json()

        if data["status"] == "degraded":
            print("⚠️  API is degraded!")
            # Send alert

        time.sleep(10)
    except Exception as e:
        print(f"Check failed: {e}")
```

### 3. Stress Testing

```bash
# Run Locust load test
locust -f stress_tests/locustfile.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 300s \
    --headless \
    --html=report.html

# Review report.html for metrics
```

---

## Testing the API

### 1. With Curl

```bash
# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"proto":"tcp","service":"http",...}'
```

### 2. With Python Requests

```python
import requests

# Load test data
sample = {
    "srcip": "192.168.1.10",
    "dstip": "10.0.0.5",
    "proto": "tcp",
    "service": "http",
    "state": "FIN",
    ...
}

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json=sample
)

print(f"Status: {response.status_code}")
print(f"Prediction: {response.json()}")
```

### 3. With Python Requests + Rate Limit Handling

```python
import requests
import time

def predict_with_retry(data, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(
            "http://localhost:8000/predict",
            json=data
        )

        if response.status_code == 200:
            return response.json()

        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            print(f"Rate limited. Retrying in {retry_after}s...")
            time.sleep(retry_after)

        elif response.status_code == 503:
            print("Service unavailable (circuit open). Waiting...")
            time.sleep(10)

        else:
            response.raise_for_status()

    raise Exception("Max retries exceeded")

result = predict_with_retry(sample)
print(result)
```

---

## Integration Examples

### Kubernetes Liveness Probe

```yaml
livenessProbe:
  httpGet:
    path: /
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
```

### Kubernetes Readiness Probe

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 2
```

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: 'ids-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

---

## Documentation Resources

- **Interactive Swagger UI**: http://localhost:8000/docs
- **ReDoc Alternative**: http://localhost:8000/redoc
- **OpenAPI JSON Schema**: http://localhost:8000/openapi.json
- **Source Code**: `/api/main.py`, `/api/stress_handling.py`
- **Tests**: `/stress_tests/test_stress_handling.py`
- **Load Tests**: `/stress_tests/locustfile.py`

---

## Support

For issues or questions:
1. Check `/health` endpoint status
2. Review `/metrics` for performance issues
3. Check `/stress/config` for rate limit settings
4. Run stress tests to identify bottlenecks
5. Review API documentation in Swagger UI
