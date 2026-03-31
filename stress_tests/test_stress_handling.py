import time
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.stress_handling import (
    RateLimiter,
    RateLimitConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    MetricsCollector,
    StressHandler,
)

# Rate Limiter Tests
class TestRateLimiter:
    """Tests for the sliding window rate limiter"""

    def test_allows_requests_under_limit(self):
        """Requests under the limit should be allowed"""
        limiter = RateLimiter(RateLimitConfig(requests_per_window=5, window_seconds=60))

        for i in range(5):
            assert limiter.is_allowed("client1") is True

    def test_blocks_requests_over_limit(self):
        """Requests over the limit should be blocked"""
        limiter = RateLimiter(RateLimitConfig(requests_per_window=3, window_seconds=60))

        # First 3 should pass
        for _ in range(3):
            assert limiter.is_allowed("client1") is True

        # 4th should be blocked
        assert limiter.is_allowed("client1") is False

    def test_separate_limits_per_client(self):
        """Each client has independent rate limits"""
        limiter = RateLimiter(RateLimitConfig(requests_per_window=2, window_seconds=60))

        # Client 1 uses their quota
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False

        # Client 2 still has full quota
        assert limiter.is_allowed("client2") is True
        assert limiter.is_allowed("client2") is True

    def test_window_expiration(self):
        """Old requests should expire and allow new ones"""
        limiter = RateLimiter(RateLimitConfig(requests_per_window=2, window_seconds=0.1))

        # Use up quota
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False

        # Wait for window to expire
        time.sleep(0.15)

        # Should be allowed again
        assert limiter.is_allowed("client1") is True

    def test_get_remaining(self):
        """Should correctly report remaining requests"""
        limiter = RateLimiter(RateLimitConfig(requests_per_window=5, window_seconds=60))

        assert limiter.get_remaining("client1") == 5

        limiter.is_allowed("client1")
        assert limiter.get_remaining("client1") == 4

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        assert limiter.get_remaining("client1") == 2

    def test_limited_count_tracking(self):
        """Should track how many requests were rate limited"""
        limiter = RateLimiter(RateLimitConfig(requests_per_window=2, window_seconds=60))

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        assert limiter.limited_count == 0

        limiter.is_allowed("client1")  # Blocked
        limiter.is_allowed("client1")  # Blocked
        assert limiter.limited_count == 2

    def test_reset(self):
        """Reset should clear all state"""
        limiter = RateLimiter(RateLimitConfig(requests_per_window=2, window_seconds=60))

        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")  # Blocked

        limiter.reset()

        assert limiter.get_remaining("client1") == 2
        assert limiter.limited_count == 0

# Circuit Breaker Tests
class TestCircuitBreaker:
    """Tests for the circuit breaker pattern"""

    def test_starts_closed(self):
        """Circuit should start in closed (normal) state"""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_threshold(self):
        """Circuit should open after failure threshold is reached"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))

        # Record failures
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()  # Threshold reached
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_failure_count(self):
        """Successes should decrement failure count in closed state"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))

        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # Should reduce count

        # Need 3 more failures to open (2 - 1 + 3 = 4, but threshold is 3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_recovery_timeout(self):
        """Circuit should transition to half-open after recovery timeout"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1
        ))

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should be half-open now
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute() is True

    def test_half_open_closes_on_success(self):
        """Successful calls in half-open should close the circuit"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=2
        ))

        # Open and wait for half-open
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Successful calls should close circuit
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Failure in half-open should reopen the circuit"""
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1
        ))

        # Open and wait for half-open
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Failure should reopen
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_manual_reset(self):
        """Manual reset should close the circuit"""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True


# Metrics Collector Tests
class TestMetricsCollector:
    """Tests for performance metrics collection"""

    def test_records_latency(self):
        """Should record latency measurements"""
        collector = MetricsCollector()

        collector.record_latency(10.0)
        collector.record_latency(20.0)
        collector.record_latency(30.0)

        snapshot = collector.get_snapshot()
        assert snapshot.total_requests == 3
        assert snapshot.avg_latency_ms == 20.0

    def test_tracks_success_failure(self):
        """Should separately track successes and failures"""
        collector = MetricsCollector()

        collector.record_latency(10.0, success=True)
        collector.record_latency(20.0, success=True)
        collector.record_latency(30.0, success=False)

        snapshot = collector.get_snapshot()
        assert snapshot.successful_requests == 2
        assert snapshot.failed_requests == 1

    def test_percentile_calculation(self):
        """Should correctly calculate percentiles"""
        collector = MetricsCollector()

        # Add 100 measurements: 1, 2, 3, ..., 100
        for i in range(1, 101):
            collector.record_latency(float(i))

        snapshot = collector.get_snapshot()
        assert snapshot.p95_latency_ms == 95.0
        assert snapshot.p99_latency_ms == 99.0

    def test_context_manager_tracking(self):
        """Context manager should track request timing"""
        collector = MetricsCollector()

        with collector.track_request():
            time.sleep(0.01)  # 10ms

        snapshot = collector.get_snapshot()
        assert snapshot.total_requests == 1
        assert snapshot.successful_requests == 1
        assert snapshot.avg_latency_ms >= 10  # At least 10ms

    def test_active_requests_tracking(self):
        """Should track concurrent active requests"""
        collector = MetricsCollector()

        snapshot = collector.get_snapshot()
        assert snapshot.active_requests == 0

    def test_throughput_calculation(self):
        """Should calculate requests per second"""
        collector = MetricsCollector()

        # Make 10 requests
        for _ in range(10):
            collector.record_latency(5.0)

        snapshot = collector.get_snapshot()
        assert snapshot.requests_per_second > 0

    def test_reset(self):
        """Reset should clear all metrics"""
        collector = MetricsCollector()

        collector.record_latency(10.0)
        collector.record_latency(20.0)

        collector.reset()

        snapshot = collector.get_snapshot()
        assert snapshot.total_requests == 0
        assert snapshot.avg_latency_ms == 0.0

# Integrated StressHandler Tests

class TestStressHandler:
    """Tests for the integrated stress handler"""

    def test_allows_normal_requests(self):
        """Normal requests should pass through"""
        handler = StressHandler(
            rate_limit_config=RateLimitConfig(requests_per_window=100, window_seconds=60),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=10),
        )

        # Should not raise
        handler.check_request("client1")

    def test_rate_limits_excessive_requests(self):
        """Should raise 429 when rate limit exceeded"""
        handler = StressHandler(
            rate_limit_config=RateLimitConfig(requests_per_window=2, window_seconds=60),
        )

        handler.check_request("client1")
        handler.check_request("client1")

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            handler.check_request("client1")

        assert exc_info.value.status_code == 429
        assert "Rate limit" in exc_info.value.detail

    def test_circuit_breaker_protection(self):
        """Should raise 503 when circuit is open"""
        handler = StressHandler(
            rate_limit_config=RateLimitConfig(requests_per_window=100, window_seconds=60),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2),
        )

        # Trip the circuit
        handler.record_failure()
        handler.record_failure()

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            handler.check_request("client1")

        assert exc_info.value.status_code == 503
        assert "circuit breaker" in exc_info.value.detail.lower()

    def test_metrics_snapshot(self):
        """Should provide combined metrics snapshot"""
        handler = StressHandler()

        snapshot = handler.get_metrics()
        assert hasattr(snapshot, 'total_requests')
        assert hasattr(snapshot, 'circuit_state')
        assert hasattr(snapshot, 'rate_limited_requests')

    def test_reset_all(self):
        """Reset should clear all components"""
        handler = StressHandler(
            rate_limit_config=RateLimitConfig(requests_per_window=2, window_seconds=60),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2),
        )

        # Use up rate limit and trip circuit
        handler.check_request("client1")
        handler.check_request("client1")
        handler.record_failure()
        handler.record_failure()

        handler.reset_all()

        # Should work again
        handler.check_request("client1")  # Should not raise


# API Endpoint Testing
class TestStressHandlingEndpoints:
    """Tests for stress handling API endpoints"""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create test client with mocked model"""
        # Mock the model loader
        monkeypatch.setattr("api.model_loader.model", MagicMock())
        monkeypatch.setattr("api.model_loader.pipeline", MagicMock())
        monkeypatch.setattr("api.model_loader.class_labels", ["Normal", "Attack"])

        # Reset stress handler before each test
        from api.stress_handling import stress_handler
        stress_handler.reset_all()

        from api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health endpoint should return status"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "circuit_breaker" in data
        assert data["status"] in ["healthy", "degraded"]

    def test_metrics_endpoint(self, client):
        """Metrics endpoint should return performance data"""
        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "requests" in data
        assert "latency_ms" in data
        assert "throughput" in data
        assert "circuit_breaker" in data

    def test_stress_config_endpoint(self, client):
        """Should return stress handling configuration"""
        response = client.get("/stress/config")
        assert response.status_code == 200

        data = response.json()
        assert "rate_limit" in data
        assert "circuit_breaker" in data
        assert "requests_per_window" in data["rate_limit"]

    def test_stress_reset_endpoint(self, client):
        """Should reset stress handler state"""
        response = client.post("/stress/reset")
        assert response.status_code == 200
        assert "reset" in response.json()["message"].lower()

    def test_rate_limit_headers(self, client):
        """Responses should include rate limit headers"""
        # Make a request to a rate-limited endpoint
        from api.stress_handling import stress_handler
        stress_handler.reset_all()

        # The /predict endpoint is rate limited
        # We need to mock the pipeline transform and model predict
        with patch('api.main.pipeline') as mock_pipeline, \
             patch('api.main.model') as mock_model:
            mock_pipeline.transform.return_value = [[0] * 10]
            mock_model.predict.return_value = [0]

            response = client.post("/predict", json={
                "proto": "tcp",
                "service": "http",
                "state": "FIN",
                "sport": 12345,
                "dsport": 80,
                "dur": 0.1,
                "sbytes": 100,
                "dbytes": 200,
                "sttl": 64,
                "dttl": 128,
                "sloss": 0,
                "dloss": 0,
                "sload": 1000,
                "dload": 2000,
                "spkts": 10,
                "dpkts": 15,
                "swin": 65535,
                "dwin": 65535,
                "stcpb": 123456,
                "dtcpb": 654321,
                "smeansz": 100,
                "dmeansz": 150,
                "trans_depth": 1,
                "res_bdy_len": 500,
                "sjit": 0.001,
                "djit": 0.002,
                "stime": 1234567890,
                "ltime": 1234567890,
                "sintpkt": 0.01,
                "dintpkt": 0.02,
                "tcprtt": 0.003,
                "synack": 0.001,
                "ackdat": 0.001,
                "is_sm_ips_ports": 0,
                "ct_state_ttl": 2,
                "ct_flw_http_mthd": 1,
                "is_ftp_login": 0,
                "ct_ftp_cmd": 0,
                "ct_srv_src": 5,
                "ct_srv_dst": 10,
                "ct_dst_ltm": 3,
                "ct_src__ltm": 2,
                "ct_src_dport_ltm": 1,
                "ct_dst_sport_ltm": 2,
                "ct_dst_src_ltm": 1,
            })

            # Check for custom headers
            assert "X-Response-Time" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
