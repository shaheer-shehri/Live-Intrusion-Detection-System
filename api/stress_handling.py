import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Circuit tripped, rejecting all requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5       # Failures before circuit opens
    recovery_timeout: float = 30.0   # Seconds before attempting recovery
    half_open_max_calls: int = 3     # Test calls in half-open state


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_window: int = 100   # Max requests per window
    window_seconds: float = 60.0     # Time window in seconds


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics snapshot"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    active_requests: int
    circuit_state: str
    rate_limited_requests: int


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
            return self._state

    def record_success(self) -> None:
        """Record a successful call - resets failure count"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self.config.half_open_max_calls:
                    # Service recovered, close circuit
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed call - may trip the circuit"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery test, reopen circuit
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    # Threshold exceeded, open circuit
                    self._state = CircuitState.OPEN

    def can_execute(self) -> bool:
        """Check if request should be allowed through"""
        state = self.state  # Triggers recovery timeout check
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            with self._lock:
                return self._half_open_calls < self.config.half_open_max_calls
        return False  # OPEN state

    def reset(self) -> None:
        """Manually reset the circuit breaker"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0


class RateLimiter:
    """
    Sliding Window Rate Limiter

    Tracks requests per client (IP) using a sliding window algorithm:
    1. Maintains a list of timestamps per client
    2. Cleans up old timestamps outside the window
    3. Reject if count exceeds limit
    """

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._limited_count = 0

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make a request"""
        now = time.time()
        cutoff = now - self.config.window_seconds

        with self._lock:
            # Clean old timestamps
            self._requests[client_id] = [
                ts for ts in self._requests[client_id] if ts > cutoff
            ]

            # Check limit
            if len(self._requests[client_id]) >= self.config.requests_per_window:
                self._limited_count += 1
                return False

            # Record this request
            self._requests[client_id].append(now)
            return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client in current window"""
        now = time.time()
        cutoff = now - self.config.window_seconds

        with self._lock:
            current = len([
                ts for ts in self._requests[client_id] if ts > cutoff
            ])
            return max(0, self.config.requests_per_window - current)

    @property
    def limited_count(self) -> int:
        """Total requests that were rate limited"""
        return self._limited_count

    def reset(self) -> None:
        """Reset rate limiter state"""
        with self._lock:
            self._requests.clear()
            self._limited_count = 0


class MetricsCollector:
    """
    Metrics Collection for Performance Monitoring

    Tracks:
    - Request counts (total, success, failure)
    - Latency distribution (avg, p95, p99)
    - Throughput (requests per second)
    - Active concurrent requests
    """

    def __init__(self, retention_seconds: float = 300.0):
        self.retention_seconds = retention_seconds
        self._latencies: List[tuple] = []  # (timestamp, latency_ms)
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        self._active_requests = 0
        self._start_time = time.time()
        self._lock = threading.Lock()

    @contextmanager
    def track_request(self):
        """Context manager to track request timing"""
        start = time.time()
        with self._lock:
            self._request_count += 1
            self._active_requests += 1

        success = False
        try:
            yield
            success = True
        finally:
            elapsed_ms = (time.time() - start) * 1000
            with self._lock:
                self._active_requests -= 1
                self._latencies.append((time.time(), elapsed_ms))
                if success:
                    self._success_count += 1
                else:
                    self._error_count += 1

    def record_latency(self, latency_ms: float, success: bool = True) -> None:
        """Manually record a latency measurement"""
        with self._lock:
            self._request_count += 1
            self._latencies.append((time.time(), latency_ms))
            if success:
                self._success_count += 1
            else:
                self._error_count += 1

    def _cleanup_old_data(self) -> None:
        """Remove data older than retention period"""
        cutoff = time.time() - self.retention_seconds
        self._latencies = [(ts, lat) for ts, lat in self._latencies if ts > cutoff]

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile from sorted values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def get_snapshot(self, circuit_state: str = "unknown", rate_limited: int = 0) -> MetricsSnapshot:
        """Get current metrics values"""
        with self._lock:
            self._cleanup_old_data()

            latencies = [lat for _, lat in self._latencies]
            elapsed = time.time() - self._start_time

            return MetricsSnapshot(
                total_requests=self._request_count,
                successful_requests=self._success_count,
                failed_requests=self._error_count,
                avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
                p95_latency_ms=self._percentile(latencies, 95),
                p99_latency_ms=self._percentile(latencies, 99),
                requests_per_second=self._request_count / elapsed if elapsed > 0 else 0.0,
                active_requests=self._active_requests,
                circuit_state=circuit_state,
                rate_limited_requests=rate_limited,
            )

    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._latencies.clear()
            self._request_count = 0
            self._success_count = 0
            self._error_count = 0
            self._active_requests = 0
            self._start_time = time.time()


class StressHandler:
    """
    Unified Stress Handling for FastAPI

    Combines rate limiting, circuit breaker, and metrics collection
    into a single easy-to-use interface.
    """

    def __init__(
        self,
        rate_limit_config: RateLimitConfig = None,
        circuit_breaker_config: CircuitBreakerConfig = None,
    ):
        self.rate_limiter = RateLimiter(rate_limit_config)
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
        self.metrics = MetricsCollector()

    def check_request(self, client_id: str) -> None:
        """
        Check if request should be allowed
        Raises HTTPException if rate limited or circuit is open
        """
        # Check rate limit first
        if not self.rate_limiter.is_allowed(client_id):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": str(int(self.rate_limiter.config.window_seconds))}
            )

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable (circuit breaker open)",
                headers={"Retry-After": str(int(self.circuit_breaker.config.recovery_timeout))}
            )

    def record_success(self) -> None:
        """Record successful request"""
        self.circuit_breaker.record_success()

    def record_failure(self) -> None:
        """Record failed request"""
        self.circuit_breaker.record_failure()

    def get_metrics(self) -> MetricsSnapshot:
        """Get current metrics snapshot"""
        return self.metrics.get_snapshot(
            circuit_state=self.circuit_breaker.state.value,
            rate_limited=self.rate_limiter.limited_count,
        )

    def reset_all(self) -> None:
        """Reset all stress handling components"""
        self.rate_limiter.reset()
        self.circuit_breaker.reset()
        self.metrics.reset()


# Global stress handler instance
stress_handler = StressHandler(
    rate_limit_config=RateLimitConfig(requests_per_window=100, window_seconds=60),
    circuit_breaker_config=CircuitBreakerConfig(failure_threshold=10, recovery_timeout=30),
)


class StressHandlingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI Middleware for Automatic Stress Handling

    Intercepts all requests and:
    1. Extracts client IP for rate limiting
    2. Checks rate limit and circuit breaker
    3. Tracks request metrics
    4. Records success/failure for circuit breaker
    """

    def __init__(self, app, handler: StressHandler = None, exclude_paths: List[str] = None):
        super().__init__(app)
        self.handler = handler or stress_handler
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip stress handling for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Track request timing
        start_time = time.time()

        try:
            # Check if request is allowed
            self.handler.check_request(client_ip)

            # Process request
            with self.handler.metrics.track_request():
                response = await call_next(request)

            # Record outcome based on status code
            if response.status_code >= 500:
                self.handler.record_failure()
            else:
                self.handler.record_success()

            # Add timing header
            elapsed_ms = (time.time() - start_time) * 1000
            response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"
            response.headers["X-RateLimit-Remaining"] = str(
                self.handler.rate_limiter.get_remaining(client_ip)
            )

            return response

        except HTTPException:
            # Re-raise rate limit and circuit breaker exceptions
            raise
        except Exception as e:
            # Record failure for unexpected errors
            self.handler.record_failure()
            raise
