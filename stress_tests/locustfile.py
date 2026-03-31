import random
from locust import HttpUser, task, between, events
import time

# Sample network flow data (realistic values from UNSW-NB15 dataset)
SAMPLE_INPUTS = [
    {
        # Normal TCP traffic
        "srcip": "192.168.1.10",
        "dstip": "10.0.0.5",
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "sport": 54321,
        "dsport": 80,
        "dur": 0.121478,
        "sbytes": 1032,
        "dbytes": 8234,
        "sttl": 64,
        "dttl": 128,
        "sloss": 0,
        "dloss": 0,
        "sload": 67891.23,
        "dload": 543210.45,
        "spkts": 12,
        "dpkts": 15,
        "swin": 65535,
        "dwin": 65535,
        "stcpb": 123456789,
        "dtcpb": 987654321,
        "smeansz": 86,
        "dmeansz": 549,
        "trans_depth": 1,
        "res_bdy_len": 8000,
        "sjit": 0.001,
        "djit": 0.002,
        "stime": 1234567890,
        "ltime": 1234567890,
        "sintpkt": 0.01,
        "dintpkt": 0.008,
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
    },
    {
        # UDP traffic (possibly attack pattern)
        "srcip": "10.0.0.100",
        "dstip": "192.168.1.1",
        "proto": "udp",
        "service": "-",
        "state": "INT",
        "sport": 12345,
        "dsport": 53,
        "dur": 0.000234,
        "sbytes": 64,
        "dbytes": 0,
        "sttl": 255,
        "dttl": 0,
        "sloss": 0,
        "dloss": 1,
        "sload": 2188034.56,
        "dload": 0.0,
        "spkts": 1,
        "dpkts": 0,
        "swin": 0,
        "dwin": 0,
        "stcpb": 0,
        "dtcpb": 0,
        "smeansz": 64,
        "dmeansz": 0,
        "trans_depth": 0,
        "res_bdy_len": 0,
        "sjit": 0.0,
        "djit": 0.0,
        "stime": 1234567891,
        "ltime": 1234567891,
        "sintpkt": 0.0,
        "dintpkt": 0.0,
        "tcprtt": 0.0,
        "synack": 0.0,
        "ackdat": 0.0,
        "is_sm_ips_ports": 0,
        "ct_state_ttl": 1,
        "ct_flw_http_mthd": 0,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_srv_src": 100,
        "ct_srv_dst": 1,
        "ct_dst_ltm": 50,
        "ct_src__ltm": 1,
        "ct_src_dport_ltm": 50,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 25,
    },
    {
        # FTP traffic
        "srcip": "172.16.0.50",
        "dstip": "172.16.0.1",
        "proto": "tcp",
        "service": "ftp",
        "state": "FIN",
        "sport": 49152,
        "dsport": 21,
        "dur": 5.234567,
        "sbytes": 512,
        "dbytes": 4096,
        "sttl": 64,
        "dttl": 64,
        "sloss": 0,
        "dloss": 0,
        "sload": 782.34,
        "dload": 6258.89,
        "spkts": 20,
        "dpkts": 25,
        "swin": 32768,
        "dwin": 32768,
        "stcpb": 111111111,
        "dtcpb": 222222222,
        "smeansz": 25,
        "dmeansz": 163,
        "trans_depth": 3,
        "res_bdy_len": 0,
        "sjit": 0.05,
        "djit": 0.03,
        "stime": 1234567892,
        "ltime": 1234567897,
        "sintpkt": 0.25,
        "dintpkt": 0.2,
        "tcprtt": 0.002,
        "synack": 0.001,
        "ackdat": 0.001,
        "is_sm_ips_ports": 0,
        "ct_state_ttl": 2,
        "ct_flw_http_mthd": 0,
        "is_ftp_login": 1,
        "ct_ftp_cmd": 5,
        "ct_srv_src": 3,
        "ct_srv_dst": 2,
        "ct_dst_ltm": 1,
        "ct_src__ltm": 1,
        "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 1,
    },
]

class IDSUser(HttpUser):
    """
    Simulated user making requests to the IDS API
    wait_time: Random delay between requests
    """
    wait_time = between(0.1, 0.5)  # 100-500ms between requests

    @task(10)
    def predict(self):
        """
        Main prediction endpoint - weighted 10x (most common operation)
        This is the heavy endpoint that runs ML inference.
        """
        # Pick a random sample input
        data = random.choice(SAMPLE_INPUTS).copy()

        # Add randomization to simulate varied traffic
        # IMPORTANT: All numeric fields must be floats (not ints) for API validation
        data["sport"] = float(random.randint(1024, 65535))
        data["dsport"] = float(random.randint(1024, 65535))
        data["sbytes"] = float(random.randint(64, 10000))
        data["dbytes"] = float(random.randint(0, 50000))
        data["spkts"] = float(random.randint(1, 100))
        data["dpkts"] = float(random.randint(1, 100))
        data["dur"] = random.uniform(0.001, 10.0)

        with self.client.post(
            "/predict",
            json=data,
            catch_response=True,
            name="/predict"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Rate limited - expected under heavy load
                response.failure(f"Rate limited: {response.text}")
            elif response.status_code == 503:
                # Circuit breaker open
                response.failure(f"Circuit breaker open: {response.text}")
            elif response.status_code == 400:
                # Validation error - likely data format issue
                response.failure(f"Validation error: {response.text}")
            else:
                response.failure(f"Error {response.status_code}: {response.text}")

    @task(3)
    def health_check(self):
        """Health check endpoint - weighted 3x"""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Error {response.status_code}: {response.text}")

    @task(2)
    def get_metrics(self):
        """Metrics endpoint - weighted 2x"""
        with self.client.get("/metrics", catch_response=True, name="/metrics") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Error {response.status_code}: {response.text}")

    @task(1)
    def root_check(self):
        """Root endpoint - weighted 1x"""
        with self.client.get("/", catch_response=True, name="/") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Error {response.status_code}: {response.text}")


class HeavyLoadUser(HttpUser):
    """
    Aggressive user for stress testing with controlled burst rate.
    Use this to find the breaking point of your API.

    Note: Even though rate limiting is 100/min, this user bursts faster
    to test circuit breaker and rate limiting behavior.
    """
    wait_time = between(0.05, 0.15)  # 50-150ms wait (more realistic than 0ms)

    @task
    def predict_burst(self):
        """Burst prediction requests with proper type conversion"""
        data = random.choice(SAMPLE_INPUTS).copy()

        # Ensure all numeric fields are floats
        data["sport"] = float(random.randint(1024, 65535))
        data["dsport"] = float(random.randint(1024, 65535))
        data["sbytes"] = float(random.randint(64, 10000))
        data["spkts"] = float(random.randint(1, 50))
        data["dpkts"] = float(random.randint(1, 50))
        data["dur"] = random.uniform(0.001, 5.0)

        with self.client.post(
            "/predict",
            json=data,
            catch_response=True,
            name="/predict [burst]"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                response.failure(f"Rate limited: {response.text}")
            elif response.status_code == 503:
                response.failure(f"Circuit breaker open: {response.text}")
            else:
                response.failure(f"Error {response.status_code}: {response.text}")


# Event hooks for custom logging
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, **kwargs):
    """Log slow requests"""
    if response_time > 1000:  # > 1 second
        print(f"SLOW REQUEST: {name} took {response_time:.2f}ms")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print("Starting stress test...")
    print(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops - print summary"""
    print("Stress test completed!")

    stats = environment.stats
    total = stats.total

    print(f"\nTotal Requests: {total.num_requests}")
    print(f"Failed Requests: {total.num_failures}")
    print(f"Failure Rate: {total.fail_ratio * 100:.2f}%")
    print("\nResponse Times:")
    print(f"  Average: {total.avg_response_time:.2f}ms")
    print(f"  Median:  {total.median_response_time:.2f}ms")
    print(f"  95th %%:  {total.get_response_time_percentile(0.95):.2f}ms")
    print(f"  99th %%:  {total.get_response_time_percentile(0.99):.2f}ms")
    print(f"\nThroughput: {total.total_rps:.2f} req/sec")
