"""
Local IDS agent.

Runs on the same machine as the user's browser. Detects DNS lookups / TCP
connections to the watched malicious-demo domains and POSTs a trigger to the
remote backend (Azure) so the cloud-hosted simulator switches to the matching
attack scenario.

Usage:
    set IDS_BACKEND_URL=https://<your-backend>.azurecontainerapps.io
    set IDS_TRIGGER_TOKEN=<same token as backend>      # optional, only if backend has it set
    python local_agent.py

Or on PowerShell:
    $env:IDS_BACKEND_URL = "https://<your-backend>.azurecontainerapps.io"
    python local_agent.py
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request

from domain_watcher import WATCH_DOMAINS, DomainWatcher

BACKEND_URL = os.environ.get("IDS_BACKEND_URL", "http://localhost:8000").rstrip("/")
TOKEN       = os.environ.get("IDS_TRIGGER_TOKEN", "").strip()
TIMEOUT     = 5.0


def remote_trigger(scenario: str) -> None:
    url = f"{BACKEND_URL}/trigger/{scenario}"
    req = urllib.request.Request(url, method="POST")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            print(f"[agent] {scenario} → HTTP {resp.status}", flush=True)
    except urllib.error.HTTPError as e:
        print(f"[agent] {scenario} → HTTP {e.code}: {e.reason}", flush=True)
    except urllib.error.URLError as e:
        print(f"[agent] {scenario} → connection error: {e.reason}", flush=True)
    except Exception as e:
        print(f"[agent] {scenario} → unexpected error: {e}", flush=True)


def main() -> int:
    print(f"[agent] backend  : {BACKEND_URL}", flush=True)
    print(f"[agent] auth     : {'enabled' if TOKEN else 'disabled'}", flush=True)
    print(f"[agent] watching : {len(WATCH_DOMAINS)} domains", flush=True)
    for d, s in WATCH_DOMAINS.items():
        print(f"           {d:30s} → {s}", flush=True)

    watcher = DomainWatcher(trigger_fn=remote_trigger)
    watcher.start()
    print("[agent] running. Press Ctrl-C to stop.", flush=True)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[agent] stopping...", flush=True)
        watcher.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
