"""
Local IDS agent.

Runs on the same machine as the user's browser.

What it does:
  1. On startup, checks backend health and confirms the simulator is active
     (normal traffic is already flowing to the live monitor page).
  2. Every 30 s prints a live traffic summary from the backend.
  3. Runs DomainWatcher in the background — detects when the browser visits
     any of the watched attack-demo domains and immediately POSTs
     /trigger/{scenario} to the cloud backend so the live monitor shows an
     attack burst, then auto-reverts to Normal.

Usage (PowerShell):
    $env:IDS_BACKEND_URL   = "https://shaheershehri-ai-project-backend.hf.space"
    $env:IDS_TRIGGER_TOKEN = "<same token set on the backend>"   # optional
    python local_agent.py

Usage (cmd):
    set IDS_BACKEND_URL=https://shaheershehri-ai-project-backend.hf.space
    set IDS_TRIGGER_TOKEN=<token>
    python local_agent.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

from domain_watcher import WATCH_DOMAINS, DomainWatcher

BACKEND_URL = os.environ.get(
    "IDS_BACKEND_URL", "https://shaheershehri-ai-project-backend.hf.space"
).rstrip("/")
TOKEN   = os.environ.get("IDS_TRIGGER_TOKEN", "").strip()
TIMEOUT = 6.0


def _request(path: str, method: str = "GET") -> dict:
    req = urllib.request.Request(f"{BACKEND_URL}{path}", method=method)
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())

def remote_trigger(scenario: str) -> None:
    print(f"\n[agent] *** DOMAIN DETECTED — triggering {scenario.upper()} ***", flush=True)
    try:
        data = _request(f"/trigger/{scenario}", method="POST")
        print(f"[agent] backend confirmed: {data.get('message', 'ok')} "
              f"({data.get('duration_seconds', '?')}s)", flush=True)
    except urllib.error.HTTPError as e:
        print(f"[agent] trigger failed — HTTP {e.code}: {e.reason}", flush=True)
    except Exception as e:
        print(f"[agent] trigger failed — {e}", flush=True)


# ── startup health check ──────────────────────────────────────────────────────

def check_backend() -> bool:
    try:
        h = _request("/health")
        sim_ok  = h.get("simulator_active", h.get("model_loaded", False))
        sim_err = h.get("simulator_error", "")
        print("[agent] backend     : reachable", flush=True)
        if sim_ok:
            print("[agent] simulator   : ACTIVE — normal traffic is flowing", flush=True)
        else:
            print(f"[agent] simulator   : INACTIVE ({sim_err})", flush=True)
            print("[agent] WARNING — no live traffic until backend pipeline loads", flush=True)
        return True
    except Exception as e:
        print(f"[agent] backend UNREACHABLE: {e}", flush=True)
        return False


# ── periodic stats ────────────────────────────────────────────────────────────

def print_stats() -> None:
    try:
        stats = _request("/monitor?n=1").get("stats", {})
        total  = stats.get("total_flows", 0)
        normal = stats.get("normal_flows", 0)
        attack = stats.get("attack_flows", 0)
        state  = stats.get("current_state", "normal")
        exp    = stats.get("attack_expires_in_sec", 0)
        if state != "normal":
            print(f"[agent] status — {total} flows | ATTACK:{state.upper()} "
                  f"(reverts in {exp}s)", flush=True)
        else:
            print(f"[agent] status — {total} flows | Normal "
                  f"({normal} normal, {attack} attack)", flush=True)
    except Exception:
        pass


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 58, flush=True)
    print("  IDS Local Agent", flush=True)
    print("=" * 58, flush=True)
    print(f"[agent] backend : {BACKEND_URL}", flush=True)
    print(f"[agent] auth    : {'token set' if TOKEN else 'no token'}\n", flush=True)

    # Startup health check — retry once if backend is slow to respond
    if not check_backend():
        print("[agent] retrying in 10 s …", flush=True)
        time.sleep(10)
        check_backend()

    # List watched domains
    print(f"\n[agent] watching {len(WATCH_DOMAINS)} domains:", flush=True)
    for domain, scenario in WATCH_DOMAINS.items():
        print(f"   {domain:38s}→  {scenario}", flush=True)

    # Start domain watcher (DNS cache poll + TCP poll + optional Scapy sniff)
    watcher = DomainWatcher(trigger_fn=remote_trigger)
    watcher.start()

    print("\n[agent] monitoring started.", flush=True)
    print("[agent] visit a watched domain to trigger an attack scenario.", flush=True)
    print("[agent] normal traffic flows continuously on the live monitor.", flush=True)
    print("[agent] press Ctrl-C to stop.\n", flush=True)

    # Main loop: print stats every 30 s
    tick = 0
    try:
        while True:
            time.sleep(5)
            tick += 1
            if tick % 6 == 0:
                print_stats()
    except KeyboardInterrupt:
        print("\n[agent] stopping …", flush=True)
        watcher.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
