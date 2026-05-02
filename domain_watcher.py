"""
DomainWatcher: triggers the simulator when a known demo-attack domain is
visited in the user's browser.

Three detection methods run in parallel; the first one to fire wins
(cooldown prevents duplicates):

  1. Windows DNS cache polling    — `ipconfig /displaydns`  (no admin)
  2. TCP connection polling       — psutil.net_connections() (no admin)
  3. Live DNS sniffing            — Scapy (needs Npcap + admin)
"""
from __future__ import annotations

import platform
import re
import socket
import subprocess
import threading
import time
from typing import Callable, Dict, Optional, Set

WATCH_DOMAINS: Dict[str, str] = {
    "testphp.vulnweb.com":  "exploits",
    "vulnweb.com":          "exploits",
    "ddostest.me":          "dos",
    "scanme.nmap.org":      "recon",
    "nmap.org":             "recon",
    "hackthissite.org":     "generic",
    "www.hackthissite.org": "generic",
    "webscantest.com":      "fuzzers",
    "www.webscantest.com":  "fuzzers",
}

_COOLDOWN_SEC  = 30
_POLL_INTERVAL = 2.0
_IS_WINDOWS    = platform.system() == "Windows"
_CREATE_NO_WIN = 0x08000000 if _IS_WINDOWS else 0


def _match_scenario(qname: str) -> Optional[str]:
    qname = qname.lower().rstrip(".")
    if qname in WATCH_DOMAINS:
        return WATCH_DOMAINS[qname]
    for domain, scenario in WATCH_DOMAINS.items():
        if qname.endswith("." + domain):
            return scenario
    return None


class DomainWatcher:
    def __init__(self, trigger_fn: Callable[[str], None]) -> None:
        self._trigger_fn = trigger_fn
        self._last_fired: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._active = False
        self._domain_ips: Dict[str, str] = {}      # ip → scenario
        self._initial_cache: Set[str] = set()      # domains in cache at startup

    def _maybe_fire(self, scenario: str, source: str = "") -> None:
        with self._lock:
            now = time.time()
            if now - self._last_fired.get(scenario, 0) >= _COOLDOWN_SEC:
                self._last_fired[scenario] = now
                fired = True
            else:
                fired = False
        if fired:
            try:
                self._trigger_fn(scenario)
                print(f"[DomainWatcher] {source}: triggered scenario '{scenario}'", flush=True)
            except Exception as exc:
                print(f"[DomainWatcher] trigger error: {exc}", flush=True)

    def _resolve_watched(self) -> None:
        for domain, scenario in WATCH_DOMAINS.items():
            try:
                infos = socket.getaddrinfo(domain, None)
                for info in infos:
                    ip = info[4][0]
                    self._domain_ips[ip] = scenario
            except Exception:
                pass

    def _read_dns_cache(self) -> Set[str]:
        if not _IS_WINDOWS:
            return set()
        try:
            out = subprocess.check_output(
                ["ipconfig", "/displaydns"],
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=_CREATE_NO_WIN,
            )
        except Exception:
            return set()

        domains: Set[str] = set()
        for line in out.splitlines():
            line = line.strip()
            m = re.match(r"Record Name\s*[.…: ]+\s*(.+)", line)
            if m:
                domains.add(m.group(1).strip().lower().rstrip("."))
        return domains

    def _poll_dns_cache(self) -> None:
        self._initial_cache = {d for d in self._read_dns_cache() if _match_scenario(d)}
        prev_seen = set(self._initial_cache)
        while self._active:
            try:
                current = {d for d in self._read_dns_cache() if _match_scenario(d)}
                newly_appeared = current - prev_seen
                for d in newly_appeared:
                    scenario = _match_scenario(d)
                    if scenario:
                        self._maybe_fire(scenario, source=f"dns-cache:{d}")
                prev_seen = current
            except Exception:
                pass
            time.sleep(_POLL_INTERVAL)

    def _poll_connections(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        prev_remote_ips: Set[str] = set()
        while self._active:
            try:
                conns = psutil.net_connections(kind="inet")
                current_ips: Set[str] = set()
                for c in conns:
                    if c.raddr and c.raddr.ip:
                        current_ips.add(c.raddr.ip)
                # Newly opened connections to watched IPs
                for ip in current_ips - prev_remote_ips:
                    scenario = self._domain_ips.get(ip)
                    if scenario:
                        self._maybe_fire(scenario, source=f"tcp:{ip}")
                prev_remote_ips = current_ips
            except Exception:
                pass
            time.sleep(_POLL_INTERVAL)

    def _scapy_handler(self, pkt) -> None:
        try:
            from scapy.layers.dns import DNS, DNSQR
            if not (pkt.haslayer(DNS) and pkt[DNS].qr == 0 and pkt.haslayer(DNSQR)):
                return
            raw = pkt[DNSQR].qname
            qname = (raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)).rstrip(".").lower()
            scenario = _match_scenario(qname)
            if scenario:
                self._maybe_fire(scenario, source=f"scapy:{qname}")
        except Exception:
            pass

    def _scapy_loop(self) -> None:
        try:
            from scapy.all import sniff
            while self._active:
                sniff(filter="udp port 53", prn=self._scapy_handler, store=False, timeout=5)
        except Exception:
            pass

    def start(self) -> None:
        self._active = True
        self._resolve_watched()
        print(
            f"[DomainWatcher] resolved {len(self._domain_ips)} IPs for "
            f"{len(WATCH_DOMAINS)} watched domains",
            flush=True,
        )
        threading.Thread(target=self._poll_dns_cache,   daemon=True, name="dw-dns-cache").start()
        threading.Thread(target=self._poll_connections, daemon=True, name="dw-conn").start()
        threading.Thread(target=self._scapy_loop,       daemon=True, name="dw-scapy").start()

    def stop(self) -> None:
        self._active = False
