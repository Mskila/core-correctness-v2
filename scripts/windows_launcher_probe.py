"""Dependency-free checks used by the Windows launcher scripts."""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


def port_is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def health_is_ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1.0) as response:
            payload = json.load(response)
        return payload.get("status") == "ok"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def wait_for_existing_service(port: int, seconds: int) -> int:
    """Return 0 when free, 2 for AlphaMaster, and 1 for another listener."""
    if not port_is_listening(port):
        return 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if health_is_ready(port):
            return 2
        time.sleep(1.0)
    return 1


def show_listeners(port: int) -> None:
    result = subprocess.run(
        ["netstat.exe", "-ano", "-p", "TCP"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    pattern = re.compile(
        rf"^\s*TCP\s+\S+:{port}\s+\S+\s+LISTENING\s+\d+\s*$",
        re.IGNORECASE,
    )
    matches = [line for line in result.stdout.splitlines() if pattern.match(line)]
    if matches:
        print("\n".join(matches))
    else:
        print("No listener is present now; the port use was probably temporary.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("wait", "show"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seconds", type=int, default=15)
    args = parser.parse_args()
    if args.action == "show":
        show_listeners(args.port)
        return 0
    return wait_for_existing_service(args.port, args.seconds)


if __name__ == "__main__":
    sys.exit(main())
