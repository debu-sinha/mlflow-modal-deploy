"""Run live integration scenarios against the configured Modal account.

Usage: python tests/e2e_test.py [--quick | --stream | --proxy-auth]
All scenarios run by default, including proxy authentication. Set
PROXY_AUTH_TOKEN_ID and PROXY_AUTH_TOKEN_SECRET for the proxy-auth scenario.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--stream", action="store_true")
    group.add_argument("--proxy-auth", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, "-m", "pytest", str(Path(__file__).with_name("test_integration.py")), "-v", "-s"]
    if args.quick:
        command.extend(["-k", "basic"])
    elif args.stream:
        command.extend(["-k", "streaming"])
    elif args.proxy_auth:
        command.extend(["-k", "proxy-auth"])
    return subprocess.call(command, env={**os.environ, "TEST_MODAL_INTEGRATION": "1"})


if __name__ == "__main__":
    sys.exit(main())
