#!/usr/bin/env python3
"""Aggregates FSCI-P2C-018 diff and metamorphic logs into the root parity report triple.

Invokes the native Rust binary `aggregate_p2c018` to produce:
  - crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-018/parity_report.json
  - crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-018/parity_report.raptorq.json
  - crates/fsci-conformance/fixtures/artifacts/FSCI-P2C-018/parity_report.decode_proof.json
"""

import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    cmd = ["cargo", "run", "-p", "fsci-conformance", "--bin", "aggregate_p2c018"]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=repo_root)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
