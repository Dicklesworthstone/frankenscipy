#!/usr/bin/env python3
"""Bounded execution of the #19 ownership probe or the full sparse test suite."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

RESULT = re.compile(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;")


def run(command: list[str], cwd: Path, output: Path, name: str,
        seconds: int, env: dict[str, str]) -> dict:
    log = output / f"{name}.log"
    started = time.monotonic()
    record = {"command": command, "timeout_seconds": seconds, "log": str(log)}
    print(f"{name}: {' '.join(command)}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            record["returncode"] = process.wait(timeout=seconds)
            record["timed_out"] = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            record.update(returncode=process.returncode, timed_out=True)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    text = log.read_text(encoding="utf-8", errors="replace")
    record["test_results"] = [dict(passed=int(p), failed=int(f), ignored=int(i))
                              for p, f, i in RESULT.findall(text)]
    (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if record["returncode"] != 0:
        print(text[-20000:], flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(RUSTFLAGS="-D warnings", CARGO_PROFILE_TEST_DEBUG="0",
               CARGO_PROFILE_DEV_DEBUG="0", CARGO_TERM_COLOR="never")
    # The child marker must never turn the parent watchdog into an unbounded test.
    env.pop("FSCI_SPARSE_ISSUE_19_REGRESSION_CHILD", None)
    results = {"status": "incomplete", "runs": []}

    def execute(command: list[str], name: str, seconds: int,
                run_env: dict[str, str] | None = None) -> dict:
        item = run(command, repo, output, name, seconds, run_env or env)
        results["runs"].append(item)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        return item

    def successful(item: dict) -> bool:
        return item["returncode"] == 0 and not item["timed_out"]

    if args.probe:
        build = execute(["cargo", "test", "--locked", "-p", "fsci-sparse", "--lib",
                         "--no-run", "--message-format=json"], "probe-build", 1800)
        if not successful(build):
            return 1
        executables = set()
        for line in Path(build["log"]).read_text().splitlines():
            try:
                artifact = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (artifact.get("reason") == "compiler-artifact"
                    and artifact.get("target", {}).get("name") == "fsci_sparse"
                    and "lib" in artifact.get("target", {}).get("kind", [])
                    and artifact.get("profile", {}).get("test")
                    and artifact.get("executable")):
                executables.add(artifact["executable"])
        if len(executables) != 1:
            raise RuntimeError(f"expected one sparse test executable: {executables}")
        probe = execute([next(iter(executables)), "--exact",
                         "linalg::tests::issue_19_original_scaffolding_abba_probe",
                         "--ignored", "--nocapture", "--test-threads=1"], "ownership-cycle", 20)
        text = Path(probe["log"]).read_text()
        markers = ["ISSUE_19_ABBA_CONFIRMED:",
                   "issue19-one-column: owns BANDED_TOGGLE_LOCK",
                   "issue19-one-column: waiting for PERF_TOGGLE_TEST_LOCK",
                   "issue19-doubled-compare: owns PERF_TOGGLE_TEST_LOCK",
                   "issue19-doubled-compare: waiting for BANDED_TOGGLE_LOCK"]
        if probe["timed_out"] or probe["returncode"] != 101 or not all(m in text for m in markers):
            return 1
        results["status"] = "negative_control_confirmed"
    else:
        build = execute(["cargo", "test", "--locked", "-p", "fsci-sparse", "--no-run"],
                        "build", 1800)
        if not successful(build):
            return 1
        # Unfiltered runs preserve every numerical assertion and the existing ignored set.
        for index, threads in enumerate([32, 32, 32, 4, 1], 1):
            item = execute(["cargo", "test", "--locked", "-p", "fsci-sparse", "--",
                            f"--test-threads={threads}"], f"full-{index}-threads-{threads}",
                           900, {**env, "RUST_TEST_THREADS": str(threads)})
            full = any(r["passed"] >= 705 and r["failed"] == 0 and r["ignored"] == 50
                       for r in item["test_results"])
            if not successful(item) or not full:
                return 1
        for name, command in [
            ("format", ["cargo", "fmt", "--all", "--", "--check"]),
            ("sparse-check", ["cargo", "check", "--locked", "-p", "fsci-sparse", "--all-targets"]),
            ("sparse-clippy", ["cargo", "clippy", "--locked", "-p", "fsci-sparse",
                               "--all-targets", "--", "-D", "warnings"]),
        ]:
            if not successful(execute(command, name, 1800)):
                return 1
        results["status"] = "passed"
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(results["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
