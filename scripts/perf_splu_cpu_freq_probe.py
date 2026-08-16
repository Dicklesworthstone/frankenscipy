#!/usr/bin/env python3
"""Sample PER-ARM CPU frequency exposure during a `perf_splu` run.

WHY THIS EXISTS. The fleet measured a cross-core frequency spread of 2.879x
(1429-3946 MHz simultaneously) and identified it, rather than ambient load, as the
reason ratios move between windows. That lands on `perf_splu_balanced_square.rs` in a
specific way the A/A null CANNOT catch: the two arms are different PROCESSES -- the
FrankenSciPy arm is the parent, the SciPy arm is a child Python interpreter -- so they
can occupy different cores at different frequencies, and each arm would still be
perfectly self-consistent while the ratio between them was biased.

WHAT IT MEASURES. It launches the harness, then samples both the parent's and the
child's current CPU (`/proc/<pid>/stat` field 39) and that CPU's
`scaling_cur_freq`, and reports each arm's frequency distribution and their ratio.

HOW TO READ IT. `PER-ARM MEAN MHz ratio` near 1.0 means both arms sampled the same
frequency distribution and the cross-core spread cancels in the ratio. A ratio far from
1.0 means the row it accompanies is biased by clock, not by code, and is not reportable.

Usage: python3 scripts/perf_splu_cpu_freq_probe.py [perf_splu args...]
"""
import collections
import glob
import os
import subprocess
import sys
import time

BIN = "./target/release/perf_splu"
SAMPLE_SECONDS = 0.01
DEADLINE = 900


def cpu_of(pid):
    """Current CPU of `pid`, from field 39 of /proc/<pid>/stat."""
    try:
        raw = open(f"/proc/{pid}/stat").read()
        # `comm` can contain spaces and parentheses, so split after the last ')'.
        return int(raw[raw.rindex(")") + 2:].split()[36])
    except Exception:
        return None


def mhz(cpu):
    try:
        path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        return int(open(path).read()) / 1000
    except Exception:
        return None


def children(pid):
    kids = []
    for path in glob.glob(f"/proc/{pid}/task/*/children"):
        try:
            kids.extend(open(path).read().split())
        except Exception:
            pass
    return kids


def summarise(label, cpus, freqs):
    if not freqs:
        print(f"{label}: no samples")
        return None
    ordered = sorted(freqs)
    mean = sum(freqs) / len(freqs)
    print(
        f"{label}: samples={len(freqs)} distinct_cpus={len(cpus)} "
        f"MHz min={ordered[0]:.0f} med={ordered[len(ordered)//2]:.0f} "
        f"max={ordered[-1]:.0f} mean={mean:.0f}"
    )
    return mean


def main():
    args = sys.argv[1:] or ["16", "11", "8", "off", "cubic", "on", "off", "off"]
    proc = subprocess.Popen(
        [BIN, *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "CARGO_TARGET_DIR": ""},
    )
    parent_cpus, child_cpus = collections.Counter(), collections.Counter()
    parent_freqs, child_freqs = [], []
    started = time.time()
    while proc.poll() is None and time.time() - started < DEADLINE:
        cpu = cpu_of(proc.pid)
        if cpu is not None:
            parent_cpus[cpu] += 1
            freq = mhz(cpu)
            if freq:
                parent_freqs.append(freq)
        for kid in children(proc.pid):
            cpu = cpu_of(kid)
            if cpu is not None:
                child_cpus[cpu] += 1
                freq = mhz(cpu)
                if freq:
                    child_freqs.append(freq)
        time.sleep(SAMPLE_SECONDS)
    proc.wait()

    fsci = summarise("FrankenSciPy arm (parent)", parent_cpus, parent_freqs)
    scipy = summarise("SciPy arm (python child) ", child_cpus, child_freqs)
    if fsci and scipy:
        print(f"\nPER-ARM MEAN MHz: fsci={fsci:.0f} scipy={scipy:.0f} ratio={fsci/scipy:.4f}x")


if __name__ == "__main__":
    main()
