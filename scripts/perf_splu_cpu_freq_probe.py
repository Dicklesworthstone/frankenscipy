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


def state_and_cpu(pid):
    """(run state, current CPU) of `pid`, from fields 3 and 39 of /proc/<pid>/stat.

    THE STATE IS NOT OPTIONAL. The two arms are time-interleaved, so while one arm is
    timing the other is BLOCKED on the pipe, and a blocked process sits on a core the
    governor has clocked down. Averaging frequency over all samples therefore measures
    "how idle was the other arm", not "how fast did this arm run" -- it read 1.1866x on
    one run and 1.0079x on another for the same machine. Conditioning on state == "R"
    gives the frequency each arm actually experienced while doing its timed work.
    """
    try:
        raw = open(f"/proc/{pid}/stat").read()
        # `comm` can contain spaces and parentheses, so split after the last ')'.
        rest = raw[raw.rindex(")") + 2:].split()
        return rest[0], int(rest[36])
    except Exception:
        return None, None


def mhz(cpu):
    try:
        path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        return int(open(path).read()) / 1000
    except Exception:
        return None


def smt_siblings():
    """logical CPU -> its SMT sibling group, from thread_siblings_list."""
    import re
    out = {}
    for path in glob.glob("/sys/devices/system/cpu/cpu*/topology/thread_siblings_list"):
        match = re.search(r"/cpu(\d+)/", path)
        if not match:
            continue
        group = []
        for part in open(path).read().strip().split(","):
            if "-" in part:
                low, high = part.split("-")
                group.extend(range(int(low), int(high) + 1))
            else:
                group.append(int(part))
        out[int(match.group(1))] = sorted(group)
    return out


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
    parent_running, child_running = [], []
    co_resident = samples = 0
    siblings = smt_siblings()
    started = time.time()
    while proc.poll() is None and time.time() - started < DEADLINE:
        pstate, pcpu = state_and_cpu(proc.pid)
        if pcpu is not None:
            samples += 1
            parent_cpus[pcpu] += 1
            freq = mhz(pcpu)
            if freq:
                parent_freqs.append(freq)
                if pstate == "R":
                    parent_running.append(freq)
        for kid in children(proc.pid):
            cstate, ccpu = state_and_cpu(kid)
            if ccpu is None:
                continue
            child_cpus[ccpu] += 1
            freq = mhz(ccpu)
            if freq:
                child_freqs.append(freq)
                if cstate == "R":
                    child_running.append(freq)
            if pcpu is not None and ccpu in siblings.get(pcpu, ()) and ccpu != pcpu:
                co_resident += 1
        time.sleep(SAMPLE_SECONDS)
    proc.wait()

    summarise("FrankenSciPy arm (parent), all samples", parent_cpus, parent_freqs)
    summarise("SciPy arm (python child), all samples  ", child_cpus, child_freqs)
    print(f"\nSMT co-residency (arms on siblings of one physical core): "
          f"{100 * co_resident / max(samples, 1):.1f}% of samples")
    if parent_running and child_running:
        fr = sum(parent_running) / len(parent_running)
        cr = sum(child_running) / len(child_running)
        print(f"\nRUNNING-ONLY per-arm MHz (the figure to record):")
        print(f"  fsci={fr:.0f} (n={len(parent_running)})  "
              f"scipy={cr:.0f} (n={len(child_running)})  ratio={fr/cr:.4f}x")
        print("  A ratio within ~2% of 1.0 means the cross-core spread cancels; "
              "outside it, the row is clock-biased and must be refused.")


if __name__ == "__main__":
    main()
