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
frequency distribution and the cross-core spread cancels in the ratio.

DO NOT DECIDE FROM ONE PROBE (frankenscipy-llywn, 2026-08-17). This script used to be
read as: one ratio outside 2% means the accompanying row is clock-biased and not
reportable. That rule was wrong and it cost two refused cells and one banked mechanism
that later failed its own falsification test. The same cell at identical settings read
0.9532 in one window and 0.9942 / 1.0050 / 0.9969 / 0.9933 in four probes in the next —
the reading's spread is comparable to the bar it is being compared against. Collect at
least three probes and decide with `--gate r1 r2 r3 ...`, which reports PASS, FAIL, or
UNDECIDED on the MEDIAN. Fewer than three probes is UNDECIDED, which is not a pass.

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


def ramp_profile(aged, edges=(0.02, 0.05, 0.10)):
    """Mean MHz by AGE WITHIN A CONTIGUOUS RUNNING BURST.

    THE HYPOTHESIS THIS TESTS (frankenscipy-llywn, 2026-08-17). Sides 10 and 14 of the
    scattered family were refused because their per-arm clock ratio failed the 2% bar --
    0.9532 and 0.9655 -- and the failure was NOT random: the ratio rose monotonically with
    both sample count and fixture size across six probes (side=10 0.8789→0.9532, side=14
    0.9330→0.9655, side=20 0.9598→0.9822). The mechanism that predicts exactly that shape
    is a governor ramp: the sampler counts only state `R`, FrankenSciPy is the faster arm
    on this family, so it runs for less wall time per burst and a larger fraction of its
    samples land before the core has boosted. `scaling_governor=powersave` makes that ramp
    real rather than hypothetical.

    IF THE HYPOTHESIS IS RIGHT this profile rises with burst age. If frequency is flat
    across age, the ramp explanation is dead and the asymmetry is something else -- which
    is the outcome that would refute the banked row, and is why the buckets are reported
    even when they say nothing.

    `aged` is a list of `(burst_age_seconds, mhz)`.
    """
    buckets = collections.OrderedDict()
    labels = [f"<{int(1000 * edges[0])}ms"]
    labels += [f"{int(1000 * lo)}-{int(1000 * hi)}ms" for lo, hi in zip(edges, edges[1:])]
    labels.append(f">={int(1000 * edges[-1])}ms")
    for label in labels:
        buckets[label] = []
    for age, freq in aged:
        index = sum(1 for edge in edges if age >= edge)
        buckets[labels[index]].append(freq)
    return buckets


def ratio_after_discard(parent_aged, child_aged, discard):
    """Per-arm MHz ratio counting only samples at least `discard` seconds into a burst.

    SYMMETRIC BY CONSTRUCTION: the same threshold is applied to both arms, so this cannot
    manufacture agreement by trimming one side. Returns `(ratio, n_parent, n_child)`, or
    `(None, ...)` when either arm has nothing left -- reporting a ratio over an empty arm
    is how a gate starts passing everything.
    """
    kept_p = [freq for age, freq in parent_aged if age >= discard]
    kept_c = [freq for age, freq in child_aged if age >= discard]
    if not kept_p or not kept_c:
        return None, len(kept_p), len(kept_c)
    mean_p = sum(kept_p) / len(kept_p)
    mean_c = sum(kept_c) / len(kept_c)
    return mean_p / mean_c, len(kept_p), len(kept_c)


CLOCK_BAR = 0.02
MIN_GATE_PROBES = 3


def clock_gate(ratios, bar=CLOCK_BAR, minimum=MIN_GATE_PROBES):
    """Decide clock bias from the MEDIAN OF SEVERAL probes, never from one.

    WHY THIS REPLACES THE SINGLE-PROBE CHECK (frankenscipy-llywn, 2026-08-17). The old
    gate refused a row whenever one probe fell outside 2%. That gate turned out to be
    unreproducible: side=10 at identical settings read 0.9532 in one window and
    0.9942 / 1.0050 / 0.9969 / 0.9933 in four probes in the next. On the strength of the
    single failing read I refused two cells and banked a mechanism -- a governor ramp --
    that later failed its own falsification test in both directions. Both mistakes trace
    to the same root: deciding from one draw of a quantity whose spread was never measured.

    That is the identical error the replicate convention fixed for RATIOS one day earlier,
    and it was not carried across to the gate. This carries it across.

    REFUSING TO DECIDE IS A VERDICT. With fewer than `minimum` probes this returns
    `UNDECIDED`, not a pass -- a gate that silently passes on thin evidence is worse than
    one that fails, because nothing downstream can tell the two apart.

    Returns `(verdict, median, n, spread)` with verdict in {PASS, FAIL, UNDECIDED}.
    """
    n = len(ratios)
    if n < minimum:
        return "UNDECIDED", None, n, None
    ordered = sorted(ratios)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    spread = ordered[-1] - ordered[0]
    verdict = "PASS" if abs(median - 1.0) <= bar else "FAIL"
    return verdict, median, n, spread


def run_clock_gate(raw):
    """`perf_splu_cpu_freq_probe.py --gate <ratio> <ratio> ...`"""
    ratios = []
    for value in raw:
        try:
            parsed = float(value)
        except ValueError:
            return f"clock ratio must be a number, got {value!r}"
        if not parsed > 0 or parsed != parsed or parsed in (float("inf"),):
            return f"clock ratio must be finite and positive, got {parsed}"
        ratios.append(parsed)
    verdict, median, n, spread = clock_gate(ratios)
    if verdict == "UNDECIDED":
        print(f"clock_gate: UNDECIDED n={n} -- needs at least {MIN_GATE_PROBES} probes. "
              "One probe cannot decide clock bias; this is not a pass.")
        return None
    print(f"clock_gate: {verdict} median={median:.4f} n={n} spread={spread:.4f} "
          f"bar=+/-{CLOCK_BAR}")
    if spread > 2 * CLOCK_BAR:
        print(f"  WARNING: spread {spread:.4f} exceeds twice the bar, so the probes "
              "disagree by more than the thing being tested. Treat the median as weak "
              "and collect more probes before relying on this verdict.")
    return None


def selftest():
    """Exercise the ramp logic on synthetic data, with an arm that must show a ramp and
    an arm that must not. Runs no measurement, so it is safe under a build/measure freeze.
    """
    # MUST SHOW A RAMP: frequency climbs with burst age.
    ramped = [(0.00, 1400), (0.01, 1400), (0.03, 3000), (0.06, 3900), (0.20, 4000)]
    profile = ramp_profile(ramped)
    means = [sum(v) / len(v) for v in profile.values() if v]
    assert means == sorted(means), f"a ramped arm must be non-decreasing in age: {means}"
    assert means[0] < means[-1], f"a ramped arm must actually rise: {means}"

    # MUST NOT: a flat arm. If this also read as rising, the bucketing would be inventing
    # a ramp out of bucket boundaries and every conclusion drawn from it would be false.
    flat = [(0.00, 3800), (0.01, 3800), (0.03, 3800), (0.06, 3800), (0.20, 3800)]
    flat_means = [sum(v) / len(v) for v in ramp_profile(flat).values() if v]
    assert max(flat_means) - min(flat_means) == 0, f"flat arm must stay flat: {flat_means}"

    # Discarding the ramp window must pull a ramped-vs-flat pair TOWARD 1.0, which is the
    # precise prediction the banked row made falsifiable.
    before, _, _ = ratio_after_discard(ramped, flat, 0.0)
    after, _, _ = ratio_after_discard(ramped, flat, 0.05)
    assert before < after <= 1.05, f"discard must move the ratio toward 1.0: {before} -> {after}"

    # An over-aggressive discard empties an arm, and that must report None rather than a
    # confident number over nothing.
    empty, np_, nc_ = ratio_after_discard(ramped, flat, 10.0)
    assert empty is None and np_ == 0 and nc_ == 0, "an emptied arm must not yield a ratio"

    # --- clock gate: must pass, must fail, must refuse, and must not be swayed by one draw

    # MUST PASS: three probes whose median sits inside the bar.
    verdict, median, n, spread = clock_gate([0.9942, 1.0050, 0.9969])
    assert verdict == "PASS", f"tight passing probes must PASS, got {verdict} {median}"
    assert n == 3 and spread is not None

    # MUST FAIL: a median outside the bar.
    verdict, _, _, _ = clock_gate([0.9500, 0.9532, 0.9610])
    assert verdict == "FAIL", f"a median outside the bar must FAIL, got {verdict}"

    # MUST REFUSE: fewer than three probes is UNDECIDED, never PASS. This is the exact
    # shape of the mistake being fixed -- one probe read 0.9532 and two cells were refused
    # and a mechanism banked on it.
    for thin in ([], [0.9942], [0.9942, 1.0050]):
        verdict, median, n, _ = clock_gate(thin)
        assert verdict == "UNDECIDED", f"n={n} must be UNDECIDED, got {verdict}"
        assert median is None, "an undecided gate must not report a median"

    # ONE OUTLIER MUST NOT FLIP A VERDICT IN EITHER DIRECTION. The median exists precisely
    # so that a single unreproducible draw cannot decide, which is what happened for real.
    verdict, _, _, _ = clock_gate([0.9942, 1.0050, 0.9969, 0.9933, 0.9532])
    assert verdict == "PASS", "one low outlier among passing probes must not force FAIL"
    verdict, _, _, _ = clock_gate([0.9500, 0.9532, 0.9610, 0.9480, 1.0050])
    assert verdict == "FAIL", "one high outlier among failing probes must not force PASS"

    print("selftest: OK -- ramp detected, flat arm stayed flat, discard moved the ratio "
          "toward 1.0, emptied arm refused; clock gate passes/fails/refuses correctly and "
          "is not flipped by a single outlier")


def main():
    args = sys.argv[1:] or ["16", "11", "8", "off", "cubic", "on", "off", "off"]
    if args and args[0] == "--selftest":
        selftest()
        return
    if args and args[0] == "--gate":
        error = run_clock_gate(args[1:])
        if error:
            print(f"invalid --gate invocation: {error}", file=sys.stderr)
            sys.exit(2)
        return
    proc = subprocess.Popen(
        [BIN, *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "CARGO_TARGET_DIR": ""},
    )
    parent_cpus, child_cpus = collections.Counter(), collections.Counter()
    parent_freqs, child_freqs = [], []
    parent_running, child_running = [], []
    # `(age within the current contiguous R burst, MHz)` per arm. A burst ends the moment
    # the arm is seen in any state other than R, because the arms are time-interleaved and
    # each hand-off lets the core clock back down.
    parent_aged, child_aged = [], []
    parent_burst_start, child_burst_start = None, None
    co_resident = samples = 0
    siblings = smt_siblings()
    started = time.time()
    while proc.poll() is None and time.time() - started < DEADLINE:
        pstate, pcpu = state_and_cpu(proc.pid)
        if pcpu is not None:
            samples += 1
            parent_cpus[pcpu] += 1
            freq = mhz(pcpu)
            now = time.time()
            if pstate == "R":
                if parent_burst_start is None:
                    parent_burst_start = now
            else:
                parent_burst_start = None
            if freq:
                parent_freqs.append(freq)
                if pstate == "R":
                    parent_running.append(freq)
                    parent_aged.append((now - parent_burst_start, freq))
        for kid in children(proc.pid):
            cstate, ccpu = state_and_cpu(kid)
            if ccpu is None:
                continue
            child_cpus[ccpu] += 1
            freq = mhz(ccpu)
            now = time.time()
            if cstate == "R":
                if child_burst_start is None:
                    child_burst_start = now
            else:
                child_burst_start = None
            if freq:
                child_freqs.append(freq)
                if cstate == "R":
                    child_running.append(freq)
                    child_aged.append((now - child_burst_start, freq))
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

        # IS THE GATE MEASURING ITS OWN SAMPLING? (frankenscipy-llywn, 2026-08-17)
        #
        # On the scattered fixture the gate FAILS reproducibly -- at 61% idle and again at
        # 90% idle -- while the cubic gate passes comfortably on the same host in the same
        # sessions. The arms do not contribute equally: FrankenSciPy is ~1.6x faster there,
        # so its arm occupies proportionally less wall time and supplies ~620 running
        # samples against SciPy's ~1000. If the mean ratio is an artefact of that asymmetry
        # rather than of clock, matching the sample counts should move it toward 1.0.
        #
        # Two controls on the same data: the MEDIAN, which is insensitive to a long tail on
        # one side, and a COUNT-MATCHED mean that subsamples the longer arm uniformly.
        def median(values):
            ordered = sorted(values)
            mid = len(ordered) // 2
            return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])

        def uniform_subsample(values, keep):
            if keep >= len(values):
                return values
            step = len(values) / keep
            return [values[int(i * step)] for i in range(keep)]

        keep = min(len(parent_running), len(child_running))
        pm = uniform_subsample(parent_running, keep)
        cm = uniform_subsample(child_running, keep)
        print(f"  median ratio      = {median(parent_running) / median(child_running):.4f}")
        print(f"  count-matched     = {(sum(pm) / len(pm)) / (sum(cm) / len(cm)):.4f}  "
              f"(both arms subsampled to n={keep})")
        print("  If these sit closer to 1.0 than the mean ratio, the gate is partly "
              "measuring sample-count asymmetry rather than clock.")

    # THE RAMP TEST. Reported unconditionally, including when it says nothing, because a
    # flat profile REFUTES the boost-ramp explanation banked for the sides 10/14 refusal
    # and that refutation is as valuable as a confirmation.
    if parent_aged and child_aged:
        print("\nMHz BY AGE WITHIN A RUNNING BURST (does the governor ramp?):")
        for arm, aged in (("fsci ", parent_aged), ("scipy", child_aged)):
            cells = []
            for label, values in ramp_profile(aged).items():
                cells.append(
                    f"{label}={sum(values) / len(values):.0f}(n={len(values)})"
                    if values else f"{label}=-"
                )
            print(f"  {arm}  " + "  ".join(cells))
        print("\nRATIO AFTER DISCARDING THE RAMP WINDOW (same threshold on BOTH arms):")
        for discard in (0.0, 0.02, 0.05, 0.10):
            ratio, np_, nc_ = ratio_after_discard(parent_aged, child_aged, discard)
            if ratio is None:
                print(f"  discard>={int(1000 * discard):3d}ms  ratio=-  "
                      f"(fsci n={np_}, scipy n={nc_}) -- an emptied arm yields no ratio")
            else:
                print(f"  discard>={int(1000 * discard):3d}ms  ratio={ratio:.4f}  "
                      f"(fsci n={np_}, scipy n={nc_})")
        print("  PREDICTION ON RECORD: if the ramp explains the sides 10/14 refusal, the "
              "ratio rises toward 1.0 with the discard threshold on a FAST fixture and "
              "stays put on a slow one. If it is flat in both, the hypothesis is dead and "
              "the banked row must be corrected.")


if __name__ == "__main__":
    main()
