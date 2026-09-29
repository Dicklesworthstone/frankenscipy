#!/usr/bin/env python3
"""FrankenSciPy performance scoreboard at HEAD against LIVE SciPy (frankenscipy-sw4p0.2).

One file, two halves, one self-test:

  run      Execute ONE live-SciPy harness binary on this host -- pinned to one logical CPU
           (`taskset -c N`, which the SciPy child inherits) or unpinned -- and capture, into
           an artifact directory:
             <tag>.log        the harness's stdout+stderr, byte-for-byte (the raw evidence)
             <tag>.lines.tsv  arrival time of every log line, seconds since launch
             <tag>.trace.tsv  1 Hz host trace: loadavg, runnable tasks split into OURS (the
                              harness process tree) and FOREIGN, pinned-CPU / SMT-sibling /
                              host-mean busy, observed thread counts of both arms
             <tag>.meta.json  provenance: host, governor, ISA, executed-ELF sha256 (hashed
                              shell-side, to cross-check the harness's self-report), the
                              incumbent interpreter's sha256, argv, env overlay, exit status

  build    Parse every captured run, normalise every ratio to ONE convention, validate the
           provenance of every row, classify it, and write the scoreboard markdown + JSON.

  --self-test   Classifier (CI-overlap rules, NaN handling), ratio normaliser (both input
                conventions), provenance validator (missing sha / worker -> INVALID), the
                parsers, and both negative cases of the bead.

RATIO CONVENTION (the only one this file emits): ratio = SciPy time / fsci time, so > 1 means
FrankenSciPy is faster. Harnesses that print fsci/SciPy (perf_fft_vs_scipy's
`fsci_over_scipy`, perf_eigh_vs_scipy's `fsci/scipy1`) are inverted here, CI endpoints swapped.

CLASSES, decided in this order (the first that applies wins):
  INVALID          provenance fails: no self-reported fsci ELF sha256, self-report disagrees
                   with the executed binary, no measuring host, incumbent not the pinned
                   genuine SciPy 1.17.1 / numpy 2.4.3, or the SciPy arm was NOT run live in
                   the same invocation (cached/copied numbers land here, never in WIN).
  SELF_COMPARISON  the incumbent arm's executable sha256 equals the fsci ELF sha256: an
                   fsci-vs-fsci comparison. Maintenance at best; never a WIN or LOSE.
  UNRESOLVED       ratio or interval non-finite; an A/A null missing or outside its band; the
                   host above the load ceiling; the harness itself refused or voided the row;
                   or the interval overlaps 1.0.
  WIN / LOSE       interval entirely above / below 1.0 with every gate above passed.

Nothing in the build step re-measures, retunes, or drops a row. A row the harness printed is a
row on the board, in one of the five classes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── Gates, stated once. Changing any of these is a change to what a row means. ──────────────
PINNED_SCIPY = "1.17.1"
PINNED_NUMPY = "2.4.3"
# Host load ceiling, applied to THREE readings, all of which must be at or below it:
#   (a) the 1-min loadavg sampled immediately before the harness is launched;
#   (b) the median, over the row's own time window, of the 1-min loadavg minus the harness
#       tree's own running threads at that sample (the loadavg the row was measured under,
#       net of what the row itself contributed);
#   (c) the median, over the same window, of FOREIGN runnable tasks (host `procs_running`
#       minus the harness tree's running threads minus the sampler).
# 20 is 31% of this host's 64 logical CPUs: at or below it a one-thread arm has idle cores.
LOAD_CEILING = 20.0
# A/A null bands. A "centered" null is a first-half/second-half (or A/A pair) median ratio,
# expected 1.0: it must sit within +/-2% (the repo's registered band). A "spread" null is the
# median over rounds of max/min of an arm's two samples, >= 1 by construction: it must be
# <= 1.05 (perf_chol_vs_scipy's own threshold for that statistic).
NULL_CENTERED_BAND = 0.02
NULL_SPREAD_MAX = 1.05
BOOTSTRAP_ITERS = 4000
# Elementwise max relative disagreement above which a row is listed (never reclassified).
AGREEMENT_FLAG = 1e-6

CLASSES = ("WIN", "LOSE", "UNRESOLVED", "SELF_COMPARISON", "INVALID")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ════════════════════════════════════════════════════════════════════════════════════════════
# Pure statistics
# ════════════════════════════════════════════════════════════════════════════════════════════

def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def median(values):
    vals = sorted(v for v in values)
    n = len(vals)
    if n == 0:
        return float("nan")
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def bootstrap_median_ci(values, iters=BOOTSTRAP_ITERS):
    """Deterministic percentile bootstrap of the median (seeded LCG, as the harnesses use).

    Any non-finite sample makes the interval (nan, nan): a NaN must never be sorted into a
    plausible-looking bound.
    """
    vals = list(values)
    if len(vals) < 3 or not all(finite(v) for v in vals):
        return (float("nan"), float("nan"))
    state = 0x2545F4914F6CDD1D
    meds = []
    n = len(vals)
    for _ in range(iters):
        sample = []
        for _ in range(n):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            sample.append(vals[(state >> 33) % n])
        meds.append(median(sample))
    meds.sort()
    return (meds[int(0.025 * iters)], meds[int(0.975 * iters)])


def normalise_ratio(value, convention, ci=None):
    """Return (ratio, (lo, hi)) in the SciPy-time / fsci-time convention.

    `convention` is what the harness printed: "scipy_over_fsci" passes through,
    "fsci_over_scipy" is inverted with the interval endpoints swapped. A zero, negative or
    non-finite input becomes NaN rather than a large or negative "ratio".
    """
    if convention not in ("scipy_over_fsci", "fsci_over_scipy"):
        raise ValueError(f"unknown ratio convention {convention!r}")

    def conv(x):
        if not finite(x) or x <= 0.0:
            return float("nan")
        return x if convention == "scipy_over_fsci" else 1.0 / x

    ratio = conv(value)
    if ci is None:
        return ratio, None
    lo, hi = conv(ci[0]), conv(ci[1])
    if convention == "fsci_over_scipy":
        lo, hi = hi, lo
    return ratio, (lo, hi)


def null_passes(value, kind):
    if not finite(value):
        return False
    if kind == "centered":
        return abs(value - 1.0) <= NULL_CENTERED_BAND
    if kind == "spread":
        return 1.0 <= value <= NULL_SPREAD_MAX
    raise ValueError(f"unknown null kind {kind!r}")


def null_envelope(ratio, null_fsci, null_scipy, kind):
    """Interval implied by the two arms' own A/A dispersion, for harnesses that print no CI
    and no per-round samples. The ratio can be off by each arm's null factor in either
    direction, so the envelope multiplies them. Conservative by construction."""
    if not (finite(ratio) and finite(null_fsci) and finite(null_scipy)):
        return (float("nan"), float("nan"))
    if kind == "spread":
        if null_fsci < 1.0 or null_scipy < 1.0:
            return (float("nan"), float("nan"))
        f = null_fsci * null_scipy
    elif kind == "centered":
        if null_fsci <= 0.0 or null_scipy <= 0.0:
            return (float("nan"), float("nan"))
        f = max(null_fsci, 1.0 / null_fsci) * max(null_scipy, 1.0 / null_scipy)
    else:
        raise ValueError(f"unknown null kind {kind!r}")
    return (ratio / f, ratio * f)


# ════════════════════════════════════════════════════════════════════════════════════════════
# Provenance validation and classification
# ════════════════════════════════════════════════════════════════════════════════════════════

def validate_provenance(row):
    """Return (status, reasons) with status in {"OK", "INVALID", "SELF_COMPARISON"}."""
    reasons = []
    sha = row.get("fsci_elf_sha256")
    if not (isinstance(sha, str) and HEX64.match(sha)):
        reasons.append("no self-reported fsci ELF sha256")
    executed = row.get("executed_elf_sha256")
    if isinstance(sha, str) and HEX64.match(sha):
        if not (isinstance(executed, str) and HEX64.match(executed)):
            reasons.append("executed binary sha256 not recorded")
        elif executed != sha:
            reasons.append("self-reported ELF sha256 differs from the executed binary")
    if not row.get("host"):
        reasons.append("no measuring host/worker recorded")
    if row.get("scipy_source") != "live_same_invocation":
        reasons.append(
            f"SciPy arm not live in the same invocation (source={row.get('scipy_source')!r})")
    inc = row.get("incumbent") or {}
    if not inc:
        reasons.append("no scipy_incumbent provenance line")
    else:
        if inc.get("genuine") is not True:
            reasons.append("incumbent not genuine")
        if inc.get("scipy") != PINNED_SCIPY or inc.get("numpy") != PINNED_NUMPY:
            reasons.append(
                f"incumbent scipy={inc.get('scipy')} numpy={inc.get('numpy')} "
                f"is not the pinned {PINNED_SCIPY}/{PINNED_NUMPY}")
    if not row.get("governor"):
        reasons.append("no CPU governor recorded")
    if not row.get("isa"):
        reasons.append("no runtime ISA recorded")
    if reasons:
        return "INVALID", reasons
    inc_sha = row.get("incumbent_elf_sha256")
    if isinstance(inc_sha, str) and inc_sha == sha:
        return "SELF_COMPARISON", [
            "incumbent executable sha256 equals the fsci ELF sha256: fsci-vs-fsci"]
    return "OK", []


def classify(row):
    """Return (class, reasons). Pure function of the row dict."""
    status, reasons = validate_provenance(row)
    if status != "OK":
        return status, reasons
    reasons = []
    ratio = row.get("ratio")
    ci = row.get("ci")
    if not finite(ratio):
        reasons.append("ratio not finite")
    if ci is None or not (finite(ci[0]) and finite(ci[1])):
        reasons.append("no finite interval for the ratio")
    kind = row.get("null_kind")
    for arm in ("fsci", "scipy"):
        value = row.get(f"null_{arm}")
        if value is None:
            reasons.append(f"no A/A null for the {arm} arm")
        elif not null_passes(value, kind):
            reasons.append(f"{arm} A/A null {value} outside its {kind} band")
    load = row.get("load") or {}
    ambient = load.get("ambient_loadavg1")
    during = load.get("loadavg1_net_median")
    foreign = load.get("foreign_running_median")
    if not finite(ambient) or ambient > LOAD_CEILING:
        reasons.append(f"ambient loadavg1 {ambient} above ceiling {LOAD_CEILING}")
    if not finite(during) or during > LOAD_CEILING:
        reasons.append(f"in-window loadavg1 (net of own running) {during} above ceiling "
                       f"{LOAD_CEILING}")
    if not finite(foreign) or foreign > LOAD_CEILING:
        reasons.append(f"foreign runnable median {foreign} above ceiling {LOAD_CEILING}")
    if row.get("harness_refused"):
        reasons.append(f"harness voided/refused the row: {row.get('harness_verdict')}")
    if reasons:
        return "UNRESOLVED", reasons
    lo, hi = ci
    if lo > 1.0:
        return "WIN", [f"interval [{lo:.4f},{hi:.4f}] entirely above 1"]
    if hi < 1.0:
        return "LOSE", [f"interval [{lo:.4f},{hi:.4f}] entirely below 1"]
    return "UNRESOLVED", [f"interval [{lo:.4f},{hi:.4f}] overlaps 1"]


# ════════════════════════════════════════════════════════════════════════════════════════════
# Log parsing. Every harness prints its own shape; each parser returns rows in the harness's
# OWN convention and names that convention, and nothing else.
# ════════════════════════════════════════════════════════════════════════════════════════════

NUM = r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|NaN|inf|-inf)"


def fnum(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def parse_header(lines):
    """Provenance common to every harness log."""
    out = {"fsci_elf_sha256": None, "incumbent": None, "ready_genuine": None,
           "scipy_engine_sha256": None}
    for line in lines:
        if out["fsci_elf_sha256"] is None:
            m = re.search(r"\belf_sha256=([0-9a-f]{64})\b", line)
            if m:
                out["fsci_elf_sha256"] = m.group(1)
        if line.startswith("scipy_incumbent:") and out["incumbent"] is None:
            fields = dict(re.findall(r"(\w+)=(\S+)", line))
            out["incumbent"] = {
                "python": fields.get("python"),
                "scipy": fields.get("scipy"),
                "numpy": fields.get("numpy"),
                "fsci_loaded": fields.get("fsci_loaded"),
                "genuine": fields.get("genuine") == "true",
                "blas": fields.get("blas"),
            }
        if "genuine=" in line and not line.startswith("scipy_incumbent:"):
            g = "genuine=True" in line
            out["ready_genuine"] = g if out["ready_genuine"] is None else (out["ready_genuine"] and g)
        m = re.search(r"\bscipy_engine_sha256=([0-9a-f]{64})\b", line)
        if m and out["scipy_engine_sha256"] is None:
            out["scipy_engine_sha256"] = m.group(1)
    return out


CASE_RE = re.compile(
    r"^case=(?P<case>\S+)(?P<mid>.*?) fsci=(?P<f>" + NUM + r")ms scipy=(?P<s>" + NUM
    + r")ms scipy/fsci=(?P<r>" + NUM + r")x null_fsci=(?P<nf>" + NUM + r") null_scipy=(?P<ns>"
    + NUM + r")(?P<tail>.*)$")


def parse_case_style(lines):
    """cluster / interpolate / ndimage / opt / signal / spatial / special / stats / eigsh.
    `case=... fsci=Xms scipy=Yms scipy/fsci=Rx null_fsci=A null_scipy=B <check>`; the nulls
    are medians over rounds of each arm's max/min pair ("spread"). Some harnesses emit result
    lines on stdout AND stderr, so identical lines are de-duplicated."""
    rows, seen = [], set()
    for no, line in enumerate(lines):
        m = CASE_RE.match(line.strip())
        if not m:
            continue
        key = line.strip()
        if key in seen:
            continue
        seen.add(key)
        mid = m.group("mid")
        op = re.search(r"\bop=(\S+)", mid)
        case = m.group("case") + (f" op={op.group(1)}" if op else "")
        tail = m.group("tail").strip()
        if not op:
            extra = " ".join(t for t in mid.split() if t.startswith(("n=", "k=", "which=")))
            case = (case + " " + extra).strip()
        rows.append({
            "case": case, "variant": "scipy", "line_no": no,
            "raw_value": fnum(m.group("r")), "convention": "scipy_over_fsci", "raw_ci": None,
            "raw_samples": None, "null_fsci": fnum(m.group("nf")),
            "null_scipy": fnum(m.group("ns")), "null_kind": "spread",
            "fsci_ms": fnum(m.group("f")), "scipy_ms": fnum(m.group("s")),
            "harness_verdict": None, "harness_refused": False, "agreement": tail,
        })
    return rows


FFT_RE = re.compile(
    r"^RESULT mode=(?P<mode>\S+) n=(?P<n>\d+) repeats=\d+ fsci_ms=(?P<f>" + NUM
    + r") scipy_ms=(?P<s>" + NUM + r") fsci_over_scipy=(?P<r>" + NUM + r") fsci_aa=(?P<nf>"
    + NUM + r") scipy_aa=(?P<ns>" + NUM + r")(?P<tail>.*)$")


def parse_fft(lines):
    rows = []
    for no, line in enumerate(lines):
        m = FFT_RE.match(line.strip())
        if not m:
            continue
        rows.append({
            "case": f"{m.group('mode')} n={m.group('n')}", "variant": "scipy", "line_no": no,
            "raw_value": fnum(m.group("r")), "convention": "fsci_over_scipy", "raw_ci": None,
            "raw_samples": None, "null_fsci": fnum(m.group("nf")),
            "null_scipy": fnum(m.group("ns")), "null_kind": "centered",
            "fsci_ms": fnum(m.group("f")), "scipy_ms": fnum(m.group("s")),
            "harness_verdict": None, "harness_refused": False,
            "agreement": "not checked: the two arms digest different algorithms' outputs",
        })
    return rows


def _list(text):
    return [fnum(v) for v in re.split(r"[,\s]+", text.strip().strip("[]")) if v]


def parse_bdf(lines):
    """perf_bdf_vs_scipy / perf_gmres_job_vs_scipy share the `Incumbent ratio:` + NULL lines."""
    fixture = None
    null_ours = null_scipy = None
    raw = None
    verdict = None
    rows = []
    for no, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^fixture=(\S+)(.*)$", s)
        if m:
            fields = dict(re.findall(r"(\w+)=(\S+)", m.group(2)))
            fixture = f"{m.group(1)} n={fields.get('n')} method={fields.get('method')}"
        m = re.match(r"^(?:\w+_)?NULL-ours(?: A/A)?\s+median=(" + NUM + r")", s)
        if m:
            null_ours = fnum(m.group(1))
        m = re.match(r"^(?:\w+_)?NULL-scipy(?: A/A)?\s+median=(" + NUM + r")", s)
        if m:
            null_scipy = fnum(m.group(1))
        m = re.search(r"raw_samples_seconds: ours=(.*?) scipy=(.*?) ratios=(.*?) null_ours=", s)
        if m:
            raw = _list(m.group(3))
        m = re.search(r"median-CI gate: .*=> (.+)$", s)
        if m:
            verdict = m.group(1).strip()
            if rows and rows[-1]["harness_verdict"] is None:
                rows[-1]["harness_verdict"] = verdict
                # NOT DECIDED and every PROVISIONAL (non-exclusive host) outcome are the
                # harness declining to decide; only its DECIDED tokens are not a refusal.
                rows[-1]["harness_refused"] = not verdict.startswith("DECIDED")
        m = re.match(r"^(Incumbent|Unpreconditioned) ratio: SciPy / FrankenSciPy = (" + NUM
                     + r")x \(bootstrap-median ci95=\[(" + NUM + r"),(" + NUM + r")\]", s)
        if m:
            rows.append({
                "case": (fixture or "default") + ("" if m.group(1) == "Incumbent"
                                                  else " unpreconditioned"),
                "variant": "scipy", "line_no": no, "raw_value": fnum(m.group(2)),
                "convention": "scipy_over_fsci",
                "raw_ci": (fnum(m.group(3)), fnum(m.group(4))), "raw_samples": raw,
                "null_fsci": null_ours, "null_scipy": null_scipy, "null_kind": "centered",
                "fsci_ms": None, "scipy_ms": None, "harness_verdict": None,
                "harness_refused": False, "agreement": None,
            })
    return rows


EIGH_LINE = re.compile(
    r"^(?P<label>NULL fsci/fsci|NULL sp1/sp1|NULL nalg/nalg|fsci/scipy1|fsci/scipyN)\s+a=\s*(?P<a>"
    + NUM + r")ms b=\s*(?P<b>" + NUM + r")ms ratio_p50=(?P<r>" + NUM + r")x ci95=\[(?P<lo>"
    + NUM + r"),(?P<hi>" + NUM + r")\]")


def parse_eigh(lines):
    rows = []
    block = None
    nulls = {}
    for no, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^--- n=(\d+) impl=(\S+) ---", s)
        if m:
            block = f"n={m.group(1)} impl={m.group(2)}"
            nulls = {}
            continue
        m = EIGH_LINE.match(s)
        if not m or block is None:
            continue
        label = m.group("label")
        if label.startswith("NULL"):
            nulls[label] = fnum(m.group("r"))
            continue
        variant = "scipy1" if label.endswith("scipy1") else "scipyN"
        rows.append({
            "case": block, "variant": variant, "line_no": no, "raw_value": fnum(m.group("r")),
            "convention": "fsci_over_scipy", "raw_ci": (fnum(m.group("lo")), fnum(m.group("hi"))),
            "raw_samples": None, "null_fsci": nulls.get("NULL fsci/fsci"),
            # scipyN has no A/A null of its own in this harness; only scipy1 does.
            "null_scipy": nulls.get("NULL sp1/sp1") if variant == "scipy1" else None,
            "null_kind": "centered", "fsci_ms": fnum(m.group("a")), "scipy_ms": fnum(m.group("b")),
            "harness_verdict": None, "harness_refused": False, "agreement": None,
        })
    return rows


def parse_chol(lines):
    reps = {}
    rows = []
    for no, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^n=(\d+) rep=\d+ fsci=(" + NUM + r")s scipy1=(" + NUM + r")s scipyN=("
                     + NUM + r")s r1=(" + NUM + r")x rN=(" + NUM + r")x", s)
        if m:
            reps.setdefault(m.group(1), []).append((fnum(m.group(5)), fnum(m.group(6)),
                                                    fnum(m.group(2)), fnum(m.group(3)),
                                                    fnum(m.group(4))))
            continue
        m = re.match(r"^n=(\d+) RESULT scipy1/fsci=(" + NUM + r")x scipyN/fsci=(" + NUM
                     + r")x null_fsci=(" + NUM + r") null_scipy1=(" + NUM + r") .*gates=(\w+)", s)
        if m:
            n = m.group(1)
            rr = reps.get(n, [])
            for variant, value, idx in (("scipy1", m.group(2), 0), ("scipyN", m.group(3), 1)):
                rows.append({
                    "case": f"n={n}", "variant": variant, "line_no": no,
                    "raw_value": fnum(value), "convention": "scipy_over_fsci", "raw_ci": None,
                    "raw_samples": [r[idx] for r in rr] or None,
                    "null_fsci": fnum(m.group(4)),
                    # scipyN has no A/A null of its own in this harness; only scipy1 does.
                    "null_scipy": fnum(m.group(5)) if variant == "scipy1" else None,
                    "null_kind": "spread",
                    "fsci_ms": median([r[2] for r in rr]) * 1e3 if rr else None,
                    "scipy_ms": median([r[3 + idx] for r in rr]) * 1e3 if rr else None,
                    "harness_verdict": f"gates={m.group(6)}",
                    "harness_refused": m.group(6) != "PASS", "agreement": None,
                })
    return rows


def parse_splu(lines):
    rows = []
    nulls = (None, None)
    fixture = None
    for no, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^fixture_sha256=\S+ n=(\d+) nnz=(\d+)", s)
        if m:
            fixture = f"n={m.group(1)} nnz={m.group(2)}"
        m = re.match(r"^NULL scipy/scipy=(" + NUM + r") NULL fsci/fsci=(" + NUM + r")", s)
        if m:
            nulls = (fnum(m.group(2)), fnum(m.group(1)))
        m = re.match(r"^Incumbent ratio: SciPy / FrankenSciPy = (" + NUM + r")x\s+ci95=\[("
                     + NUM + r"),(" + NUM + r")\]\s+rounds=(\d+)\s+verdict=(.*)$", s)
        if m:
            verdict = m.group(5).strip()
            rows.append({
                "case": fixture or "default", "variant": "scipy", "line_no": no,
                "raw_value": fnum(m.group(1)), "convention": "scipy_over_fsci",
                "raw_ci": (fnum(m.group(2)), fnum(m.group(3))), "raw_samples": None,
                "null_fsci": nulls[0], "null_scipy": nulls[1], "null_kind": "centered",
                "fsci_ms": None, "scipy_ms": None, "harness_verdict": verdict,
                "harness_refused": verdict.startswith(("NULL-FAILED", "IN-FLOOR")),
                "agreement": None,
            })
    return rows


def parse_sparse(lines):
    """perf_sparse_vs_scipy. Its single-method cells print the same `NULL-ours` /
    `Incumbent ratio:` / `median-CI gate: => outcome` block as perf_bdf_vs_scipy; its
    registered batch cells print `<label>_ratio_median=R ci95=[lo,hi]` after a
    `<label>_corrected_null_gate:` line carrying both null medians and the outcome."""
    single = parse_bdf(lines)
    if single:
        return single
    rows = []
    gate = {}
    for no, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^(\w+)_corrected_null_gate: (.*)$", s)
        if m:
            fields = dict(re.findall(r"(\w+)=(\S+)", m.group(2)))
            outcome = re.search(r"=> (.*)$", s)
            gate[m.group(1)] = (fields, outcome.group(1).strip() if outcome else None)
        m = re.match(r"^(\w+)_ratio_median=(" + NUM + r") ci95=\[(" + NUM + r"),(" + NUM + r")\]", s)
        if m:
            rows.append({
                "case": m.group(1), "variant": "scipy", "line_no": no,
                "raw_value": fnum(m.group(2)), "convention": "scipy_over_fsci",
                "raw_ci": (fnum(m.group(3)), fnum(m.group(4))), "raw_samples": None,
                "null_fsci": None, "null_scipy": None, "null_kind": "centered",
                "fsci_ms": None, "scipy_ms": None, "harness_verdict": None,
                "harness_refused": False, "agreement": None,
            })
    for row in rows:
        fields, outcome = gate.get(row["case"], ({}, None))
        row["null_fsci"] = fnum(fields.get("left_null_median")) if fields else None
        row["null_scipy"] = fnum(fields.get("right_null_median")) if fields else None
        row["harness_verdict"] = outcome
        row["harness_refused"] = outcome is None or not outcome.startswith("DECIDED")
    return rows


def parse_minres(lines):
    """perf_minres_vs_scipy times ours-minres, ours-gmres20 and scipy-minres; only ours-minres
    carries an A/A null. The interval is the quotient of the two arms' own median CIs."""
    arms = {}
    null = None
    rows = []
    for no, line in enumerate(lines):
        s = line.strip()
        m = re.match(r"^(ours-minres|ours-gmres20|scipy-minres)\s+median_ms_per_solve=(" + NUM
                     + r") ci95=\[(" + NUM + r"),(" + NUM + r")\]", s)
        if m:
            arms[m.group(1)] = (fnum(m.group(2)), fnum(m.group(3)), fnum(m.group(4)))
        m = re.match(r"^null_pair_median=(" + NUM + r")", s)
        if m:
            null = fnum(m.group(1))
        m = re.match(r"^SPEEDUP minres_vs_scipy_minres=(" + NUM + r")x", s)
        if m and "ours-minres" in arms and "scipy-minres" in arms:
            f, s_ = arms["ours-minres"], arms["scipy-minres"]
            ci = (s_[1] / f[2], s_[2] / f[1]) if f[1] > 0 and f[2] > 0 else None
            rows.append({
                "case": "minres", "variant": "scipy", "line_no": no,
                "raw_value": fnum(m.group(1)), "convention": "scipy_over_fsci",
                "raw_ci": ci, "raw_samples": None, "null_fsci": null, "null_scipy": None,
                "null_kind": "centered", "fsci_ms": f[0], "scipy_ms": s_[0],
                "harness_verdict": None, "harness_refused": False, "agreement": None,
            })
    return rows


def parse_eig(lines):
    """perf_eig_vs_scipy has NO SciPy arm: it times two fsci Schur arms and an fsci A/A.
    Rows are emitted so the board shows it, with no incumbent timing at all."""
    rows = []
    for no, line in enumerate(lines):
        m = re.match(r"^T n=(\d+) arm=(francis) (.*)$", line.strip())
        if m:
            rows.append({
                "case": f"n={m.group(1)}", "variant": "none (fsci arms only)", "line_no": no,
                "raw_value": float("nan"), "convention": "scipy_over_fsci", "raw_ci": None,
                "raw_samples": None, "null_fsci": None, "null_scipy": None,
                "null_kind": "centered", "fsci_ms": median(_list(m.group(3))) * 1e3,
                "scipy_ms": None, "harness_verdict": "no SciPy arm in harness",
                "harness_refused": True, "agreement": None,
            })
    return rows


HARNESSES = {
    "perf_cluster_vs_scipy": ("cluster", parse_case_style, "ABBA (not flipped), 5 rounds"),
    "perf_fft_vs_scipy": ("fft", parse_fft, "ABBA (not flipped), 3 rounds"),
    "perf_bdf_vs_scipy": ("integrate", parse_bdf, "interleaved, arm order alternating, per-round A/A pairs"),
    "perf_interpolate_vs_scipy": ("interpolate", parse_case_style, "ABBA/BAAB flipped, 5 rounds"),
    "perf_chol_vs_scipy": ("linalg", parse_chol, "ABBA halves per replicate, 5 replicates"),
    "perf_eig_vs_scipy": ("linalg", parse_eig, "fsci arms only, no SciPy"),
    "perf_eigh_vs_scipy": ("linalg", parse_eigh, "per-round pairs, A/B order alternating"),
    "perf_ndimage_vs_scipy": ("ndimage", parse_case_style, "ABBA/BAAB flipped, 5 rounds"),
    "perf_opt_vs_scipy": ("optimize", parse_case_style, "ABBA/BAAB flipped, 5 rounds"),
    "perf_signal_vs_scipy": ("signal", parse_case_style, "ABBA (not flipped), 5 rounds"),
    "perf_eigsh_vs_scipy": ("sparse", parse_case_style, "ABBA/BAAB flipped, 5 rounds"),
    "perf_gmres_job_vs_scipy": ("sparse", parse_bdf, "interleaved whole-job rounds"),
    "perf_minres_vs_scipy": ("sparse", parse_minres, "3-arm rotation, fsci A/A pair per round"),
    "perf_sparse_vs_scipy": ("sparse", parse_sparse, "interleaved rounds with A/A nulls"),
    "perf_splu": ("sparse", parse_splu, "balanced square ABBAABBA"),
    "perf_spatial_vs_scipy": ("spatial", parse_case_style, "ABBA/BAAB flipped, 5 rounds"),
    "perf_special_vs_scipy": ("special", parse_case_style, "ABBA/BAAB flipped, 5 rounds"),
    "perf_stats_vs_scipy": ("stats", parse_case_style, "ABBA/BAAB flipped, 5 rounds"),
}


# ════════════════════════════════════════════════════════════════════════════════════════════
# Host sampling (runner)
# ════════════════════════════════════════════════════════════════════════════════════════════

def read_loadavg():
    parts = pathlib.Path("/proc/loadavg").read_text().split()
    running, total = parts[3].split("/")
    return float(parts[0]), float(parts[1]), float(parts[2]), int(running), int(total)


def read_cpu_ticks():
    ticks = {}
    for line in pathlib.Path("/proc/stat").read_text().splitlines():
        if not line.startswith("cpu"):
            break
        name, *vals = line.split()
        vals = [int(v) for v in vals[:8]]
        idle = vals[3] + vals[4]
        ticks[name] = (sum(vals), idle)
    return ticks


def busy_between(a, b, name):
    if name not in a or name not in b:
        return float("nan")
    total = b[name][0] - a[name][0]
    idle = b[name][1] - a[name][1]
    return float("nan") if total <= 0 else (total - idle) / total


def process_tree(root_pid):
    """PIDs of `root_pid` and all descendants, via /proc/<pid>/task/<tid>/children."""
    out, stack = [], [root_pid]
    while stack:
        pid = stack.pop()
        out.append(pid)
        try:
            for task in os.listdir(f"/proc/{pid}/task"):
                try:
                    kids = pathlib.Path(f"/proc/{pid}/task/{task}/children").read_text().split()
                except OSError:
                    continue
                stack.extend(int(k) for k in kids)
        except OSError:
            continue
    return out


def tree_stats(root_pid):
    """(own running threads, threads of root, threads of descendants)."""
    running, root_threads, child_threads = 0, 0, 0
    for pid in process_tree(root_pid):
        try:
            tasks = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        if pid == root_pid:
            root_threads = len(tasks)
        else:
            child_threads += len(tasks)
        for task in tasks:
            try:
                stat = pathlib.Path(f"/proc/{pid}/task/{task}/stat").read_text()
            except OSError:
                continue
            state = stat[stat.rfind(")") + 2:].split(" ", 1)[0]
            if state == "R":
                running += 1
    return running, root_threads, child_threads


def sibling_of(cpu):
    text = pathlib.Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()
    sibs = []
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            sibs.extend(range(int(a), int(b) + 1))
        else:
            sibs.append(int(part))
    return [s for s in sibs if s != cpu]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def host_provenance():
    govs = {}
    for gov in sorted(pathlib.Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")):
        g = gov.read_text().strip()
        govs[g] = govs.get(g, 0) + 1
    driver = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver")
    epp = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
    flags, model = set(), None
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("flags") and not flags:
            flags = set(line.split(":", 1)[1].split())
        if line.startswith("model name") and model is None:
            model = line.split(":", 1)[1].strip()
    isa = "+".join(f for f in ("sse4_2", "avx", "avx2", "fma", "avx512f") if f in flags)
    return {
        "host": socket.gethostname(),
        "governor": ",".join(f"{g}x{c}" for g, c in govs.items()),
        "scaling_driver": driver.read_text().strip() if driver.exists() else None,
        "epp": epp.read_text().strip() if epp.exists() else None,
        "isa": isa,
        "cpu_model": model,
        "logical_cpus": os.cpu_count(),
        "kernel": os.uname().release,
    }


def git_head():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def cmd_run(args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    binary = pathlib.Path(args.binary).resolve()
    extra_env = dict(kv.split("=", 1) for kv in args.env)
    env = dict(os.environ)
    env.update(extra_env)
    argv = [str(binary), *args.harness_args]
    siblings = []
    if args.pin is not None:
        argv = ["taskset", "-c", str(args.pin), *argv]
        siblings = sibling_of(args.pin)
    prov = host_provenance()
    # Wait (bounded) for the ambient gate to be satisfiable before launching: a run started
    # on a host already above the ceiling produces rows that are UNRESOLVED by construction.
    # The launch happens either way once the bound expires, and the wait is recorded.
    waited = 0.0
    wait_start = time.monotonic()
    while read_loadavg()[0] > LOAD_CEILING and waited < args.max_wait:
        time.sleep(10.0)
        waited = time.monotonic() - wait_start
    la_pre = read_loadavg()
    pre_ticks = read_cpu_ticks()
    python = pathlib.Path(os.path.expanduser("~/.local/bin/python3.13")).resolve()
    meta = {
        "schema": "fsci-scoreboard-run/1",
        "tag": tag, "harness": args.harness, "binary": str(binary),
        "executed_elf_sha256": sha256_file(binary),
        "build_manifest": args.build_manifest,
        "argv": argv, "cwd": str(ROOT), "env_overlay": extra_env,
        "pin_cpu": args.pin, "sibling_cpus": siblings,
        "mode": f"pinned:{args.pin}" if args.pin is not None else "unpinned",
        "git_head": git_head(),
        "incumbent_interpreter": str(python),
        "incumbent_interpreter_sha256": sha256_file(python) if python.exists() else None,
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "loadavg_pre": list(la_pre[:3]), "procs_running_pre": la_pre[3],
        "ambient_wait_s": round(waited, 1), "ambient_wait_bound_s": args.max_wait,
        **prov,
    }
    log_path = out_dir / f"{tag}.log"
    lines_path = out_dir / f"{tag}.lines.tsv"
    trace_path = out_dir / f"{tag}.trace.tsv"
    t0 = time.monotonic()
    proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    stop = threading.Event()

    def sampler():
        cpus = ["cpu"] + ([f"cpu{args.pin}"] + [f"cpu{s}" for s in siblings]
                          if args.pin is not None else [])
        with open(trace_path, "w", encoding="utf-8") as fh:
            fh.write("t\tla1\tla5\tla15\tprocs_running\town_running\tforeign_running\t"
                     "fsci_threads\tchild_threads\tpinned_busy\tsibling_busy\thost_busy\n")
            prev = pre_ticks
            # Sample FIRST, then wait: a harness that finishes in under a second must still
            # leave at least one sample, or its rows have no load evidence at all.
            first = True
            while first or not stop.wait(1.0):
                first = False
                try:
                    la = read_loadavg()
                    own, rt, ct = tree_stats(proc.pid)
                    cur = read_cpu_ticks()
                except OSError:
                    continue
                pinned = busy_between(prev, cur, cpus[1]) if len(cpus) > 1 else float("nan")
                sib = (median([busy_between(prev, cur, c) for c in cpus[2:]])
                       if len(cpus) > 2 else float("nan"))
                host = busy_between(prev, cur, "cpu")
                prev = cur
                foreign = max(0, la[3] - own - 1)
                fh.write(f"{time.monotonic() - t0:.2f}\t{la[0]}\t{la[1]}\t{la[2]}\t{la[3]}\t"
                         f"{own}\t{foreign}\t{rt}\t{ct}\t{pinned:.3f}\t{sib:.3f}\t{host:.3f}\n")
                fh.flush()

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    timed_out = False
    with open(log_path, "wb") as log, open(lines_path, "w", encoding="utf-8") as lines:
        lines.write("line_no\tt\n")
        no = 0
        deadline = t0 + args.timeout
        for raw in iter(proc.stdout.readline, b""):
            log.write(raw)
            log.flush()
            lines.write(f"{no}\t{time.monotonic() - t0:.3f}\n")
            no += 1
            if time.monotonic() > deadline:
                timed_out = True
                proc.kill()
                break
    rc = proc.wait()
    stop.set()
    th.join(timeout=5)
    la_post = read_loadavg()
    meta.update({
        "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.monotonic() - t0, 2),
        "exit_code": rc, "timed_out": timed_out,
        "loadavg_post": list(la_post[:3]),
        "host_busy_whole_run": busy_between(pre_ticks, read_cpu_ticks(), "cpu"),
        "log_sha256": sha256_file(log_path),
    })
    (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{tag}: exit={rc} elapsed={meta['elapsed_s']}s timed_out={timed_out} "
          f"loadavg_pre={la_pre[0]} post={la_post[0]}")
    return 0


# ════════════════════════════════════════════════════════════════════════════════════════════
# Build
# ════════════════════════════════════════════════════════════════════════════════════════════

def read_tsv(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        return []
    head = text[0].split("\t")
    out = []
    for line in text[1:]:
        vals = line.split("\t")
        if len(vals) != len(head):
            continue
        out.append({k: fnum(v) for k, v in zip(head, vals)})
    return out


def window_stats(trace, t_start, t_end):
    window = [s for s in trace if t_start <= s["t"] <= t_end]
    if not window and trace:
        # A row shorter than one sample: take the nearest sample after its start.
        after = [s for s in trace if s["t"] >= t_start]
        window = [after[0]] if after else [trace[-1]]

    def col(name):
        return [s[name] for s in window if finite(s.get(name))]

    def summ(name):
        vals = col(name)
        return {"min": min(vals), "median": median(vals), "max": max(vals)} if vals else None

    fr = col("foreign_running")
    net = [s["la1"] - s["own_running"] for s in window
           if finite(s.get("la1")) and finite(s.get("own_running"))]
    return {
        "samples": len(window),
        "loadavg1": summ("la1"),
        "loadavg1_net_median": median(net) if net else float("nan"),
        "foreign_running_median": median(fr) if fr else float("nan"),
        "pinned_busy_mean": (sum(col("pinned_busy")) / len(col("pinned_busy"))
                             if col("pinned_busy") else None),
        "sibling_busy_mean": (sum(col("sibling_busy")) / len(col("sibling_busy"))
                              if col("sibling_busy") else None),
        "host_busy_mean": (sum(col("host_busy")) / len(col("host_busy"))
                           if col("host_busy") else None),
        "fsci_threads_peak": max(col("fsci_threads")) if col("fsci_threads") else None,
        "child_threads_peak": max(col("child_threads")) if col("child_threads") else None,
    }


def finish_row(row):
    """Normalise, choose the interval, attach its kind, classify. Mutates and returns row."""
    ratio, ci = normalise_ratio(row["raw_value"], row["convention"], row.get("raw_ci"))
    kind = None
    if ci is not None:
        kind = "harness_bootstrap95" if row.get("raw_ci_kind") is None else row["raw_ci_kind"]
    elif row.get("raw_samples"):
        samples = [normalise_ratio(v, row["convention"])[0] for v in row["raw_samples"]]
        ci = bootstrap_median_ci(samples)
        kind = "builder_bootstrap95_over_rounds"
    elif row.get("null_fsci") is not None and row.get("null_scipy") is not None:
        ci = null_envelope(ratio, row["null_fsci"], row["null_scipy"], row["null_kind"])
        kind = "null_envelope"
    row["ratio"], row["ci"], row["ci_kind"] = ratio, ci, kind
    m = re.search(r"max_rel=(" + NUM + r")", row.get("agreement") or "")
    # Recorded, not gated: the harnesses print the elementwise disagreement but carry no
    # per-op tolerance contract, so the board flags large values instead of judging them.
    row["agreement_max_rel"] = fnum(m.group(1)) if m else None
    row["class"], row["class_reasons"] = classify(row)
    return row


def invocation_key(meta):
    """What was run, independent of WHEN: harness, args, env overlay, pin mode. Two runs with
    the same key are replicates of one invocation; a retry is only ever such a replicate."""
    args = meta["argv"][meta["argv"].index(meta["binary"]) + 1:]
    env = " ".join(f"{k}={v}" for k, v in sorted(meta["env_overlay"].items()))
    return f"{meta['harness']} [{' '.join(args)}] {{{env}}} {meta['mode'].split(':')[0]}"


def load_gated(row):
    return any("loadavg" in r or "foreign runnable" in r for r in row.get("class_reasons", []))


def rows_from_run(meta_path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Tags contain dots (`fft.pinned`), so build sibling names by concatenation, never with
    # `with_suffix`, which would eat the mode.
    log_path = meta_path.parent / f"{meta['tag']}.log"
    text = log_path.read_bytes().decode("utf-8", errors="replace")
    lines = text.splitlines()
    line_times = {int(r["line_no"]): r["t"]
                  for r in read_tsv(meta_path.parent / f"{meta['tag']}.lines.tsv")}
    trace = read_tsv(meta_path.parent / f"{meta['tag']}.trace.tsv")
    header = parse_header(lines)
    family, parser, schedule = HARNESSES[meta["harness"]]
    parsed = parser(lines)
    incumbent = header["incumbent"]
    if incumbent is not None and header["ready_genuine"] is False:
        incumbent = dict(incumbent, genuine=False)
    key = invocation_key(meta)
    args = meta["argv"][meta["argv"].index(meta["binary"]) + 1:]
    invocation = " ".join(
        [f"{k}={v}" for k, v in sorted(meta["env_overlay"].items())]
        + [meta["harness"]] + args)
    status = {
        "tag": meta["tag"], "harness": meta["harness"], "family": family, "key": key,
        "invocation": invocation, "mode": meta["mode"], "exit_code": meta["exit_code"],
        "timed_out": meta["timed_out"], "elapsed_s": meta["elapsed_s"], "rows": len(parsed),
        "log": log_path.name, "log_sha256": meta.get("log_sha256"),
        "started_utc": meta.get("started_utc"),
        "fsci_elf_sha256": header["fsci_elf_sha256"],
        "executed_elf_sha256": meta["executed_elf_sha256"],
        "incumbent": incumbent, "loadavg_pre": meta["loadavg_pre"],
        "loadavg_post": meta["loadavg_post"], "schedule": schedule,
        "ambient_wait_s": meta.get("ambient_wait_s"),
        "peak_threads_fsci": max((s["fsci_threads"] for s in trace
                                  if finite(s.get("fsci_threads"))), default=None),
        "peak_threads_scipy_child": max((s["child_threads"] for s in trace
                                         if finite(s.get("child_threads"))), default=None),
        "refusal": next((line.strip() for line in lines
                         if line.startswith(("ABORT", "Error", "error:"))
                         or "panicked at" in line), None),
        "tail": [line for line in lines[-3:]],
    }
    rows = []
    prev_t = 0.0
    for r in parsed:
        t_start = prev_t
        t_end = line_times.get(r["line_no"], meta["elapsed_s"])
        load = window_stats(trace, t_start, t_end)
        prev_t = t_end
        load["ambient_loadavg1"] = meta["loadavg_pre"][0]
        row = dict(r)
        row.update({
            "harness": meta["harness"], "family": family, "tag": meta["tag"], "key": key,
            "invocation": invocation, "started_utc": meta.get("started_utc"),
            "mode": meta["mode"], "host": meta.get("host"), "governor": meta.get("governor"),
            "scaling_driver": meta.get("scaling_driver"), "epp": meta.get("epp"),
            "isa": meta.get("isa"), "fsci_elf_sha256": header["fsci_elf_sha256"],
            "executed_elf_sha256": meta["executed_elf_sha256"],
            "incumbent": incumbent,
            "incumbent_elf_sha256": meta.get("incumbent_interpreter_sha256"),
            "scipy_engine_sha256": header["scipy_engine_sha256"],
            "scipy_source": ("live_same_invocation" if incumbent is not None
                             else "absent"),
            "schedule": schedule, "window_s": [round(t_start, 2), round(t_end, 2)],
            "load": load, "raw_log": log_path.name, "log_line": r["line_no"] + 1,
            "git_head": meta.get("git_head"), "build_manifest": meta.get("build_manifest"),
        })
        rows.append(finish_row(row))
    return status, rows


def select_primary(rows):
    """Replicates of one invocation are all kept; exactly one row per (key, case, variant) is
    PRIMARY. The earliest run's row is primary unless it was load-gated, in which case the
    earliest later replicate whose row was not load-gated takes over. The choice never looks
    at the ratio. Superseded rows stay in the JSON with `superseded_by`."""
    groups = {}
    for r in rows:
        groups.setdefault((r["key"], r["case"], r["variant"]), []).append(r)
    for group in groups.values():
        group.sort(key=lambda r: r["started_utc"] or "")
        primary = group[0]
        if load_gated(primary):
            for later in group[1:]:
                if not load_gated(later):
                    primary = later
                    break
        for r in group:
            r["primary"] = r is primary
            r["superseded_by"] = None if r is primary else primary["tag"]


def inventory():
    """Every live-incumbent harness source in the tree, so the board says what it did NOT run."""
    found = []
    for path in sorted(ROOT.glob("crates/*/src/bin/perf_*.rs")):
        if "scipy" in path.name or path.name == "perf_splu_balanced_square.rs":
            found.append(str(path.relative_to(ROOT)))
    return found


def read_build_manifest(path):
    out = {"path": path, "worker": None, "built": {}, "missing": []}
    if not path or not pathlib.Path(ROOT / path).exists():
        return out
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if line.startswith("worker="):
            out["worker"] = line.split()[0].split("=", 1)[1]
        elif line.startswith("BUILT "):
            _, name, sha = line.split()
            out["built"][name] = sha
        elif line.startswith("MISSING "):
            out["missing"].append(line.split()[1])
    return out


# The W7 epic's named standing losses (frankenscipy-sw4p0 evidence, ratio SciPy/fsci), and how
# each maps onto rows of this board. A loss with no matching row is reported as such.
W7_LOSSES = [
    ("add_coo 28672^2", "~0.062 (perf_ledger_cc.md:13837-13870)", None, None),
    ("splu convection 16-RHS solve stage", "0.442 (NEGATIVE_EVIDENCE.md:42600-42625)",
     "perf_splu", r"FSCI_SPLU_STAGE=solve .*convection"),
    ("eigh n=768", "0.276 (commit 932bd498a)", "perf_eigh_vs_scipy", r"n=768"),
    ("dense BDF n=512", "0.497 (perf_ledger_cc.md:5985)", "perf_bdf_vs_scipy",
     r"dense-allpairs n=512"),
    ("erfinv 200k", "0.728 (commit 42c898d99)", "perf_special_vs_scipy", r"op=erfinv$"),
    ("erfcinv 200k", "0.721 (commit 42c898d99)", "perf_special_vs_scipy", r"op=erfcinv$"),
]


def w7_rows(primary, missing):
    out = []
    for name, prior, harness, pattern in W7_LOSSES:
        hits = [r for r in primary if harness and r["harness"] == harness
                and re.search(pattern, r["invocation"] + " | " + r["case"])]
        if not harness:
            verdict = "UNRESOLVED: no live-SciPy harness for add_coo exists at HEAD"
        elif harness in missing:
            verdict = f"UNRESOLVED: {harness} does not build at HEAD"
        elif not hits:
            verdict = "UNRESOLVED: harness ran but produced no row for this cell"
        else:
            classes = sorted({r["class"] for r in hits})
            verdict = "/".join(classes)
        out.append({"loss": name, "prior": prior, "verdict": verdict,
                    "rows": [{"mode": r["mode"], "variant": r["variant"], "class": r["class"],
                              "ratio": r["ratio"], "ci": r["ci"], "tag": r["tag"],
                              "log_line": r["log_line"]} for r in hits]})
    return out


def fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        if not math.isfinite(x):
            return "nan"
        return f"{x:.{digits}f}"
    return str(x)


def fmt_ci(ci):
    if not ci:
        return "-"
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def short(sha):
    return sha[:12] if isinstance(sha, str) else "-"


def row_md(r, rank=None):
    load = r.get("load") or {}
    la = load.get("loadavg1") or {}
    cells = [
        str(rank) if rank is not None else None,
        f"`{r['harness']}`", r["case"], r["variant"], r["mode"], f"**{fmt(r['ratio'])}**",
        f"{fmt_ci(r['ci'])} {r['ci_kind'] or ''}",
        f"{fmt(r.get('null_fsci'))} / {fmt(r.get('null_scipy'))} ({r.get('null_kind')})",
        f"{fmt(load.get('ambient_loadavg1'), 1)} / {fmt(load.get('loadavg1_net_median'), 1)}"
        f" / {fmt(load.get('foreign_running_median'), 1)} (raw la1 max {fmt(la.get('max'), 1)})",
        f"{fmt(load.get('sibling_busy_mean'), 2)}",
        f"{fmt(load.get('fsci_threads_peak'), 0)}/{fmt(load.get('child_threads_peak'), 0)}",
        f"`{short(r.get('fsci_elf_sha256'))}`",
        f"`raw/{r['raw_log']}:{r['log_line']}`",
    ]
    return "| " + " | ".join(c for c in cells if c is not None) + " |"


ROW_HEAD = ("| harness | case | vs | mode | SciPy/fsci | interval | A/A nulls fsci / scipy | "
            "load: ambient / la1 net med / foreign med | sibling busy | threads (1 Hz) fsci/scipy | fsci ELF | "
            "evidence |\n|" + "---|" * 12)


def render_markdown(doc):
    out = []
    w = out.append
    prov = doc["provenance"]
    w("# FrankenSciPy performance scoreboard at HEAD vs live SciPy")
    w("")
    w(f"Bead `frankenscipy-sw4p0.2`. Generated {doc['generated_utc']} by "
      "`scripts/perf_scoreboard.py build` from the raw run logs in `raw/`. "
      f"HEAD `{prov['git_head']}`, measured on `{prov['host']}`.")
    w("")
    w("**Ratio convention: ratio = SciPy time / fsci time. Above 1 means FrankenSciPy is "
      "faster.** Harnesses that print fsci/SciPy were inverted and their intervals swapped.")
    w("")
    w("This file replaces the headline role of `docs/GAUNTLET_RELEASE_SCORECARD.md` (June "
      "2026). The ledgers keep their history; nothing in them was rewritten.")
    w("")
    w("## Headline")
    w("")
    w("Primary rows only (one per invocation, case and incumbent variant; planted "
      "negative-case rows and superseded replicates are excluded).")
    w("")
    w("| class | all | pinned (1 CPU) | unpinned |")
    w("|---|---|---|---|")
    for c in CLASSES:
        w(f"| {c} | {doc['counts'][c]} | {doc['counts_pinned'][c]} | "
          f"{doc['counts_unpinned'][c]} |")
    no_rows = [s for s in doc["runs"] if s["rows"] == 0]
    w(f"| runs that produced no row (harness refused or failed), plus harnesses not built | "
      f"{len(no_rows) + len(doc['build']['missing'])} | "
      f"{sum(1 for s in no_rows if s['mode'].startswith('pinned'))} | "
      f"{sum(1 for s in no_rows if s['mode'] == 'unpinned')} |")
    w("")
    w("A WIN or LOSE needs: SciPy live in the same invocation (pinned 1.17.1 / numpy 2.4.3, "
      "`genuine=true`), a self-reported fsci ELF sha256 equal to the executed binary's, both "
      f"arms' A/A nulls inside their band (centered: within +/-{NULL_CENTERED_BAND:.0%}; spread "
      f"max/min: <= {NULL_SPREAD_MAX}), the host under the load ceiling of {LOAD_CEILING:.0f} "
      "on all three readings (1-min loadavg just before launch; median over the row's own time "
      "window of the 1-min loadavg net of the harness's own running threads; median over that "
      "window of foreign runnable tasks), no refusal from the harness's own gate, and an "
      "interval that does not touch 1.0.")
    w("")
    w("Intervals come from the harness's own bootstrap CI where it prints one "
      "(`harness_bootstrap95`), else a bootstrap over the harness's per-round ratios "
      "(`builder_bootstrap95_over_rounds`), else the null envelope ratio/(nf*ns) .. "
      "ratio*(nf*ns) built from the two arms' own A/A nulls (`null_envelope`). A replicate of "
      "the same invocation replaces a row only when the earlier row was load-gated; the "
      "choice never looks at the ratio.")
    w("")
    w("## LOSE rows, ranked by magnitude")
    w("")
    loses = sorted((r for r in doc["rows"] if r["primary"] and r["class"] == "LOSE"),
                   key=lambda r: r["ratio"])
    if loses:
        w("| rank |" + ROW_HEAD[1:].replace("|\n|", "|\n|---|", 1))
        for i, r in enumerate(loses, 1):
            w(row_md(r, i))
    else:
        w("No primary row classified LOSE.")
    w("")
    w("## WIN rows")
    w("")
    wins = sorted((r for r in doc["rows"] if r["primary"] and r["class"] == "WIN"),
                  key=lambda r: -r["ratio"])
    if wins:
        w(ROW_HEAD)
        for r in wins:
            w(row_md(r))
    else:
        w("No primary row classified WIN.")
    w("")
    w("## W7 epic standing losses, re-measured at HEAD")
    w("")
    w("| named loss | prior figure | HEAD verdict | HEAD rows (mode, vs, class, ratio, interval) |")
    w("|---|---|---|---|")
    for item in doc["w7"]:
        rows = "; ".join(
            f"{r['mode']} {r['variant']} {r['class']} {fmt(r['ratio'])} {fmt_ci(r['ci'])} "
            f"(`raw/{r['tag']}.log:{r['log_line']}`)" for r in item["rows"]) or "-"
        w(f"| {item['loss']} | {item['prior']} | {item['verdict']} | {rows} |")
    w("")
    w("## Harness status")
    w("")
    w("Every run is listed, replicates included. `peak threads` is the largest task count seen "
      "by the 1 Hz sampler over the whole run (fsci process / SciPy child); worker threads that "
      "live for less than a sampling interval can be missed, so it is a lower bound.")
    w("")
    w("| invocation | family | mode | log | exit | rows | classes of primary rows | elapsed s | "
      "loadavg pre | peak threads | note |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in doc["runs"]:
        classes = {}
        for r in doc["rows"]:
            if r["tag"] == s["tag"] and r["primary"]:
                classes[r["class"]] = classes.get(r["class"], 0) + 1
        cls = ", ".join(f"{k} {v}" for k, v in sorted(classes.items())) or "-"
        note = s.get("note") or s.get("refusal") or ""
        w(f"| `{s['invocation']}` | {s['family']} | {s['mode']} | `raw/{s['log']}` | "
          f"{s['exit_code']} | {s['rows']} | {cls} | {fmt(s['elapsed_s'], 0)} | "
          f"{fmt(s['loadavg_pre'][0], 1)} | {fmt(s['peak_threads_fsci'], 0)}/"
          f"{fmt(s['peak_threads_scipy_child'], 0)} | {note} |")
    for name in doc["build"]["missing"]:
        w(f"| `{name}` | - | - | - | not built | 0 | - | - | - | - | "
          f"{doc['build_failures'].get(name, 'did not compile at HEAD')} |")
    w("")
    w("Live-incumbent harness sources present in the tree but NOT run in this pass: "
      + (", ".join(f"`{p}`" for p in doc["not_run"]) or "none") + ".")
    w("")
    w("## Harness defects observed")
    w("")
    for d in doc["defects"]:
        w(f"- {d}")
    flagged = [r for r in doc["rows"] if r["primary"] and finite(r.get("agreement_max_rel"))
               and r["agreement_max_rel"] > AGREEMENT_FLAG]
    for r in flagged:
        w(f"- Agreement flag (recorded, not gated): `{r['harness']}` {r['case']} {r['mode']} "
          f"reports `max_rel={r['agreement_max_rel']:.3g}` against SciPy's output "
          f"(`raw/{r['raw_log']}:{r['log_line']}`: `{(r.get('agreement') or '')[:80]}`).")
    w("")
    if doc.get("observations"):
        w("## Observations")
        w("")
        for o in doc["observations"]:
            w(f"- {o}")
        w("")
    w("## Negative cases (planted, never counted)")
    w("")
    w("Each planted row is built from real HEAD run data (generator and inputs named in the "
      "row's `case`); `planted_rows.json` is the builder's input. The last column is what the "
      "same row would have been called with the source and executable-identity checks "
      "skipped, which is the failure each case exists to catch.")
    w("")
    w("| planted row | ratio (SciPy/fsci) | class | reason | class without those checks |")
    w("|---|---|---|---|---|")
    for p in doc["planted_negative_cases"]:
        w(f"| {p['case']} | {fmt(p['ratio'])} {fmt_ci(p['ci'])} {p.get('ci_kind') or ''} | "
          f"**{p['class']}** | {'; '.join(p['class_reasons'])} | "
          f"{p.get('class_without_source_and_identity_checks')} |")
    w("")
    w("## All primary rows")
    w("")
    for fam in sorted({r["family"] for r in doc["rows"]}):
        w(f"### {fam}")
        w("")
        w("| class |" + ROW_HEAD[1:].replace("|\n|", "|\n|---|", 1))
        for r in sorted((r for r in doc["rows"] if r["family"] == fam and r["primary"]),
                        key=lambda r: (r["harness"], r["mode"], r["case"], r["variant"])):
            reason = "" if r["class"] in ("WIN", "LOSE") else f" ({'; '.join(r['class_reasons'])})"
            w(f"| {r['class']}{reason} |" + row_md(r)[1:])
        w("")
    w("## Provenance")
    w("")
    for k in ("host", "cpu_model", "logical_cpus", "kernel", "governor", "scaling_driver",
              "epp", "isa", "build_target_features", "build_worker", "git_head",
              "incumbent_interpreter", "incumbent_interpreter_sha256", "incumbent"):
        w(f"- `{k}`: {prov.get(k)}")
    w("")
    w("## Regenerating")
    w("")
    w("```")
    w("# per harness, from the worktree root, binaries built remotely and brought back:")
    w("python3.13 scripts/perf_scoreboard.py run --out-dir <dir>/raw --tag <tag> "
      "--harness <bin> --binary <path> [--pin <cpu>] [--env K=V ...] -- <harness args>")
    w("python3.13 scripts/perf_scoreboard.py plant --artifact-dir <dir> --gmres-tag <tag> "
      "--bdf-tag <tag> --splu-tag <tag>")
    w("python3.13 scripts/perf_scoreboard.py build --artifact-dir <dir> --out-json "
      "<dir>/scoreboard.json --out-md <dir>/SCOREBOARD.md --build-manifest <dir>/build/manifest_b1.txt")
    w("python3.13 scripts/perf_scoreboard.py --self-test")
    w("```")
    w("")
    w("Per row, the JSON carries: raw log path and line, both ELF identities (self-reported "
      "and executed), the incumbent interpreter sha256 and SciPy engine sha256 where the "
      "harness prints one, the interval kind, both null values and their kind, the row's time "
      "window, its loadavg trace summary, foreign runnable median, pinned-CPU and SMT-sibling "
      "busy, peak observed thread counts of both arms, the interleaving schedule, and the "
      "classification reasons.")
    w("")
    return "\n".join(out) + "\n"


def cmd_build(args):
    art = pathlib.Path(args.artifact_dir)
    runs = sorted((art / "raw").glob("*.meta.json"))
    statuses, rows = [], []
    for meta_path in runs:
        status, rr = rows_from_run(meta_path)
        statuses.append(status)
        rows.extend(rr)
    select_primary(rows)
    notes_path = art / "run_notes.json"
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
    for s in statuses:
        s["note"] = notes.get("runs", {}).get(s["tag"])
    build = read_build_manifest(args.build_manifest)
    planted = []
    planted_path = art / "planted_rows.json"
    if planted_path.exists():
        for p in json.loads(planted_path.read_text(encoding="utf-8")):
            p = dict(p)
            # A planted row inherits the load evidence of the real run its data came from.
            src = next((r for r in rows if r["tag"] == p.get("tag")), None)
            if src is not None:
                p["load"] = src["load"]
            done = finish_row(p)
            # What the row would have been called had the builder NOT checked where the SciPy
            # number came from or what executable sat in the incumbent slot.
            naive = dict(done, scipy_source="live_same_invocation",
                         incumbent_elf_sha256="0" * 64, harness_refused=False)
            done["class_without_source_and_identity_checks"] = classify(naive)[0]
            planted.append(done)
    primary = [r for r in rows if r["primary"]]

    def count(sel):
        c = {k: 0 for k in CLASSES}
        for r in sel:
            c[r["class"]] += 1
        return c

    first = runs and json.loads(runs[0].read_text(encoding="utf-8"))
    run_harnesses = {s["harness"] for s in statuses}
    src_for = {"perf_splu": "perf_splu_balanced_square"}
    not_run = [p for p in inventory()
               if not any(p.endswith(f"/{src_for.get(h, h)}.rs") for h in run_harnesses)
               and not any(p.endswith(f"/{m}.rs") for m in build["missing"])]
    doc = {
        "schema": "fsci-scoreboard/1",
        "bead": "frankenscipy-sw4p0.2",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "ratio_convention": "ratio = SciPy time / fsci time; > 1 means FrankenSciPy is faster",
        "gates": {
            "load_ceiling": LOAD_CEILING, "null_centered_band": NULL_CENTERED_BAND,
            "null_spread_max": NULL_SPREAD_MAX, "pinned_scipy": PINNED_SCIPY,
            "pinned_numpy": PINNED_NUMPY, "bootstrap_iters": BOOTSTRAP_ITERS,
            "primary_row_rule": select_primary.__doc__.split("\n\n")[0].strip(),
        },
        "provenance": {
            **{k: (first or {}).get(k) for k in (
                "host", "cpu_model", "logical_cpus", "kernel", "governor", "scaling_driver",
                "epp", "isa", "git_head", "incumbent_interpreter",
                "incumbent_interpreter_sha256")},
            "incumbent": next((s["incumbent"] for s in statuses if s["incumbent"]), None),
            "build_worker": build["worker"],
            "build_manifest": build["path"],
            "build_target_features": notes.get("build_target_features"),
            "binaries": build["built"],
        },
        "counts": count(primary),
        "counts_pinned": count(r for r in primary if r["mode"].startswith("pinned")),
        "counts_unpinned": count(r for r in primary if r["mode"] == "unpinned"),
        "build": build,
        "build_failures": notes.get("build_failures", {}),
        "defects": notes.get("defects", []),
        "observations": notes.get("observations", []),
        "not_run": not_run,
        "runs": statuses,
        "w7": w7_rows(primary, build["missing"]),
        "rows": rows,
        "planted_negative_cases": planted,
    }
    pathlib.Path(args.out_json).write_text(json.dumps(doc, indent=1, default=str) + "\n",
                                           encoding="utf-8")
    if args.out_md:
        pathlib.Path(args.out_md).write_text(render_markdown(doc), encoding="utf-8")
    print(json.dumps({"all": doc["counts"], "pinned": doc["counts_pinned"],
                      "unpinned": doc["counts_unpinned"]}))
    return 0


# ════════════════════════════════════════════════════════════════════════════════════════════
# Planted negative cases (the bead's two), built from REAL run data
# ════════════════════════════════════════════════════════════════════════════════════════════

# Ledger rows whose SciPy p50 is copied into negative case 1. Both are genuine same-invocation
# figures IN THEIR OWN RUN; copied into a HEAD row they are cached numbers.
CACHED_SCIPY = {
    "gmres": (20.786496, "docs/perf_ledger_cc.md:4467",
              "GMRES side=64 SciPy p50, frankenscipy-felow row, 2026-07-29"),
    "bdf_dense512": (511.248, "docs/perf_ledger_cc.md:5979",
                     "dense BDF n=512 live SciPy p50, 2026-08-01"),
}


def cmd_plant(args):
    """Write planted_rows.json.

    1. CACHED SciPy arm: the fsci arm is a live HEAD measurement, the SciPy arm is a number
       copied from a ledger. One case is chosen whose naive ratio reads as a WIN and one as a
       LOSS, so the rejection is seen to key on the SOURCE, not on the direction.
    2. SELF-COMPARISON: perf_splu's `NULL fsci/fsci` is a genuine fsci-vs-fsci comparison (the
       fsci arm's first-half slots against its second-half slots, same ELF, same invocation).
       It is presented in the shape of a vs-SciPy row -- the log's own `genuine=true` incumbent
       line included -- with the incumbent slot's executable identity set to what actually ran
       there, the fsci ELF. Only the ELF-sha check can tell; the ratio must sit at ~1.0.
    """
    raw = pathlib.Path(args.artifact_dir) / "raw"

    def load_run(tag):
        meta = json.loads((raw / f"{tag}.meta.json").read_text(encoding="utf-8"))
        text = (raw / f"{tag}.log").read_text(encoding="utf-8", errors="replace")
        head = parse_header(text.splitlines())
        return meta, text, head

    def base(tag, meta, head):
        return {
            "harness": meta["harness"], "family": "planted", "tag": tag, "mode": meta["mode"],
            "host": meta["host"], "governor": meta["governor"], "isa": meta["isa"],
            "fsci_elf_sha256": head["fsci_elf_sha256"],
            "executed_elf_sha256": meta["executed_elf_sha256"],
            "incumbent": head["incumbent"],
            "incumbent_elf_sha256": meta["incumbent_interpreter_sha256"],
            "scipy_source": "live_same_invocation", "variant": "scipy",
            "convention": "scipy_over_fsci", "raw_ci": None, "raw_samples": None,
            "null_kind": "centered", "harness_verdict": None, "harness_refused": False,
            "load": None, "raw_log": f"{tag}.log", "planted": True,
        }

    rows = []
    for tag, key, ours_re, null_re in (
        (args.gmres_tag, "gmres", r"OURS p50=([0-9.]+)ms",
         r"NULL-ours A/A median=([0-9.]+) ci95=\[([0-9.]+),([0-9.]+)\]"),
        (args.bdf_tag, "bdf_dense512", r"OURS\s+p50=([0-9.]+)ms",
         r"NULL-ours\s+median=([0-9.]+) ci95=\[([0-9.]+),([0-9.]+)\]"),
    ):
        meta, text, head = load_run(tag)
        ours_ms = float(re.search(ours_re, text).group(1))
        null_med, null_lo, null_hi = map(float, re.search(null_re, text).group(1, 2, 3))
        cached_ms, cite, what = CACHED_SCIPY[key]
        row = base(tag, meta, head)
        row.update({
            "case": (f"NEGATIVE CASE 1 ({key}): fsci p50 {ours_ms:.6f} ms live at HEAD "
                     f"(`raw/{tag}.log`), SciPy p50 {cached_ms} ms COPIED from {cite} ({what})"),
            "scipy_source": f"cached:{cite}",
            "raw_value": cached_ms / ours_ms,
            # The only interval a copied number admits: the live fsci arm's own A/A spread.
            "raw_ci": (cached_ms / (ours_ms * null_hi), cached_ms / (ours_ms * null_lo)),
            "raw_ci_kind": "fsci_null_ci_propagated",
            "null_fsci": null_med, "null_scipy": 1.0, "fsci_ms": ours_ms, "scipy_ms": cached_ms,
        })
        rows.append(row)
    meta, text, head = load_run(args.splu_tag)
    rounds = int(re.search(r"rounds=(\d+)", text).group(1))
    aa = float(re.search(r"NULL fsci/fsci=([0-9.]+)", text).group(1))
    row = base(args.splu_tag, meta, head)
    row.update({
        "case": (f"NEGATIVE CASE 2: perf_splu fsci arm against the SAME fsci ELF (its own "
                 f"`NULL fsci/fsci` over {rounds} balanced-square rounds, `raw/{args.splu_tag}"
                 f".log`), labelled as a SciPy row"),
        "raw_value": aa, "null_fsci": aa, "null_scipy": aa,
        "incumbent_elf_sha256": head["fsci_elf_sha256"],
    })
    rows.append(row)
    out = pathlib.Path(args.artifact_dir) / "planted_rows.json"
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} planted rows to {out}")
    return 0


# ════════════════════════════════════════════════════════════════════════════════════════════
# Self-test
# ════════════════════════════════════════════════════════════════════════════════════════════

def _good_row(**over):
    sha = "a" * 64
    row = {
        "fsci_elf_sha256": sha, "executed_elf_sha256": sha, "host": "thinkstation1",
        "scipy_source": "live_same_invocation",
        "incumbent": {"genuine": True, "scipy": PINNED_SCIPY, "numpy": PINNED_NUMPY},
        "incumbent_elf_sha256": "b" * 64, "governor": "powersavex64", "isa": "avx2+fma",
        "ratio": 1.5, "ci": (1.4, 1.6), "null_fsci": 1.01, "null_scipy": 0.99,
        "null_kind": "centered",
        "load": {"ambient_loadavg1": 10.0, "loadavg1_net_median": 11.0,
                 "foreign_running_median": 5.0},
        "harness_refused": False,
    }
    row.update(over)
    return row


def _with_provenance(parsed):
    """A parsed row plus complete provenance; the PARSED fields win (nulls, kind, values)."""
    return {**_good_row(ratio=None, ci=None), **parsed}


def run_self_test():
    import unittest

    class Normaliser(unittest.TestCase):
        def test_scipy_over_fsci_passes_through(self):
            r, ci = normalise_ratio(2.0, "scipy_over_fsci", (1.5, 2.5))
            self.assertEqual((r, ci), (2.0, (1.5, 2.5)))

        def test_fsci_over_scipy_inverts_and_swaps(self):
            r, ci = normalise_ratio(2.0, "fsci_over_scipy", (1.6, 2.5))
            self.assertAlmostEqual(r, 0.5)
            self.assertAlmostEqual(ci[0], 0.4)
            self.assertAlmostEqual(ci[1], 0.625)
            self.assertLess(ci[0], ci[1])

        def test_bad_values_become_nan_not_ratios(self):
            for bad in (0.0, -1.0, float("nan"), float("inf"), None):
                r, _ = normalise_ratio(bad, "fsci_over_scipy")
                self.assertTrue(math.isnan(r), bad)

        def test_unknown_convention_raises(self):
            with self.assertRaises(ValueError):
                normalise_ratio(1.0, "scipy/fsci")

    class Classifier(unittest.TestCase):
        def test_win_lose_and_overlap(self):
            self.assertEqual(classify(_good_row())[0], "WIN")
            self.assertEqual(classify(_good_row(ratio=0.5, ci=(0.4, 0.6)))[0], "LOSE")
            self.assertEqual(classify(_good_row(ratio=1.0, ci=(0.9, 1.1)))[0], "UNRESOLVED")

        def test_interval_touching_one_is_not_decided(self):
            self.assertEqual(classify(_good_row(ci=(1.0, 1.2)))[0], "UNRESOLVED")
            self.assertEqual(classify(_good_row(ratio=0.9, ci=(0.8, 1.0)))[0], "UNRESOLVED")

        def test_nan_never_wins(self):
            for over in ({"ratio": float("nan")}, {"ci": (float("nan"), 2.0)},
                         {"ci": (1.2, float("nan"))}, {"ci": None},
                         {"null_fsci": float("nan")}, {"null_scipy": float("nan")},
                         {"load": {"ambient_loadavg1": float("nan"),
                                   "foreign_running_median": 1.0}}):
                self.assertEqual(classify(_good_row(**over))[0], "UNRESOLVED", over)

        def test_null_failures_and_missing_nulls(self):
            self.assertEqual(classify(_good_row(null_fsci=1.03))[0], "UNRESOLVED")
            self.assertEqual(classify(_good_row(null_scipy=None))[0], "UNRESOLVED")
            spread = dict(null_kind="spread", null_fsci=1.04, null_scipy=1.02)
            self.assertEqual(classify(_good_row(**spread))[0], "WIN")
            self.assertEqual(classify(_good_row(**dict(spread, null_scipy=1.06)))[0], "UNRESOLVED")
            # A spread null below 1 is impossible for max/min, so it is a broken null.
            self.assertEqual(classify(_good_row(**dict(spread, null_fsci=0.99)))[0], "UNRESOLVED")

        def test_load_ceiling(self):
            calm = {"ambient_loadavg1": 1.0, "loadavg1_net_median": 1.0,
                    "foreign_running_median": 1.0}
            self.assertEqual(classify(_good_row(load=calm))[0], "WIN")
            for key in ("ambient_loadavg1", "loadavg1_net_median", "foreign_running_median"):
                hot = dict(calm, **{key: LOAD_CEILING + 0.1})
                self.assertEqual(classify(_good_row(load=hot))[0], "UNRESOLVED", key)
                missing = dict(calm, **{key: float("nan")})
                self.assertEqual(classify(_good_row(load=missing))[0], "UNRESOLVED", key)
            edge = {k: LOAD_CEILING for k in calm}
            self.assertEqual(classify(_good_row(load=edge))[0], "WIN")

        def test_harness_refusal_is_honoured(self):
            self.assertEqual(classify(_good_row(harness_refused=True))[0], "UNRESOLVED")

        def test_envelope(self):
            lo, hi = null_envelope(1.2, 1.02, 1.03, "spread")
            self.assertAlmostEqual(lo, 1.2 / (1.02 * 1.03))
            self.assertAlmostEqual(hi, 1.2 * 1.02 * 1.03)
            lo, hi = null_envelope(1.2, 0.98, 1.01, "centered")
            self.assertAlmostEqual(lo, 1.2 / ((1 / 0.98) * 1.01))
            self.assertTrue(all(math.isnan(v) for v in null_envelope(1.2, float("nan"), 1.0, "spread")))

        def test_bootstrap_is_deterministic_and_nan_safe(self):
            data = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 1.02]
            self.assertEqual(bootstrap_median_ci(data), bootstrap_median_ci(data))
            lo, hi = bootstrap_median_ci(data)
            self.assertLessEqual(lo, median(data))
            self.assertGreaterEqual(hi, median(data))
            self.assertTrue(all(math.isnan(v) for v in bootstrap_median_ci(data + [float("nan")])))
            self.assertTrue(all(math.isnan(v) for v in bootstrap_median_ci([1.0, 2.0])))

    class Provenance(unittest.TestCase):
        def test_complete_row_is_ok(self):
            self.assertEqual(validate_provenance(_good_row())[0], "OK")

        def test_missing_sha_is_invalid(self):
            for over in ({"fsci_elf_sha256": None}, {"fsci_elf_sha256": "abc"},
                         {"executed_elf_sha256": None}):
                self.assertEqual(classify(_good_row(**over))[0], "INVALID", over)

        def test_sha_mismatch_is_invalid(self):
            self.assertEqual(classify(_good_row(executed_elf_sha256="c" * 64))[0], "INVALID")

        def test_missing_worker_is_invalid(self):
            self.assertEqual(classify(_good_row(host=None))[0], "INVALID")
            self.assertEqual(classify(_good_row(host=""))[0], "INVALID")

        def test_incumbent_must_be_pinned_and_genuine(self):
            self.assertEqual(classify(_good_row(incumbent=None))[0], "INVALID")
            bad = {"genuine": True, "scipy": "1.18.1", "numpy": PINNED_NUMPY}
            self.assertEqual(classify(_good_row(incumbent=bad))[0], "INVALID")
            fake = {"genuine": False, "scipy": PINNED_SCIPY, "numpy": PINNED_NUMPY}
            self.assertEqual(classify(_good_row(incumbent=fake))[0], "INVALID")

        def test_negative_case_1_cached_scipy_arm_is_invalid_not_win(self):
            # A row whose SciPy number was copied from a ledger, with a ratio that would
            # otherwise be a clean WIN.
            row = _good_row(scipy_source="cached:docs/perf_ledger_cc.md", ratio=3.0, ci=(2.9, 3.1))
            cls, reasons = classify(row)
            self.assertEqual(cls, "INVALID")
            self.assertTrue(any("not live in the same invocation" in r for r in reasons))

        def test_negative_case_2_self_comparison_flagged_by_sha(self):
            sha = "a" * 64
            # At ratio ~1 inside its null ...
            row = _good_row(incumbent_elf_sha256=sha, ratio=1.003, ci=(0.99, 1.02))
            self.assertEqual(classify(row)[0], "SELF_COMPARISON")
            # ... and even when noise hands it a ratio that would otherwise WIN: the flag is
            # the sha check, not the ratio.
            row = _good_row(incumbent_elf_sha256=sha, ratio=1.3, ci=(1.25, 1.35))
            self.assertEqual(classify(row)[0], "SELF_COMPARISON")
            # The same row with a different incumbent sha is an ordinary WIN, so the check
            # really keys on the sha.
            self.assertEqual(classify(dict(row, incumbent_elf_sha256="b" * 64))[0], "WIN")

    class Parsers(unittest.TestCase):
        def test_case_style_dedupes_and_parses(self):
            line = ("case=n200000 op=erfinv fsci=3.100ms scipy=2.200ms scipy/fsci=0.710x "
                    "null_fsci=1.010 null_scipy=1.020 CHECK max_abs=0 max_rel=0")
            rows = parse_case_style([line, line, "noise"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case"], "n200000 op=erfinv")
            self.assertAlmostEqual(rows[0]["raw_value"], 0.71)
            self.assertEqual(rows[0]["null_kind"], "spread")

        def test_eigsh_case_line(self):
            line = ("case=shift_invert n=4096 nnz=20224 k=6 which=LM sigma=Some(0.5) "
                    "fsci=10.000ms scipy=11.000ms scipy/fsci=1.100x null_fsci=1.010 "
                    "null_scipy=1.030 iterations=3 nmatvec=40 CHECK ok")
            rows = parse_case_style([line])
            self.assertEqual(rows[0]["case"], "shift_invert n=4096 k=6 which=LM")

        def test_fft_is_fsci_over_scipy(self):
            line = ("RESULT mode=rfft n=65536 repeats=6 fsci_ms=1.0 scipy_ms=2.0 "
                    "fsci_over_scipy=0.500000 fsci_aa=1.001 scipy_aa=0.999 fsci_checksum=0 "
                    "scipy_checksum=1")
            row = finish_row(_with_provenance(parse_fft([line])[0]))
            self.assertAlmostEqual(row["ratio"], 2.0)
            self.assertEqual(row["ci_kind"], "null_envelope")

        def test_bdf_block(self):
            lines = [
                "fixture=dense-allpairs n=512 rounds=21 reps=1 method=BDF t_span=[0,1]",
                "NULL-ours   median=1.001 ci95=[0.99,1.01] cv=1% (provenance only)",
                "NULL-scipy  median=0.998 ci95=[0.99,1.01] cv=1% (provenance only)",
                "raw_samples_seconds: ours=[1.0, 1.0] scipy=[0.5, 0.5] ratios=[0.5, 0.5] "
                "null_ours=[1.0] null_scipy=[1.0]",
                "Incumbent ratio: SciPy / FrankenSciPy = 0.4972x (bootstrap-median "
                "ci95=[0.4902,0.5044], cv=1% provenance only)",
                "median-CI gate: worst_null_edge=1.01 required=1.02 ratio_ci=[0.49,0.50] "
                "null_margin=2x cv_used_for_decision=false => DECIDED FRANKENSCIPY LOSS",
            ]
            rows = parse_bdf(lines)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case"], "dense-allpairs n=512 method=BDF")
            self.assertEqual(rows[0]["raw_ci"], (0.4902, 0.5044))
            self.assertEqual(rows[0]["harness_verdict"], "DECIDED FRANKENSCIPY LOSS")
            self.assertFalse(rows[0]["harness_refused"])

        def test_eigh_inverts_and_scipyN_has_no_scipy_null(self):
            lines = [
                "--- n=768 impl=native ---",
                "NULL fsci/fsci   a=  100.000ms b=  100.000ms ratio_p50=1.0010x ci95=[0.9950,1.0060] cv=1.00% rounds=9",
                "NULL sp1/sp1     a=   30.000ms b=   30.000ms ratio_p50=0.9990x ci95=[0.9900,1.0100] cv=1.00% rounds=9",
                "fsci/scipy1      a=  100.000ms b=   30.000ms ratio_p50=3.3000x ci95=[3.2000,3.4000] cv=1.00% rounds=9",
                "fsci/scipyN      a=  100.000ms b=   10.000ms ratio_p50=10.0000x ci95=[9.0000,11.0000] cv=1.00% rounds=9",
            ]
            rows = parse_eigh(lines)
            self.assertEqual([r["variant"] for r in rows], ["scipy1", "scipyN"])
            done = finish_row(_with_provenance(rows[0]))
            self.assertAlmostEqual(done["ratio"], 1 / 3.3)
            self.assertEqual(done["class"], "LOSE")
            self.assertIsNone(rows[1]["null_scipy"])
            done = finish_row(_with_provenance(rows[1]))
            self.assertEqual(done["class"], "UNRESOLVED")

        def test_chol_rows_get_replicate_bootstrap(self):
            lines = [f"n=512 rep={i} fsci=1.0e-2s scipy1=1.{i}e-2s scipyN=5.0e-3s r1=1.{i}00x "
                     f"rN=0.500x | x" for i in range(5)]
            lines.append("n=512 RESULT scipy1/fsci=1.200x scipyN/fsci=0.500x null_fsci=1.010 "
                         "null_scipy1=1.020 ambient=10.00/30.00 load_peak=12[ours, not gated] "
                         "mhz_fsci=3000 mhz_scipy=3000 clock_ratio=1.000 gates=PASS loadavg_post=x")
            rows = parse_chol(lines)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(rows[0]["raw_samples"]), 5)
            done = finish_row(_with_provenance(rows[0]))
            self.assertEqual(done["ci_kind"], "builder_bootstrap95_over_rounds")

        def test_splu_verdicts(self):
            base = ["fixture_sha256=ab n=16384 nnz=81408",
                    "NULL scipy/scipy=1.0040 NULL fsci/fsci=0.9990 bound=+/-0.02 null_edge=0.004 x"]
            ok = parse_splu(base + ["Incumbent ratio: SciPy / FrankenSciPy = 0.4421x  "
                                    "ci95=[0.4355,0.4492]  rounds=21  verdict=ADMISSIBLE: "
                                    "FrankenSciPy SLOWER"])
            self.assertFalse(ok[0]["harness_refused"])
            self.assertEqual((ok[0]["null_fsci"], ok[0]["null_scipy"]), (0.999, 1.004))
            void = parse_splu(base + ["Incumbent ratio: SciPy / FrankenSciPy = 0.9000x  "
                                      "ci95=[0.8,1.0]  rounds=21  verdict=NULL-FAILED (row void)"])
            self.assertTrue(void[0]["harness_refused"])

        def test_header(self):
            h = parse_header([
                "elf_sha256=" + "d" * 64,
                "scipy_incumbent: python=/p pythonpath=<default> scipy=1.17.1 numpy=2.4.3 "
                "fsci_loaded=false genuine=true pinned_scipy=1.17.1 pinned_numpy=2.4.3 blas=x",
                "READY scipy=1.17.1 numpy=2.4.3 genuine=True"])
            self.assertEqual(h["fsci_elf_sha256"], "d" * 64)
            self.assertTrue(h["incumbent"]["genuine"])
            self.assertTrue(h["ready_genuine"])
            h = parse_header(["READY scipy=1.18.1 genuine=False"])
            self.assertIsNone(h["incumbent"])
            self.assertFalse(h["ready_genuine"])

    class PrimarySelection(unittest.TestCase):
        @staticmethod
        def _r(tag, started, reasons, ratio=1.0):
            return {"key": "k", "case": "c", "variant": "v", "tag": tag, "started_utc": started,
                    "class_reasons": reasons, "ratio": ratio}

        def test_earliest_replicate_is_primary_whatever_its_ratio(self):
            a = self._r("a", "1", ["interval [0.9,1.1] overlaps 1"], ratio=0.9)
            b = self._r("b", "2", ["interval [1.4,1.6] entirely above 1"], ratio=1.5)
            select_primary([b, a])
            self.assertTrue(a["primary"])
            self.assertEqual(b["superseded_by"], "a")

        def test_only_a_load_gated_row_is_superseded(self):
            a = self._r("a", "1", ["ambient loadavg1 25.0 above ceiling 20.0"])
            b = self._r("b", "2", ["foreign runnable median 22 above ceiling 20.0"])
            c = self._r("c", "3", ["interval [0.4,0.5] entirely below 1"])
            d = self._r("d", "4", ["interval [1.4,1.6] entirely above 1"])
            select_primary([a, b, c, d])
            self.assertEqual([r["primary"] for r in (a, b, c, d)], [False, False, True, False])

        def test_all_load_gated_keeps_the_earliest(self):
            a = self._r("a", "1", ["ambient loadavg1 25.0 above ceiling 20.0"])
            b = self._r("b", "2", ["ambient loadavg1 24.0 above ceiling 20.0"])
            select_primary([b, a])
            self.assertTrue(a["primary"])

    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for case in (Normaliser, Classifier, Provenance, Parsers, PrimarySelection):
        suite.addTests(loader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--self-test":
        return run_self_test()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="execute one harness and capture its evidence")
    run.add_argument("--out-dir", required=True)
    run.add_argument("--tag", required=True)
    run.add_argument("--harness", required=True, choices=sorted(HARNESSES))
    run.add_argument("--binary", required=True)
    run.add_argument("--pin", type=int, default=None)
    run.add_argument("--env", action="append", default=[])
    run.add_argument("--build-manifest", default=None)
    run.add_argument("--timeout", type=float, default=5400.0)
    run.add_argument("--max-wait", type=float, default=1200.0,
                     help="seconds to wait for 1-min loadavg <= the ceiling before launching")
    run.add_argument("harness_args", nargs="*")
    build = sub.add_parser("build", help="parse, classify and emit the scoreboard")
    build.add_argument("--artifact-dir", required=True)
    build.add_argument("--out-json", required=True)
    build.add_argument("--out-md", required=False)
    build.add_argument("--build-manifest", required=False,
                       help="rch build manifest (worktree-relative) naming BUILT/MISSING bins")
    plant = sub.add_parser("plant", help="write the bead's two negative cases from run data")
    plant.add_argument("--artifact-dir", required=True)
    plant.add_argument("--gmres-tag", required=True)
    plant.add_argument("--bdf-tag", required=True)
    plant.add_argument("--splu-tag", required=True)
    args = parser.parse_args(argv)
    return {"run": cmd_run, "build": cmd_build, "plant": cmd_plant}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
