#!/usr/bin/env python3
"""Per-iteration a/b decomposition for the qmr campaign.

Reads the raw cell files produced by perf_sparse_vs_scipy and fits
    us_per_iteration(n) = a + b * n
independently for each arm, exactly as the lsqr run did. Cells whose iteration
counts differ between arms are excluded from the fit automatically, and cells
that aborted are skipped with a printed reason. No cell is dropped silently.
"""

import re
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent


def parse(path):
    text = path.read_text()
    if "ABORT:" in text:
        reason = re.search(r"ABORT: (.*)", text).group(1)
        exec_m = re.search(
            r"execution: ours converged=(\S+) iterations=(\d+) "
            r"reported_residual=(\S+) \| scipy info=(-?\d+) "
            r"counted_inner_iterations=(\d+)",
            text,
        )
        return {
            "aborted": True,
            "reason": reason,
            "ours_converged": exec_m.group(1) if exec_m else "?",
            "it_o": int(exec_m.group(2)) if exec_m else 0,
            "it_s": int(exec_m.group(5)) if exec_m else 0,
            "resid_o": exec_m.group(3) if exec_m else "?",
        }
    if "Incumbent ratio:" not in text:
        return {"aborted": True, "reason": "cell incomplete (no Incumbent ratio line)",
                "ours_converged": "?", "it_o": 0, "it_s": 0, "resid_o": "?"}
    n = int(re.search(r"\bn=(\d+)\b", text).group(1))
    side = int(re.search(r"\bside=(\d+)\b", text).group(1))
    it_o = int(re.search(r"ours converged=\S+ iterations=(\d+)", text).group(1))
    it_s = int(re.search(r"counted_inner_iterations=(\d+)", text).group(1))
    reps = int(re.search(r"calibration repetitions=(\d+)", text).group(1))
    ours_ms = float(re.search(r"OURS p50=([\d.]+)ms/rep", text).group(1))
    scipy_ms = float(re.search(r"SCIPY p50=([\d.]+)ms/rep", text).group(1))
    ratio = float(re.search(r"SciPy / FrankenSciPy = ([\d.]+)x", text).group(1))
    # p50 is already per repetition; convert to microseconds per iteration.
    return {
        "aborted": False,
        "side": side,
        "n": n,
        "it_o": it_o,
        "it_s": it_s,
        "reps": reps,
        "us_it_o": ours_ms * 1000.0 / it_o,
        "us_it_s": scipy_ms * 1000.0 / it_s,
        "ratio": ratio,
        "matched": it_o == it_s,
    }


def ols(xs, ys):
    k = len(xs)
    mx = sum(xs) / k
    my = sum(ys) / k
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def report(method, paths):
    print(f"\n===== {method} =====")
    cells = []
    for p in paths:
        c = parse(p)
        if c["aborted"]:
            print(
                f"  {p.name}: SKIPPED — {c['reason']} "
                f"(ours converged={c['ours_converged']} it {c['it_o']}/{c['it_s']} "
                f"resid={c['resid_o']})"
            )
            continue
        cells.append(c)
    if not cells:
        print("  no usable cells")
        return None
    # "whole-job" is the measured wall-clock ratio a caller experiences and it
    # includes any difference in counted work. "per-iteration" divides that out.
    print(f"  {'side':>5} {'n':>7} {'it o/s':>12} {'match':>6} "
          f"{'us/it ours':>11} {'us/it scipy':>12} {'whole-job':>10} "
          f"{'per-iter':>9}")
    for c in cells:
        per_it = c["us_it_s"] / c["us_it_o"]
        print(
            f"  {c['side']:>5} {c['n']:>7} {c['it_o']:>5}/{c['it_s']:<6} "
            f"{'YES' if c['matched'] else 'NO':>6} "
            f"{c['us_it_o']:>11.3f} {c['us_it_s']:>12.3f} {c['ratio']:>9.4f}x "
            f"{per_it:>8.4f}x"
        )
    fit = [c for c in cells if c["matched"]]
    dropped = [c for c in cells if not c["matched"]]
    for c in dropped:
        print(f"  EXCLUDED from fit (counts differ): side={c['side']}")
    if len(fit) < 2:
        print("  fewer than 2 matched cells — no fit")
        return None
    ns = [c["n"] for c in fit]
    ao, bo, r2o = ols(ns, [c["us_it_o"] for c in fit])
    a_s, bs, r2s = ols(ns, [c["us_it_s"] for c in fit])
    print(f"  fit over {len(fit)} matched cells, n range {min(ns)}..{max(ns)}")
    print(f"    ours   a={ao:+9.3f} us   b={bo:.6f} us/unknown   R2={r2o:.4f}")
    print(f"    scipy  a={a_s:+9.3f} us   b={bs:.6f} us/unknown   R2={r2s:.4f}")
    print(f"    marginal per-unknown ours/scipy = {bo / bs:.3f}x")
    if bo > bs:
        print(f"    crossover n* = {a_s / (bo - bs):,.0f} "
              f"(side ~{(a_s / (bo - bs)) ** 0.5:.0f})")
    else:
        print("    ours is marginally CHEAPER — no crossover; ratio asymptotes "
              f"to {bs / bo:.3f}x")
    return {"a_scipy": a_s, "b_scipy": bs, "a_ours": ao, "b_ours": bo}


def main():
    q = report("qmr", sorted(RAW.glob("qmr_side_*.txt"),
                             key=lambda p: int(re.search(r"(\d+)\.txt", p.name).group(1))))
    ls = report("lsqr (same-session re-measurement)",
                sorted(RAW.glob("lsqr_resample_side_*.txt"),
                       key=lambda p: int(re.search(r"(\d+)\.txt", p.name).group(1))))
    if q and ls:
        print("\n===== P1: does the fixed tax scale with the dispatch count? =====")
        print(f"  a_scipy(qmr)  = {q['a_scipy']:.3f} us   (D=43)")
        print(f"  a_scipy(lsqr) = {ls['a_scipy']:.3f} us   (D=18)")
        print(f"  measured ratio = {q['a_scipy'] / ls['a_scipy']:.3f}x   "
              f"predicted 2.389x, interval [2.0, 2.8]")
        print(f"  implied us per dispatch unit: qmr={q['a_scipy'] / 43:.3f}  "
              f"lsqr={ls['a_scipy'] / 18:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
