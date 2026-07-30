#!/usr/bin/env python3
"""Decompose an iteration-matched GMRES incumbent ratio into
   per-iteration cost = fixed overhead + marginal cost per unknown.

Per-cell only: time is divided by that cell's OWN matched iteration count.
No ratio or time is ever averaged across cells with different counts.
"""
import re
import sys
import glob


def parse(path):
    txt = open(path).read()
    cells = []
    for blk in txt.split("fixture=")[1:]:
        m_n = re.search(r"side=(\d+) n=(\d+) nnz=(\d+)", blk)
        m_it = re.search(
            r"ours converged=(\w+) iterations=(\d+).*?counted_inner_iterations=(\d+)", blk
        )
        m_t = re.search(r"OURS p50=([\d.]+)ms/rep SCIPY p50=([\d.]+)ms/rep", blk)
        m_r = re.search(r"Incumbent ratio: SciPy / FrankenSciPy = ([\d.]+)x", blk)
        m_q = re.search(r"host_wide_quiescence_measurement=clear.*?maximum_busy_fraction=([\d.]+)", blk)
        if not (m_n and m_it and m_t):
            continue
        cells.append(
            dict(
                side=int(m_n.group(1)),
                n=int(m_n.group(2)),
                nnz=int(m_n.group(3)),
                conv=m_it.group(1),
                it_ours=int(m_it.group(2)),
                it_scipy=int(m_it.group(3)),
                t_ours=float(m_t.group(1)),
                t_scipy=float(m_t.group(2)),
                ratio=float(m_r.group(1)) if m_r else None,
                busy=float(m_q.group(1)) if m_q else None,
            )
        )
    return cells


def ols(xs, ys):
    k = len(xs)
    mx = sum(xs) / k
    my = sum(ys) / k
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    icpt = my - slope * mx
    ss_t = sum((y - my) ** 2 for y in ys)
    ss_r = sum((y - (icpt + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_r / ss_t if ss_t else float("nan")
    return icpt, slope, r2


def main(paths):
    cells = []
    for p in paths:
        cells += parse(p)
    cells.sort(key=lambda c: c["n"])
    keep = []
    print(f"{'side':>5} {'n':>7} {'it_o':>5} {'it_s':>5} {'match':>6} "
          f"{'us/it ours':>11} {'us/it scipy':>12} {'ratio':>8} {'maxbusy':>8}")
    for c in cells:
        matched = c["it_ours"] == c["it_scipy"]
        po = c["t_ours"] * 1000.0 / c["it_ours"]
        ps = c["t_scipy"] * 1000.0 / c["it_scipy"]
        c["po"], c["ps"], c["matched"] = po, ps, matched
        print(f"{c['side']:>5} {c['n']:>7} {c['it_ours']:>5} {c['it_scipy']:>5} "
              f"{('YES' if matched else 'NO'):>6} {po:>11.3f} {ps:>12.3f} "
              f"{(c['ratio'] if c['ratio'] else float('nan')):>8.4f} "
              f"{(c['busy'] if c['busy'] is not None else float('nan')):>8.3f}")
        if matched:
            keep.append(c)
        else:
            print(f"      ^^ EXCLUDED from fit: iteration counts differ "
                  f"({c['it_ours']} vs {c['it_scipy']})")

    if len(keep) < 3:
        print(f"\nonly {len(keep)} iteration-matched cells; need >=3 for a fit")
        return
    ns = [c["n"] for c in keep]
    io, so, r2o = ols(ns, [c["po"] for c in keep])
    isc, ssc, r2s = ols(ns, [c["ps"] for c in keep])
    print(f"\nper-iteration cost model  us/iter = a + b*n   ({len(keep)} matched cells)")
    print(f"  FrankenSciPy : a={io:9.3f} us   b={so:.6f} us/unknown   R2={r2o:.4f}")
    print(f"  SciPy 1.17.1 : a={isc:9.3f} us   b={ssc:.6f} us/unknown   R2={r2s:.4f}")
    print(f"\n  fixed per-iteration overhead SciPy - ours = {isc - io:.3f} us/iter")
    if ssc != so:
        print(f"  marginal per-unknown ours / SciPy         = {so / ssc:.3f}x")
        xc = (isc - io) / (so - ssc)
        if xc > 0:
            print(f"  predicted crossover n (equal per-iteration cost) = {xc:,.0f}"
                  f"  (side ~= {xc ** 0.5:.0f})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("sweep/side_*.txt"))
    main(args)
