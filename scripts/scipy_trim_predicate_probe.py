#!/usr/bin/env python3
"""Probe scipy's ACTUAL rejection predicate for the trim family (frankenscipy-tb5es).

Behavioural probe: prints observed values only. It times nothing and makes no
performance claim.

WHY THIS EXISTS. `scipy.stats.trim_mean` and `scipy.stats.trimboth` both raise
`ValueError: Proportion too big.`, and the obvious reading -- reject when
`proportiontocut > 0.5` -- is WRONG in both directions. The real rule, from
scipy/stats/_stats_py.py, is on the derived cut indices:

    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    trim_mean raises iff lowercut >  uppercut
    trimboth  raises iff lowercut >= uppercut

Two consequences a proportion-only threshold gets wrong:

  * at n=5, prop=0.51 scipy ANSWERS (lowercut=int(2.55)=2, uppercut=3), so a
    `> 0.5` rejection refuses input the incumbent accepts;
  * at n=10, prop=0.5 `trim_mean` returns nan while `trimboth` RAISES -- the two
    functions genuinely disagree wherever lowercut == uppercut, so a shared
    validation helper is necessarily wrong for one of them.

Run:  python3 scripts/scipy_trim_predicate_probe.py
Exits non-zero if live scipy disagrees with the predicate above, which is the
point: it is a re-runnable check, not a transcript.
"""

from __future__ import annotations

import sys
import warnings


def main() -> int:
    warnings.simplefilter("ignore")
    try:
        import numpy as np
        import scipy
        from scipy.stats import trim_mean, trimboth
    except Exception as exc:  # pragma: no cover - environment probe
        print(f"SKIP: scipy/numpy unavailable ({exc})")
        return 0

    print(f"scipy {scipy.__version__} / numpy {np.__version__}")
    header = f"{'n':>4} {'prop':>5} {'lowcut':>7} {'upcut':>6} | {'trim_mean':>22} | {'trimboth':>22}"
    print(header)
    print("-" * len(header))

    mismatches = 0
    for n in (4, 5, 10, 11):
        data = [float(i) for i in range(1, n + 1)]
        for prop in (0.4, 0.45, 0.5, 0.51, 0.6):
            lowercut = int(prop * n)
            uppercut = n - lowercut

            def observe(fn):
                try:
                    return repr(fn(data, prop))[:22], False
                except ValueError:
                    return "RAISES", True

            mean_txt, mean_raised = observe(trim_mean)
            both_txt, both_raised = observe(trimboth)

            # The predicate this probe asserts.
            mean_should_raise = lowercut > uppercut
            both_should_raise = lowercut >= uppercut

            flag = ""
            if mean_raised != mean_should_raise:
                flag += "  <-- trim_mean DISAGREES with lowercut > uppercut"
                mismatches += 1
            if both_raised != both_should_raise:
                flag += "  <-- trimboth DISAGREES with lowercut >= uppercut"
                mismatches += 1

            print(
                f"{n:>4} {prop:>5} {lowercut:>7} {uppercut:>6} | "
                f"{mean_txt:>22} | {both_txt:>22}{flag}"
            )

    print()
    if mismatches:
        print(f"FAIL: {mismatches} observed outcome(s) contradict the documented predicate")
        return 1
    print("OK: every observed outcome matches the cut-index predicate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
