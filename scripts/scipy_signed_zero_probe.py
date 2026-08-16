#!/usr/bin/env python3
"""Enumerate scipy.special functions that DISTINGUISH +0.0 from -0.0 (frankenscipy-eaqem).

Behavioural probe: prints observed values only. It times nothing and makes no
performance claim.

WHY THIS EXISTS. In IEEE arithmetic `x == 0.0` is TRUE for -0.0, so any branch
written that way silently collapses the two zeros. That only matters where the
incumbent distinguishes them -- and scipy does, in 14 unary functions, three of
which flip SIGN rather than merely returning a signed zero:

    gamma     (+0.0) = +inf   (-0.0) = -inf
    psi       (+0.0) = -inf   (-0.0) = +inf     <-- note the inversion
    gammasgn  (+0.0) =  1.0   (-0.0) = -1.0

`gammasgn_scalar` in crates/fsci-special/src/gamma.rs already handles this
correctly (`if x == 0.0 { if x.is_sign_negative() { -1.0 } else { 1.0 } }`),
which is the positive control proving the pattern is known here.
`digamma_scalar` in crates/fsci-special/src/convenience.rs does not: its guard
`x <= 0.0 && x == x.floor()` returns NaN for both zeros where scipy returns a
signed infinity.

Run:  python3 scripts/scipy_signed_zero_probe.py
Exits non-zero if the set of sign-distinguishing functions ever changes, which
is the point: it is a re-runnable check, not a transcript.
"""

from __future__ import annotations

import struct
import sys
import warnings

# Measured against scipy 1.17.1 on 2026-08-16. A change here is a real event:
# either scipy changed its zero handling or the installed build differs.
EXPECTED = {
    "cbrt", "dawsn", "digamma", "erf", "erfi", "expm1", "gamma", "gammasgn",
    "itairy", "j1", "lambertw", "log1p", "psi", "rgamma", "round",
}


def _bits(value) -> bytes | None:
    try:
        return struct.pack("<d", float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def main() -> int:
    warnings.simplefilter("ignore")
    try:
        import scipy
        import scipy.special as sp
    except Exception as exc:  # pragma: no cover - environment probe
        print(f"SKIP: scipy unavailable ({exc})")
        return 0

    print(f"scipy {scipy.__version__}")
    found: dict[str, tuple[str, str]] = {}
    for name in dir(sp):
        if name.startswith("_"):
            continue
        fn = getattr(sp, name)
        # Exception classes are callable; they are not functions under test.
        if not callable(fn) or isinstance(fn, type):
            continue
        try:
            pos, neg = fn(0.0), fn(-0.0)
        except Exception:
            continue
        # Tuple-returning functions (itairy) distinguish if ANY element does.
        if isinstance(pos, tuple) and isinstance(neg, tuple):
            if any(_bits(a) != _bits(b) for a, b in zip(pos, neg)):
                found[name] = (repr(pos)[:40], repr(neg)[:40])
            continue
        bp, bn = _bits(pos), _bits(neg)
        if bp is None or bn is None:
            # complex results (lambertw) compare by repr instead
            if repr(pos) != repr(neg):
                found[name] = (repr(pos), repr(neg))
            continue
        if bp != bn:
            found[name] = (repr(pos), repr(neg))

    for name in sorted(found):
        pos, neg = found[name]
        print(f"  {name:12s} f(+0.0)={pos:24s} f(-0.0)={neg}")

    got = set(found)
    missing, extra = EXPECTED - got, got - EXPECTED
    print()
    if missing or extra:
        if missing:
            print(f"FAIL: no longer sign-distinguishing: {sorted(missing)}")
        if extra:
            print(f"FAIL: newly sign-distinguishing: {sorted(extra)}")
        return 1
    print(f"OK: {len(got)} sign-distinguishing functions, exactly as recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
