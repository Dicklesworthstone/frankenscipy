#!/usr/bin/env python3
"""Regenerate the SciPy symbol-coverage census. Read-only."""
import importlib, inspect, json, re, subprocess, sys
from pathlib import Path

REPO = Path("/data/projects/frankenscipy")
MODMAP = {
    "scipy.cluster": "fsci-cluster", "scipy.constants": "fsci-constants",
    "scipy.datasets": "fsci-datasets", "scipy.fft": "fsci-fft",
    "scipy.integrate": "fsci-integrate", "scipy.interpolate": "fsci-interpolate",
    "scipy.io": "fsci-io", "scipy.linalg": "fsci-linalg", "scipy.ndimage": "fsci-ndimage",
    "scipy.odr": "fsci-odr", "scipy.optimize": "fsci-opt", "scipy.signal": "fsci-signal",
    "scipy.sparse": "fsci-sparse", "scipy.spatial": "fsci-spatial",
    "scipy.special": "fsci-special", "scipy.stats": "fsci-stats",
}

def scipy_symbols(mod):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        return {}, str(e)
    names = getattr(m, "__all__", None) or [n for n in dir(m) if not n.startswith("_")]
    out = {}
    for n in names:
        try:
            obj = getattr(m, n)
        except Exception:
            continue
        if inspect.ismodule(obj):
            continue
        if callable(obj) or isinstance(obj, type):
            out[n] = "class" if inspect.isclass(obj) else "fn"
    return out, None

def rust_symbols(crate):
    d = REPO / "crates" / crate / "src"
    if not d.exists():
        return set()
    names = set()
    # NOTE: fn names are NOT always snake_case here — scipy spellings like `check_COLA`
    # are preserved verbatim behind #[allow(non_snake_case)]. A lowercase-only pattern
    # reported those as MISSING when they are implemented, understating coverage.
    pat = re.compile(
        r"pub (?:async )?fn ([A-Za-z0-9_]+)|pub struct ([A-Za-z0-9_]+)|"
        r"pub enum ([A-Za-z0-9_]+)|pub type ([A-Za-z0-9_]+)|pub const ([A-Za-z0-9_]+)")
    for f in d.rglob("*.rs"):
        try:
            for m in pat.finditer(f.read_text(errors="replace")):
                names.add(next(g for g in m.groups() if g))
        except Exception:
            pass
    return names

def reexports(crate):
    """`pub use` re-export names — several crates expose their surface this way."""
    d = REPO / "crates" / crate / "src"
    names = set()
    if not d.exists():
        return names
    for f in d.rglob("*.rs"):
        try:
            txt = f.read_text(errors="replace")
        except Exception:
            continue
        for m in re.finditer(r"pub use [^;]+;", txt):
            names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", m.group(0)))
    return names


def norm(n):
    return n.lower().replace("_", "")

rows, detail = [], {}
for mod, crate in sorted(MODMAP.items()):
    syms, err = scipy_symbols(mod)
    if err:
        rows.append((mod, crate, 0, 0, err)); continue
    have = {norm(x) for x in (rust_symbols(crate) | reexports(crate))}
    missing = sorted(n for n in syms if norm(n) not in have)
    covered = len(syms) - len(missing)
    rows.append((mod, crate, len(syms), covered, ""))
    detail[mod] = missing

tot = sum(r[2] for r in rows); cov = sum(r[3] for r in rows)
print(f"scipy {importlib.import_module('scipy').__version__}   TOTAL {cov}/{tot} = {100*cov/tot:.1f}%\n")
print(f"{'module':22s} {'scipy':>6s} {'covered':>8s} {'missing':>8s}  cov%")
for mod, crate, n, c, err in sorted(rows, key=lambda r: (r[3]-r[2])):
    if err: print(f"{mod:22s} ERROR {err[:40]}"); continue
    print(f"{mod:22s} {n:6d} {c:8d} {n-c:8d}  {100*c/n if n else 0:5.1f}%")
Path(sys.argv[1]).write_text(json.dumps(detail, indent=1))
