"""Which public fsci entry points with SciPy-shaped names have no differential coverage?

Motivated by frankenscipy-icozs: `RbfInterpolator` implements SciPy's non-default `degree=-1`
variant under SciPy's default name, verified numerically, and it survived because nothing anywhere
compared it to SciPy. This finds what else is in that position.

CORPUS -- and getting this wrong is the easy mistake. Differential tests do NOT all live under
`crates/fsci-conformance/tests/`. They are split:

    crates/fsci-conformance/tests/     731 diff_*.rs
    crates/<crate>/src/bin/            200 more diff_*.rs, per-crate

A first pass over only the conformance crate reported 201 uncovered entry points. That was
INFLATED: it counted `cholesky_banded`, `expm_frechet`, `convolve1d`, `fourier_gaussian` and
others that have a `diff_*.rs` in their own crate's `src/bin/`. This scans both, plus the python
oracles and the conformance lib.

A public `fn`/`struct`/`enum` counts UNCOVERED only if BOTH hold:
  1. neither its name nor its snake_case form occurs anywhere in that corpus, and
  2. the name actually EXISTS in the corresponding SciPy module.

Filter 2 is what keeps the number honest: without it the raw count includes internal plumbing
exposed as `pub` for benchmarking (`bench_trailing_syrk_prepare`, `correlate1d_perwindow_ref`).

The result is a TRIAGE BACKLOG, not a defect count. It is a name-level heuristic: a name may be
exercised indirectly or under another name. Verify any individual claim with
`grep -rli <name> crates/fsci-conformance/{tests,src} crates/*/src/bin`.

FAIL-CLOSED (frankenscipy-olv0j.6). This script used to `continue` past any SciPy module it
could not import, so under an interpreter without SciPy it skipped every crate and reported
"0 uncovered" -- and bead frankenscipy-ivxx6 was closed on that 0 (the pinned interpreter
reports 167). It now:
  * refuses to run unless scipy and numpy are the pinned pair (override the pin with
    FSCI_AUDIT_SCIPY / FSCI_AUDIT_NUMPY only on purpose), and exits 2 if any module is missing;
  * counts a name as referenced only from RUST sources (tests, src/bin diff files, the
    conformance lib), not from the Python oracle scripts: a name that appears only on the SciPy
    side of an oracle string is exactly the uncompared case this audit exists to find;
  * scans every `src/**/*.rs` of a crate (not only lib.rs), skipping `src/bin/`.
Run it with the pinned incumbent: /home/ubuntu/.local/bin/python3.13 scripts/conformance_coverage_audit.py
"""
import importlib
import os
import pathlib
import re
import sys

ROOT = pathlib.Path('/data/projects/frankenscipy')
PINNED_SCIPY = os.environ.get('FSCI_AUDIT_SCIPY', '1.17.1')
PINNED_NUMPY = os.environ.get('FSCI_AUDIT_NUMPY', '2.4.3')

try:
    import numpy
    import scipy
except ImportError as err:
    sys.exit(f"REFUSING: {sys.executable} cannot import scipy/numpy ({err}); a run without the "
             f"oracle would report every entry point as covered.")
if scipy.__version__ != PINNED_SCIPY or numpy.__version__ != PINNED_NUMPY:
    sys.exit(f"REFUSING: {sys.executable} has scipy {scipy.__version__} / numpy {numpy.__version__}, "
             f"pinned pair is {PINNED_SCIPY} / {PINNED_NUMPY}")
print(f"interpreter: {sys.executable}  scipy {scipy.__version__}  numpy {numpy.__version__}")

corpus_files = []
conf = ROOT / 'crates' / 'fsci-conformance'
for pat in ('tests/*.rs', 'src/*.rs'):
    corpus_files.extend(conf.glob(pat))
# per-crate differential tests, which the first pass missed entirely
corpus_files.extend(ROOT.glob('crates/*/src/bin/diff_*.rs'))

texts = []
for f in corpus_files:
    try:
        texts.append(f.read_text(errors='ignore').lower())
    except OSError:
        pass
blob = '\n'.join(texts)
n_diff = len(list(conf.glob('tests/diff_*.rs'))) + len(list(ROOT.glob('crates/*/src/bin/diff_*.rs')))
print(f"corpus: {len(corpus_files)} files, {len(blob):,} chars, {n_diff} diff_*.rs total")
print()

PUB_RE = re.compile(r'^pub (?:fn|struct|enum) ([A-Za-z_][A-Za-z0-9_]*)', re.MULTILINE)
CRATE_TO_MOD = {
    'fsci-signal': 'scipy.signal', 'fsci-linalg': 'scipy.linalg',
    'fsci-ndimage': 'scipy.ndimage', 'fsci-interpolate': 'scipy.interpolate',
    'fsci-stats': 'scipy.stats', 'fsci-spatial': 'scipy.spatial',
    'fsci-cluster': 'scipy.cluster.hierarchy', 'fsci-io': 'scipy.io',
    'fsci-opt': 'scipy.optimize', 'fsci-sparse': 'scipy.sparse',
    'fsci-integrate': 'scipy.integrate', 'fsci-fft': 'scipy.fft',
    'fsci-special': 'scipy.special', 'fsci-constants': 'scipy.constants',
    'fsci-odr': 'scipy.odr', 'fsci-datasets': 'scipy.datasets',
}


def snake(n):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', n).lower()


total = 0
findings = []
missing_modules = []
for crate, modname in sorted(CRATE_TO_MOD.items()):
    src = ROOT / 'crates' / crate / 'src'
    sources = [p for p in src.rglob('*.rs') if 'bin' not in p.relative_to(src).parts]
    if not sources:
        missing_modules.append(f"{crate}: no sources under {src}")
        continue
    try:
        mod = importlib.import_module(modname)
    except Exception as err:
        missing_modules.append(f"{modname}: {err}")
        continue
    names = set()
    for path in sources:
        names.update(PUB_RE.findall(path.read_text(errors='ignore')))
    hits = set()
    for n in sorted(names):
        if n.lower() in blob or snake(n) in blob:
            continue
        if hasattr(mod, n):
            hits.add(n)
        elif hasattr(mod, snake(n)):
            hits.add(snake(n))
    if hits:
        findings.append((crate, modname, sorted(hits)))
        total += len(hits)

if missing_modules:
    print("REFUSING: could not audit every crate; a partial audit under-reports:", file=sys.stderr)
    for line in missing_modules:
        print(f"  {line}", file=sys.stderr)
    sys.exit(2)

print(f"SciPy-named public entry points with NO differential coverage: {total}\n")
for crate, modname, hits in sorted(findings, key=lambda r: -len(r[2])):
    print(f"{crate}  ->  {modname}   ({len(hits)})")
    print('  ' + ', '.join(hits))
    print()
