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
`grep -rli <name> crates/fsci-conformance/{tests,python_oracle,src} crates/*/src/bin`.
"""
import importlib
import pathlib
import re

ROOT = pathlib.Path('/data/projects/frankenscipy')

corpus_files = []
conf = ROOT / 'crates' / 'fsci-conformance'
for pat in ('tests/*.rs', 'python_oracle/*.py', 'src/*.rs'):
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
for crate, modname in sorted(CRATE_TO_MOD.items()):
    lib = ROOT / 'crates' / crate / 'src' / 'lib.rs'
    if not lib.exists():
        continue
    try:
        mod = importlib.import_module(modname)
    except Exception:
        continue
    hits = set()
    for n in sorted(set(PUB_RE.findall(lib.read_text(errors='ignore')))):
        if n.lower() in blob or snake(n) in blob:
            continue
        if hasattr(mod, n):
            hits.add(n)
        elif hasattr(mod, snake(n)):
            hits.add(snake(n))
    if hits:
        findings.append((crate, modname, sorted(hits)))
        total += len(hits)

print(f"SciPy-named public entry points with NO differential coverage: {total}\n")
for crate, modname, hits in sorted(findings, key=lambda r: -len(r[2])):
    print(f"{crate}  ->  {modname}   ({len(hits)})")
    print('  ' + ', '.join(hits))
    print()
