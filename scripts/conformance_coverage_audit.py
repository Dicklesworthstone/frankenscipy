"""How many public fsci names that ARE SciPy APIs have no conformance reference at all?

Two filters, both needed to avoid overclaiming:
  1. the name must have NO occurrence anywhere in the conformance corpus (731 diff_*.rs files,
     the python oracles, and the conformance lib) -- verified on both arms: six sampled negatives
     return 0 referencing files, and griddata/splu/cholesky/eigh return 1/2/6/70.
  2. the name must actually EXIST in the corresponding SciPy module, which drops internal helpers
     like `bench_trailing_syrk_prepare` and `correlate1d_perwindow_ref` that are `pub` for
     benchmarking rather than as API surface.

Filter 2 is what makes the count defensible: it is not "half the public surface is unverified",
it is "these specific SciPy-named entry points have never been compared to SciPy".
"""
import importlib
import pathlib
import re

ROOT = pathlib.Path('/data/projects/frankenscipy')
CONF = ROOT / 'crates' / 'fsci-conformance'

blob = []
for pat in ('tests/*.rs', 'python_oracle/*.py', 'src/*.rs'):
    for f in CONF.glob(pat):
        try:
            blob.append(f.read_text(errors='ignore').lower())
        except OSError:
            pass
blob = '\n'.join(blob)

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


total_uncovered_real = 0
findings = []
for crate, modname in sorted(CRATE_TO_MOD.items()):
    lib = ROOT / 'crates' / crate / 'src' / 'lib.rs'
    if not lib.exists():
        continue
    try:
        mod = importlib.import_module(modname)
    except Exception:
        continue
    names = sorted(set(PUB_RE.findall(lib.read_text(errors='ignore'))))
    hits = []
    for n in names:
        if n.lower() in blob or snake(n) in blob:
            continue
        # exists in SciPy under this name, or its snake_case form?
        if hasattr(mod, n) or hasattr(mod, snake(n)):
            hits.append(n if hasattr(mod, n) else snake(n))
    hits = sorted(set(hits))
    if hits:
        findings.append((crate, modname, hits))
        total_uncovered_real += len(hits)

print(f"SciPy-named public entry points with NO conformance reference: {total_uncovered_real}\n")
for crate, modname, hits in sorted(findings, key=lambda r: -len(r[2])):
    print(f"{crate}  ->  {modname}   ({len(hits)})")
    print('  ' + ', '.join(sorted(hits)))
    print()
