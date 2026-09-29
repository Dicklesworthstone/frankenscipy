"""Which public fsci entry points with SciPy-shaped names have no differential coverage?

Motivated by frankenscipy-icozs: `RbfInterpolator` implements SciPy's non-default `degree=-1`
variant under SciPy's default name, verified numerically, and it survived because nothing anywhere
compared it to SciPy. This finds what else is in that position.

CORPUS -- and getting this wrong is the easy mistake. Differential tests do NOT all live under
`crates/fsci-conformance/tests/`. They are split:

    crates/fsci-conformance/tests/     diff_*.rs
    crates/<crate>/src/bin/            more diff_*.rs, per-crate

A first pass over only the conformance crate reported 201 uncovered entry points. That was
INFLATED: it counted `cholesky_banded`, `expm_frechet`, `convolve1d`, `fourier_gaussian` and
others that have a `diff_*.rs` in their own crate's `src/bin/`. This scans both, plus the
conformance lib.

A public `fn`/`struct`/`enum` counts UNCOVERED only if BOTH hold:
  1. neither its name nor its snake_case form occurs in the Rust code of that corpus, and
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
  * counts a name as referenced only from RUST code: raw-string literals (the inline Python
    oracles) and `//` comments are removed first, and the python_oracle scripts are not read. A
    name that appears only on the SciPy side of an oracle is exactly the uncompared case this
    audit exists to find;
  * scans every `src/**/*.rs` of a crate (not only lib.rs), skipping `src/bin/`, and prints the
    defining file:line of every unreferenced name;
  * self-tests both arms before every run (`--self-test` stops after it).
Run it with the pinned incumbent: /home/ubuntu/.local/bin/python3.13 scripts/conformance_coverage_audit.py
"""
import importlib
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PINNED_SCIPY = os.environ.get('FSCI_AUDIT_SCIPY', '1.17.1')
PINNED_NUMPY = os.environ.get('FSCI_AUDIT_NUMPY', '2.4.3')

PUB_RE = re.compile(r'^pub (?:fn|struct|enum) ([A-Za-z_][A-Za-z0-9_]*)', re.MULTILINE)
RAW_STRING = re.compile(r'r(#+)".*?"\1', re.S)
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


class Refused(Exception):
    """The audit cannot produce an honest number."""


def snake(n):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', n).lower()


def rust_code(text):
    """Lower-cased Rust code with raw strings (the inline oracles) and `//` comments removed."""
    return re.sub(r'//[^\n]*', '', RAW_STRING.sub(' ', text)).lower()


def corpus_blob(files):
    return '\n'.join(rust_code(f.read_text(errors='ignore')) for f in files)


def unreferenced(sources, module, blob):
    """{scipy name: 'file:line'} for public items of `sources` that exist in `module` but whose
    name is referenced nowhere in `blob`."""
    out = {}
    for path in sources:
        text = path.read_text(errors='ignore')
        for m in PUB_RE.finditer(text):
            n = m.group(1)
            if n.lower() in blob or snake(n) in blob:
                continue
            scipy_name = n if hasattr(module, n) else snake(n) if hasattr(module, snake(n)) else None
            if scipy_name and scipy_name not in out:
                out[scipy_name] = f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}:{text.count(chr(10), 0, m.start()) + 1}"
    return out


def audit(crate_to_mod, root):
    findings, refusals = [], []
    for crate, modname in sorted(crate_to_mod.items()):
        src = root / 'crates' / crate / 'src'
        sources = [p for p in src.rglob('*.rs') if 'bin' not in p.relative_to(src).parts]
        if not sources:
            refusals.append(f"{crate}: no sources under {src}")
            continue
        try:
            mod = importlib.import_module(modname)
        except Exception as err:
            refusals.append(f"{modname}: {err}")
            continue
        findings.append((crate, modname, sources, mod))
    if refusals:
        raise Refused("could not audit every crate; a partial audit under-reports:\n  "
                      + "\n  ".join(refusals))
    return findings


def self_test():
    """Must-hit and must-miss arms on a synthetic crate and corpus."""
    tmp = ROOT / 'target' / 'coverage_audit_selftest'
    src = tmp / 'crates' / 'fsci-fake' / 'src'
    src.mkdir(parents=True, exist_ok=True)
    (src / 'lib.rs').write_text('pub fn solve() {}\npub fn det() {}\npub fn inv() {}\n')
    diff = tmp / 'diff_fake.rs'
    diff.write_text('fn t() {\n    let x = fsci_fake::solve();\n'
                    '    let script = r#"\nimport scipy.linalg as la\nla.det([[1.0]])\n"#;\n'
                    '    // fsci_fake::inv() is only mentioned in a comment\n}\n')
    import scipy.linalg
    got = unreferenced([src / 'lib.rs'], scipy.linalg, corpus_blob([diff]))
    checks = {
        'a Rust call references a name': 'solve' not in got,
        'a name only in the Python oracle is unreferenced': 'det' in got,
        'a name only in a comment is unreferenced': 'inv' in got,
        'the defining line is reported': got.get('det', '').endswith('lib.rs:2'),
    }
    saved = sys.modules.get('scipy.odr')
    sys.modules['scipy.odr'] = None  # the import system raises ImportError for a None entry
    try:
        audit({'fsci-odr': 'scipy.odr'}, ROOT)
        checks['an unimportable module refuses the audit'] = False
    except Refused:
        checks['an unimportable module refuses the audit'] = True
    finally:
        sys.modules.pop('scipy.odr')
        if saved is not None:
            sys.modules['scipy.odr'] = saved
    for label, ok in checks.items():
        if not ok:
            print(f"self-test FAILED: {label}")
    print(f"self-test: unreferenced on the synthetic corpus = {got}")
    return all(checks.values())


def main():
    try:
        import numpy
        import scipy
    except ImportError as err:
        sys.exit(f"REFUSING: {sys.executable} cannot import scipy/numpy ({err}); a run without the "
                 f"oracle would report every entry point as covered.")
    if scipy.__version__ != PINNED_SCIPY or numpy.__version__ != PINNED_NUMPY:
        sys.exit(f"REFUSING: {sys.executable} has scipy {scipy.__version__} / numpy {numpy.__version__}, "
                 f"pinned pair is {PINNED_SCIPY} / {PINNED_NUMPY}")
    if not self_test():
        print("REFUSING: self-test failed", file=sys.stderr)
        return 2
    if '--self-test' in sys.argv:
        return 0
    print(f"interpreter: {sys.executable}  scipy {scipy.__version__}  numpy {numpy.__version__}")

    conf = ROOT / 'crates' / 'fsci-conformance'
    corpus_files = [*conf.glob('tests/*.rs'), *conf.glob('src/*.rs'),
                    *ROOT.glob('crates/*/src/bin/diff_*.rs')]
    blob = corpus_blob(corpus_files)
    n_diff = len(list(conf.glob('tests/diff_*.rs'))) + len(list(ROOT.glob('crates/*/src/bin/diff_*.rs')))
    print(f"corpus: {len(corpus_files)} files, {len(blob):,} chars of Rust code, {n_diff} diff_*.rs total")
    print()
    try:
        crates = audit(CRATE_TO_MOD, ROOT)
    except Refused as err:
        print(f"REFUSING: {err}", file=sys.stderr)
        return 2
    rows = [(crate, modname, unreferenced(sources, mod, blob)) for crate, modname, sources, mod in crates]
    rows = [r for r in rows if r[2]]
    total = sum(len(r[2]) for r in rows)
    print(f"SciPy-named public entry points with NO differential coverage: {total}\n")
    for crate, modname, hits in sorted(rows, key=lambda r: -len(r[2])):
        print(f"{crate}  ->  {modname}   ({len(hits)})")
        for name, where in sorted(hits.items()):
            print(f"  {name:32s} {where}")
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
