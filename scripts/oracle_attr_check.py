#!/usr/bin/env python3
"""Every `scipy.*` attribute an oracle names must exist in the PINNED SciPy.

Observed defect class (frankenscipy-olv0j.2, checker bead frankenscipy-olv0j.3): inline oracle
scripts called `signal.gauspuls`, `special.stdtrc` / `btdtr` / `btdtrc` / `btdtri`,
`special.lpmn`, `special.lpn`, `integrate.romberg` and `distance.wminkowski`, none of which
exists in SciPy 1.17.1. Each raised AttributeError, the harness mapped that to "no SciPy value",
and the case was skipped, so whole columns compared nothing while their tests passed.

This walks the Python embedded in `crates/fsci-conformance/tests/diff_*.rs` (raw string literals)
and `crates/fsci-conformance/python_oracle/*.py`, collects the SciPy module aliases each snippet
imports and every `alias.attr` it uses, and resolves them all under the pinned interpreter.

Exit status: 0 = every attribute resolves; 1 = at least one MISSING attribute that is not
allowlisted; 2 = refused (not the pinned scipy/numpy pair, or the self-test failed).

Run with the pinned incumbent:
    /home/ubuntu/.local/bin/python3.13 scripts/oracle_attr_check.py
    /home/ubuntu/.local/bin/python3.13 scripts/oracle_attr_check.py --self-test
"""
import importlib
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFORMANCE = ROOT / "crates" / "fsci-conformance"
PINNED_SCIPY = "1.17.1"
PINNED_NUMPY = "2.4.3"

# (file name, "module.attr") -> reason. Only for references that are MEANT to be absent, e.g. a
# guarded `hasattr` probe. Every entry names its bead.
ALLOWLIST: dict[tuple[str, str], str] = {}

RAW_STRING = re.compile(r'r(#+)"(.*?)"\1', re.S)
IMPORT_AS = re.compile(r"^\s*import\s+(scipy(?:\.\w+)*)(?:\s+as\s+(\w+))?\s*$", re.M)
FROM_IMPORT = re.compile(r"^\s*from\s+(scipy(?:\.\w+)*)\s+import\s+\(?([^\n)]+)\)?", re.M)
IDENT = re.compile(r"[A-Za-z_]\w*")


def aliases_in(code: str) -> dict[str, str]:
    """alias -> fully qualified scipy module (or scipy object) it names."""
    out: dict[str, str] = {}
    for module, alias in IMPORT_AS.findall(code):
        if alias:
            out[alias] = module
        else:
            out["scipy"] = "scipy"  # `import scipy.special` binds the name `scipy`
    for module, names in FROM_IMPORT.findall(code):
        for part in names.split(","):
            part = part.strip()
            if not part or part == "*":
                continue
            bits = part.split()
            name, alias = bits[0], (bits[2] if len(bits) == 3 and bits[1] == "as" else bits[0])
            if IDENT.fullmatch(name) and IDENT.fullmatch(alias):
                out[alias] = f"{module}.{name}"
    return out


def references(code: str, aliases: dict[str, str]) -> list[tuple[int, str, str]]:
    """(line within snippet, qualified module, attribute) for every alias.attr use, plus the
    names pulled in by `from scipy.x import name` (which must exist themselves)."""
    refs = []
    for alias, target in aliases.items():
        if alias == "scipy":
            pattern = re.compile(r"(?<![\w.])scipy((?:\.\w+)+)")
            for m in pattern.finditer(code):
                chain = m.group(1).lstrip(".").split(".")
                line = code.count("\n", 0, m.start()) + 1
                refs.append((line, "scipy", ".".join(chain)))
            continue
        pattern = re.compile(rf"(?<![\w.]){re.escape(alias)}\.(\w+)")
        for m in pattern.finditer(code):
            line = code.count("\n", 0, m.start()) + 1
            refs.append((line, target, m.group(1)))
    for module, names in FROM_IMPORT.findall(code):
        for part in names.split(","):
            name = part.strip().split(" ")[0] if part.strip() else ""
            if IDENT.fullmatch(name):
                refs.append((0, module, name))
    return refs


def resolve(target: str, attr_chain: str) -> bool:
    """Does `target.attr_chain` exist? Walks submodules the way a Python script would reach them
    (`scipy.sparse.linalg`, `scipy.spatial.distance`)."""
    parts = target.split(".")
    try:
        obj = importlib.import_module(parts[0])
    except ImportError:
        return False
    path = parts[0]
    for name in parts[1:] + attr_chain.split("."):
        path = f"{path}.{name}"
        if hasattr(obj, name):
            obj = getattr(obj, name)
            continue
        try:
            obj = importlib.import_module(path)
        except ImportError:
            return False
        # Stop descending once we reach a non-module object's attributes chain end.
    return True


def strip_comments(code: str) -> str:
    """Blank out `#` comments (outside string literals) line by line, keeping line numbers: a
    comment that NAMES a removed function (to explain its replacement) is not a use of it."""
    out = []
    for line in code.split("\n"):
        quote = None
        cut = len(line)
        i = 0
        while i < len(line):
            c = line[i]
            if quote:
                if c == "\\":
                    i += 2
                    continue
                if c == quote:
                    quote = None
            elif c in "'\"":
                quote = c
            elif c == "#":
                cut = i
                break
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


def snippets(path: pathlib.Path) -> list[tuple[int, str]]:
    """(first line number, python text without comments) blocks from a diff test or an oracle
    script."""
    text = path.read_text(errors="replace")
    if path.suffix == ".py":
        return [(1, strip_comments(text))]
    out = []
    for m in RAW_STRING.finditer(text):
        body = m.group(2)
        if "import" in body and "scipy" in body:
            start = text.count("\n", 0, m.start(2)) + 1
            out.append((start, strip_comments(body)))
    return out


def check(paths: list[pathlib.Path]) -> list[str]:
    misses = []
    for path in paths:
        for first_line, code in snippets(path):
            aliases = aliases_in(code)
            seen = set()
            for line, target, attr in references(code, aliases):
                key = (target, attr)
                if key in seen:
                    continue
                seen.add(key)
                if resolve(target, attr):
                    continue
                qualified = f"{target}.{attr}"
                short = f"{target.split('.')[-1]}.{attr}"
                if (path.name, short) in ALLOWLIST or (path.name, qualified) in ALLOWLIST:
                    continue
                where = first_line + line - 1 if line else first_line
                misses.append(f"{path.relative_to(ROOT)}:{where} {qualified} MISSING")
    return misses


def self_test() -> bool:
    """Both arms: a removed name must be reported, the current one must resolve."""
    must_miss = "from scipy import signal\nsignal.gauspuls(0.1)\nimport scipy.special as sp\nsp.lpmn(1, 1, 0.5)\n"
    must_hit = (
        "from scipy import signal\nsignal.gausspulse(0.1)  # not signal.gauspuls\n"
        "from scipy.spatial import distance\ndistance.minkowski([0], [1])\n"
        "# sp.lpmn was removed; this comment must not count\n"
        "s = 'a # inside a string'\nimport scipy\nscipy.sparse.linalg.cg\n"
    )
    tmp = ROOT / "target" / "oracle_attr_check_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    miss_file, hit_file = tmp / "must_miss.py", tmp / "must_hit.py"
    miss_file.write_text(must_miss)
    hit_file.write_text(must_hit)
    missed = check([miss_file])
    hit = check([hit_file])
    ok = (
        any("scipy.signal.gauspuls" in m for m in missed)
        and any("scipy.special.lpmn" in m for m in missed)
        and not hit
    )
    print(f"self-test: must-miss reported {len(missed)} ({missed}); must-hit reported {len(hit)}")
    return ok


def main() -> int:
    try:
        import numpy
        import scipy
    except ImportError as err:
        print(f"REFUSING: {sys.executable} cannot import scipy/numpy ({err})", file=sys.stderr)
        return 2
    if scipy.__version__ != PINNED_SCIPY or numpy.__version__ != PINNED_NUMPY:
        print(
            f"REFUSING: {sys.executable} has scipy {scipy.__version__} / numpy "
            f"{numpy.__version__}; the pin is {PINNED_SCIPY} / {PINNED_NUMPY}",
            file=sys.stderr,
        )
        return 2
    if not self_test():
        print("REFUSING: self-test failed; the checker cannot tell a miss from a hit", file=sys.stderr)
        return 2
    if "--self-test" in sys.argv:
        return 0
    paths = sorted((CONFORMANCE / "tests").glob("diff_*.rs")) + sorted(
        (CONFORMANCE / "python_oracle").glob("*.py")
    )
    misses = check(paths)
    print(f"{len(paths)} files scanned under scipy {scipy.__version__} / numpy {numpy.__version__}")
    for miss in misses:
        print(miss)
    print(f"{len(misses)} missing attribute reference(s)")
    return 1 if misses else 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.exit(main())
