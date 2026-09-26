#!/usr/bin/env python3
"""SciPy public-name census: declared / real / compared, per module (frankenscipy-8dndw.2).

Observed defect classes. The 2026-09-15 census reported 1,300/1,300 SciPy names covered
(b5d699289). 25 of those names were no-op stand-ins (frankenscipy-8dndw.1), four were `pub type`
aliases to a DIFFERENT distribution (frankenscipy-szq1n.2), and the same commit widened the name
regex to `pub trait`. The old scan also swallowed a failing module import, which dropped that
module from the numerator AND the denominator, so a partial SciPy install printed a
plausible-looking percentage over fewer modules.

The three numbers:
  declared  a public Rust item's name matches the SciPy name, ignoring case and underscores.
            Public means a top-level item of a module reachable from lib.rs through `pub mod` or
            a glob `pub use child::*`, or the final name of a `pub use` in such a module.
            `src/bin/`, private (including test) modules and methods do not count. `matched_by` records which kind of match carried each name
            (exact, case_fold, reexport, type_alias, trait), so the reader can see what drove it.
  real      declared, minus the names docs/planning/census_dispositions.toml records as not
            applicable (`na`), `missing` or a `wrong_alias`, plus the names `adapted` under a
            different Rust name that exists. `na` names also leave the denominator.
  compared  real, and some diff_*.rs test's inline SciPy oracle names it while the same file's
            Rust code calls a matching item. A call-site heuristic and a lower bound: rows that
            reach a SciPy name through `getattr` are not seen.

The committed numbers live in docs/planning/symbol_census.json. Nightly CI (G9) runs `--check`,
which fails when a regeneration differs from the committed copy; `--write` regenerates it.

Exit status: 0 = ok; 1 = `--check` found the artifact stale, or a disposition no longer matches
the tree; 2 = refused (not the pinned scipy/numpy pair, a module failed to import, or the
self-test failed).

    /home/ubuntu/.local/bin/python3.13 scripts/symbol_census.py [--write | --check | --self-test]
"""
import copy
import importlib
import json
import pathlib
import re
import subprocess
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import oracle_attr_check as oracle  # noqa: E402  (the pin, and the oracle-snippet parser)

ROOT = oracle.ROOT
CRATES = ROOT / "crates"
DIFF_TESTS = CRATES / "fsci-conformance" / "tests"
DISPOSITIONS = ROOT / "docs" / "planning" / "census_dispositions.toml"
ARTIFACT = ROOT / "docs" / "planning" / "symbol_census.json"

# Every public SciPy module with an `__all__` (frankenscipy-9fbpm: the 16 top-level ones alone left
# 245 names out). Left out: scipy.cluster itself (its `__all__` is just vq and hierarchy, so it
# read 0/0), scipy.linalg.blas/lapack and scipy.stats.distributions (every name is already in the
# parent), and scipy.optimize.cython_optimize (Cython-level API with no `__all__`).
MODMAP = {
    "scipy.cluster.hierarchy": "fsci-cluster",
    "scipy.cluster.vq": "fsci-cluster", "scipy.constants": "fsci-constants",
    "scipy.datasets": "fsci-datasets", "scipy.differentiate": "fsci-opt",
    "scipy.fft": "fsci-fft", "scipy.fftpack": "fsci-fft",
    "scipy.integrate": "fsci-integrate", "scipy.interpolate": "fsci-interpolate",
    "scipy.io": "fsci-io", "scipy.io.arff": "fsci-io", "scipy.io.matlab": "fsci-io",
    "scipy.io.wavfile": "fsci-io", "scipy.linalg": "fsci-linalg",
    "scipy.linalg.interpolative": "fsci-linalg", "scipy.ndimage": "fsci-ndimage",
    "scipy.odr": "fsci-odr", "scipy.optimize": "fsci-opt", "scipy.optimize.elementwise": "fsci-opt",
    "scipy.signal": "fsci-signal", "scipy.signal.windows": "fsci-signal",
    "scipy.sparse": "fsci-sparse", "scipy.sparse.csgraph": "fsci-sparse",
    "scipy.sparse.linalg": "fsci-sparse", "scipy.spatial": "fsci-spatial",
    "scipy.spatial.distance": "fsci-spatial", "scipy.spatial.transform": "fsci-spatial",
    "scipy.special": "fsci-special", "scipy.stats": "fsci-stats",
    "scipy.stats.contingency": "fsci-stats", "scipy.stats.mstats": "fsci-stats",
    "scipy.stats.qmc": "fsci-stats", "scipy.stats.sampling": "fsci-stats",
}
# A name a module shares with its parent (or, for the legacy fftpack, with scipy.fft) is counted
# once, in the owner's row: signal.windows.get_window is signal.get_window, and the mstats names
# that stats also exports are the same functions over masked arrays.
SHARED_WITH = {"scipy.fftpack": "scipy.fft"}


def owner_of(module: str, modmap: dict[str, str]) -> str | None:
    owner = SHARED_WITH.get(module) or module.rsplit(".", 1)[0]
    return owner if owner != module and owner in modmap else None
DISPOSITION_KINDS = ("adapted", "na", "missing", "wrong_alias")
DISPOSITION_KEYS = {"module", "name", "disposition", "reason", "bead", "rust"}
MATCH_KINDS = ("exact", "case_fold", "reexport", "type_alias", "trait")
COUNTS = ("scipy", "na", "applicable", "declared", "real", "compared")

ITEM = re.compile(
    r"^pub (?:const |async |unsafe )*fn (\w+)"
    r"|^pub (struct|enum|trait|const) (\w+)"
    r"|^pub type (\w+)[^=;]*=\s*([^;]+);",
    re.M,
)
MOD_DECL = re.compile(r"^(pub )?mod (\w+)( \{|;)", re.M)
USE_STMT = re.compile(r"^pub use ([^;]+);", re.M)
CALL = re.compile(r"(\w+)\s*(?:\(|::|<|\{|!)|::(\w+)")


class CensusRefused(Exception):
    """No honest number can be produced (a module failed to import)."""


class CensusError(Exception):
    """A disposition or input no longer matches the tree."""


def norm(name: str) -> str:
    return name.lower().replace("_", "")


def scipy_symbols(module: str) -> set[str]:
    """Callable names in `module.__all__`. Every failure is fatal: skipping a module would drop
    it from both the numerator and the denominator."""
    try:
        mod = importlib.import_module(module)
    except Exception as err:  # ImportError, or anything the module raises while importing
        raise CensusRefused(f"cannot import {module}: {err!r}") from err
    names = getattr(mod, "__all__", None)
    if not names:
        raise CensusRefused(f"{module} has no __all__; the census counts declared public names")
    out = set()
    for name in names:
        try:
            obj = getattr(mod, name)
        except Exception as err:
            raise CensusRefused(f"{module}.{name} is in __all__ but not retrievable: {err!r}") from err
        if callable(obj):  # classes, functions, distribution instances; never modules
            out.add(name)
    return out


def crate_modules(src: pathlib.Path) -> list[tuple[str, bool]]:
    """(source text, public) for lib.rs and every module under it, in files or inline. A module
    is public when its parent is and it is declared `pub mod` or glob re-exported
    (`pub use child::*`). Private modules, test modules included, reach the surface only through
    named `pub use` leaves."""
    root = src / "lib.rs"
    if not root.exists():
        raise CensusError(f"{root} does not exist")
    out, stack = [], [(root.read_text(errors="replace"), src, True)]
    while stack:
        text, base, public = stack.pop()
        out.append((text, public))
        globbed = {m.group(1) for m in (re.fullmatch(r"\s*(?:self::)?(\w+)::\*\s*", body)
                                        for body in USE_STMT.findall(text)) if m}
        for m in MOD_DECL.finditer(text):
            name = m.group(2)
            child_public = public and (bool(m.group(1)) or name in globbed)
            if m.group(3) == ";":
                path = next((c for c in (base / f"{name}.rs", base / name / "mod.rs") if c.exists()), None)
                if path is None:
                    raise CensusError(f"{base}: `mod {name};` resolves to no file")
                stack.append((path.read_text(errors="replace"), base / name, child_public))
            else:
                end = text.find("\n}", m.end())
                if end < 0:
                    raise CensusError(f"{base}: inline `mod {name}` has no closing brace at column 0")
                stack.append((re.sub(r"(?m)^    ", "", text[m.end():end]), base / name, child_public))
    return out


def use_leaves(body: str) -> list[tuple[str, str]]:
    """(exported name, source name) for each final name of a `pub use` tree."""
    tokens = re.findall(r"\w+|[{},*]", body)
    out = []
    for i, tok in enumerate(tokens):
        if tok in ("{", "}", ",", "*", "as", "self", "super", "crate", "_"):
            continue
        prev = tokens[i - 1] if i else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if prev == "as":
            out.append((tok, tokens[i - 2]))
        elif nxt in ("", ",", "}"):
            out.append((tok, tok))
    return out


def crate_surface(src: pathlib.Path) -> dict[str, list[tuple[str, str, str]]]:
    """normalized name -> [(kind, Rust name, name it stands for)] over the public surface."""
    surface: dict[str, list[tuple[str, str, str]]] = {}
    for text, public in crate_modules(src):
        if not public:
            continue
        for m in ITEM.finditer(text):
            if m.group(1):
                kind, name, target = "fn", m.group(1), m.group(1)
            elif m.group(3):
                kind, name, target = m.group(2), m.group(3), m.group(3)
            else:
                rhs = re.sub(r"<.*", "", m.group(5)).strip()
                kind, name, target = "type", m.group(4), rhs.split("::")[-1]
            surface.setdefault(norm(name), []).append((kind, name, target))
        for m in USE_STMT.finditer(text):
            for name, source in use_leaves(m.group(1)):
                surface.setdefault(norm(name), []).append(("reexport", name, source))
    return surface


def classify(name: str, candidates: list[tuple[str, str, str]]) -> str | None:
    items = [c for c in candidates if c[0] in ("fn", "struct", "enum", "const")]
    if any(c[1] == name for c in items):
        return "exact"
    if items:
        return "case_fold"
    for kind, label in (("reexport", "reexport"), ("type", "type_alias"), ("trait", "trait")):
        if any(c[0] == kind for c in candidates):
            return label
    return None


def load_dispositions(path: pathlib.Path) -> dict[tuple[str, str], dict]:
    entries = tomllib.loads(path.read_text()).get("entry", [])
    out: dict[tuple[str, str], dict] = {}
    for entry in entries:
        key = (entry.get("module"), entry.get("name"))
        where = f"{path.name}: {key[0]}.{key[1]}"
        if set(entry) - DISPOSITION_KEYS:
            raise CensusError(f"{where}: unknown keys {sorted(set(entry) - DISPOSITION_KEYS)}")
        if key[0] not in MODMAP:
            raise CensusError(f"{where}: module is not censused")
        if entry.get("disposition") not in DISPOSITION_KINDS:
            raise CensusError(f"{where}: disposition must be one of {DISPOSITION_KINDS}")
        if not entry.get("reason"):
            raise CensusError(f"{where}: no reason")
        if (entry["disposition"] == "adapted") != bool(entry.get("rust")):
            raise CensusError(f"{where}: `rust` is required for adapted and only for adapted")
        if key in out:
            raise CensusError(f"{where}: duplicate entry")
        out[key] = entry
    return out


def resolve_adapted(path: str, crates: pathlib.Path) -> str:
    """`fsci_crate::Item[::Member...]` must name a public item of that crate (and each further
    segment must occur in its source). Returns the last segment, the name call sites use."""
    head, *rest = path.split("::")
    src = crates / head.replace("_", "-") / "src"
    if not rest or not src.exists():
        raise CensusError(f"adapted target {path}: expected fsci_crate::Item")
    surface = crate_surface(src)
    if not any(c[1] == rest[0] for cands in surface.values() for c in cands):
        raise CensusError(f"adapted target {path}: {rest[0]} is not a public item of {head}")
    text = "\n".join(t for t, _ in crate_modules(src))
    for seg in rest[1:]:
        if not re.search(rf"\b{re.escape(seg)}\b", text):
            raise CensusError(f"adapted target {path}: {seg} does not occur in {head}")
    return rest[-1]


def diff_uses(path: pathlib.Path, modmap: dict[str, str]) -> tuple[set[tuple[str, str]], set[str]]:
    """((scipy module, name) the file's inline oracles reference, identifiers its Rust calls)."""
    refs = set()
    for _, code in oracle.snippets(path):
        for _, target, attr in oracle.references(code, oracle.aliases_in(code)):
            parts = f"{target}.{attr}".split(".")
            for i in range(len(parts) - 1, 0, -1):
                if ".".join(parts[:i]) in modmap:
                    refs.add((".".join(parts[:i]), parts[i]))
                    break
    rust = oracle.RAW_STRING.sub(" ", path.read_text(errors="replace"))
    rust = re.sub(r"//[^\n]*", "", rust)
    calls = {a or b for a, b in CALL.findall(rust)}
    # An imported name is a used one: clippy -D warnings rejects unused imports in these tests,
    # and a unit struct is used as `Semicircular.pdf(x)`, which CALL does not see.
    for m in re.finditer(r"^\s*(?:pub )?use ([^;]+);", rust, re.M):
        calls.update(name for pair in use_leaves(m.group(1)) for name in pair)
    return refs, calls


def census(modmap, crates, dispositions, diff_paths, symbols=scipy_symbols) -> dict:
    uses = [diff_uses(p, modmap) for p in diff_paths]
    modules, total = {}, dict.fromkeys(COUNTS, 0) | {"matched_by": dict.fromkeys(MATCH_KINDS, 0)}
    exported = {module: symbols(module) for module in modmap}
    for module, crate in sorted(modmap.items()):
        owner = owner_of(module, modmap)
        names = exported[module] - (exported[owner] if owner else set())
        for m, n in dispositions:
            if m == module and n not in names:
                raise CensusError(f"disposition {m}.{n}: not a callable counted in {module}'s row")
        surface = crate_surface(crates / crate / "src")
        row = dict.fromkeys(COUNTS, 0) | {"crate": crate, "matched_by": dict.fromkeys(MATCH_KINDS, 0)}
        row |= {"not_real": {}, "real_not_compared": []}
        row["scipy"] = len(names)
        for name in sorted(names):
            entry = dispositions.get((module, name))
            kind = entry["disposition"] if entry else None
            candidates = surface.get(norm(name), [])
            how = classify(name, candidates)
            rust_names = {c[1] for c in candidates} | {c[2] for c in candidates}
            if how:
                row["declared"] += 1
                row["matched_by"][how] += 1
            where = f"{module}.{name}"
            if kind == "adapted":
                if how:
                    raise CensusError(f"{where}: dispositioned adapted but already declared ({how})")
                rust_names = {resolve_adapted(entry["rust"], crates)}
            elif kind == "wrong_alias" and not how:
                raise CensusError(f"{where}: dispositioned wrong_alias but nothing matches; record it as missing")
            elif kind == "missing" and how:
                raise CensusError(f"{where}: dispositioned missing but a Rust item matches ({how})")
            if kind == "na":
                row["na"] += 1
            is_real = kind == "adapted" or (bool(how) and kind is None)
            if is_real:
                row["real"] += 1
                if any((module, name) in refs and calls & rust_names for refs, calls in uses):
                    row["compared"] += 1
                else:
                    row["real_not_compared"].append(name)
            else:
                bead = f" ({entry['bead']})" if entry and entry.get("bead") else ""
                row["not_real"][name] = f"{kind}: {entry['reason']}{bead}" if kind else "unmatched"
        row["applicable"] = row["scipy"] - row["na"]
        modules[module] = row
        for key in COUNTS:
            total[key] += row[key]
        for key in MATCH_KINDS:
            total["matched_by"][key] += row["matched_by"][key]
    return {"total": total, "modules": modules}


def stale_fields(fresh: dict, path: pathlib.Path) -> list[str]:
    if not path.exists():
        return [f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} does not exist"]
    committed = json.loads(path.read_text())
    out = []
    for key in sorted(set(committed) | set(fresh)):
        a, b = committed.get(key), fresh.get(key)
        if a == b:
            continue
        if key == "total" and isinstance(a, dict) and isinstance(b, dict):
            out += [f"total.{f}: committed {a.get(f)} != regenerated {b.get(f)}"
                    for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
            continue
        if key != "modules" or not (isinstance(a, dict) and isinstance(b, dict)):
            out.append(f"{key}: committed {a} != regenerated {b}")
            continue
        for mod in sorted(set(a) | set(b)):
            ra, rb = a.get(mod, {}), b.get(mod, {})
            for field in sorted(set(ra) | set(rb)):
                fa, fb = ra.get(field), rb.get(field)
                if fa == fb:
                    continue
                if isinstance(fa, (list, dict)) and isinstance(fb, (list, dict)):
                    changed = sorted(k for k in set(fa) & set(fb) if isinstance(fa, dict) and fa[k] != fb[k])
                    out.append(f"{mod}.{field}: +{sorted(set(fb) - set(fa))} -{sorted(set(fa) - set(fb))}"
                               + (f" changed {changed}" if changed else ""))
                else:
                    out.append(f"{mod}.{field}: committed {fa} != regenerated {fb}")
    return out


def pin_refusal() -> str | None:
    try:
        import numpy
        import scipy
    except ImportError as err:
        return f"{sys.executable} cannot import scipy/numpy ({err})"
    if scipy.__version__ != oracle.PINNED_SCIPY or numpy.__version__ != oracle.PINNED_NUMPY:
        return (f"{sys.executable} has scipy {scipy.__version__} / numpy {numpy.__version__}; "
                f"the pin is {oracle.PINNED_SCIPY} / {oracle.PINNED_NUMPY}")
    return None


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def self_test() -> bool:
    """Both arms of every claim the census makes, on a synthetic crate and SciPy module."""
    tmp = ROOT / "target" / "symbol_census_selftest"
    lib = ("pub fn solve() {}\npub trait Onlytrait {}\npub struct Moyal;\npub type Landau = Moyal;\n"
           "impl Moyal {\n    pub fn indented() {}\n}\nmod private;\npub use private::reexported;\n"
           "mod aliases {\n    pub type inline_alias = super::Moyal;\n}\npub use aliases::*;\n"
           "#[cfg(test)]\nmod tests {\n    pub fn hidden_inline() {}\n}\n")
    for crate, extra in (("fsci-fake", ""), ("fsci-noop", "pub fn get_blas_funcs() {}\n")):
        src = tmp / "crates" / crate / "src"
        (src / "bin").mkdir(parents=True, exist_ok=True)
        (src / "lib.rs").write_text(lib + extra)
        (src / "private.rs").write_text("pub fn hidden() {}\npub fn reexported() {}\n")
        (src / "bin" / "tool.rs").write_text("pub fn binonly() {}\n")
    diff = tmp / "diff_fake.rs"
    diff.write_text(
        "use fsci_fake::{Onlytrait, other};\n"
        'fn t() {\n    let script = r#"\nfrom scipy import fake\n'
        "print(fake.solve(1), fake.moyal(2), fake.landau(3), fake.reexported(4), fake.onlytrait)\n\"#;\n"
        "    let x = fsci_fake::solve();\n    let y = fsci_fake::Landau::new();\n"
        "    // fsci_fake::reexported() in a comment is not a call\n}\n")
    names = {"solve", "onlytrait", "moyal", "landau", "indented", "hidden", "reexported",
             "binonly", "get_blas_funcs", "absent", "inline_alias", "hidden_inline"}
    modmap = {"scipy.fake": "fsci-fake"}
    crates = tmp / "crates"

    def entry(name, kind, **extra):
        return {("scipy.fake", name): {"disposition": kind, "reason": "self-test"} | extra}

    disps = entry("get_blas_funcs", "na") | entry("landau", "wrong_alias")

    def run(dispositions, crate="fsci-fake"):
        return census({"scipy.fake": crate}, crates, dispositions, [diff], lambda _m: names)["total"]

    t = run(disps)
    noop = run(disps, "fsci-noop")
    bare = run({})
    adapted = run(disps | entry("absent", "adapted", rust="fsci_fake::Moyal"))
    uncompared = census(modmap, crates, disps, [diff], lambda _m: names)["modules"]["scipy.fake"][
        "real_not_compared"]
    nested = census({"scipy.fake": "fsci-fake", "scipy.fake.sub": "fsci-fake"}, crates, {}, [],
                    lambda m: names if m == "scipy.fake" else {"solve", "subonly"})
    checks = {
        # inline_alias is public through the glob; hidden (private file module), hidden_inline
        # (private inline module), indented (method), binonly (src/bin) and absent never match
        "declared counts public top-level names only": t["declared"] == 6,
        "each match kind has its own sub-count": t["matched_by"] == {
            "exact": 1, "case_fold": 1, "reexport": 1, "type_alias": 2, "trait": 1},
        "wrong_alias is not real": t["real"] == 5 and bare["real"] == 6,
        "na leaves the denominator": t["applicable"] == 11 and bare["applicable"] == 12,
        "a no-op name listed na leaves real unchanged": noop["declared"] == 7 and noop["real"] == t["real"],
        "adapted counts when its Rust item exists": adapted["real"] == 6,
        # the submodule re-exports solve; only subonly is its own
        "a submodule name shared with its parent is counted once": nested["modules"][
            "scipy.fake.sub"]["scipy"] == 1 and nested["total"]["scipy"] == len(names) + 1,
        # solve is called and Onlytrait imported; Landau::new counts only once landau is real;
        # moyal is named by the oracle but never used; reexported is "called" only in a comment
        "a Rust call or import next to an oracle reference is compared": t["compared"] == 2
        and bare["compared"] == 3,
        "an oracle reference alone, or a call in a comment, is not": "moyal" in uncompared
        and "reexported" in uncompared,
        "adapted to a missing item fails": raises(
            lambda: run(disps | entry("absent", "adapted", rust="fsci_fake::Nope")), CensusError),
        "missing on a declared name fails": raises(lambda: run(entry("solve", "missing")), CensusError),
        "wrong_alias on an absent name fails": raises(lambda: run(entry("absent", "wrong_alias")), CensusError),
        "a disposition for a non-SciPy name fails": raises(lambda: run(entry("typo", "na")), CensusError),
    }

    saved = sys.modules.get("scipy.odr")
    sys.modules["scipy.odr"] = None  # the import system raises ImportError for a None entry
    try:
        checks["an unimportable module fails the run"] = raises(
            lambda: census({"scipy.odr": "fsci-odr"}, CRATES, {}, []), CensusRefused)
    finally:
        sys.modules.pop("scipy.odr")
        if saved is not None:
            sys.modules["scipy.odr"] = saved
    checks["the same module imports when not patched"] = bool(scipy_symbols("scipy.odr"))
    saved = sys.modules["scipy"]
    sys.modules["scipy"] = None
    try:
        checks["missing SciPy is refused"] = pin_refusal() is not None
    finally:
        sys.modules["scipy"] = saved
    checks["the pinned SciPy is accepted"] = pin_refusal() is None

    fresh = census(modmap, crates, disps, [diff], lambda _m: names)
    artifact = tmp / "artifact.json"
    artifact.write_text(json.dumps(fresh))
    checks["an identical artifact is not stale"] = stale_fields(fresh, artifact) == []
    moved = copy.deepcopy(fresh)
    moved["modules"]["scipy.fake"]["real"] += 1
    artifact.write_text(json.dumps(moved))
    checks["a moved count is stale"] = bool(stale_fields(fresh, artifact))
    checks["a missing artifact is stale"] = bool(stale_fields(fresh, tmp / "absent.json"))

    print(f"self-test: synthetic totals {t}; with a no-op na name {noop['declared']}/{noop['real']} "
          f"declared/real; without dispositions {bare['real']} real")
    for label, ok in checks.items():
        if not ok:
            print(f"self-test FAILED: {label}")
    return all(checks.values())


def print_report(report: dict) -> None:
    def pct(n, d):
        return f"{100 * n / d:5.1f}" if d else "  n/a"

    print(f"{'module':26s} {'crate':17s} {'scipy':>5s} {'na':>3s} {'appl':>5s} {'decl':>5s} "
          f"{'real':>5s} {'cmp':>5s} {'real%':>6s} {'cmp%':>6s}")
    rows = list(report["modules"].items()) + [("TOTAL", report["total"] | {"crate": ""})]
    for module, r in rows:
        print(f"{module:26s} {r['crate']:17s} {r['scipy']:5d} {r['na']:3d} {r['applicable']:5d} "
              f"{r['declared']:5d} {r['real']:5d} {r['compared']:5d} {pct(r['real'], r['applicable']):>6s} "
              f"{pct(r['compared'], r['applicable']):>6s}")
    print("declared, by match kind: " + ", ".join(f"{k} {v}" for k, v in report["total"]["matched_by"].items()))
    print("not real:")
    for module, r in report["modules"].items():
        for name, why in r["not_real"].items():
            print(f"  {module}.{name}: {why}")


def main() -> int:
    refusal = pin_refusal()
    if refusal:
        print(f"REFUSING: {refusal}", file=sys.stderr)
        return 2
    try:
        passed = self_test()
    except Exception as err:  # a crashing self-test has not shown the arms apart either
        print(f"self-test raised {err!r}")
        passed = False
    if not passed:
        print("REFUSING: self-test failed; the census cannot tell its arms apart", file=sys.stderr)
        return 2
    if "--self-test" in sys.argv:
        return 0
    import numpy
    import scipy

    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "unknown"
    print(f"{sys.executable}: scipy {scipy.__version__} / numpy {numpy.__version__}, tree {sha}")
    try:
        dispositions = load_dispositions(DISPOSITIONS)
        report = census(MODMAP, CRATES, dispositions, sorted(DIFF_TESTS.glob("diff_*.rs")))
    except CensusRefused as err:
        print(f"REFUSING: {err}", file=sys.stderr)
        return 2
    except CensusError as err:
        print(f"DISPOSITION ERROR: {err}", file=sys.stderr)
        return 1
    report = {"scipy": scipy.__version__, "numpy": numpy.__version__} | report
    print_report(report)
    if "--write" in sys.argv:
        ARTIFACT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        print(f"wrote {ARTIFACT.relative_to(ROOT)}")
    if "--check" in sys.argv:
        stale = stale_fields(report, ARTIFACT)
        for line in stale:
            print(f"STALE {line}")
        if stale:
            print(f"{ARTIFACT.relative_to(ROOT)} is stale: regenerate it with --write", file=sys.stderr)
            return 1
        print(f"{ARTIFACT.relative_to(ROOT)} is current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
