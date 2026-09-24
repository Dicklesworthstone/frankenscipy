#!/usr/bin/env python3
"""A literal `converged: true` / `success: true` must say which check earned it.

Observed defect class (frankenscipy-szq1n.7): a sweep of the 102 such literals in non-test code
found about 30 where the flag was not tied to its criterion: nquad reported converged with error
0.0 unconditionally, GMRES/LGMRES reported convergence on a lucky breakdown of a singular matrix,
SLSQP and trust-constr reported success at an infeasible point, basinhopping was always
successful, ODR's tests depended on the scale of the data, and so on. Callers and the CASP
fallbacks trust these flags.

Rule: every such literal in library code (not `#[cfg(test)]` modules, not `src/bin`) carries a
`// status: <criterion>` comment on the same line or on one of the two lines above it, naming the
test that established the flag (e.g. `// status: |r|/|b| < tol`). A literal `false` needs nothing.

Exit status: 0 clean, 1 unannotated literals found, 2 self-test failed.
    python3 scripts/status_literal_guard.py            # scan crates/*/src
    python3 scripts/status_literal_guard.py --self-test
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LITERAL = re.compile(r"\b(converged|success)\s*(?::|=)\s*true\b")
ANNOTATION = re.compile(r"//\s*status:\s*\S")
CFG_TEST = re.compile(r"#\[cfg\(test\)\]")


def test_line_mask(lines: list[str]) -> list[bool]:
    """True for lines inside a `#[cfg(test)]` item (tracked by brace depth from the item's
    opening brace). Brace counting ignores `//` comments; that is enough for this codebase."""
    mask = [False] * len(lines)
    i = 0
    while i < len(lines):
        if CFG_TEST.search(lines[i]):
            depth = 0
            opened = False
            j = i
            while j < len(lines):
                code = lines[j].split("//", 1)[0]
                depth += code.count("{") - code.count("}")
                if "{" in code:
                    opened = True
                mask[j] = True
                if opened and depth <= 0:
                    break
                if not opened and code.rstrip().endswith(";"):
                    break  # `#[cfg(test)] use ...;` or a one-line item
                j += 1
            i = j + 1
            continue
        i += 1
    return mask


def scan_text(text: str) -> list[int]:
    """1-based line numbers of unannotated literals outside test code."""
    lines = text.split("\n")
    in_test = test_line_mask(lines)
    bad = []
    for n, line in enumerate(lines):
        if in_test[n]:
            continue
        code = line.split("//", 1)[0]
        if not LITERAL.search(code):
            continue
        window = lines[max(0, n - 2) : n + 1]
        if any(ANNOTATION.search(w) for w in window):
            continue
        bad.append(n + 1)
    return bad


def self_test() -> bool:
    unannotated = "fn f() -> R {\n    R { converged: true }\n}\n"
    annotated = (
        "fn f() -> R {\n    // status: residual < tol\n    R { converged: true }\n}\n"
        "fn g() -> R { R { success: true } } // status: exact zero\n"
        "#[cfg(test)]\nmod tests {\n    fn h() { let _ = R { converged: true }; }\n}\n"
        "fn k() -> R { R { converged: false } }\n"
    )
    miss = scan_text(unannotated)
    hit = scan_text(annotated)
    ok = miss == [2] and hit == []
    print(f"self-test: unannotated -> {miss} (want [2]); annotated/test/false -> {hit} (want [])")
    return ok


def main() -> int:
    if not self_test():
        print("self-test FAILED: the guard cannot tell an annotated literal from a bare one")
        return 2
    if "--self-test" in sys.argv:
        return 0
    findings = []
    files = sorted(ROOT.glob("crates/*/src/**/*.rs"))
    for path in files:
        rel = path.relative_to(ROOT)
        if "bin" in rel.parts:
            continue
        for line in scan_text(path.read_text(errors="replace")):
            findings.append(f"{rel}:{line}")
    print(f"{len(files)} files scanned")
    for f in findings:
        print(f"{f} unannotated converged/success literal (add `// status: <criterion>`)")
    print(f"{len(findings)} unannotated literal(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
