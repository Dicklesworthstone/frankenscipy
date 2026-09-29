#!/usr/bin/env python3
"""Every differential test ends in a compared-case ledger (frankenscipy-olv0j.1).

Observed defect class: nearly every `crates/fsci-conformance/tests/diff_*.rs` asserted
`diffs.iter().all(|d| d.pass)`, which is true over an empty iterator, and reached it through
silent skips (Python `except` -> None, Rust `Err(_) => continue`, `is_finite()` guards). A test
whose oracle raised on every case, or whose Rust call always failed, compared nothing and passed;
four such columns shipped under SciPy 1.17.1 (frankenscipy-olv0j.2). `fsci_conformance::
CompareLedger::finish` fails a test that compared nothing, or where fsci failed against a SciPy
value.

A diff file is ledgered when its Rust code (comments stripped) constructs `CompareLedger::new(`
and calls `.finish(`. Files not yet migrated are listed in scripts/unledgered_diff_tests.txt,
and that list only shrinks: an unlisted unledgered file fails (new tests start ledgered), and a
listed file that is ledgered or no longer exists fails (take it off the list). There is no write
mode; entries are removed by hand as files migrate, and the list and this check's exemption go
when it is empty.

Exit status: 0 clean, 1 violations, 2 self-test failed.
    python3 scripts/unledgered_diff_tests.py
    python3 scripts/unledgered_diff_tests.py --self-test
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "crates" / "fsci-conformance" / "tests"
REGISTER = pathlib.Path(__file__).resolve().parent / "unledgered_diff_tests.txt"


def is_ledgered(text: str) -> bool:
    code = re.sub(r"//[^\n]*", "", text)
    return "CompareLedger::new(" in code and ".finish(" in code


def read_register(path: pathlib.Path) -> list[str]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def violations(tests: pathlib.Path, register: list[str]) -> list[str]:
    out = []
    listed = set(register)
    if len(listed) != len(register):
        out.append("register lists a file twice")
    files = {p.name: p for p in sorted(tests.glob("diff_*.rs"))}
    for name, path in files.items():
        ledgered = is_ledgered(path.read_text(errors="replace"))
        if not ledgered and name not in listed:
            out.append(f"UNLEDGERED {name}: end the test in CompareLedger::finish (olv0j.1)")
        if ledgered and name in listed:
            out.append(f"STALE {name}: it is ledgered now; remove it from {REGISTER.name}")
    for name in sorted(listed - set(files)):
        out.append(f"STALE {name}: no such diff test; remove it from {REGISTER.name}")
    return out


def self_test() -> bool:
    tmp = ROOT / "target" / "unledgered_diff_tests_selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    for old in tmp.glob("diff_*.rs"):
        old.write_text("")  # reset the fixtures in place; the checks below rewrite each one
    ledgered = 'let mut l = CompareLedger::new("t", &["x"]);\nl.compared("x", "a", true);\nl.finish(1);\n'
    files = {
        "diff_ledgered.rs": ledgered,
        "diff_bare.rs": "let all_pass = diffs.iter().all(|d| d.pass);\nassert!(all_pass);\n",
        "diff_half.rs": 'let l = CompareLedger::new("t", &["x"]);\n// l.finish(1) is only a comment\n',
    }
    for name, text in files.items():
        (tmp / name).write_text(text)
    checks = {
        "an unlisted bare test is reported": any("UNLEDGERED diff_bare.rs" in v for v in violations(tmp, [])),
        "a ledger without finish is not ledgered": any(
            "UNLEDGERED diff_half.rs" in v for v in violations(tmp, [])),
        "a listed bare test passes": violations(tmp, ["diff_bare.rs", "diff_half.rs"]) == [],
        "a listed ledgered test is stale": any(
            "STALE diff_ledgered.rs" in v for v in violations(tmp, ["diff_bare.rs", "diff_half.rs", "diff_ledgered.rs"])),
        "a listed missing test is stale": any(
            "STALE diff_gone.rs" in v for v in violations(tmp, ["diff_bare.rs", "diff_half.rs", "diff_gone.rs"])),
    }
    for label, ok in checks.items():
        if not ok:
            print(f"self-test FAILED: {label}")
    return all(checks.values())


def main() -> int:
    if not self_test():
        print("REFUSING: self-test failed", file=sys.stderr)
        return 2
    if "--self-test" in sys.argv:
        print("self-test passed")
        return 0
    register = read_register(REGISTER)
    found = violations(TESTS, register)
    total = len(list(TESTS.glob("diff_*.rs")))
    print(f"{total} diff tests, {total - len(register)} ledgered, {len(register)} still listed")
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
