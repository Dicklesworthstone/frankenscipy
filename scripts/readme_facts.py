#!/usr/bin/env python3
"""Verify README.md numeric claims against live repository counts (frankenscipy-1yifk.4).

Reads the repository tree and `docs/planning/symbol_census.json` (no network or live SciPy
required) and checks every `<!-- fact:<name> -->` marker in `README.md`.

Fails closed on:
  1. Any `<!-- fact:<name> -->` marker whose `<name>` is not a known computed fact.
  2. Any marker without a preceding numeric literal on the same line.
  3. Any marker whose value differs from the computed fact by more than its allowed tolerance.
  4. Any required fact that has zero markers in `README.md`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_ATTR_RE = re.compile(r"#\[(?:[\w:]+::)?test\b")
MARKER_RE = re.compile(r"<!--\s*fact:([A-Za-z0-9_.-]+)\s*-->")
NUM_BEFORE_MARKER_RE = re.compile(
    r"(~?\d[\d,]*(?:\.\d+)?[Kk]?)\+?(?:[`*~\s]|</?[a-zA-Z][^>]*>)*$"
)


def compute_facts(root: pathlib.Path = ROOT) -> dict[str, tuple[int, int]]:
    """Return {fact_name: (computed_value, allowed_tolerance)}."""
    crates_dir = root / "crates"
    crates = sorted(
        p for p in crates_dir.iterdir() if p.is_dir() and (p / "Cargo.toml").exists()
    )

    facts: dict[str, tuple[int, int]] = {
        "crates": (len(crates), 0),
    }

    total_src_lines = 0
    inline_tests = 0
    all_tests = 0

    for crate in crates:
        src_dir = crate / "src"
        crate_lines = 0
        for rs in src_dir.rglob("*.rs"):
            text = rs.read_text(errors="replace")
            crate_lines += len(text.splitlines())
            t_count = len(TEST_ATTR_RE.findall(text))
            inline_tests += t_count
            all_tests += t_count
        total_src_lines += crate_lines
        tol = 200 if crate_lines < 2000 else 3000
        key = f"src_lines_{crate.name.replace('-', '_')}"
        facts[key] = (crate_lines, tol)

        tests_dir = crate / "tests"
        if tests_dir.exists():
            for rs in tests_dir.rglob("*.rs"):
                text = rs.read_text(errors="replace")
                all_tests += len(TEST_ATTR_RE.findall(text))

    conf_tests_dir = crates_dir / "fsci-conformance" / "tests"
    conf_test_files = len(list(conf_tests_dir.glob("*.rs")))
    diff_files = len(list(conf_tests_dir.glob("diff_*.rs")))
    oracle_scripts = len(
        list((crates_dir / "fsci-conformance" / "python_oracle").glob("*.py"))
    )

    fuzz_targets = len(list((root / "fuzz" / "fuzz_targets").glob("*.rs")))
    fuzz_corpus = sum(
        1 for f in (root / "fuzz" / "corpus").rglob("*") if f.is_file()
    )
    fuzz_seeds = sum(
        1 for f in (root / "fuzz" / "seeds").rglob("*") if f.is_file()
    )
    fsci_packets = sum(
        1
        for d in (
            crates_dir / "fsci-conformance" / "fixtures" / "artifacts"
        ).glob("FSCI-P2C-*")
        if d.is_dir()
    )
    stats_lib_lines = len(
        (crates_dir / "fsci-stats" / "src" / "lib.rs")
        .read_text(errors="replace")
        .splitlines()
    )

    census_path = root / "docs" / "planning" / "symbol_census.json"
    census_doc = json.loads(census_path.read_text(encoding="utf-8"))
    totals = census_doc["total"]

    facts.update(
        {
            "total_src_lines": (total_src_lines, 15000),
            "all_tests": (all_tests, 250),
            "inline_tests": (inline_tests, 250),
            "conf_test_files": (conf_test_files, 0),
            "diff_files": (diff_files, 0),
            "oracle_scripts": (oracle_scripts, 0),
            "fuzz_targets": (fuzz_targets, 0),
            "fuzz_corpus": (fuzz_corpus, 0),
            "fuzz_seeds": (fuzz_seeds, 0),
            "fsci_packets": (fsci_packets, 0),
            "stats_lib_lines": (stats_lib_lines, 5000),
            "census_modules": (len(census_doc["modules"]), 0),
            "census_scipy": (int(totals["scipy"]), 0),
            "census_na": (int(totals["na"]), 0),
            "census_applicable": (int(totals["applicable"]), 0),
            "census_declared": (int(totals["declared"]), 0),
            "census_real": (int(totals["real"]), 0),
            "census_compared": (int(totals["compared"]), 0),
        }
    )
    return facts


def parse_readme_number(raw: str) -> int:
    s = raw.lstrip("~").replace(",", "")
    if s.endswith(("K", "k")):
        return int(round(float(s[:-1]) * 1000))
    return int(round(float(s)))


def check_readme_text(
    text: str,
    facts: dict[str, tuple[int, int]],
    source_name: str = "README.md",
    require_all_facts: bool = True,
) -> list[str]:
    errors: list[str] = []
    seen_facts: set[str] = set()
    in_fence = False

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence

        for match in MARKER_RE.finditer(line):
            fact_name = match.group(1)
            if in_fence:
                errors.append(
                    f"{source_name}:{line_no}: marker <!-- fact:{fact_name} --> is inside a fenced code block"
                )
            if fact_name not in facts:
                errors.append(
                    f"{source_name}:{line_no}: unknown fact marker <!-- fact:{fact_name} -->"
                )
                continue

            prefix = line[: match.start()]
            num_match = NUM_BEFORE_MARKER_RE.search(prefix)
            if not num_match:
                errors.append(
                    f"{source_name}:{line_no}: no numeric value immediately precedes <!-- fact:{fact_name} -->"
                )
                continue

            raw_val = num_match.group(1)
            readme_val = parse_readme_number(raw_val)
            computed_val, tol = facts[fact_name]
            seen_facts.add(fact_name)

            if abs(readme_val - computed_val) > tol:
                errors.append(
                    f"{source_name}:{line_no}: fact '{fact_name}' drifted: README has {raw_val} ({readme_val}), "
                    f"computed {computed_val} (allowed tolerance ±{tol})"
                )

    if require_all_facts:
        missing = sorted(set(facts) - seen_facts)
        if missing:
            errors.append(
                f"{source_name}: missing markers for computed facts: {', '.join(missing)}"
            )

    return errors


def run_self_test() -> None:
    sample_facts = {
        "diff_files": (781, 0),
        "total_src_lines": (722531, 15000),
        "stats_lib_lines": (114773, 5000),
    }
    good_readme = (
        "We have **781**<!-- fact:diff_files --> diff files across "
        "~722,500<!-- fact:total_src_lines --> lines and "
        "~115K<!-- fact:stats_lib_lines --> lines in stats.\n"
    )
    errs_good = check_readme_text(good_readme, sample_facts, "good.md")
    if errs_good:
        raise AssertionError(f"self-test must-hit failed: {errs_good}")

    drifted_readme = good_readme.replace("781", "780")
    errs_drift = check_readme_text(drifted_readme, sample_facts, "drift.md")
    if not any("fact 'diff_files' drifted" in e for e in errs_drift):
        raise AssertionError(
            f"self-test must-miss (drifted number) did not catch drift: {errs_drift}"
        )

    typo_readme = good_readme + "Extra 42<!-- fact:typo_nonexistent --> item.\n"
    errs_typo = check_readme_text(typo_readme, sample_facts, "typo.md")
    if not any("unknown fact marker <!-- fact:typo_nonexistent -->" in e for e in errs_typo):
        raise AssertionError(
            f"self-test must-miss (unknown marker) did not catch typo: {errs_typo}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", dest="do_print", action="store_true", help="Print computed facts")
    parser.add_argument("--check", action="store_true", help="Check README.md against computed facts")
    parser.add_argument("--self-test", action="store_true", help="Run built-in must-hit / must-miss self-test")
    parser.add_argument(
        "--readme",
        type=pathlib.Path,
        default=ROOT / "README.md",
        help="Path to README file to check (default: README.md)",
    )
    args = parser.parse_args()

    run_self_test()
    if args.self_test and not (args.do_print or args.check):
        print("self-test: ok (valid pass, drifted-number fail, unknown-marker fail)")
        return 0

    facts = compute_facts(ROOT)
    if args.do_print or not args.check:
        for name, (val, tol) in sorted(facts.items()):
            print(f"{name:28s} = {val:8d}  (±{tol})")
        if not args.check:
            return 0

    readme_text = args.readme.read_text(encoding="utf-8")
    errors = check_readme_text(readme_text, facts, str(args.readme.relative_to(ROOT) if args.readme.is_relative_to(ROOT) else args.readme))
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(
        f"self-test: ok; {args.readme.name}: all {len(facts)} facts verified across "
        f"{len(MARKER_RE.findall(readme_text))} markers"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
