#!/usr/bin/env python3
"""Ledger preflight — refuse a perf lever whose evidence cannot decide it.

Read-only. Exit 0 = CLEAR, exit 2 = BLOCKED, exit 64 = usage/internal error.
Modelled on frankensqlite's `sql_pipeline_candidate_preflight`, which is why that
repo's void rate is 1.7% while repos that audited once and stopped sit at 25-91%
(fleet broadcast 2026-07-25). **Ledger integrity decays.** A one-time cleanup buys a
month; a check that runs every time is what holds.

Four modes.

  --propose "<free text describing the lever>" --surface "<target surface>"
      Run BEFORE touching source. Greps the negative-evidence ledgers for prior
      rows on the target surface, prints every recorded retry predicate, and exits
      2 if a sound prior REJECT already covers the proposal.

  --check-row <file> [--row <n>]
      Check one ledger row. Refuses a REJECT that records neither an A/A null
      control nor a counted mechanism, refuses a cv-only REJECT, and refuses a
      KEEP without an executed-binary SHA-256 and an explicit result class.
      CAMPAIGN-WIN requires a SciPy legacy-incumbent arm, side-by-side
      same-invocation evidence, and an unambiguous incumbent ratio. Every KEEP
      and every A/A-timed REJECT also records host identity, physical cores,
      logical threads, actual threads used, runtime-detected ISA, and
      affinity/cpuset.

  --check-staged
      Pre-commit mode. Reads ledger blobs from Git's INDEX, finds every newly
      staged row, and applies the same fail-closed checks. Historical rows are
      not grandfathered into a new commit merely because they share a file.

  --self-test
      Run deterministic contract tests without Cargo or a worker.

The taxonomy and median-CI rule are fleet policy; see
`docs/LEDGER_RESURRECTION.md` and `docs/OPTIMIZATION_PROTOCOL.md`.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGERS = [
    REPO / "docs/NEGATIVE_EVIDENCE.md",
    REPO / "docs/progress/perf-negative-results.md",
    REPO / "docs/perf_ledger_cc.md",
]

# A REJECT is decidable only if it carries EITHER a measured A/A null control OR
# a counted mechanism. Mentioning the words without a value is not evidence.
NULL_RE = re.compile(
    r"(?:A/A|A-A null|null (?:control|floor|range|band|p95|median)|"
    r"self-A/B|identical-arm|base-vs-base|paired\(base, ?base\))"
    r"(?:(?!\n## ).){0,240}?(?:\d+\.\d+|\[[^\]\n]*\d[^\]\n]*\])",
    re.IGNORECASE | re.DOTALL,
)
# "Counted" means a hardware or OS counter, not a wall clock: a null control cannot
# change the fact that no work was removed.
COUNTED_RE = re.compile(
    r"(?:instructions?|cycles|syscalls?|allocations?|(?:page|minor)[- ]faults)"
    r"[^.\n]{0,180}?"
    r"(?:unchanged|identical|no change|same|more|fewer|eliminated but|"
    r"\d+(?:\.\d+)?(?:e[+-]?\d+)?[^.\n]{0,50}(?:vs\.?|→)"
    r"[^.\n]{0,50}\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
    r"|zero `?idiv`? emitted|no work (?:was )?removed|nothing was removed",
    re.IGNORECASE,
)

# NOTE: a PROFILE attribution ("80.09% self cycles in dense LU") is deliberately NOT a
# counted mechanism. Self-time says where the time IS; VALID-MECHANISM requires evidence
# that the candidate REMOVED NO WORK. Conflating them classified `.165` — a 17-19x lever
# killed on a cv gate — as sound, which is the exact failure this file exists to prevent.

CV_ONLY_RE = re.compile(
    r"(CV\s*>\s*5|CV ceiling|CVs? (?:above|exceed|remained above)|"
    r"failed the (?:mandatory )?CV|mandatory CV|all three CVs)",
    re.IGNORECASE,
)
MEDIAN_CI_RE = re.compile(
    r"(?:bootstrap[- ]median(?: 95%)? CIs?|median[- ]CI GATE|"
    r"candidate CI|cand(?:idate)?[_ -]?ci)"
    r"(?:(?!\n## ).){0,800}?"
    r"(?:DECIDED|IN-FLOOR|NOT DECIDED|\[[^\]\n]*\d[^\]\n]*\])",
    re.IGNORECASE | re.DOTALL,
)
# KEEP rows use an explicit, machine-checked result class. "CAMPAIGN-WIN" is
# deliberately narrow; a self-comparison remains maintenance even when a
# separately invoked SciPy process suggests the current code is competitive.
RESULT_CLASS_RE = re.compile(
    r"\bresult class(?:\*\*)?\s*:\s*(?:\*\*)?`?\s*"
    r"(CAMPAIGN-WIN|SELF-SPEEDUP)\s*`?",
    re.IGNORECASE,
)
LEGACY_INCUMBENT_ARM_RE = re.compile(
    r"\blegacy incumbent arm(?:\*\*)?\s*:\s*(?:\*\*)?"
    r"(?:`)?SciPy(?:\s+[0-9][0-9A-Za-z.+-]*)?"
    r"(?:(?!\n#{2,6} ).){0,180}?\bsame[- ]invocation\b",
    re.IGNORECASE | re.DOTALL,
)
SIDE_BY_SIDE_RE = re.compile(
    r"\bside[- ]by[- ]side\b(?:(?!\n#{2,6} ).){0,100}?\bsame[- ]invocation\b"
    r"|\bsame[- ]invocation\b(?:(?!\n#{2,6} ).){0,100}?\bside[- ]by[- ]side\b",
    re.IGNORECASE | re.DOTALL,
)
INCUMBENT_RATIO_RE = re.compile(
    r"\bincumbent ratio(?:\*\*)?\s*:\s*(?:\*\*)?"
    r"(?:`)?SciPy\s*/\s*(?:FrankenSciPy|fsci)\s*=\s*"
    r"\d+(?:\.\d+)?x(?:`)?\b",
    re.IGNORECASE,
)
HOST_IDENTITY_RE = re.compile(
    r"\bhost(?:_identity|\s+identity)?\s*(?:=|:)\s*`?[a-z0-9][a-z0-9_.-]*"
    r"|\bhost\s+`[a-z0-9][a-z0-9_.-]*`",
    re.IGNORECASE,
)
PHYSICAL_CORES_RE = re.compile(
    r"\bphysical_cores\s*=\s*\d+\b|\b\d+\s+physical cores?\b",
    re.IGNORECASE,
)
LOGICAL_THREADS_RE = re.compile(
    r"\blogical_threads\s*=\s*\d+\b|\b\d+\s+logical threads?\b",
    re.IGNORECASE,
)
ACTUAL_THREADS_RE = re.compile(
    r"\bthreads_(?:actually_)?used\s*=\s*\d+\b"
    r"|\bactual(?:ly)?(?:\s+[a-z]+){0,3}\s+threads?(?:\s+used)?\s*(?:=|:)?\s*"
    r"`?\d+(?:[/,]\d+)*`?\b",
    re.IGNORECASE,
)
RUNTIME_ISA_RE = re.compile(
    r"\bruntime(?:[-_ ]detected)?[-_ ]isa(?:[-_ ]features?)?\s*(?:=|:)?\s*"
    r"`?[a-z0-9]",
    re.IGNORECASE,
)
AFFINITY_CPUSET_RE = re.compile(
    r"\b(?:affinity|affinities|cpuset(?:_logical_cap)?)\s*(?:=|:)?\s*"
    r"`?[0-9]",
    re.IGNORECASE,
)

REJECT_HEAD_RE = re.compile(
    r"\b(REJECT|REJECTED|INVALID|NO-SHIP|NEGATIVE RESULT|DEAD END|ABANDON)\b",
    re.IGNORECASE,
)
KEEP_HEAD_RE = re.compile(r"\b(KEEP|KEPT)\b", re.IGNORECASE)
WIN_HEAD_RE = re.compile(r"\bWIN\b", re.IGNORECASE)
ELF_SHA256_RE = re.compile(
    r"\b(?:executed[- ]binary|binary|elf)(?:(?!\n).){0,40}?"
    r"(?:sha(?:-?256)?)(?:\s*[:=]\s*|\s+)"
    r"(?:`)?([0-9a-f]{64})(?:`)?\b",
    re.IGNORECASE,
)
# 2+ hashes: docs/NEGATIVE_EVIDENCE.md and perf-negative-results.md use `##`,
# docs/perf_ledger_cc.md uses `###`. Matching only `##` silently parsed ZERO
# entries from the cc ledger, so every gate below was a no-op on it.
HEAD_RE = re.compile(r"^#{2,6} (.+)$")
STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "is", "are",
    "per", "via", "into", "over", "under", "from", "by", "at", "as", "that", "this",
    "reject", "keep", "win", "lever", "perf", "fsci", "candidate", "arm", "run",
}


def entries_from_text(path: Path, text: str):
    lines = text.split("\n")
    idx = [i for i, line in enumerate(lines) if HEAD_RE.match(line)]
    for n, i in enumerate(idx):
        end = idx[n + 1] if n + 1 < len(idx) else len(lines)
        head = lines[i].lstrip("#").strip()
        yield path, i + 1, head, "\n".join(lines[i + 1 : end])


def entries(path: Path):
    if path.exists():
        yield from entries_from_text(path, path.read_text(errors="replace"))


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z_][a-z0-9_]{2,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


def verdict_class(head: str, body: str) -> str:
    blob = head + "\n" + body
    has_null = bool(NULL_RE.search(blob))
    has_counted = bool(COUNTED_RE.search(blob))
    if CV_ONLY_RE.search(blob) and not MEDIAN_CI_RE.search(blob):
        return "VOID-CV"
    if has_null:
        return "VALID-AB"
    if has_counted:
        return "VALID-MECHANISM"
    return "VOID-NONULL"


def retry_predicate(body: str) -> str:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if not re.search(
            r"\bretry(?: predicate| condition)?\b", line, re.IGNORECASE
        ):
            continue
        selected = [line.strip()]
        for continuation in lines[index + 1 : index + 4]:
            stripped = continuation.strip()
            if not stripped or stripped.startswith(("#", "|")):
                break
            if re.match(r"[-*]\s+", stripped):
                break
            selected.append(stripped)
        return " ".join(selected)[:500]
    return "NOT_RECORDED"


def row_evidence(head: str, body: str) -> tuple[bool, bool, bool, bool]:
    blob = head + "\n" + body
    return (
        bool(NULL_RE.search(blob)),
        bool(COUNTED_RE.search(blob)),
        bool(MEDIAN_CI_RE.search(blob)),
        bool(ELF_SHA256_RE.search(blob)),
    )


def result_class(head: str, body: str) -> str | None:
    matches = RESULT_CLASS_RE.findall(head + "\n" + body)
    return matches[0].upper() if len(matches) == 1 else None


def hardware_provenance_missing(head: str, body: str) -> list[str]:
    blob = head + "\n" + body
    fields = [
        ("host identity", HOST_IDENTITY_RE),
        ("physical cores", PHYSICAL_CORES_RE),
        ("logical threads", LOGICAL_THREADS_RE),
        ("actual threads used", ACTUAL_THREADS_RE),
        ("runtime-detected ISA", RUNTIME_ISA_RE),
        ("affinity/cpuset", AFFINITY_CPUSET_RE),
    ]
    return [name for name, pattern in fields if not pattern.search(blob)]


def row_errors(head: str, body: str) -> list[str]:
    blob = head + "\n" + body
    has_null, has_counted, has_median_ci, has_elf_sha = row_evidence(head, body)
    errors = []
    is_reject = bool(REJECT_HEAD_RE.search(head))
    is_keep = bool(KEEP_HEAD_RE.search(head))
    if is_keep or (is_reject and has_null):
        missing_provenance = hardware_provenance_missing(head, body)
        if missing_provenance:
            errors.append(
                "timed result lacks mandatory hardware/thread provenance: "
                + ", ".join(missing_provenance)
            )
    if is_reject:
        if not has_null and not has_counted:
            errors.append(
                "REJECT records neither a measured same-invocation A/A null "
                "nor a counted mechanism"
            )
        if CV_ONLY_RE.search(blob) and not has_median_ci:
            errors.append(
                "REJECT invokes a cv ceiling without a bootstrap-median CI decision"
            )
    if is_keep:
        if not has_elf_sha:
            errors.append("KEEP has no executed-binary ELF SHA-256")
        classifications = RESULT_CLASS_RE.findall(blob)
        classification = result_class(head, body)
        if len(classifications) != 1:
            errors.append(
                "KEEP must record exactly one result class — use "
                "`Result class: CAMPAIGN-WIN` or `Result class: SELF-SPEEDUP`"
            )
        elif classification == "CAMPAIGN-WIN":
            if not LEGACY_INCUMBENT_ARM_RE.search(blob):
                errors.append(
                    "CAMPAIGN-WIN has no named SciPy legacy-incumbent arm "
                    "recorded in the same invocation"
                )
            if not SIDE_BY_SIDE_RE.search(blob):
                errors.append(
                    "CAMPAIGN-WIN lacks side-by-side same-invocation harness evidence"
                )
            if not INCUMBENT_RATIO_RE.search(blob):
                errors.append(
                    "CAMPAIGN-WIN lacks an unambiguous "
                    "`Incumbent ratio: SciPy / FrankenSciPy = <ratio>x`"
                )
        elif WIN_HEAD_RE.search(head):
            errors.append(
                "SELF-SPEEDUP is maintenance and may not be titled as a WIN"
            )
    return errors


def cmd_propose(text: str, surface: str, threshold: float) -> int:
    want = tokens(text)
    if len(want) < 2:
        print("preflight: describe the lever in a few words (need >=2 content tokens)")
        return 64
    surface_want = tokens(surface)
    if not surface_want:
        print("preflight: name a concrete target surface")
        return 64
    hits = []
    for path, line, head, body in (e for p in LEDGERS for e in entries(p)):
        row_tokens = tokens(head + "\n" + body)
        lever_overlap = want & row_tokens
        surface_overlap = surface_want & row_tokens
        if not lever_overlap or not surface_overlap:
            continue
        lever_score = len(lever_overlap) / len(want)
        surface_score = len(surface_overlap) / len(surface_want)
        score = 0.35 * lever_score + 0.65 * surface_score
        if score >= threshold:
            hits.append((score, path, line, head, body))
    hits.sort(reverse=True, key=lambda h: h[0])

    if not hits:
        print(
            f"preflight: CLEAR — no prior row matches surface={surface!r} "
            f"lever={text!r}."
        )
        print("  Reminder: a weak match is not a clean bill of health. Grep by hand too.")
        return 0

    blocking = []
    print(
        f"preflight: {len(hits)} prior ledger row(s) match "
        f"surface={surface!r} lever={text!r}\n"
    )
    for score, path, line, head, body in hits[:12]:
        cls = verdict_class(head, body) if REJECT_HEAD_RE.search(head) else "KEEP/other"
        mark = ""
        if REJECT_HEAD_RE.search(head) and cls.startswith("VALID"):
            mark = "  <-- BLOCKING: already decided on sound evidence"
            blocking.append((path, line, head, cls))
        elif REJECT_HEAD_RE.search(head):
            mark = "  <-- prior REJECT, but its own evidence is VOID: re-running is legitimate"
        print(f"  [{score:.0%}] {path.name}:{line}  ({cls}){mark}")
        print(f"        {head[:150]}")
        print(f"        retry_predicate={retry_predicate(body)}")
    if blocking:
        print("\nBLOCKED. Sound prior rejections exist for this lever:")
        for path, line, head, cls in blocking:
            print(f"  - {path.name}:{line} [{cls}] {head[:120]}")
        print(
            "\nProceed only if you can state what is DIFFERENT from the prior attempt,\n"
            "and record that difference in the new row. Otherwise pick another lever."
        )
        return 2
    print("\npreflight: CLEAR — prior rows exist but none rejected this on sound evidence.")
    return 0


def report_row(path: Path, line: int, head: str, body: str) -> int:
    has_null, has_counted, has_median_ci, has_elf_sha = row_evidence(head, body)
    errors = row_errors(head, body)
    row_kind = []
    if REJECT_HEAD_RE.search(head):
        row_kind.append(verdict_class(head, body))
    if KEEP_HEAD_RE.search(head):
        row_kind.append("KEEP")
        row_kind.append(result_class(head, body) or "UNCLASSED")
    print(f"preflight: checking {path}:{line}\n  {head[:160]}")
    print(
        "  class={}  A/A-null={}  counted-mechanism={}  "
        "median-CI={}  executed-ELF-sha256={}".format(
            "+".join(row_kind) if row_kind else "OTHER",
            has_null,
            has_counted,
            has_median_ci,
            has_elf_sha,
        )
    )
    if not errors:
        print("preflight: CLEAR — row satisfies every applicable evidence gate.")
        return 0
    print("\nBLOCKED: ledger row is inadmissible:")
    for error in errors:
        print(f"  - {error}")
    print(
        "\nRequired repairs:\n"
        "  REJECT: record same-invocation A/A values or a counted mechanism; decide\n"
        "          timing only from the bootstrap-median CI, never cv.\n"
        "  TIMED:  name host, physical cores, logical threads, actual threads used,\n"
        "          runtime ISA, and affinity/cpuset for every KEEP or A/A REJECT.\n"
        "  KEEP:   record the 64-hex SHA-256 self-reported by the executed ELF and\n"
        "          exactly one result class.\n"
        "          CAMPAIGN-WIN additionally requires\n"
        "          a SciPy legacy-incumbent arm, side-by-side in the same invocation,\n"
        "          plus `Incumbent ratio: SciPy / FrankenSciPy = <ratio>x`."
    )
    return 2


def cmd_check_row(path: Path, row: int | None) -> int:
    found = list(entries(path))
    if not found:
        print(f"preflight: no ledger entries parsed from {path}")
        return 64
    if row is None:
        _, line, head, body = found[-1]  # newest entry appended at EOF
    else:
        match = [e for e in found if e[1] == row]
        if not match:
            print(f"preflight: no entry begins at line {row} of {path}")
            return 64
        _, line, head, body = match[0]
    return report_row(path, line, head, body)


def git_text(*args: str, allow_missing: bool = False) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode == 0:
        return result.stdout
    if allow_missing:
        return ""
    raise RuntimeError(
        f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
    )


def new_index_entries(path: Path, staged_text: str, head_text: str):
    existing = Counter(head for _, _, head, _ in entries_from_text(path, head_text))
    for entry in entries_from_text(path, staged_text):
        head = entry[2]
        if existing[head]:
            existing[head] -= 1
        else:
            yield entry


def cmd_check_staged() -> int:
    relative = [str(path.relative_to(REPO)) for path in LEDGERS]
    changed = set(
        git_text(
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "--",
            *relative,
        ).splitlines()
    )
    if not changed:
        print("ledger-preflight hook: CLEAR — no staged ledger rows.")
        return 0

    blocked = False
    checked = 0
    for rel in relative:
        if rel not in changed:
            continue
        path = REPO / rel
        staged_text = git_text("show", f":{rel}")
        head_text = git_text("show", f"HEAD:{rel}", allow_missing=True)
        for _, line, head, body in new_index_entries(path, staged_text, head_text):
            checked += 1
            blocked |= report_row(Path(rel), line, head, body) == 2
    if blocked:
        print("\nledger-preflight hook: BLOCKED — repair the staged row(s) and re-stage.")
        return 2
    print(f"ledger-preflight hook: CLEAR — {checked} newly staged row(s) checked.")
    return 0


def cmd_self_test() -> int:
    sha = "a" * 64
    provenance = (
        "Host identity: trj. 64 physical cores / 128 logical threads. "
        "Actual fsci threads 16. Runtime-detected ISA: avx2=true. Affinity: 0-15."
    )
    cases = [
        (
            "reject_without_control",
            "2026-07-25 REJECT: neutral wall ratio",
            "A/B was 1.001x. Retry on a quieter worker.",
            True,
        ),
        (
            "reject_with_null_and_median_ci",
            "2026-07-25 REJECT: inside floor",
            (
                "A/A null CI [0.99, 1.01]. Candidate CI [1.00, 1.01]. "
                f"bootstrap-median CI verdict IN-FLOOR. {provenance}"
            ),
            False,
        ),
        (
            "timed_reject_without_hardware_provenance",
            "2026-07-25 REJECT: inside floor",
            (
                "A/A null CI [0.99, 1.01]. Candidate CI [1.00, 1.01]. "
                "bootstrap-median CI verdict IN-FLOOR."
            ),
            True,
        ),
        (
            "reject_with_counted_mechanism",
            "2026-07-25 REJECT: work count unchanged",
            "instructions 1.20e9 vs 1.20e9; no work was removed.",
            False,
        ),
        (
            "cv_only_reject",
            "2026-07-25 REJECT: CV > 5%",
            "A/A null median 1.003. Candidate CV remained above 5%.",
            True,
        ),
        (
            "keep_without_elf_sha",
            "2026-07-25 KEEP: candidate retained",
            "Result class: SELF-SPEEDUP. Candidate median-CI clears the null.",
            True,
        ),
        (
            "keep_without_result_class",
            "2026-07-25 KEEP: candidate retained",
            f"executed ELF sha256={sha}. Candidate median-CI clears the null.",
            True,
        ),
        (
            "self_speedup_keep",
            "2026-07-25 KEEP: candidate retained",
            f"Result class: SELF-SPEEDUP. executed ELF sha256={sha}. {provenance}",
            False,
        ),
        (
            "self_speedup_titled_win",
            "2026-07-25 KEEP WIN: candidate retained",
            f"Result class: SELF-SPEEDUP. executed ELF sha256={sha}. {provenance}",
            True,
        ),
        (
            "keep_without_hardware_provenance",
            "2026-07-25 KEEP: candidate retained",
            f"Result class: SELF-SPEEDUP. executed ELF sha256={sha}",
            True,
        ),
        (
            "keep_with_conflicting_result_classes",
            "2026-07-25 KEEP: candidate retained",
            (
                f"Result class: SELF-SPEEDUP. Result class: CAMPAIGN-WIN. "
                f"executed ELF sha256={sha}"
            ),
            True,
        ),
        (
            "campaign_win_without_incumbent_arm",
            "2026-07-25 KEEP WIN: candidate beats incumbent",
            (
                f"Result class: CAMPAIGN-WIN. executed ELF sha256={sha}. "
                "Incumbent ratio: SciPy / FrankenSciPy = 1.23x."
            ),
            True,
        ),
        (
            "campaign_win_cross_invocation",
            "2026-07-25 KEEP WIN: candidate beats incumbent",
            (
                f"Result class: CAMPAIGN-WIN. executed ELF sha256={sha}. "
                "Legacy incumbent arm: SciPy 1.17.1, measured in a separate invocation. "
                "Incumbent ratio: SciPy / FrankenSciPy = 1.23x."
            ),
            True,
        ),
        (
            "campaign_win_same_invocation",
            "2026-07-25 KEEP WIN: candidate beats incumbent",
            (
                f"Result class: CAMPAIGN-WIN. executed ELF sha256={sha}. "
                "Legacy incumbent arm: SciPy 1.17.1, side-by-side in the same invocation. "
                "Incumbent ratio: SciPy / FrankenSciPy = 1.23x. "
                f"{provenance}"
            ),
            False,
        ),
        (
            "cv_provenance_never_decides",
            "2026-07-25 REJECT: candidate remains in floor",
            (
                "A/A null CI [0.99, 1.01]. bootstrap-median candidate CI "
                f"[1.00, 1.01] verdict IN-FLOOR. CV > 5% is provenance only. "
                f"{provenance}"
            ),
            False,
        ),
    ]
    for name, head, body, should_block in cases:
        blocked = bool(row_errors(head, body))
        if blocked != should_block:
            print(
                f"self-test FAILED: {name}: expected blocked={should_block}, "
                f"got blocked={blocked}: {row_errors(head, body)}"
            )
            return 1
    if retry_predicate("Retry only if target self-time exceeds 5%.") == "NOT_RECORDED":
        print("self-test FAILED: retry predicate extraction")
        return 1
    old = "## old KEEP\nexecuted ELF sha256=" + sha + "\n"
    staged = old + "\n## new REJECT\nwall ratio 1.001x\n"
    additions = list(new_index_entries(Path("ledger.md"), staged, old))
    if len(additions) != 1 or additions[0][2] != "new REJECT":
        print(f"self-test FAILED: staged-row detection: {additions}")
        return 1
    print(f"ledger-preflight self-test: PASS ({len(cases) + 2} checks)")
    return 0


class ContractArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(64, f"{self.prog}: error: {message}\n")


def main() -> int:
    # A no-argument invocation is the hook entry point. This lets the checked-in
    # script itself be linked into an existing hook chain without a wrapper.
    if len(sys.argv) == 1:
        return cmd_check_staged()

    ap = ContractArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--propose", metavar="TEXT", help="describe a lever before writing source")
    g.add_argument(
        "--check-row",
        metavar="FILE",
        help="check a ledger file's newest (or --row) entry",
    )
    g.add_argument("--check-staged", action="store_true", help="check newly staged ledger rows")
    g.add_argument("--self-test", action="store_true", help="run deterministic contract tests")
    ap.add_argument("--surface", help="target source/API surface (required with --propose)")
    ap.add_argument("--row", type=int, default=None, help="line number the entry starts at")
    ap.add_argument("--threshold", type=float, default=0.34, help="token-overlap match threshold")
    args = ap.parse_args()

    if args.propose:
        if not args.surface:
            ap.error("--surface is required with --propose")
        return cmd_propose(args.propose, args.surface, args.threshold)
    if args.check_row:
        return cmd_check_row(Path(args.check_row), args.row)
    if args.self_test:
        return cmd_self_test()
    return cmd_check_staged()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ledger-preflight internal error: {error}", file=sys.stderr)
        sys.exit(64)
