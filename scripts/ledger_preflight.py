#!/usr/bin/env python3
"""Ledger preflight — refuse a perf lever whose evidence cannot decide it.

Read-only. Exit 0 = CLEAR, exit 2 = BLOCKED, exit 1 = usage/internal error.
Modelled on frankensqlite's `sql_pipeline_candidate_preflight`, which is why that
repo's void rate is 1.7% while repos that audited once and stopped sit at 25-91%
(fleet broadcast 2026-07-25). **Ledger integrity decays.** A one-time cleanup buys a
month; a check that runs every time is what holds.

Two modes.

  --propose "<free text describing the lever>"
      Run BEFORE touching source. Greps both ledgers for prior art and prints any
      row that already decided this lever. Exit 2 if a prior REJECT with a *sound*
      verdict class covers it — re-deriving a closed lever is the single most
      common waste in this repo's history (two rows rejected in July had already
      been shipped by another agent twelve days earlier).

  --check-row <file> [--row <n>]
      Run BEFORE committing a new ledger row. Refuses a REJECT that records neither
      an A/A null control nor a counted mechanism, because such a row cannot
      distinguish the lever from the harness. That is the VOID-NONULL class, 30 of
      this repo's 48 void rows and 214 of frankenfs's 219.

The taxonomy is frankenfs's, adopted fleet-wide 2026-07-25; see
`docs/LEDGER_RESURRECTION.md` and `docs/OPTIMIZATION_PROTOCOL.md`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGERS = [
    REPO / "docs/NEGATIVE_EVIDENCE.md",
    REPO / "docs/progress/perf-negative-results.md",
    REPO / "docs/perf_ledger_cc.md",
]

# A row is decidable if it carries EITHER an A/A null control OR a counted mechanism.
NULL_RE = re.compile(
    r"(A/A|A-A null|null control|null floor|null range|null band|null p95|null median|"
    r"NULL median|null \[|inside the null|self-A/B|identical-arm|base-vs-base|"
    r"CONTROL(?: arm| column| row| \()|control arm|placebo|paired\(base, ?base\))",
    re.I,
)
# "Counted" means a hardware or OS counter, not a wall clock: a null control cannot
# change the fact that no work was removed.
COUNTED_RE = re.compile(
    r"(instructions?[^.\n]{0,30}(?:unchanged|identical|no change|same|more|fewer|MORE|FEWER)|"
    r"cycles[^.\n]{0,30}(?:unchanged|identical|no change|same|more|fewer|MORE|FEWER)|"
    r"syscalls?[^.\n]{0,30}(?:unchanged|identical|no change|same|more|fewer)|"
    r"allocations?[^.\n]{0,30}(?:unchanged|identical|no change|same|eliminated but)|"
    r"(?:page|minor)[- ]faults[^.\n]{0,30}(?:unchanged|identical|no change|same)|"
    r"perf stat[^.\n]{0,40}(?:unchanged|identical|no change)|"
    r"zero `?idiv`? emitted|no work (?:was )?removed|nothing was removed)",
    re.I,
)

# NOTE: a PROFILE attribution ("80.09% self cycles in dense LU") is deliberately NOT a
# counted mechanism. Self-time says where the time IS; VALID-MECHANISM requires evidence
# that the candidate REMOVED NO WORK. Conflating them classified `.165` — a 17-19x lever
# killed on a cv gate — as sound, which is the exact failure this file exists to prevent.

# Feasibility / correctness refutations are sound without any timing evidence.
INFEASIBLE_RE = re.compile(
    r"(inexpressible|not expressible|safe Rust cannot|correctness blocker|"
    r"premise was FALSE|premise is false|NOT bit-identical|not bit-identical|"
    r"invalid comparator|measurement artifact|parity (?:break|violat)|dead arm|"
    r"never (?:ran|executed))",
    re.I,
)
CV_ONLY_RE = re.compile(
    r"(CV\s*>\s*5|CV ceiling|CVs? (?:above|exceed|remained above)|"
    r"failed the (?:mandatory )?CV|mandatory CV|all three CVs)",
    re.I,
)
REJECT_HEAD_RE = re.compile(
    r"\b(REJECT|REJECTED|INVALID|NO-SHIP|NEGATIVE RESULT|DEAD END|ABANDON)\b", re.I
)
HEAD_RE = re.compile(r"^## (.+)$")
STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "is", "are",
    "per", "via", "into", "over", "under", "from", "by", "at", "as", "that", "this",
    "reject", "keep", "win", "lever", "perf", "fsci", "candidate", "arm", "run",
}


def entries(path: Path):
    if not path.exists():
        return
    lines = path.read_text(errors="replace").split("\n")
    idx = [i for i, l in enumerate(lines) if HEAD_RE.match(l) or l.startswith("### ")]
    for n, i in enumerate(idx):
        end = idx[n + 1] if n + 1 < len(idx) else len(lines)
        head = lines[i].lstrip("#").strip()
        yield path, i + 1, head, "\n".join(lines[i + 1 : end])


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z_][a-z0-9_]{2,}", text.lower())
    return {w for w in words if w not in STOPWORDS}


# The fleet null band. A ratio outside it in the losing direction is decisive even
# without a recorded null — VOID-NONULL is specifically about NEAR-1.0 ratios, and
# calling a measured 2.5x regression "void" would send the next agent to re-run a
# settled loss.
BAND_LO, BAND_HI = 0.905, 1.105
RATIO_RE = re.compile(r"(\d+\.\d+)\s*[x×]")
REGRESSION_RE = re.compile(r"(regress|slower|SLOWER|no-ship|worse)", re.I)
# A row that says its own effect overlaps the floor is near-1.0 whatever ratios its prose
# happens to quote elsewhere.
# Checked BEFORE the in-floor language, because rows legitimately write sentences like
# "far outside the centered null, so this is NOT an IN-FLOOR result" — matching the
# phrase inside its own denial is how `.165` (17-19x, killed on cv) read as sound.
OUTSIDE_NULL_RE = re.compile(
    r"(far outside the[^.\n]{0,30}null|outside the[^.\n]{0,20}null|clears? the null|"
    r"above null p95|not an IN-FLOOR|is not IN-FLOOR|not in-floor|"
    r"speedup p50 \*\*1[0-9]|inadmissible evidence)",
    re.I,
)
INFLOOR_LANG_RE = re.compile(
    r"(IN-FLOOR|in-floor|overlapping intervals|intervals overlap|within noise|"
    r"inside the null|within the null|NOT DECIDED|did not clear|below the noise floor|"
    r"no gain|zero-gain|~0-gain|median-neutral)",
    re.I,
)


def decisive_ratio(blob: str) -> float | None:
    """The ratio the row's verdict rests on, or None.

    Rows quote many ratios (per-size tables, vs-SciPy context, prior art). Taking the
    one furthest from unity picked up unrelated numbers, so: if the row says in its own
    words that the effect overlaps the floor, report 1.0; otherwise prefer a ratio
    attached to decision language and fall back to the furthest-from-unity value.
    """
    if INFLOOR_LANG_RE.search(blob) and not OUTSIDE_NULL_RE.search(blob):
        return 1.0
    for rx in (
        re.compile(r"paired median[^0-9]{0,20}(\d+\.\d+)", re.I),
        re.compile(r"median[^0-9]{0,15}(\d+\.\d+)\s*[x×]", re.I),
        re.compile(r"centered[^0-9]{0,15}(\d+\.\d+)\s*[x×]", re.I),
    ):
        m = rx.search(blob)
        if m:
            return float(m.group(1))
    vals = [v for v in (float(x) for x in RATIO_RE.findall(blob)) if 0.0 < v < 1e4]
    return max(vals, key=lambda v: abs(v - 1.0)) if vals else None


def verdict_class(head: str, body: str) -> str:
    blob = head + "\n" + body
    if INFEASIBLE_RE.search(blob):
        return "VALID-INFEASIBLE"
    if COUNTED_RE.search(blob):
        return "VALID-MECHANISM"
    r = decisive_ratio(blob)
    if CV_ONLY_RE.search(blob):
        # A cv ceiling is never a sound cause of death. If a null WAS recorded and the
        # effect sits inside it, the row is a legitimate in-floor call that merely also
        # mentions cv; but if the effect is large, cv is what killed it and the row is
        # VOID regardless of the null.
        #
        # DELIBERATE ASYMMETRY: when in doubt this returns VOID-CV. The false positive
        # costs one wasted re-run (`.168` re-ran and came back 1.118x IN-FLOOR). The
        # false negative cost this repo 109x (`.165` recorded a null of 1.002, measured
        # 17-19x, and was rejected on cv anyway). Prefer the cheap error.
        if r is None or r < BAND_LO or r > BAND_HI:
            return "VOID-CV"
    if NULL_RE.search(blob):
        return "VALID-AB"
    if r is not None and (r < BAND_LO or r > BAND_HI) and REGRESSION_RE.search(blob):
        return "VALID-DECISIVE"
    return "VOID-NONULL"


def cmd_propose(text: str, threshold: float) -> int:
    want = tokens(text)
    if len(want) < 2:
        print("preflight: describe the lever in a few words (need >=2 content tokens)")
        return 1
    hits = []
    for path, line, head, body in (e for p in LEDGERS for e in entries(p)):
        overlap = want & tokens(head)
        if not overlap:
            continue
        score = len(overlap) / len(want)
        if score >= threshold:
            hits.append((score, path, line, head, body))
    hits.sort(reverse=True, key=lambda h: h[0])

    if not hits:
        print("preflight: CLEAR — no prior ledger row matches this description.")
        print("  Reminder: a weak match is not a clean bill of health. Grep by hand too.")
        return 0

    blocking = []
    print(f"preflight: {len(hits)} prior ledger row(s) overlap this description\n")
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


def cmd_check_row(path: Path, row: int | None) -> int:
    found = list(entries(path))
    if not found:
        print(f"preflight: no ledger entries parsed from {path}")
        return 1
    if row is None:
        _, line, head, body = found[-1]  # newest entry appended at EOF
    else:
        match = [e for e in found if e[1] == row]
        if not match:
            print(f"preflight: no entry begins at line {row} of {path}")
            return 1
        _, line, head, body = match[0]

    print(f"preflight: checking {path.name}:{line}\n  {head[:160]}")
    if not REJECT_HEAD_RE.search(head):
        print("preflight: CLEAR — not a REJECT row (KEEP rows are gated by the A/B itself).")
        return 0

    cls = verdict_class(head, body)
    has_null = bool(NULL_RE.search(head + body))
    has_counted = bool(COUNTED_RE.search(head + body))
    print(f"  class={cls}  A/A null recorded={has_null}  counted mechanism={has_counted}")

    if cls == "VOID-CV":
        print(
            "\nBLOCKED: this REJECT rests on a `cv` ceiling.\n"
            "  `cv < 5%` is unreachable on this hardware (floor ~12%) and is NOT a\n"
            "  decidability criterion. Decide on the median-CI gate against an A/A null\n"
            "  (docs/OPTIMIZATION_PROTOCOL.md); report cv as provenance only.\n"
            "  Every row killed on cv alone in this repo had a signal that cleared its null."
        )
        return 2
    if cls == "VOID-NONULL":
        print(
            "\nBLOCKED: this REJECT records neither an A/A null control nor a counted\n"
            "  mechanism, so it cannot distinguish the lever from the harness.\n"
            "  Add ONE of:\n"
            "    - an A/A null in the SAME invocation (fsci_conformance::perf_gate::paired), or\n"
            "    - a counted mechanism showing no work was removed (perf stat: instructions,\n"
            "      cycles, minor-faults, syscalls, allocations), or\n"
            "    - a feasibility/correctness refutation, if that is the real reason.\n"
            "  This is the class that voided 30 of this repo's 48 void rows."
        )
        return 2
    print("\npreflight: CLEAR — this REJECT records evidence that can decide it.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--propose", metavar="TEXT", help="describe a lever before writing source")
    g.add_argument("--check-row", metavar="FILE", help="check a ledger file's newest (or --row) entry")
    ap.add_argument("--row", type=int, default=None, help="line number the entry starts at")
    ap.add_argument("--threshold", type=float, default=0.34, help="token-overlap match threshold")
    args = ap.parse_args()

    if args.propose:
        return cmd_propose(args.propose, args.threshold)
    return cmd_check_row(Path(args.check_row), args.row)


if __name__ == "__main__":
    sys.exit(main())
