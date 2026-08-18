"""Flag ledger rows whose COST figures were taken on a `cargo test` binary.

WHY (frankenscipy-llywn, 2026-08-17). `cfg(test)` instrumentation sits inside this
project's hot paths and acts as an optimisation barrier: it inflated one measured effect
34-fold, forcing three cost rows to be withdrawn, and it was later found under the
supernodal closure's 15.06/15.00 pair. Structural counts (run lengths, call counts, drop
counts) are unaffected -- barriers do not change how often a Vec outgrows its capacity.
COST figures are.

So the audit is not "which rows mention cargo test" but "which rows report an
instruction-or-cycle cost AND name a cargo-test harness".
"""
import re
import sys

def detect_row_level(text):
    """Ledgers disagree on which heading level is a ROW.

    NEGATIVE_EVIDENCE.md uses `## ` (1154 rows, 10 `### ` subheads); perf_ledger_cc.md
    uses `### ` (288 rows) with `## ` reserved for sections. Splitting the latter on
    `## ` yielded 44 "rows" of ~22KB each -- the probe was not wrong, it was BLIND, and
    a blind probe prints a clean number. Pick the level that actually enumerates rows.
    """
    counts = {n: len(re.findall(r"^#{%d} " % n, text, re.M)) for n in (2, 3, 4)}
    level = max(counts, key=lambda n: counts[n])
    return level, counts


def split_rows(text, level):
    marker = "#" * level
    return re.split(r"\n(?=%s )" % marker, text)

HARNESS = re.compile(r"cargo test", re.I)
COST = re.compile(
    r"\b\d[\d,]{4,}\s*(?:Ir\b|instructions?\b)"      # a big instruction count
    r"|\bIr\s*(?:/|per)\s*element"                    # Ir per element-update
    r"|\binstructions? per\b"
    r"|\bIr\b[^.\n]{0,40}\bvs\b",
    re.I,
)
# Rows that explicitly say the measurement was NOT on a test binary, or that are
# about the hazard itself, are not defects.
EXEMPT = re.compile(
    r"no `?cfg\(test\)`? code in the measured region"
    r"|shipping[- ]profile binary"
    r"|not transferable"
    r"|were instrumented",
    re.I,
)

def audit(text, level=None):
    detected, counts = detect_row_level(text)
    level = level or detected
    marker = "#" * level + " "
    rows = [r for r in split_rows(text, level) if r.startswith(marker)]
    flagged, harnessed = [], 0
    for row in rows:
        head = row.split("\n", 1)[0][level + 1:]
        if HARNESS.search(row):
            harnessed += 1
            if COST.search(row) and not EXEMPT.search(row):
                flagged.append(head)
    return level, counts, len(rows), harnessed, flagged


SELFTEST = [
    # (dialect level, text, must_flag)
    (2, "## hit\n\nUnder `cargo test -p x`: 3,510,334 Ir on the merge.\n", True),
    (2, "## miss-no-harness\n\nShipping-profile binary: 2,650,477 instructions.\n", False),
    (2, "## miss-structural\n\nUnder `cargo test -p x` the run length was 7, 412 drops.\n", False),
    (3, "### hit\n\nUnder `cargo test -p x`: 3,510,334 Ir on the merge.\n", True),
    (3, "### miss-no-harness\n\nShipping-profile binary: 2,650,477 instructions.\n", False),
]


def selftest():
    ok = True
    for level, text, must_flag in SELFTEST:
        # pad so the intended level wins detection
        padded = text + ("\n" + "#" * level + " filler\n") * 3
        _, _, _, _, flagged = audit(padded, level=level)
        got = bool(flagged)
        status = "ok" if got == must_flag else "FAIL"
        if got != must_flag:
            ok = False
        print(f"  [{status}] h{level} expect_flag={must_flag} got={got}  {text.splitlines()[0]}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = [a for a in sys.argv[1:] if a.startswith("--")]
    if "--selftest" in opts:
        sys.exit(selftest())
    override = None
    for o in opts:
        if o.startswith("--level="):
            override = int(o.split("=", 1)[1])
    text = open(args[0], errors="replace").read()
    level, counts, n_rows, harnessed, flagged = audit(text, override)
    print(f"row heading level detected        : h{level}   (counts {counts})")
    print(f"rows examined                     : {n_rows:>4}")
    print(f"rows citing a cargo-test harness  : {harnessed:>4}")
    print(f"of those, reporting a COST figure : {len(flagged):>4}   <- these need a bias check")
    for head in flagged:
        print(f"  - {head[:132]}")
