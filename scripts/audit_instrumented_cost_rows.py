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

text = open(sys.argv[1], errors="replace").read()
rows = re.split(r"\n(?=## )", text)

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

flagged = []
for row in rows:
    if not row.startswith("## "):
        continue
    head = row.split("\n", 1)[0][3:]
    if HARNESS.search(row) and COST.search(row) and not EXEMPT.search(row):
        flagged.append(head)

print(f"rows examined                     : {sum(1 for r in rows if r.startswith('## ')):>4}")
print(f"rows citing a cargo-test harness  : {sum(1 for r in rows if r.startswith('## ') and HARNESS.search(r)):>4}")
print(f"of those, reporting a COST figure : {len(flagged):>4}   <- these need a bias check")
for head in flagged:
    print(f"  - {head[:132]}")
