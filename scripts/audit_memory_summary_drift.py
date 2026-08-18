"""Find memory files whose SUMMARY advertises open work that the BODY says is closed.

WHY. The summary line is what gets read when choosing what to work on; the body is what
gets read after committing to it. When they disagree, the summary wins the decision and
the body wins the argument, so the cost is a wasted turn. Hit twice on 2026-08-18:

  - perf_splu_fill_scaling_elimination_kernel: summary said "107x LOSS ... the wall is the
    Vec<Vec<(usize,f64)>> kernel" and recommended a dense-scatter rewrite that had since
    been MEASURED AND REFUTED. Following it would have burned builds.
  - perf_integrate_stiff_newton_alloc_hoist: description + index said "LIVE AUDIT: BDF
    corrector, kmedoids dmat, hessian_approx" while the body recorded BDF and kmedoids as
    DONE with commit hashes and measured ratios. Cost a turn re-deriving them.

Both were invisible from the summary alone, which is the point: a stale summary reads
exactly like a fresh one.

The check is deliberately CRUDE and reports CANDIDATES, not verdicts -- every hit must be
read. It exists to bound the search, not to replace it.

KNOWN BLIND SPOT, found the hard way 2026-08-18. This compares a file's SUMMARY against
its own BODY, so it cannot see a file that is internally CONSISTENT and externally WRONG.
conformance_fsci_iterative_solver_stubs passed this check cleanly -- index and body both
said "lsmr -> lsqr still open" -- while a cross-reference buried in a DIFFERENT memory
(correctness_absolute_epsilon_on_dimensioned_quantity) said that note was stale, and a
source read settled it: lsmr is a real ~219-line Fong & Saunders Algorithm 6.1
implementation that never calls lsqr.

Agreement between a summary and its body is not evidence that either is true. Cross-file
contradictions need a different probe, and any claim ABOUT SOURCE has to be settled by
reading the source.
"""
import os
import re
import sys

MEM = "/home/ubuntu/.claude/projects/-data-projects-frankenscipy/memory"

# Language that ADVERTISES remaining work in a summary.
OPEN = re.compile(
    r"\bLIVE AUDIT\b|\bAUDIT\b|\bTODO\b|\bNOT yet\b|\bpending\b|\bopen\b|\bnext\b|"
    r"\bunmeasured\b|\bneeds\b|\bawaiting\b|\bre-?try\b",
    re.I,
)
# Language that says the SAME work is finished or dead.
CLOSED = re.compile(
    r"\bDONE\b|\bCLOSED\b|\bREFUTED\b|\bLANDED\b|\bEXHAUSTED\b|\bWITHDRAWN\b|"
    r"\bmeasured and closed\b|\bdo not re-?open\b|\bdo not re-?try\b|\bis dead\b",
    re.I,
)


def frontmatter_description(text):
    m = re.search(r"^---\s*$(.*?)^---\s*$", text, re.M | re.S)
    if not m:
        return ""
    d = re.search(r"^description:\s*(.*)$", m.group(1), re.M)
    return d.group(1).strip().strip('"') if d else ""


def body_of(text):
    parts = re.split(r"^---\s*$", text, maxsplit=2, flags=re.M)
    return parts[2] if len(parts) > 2 else text


def index_lines():
    p = os.path.join(MEM, "MEMORY.md")
    out = {}
    for line in open(p, errors="replace"):
        for fn in re.findall(r"\(([a-zA-Z0-9_]+\.md)\)", line):
            out.setdefault(fn, []).append(line.strip())
    return out


SELFTEST = [
    # (name, description, body, must_flag)
    ("hit-desc-open-body-closed",
     "LIVE AUDIT: foo and bar have the same shape",
     "foo DONE (abc123) 1.09x. bar DONE (def456) 2.4x.", True),
    ("miss-desc-open-body-open",
     "LIVE AUDIT: foo and bar have the same shape",
     "Neither has been looked at. Both remain to be measured.", False),
    ("miss-desc-closed-body-closed",
     "EXHAUSTED: every target closed",
     "foo DONE. bar REFUTED.", False),
]


def flag(desc, index, body):
    summary = (desc or "") + " " + " ".join(index or [])
    return bool(OPEN.search(summary)) and bool(CLOSED.search(body)) and not CLOSED.search(summary)


def selftest():
    ok = True
    for name, desc, body, want in SELFTEST:
        got = flag(desc, [], body)
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name:<30} flagged={got} (want {want})")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    idx = index_lines()
    files = sorted(f for f in os.listdir(MEM) if f.endswith(".md") and f != "MEMORY.md")
    hits = []
    for f in files:
        text = open(os.path.join(MEM, f), errors="replace").read()
        desc = frontmatter_description(text)
        body = body_of(text)
        if flag(desc, idx.get(f, []), body):
            hits.append((f, desc))
    print(f"memory files scanned : {len(files)}")
    print(f"summary/body drift   : {len(hits)}   <- read each; these are CANDIDATES\n")
    for f, desc in hits:
        print(f"  - {f}")
        print(f"      desc: {desc[:150]}")
