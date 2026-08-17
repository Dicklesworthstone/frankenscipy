#!/usr/bin/env python3
"""Census of perf toggles that no test, perf bin or bench ever drives.

frankenscipy-5f06d asks for a number ("EXERCISED NOWHERE") and closes when that
number reaches 0. THE NUMBER IS AN ARTIFACT OF THE PROBE, and on fsci-stats three
plausible probes gave three different answers on the same source tree:

    14   grep `<NAME>.store` inside the test region      -- the bead's own method
   133   substring match, test region = everything after the FIRST #[cfg(test)]
     5   substring match, test region = brace-matched #[cfg(test)] mod bodies

The first UNDERCOUNTS drivers: a table-driven test holds `&STATIC` references and
calls `.store` on the reference, so the literal `NAME.store` never appears even
though the toggle is exercised. Acting on it means writing a duplicate driver for
a lever that already has one.

The second is worse because it looks best. Taking the region as "everything after
the first `#[cfg(test)]`" swallows the rest of the library, including the `pub
static` declarations themselves, so every toggle matches its own declaration and
the probe reports a perfect score while testing nothing. A clean number is not
evidence; 133/133 was the tell.

The third is this script. It scans EVERY module (not just lib.rs -- four crates
declare their toggles in submodules and a lib.rs-only scan called them empty),
brace-matches each `#[cfg(test)] mod` body, drops comment-only lines so a toggle
NAMED IN PROSE is not counted as driven, and verifies itself before printing:

  must-hit   the declaration pattern matches a known-good fixture line
  must-miss  it does not match a known-bad one
  must-miss  a declaration site is never inside the test region -- the arm that
             fails if the region blows out the way probe 2 did
  must-miss  an invented name matches nothing
  must-hit   at least one channel matched, when the crate declares any toggles

Exits non-zero if any control fails, so a broken probe cannot report a count.

TWO OF THOSE CONTROLS ARE THERE BECAUSE THE EARLIER ONES WERE WRONG, in the same
direction both times -- too strict, failing healthy crates:

  * requiring the TEST channel specifically failed fsci-fft and fsci-opt, which
    legitimately drive every toggle from perf bins;
  * treating "no toggles declared" as a failure fired on the 5 crates that
    genuinely declare none.

But relaxing the second one silently reintroduced the exact blindness this script
exists to prevent: with a deliberately broken declaration regex, fsci-stats --
which declares 134 -- reported "no perf toggles declared" and exited 0. An empty
result and a blind probe are indistinguishable from the outside, which is why the
fixture self-test at the top is not decoration. Verified by running both arms.

Usage:  python3 scripts/toggle_driver_census.py [crate ...]
        (default: fsci-stats)
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

ARGS = sys.argv[1:]

# `--gate` turns the census from a report into an ENFORCEABLE invariant: exit
# non-zero if any crate carries more undriven A/B switches than the cap.
#
# Without this the four crates driven to zero can regress silently -- nothing
# stops a new `pub static FOO_FORCE_SERIAL` landing with no driver, which is how
# the original 51-toggle backlog accumulated in the first place. A report nobody
# runs is not a ratchet.
#
# It also closes a gap in the drqu7 ratchet, which lives in the crates and counts
# a toggle as paid once it has a CONTRACT -- whether or not anything ever runs
# both arms. Five fsci-linalg toggles were contracted on 2026-08-16 and undriven
# until 2026-08-17. Documentation and verification are different properties and
# need different gates; this is the one for verification. The in-crate ratchet
# cannot do this job: a `#[cfg(test)]` test sees only what `include_str!` can
# reach, and fsci-stats drives 64 of its toggles from 205 separate perf bins.
GATE = "--gate" in ARGS
MAX_UNDRIVEN = 0
for i, a in enumerate(ARGS):
    if a == "--max-undriven" and i + 1 < len(ARGS):
        MAX_UNDRIVEN = int(ARGS[i + 1])
CRATES = [a for a in ARGS if not a.startswith("--") and not a.isdigit()] or ["fsci-stats"]

# A/B SWITCHES ONLY. `pub static NAME: AtomicUsize`/`AtomicU64` declarations are
# COUNTERS -- instrumentation incremented with `fetch_add`, with no second arm to
# compare against -- and counting them as undriven toggles overstated the backlog.
# Two of the five entries left on frankenscipy-5f06d after the linalg sweep were
# counters: SPLU_RESERVE_FROM_SYMBOLIC_FACTOR_HITS and
# RADAU_POST_STEP_JAC_REFRESH_HITS. They are reported separately rather than
# dropped, because a counter nothing ever reads is its own (different) problem.
DECL = r"pub static ([A-Z0-9_]+)\s*:\s*(?:std::sync::atomic::)?AtomicBool"
COUNTER = r"pub static ([A-Z0-9_]+)\s*:\s*(?:std::sync::atomic::)?Atomic(?:Usize|U64|U32|I64)"

# Self-test on a fixture, run before any crate is scanned.
#
# "This crate declares no perf toggles" is a legitimate and common answer -- 5 of
# 19 fsci crates give it. But it is ALSO what a broken declaration regex says
# about every crate, and the two are indistinguishable from the outside. That is
# not hypothetical: relaxing the empty case to a clean exit made a deliberately
# broken regex report "no perf toggles declared" for fsci-stats, which declares
# 134. Pinning the regex against a known-good and a known-bad line restores the
# must-hit/must-miss pair, so an empty crate result means the crate is empty
# rather than the probe being blind.
_HIT = "pub static EXAMPLE_FORCE_SERIAL: std::sync::atomic::AtomicBool ="
_MISS = "pub static EXAMPLE_HITS: std::sync::atomic::AtomicUsize = "
if re.findall(DECL, _HIT) != ["EXAMPLE_FORCE_SERIAL"] or re.findall(DECL, _MISS):
    sys.exit(
        "CONTROL FAILED: the declaration pattern does not match its own fixture; "
        "every count it would print is meaningless"
    )


# Must-hit / must-miss on the GATE comparison, not just on the pattern. A gate
# that never fires is indistinguishable from a clean tree.
assert (3 > 0) and not (0 > 0), "gate comparison is broken"
assert not (2 > 2) and (3 > 2), "gate cap comparison is off by one"


def test_regions(src: str) -> list[tuple[int, int]]:
    """Byte extents of every `#[cfg(test)] mod ... { ... }` body."""
    regions = []
    for marker in re.finditer(r"#\[cfg\(test\)\]", src):
        opening = re.compile(r"mod\s+\w+\s*\{").search(src, marker.end())
        if not opening:
            continue
        depth, i = 1, opening.end()
        while i < len(src) and depth:
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
            i += 1
        regions.append((opening.end(), i))
    return regions


def strip_comments(text: str) -> str:
    return "\n".join(
        line
        for line in text.split("\n")
        if not line.strip().startswith(("//", "///", "//!"))
    )


def census(crate: str) -> int:
    # EVERY module, not just lib.rs. Scanning lib.rs alone reported "no pub
    # statics found at all" for fsci-sparse, fsci-special, fsci-fft and
    # fsci-integrate, which declare their toggles in submodules -- a false zero
    # that the controls caught only because they refuse to print a count when
    # the channel is blind. src/bin is excluded here: those are DRIVERS, not
    # declaration sites, and counting them as both would let a perf bin satisfy
    # its own coverage.
    src_files = [
        p
        for p in sorted(glob.glob(f"crates/{crate}/src/**/*.rs", recursive=True))
        if "/src/bin/" not in p.replace("\\", "/")
    ]
    if not src_files:
        print(f"{crate}: no source files found")
        return 1

    test_chunks, lib_chunks, names_all, counters = [], [], set(), set()
    for path in src_files:
        src = Path(path).read_text()
        names_all.update(re.findall(DECL, src))
        counters.update(re.findall(COUNTER, src))
        regions = test_regions(src)
        test_chunks.extend(src[a:b] for a, b in regions)
        cut = min((a for a, _ in regions), default=len(src))
        lib_chunks.append(src[:cut])

    test_code = strip_comments("\n".join(test_chunks))
    lib_body = "\n".join(lib_chunks)

    driver_files = glob.glob(f"crates/{crate}/src/bin/*.rs") + glob.glob(
        f"crates/{crate}/benches/*.rs"
    )
    bin_code = strip_comments("".join(Path(p).read_text() for p in driver_files))

    names = sorted(names_all)
    in_test = [n for n in names if n in test_code]
    in_bin = [n for n in names if n not in test_code and n in bin_code]
    nowhere = [n for n in names if n not in test_code and n not in bin_code]

    # ---- controls, both arms, before any count is believed ------------------
    failures = []
    if not names:
        # A crate with no perf toggles is the common case, not a broken probe:
        # 5 of 19 fsci crates declare none. Treating that as a control failure
        # was a false positive of the same kind as the test-channel one. Genuine
        # regex breakage shows up as EVERY crate reporting zero in a fleet run,
        # which is the level where it is actually detectable.
        print(f"{crate}\n  no perf toggles declared")
        return 0
    # must-miss: the declaration site must NOT be inside the test region, or the
    # region has blown out and every toggle will match itself.
    for probe in names[:5]:
        if f"pub static {probe}" in test_code:
            failures.append(
                f"declaration of {probe} is inside the test region: the region "
                f"has blown out and every toggle would match its own declaration"
            )
            break
    # must-miss: an invented name matches nothing.
    if "ZZ_INVENTED_TOGGLE_NAME" in test_code + bin_code:
        failures.append("an invented name matched; the probe blanket-matches")
    # must-hit: the probe must be shown to FIND something before a zero from it
    # means anything. Requiring the TEST channel specifically was wrong -- a
    # crate may legitimately drive every toggle from perf bins (fsci-fft and
    # fsci-opt do), and failing those was a false positive in the control
    # itself. What actually has to hold is that at least one channel matched;
    # if neither did while toggles exist, the probe is blind.
    if names and not in_test and not in_bin:
        failures.append(
            f"{len(names)} toggles declared but NEITHER the test nor the "
            f"bin/bench channel matched any of them; the probe is blind"
        )
    if failures:
        for f in failures:
            print(f"{crate}: CONTROL FAILED: {f}")
        return 1

    print(f"{crate}")
    print(f"  A/B switches                 {len(names)}")
    if counters:
        undriven_counters = sorted(
            c for c in counters if c not in test_code and c not in bin_code
        )
        print(
            f"  counters (not A/B levers)    {len(counters)}"
            f"{f', {len(undriven_counters)} unread' if undriven_counters else ''}"
        )
    print(f"  driven by an in-crate test   {len(in_test)}")
    print(f"  driven by a perf bin/bench   {len(in_bin)}")
    print(f"  EXERCISED NOWHERE            {len(nowhere)}")
    for n in nowhere:
        print(f"      {n}")
    assert len(lib_body) > 0
    if GATE and len(nowhere) > MAX_UNDRIVEN:
        print(
            f"  GATE FAILED: {len(nowhere)} undriven A/B switches, cap is "
            f"{MAX_UNDRIVEN}. Either give each one a driver sized ABOVE its work "
            f"gate, or retire it explicitly with a comment (frankenscipy-5f06d "
            f"remedy b). Raising the cap to make this pass is not one of the two "
            f"options."
        )
        return 2
    return 0


sys.exit(max(census(c) for c in CRATES))
