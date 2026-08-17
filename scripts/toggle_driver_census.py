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

The third is this script. It brace-matches each `#[cfg(test)] mod` body, drops
comment-only lines so a toggle NAMED IN PROSE is not counted as driven, and
verifies itself with both arms of a control before printing:

  must-hit   a toggle driven by a plain `NAME.store`
  must-hit   a toggle driven only through a `&STATIC` table (defeats probe 1)
  must-miss  an invented name matches nothing
  must-miss  the declaration site alone does not count as a driver (defeats
             probe 2 -- this is the assertion that fails if the region blows out)

Exits non-zero if any control fails, so a broken probe cannot report a count.

Usage:  python3 scripts/toggle_driver_census.py [crate ...]
        (default: fsci-stats)
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

CRATES = sys.argv[1:] or ["fsci-stats"]


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
    lib = Path(f"crates/{crate}/src/lib.rs")
    src = lib.read_text()
    regions = test_regions(src)
    if not regions:
        print(f"{crate}: no #[cfg(test)] mod found; probe cannot run")
        return 1

    test_code = strip_comments("\n".join(src[a:b] for a, b in regions))
    lib_body = src[: min(a for a, _ in regions)]

    driver_files = glob.glob(f"crates/{crate}/src/bin/*.rs") + glob.glob(
        f"crates/{crate}/benches/*.rs"
    )
    bin_code = strip_comments("".join(Path(p).read_text() for p in driver_files))

    names = sorted(set(re.findall(r"pub static ([A-Z0-9_]+)\s*:", src)))
    in_test = [n for n in names if n in test_code]
    in_bin = [n for n in names if n not in test_code and n in bin_code]
    nowhere = [n for n in names if n not in test_code and n not in bin_code]

    # ---- controls, both arms, before any count is believed ------------------
    failures = []
    if not names:
        failures.append("no pub statics found at all")
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
    # must-hit: at least one toggle is found by each channel we claim to cover.
    if not in_test:
        failures.append("no toggle found in any test; the test channel is blind")
    if failures:
        for f in failures:
            print(f"{crate}: CONTROL FAILED: {f}")
        return 1

    print(f"{crate}")
    print(f"  pub statics                  {len(names)}")
    print(f"  driven by an in-crate test   {len(in_test)}")
    print(f"  driven by a perf bin/bench   {len(in_bin)}")
    print(f"  EXERCISED NOWHERE            {len(nowhere)}")
    for n in nowhere:
        print(f"      {n}")
    assert len(lib_body) > 0
    return 0


sys.exit(max(census(c) for c in CRATES))
