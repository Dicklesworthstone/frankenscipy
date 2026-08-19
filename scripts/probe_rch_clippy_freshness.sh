#!/usr/bin/env bash
# frankenscipy-ozg54, the CLIPPY variant -- the gap my build-based probe left open.
#
# ozg54 was a `cargo clippy` run that reported PRE-EDIT line numbers after 65 lines had been
# inserted above them. My first reproducer (scripts/probe_rch_source_freshness.sh) used
# `cargo build` and found 0 stale in 14 iterations, but clippy has its own caching path, so a
# clean build result says nothing about it. This closes that gap by reproducing ozg54's own
# detector -- the line-number shift -- under control.
#
# Per iteration: rewrite the probe source with K padding lines inserted above a function
# carrying a deliberate clippy lint, so the lint's line number is known exactly and moves by
# K. Run clippy remotely, parse the line number it reports, compare to the line the lint
# actually occupies on local disk.
#
#   equal    -> clippy linted the working tree
#   unequal  -> clippy linted a different revision; the delta says how stale
#
# The deliberate lint is INSERTED and removed on EXIT rather than committed: a permanent
# warning in a tracked file would fail anyone running clippy with -D warnings. Editing a
# shared checkout at all is the hazard behind hld7v, so the dirty window is kept short and
# always closed.
#
# Counted, not timed: a line number matches or it does not, identically under any load. No
# build slot needed, which matters while acquire_build_slot is disabled (frankenscipy-fr78g).
#
# SCALE: pass a second argument to bulk the probe file up to a comparable size before
# testing. ozg54's edit was 65 lines inserted into fsci-interpolate's lib.rs, which is 15,476
# lines / 558 KB; the bare probe is 59 lines / 3 KB, a 180x difference. A chunked-transfer or
# fingerprint bug could easily be size-dependent, so a clean result on a 3 KB file does not
# generalise. With bulk lines the lint also sits DEEP in the file, as the original's line
# 2185 and 7065 did, rather than near the top where a partial transfer would still cover it.
#
# Usage: scripts/probe_rch_clippy_freshness.sh [iterations] [bulk_lines]   (default 5, 0)
set -u

ITERATIONS="${1:-5}"
BULK="${2:-0}"
SRC="crates/fsci-sparse/src/bin/probe_rch_source_freshness.rs"
BACKUP="$(mktemp)"

if [ ! -f "$SRC" ]; then
    echo "FAIL: probe source missing at $SRC" >&2
    exit 1
fi
cp "$SRC" "$BACKUP"

restore() {
    cp "$BACKUP" "$SRC"
    rm -f "$BACKUP"
}
trap restore EXIT

stale=0
fresh=0
failures=0

for i in $(seq 1 "$ITERATIONS"); do
    # K varies per iteration so a cached answer from the previous round is wrong by a
    # DIFFERENT amount each time -- a fixed K could coincide with a stale value by luck.
    pad=$(( i * 7 ))

    cp "$BACKUP" "$SRC"
    if [ "$BULK" -gt 0 ]; then
        # Bulk FIRST, so the lint lands deep in a large file rather than in its first
        # kilobyte. Generated per iteration and never committed.
        seq 1 "$BULK" | sed 's|.*|// ozg54 bulk padding line to reach fsci-interpolate scale|' >> "$SRC"
    fi
    {
        for _ in $(seq 1 "$pad"); do echo "// ozg54 clippy padding line"; done
        echo ""
        echo "#[allow(dead_code)]"
        echo "fn ozg54_deliberate_lint(values: &[u8]) -> bool {"
        echo "    values.len() == 0"
        echo "}"
    } >> "$SRC"

    # The line the lint actually occupies right now, read from disk rather than computed,
    # so the expectation cannot drift from the file.
    expected=$(grep -n "values.len() == 0" "$SRC" | head -1 | cut -d: -f1)

    out=$(RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR timeout 600 \
        rch exec -- cargo clippy --release -p fsci-sparse \
        --bin probe_rch_source_freshness --no-deps 2>&1)

    # clippy cites `--> path:line:col`; take the line for our probe file.
    observed=$(printf '%s\n' "$out" \
        | grep -oE "probe_rch_source_freshness\.rs:[0-9]+:[0-9]+" \
        | head -1 | cut -d: -f2)

    if [ -z "$observed" ]; then
        # No diagnostic at all means the detector did not fire; that is a harness failure,
        # not evidence of freshness, and must never be counted as a clean iteration.
        echo "iter=${i} NO_DIAGNOSTIC pad=${pad} expected_line=${expected} (detector did not fire)"
        failures=$((failures + 1))
    elif [ "$observed" = "$expected" ]; then
        echo "iter=${i} FRESH   pad=${pad} line=${observed}"
        fresh=$((fresh + 1))
    else
        echo "iter=${i} *** STALE *** pad=${pad} expected_line=${expected} observed_line=${observed} delta=$((expected - observed))"
        stale=$((stale + 1))
    fi
done

echo
echo "VERDICT iterations=${ITERATIONS} bulk_lines=${BULK} file_bytes=$(wc -c < "$SRC") fresh=${fresh} stale=${stale} no_diagnostic=${failures}"
if [ "$fresh" -eq 0 ]; then
    echo "CONTROL FAILED: no iteration was observed FRESH; this run is VOID, not a negative" >&2
    exit 2
fi
[ "$stale" -eq 0 ]
