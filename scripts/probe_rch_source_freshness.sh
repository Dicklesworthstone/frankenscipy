#!/usr/bin/env bash
# frankenscipy-ozg54 reproducer: does `rch exec --` ever compile a STALE working tree?
#
# ozg54 observed a clippy run reporting pre-edit line numbers after 65 lines were inserted
# above them, then correct locations on the next identical invocation with no local edit
# between. Cause NOT DIAGNOSED. This drives the question directly instead of waiting for the
# symptom to recur.
#
# Each iteration rewrites the probe's MARKER to a value no previous build could have seen,
# records the local SHA-256 of the probe source, builds it remotely through plain
# `rch exec --` (the exact form ozg54 used -- NOT --base/--no-overlay, which legitimately
# builds the committed tree and would be a different question), and asks the built binary
# for the SHA-256 of the source the COMPILER actually read, captured via include_bytes!.
#
# Equal    -> the remote build saw the working tree.
# Unequal  -> it compiled something else, and the reported marker names which revision.
#
# This is a COUNT, not a timing: it reads identically on an idle or saturated host, needs no
# build slot, and cannot be invalidated by load. That matters because acquire_build_slot is
# disabled fleet-wide (frankenscipy-fr78g).
#
# Usage: scripts/probe_rch_source_freshness.sh [iterations]   (default 6)
set -u

ITERATIONS="${1:-6}"
SRC="crates/fsci-sparse/src/bin/probe_rch_source_freshness.rs"
BIN="./target/release/probe_rch_source_freshness"
ORIGINAL_MARKER="ozg54-baseline"

if [ ! -f "$SRC" ]; then
    echo "FAIL: probe source missing at $SRC" >&2
    exit 1
fi

restore() {
    # Leave the tree exactly as found. The probe edits a source file in a SHARED checkout,
    # which is the very hazard that produced frankenscipy-hld7v -- an in-flight control edit
    # captured by another agent's commit. Keep the dirty window short and always restore.
    sed -i "s/^const MARKER: &str = \".*\";/const MARKER: \&str = \"${ORIGINAL_MARKER}\";/" "$SRC"
}
trap restore EXIT

stale=0
fresh=0
build_failures=0

for i in $(seq 1 "$ITERATIONS"); do
    marker="ozg54-iter-${i}-$$"
    sed -i "s/^const MARKER: &str = \".*\";/const MARKER: \&str = \"${marker}\";/" "$SRC"
    local_sha=$(sha256sum "$SRC" | cut -d' ' -f1)

    if ! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR timeout 600 \
        rch exec -- cargo build --release -p fsci-sparse \
        --bin probe_rch_source_freshness >/dev/null 2>&1; then
        echo "iter=${i} BUILD_FAILED"
        build_failures=$((build_failures + 1))
        continue
    fi

    out=$("$BIN" 2>&1)
    got_marker=$(printf '%s\n' "$out" | sed -n 's/^marker=//p')
    got_sha=$(printf '%s\n' "$out" | sed -n 's/^compiled_source_sha256=//p')

    if [ "$got_sha" = "$local_sha" ] && [ "$got_marker" = "$marker" ]; then
        echo "iter=${i} FRESH   marker=${got_marker}"
        fresh=$((fresh + 1))
    else
        # The interesting outcome. Report BOTH what was expected and what arrived, because
        # the stale marker identifies which earlier revision was served.
        echo "iter=${i} *** STALE *** expected_marker=${marker} got_marker=${got_marker}"
        echo "                        expected_sha=${local_sha}"
        echo "                        got_sha=${got_sha}"
        stale=$((stale + 1))
    fi
done

echo
echo "VERDICT iterations=${ITERATIONS} fresh=${fresh} stale=${stale} build_failures=${build_failures}"
# A run with zero fresh iterations proves nothing about staleness -- it means the probe never
# worked. Require at least one observed FRESH before reporting a clean result, so a broken
# harness cannot masquerade as a negative finding.
if [ "$fresh" -eq 0 ]; then
    echo "CONTROL FAILED: no iteration was observed FRESH; this run is VOID, not a negative" >&2
    exit 2
fi
[ "$stale" -eq 0 ]
