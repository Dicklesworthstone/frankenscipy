#!/usr/bin/env bash
# frankenscipy-ozg54, the CONCURRENCY variant -- the last untested variable.
#
# ozg54 happened while several agents wrote the checkout continuously, and "rsync raciness
# against concurrent writes" is the hypothesis that condition would be needed to test. My
# earlier probes changed the file only BETWEEN invocations, so they could not have exercised
# it. This rewrites the same path continuously WHILE a remote build is in flight.
#
# ## The expectation has to be defined carefully, or this test is meaningless
#
# Under concurrent writes there is no single "correct" answer: a transfer may legitimately
# capture any revision that existed during its window, and which one it wins is a race, not a
# defect. What is NEVER legitimate is compiling a revision from BEFORE the command was
# issued. So:
#
#   generation N0 is written and fsync'd, THEN the build is launched
#   generations N0+1, N0+2, ... are written continuously while it runs
#
#   observed >= N0  -> VALID   (the build saw the tree as of launch, or fresher)
#   observed <  N0  -> STALE   (it compiled a revision that predates the invocation)
#
# That criterion is the reason this probe can report a negative at all. A naive
# "observed == last written" check would flag every ordinary race as a defect and produce a
# stream of false positives.
#
# Writes use write-to-temp + atomic rename. A partially-written source file would be MY race,
# not rch's, and would show up as a compile error rather than staleness -- testing my own bug
# instead of the one under investigation.
#
# Counted, not timed. No build slot needed (acquire_build_slot is disabled, frankenscipy-fr78g).
#
# Usage: scripts/probe_rch_concurrent_freshness.sh [iterations]   (default 4)
set -u

ITERATIONS="${1:-4}"
SRC="crates/fsci-sparse/src/bin/probe_rch_source_freshness.rs"
BIN="./target/release/probe_rch_source_freshness"
BACKUP="$(mktemp)"
ORIGINAL_MARKER="ozg54-baseline"

[ -f "$SRC" ] || { echo "FAIL: probe source missing at $SRC" >&2; exit 1; }
cp "$SRC" "$BACKUP"

CHURN_PID=""
cleanup() {
    [ -n "$CHURN_PID" ] && kill "$CHURN_PID" 2>/dev/null
    cp "$BACKUP" "$SRC"
    rm -f "$BACKUP"
}
trap cleanup EXIT

write_generation() {
    # Atomic: build the new content in a temp file on the same filesystem, then rename.
    local gen="$1" tmp
    tmp="$(mktemp "${SRC}.XXXXXX")"
    sed "s/^const MARKER: &str = \".*\";/const MARKER: \&str = \"ozg54-gen-${gen}\";/" \
        "$BACKUP" > "$tmp"
    mv -f "$tmp" "$SRC"
}

stale=0
valid=0
failures=0

for i in $(seq 1 "$ITERATIONS"); do
    base=$(( i * 1000 ))
    write_generation "$base"
    sync

    # Churn the SAME path while the build runs. Generations strictly increase so the
    # observed marker locates itself unambiguously on the timeline.
    (
        gen=$(( base + 1 ))
        while :; do
            write_generation "$gen"
            gen=$(( gen + 1 ))
            sleep 0.05
        done
    ) &
    CHURN_PID=$!

    build_ok=0
    if RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR timeout 600 \
        rch exec -- cargo build --release -p fsci-sparse \
        --bin probe_rch_source_freshness >/dev/null 2>&1; then
        build_ok=1
    fi

    kill "$CHURN_PID" 2>/dev/null
    wait "$CHURN_PID" 2>/dev/null
    CHURN_PID=""

    if [ "$build_ok" -ne 1 ]; then
        echo "iter=${i} BUILD_FAILED base_gen=${base} (loud, not a staleness finding)"
        failures=$((failures + 1))
        continue
    fi

    observed=$("$BIN" 2>/dev/null | sed -n 's/^marker=ozg54-gen-//p')
    if [ -z "$observed" ]; then
        echo "iter=${i} NO_MARKER base_gen=${base} (detector did not fire)"
        failures=$((failures + 1))
    elif [ "$observed" -ge "$base" ]; then
        echo "iter=${i} VALID   base_gen=${base} observed_gen=${observed} (ahead by $((observed - base)))"
        valid=$((valid + 1))
    else
        echo "iter=${i} *** STALE *** base_gen=${base} observed_gen=${observed} predates launch by $((base - observed))"
        stale=$((stale + 1))
    fi
done

echo
echo "VERDICT iterations=${ITERATIONS} valid=${valid} stale=${stale} failures=${failures}"
if [ "$valid" -eq 0 ]; then
    echo "CONTROL FAILED: no iteration was observed VALID; this run is VOID, not a negative" >&2
    exit 2
fi
[ "$stale" -eq 0 ]
