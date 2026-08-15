#!/usr/bin/env bash
# Run one performance measurement visibly and fail closed when RCH refuses to
# schedule it.  Do not redirect this wrapper: the terminal is the artifact.
set -o pipefail

if (( $# == 0 )); then
    echo "usage: $0 -- <cargo command and arguments>" >&2
    exit 64
fi

if [[ $1 == -- ]]; then
    shift
fi

if (( $# == 0 )); then
    echo "usage: $0 -- <cargo command and arguments>" >&2
    exit 64
fi

rch_bin=${RCH_BIN:-rch}
set +e
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR "$rch_bin" exec --base HEAD --clean-overlay --no-overlay -- "$@" \
    2>&1 | tee /dev/stderr | grep -Eiq 'no (admissible )?workers|no admissible workers'
statuses=("${PIPESTATUS[@]}")
set -e

if (( statuses[2] == 0 )); then
    echo "BLOCKED: RCH reported no workers; no measurement was run." >&2
    exit 75
fi

exit "${statuses[0]}"
