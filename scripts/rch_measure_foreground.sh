#!/usr/bin/env bash
# Run one performance measurement visibly and fail closed when RCH refuses to
# schedule it.  Do not redirect this wrapper: the terminal is the artifact.
set -o pipefail

report_rch_scheduling_refusal() {
    local transcript
    transcript=$(cat)

    if grep -Eiq 'remote build admission is paused' <<<"$transcript"; then
        echo "BLOCKED: RCH admission is paused for daemon remediation; no measurement was run." >&2
        return 0
    fi
    if grep -Eiq 'all workers failed preflight checks' <<<"$transcript"; then
        echo "BLOCKED: all RCH workers failed preflight; no measurement was run." >&2
        return 0
    fi
    if grep -Eiq 'no (admissible )?workers' <<<"$transcript"; then
        echo "BLOCKED: RCH has no admissible workers; no measurement was run." >&2
        return 0
    fi
    return 1
}

main() {
    if (( $# == 0 )); then
        echo "usage: $0 -- <cargo command and arguments>" >&2
        return 64
    fi

    if [[ $1 == -- ]]; then
        shift
    fi

    if (( $# == 0 )); then
        echo "usage: $0 -- <cargo command and arguments>" >&2
        return 64
    fi

    local rch_bin=${RCH_BIN:-rch}
    local -a statuses
    set +e
    RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR "$rch_bin" exec --base HEAD --clean-overlay --no-overlay -- "$@" \
        2>&1 | tee /dev/stderr | report_rch_scheduling_refusal
    statuses=("${PIPESTATUS[@]}")
    set -e

    if (( statuses[2] == 0 )); then
        return 75
    fi

    return "${statuses[0]}"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
