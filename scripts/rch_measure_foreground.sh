#!/usr/bin/env bash
# Run one performance measurement visibly and fail closed when RCH refuses to
# schedule it.  Do not redirect this wrapper: the terminal is the artifact.
set -o pipefail

readonly refusal_pattern='no (admissible )?workers|remote build admission is paused|all workers failed preflight checks'

is_rch_scheduling_refusal() {
    grep -Eiq "$refusal_pattern"
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
        2>&1 | tee /dev/stderr | is_rch_scheduling_refusal
    statuses=("${PIPESTATUS[@]}")
    set -e

    if (( statuses[2] == 0 )); then
        echo "BLOCKED: RCH scheduling refused the run; no measurement was run." >&2
        return 75
    fi

    return "${statuses[0]}"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    main "$@"
fi
