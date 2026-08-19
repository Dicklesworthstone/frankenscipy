#!/usr/bin/env python3
"""Closing evidence for frankenscipy-hld7v: the reservation guard's bypass routes are shut.

hld7v recorded that an auto-commit swept an in-flight negative control -- a deliberately
wrong golden constant, transited through on purpose by the project's own two-arm
verification standard -- and pushed a RED value to main as 21c11204f. The root-cause
narrowing found the mechanism: `50-agent-mail.py`'s `main()` had two early exits that
returned 0 SILENTLY, one on `AGENT_MAIL_BYPASS` and one on a falsy
`FILE_RESERVATIONS_ENFORCEMENT_ENABLED`. As that comment put it, "a bypass that leaves no
trace is indistinguishable from a guard that ran and approved."

Both bypasses are now gone from the hook and inverted by `scripts/agent_mail_guard_policy.py`,
which the hook invokes as a fail-closed precondition before any reservation check. This
script is the re-runnable proof of that, because "I read the source and it looks right" is
not the standard this repo holds anything else to.

BOTH ARMS, and the must-miss half is the one that matters most here. A policy that returned 2
unconditionally would pass every must-hit below while breaking every commit in the repo, so
the clean-environment and correctly-set cases must be shown to return 0 SILENTLY -- otherwise
this script would be reporting "the guard is strict" about a guard that is merely broken.

Exit 0 if every arm behaves; 1 otherwise, naming the arm that failed.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

POLICY = pathlib.Path(__file__).resolve().parent / "agent_mail_guard_policy.py"

# (label, env overrides, expected exit, expect_stderr)
#
# The truthy spellings are enumerated rather than sampled: `is_truthy` accepts
# 1/true/t/yes/y, and a policy that only caught "1" would leave "yes" as a live bypass.
ARMS = [
    ("clean environment", {}, 0, False),
    ("AGENT_MAIL_BYPASS=1", {"AGENT_MAIL_BYPASS": "1"}, 2, True),
    ("AGENT_MAIL_BYPASS=true", {"AGENT_MAIL_BYPASS": "true"}, 2, True),
    ("AGENT_MAIL_BYPASS=yes", {"AGENT_MAIL_BYPASS": "yes"}, 2, True),
    ("AGENT_MAIL_BYPASS=y", {"AGENT_MAIL_BYPASS": "y"}, 2, True),
    ("AGENT_MAIL_BYPASS=t", {"AGENT_MAIL_BYPASS": "t"}, 2, True),
    # Falsy/empty must NOT trip it: an agent with the variable present but unset is not
    # attempting a bypass, and blocking them would train people to unset it blindly.
    ("AGENT_MAIL_BYPASS=0", {"AGENT_MAIL_BYPASS": "0"}, 0, False),
    ("AGENT_MAIL_BYPASS=<empty>", {"AGENT_MAIL_BYPASS": ""}, 0, False),
    ("FILE_RESERVATIONS_ENFORCEMENT_ENABLED=0", {"FILE_RESERVATIONS_ENFORCEMENT_ENABLED": "0"}, 2, True),
    ("FILE_RESERVATIONS_ENFORCEMENT_ENABLED=false", {"FILE_RESERVATIONS_ENFORCEMENT_ENABLED": "false"}, 2, True),
    ("FILE_RESERVATIONS_ENFORCEMENT_ENABLED=1", {"FILE_RESERVATIONS_ENFORCEMENT_ENABLED": "1"}, 0, False),
    ("AGENT_MAIL_GUARD_MODE=warn", {"AGENT_MAIL_GUARD_MODE": "warn"}, 2, True),
    ("AGENT_MAIL_GUARD_MODE=off", {"AGENT_MAIL_GUARD_MODE": "off"}, 2, True),
    ("AGENT_MAIL_GUARD_MODE=block", {"AGENT_MAIL_GUARD_MODE": "block"}, 0, False),
]

SCRUB = ("AGENT_MAIL_BYPASS", "FILE_RESERVATIONS_ENFORCEMENT_ENABLED", "AGENT_MAIL_GUARD_MODE")


def main() -> int:
    if not POLICY.is_file():
        print(f"FAIL: policy script missing at {POLICY}", file=sys.stderr)
        return 1

    failures = []
    for label, overrides, expected_exit, expect_stderr in ARMS:
        env = {k: v for k, v in os.environ.items() if k not in SCRUB}
        env.update(overrides)
        done = subprocess.run(
            [sys.executable, str(POLICY)], env=env, capture_output=True, text=True, check=False
        )
        got_stderr = bool(done.stderr.strip())
        ok = done.returncode == expected_exit and got_stderr == expect_stderr
        print(
            f"{'ok  ' if ok else 'FAIL'}  {label:44s} exit={done.returncode} "
            f"stderr={'yes' if got_stderr else 'no ':3s} "
            f"(want exit={expected_exit} stderr={'yes' if expect_stderr else 'no'})"
        )
        if not ok:
            failures.append(label)

    # A refusal must SAY SO. This is the specific property hld7v asked for -- a silent
    # exit 0 was indistinguishable from an approval, so every blocking arm is required to
    # leave a trace on stderr, which the per-arm check above already enforces.
    if failures:
        print(f"\nFAILED arms: {failures}", file=sys.stderr)
        return 1
    print(f"\nall {len(ARMS)} arms behaved; bypass routes are shut and refusals are audible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
