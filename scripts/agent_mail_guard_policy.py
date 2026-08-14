"""Fail-closed policy for the installed Agent Mail reservation hooks."""

import os
import sys


def is_truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() in {"1", "true", "t", "yes", "y"}


def main() -> int:
    if is_truthy(os.environ.get("AGENT_MAIL_BYPASS")):
        print(
            "mcp-agent-mail: AGENT_MAIL_BYPASS is forbidden; reservation enforcement is fail-closed.",
            file=sys.stderr,
        )
        return 2
    if "FILE_RESERVATIONS_ENFORCEMENT_ENABLED" in os.environ and not is_truthy(
        os.environ["FILE_RESERVATIONS_ENFORCEMENT_ENABLED"]
    ):
        print(
            "mcp-agent-mail: disabling reservation enforcement is forbidden; reservation enforcement is fail-closed.",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("AGENT_MAIL_GUARD_MODE", "block") != "block":
        print(
            "mcp-agent-mail: AGENT_MAIL_GUARD_MODE must be block; reservation enforcement is fail-closed.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
