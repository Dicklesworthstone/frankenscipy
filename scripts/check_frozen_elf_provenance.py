#!/usr/bin/env python3
"""frankenscipy-y9bzw item 3: are cited frozen-ELF artifacts locatable, and on which host?

Several banked rows and beads cite an absolute path inside `/data/tmp/cargo-target/frozen/`
as the provenance of a measured binary. That directory does not exist on this host. Two
readings, and this script does not claim to distinguish them: the builds ran on remote
workers, so the frozen tree may live on the worker rather than here; or it existed here and
is gone.

The consequence is the same either way, and it is the point. The ledger's
executed-ELF-sha256 clause exists so a row's binary can be re-verified against the artifact.
An absolute path with no host attached is ambiguous across an eleven-worker fleet, so the
clause cannot do its job -- the SHA is recorded but unfalsifiable.

WHAT THIS CHECKS, per citation:
  * does the path resolve on THIS host?
  * does the citing record name exactly one host, so the path is unambiguous?

A record naming two hosts is reported as ambiguous, not as attributed. `dw6du` mentions both
thinkstation1 and vmi1153651, so "a host is mentioned somewhere in the text" is not the same
as "the path is attributed" -- which is precisely the distinction that makes this worth
scripting rather than eyeballing.

NOT A DELETION TOOL and not an authorisation for one. y9bzw escalates reclaim to the user;
this only reports.

Exit 0 if every citation either resolves locally or carries exactly one host; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FROZEN = re.compile(r"/data/tmp/cargo-target/frozen/[A-Za-z0-9._-]+")

# The fleet as configured. Kept explicit rather than inferred: a regex for "hostlike token"
# would match crate names and commit prefixes and quietly inflate the attributed count.
WORKERS = [
    "threadripperje", "thinkstation1", "fixmydocuments", "ovh-a", "hz1", "hz2",
    "vmi1149989", "vmi1152480", "vmi1153651", "vmi1156319", "vmi1227854",
    "vmi1264463", "vmi1293453",
]


# How much text around a citation counts as "the record" for host attribution.
#
# This is not a tuning knob, it is a correctness fix. My first version scoped the host search
# to the WHOLE document, and perf_ledger_cc.md mentions every worker in the fleet somewhere
# across thousands of lines -- so all eight of its citations were reported as naming twelve
# hosts, which is a probe blanket-matching rather than a finding. A citation is attributed by
# text NEAR it, not by anything that happens to appear in the same file.
DOC_CONTEXT_LINES = 40


def citation_context(text: str, match: re.Match[str]) -> str:
    """Return the line window that can attribute one frozen-ELF citation."""
    lines = text.splitlines()
    line_index = text[: match.start()].count("\n")
    lo = max(0, line_index - DOC_CONTEXT_LINES)
    hi = min(len(lines), line_index + DOC_CONTEXT_LINES + 1)
    return "\n".join(lines[lo:hi])


def contexts_from_text(origin: str, text: str):
    """Yield one (origin, path, context) tuple for every citation in one record field."""
    for match in FROZEN.finditer(text):
        yield origin, match.group(), citation_context(text, match)


def parse_issue_record(line: str) -> dict[str, object] | None:
    """Parse one durable Beads record without abandoning the whole audit on bad data."""
    try:
        return json.JSONDecoder().decode(line)
    except json.JSONDecodeError as error:
        print(f"Skipping malformed Beads record: {error}", file=sys.stderr)
        return None


def sources():
    """Yield citation-local (origin, path, context) tuples.

    A whole Bead JSON object is not a provenance record: its description, notes, and every
    historical comment can name unrelated workers. Attribute a path only from its own field's
    local window. Markdown ledgers follow the same rule, one window per matching line.
    """
    beads = ROOT / ".beads" / "issues.jsonl"
    if beads.is_file():
        for line in beads.read_text(errors="replace").splitlines():
            obj = parse_issue_record(line)
            if obj is None:
                continue
            bead_id = str(obj.get("id", "<bead>"))
            for field in ("description", "notes", "close_reason"):
                text = obj.get(field)
                if isinstance(text, str):
                    yield from contexts_from_text(f"{bead_id}:{field}", text)
            for comment in obj.get("comments", []):
                if not isinstance(comment, dict):
                    continue
                text = comment.get("text")
                if isinstance(text, str):
                    comment_id = comment.get("id", "<comment>")
                    yield from contexts_from_text(f"{bead_id}:comment-{comment_id}", text)
    for doc in sorted((ROOT / "docs").glob("*.md")):
        lines = doc.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            for match in FROZEN.finditer(line):
                lo = max(0, index - DOC_CONTEXT_LINES)
                hi = min(len(lines), index + DOC_CONTEXT_LINES + 1)
                yield f"{doc.name}:{index + 1}", match.group(), "\n".join(lines[lo:hi])


def main() -> int:
    rows = []
    for origin, path, context in sources():
        hosts = sorted({worker for worker in WORKERS if worker in context})
        rows.append((origin, path, pathlib.Path(path).exists(), hosts))

    if not rows:
        print("no frozen-ELF citations found")
        return 0

    unresolved_unattributed = []
    print(f"{'origin':<24} {'artifact':<48} {'local':<7} hosts named in record")
    for origin, path, exists, hosts in sorted(rows):
        name = path.rsplit("/", 1)[-1]
        label = ",".join(hosts) if hosts else "-- NONE --"
        if len(hosts) > 1:
            label += "  (AMBIGUOUS)"
        print(f"{origin:<24} {name:<48} {exists!s:<7} {label}")
        if not exists and len(hosts) != 1:
            unresolved_unattributed.append((origin, name, hosts))

    resolvable = sum(1 for r in rows if r[2])
    print(
        f"\ncitations={len(rows)} resolvable_here={resolvable} "
        f"unresolved_and_unattributed={len(unresolved_unattributed)}"
    )
    if unresolved_unattributed:
        print(
            "\nThese cite a binary that cannot be located from this host and do not name a "
            "single host it is relative to, so their recorded ELF SHA cannot be re-verified:",
            file=sys.stderr,
        )
        for origin, name, hosts in unresolved_unattributed:
            why = "no host named" if not hosts else f"ambiguous: {','.join(hosts)}"
            print(f"  {origin}: {name} ({why})", file=sys.stderr)
        return 1
    return 0


def cmd_self_test() -> int:
    artifact = "/data/tmp/cargo-target/frozen/perf_fixture-01234567"
    distant_host_record = "\n".join(
        [f"artifact={artifact} built on hz1"]
        + [f"unrelated line {number}" for number in range(DOC_CONTEXT_LINES + 1)]
        + ["old unrelated run used ovh-a"]
    )
    match = FROZEN.search(distant_host_record)
    if match is None:
        print("self-test FAILED: fixture has no frozen-ELF path", file=sys.stderr)
        return 1
    local_hosts = sorted({worker for worker in WORKERS if worker in citation_context(distant_host_record, match)})
    whole_record_hosts = sorted({worker for worker in WORKERS if worker in distant_host_record})
    if local_hosts != ["hz1"] or whole_record_hosts != ["hz1", "ovh-a"]:
        print("self-test FAILED: distant host contaminated citation context", file=sys.stderr)
        return 1

    ambiguous_record = f"artifact={artifact} built by hz1, later executed on ovh-a"
    ambiguous_match = FROZEN.search(ambiguous_record)
    if ambiguous_match is None:
        print("self-test FAILED: ambiguous fixture has no frozen-ELF path", file=sys.stderr)
        return 1
    ambiguous_hosts = sorted(
        {worker for worker in WORKERS if worker in citation_context(ambiguous_record, ambiguous_match)}
    )
    if ambiguous_hosts != ["hz1", "ovh-a"]:
        print("self-test FAILED: nearby multiple hosts were not ambiguous", file=sys.stderr)
        return 1

    print("frozen-ELF provenance self-test: PASS (2 attribution-boundary checks)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run deterministic attribution-boundary checks")
    args = parser.parse_args()
    raise SystemExit(cmd_self_test() if args.self_test else main())
