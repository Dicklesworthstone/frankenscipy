#!/usr/bin/env python3
"""A public option field must be read by the library that declares it.

Observed defect class (frankenscipy-szq1n.12): public option fields were accepted, documented,
sometimes validated, and never read: `LstsqOptions.driver`, `SolveOptions.lower`,
`InvOptions.lower`, `CubatureOptions.rule`, ODR `fjacb`/`fjacd`/`estimate`,
`FindPeaksOptions.width`, `PlanCacheConfig.planning_strategy`, `FftOptions.overwrite_input`.
A caller who set one believed they had chosen a behaviour they did not get.

Rule: every `pub` field of a `pub struct *Options / *Config / *Opts / *Params / *Settings` in
`crates/*/src` has at least one `.field` read in the same crate's non-test library code
(`#[cfg(test)]` items and `src/bin` excluded). A field that is only validated still counts as
read; this check catches the silent case, not every partial one.

Scope: fsci-conformance is excluded. It is the harness, not library API, and its
`quality_gates.toml` config is outside this defect class.

KNOWN_OPEN lists fields that are unread today and tracked by an open bead. An entry leaves
only by being wired (the check then reports it as stale) or removed; it never grows to make
a red run green.

Exit status: 0 clean, 1 unread fields (or a stale KNOWN_OPEN entry), 2 self-test failed.
    python3 scripts/unread_option_fields.py
    python3 scripts/unread_option_fields.py --self-test
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from status_literal_guard import test_line_mask  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXCLUDED_CRATES = {"fsci-conformance"}
STRUCT = re.compile(r"pub struct (\w+(?:Options|Config|Opts|Params|Settings))\b[^{;]*\{")
FIELD = re.compile(r"pub\s+(\w+)\s*:")

# (crate, struct, field) -> the open bead that owns wiring or removing it. Empty: the nine
# SpecialErrConfig fields listed here were wired by frankenscipy-8dndw.1 (errstate is real).
KNOWN_OPEN: dict[tuple[str, str, str], str] = {}


def library_code(text: str) -> str:
    """Non-test code with `//` comments stripped."""
    lines = text.split("\n")
    mask = test_line_mask(lines)
    return "\n".join(line.split("//", 1)[0] for line, in_test in zip(lines, mask) if not in_test)


def unread_fields(sources: dict[str, str]) -> list[tuple[str, str]]:
    """`(struct, field)` pairs declared in `sources` (path -> library code) and never read."""
    all_code = "\n".join(sources.values())
    unread = []
    for text in sources.values():
        for match in STRUCT.finditer(text):
            depth, end = 1, match.end()
            while depth and end < len(text):
                depth += {"{": 1, "}": -1}.get(text[end], 0)
                end += 1
            for field in FIELD.findall(text[match.end() : end - 1]):
                read = re.compile(r"\.\s*" + re.escape(field) + r"\b(?!\s*\()")
                if not read.search(all_code):
                    unread.append((match.group(1), field))
    return unread


def scan() -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    found = []
    for crate in sorted(p for p in (ROOT / "crates").iterdir() if (p / "src").is_dir()):
        if crate.name in EXCLUDED_CRATES:
            continue
        sources = {
            str(path): library_code(path.read_text())
            for path in sorted((crate / "src").rglob("*.rs"))
            if "/src/bin/" not in str(path)
        }
        found.extend((crate.name, struct, field) for struct, field in unread_fields(sources))
    new = [row for row in found if row not in KNOWN_OPEN]
    stale = [row for row in KNOWN_OPEN if row not in found]
    return new, stale


def self_test() -> bool:
    unread = "pub struct FooOptions {\n    pub used: bool,\n    pub ignored: f64,\n}\nfn f(o: &FooOptions) -> bool { o.used }\n"
    only_in_test = unread + "#[cfg(test)]\nmod tests {\n    fn t(o: &super::FooOptions) -> f64 { o.ignored }\n}\n"
    method_call = unread + "fn g(o: &FooOptions) -> f64 { o.ignored() }\n"
    read = unread + "fn h(o: &FooOptions) -> f64 { o.ignored }\n"
    cases = [
        ("unread field is flagged", unread, [("FooOptions", "ignored")]),
        ("a read inside #[cfg(test)] does not count", only_in_test, [("FooOptions", "ignored")]),
        ("a same-named method call does not count", method_call, [("FooOptions", "ignored")]),
        ("a library read clears it", read, []),
    ]
    ok = True
    for name, text, want in cases:
        got = unread_fields({"x.rs": library_code(text)})
        print(f"self-test: {name}: {got} (want {want})")
        ok &= got == want
    return ok


def main() -> int:
    if not self_test():
        print("self-test FAILED")
        return 2
    if "--self-test" in sys.argv:
        return 0
    new, stale = scan()
    for crate, struct, field in new:
        print(f"UNREAD {crate}: {struct}.{field} is never read by library code")
    for crate, struct, field in stale:
        print(f"STALE KNOWN_OPEN {crate}: {struct}.{field} is read now; drop the entry")
    print(f"{len(new)} unread field(s), {len(stale)} stale KNOWN_OPEN entr(ies), {len(KNOWN_OPEN)} known open")
    return 1 if new or stale else 0


if __name__ == "__main__":
    sys.exit(main())
