"""Find perf-toggle reads that sit INSIDE a loop body.

WHY. A `PerfToggle`/`AtomicBool` load inside a per-item loop is an optimisation
BARRIER, not merely a cheap load: measured on frankenscipy splu, moving one read out
of the per-update branch and passing a plain `bool` moved the median from 0.498 to
0.562 (~13%), both shipping binaries, alternated in one window. The load executed
555,096 times per factorization; the cost was the specialisation it prevented, not
the atomic.

So this scan is not "which toggles exist" but "which toggle READS are inside a loop",
which is a static property and needs no build to find.

Judgment still required per hit: a read inside a loop that runs once, or in a cold
path, costs nothing. The scan reports loop nesting depth and the enclosing function so
that judgment has something to work with. Test code is excluded -- barriers there are
harmless and are often deliberate instrumentation.
"""
import re
import sys
import os

TOGGLE_DEF = re.compile(r"^\s*(?:pub\s+)?(?:pub\([^)]*\)\s+)?static\s+([A-Z_0-9]+)\s*:\s*(?:[A-Za-z:]*\s*)?(AtomicBool|PerfToggle)\b")
FN_DEF = re.compile(r"^\s*(?:pub\s+)?(?:pub\([^)]*\)\s+)?(?:const\s+|async\s+|unsafe\s+|extern\s+\"[^\"]*\"\s+)*fn\s+([A-Za-z_0-9]+)")
TEST_MOD = re.compile(r"^\s*(?:pub\s+)?mod\s+tests?\b")
LOOP = re.compile(r"^\s*(?:\}\s*)?(?:'[a-z_0-9]+:\s*)?(for|while|loop)\b")
CFG_TEST = re.compile(r"#\[cfg\(test\)\]|#\[cfg\(any\([^)]*test|#\[test\]|#\[rstest")


def strip_comment(line):
    # crude but adequate: no string literals in this codebase contain "//" in a way
    # that matters for brace counting of loops
    i = line.find("//")
    return line[:i] if i >= 0 else line


def scan_text(text, toggles, path="<mem>"):
    """Return list of hits: (path, lineno, toggle, fn, loop_depth)."""
    hits = []
    depth = 0
    loop_stack = []        # depths at which a loop body opened
    fn_stack = []          # (depth, name)
    in_test_mod_depth = None
    pending_test_attr = False
    pending_loop = False
    pending_fn = None

    for n, raw in enumerate(text.splitlines(), 1):
        line = strip_comment(raw)

        if CFG_TEST.search(raw):
            pending_test_attr = True

        if TEST_MOD.search(line):
            in_test_mod_depth = depth

        m = FN_DEF.search(line)
        if m:
            pending_fn = m.group(1)
            if pending_test_attr:
                # a #[cfg(test)] fn: mark by giving it a sentinel name
                pending_fn = "#test#" + pending_fn

        if LOOP.search(line):
            pending_loop = True

        opens = line.count("{")
        closes = line.count("}")

        # toggle reads on this line, evaluated at CURRENT depth.
        # Cheap pre-filter first: without it this is 332 regexes per line and the
        # scan does not finish.
        for tname in (toggles if ".load" in line else ()):
            if re.search(r"\b" + re.escape(tname) + r"\s*\.\s*load\s*\(", line):
                in_test = (in_test_mod_depth is not None) or (
                    fn_stack and fn_stack[-1][1].startswith("#test#")
                )
                if not in_test and loop_stack:
                    fn = fn_stack[-1][1] if fn_stack else "?"
                    hits.append((path, n, tname, fn, len(loop_stack)))

        for _ in range(opens):
            depth += 1
            if pending_loop:
                loop_stack.append(depth)
                pending_loop = False
            elif pending_fn is not None:
                fn_stack.append((depth, pending_fn))
                pending_fn = None
        if opens:
            pending_test_attr = False

        for _ in range(closes):
            if loop_stack and loop_stack[-1] == depth:
                loop_stack.pop()
            if fn_stack and fn_stack[-1][0] == depth:
                fn_stack.pop()
            if in_test_mod_depth is not None and depth == in_test_mod_depth + 1:
                in_test_mod_depth = None
            depth -= 1

    return hits


def collect_toggles(files):
    toggles = {}
    for p in files:
        try:
            text = open(p, errors="replace").read()
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            m = TOGGLE_DEF.search(line)
            if m:
                toggles[m.group(1)] = (p, n, m.group(2))
    return toggles


MUST_HIT = """
static FOO_ENABLE: AtomicBool = AtomicBool::new(true);
fn kernel(xs: &mut [f64]) {
    for x in xs.iter_mut() {
        if FOO_ENABLE.load(Ordering::Relaxed) {
            *x += 1.0;
        }
    }
}
"""

MUST_MISS_HOISTED = """
static FOO_ENABLE: AtomicBool = AtomicBool::new(true);
fn kernel(xs: &mut [f64]) {
    let on = FOO_ENABLE.load(Ordering::Relaxed);
    for x in xs.iter_mut() {
        if on { *x += 1.0; }
    }
}
"""

MUST_MISS_TEST = """
static FOO_ENABLE: AtomicBool = AtomicBool::new(true);
#[cfg(test)]
mod tests {
    fn t() {
        for _ in 0..3 {
            assert!(FOO_ENABLE.load(Ordering::Relaxed));
        }
    }
}
"""

MUST_HIT_NESTED = """
static FOO_ENABLE: AtomicBool = AtomicBool::new(true);
fn kernel(m: &mut [[f64; 4]]) {
    for row in m.iter_mut() {
        for x in row.iter_mut() {
            if FOO_ENABLE.load(Ordering::Relaxed) { *x += 1.0; }
        }
    }
}
"""


MUST_MISS_TEST_MOD_DETACHED = """
static FOO_ENABLE: AtomicBool = AtomicBool::new(true);
#[cfg(test)]
#[allow(clippy::all)]
mod tests {
    use super::*;
    fn t() {
        for _ in 0..3 {
            assert!(FOO_ENABLE.load(Ordering::Relaxed));
        }
    }
}
"""

def selftest():
    cases = [
        ("must-hit  read in loop", MUST_HIT, 1, 1),
        ("must-hit  read in nested loop", MUST_HIT_NESTED, 1, 2),
        ("must-miss hoisted above loop", MUST_MISS_HOISTED, 0, None),
        ("must-miss read inside cfg(test) mod", MUST_MISS_TEST, 0, None),
        ("must-miss cfg(test) attr NOT adjacent to mod", MUST_MISS_TEST_MOD_DETACHED, 0, None),
    ]
    ok = True
    for name, src, want_n, want_depth in cases:
        hits = scan_text(src, {"FOO_ENABLE"})
        got_n = len(hits)
        good = got_n == want_n and (want_depth is None or hits[0][4] == want_depth)
        ok &= good
        d = hits[0][4] if hits else "-"
        print(f"  [{'ok' if good else 'FAIL'}] {name:<38} hits={got_n} (want {want_n})  depth={d} (want {want_depth})")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1



# ---------------------------------------------------------------------------
# PASS 2: the gap the first pass explicitly does NOT cover.
#
# A toggle read once per CALL, in a function that is itself called from inside a
# per-element loop, costs exactly what a read inside the loop costs -- the barrier
# lands in the same place. Pass 1 is lexical and clears such a function, because the
# read really is at the top of its body. So: find functions holding a top-level
# toggle read, then look for CALL SITES of those functions that sit inside a loop.
#
# This is a name-based approximation of a call graph. It over-selects (any `name(`
# token counts) and under-selects (method calls through traits, function pointers,
# closures stored and invoked later are all invisible). It is a place to LOOK, not a
# verdict -- same discipline as pass 1, where all 8 hits had to be read by hand.
# ---------------------------------------------------------------------------


def fns_with_toplevel_toggle_read(text, toggles):
    """fn name -> (lineno, toggle) for reads at loop_depth 0 inside a fn body."""
    found = {}
    depth = 0
    loop_stack = []
    fn_stack = []
    pending_loop = False
    pending_fn = None
    pending_test_attr = False
    in_test_mod_depth = None

    for n, raw in enumerate(text.splitlines(), 1):
        line = strip_comment(raw)
        if CFG_TEST.search(raw):
            pending_test_attr = True
        if TEST_MOD.search(line):
            in_test_mod_depth = depth
        m = FN_DEF.search(line)
        if m:
            pending_fn = m.group(1)
            if pending_test_attr:
                pending_fn = "#test#" + pending_fn
        if LOOP.search(line):
            pending_loop = True

        if ".load" in line and not loop_stack and fn_stack:
            fname = fn_stack[-1][1]
            if not fname.startswith("#test#") and in_test_mod_depth is None:
                for t in toggles:
                    if re.search(r"\b" + re.escape(t) + r"\s*\.\s*load\s*\(", line):
                        found.setdefault(fname, (n, t))

        opens, closes = line.count("{"), line.count("}")
        for _ in range(opens):
            depth += 1
            if pending_loop:
                loop_stack.append(depth); pending_loop = False
            elif pending_fn is not None:
                fn_stack.append((depth, pending_fn)); pending_fn = None
        if opens:
            pending_test_attr = False
        for _ in range(closes):
            if loop_stack and loop_stack[-1] == depth: loop_stack.pop()
            if fn_stack and fn_stack[-1][0] == depth: fn_stack.pop()
            if in_test_mod_depth is not None and depth == in_test_mod_depth + 1:
                in_test_mod_depth = None
            depth -= 1
    return found


def callsites_in_loops(text, names, path="<mem>"):
    """Call sites of `names` that sit lexically inside a loop."""
    hits = []
    depth = 0
    loop_stack = []
    fn_stack = []
    pending_loop = False
    pending_fn = None
    pending_test_attr = False
    in_test_mod_depth = None

    for n, raw in enumerate(text.splitlines(), 1):
        line = strip_comment(raw)
        if CFG_TEST.search(raw):
            pending_test_attr = True
        if TEST_MOD.search(line):
            in_test_mod_depth = depth
        m = FN_DEF.search(line)
        if m:
            pending_fn = m.group(1)
            if pending_test_attr:
                pending_fn = "#test#" + pending_fn
        if LOOP.search(line):
            pending_loop = True

        if loop_stack and in_test_mod_depth is None:
            caller = fn_stack[-1][1] if fn_stack else "?"
            if not caller.startswith("#test#"):
                for name in names:
                    # a call, not the definition
                    if FN_DEF.search(line) and m and m.group(1) == name:
                        continue
                    if re.search(r"(?<![A-Za-z_0-9.])" + re.escape(name) + r"\s*\(", line):
                        hits.append((path, n, name, caller, len(loop_stack)))

        opens, closes = line.count("{"), line.count("}")
        for _ in range(opens):
            depth += 1
            if pending_loop:
                loop_stack.append(depth); pending_loop = False
            elif pending_fn is not None:
                fn_stack.append((depth, pending_fn)); pending_fn = None
        if opens:
            pending_test_attr = False
        for _ in range(closes):
            if loop_stack and loop_stack[-1] == depth: loop_stack.pop()
            if fn_stack and fn_stack[-1][0] == depth: fn_stack.pop()
            if in_test_mod_depth is not None and depth == in_test_mod_depth + 1:
                in_test_mod_depth = None
            depth -= 1
    return hits


# A name-based call graph is only meaningful for names that resolve UNIQUELY. The
# first run of this pass reported 1533 "hits" that were almost entirely `new(`,
# `solve(`, `run(` and `median(` -- generic names defined in many places, each
# attributed to one arbitrary definition. The controls passed anyway because the
# control helper had a unique name, so they exercised loop-context discrimination and
# never name resolution. That is the failure the two-arm rule is meant to catch and
# the arms were chosen too kindly. Hence: drop every ambiguous name, and confine the
# scan to shipping code.

NON_SHIPPING = ("/bin/", "/tests/", "/benches/", "/examples/", "/build.rs")


def is_shipping(path):
    return not any(m in path.replace("\\", "/") for m in NON_SHIPPING)


def unique_fn_definitions(files):
    """fn name -> defining path, keeping ONLY names defined exactly once tree-wide."""
    seen = {}
    for p in files:
        try:
            text = open(p, errors="replace").read()
        except OSError:
            continue
        for line in text.splitlines():
            m = FN_DEF.search(strip_comment(line))
            if m:
                seen.setdefault(m.group(1), set()).add(p)
    return {n: next(iter(ps)) for n, ps in seen.items() if len(ps) == 1}


PASS2_AMBIGUOUS = """
static BAR_ENABLE: AtomicBool = AtomicBool::new(true);
fn build(x: f64) -> f64 {
    let on = BAR_ENABLE.load(Ordering::Relaxed);
    if on { x } else { -x }
}
fn hot(xs: &mut [f64]) {
    for x in xs.iter_mut() {
        *x = build(*x);
    }
}
"""

PASS2_AMBIGUOUS_TWIN = """
fn build(y: i64) -> i64 { y + 1 }
"""

PASS2_SRC = """
static FOO_ENABLE: AtomicBool = AtomicBool::new(true);
fn helper(x: f64) -> f64 {
    let on = FOO_ENABLE.load(Ordering::Relaxed);
    if on { x + 1.0 } else { x }
}
fn hot(xs: &mut [f64]) {
    for x in xs.iter_mut() {
        *x = helper(*x);
    }
}
fn cold(x: f64) -> f64 {
    helper(x)
}
"""


def selftest_pass2():
    toggles = {"FOO_ENABLE"}
    fns = fns_with_toplevel_toggle_read(PASS2_SRC, toggles)
    ok = "helper" in fns and "hot" not in fns
    print(f"  [{'ok' if ok else 'FAIL'}] pass2 must-hit  `helper` has a top-level toggle read   got={sorted(fns)}")
    sites = callsites_in_loops(PASS2_SRC, set(fns))
    got = [(s[2], s[3]) for s in sites]
    ok2 = ("helper", "hot") in got and ("helper", "cold") not in got
    print(f"  [{'ok' if ok2 else 'FAIL'}] pass2 must-hit  call in `hot` / must-miss call in `cold`  got={got}")
    # THIRD ARM: a name defined in more than one place must be DROPPED, not
    # attributed to whichever definition happened to be seen first.
    import tempfile, os as _os
    d = tempfile.mkdtemp()
    a, b = _os.path.join(d, "a.rs"), _os.path.join(d, "b.rs")
    open(a, "w").write(PASS2_AMBIGUOUS)
    open(b, "w").write(PASS2_AMBIGUOUS_TWIN)
    uniq = unique_fn_definitions([a, b])
    ok3 = "build" not in uniq and "hot" in uniq
    print(f"  [{'ok' if ok3 else 'FAIL'}] pass2 must-miss ambiguous name `build` dropped     uniq={sorted(uniq)}")

    ok4 = is_shipping("crates/x/src/lib.rs") and not is_shipping("crates/x/src/bin/p.rs") \
        and not is_shipping("crates/x/tests/t.rs") and not is_shipping("crates/x/benches/b.rs")
    print(f"  [{'ok' if ok4 else 'FAIL'}] pass2 shipping-scope filter (src yes; bin/tests/benches no)")

    allok = ok and ok2 and ok3 and ok4
    print("PASS2 SELFTEST", "PASS" if allok else "FAIL")
    return 0 if allok else 1

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        rc = selftest()
        rc |= selftest_pass2()
        sys.exit(rc)
    roots = [a for a in sys.argv[1:] if not a.startswith("--")] or ["crates"]
    files = []
    for root in roots:
        for dirpath, _, names in os.walk(root):
            if "/target/" in dirpath:
                continue
            files += [os.path.join(dirpath, f) for f in names if f.endswith(".rs")]
    toggles = collect_toggles(files)
    print(f"toggle statics found : {len(toggles)}")
    all_hits = []
    for p in files:
        try:
            all_hits += scan_text(open(p, errors="replace").read(), toggles, p)
        except OSError:
            pass
    print(f"reads INSIDE a loop  : {len(all_hits)}\n")
    for p, n, t, fn, d in sorted(all_hits, key=lambda h: (-h[4], h[0], h[1])):
        print(f"  loop_depth={d}  {p}:{n}  {t}  in fn `{fn}`")

    if "--callsites" in sys.argv:
        print("\n=== PASS 2: fns with a TOP-LEVEL toggle read, called from inside a loop ===")
        ship = [p for p in files if is_shipping(p)]
        uniq = unique_fn_definitions(ship)
        owners = {}
        for p in ship:
            try:
                t = open(p, errors="replace").read()
            except OSError:
                continue
            for fn, (n, tog) in fns_with_toplevel_toggle_read(t, toggles).items():
                if fn in uniq:                     # unambiguous names only
                    owners[fn] = (p, n, tog)
        print(f"shipping files scanned              : {len(ship)} of {len(files)}")
        print(f"fns holding a top-level toggle read : {len(owners)} (unambiguous names only)")
        sites = []
        for p in ship:
            try:
                sites += callsites_in_loops(open(p, errors="replace").read(), set(owners), p)
            except OSError:
                pass
        print(f"such fns CALLED from inside a loop  : {len(sites)}\n")
        for p, n, name, caller, d in sorted(sites, key=lambda h: (-h[4], h[0], h[1])):
            op, on_, tog = owners[name]
            print(f"  loop_depth={d}  {p}:{n}  calls `{name}` (reads {tog} at {op}:{on_}) from fn `{caller}`")
