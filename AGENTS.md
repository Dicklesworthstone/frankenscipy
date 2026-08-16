# AGENTS.md — FrankenSciPy

> Guidelines for AI coding agents working in this Rust codebase.

---

## RULE 0 - THE FUNDAMENTAL OVERRIDE PREROGATIVE

If I tell you to do something, even if it goes against what follows below, YOU MUST LISTEN TO ME. I AM IN CHARGE, NOT YOU.

---

## RULE 0.5 - SUITE-WIDE RULES LIVE IN /data/projects/AGENTS.md

The suite-wide rules in **`/data/projects/AGENTS.md`** bind you here too. Read it. Two sections
are load-bearing for perf work and are NOT duplicated below, so they cannot drift out of sync:

- **`## Named Reward-Hacking Patterns (ALL FORBIDDEN)`** — 12 named patterns, several already
  observed in this suite: gate self-weakening (and the exact price of a legitimate gate fix),
  proof-class inflation, golden regeneration reflex, commit-stream pumping, tautological tests,
  easy-lever cherry-picking, close-pump abuse, scope-splitting, spec-editing as progress,
  conformance metastasis, dependency smuggling, bench-path hardcoding.
- **`### Work-Graph Discipline`** — JSONL is truth and `beads.db` is disposable, `br sync
  --import-only` after every pull, single-writer on graph structure, closure on cited evidence
  with blocker beads gated on their named probe, `br dep cycles` stays empty.

The three that most often decide whether a number here is real: a **self-speedup is
MAINTENANCE, not a win** — a win needs the incumbent live in the SAME invocation; **never
weaken a gate to land a change**, and if a gate is genuinely defective, meet the evidence
standard and publish the win/lose split of what the fix admits; and **reporting a loss is a
success** — one line, revert, next lever, no retraction narrative.

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a new file that you yourself created, such as a test code file. You have a horrible track record of deleting critically important files or otherwise throwing away tons of expensive work. As a result, you have permanently lost any and all rights to determine that a file or folder should be deleted.

**YOU MUST ALWAYS ASK AND RECEIVE CLEAR, WRITTEN PERMISSION BEFORE EVER DELETING A FILE OR FOLDER OF ANY KIND.**

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval. "I think it's safe" is never acceptable.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options (`git status`, `git diff`, `git stash`, copying to backups) before ever considering a destructive command.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for a confirmation that your understanding is correct. Only then may you execute it—if anything remains ambiguous, refuse and escalate.
5. **Document the confirmation:** When running any approved destructive command, record (in the session notes / final response) the exact user text that authorized it, the command actually run, and the execution time. If that record is absent, the operation did not happen.

---

## Git Branch: ONLY Use `main`, NEVER `master`

**The default branch is `main`. The `master` branch exists only for legacy URL compatibility.**

- **All work happens on `main`** — commits, PRs, feature branches all merge to `main`
- **Never reference `master` in code or docs** — if you see `master` anywhere, it's a bug that needs fixing
- **The `master` branch must stay synchronized with `main`** — after pushing to `main`, also push to `master`:
  ```bash
  git push origin main:master
  ```

**If you see `master` referenced anywhere:**
1. Update it to `main`
2. Ensure `master` is synchronized: `git push origin main:master`

---

## Toolchain: Rust & Cargo

We only use **Cargo** in this project, NEVER any other package manager.

- **Edition:** Rust 2024 (nightly required — see `rust-toolchain.toml`)
- **Dependency versions:** Explicit versions for stability
- **Configuration:** Cargo.toml workspace with `workspace = true` pattern
- **Unsafe code:** Forbidden (`#![forbid(unsafe_code)]`)
- If narrow unsafe usage is unavoidable, isolate it behind audited interfaces and tests

### Async Runtime: asupersync (MANDATORY — NO TOKIO)

**This project uses [asupersync](/dp/asupersync) exclusively for all async/concurrent operations. Tokio and the entire tokio ecosystem are FORBIDDEN.**

- **Structured concurrency**: `Cx`, `Scope`, `region()` — no orphan tasks
- **Cancel-correct channels**: Two-phase `reserve()/send()` — no data loss on cancellation
- **Sync primitives**: `asupersync::sync::Mutex`, `RwLock`, `OnceCell`, `Pool` — cancel-aware
- **Deterministic testing**: `LabRuntime` with virtual time, DPOR, oracles

**Forbidden crates**: `tokio`, `hyper`, `reqwest`, `axum`, `tower` (tokio adapter), `async-std`, `smol`, or any crate that transitively depends on tokio.

**Pattern**: All async functions take `&Cx` as first parameter. The `Cx` flows down from the consumer's runtime — FrankenSciPy does NOT create its own runtime.

### Key Dependencies

| Crate | Purpose |
|-------|---------|
| `asupersync` | Structured async runtime (channels, sync, regions, testing) |
| `fsci-arrayapi` | Array API abstraction layer |
| `fsci-linalg` | Linear algebra routines |
| `fsci-sparse` | Sparse matrix operations |
| `fsci-opt` | Optimization solvers |
| `fsci-integrate` | Numerical integration and ODE solvers |
| `fsci-fft` | Fast Fourier Transform |
| `fsci-special` | Special mathematical functions |
| `fsci-runtime` | Runtime solver portfolio and algorithm selection |
| `fsci-conformance` | Conformance testing against SciPy reference |
| `ftui` | frankentui TUI rendering |
| `blake3` | Cryptographic hashing for artifact integrity |
| `serde` + `serde_json` | Serialization |
| `thiserror` | Ergonomic error type derivation |
| `proptest` | Property-based testing |
| `criterion` | Benchmarking |
| `rand` | Random number generation |

### Release Profile

The release build optimizes for performance (this is a library, not a binary):

```toml
[profile.release]
opt-level = 3       # Maximum performance optimization
lto = true          # Link-time optimization
codegen-units = 1   # Single codegen unit for better optimization
strip = true        # Remove debug symbols
```

---

## Code Editing Discipline

### No Script-Based Changes

**NEVER** run a script that processes/changes code files in this repo. Brittle regex-based transformations create far more problems than they solve.

- **Always make code changes manually**, even when there are many instances
- For many simple changes: use parallel subagents
- For subtle/complex changes: do them methodically yourself

### No File Proliferation

If you want to change something or add a feature, **revise existing code files in place**.

**NEVER** create variations like:
- `mainV2.rs`
- `main_improved.rs`
- `main_enhanced.rs`

New files are reserved for **genuinely new functionality** that makes zero sense to include in any existing file. The bar for creating new files is **incredibly high**.

---

## Backwards Compatibility

We do not care about backwards compatibility—we're in early development with no users. We want to do things the **RIGHT** way with **NO TECH DEBT**.

- Never create "compatibility shims"
- Never create wrapper functions for deprecated APIs
- Just fix the code directly

---

## Compiler Checks (CRITICAL)

**After any substantive code changes, you MUST verify no errors were introduced:**

```bash
# Check for compiler errors and warnings (workspace-wide)
cargo check --workspace --all-targets

# Check for clippy lints (pedantic + nursery are enabled)
cargo clippy --workspace --all-targets -- -D warnings

# Verify formatting
cargo fmt --check
```

If you see errors, **carefully understand and resolve each issue**. Read sufficient context to fix them the RIGHT way.

If conformance/bench crates exist, also run:

```bash
cargo test -p fsci-conformance -- --nocapture
cargo bench
```

---

## Testing

### Testing Policy

Every component crate includes inline `#[cfg(test)]` unit tests alongside the implementation. Tests must cover:
- Happy path
- Edge cases (empty input, max values, boundary conditions)
- Error conditions

Cross-component integration tests live in the workspace `tests/` directory.

### Unit Tests

```bash
# Run all tests across the workspace
cargo test --workspace

# Run with output
cargo test --workspace -- --nocapture

# Run tests for a specific crate
cargo test -p fsci-linalg
cargo test -p fsci-sparse
cargo test -p fsci-opt
cargo test -p fsci-integrate
cargo test -p fsci-fft
cargo test -p fsci-special
cargo test -p fsci-arrayapi
cargo test -p fsci-runtime
cargo test -p fsci-conformance

# Run tests with all features enabled
cargo test --workspace --all-features
```

### Test Categories

| Crate | Focus Areas |
|-------|-------------|
| `fsci-linalg` | Matrix decompositions, solvers, eigenvalue routines, conditioning diagnostics |
| `fsci-sparse` | Sparse matrix formats, sparse solvers, fill-in strategies |
| `fsci-opt` | Optimization convergence, constraint handling, line search |
| `fsci-integrate` | ODE/IVP solvers, quadrature, step-size control, stiffness detection |
| `fsci-fft` | Transform correctness, Parseval's theorem, inverse round-trip |
| `fsci-special` | Special function accuracy vs reference values, domain edge cases |
| `fsci-arrayapi` | Array operations, broadcasting, type coercion |
| `fsci-runtime` | Solver portfolio selection, CASP diagnostics, stability certificates |
| `fsci-conformance` | Differential conformance against SciPy legacy oracle |

---

## Third-Party Library Usage

If you aren't 100% sure how to use a third-party library, **SEARCH ONLINE** to find the latest documentation and current best practices.

---

## FrankenSciPy — This Project

**This is the project you're working on.** FrankenSciPy is a ground-up Rust reimplementation of SciPy's core numerical computing routines, designed for correctness, performance, and safety.

### Crown-Jewel Innovation: Condition-Aware Solver Portfolio (CASP)

Runtime algorithm selection driven by conditioning diagnostics and stability certificates. CASP inspects the numerical properties of each problem instance (condition number, sparsity pattern, stiffness ratio, etc.) and selects the optimal solver from a portfolio, providing stability guarantees via formal certificates.

### Legacy Behavioral Oracle

- `/dp/frankenscipy/legacy_scipy_code/scipy`
- upstream: https://github.com/scipy/scipy

**CRITICAL NON-REGRESSION RULE:** Numerical stability guarantees outrank speed. No optimization may weaken tolerance contracts.

### Architecture

```
high-level API -> domain module -> algorithm selector -> numeric kernel -> diagnostics
```

### Workspace Structure

```
frankenscipy/
├── Cargo.toml                         # Workspace root
├── crates/
│   ├── fsci-arrayapi/                 # Array API abstraction layer
│   ├── fsci-linalg/                   # Linear algebra (decompositions, solvers, eigenvalues)
│   ├── fsci-sparse/                   # Sparse matrix operations
│   ├── fsci-opt/                      # Optimization solvers
│   ├── fsci-integrate/                # Numerical integration and ODE solvers
│   ├── fsci-fft/                      # Fast Fourier Transform
│   ├── fsci-special/                  # Special mathematical functions
│   ├── fsci-runtime/                  # CASP solver portfolio and runtime selection
│   └── fsci-conformance/             # Conformance testing against SciPy oracle
├── docs/                              # Architecture, schemas, artifact topology
├── legacy_scipy_code/                 # SciPy reference for behavioral oracle
├── reference/                         # Reference materials
├── fuzz/                              # Fuzz testing harnesses
└── .beads/                            # Issue tracking
```

### Compatibility Doctrine (Mode-Split)

- **Strict mode:**
  - Maximize observable compatibility for V1 scoped APIs
  - No behavior-altering repairs
- **Hardened mode:**
  - Preserve API contract while adding safety guards
  - Bounded defensive recovery for malformed inputs and hostile edge cases

Compatibility focus: Preserve SciPy-observable behavior for scoped routines with explicit tolerance/equality policies.

### Security Doctrine

Security focus: Defend against numerical instability abuse, malformed array metadata, and unsafe fallback paths under ill-conditioned inputs.

Minimum security bar:

1. Threat model notes for each major subsystem.
2. Fail-closed behavior for unknown incompatible features.
3. Adversarial fixture coverage and fuzz/property tests for high-risk parsers/state transitions.
4. Deterministic audit logs for recoveries and policy overrides.

### RaptorQ-Everywhere Contract

RaptorQ sidecar durability applies to:

- conformance fixture bundles
- benchmark baseline bundles
- migration manifests
- reproducibility ledgers
- long-lived state snapshots

Required outputs:

1. Repair-symbol generation manifest.
2. Integrity scrub report.
3. Decode proof artifact for each recovery event.

### Performance Doctrine

Track solver runtime tails, convergence costs, and memory budgets; gate regressions for core routine families.

Mandatory optimization loop:

1. Baseline: record p50/p95/p99 and memory.
2. Profile: identify real hotspots.
3. Implement one optimization lever.
4. Prove behavior unchanged via conformance + invariant checks.
5. Re-baseline and emit delta artifact.

### Correctness Doctrine

Maintain conditioning-aware fallback, convergence, and tolerance invariants for scoped algorithms.

Required evidence for substantive changes:

- differential conformance report
- invariant checklist update
- benchmark delta report
- risk-note update if threat or compatibility surface changed

### Artifact Topology Governance (Locked)

The artifact directory structure defined in `docs/ARTIFACT_TOPOLOGY.md` and the contract schemas in `docs/schemas/` are topology-locked. Changes require:

1. An explicit governance proposal in a bead or issue.
2. Review and approval by the project owner.
3. Update to `docs/ARTIFACT_TOPOLOGY.md` and the schema validation test.
4. Zero-regression confirmation on all existing artifacts.

Locked schemas: `behavior_ledger.schema.json`, `contract_table.schema.json`, `threat_matrix.schema.json`.

---

## MCP Agent Mail — Multi-Agent Coordination

A mail-like layer that lets coding agents coordinate asynchronously via MCP tools and resources. Provides identities, inbox/outbox, searchable threads, and advisory file reservations with human-auditable artifacts in Git.

### Why It's Useful

- **Prevents conflicts:** Explicit file reservations (leases) for files/globs
- **Token-efficient:** Messages stored in per-project archive, not in context
- **Quick reads:** `resource://inbox/...`, `resource://thread/...`

### Same Repository Workflow

1. **Register identity:**
   ```
   ensure_project(project_key=<abs-path>)
   register_agent(project_key, program, model)
   ```

2. **Reserve files before editing:**
   ```
   file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true)
   ```

3. **Communicate with threads:**
   ```
   send_message(..., thread_id="FEAT-123")
   fetch_inbox(project_key, agent_name)
   acknowledge_message(project_key, agent_name, message_id)
   ```

4. **Quick reads:**
   ```
   resource://inbox/{Agent}?project=<abs-path>&limit=20
   resource://thread/{id}?project=<abs-path>&include_bodies=true
   ```

### Macros vs Granular Tools

- **Prefer macros for speed:** `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`
- **Use granular tools for control:** `register_agent`, `file_reservation_paths`, `send_message`, `fetch_inbox`, `acknowledge_message`

### Common Pitfalls

- `"from_agent not registered"`: Always `register_agent` in the correct `project_key` first
- `"FILE_RESERVATION_CONFLICT"`: Adjust patterns, wait for expiry, or use non-exclusive reservation
- **Auth errors:** If JWT+JWKS enabled, include bearer token with matching `kid`

### What the pre-commit reservation guard actually protects (frankenscipy-hld7v)

An auto-commit once swept an in-flight negative control — a deliberately-wrong
golden — and pushed a RED value to `main` (`21c11204f`, reverted by
`130b32245`). Measured 2026-08-08, here is what the installed guard really does,
because the exposure is narrower and sharper than "auto-commits aren't covered":

**It does block.** `.git/hooks/hooks.d/{pre-commit,pre-push}/50-agent-mail.py`
refuses any commit touching a path held under an active exclusive reservation by
a *different* agent, and it fails closed (exit 2) when `AGENT_NAME` is unset.
Observed both arms on real commits: with the ambient identity the commit was
refused and every reserved path listed with its holder; re-running the identical
command with `AGENT_NAME=<your registered agent name>` committed cleanly.

**So set `AGENT_NAME` to the agent name you registered with**, e.g.
`AGENT_NAME=RubyBeacon git commit -- <paths>`. Reserving as one identity and
committing as another is what produces the confusing self-conflict.

**The hazard: `AGENT_NAME` is already set to `BlackThrush` in this environment**
— the git user, not a registered agent. Every process inherits it. The guard
self-exempts a committer whose `AGENT_NAME` equals the reservation holder, so if
you reserve paths while operating under that inherited default, your reservation
silently protects nothing from any other process that also inherited it. Reserve
under your own agent name, never the default.

**Four ways a commit still gets through**, none of which the guard can see:
`git commit --no-verify`, `AGENT_MAIL_BYPASS=1`, `AGENT_MAIL_GUARD_MODE=warn`,
and a reservation whose TTL expired mid-window (`is_expired` drops it from the
active set). Keep control windows short and renew long ones — a control held
past its TTL is unprotected exactly when you stopped watching.

**Prefer a control that cannot be committed wrong in the first place.** See the
freshness-probe recipe in the RCH section: flip a marker in a `src/bin` probe
rather than a golden constant. A swept marker is a harmless diff; a swept golden
is a red `main` under a plausible-looking commit message.

---

## Beads (br) — Dependency-Aware Issue Tracking

Beads provides a lightweight, dependency-aware issue database and CLI (`br` - beads_rust) for selecting "ready work," setting priorities, and tracking status. It complements MCP Agent Mail's messaging and file reservations.

**Important:** `br` is non-invasive—it NEVER runs git commands automatically. You must manually commit changes after `br sync --flush-only`.

### Conventions

- **Single source of truth:** Beads for task status/priority/dependencies; Agent Mail for conversation and audit
- **Shared identifiers:** Use Beads issue ID (e.g., `br-123`) as Mail `thread_id` and prefix subjects with `[br-123]`
- **Reservations:** When starting a task, call `file_reservation_paths()` with the issue ID in `reason`

### Typical Agent Flow

1. **Pick ready work (Beads):**
   ```bash
   br ready --json  # Choose highest priority, no blockers
   ```

2. **Reserve edit surface (Mail):**
   ```
   file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true, reason="br-123")
   ```

3. **Announce start (Mail):**
   ```
   send_message(..., thread_id="br-123", subject="[br-123] Start: <title>", ack_required=true)
   ```

4. **Work and update:** Reply in-thread with progress

5. **Complete and release:**
   ```bash
   br close 123 --reason "Completed"
   br sync --flush-only  # Export to JSONL (no git operations)
   ```
   ```
   release_file_reservations(project_key, agent_name, paths=["src/**"])
   ```
   Final Mail reply: `[br-123] Completed` with summary

### Mapping Cheat Sheet

| Concept | Value |
|---------|-------|
| Mail `thread_id` | `br-###` |
| Mail subject | `[br-###] ...` |
| File reservation `reason` | `br-###` |
| Commit messages | Include `br-###` for traceability |

---

## bv — Graph-Aware Triage Engine

bv is a graph-aware triage engine for Beads projects (`.beads/beads.jsonl`). It computes PageRank, betweenness, critical path, cycles, HITS, eigenvector, and k-core metrics deterministically.

**Scope boundary:** bv handles *what to work on* (triage, priority, planning). For agent-to-agent coordination (messaging, work claiming, file reservations), use MCP Agent Mail.

**CRITICAL: Use ONLY `--robot-*` flags. Bare `bv` launches an interactive TUI that blocks your session.**

### The Workflow: Start With Triage

**`bv --robot-triage` is your single entry point.** It returns:
- `quick_ref`: at-a-glance counts + top 3 picks
- `recommendations`: ranked actionable items with scores, reasons, unblock info
- `quick_wins`: low-effort high-impact items
- `blockers_to_clear`: items that unblock the most downstream work
- `project_health`: status/type/priority distributions, graph metrics
- `commands`: copy-paste shell commands for next steps

```bash
bv --robot-triage        # THE MEGA-COMMAND: start here
bv --robot-next          # Minimal: just the single top pick + claim command
```

### Command Reference

**Planning:**
| Command | Returns |
|---------|---------|
| `--robot-plan` | Parallel execution tracks with `unblocks` lists |
| `--robot-priority` | Priority misalignment detection with confidence |

**Graph Analysis:**
| Command | Returns |
|---------|---------|
| `--robot-insights` | Full metrics: PageRank, betweenness, HITS, eigenvector, critical path, cycles, k-core, articulation points, slack |
| `--robot-label-health` | Per-label health: `health_level`, `velocity_score`, `staleness`, `blocked_count` |
| `--robot-label-flow` | Cross-label dependency: `flow_matrix`, `dependencies`, `bottleneck_labels` |
| `--robot-label-attention [--attention-limit=N]` | Attention-ranked labels |

**History & Change Tracking:**
| Command | Returns |
|---------|---------|
| `--robot-history` | Bead-to-commit correlations |
| `--robot-diff --diff-since <ref>` | Changes since ref: new/closed/modified issues, cycles |

**Other:**
| Command | Returns |
|---------|---------|
| `--robot-burndown <sprint>` | Sprint burndown, scope changes, at-risk items |
| `--robot-forecast <id\|all>` | ETA predictions with dependency-aware scheduling |
| `--robot-alerts` | Stale issues, blocking cascades, priority mismatches |
| `--robot-suggest` | Hygiene: duplicates, missing deps, label suggestions |
| `--robot-graph [--graph-format=json\|dot\|mermaid]` | Dependency graph export |
| `--export-graph <file.html>` | Interactive HTML visualization |

### Scoping & Filtering

```bash
bv --robot-plan --label backend              # Scope to label's subgraph
bv --robot-insights --as-of HEAD~30          # Historical point-in-time
bv --recipe actionable --robot-plan          # Pre-filter: ready to work
bv --recipe high-impact --robot-triage       # Pre-filter: top PageRank
bv --robot-triage --robot-triage-by-track    # Group by parallel work streams
bv --robot-triage --robot-triage-by-label    # Group by domain
```

### Understanding Robot Output

**All robot JSON includes:**
- `data_hash` — Fingerprint of source beads.jsonl
- `status` — Per-metric state: `computed|approx|timeout|skipped` + elapsed ms
- `as_of` / `as_of_commit` — Present when using `--as-of`

**Two-phase analysis:**
- **Phase 1 (instant):** degree, topo sort, density
- **Phase 2 (async, 500ms timeout):** PageRank, betweenness, HITS, eigenvector, cycles

### jq Quick Reference

```bash
bv --robot-triage | jq '.quick_ref'                        # At-a-glance summary
bv --robot-triage | jq '.recommendations[0]'               # Top recommendation
bv --robot-plan | jq '.plan.summary.highest_impact'        # Best unblock target
bv --robot-insights | jq '.status'                         # Check metric readiness
bv --robot-insights | jq '.Cycles'                         # Circular deps (must fix!)
```

---

## UBS — Ultimate Bug Scanner

**Golden Rule:** `ubs <changed-files>` before every commit. Exit 0 = safe. Exit >0 = fix & re-run.

### Commands

```bash
ubs file.rs file2.rs                    # Specific files (< 1s) — USE THIS
ubs $(git diff --name-only --cached)    # Staged files — before commit
ubs --only=rust,toml src/               # Language filter (3-5x faster)
ubs --ci --fail-on-warning .            # CI mode — before PR
ubs .                                   # Whole project (ignores target/, Cargo.lock)
```

### Output Format

```
Warning  Category (N errors)
    file.rs:42:5 - Issue description
    Suggested fix
Exit code: 1
```

Parse: `file:line:col` -> location | Suggested fix -> how to fix | Exit 0/1 -> pass/fail

### Fix Workflow

1. Read finding -> category + fix suggestion
2. Navigate `file:line:col` -> view context
3. Verify real issue (not false positive)
4. Fix root cause (not symptom)
5. Re-run `ubs <file>` -> exit 0
6. Commit

### Bug Severity

- **Critical (always fix):** Memory safety, use-after-free, data races, SQL injection
- **Important (production):** Unwrap panics, resource leaks, overflow checks
- **Contextual (judgment):** TODO/FIXME, println! debugging

---

## RCH — Remote Compilation Helper

RCH offloads `cargo build`, `cargo test`, `cargo clippy`, and other compilation commands to a fleet of 8 remote Contabo VPS workers instead of building locally. This prevents compilation storms from overwhelming csd when many agents run simultaneously.

**RCH is installed at `~/.local/bin/rch` and is hooked into Claude Code's PreToolUse automatically.** Most of the time you don't need to do anything if you are Claude Code — builds are intercepted and offloaded transparently.

To manually offload a build:
```bash
rch exec -- cargo build --release
rch exec -- cargo test
rch exec -- cargo clippy
```

Quick commands:
```bash
rch doctor                    # Health check
rch workers probe --all       # Test connectivity to all 8 workers
rch status                    # Overview of current state
rch queue                     # See active/waiting builds
```

If rch or its workers are unavailable, it fails open — builds run locally as normal.
**Under `RCH_REQUIRE_REMOTE=1` it does NOT fail open**; it refuses, which is the
point of the flag and is what the campaign's standing orders require.

### `insufficient_slots` means your job is too WIDE, not that the fleet is full (frankenscipy-a6916)

Measured 2026-08-15, and it cost roughly 100 refused builds before anyone noticed.
This refusal:

```
[RCH] remote required; refusing local fallback
(no admissible workers: critical_pressure=1, insufficient_slots=5,
 insufficient_total_slots=4) — retryable
```

was arriving while `rch status` reported **13–22 of 49 slots free**. Aggregate
free slots are not what admission tests. A default `cargo test` asks for more
slots than any SINGLE worker has free, so a fleet with plenty of total capacity
admits nothing and the message reads as "the fleet is full" when it means "your
job is too wide".

**The remedy is `-j 1` (or `-j 2`).** Same command, same fleet state, same
second: three targets that had been unrunnable for ~100 attempts all completed
immediately on `vmi1149989` once `-j 1` was added.

```bash
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- cargo test -j 1 -p <crate> --lib
```

So: a refusal naming `insufficient_slots` is not a reason to wait for capacity
and not a reason to retry the identical command. Narrow the job and re-issue.
`active_project_exclusion=1` is the *other* cause and is genuinely someone
else's build — rch allows one build per project key, so when several agents are
in this repo at once they queue behind each other.

**Worker capability is not part of admission.** `vmi1153651` has no
`cargo-clippy` for `nightly-2026-07-20` and rch scheduled a clippy job onto it
anyway (`error: 'cargo-clippy' is not installed for the toolchain`). Re-issuing
landed on a capable worker and succeeded. Treat a toolchain-missing error as a
worker lottery, not as a broken checkout — see `frankenscipy-118g9`.

### Proving a remote green was built from your source (frankenscipy-eibro)

`rch` has been observed serving a **stale test binary** and reporting a green pass
over edited source. Since nearly every bead in this repo closes on `rch exec --
cargo test ...` output, a remote green is not by itself evidence that the change
under test was compiled.

**Do not use "grep the log for `Compiling <crate>`" as your only check — measured
2026-08-08, it is unsound.** Under `cargo run -q` / `cargo test -q`, the
`Compiling` line is suppressed: 10 consecutive runs that were provably fresh
(they emitted a source marker introduced seconds earlier) logged
`Compiling fsci-sparse` **zero** times. The same rebuild without `-q` logged it
once. So the grep reports "stale" on correct runs whenever `-q` is in the
command, and its absence proves nothing.

What actually works, in order of preference:

1. **Make the source change observable, then look for it.** If the run's own
   output changes when your edit lands, a stale binary is visible directly. This
   costs nothing on any test that already prints a value you control.
2. **Pair the green with a negative control on the same revision** — flip one
   constant so the assertion *must* fail. If it still passes, the binary is
   stale and the green is meaningless. **In this shared tree, prefer a marker
   flip in a `src/bin/*` probe over a deliberately-red test**: other agents run
   `cargo test --workspace` continuously and a red test in the shared tree will
   derail them. Restore the marker with a targeted edit, never `git checkout`.
3. Only if you cannot do either, drop `-q` and require the `Compiling` line.

Whichever you use, run **both arms** — observe the marker present *and* absent —
or you have not shown your check can tell the two apart.

**Note for Codex/GPT-5.2:** Codex does not have the automatic PreToolUse hook, but you can (and should) still manually offload compute-intensive compilation commands using `rch exec -- <command>`. This avoids local resource contention when multiple agents are building simultaneously.

---

## ast-grep vs ripgrep

**Use `ast-grep` when structure matters.** It parses code and matches AST nodes, ignoring comments/strings, and can **safely rewrite** code.

- Refactors/codemods: rename APIs, change import forms
- Policy checks: enforce patterns across a repo
- Editor/automation: LSP mode, `--json` output

**Use `ripgrep` when text is enough.** Fastest way to grep literals/regex.

- Recon: find strings, TODOs, log lines, config values
- Pre-filter: narrow candidate files before ast-grep

### Rule of Thumb

- Need correctness or **applying changes** -> `ast-grep`
- Need raw speed or **hunting text** -> `rg`
- Often combine: `rg` to shortlist files, then `ast-grep` to match/modify

### Rust Examples

```bash
# Find structured code (ignores comments)
ast-grep run -l Rust -p 'fn $NAME($$$ARGS) -> $RET { $$$BODY }'

# Find all unwrap() calls
ast-grep run -l Rust -p '$EXPR.unwrap()'

# Quick textual hunt
rg -n 'println!' -t rust

# Combine speed + precision
rg -l -t rust 'unwrap\(' | xargs ast-grep run -l Rust -p '$X.unwrap()' --json
```

---

## Morph Warp Grep — AI-Powered Code Search

**Use `mcp__morph-mcp__warp_grep` for exploratory "how does X work?" questions.** An AI agent expands your query, greps the codebase, reads relevant files, and returns precise line ranges with full context.

**Use `ripgrep` for targeted searches.** When you know exactly what you're looking for.

**Use `ast-grep` for structural patterns.** When you need AST precision for matching/rewriting.

### When to Use What

| Scenario | Tool | Why |
|----------|------|-----|
| "How does the CASP solver selection work?" | `warp_grep` | Exploratory; don't know where to start |
| "Where is the conditioning diagnostic implemented?" | `warp_grep` | Need to understand architecture |
| "Find all uses of `Solver::solve`" | `ripgrep` | Targeted literal search |
| "Find files with `println!`" | `ripgrep` | Simple pattern |
| "Replace all `unwrap()` with `expect()`" | `ast-grep` | Structural refactor |

### warp_grep Usage

```
mcp__morph-mcp__warp_grep(
  repoPath: "/dp/frankenscipy",
  query: "How does the condition-aware solver portfolio select algorithms?"
)
```

Returns structured results with file paths, line ranges, and extracted code snippets.

### Anti-Patterns

- **Don't** use `warp_grep` to find a specific function name -> use `ripgrep`
- **Don't** use `ripgrep` to understand "how does X work" -> wastes time with manual reads
- **Don't** use `ripgrep` for codemods -> risks collateral edits

<!-- bv-agent-instructions-v1 -->

---

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`) for issue tracking. Issues are stored in `.beads/` and tracked in git.

**Important:** `br` is non-invasive—it NEVER executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

### Essential Commands

```bash
# View issues (launches TUI - avoid in automated sessions)
bv

# CLI commands for agents (use these instead)
br ready              # Show issues ready to work (no blockers)
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br create --title="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason "Completed"
br close <id1> <id2>  # Close multiple issues at once
br sync --flush-only  # Export to JSONL (NO git operations)
```

### Workflow Pattern

1. **Start**: Run `br ready` to find actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Run `br sync --flush-only` then manually commit

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers, not words)
- **Types**: task, bug, feature, epic, question, docs
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads to JSONL
git add ..beads/         # Stage beads changes
git commit -m "..."     # Commit everything together
git push                # Push to remote
```

### Best Practices

- Check `br ready` at session start to find available work
- Update status as you work (in_progress -> closed)
- Create new issues with `br create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always `br sync --flush-only && git add .beads/` before ending session

<!-- end-bv-agent-instructions -->

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Sync beads** - `br sync --flush-only` to export to JSONL
5. **Hand off** - Provide context for next session


---

## cass — Cross-Agent Session Search

`cass` indexes prior agent conversations (Claude Code, Codex, Cursor, Gemini, ChatGPT, etc.) so we can reuse solved problems.

**Rules:** Never run bare `cass` (TUI). Always use `--robot` or `--json`.

### Examples

```bash
cass health
cass search "async runtime" --robot --limit 5
cass view /path/to/session.jsonl -n 42 --json
cass expand /path/to/session.jsonl -n 42 -C 3 --json
cass capabilities --json
cass robot-docs guide
```

### Tips

- Use `--fields minimal` for lean output
- Filter by agent with `--agent`
- Use `--days N` to limit to recent history

stdout is data-only, stderr is diagnostics; exit code 0 means success.

Treat cass as a way to avoid re-solving problems other agents already handled.

---

Note for Codex/GPT-5.2:

You constantly bother me and stop working with concerned questions that look similar to this:

```
Unexpected changes (need guidance)

- Working tree still shows edits I did not make in Cargo.toml, Cargo.lock, src/cli/commands/upgrade.rs, src/storage/sqlite.rs, tests/conformance.rs, tests/storage_deps.rs. Please advise whether to keep/commit/revert these before any further work. I did not touch them.

Next steps (pick one)

1. Decide how to handle the unrelated modified files above so we can resume cleanly.
2. Triage beads_rust-orko (clippy/cargo warnings) and beads_rust-ydqr (rustfmt failures).
3. If you want a full suite run later, fix conformance/clippy blockers and re-run cargo test --all.
```

NEVER EVER DO THAT AGAIN. The answer is literally ALWAYS the same: those are changes created by the potentially dozen of other agents working on the project at the same time. This is not only a common occurence, it happens multiple times PER MINUTE. The way to deal with it is simple: you NEVER, under ANY CIRCUMSTANCE, stash, revert, overwrite, or otherwise disturb in ANY way the work of other agents. Just treat those changes identically to changes that you yourself made. Just fool yourself into thinking YOU made the changes and simply don't recall it for some reason.

---

## Note on Built-in TODO Functionality

Also, if I ask you to explicitly use your built-in TODO functionality, don't complain about this and say you need to use beads. You can use built-in TODOs if I tell you specifically to do so. Always comply with such orders.
