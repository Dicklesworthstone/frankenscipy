# Parked A/B drivers — written, never compiled

`frankenscipy-5f06d`. These two `#[cfg(test)] mod` blocks are finished driver
code for the last two A/B switches outside the sparse lane. They are parked here
rather than in their crates because **the rch fleet refused the build twice**:

```
[RCH] remote required; refusing local fallback
  (no admissible workers: critical_pressure=1,insufficient_slots=2,insufficient_total_slots=7)
```

That is a REFUSAL, not a failure — it wrote no compiler output at all, so these
blocks have never been type-checked, let alone run.

## Why parked rather than committed into the crates

Uncompiled *test* code in a crate is not the same risk as an uncompiled doc
comment. If either block fails to compile, `cargo test -p fsci-ndimage` and
`cargo test -p fsci-spatial` break for **every pane in the fleet**, not just mine.
That blast radius is not mine to spend on unverified code, so the crates were
restored to exactly their committed state (verified: empty `git diff`).

Nothing under `tests/artifacts/` is a Cargo target — Cargo only picks up
`tests/*.rs` at the top level of a package, not nested subdirectories — so parking
them here compiles nothing and breaks nothing.

## How to land them

Append each file verbatim to the end of the named crate's `lib.rs`, then run one
build. No other edit is needed; the imports at the top of each module are written
against the current API.

| file | append to | drives |
|---|---|---|
| `5f06d_ndimage_nd_filter_driver.rs` | `crates/fsci-ndimage/src/lib.rs` | `ND_FILTER_FORCE_SCALAR` |
| `5f06d_spatial_mahalanobis_driver.rs` | `crates/fsci-spatial/src/lib.rs` | `MAHALANOBIS_ASSEMBLY_FORCE_SERIAL` |

```
RCH_REQUIRE_REMOTE=1 RCH_CARGO_WRAPPER_BYPASS=1 env -u CARGO_TARGET_DIR \
  rch exec -- cargo test -p fsci-ndimage -p fsci-spatial --lib -- --nocapture toggle_ab_
```

Then re-run `python3 scripts/toggle_driver_census.py fsci-ndimage fsci-spatial`,
which should report 0 in the "EXERCISED NOWHERE" column for both.

## What to expect, and what would be a real finding

Both toggles are documented bit-identical and both are structurally entitled to
be: the ndimage SIMD interior sums the same taps in the same k-order with no FMA
contraction, and the mahalanobis parallel arm writes disjoint row slices and
combines no floats across threads. **A red here is a defect, not rounding.**

Both drivers deliberately assert more than "the two arms agree", because an
arms-only comparison cannot catch a fault both arms share:

* the ndimage driver checks one interior pixel against the correlation definition
  computed directly from the input;
* the mahalanobis driver checks the final cell against the closed form, which is
  valid because an identity `vi` makes the Mahalanobis distance the Euclidean one.

Fixture sizing is the part most likely to be "simplified" into uselessness:

* `ND_FILTER_FORCE_SCALAR` has **no size gate** but needs an innermost run wider
  than `8 + kernel_width`, or every pixel takes the border fallback and the SIMD
  interior — the thing the toggle gates — never runs. Guarded by `const` assert.
* `MAHALANOBIS_ASSEMBLY_FORCE_SERIAL` gates on **cells**, `na * nb >= 1 << 22`.
  Dimension is free, so `d` is 2 on purpose: the compute underneath is
  `O(na*nb*d)` and a bigger `d` costs time without strengthening the comparison.
