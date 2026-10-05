# Genuine published v0.2.0 writer fixtures

These five inputs preserve the exact bytes emitted by an independently built
consumer of published `fsci-*` 0.2.0 registry packages. Keep them unchanged;
regenerating them with a current writer would lose the previous-writer control.

`real-v4.mat` contains a real double 2×3 rectangular matrix and a 3×3 linear
system. `system-stored.npz` and `system-deflated.npz` contain the same seven-entry
CSR matrix in ZIP stored and deflated forms. `golden.json` retains numerical
outputs, FFT bins, an ODE state, filter continuation state, and CoreArray physical
and logical layouts. `audited-solve.json` preserves an actual serialized old
audit event, including its timestamp, fingerprint, action, and outcome.

The old `rectangular.mtx` is byte-for-byte identical to the existing
[`../mmwrite_dense.mtx`](../mmwrite_dense.mtx), so that file is reused. The
original name in `golden.json` is a producer declaration; the provenance maps it
to its existing repository fixture.

The custody digests and exact input hashes are in `PROVENANCE.json`. Stock SciPy
1.18.1 and NumPy 2.5.3 independently decoded MAT, Matrix Market, and both NPZ files
and checked historical numerical-state invariants. Those checks establish input
authenticity/readability. They do not qualify current FrankenSciPy readers,
restore adaptive ODE internals, or constitute current numerical/performance proof.

Native current-reader and serialized-state controls are follow-up work. Existing
same-version round-trip tests and the current shifted-Laplacian GMRES case remain
the active coverage; no unqualified Rust tests or release-driver assumptions were
imported. The timestamped producer, registry receipts, original report and full
consumer remain in the preserved recovery evidence.
