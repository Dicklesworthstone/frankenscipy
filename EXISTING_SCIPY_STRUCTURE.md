# EXISTING_SCIPY_STRUCTURE

## 1. Legacy Oracle

- Root: /dp/frankenscipy/legacy_scipy_code/scipy
- Upstream: scipy/scipy

## 2. Subsystem Map

- scipy/integrate: IVP and integration routines.
- scipy/linalg: dense algebra wrappers over BLAS/LAPACK.
- scipy/optimize: root/minimize ecosystem with MINPACK and newer methods.
- scipy/sparse (+ sparse/linalg): sparse structures and solver paths.
- scipy/special: special-function wrappers and error handling.
- scipy/fft with subprojects/pocketfft: transform backend and wrappers.
- scipy/_lib: array-api negotiation, callbacks, utility internals.
- scipy/subprojects: bundled third-party numeric runtimes.

## 3. Semantic Hotspots (Must Preserve)

1. solve_ivp adaptive stepping, event handling, and tolerance scaling.
2. linalg factorization workspace/pivot semantics and error signaling.
3. optimize root/minpack option handling and status interpretation.
4. special-function error state controls and propagation behavior.
5. sparse array/matrix dual semantics and index dtype constraints.
6. array API backend negotiation behavior in _lib.

## 4. Compatibility-Critical Behaviors

- API-level option dictionaries and result shapes for optimize/integrate/linalg routines.
- tolerance and convergence semantics for scoped solver families.
- sparse constructor and operator behavior across formats.
- backend namespace compatibility expectations.

## 5. Security and Stability Risk Areas

- BLAS/LAPACK and Fortran wrapper memory-safety assumptions.
- special-function wrappers with mixed C/Cython/Fortran layers.
- third-party subproject integration and thread/memory semantics.
- callback lifecycle correctness in _lib callback helpers.

## 6. V1 Extraction Boundary

Include now:
- integrate/linalg/optimize/sparse/fft scoped families and array-api helpers needed for them.

Exclude for V1:
- full breadth modules (stats/spatial/ndimage/io etc.), docs/tooling/bench datasets, full third-party replacement breadth.

## 7. High-Value Conformance Fixture Families

- integrate/_ivp/tests for solver and event behavior.
- linalg/tests for decomposition and low-level wrappers.
- optimize/tests for method option and convergence contracts.
- fft/tests for backend and transform parity.
- sparse/tests and _lib/tests for structure and utility parity.

## 8. Extraction Notes for Rust Spec

- Maintain per-algorithm tolerance contracts explicitly.
- Prioritize deterministic convergence/error semantics over speed.
- Use differential fixture bundles before introducing deep optimization.
