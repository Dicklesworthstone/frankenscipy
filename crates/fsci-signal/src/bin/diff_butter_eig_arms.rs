//! Which `eig` arm designs filters closer to SciPy? — blocks shipping frankenscipy-sez4r.
//!
//! `butter` places its poles through `poly_roots`, which builds a companion matrix and
//! calls `fsci_linalg::eig`. So the Schur routine chosen by `EIG_USE_FRANCIS_SCHUR`
//! propagates into every filter coefficient, and flipping that default moved a
//! Butterworth SOS coefficient by 1.3e-6 against SciPy's reference — enough to fail
//! conformance and to block an otherwise-good change.
//!
//! This prints the SOS sections under BOTH arms so the Python side can score each
//! against `scipy.signal.butter` in the same invocation. The question is not "do they
//! differ" — they do — but "which one is closer to the incumbent, and by how much",
//! which is the only form that decides whether the Francis path can ship.
//!
//! Exact coefficient values, not timings: this is a correctness comparison and reads the
//! same on a loaded host.

use fsci_linalg::EIG_USE_FRANCIS_SCHUR;
use fsci_signal::{FilterType, butter};
use std::sync::atomic::Ordering;

const MARKER: &str = "butter-eig-arms-v1";

fn main() {
    println!("MARKER={MARKER}");
    let cases: Vec<(usize, f64, FilterType, &str)> = vec![
        (4, 0.2, FilterType::Lowpass, "lp4_0.2"),
        (6, 0.3, FilterType::Lowpass, "lp6_0.3"),
        (8, 0.15, FilterType::Lowpass, "lp8_0.15"),
        (5, 0.4, FilterType::Highpass, "hp5_0.4"),
        (10, 0.25, FilterType::Lowpass, "lp10_0.25"),
        (12, 0.1, FilterType::Lowpass, "lp12_0.1"),
    ];

    for (arm, francis) in [("nalgebra", false), ("francis", true)] {
        EIG_USE_FRANCIS_SCHUR.store(francis, Ordering::Relaxed);
        for (order, wn, btype, label) in &cases {
            // The BA (transfer-function) form, NOT sos. Second-order sections carry no
            // canonical ORDER, so an element-wise diff against SciPy compares different
            // sections and reports nonsense — the first pass at this printed relative
            // errors of 1e11 for exactly that reason. `b` and `a` are ordered by
            // descending power, which is canonical, so the comparison is meaningful.
            match butter(*order, &[*wn], *btype) {
                Ok(ba) => {
                    let flat: Vec<String> = ba
                        .b
                        .iter()
                        .chain(ba.a.iter())
                        .map(|v| format!("{v:.17e}"))
                        .collect();
                    println!("BA arm={arm} case={label} nb={} {}", ba.b.len(), flat.join(","));
                }
                Err(e) => println!("SOS arm={arm} case={label} ERROR {e:?}"),
            }
        }
    }
    EIG_USE_FRANCIS_SCHUR.store(false, Ordering::Relaxed);
}
