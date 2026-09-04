//! scipy-parity probe for `map_coordinates` at order-0 boundary coordinates.
//!
//! Prints fsci output beside the values the pinned scipy 1.17.1 incumbent
//! returns (probed 2026-09-04 for the 5x5 arange input, bead frankenscipy-0y8z8
//! follow-up):
//!
//! ```text
//! coords (x=row, y=col): (0,4) (0,0) (4,0) (4,4) (2,2)
//! scipy wrap     order 0: [4, 0, 20, 24, 12]
//! scipy constant order 0: [4, 0, 20, 24, 12]
//! scipy wrap     order 3: [4.000000000000001, 8.287021862380636e-16, 20.0, 24.0, 12.000000000000005]
//! ```
//!
//! Run from the repo root:
//! `cargo run -p fsci-ndimage --bin rotprobe`
#![forbid(unsafe_code)]

use fsci_ndimage::{map_coordinates, BoundaryMode, NdArray};

fn main() {
    let data: Vec<f64> = (0..25).collect();
    let arr = NdArray::new(data, vec![5, 5]).unwrap();
    let coords: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 4.0, 4.0, 2.0],
        vec![4.0, 0.0, 0.0, 4.0, 2.0],
    ];
    for (order, mode, label) in [
        (0, BoundaryMode::Wrap, "wrap o0"),
        (0, BoundaryMode::Constant, "const o0"),
        (3, BoundaryMode::Wrap, "wrap o3"),
    ] {
        let r = map_coordinates(&arr, &coords, order, mode, 0.0).unwrap();
        println!("PROBE {label}: {r:?}");
    }
    println!("SCIPY wrap o0: [4.0, 0.0, 20.0, 24.0, 12.0]");
    println!("SCIPY const o0: [4.0, 0.0, 20.0, 24.0, 12.0]");
    println!(
        "SCIPY wrap o3: [4.000000000000001, 8.287021862380636e-16, 20.0, 24.0, 12.000000000000005]"
    );
}
