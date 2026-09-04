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

use fsci_ndimage::{BoundaryMode, NdArray, map_coordinates, rotate};

fn main() {
    let data: Vec<f64> = (0..25).map(|i| i as f64).collect();
    let arr = NdArray::new(data, vec![5, 5]).unwrap();
    let coords: Vec<Vec<f64>> = vec![vec![0.0, 0.0, 4.0, 4.0, 2.0], vec![4.0, 0.0, 0.0, 4.0, 2.0]];
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
    for (name, ang, order, mode) in [
        ("rot0 o0 wrap", 0.0, 0, BoundaryMode::Wrap),
        ("rot90 o0 wrap", 90.0, 0, BoundaryMode::Wrap),
        ("rot90 o0 const", 90.0, 0, BoundaryMode::Constant),
        ("rot180 o0 wrap", 180.0, 0, BoundaryMode::Wrap),
        ("rot90 o3 wrap", 90.0, 3, BoundaryMode::Wrap),
    ] {
        let r = rotate(&arr, ang, false, order, mode, 0.0).unwrap();
        println!("PROBE {name}: {:?}", r.data);
    }
    println!(
        "SCIPY rot90 o0: [4, 9, 14, 19, 24, 3, 8, 13, 18, 23, 2, 7, 12, 17, 22, 1, 6, 11, 16, 21, 0, 5, 10, 15, 20]"
    );
    println!("SCIPY rot0: identity; SCIPY rot180: reversed");
}
