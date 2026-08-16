use fsci_ndimage::{BoundaryMode, NdArray, map_coordinates, zoom};
fn dump(name: &str, v: &[f64]) {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
    println!("{name}|{}", s.join(";"));
}
fn main() {
    let m = BoundaryMode::Mirror;
    let input = NdArray::new(
        vec![1.0, 2.0, 4.0, 7.0, 3.0, 9.0, 5.0, 8.0, 2.0, 6.0],
        vec![10],
    )
    .unwrap();
    let coords = vec![-0.7, 0.5, 2.3, 5.5, 7.9, 9.0, 1.1, 8.8];
    for order in [2usize, 3, 4, 5] {
        dump(
            &format!("mc_o{order}"),
            &map_coordinates(&input, std::slice::from_ref(&coords), order, m, 0.0).unwrap(),
        );
    }
    // 2-D map_coordinates
    let img = NdArray::new(
        vec![
            1.0, 2.0, 4.0, 7.0, 3.0, 9.0, 5.0, 8.0, 1.0, 4.0, 3.0, 7.0, 2.0, 6.0, 1.0, 5.0, 8.0,
            3.0, 4.0, 9.0,
        ],
        vec![4, 5],
    )
    .unwrap();
    let r = vec![0.5, 1.7, 2.3, 3.1];
    let c = vec![1.2, 3.4, 0.6, 4.2];
    dump(
        "mc2d_o3",
        &map_coordinates(&img, &[r, c], 3, m, 0.0).unwrap(),
    );
    // zoom 2x order 3
    if let Ok(z) = zoom(&input, &[2.0], 3, m, 0.0) {
        dump("zoom_o3", &z.data);
    }
}
