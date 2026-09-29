use fsci_spatial::Delaunay;
fn main() {
    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![f64::NAN, f64::NAN],
        vec![0.0, 1.0],
    ];
    let delaunay = Delaunay::new(&points);
    println!("{:?}", delaunay.is_err());
}
