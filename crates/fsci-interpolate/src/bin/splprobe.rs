//! Probe: fsci splrep/splder/splantider shapes and values vs scipy 1.17.1.
fn main() {
    let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| (v - 3.0) * (v - 3.0)).collect();
    for k in [1usize, 2, 3] {
        let (t, c, kk) = fsci_interpolate::splrep(&x, &y, k, 0.0).unwrap();
        println!("PROBE splrep k={k}: len(t)={} len(c)={}", t.len(), c.len());
        let (dt, dc, dk) = fsci_interpolate::splder(&(t.clone(), c.clone(), kk)).unwrap();
        println!(
            "PROBE splder  k={k}: len(t)={} len(c)={} k'={dk}",
            dt.len(),
            dc.len()
        );
        let (at, ac, ak) = fsci_interpolate::splantider(&(t.clone(), c.clone(), kk)).unwrap();
        println!(
            "PROBE spanti  k={k}: len(t)={} len(c)={} k'={ak}",
            at.len(),
            ac.len()
        );
    }
}
