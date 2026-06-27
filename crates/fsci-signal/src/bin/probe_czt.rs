use fsci_signal::{CZT, ZoomFFT};
fn xin(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|k| ((0.3 * k as f64 + 1.0).sin(), (0.17 * k as f64).cos()))
        .collect()
}
fn pr(tag: &str, v: &[(f64, f64)]) {
    print!("{tag}");
    for c in v {
        print!(" {:.12} {:.12}", c.0, c.1);
    }
    println!();
}
fn main() {
    // CZT default (FFT-like), n=7 m=7
    let c = CZT::new(7, None, None, None).unwrap();
    pr("CZT_default", &c.transform(&xin(7)).unwrap());
    pr("CZT_default_pts", &c.points());
    // CZT m>n, n=6 m=10
    let c = CZT::new(6, Some(10), None, None).unwrap();
    pr("CZT_m10", &c.transform(&xin(6)).unwrap());
    // CZT custom w,a (spiral): w=0.99*exp(-2pi i/12), a=1.1*exp(0.1i)
    let wm = 0.99f64;
    let wa = -2.0 * std::f64::consts::PI / 12.0;
    let w = (wm * wa.cos(), wm * wa.sin());
    let am = 1.1f64;
    let aa = 0.1f64;
    let a = (am * aa.cos(), am * aa.sin());
    let c = CZT::new(8, Some(5), Some(w), Some(a)).unwrap();
    pr("CZT_spiral", &c.transform(&xin(8)).unwrap());
    pr("CZT_spiral_pts", &c.points());
    // ZoomFFT n=16 m=8 fn=(0.2,0.6) fs=2 endpoint=false
    let z = ZoomFFT::new(16, (0.2, 0.6), Some(8), 2.0, false).unwrap();
    pr("ZOOM", &z.transform(&xin(16)).unwrap());
    // ZoomFFT endpoint=true, scalar fn -> (0,0.5)
    let z = ZoomFFT::new(16, (0.0, 0.5), Some(8), 2.0, true).unwrap();
    pr("ZOOM_ep", &z.transform(&xin(16)).unwrap());
}
