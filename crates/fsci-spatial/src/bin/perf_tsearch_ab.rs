use fsci_spatial::{Delaunay, tsearch};
use std::io::Write;
use std::time::Instant;
struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}
fn main() {
    let mut r = Lcg(11);
    for &n in &[3000usize, 10000] {
        let pts: Vec<Vec<f64>> = (0..n).map(|_| vec![r.u(), r.u()]).collect();
        let nq = 200000usize;
        let xi: Vec<Vec<f64>> = (0..nq).map(|_| vec![r.u(), r.u()]).collect();
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(format!("/tmp/del_{n}.f64")).unwrap());
        for p in pts.iter().chain(&xi) {
            f.write_all(&p[0].to_le_bytes()).unwrap();
            f.write_all(&p[1].to_le_bytes()).unwrap();
        }
        drop(f);
        let tb = {
            let mut b = std::time::Duration::MAX;
            for _ in 0..4 {
                let t = Instant::now();
                let d = Delaunay::new(&pts).unwrap();
                let e = t.elapsed();
                std::hint::black_box(d.simplices.len());
                if e < b {
                    b = e;
                }
            }
            b
        };
        let tri = Delaunay::new(&pts).unwrap();
        let tq = {
            let mut b = std::time::Duration::MAX;
            for _ in 0..6 {
                let t = Instant::now();
                let s = tsearch(&tri, &xi).unwrap();
                let e = t.elapsed();
                std::hint::black_box(s.len());
                if e < b {
                    b = e;
                }
            }
            b
        };
        println!(
            "n={n} nq={nq}  fsci_build={tb:>9.3?}  fsci_tsearch={tq:>9.3?}  nsimplex={}",
            tri.simplices.len()
        );
    }
}
