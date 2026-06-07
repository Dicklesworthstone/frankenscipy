use std::hint::black_box;
use std::time::Instant;
use fsci_stats::multiscale_graphcorr;
fn data(n: usize, d: usize, s: f64) -> Vec<Vec<f64>> {
    (0..n).map(|i| (0..d).map(|j| (i as f64*0.013+j as f64*0.7+s).sin()).collect()).collect()
}
fn main() {
    let sizes: Vec<(usize, usize)> = match std::env::args().nth(1) {
        Some(s) => s
            .split(',')
            .map(|p| {
                let mut it = p.split('x');
                let n = it.next().unwrap().parse().unwrap();
                let d = it.next().unwrap().parse().unwrap();
                (n, d)
            })
            .collect(),
        None => vec![(1500usize, 16usize), (2500, 24)],
    };
    for (n, d) in sizes {
        let x = data(n,d,0.0); let y = data(n,d,1.0);
        black_box(multiscale_graphcorr(&x,&y,0,None).unwrap());
        let it=3; let st=Instant::now();
        for _ in 0..it { black_box(multiscale_graphcorr(&x,&y,0,None).unwrap()); }
        println!("mgc n={n} d={d}: {:.1} ms/run", st.elapsed().as_secs_f64()*1e3/it as f64);
    }
}
