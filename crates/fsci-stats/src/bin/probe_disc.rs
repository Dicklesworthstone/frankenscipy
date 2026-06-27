use fsci_stats::{
    centered_discrepancy, centered_discrepancy_iterative, l2_star_discrepancy, mixture_discrepancy,
    wraparound_discrepancy,
};
fn main() {
    // 6x3 sample in [0,1], deterministic
    let n = 6;
    let d = 3;
    let s: Vec<f64> = (0..n * d)
        .map(|i| ((i as f64 * 0.123 + 0.07).sin().abs()))
        .collect();
    println!("CD {:.12}", centered_discrepancy(&s, d).unwrap());
    println!("WD {:.12}", wraparound_discrepancy(&s, d).unwrap());
    println!("MD {:.12}", mixture_discrepancy(&s, d).unwrap());
    println!("L2 {:.12}", l2_star_discrepancy(&s, d).unwrap());
    println!(
        "CDit {:.12}",
        centered_discrepancy_iterative(&s, d).unwrap()
    );
    // print sample for python
    print!("S");
    for v in &s {
        print!(" {:.17}", v);
    }
    println!();
}
