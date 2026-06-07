// Debug scratch bin: scipy reference-value literals carry extra precision.
#![allow(clippy::excessive_precision)]

fn main() {
    // scipy.special.zeta(s) reference values
    let r_cases = [
        (1.05_f64, 20.580844302036990_f64),
        (1.1, 10.584448464950802),
        (1.2, 5.591582441177753),
        (1.5, 2.6123753486854882),
        (2.0, 1.6449340668482264),
        (3.0, 1.2020569031595942),
        (5.0, 1.0369277551433700),
        (10.0, 1.0009945751278180),
    ];
    println!("=== Riemann zeta (single arg) ===");
    for (s, expected) in r_cases {
        let z = fsci_special::zeta(s);
        let diff = (z - expected).abs();
        println!("zeta({:>4}) = {:.16} (diff {:.2e})", s, z, diff);
    }

    println!("=== Hurwitz zeta ===");
    // scipy.special.zeta(s, a) reference values
    let h_cases = [
        (1.1_f64, 1.0_f64, 10.584448464950802_f64),
        (1.1, 2.0, 9.584448464950798),
        (1.1, 0.5, 12.103813495683745),
        (1.5, 2.0, 1.6123753486854886),
        (2.0, 0.5, 4.934802200544680),
        (2.0, 3.0, 0.3949340668482265),
        (3.0, 1.5, 0.4143983221171600),
    ];
    for (s, a, expected) in h_cases {
        let z = fsci_special::hurwitz_zeta(s, a);
        let diff = (z - expected).abs();
        println!("zeta({:>4}, {}) = {:.16} (diff {:.2e})", s, a, z, diff);
    }
}
