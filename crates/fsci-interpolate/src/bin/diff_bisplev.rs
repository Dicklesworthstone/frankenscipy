//! bisplev probe vs scipy.interpolate.bisplev — the SAME tck (from
//! scipy.bisplrep) is fed to both libraries so only the evaluation math is
//! compared. Lines: `i,j,value`.
use fsci_interpolate::bisplev;

fn main() {
    let tx = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let ty = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let c = vec![
        0.0007036914835632136,
        0.005926321740951743,
        -0.06160310345403084,
        -0.0001221843016128943,
        0.9557129046360605,
        1.62140389861632,
        0.23776490140582782,
        -0.26516573732626564,
        1.6567096611867258,
        1.262245622926786,
        0.6532855323355665,
        -0.402172884515181,
        0.14105879414771977,
        0.08629198314637655,
        0.5637006125242106,
        0.5145217038568174,
    ];
    let tck = (tx, ty, c, 3usize, 3usize);

    let x = [0.05_f64, 0.25, 0.5, 0.7, 0.9, 1.0];
    let y = [0.0_f64, 0.15, 0.5, 0.65, 1.0];
    let z = bisplev(&x, &y, &tck).expect("bisplev");
    for (i, row) in z.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            println!("{i},{j},{v:.17e}");
        }
    }
}
