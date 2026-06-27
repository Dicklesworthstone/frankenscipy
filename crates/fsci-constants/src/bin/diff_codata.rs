//! Differential: fsci-constants vs scipy.constants (CODATA 2022). Gitignored.
use fsci_constants::*;
fn main() {
    let pairs: &[(&str, f64)] = &[
        ("vacuum mag. permeability", MU_0),
        ("vacuum electric permittivity", EPSILON_0),
        ("Rydberg constant", RYDBERG),
        ("electron mass", ELECTRON_MASS),
        ("proton mass", PROTON_MASS),
        ("neutron mass", NEUTRON_MASS),
        ("atomic mass constant", ATOMIC_MASS),
        ("Bohr magneton", BOHR_MAGNETON),
        ("nuclear magneton", NUCLEAR_MAGNETON),
        ("mag. flux quantum", MAGNETIC_FLUX_QUANTUM),
        ("Josephson constant", JOSEPHSON),
        ("von Klitzing constant", VON_KLITZING),
        ("fine-structure constant", FINE_STRUCTURE),
        ("Bohr radius", BOHR_RADIUS),
        ("Hartree energy", HARTREE),
        ("classical electron radius", CLASSICAL_ELECTRON_RADIUS),
        ("Wien wavelength displacement law constant", WIEN),
        ("Stefan-Boltzmann constant", STEFAN_BOLTZMANN),
        ("Compton wavelength", COMPTON_WAVELENGTH),
    ];
    for (k, v) in pairs {
        println!("{k}\t{v:.17e}");
    }
    println!(
        "inverse fine-structure constant\t{:.17e}",
        INVERSE_FINE_STRUCTURE
    );
    println!("CHECK inv_alpha\t{:.17e}", 1.0 / FINE_STRUCTURE);
}
