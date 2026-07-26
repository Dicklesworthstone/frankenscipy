#![forbid(unsafe_code)]

//! Physical and mathematical constants for FrankenSciPy.
//!
//! Matches `scipy.constants` — CODATA 2022 recommended values (SciPy 1.17.x).
//!
//! # Usage
//! ```
//! use fsci_constants::*;
//! let energy = ELECTRON_MASS * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
//! ```

// ══════════════════════════════════════════════════════════════════════
// Mathematical Constants
// ══════════════════════════════════════════════════════════════════════

/// π
pub const PI: f64 = std::f64::consts::PI;
/// 2π (tau)
pub const TAU: f64 = 2.0 * std::f64::consts::PI;
/// Euler's number e
pub const E: f64 = std::f64::consts::E;
/// Golden ratio φ = (1 + √5)/2
pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;
/// Euler-Mascheroni constant γ
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

// ══════════════════════════════════════════════════════════════════════
// SI Prefixes
// ══════════════════════════════════════════════════════════════════════

pub const YOTTA: f64 = 1e24;
pub const ZETTA: f64 = 1e21;
pub const EXA: f64 = 1e18;
pub const PETA: f64 = 1e15;
pub const TERA: f64 = 1e12;
pub const GIGA: f64 = 1e9;
pub const MEGA: f64 = 1e6;
pub const KILO: f64 = 1e3;
pub const HECTO: f64 = 1e2;
pub const DEKA: f64 = 1e1;
pub const DECI: f64 = 1e-1;
pub const CENTI: f64 = 1e-2;
pub const MILLI: f64 = 1e-3;
pub const MICRO: f64 = 1e-6;
pub const NANO: f64 = 1e-9;
pub const PICO: f64 = 1e-12;
pub const FEMTO: f64 = 1e-15;
pub const ATTO: f64 = 1e-18;
pub const ZEPTO: f64 = 1e-21;
pub const YOCTO: f64 = 1e-24;

// ══════════════════════════════════════════════════════════════════════
// Fundamental Physical Constants (CODATA 2022)
// ══════════════════════════════════════════════════════════════════════

/// Speed of light in vacuum [m/s]
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// Shorthand alias
pub const C: f64 = SPEED_OF_LIGHT;

/// Magnetic constant (vacuum permeability) μ₀ [N/A²]
pub const MU_0: f64 = 1.256_637_061_27e-6;

/// Electric constant (vacuum permittivity) ε₀ [F/m]
pub const EPSILON_0: f64 = 8.854_187_818_8e-12;

/// Planck constant `J·s`
pub const PLANCK: f64 = 6.626_070_15e-34;
/// Shorthand alias
pub const H: f64 = PLANCK;

/// Reduced Planck constant ℏ = h/(2π) `J·s`
///
/// Defined as `PLANCK / TAU` so precision is automatically correct to
/// machine epsilon. The prior hand-typed `1.054_571_817e-34` truncated
/// 4-5 significant digits vs scipy.constants.hbar (6.13e-10 relative
/// error). Per frankenscipy-h0su.
pub const HBAR: f64 = PLANCK / TAU;

/// Newtonian constant of gravitation [m³/(kg·s²)]
pub const GRAVITATIONAL_CONSTANT: f64 = 6.674_30e-11;
/// Shorthand alias
pub const G: f64 = GRAVITATIONAL_CONSTANT;

/// Standard acceleration of gravity [m/s²]
pub const G_N: f64 = 9.806_65;

/// Elementary charge `C`
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
/// Shorthand alias
pub const E_CHARGE: f64 = ELEMENTARY_CHARGE;

/// Molar gas constant R [J/(mol·K)]
pub const GAS_CONSTANT: f64 = 8.314_462_618;
/// Shorthand alias
pub const R: f64 = GAS_CONSTANT;

/// Avogadro constant [1/mol]
pub const AVOGADRO: f64 = 6.022_140_76e23;
/// Shorthand alias
pub const N_A: f64 = AVOGADRO;

/// Boltzmann constant [J/K]
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// Shorthand alias
pub const K_B: f64 = BOLTZMANN;

/// Stefan-Boltzmann constant [W/(m²·K⁴)]
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419_184_431_4e-8;
/// Shorthand alias
pub const SIGMA: f64 = STEFAN_BOLTZMANN;

/// Wien displacement law constant `m·K`
pub const WIEN: f64 = 2.897_771_955_185_172_7e-3;

/// Rydberg constant [1/m]
pub const RYDBERG: f64 = 1.097_373_156_815_7e7;

// ══════════════════════════════════════════════════════════════════════
// Particle Masses
// ══════════════════════════════════════════════════════════════════════

/// Electron mass `kg`
pub const ELECTRON_MASS: f64 = 9.109_383_713_9e-31;
/// Shorthand alias
pub const M_E: f64 = ELECTRON_MASS;

/// Proton mass `kg`
pub const PROTON_MASS: f64 = 1.672_621_925_95e-27;
/// Shorthand alias
pub const M_P: f64 = PROTON_MASS;

/// Neutron mass `kg`
pub const NEUTRON_MASS: f64 = 1.674_927_500_56e-27;
/// Shorthand alias
pub const M_N: f64 = NEUTRON_MASS;

/// Atomic mass constant (1/12 of ¹²C mass) `kg`
pub const ATOMIC_MASS: f64 = 1.660_539_068_92e-27;
/// Shorthand alias
pub const U: f64 = ATOMIC_MASS;

// ══════════════════════════════════════════════════════════════════════
// Electromagnetic Constants
// ══════════════════════════════════════════════════════════════════════

/// Bohr magneton [J/T]
pub const BOHR_MAGNETON: f64 = 9.274_010_065_7e-24;

/// Nuclear magneton [J/T]
pub const NUCLEAR_MAGNETON: f64 = 5.050_783_739_3e-27;

/// Magnetic flux quantum Φ₀ = h/(2e) `Wb`
pub const MAGNETIC_FLUX_QUANTUM: f64 = 2.067_833_848_461_929_5e-15;

/// Conductance quantum G₀ = 2e²/h `S`
///
/// Derived as `2 * ELEMENTARY_CHARGE^2 / PLANCK` so precision tracks the
/// SI-exact inputs. Per frankenscipy-h0su.
pub const CONDUCTANCE_QUANTUM: f64 = 2.0 * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE / PLANCK;

/// Josephson constant K_J = 2e/h [Hz/V]
pub const JOSEPHSON: f64 = 4.835_978_484_169_836e14;

/// Von Klitzing constant R_K = h/e² `Ω`
pub const VON_KLITZING: f64 = 2.581_280_745_930_451_3e4;

// ══════════════════════════════════════════════════════════════════════
// Atomic & Nuclear Constants
// ══════════════════════════════════════════════════════════════════════

/// Fine-structure constant α
pub const FINE_STRUCTURE: f64 = 7.297_352_564_3e-3;
/// Shorthand alias
pub const ALPHA: f64 = FINE_STRUCTURE;

/// Bohr radius `m`
pub const BOHR_RADIUS: f64 = 5.291_772_105_44e-11;

/// Hartree energy `J`
pub const HARTREE: f64 = 4.359_744_722_206e-18;

/// Classical electron radius `m`
pub const CLASSICAL_ELECTRON_RADIUS: f64 = 2.817_940_320_5e-15;

/// Compton wavelength of electron `m`
pub const COMPTON_WAVELENGTH: f64 = 2.426_310_235_38e-12;
/// Explicit electron alias for the physical_constants table key.
pub const ELECTRON_COMPTON_WAVELENGTH: f64 = COMPTON_WAVELENGTH;

// ══════════════════════════════════════════════════════════════════════
// Additional Physical Constants — br-wada
// ══════════════════════════════════════════════════════════════════════
//
// CODATA 2022 / scipy.constants.physical_constants. Bit-exact parity
// against scipy is locked in via FSCI-P2C-016 fixture cases.

/// Faraday constant F = N_A · e [C·mol⁻¹]
pub const FARADAY: f64 = 96_485.332_123_310_01;

/// Electron g-factor (dimensionless, negative)
pub const ELECTRON_G_FACTOR: f64 = -2.002_319_304_360_92;

/// Proton g-factor (dimensionless)
pub const PROTON_G_FACTOR: f64 = 5.585_694_689_3;

/// Neutron g-factor (dimensionless, negative)
pub const NEUTRON_G_FACTOR: f64 = -3.826_085_52;

/// Muon g-factor (dimensionless, negative)
pub const MUON_G_FACTOR: f64 = -2.002_331_841_23;

/// Thomson cross section `m²`
pub const THOMSON_CROSS_SECTION: f64 = 6.652_458_705_1e-29;

/// Characteristic impedance of vacuum `Ω`
pub const CHARACTERISTIC_IMPEDANCE_OF_VACUUM: f64 = 376.730_313_412;

/// Deuteron mass `kg`
pub const DEUTERON_MASS: f64 = 3.343_583_776_8e-27;

/// Alpha particle mass `kg`
pub const ALPHA_PARTICLE_MASS: f64 = 6.644_657_345e-27;

/// Muon mass `kg`
pub const MUON_MASS: f64 = 1.883_531_627e-28;

/// Tau mass `kg`
pub const TAU_MASS: f64 = 3.167_54e-27;

/// Helion mass `kg`
pub const HELION_MASS: f64 = 5.006_412_786_2e-27;

/// Triton mass `kg`
pub const TRITON_MASS: f64 = 5.007_356_751_2e-27;

/// Molar volume of ideal gas (273.15 K, 101.325 kPa) [m³·mol⁻¹]
pub const MOLAR_VOLUME_IDEAL_GAS: f64 = 0.022_413_969_545_014_137;

/// Molar Planck constant [J·Hz⁻¹·mol⁻¹]
pub const MOLAR_PLANCK: f64 = 3.990_312_712_893_431e-10;

/// Rydberg constant times c in Hz `Hz`
pub const RYDBERG_HZ: f64 = 3_289_841_960_250_000.0;

/// Inverse fine-structure constant α⁻¹ (dimensionless)
pub const INVERSE_FINE_STRUCTURE: f64 = 137.035_999_177;

/// First radiation constant c₁ = 2π·h·c² [W·m²]
pub const FIRST_RADIATION_CONSTANT: f64 = 3.741_771_852_192_757_3e-16;

/// Second radiation constant c₂ = h·c/k `m·K`
pub const SECOND_RADIATION_CONSTANT: f64 = 0.014_387_768_775_039_337;

/// Electron-proton mass ratio (dimensionless)
pub const ELECTRON_PROTON_MASS_RATIO: f64 = 0.000_544_617_021_488_9;

/// Proton-electron mass ratio (dimensionless)
pub const PROTON_ELECTRON_MASS_RATIO: f64 = 1_836.152_673_426;

/// Bohr magneton in eV/T [eV·T⁻¹]
pub const BOHR_MAGNETON_EV_T: f64 = 5.788_381_798_2e-5;

// ══════════════════════════════════════════════════════════════════════
// Additional Physical Constants — br-sl60
// ══════════════════════════════════════════════════════════════════════

/// Electron mass energy equivalent `MeV`
pub const ELECTRON_MASS_MEV: f64 = 0.510_998_950_69;

/// Proton mass energy equivalent `MeV`
pub const PROTON_MASS_MEV: f64 = 938.272_089_43;

/// Neutron mass energy equivalent `MeV`
pub const NEUTRON_MASS_MEV: f64 = 939.565_421_94;

/// Muon mass energy equivalent `MeV`
pub const MUON_MASS_MEV: f64 = 105.658_375_5;

/// Tau mass energy equivalent `MeV`
pub const TAU_MASS_MEV: f64 = 1_776.82;

/// Deuteron mass energy equivalent `MeV`
pub const DEUTERON_MASS_MEV: f64 = 1_875.612_945;

/// Alpha particle mass energy equivalent `MeV`
pub const ALPHA_PARTICLE_MASS_MEV: f64 = 3_727.379_411_8;

/// Helion mass energy equivalent `MeV`
pub const HELION_MASS_MEV: f64 = 2_808.391_611_12;

/// Triton mass energy equivalent `MeV`
pub const TRITON_MASS_MEV: f64 = 2_808.921_136_68;

/// Proton Compton wavelength `m`
pub const PROTON_COMPTON_WAVELENGTH: f64 = 1.321_409_853_6e-15;

/// Neutron Compton wavelength `m`
pub const NEUTRON_COMPTON_WAVELENGTH: f64 = 1.319_590_903_82e-15;

/// Muon Compton wavelength `m`
pub const MUON_COMPTON_WAVELENGTH: f64 = 1.173_444_11e-14;

/// Tau Compton wavelength `m`
pub const TAU_COMPTON_WAVELENGTH: f64 = 6.977_71e-16;

/// Neutron-electron mass ratio (dimensionless)
pub const NEUTRON_ELECTRON_MASS_RATIO: f64 = 1_838.683_662;

/// Electron-neutron mass ratio (dimensionless)
pub const ELECTRON_NEUTRON_MASS_RATIO: f64 = 0.000_543_867_344_16;

/// Muon-electron mass ratio (dimensionless)
pub const MUON_ELECTRON_MASS_RATIO: f64 = 206.768_282_7;

/// Electron-muon mass ratio (dimensionless)
pub const ELECTRON_MUON_MASS_RATIO: f64 = 0.004_836_331_7;

/// Proton-neutron mass ratio (dimensionless)
pub const PROTON_NEUTRON_MASS_RATIO: f64 = 0.998_623_477_97;

/// Neutron-proton mass ratio (dimensionless)
pub const NEUTRON_PROTON_MASS_RATIO: f64 = 1.001_378_419_46;

/// Deuteron-electron mass ratio (dimensionless)
pub const DEUTERON_ELECTRON_MASS_RATIO: f64 = 3_670.482_967_655;

/// Alpha particle-electron mass ratio (dimensionless)
pub const ALPHA_PARTICLE_ELECTRON_MASS_RATIO: f64 = 7_294.299_541_71;

/// Helion-electron mass ratio (dimensionless)
pub const HELION_ELECTRON_MASS_RATIO: f64 = 5_495.885_279_84;

/// Triton-electron mass ratio (dimensionless)
pub const TRITON_ELECTRON_MASS_RATIO: f64 = 5_496.921_535_51;

/// Tau-electron mass ratio (dimensionless)
pub const TAU_ELECTRON_MASS_RATIO: f64 = 3_477.23;

/// Electron-tau mass ratio (dimensionless)
pub const ELECTRON_TAU_MASS_RATIO: f64 = 0.000_287_585;

// ══════════════════════════════════════════════════════════════════════
// Conversion Factors
// ══════════════════════════════════════════════════════════════════════

/// Electron volt `J`
pub const ELECTRON_VOLT: f64 = 1.602_176_634e-19;
/// Shorthand alias
pub const EV: f64 = ELECTRON_VOLT;

/// Calorie (thermochemical) `J`
pub const CALORIE: f64 = 4.184;

/// Standard atmosphere `Pa`
pub const ATMOSPHERE: f64 = 101_325.0;
/// Shorthand alias
pub const ATM: f64 = ATMOSPHERE;

/// Torr (mmHg) `Pa`
///
/// Defined as `ATMOSPHERE / 760.0` — the physics-exact ratio.
/// Per frankenscipy-h0su.
pub const TORR: f64 = ATMOSPHERE / 760.0;
/// Shorthand alias
pub const MMHG: f64 = TORR;

/// Bar `Pa`
pub const BAR: f64 = 1e5;

/// Pound-force `N`
pub const POUND_FORCE: f64 = 4.448_221_615_260_5;

/// Angstrom `m`
pub const ANGSTROM: f64 = 1e-10;

/// Nautical mile `m`
pub const NAUTICAL_MILE: f64 = 1852.0;

/// Light year `m`
pub const LIGHT_YEAR: f64 = 9.460_730_472_580_8e15;

/// Astronomical unit `m`
pub const ASTRONOMICAL_UNIT: f64 = 1.495_978_707e11;
/// Shorthand alias
pub const AU: f64 = ASTRONOMICAL_UNIT;

/// Parsec `m`
pub const PARSEC: f64 = 3.085_677_581_28e16;

/// Inch `m`
pub const INCH: f64 = 0.0254;

/// Foot `m`
pub const FOOT: f64 = 0.3048;

/// Yard `m`
pub const YARD: f64 = 0.9144;

/// Mile `m`
pub const MILE: f64 = 1609.344;

/// Pound mass (avoirdupois) `kg`
pub const POUND: f64 = 0.453_592_37;

/// Ounce (avoirdupois) `kg`
pub const OUNCE: f64 = 0.028_349_523_125;

/// Gallon (US liquid) `m³`
pub const GALLON: f64 = 3.785_411_784e-3;

/// Liter `m³`
pub const LITER: f64 = 1e-3;

/// Degree (angle) `rad`
pub const DEGREE: f64 = std::f64::consts::PI / 180.0;

/// Arc minute `rad`
pub const ARCMINUTE: f64 = DEGREE / 60.0;

/// Arc second `rad`
pub const ARCSECOND: f64 = DEGREE / 3600.0;

// ══════════════════════════════════════════════════════════════════════
// Temperature Conversion Functions
// ══════════════════════════════════════════════════════════════════════

/// Convert Celsius to Kelvin.
pub fn celsius_to_kelvin(c: f64) -> f64 {
    c + 273.15
}

/// Convert Kelvin to Celsius.
pub fn kelvin_to_celsius(k: f64) -> f64 {
    k - 273.15
}

/// Convert Fahrenheit to Kelvin.
pub fn fahrenheit_to_kelvin(f: f64) -> f64 {
    (f - 32.0) * 5.0 / 9.0 + 273.15
}

/// Convert Kelvin to Fahrenheit.
pub fn kelvin_to_fahrenheit(k: f64) -> f64 {
    (k - 273.15) * 9.0 / 5.0 + 32.0
}

/// Convert Fahrenheit to Celsius.
pub fn fahrenheit_to_celsius(f: f64) -> f64 {
    (f - 32.0) * 5.0 / 9.0
}

/// Convert Celsius to Fahrenheit.
pub fn celsius_to_fahrenheit(c: f64) -> f64 {
    c * 9.0 / 5.0 + 32.0
}

/// Convert Rankine to Kelvin.
pub fn rankine_to_kelvin(r: f64) -> f64 {
    r * 5.0 / 9.0
}

/// Convert Kelvin to Rankine.
pub fn kelvin_to_rankine(k: f64) -> f64 {
    k * 9.0 / 5.0
}

// ══════════════════════════════════════════════════════════════════════
// Energy Conversion
// ══════════════════════════════════════════════════════════════════════

/// Convert electron volts to Joules.
pub fn ev_to_joules(ev: f64) -> f64 {
    ev * ELECTRON_VOLT
}

/// Convert Joules to electron volts.
pub fn joules_to_ev(j: f64) -> f64 {
    j / ELECTRON_VOLT
}

/// Convert wavelength `m` to frequency `Hz`.
pub fn wavelength_to_freq(wavelength: f64) -> f64 {
    SPEED_OF_LIGHT / wavelength
}

/// Convert frequency `Hz` to wavelength `m`.
pub fn freq_to_wavelength(freq: f64) -> f64 {
    SPEED_OF_LIGHT / freq
}

/// Convert wavelength (m) to optical frequency (Hz).
///
/// Matches `scipy.constants.lambda2nu(lambda_)` — `nu = c / lambda`.
pub fn lambda2nu(lambda_: f64) -> f64 {
    SPEED_OF_LIGHT / lambda_
}

/// Convert optical frequency (Hz) to wavelength (m).
///
/// Matches `scipy.constants.nu2lambda(nu)` — `lambda = c / nu`.
pub fn nu2lambda(nu: f64) -> f64 {
    SPEED_OF_LIGHT / nu
}

#[derive(Debug, Clone, Copy)]
struct PhysicalConstant {
    name: &'static str,
    value: f64,
    unit: &'static str,
    uncertainty: f64,
    aliases: &'static [&'static str],
}

const fn constant_record(
    name: &'static str,
    value: f64,
    unit: &'static str,
    uncertainty: f64,
    aliases: &'static [&'static str],
) -> PhysicalConstant {
    PhysicalConstant {
        name,
        value,
        unit,
        uncertainty,
        aliases,
    }
}

// Units and standard uncertainties are transcribed from
// scipy.constants.physical_constants in SciPy 1.17.1 (CODATA 2022).
// Keep the metadata beside the value so value/unit/precision cannot drift
// into three independently maintained lookup tables.
const PHYSICAL_CONSTANTS: &[PhysicalConstant] = &[
    constant_record(
        "speed of light in vacuum",
        SPEED_OF_LIGHT,
        "m s^-1",
        0.0,
        &["speed of light", "c"],
    ),
    constant_record("Planck constant", PLANCK, "J Hz^-1", 0.0, &["planck", "h"]),
    constant_record("reduced Planck constant", HBAR, "J s", 0.0, &["hbar"]),
    constant_record(
        "Newtonian constant of gravitation",
        GRAVITATIONAL_CONSTANT,
        "m^3 kg^-1 s^-2",
        1.5e-15,
        &["gravitational constant", "g"],
    ),
    constant_record(
        "standard acceleration of gravity",
        G_N,
        "m s^-2",
        0.0,
        &["g_n"],
    ),
    constant_record("elementary charge", ELEMENTARY_CHARGE, "C", 0.0, &["e"]),
    constant_record(
        "molar gas constant",
        GAS_CONSTANT,
        "J mol^-1 K^-1",
        0.0,
        &["gas constant", "r"],
    ),
    constant_record(
        "Avogadro constant",
        AVOGADRO,
        "mol^-1",
        0.0,
        &["avogadro", "n_a"],
    ),
    constant_record(
        "Boltzmann constant",
        BOLTZMANN,
        "J K^-1",
        0.0,
        &["boltzmann", "k", "k_b"],
    ),
    constant_record(
        "Stefan-Boltzmann constant",
        STEFAN_BOLTZMANN,
        "W m^-2 K^-4",
        0.0,
        &["stefan-boltzmann", "sigma"],
    ),
    constant_record(
        "Wien wavelength displacement law constant",
        WIEN,
        "m K",
        0.0,
        &["wien"],
    ),
    constant_record("Rydberg constant", RYDBERG, "m^-1", 1.2e-5, &["rydberg"]),
    constant_record("electron mass", ELECTRON_MASS, "kg", 2.8e-40, &["m_e"]),
    constant_record("proton mass", PROTON_MASS, "kg", 5.2e-37, &["m_p"]),
    constant_record("neutron mass", NEUTRON_MASS, "kg", 8.5e-37, &["m_n"]),
    constant_record(
        "atomic mass constant",
        ATOMIC_MASS,
        "kg",
        5.2e-37,
        &["atomic mass", "u"],
    ),
    constant_record(
        "vacuum mag. permeability",
        MU_0,
        "N A^-2",
        2.0e-16,
        &["mu_0", "magnetic constant"],
    ),
    constant_record(
        "vacuum electric permittivity",
        EPSILON_0,
        "F m^-1",
        1.4e-21,
        &["epsilon_0", "electric constant"],
    ),
    constant_record("Bohr magneton", BOHR_MAGNETON, "J T^-1", 2.9e-33, &[]),
    constant_record("nuclear magneton", NUCLEAR_MAGNETON, "J T^-1", 1.6e-36, &[]),
    constant_record("mag. flux quantum", MAGNETIC_FLUX_QUANTUM, "Wb", 0.0, &[]),
    constant_record("conductance quantum", CONDUCTANCE_QUANTUM, "S", 0.0, &[]),
    constant_record("Josephson constant", JOSEPHSON, "Hz V^-1", 0.0, &[]),
    constant_record("von Klitzing constant", VON_KLITZING, "ohm", 0.0, &[]),
    constant_record("Hartree energy", HARTREE, "J", 4.8e-30, &[]),
    constant_record(
        "classical electron radius",
        CLASSICAL_ELECTRON_RADIUS,
        "m",
        1.3e-24,
        &[],
    ),
    constant_record(
        "fine-structure constant",
        FINE_STRUCTURE,
        "",
        1.1e-12,
        &["fine-structure", "alpha"],
    ),
    constant_record("Bohr radius", BOHR_RADIUS, "m", 8.2e-21, &[]),
    constant_record("electron volt", ELECTRON_VOLT, "J", 0.0, &["ev"]),
    constant_record(
        "standard atmosphere",
        ATMOSPHERE,
        "Pa",
        0.0,
        &["atmosphere", "atm"],
    ),
    constant_record("Faraday constant", FARADAY, "C mol^-1", 0.0, &["faraday"]),
    constant_record("electron g factor", ELECTRON_G_FACTOR, "", 3.6e-13, &[]),
    constant_record("proton g factor", PROTON_G_FACTOR, "", 1.6e-9, &[]),
    constant_record("neutron g factor", NEUTRON_G_FACTOR, "", 9.0e-7, &[]),
    constant_record("muon g factor", MUON_G_FACTOR, "", 8.2e-10, &[]),
    constant_record(
        "Thomson cross section",
        THOMSON_CROSS_SECTION,
        "m^2",
        6.2e-38,
        &[],
    ),
    constant_record(
        "characteristic impedance of vacuum",
        CHARACTERISTIC_IMPEDANCE_OF_VACUUM,
        "ohm",
        5.9e-8,
        &[],
    ),
    constant_record("deuteron mass", DEUTERON_MASS, "kg", 1.0e-36, &[]),
    constant_record(
        "alpha particle mass",
        ALPHA_PARTICLE_MASS,
        "kg",
        2.1e-36,
        &[],
    ),
    constant_record("muon mass", MUON_MASS, "kg", 4.2e-36, &[]),
    constant_record("tau mass", TAU_MASS, "kg", 2.1e-31, &[]),
    constant_record("helion mass", HELION_MASS, "kg", 1.6e-36, &[]),
    constant_record("triton mass", TRITON_MASS, "kg", 1.6e-36, &[]),
    constant_record(
        "molar volume of ideal gas (273.15 K, 101.325 kPa)",
        MOLAR_VOLUME_IDEAL_GAS,
        "m^3 mol^-1",
        0.0,
        &[],
    ),
    constant_record(
        "molar Planck constant",
        MOLAR_PLANCK,
        "J Hz^-1 mol^-1",
        0.0,
        &[],
    ),
    constant_record(
        "Rydberg constant times c in Hz",
        RYDBERG_HZ,
        "Hz",
        3_600.0,
        &[],
    ),
    constant_record(
        "inverse fine-structure constant",
        INVERSE_FINE_STRUCTURE,
        "",
        2.1e-8,
        &[],
    ),
    constant_record(
        "first radiation constant",
        FIRST_RADIATION_CONSTANT,
        "W m^2",
        0.0,
        &[],
    ),
    constant_record(
        "second radiation constant",
        SECOND_RADIATION_CONSTANT,
        "m K",
        0.0,
        &[],
    ),
    constant_record(
        "electron-proton mass ratio",
        ELECTRON_PROTON_MASS_RATIO,
        "",
        9.4e-15,
        &[],
    ),
    constant_record(
        "proton-electron mass ratio",
        PROTON_ELECTRON_MASS_RATIO,
        "",
        3.2e-8,
        &[],
    ),
    constant_record(
        "Bohr magneton in eV/T",
        BOHR_MAGNETON_EV_T,
        "eV T^-1",
        1.8e-14,
        &[],
    ),
    constant_record(
        "Compton wavelength",
        COMPTON_WAVELENGTH,
        "m",
        7.6e-22,
        &["electron compton wavelength"],
    ),
    constant_record(
        "electron mass energy equivalent in MeV",
        ELECTRON_MASS_MEV,
        "MeV",
        1.6e-10,
        &[],
    ),
    constant_record(
        "proton mass energy equivalent in MeV",
        PROTON_MASS_MEV,
        "MeV",
        2.9e-7,
        &[],
    ),
    constant_record(
        "neutron mass energy equivalent in MeV",
        NEUTRON_MASS_MEV,
        "MeV",
        4.8e-7,
        &[],
    ),
    constant_record(
        "muon mass energy equivalent in MeV",
        MUON_MASS_MEV,
        "MeV",
        2.3e-6,
        &[],
    ),
    constant_record(
        "tau mass energy equivalent in MeV",
        TAU_MASS_MEV,
        "MeV",
        0.16,
        &[],
    ),
    constant_record(
        "deuteron mass energy equivalent in MeV",
        DEUTERON_MASS_MEV,
        "MeV",
        5.8e-7,
        &[],
    ),
    constant_record(
        "alpha particle mass energy equivalent in MeV",
        ALPHA_PARTICLE_MASS_MEV,
        "MeV",
        1.2e-6,
        &[],
    ),
    constant_record(
        "helion mass energy equivalent in MeV",
        HELION_MASS_MEV,
        "MeV",
        8.8e-7,
        &[],
    ),
    constant_record(
        "triton mass energy equivalent in MeV",
        TRITON_MASS_MEV,
        "MeV",
        8.8e-7,
        &[],
    ),
    constant_record(
        "proton Compton wavelength",
        PROTON_COMPTON_WAVELENGTH,
        "m",
        4.1e-25,
        &[],
    ),
    constant_record(
        "neutron Compton wavelength",
        NEUTRON_COMPTON_WAVELENGTH,
        "m",
        6.7e-25,
        &[],
    ),
    constant_record(
        "muon Compton wavelength",
        MUON_COMPTON_WAVELENGTH,
        "m",
        2.6e-22,
        &[],
    ),
    constant_record(
        "tau Compton wavelength",
        TAU_COMPTON_WAVELENGTH,
        "m",
        4.7e-20,
        &[],
    ),
    constant_record(
        "neutron-electron mass ratio",
        NEUTRON_ELECTRON_MASS_RATIO,
        "",
        7.4e-7,
        &[],
    ),
    constant_record(
        "electron-neutron mass ratio",
        ELECTRON_NEUTRON_MASS_RATIO,
        "",
        2.2e-13,
        &[],
    ),
    constant_record(
        "muon-electron mass ratio",
        MUON_ELECTRON_MASS_RATIO,
        "",
        4.6e-6,
        &[],
    ),
    constant_record(
        "electron-muon mass ratio",
        ELECTRON_MUON_MASS_RATIO,
        "",
        1.1e-10,
        &[],
    ),
    constant_record(
        "proton-neutron mass ratio",
        PROTON_NEUTRON_MASS_RATIO,
        "",
        4.0e-10,
        &[],
    ),
    constant_record(
        "neutron-proton mass ratio",
        NEUTRON_PROTON_MASS_RATIO,
        "",
        4.0e-10,
        &[],
    ),
    constant_record(
        "deuteron-electron mass ratio",
        DEUTERON_ELECTRON_MASS_RATIO,
        "",
        6.3e-8,
        &[],
    ),
    constant_record(
        "alpha particle-electron mass ratio",
        ALPHA_PARTICLE_ELECTRON_MASS_RATIO,
        "",
        1.7e-7,
        &[],
    ),
    constant_record(
        "helion-electron mass ratio",
        HELION_ELECTRON_MASS_RATIO,
        "",
        1.6e-7,
        &[],
    ),
    constant_record(
        "triton-electron mass ratio",
        TRITON_ELECTRON_MASS_RATIO,
        "",
        2.1e-7,
        &[],
    ),
    constant_record(
        "tau-electron mass ratio",
        TAU_ELECTRON_MASS_RATIO,
        "",
        0.23,
        &[],
    ),
    constant_record(
        "electron-tau mass ratio",
        ELECTRON_TAU_MASS_RATIO,
        "",
        1.9e-8,
        &[],
    ),
];

fn physical_constant_normalized(name: &str) -> Option<PhysicalConstant> {
    PHYSICAL_CONSTANTS
        .iter()
        .copied()
        .find(|entry| entry.name.eq_ignore_ascii_case(name) || entry.aliases.contains(&name))
}

fn physical_constant(name: &str) -> Option<PhysicalConstant> {
    let normalized;
    let key = if name
        .bytes()
        .all(|byte| byte.is_ascii() && !byte.is_ascii_uppercase())
    {
        name
    } else {
        normalized = name.to_lowercase();
        normalized.as_str()
    };
    physical_constant_normalized(key)
}

/// Lookup a physical constant by name (case-insensitive).
///
/// Matches `scipy.constants.value(name)`. FrankenSciPy additionally accepts
/// the historical short aliases already exposed by this crate.
pub fn value(name: &str) -> Option<f64> {
    physical_constant(name).map(|entry| entry.value)
}

#[cfg(test)]
fn value_normalized(name: &str) -> Option<f64> {
    physical_constant_normalized(name).map(|entry| entry.value)
}

/// Return the CODATA unit for a physical constant.
///
/// Matches `scipy.constants.unit(name)`. Dimensionless constants return the
/// empty string; unknown names return `None`.
pub fn unit(name: &str) -> Option<&'static str> {
    physical_constant(name).map(|entry| entry.unit)
}

/// Return the CODATA relative precision for a physical constant.
///
/// This deliberately follows SciPy's signed `uncertainty / value`
/// calculation, so constants with negative values have negative precision.
/// Unknown names return `None`.
pub fn precision(name: &str) -> Option<f64> {
    physical_constant(name).map(|entry| entry.uncertainty / entry.value)
}

/// Find a physical constant by partial name match.
///
/// Returns all constants whose name contains the query string.
/// Matches `scipy.constants.find(query)`.
pub fn find(query: &str) -> Vec<(&'static str, f64)> {
    let q = query.to_lowercase();
    let mut matches: Vec<_> = PHYSICAL_CONSTANTS
        .iter()
        .filter(|entry| entry.name.to_lowercase().contains(&q))
        .map(|entry| (entry.name, entry.value))
        .collect();
    matches.sort_by_key(|(name, _)| *name);
    matches
}

/// Convert between physical units.
///
/// Matches `scipy.constants.convert_temperature` (generalized).
pub fn convert_temperature(val: f64, from: &str, to: &str) -> Result<f64, String> {
    // Convert to Kelvin first
    let kelvin = match from.to_lowercase().as_str() {
        "c" | "celsius" => celsius_to_kelvin(val),
        "f" | "fahrenheit" => fahrenheit_to_kelvin(val),
        "k" | "kelvin" => val,
        "r" | "rankine" => rankine_to_kelvin(val),
        s => return Err(format!("unsupported temperature scale: {}", s)),
    };
    // Convert from Kelvin to target
    match to.to_lowercase().as_str() {
        "c" | "celsius" => Ok(kelvin_to_celsius(kelvin)),
        "f" | "fahrenheit" => Ok(kelvin_to_fahrenheit(kelvin)),
        "k" | "kelvin" => Ok(kelvin),
        "r" | "rankine" => Ok(kelvin_to_rankine(kelvin)),
        s => Err(format!("unsupported temperature scale: {}", s)),
    }
}

/// Convert angle from degrees to radians.
pub fn deg2rad(degrees: f64) -> f64 {
    degrees * DEGREE
}

/// Convert angle from radians to degrees.
pub fn rad2deg(radians: f64) -> f64 {
    radians / DEGREE
}

/// Convert speed from mph to m/s.
pub fn mph_to_mps(mph: f64) -> f64 {
    mph * MILE / 3600.0
}

/// Convert speed from km/h to m/s.
pub fn kmh_to_mps(kmh: f64) -> f64 {
    kmh * 1000.0 / 3600.0
}

/// Convert speed from knots to m/s.
pub fn knots_to_mps(knots: f64) -> f64 {
    knots * NAUTICAL_MILE / 3600.0
}

/// Convert pressure from psi to Pa.
pub fn psi_to_pa(psi: f64) -> f64 {
    psi * POUND_FORCE / (INCH * INCH)
}

/// Convert mass from kg to lb.
pub fn kg_to_lb(kg: f64) -> f64 {
    kg / POUND
}

/// Convert mass from lb to kg.
pub fn lb_to_kg(lb: f64) -> f64 {
    lb * POUND
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambda2nu_nu2lambda_convert_temp_match_scipy() {
        // scipy.constants helpers.
        assert!(
            (lambda2nu(1e-6) - 299_792_458_000_000.0).abs() < 1.0,
            "lambda2nu(1e-6)"
        );
        assert!(
            (nu2lambda(3e14) - 9.993_081_933_333_334e-7).abs() < 1e-19,
            "nu2lambda(3e14)"
        );
        assert!(
            (convert_temperature(100.0, "Celsius", "Kelvin").unwrap() - 373.15).abs() < 1e-10,
            "convert C->K"
        );
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64, relation: &str) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tolerance,
            "{relation}: actual={actual:.17e}, expected={expected:.17e}, delta={delta:.17e}, tolerance={tolerance:.17e}"
        );
    }

    fn former_value(name: &str) -> Option<f64> {
        let normalized = name.to_lowercase();
        value_normalized(&normalized)
    }

    #[test]
    fn speed_of_light_exact() {
        assert_eq!(SPEED_OF_LIGHT, 299_792_458.0);
    }

    #[test]
    fn planck_order_of_magnitude() {
        const {
            assert!(PLANCK > 6e-34 && PLANCK < 7e-34);
        }
    }

    #[test]
    fn temperature_conversions() {
        assert!((celsius_to_kelvin(0.0) - 273.15).abs() < 1e-10);
        assert!((celsius_to_kelvin(100.0) - 373.15).abs() < 1e-10);
        assert!((fahrenheit_to_celsius(32.0) - 0.0).abs() < 1e-10);
        assert!((fahrenheit_to_celsius(212.0) - 100.0).abs() < 1e-10);
        assert!((kelvin_to_fahrenheit(0.0) - (-459.67)).abs() < 0.01);
    }

    #[test]
    fn energy_conversions() {
        let j = ev_to_joules(1.0);
        assert!((j - ELECTRON_VOLT).abs() < 1e-30);
        assert!((joules_to_ev(j) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn constant_lookup() {
        assert_eq!(value("speed of light"), Some(SPEED_OF_LIGHT));
        assert_eq!(value("Planck"), Some(PLANCK));
        assert_eq!(value("nonexistent"), None);
    }

    #[test]
    fn unit_and_precision_match_codata_metadata_and_aliases() {
        assert_eq!(unit("proton mass"), Some("kg"));
        assert_eq!(unit("Bohr magneton in eV/T"), Some("eV T^-1"));
        assert_eq!(unit("fine-structure constant"), Some(""));
        assert_eq!(unit("M_E"), Some("kg"));
        assert_eq!(unit("nonexistent"), None);

        assert_eq!(
            precision("proton mass").map(f64::to_bits),
            Some((5.2e-37_f64 / PROTON_MASS).to_bits())
        );
        assert_eq!(
            precision("electron g factor").map(f64::to_bits),
            Some((3.6e-13_f64 / ELECTRON_G_FACTOR).to_bits()),
            "SciPy preserves the sign when dividing uncertainty by value"
        );
        assert_eq!(precision("elementary charge"), Some(0.0));
        assert_eq!(precision("M_E"), precision("electron mass"));
        assert_eq!(precision("nonexistent"), None);
    }

    #[test]
    fn physical_constant_metadata_table_is_unique_and_fully_lookupable() {
        for (index, entry) in PHYSICAL_CONSTANTS.iter().enumerate() {
            assert_eq!(value(entry.name), Some(entry.value), "{}", entry.name);
            assert_eq!(unit(entry.name), Some(entry.unit), "{}", entry.name);
            assert_eq!(
                precision(entry.name).map(f64::to_bits),
                Some((entry.uncertainty / entry.value).to_bits()),
                "{}",
                entry.name
            );
            assert!(
                PHYSICAL_CONSTANTS[..index]
                    .iter()
                    .all(|prior| !prior.name.eq_ignore_ascii_case(entry.name)),
                "duplicate physical-constant key: {}",
                entry.name
            );
            for alias in entry.aliases {
                assert_eq!(value(alias), Some(entry.value), "alias {alias}");
                assert_eq!(unit(alias), Some(entry.unit), "alias {alias}");
                assert_eq!(
                    precision(alias).map(f64::to_bits),
                    Some((entry.uncertainty / entry.value).to_bits()),
                    "alias {alias}"
                );
            }
        }
    }

    #[test]
    fn normalized_ascii_value_lookup_matches_former_bits() {
        const QUERIES: &[&str] = &[
            "speed of light",
            "planck",
            "hbar",
            "gravitational constant",
            "elementary charge",
            "boltzmann",
            "stefan-boltzmann",
            "electron mass",
            "fine-structure",
            "bohr magneton",
            "conductance quantum",
            "characteristic impedance of vacuum",
            "electron mass energy equivalent in mev",
            "alpha particle-electron mass ratio",
            "c",
            "k_b",
            "Planck",
            "ELECTRON MASS",
            "Molar Volume Of Ideal Gas (273.15 K, 101.325 KPa)",
            "K",
            "İ",
            "é",
            "",
            "not a real constant",
        ];
        for query in QUERIES {
            assert_eq!(
                value(query).map(f64::to_bits),
                former_value(query).map(f64::to_bits),
                "value({query:?})"
            );
        }
    }

    #[test]
    fn golden_ratio_value() {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert!((GOLDEN_RATIO - phi).abs() < 1e-15);
    }

    #[test]
    fn lambda_nu_round_trip_matches_scipy() {
        // scipy.constants.lambda2nu(500e-9) == 599584916000000.0
        assert!((lambda2nu(500e-9) - 599_584_916_000_000.0).abs() <= 1.0);
        // scipy.constants.nu2lambda(5e14) == 5.99584916e-07
        assert!((nu2lambda(5e14) - 5.99584916e-07).abs() <= 1e-19);
        // round-trip
        assert!((nu2lambda(lambda2nu(632.8e-9)) - 632.8e-9).abs() <= 1e-18);
    }

    #[test]
    fn si_prefixes() {
        assert_eq!(KILO, 1e3);
        assert_eq!(MILLI, 1e-3);
        assert_eq!(NANO, 1e-9);
    }

    #[test]
    fn metamorphic_derived_constants_are_self_consistent() {
        let gas_from_micro_constants = BOLTZMANN * AVOGADRO;
        assert!(
            (gas_from_micro_constants - GAS_CONSTANT).abs() < 1e-9,
            "k_B * N_A should recover R"
        );

        let faraday_from_charge_count = ELEMENTARY_CHARGE * AVOGADRO;
        assert!(
            (faraday_from_charge_count - FARADAY).abs() < 1e-9,
            "e * N_A should recover Faraday"
        );

        assert!(
            (HBAR * TAU - PLANCK).abs() < 1e-48,
            "hbar * tau should recover h"
        );

        assert!(
            (CONDUCTANCE_QUANTUM * VON_KLITZING - 2.0).abs() < 1e-9,
            "G0 * R_K should equal 2"
        );
    }

    #[test]
    fn metamorphic_temperature_conversions_are_roundtrip_stable() {
        let kelvin_cases = [0.0, 1.0, 273.15, 310.15, 373.15, 1_000.0];
        let scales = ["kelvin", "celsius", "fahrenheit", "rankine"];

        for kelvin in kelvin_cases {
            for from_scale in scales {
                let source = convert_temperature(kelvin, "kelvin", from_scale)
                    .expect("kelvin should convert to every supported scale");
                for to_scale in scales {
                    let converted = convert_temperature(source, from_scale, to_scale)
                        .expect("supported source and target scales should convert");
                    let roundtrip = convert_temperature(converted, to_scale, from_scale)
                        .expect("supported target scale should convert back");
                    assert_close(
                        roundtrip,
                        source,
                        1e-10,
                        "temperature conversion should roundtrip through any supported scale",
                    );
                }
            }
        }
    }

    #[test]
    fn metamorphic_unit_conversions_are_inverse_and_linear() {
        for ev in [1e-9, 1.0, 13.605_693_122_994, 1e9] {
            assert_close(
                joules_to_ev(ev_to_joules(ev)),
                ev,
                ev.abs().max(1.0) * 1e-15,
                "electron-volt conversion should invert joule conversion",
            );
            assert_close(
                ev_to_joules(ev * 2.0),
                ev_to_joules(ev) * 2.0,
                ev_to_joules(ev.abs()).max(1.0) * 1e-15,
                "electron-volt conversion should be linear",
            );
        }

        for degrees in [-720.0, -45.0, 0.0, 90.0, 360.0] {
            assert_close(
                rad2deg(deg2rad(degrees)),
                degrees,
                1e-12,
                "degree/radian conversion should roundtrip",
            );
        }

        for pounds in [0.0, 1.0, 2.2, 150.5] {
            assert_close(
                kg_to_lb(lb_to_kg(pounds)),
                pounds,
                1e-12,
                "pound/kilogram conversion should roundtrip",
            );
        }

        for wavelength in [1e-12, 532e-9, 21.106_114_054_160e-2] {
            assert_close(
                freq_to_wavelength(wavelength_to_freq(wavelength)),
                wavelength,
                wavelength * 1e-15,
                "wavelength/frequency conversion should roundtrip",
            );
        }
    }

    #[test]
    fn metamorphic_particle_energy_and_wavelength_constants_are_self_consistent() {
        let mass_energy_cases = [
            (
                ELECTRON_MASS,
                ELECTRON_MASS_MEV,
                5e-9,
                "electron mass energy",
            ),
            (PROTON_MASS, PROTON_MASS_MEV, 5e-9, "proton mass energy"),
            (NEUTRON_MASS, NEUTRON_MASS_MEV, 5e-9, "neutron mass energy"),
            (MUON_MASS, MUON_MASS_MEV, 5e-9, "muon mass energy"),
            (TAU_MASS, TAU_MASS_MEV, 5e-5, "tau mass energy"),
            (
                DEUTERON_MASS,
                DEUTERON_MASS_MEV,
                5e-9,
                "deuteron mass energy",
            ),
            (
                ALPHA_PARTICLE_MASS,
                ALPHA_PARTICLE_MASS_MEV,
                5e-9,
                "alpha particle mass energy",
            ),
            (HELION_MASS, HELION_MASS_MEV, 5e-9, "helion mass energy"),
            (TRITON_MASS, TRITON_MASS_MEV, 5e-9, "triton mass energy"),
        ];

        for (mass_kg, energy_mev, relative_tolerance, relation) in mass_energy_cases {
            let derived_mev = joules_to_ev(mass_kg * SPEED_OF_LIGHT * SPEED_OF_LIGHT) / 1e6;
            assert_close(
                derived_mev,
                energy_mev,
                energy_mev.abs().max(1.0) * relative_tolerance,
                relation,
            );
        }

        let compton_cases = [
            (
                ELECTRON_MASS,
                COMPTON_WAVELENGTH,
                5e-9,
                "electron Compton wavelength",
            ),
            (
                PROTON_MASS,
                PROTON_COMPTON_WAVELENGTH,
                5e-9,
                "proton Compton wavelength",
            ),
            (
                NEUTRON_MASS,
                NEUTRON_COMPTON_WAVELENGTH,
                5e-9,
                "neutron Compton wavelength",
            ),
            (
                MUON_MASS,
                MUON_COMPTON_WAVELENGTH,
                5e-9,
                "muon Compton wavelength",
            ),
            (
                TAU_MASS,
                TAU_COMPTON_WAVELENGTH,
                5e-5,
                "tau Compton wavelength",
            ),
        ];

        for (mass_kg, wavelength, relative_tolerance, relation) in compton_cases {
            let derived = PLANCK / (mass_kg * SPEED_OF_LIGHT);
            assert_close(
                derived,
                wavelength,
                wavelength * relative_tolerance,
                relation,
            );
        }
    }

    #[test]
    fn metamorphic_reciprocal_and_radiation_constants_are_self_consistent() {
        let reciprocal_cases = [
            (
                ELECTRON_PROTON_MASS_RATIO,
                PROTON_ELECTRON_MASS_RATIO,
                "electron/proton mass ratios",
            ),
            (
                ELECTRON_NEUTRON_MASS_RATIO,
                NEUTRON_ELECTRON_MASS_RATIO,
                "electron/neutron mass ratios",
            ),
            (
                ELECTRON_MUON_MASS_RATIO,
                MUON_ELECTRON_MASS_RATIO,
                "electron/muon mass ratios",
            ),
            (
                PROTON_NEUTRON_MASS_RATIO,
                NEUTRON_PROTON_MASS_RATIO,
                "proton/neutron mass ratios",
            ),
        ];
        for (forward, inverse, relation) in reciprocal_cases {
            assert_close(forward * inverse, 1.0, 5e-9, relation);
        }

        assert_close(
            PLANCK * AVOGADRO,
            MOLAR_PLANCK,
            MOLAR_PLANCK * 1e-15,
            "molar Planck constant should equal h * N_A",
        );
        assert_close(
            2.0 * PI * PLANCK * SPEED_OF_LIGHT * SPEED_OF_LIGHT,
            FIRST_RADIATION_CONSTANT,
            FIRST_RADIATION_CONSTANT * 1e-15,
            "first radiation constant should equal 2*pi*h*c^2",
        );
        assert_close(
            PLANCK * SPEED_OF_LIGHT / BOLTZMANN,
            SECOND_RADIATION_CONSTANT,
            SECOND_RADIATION_CONSTANT * 1e-15,
            "second radiation constant should equal h*c/k",
        );
    }

    #[test]
    fn unit_conversions_match_scipy_constants() {
        // These conversion helpers were previously untested.
        assert!(
            (celsius_to_fahrenheit(100.0) - 212.0).abs() < 1e-12,
            "C->F boiling"
        );
        assert!(
            (celsius_to_fahrenheit(0.0) - 32.0).abs() < 1e-12,
            "C->F freezing"
        );
        assert!((kmh_to_mps(36.0) - 10.0).abs() < 1e-12, "36 km/h = 10 m/s");
        // scipy.constants.knot = 1852/3600.
        assert!(
            (knots_to_mps(1.0) - 0.514_444_444_444_444_5).abs() < 1e-12,
            "knot"
        );
        // scipy.constants.mph = mile/3600.
        assert!((mph_to_mps(1.0) - 0.44704).abs() < 1e-9, "mph");
        // scipy.constants.psi.
        assert!((psi_to_pa(1.0) - 6_894.757_293_168_361).abs() < 1e-6, "psi");
    }

    #[test]
    fn physical_constants_match_scipy_reference_values() {
        // scipy.constants values (CODATA 2022, SciPy 1.17.1)
        // Use relative tolerances for physical constants since different CODATA years may differ slightly
        assert_eq!(
            SPEED_OF_LIGHT, 299792458.0,
            "c should match scipy.constants.c exactly"
        );
        assert_close(
            MU_0,
            1.25663706127e-6,
            1e-15,
            "mu_0 vs scipy.constants.mu_0",
        );
        assert_close(
            EPSILON_0,
            8.8541878188e-12,
            1e-20,
            "epsilon_0 vs scipy.constants.epsilon_0",
        );
        assert_close(PLANCK, 6.62607015e-34, 1e-43, "h vs scipy.constants.h");
        assert_close(
            GRAVITATIONAL_CONSTANT,
            6.6743e-11,
            1e-14,
            "G vs scipy.constants.G",
        );
        assert_close(
            ELEMENTARY_CHARGE,
            1.602176634e-19,
            1e-28,
            "e vs scipy.constants.e",
        );
        assert_close(
            GAS_CONSTANT,
            8.31446261815324,
            2e-10,
            "R vs scipy.constants.R",
        );
        assert_close(AVOGADRO, 6.02214076e23, 1e14, "N_A vs scipy.constants.N_A");
        assert_close(BOLTZMANN, 1.380649e-23, 1e-29, "k vs scipy.constants.k");
        assert_close(
            GOLDEN_RATIO,
            1.618033988749895,
            1e-15,
            "golden_ratio vs scipy.constants.golden_ratio",
        );
    }

    #[test]
    fn measured_constants_match_scipy_codata_2022_exactly() {
        // The core measured/derived constants must equal scipy.constants
        // (CODATA 2022, SciPy 1.17.1) bit-for-bit. These were previously CODATA
        // 2018 values, inconsistent with the already-2022 additions (e.g.
        // FINE_STRUCTURE vs INVERSE_FINE_STRUCTURE). frankenscipy-i5rk3.
        assert_eq!(MU_0, 1.256_637_061_27e-6);
        assert_eq!(EPSILON_0, 8.854_187_818_8e-12);
        assert_eq!(RYDBERG, 1.097_373_156_815_7e7);
        assert_eq!(ELECTRON_MASS, 9.109_383_713_9e-31);
        assert_eq!(PROTON_MASS, 1.672_621_925_95e-27);
        assert_eq!(NEUTRON_MASS, 1.674_927_500_56e-27);
        assert_eq!(ATOMIC_MASS, 1.660_539_068_92e-27);
        assert_eq!(BOHR_MAGNETON, 9.274_010_065_7e-24);
        assert_eq!(NUCLEAR_MAGNETON, 5.050_783_739_3e-27);
        assert_eq!(MAGNETIC_FLUX_QUANTUM, 2.067_833_848_461_929_5e-15);
        assert_eq!(JOSEPHSON, 4.835_978_484_169_836e14);
        assert_eq!(VON_KLITZING, 2.581_280_745_930_451_3e4);
        assert_eq!(FINE_STRUCTURE, 7.297_352_564_3e-3);
        assert_eq!(BOHR_RADIUS, 5.291_772_105_44e-11);
        assert_eq!(HARTREE, 4.359_744_722_206e-18);
        assert_eq!(CLASSICAL_ELECTRON_RADIUS, 2.817_940_320_5e-15);
        assert_eq!(WIEN, 2.897_771_955_185_172_7e-3);
        assert_eq!(STEFAN_BOLTZMANN, 5.670_374_419_184_431_4e-8);
        // Internal consistency: 1/alpha agrees with the independent inverse
        // constant to CODATA rounding (scipy itself differs by ~4e-12).
        assert!(
            (1.0 / FINE_STRUCTURE - INVERSE_FINE_STRUCTURE).abs() / INVERSE_FINE_STRUCTURE < 1e-11
        );
    }

    #[test]
    fn convert_temperature_matches_scipy_reference_values() {
        // scipy.constants.convert_temperature(100, 'C', 'K') = 373.15
        // scipy.constants.convert_temperature(0, 'C', 'F') = 32
        // scipy.constants.convert_temperature(212, 'F', 'C') = 100
        let c_to_k = convert_temperature(100.0, "C", "K").expect("C to K");
        assert_close(c_to_k, 373.15, 1e-10, "convert_temperature C to K");

        let c_to_f = convert_temperature(0.0, "C", "F").expect("C to F");
        assert_close(c_to_f, 32.0, 1e-10, "convert_temperature C to F");

        let f_to_c = convert_temperature(212.0, "F", "C").expect("F to C");
        assert_close(f_to_c, 100.0, 1e-10, "convert_temperature F to C");
    }

    #[test]
    fn value_lookup_matches_scipy_codata_2022() {
        // Each key must return scipy.constants.physical_constants[key][0]
        // (CODATA 2022, SciPy 1.17.1) bit-for-bit. Locks both the constant
        // values and the value() key→symbol mapping. frankenscipy-nwzcn.
        let cases: &[(&str, f64)] = &[
            ("faraday constant", 96485.33212331001),
            ("electron g factor", -2.00231930436092),
            ("proton g factor", 5.5856946893),
            ("neutron g factor", -3.82608552),
            ("muon g factor", -2.00233184123),
            ("thomson cross section", 6.6524587051e-29),
            ("characteristic impedance of vacuum", 376.730313412),
            ("molar planck constant", 3.990312712893431e-10),
            ("rydberg constant times c in hz", 3289841960250000.0),
            ("inverse fine-structure constant", 137.035999177),
            ("first radiation constant", 3.7417718521927573e-16),
            ("second radiation constant", 0.014387768775039337),
            ("electron-proton mass ratio", 0.0005446170214889),
            ("proton-electron mass ratio", 1836.152673426),
            ("bohr magneton in ev/t", 5.7883817982e-05),
            ("electron mass energy equivalent in mev", 0.51099895069),
            ("proton mass energy equivalent in mev", 938.27208943),
            ("neutron mass energy equivalent in mev", 939.56542194),
            ("proton compton wavelength", 1.3214098536e-15),
            ("neutron compton wavelength", 1.31959090382e-15),
            ("neutron-electron mass ratio", 1838.683662),
            ("muon-electron mass ratio", 206.7682827),
            ("proton-neutron mass ratio", 0.99862347797),
            ("deuteron-electron mass ratio", 3670.482967655),
            ("alpha particle-electron mass ratio", 7294.29954171),
            ("bohr magneton", 9.2740100657e-24),
            ("nuclear magneton", 5.0507837393e-27),
            ("mag. flux quantum", 2.0678338484619295e-15),
            ("conductance quantum", 7.748091729863649e-05),
            ("josephson constant", 483597848416983.6),
            ("von klitzing constant", 25812.807459304513),
            ("hartree energy", 4.359744722206e-18),
            ("classical electron radius", 2.8179403205e-15),
        ];
        for &(key, want) in cases {
            let got = value(key).expect("known constant value lookup should succeed");
            assert_eq!(got, want, "value({key:?})");
            // Case-insensitivity must hold (scipy keys are case-sensitive but
            // fsci's value() lowercases): an upper-cased key resolves the same.
            assert_eq!(
                value(&key.to_uppercase()),
                Some(want),
                "value({key:?}) upper"
            );
        }
        // Unknown keys return None (scipy raises; fsci is Option-typed).
        assert_eq!(value("not a real constant"), None);
    }

    #[test]
    fn value_matches_scipy_reference_values() {
        // scipy.constants.value('speed of light in vacuum') = 299792458.0
        // Our API uses simpler keys
        let c = value("speed of light").expect("speed of light");
        assert_eq!(c, 299792458.0, "value('speed of light')");

        let h = value("planck").expect("planck");
        assert_close(h, 6.62607015e-34, 1e-43, "value('planck')");
    }

    #[test]
    fn find_matches_scipy_reference_values() {
        // scipy.constants.find('light') returns constants containing 'light'
        let results = find("light");
        assert!(!results.is_empty(), "find('light') should return results");
        // Should include speed of light
        let has_speed_of_light = results
            .iter()
            .any(|(name, _)| name.contains("speed of light"));
        assert!(
            has_speed_of_light,
            "find('light') should include speed of light"
        );
    }

    #[test]
    fn find_lists_em_atomic_constants_exposed_by_value() {
        // frankenscipy-g9e37: value() and find() should expose the same
        // CODATA-2022 EM/atomic constants that SciPy lists by name.
        let cases: &[(&str, &str, f64)] = &[
            ("magneton", "Bohr magneton", BOHR_MAGNETON),
            ("magneton", "nuclear magneton", NUCLEAR_MAGNETON),
            ("flux", "mag. flux quantum", MAGNETIC_FLUX_QUANTUM),
            ("conductance", "conductance quantum", CONDUCTANCE_QUANTUM),
            ("josephson", "Josephson constant", JOSEPHSON),
            ("klitzing", "von Klitzing constant", VON_KLITZING),
            ("hartree", "Hartree energy", HARTREE),
            (
                "classical electron",
                "classical electron radius",
                CLASSICAL_ELECTRON_RADIUS,
            ),
        ];

        for &(query, expected_name, expected_value) in cases {
            let results = find(query);
            assert!(
                results
                    .iter()
                    .any(|&(name, value)| name == expected_name && value == expected_value),
                "find({query:?}) should include {expected_name}"
            );
        }
    }
}
