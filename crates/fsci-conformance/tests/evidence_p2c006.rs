//! bd-3jh.17.9: [FSCI-P2C-006-I] Final Evidence Pack
//!
//! Produces a self-contained evidence bundle at:
//!   fixtures/artifacts/FSCI-P2C-006/evidence/
//!
//! Covers gamma, erf, beta, bessel families with parity gates and risk notes.

use blake3::hash;
use fsci_conformance::{RaptorQSidecar, generate_raptorq_sidecar};
use fsci_runtime::RuntimeMode;
use fsci_special::{
    SpecialTensor, beta, erf, erfc, erfinv, gamma, gammainc, gammaincc, j0, j1, rgamma, y0,
};
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize)]
struct FixtureManifest {
    packet_id: &'static str,
    generated_at: String,
    fixtures: Vec<FixtureEntry>,
}

#[derive(Debug, Serialize)]
struct FixtureEntry {
    fixture_id: String,
    function: &'static str,
    input_range: String,
    expected_properties: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ParityGateReport {
    packet_id: &'static str,
    all_gates_pass: bool,
    gates: Vec<ParityGate>,
}

#[derive(Debug, Serialize)]
struct ParityGate {
    fixture_id: String,
    operation: &'static str,
    pass: bool,
    max_abs_diff: f64,
    tolerance_used: f64,
    note: String,
}

#[derive(Debug, Serialize)]
struct RiskNote {
    category: &'static str,
    description: String,
    affected_operations: Vec<&'static str>,
    mitigation: String,
}

#[derive(Debug, Serialize)]
struct RiskNotesReport {
    packet_id: &'static str,
    notes: Vec<RiskNote>,
}

#[derive(Debug, Serialize)]
struct ParityReport {
    packet_id: &'static str,
    operation_summaries: Vec<OperationParitySummary>,
}

#[derive(Debug, Serialize)]
struct OperationParitySummary {
    operation: &'static str,
    total_fixtures: usize,
    passed: usize,
    failed: usize,
    max_abs_diff_across_all: f64,
}

#[derive(Debug, Serialize)]
struct EvidenceBundle {
    manifest: FixtureManifest,
    parity_gates: ParityGateReport,
    risk_notes: RiskNotesReport,
    parity_report: ParityReport,
    sidecar: Option<RaptorQSidecar>,
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn scalar(x: f64) -> SpecialTensor {
    SpecialTensor::RealScalar(x)
}

fn real_val(t: &SpecialTensor) -> f64 {
    match t {
        SpecialTensor::RealScalar(v) => *v,
        _ => panic!("expected RealScalar"),
    }
}

fn now_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("unix:{secs}")
}

// ── Parity checks ──────────────────────────────────────────────────────────────

fn check_gamma_known_values() -> Vec<ParityGate> {
    let cases: &[(&str, f64, f64, f64)] = &[
        ("gamma_1", 1.0, 1.0, 1e-12),
        ("gamma_2", 2.0, 1.0, 1e-12),
        ("gamma_5", 5.0, 24.0, 1e-10),
        ("gamma_half", 0.5, std::f64::consts::PI.sqrt(), 1e-10),
    ];
    cases
        .iter()
        .map(|&(id, x, expected, tol)| {
            let result = real_val(&gamma(&scalar(x), RuntimeMode::Strict).unwrap());
            let diff = (result - expected).abs();
            ParityGate {
                fixture_id: id.into(),
                operation: "gamma",
                pass: diff < tol,
                max_abs_diff: diff,
                tolerance_used: tol,
                note: format!("gamma({x})={result:.6e}, expected={expected:.6e}"),
            }
        })
        .collect()
}

fn check_erf_erfc_complement() -> Vec<ParityGate> {
    let xs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
    xs.iter()
        .map(|&x| {
            let e = real_val(&erf(&scalar(x), RuntimeMode::Strict).unwrap());
            let ec = real_val(&erfc(&scalar(x), RuntimeMode::Strict).unwrap());
            let sum = e + ec;
            let diff = (sum - 1.0).abs();
            ParityGate {
                fixture_id: format!("erf_erfc_x{x}"),
                operation: "erf_erfc",
                pass: diff < 1e-12,
                max_abs_diff: diff,
                tolerance_used: 1e-12,
                note: format!("erf({x})+erfc({x})={sum:.6e}"),
            }
        })
        .collect()
}

fn check_erfinv_roundtrip() -> Vec<ParityGate> {
    let xs = [-0.9, -0.5, 0.0, 0.5, 0.9];
    xs.iter()
        .map(|&x| {
            let inv = real_val(&erfinv(&scalar(x), RuntimeMode::Strict).unwrap());
            let rt = real_val(&erf(&scalar(inv), RuntimeMode::Strict).unwrap());
            let diff = (rt - x).abs();
            ParityGate {
                fixture_id: format!("erfinv_rt_x{x}"),
                operation: "erfinv",
                pass: diff < 1e-10,
                max_abs_diff: diff,
                tolerance_used: 1e-10,
                note: format!("erf(erfinv({x}))={rt:.6e}"),
            }
        })
        .collect()
}

fn check_gammainc_complement() -> Vec<ParityGate> {
    let pairs: &[(f64, f64)] = &[(1.0, 1.0), (2.0, 2.0), (3.0, 2.0), (5.0, 5.0)];
    pairs
        .iter()
        .map(|&(a, x)| {
            let inc = real_val(&gammainc(&scalar(a), &scalar(x), RuntimeMode::Strict).unwrap());
            let incc = real_val(&gammaincc(&scalar(a), &scalar(x), RuntimeMode::Strict).unwrap());
            let sum = inc + incc;
            let diff = (sum - 1.0).abs();
            ParityGate {
                fixture_id: format!("gammainc_a{a}_x{x}"),
                operation: "gammainc",
                pass: diff < 1e-10,
                max_abs_diff: diff,
                tolerance_used: 1e-10,
                note: format!("P({a},{x})+Q({a},{x})={sum:.6e}"),
            }
        })
        .collect()
}

fn check_beta_gamma_relation() -> Vec<ParityGate> {
    let pairs: &[(f64, f64)] = &[(1.0, 1.0), (2.0, 3.0), (0.5, 0.5)];
    pairs
        .iter()
        .map(|&(a, b)| {
            let b_ab = real_val(&beta(&scalar(a), &scalar(b), RuntimeMode::Strict).unwrap());
            let ga = real_val(&gamma(&scalar(a), RuntimeMode::Strict).unwrap());
            let gb = real_val(&gamma(&scalar(b), RuntimeMode::Strict).unwrap());
            let gab = real_val(&gamma(&scalar(a + b), RuntimeMode::Strict).unwrap());
            let expected = ga * gb / gab;
            let rel_err = (b_ab - expected).abs() / expected.abs().max(1e-30);
            ParityGate {
                fixture_id: format!("beta_gamma_a{a}_b{b}"),
                operation: "beta",
                pass: rel_err < 1e-10,
                max_abs_diff: rel_err,
                tolerance_used: 1e-10,
                note: format!("B({a},{b})={b_ab:.6e}, Γ(a)Γ(b)/Γ(a+b)={expected:.6e}"),
            }
        })
        .collect()
}

fn check_bessel_boundary() -> Vec<ParityGate> {
    let mut gates = Vec::new();
    let j0_0 = real_val(&j0(&scalar(0.0), RuntimeMode::Strict).unwrap());
    gates.push(ParityGate {
        fixture_id: "j0_zero".into(),
        operation: "bessel",
        pass: (j0_0 - 1.0).abs() < 1e-8,
        max_abs_diff: (j0_0 - 1.0).abs(),
        tolerance_used: 1e-8,
        note: format!("j0(0)={j0_0:.6e}"),
    });
    let j1_0 = real_val(&j1(&scalar(0.0), RuntimeMode::Strict).unwrap());
    gates.push(ParityGate {
        fixture_id: "j1_zero".into(),
        operation: "bessel",
        pass: j1_0.abs() < 1e-8,
        max_abs_diff: j1_0.abs(),
        tolerance_used: 1e-8,
        note: format!("j1(0)={j1_0:.6e}"),
    });
    let y0_1 = real_val(&y0(&scalar(1.0), RuntimeMode::Strict).unwrap());
    gates.push(ParityGate {
        fixture_id: "y0_one".into(),
        operation: "bessel",
        pass: (y0_1 - 0.0882569642).abs() < 1e-4,
        max_abs_diff: (y0_1 - 0.0882569642).abs(),
        tolerance_used: 1e-4,
        note: format!("y0(1)={y0_1:.6e}"),
    });
    gates
}

fn check_gamma_rgamma_inverse() -> Vec<ParityGate> {
    let xs = [1.0, 2.0, 3.0, 5.0, 10.0];
    xs.iter()
        .map(|&x| {
            let g = real_val(&gamma(&scalar(x), RuntimeMode::Strict).unwrap());
            let rg = real_val(&rgamma(&scalar(x), RuntimeMode::Strict).unwrap());
            let product = g * rg;
            let diff = (product - 1.0).abs();
            ParityGate {
                fixture_id: format!("gamma_rgamma_x{x}"),
                operation: "rgamma",
                pass: diff < 1e-10,
                max_abs_diff: diff,
                tolerance_used: 1e-10,
                note: format!("gamma({x})*rgamma({x})={product:.6e}"),
            }
        })
        .collect()
}

// ── Main test ──────────────────────────────────────────────────────────────────

#[test]
fn evidence_p2c006_final_pack() {
    let evidence_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-006/evidence");
    std::fs::create_dir_all(&evidence_dir).expect("evidence directory");

    // Fixture manifest
    let fixture_entries = vec![
        FixtureEntry {
            fixture_id: "gamma_known".into(),
            function: "gamma",
            input_range: "x in {0.5, 1, 2, 5}".into(),
            expected_properties: vec!["Γ(1)=1".into(), "Γ(n)=(n-1)!".into(), "Γ(0.5)=√π".into()],
        },
        FixtureEntry {
            fixture_id: "erf_erfc_complement".into(),
            function: "erf+erfc",
            input_range: "x in [0, 3]".into(),
            expected_properties: vec!["erf(x)+erfc(x)=1".into()],
        },
        FixtureEntry {
            fixture_id: "erfinv_roundtrip".into(),
            function: "erfinv",
            input_range: "x in (-1, 1)".into(),
            expected_properties: vec!["erf(erfinv(x))=x".into()],
        },
        FixtureEntry {
            fixture_id: "gammainc_complement".into(),
            function: "gammainc+gammaincc",
            input_range: "a,x > 0".into(),
            expected_properties: vec!["P(a,x)+Q(a,x)=1".into()],
        },
        FixtureEntry {
            fixture_id: "beta_gamma_relation".into(),
            function: "beta",
            input_range: "a,b > 0".into(),
            expected_properties: vec!["B(a,b)=Γ(a)Γ(b)/Γ(a+b)".into()],
        },
        FixtureEntry {
            fixture_id: "bessel_boundary".into(),
            function: "j0,j1,y0",
            input_range: "x=0 and x=1".into(),
            expected_properties: vec!["J0(0)=1".into(), "J1(0)=0".into()],
        },
        FixtureEntry {
            fixture_id: "gamma_rgamma_inverse".into(),
            function: "gamma*rgamma",
            input_range: "x in {1,2,3,5,10}".into(),
            expected_properties: vec!["Γ(x)·(1/Γ(x))=1".into()],
        },
    ];

    let manifest = FixtureManifest {
        packet_id: "FSCI-P2C-006",
        generated_at: now_str(),
        fixtures: fixture_entries,
    };

    // Run parity gates
    let mut gates = Vec::new();
    gates.extend(check_gamma_known_values());
    gates.extend(check_erf_erfc_complement());
    gates.extend(check_erfinv_roundtrip());
    gates.extend(check_gammainc_complement());
    gates.extend(check_beta_gamma_relation());
    gates.extend(check_bessel_boundary());
    gates.extend(check_gamma_rgamma_inverse());

    let all_pass = gates.iter().all(|g| g.pass);
    let parity_gates = ParityGateReport {
        packet_id: "FSCI-P2C-006",
        all_gates_pass: all_pass,
        gates,
    };

    // Operation summaries
    let ops = [
        "gamma", "erf_erfc", "erfinv", "gammainc", "beta", "bessel", "rgamma",
    ];
    let operation_summaries: Vec<_> = ops
        .iter()
        .map(|&op| {
            let matched: Vec<_> = parity_gates
                .gates
                .iter()
                .filter(|g| g.operation == op)
                .collect();
            OperationParitySummary {
                operation: op,
                total_fixtures: matched.len(),
                passed: matched.iter().filter(|g| g.pass).count(),
                failed: matched.iter().filter(|g| !g.pass).count(),
                max_abs_diff_across_all: matched.iter().map(|g| g.max_abs_diff).fold(
                    0.0_f64,
                    |a: f64, b: f64| {
                        if a.is_nan() || b.is_nan() {
                            f64::NAN
                        } else {
                            a.max(b)
                        }
                    },
                ),
            }
        })
        .collect();

    let parity_report = ParityReport {
        packet_id: "FSCI-P2C-006",
        operation_summaries,
    };

    let risk_notes = RiskNotesReport {
        packet_id: "FSCI-P2C-006",
        notes: vec![
            RiskNote {
                category: "pole_handling",
                description: "Gamma function has poles at non-positive integers. Strict mode returns NaN; Hardened mode returns error.".into(),
                affected_operations: vec!["gamma", "gammaln", "rgamma"],
                mitigation: "Domain validation in Hardened mode. Strict mode NaN propagation matches SciPy behavior.".into(),
            },
            RiskNote {
                category: "bessel_rational_approximation",
                description: "J0/J1 use rational polynomial approximation (Numerical Recipes). Accuracy is ~1e-8 near x=0 due to coefficient precision limits.".into(),
                affected_operations: vec!["j0", "j1", "jn"],
                mitigation: "Tolerances set to 1e-8 for bessel boundary checks. Higher-precision coefficients can be substituted.".into(),
            },
            RiskNote {
                category: "erfinv_domain",
                description: "erfinv is only defined on (-1, 1). Values at ±1 produce ±Inf, values outside produce NaN in Strict mode.".into(),
                affected_operations: vec!["erfinv", "erfcinv"],
                mitigation: "Hardened mode rejects out-of-domain inputs. E2E tests cover boundary and out-of-domain cases.".into(),
            },
            RiskNote {
                category: "gammainc_convergence",
                description: "Incomplete gamma uses series/continued fraction expansion. Convergence can be slow for large a or x.".into(),
                affected_operations: vec!["gammainc", "gammaincc"],
                mitigation: "Complement identity P+Q=1 verified in parity gates. Series truncation monitored by iteration count.".into(),
            },
        ],
    };

    let evidence = EvidenceBundle {
        manifest,
        parity_gates,
        risk_notes,
        parity_report,
        sidecar: None,
    };

    let json = serde_json::to_string_pretty(&evidence).unwrap();
    std::fs::write(evidence_dir.join("evidence_bundle.json"), &json).unwrap();

    // Generate RaptorQ sidecar from the bundle on disk
    let sidecar = generate_raptorq_sidecar(json.as_bytes()).unwrap();
    std::fs::write(
        evidence_dir.join("evidence_bundle.raptorq.json"),
        serde_json::to_string_pretty(&sidecar).unwrap(),
    )
    .unwrap();

    std::fs::write(
        evidence_dir.join("fixture_manifest.json"),
        serde_json::to_string_pretty(&evidence.manifest).unwrap(),
    )
    .unwrap();
    std::fs::write(
        evidence_dir.join("parity_gates.json"),
        serde_json::to_string_pretty(&evidence.parity_gates).unwrap(),
    )
    .unwrap();
    std::fs::write(
        evidence_dir.join("risk_notes.json"),
        serde_json::to_string_pretty(&evidence.risk_notes).unwrap(),
    )
    .unwrap();
    std::fs::write(
        evidence_dir.join("parity_report.json"),
        serde_json::to_string_pretty(&evidence.parity_report).unwrap(),
    )
    .unwrap();

    let bundle_hash = hash(json.as_bytes()).to_hex().to_string();
    std::fs::write(evidence_dir.join("evidence_bundle.blake3"), &bundle_hash).unwrap();

    for g in &evidence.parity_gates.gates {
        if g.pass {
            eprintln!("  PASS: {} {} — {}", g.operation, g.fixture_id, g.note);
        } else {
            eprintln!("  FAIL: {} {} — {}", g.operation, g.fixture_id, g.note);
        }
    }
    assert!(all_pass, "All parity gates must pass");

    eprintln!("\n── P2C-006 Evidence Pack ──");
    for s in &evidence.parity_report.operation_summaries {
        eprintln!(
            "  {}: {}/{} pass, max_abs={:.2e}",
            s.operation, s.passed, s.total_fixtures, s.max_abs_diff_across_all
        );
    }
    eprintln!("  RaptorQ sidecar: generated (external)");
}
