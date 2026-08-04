use fsci_conformance::{HarnessConfig, oracle_checkout_present, run_smoke};
use std::fs;

#[test]
fn smoke_report_is_stable() {
    let cfg = HarnessConfig::default_paths();
    let report = run_smoke(&cfg).expect("smoke packet should run");
    assert_eq!(report.suite, "smoke");
    assert!(report.cases_run >= 1);
    assert_eq!(report.failed_cases, 0);
    assert!(report.strict_mode);
}

/// The legacy SciPy oracle clone is gitignored, so whether it is populated is a
/// property of the machine rather than of this crate. What must hold everywhere
/// is that `run_smoke` reports it accurately across both states — including
/// reporting an empty `legacy_scipy_code/scipy` directory as absent, which the
/// previous `Path::exists()` implementation got wrong.
#[test]
fn smoke_report_oracle_flag_tracks_checkout_contents() {
    let defaults = HarnessConfig::default_paths();
    let root = std::env::temp_dir().join(format!(
        "fsci-smoke-oracle-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock after epoch")
            .as_nanos()
    ));
    let empty = root.join("empty");
    let populated = root.join("populated");
    fs::create_dir_all(&empty).expect("create empty oracle root");
    fs::create_dir_all(populated.join("scipy")).expect("create populated oracle root");
    fs::write(populated.join("scipy").join("__init__.py"), "# scipy\n")
        .expect("write scipy package marker");

    let report_for = |oracle_root: std::path::PathBuf| {
        let cfg = HarnessConfig {
            oracle_root,
            fixture_root: defaults.fixture_root.clone(),
            strict_mode: true,
        };
        run_smoke(&cfg).expect("smoke packet should run")
    };

    // A directory that exists but carries no SciPy source is NOT an oracle.
    assert!(!report_for(empty).oracle_present);
    // A checkout carrying the SciPy package marker is.
    assert!(report_for(populated).oracle_present);
    // The free function agrees with the reported flag.
    assert!(!oracle_checkout_present(&root.join("absent")));
}
