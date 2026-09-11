#![forbid(unsafe_code)]
//! Aggregates FSCI-P2C-018 diff and metamorphic logs into the root parity report triple.
//!
//! Usage:
//!     cargo run -p fsci-conformance --bin aggregate_p2c018

use fsci_conformance::{HarnessConfig, write_p2c018_root_parity_artifacts};

fn main() {
    let config = HarnessConfig::default_paths();
    match write_p2c018_root_parity_artifacts(&config) {
        Ok(bundle) => {
            println!("FSCI-P2C-018 root parity report generated successfully:");
            println!("  report:       {}", bundle.report_path.display());
            println!("  raptorq:      {}", bundle.sidecar_path.display());
            println!("  decode_proof: {}", bundle.decode_proof_path.display());
        }
        Err(e) => {
            eprintln!("Error generating FSCI-P2C-018 parity report: {e}");
            std::process::exit(1);
        }
    }
}
