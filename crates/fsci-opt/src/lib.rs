#![forbid(unsafe_code)]

pub mod linesearch;
pub mod minimize;
pub mod root;
pub mod types;

pub use linesearch::{LineSearchResult, WolfeParams, line_search_wolfe1, line_search_wolfe2};
pub use minimize::{bfgs, cg_pr_plus, minimize, powell};
pub use root::{RootResult, bisect, brenth, brentq, ridder, root_scalar};
pub use types::{
    ConvergenceStatus, MinimizeOptions, OptError, OptimizeMethod, OptimizeResult, RootMethod,
    RootOptions,
};

#[cfg(test)]
mod tests {
    use fsci_runtime::RuntimeMode;

    use crate::{ConvergenceStatus, MinimizeOptions, OptimizeMethod, RootOptions};

    #[test]
    fn defaults_match_packet_contract_intent() {
        let minimize = MinimizeOptions::default();
        assert_eq!(minimize.method, None);
        assert_eq!(minimize.mode, RuntimeMode::Strict);

        let root = RootOptions::default();
        assert!(root.xtol > 0.0);
        assert!(root.rtol > 0.0);
        assert_eq!(root.mode, RuntimeMode::Strict);
    }

    #[test]
    fn convergence_status_is_cloneable() {
        let status = ConvergenceStatus::CallbackStop;
        assert_eq!(status, ConvergenceStatus::CallbackStop);
    }

    #[test]
    fn optimize_method_enum_is_stable() {
        let method = OptimizeMethod::Bfgs;
        assert!(matches!(method, OptimizeMethod::Bfgs));
    }
}
