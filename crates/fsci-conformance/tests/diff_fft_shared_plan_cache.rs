#![forbid(unsafe_code)]
//! Cover fsci_fft::plan public API surface from outside the crate.
//!
//! Resolves [frankenscipy-4i2eq]. Exercises the shared plan cache
//! round-trip, the three admission policies (Disabled, AlwaysInsert,
//! CostWeightedLru), the capacity bound, and the working-set bytes
//! accounting + clear. Runs as a single test function so the
//! process-global cache state is observed sequentially.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::plan::{
    clear_shared_plan_cache, lookup_shared_plan, shared_plan_cache_len,
    shared_plan_cache_working_set_bytes, store_shared_plan, store_shared_plan_with_config,
};
use fsci_fft::{
    CacheAdmissionPolicy, Normalization, PlanCacheConfig, PlanFingerprint, PlanKey, PlanMetadata,
    PlanningStrategy, TransformKind,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create plan_cache diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn mk_key(n: usize) -> PlanKey {
    PlanKey::new(
        TransformKind::Fft,
        vec![n],
        vec![0],
        Normalization::Backward,
        false,
    )
}

fn mk_metadata(n: usize, estimated_flops: u64, scratch_bytes: usize) -> PlanMetadata {
    PlanMetadata {
        key: mk_key(n),
        fingerprint: PlanFingerprint {
            radix_path: vec![2; n.trailing_zeros() as usize],
            estimated_flops,
            scratch_bytes,
        },
        generated_by: PlanningStrategy::EstimateOnly,
    }
}

#[test]
fn diff_fft_shared_plan_cache() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === 1. Basic round-trip ===
    clear_shared_plan_cache();
    check(
        "cache_starts_empty",
        shared_plan_cache_len() == 0,
        format!("len={}", shared_plan_cache_len()),
    );
    check(
        "working_set_zero_when_empty",
        shared_plan_cache_working_set_bytes() == 0,
        format!("bytes={}", shared_plan_cache_working_set_bytes()),
    );

    let m_basic = mk_metadata(64, 64 * 6 * 5, 64 * 16);
    let k_basic = m_basic.key.clone();
    let stored = store_shared_plan(m_basic.clone());
    check(
        "store_basic_returns_true",
        stored,
        format!("stored={stored}"),
    );
    check(
        "cache_len_one_after_store",
        shared_plan_cache_len() == 1,
        format!("len={}", shared_plan_cache_len()),
    );

    let looked_up = lookup_shared_plan(&k_basic);
    check(
        "lookup_returns_some",
        looked_up.is_some(),
        format!("found={}", looked_up.is_some()),
    );
    check(
        "lookup_metadata_matches",
        looked_up.as_ref() == Some(&m_basic),
        format!("eq={}", looked_up.as_ref() == Some(&m_basic)),
    );

    let working_after_one = shared_plan_cache_working_set_bytes();
    check(
        "working_set_nonzero_after_store",
        working_after_one > 0,
        format!("bytes={working_after_one}"),
    );

    // === 2. Disabled admission policy: store should fail ===
    clear_shared_plan_cache();
    let cfg_disabled = PlanCacheConfig {
        admission_policy: CacheAdmissionPolicy::Disabled,
        ..PlanCacheConfig::default()
    };
    let m16 = mk_metadata(16, 320, 16 * 16);
    let disabled_ok = !store_shared_plan_with_config(m16.clone(), cfg_disabled);
    check(
        "disabled_admission_rejects",
        disabled_ok,
        format!("rejected={disabled_ok}"),
    );
    check(
        "disabled_cache_remains_empty",
        shared_plan_cache_len() == 0,
        format!("len={}", shared_plan_cache_len()),
    );

    // === 3. AlwaysInsert capacity enforcement ===
    clear_shared_plan_cache();
    let cfg_cap2 = PlanCacheConfig {
        capacity: 2,
        admission_policy: CacheAdmissionPolicy::AlwaysInsert,
        ..PlanCacheConfig::default()
    };
    let stored_a = store_shared_plan_with_config(mk_metadata(16, 320, 16 * 16), cfg_cap2.clone());
    let stored_b = store_shared_plan_with_config(mk_metadata(32, 800, 32 * 16), cfg_cap2.clone());
    let stored_c = store_shared_plan_with_config(mk_metadata(64, 1_920, 64 * 16), cfg_cap2);
    check(
        "alwaysinsert_a_b_c_all_stored",
        stored_a && stored_b && stored_c,
        format!("a={stored_a} b={stored_b} c={stored_c}"),
    );
    check(
        "alwaysinsert_cap2_holds_2",
        shared_plan_cache_len() == 2,
        format!("len={}", shared_plan_cache_len()),
    );
    // After eviction, key 16 (oldest) should be evicted; 32 and 64 remain.
    let after_lookup_16 = lookup_shared_plan(&mk_key(16));
    let after_lookup_32 = lookup_shared_plan(&mk_key(32));
    let after_lookup_64 = lookup_shared_plan(&mk_key(64));
    check(
        "alwaysinsert_evicted_oldest_16",
        after_lookup_16.is_none(),
        format!("16_present={}", after_lookup_16.is_some()),
    );
    check(
        "alwaysinsert_kept_32",
        after_lookup_32.is_some(),
        format!("32_present={}", after_lookup_32.is_some()),
    );
    check(
        "alwaysinsert_kept_64",
        after_lookup_64.is_some(),
        format!("64_present={}", after_lookup_64.is_some()),
    );

    // === 4. CostWeightedLru rejects cheap plan when full ===
    clear_shared_plan_cache();
    let cfg_cw = PlanCacheConfig {
        capacity: 1,
        admission_policy: CacheAdmissionPolicy::CostWeightedLru,
        ..PlanCacheConfig::default()
    };
    let stored_big =
        store_shared_plan_with_config(mk_metadata(128, 4_480, 128 * 16), cfg_cw.clone());
    let stored_cheap = store_shared_plan_with_config(mk_metadata(8, 120, 8 * 16), cfg_cw);
    check(
        "cw_lru_accepts_first",
        stored_big,
        format!("big={stored_big}"),
    );
    check(
        "cw_lru_rejects_cheaper_when_full",
        !stored_cheap,
        format!("cheap={stored_cheap}"),
    );
    check(
        "cw_lru_big_still_present",
        lookup_shared_plan(&mk_key(128)).is_some(),
        format!("big_present={}", lookup_shared_plan(&mk_key(128)).is_some()),
    );
    check(
        "cw_lru_cheap_absent",
        lookup_shared_plan(&mk_key(8)).is_none(),
        format!("cheap_present={}", lookup_shared_plan(&mk_key(8)).is_some()),
    );

    // === 5. Working set bytes enforced by max_working_set_bytes ===
    clear_shared_plan_cache();
    let cfg_ws = PlanCacheConfig {
        capacity: 8,
        max_working_set_bytes: 160,
        admission_policy: CacheAdmissionPolicy::AlwaysInsert,
    };
    let stored_ws_1 = store_shared_plan_with_config(mk_metadata(16, 320, 64), cfg_ws.clone());
    let stored_ws_2 = store_shared_plan_with_config(mk_metadata(32, 800, 64), cfg_ws);
    let ws_bytes = shared_plan_cache_working_set_bytes();
    let len_after = shared_plan_cache_len();
    check(
        "ws_first_stored",
        stored_ws_1,
        format!("first={stored_ws_1}"),
    );
    check(
        "ws_second_stored_with_eviction",
        stored_ws_2,
        format!("second={stored_ws_2}"),
    );
    check(
        "ws_bytes_within_cap",
        ws_bytes <= 160,
        format!("bytes={ws_bytes}"),
    );
    check(
        "ws_only_one_fits",
        len_after == 1,
        format!("len={len_after}"),
    );

    // === 6. clear_shared_plan_cache resets to defaults ===
    clear_shared_plan_cache();
    check(
        "clear_empties_cache",
        shared_plan_cache_len() == 0 && shared_plan_cache_working_set_bytes() == 0,
        format!(
            "len={} bytes={}",
            shared_plan_cache_len(),
            shared_plan_cache_working_set_bytes()
        ),
    );

    // === 7. After clear, lookups return None ===
    check(
        "lookup_after_clear_none",
        lookup_shared_plan(&mk_key(64)).is_none() && lookup_shared_plan(&mk_key(32)).is_none(),
        String::new(),
    );

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_fft_shared_plan_cache".into(),
        category: "fsci_fft::plan shared plan cache public-API coverage".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("plan_cache mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "shared plan cache coverage failed: {} cases",
        diffs.len(),
    );
}
