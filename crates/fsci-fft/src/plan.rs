use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::{Normalization, TransformKind};

/// How planning heuristics are produced for a transform key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PlanningStrategy {
    /// Static estimate only; cheapest and deterministic.
    #[default]
    EstimateOnly,
    /// Measure candidate paths and persist chosen plan metadata.
    MeasureAndPersist,
}

/// Admission mode controlling what enters the plan cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CacheAdmissionPolicy {
    Disabled,
    #[default]
    CostWeightedLru,
    AlwaysInsert,
}

/// Stable cache key for FFT planning decisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlanKey {
    pub kind: TransformKind,
    pub shape: Vec<usize>,
    pub axes: Vec<usize>,
    pub normalization: Normalization,
    pub real_input: bool,
}

impl PlanKey {
    #[must_use]
    pub fn new(
        kind: TransformKind,
        shape: Vec<usize>,
        axes: Vec<usize>,
        normalization: Normalization,
        real_input: bool,
    ) -> Self {
        Self {
            kind,
            shape,
            axes,
            normalization,
            real_input,
        }
    }
}

/// Fingerprint proving how a concrete FFT plan was selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanFingerprint {
    pub radix_path: Vec<usize>,
    pub estimated_flops: u64,
    pub scratch_bytes: usize,
}

/// Persistent metadata associated with a cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanMetadata {
    pub key: PlanKey,
    pub fingerprint: PlanFingerprint,
    pub generated_by: PlanningStrategy,
}

/// Control-plane configuration for plan caching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanCacheConfig {
    pub capacity: usize,
    pub max_working_set_bytes: usize,
    pub planning_strategy: PlanningStrategy,
    pub admission_policy: CacheAdmissionPolicy,
}

impl Default for PlanCacheConfig {
    fn default() -> Self {
        Self {
            capacity: 128,
            max_working_set_bytes: 64 * 1024 * 1024,
            planning_strategy: PlanningStrategy::EstimateOnly,
            admission_policy: CacheAdmissionPolicy::CostWeightedLru,
        }
    }
}

/// Storage interface to decouple planning from cache implementation
/// details.
pub trait PlanCacheBackend {
    fn lookup(&self, key: &PlanKey) -> Option<PlanMetadata>;
    fn store(&mut self, metadata: PlanMetadata) -> bool;
    fn config(&self) -> &PlanCacheConfig;
}

#[derive(Debug, Clone)]
pub struct BoundedPlanCache {
    config: PlanCacheConfig,
    entries: HashMap<PlanKey, PlanMetadata>,
    lru: VecDeque<PlanKey>,
    working_set_bytes: usize,
}

impl BoundedPlanCache {
    #[must_use]
    pub fn new(config: PlanCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            lru: VecDeque::new(),
            working_set_bytes: 0,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn working_set_bytes(&self) -> usize {
        self.working_set_bytes
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.working_set_bytes = 0;
    }

    fn lookup_and_touch(&mut self, key: &PlanKey) -> Option<PlanMetadata> {
        let metadata = self.entries.get(key).cloned()?;
        self.touch_key(key);
        Some(metadata)
    }

    fn contains_and_touch(&mut self, key: &PlanKey) -> bool {
        if !self.entries.contains_key(key) {
            return false;
        }
        self.touch_key(key);
        true
    }

    fn store_with_config(&mut self, metadata: PlanMetadata, config: PlanCacheConfig) -> bool {
        self.config = config;
        self.store(metadata)
    }

    fn touch_key(&mut self, key: &PlanKey) {
        self.remove_from_lru(key);
        self.lru.push_back(key.clone());
    }

    fn remove_from_lru(&mut self, key: &PlanKey) {
        if let Some(index) = self.lru.iter().position(|candidate| candidate == key) {
            self.lru.remove(index);
        }
    }

    fn metadata_working_set_bytes(metadata: &PlanMetadata) -> usize {
        metadata.fingerprint.scratch_bytes.saturating_add(
            metadata
                .fingerprint
                .radix_path
                .len()
                .saturating_mul(std::mem::size_of::<usize>()),
        )
    }

    fn can_consider(&self, metadata: &PlanMetadata) -> bool {
        if self.config.capacity == 0
            || self.config.max_working_set_bytes == 0
            || matches!(self.config.admission_policy, CacheAdmissionPolicy::Disabled)
        {
            return false;
        }

        let entry_bytes = Self::metadata_working_set_bytes(metadata);
        if entry_bytes > self.config.max_working_set_bytes {
            return false;
        }

        if matches!(
            self.config.admission_policy,
            CacheAdmissionPolicy::AlwaysInsert
        ) {
            return true;
        }

        if self.entries.len() < self.config.capacity
            && self.working_set_bytes.saturating_add(entry_bytes)
                <= self.config.max_working_set_bytes
        {
            return true;
        }

        let Some(min_existing_flops) = self
            .entries
            .values()
            .map(|existing| existing.fingerprint.estimated_flops)
            .min()
        else {
            return true;
        };

        metadata.fingerprint.estimated_flops >= min_existing_flops
    }

    fn evict_until_fit(&mut self, incoming_bytes: usize) {
        while self.entries.len() >= self.config.capacity
            || self.working_set_bytes.saturating_add(incoming_bytes)
                > self.config.max_working_set_bytes
        {
            if !self.evict_one() {
                break;
            }
        }
    }

    fn evict_one(&mut self) -> bool {
        if self.lru.is_empty() {
            return false;
        }

        let victim_index = if matches!(
            self.config.admission_policy,
            CacheAdmissionPolicy::CostWeightedLru
        ) {
            self.lru
                .iter()
                .take(8)
                .enumerate()
                .min_by_key(|(_, key)| {
                    self.entries
                        .get(*key)
                        .map_or(0, |metadata| metadata.fingerprint.estimated_flops)
                })
                .map_or(0, |(index, _)| index)
        } else {
            0
        };

        let Some(victim_key) = self.lru.remove(victim_index) else {
            return false;
        };
        if let Some(victim) = self.entries.remove(&victim_key) {
            self.working_set_bytes = self
                .working_set_bytes
                .saturating_sub(Self::metadata_working_set_bytes(&victim));
        }
        true
    }
}

impl Default for BoundedPlanCache {
    fn default() -> Self {
        Self::new(PlanCacheConfig::default())
    }
}

impl PlanCacheBackend for BoundedPlanCache {
    fn lookup(&self, key: &PlanKey) -> Option<PlanMetadata> {
        self.entries.get(key).cloned()
    }

    fn store(&mut self, metadata: PlanMetadata) -> bool {
        if !self.can_consider(&metadata) {
            return false;
        }

        let entry_bytes = Self::metadata_working_set_bytes(&metadata);
        if let Some(previous) = self.entries.remove(&metadata.key) {
            self.working_set_bytes = self
                .working_set_bytes
                .saturating_sub(Self::metadata_working_set_bytes(&previous));
            self.remove_from_lru(&metadata.key);
        }

        self.evict_until_fit(entry_bytes);
        if self.entries.len() >= self.config.capacity
            || self.working_set_bytes.saturating_add(entry_bytes)
                > self.config.max_working_set_bytes
        {
            return false;
        }

        self.working_set_bytes = self.working_set_bytes.saturating_add(entry_bytes);
        self.lru.push_back(metadata.key.clone());
        self.entries.insert(metadata.key.clone(), metadata);
        true
    }

    fn config(&self) -> &PlanCacheConfig {
        &self.config
    }
}

static SHARED_PLAN_CACHE: OnceLock<Mutex<BoundedPlanCache>> = OnceLock::new();

fn shared_cache() -> &'static Mutex<BoundedPlanCache> {
    SHARED_PLAN_CACHE.get_or_init(|| Mutex::new(BoundedPlanCache::default()))
}

/// Lock `mutex`, recovering from poison instead of propagating it.
///
/// A thread that panics while holding the plan-cache lock must not wedge the
/// cache for the rest of the process: the cache holds only derived planning
/// metadata, so the worst a partial update can cost is a stale or missing
/// plan, never a wrong transform. Clearing the flag (rather than only taking
/// `into_inner`) is what makes the recovery visible to a later plain `lock()`.
///
/// Factored out of [`lock_shared_cache`] so the recovery can be tested on a
/// caller-owned mutex; the process-global one cannot be tested deterministically
/// because every `fft()` in the suite touches it (frankenscipy-6d400).
fn lock_recovering<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            mutex.clear_poison();
            poisoned.into_inner()
        }
    }
}

fn lock_shared_cache() -> MutexGuard<'static, BoundedPlanCache> {
    lock_recovering(shared_cache())
}

#[must_use]
pub fn lookup_shared_plan(key: &PlanKey) -> Option<PlanMetadata> {
    lock_shared_cache().lookup_and_touch(key)
}

/// Return whether `key` is cached while updating its recency, without cloning
/// the cached plan metadata.
#[must_use]
pub fn touch_shared_plan(key: &PlanKey) -> bool {
    lock_shared_cache().contains_and_touch(key)
}

#[must_use]
pub fn store_shared_plan(metadata: PlanMetadata) -> bool {
    lock_shared_cache().store(metadata)
}

#[must_use]
pub fn store_shared_plan_with_config(metadata: PlanMetadata, config: PlanCacheConfig) -> bool {
    lock_shared_cache().store_with_config(metadata, config)
}

#[must_use]
pub fn shared_plan_cache_len() -> usize {
    lock_shared_cache().len()
}

#[must_use]
pub fn shared_plan_cache_working_set_bytes() -> usize {
    lock_shared_cache().working_set_bytes()
}

pub fn clear_shared_plan_cache() {
    *lock_shared_cache() = BoundedPlanCache::default();
}

/// Acquire a workspace-wide guard that serialises tests touching the
/// shared FFT plan cache. The cache is a process-global Mutex; cargo
/// test schedules unit tests in parallel by default, and unrelated tests
/// in different modules (notably plan.rs and transforms.rs) can race
/// while inserting / clearing / inspecting cache state. Every test
/// that calls `clear_shared_plan_cache`, `store_shared_plan*`, or
/// `lookup_shared_plan` should hold this guard for its duration.
/// Recovers from a poisoned mutex so a panicking sibling can't wedge
/// the rest of the suite (the underlying cache also handles poison via
/// `lock_shared_cache`). (frankenscipy-lw3rl)
#[cfg(test)]
pub(crate) fn shared_cache_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    match LOCK.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BoundedPlanCache, CacheAdmissionPolicy, PlanCacheBackend, PlanCacheConfig, PlanFingerprint,
        PlanKey, PlanMetadata, PlanningStrategy, clear_shared_plan_cache, lock_recovering,
        lookup_shared_plan, shared_cache, shared_plan_cache_len, store_shared_plan_with_config,
    };
    use crate::{Normalization, TransformKind};
    use std::sync::{Mutex, MutexGuard};

    // The shared_cache_* tests all mutate the same static Mutex<BoundedPlanCache>
    // exposed via shared_cache(). cargo test runs tests in parallel by default,
    // so they intermittently race and corrupt each other's expected state. We
    // serialize them via this test-only mutex; each test acquires the guard
    // first. Resolves [frankenscipy-a6f8b].
    //
    // This guard is necessary but NOT sufficient, and that is what made
    // frankenscipy-6d400 flaky: it only serialises tests that take it, while the
    // PRODUCTION path writes to the same process-global cache. Every `fft()`,
    // `rfft()`, `fftn()` and friend calls `transforms::touch_plan_cache`, which
    // stores a plan and clears cache-lock poison — so any of the ~190 other
    // tests in this crate can perturb the global cache's contents, its LRU
    // order, and its poison flag mid-test, no matter what guard is held. No
    // test-only mutex can fix that.
    //
    // The rule that follows: assertions about cache SEMANTICS (capacity, LRU
    // order, working-set limits, admission policy, poison recovery) run against
    // a caller-owned `BoundedPlanCache` / `Mutex`, where small capacities are
    // meaningful and nothing else can write. The global cache keeps only the
    // assertions concurrent traffic cannot invalidate: that the public wrappers
    // are really wired to it, and that it stays usable after a panic.
    fn serial_cache_lock() -> MutexGuard<'static, ()> {
        super::shared_cache_test_lock()
    }

    fn test_key(n: usize) -> PlanKey {
        PlanKey::new(
            TransformKind::Fft,
            vec![n],
            vec![0],
            Normalization::Backward,
            false,
        )
    }

    fn test_metadata(n: usize, estimated_flops: u64, scratch_bytes: usize) -> PlanMetadata {
        PlanMetadata {
            key: test_key(n),
            fingerprint: PlanFingerprint {
                radix_path: vec![2; n.trailing_zeros() as usize],
                estimated_flops,
                scratch_bytes,
            },
            generated_by: PlanningStrategy::EstimateOnly,
        }
    }

    #[test]
    fn default_cache_config_is_bounded_and_deterministic() {
        let config = PlanCacheConfig::default();
        assert_eq!(config.capacity, 128);
        assert_eq!(config.max_working_set_bytes, 64 * 1024 * 1024);
    }

    #[test]
    fn plan_key_captures_contract_surface() {
        let key = PlanKey::new(
            TransformKind::Fftn,
            vec![32, 32, 16],
            vec![0, 2],
            Normalization::Backward,
            false,
        );
        assert_eq!(key.kind, TransformKind::Fftn);
        assert_eq!(key.axes, vec![0, 2]);
    }

    /// The public wrappers really read and write the process-global cache
    /// (a no-op wrapper would be a live defect). Sized so concurrent traffic
    /// cannot invalidate it: the config below leaves room for far more entries
    /// than the whole suite can insert, so nothing can evict the stored key
    /// between the store and the lookup.
    #[test]
    fn shared_cache_roundtrip_works() {
        let _g = serial_cache_lock();
        clear_shared_plan_cache();
        let roomy = PlanCacheConfig {
            capacity: 4096,
            max_working_set_bytes: 64 * 1024 * 1024,
            admission_policy: CacheAdmissionPolicy::AlwaysInsert,
            ..PlanCacheConfig::default()
        };
        let key = PlanKey::new(
            TransformKind::Fft,
            vec![64],
            vec![0],
            Normalization::Backward,
            false,
        );
        let metadata = PlanMetadata {
            key: key.clone(),
            fingerprint: PlanFingerprint {
                radix_path: vec![2, 2, 2, 2, 2, 2],
                estimated_flops: 64 * 6 * 5,
                scratch_bytes: 64 * 16,
            },
            generated_by: PlanningStrategy::EstimateOnly,
        };
        assert!(store_shared_plan_with_config(metadata, roomy));
        assert!(
            shared_plan_cache_len() >= 1,
            "cache should contain at least the stored plan"
        );
        assert!(
            lookup_shared_plan(&key).is_some(),
            "the shared wrappers must read back what they wrote"
        );
    }

    /// Exact poison-recovery semantics, on a caller-owned mutex so the result
    /// is deterministic.
    ///
    /// This is the deterministic half of frankenscipy-6d400. The old test
    /// asserted `shared_cache().lock().is_err()` on the PROCESS-GLOBAL mutex
    /// right after poisoning it, which any concurrently running `fft()` could
    /// falsify by calling `lock_shared_cache` and clearing the flag first. The
    /// invariant it meant to pin lives in `lock_recovering`, so it is pinned
    /// here instead, where no other thread exists.
    #[test]
    fn lock_recovering_clears_poison_and_returns_the_value() {
        let mutex = Mutex::new(7u32);
        let poisoned = std::panic::catch_unwind(|| {
            let _guard = mutex.lock().expect("fresh mutex is unpoisoned");
            std::panic::resume_unwind(Box::new("poison the plan-cache mutex"));
        });
        assert!(poisoned.is_err(), "the probe must actually panic");

        // MUST-HIT: the mutex really is poisoned before recovery, and a plain
        // `lock()` alone would leave it that way — so it is `lock_recovering`,
        // not the passage of time or the guard drop, that clears the flag.
        assert!(mutex.lock().is_err(), "panic while held must poison");
        assert!(
            mutex.lock().is_err(),
            "a plain lock() must not clear the poison flag"
        );

        assert_eq!(*lock_recovering(&mutex), 7, "value survives the panic");
        assert!(
            mutex.lock().is_ok(),
            "lock_recovering must clear the poison flag"
        );
    }

    /// MUST-MISS arm: on an unpoisoned mutex `lock_recovering` is a plain lock
    /// and leaves the mutex healthy. Without this, a `lock_recovering` that
    /// somehow poisoned or corrupted the mutex would still pass the test above.
    #[test]
    fn lock_recovering_is_transparent_when_not_poisoned() {
        let mutex = Mutex::new(BoundedPlanCache::default());
        assert!(lock_recovering(&mutex).is_empty());
        lock_recovering(&mutex).store(test_metadata(64, 1_920, 64 * 16));
        assert_eq!(lock_recovering(&mutex).len(), 1);
        assert!(
            mutex.lock().is_ok(),
            "an unpoisoned mutex must stay unpoisoned"
        );
    }

    /// The shared cache stays usable after a thread panics while holding it.
    ///
    /// Only race-immune facts are asserted. Whether THIS test's panic is the
    /// one that leaves the flag set is not observable — a concurrent `fft()`
    /// may clear it first — but "usable afterwards" holds either way.
    #[test]
    fn shared_cache_stays_usable_after_a_panic_while_held() {
        let _g = serial_cache_lock();
        let poison_result = std::panic::catch_unwind(|| {
            let _guard = match shared_cache().lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            std::panic::resume_unwind(Box::new("poison shared FFT plan cache"));
        });
        assert!(poison_result.is_err());

        // Any shared-cache call routes through `lock_recovering`, so it must
        // neither panic nor leave the mutex poisoned for the rest of the suite.
        let _ = shared_plan_cache_len();
        assert!(
            shared_cache().lock().is_ok(),
            "shared cache operations should clear poison"
        );
    }

    #[test]
    fn cache_respects_disabled_admission_policy() {
        let mut cache = BoundedPlanCache::default();
        let config = PlanCacheConfig {
            admission_policy: CacheAdmissionPolicy::Disabled,
            ..PlanCacheConfig::default()
        };

        assert!(!cache.store_with_config(test_metadata(16, 320, 16 * 16), config));
        assert_eq!(cache.len(), 0, "disabled policy should not add entries");
    }

    #[test]
    fn cache_enforces_capacity_limit() {
        let mut cache = BoundedPlanCache::default();
        let config = PlanCacheConfig {
            capacity: 2,
            admission_policy: CacheAdmissionPolicy::AlwaysInsert,
            ..PlanCacheConfig::default()
        };

        assert!(cache.store_with_config(test_metadata(16, 320, 16 * 16), config.clone()));
        assert!(cache.store_with_config(test_metadata(32, 800, 32 * 16), config.clone()));
        assert!(cache.store_with_config(test_metadata(64, 1_920, 64 * 16), config));

        // Exact now that nothing else can insert: capacity 2, oldest evicted.
        assert_eq!(cache.len(), 2);
        assert!(cache.lookup_and_touch(&test_key(16)).is_none());
        assert!(cache.lookup_and_touch(&test_key(32)).is_some());
        assert!(cache.lookup_and_touch(&test_key(64)).is_some());
    }

    #[test]
    fn clone_free_touch_preserves_lru_eviction_order() {
        let mut cache = BoundedPlanCache::default();
        let config = PlanCacheConfig {
            capacity: 2,
            admission_policy: CacheAdmissionPolicy::AlwaysInsert,
            ..PlanCacheConfig::default()
        };

        assert!(cache.store_with_config(test_metadata(16, 320, 16 * 16), config.clone()));
        assert!(cache.store_with_config(test_metadata(32, 800, 32 * 16), config.clone()));
        // Touching 16 without cloning its metadata must still make it the most
        // recently used, so the next insert evicts 32 rather than 16.
        assert!(cache.contains_and_touch(&test_key(16)));
        assert!(!cache.contains_and_touch(&test_key(8)));
        assert!(cache.store_with_config(test_metadata(64, 1_920, 64 * 16), config));

        assert!(cache.lookup_and_touch(&test_key(16)).is_some());
        assert!(cache.lookup_and_touch(&test_key(32)).is_none());
        assert!(cache.lookup_and_touch(&test_key(64)).is_some());
    }

    #[test]
    fn cache_enforces_working_set_limit() {
        let mut cache = BoundedPlanCache::default();
        let config = PlanCacheConfig {
            capacity: 8,
            max_working_set_bytes: 160,
            admission_policy: CacheAdmissionPolicy::AlwaysInsert,
            ..PlanCacheConfig::default()
        };

        assert!(cache.store_with_config(test_metadata(16, 320, 64), config.clone()));
        assert!(cache.store_with_config(test_metadata(32, 800, 64), config));

        assert!(
            !cache.is_empty(),
            "working set eviction should leave at least one plan"
        );
        assert!(
            cache.working_set_bytes() <= 160,
            "working set must stay within its byte budget"
        );
    }

    #[test]
    fn cost_weighted_cache_rejects_cheap_plan_when_full() {
        let mut cache = BoundedPlanCache::default();
        let config = PlanCacheConfig {
            capacity: 1,
            admission_policy: CacheAdmissionPolicy::CostWeightedLru,
            ..PlanCacheConfig::default()
        };

        assert!(cache.store_with_config(test_metadata(128, 4_480, 128 * 16), config.clone()));
        assert!(!cache.store_with_config(test_metadata(8, 120, 8 * 16), config));

        assert!(cache.lookup_and_touch(&test_key(128)).is_some());
        assert!(cache.lookup_and_touch(&test_key(8)).is_none());
    }
}
