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

fn lock_shared_cache() -> MutexGuard<'static, BoundedPlanCache> {
    let cache = shared_cache();
    match cache.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            cache.clear_poison();
            poisoned.into_inner()
        }
    }
}

#[must_use]
pub fn lookup_shared_plan(key: &PlanKey) -> Option<PlanMetadata> {
    lock_shared_cache().lookup_and_touch(key)
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
        CacheAdmissionPolicy, PlanCacheConfig, PlanFingerprint, PlanKey, PlanMetadata,
        PlanningStrategy, clear_shared_plan_cache, lookup_shared_plan, shared_cache,
        shared_plan_cache_len, store_shared_plan, store_shared_plan_with_config,
    };
    use crate::{Normalization, TransformKind};
    use std::sync::MutexGuard;

    // The shared_cache_* tests all mutate the same static Mutex<BoundedPlanCache>
    // exposed via shared_cache(). cargo test runs tests in parallel by default,
    // so they intermittently race and corrupt each other's expected state. We
    // serialize them via this test-only mutex; each test acquires the guard
    // first. Resolves [frankenscipy-a6f8b].
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

    #[test]
    fn shared_cache_roundtrip_works() {
        let _g = serial_cache_lock();
        clear_shared_plan_cache();
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
        assert!(store_shared_plan(metadata));
        assert!(
            shared_plan_cache_len() >= 1,
            "cache should contain at least the stored plan"
        );
        assert!(lookup_shared_plan(&key).is_some());
    }

    #[test]
    fn shared_cache_recovers_after_poisoned_lock() {
        let _g = serial_cache_lock();
        clear_shared_plan_cache();
        let poison_result = std::panic::catch_unwind(|| {
            let _guard = match shared_cache().lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            std::panic::resume_unwind(Box::new("poison shared FFT plan cache"));
        });
        assert!(poison_result.is_err());
        assert!(shared_cache().lock().is_err(), "test should poison cache");

        let metadata = test_metadata(256, 10_240, 256 * 16);
        let key = metadata.key.clone();
        assert!(store_shared_plan(metadata));
        assert!(
            shared_plan_cache_len() >= 1,
            "cache should contain at least the stored plan"
        );
        assert!(lookup_shared_plan(&key).is_some());
        assert!(
            shared_cache().lock().is_ok(),
            "shared cache operations should clear poison"
        );

        clear_shared_plan_cache();
    }

    #[test]
    fn shared_cache_respects_disabled_admission_policy() {
        let _g = serial_cache_lock();
        clear_shared_plan_cache();
        let config = PlanCacheConfig {
            admission_policy: CacheAdmissionPolicy::Disabled,
            ..PlanCacheConfig::default()
        };
        let metadata = test_metadata(16, 320, 16 * 16);

        let len_before = shared_plan_cache_len();
        assert!(!store_shared_plan_with_config(metadata, config));
        assert_eq!(
            shared_plan_cache_len(),
            len_before,
            "disabled policy should not add entries"
        );
    }

    #[test]
    fn shared_cache_enforces_capacity_limit() {
        let _g = serial_cache_lock();
        clear_shared_plan_cache();
        let config = PlanCacheConfig {
            capacity: 2,
            admission_policy: CacheAdmissionPolicy::AlwaysInsert,
            ..PlanCacheConfig::default()
        };

        assert!(store_shared_plan_with_config(
            test_metadata(16, 320, 16 * 16),
            config.clone()
        ));
        assert!(store_shared_plan_with_config(
            test_metadata(32, 800, 32 * 16),
            config.clone()
        ));
        assert!(store_shared_plan_with_config(
            test_metadata(64, 1_920, 64 * 16),
            config
        ));

        assert!(
            shared_plan_cache_len() >= 2,
            "cache should contain at least 2 entries after capacity limit"
        );
        assert!(lookup_shared_plan(&test_key(16)).is_none());
        assert!(lookup_shared_plan(&test_key(32)).is_some());
        assert!(lookup_shared_plan(&test_key(64)).is_some());
    }

    #[test]
    fn shared_cache_enforces_working_set_limit() {
        let _g = serial_cache_lock();
        clear_shared_plan_cache();
        let config = PlanCacheConfig {
            capacity: 8,
            max_working_set_bytes: 160,
            admission_policy: CacheAdmissionPolicy::AlwaysInsert,
            ..PlanCacheConfig::default()
        };

        assert!(store_shared_plan_with_config(
            test_metadata(16, 320, 64),
            config.clone()
        ));
        assert!(store_shared_plan_with_config(
            test_metadata(32, 800, 64),
            config
        ));

        assert!(
            shared_plan_cache_len() >= 1,
            "working set eviction should leave at least one plan"
        );
    }

    #[test]
    fn cost_weighted_cache_rejects_cheap_plan_when_full() {
        let _g = serial_cache_lock();
        clear_shared_plan_cache();
        let config = PlanCacheConfig {
            capacity: 1,
            admission_policy: CacheAdmissionPolicy::CostWeightedLru,
            ..PlanCacheConfig::default()
        };

        assert!(store_shared_plan_with_config(
            test_metadata(128, 4_480, 128 * 16),
            config.clone()
        ));
        assert!(!store_shared_plan_with_config(
            test_metadata(8, 120, 8 * 16),
            config
        ));

        assert!(lookup_shared_plan(&test_key(128)).is_some());
        assert!(lookup_shared_plan(&test_key(8)).is_none());
    }
}
