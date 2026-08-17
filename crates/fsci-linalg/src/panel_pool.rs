//! Worker reuse across a factorization's panels — frankenscipy-ua3gn.
//!
//! ## Why this exists
//!
//! `fsci-linalg` spawns a fresh `std::thread::scope` per panel; there are 49 such
//! sites. At n=1000 the measured obstacle on ua3gn is that per-panel spawn/join
//! makes fan-out NET NEGATIVE below ~15M MACs per panel, so the parallel TRSM and
//! SYRK are gated off at exactly the sizes where the dense-BLAS wall is worst
//! (fsci 30.8 GF/s vs scipy-default 72.6, while per-core arithmetic is at parity).
//! The cost being amortized is the spawn, not the arithmetic.
//!
//! This module provides the substrate the bead asks for: a std-only scoped pool
//! whose workers stay alive across the whole k-loop, so a panel dispatch costs a
//! queue push and a condvar wake rather than a thread spawn.
//!
//! ## What it is NOT
//!
//! It is not wired into any factorization yet, and this file makes no performance
//! claim. Wiring it changes which code path runs at a given size, which is a
//! measurement question and needs a quiet window; landing the substrate with its
//! own tests is the separable half. No `tokio`, no `unsafe` — the crate is
//! `#![forbid(unsafe_code)]` and this module holds to that.
//!
//! ## Design
//!
//! Everything lives inside one `std::thread::scope`, which is what makes borrowing
//! caller data from tasks sound without `unsafe`: the scope cannot exit until every
//! worker has joined, so a task's borrows are guaranteed to outlive it.
//!
//! Workers park on a condvar and pull boxed closures from a shared queue. A batch
//! completes when its outstanding counter reaches zero. The boxing is per TASK,
//! which here means per panel-chunk, not per matrix element — an allocation at that
//! granularity is far below the spawn it replaces.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, PoisonError};

type Task<'env> = Box<dyn FnOnce() + Send + 'env>;

struct QueueState<'env> {
    tasks: VecDeque<Task<'env>>,
    /// Tasks dispatched in the current batch that have not yet finished. A batch
    /// is done when this reaches zero.
    outstanding: usize,
    shutdown: bool,
}

struct Shared<'env> {
    state: Mutex<QueueState<'env>>,
    /// Signalled when tasks are pushed or shutdown is requested.
    work_ready: Condvar,
    /// Signalled when `outstanding` reaches zero.
    batch_done: Condvar,
}

impl<'env> Shared<'env> {
    fn lock(&self) -> std::sync::MutexGuard<'_, QueueState<'env>> {
        // Poisoning is deliberately ignored. A panicking TASK must not convert
        // every later batch into a poisoned-mutex failure that buries the original
        // panic — the same reasoning as the toggle locks elsewhere in this crate.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Decrements the outstanding counter when a task finishes, INCLUDING when it
/// finishes by panicking.
///
/// This is the reason the pool cannot hang: without a `Drop` guard, a panicking
/// task would skip the decrement, `outstanding` would never reach zero, and
/// `run_batch` would wait forever. A deadlock is a far worse failure than a
/// propagated panic, because it produces no output to diagnose.
struct BatchGuard<'a, 'env> {
    shared: &'a Shared<'env>,
}

impl Drop for BatchGuard<'_, '_> {
    fn drop(&mut self) {
        let mut state = self.shared.lock();
        state.outstanding = state.outstanding.saturating_sub(1);
        if state.outstanding == 0 {
            self.shared.batch_done.notify_all();
        }
    }
}

/// A pool of workers that stay alive across many batches.
pub(crate) struct PanelPool<'a, 'env> {
    shared: &'a Shared<'env>,
    workers: usize,
}

impl<'env> PanelPool<'_, 'env> {
    /// Number of worker threads. A caller that wants to keep its own chunking
    /// decisions consistent with the pool should size chunks against this.
    pub(crate) fn workers(&self) -> usize {
        self.workers
    }

    /// Run every task and return once all of them have finished.
    ///
    /// Tasks may run in any order and on any worker; the caller is responsible for
    /// making them independent, exactly as it already must be with
    /// `thread::scope`. This method does not itself impose an order, so it cannot
    /// change the numerics of a caller whose tasks write disjoint slots.
    pub(crate) fn run_batch<I>(&self, tasks: I)
    where
        I: IntoIterator<Item = Task<'env>>,
    {
        let mut state = self.shared.lock();
        debug_assert_eq!(
            state.outstanding, 0,
            "run_batch called while a batch is still outstanding"
        );
        let before = state.tasks.len();
        state.tasks.extend(tasks);
        let dispatched = state.tasks.len() - before;
        if dispatched == 0 {
            return;
        }
        state.outstanding = dispatched;
        drop(state);
        self.shared.work_ready.notify_all();

        let mut state = self.shared.lock();
        while state.outstanding > 0 {
            state = self
                .shared
                .batch_done
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

/// Run `body` with a pool of `workers` threads that persist for its duration.
///
/// `workers` is clamped to at least 1. With one worker the tasks simply run on
/// that worker; the caller does not need a separate serial path.
pub(crate) fn with_panel_pool<'env, R>(
    workers: usize,
    body: impl FnOnce(&PanelPool<'_, 'env>) -> R,
) -> R {
    let workers = workers.max(1);
    let shared = Shared {
        state: Mutex::new(QueueState {
            tasks: VecDeque::new(),
            outstanding: 0,
            shutdown: false,
        }),
        work_ready: Condvar::new(),
        batch_done: Condvar::new(),
    };

    std::thread::scope(|scope| {
        let shared = &shared;
        for _ in 0..workers {
            scope.spawn(move || {
                loop {
                    let task = {
                        let mut state = shared.lock();
                        loop {
                            if let Some(task) = state.tasks.pop_front() {
                                break Some(task);
                            }
                            if state.shutdown {
                                break None;
                            }
                            state = shared
                                .work_ready
                                .wait(state)
                                .unwrap_or_else(PoisonError::into_inner);
                        }
                    };
                    match task {
                        Some(task) => {
                            // Guard first, so a panic inside `task` still decrements.
                            let _guard = BatchGuard { shared };
                            task();
                        }
                        None => break,
                    }
                }
            });
        }

        let pool = PanelPool { shared, workers };
        let result = body(&pool);

        // Retire the workers so `scope` can join them.
        {
            let mut state = shared.lock();
            state.shutdown = true;
        }
        shared.work_ready.notify_all();
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_task_in_a_batch_runs_exactly_once() {
        let counter = AtomicUsize::new(0);
        with_panel_pool(4, |pool| {
            let tasks: Vec<Task<'_>> = (0..64)
                .map(|_| Box::new(|| { counter.fetch_add(1, Ordering::Relaxed); }) as Task<'_>)
                .collect();
            pool.run_batch(tasks);
        });
        assert_eq!(
            counter.load(Ordering::Relaxed),
            64,
            "a task was dropped or run twice"
        );
    }

    #[test]
    fn tasks_write_their_own_disjoint_slots() {
        // The property a factorization actually depends on: task i writes slot i,
        // whichever worker picks it up. Order-independence is why this can replace
        // a per-panel scope without touching the numerics.
        let mut out = vec![0usize; 256];
        with_panel_pool(8, |pool| {
            let tasks: Vec<Task<'_>> = out
                .iter_mut()
                .enumerate()
                .map(|(i, slot)| Box::new(move || *slot = i * i) as Task<'_>)
                .collect();
            pool.run_batch(tasks);
        });
        assert!(
            out.iter().enumerate().all(|(i, &v)| v == i * i),
            "a task wrote the wrong slot, or a write was lost"
        );
    }

    #[test]
    fn workers_are_reused_across_batches_rather_than_respawned() {
        // THE POINT OF THE MODULE, so it is asserted rather than assumed: across
        // many batches the number of DISTINCT threads must stay bounded by the pool
        // size. A per-panel `thread::scope` would produce a fresh thread id per
        // batch and fail this.
        use std::collections::BTreeSet;
        let seen = Mutex::new(BTreeSet::new());
        const BATCHES: usize = 40;
        with_panel_pool(4, |pool| {
            for _ in 0..BATCHES {
                let tasks: Vec<Task<'_>> = (0..8)
                    .map(|_| {
                        Box::new(|| {
                            let id = format!("{:?}", std::thread::current().id());
                            seen.lock().unwrap_or_else(PoisonError::into_inner).insert(id);
                        }) as Task<'_>
                    })
                    .collect();
                pool.run_batch(tasks);
            }
        });
        let distinct = seen.lock().unwrap_or_else(PoisonError::into_inner).len();
        assert!(
            distinct <= 4,
            "{distinct} distinct threads ran {BATCHES} batches on a 4-worker pool; \
             the workers are being respawned, which is the cost this module exists \
             to remove"
        );
        // MUST-HIT on the other side: if only ONE thread ever ran, the pool is
        // serial and the reuse assertion above would hold vacuously.
        assert!(
            distinct > 1,
            "only one thread ran; the pool never fanned out, so the bound above \
             proves nothing"
        );
    }

    #[test]
    fn a_single_worker_pool_still_runs_everything() {
        let counter = AtomicUsize::new(0);
        with_panel_pool(1, |pool| {
            let tasks: Vec<Task<'_>> = (0..16)
                .map(|_| Box::new(|| { counter.fetch_add(1, Ordering::Relaxed); }) as Task<'_>)
                .collect();
            pool.run_batch(tasks);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 16);
    }

    #[test]
    fn an_empty_batch_returns_instead_of_waiting_forever() {
        with_panel_pool(4, |pool| {
            pool.run_batch(Vec::new());
        });
    }

    #[test]
    fn a_panicking_task_does_not_hang_the_batch() {
        // THIS TEST COMPLETING AT ALL IS THE ASSERTION. Without the `BatchGuard`,
        // a panicking task would skip the outstanding-counter decrement and
        // `run_batch` would block forever — the failure would show up as a hung
        // suite rather than a red, which is precisely why it is worth pinning.
        //
        // The panic is expected to propagate out of the scope; what must NOT
        // happen is a wait that never returns.
        let outcome = std::panic::catch_unwind(|| {
            with_panel_pool(4, |pool| {
                let tasks: Vec<Task<'_>> = vec![
                    Box::new(|| {}) as Task<'_>,
                    // The explicit `-> ()` is required, and a trailing `;` is NOT
                    // enough: a closure whose body diverges still infers `!` as its
                    // return type, and `!` does not coerce inside the boxed cast.
                    // Annotating the return type pins it to `()`.
                    Box::new(|| -> () {
                        panic!("task blew up");
                    }) as Task<'_>,
                    Box::new(|| {}) as Task<'_>,
                ];
                pool.run_batch(tasks);
            });
        });
        assert!(
            outcome.is_err(),
            "the panicking task did not surface; if it had been swallowed, a real \
             fault in a panel would vanish silently"
        );
    }
}
