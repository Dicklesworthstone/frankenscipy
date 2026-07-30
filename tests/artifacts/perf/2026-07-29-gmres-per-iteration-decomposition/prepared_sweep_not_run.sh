#!/usr/bin/env bash
# GMRES per-iteration decomposition sweep on thinkstation1.
# Retries ONLY on the fail-closed host-wide exclusivity gate (never on an
# unfavourable ratio) so the retry loop selects measurement windows, not results.
set -u

BIN=/data/projects/frankenscipy/target/release-perf/perf_sparse_vs_scipy
ORACLE=/data/projects/frankenscipy/crates/fsci-sparse/python/scipy_sparse_arm.py
OUT=/data/tmp/claude-1000/-data-projects-frankenscipy/75034167-be21-45be-8107-0ef93153876b/scratchpad/sweep
CPU=63
ROUNDS=21
MAXATT=40

mkdir -p "$OUT"
for side in 24 32 40 48 56 64 80 96 112 128; do
  log="$OUT/side_${side}.txt"
  att=0
  while :; do
    att=$((att+1))
    if taskset -c "$CPU" "$BIN" "$side" "$ROUNDS" gmres "$ORACLE" >"$log" 2>&1; then
      echo "side=$side OK attempts=$att"
      echo "gate_attempts=$att" >>"$log"
      break
    fi
    if ! grep -q "exclusivity failed" "$log"; then
      echo "side=$side HARD-FAIL attempts=$att"
      tail -3 "$log"
      break
    fi
    if [ "$att" -ge "$MAXATT" ]; then
      echo "side=$side GATE-EXHAUSTED attempts=$att"
      break
    fi
    sleep 3
  done
done
echo "sweep done"
