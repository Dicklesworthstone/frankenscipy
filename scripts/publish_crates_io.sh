#!/usr/bin/env bash
set -u

TOKEN="${CARGO_REGISTRY_TOKEN:-cioYJQTi6A08oHgh2x6XEiQR3PEMM0oKFnq}"

CRATES=(
  fsci-runtime
  fsci-constants
  fsci-datasets
  fsci-odr
  fsci-arrayapi
  fsci-fft
  fsci-io
  fsci-opt
  fsci-linalg
  fsci-integrate
  fsci-special
  fsci-sparse
  fsci-signal
  fsci-spatial
  fsci-stats
  fsci-interpolate
  fsci-ndimage
  fsci-cluster
  fsci-conformance
)

echo "=== FrankenSciPy Crates.io Publisher ==="
for c in "${CRATES[@]}"; do
  # Check if already published on crates.io
  if RCH_LOCAL=1 cargo search "$c" 2>/dev/null | grep -E "^${c} = \"0\.2\.0\"" >/dev/null; then
    echo "✓ $c v0.2.0 is already published on crates.io"
    continue
  fi

  echo "Targeting crate: $c"
  # Attempt publication with up to 5 immediate retries if close to boundary
  for attempt in {1..5}; do
    echo "Attempt $attempt for $c at $(date -u +%T)..."
    OUTPUT=$(CARGO_REGISTRY_TOKEN="$TOKEN" cargo publish -p "$c" 2>&1)
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
      echo "★ Successfully published $c v0.2.0 to crates.io!"
      exit 0
    fi

    if echo "$OUTPUT" | grep -q "429 Too Many Requests"; then
      RESET_TIME=$(echo "$OUTPUT" | grep -o "after [^.]*" | sed 's/after //')
      echo ">> Rate limited: $RESET_TIME"
      sleep 1
    else
      echo "$OUTPUT" | tail -n 5
      exit $STATUS
    fi
  done

  exit 42
done

echo "All 19 crates are published!"
exit 0
