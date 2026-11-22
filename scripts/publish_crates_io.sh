#!/usr/bin/env bash
set -euo pipefail

# Whole-workspace crates.io publisher for FrankenSciPy.
#
# Derives workspace crate names, versions, and a deterministic leaves-first
# topological order from locked Cargo metadata, checks exact crates.io version
# presence, runs verified publication by default (never silently bypassing
# package verification), continues across every crate, and emits a per-package
# reconciliation summary without ever logging credentials.

usage() {
  cat <<'EOF'
Usage: scripts/publish_crates_io.sh [--plan | --dry-run] [--allow-dirty]
                                    [--no-verify --verified-receipt FILE]
                                    [--max-retries N] [--retry-delay SEC]

Options:
  --plan                     Print resolved topological publish order and exit 0.
  --dry-run                  Check crates.io presence and run `cargo publish --dry-run`
                             without uploading; does not require CARGO_REGISTRY_TOKEN.
  --allow-dirty              Allow running on a dirty git worktree (mainly for --plan/--dry-run).
  --no-verify                Pass `--no-verify` to `cargo publish`; requires `--verified-receipt`.
  --verified-receipt FILE    Non-empty receipt proving package verification already passed.
  --max-retries N            Max 429 retry attempts per crate (default: 30).
  --retry-delay SEC          Seconds to sleep after a 429 response when no header gives a wait (default: 15).
  -h, --help                 Show this help message.
EOF
}

MODE="publish"
ALLOW_DIRTY=0
NO_VERIFY=0
VERIFIED_RECEIPT=""
MAX_RETRIES=30
RETRY_DELAY=15

while [ "$#" -gt 0 ]; do
  case "$1" in
    --plan)
      MODE="plan"
      shift
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --no-verify)
      NO_VERIFY=1
      shift
      ;;
    --verified-receipt)
      VERIFIED_RECEIPT="${2:-}"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="${2:-}"
      shift 2
      ;;
    --retry-delay)
      RETRY_DELAY="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$NO_VERIFY" -eq 1 ]; then
  if [ -z "$VERIFIED_RECEIPT" ] || [ ! -s "$VERIFIED_RECEIPT" ]; then
    echo "ERROR: --no-verify requires a non-empty --verified-receipt FILE" >&2
    exit 2
  fi
fi

if [ "$MODE" = "publish" ]; then
  : "${CARGO_REGISTRY_TOKEN:?Set CARGO_REGISTRY_TOKEN from your existing registry credentials}"
fi

if [ "$ALLOW_DIRTY" -eq 0 ] && command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ERROR: tracked git worktree has uncommitted changes; commit cleanly or pass --allow-dirty" >&2
    exit 1
  fi
fi

# Resolve exact workspace member names, versions, and a deterministic leaves-first
# topological sort over normal + build workspace dependencies from locked metadata.
PLAN_LINES=()
while IFS= read -r line; do
  [ -n "$line" ] && PLAN_LINES+=("$line")
done < <(
  cargo metadata --locked --no-deps --format-version 1 | python3 -c '
import json, sys
from collections import defaultdict

meta = json.load(sys.stdin)
ws_ids = set(meta["workspace_members"])
pkgs = {p["name"]: p for p in meta["packages"] if p["id"] in ws_ids and p.get("publish") != []}

# Verify all publishable workspace crates share a single consistent release version.
versions = {p["version"] for p in pkgs.values()}
if len(versions) != 1:
    sys.stderr.write(f"ERROR: workspace publishable crates have mismatched versions: {sorted(versions)}\n")
    sys.exit(1)

deps = {name: set() for name in pkgs}
rdeps = defaultdict(set)
for name, p in pkgs.items():
    for d in p.get("dependencies", []):
        kind = d.get("kind")
        if kind in (None, "build") and d["name"] in pkgs:
            deps[name].add(d["name"])
            rdeps[d["name"]].add(name)

# Deterministic Kahn topological sort (lexicographical tie-break within each layer).
indeg = {name: len(ds) for name, ds in deps.items()}
ready = sorted(n for n, deg in indeg.items() if deg == 0)
order = []
while ready:
    cur = ready.pop(0)
    order.append(cur)
    for nxt in sorted(rdeps[cur]):
        indeg[nxt] -= 1
        if indeg[nxt] == 0:
            ready.append(nxt)
            ready.sort()

if len(order) != len(pkgs):
    cycle = sorted(set(pkgs) - set(order))
    sys.stderr.write(f"ERROR: workspace dependency cycle detected among: {cycle}\n")
    sys.exit(1)

for name in order:
    ver = pkgs[name]["version"]
    print(f"{name}\t{ver}")
'
)

if [ "${#PLAN_LINES[@]}" -eq 0 ]; then
  echo "ERROR: no publishable workspace crates found in cargo metadata" >&2
  exit 1
fi

echo "=== FrankenSciPy Crates.io Publisher (mode=${MODE}, crates=${#PLAN_LINES[@]}) ==="
if [ "$MODE" = "plan" ]; then
  idx=0
  for entry in "${PLAN_LINES[@]}"; do
    idx=$((idx + 1))
    IFS=$'\t' read -r crate ver <<<"$entry"
    printf "%2d. %s %s\n" "$idx" "$crate" "$ver"
  done
  exit 0
fi

# Check whether an exact crate version is already live (and not yanked) on crates.io.
# Prints the published checksum on stdout and returns 0 if live, 1 if absent, 2 on query error.
check_registry_version() {
  local crate="$1"
  local ver="$2"
  local url="https://crates.io/api/v1/crates/${crate}/${ver}"
  local resp http_code body
  resp="$(curl -sS -w $'\n%{http_code}' "$url" || true)"
  http_code="${resp##*$'\n'}"
  body="${resp%$'\n'*}"
  if [ "$http_code" = "404" ]; then
    return 1
  fi
  if [ "$http_code" != "200" ]; then
    echo "WARNING: crates.io returned HTTP ${http_code} for ${crate} ${ver}" >&2
    return 2
  fi
  python3 -c '
import json, sys
doc = json.load(sys.stdin)
v = doc.get("version") or {}
if v.get("num") != sys.argv[1]:
    sys.exit(1)
if v.get("yanked"):
    sys.stderr.write(f"ERROR: {sys.argv[2]} {sys.argv[1]} exists on crates.io but is YANKED\n")
    sys.exit(2)
print(v.get("checksum", ""))
' "$ver" "$crate" <<<"$body"
}

PUBLISHED=()
ALREADY_PUBLISHED=()
FAILED=()
PENDING=()

ALL_NAMES=()
for entry in "${PLAN_LINES[@]}"; do
  IFS=$'\t' read -r crate _ver <<<"$entry"
  ALL_NAMES+=("$crate")
done

ABORTED=0
for i in "${!PLAN_LINES[@]}"; do
  IFS=$'\t' read -r crate ver <<<"${PLAN_LINES[$i]}"

  if [ "$ABORTED" -eq 1 ]; then
    PENDING+=("${crate}@${ver}")
    continue
  fi

  if checksum="$(check_registry_version "$crate" "$ver")"; then
    echo "[already-published] ${crate} ${ver} (sha256:${checksum})"
    ALREADY_PUBLISHED+=("${crate}@${ver}")
    continue
  else
    rc=$?
    if [ "$rc" -eq 2 ]; then
      echo "[failed] ${crate} ${ver}: registry presence check failed" >&2
      FAILED+=("${crate}@${ver}")
      ABORTED=1
      continue
    fi
  fi

  publish_args=(publish --locked -p "$crate")
  if [ "$MODE" = "dry-run" ]; then
    publish_args+=(--dry-run)
  fi
  if [ "$NO_VERIFY" -eq 1 ]; then
    publish_args+=(--no-verify)
  fi

  echo "[targeting] ${crate} ${ver}"
  succeeded=0
  for ((attempt = 1; attempt <= MAX_RETRIES; attempt++)); do
    echo "  attempt ${attempt}/${MAX_RETRIES} at $(date -u +%T) UTC..."
    set +e
    OUTPUT="$(cargo "${publish_args[@]}" 2>&1)"
    STATUS=$?
    set -e

    if [ "$STATUS" -eq 0 ]; then
      if [ "$MODE" = "dry-run" ]; then
        echo "[dry-run-ok] ${crate} ${ver}"
      else
        echo "[published] ${crate} ${ver}"
      fi
      PUBLISHED+=("${crate}@${ver}")
      succeeded=1
      break
    fi

    if printf '%s' "$OUTPUT" | grep -qiE "429 Too Many Requests|rate limit"; then
      echo "  rate-limited by crates.io; sleeping ${RETRY_DELAY}s before retry..."
      sleep "$RETRY_DELAY"
      continue
    fi

    if printf '%s' "$OUTPUT" | grep -qiE "already exists|already uploaded"; then
      echo "[already-published] ${crate} ${ver} (confirmed by registry response)"
      ALREADY_PUBLISHED+=("${crate}@${ver}")
      succeeded=1
      break
    fi

    printf '%s\n' "$OUTPUT" | tail -n 20 >&2
    FAILED+=("${crate}@${ver}")
    ABORTED=1
    break
  done

  if [ "$succeeded" -eq 0 ] && [ "$ABORTED" -eq 0 ]; then
    echo "[failed] ${crate} ${ver}: exhausted ${MAX_RETRIES} retries" >&2
    FAILED+=("${crate}@${ver}")
    ABORTED=1
  fi
done

echo ""
echo "=== Publication Summary ==="
echo "  published (${#PUBLISHED[@]}): ${PUBLISHED[*]:-none}"
echo "  already-published (${#ALREADY_PUBLISHED[@]}): ${ALREADY_PUBLISHED[*]:-none}"
echo "  failed (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "  pending (${#PENDING[@]}): ${PENDING[*]:-none}"

if [ "${#FAILED[@]}" -gt 0 ] || [ "${#PENDING[@]}" -gt 0 ]; then
  exit 1
fi
exit 0
