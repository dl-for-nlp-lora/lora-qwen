#!/usr/bin/env bash
# Pull instruct + stage6 checkpoints from exited RunPod pods and pack a shareable zip.
# Run locally after: export RUNPOD_API_KEY=...
#
# Usage:
#   bash scripts/export_checkpoints.sh
#   bash scripts/export_checkpoints.sh --skip-start   # pods already RUNNING
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/checkpoints-export"
ZIP="$ROOT/lora-qwen-checkpoints.zip"
API="https://rest.runpod.io/v1/pods"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15)

# pod_id:remote_subtree pairs (under /workspace/lora-qwen/)
PODS=(
  "hx5vb7neddwgs5:checkpoints/v5"
  "e3fe2d31z9l81e:checkpoints/v5"
  "028yeahc5upnw6:checkpoints/v4_instruct"
)

SKIP_START=0
[[ "${1:-}" == "--skip-start" ]] && SKIP_START=1

: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"

start_pod() {
  local id=$1
  local tries=${2:-45}
  local i resp
  if (( SKIP_START )); then
    return 0
  fi
  echo "[export] starting pod $id ..."
  for ((i = 1; i <= tries; i++)); do
    resp=$(curl -sS -X POST "$API/$id/start" -H "Authorization: Bearer $RUNPOD_API_KEY" || true)
    if echo "$resp" | grep -q '"desiredStatus":"RUNNING"'; then
      echo "[export]   $id RUNNING"
      return 0
    fi
    if echo "$resp" | grep -q 'not enough free GPUs'; then
      echo "[export]   $id waiting for host GPU ($i/$tries) ..."
      sleep 20
      continue
    fi
    echo "[export]   $id start response: $resp"
    sleep 10
  done
  echo "[export] ERROR: could not start $id" >&2
  return 1
}

wait_ssh() {
  local id=$1
  local ip port i
  for ((i = 1; i <= 60; i++)); do
    read -r ip port < <(curl -sS "$API/$id" -H "Authorization: Bearer $RUNPOD_API_KEY" \
      | python3 -c "import sys,json; p=json.load(sys.stdin); m=p.get('portMappings') or {}; print(p.get('publicIp') or '', m.get('22',''))")
    if [[ -n "$ip" && -n "$port" ]] && ssh "${SSH_OPTS[@]}" -p "$port" "root@$ip" 'echo ok' &>/dev/null; then
      echo "$ip" "$port"
      return 0
    fi
    sleep 5
  done
  echo "[export] ERROR: SSH not ready for $id" >&2
  return 1
}

pull_from_pod() {
  local id=$1 remote_sub=$2
  local ip port
  read -r ip port < <(wait_ssh "$id") || return 1
  echo "[export] rsync from $id ($ip:$port) $remote_sub ..."
  mkdir -p "$OUT_DIR"
  ssh "${SSH_OPTS[@]}" -p "$port" "root@$ip" \
    "test -d /workspace/lora-qwen/$remote_sub && du -sh /workspace/lora-qwen/$remote_sub || echo MISSING"
  rsync -az --no-owner --no-group \
    -e "ssh ${SSH_OPTS[*]} -p $port" \
    "root@$ip:/workspace/lora-qwen/$remote_sub/" \
    "$OUT_DIR/$remote_sub/"
}

mkdir -p "$OUT_DIR"
rm -f "$ZIP"

FAILED=0
for entry in "${PODS[@]}"; do
  id="${entry%%:*}"
  sub="${entry#*:}"
  if start_pod "$id"; then
    if ! pull_from_pod "$id" "$sub"; then
      echo "[export] WARN: pull failed for $id ($sub)" >&2
      FAILED=1
    fi
    curl -sS -X POST "$API/$id/stop" -H "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null || true
  else
    FAILED=1
  fi
done

python3 - "$OUT_DIR/MANIFEST.txt" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
paths = set()
for base in [Path("results/instruct"), Path("results/stage6")]:
    if not base.exists():
        continue
    for p in base.glob("*.json"):
        d = json.loads(p.read_text())
        for k in ("adapter_dir", "full_model_dir"):
            if d.get(k):
                paths.add(d[k])
        for pt in d.get("step_curve") or []:
            if pt.get("checkpoint"):
                paths.add(pt["checkpoint"])
lines = [
    "Expected checkpoint paths (from results JSONs in the PR):",
    "",
    "Extract the zip into the repo root so paths match eval commands, e.g.",
    "  checkpoints/v4_instruct/e1_v_proj",
    "  checkpoints/v5/full_ft_gsm8k_instruct",
    "",
]
for p in sorted(paths):
    lines.append(p)
out.write_text("\n".join(lines) + "\n")
print(f"Wrote {out}")
PY

if [[ ! -d "$OUT_DIR/checkpoints" ]]; then
  echo "[export] No checkpoints pulled — hosts may be out of GPU capacity." >&2
  echo "[export] Retry later: bash scripts/export_checkpoints.sh" >&2
  exit 1
fi

(
  cd "$OUT_DIR"
  zip -rq "$ZIP" checkpoints MANIFEST.txt
)
du -sh "$ZIP" "$OUT_DIR"
echo "[export] wrote $ZIP"
(( FAILED )) && echo "[export] completed with warnings (some pods unavailable)" >&2
