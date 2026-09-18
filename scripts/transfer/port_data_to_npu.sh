#!/usr/bin/env bash
# Resumable H200 -> local -> NPU transfer for the CoRT data.
#
# The H200 has no useful egress (~1.4 MB/s aggregate) and no OBS access, so the
# only route is streaming through this machine. Everything is sharded and
# resumable: completed image shards leave a stamp file and are skipped on rerun.
#
# Usage:
#   bash scripts/transfer/port_data_to_npu.sh manifests   # ~400MB, minutes
#   bash scripts/transfer/port_data_to_npu.sh images      # ~45GB, many hours
#   bash scripts/transfer/port_data_to_npu.sh images 3    # 3 parallel shards
#
# Monitor:  tail -f ~/.cache/port_to_npu/transfer.log

set -uo pipefail

SRC_HOST="root@vr.turbo-ai.com"
SRC_PORT="${SRC_PORT:-20470}"
SRC_DIR="${SRC_DIR:-/private/codes/exp/deep_learning/data/datasets/cort_sft_133k/qwen_latent_cot_v3_adapted_20260816}"

NPU_KEY="${NPU_KEY:-$HOME/Downloads/KeyPair-zyd.pem}"
NPU_HOST="${NPU_HOST:-ma-user@dev.modelarts.cnszaismartcity01.api-ai.smartcitysz.com}"
NPU_PORT="${NPU_PORT:-32742}"
NPU_DIR="${NPU_DIR:-/home/ma-user/work/datasets/cort_sft_133k/qwen_latent_cot_v3_adapted_20260816}"

STATE_DIR="${STATE_DIR:-$HOME/.cache/port_to_npu}"
LOG="$STATE_DIR/transfer.log"
SHARD_SIZE="${SHARD_SIZE:-150}"

mkdir -p "$STATE_DIR/shards"

SRC_SSH=(ssh -p "$SRC_PORT" -o BatchMode=yes -o ConnectTimeout=20 "$SRC_HOST")
NPU_SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o BatchMode=yes -o ConnectTimeout=20 -i "$NPU_KEY" "$NPU_HOST" -p "$NPU_PORT")

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

prepare_dest() {
  "${NPU_SSH[@]}" "mkdir -p '$NPU_DIR/manifests' '$NPU_DIR/images/cort36k'"
}

transfer_manifests() {
  # Per file + byte verification: a single 400MB tar stream stalled in testing.
  local f
  for f in $("${SRC_SSH[@]}" "ls '$SRC_DIR/manifests'"); do
    local want got
    want=$("${SRC_SSH[@]}" "stat -c %s '$SRC_DIR/manifests/$f'")
    if [ "$("${NPU_SSH[@]}" "stat -c %s '$NPU_DIR/manifests/$f' 2>/dev/null || echo 0")" = "$want" ]; then
      log "manifest $f: skip (size ok)"; continue
    fi
    log "manifest $f: $want bytes"
    "${SRC_SSH[@]}" "cat '$SRC_DIR/manifests/$f'" \
      | "${NPU_SSH[@]}" "cat > '$NPU_DIR/manifests/$f'" || { log "manifest $f: FAILED"; return 1; }
    got=$("${NPU_SSH[@]}" "stat -c %s '$NPU_DIR/manifests/$f'")
    [ "$got" = "$want" ] || { log "manifest $f: size mismatch $got != $want"; return 1; }
  done
  "${SRC_SSH[@]}" "cat '$SRC_DIR/audit.json'" | "${NPU_SSH[@]}" "cat > '$NPU_DIR/audit.json'" || true

  log "manifests: rewriting paths -> $NPU_DIR"
  "${NPU_SSH[@]}" "cd '$NPU_DIR/manifests' && sed -i 's|$SRC_DIR|$NPU_DIR|g' *.jsonl && \
    head -n1 cort36k_train.jsonl | grep -o '\"image_paths\":\[[^]]*\]' | head -c 200"
}

shard_transfer() {
  local shard_file="$1"
  local shard_name="$2"
  local stamp="$STATE_DIR/shards/${shard_name}.done"
  [ -f "$stamp" ] && { log "shard $shard_name: skip (done)"; return 0; }

  local ids
  ids=$(tr '\n' ' ' < "$shard_file")
  if "${SRC_SSH[@]}" "cd '$SRC_DIR/images/cort36k' && tar -cf - $ids" \
      | "${NPU_SSH[@]}" "tar -C '$NPU_DIR/images/cort36k' -xf -"; then
    touch "$stamp"
    log "shard $shard_name: done"
  else
    log "shard $shard_name: FAILED (will retry on rerun)"
    return 1
  fi
}

transfer_images() {
  local parallel="${1:-6}"
  log "images: listing samples"
  "${SRC_SSH[@]}" "ls '$SRC_DIR/images/cort36k'" > "$STATE_DIR/samples.txt"
  local total; total=$(wc -l < "$STATE_DIR/samples.txt")
  log "images: $total samples, shard=$SHARD_SIZE, parallel=$parallel"

  split -l "$SHARD_SIZE" -d -a 4 "$STATE_DIR/samples.txt" "$STATE_DIR/shards/shard_"

  # Wave-based parallelism (bash 3.2 compatible: no `wait -n`).
  local batch=()
  for shard in "$STATE_DIR"/shards/shard_[0-9][0-9][0-9][0-9]; do
    batch+=("$shard")
    if [ "${#batch[@]}" -ge "$parallel" ]; then
      for s in "${batch[@]}"; do shard_transfer "$s" "$(basename "$s")" & done
      wait
      batch=()
    fi
  done
  if [ "${#batch[@]}" -gt 0 ]; then
    for s in "${batch[@]}"; do shard_transfer "$s" "$(basename "$s")" & done
    wait
  fi

  local done_n; done_n=$(ls "$STATE_DIR"/shards/*.done 2>/dev/null | wc -l)
  log "images: $done_n shards complete (rerun to resume)"
}

case "${1:-all}" in
  manifests) prepare_dest; transfer_manifests ;;
  images)    prepare_dest; transfer_images "${2:-6}" ;;
  all)       prepare_dest; transfer_manifests && transfer_images "${2:-6}" ;;
  *) echo "usage: $0 [manifests|images|all] [parallel]" >&2; exit 2 ;;
esac
