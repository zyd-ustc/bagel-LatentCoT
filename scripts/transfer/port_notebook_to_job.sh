#!/usr/bin/env bash
# Local driver: H200 (source) -> this Mac -> ModelArts training job (dest).
#
# The H200 egress is capped (~1 MB/s per stream), so everything is chunked and
# parallelised; the model's 29 GB single file is split ON THE SOURCE first to
# avoid random-access dd reads.
#
# Usage:
#   bash scripts/transfer/port_h200_to_job.sh code
#   bash scripts/transfer/port_h200_to_job.sh model [parallel]
#   bash scripts/transfer/port_h200_to_job.sh data  [parallel]
#   bash scripts/transfer/port_h200_to_job.sh all   [parallel]

set -uo pipefail

# ---- source: H200 ---------------------------------------------------------
SRC_HOST="${SRC_HOST:-root@vr.turbo-ai.com}"
SRC_PORT="${SRC_PORT:-20470}"
SRC_KEY="${SRC_KEY:-}"                      # empty -> default ssh key
SRC_MODEL="${SRC_MODEL:-/private/yida_workspace/models/BAGEL-7B-MoT}"
SRC_DATA="${SRC_DATA:-/private/codes/exp/deep_learning/data/datasets/cort_sft_133k/qwen_latent_cot_v3_adapted_20260816}"

# ---- dest: training job ---------------------------------------------------
DST_HOST="${DST_HOST:-root@dev.modelarts.cnszaismartcity01.api-ai.smartcitysz.com}"
DST_PORT="${DST_PORT:-32692}"
DST_KEY="${DST_KEY:-$HOME/Downloads/KeyPair-zyd.pem}"
DST_ROOT="${DST_ROOT:-/cache/bagel-LatentCoT}"
DST_MODEL="${DST_MODEL:-/data/bagel-LatentCoT/models/Bagel-7B-MoT}"
DST_DATA="${DST_DATA:-/data/bagel-LatentCoT/datasets/qwen_latent_cot_v3_adapted_20260816}"

STATE_DIR="${STATE_DIR:-$HOME/.cache/port_h200_to_job}"
LOG="$STATE_DIR/transfer.log"
mkdir -p "$STATE_DIR/chunks"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

if [ -n "$SRC_KEY" ]; then SRC_OPTS="-i $SRC_KEY"; else SRC_OPTS=""; fi
src_ssh() { ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes \
              $SRC_OPTS -p "$SRC_PORT" "$SRC_HOST" "$@"; }
dst_ssh() { ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes \
              -i "$DST_KEY" -p "$DST_PORT" "$DST_HOST" "$@"; }

check_src() {
  src_ssh 'echo UP' 2>&1 | grep -q UP || { log "SOURCE $SRC_HOST:$SRC_PORT unreachable"; exit 3; }
  log "source OK ($SRC_HOST)"
}
check_dst() {
  dst_ssh 'echo UP' 2>&1 | grep -q UP || { log "DEST $DST_HOST:$DST_PORT unreachable"; exit 3; }
  log "dest OK ($DST_HOST)"
}

do_code() {
  local repo="$HOME/Documents/LCoT-codex/bagel-LatentCoT"
  check_dst
  log "code: local -> job"
  ( cd "$repo" && tar --exclude='._*' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='.pytest_cache' --exclude='.ruff_cache' \
      -cf - qwen_latent_cot scripts configs tests experiments/data docs \
      pyproject.toml requirements.txt PLAN.md CHECKLIST.md README.md 2>/dev/null ) \
    | dst_ssh "mkdir -p '$DST_ROOT' && tar -C '$DST_ROOT' -xf - && find '$DST_ROOT' -name '._*' -delete && chmod +x '$DST_ROOT'/scripts/evaluate/*.sh 2>/dev/null; echo OK"
  log "code: done"
}

do_model() {
  local parallel="${1:-6}"
  check_src; check_dst
  dst_ssh "mkdir -p '$DST_MODEL'"

  log "model: small files"
  src_ssh "cd '$SRC_MODEL' && tar -cf - \$(find . -maxdepth 1 -type f ! -name 'ema.safetensors' ! -name 'ema.safetensors.part.*' | sed 's|^\./||')" \
    | dst_ssh "tar -C '$DST_MODEL' -xf -" && log "model: small files done"

  local big="ema.safetensors"
  log "model: split $big on source (sequential, skipped if present)"
  src_ssh "cd '$SRC_MODEL' && if ! ls $big.part.* >/dev/null 2>&1; then split -b 1G -d -a 3 $big $big.part.; fi; ls $big.part.* | wc -l"

  local parts
  parts=$(src_ssh "ls '$SRC_MODEL/$big.part.'*" | tr -d '\r')
  local n; n=$(printf '%s\n' $parts | wc -l | tr -d ' ')
  log "model: $n parts, parallel=$parallel"

  # md5 is the source of truth (size-only checks miss truncated writes).
  part_ok() {
    local src_path="$1" dst_path="$2" want got
    want=$(src_ssh "md5sum '$src_path' 2>/dev/null | awk '{print \$1}'" | tr -d '\r')
    got=$(dst_ssh "md5sum '$dst_path' 2>/dev/null | awk '{print \$1}'" | tr -d '\r')
    [ -n "$want" ] && [ "$want" = "$got" ]
  }

  local batch=()
  for p in $parts; do
    local base; base=$(basename "$p")
    if part_ok "$p" "$DST_MODEL/$base"; then log "part $base skip (md5 ok)"; continue; fi
    (
      src_ssh "cat '$p'" | dst_ssh "cat > '$DST_MODEL/$base'" \
        && log "part $base transferred" || log "part $base transfer FAILED"
    ) &
    batch+=($!)
    if [ "${#batch[@]}" -ge "$parallel" ]; then wait; batch=(); fi
  done
  wait

  # retry rounds until every part passes md5
  local attempt missing
  for attempt in 1 2 3; do
    missing=0
    for p in $parts; do
      local base; base=$(basename "$p")
      part_ok "$p" "$DST_MODEL/$base" && continue
      missing=$((missing+1))
      src_ssh "cat '$p'" | dst_ssh "cat > '$DST_MODEL/$base'" \
        && log "part $base ok (retry)" || log "part $base still failing"
    done
    [ "$missing" -eq 0 ] && break
    log "model: retry round $attempt, $missing part(s) missing"
  done

  # final per-part verification
  local ok_n=0
  for p in $parts; do part_ok "$p" "$DST_MODEL/$(basename "$p")" && ok_n=$((ok_n+1)); done
  log "model: $ok_n/$n parts verified by md5"

  if [ "$ok_n" -eq "$n" ]; then
    log "model: assembling"
    dst_ssh "cd '$DST_MODEL' && cat $big.part.* > $big && rm -f $big.part.*"
    if [ ! -f "$STATE_DIR/ema.md5" ]; then
      log "model: computing source md5 (one-off)"
      src_ssh "md5sum '$SRC_MODEL/$big'" > "$STATE_DIR/ema.md5"
    fi
    local src_md5 dst_md5
    src_md5=$(awk '{print $1}' "$STATE_DIR/ema.md5" | tr -d '\r')
    dst_md5=$(dst_ssh "md5sum '$DST_MODEL/$big' | awk '{print \$1}'" | tr -d '\r')
    if [ -n "$src_md5" ] && [ "$src_md5" = "$dst_md5" ]; then
      log "model: MD5 MATCH ($src_md5)"
      src_ssh "cd '$SRC_MODEL' && rm -f $big.part.*"
    else
      log "model: MD5 MISMATCH src=$src_md5 dst=$dst_md5 -- parts kept on dest, rerun to refetch"
    fi
  else
    log "model: incomplete ($ok_n/$n) -- rerun to resume"
  fi
}

do_data() {
  local parallel="${1:-6}"
  local shard_size="${SHARD_SIZE:-150}"
  check_src; check_dst
  dst_ssh "mkdir -p '$DST_DATA/manifests' '$DST_DATA/images/cort36k'"

  log "data: manifests"
  for f in $(src_ssh "ls '$SRC_DATA/manifests'"); do
    local want got
    want=$(src_ssh "stat -c %s '$SRC_DATA/manifests/$f'" | tr -d '\r')
    got=$(dst_ssh "stat -c %s '$DST_DATA/manifests/$f' 2>/dev/null || echo 0" | tr -d '\r')
    [ "$want" = "$got" ] && { log "manifest $f skip"; continue; }
    src_ssh "cat '$SRC_DATA/manifests/$f'" | dst_ssh "cat > '$DST_DATA/manifests/$f'"
    got=$(dst_ssh "stat -c %s '$DST_DATA/manifests/$f'" | tr -d '\r')
    [ "$want" = "$got" ] && log "manifest $f ok" || log "manifest $f SIZE MISMATCH"
  done
  dst_ssh "cd '$DST_DATA/manifests' && sed -i 's|$SRC_DATA|$DST_DATA|g' *.jsonl"

  log "data: listing samples"
  src_ssh "ls '$SRC_DATA/images/cort36k'" > "$STATE_DIR/samples.txt"
  split -l "$shard_size" -d -a 4 "$STATE_DIR/samples.txt" "$STATE_DIR/chunks/shard_" 2>/dev/null
  local total; total=$(ls "$STATE_DIR"/chunks/shard_[0-9][0-9][0-9][0-9] 2>/dev/null | wc -l | tr -d ' ')
  log "data: $total shards, parallel=$parallel"

  local batch=()
  for shard in "$STATE_DIR"/chunks/shard_[0-9][0-9][0-9][0-9]; do
    local stamp="$STATE_DIR/chunks/$(basename "$shard").done"
    [ -f "$stamp" ] && continue
    (
      ids=$(tr '\n' ' ' < "$shard")
      src_ssh "cd '$SRC_DATA/images/cort36k' && tar -cf - $ids" \
        | dst_ssh "tar -C '$DST_DATA/images/cort36k' -xf -" \
        && touch "$stamp" && log "shard $(basename "$shard") ok" \
        || log "shard $(basename "$shard") FAILED"
    ) &
    batch+=($!)
    if [ "${#batch[@]}" -ge "$parallel" ]; then wait; batch=(); fi
  done
  wait
  local done_n; done_n=$(ls "$STATE_DIR"/chunks/shard_*.done 2>/dev/null | wc -l | tr -d ' ')
  log "data: $done_n/$total shards done (rerun to resume)"
}

case "${1:-}" in
  code)  do_code ;;
  model) do_model "${2:-6}" ;;
  data)  do_data  "${2:-6}" ;;
  all)   do_code; do_model "${2:-6}"; do_data "${2:-6}" ;;
  *) echo "usage: $0 {code|model|data|all} [parallel]" >&2; exit 2 ;;
esac
