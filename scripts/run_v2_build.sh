#!/usr/bin/env bash
# Full v2 data build on modelarts-64 (16 chips).
#
# Ordered so no chip time is wasted: scoring is CPU-only and runs while nothing
# else holds the card, then teacher production takes the card, then acceptance
# (also CPU) runs over whatever came out.
set -u
ROOT=/data/zyd_workspace/isochronic_data_20261107
CODE=/data/zyd_workspace/bagel-LatentCoT/code/isochronic-v2-20261110
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python
MODEL=/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT
SRC=/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b
CFG=$CODE/configs/u15_isochronic_state/terminal_active.json
ACTIVE=$ROOT/active

stage() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$ROOT/v2_build.log"; }

# The pilot and the full run must never share a file: rebuilding the pilot list
# used to overwrite the full one, so a later run silently produced 48 records.
build_requests() {
  local out=$1 limit=${2:-}
  stage "build requests -> $out"
  python3 "$CODE/scripts/build_requests.py" \
    --scores "$ROOT/parent_scores.jsonl" \
    --plan "$ACTIVE/plan/all42k.jsonl" \
    --out "$out" \
    --parent-cache "$ACTIVE/mt" "$ACTIVE/mt_b" \
    ${limit:+--limit "$limit"} 2>&1 | tee -a "$ROOT/v2_build.log"
}

produce() {
  local out=$1 limit=${2:-}
  stage "teacher production -> $out"
  cd "$CODE" || exit 1
  PYTHONPATH="$CODE" ASCEND_RT_VISIBLE_DEVICES=0 \
    PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" \
    OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    "$PY" -m scripts.prepare_coupled \
      --model "$MODEL" --source "$SRC" --config "$CFG" \
      --requests "$REQUESTS" \
      --source-cache "$ACTIVE/mt" "$ACTIVE/mt_b" \
      --output "$out" ${limit:+--limit "$limit"} \
      >> "$ROOT/v2_build.log" 2>&1
  stage "teacher production done rc=$?"
}

accept() {
  local cache=$1 tag=$2
  stage "acceptance -> $tag"
  METTO_BASE_URL="https://metto-api.com" METTO_GLM_KEY="${METTO_GLM_KEY:?}" \
    python3 "$CODE/scripts/accept_final.py" --cache "$cache" \
      --out-approved "$ROOT/${tag}_approved.jsonl" \
      --out-rejected "$ROOT/${tag}_rejected.jsonl" \
      --report "$ROOT/${tag}_report.json" \
      --workers 96 2>&1 | tee -a "$ROOT/v2_build.log"
}

case "${1:-all}" in
  requests) build_requests "$ROOT/requests_full.jsonl" "${2:-}" ;;
  pilot)    REQUESTS=$ROOT/requests_pilot.jsonl
            build_requests "$REQUESTS" 48
            produce "$ROOT/v2_pilot" 48
            accept "$ROOT/v2_pilot" "v2_pilot" ;;
  full)     REQUESTS=$ROOT/requests_full.jsonl
            [ -f "$REQUESTS" ] || build_requests "$REQUESTS" "${2:-}"
            produce "$ROOT/v2_full"
            accept "$ROOT/v2_full" "v2_full" ;;
  *)        echo "usage: $0 {requests|pilot|full} [limit]" >&2 ; exit 2 ;;
esac
