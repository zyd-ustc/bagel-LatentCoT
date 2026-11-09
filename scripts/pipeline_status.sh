#!/bin/bash
# One-screen status for the isochronic data and training pipeline.
set -u
D=/data/zyd_workspace/isochronic_data_20261107
LOG=/data/zyd_workspace/isochronic_data_20261107
count() { ls "$1" 2>/dev/null | wc -l; }
line() { printf '%-22s %s\n' "$1" "$2"; }

echo "== caches =="
for cache in i36b r r_edit; do
  [ -d "$D/$cache" ] || continue
  line "$cache markers" "$(count "$D/$cache/progress")"
  line "$cache manifests" "$(count "$D/$cache/manifest.shard_"* 2>/dev/null)"
  line "$cache records" "$(wc -l < "$D/$cache/production.jsonl" 2>/dev/null || echo 0)"
done
line "i cache size" "$(du -sh "$D/i36b" 2>/dev/null | cut -f1)"

echo "== processes =="
echo "local shards: $(ps aux | grep -c '[p]repare-state-pairs\|[u]15_target_correction.cli')"
echo "train runs:   $(ps aux | grep -c '[t]rain --distributed')"

echo "== runs =="
for run in "$D"/runs/*/; do
  [ -d "$run" ] || continue
  steps=$(count "$run"/step_*)
  echo "$(basename "$run"): steps=$steps $(tail -1 "$run/train.jsonl" 2>/dev/null | head -c 220)"
done

echo "== eval =="
for album in linear32 sample64 keep16; do
  [ -d "$D/eval/$album" ] && line "$album" "$(count "$D/eval/$album"/case_*.json) cases"
done
