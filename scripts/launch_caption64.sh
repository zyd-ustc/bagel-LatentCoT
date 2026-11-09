#!/bin/bash
# Launch 64 Qwen3-VL caption shards: 4 workers x 16 chips.
set -u
CODE=/data/zyd_workspace/bagel-LatentCoT/code/isochronic-state-flow-20261106
PY=/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python
INPUT=/data/zyd_workspace/isochronic_data_20261106/caption_inputs_30k.jsonl
OUT=/data/zyd_workspace/isochronic_data_20261106/captions
LOG=/data/zyd_workspace/isochronic_data_20261106/logs
mkdir -p "$OUT" "$LOG"
HOSTS=(${VC_WORKER_HOSTS//,/ })
for W in 0 1 2 3; do
  H=${HOSTS[$W]}
  BASE=$((W * 16))
  CMD="cd $CODE || exit 1; mkdir -p $OUT $LOG; for j in \$(seq 0 15); do k=\$(( $BASE + j )); ASCEND_RT_VISIBLE_DEVICES=\$j setsid nohup $PY scripts/caption_sources.py --input $INPUT --out $OUT/shard_\$(printf '%02d' \$k).jsonl --shard \$k --shards 64 >> $LOG/caption_w${W}_s\${k}.log 2>&1 & sleep 2; done"
  if [ "$W" -eq 0 ]; then
    bash -lc "$CMD"
  else
    ssh -p 2222 -o BatchMode=yes root@"$H" "$CMD"
  fi
done
echo "launched 64 caption shards"
