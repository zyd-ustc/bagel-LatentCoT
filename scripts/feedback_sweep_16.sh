#!/usr/bin/env bash
set -euo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?usage: feedback_sweep_16.sh PREPARED_OUTPUT}
PYTHON=${MODEL_PYTHON:-/data/zyd_workspace/bagel-LatentCoT/venvs/bagel-NPU/bin/python}
WORLD=${SAMPLE_WORKERS:-16}
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
cd "$CODE"
[[ ! -e "$RUN/launch" ]] || { echo 'Use a fresh prepared run' >&2; exit 1; }
mkdir "$RUN/launch"
pids=()
for ((rank=0;rank<WORLD;rank++)); do
  printf -v name 'worker_%02d' "$rank"
  env ASCEND_RT_VISIBLE_DEVICES="$rank" OMP_NUM_THREADS=4 "$PYTHON" scripts/sample_feedback_sweep.py worker \
    --output "$RUN" --rank "$rank" --world "$WORLD" > "$RUN/launch/$name.log" 2>&1 < /dev/null &
  pids+=("$!");echo "$!" > "$RUN/launch/$name.pid"
done
failed=0
for pid in "${pids[@]}";do wait "$pid" || failed=1;done
[[ "$failed" == 0 ]] || { echo 'Sampling worker failed' >&2;exit 1; }
exec "$PYTHON" scripts/sample_feedback_sweep.py render --output "$RUN"
