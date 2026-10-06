#!/usr/bin/env bash
# User-run evaluation; GPUS can select four or eight allocated cards.
set -euo pipefail
export ARMS=${ARMS:-BASE,MEMORY_LOOP,LAYERWISE_MEMORY_KV}
exec bash "$(dirname "$0")/run_memory_loop_8gpu.sh" "$@"
