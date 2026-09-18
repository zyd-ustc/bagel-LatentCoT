#!/usr/bin/env bash
# Run ON the ModelArts training container (root@...:32692).
# Sets up paths + env file and verifies the environment. Idempotent.
set -uo pipefail

ROOT=${ROOT:-/cache/bagel-LatentCoT}
MODEL=${MODEL:-/cache/models/Bagel-7B-MoT}
DATA=${DATA:-/data/bagel-LatentCoT/datasets/qwen_latent_cot_v3_adapted_20260816}
PY=${PY:-/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python}

mkdir -p "$ROOT" "$(dirname "$MODEL")" "$(dirname "$DATA")"
cat > "$ROOT/env.sh" <<ENVEOF
export LCOT_ROOT=$ROOT
export LCOT_MODEL=$MODEL
export LCOT_DATA=$DATA
export LCOT_PY=$PY
export ASCEND_RT_VISIBLE_DEVICES=\${ASCEND_RT_VISIBLE_DEVICES:-0}
ENVEOF
echo "[env] wrote $ROOT/env.sh"

echo "=== npu ==="; npu-smi info 2>/dev/null | sed -n '5,12p'
echo "=== python ==="; "$PY" -c "import torch,torch_npu,transformers,safetensors;print('torch',torch.__version__,'| npu',torch.npu.is_available(),'| transformers',transformers.__version__)"
echo "=== code ==="; [ -d "$ROOT/qwen_latent_cot" ] && echo "code OK ($ROOT)" || echo "code MISSING -> run sync"
echo "=== model ==="; [ -f "$MODEL/ema.safetensors" ] && echo "model OK ($MODEL)" || echo "model MISSING ($MODEL)"
echo "=== data ==="; [ -f "$DATA/manifests/cort36k_train.jsonl" ] && echo "data OK ($DATA)" || echo "data MISSING ($DATA)"
echo "=== eval scripts ==="; ls "$ROOT/scripts/evaluate" 2>/dev/null | grep -v pycache | head
