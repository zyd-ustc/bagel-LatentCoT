# Phase 0.5: Frozen BAGEL T2I Read→Write Validation

## 1. Research question

Phase 0.5 tests one mechanism only:

> Given the same prompt and initial noise, does one latent Read→Write cycle improve
> complex composition without damaging BAGEL's native T2I prior?

Paired semantic editing is intentionally excluded. The archived edit runner and
`experiments/data/semantic_edit_phase05.jsonl` remain available for Phase 2.

## 2. Fixed T2I path

Every arm calls the native prompt-only path:

```python
inferencer(
    image=None,
    text=prompt,
    image_shapes=(height, width),
    init_noise=init_noise,
    cfg_text_scale=4.0,
    cfg_img_scale=1.0,
    cfg_interval=[0.4, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
)
```

For one prompt, every arm shares prompt, seed, initial noise, geometry, CFG, NFE,
and timestep schedule. Only `K`, `R`, loop body, persistence, and read/write timing
may change. Noise seeds use schema `bagel-loop-t2i-v1`.

## 3. Main arm matrix

| Arm | K | R | Body | Persist | Round 0 | Purpose |
|---|---:|---:|---|---|---|---|
| Z0 | 0 | 1 | — | off | — | native BAGEL T2I |
| Z1 | 8 | 2 | `[20,28)` | on | write | historical loop |
| Z2 | 8 | 2 | `[16,24)` | off | read | primary 1R+1W |
| Z3 | 8 | 2 | `[12,20)` | off | read | early body |
| Z4 | 8 | 2 | `[20,28)` | off | read | late body |
| Z5 | 8 | 2 | `[16,24)` | on | read | persistence |
| C0 | 8 | 2 | full depth | off | read | repeated-compute control |
| C1 | 8 | 1 | `[16,24)` | off | read only | causal isolation |
| C2 | 8 | 2 | `[16,24)` | off | write | immediate-write control |

`Z2` and `C2` differ only in whether Round 0 may write memory to non-memory
queries. `C1` reads generation state but never writes back, so its image and
velocity should remain close to `Z0` while memory diagnostics still change.

## 4. K-scaling matrix

Run `Z0`, `K1`, `K4`, and `K8` separately from the main matrix. Every K arm uses
`R=2`, strict Round-0 read, body `[16,24)`, and `persist=False`.

```bash
bash scripts/evaluate/run_bagel_loop_t2i_zeroshot.sh /data/outputs/t2i_main

K_VALUES=1,4,8 \
  bash scripts/evaluate/run_bagel_loop_t2i_zeroshot.sh /data/outputs/t2i_k
```

The launcher defaults to 16 NPU shards, 1024×1024, seed 42, and
`experiments/data/geneval2_hard_16.txt`. Override with `NUM_SHARDS`, `HEIGHT`,
`WIDTH`, `SEED`, `MODEL_PATH`, or `PROMPT_FILE`.

## 5. Outputs and scoring

The merge step writes:

- `index.html`: visual arm comparison;
- `manifest.json`: run metadata and per-arm GenEval image maps;
- `mechanism_summary.json`: aggregated `ΔM`, `ΔG`, and `Δv`;
- `geneval2_image_maps/*.json`: prompt-to-image maps for semantic scoring.

Pixel MAE versus Z0 is behavioral distance, not a quality score. Semantic quality
comes from GenEval2 Soft-TIFA, especially count, spatial relation, attribute
binding, and multi-object composition.

After starting the existing Soft-TIFA server:

```bash
python scripts/evaluate/score_bagel_loop_t2i_geneval2.py \
  --output-dir /data/outputs/t2i_main \
  --benchmark-data experiments/data/geneval2_hard_16.jsonl \
  --server-url http://127.0.0.1:5000
```

The scorer writes aggregate AM/GM, per-skill, per-atomicity, CSV, JSON, and a
Markdown summary using Z0 as the baseline.

## 6. Decision rule

Promote Z2 only if diagnostics establish `memory → GEN → velocity`, C1 remains
causally isolated from generation, and Z2 improves structural semantic scores
over both Z0 and the immediate-write C2 without unacceptable visual degradation.
