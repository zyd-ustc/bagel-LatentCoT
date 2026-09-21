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
| Z2 | 8 | 2 | `[16,24)` | off | read | mid body, fresh memory |
| Z3 | 8 | 2 | `[12,20)` | off | read | early body |
| Z4 | 8 | 2 | `[20,28)` | off | read | late body |
| Z6 | 8 | 2 | `[12,20)` | on | read | early body, persistent memory |

`Z3↔Z6` is the retained matched persistence comparison. It differs only in whether
the final memory state is carried into the next denoising timestep. Z5 and Z7 were
removed after the 128-prompt pilot did not support retaining their persistence
windows. All loop arms use strict `1R+1W` and `K=8`.

## 4. K-scaling matrix

Run `Z0`, `K1`, `K4`, and `K8` separately from the main matrix. Every K arm uses
`R=2`, strict Round-0 read, body `[16,24)`, and `persist=False`.

```bash
bash scripts/evaluate/run_bagel_loop_t2i_zeroshot.sh /data/outputs/t2i_main

K_VALUES=1,4,8 \
  bash scripts/evaluate/run_bagel_loop_t2i_zeroshot.sh /data/outputs/t2i_k
```

The launcher defaults to 16 NPU shards, 1024×1024, seed 42, and all 800 official
GenEval2 prompts. Atomicities 3–10 each contribute 100 prompts. The source is
official GenEval2 commit `a6e82d2289e8d418f27f0adee77908b07060eea3`.
Override with `NUM_SHARDS`, `HEIGHT`, `WIDTH`, `SEED`, `MODEL_PATH`, or
`PROMPT_FILE`.

The multi-seed protocol runs the five-arm main matrix at seed 42, then Z0-only at
seeds 43 and 44:

```bash
bash scripts/evaluate/run_bagel_loop_t2i_full800_multiseed.sh \
  /data/outputs/bagel_loop_t2i_full800_multiseed
```

The root-level `multiseed_summary.{json,md}` reports Z0 mean, sample standard
deviation, and range. Loop deltas remain same-seed comparisons against seed-42 Z0;
the extra Z0 seeds do not make the loop arms multi-seed evaluations.

## 5. Outputs and scoring

The merge step writes:

- `index.html`: visual arm comparison;
- `run_manifest.json`: run metadata and per-arm GenEval image maps;
- `mechanism_summary.json`: aggregated `ΔM`, `ΔG`, and `Δv`;
- `geneval2_image_maps/*.json`: prompt-to-image maps for semantic scoring.

Pixel MAE versus Z0 is behavioral distance, not a quality score. Semantic quality
comes from GenEval2 Soft-TIFA, especially count, spatial relation, attribute
binding, and multi-object composition.

The launcher uses `SCORE=auto`: it scores when a Soft-TIFA server is already live,
or starts one after generation when `VLM_PATH` contains Qwen3-VL. Use `SCORE=1`
to make a missing evaluator a hard error:

```bash
SCORE=1 \
VLM_PATH=/data/bagel-LatentCoT/models/Qwen3-VL-8B-Instruct \
bash scripts/evaluate/run_bagel_loop_t2i_zeroshot.sh \
  /data/outputs/bagel_loop_t2i_phase05_full800
```

The scorer writes aggregate AM/GM, per-skill, per-atomicity, CSV, JSON, and a
Markdown summary using Z0 as the baseline.

## 6. Decision rule

Select a body/persistence configuration only if diagnostics establish
`memory → GEN → velocity` and the arm improves GenEval2 AM/GM over Z0 without
unacceptable visual degradation. Treat a loop gain smaller than ordinary Z0 seed
variation as inconclusive until that loop arm is repeated across multiple seeds.
