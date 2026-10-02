"""CPU-only analysis of existing reader logs; never loads a model or changes gates."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean


ARMS = ("correct", "shuffled", "zero")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{index}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{index}: expected an object")
                yield row


def finite(value, *, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError(f"invalid finite metric: {value!r}")
    return float(value)


def paired_prompt_summary(values, *, seed=42, resamples=4000):
    """Mean within prompt first; bootstrap prompt clusters (not states/layers)."""
    if not values or resamples < 1:
        raise ValueError("paired summary requires observations and positive resamples")
    clusters = [mean(finite(value) for value in values[key]) for key in sorted(values)]
    rng = random.Random(seed)
    if len(clusters) > 1:
        samples = sorted(mean(rng.choices(clusters, k=len(clusters))) for _ in range(resamples))
        ci = [samples[int((resamples - 1) * .025)], samples[int((resamples - 1) * .975)]]
    else:
        ci = None  # One prompt cannot estimate across-prompt uncertainty.
    return dict(mean=mean(clusters), ci95=ci, num_prompts=len(clusters),
                num_states=sum(len(value) for value in values.values()),
                prompt_win_fraction=mean(value < 0 for value in clusters),
                unit="MSE delta; negative favors correct", resampling_unit="prompt",
                resamples=resamples, seed=seed)


def state_key(row):
    return row["prompt_id"], row["step_index"], finite(row["timestep"])


def step_bucket(index, total):
    # Same step-index boundaries as sample_replay_step_indices, not shifted t.
    early = max(1, min(total - 2, round(total * .4)))
    middle = max(early + 1, min(total - 1, round(total * .75)))
    return "early" if index < early else "middle" if index < middle else "late"


def validate_evaluation(row):
    if isinstance(row.get("step"), bool) or not isinstance(row.get("step"), int) or row["step"] < 0:
        raise ValueError("evaluation step must be a nonnegative integer")
    states = row.get("per_state", [])
    if not states or not row.get("per_layer"):
        raise ValueError("evaluation requires per_state and per_layer evidence")
    keys = set()
    for state in states:
        if (not isinstance(state.get("prompt_id"), str) or not state["prompt_id"]
                or isinstance(state.get("step_index"), bool)
                or not isinstance(state.get("step_index"), int) or state["step_index"] < 0):
            raise ValueError("invalid prompt/state identity")
        key = state_key(state)
        if key in keys:
            raise ValueError("duplicate prompt/timestep state")
        keys.add(key)
        for arm in ARMS:
            finite(state["errors"][arm], nonnegative=True)
    for arm in ARMS:
        observed = mean(state["errors"][arm] for state in states)
        reported = finite(row["heldout_error_mse"][arm], nonnegative=True)
        if not math.isclose(observed, reported, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(f"aggregate/{arm} does not match per-state errors")
    finite(row["native_parity_max_abs"], nonnegative=True)
    for metrics in row["per_layer"].values():
        for value in metrics.values():
            finite(value)


def analyze_reader(main_dir, heldout_path, *, seed=42, resamples=4000):
    main = Path(main_dir)
    config = json.loads((main / "resolved_config.json").read_text())
    manifest = json.loads((main / "run_manifest.json").read_text())
    # num_steps is the number of schedule grid points; native Euler has N-1 steps.
    total_steps = int(config["num_steps"]) - 1
    if total_steps < 3:
        raise ValueError("at least three rollout steps required")
    heldout = list(read_jsonl(heldout_path))
    lookup = {row["prompt_id"]: row for row in heldout}
    if not heldout or len(lookup) != len(heldout):
        raise ValueError("heldout prompt IDs must be unique and nonempty")
    if manifest.get("heldout_prompt_data_sha256") != sha256(heldout_path):
        raise ValueError("heldout SHA-256 does not match run manifest")
    evaluations = list(read_jsonl(main / "heldout_diagnostics.jsonl"))
    if not evaluations or evaluations[0].get("step") != 0:
        raise ValueError("original step-zero evaluation required")
    for row in evaluations:
        validate_evaluation(row)
    steps = [row["step"] for row in evaluations]
    if steps != sorted(set(steps)):
        raise ValueError("evaluation steps must be unique and increasing")
    baseline = {state_key(row): row for row in evaluations[0]["per_state"]}
    baseline_layers = set(evaluations[0]["per_layer"])
    checkpoints = []
    for evaluation in evaluations:
        states = evaluation["per_state"]
        if set(map(state_key, states)) != set(baseline) or set(evaluation["per_layer"]) != baseline_layers:
            raise ValueError("checkpoint evaluation state/layer grids differ")
        groups = defaultdict(list)
        prompt_rows = defaultdict(list)
        for state in states:
            if state["prompt_id"] not in lookup or state["step_index"] >= total_steps:
                raise ValueError("unknown heldout prompt or rollout step")
            if baseline[state_key(state)].get("donor_prompt_id") != state.get("donor_prompt_id"):
                raise ValueError("shuffled donor changed across checkpoints")
            category = lookup[state["prompt_id"]]["category"]
            bucket = step_bucket(state["step_index"], total_steps)
            for name in ("all", f"category:{category}", f"step_bucket:{bucket}"):
                groups[name].append(state)
            prompt_rows[state["prompt_id"]].append(state)
        comparisons = {}
        for name, rows in sorted(groups.items()):
            comparisons[name] = {}
            for control in ("initial", "shuffled", "zero"):
                differences = defaultdict(list)
                for state in rows:
                    target = (baseline[state_key(state)]["errors"]["correct"] if control == "initial"
                              else state["errors"][control])
                    differences[state["prompt_id"]].append(state["errors"]["correct"] - target)
                comparisons[name][control] = paired_prompt_summary(differences, seed=seed, resamples=resamples)
        layers = {}
        for layer, metrics in sorted(evaluation["per_layer"].items(), key=lambda pair: int(pair[0])):
            rms = finite(metrics["prompt_target_rms"], nonnegative=True)
            # This is a ratio of aggregate diagnostics, not a per-state NMSE.
            layers[layer] = {**metrics, "mse_over_mean_target_rms_squared":
                             metrics["reader_mse"] / (rms * rms) if rms else None}
        prompts = [dict(prompt_id=key, category=lookup[key]["category"], num_states=len(rows),
                        errors={arm: mean(row["errors"][arm] for row in rows) for arm in ARMS})
                   for key, rows in sorted(prompt_rows.items())]
        checkpoints.append(dict(step=evaluation["step"], errors=evaluation["heldout_error_mse"],
            native_parity_max_abs=evaluation["native_parity_max_abs"], comparisons=comparisons,
            per_prompt=prompts, per_layer=layers,
            useful_vs_zero=evaluation["heldout_error_mse"]["correct"] < evaluation["heldout_error_mse"]["zero"]))
    training = []
    for index, row in enumerate(read_jsonl(main / "metrics.jsonl")):
        if isinstance(row.get("step"), bool) or row.get("step") != index + 1:
            raise ValueError("training steps must be contiguous from one")
        training.append(dict(step=row["step"], loss=finite(row["loss_reader_mse"], nonnegative=True),
                             grad_norm=finite(row["grad_norm"], nonnegative=True)))
    if not training or evaluations[-1]["step"] > training[-1]["step"]:
        raise ValueError("missing training metrics or evaluation exceeds training")
    latest = checkpoints[-1]
    gate = json.loads((main / "warmup_gate.json").read_text())
    if gate["heldout_error_mse"] != latest["errors"]:
        raise ValueError("warmup_gate disagrees with latest heldout diagnostics")
    return dict(schema="bagel-reader-offline-v1", parent_run=Path(config.get("output_dir", main)).parent.name,
        provenance={str(path): sha256(path) for path in sorted(main.glob("*.json*"))} | {str(heldout_path): sha256(heldout_path)},
        manifest=manifest, checkpoints=checkpoints, latest_step=training[-1]["step"],
        training=dict(num_updates=len(training), last=training[-1],
                      first_50_mean_loss=mean(row["loss"] for row in training[:50]),
                      last_50_mean_loss=mean(row["loss"] for row in training[-50:]),
                      maximum_preclip_grad_norm=max(row["grad_norm"] for row in training)),
        original_gate_checks=gate["checks"], original_ready_for_opd=gate["ready_for_opd"],
        limitations=["One training seed; bootstrap covers these heldout prompts, not training-seed variation.",
                    f"Only {len(prompt_rows)} of {len(heldout)} heldout prompts evaluated; no new forward passes.",
                    "Layer diagnostics are existing aggregate means; no per-state layer CI is recoverable.",
                    "Step buckets use rollout-index boundaries 40%/75%, not shifted timestep magnitudes.",
                    "Reader reconstruction MSE and gate passage are not image semantic scores.",
                    "No multiplicity correction: bucket/CI analysis is exploratory.",
                    "Training-window loss averages involve different prompts; not a heldout convergence test."],
        verdict="partial_support" if latest["useful_vs_zero"] else "reconstruction_improves_but_zero_control_is_better")


def render_reader_report(report):
    lines = ["# Reader warm-up offline diagnosis", "",
             "CPU-only analysis of original logs. No model inference, no gate/objective change.", "",
             "| Step | Correct MSE | Shuffled MSE | Zero MSE | Correct < zero |",
             "|---|---:|---:|---:|---|"]
    for row in report["checkpoints"]:
        lines.append(f'| {row["step"]} | {row["errors"]["correct"]:.6f} | {row["errors"]["shuffled"]:.6f} | {row["errors"]["zero"]:.6f} | {row["useful_vs_zero"]} |')
    latest = report["checkpoints"][-1]
    lines += ["", "## Latest paired comparisons", "",
              "Negative delta favors correct. Average within prompt, then bootstrap prompts.", "",
              "| Group | Control | Prompts | States | ΔMSE | 95% CI | Prompt win fraction |",
              "|---|---|---:|---:|---:|---|---:|"]
    for group, controls in latest["comparisons"].items():
        for name, value in controls.items():
            ci = value["ci95"]
            interval = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "not estimable"
            lines.append(f'| {group} | {name} | {value["num_prompts"]} | {value["num_states"]} | {value["mean"]:.4f} | {interval} | {value["prompt_win_fraction"]:.3f} |')
    lines += ["", "## Layer readout diagnostics", "",
              "These are correct-arm aggregate diagnostics, not paired layer-control results.", "",
              "| Layer | MSE | Cosine | Readout RMS | Target RMS | Adapter RMS | Effective slots |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for layer, metrics in latest["per_layer"].items():
        fields = [metrics.get(key) for key in ("reader_mse", "cosine", "memory_readout_rms", "prompt_target_rms", "adapter_residual_rms", "slot_effective_count")]
        lines.append(f"| {layer} | " + " | ".join("n/a" if value is None else f"{value:.4f}" for value in fields) + " |")
    lines += ["", "## Interpretation", "",
              f'Verdict: `{report["verdict"]}`. Original operational gate: `{report["original_ready_for_opd"]}`.',
              "Zero control is diagnostic only and was not a training loss or gate criterion.",
              "No semantic memory-usefulness or image-quality claim follows from these metrics.",
              f'Training log: {report["training"]["num_updates"]} updates; latest heldout checkpoint: {latest["step"]}.',
              "", "## Limitations", ""]
    lines += [f"- {value}" for value in report["limitations"]]
    return "\n".join(lines) + "\n"
