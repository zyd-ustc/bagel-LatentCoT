"""Within-step anchored loop. The sampler does not advance inside a round."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import nn

from .adapter import AnchoredReentryAdapter, EntryGate, written_residual
from .config import LoopConfig
from .diagnostics import relative_change, tensor_record
from .feedback import FeedbackState, FeedbackStateUpdater, intervene_feedback
from .legacy import refuse_legacy_checkpoint
from .native_path import (
    BranchContext,
    NativePathWrapper,
    pack_flow_hidden,
    resolve_decoder,
    snapshot_cache,
    velocity_from_hidden,
)
from .observer import TrainedObservationObserver, boundary_attention_entropy
from .schedule import in_loop_schedule


def _observation_record(captured, pre_kv, body_kv, post_kv, *, round_index: int, keep: bool) -> dict:
    """One reader input. ``_kept`` pins KV objects when intermediates are logged."""
    def pinned(bucket):
        return tuple(bucket[index] for index in sorted(bucket))

    layers = sorted(captured)
    pre_kept, body_kept, post_kept = pinned(pre_kv), pinned(body_kv), pinned(post_kv)
    record = {
        "round": round_index,
        "kind": "full_native" if round_index == 0 else "anchored",
        "layers": layers,
        "experts": [captured[index].expert for index in layers],
        "roles": [tuple(sorted(set(captured[index].token_roles))) for index in layers],
        "pre_ptrs": [item.keys.data_ptr() for item in pre_kept],
        "body_ptrs": [item.keys.data_ptr() for item in body_kept],
        "post_ptrs": [item.keys.data_ptr() for item in post_kept],
    }
    if keep:
        record["_kept"] = (pre_kept, body_kept, post_kept)
    return record


@dataclass
class LoopOutput:
    velocity: torch.Tensor
    native_velocity: torch.Tensor
    anchor: torch.Tensor
    hidden: torch.Tensor
    delta: torch.Tensor | None
    feedback: FeedbackState | None
    corrected: bool
    entries: list = field(default_factory=list)
    records: list = field(default_factory=list)
    # v^{(0)} is the native velocity. Later entries are the correction rounds.
    round_velocities: list = field(default_factory=list)
    round_deltas: list = field(default_factory=list)
    round_observations: list = field(default_factory=list)


class AnchoredDualStateLoop(nn.Module):
    """Native pass plus optional anchored corrections on one frozen decoder.

    ``variant='dual'`` reads GEN hidden through the UND side branch and carries ``M_r``.
    ``variant='gen_only'`` repeats the body from ``P + ΔH(H_r, P_t, t)`` with no
    memory state and no UND read.
    ``state_mode='direct_hidden'`` is the unsafe ablation, not the default.
    """

    def __init__(self, decoder, config: LoopConfig):
        super().__init__()
        decoder = resolve_decoder(decoder)
        config.validate(len(decoder.layers))
        object.__setattr__(self, "decoder", decoder)
        self.config = config
        hidden = decoder.config.hidden_size
        self.path = NativePathWrapper(decoder, config)
        observer = TrainedObservationObserver(
            decoder, config.slots, config.writer_position_start, config.seed_token_ids,
        )
        observer.source_mode = config.observation_source_mode
        if config.observation_checkpoint:
            observer.load_checkpoint(
                config.observation_checkpoint, source_mode=config.observation_source_mode,
            )
        # The reader is not an ADSL checkpoint tensor. Its trained copy is loaded
        # from the observation checkpoint and must stay out of this module's state.
        object.__setattr__(self, "observer", observer)
        self.observation_trace = []
        self.updater = FeedbackStateUpdater(hidden, config.slots, config.rank)
        self.adapter = AnchoredReentryAdapter(hidden, config.rank)
        self.gate = EntryGate()
        self.freeze_native()

    def freeze_native(self) -> None:
        self.decoder.eval()
        self.decoder.requires_grad_(False)
        if getattr(self, "observer", None) is not None:
            self.observer.freeze()

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        self.observer.freeze()
        return self

    def trainable_parameters(self):
        if self.config.variant != "gen_only":
            return [parameter for parameter in self.parameters() if parameter.requires_grad]
        # The state-free residual and the gate. Memory slots are not in this control.
        prefixes = (
            "gate.",
            "adapter.norm_h.",
            "adapter.norm_p.",
            "adapter.h_down.",
            "adapter.p_down.",
            "adapter.time_down.",
            "adapter.up.",
        )
        return [
            parameter for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith(prefixes)
        ]

    def assert_native_frozen(self) -> None:
        live = [name for name, parameter in self.decoder.named_parameters() if parameter.requires_grad]
        if live:
            raise RuntimeError("native weights are trainable: " + live[0])

    def export_state(self) -> dict:
        return {
            "experiment": "anchored_dual_state_loop",
            "config": self.config.to_dict(),
            "observation": self.observer.status(),
            "model": {key: value.detach().cpu() for key, value in self.state_dict().items()},
        }

    def load_state(self, blob: dict) -> None:
        refuse_legacy_checkpoint(blob)
        self.load_state_dict(blob["model"])

    def _scale(self, like: torch.Tensor, *, train: bool) -> torch.Tensor:
        # The full-zero path writes a constant zero. It does not read the gate.
        if self.config.force_full_zero_path:
            return like.new_zeros(())
        scale = self.gate()
        if train:
            return scale
        return scale.detach()

    def _fast_bypass(self, flow_time, step_index, num_steps, *, train: bool) -> bool:
        # Out-of-schedule steps stay native, including during training.
        # A scheduled training step enters so the gate tensor keeps its gradient.
        # Inference reads that same parameter. An exact zero may skip the
        # second body. force_native_bypass skips even when the gate is open.
        if not in_loop_schedule(
            self.config, flow_time=flow_time, step_index=step_index, num_steps=num_steps,
        ):
            return True
        if self.config.force_native_bypass:
            return True
        if train:
            return False
        if self.config.force_full_zero_path:
            return False
        return float(self.gate.raw.detach()) == 0.0

    def forward_branch(
        self,
        hidden: torch.Tensor,
        branch: BranchContext,
        *,
        flow_time: float | None = None,
        step_index: int | None = None,
        num_steps: int | None = None,
        velocity_head=None,
        train: bool | None = None,
    ) -> LoopOutput:
        """One CFG branch at one ``(x_t, t)``. Rounds reuse that same anchor.

        Round 0 reads the full native forward. Its Pre, Body, and Post captures
        exist so layers ``0 .. depth-1`` of full-query GEN KV are all present.
        A later round is Anchored Observation, which keeps the native Pre KV
        and replaces Body and Post with this round's recomputation:

            O_{r+1} = UND(KV_Pre^{native}, KV_Body^{(r)}, KV_Post^{(r)})

        Pre is not run again. On 28 layers with body ``[20, 28)``, Post is
        empty, so the read is native Pre KV plus the recomputed Body KV.

        ``round_velocities[0]`` is the native velocity. Each correction appends
        the velocity after that round's single post. The sampler uses the last
        one. ``round_deltas`` and ``round_observations`` have one entry per
        correction. A gen-only observation entry is ``None``.
        """
        if self.config.observation_source_mode != "gen_full_query":
            raise RuntimeError(
                "observation_source_mode prompt_and_gen is reserved and not wired. "
                "The reader does not concatenate prompt KV onto GEN KV."
            )
        self.observation_trace = []
        self.freeze_native()
        self.assert_native_frozen()
        if flow_time is None:
            flow_time = branch.flow_time
        if flow_time is None:
            raise ValueError("flow time is required")
        flow_time = float(flow_time)
        if train is None:
            train = torch.is_grad_enabled() and any(parameter.requires_grad for parameter in self.parameters())
        positions = branch.packed_query_position_ids.detach().clone()
        lengths = branch.query_lens.detach().clone()
        cache_snapshot = snapshot_cache(branch.past_key_values) if self.config.verify_prompt_cache else None
        embeddings = self.path.rope(hidden, branch.packed_query_position_ids)
        need_capture = not self._fast_bypass(flow_time, step_index, num_steps, train=train)
        capture_reader = need_capture and self.config.variant == "dual"
        pre_kv, body_kv, post_kv = {}, {}, {}
        anchor = self.path.pre(hidden, branch, embeddings, pre_kv if capture_reader else None)
        body = self.path.body(anchor, branch, embeddings, body_kv if capture_reader else None)
        native_hidden = self.path.post(body, branch, embeddings, post_kv if capture_reader else None)
        captured = {**pre_kv, **body_kv, **post_kv}
        native_velocity = velocity_from_hidden(native_hidden, branch.packed_vae_token_indexes, velocity_head)
        records = [tensor_record("native_velocity", native_velocity)]
        if not need_capture:
            self.path.guard_layout(branch, hidden, positions, lengths, cache_snapshot)
            return LoopOutput(
                native_velocity, native_velocity, anchor, body, None, None, False, [], records,
                round_velocities=[native_velocity], round_deltas=[], round_observations=[],
            )

        samples = int(branch.query_lens.shape[0])
        state = None
        initial = None
        if self.config.variant == "dual":
            initial = self.updater.initial(samples, anchor, branch.name, flow_time)
            state = initial
        current = body
        delta = None
        written = None
        entries = []
        round_velocities = [native_velocity]
        round_deltas = []
        round_observations = []
        scale = self._scale(anchor, train=train)
        reader_prefix = None
        boundary = self.config.body_start
        for round_index in range(self.config.correction_rounds):
            observation = None
            if self.config.variant == "dual":
                if not self.observer.loaded and not self.config.allow_unloaded_observer:
                    raise RuntimeError(
                        "the dual-state reader has no trained observation checkpoint. "
                        "A frozen copy of the native UND projections is not the verified readout."
                    )
                record = _observation_record(
                    captured, pre_kv, body_kv, post_kv, round_index=round_index,
                    keep=self.config.log_intermediates,
                )
                self.observation_trace.append(record)
                pre_ptrs = record["pre_ptrs"]
                if round_index == 0:
                    observation, reader_prefix = self.observer.read_with_boundary(
                        captured, branch, boundary,
                    )
                    reader_prefix = replace(reader_prefix, pre_ptrs=tuple(pre_ptrs))
                else:
                    # Pre KV is unchanged, so the reader state entering the body
                    # is unchanged. Resume there. The stitched Body and Post KV
                    # are still a different source from the C2 capture.
                    observation = self.observer.read_from_boundary(
                        captured, branch, reader_prefix, pre_ptrs,
                    )
                    native_entropy = boundary_attention_entropy(
                        self.observer.layers[boundary], reader_prefix.queries, self.decoder,
                        reader_prefix.positions, reader_prefix.lengths, reader_prefix.native_boundary,
                    )
                    anchored_entropy = boundary_attention_entropy(
                        self.observer.layers[boundary], reader_prefix.queries, self.decoder,
                        reader_prefix.positions, reader_prefix.lengths, captured[boundary],
                    )
                    record["boundary_layer"] = boundary
                    record["boundary_entropy"] = anchored_entropy
                    record["boundary_entropy_native"] = native_entropy
                    record["boundary_entropy_delta"] = anchored_entropy - native_entropy
                    self.observation_trace[0]["boundary_layer"] = boundary
                    self.observation_trace[0]["boundary_entropy"] = native_entropy
                state = self.updater(state, observation)
                state = intervene_feedback(
                    state, self.config.feedback_mode, initial=initial,
                )
                delta = self.adapter(current, state, anchor, flow_time, branch)
            elif self.config.variant == "gen_only":
                # H_{r+1} = B(P + ΔH(H_r, P_t, t)). No M_r and no UND read.
                delta = self.adapter.state_free_delta(current, anchor, flow_time, branch)
            else:
                raise ValueError(self.config.variant)
            if self.config.state_mode == "anchored":
                # Cap gamma * delta. A zero gate writes an exact zero. Training
                # warmup still carries a residual gradient to the adapter.
                written = written_residual(
                    delta, scale, anchor, self.config.max_delta_rms_ratio,
                    branch.packed_vae_token_indexes,
                )
                entry = anchor + written
            elif self.config.state_mode == "direct_hidden":
                entry = current
            else:
                raise ValueError(self.config.state_mode)
            entries.append(entry)
            round_deltas.append(written if written is not None else delta)
            round_observations.append(observation)
            capture_next = (
                self.config.variant == "dual" and round_index + 1 < self.config.correction_rounds
            )
            if capture_next:
                # Same neutral-seed, full-depth reader. Body and Post KV here are
                # recomputed from the anchored entry, so this source is not the
                # verified C2 observation capture. Native Pre KV stays.
                # One post both refreshes that KV and yields this round's velocity.
                body_kv, post_kv = {}, {}
                current = self.path.body(entry, branch, embeddings, body_kv)
                posted = self.path.post(current, branch, embeddings, post_kv)
                captured = {**pre_kv, **body_kv, **post_kv}
            else:
                current = self.path.body(entry, branch, embeddings)
                posted = self.path.post(current, branch, embeddings)
            velocity = velocity_from_hidden(
                posted, branch.packed_vae_token_indexes, velocity_head,
            )
            round_velocities.append(velocity)
            if self.config.log_intermediates:
                record = {
                    "round": round_index + 1,
                    "hidden": tensor_record("H", current),
                    "delta": tensor_record("delta", written if written is not None else delta),
                    "velocity_drift": relative_change(velocity, native_velocity),
                }
                if observation is not None:
                    record["observation"] = tensor_record("O", observation)
                if state is not None:
                    record["feedback"] = tensor_record("M", state.value)
                records.append(record)
        self.path.guard_layout(branch, hidden, positions, lengths, cache_snapshot)
        return LoopOutput(
            velocity, native_velocity, anchor, current, written if written is not None else delta,
            state, True, entries, records,
            round_velocities=round_velocities,
            round_deltas=round_deltas,
            round_observations=round_observations,
        )

    def predict_flow(
        self,
        model,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_vae_token_indexes: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_seqlens: torch.Tensor,
        branches: dict[str, BranchContext],
        *,
        cfg_text_scale: float = 1.0,
        cfg_img_scale: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        step_index: int | None = None,
        num_steps: int | None = None,
        train: bool | None = None,
    ) -> torch.Tensor:
        """CFG velocities for one latent state. Branches do not share feedback."""
        hidden = pack_flow_hidden(
            model, x_t, timestep, packed_text_ids, packed_text_indexes,
            packed_vae_token_indexes, packed_vae_position_ids, packed_seqlens,
        )
        flow_time = float(timestep.reshape(-1)[0])
        head = model.llm2vae
        shared = dict(flow_time=flow_time, step_index=step_index, num_steps=num_steps,
                      velocity_head=head, train=train)
        if "cond" not in branches:
            raise ValueError("the conditional branch is required")
        seen = {}
        outputs = {}
        for name in ("cond", "cfg_text", "cfg_img"):
            branch = branches.get(name)
            if branch is None:
                continue
            if branch.name != name:
                raise ValueError("branch object does not match its CFG name")
            identity = id(branch.past_key_values)
            if identity in seen:
                raise ValueError(f"CFG branch {name} reused the cache of {seen[identity]}")
            seen[identity] = name
            outputs[name] = self.forward_branch(hidden, branch, **shared)
        from .cfg import combine_cfg
        text = outputs["cfg_text"].velocity if cfg_text_scale > 1.0 else None
        image = outputs["cfg_img"].velocity if cfg_img_scale > 1.0 else None
        if cfg_text_scale > 1.0 and text is None:
            raise ValueError("text CFG is enabled without a cfg_text branch")
        if cfg_img_scale > 1.0 and image is None:
            raise ValueError("image CFG is enabled without a cfg_img branch")
        return combine_cfg(
            outputs["cond"].velocity, text, image,
            cfg_text_scale, cfg_img_scale, cfg_renorm_type, cfg_renorm_min,
        )
