"""Within-step anchored loop. The sampler does not advance inside a round."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .adapter import AnchoredReentryAdapter, EntryGate, bound_residual
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
from .observer import VerifiedUndObserver
from .schedule import in_loop_schedule


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


class AnchoredDualStateLoop(nn.Module):
    """Native pass plus optional anchored corrections on one frozen decoder.

    ``variant='dual'`` reads GEN hidden through the UND side branch.
    ``variant='gen_only'`` keeps the same adapter capacity and drops that read.
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
        self.observer = VerifiedUndObserver(
            decoder,
            config.slots,
            tuple(range(config.body_start, config.body_end)),
            config.writer_position_start,
        )
        self.updater = FeedbackStateUpdater(hidden, config.slots, config.rank)
        self.adapter = AnchoredReentryAdapter(hidden, config.rank)
        self.gate = EntryGate()
        self.freeze_native()

    def freeze_native(self) -> None:
        self.decoder.eval()
        self.decoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def assert_native_frozen(self) -> None:
        live = [name for name, parameter in self.decoder.named_parameters() if parameter.requires_grad]
        if live:
            raise RuntimeError("native weights are trainable: " + live[0])

    def export_state(self) -> dict:
        return {
            "experiment": "anchored_dual_state_loop",
            "config": self.config.to_dict(),
            "model": {key: value.detach().cpu() for key, value in self.state_dict().items()},
        }

    def load_state(self, blob: dict) -> None:
        refuse_legacy_checkpoint(blob)
        self.load_state_dict(blob["model"])

    def _scale(self, like: torch.Tensor, *, train: bool) -> torch.Tensor:
        if train:
            return self.gate()
        if self.config.force_full_zero_path or self.config.entry_scale == 0.0:
            return like.new_zeros(())
        return like.new_tensor(self.config.entry_scale)

    def _fast_bypass(self, flow_time, step_index, num_steps, *, train: bool) -> bool:
        # Out-of-schedule steps stay native, including during training.
        # The learned gate is a tensor on the scheduled path so a Python zero
        # cannot cut its gradient.
        if not in_loop_schedule(
            self.config, flow_time=flow_time, step_index=step_index, num_steps=num_steps,
        ):
            return True
        if train:
            return False
        if self.config.force_full_zero_path:
            return False
        return self.config.entry_scale == 0.0

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
        """One CFG branch at one ``(x_t, t)``. Rounds reuse that same anchor."""
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
        anchor = self.path.pre(hidden, branch, embeddings)
        need_capture = not self._fast_bypass(flow_time, step_index, num_steps, train=train)
        captured_now = need_capture and self.config.variant == "dual"
        body, captured = self.path.body(anchor, branch, embeddings, capture=captured_now)
        native_hidden = self.path.post(body, branch, embeddings)
        native_velocity = velocity_from_hidden(native_hidden, branch.packed_vae_token_indexes, velocity_head)
        records = [tensor_record("native_velocity", native_velocity)]
        if not need_capture:
            self.path.guard_layout(branch, hidden, positions, lengths, cache_snapshot)
            return LoopOutput(
                native_velocity, native_velocity, anchor, body, None, None, False, [], records,
            )

        samples = int(branch.query_lens.shape[0])
        initial = self.updater.initial(samples, anchor, branch.name, flow_time)
        state = initial
        current = body
        delta = None
        written = None
        entries = []
        scale = self._scale(anchor, train=train)
        for round_index in range(self.config.correction_rounds):
            if self.config.variant == "dual":
                observation = self.observer.read(current, captured, branch)
            elif self.config.variant == "gen_only":
                observation = self.observer.zeros(samples, anchor)
            else:
                raise ValueError(self.config.variant)
            state = self.updater(state, observation)
            state = intervene_feedback(
                state, self.config.feedback_mode, initial=initial,
            )
            delta = self.adapter(current, state, anchor, flow_time, branch)
            delta = bound_residual(
                delta, anchor, self.config.max_delta_rms_ratio, branch.packed_vae_token_indexes,
            )
            if self.config.state_mode == "anchored":
                # Keep the multiplier in the anchor dtype so a zero gate is an exact add of zero.
                # The loss sees this written residual, so a closed gate blocks adapter gradients.
                scale_cast = scale.to(dtype=anchor.dtype)
                written = scale_cast * delta.to(dtype=anchor.dtype)
                entry = anchor + written
            elif self.config.state_mode == "direct_hidden":
                entry = current
            else:
                raise ValueError(self.config.state_mode)
            entries.append(entry)
            capture_next = (
                self.config.variant == "dual" and round_index + 1 < self.config.correction_rounds
            )
            current, captured = self.path.body(entry, branch, embeddings, capture=capture_next)
            if self.config.log_intermediates:
                records.append({
                    "round": round_index + 1,
                    "hidden": tensor_record("H", current),
                    "feedback": tensor_record("M", state.value),
                    "delta": tensor_record("delta", delta),
                    "velocity_drift": None,
                })
        corrected_hidden = self.path.post(current, branch, embeddings)
        velocity = velocity_from_hidden(
            corrected_hidden, branch.packed_vae_token_indexes, velocity_head,
        )
        if self.config.log_intermediates and records:
            records[-1]["velocity_drift"] = relative_change(velocity, native_velocity)
        self.path.guard_layout(branch, hidden, positions, lengths, cache_snapshot)
        return LoopOutput(
            velocity, native_velocity, anchor, current, written if written is not None else delta,
            state, True, entries, records,
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
