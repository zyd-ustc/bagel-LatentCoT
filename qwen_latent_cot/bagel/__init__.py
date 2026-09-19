"""BAGEL same-timestep internal-loop training and Flow-GRPO primitives."""

from importlib import import_module


_LAZY_EXPORTS = {
    "BagelBackbone": (".backbone", "BagelBackbone"),
    "LoopLoRALinear": (".loop", "LoopLoRALinear"),
    "LoopFlowDataset": (".loop_data", "LoopFlowDataset"),
    "LoopFlowCollator": (".loop_data", "LoopFlowCollator"),
    "sde_step_with_logprob": (".flow_grpo", "sde_step_with_logprob"),
    "paired_group_advantages": (".flow_grpo", "paired_group_advantages"),
    "clipped_grpo_loss": (".flow_grpo", "clipped_grpo_loss"),
    "replay_group": (".loop_grpo", "replay_group"),
    "GenEvalRewardClient": (".rewards", "GenEvalRewardClient"),
    "FluxLatentReward": (".rewards", "FluxLatentReward"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = list(_LAZY_EXPORTS)
