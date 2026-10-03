"""BAGEL native-prior-preserving recurrent visual refinement."""
from importlib import import_module

_LAZY_EXPORTS = {
    "BagelBackbone": (".backbone", "BagelBackbone"),
    "LoopConfig": (".anchored_loop", "LoopConfig"),
    "LoopModules": (".anchored_loop", "LoopModules"),
    "direct_flow_loss": (".anchored_loop", "direct_flow_loss"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
