"""Explicit depth labels keep paired generation and resume identities distinct."""
DEPTH_PREFIX='LAYERWISE_MEMORY_KV_R'


def parse_depths(value):
    if value is None:return ()
    depths=tuple(int(x) for x in value.split(','))
    if not depths or any(r<1 for r in depths) or len(set(depths))!=len(depths):
        raise ValueError('loop depths must be distinct positive integers; BASE represents R0')
    return tuple(sorted(depths))


def expand_arms(arms,depths,default_rounds):
    if depths:
        if set(arms)!={'BASE','LAYERWISE_MEMORY_KV'}:
            raise ValueError('multi-depth comparison requires exactly BASE,LAYERWISE_MEMORY_KV')
        return [('BASE','BASE',0)]+[(f'{DEPTH_PREFIX}{r}','LAYERWISE_MEMORY_KV',r) for r in depths]
    return [(arm,'BASE' if arm=='BASE_MATCHED_LATENCY' else arm,
             0 if arm.startswith('BASE') else default_rounds) for arm in arms]


def depth_of(arm):
    if arm.startswith(DEPTH_PREFIX):return int(arm[len(DEPTH_PREFIX):])
    return None
