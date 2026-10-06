"""Explicit depth labels keep paired generation and resume identities distinct."""
DEPTH_PREFIX='LAYERWISE_MEMORY_KV_R'
DEPTH_MODES=('LAYERWISE_MEMORY_KV','LAYERWISE_MEMORY_REPLACE','LAYERWISE_SEED_REPLACE')


def parse_depths(value):
    if value is None:return ()
    depths=tuple(int(x) for x in value.split(','))
    if not depths or any(r<1 for r in depths) or len(set(depths))!=len(depths):
        raise ValueError('loop depths must be distinct positive integers; BASE represents R0')
    return tuple(sorted(depths))


def expand_arms(arms,depths,default_rounds):
    if depths:
        if 'BASE' not in arms or len(arms)<2 or any(a not in ('BASE',*DEPTH_MODES) for a in arms):
            raise ValueError('multi-depth comparison requires BASE and supported layerwise modes')
        return [('BASE','BASE',0)]+[(f'{mode}_R{r}',mode,r) for mode in arms if mode!='BASE' for r in depths]
    return [(arm,'BASE' if arm=='BASE_MATCHED_LATENCY' else arm,
             0 if arm.startswith('BASE') else default_rounds) for arm in arms]


def depth_of(arm):
    for mode in DEPTH_MODES:
        if arm.startswith(mode+'_R'):return int(arm[len(mode)+2:])
    return None


def mode_of(arm):
    return arm.rsplit('_R',1)[0] if depth_of(arm) is not None else arm
