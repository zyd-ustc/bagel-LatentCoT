"""Explicit depth identities for one architecture and its native Base control."""
from ..bagel.internal_loop import MODE
DEPTH_MODES = (MODE,)


def parse_depths(value):
    if value is None:return ()
    depths=tuple(int(x) for x in value.split(','))
    if not depths or any(r<1 for r in depths) or len(set(depths))!=len(depths):
        raise ValueError('loop depths must be distinct positive integers; BASE represents R0')
    return tuple(sorted(depths))

def depth_of(arm):
    for mode in DEPTH_MODES:
        if arm.startswith(mode+'_R'):return int(arm[len(mode)+2:])
    return None

def mode_of(arm):
    return arm.rsplit('_R',1)[0] if depth_of(arm) is not None else arm


def expand_arms(arms, depths, default_rounds):
    if len(set(arms)) != len(arms) or 'BASE' not in arms or any(a not in ('BASE', MODE) for a in arms):
        raise ValueError('evaluation supports only Base and persistent UND Memory loop')
    rounds = depths or (default_rounds,)
    if any(r < 1 for r in rounds):
        raise ValueError('positive loop depth required; BASE represents R0')
    return [('BASE', 'BASE', 0)] + [(f'{MODE}_R{r}', MODE, r) for r in rounds if MODE in arms]
