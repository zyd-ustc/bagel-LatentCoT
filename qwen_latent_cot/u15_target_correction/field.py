"""Deployment-exact forward with explicit biased coarse-grid gradient surrogates."""
import torch
from torch.utils.checkpoint import checkpoint
from . import KIND



def validate_config(cfg):
    if cfg["kind"] != KIND:
        raise ValueError("Target correction requires its own configuration/checkpoint")
    if cfg["cfg_norm"] not in ("none", "global", "channel"):
        raise ValueError("Supported none/global/channel CFG normalization")
    lo, hi = cfg["cfg_interval"]
    if not 0 <= lo < hi <= 1 or cfg["cfg_scale"] < 1:
        raise ValueError("Invalid native CFG interval or scale")
    if not lo <= cfg["deep_time"] < cfg["handoff_time"] <= hi:
        raise ValueError("CFG interval must cover the complete correction window")
    if not 0 < cfg["deep_time"] < cfg["shallow_time"] < cfg["read_time"] < cfg["handoff_time"] < 1:
        raise ValueError("Require deep < shallow < read < handoff in (0,1)")
    if not 0 < cfg["t_eps"] < 1-cfg["handoff_time"]:
        raise ValueError("Velocity denominator must be unclipped throughout correction")
    if cfg["lookahead_steps"] != 2 or cfg["num_steps"] < 2:
        raise ValueError("Target correction uses two lookahead intervals and a native multi-step grid")
    if cfg["rho_deep"] != 0 or cfg["rho_shallow"] != 0:
        raise ValueError("Target correction requires deterministic history anchors")
    if not 0 <= cfg["pass_threshold"] <= 1:
        raise ValueError("Invalid diagnostic threshold")


def protocol(rt, cfg):
    """Bind cached trajectories to the effective numerical process, not a label."""
    return dict(native=rt.identity, width=rt.width, height=rt.height,
                noise_scale=rt.noise_scale(), times=[float(t) for t in rt.times()],
                sampler="heun_final_euler", cfg_scale=cfg["cfg_scale"],
                cfg_norm=cfg["cfg_norm"], cfg_interval=cfg["cfg_interval"],
                t_eps=cfg["t_eps"], deep_time=cfg["deep_time"],
                shallow_time=cfg["shallow_time"], read_time=cfg["read_time"],
                handoff_time=cfg["handoff_time"], lookahead_steps=cfg["lookahead_steps"],
                rho_deep=cfg["rho_deep"], rho_shallow=cfg["rho_shallow"],
                pass_threshold=cfg["pass_threshold"], diagnosis="atomic_entailment_affirmative_v4",
                rollback="history_identity", window="linear_to_handoff",
                bridge="target_driven", prefix="image_first", kind=KIND)


def phase_grid(rt, cfg):
    # Insert exact phase boundaries; do not silently round requested times.
    values = [float(t) for t in rt.times()]
    for name in ("deep_time", "shallow_time", "read_time", "handoff_time"):
        value = cfg[name]
        values = [t for t in values if abs(t-value) > 1e-7] + [value]
    return sorted(values)


def segment_times(grid, start, end):
    return [start] + [t for t in grid if start+1e-7 < t < end-1e-7] + [end]


class NativeCFG:
    def __init__(self, rt, raw, cfg):
        self.rt, self.cfg = rt, cfg
        self.positive = rt.prefix(raw)
        # None selects the native unconditional template, not an empty user prompt.
        self.negative = rt.prefix(None) if cfg["cfg_scale"] > 1 else None
        self.calls = 0

    def conditional(self, z, t, prefix):
        self.calls += 1
        return self.rt.predict(z, t, prefix).velocity.float()

    def with_conditional(self, z, t):
        positive = self.conditional(z, t, self.positive)
        lo, hi = self.cfg["cfg_interval"]
        if self.negative is None or not lo <= float(t) <= hi:
            return positive, positive
        negative = self.conditional(z, t, self.negative)
        return self.rt.guided(positive, negative, self.cfg["cfg_scale"]).float(), positive

    def __call__(self, z, t):
        return self.with_conditional(z, t)[0]


def integrate(z, times, velocity, *, capture=None):
    """Differentiable Heun; final t=1 step is Euler (no singular endpoint call)."""
    z = z.float()
    if capture is not None:
        capture[float(times[0])] = z.detach().clone()
    for a, b in zip(times[:-1], times[1:]):
        ta, tb = z.new_tensor(a), z.new_tensor(b)
        va = velocity(z, ta)
        proposal = z + (b-a)*va
        z = proposal if b >= 1 else z + (b-a)*0.5*(va+velocity(proposal, tb))
        if capture is not None:
            capture[float(b)] = z.detach().clone()
    return z


def window(t, tau, handoff):
    # Linear terminal taper matches the O(h-t) residual of the Hermite path.
    return ((handoff-t)/(handoff-tau)).clamp(0, 1)


def training_free_velocity(native, corrected_prefix, z, t, tau, handoff, scale, reference_prefix=None):
    """CFG plus a tapered gap. Text subtracts v(c). Sketch subtracts v(sketch, c)."""
    value, raw_conditional = native.with_conditional(z, t)
    if float(t) >= handoff:
        return value
    if reference_prefix is None:
        reference = raw_conditional
    else:
        reference = native.conditional(z, t, reference_prefix)
    delta = native.conditional(z, t, corrected_prefix)-reference
    return value + scale*window(t, tau, handoff)*delta


def rollback(rt,history,noise,tau,arm,perturbation_seed,cfg):
    return history.detach().float().clone()


def total_velocity(native,corrector,z,t,tau,handoff,sketch,*,taper=True):
    base = native(z,t)
    if taper and float(t)>=handoff:
        return base
    proposal = (z+(1-t).clamp_min(corrector.config['t_eps'])*base).detach()
    residual = corrector(z,t,tau,sketch,proposal)
    return base+(window(t,tau,handoff) if taper else 1.)*residual


def coarse_solve(z,start,end,steps,velocity):
    times = torch.linspace(start,end,steps+1).tolist()
    for a,b in zip(times[:-1],times[1:]):
        def step(x,a=a,b=b):
            return x+(b-a)*velocity(x,x.new_tensor(a))
        z = checkpoint(step,z,use_reentrant=False)
    return z


def target_rollout(rt,cfg,anchor,tau,native,corrector,sketch,keep):
    h = cfg['handoff_time']; grid = phase_grid(rt,cfg)
    velocity = lambda z,t: total_velocity(native,corrector,z,t,tau,h,sketch)
    with torch.no_grad():
        fine_h = integrate(anchor,segment_times(grid,tau,h),velocity)
        fine_x = integrate(fine_h,segment_times(grid,h,1.),native)
    from .loss import weighted
    coarse_h=anchor
    penalty=anchor.new_zeros(())
    times=torch.linspace(tau,h,cfg['window_steps']+1).tolist()
    for a,b in zip(times[:-1],times[1:]):
        def step(z,a=a,b=b):
            t=z.new_tensor(a)
            base=native(z,t)
            delta=window(t,tau,h)*corrector(z,t,tau,sketch,(z+(1-t)*base).detach())
            return z+(b-a)*(base+delta),weighted(delta.square(),keep)
        coarse_h,cost=checkpoint(step,coarse_h,use_reentrant=False)
        penalty=penalty+cost/cfg['window_steps']
    # Values follow deployment; Jacobians follow coarse Euler integration.
    state = coarse_h+(fine_h-coarse_h).detach()
    coarse_x = coarse_solve(state,h,1.,cfg['tail_steps'],native)
    return coarse_x+(fine_x-coarse_x).detach(),penalty
