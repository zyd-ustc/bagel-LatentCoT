"""Draw the current method (frozen GEN + spatial UND velocity correction).

Style follows the legacy Loop-figure diagrams: rounded panels, pill headers,
green = correction path, blue = frozen native, beige = trainable.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.family': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'mathtext.fontset': 'dejavusans',
})

W, H = 22, 13.2
fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(left=.008, right=.992, top=.992, bottom=.008)
ax.set(xlim=(0, W), ylim=(0, H)); ax.axis('off')

C = {
    'ink': '#272727', 'gray': '#767676', 'line': '#CFCECE',
    'u': '#39744C', 'uf': '#DDF3DE', 'ub': '#F4F9F5',
    'g': '#0F4D92', 'gf': '#EAF1F8', 'gb': '#F4F7FC',
    't': '#AE7147', 'tf': '#F3E4CB', 'red': '#B64342',
}


def txt(x, y, s, fs=12, c='ink', ha='center', va='center', bold=False, **kw):
    kw.setdefault('zorder', 6)
    ax.text(x, y, s, fontsize=fs, color=C.get(c, c), ha=ha, va=va,
            weight='bold' if bold else 'normal', linespacing=1.45, **kw)


def rbox(x, y, w, h, edge='line', face='white', lw=1.6, dash=False, z=2, rs=.07):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f'round,pad=0.012,rounding_size={rs}',
                                facecolor=C.get(face, face), edgecolor=C.get(edge, edge),
                                lw=lw, ls=(0, (5, 3)) if dash else '-', zorder=z))


def arrow(p, q, c='ink', dash=False, rad=0, ms=13, lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms,
                                 color=C.get(c, c), lw=lw,
                                 ls=(0, (4, 3)) if dash else '-',
                                 connectionstyle=f'arc3,rad={rad}', zorder=5))


def path(pts, c='ink', dash=False, lw=1.5, head=True):
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], lw=lw, color=C.get(c, c),
                ls=(0, (4, 3)) if dash else '-', zorder=4,
                solid_capstyle='round')
    if head:
        arrow(pts[-2], pts[-1], c, dash, lw=lw)
            # noqa: E999


def pill(cx, cy, label, w, face='#F1F1F1'):
    rbox(cx - w / 2, cy - .22, w, .44, edge='line', face=face, lw=1.3, rs=.2)
    txt(cx, cy, label, 13.5, bold=True)


def snow(x, y, fs=12):
    txt(x, y, '❄', fs, c='g', fontfamily='DejaVu Sans')


# ---------------------------------------------------------------- canvas head
txt(.5, 12.85, 'BAGEL · Frozen GEN + Spatial UND Velocity Correction', 21, ha='left', bold=True)
txt(21.5, 12.85, 'Method', 12, c='gray', ha='right')

# legend
lx = 13.6
snow(lx, 12.42, 13)
txt(lx + .28, 12.42, 'Frozen', 11, c='g', ha='left')
rbox(lx + 1.6, 12.28, .34, .28, edge='t', face='tf', lw=1.2)
txt(lx + 2.06, 12.42, 'Trainable', 11, c='t', ha='left')
rbox(lx + 3.5, 12.28, .34, .28, edge='u', face='uf', lw=1.2)
txt(lx + 3.96, 12.42, 'Correction path', 11, c='u', ha='left')
arrow((lx + 5.7, 12.42), (lx + 6.25, 12.42), 'g', dash=True)
txt(lx + 6.45, 12.42, 'Frozen KV context', 11, c='g', ha='left')

# ---------------------------------------------------------------- panel A
rbox(.4, 8.7, 21.2, 3.35, edge='line', face='#FCFCFC', lw=1.4)
pill(2.1, 12.05, 'A  Sampling step', 2.9)

# input
rbox(.7, 9.85, 1.75, 1.2, edge='ink', face='white')
txt(1.575, 10.45, r'$x_t$' + '\n' + r'$t$, prompt', 12)
arrow((2.45, 10.45), (2.8, 10.45))

# frozen GEN panel
rbox(2.8, 9.35, 4.5, 2.4, edge='g', face=C['gb'], lw=1.7)
snow(3.02, 11.5, 12)
txt(4.98, 11.5, 'Frozen GEN · MMDiT', 13.5, c='g', bold=True)
rbox(2.98, 10.93, 2.02, .45, edge='g', face='white')
txt(3.99, 11.155, r'$\epsilon_\theta(x_t,\ \mathrm{prompt})$', 10.5)
rbox(5.1, 10.93, 2.02, .45, edge='g', face='white')
txt(6.11, 11.155, r'$\epsilon_\theta(x_t,\ \varnothing)$', 10.5)
rbox(3.4, 10.12, 3.3, .53, edge='g', face=C['gf'])
txt(5.05, 10.385, 'CFG  s = 4 · global renorm', 10.5)
arrow((3.99, 10.93), (3.99, 10.69), 'g')
arrow((6.11, 10.93), (6.11, 10.69), 'g')

# proposal
arrow((6.7, 10.405), (7.85, 10.405))
rbox(7.85, 9.85, 2.55, 1.15, edge='g', face='white')
txt(9.125, 10.52, r'$\hat{x}=x_t-t\,v_{\mathrm{native}}$', 11.5)
txt(9.125, 10.12, 'clean proposal', 9.5, c='gray')
arrow((10.4, 10.405), (10.75, 10.405))

# corrector
rbox(10.75, 9.35, 3.3, 2.4, edge='u', face=C['ub'], lw=1.7)
txt(12.4, 11.5, 'Spatial UND corrector', 13, c='u', bold=True)
txt(12.4, 11.05, '28 × UND layer (base frozen) + LoRA r64', 9.8)
txt(12.4, 10.72, 'one query per image patch', 10.5)
txt(12.4, 10.39, 'attends to prompt KV and GEN KV', 10.5)
txt(12.4, 9.93, '53.4M trainable', 10, c='t', bold=True)

# guidance projection
arrow((14.05, 10.405), (14.35, 10.405))
rbox(14.35, 9.8, 2.85, 1.25, edge='t', face=C['tf'])
txt(15.775, 10.85, r'$H_g$ projection', 12, c='t', bold=True)
txt(15.775, 10.45, r'$H_g(z)=z-\min(\langle z,g\rangle/\|g\|^2,0)\,g$', 9.5)
txt(15.775, 10.04, r'$g=v_{\mathrm{guided}}-v_{\mathrm{conditional}}$', 9.5, c='gray')

# sum node and Euler
arrow((17.2, 10.405), (17.38, 10.405), 't')
ax.add_patch(Circle((17.55, 10.405), .18, facecolor='white', edgecolor=C['ink'], lw=1.5, zorder=6))
txt(17.55, 10.4, '+', 13, zorder=7)
arrow((17.73, 10.405), (18.05, 10.405))
rbox(18.05, 9.85, 3.3, 1.15, edge='ink', face='white')
txt(19.7, 10.62, 'Euler step', 12, bold=True)
txt(19.7, 10.22, r'$x\leftarrow x-\Delta t\,v_{\mathrm{native}}-\Delta t\,r_{\mathrm{used}}$', 9.8)

# v_native bus into the sum node
path([(7.6, 10.405), (7.6, 9.15), (17.55, 9.15), (17.55, 10.22)], 'g')
txt(12.2, 9.27, r'$v_{\mathrm{native}}$', 10.5, c='g')

# loop back
path([(19.7, 9.85), (19.7, 8.88), (1.55, 8.88), (1.55, 9.85)])
txt(10.6, 8.98, 'repeat for 50 nodes (shift 3) · correction at the first 25 updates', 10.5, c='gray')

# ---------------------------------------------------------------- panel B
rbox(.4, .5, 11.9, 7.8, edge='line', face='#FCFCFC', lw=1.4)
pill(2.0, 8.12, 'B  Corrector layer', 3.0)

rbox(1.1, 7.0, 3.0, .68, edge='u', face=C['uf'])
txt(2.6, 7.34, r'$\hat{x}$ patch tokens · $t$', 11.5)
arrow((2.6, 7.0), (2.6, 6.72), 'u')
rbox(1.1, 6.2, 3.0, .48, edge='u', face='white')
txt(2.6, 6.44, 'LayerNorm', 11)
arrow((2.6, 6.2), (2.6, 5.96), 'u')
rbox(1.1, 5.3, 3.0, .7, edge='t', face=C['tf'])
txt(2.6, 5.74, 'Q K V · LoRA r64', 11.5, c='t', bold=True)
txt(2.6, 5.5, 'base frozen · B = 0', 8.8, c='gray')
arrow((2.6, 5.3), (2.6, 5.03), 't')
rbox(1.1, 4.52, 3.0, .48, edge='u', face='white')
txt(2.6, 4.76, 'RoPE · native 2D position', 10.5)
arrow((4.1, 4.76), (4.75, 4.76), 'u')

# attention with frozen context
rbox(4.75, 3.5, 3.5, 1.55, edge='u', face=C['uf'], lw=1.6)
txt(6.5, 4.74, 'SDPA', 12, c='u', bold=True)
txt(6.5, 4.34, r'$[\,P_l\,;\,G_l\,;\ \mathrm{patch}\ ]$', 11)
txt(6.5, 3.9, r'$q$ from patch · $kv$ concat', 9.5, c='gray')
rbox(9.0, 5.2, 2.85, .55, edge='g', face=C['gf'])
txt(10.42, 5.475, r'$P_l$  prompt KV', 10.5, c='g')
rbox(9.0, 4.5, 2.85, .55, edge='g', face=C['gf'])
txt(10.42, 4.775, r'$G_l$  GEN image KV', 10.5, c='g')
arrow((9.0, 5.3), (8.3, 4.85), 'g', dash=True)
arrow((9.0, 4.62), (8.3, 4.55), 'g', dash=True)
snow(9.35, 5.9, 11)
txt(9.55, 5.9, 'frozen context', 9.8, c='g', ha='left')
txt(9.15, 3.95, 'captured in the same GEN pass', 9.5, c='gray', ha='left')
txt(9.15, 3.68, 'detached before the corrector', 9.5, c='gray', ha='left')

arrow((6.5, 3.5), (6.5, 3.32), 'u')
rbox(4.75, 2.62, 3.5, .6, edge='u', face='white')
txt(6.5, 2.92, 'o_proj (LoRA) + residual', 10.5)
arrow((6.5, 2.62), (6.5, 2.42), 'u')
rbox(4.75, 1.86, 3.5, .5, edge='u', face='white')
txt(6.5, 2.11, 'MLP · native', 10.5)
arrow((6.5, 1.86), (6.5, 1.68), 'u')
rbox(4.75, 1.0, 3.5, .62, edge='t', face=C['tf'])
txt(6.5, 1.31, 'norm → FP32 head', 11, c='t', bold=True)
txt(6.5, .8, 'zero-init: r = 0 at initialization', 9, c='gray')
arrow((8.25, 1.31), (8.75, 1.31), 't')
rbox(8.75, 1.0, 1.75, .62, edge='t', face='white')
txt(9.625, 1.31, r'$r_{\mathrm{raw}}$', 12, c='t')

ax.plot([1.0, .86, .86, 1.0], [1.86, 1.86, 6.68, 6.68], lw=1.0, color=C['u'], zorder=3)
txt(.68, 4.42, '× 28', 10, c='u', rotation=90)

# ---------------------------------------------------------------- panel C
rbox(12.55, .5, 9.05, 7.8, edge='line', face='#FCFCFC', lw=1.4)
pill(14.85, 8.12, 'C  Guidance-safe objective', 4.2)

# geometry inset
rbox(12.85, 4.62, 8.45, 3.05, edge='line', face='#FAFBFC', lw=1.3)
txt(13.05, 7.42, 'projection geometry', 10, c='gray', ha='left')
O = (15.05, 5.55)
arrow(O, (18.35, 5.55), 'ink', ms=14)
txt(18.5, 5.55, r'$g$', 12.5, ha='left')
arrow(O, (14.15, 6.95), 't', ms=14)
txt(13.85, 7.08, r'$r_{\mathrm{raw}}$', 11.5, c='t', ha='right')
arrow(O, (15.05, 6.95), 'u', ms=14)
txt(15.18, 6.9, r'$r_{\mathrm{used}}=H_g(r_{\mathrm{raw}})$', 11, c='u', ha='left')
ax.plot([14.15, 15.05], [6.95, 6.95], lw=1.2, color=C['red'], ls=(0, (4, 3)), zorder=4)
ax.plot([14.15, 14.15], [5.55, 6.95], lw=1.0, color=C['line'], ls=(0, (2, 3)), zorder=3)
arrow((15.05, 5.55), (14.15, 5.55), 'red', ms=11, lw=1.3)
txt(14.55, 5.3, r'removed ($\langle r,g\rangle<0$)', 9.5, c='red')

# formulas
rbox(12.85, 2.12, 8.45, 2.3, edge='line', face='white', lw=1.3)
lines = [
    (r'$g=v_{\mathrm{guided}}-v_{\mathrm{conditional}}$', 'ink'),
    (r'$H_g(z)=z-\min(\langle z,g\rangle/\|g\|^2,\ 0)\,g$', 'ink'),
    (r'$\mathrm{target}=H_g(\varepsilon-x_0-v_{\mathrm{conditional}})$', 'ink'),
    (r'$\mathrm{loss}=\mathrm{MSE}(r_{\mathrm{raw}},\ \mathrm{target})$', 'ink'),
    (r'$r_{\mathrm{used}}=H_g(r_{\mathrm{raw}})\ \ \mathrm{at\ sampling}$', 'u'),
]
yy = 4.12
for s, c in lines:
    txt(13.1, yy, s, 11, c=c if c != 'ink' else 'ink', ha='left')
    yy -= .4

# trainable / frozen summary
rbox(12.85, .72, 8.45, 1.2, edge='line', face='white', lw=1.3)
rbox(13.05, 1.42, .3, .24, edge='t', face=C['tf'], lw=1.1)
txt(13.48, 1.54, 'Trainable  53.4M · LoRA (28 layers) · input projection · FP32 head',
    10.5, ha='left')
snow(13.12, 1.0, 11)
rbox(13.36, .88, .3, .24, edge='g', face=C['gf'], lw=1.1)
txt(13.78, 1.0, 'Frozen  GEN (MMDiT) · UND base · prompt KV · GEN KV', 10.5, ha='left')

# ---------------------------------------------------------------- export
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'method_und.{ext}', dpi=300, facecolor='white')
fig.savefig(OUT / 'method_und_preview.png', dpi=100, facecolor='white')
print('saved', OUT)
