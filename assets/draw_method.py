"""Draw the method: native BAGEL MoT, training objective, inference loop.

Panel A keeps the native two-expert MoT drawing; B and C tell the final
training and inference story. Implementation details (ranks, precision,
initialisation, parameter counts) stay out of the figure.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.family': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'mathtext.fontset': 'dejavusans',
})

W, H = 22, 11.8
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
            weight='bold' if bold else 'normal', linespacing=1.5, **kw)


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


def path(pts, c='ink', dash=False, lw=1.5):
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], lw=lw, color=C.get(c, c),
                ls=(0, (4, 3)) if dash else '-', zorder=4, solid_capstyle='round')
    arrow(pts[-2], pts[-1], c, dash, lw=lw)


def pill(cx, cy, label, w, face='#F1F1F1', fs=13):
    rbox(cx - w / 2, cy - .22, w, .44, edge='line', face=face, lw=1.3, rs=.2)
    txt(cx, cy, label, fs, bold=True)


def snow(x, y, fs=12):
    txt(x, y, '❄', fs, c='g', fontfamily='DejaVu Sans')


# ------------------------------------------------------------------ head
txt(.5, 11.5, 'BAGEL · Frozen GEN + Spatial UND Velocity Correction', 21, ha='left', bold=True)
txt(21.5, 11.5, 'Method', 12, c='gray', ha='right')

snow(16.35, 11.08, 13)
txt(16.6, 11.08, 'Frozen', 11, c='g', ha='left')
rbox(17.9, 10.94, .34, .28, edge='t', face='tf', lw=1.2)
txt(18.35, 11.08, 'Trainable', 11, c='t', ha='left')
arrow((19.5, 11.08), (20.1, 11.08), 'g', dash=True)
txt(20.25, 11.08, 'Frozen context', 11, c='g', ha='left')

# ------------------------------------------------------------------ panel A
rbox(.4, 6.75, 21.2, 4.05, edge='line', face='#FCFCFC', lw=1.4)
pill(2.6, 10.8, 'A   BAGEL MoT · T2I', 4.2)

# UND expert
snow(1.75, 10.2, 12)
txt(2.0, 10.2, 'UND expert', 13, c='u', bold=True, ha='left')
rbox(1.6, 9.38, 6.4, .55, edge='u', face=C['uf'])
txt(4.8, 9.655, 'Prompt → embeddings', 11.5)
rbox(1.6, 8.73, 6.4, .47, edge='u', face=C['uf'])
txt(4.8, 8.965, 'QKV', 11.5)
rbox(1.6, 7.22, 6.4, .47, edge='u', face=C['uf'])
txt(4.8, 7.455, 'FFN', 11.5)

# GEN expert
snow(10.75, 10.2, 12)
txt(11.0, 10.2, 'GEN expert', 13, c='g', bold=True, ha='left')
rbox(10.6, 9.38, 6.4, .55, edge='g', face=C['gf'])
txt(13.8, 9.655, r'vae2llm$(x_t)$ + $T(t)$', 11.5)
rbox(10.6, 8.73, 6.4, .47, edge='g', face=C['gf'])
txt(13.8, 8.965, 'QKV', 11.5)
rbox(10.6, 7.22, 6.4, .47, edge='g', face=C['gf'])
txt(13.8, 7.455, 'FFN', 11.5)

# shared multimodal attention
rbox(1.6, 7.85, 15.4, .78, edge='ink', face='white', lw=1.4)
txt(7.25, 8.38, 'Multi-modal attention', 12.5)
txt(1.85, 8.09, 'prompt KV cached', 9.5, c='u', ha='left')
arrow((5.3, 8.09), (9.35, 8.09), 'u', dash=True)
txt(9.55, 8.09, 'GEN reads', 9.5, c='g', ha='left')

for x in (4.8, 13.8):
    arrow((x, 9.38), (x, 9.2), 'ink')
    arrow((x, 8.73), (x, 8.63), 'ink')
    arrow((x, 7.85), (x, 7.69), 'ink')

txt(17.3, 8.3, '× 28', 11, c='gray', ha='left')
txt(4.8, 6.98, 'prompt KV per layer', 9.5, c='u')
txt(13.8, 6.98, 'image features per layer · $v_t$', 9.5, c='g')

# frozen states reused downstream
rbox(18.0, 7.12, 3.3, 3.05, edge='g', face=C['gb'], lw=1.4)
snow(18.2, 9.87, 11)
txt(18.45, 9.87, 'frozen, reused below', 9.8, c='g', ha='left')
txt(18.2, 9.4, r'$v_{\mathrm{conditional}},\ v_{\mathrm{guided}}$', 9.2, ha='left')
txt(18.2, 9.0, 'prompt KV (per layer)', 9.4, ha='left')
txt(18.2, 8.6, 'image KV (per layer)', 9.4, ha='left')
txt(18.2, 8.2, r'velocity $v_t$', 9.4, ha='left')
txt(18.2, 7.7, 'CFG and the corrector', 9.0, c='gray', ha='left')
txt(18.2, 7.45, 'read these states', 9.0, c='gray', ha='left')

# ------------------------------------------------------------------ panel B
rbox(.4, .4, 10.4, 5.95, edge='line', face='#FCFCFC', lw=1.4)
pill(1.75, 6.35, 'B   Training', 2.5)

bx = [0.75, 3.30, 5.85, 8.40]
bw, by, bh = 2.2, 2.95, 2.4
for x, title, c, f in [
    (bx[0], '1  Sample', 'ink', 'white'),
    (bx[1], '2  Frozen GEN + CFG', 'g', C['gf']),
    (bx[2], '3  Target', 't', C['tf']),
    (bx[3], '4  Fit', 'u', C['uf']),
]:
    rbox(x, by, bw, bh, edge=c if c != 'ink' else 'line', face=f, lw=1.5)
    txt(x + bw / 2, by + bh - .3, title, 11, c=c if c != 'ink' else 'ink', bold=True)

txt(bx[0] + bw / 2, 4.55, 'image $x_0$\ncaption\n$\\epsilon,\\ t$', 9.8)
txt(bx[0] + bw / 2, 3.35, '$x_t=(1-t)x_0+t\\epsilon$', 9.0)
txt(bx[1] + bw / 2, 4.5, 'conditional pass\nreference pass', 8.9)
txt(bx[1] + bw / 2, 3.85, '$v_{\\mathrm{conditional}}$\n$v_{\\mathrm{guided}}$', 9.2)
txt(bx[1] + bw / 2, 3.3, '$\\hat{x}=x_t-t\\,v_{\\mathrm{guided}}$', 8.6)
txt(bx[2] + bw / 2, 4.45, '$g=v_{\\mathrm{guided}}-v_{\\mathrm{conditional}}$', 8.6)
txt(bx[2] + bw / 2, 3.9, '$H_g(z)=z-\\min(\\langle z,g\\rangle/\\|g\\|^2,0)\\,g$', 8.4)
txt(bx[2] + bw / 2, 3.25, '$\\mathrm{target}=H_g(\\epsilon-x_0-v_{\\mathrm{conditional}})$', 8.2)
txt(bx[3] + bw / 2, 4.5, '$r=\\mathrm{Corrector}(\\hat{x})$', 9.8)
txt(bx[3] + bw / 2, 3.9, '$\\mathrm{loss}=\\mathrm{MSE}(r,\\ \\mathrm{target})$', 9.0)
txt(bx[3] + bw / 2, 3.3, 'gradients reach the corrector', 8.6, c='gray')

for i in range(3):
    arrow((bx[i] + bw, 4.15), (bx[i + 1], 4.15))

# the sample also feeds the target
path([(bx[0] + 1.1, by + bh), (bx[0] + 1.1, 5.75), (bx[2] + 1.1, 5.75), (bx[2] + 1.1, by + bh)], 'gray', dash=True)
txt(4.4, 5.9, '$x_0,\\ \\epsilon$', 9.5, c='gray')

txt(5.6, 2.45, 'The target keeps the guidance-preserving residual.', 9.6, c='gray')
txt(5.6, 2.1, 'Sampling applies the same projection to the prediction.', 9.6, c='gray')

snow(1.0, 1.5, 11)
txt(1.25, 1.5, 'frozen: GEN and UND backbone', 9.4, c='g', ha='left')
rbox(6.35, 1.37, .3, .24, edge='t', face=C['tf'], lw=1.1)
txt(6.78, 1.5, 'trainable: spatial corrector', 9.4, c='t', ha='left')

# ------------------------------------------------------------------ panel C
rbox(11.2, .4, 10.4, 5.95, edge='line', face='#FCFCFC', lw=1.4)
pill(12.55, 6.35, 'C   Inference', 2.5)

cx = [11.55, 14.15, 16.75, 19.35]
cw, cy, ch = 2.2, 3.55, 1.5
for x, title, c, f in [
    (cx[0], 'Frozen GEN + CFG', 'g', C['gf']),
    (cx[1], 'Proposal', 'g', 'white'),
    (cx[2], 'Corrector', 'u', C['uf']),
    (cx[3], 'Euler step', 'ink', 'white'),
]:
    rbox(x, cy, cw, ch, edge=c if c != 'ink' else 'line', face=f, lw=1.5)
    txt(x + cw / 2, cy + ch - .32, title, 10.5, c=c if c != 'ink' else 'ink', bold=True)

txt(cx[0] + cw / 2, 4.05, '$v_{\\mathrm{native}}$\nKV', 9.6)
txt(cx[1] + cw / 2, 4.1, '$\\hat{x}=x_t-t\\,v_{\\mathrm{native}}$', 8.8)
txt(cx[2] + cw / 2, 4.2, '$r\\ \\rightarrow\\ H_g(r)$', 9.4)
txt(cx[2] + cw / 2, 3.85, '$r_{\\mathrm{used}}$', 9.0, c='u')
txt(cx[3] + cw / 2, 4.05, '$x\\leftarrow x-\\Delta t\\,(v_{\\mathrm{native}}+r_{\\mathrm{used}})$', 8.2)

for i in range(3):
    arrow((cx[i] + cw, 4.3), (cx[i + 1], 4.3))

# frozen context into the corrector
rbox(16.45, 5.45, 2.8, .55, edge='g', face=C['gf'], lw=1.3)
snow(16.65, 5.725, 10)
txt(17.85, 5.725, 'prompt + image KV', 9.6, c='g')
arrow((17.85, 5.45), (17.85, 5.05), 'g', dash=True)

# decode exit
rbox(19.4, 5.45, 1.9, .55, edge='ink', face='white', lw=1.3)
txt(20.35, 5.725, 'VAE → image', 9.8)
arrow((20.45, 5.05), (20.45, 5.45), 'ink')

# sampling loop
path([(20.45, 3.55), (20.45, 2.65), (12.65, 2.65), (12.65, 3.55)])
txt(16.55, 2.78, r'repeat $t: 1 \rightarrow 0$ · 50 nodes · $\Delta t=t_i-t_{i+1}$', 9.4, c='gray')

txt(16.4, 1.95, '$v=v_{\\mathrm{native}}+r_{\\mathrm{used}}$', 10.5)
txt(16.4, 1.45, '$r_{\\mathrm{used}}=H_g(r)$', 10.5, c='u')
txt(16.4, .95, 'same projection as training', 9.2, c='gray')

# ------------------------------------------------------------------ export
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'method_und.{ext}', dpi=300, facecolor='white')
fig.savefig(OUT / 'method_und_preview.png', dpi=100, facecolor='white')
print('saved', OUT)
