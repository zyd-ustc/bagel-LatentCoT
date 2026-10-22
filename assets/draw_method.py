"""Draw the method: native BAGEL MoT, training objective, inference loop.

Panel A keeps the native two-expert MoT drawing; B and C tell the final
training and inference story. Few labels, large type, no implementation
details.
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
    'u': '#39744C', 'uf': '#DDF3DE',
    'g': '#0F4D92', 'gf': '#EAF1F8',
    't': '#AE7147', 'tf': '#F3E4CB',
}


SCALE = 1.35


def txt(x, y, s, fs=12, c='ink', ha='center', va='center', bold=False, **kw):
    kw.setdefault('zorder', 6)
    ax.text(x, y, s, fontsize=fs * SCALE, color=C.get(c, c), ha=ha, va=va,
            weight='bold' if bold else 'normal', linespacing=1.55, **kw)


def rbox(x, y, w, h, edge='line', face='white', lw=1.7, z=2, rs=.07):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f'round,pad=0.012,rounding_size={rs}',
                                facecolor=C.get(face, face), edgecolor=C.get(edge, edge),
                                lw=lw, zorder=z))


def arrow(p, q, c='ink', dash=False, rad=0, ms=19, lw=2.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms,
                                 color=C.get(c, c), lw=lw,
                                 ls=(0, (4, 3)) if dash else '-',
                                 connectionstyle=f'arc3,rad={rad}', zorder=5))


def path(pts, c='ink', dash=False, lw=1.9):
    for p, q in zip(pts[:-2], pts[1:-1]):
        ax.plot([p[0], q[0]], [p[1], q[1]], lw=lw, color=C.get(c, c),
                ls=(0, (4, 3)) if dash else '-', zorder=4, solid_capstyle='round')
    arrow(pts[-2], pts[-1], c, dash, lw=lw)


def pill(cx, cy, label, w, face='#F1F1F1', fs=14):
    rbox(cx - w / 2, cy - .23, w, .46, edge='line', face=face, lw=1.3, rs=.2)
    txt(cx, cy, label, fs, bold=True)


def snow(x, y, fs=13):
    txt(x, y, '❄', fs, c='g', fontfamily='DejaVu Sans')


# ------------------------------------------------------------------ head
txt(.5, 11.5, 'BAGEL · Frozen GEN + Spatial UND Velocity Correction', 18, ha='left', bold=True)
txt(21.5, 11.5, 'Method', 12, c='gray', ha='right')

snow(16.3, 11.06, 14)
txt(16.6, 11.06, 'Frozen', 12, c='g', ha='left')
rbox(17.95, 10.91, .36, .3, edge='t', face='tf', lw=1.2)
txt(18.45, 11.06, 'Trainable', 12, c='t', ha='left')
arrow((19.6, 11.06), (20.2, 11.06), 'gray', dash=True)
txt(20.35, 11.06, 'KV context', 12, c='gray', ha='left')

# ------------------------------------------------------------------ panel A
rbox(.4, 6.75, 21.2, 4.05, edge='line', face='#FCFCFC', lw=1.4)
pill(2.9, 10.8, 'A   BAGEL MoT · T2I', 5.0, fs=13)

snow(2.55, 10.2, 13)
txt(2.85, 10.2, 'UND expert', 14, c='u', bold=True, ha='left')
rbox(2.4, 9.4, 7.6, .6, edge='u', face=C['uf'])
txt(6.2, 9.7, 'Prompt → embeddings', 13)
rbox(2.4, 8.72, 7.6, .5, edge='u', face=C['uf'])
txt(6.2, 8.97, 'QKV', 13)
rbox(2.4, 7.1, 7.6, .5, edge='u', face=C['uf'])
txt(6.2, 7.35, 'FFN', 13)

snow(12.15, 10.2, 13)
txt(12.45, 10.2, 'GEN expert', 14, c='g', bold=True, ha='left')
rbox(12.0, 9.4, 7.6, .6, edge='g', face=C['gf'])
txt(15.8, 9.7, r'vae2llm$(x_t)$ + $T(t)$', 13)
rbox(12.0, 8.72, 7.6, .5, edge='g', face=C['gf'])
txt(15.8, 8.97, 'QKV', 13)
rbox(12.0, 7.1, 7.6, .5, edge='g', face=C['gf'])
txt(15.8, 7.35, 'FFN', 13)

rbox(2.4, 7.78, 17.2, .86, edge='ink', face='white', lw=1.5)
txt(11.0, 8.36, 'Multi-modal attention', 14.5)
arrow((4.9, 8.03), (16.3, 8.03), 'u', dash=True)

for x in (6.2, 15.8):
    arrow((x, 9.4), (x, 9.22), 'ink')
    arrow((x, 8.72), (x, 8.64), 'ink')
    arrow((x, 7.78), (x, 7.6), 'ink')

txt(19.85, 8.2, '× 28', 13, c='gray', ha='left')
txt(6.2, 6.92, 'prompt KV', 10, c='u')
txt(15.8, 6.92, 'image features · $v_t$', 10, c='g')

# ------------------------------------------------------------------ panel B
rbox(.4, .4, 10.4, 5.95, edge='line', face='#FCFCFC', lw=1.4)
pill(1.95, 6.35, 'B   Training', 2.9, fs=13)

bx = [0.75, 3.23, 5.71, 8.19]
bw, by, bh = 2.25, 3.1, 2.4
for x, title, c, f in [
    (bx[0], '1  Sample', 'ink', 'white'),
    (bx[1], '2  GEN + CFG', 'g', C['gf']),
    (bx[2], '3  Target', 'ink', 'white'),
    (bx[3], '4  Fit', 't', C['tf']),
]:
    rbox(x, by, bw, bh, edge=c if c != 'ink' else 'line', face=f, lw=1.6)
    txt(x + bw / 2, by + bh - .33, title, 12.5, c=c if c != 'ink' else 'ink', bold=True)

txt(bx[0] + bw / 2, 4.4, '$x_0,\\ \\epsilon$, caption', 11)
txt(bx[0] + bw / 2, 3.75, '$x_t=(1-t)x_0+t\\epsilon$', 10)
txt(bx[1] + bw / 2, 4.5, '$v_{\\mathrm{guided}}$', 11.5)
txt(bx[1] + bw / 2, 3.95, '$v_{\\mathrm{cond}}$', 11.5)
txt(bx[1] + bw / 2, 3.4, '$\\hat{x}$', 11.5)
txt(bx[2] + bw / 2, 4.1, '$H_g(\\epsilon-x_0-v_{\\mathrm{cond}})$', 10.5)
txt(bx[3] + bw / 2, 4.5, '$r=\\mathrm{Corrector}(\\hat{x})$', 10.8)
txt(bx[3] + bw / 2, 3.7, 'loss = MSE($r$, target)', 10)

for i in range(3):
    arrow((bx[i] + bw, 4.3), (bx[i + 1], 4.3))

path([(bx[0] + 1.1, by + bh), (bx[0] + 1.1, 5.8), (bx[2] + 1.1, 5.8), (bx[2] + 1.1, by + bh)], 'gray', dash=True)

txt(5.6, 2.5, '$g=v_{\\mathrm{guided}}-v_{\\mathrm{cond}}$   ·   $H_g(z)=z-\\min(\\langle z,g\\rangle/\\|g\\|^2,0)\\,g$', 11, c='gray')

snow(1.0, 1.55, 12)
txt(1.3, 1.55, 'frozen: GEN · UND', 10.5, c='g', ha='left')
rbox(4.3, 1.41, .32, .26, edge='t', face=C['tf'], lw=1.1)
txt(4.78, 1.55, 'trainable: corrector', 10.5, c='t', ha='left')

# ------------------------------------------------------------------ panel C
rbox(11.2, .4, 10.4, 5.95, edge='line', face='#FCFCFC', lw=1.4)
pill(12.75, 6.35, 'C   Inference', 2.9, fs=13)

cx = [11.7, 13.72, 15.74, 17.76, 19.78]
cw, cy, ch = 1.82, 3.9, 1.35
for x, title, c, f in [
    (cx[0], 'GEN + CFG', 'g', C['gf']),
    (cx[1], 'Proposal', 'g', 'white'),
    (cx[2], 'Corrector', 't', C['tf']),
    (cx[3], 'Projection', 'ink', 'white'),
    (cx[4], 'Euler', 'ink', 'white'),
]:
    rbox(x, cy, cw, ch, edge=c if c != 'ink' else 'line', face=f, lw=1.6)
    txt(x + cw / 2, cy + ch - .32, title, 11.5, c=c if c != 'ink' else 'ink', bold=True)

txt(cx[0] + cw / 2, 4.36, '$v_{\\mathrm{native}}$', 11)
txt(cx[1] + cw / 2, 4.36, '$\\hat{x}=x_t-t\\,v_{\\mathrm{native}}$', 9.2)
txt(cx[2] + cw / 2, 4.3, '$r$', 20, c='t', bold=True)
txt(cx[3] + cw / 2, 4.36, '$r_{\\mathrm{used}}=H_g(r)$', 9.8)
txt(cx[4] + cw / 2, 4.36, '$v=v_{\\mathrm{native}}+r_{\\mathrm{used}}$', 9.2)

for i in range(4):
    arrow((cx[i] + cw, 4.55), (cx[i + 1], 4.55))

# frozen context
rbox(15.35, 5.5, 3.1, .55, edge='g', face=C['gf'], lw=1.4)
snow(15.55, 5.775, 11)
txt(17.1, 5.775, 'prompt + image KV', 10.5, c='g')
arrow((16.65, 5.5), (16.65, 5.25), 'g', dash=True)

# decode
rbox(19.55, 5.5, 1.9, .55, edge='ink', face='white', lw=1.4)
txt(20.5, 5.775, 'VAE → image', 10.5)
arrow((20.65, 5.25), (20.65, 5.5), 'ink')

# loop
path([(20.69, 3.9), (20.69, 3.05), (12.61, 3.05), (12.61, 3.9)])
txt(16.65, 3.2, 'repeat $t: 1 \\rightarrow 0$ · 50 nodes', 10.5, c='gray', bbox=dict(facecolor='white', edgecolor='none', pad=2))

txt(16.5, 2.2, '$r$: velocity correction predicted by the corrector', 11.5)

# ------------------------------------------------------------------ export
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'method_und.{ext}', dpi=300, facecolor='white')
fig.savefig(OUT / 'method_und_preview.png', dpi=100, facecolor='white')
print('saved', OUT)
