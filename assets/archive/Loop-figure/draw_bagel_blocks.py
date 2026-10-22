from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
OUT=Path('/Users/zyd/Documents/LCoT-codex/bagel-LatentCoT-phase1a-opd/assets/Loop-figure')
plt.rcParams.update({'font.family':['Arial','Helvetica','DejaVu Sans'],'font.size':13,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
fig,ax=plt.subplots(figsize=(24,16));fig.subplots_adjust(left=.012,right=.99,top=.985,bottom=.015)
ax.set(xlim=(0,24),ylim=(0,16));ax.axis('off')
C={'ink':'#272727','gray':'#767676','line':'#CFCECE','u':'#39744C','uf':'#DDF3DE','g':'#0F4D92','gf':'#EAF1F8','loop':'#AE7147','lf':'#F3E4CB','red':'#B64342'}

TF=(1,1,0,0,1)
def xy(x,y):
 sx,sy,ox,oy,fs=TF
 return sx*x+ox,sy*y+oy
def t(x,y,s,fs=13,c='ink',ha='center',bold=False):
 x,y=xy(x,y)
 ax.text(x,y,s,fontsize=fs*TF[4]*1.24,color=C.get(c,c),ha=ha,va='center',weight='bold' if bold else 'normal',linespacing=1.35,zorder=6)
def b(x,y,w,h,s='',c='line',f='white',fs=13,d=False):
 xx,yy=xy(x,y);ww,hh=w*TF[0],h*TF[1]
 ax.add_patch(FancyBboxPatch((xx,yy),ww,hh,boxstyle='round,pad=0.015,rounding_size=.08',facecolor=C.get(f,f),edgecolor=C.get(c,c),lw=1.7,ls=(0,(5,3)) if d else '-',zorder=2))
 if s:t(x+w/2,y+h/2,s,fs)
def a(p,q,c='ink',d=False,rad=0):
 ax.add_patch(FancyArrowPatch(xy(*p),xy(*q),arrowstyle='-|>',mutation_scale=13,color=C.get(c,c),lw=1.7,ls=(0,(4,3)) if d else '-',connectionstyle=f'arc3,rad={rad}',zorder=3))
def path(pts,c='ink',d=False):
 if len(pts)>3:raise ValueError('An arrow may have at most one bend')
 for p,q in zip(pts[:-2],pts[1:-1]):
  px,py=xy(*p);qx,qy=xy(*q)
  ax.plot([px,qx],[py,qy],lw=1.6,color=C.get(c,c),ls=(0,(4,3)) if d else '-',zorder=3)
 a(pts[-2],pts[-1],c,d)
def block(x,y,w,label,c):
 b(x,y,w,.67,c=c,f='uf' if c=='u' else 'gf');t(x+w/2,y+.335,label,15,c,bold=True)

t(.5,15.65,'BAGEL Memory Loop',25,ha='left',bold=True)
t(23.4,15.65,'Legacy R2',14,ha='right',c='gray')

# A: compact native MoT block, inspired by the official BAGEL architecture.
TF=(10.8/23,5.55/6.2,.5-.5*10.8/23,9.55-8.95*5.55/6.2,.9)
b(.5,8.95,23.0,6.2)
t(.85,14.8,'A   BAGEL MoT · T2I',18,ha='left',bold=True)
t(6.5,14.12,'UND expert',16,'u',bold=True)
t(18.0,14.12,'GEN expert',16,'g',bold=True)
b(2.8,13.08,7.4,.72,'Prompt → embeddings','u','uf',15)
b(14.3,13.08,7.4,.72,r'vae2llm$(x_t)$ + $T(t)$','g','gf',15)
# Separate expert parameters; shared multimodal attention operation.
b(2.8,9.62,7.4,3.12,c='u',f='#F4F9F5')
b(14.3,9.62,7.4,3.12,c='g',f='#F4F7FC')
b(3.3,11.77,6.4,.7,'QKV','u','uf',17)
b(14.8,11.77,6.4,.7,'QKV','g','gf',17)
a((6.5,13.08),(6.5,12.47),'u')
a((18.0,13.08),(18.0,12.47),'g')
a((6.5,11.77),(6.5,11.36),'u')
a((18.0,11.77),(18.0,11.36),'g')
b(2.8,10.48,18.9,.87,c='ink',f='white')
t(12.25,11.12,'Multi-modal attention',16)
# Native T2I freezes prompt KV after prefill; GEN reads the cached condition.
t(6.7,10.72,r'$P_l$ · cached',12,'u')
a((10.0,10.72),(14.4,10.72),'u',True)
t(18.0,10.72,'GEN reads',12,'g')
b(3.3,9.84,6.4,.5,'FFN','u','uf',17)
b(14.8,9.84,6.4,.5,'FFN','g','gf',17)
a((6.5,10.48),(6.5,10.34),'u')
a((18.0,10.48),(18.0,10.34),'g')
t(22.6,11.19,'×28',13,'gray')
t(6.5,9.27,r'$H_{prompt,l}$ → Memory',12,'u')
t(18.0,9.27,'llm2vae → $v_t$',12,'g')

# C: top-right visibility
TF=(11.8/8.1,5.55/6.08,11.7-15.4*11.8/8.1,9.55-2.13*5.55/6.08,1.08)
# C: Magnify native blocks, make exact attention visibility graphical.
b(15.4,2.13,8.1,6.08,c='line',f='#FAFBFC')
t(15.75,7.86,'C   KV VISIBILITY',17,ha='left',bold=True)

# UND native block contains explicit attention read set.
b(15.95,6.67,3.1,.65,r'$H_l^{(r)}$','u','uf',12)
b(19.55,6.67,3.3,.65,r'$P_l$ · fixed','u','white',12)
a((17.5,6.67),(17.5,6.28),'u');a((19.55,6.995),(19.05,6.02),'u',True)
b(15.95,4.86,3.1,1.4,c='u',f='uf')
t(17.5,5.98,'UND $l$',14,'u',bold=True)
t(17.5,5.53,'Q: H\nKV: P + G + H',12)

b(19.55,5.85,3.3,.57,r'$G_l^{(r)}$ KV','g','gf',12)
a((19.55,6.135),(19.05,5.55),'g',True)
a((17.5,4.86),(17.5,4.43),'u')
b(15.95,3.63,3.1,.77,r'$H_l^{(r+1)}$','u','uf',12)
a((19.05,4.015),(19.55,4.015),'u')
b(19.55,3.63,3.3,.77,'UND K/V → $M_l^{(r+1)}$','u','white',12)
# M hidden returns same layer next writer; arrow outside box.
a((15.95,4.015),(15.95,6.995),'u',rad=-.18)
# Next round GEN block: exact read scope, prompt excluded.
b(19.55,2.48,3.3,.77,'GEN $l$, $r+1$\nKV: M + G','g','gf',12)
a((21.2,3.63),(21.2,3.25),'u',True)
t(17.5,2.84,'$P_l$ excluded',11,'red')


# B: full-width lower loop; native experts decomposed as in the official diagram.
TF=(23/14.5,8.4/7.55,.5-.5*23/14.5,.7-2.13*8.4/7.55,1.13)
b(.5,2.13,14.5,7.55,c='loop',f='#FFFCF8')
t(.85,9.32,'B   NATIVE BLOCK REUSE · R2',17,ha='left',bold=True)
t(.85,8.92,r'Same GEN entrance · fixed $x_t,t$',12,'gray',ha='left')

def expert(x,w,name,c):
    # Distinct attention calls: KV arrows link the experts, not a simultaneous joint update.
    b(x,4.25,w,2.55,c=c,f='#F4F9F5' if c=='u' else '#F4F7FC')
    t(x+w/2,6.65,name,13,c,bold=True)
    for yy,hh,label in [(6.0,.43,'QKV'),(5.15,.57,'Attention'),(4.42,.43,'FFN')]:
        b(x+.10,yy,w-.20,hh,label,c,'uf' if c=='u' else 'gf',13 if w>2 else 11)
    a((x+w/2,6.0),(x+w/2,5.72),c)
    a((x+w/2,5.15),(x+w/2,4.85),c)

for x,r in zip([1.0,6.0,11.0],[0,1,2]):
    t(x+1.4,8.43,f'Round {r}',14,'g',bold=True)
    b(x,7.37,2.8,.7,('Prompt KV $P_l$' if r==0 else rf'Memory KV $M_l^{{({r})}}$'),c='u',f='uf',fs=12)
    path([(x,7.37),(x-.20,5.435),(x+.10,5.435)],'u',True)
    t(x+1.4,7.05,r'$G_{entrance}$',11,'g')
    a((x+1.4,6.92),(x+1.4,6.80),'g')
    expert(x,2.8,'GEN 0–7','g')

for x,r in [(4.0,1),(9.0,2)]:
    t(x+.725,8.43,f'Writer {r}',12,'u',bold=True)
    b(x,7.37,1.45,.7,rf'$H_l^{{({r-1})}}$',c='u',f='uf',fs=13)
    a((x+.725,7.37),(x+.725,6.80),'u')
    expert(x,1.45,'UND 0–7','u')
    # Feedback is captured from GEN's layer input QKV and read by UND attention.
    prev=x-3.0; nxt=x+2.0
    a((prev+2.70,6.215),(x+.10,5.435),'g',True)
    b(x,3.55,1.45,.43,rf'$H_l^{{({r})}}$',c='u',f='uf',fs=12)
    a((x+.725,4.42),(x+.725,3.98),'u')
    # Same-layer native UND projection supplies the next round's Memory KV.

    path([(x+1.45,3.765),(x+1.72,7.72),(nxt,7.72)],'u',True)

# Only the last writer continues through the suffix, without GEN feedback.
b(8.68,2.48,2.0,.72,'UND 8–27\nP + self','u','uf',11)
a((9.725,3.55),(9.725,3.20),'u')
b(11.0,2.48,2.8,.72,'GEN 8–27 → $v_t$','g','gf',12)
a((12.4,4.42),(12.4,3.20),'g')
a((10.68,2.84),(11.0,2.84),'u',True)

# Native full-prompt layer-input hidden initializes each independent Memory state.
b(1.0,2.90,2.85,.62,'Prompt prefill\nLayer-input hidden','u','white',12)
a((3.85,3.21),(4.35,3.21),'u')
b(4.35,2.90,2.6,.62,r'$H_l^{(0)}=H_{prompt,l}$','u','uf',13)
t(4.0,2.60,'Full prompt · one state per layer',11,'u')
t(4.0,2.32,'Reinitialize each denoising step',10,'gray')

TF=(1,1,0,0,1)
t(.5,.25,'Solid: hidden     Dashed: KV     H: Memory hidden     M: Memory KV     G: GEN hidden     P: prompt KV',10,'gray',ha='left')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'bagel_legacy_loop_blocks.{ext}',dpi=300,facecolor='white')
fig.savefig(OUT/'bagel_legacy_loop_blocks_preview.png',dpi=85,facecolor='white')
