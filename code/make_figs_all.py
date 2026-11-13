import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, os
from matplotlib.lines import Line2D
from anlib import LABEL, VENDOR, ORDER
BLUE,ORANGE,AQUA,INK,INK2,MUTED='#2a78d6','#eb6834','#1baf7a','#1a1a19','#52514e','#b8b7b2'
VCOL={'OpenAI':BLUE,'Anthropic':ORANGE,'Google':AQUA,'open':'#4a3aa7'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':MUTED,
 'axes.linewidth':0.6,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,'axes.labelcolor':INK,
 'xtick.major.width':0.6,'ytick.major.width':0.6,'figure.facecolor':'white','axes.facecolor':'white',
 'savefig.facecolor':'white'})
def clean(ax,axis='x'):
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    ax.grid(axis=axis,color='#e8e7e3',lw=0.6); ax.set_axisbelow(True)
os.makedirs('paper/figures',exist_ok=True)

# --- FIG 1: headline, Delta by model, coloured by vendor ---
h=pd.read_csv('results/headline.csv')
h['ord']=h.model.map({m:i for i,m in enumerate(ORDER)}); h=h.sort_values('delta',ascending=True)
fig,ax=plt.subplots(figsize=(6.4,0.34*len(h)+1.1))
y=np.arange(len(h))
for i,(_,r) in enumerate(h.iterrows()):
    c=VCOL.get(r.vendor,MUTED)
    ax.barh(y[i],r.delta,height=0.62,color=c,zorder=3,edgecolor='white',linewidth=0.8)
    ax.plot([r.d_lo,r.d_hi],[y[i],y[i]],color=INK,lw=1.2,zorder=4,solid_capstyle='butt')
    ax.text(max(r.d_hi,r.delta)+0.012,y[i],f"{r.delta:+.0%}",va='center',ha='left',fontsize=8,color=INK)
ax.axvline(0,color=INK2,lw=0.8,zorder=2)
ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in h.model],fontsize=8)
ax.set_xlim(-0.18,0.62); ax.set_xticks([-0.1,0,0.2,0.4,0.6])
ax.set_xticklabels(['-10%','0','20%','40%','60%'])
ax.set_xlabel('Matched discrimination $\\Delta$ (sensitivity controls minus resistance cases)',fontsize=8,labelpad=4)
ax.legend(handles=[plt.Rectangle((0,0),1,1,fc=VCOL[v],ec='white') for v in ['OpenAI','Anthropic','Google','open']],
          labels=['OpenAI','Anthropic','Google','Open-weight'],frameon=False,fontsize=7.5,
          loc='lower right',ncol=2,columnspacing=1.0,handlelength=1.2)
clean(ax)
fig.savefig('paper/figures/fig1_delta.pdf',bbox_inches='tight',dpi=300); plt.close(fig); print("fig1")

# --- FIG 2: Delta by evidence level, per KB ---
bl=pd.read_csv('results/bylevel.csv')
kbs=[k for k in ['CIViC','JAX-CKB','CGI'] if k in set(bl.kb)]
fig,axes=plt.subplots(1,len(kbs),figsize=(2.35*len(kbs)+0.6,2.4),sharey=True)
if len(kbs)==1: axes=[axes]
LV=['A','B','C','D']
show=[m for m in ORDER if m in set(bl.model)][:6]
for ax,kb in zip(axes,kbs):
    for m in show:
        s=bl[(bl.kb==kb)&(bl.model==m)].set_index('level').reindex(LV)
        if s.delta.notna().sum()<2: continue
        ax.plot(range(4),s.delta.values,color=VCOL.get(VENDOR[m],MUTED),lw=1.7,marker='o',ms=4.5,
                mec='white',mew=0.8,alpha=0.9,zorder=3)
    ax.axhline(0,color=INK2,lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(LV,fontsize=8)
    ax.set_ylim(-0.35,1.05); ax.set_title(kb,fontsize=8.5,loc='left',pad=6)
    ax.set_xlabel('Resistance evidence level',fontsize=8,labelpad=3)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    ax.grid(axis='y',color='#e8e7e3',lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel('Matched discrimination $\\Delta$',fontsize=8)
axes[0].set_yticks([0,.25,.5,.75,1.0]); axes[0].set_yticklabels(['0','25%','50%','75%','100%'])
axes[-1].legend(handles=[Line2D([],[],color=VCOL[v],lw=1.8) for v in ['OpenAI','Anthropic','Google','open']],
    labels=['OpenAI','Anthropic','Google','Open'],frameon=False,fontsize=7,loc='lower left',ncol=2,columnspacing=0.8)
fig.savefig('paper/figures/fig2_bylevel.pdf',bbox_inches='tight',dpi=300); plt.close(fig); print("fig2")

# --- FIG 3: mitigation before/after ---
if os.path.exists('results/mitigation.csv'):
    mt=pd.read_csv('results/mitigation.csv')
    if len(mt):
        mt=mt.sort_values('d_b')
        fig,ax=plt.subplots(figsize=(5.6,0.42*len(mt)+1.0))
        y=np.arange(len(mt))
        for i,(_,r) in enumerate(mt.iterrows()):
            c=VCOL.get(r.vendor,MUTED)
            ax.plot([r.d_b,r.d_r],[y[i],y[i]],color=c,lw=2.2,alpha=0.45,zorder=2,solid_capstyle='round')
            ax.plot(r.d_b,y[i],'o',color='white',mec=c,mew=1.8,ms=7,zorder=3)
            ax.plot(r.d_r,y[i],'o',color=c,ms=7,mec='white',mew=1.0,zorder=4)
            ax.text(r.d_r+0.02,y[i],f"{r.d_r:+.0%}",va='center',fontsize=8,color=INK)
        ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in mt.model],fontsize=8)
        ax.set_xlim(-0.05,1.0); ax.set_xticks([0,.25,.5,.75,1.0]); ax.set_xticklabels(['0','25%','50%','75%','100%'])
        ax.set_xlabel('Matched discrimination $\\Delta$',fontsize=8,labelpad=4)
        ax.legend(handles=[Line2D([],[],color=INK2,lw=0,marker='o',mfc='white',mec=INK2,ms=6),
                           Line2D([],[],color=INK2,lw=0,marker='o',ms=6)],
                  labels=['Vignette only','With evidence in context'],frameon=False,fontsize=7.5,loc='lower right')
        clean(ax)
        fig.savefig('paper/figures/fig3_mitigation.pdf',bbox_inches='tight',dpi=300); plt.close(fig); print("fig3")
print("done")
