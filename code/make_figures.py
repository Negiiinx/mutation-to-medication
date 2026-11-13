import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, json, glob, os
from matplotlib.lines import Line2D
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
INK,INK2,MUTED='#1a1a19','#52514e','#b8b7b2'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.edgecolor':MUTED,
 'axes.linewidth':0.6,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,
 'axes.labelcolor':INK,'xtick.major.width':0.6,'ytick.major.width':0.6,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
REC={'Strongly recommend':4,'Consider':3,'Insufficient evidence':2,'Do not recommend':1}
def load(c,tag):
    if not os.path.exists(c): return None
    co=pd.read_csv(c); rows=[]
    for f in glob.glob(f'raw_responses/{tag}_*.jsonl'):
        for L in open(f):
            r=json.loads(L)
            if 'error' in r['resp']: continue
            rows.append({'uid':r['uid'],'model':r['model'],**r['resp']})
    if not rows: return None
    d=pd.DataFrame(rows).drop_duplicates(['uid','model']).merge(co,on='uid',how='inner')
    d['rec']=d.recommendation.map(REC); d['endorse']=(d.rec>=3).astype(int)
    d['strongrec']=(d.recommendation=='Strongly recommend').astype(int); return d
def cb(s,col,B=3000,seed=5):
    rng=np.random.default_rng(seed); gs=s.gene.unique(); byg={g:s[s.gene==g][col].values for g in gs}
    o=[np.concatenate([byg[g] for g in rng.choice(gs,len(gs),replace=True)]).mean() for _ in range(B)]
    return np.percentile(o,[2.5,97.5])
def clean(ax):
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    ax.grid(axis='x',color='#e8e7e3',lw=0.6,zorder=0); ax.set_axisbelow(True)

D=load('data/cohort_direction.csv','dir'); R=D[D.label=='RESISTANT']; S=D[D.label=='SENSITIVE']
MODELS=['o4-mini','gpt-4.1','gpt-4o','gpt-4o-mini','qwen2.5:7b','medgemma:4b']
LBL={'o4-mini':'o4-mini (reasoning)','gpt-4.1':'GPT-4.1','gpt-4o':'GPT-4o','gpt-4o-mini':'GPT-4o-mini',
     'qwen2.5:7b':'Qwen 2.5 7B','medgemma:4b':'MedGemma 4B'}
OPEN={'qwen2.5:7b','medgemma:4b'}

# ---------- FIGURE 1 ----------
fig,axes=plt.subplots(1,2,figsize=(7.0,3.1),gridspec_kw={'wspace':0.46})
for ax,col,title in zip(axes,['endorse','strongrec'],
      ['Any endorsement despite curated resistance','Unqualified "strongly recommend"']):
    y=np.arange(len(MODELS))[::-1]
    for i,m in enumerate(MODELS):
        g=R[R.model==m]; v=g[col].mean(); lo,hi=cb(g,col)
        ax.barh(y[i],v,height=0.62,color=BLUE,alpha=1.0,zorder=3,
                hatch='///' if m in OPEN else None,edgecolor='white',linewidth=0.8)
        ax.plot([lo,hi],[y[i],y[i]],color=INK,lw=1.2,zorder=4,solid_capstyle='butt')
        ax.text(hi+0.015,y[i],f"{v:.0%}",va='center',ha='left',fontsize=8,color=INK)
    ax.set_yticks(y); ax.set_yticklabels([LBL[m] for m in MODELS],fontsize=8)
    ax.set_xlim(0,1.0); ax.set_xticks([0,.25,.5,.75,1.0])
    ax.set_xticklabels(['0','25%','50%','75%','100%'])
    ax.set_title(title,fontsize=8.5,color=INK,pad=7,loc='left')
    clean(ax)
axes[1].legend(handles=[plt.Rectangle((0,0),1,1,fc=BLUE,ec='white'),
                        plt.Rectangle((0,0),1,1,fc=BLUE,ec='white',hatch='///')],
               labels=['API model','Open-weight'],frameon=False,fontsize=7.5,
               loc='center right',handlelength=1.4,handleheight=1.1,
               bbox_to_anchor=(1.0,0.62))
fig.savefig('paper/figures/fig1_headline.pdf',bbox_inches='tight',dpi=300)
plt.close(fig); print("fig1 done")

# ---------- FIGURE 2 ----------
J=load('data/cohort_jax_clean.csv','jax2'); JR=J[J.label=='RESISTANT']
C=R.copy()
fig,axes=plt.subplots(1,2,figsize=(7.0,2.7),gridspec_kw={'wspace':0.3})
LV=['A','B','C','D']; LVN={'A':'A\nvalidated','B':'B\nclinical','C':'C\ncase study','D':'D\npreclinical'}
for ax,data,name in zip(axes,[JR,C],['JAX-CKB (n=998)','CIViC (n=312)']):
    for m,col in zip(['gpt-4.1','gpt-4o','gpt-4o-mini'],[BLUE,ORANGE,AQUA]):
        g=data[data.model==m]
        vals=[g[g.best_level==l].endorse.mean() if (g.best_level==l).any() else np.nan for l in LV]
        ax.plot(range(4),vals,color=col,lw=2.0,marker='o',ms=5.5,mec='white',mew=1.0,zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([LVN[l] for l in LV],fontsize=7.5)
    ax.set_ylim(-0.03,1.0); ax.set_yticks([0,.25,.5,.75,1.0])
    ax.set_yticklabels(['0','25%','50%','75%','100%'])
    ax.set_title(name,fontsize=8.5,color=INK,pad=7,loc='left')
    ax.set_xlabel('Strength of the resistance evidence',fontsize=8,labelpad=4)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    ax.grid(axis='y',color='#e8e7e3',lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel('Endorsement of a therapy for an\nalteration curated as resistant',fontsize=8)
axes[1].legend(handles=[Line2D([],[],color=c,lw=2,marker='o',ms=5,mec='white') for c in (BLUE,ORANGE,AQUA)],
               labels=['GPT-4.1','GPT-4o','GPT-4o-mini'],frameon=False,fontsize=7.5,loc='upper left')
fig.savefig('paper/figures/fig2_stratified.pdf',bbox_inches='tight',dpi=300)
plt.close(fig); print("fig2 done")

# ---------- FIGURE 3 ----------
KB=[('CIViC','data/cohort_direction.csv','dir'),('JAX-CKB','data/cohort_jax_clean.csv','jax2'),
    ('CGI','data/cohort_cgi_clean.csv','cgi2'),('OncoKB','data/cohort_oncokb.csv','oncokb'),
    ('MOAlmanac','data/cohort_moa_clean.csv','moa2')]
fig,ax=plt.subplots(figsize=(5.4,2.9))
ys=[]; labels=[]
for k,(name,c,tag) in enumerate(KB):
    x=load(c,tag)
    if x is None: continue
    r=x[x.label=='RESISTANT']; y=len(KB)-1-k; ys.append(y)
    labels.append(f"{name}\nn={r.uid.nunique()}, {r.gene.nunique()} genes")
    for m,col,off in zip(['gpt-4.1','gpt-4o','gpt-4o-mini'],[BLUE,ORANGE,AQUA],[0.22,0,-0.22]):
        g=r[r.model==m]
        if not len(g): continue
        v=g.endorse.mean(); lo,hi=cb(g,'endorse')
        ax.plot([lo,hi],[y+off,y+off],color=col,lw=1.4,alpha=0.55,solid_capstyle='butt',zorder=2)
        ax.plot(v,y+off,'o',color=col,ms=6.5,mec='white',mew=1.0,zorder=3)
ax.set_yticks(ys); ax.set_yticklabels(labels,fontsize=7.5)
ax.set_xlim(0,1.0); ax.set_xticks([0,.25,.5,.75,1.0]); ax.set_xticklabels(['0','25%','50%','75%','100%'])
ax.set_xlabel('Endorsement despite curated resistance evidence',fontsize=8,labelpad=4)
ax.legend(handles=[Line2D([],[],color=c,lw=0,marker='o',ms=5.5,mec='white') for c in (BLUE,ORANGE,AQUA)],
          labels=['GPT-4.1','GPT-4o','GPT-4o-mini'],frameon=False,fontsize=7.5,
          loc='lower center',bbox_to_anchor=(0.5,1.01),ncol=3,columnspacing=1.6,handletextpad=0.3)
ax.set_ylim(-0.6,len(KB)-0.4)
clean(ax)
fig.savefig('paper/figures/fig3_crosskb.pdf',bbox_inches='tight',dpi=300)
plt.close(fig); print("fig3 done")

# ---------- FIGURE 4: mitigation ----------
import glob as _g
def _load(pat,arm):
    rows=[]
    for f in _g.glob(pat):
        for L in open(f):
            r=json.loads(L)
            if 'error' in r['resp']: continue
            rows.append({'uid':r['uid'],'model':r['model'],'arm':arm,**r['resp']})
    return rows
_co=pd.read_csv('data/cohort_direction.csv')
_d=pd.DataFrame(_load('raw_responses/dir_*.jsonl','base')+_load('raw_responses/rag_*.jsonl','rag'))
_d=_d.drop_duplicates(['uid','model','arm']).merge(_co,on='uid')
_d['endorse']=(_d.recommendation.map(REC)>=3).astype(int)
_R=_d[_d.label=='RESISTANT']; _S=_d[_d.label=='SENSITIVE']
MM=['gpt-4.1','gpt-4o','gpt-4o-mini']; LB={'gpt-4.1':'GPT-4.1','gpt-4o':'GPT-4o','gpt-4o-mini':'GPT-4o-mini'}
fig,axes=plt.subplots(1,2,figsize=(6.6,2.1),gridspec_kw={'wspace':0.30})
LV=['A','B','C','D']; LVN={'A':'A','B':'B','C':'C','D':'D'}
for ax,arm,ttl in zip(axes,['base','rag'],['Vignette only','With retrieved evidence in context']):
    for m,col in zip(MM,[BLUE,ORANGE,AQUA]):
        g=_R[(_R.model==m)&(_R.arm==arm)]
        vals=[g[g.best_level==l].endorse.mean() if (g.best_level==l).sum()>=5 else np.nan for l in LV]
        ax.plot(range(4),vals,color=col,lw=2.0,marker='o',ms=5.5,mec='white',mew=1.0,zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([LVN[l] for l in LV],fontsize=8)
    ax.set_ylim(-0.03,1.0); ax.set_yticks([0,.25,.5,.75,1.0]); ax.set_yticklabels(['0','25%','50%','75%','100%'])
    ax.set_title(ttl,fontsize=8.5,color=INK,pad=7,loc='left')
    ax.set_xlabel('Strength of the resistance evidence',fontsize=8,labelpad=3)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
    ax.grid(axis='y',color='#e8e7e3',lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel('Endorsement of a therapy for an\nalteration curated as resistant',fontsize=8)
axes[1].legend(handles=[Line2D([],[],color=c,lw=2,marker='o',ms=5,mec='white') for c in (BLUE,ORANGE,AQUA)],
               labels=[LB[m] for m in MM],frameon=False,fontsize=7.5,loc='upper left')
fig.savefig('paper/figures/fig4_mitigation.pdf',bbox_inches='tight',dpi=300)
plt.close(fig); print("fig4 done")
