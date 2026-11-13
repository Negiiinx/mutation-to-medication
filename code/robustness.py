import pandas as pd, numpy as np, sys
sys.path.insert(0,'.'); from stats_core import ols_cluster
rng=np.random.default_rng(7)

# ---------- data ----------
d=pd.read_csv('data/probe_results.csv'); d=d[d.best_level!='E'].copy()
ev=pd.read_csv('data/nightly-ClinicalEvidenceSummaries.tsv',sep='\t',dtype=str)
import re
def gene_from_mp(mp):
    s=str(mp)
    if '::' in s:
        parts=[p for p in re.split(r'::',s.split(' ')[0]) if p and p.lower()!='v']
        if parts: return parts[-1]
    tok=re.sub(r'[^A-Za-z0-9\-]','',s.split(' ')[0]); return tok if len(tok)>1 else None
ev['gene']=ev.molecular_profile.map(gene_from_mp)
# CURATION DENSITY: how many CIViC evidence items exist for this gene (any type)
dens=ev.groupby('gene').size().rename('civic_n')
d=d.merge(dens,left_on='gene',right_index=True,how='left')
d['civic_n']=d.civic_n.fillna(0)

d['lvl']=d.best_lvl.astype(float); d['lg']=np.log10(d.n_gene+1)
d['lv']=np.log10(d.n_variant+1); d['lc']=np.log10(d.civic_n+1)
for c in ['lvl','lg','lv','lc']: d[c+'_c']=d[c]-d[c].mean()
print(f"CIViC curation density per gene: median {d.civic_n.median():.0f}, range {d.civic_n.min():.0f}-{d.civic_n.max():.0f}")
print(f"corr(log CIViC entries, log PubMed) = {d[['lc','lg']].corr('spearman').iloc[0,1]:+.3f}   <- how confounded they are\n")

def design(g, with_density=False):
    cols=[np.ones(len(g)), g.lvl_c, g.lg_c, g.lvl_c*g.lg_c, g.lv_c]
    names=['b0','bE','bP','bEP','bV']
    if with_density:
        cols += [g.lc_c, g.lvl_c*g.lc_c]; names += ['bC','bE_x_C']
    return np.column_stack(cols), names

# ---------- 1. curation-density control ----------
print("="*78)
print("1. DOES THE INTERACTION SURVIVE CONTROLLING FOR CIViC CURATION DENSITY?")
print("   (adds CIViC entries per gene, and its interaction with evidence)")
for m,g in d.groupby('model'):
    g=g.reset_index(drop=True)
    X0,n0=design(g,False); r0,_=ols_cluster(g.endorse,X0,g.gene,n0)
    X1,n1=design(g,True);  r1,_=ols_cluster(g.endorse,X1,g.gene,n1)
    print(f"  {m:13s} bEP base = {r0.loc['bEP','coef']:+.4f} (p={r0.loc['bEP','p']:.4f})   "
          f"-> controlled = {r1.loc['bEP','coef']:+.4f} (p={r1.loc['bEP','p']:.4f})")

# ---------- 2. wild cluster bootstrap ----------
def wild_boot(y,X,clusters,k_test,B=4999):
    """Cameron-Gelbach-Miller wild cluster bootstrap, restricted null, Rademacher."""
    def fit(yy):
        b=np.linalg.pinv(X.T@X)@X.T@yy; return b, yy-X@b
    b_hat,e_hat=fit(y)
    r,_=ols_cluster(y,X,clusters,[f'v{i}' for i in range(X.shape[1])])
    t_obs=r.iloc[k_test]['t']
    # restricted fit: drop the tested column
    Xr=np.delete(X,k_test,axis=1)
    br=np.linalg.pinv(Xr.T@Xr)@Xr.T@y; er=y-Xr@br; yr_hat=Xr@br
    cl=pd.Series(clusters).reset_index(drop=True); groups={g:idx.values for g,idx in cl.groupby(cl).groups.items()}
    cnt=0
    for b_i in range(B):
        w={g:(1.0 if rng.random()<0.5 else -1.0) for g in groups}
        ystar=yr_hat.copy()
        for g,idx in groups.items(): ystar[idx]=yr_hat[idx]+w[g]*er[idx]
        rb,_=ols_cluster(ystar,X,clusters,[f'v{i}' for i in range(X.shape[1])])
        if abs(rb.iloc[k_test]['t'])>=abs(t_obs): cnt+=1
    return t_obs,(cnt+1)/(B+1)

print("\n"+"="*78)
print("2. WILD CLUSTER BOOTSTRAP on bEP  (27 clusters; 4999 reps, Rademacher, restricted null)")
for m,g in d.groupby('model'):
    g=g.reset_index(drop=True); X,names=design(g,False)
    t,p=wild_boot(g.endorse.values.astype(float),X,g.gene.values,k_test=3,B=4999)
    print(f"  {m:13s} t = {t:+.3f}   bootstrap p = {p:.4f}")

# ---------- 3. logistic with cluster-robust SEs ----------
def logit_cluster(y,X,clusters,names,iters=60):
    b=np.zeros(X.shape[1])
    for _ in range(iters):
        eta=X@b; p=1/(1+np.exp(-eta)); W=p*(1-p)
        H=X.T@(X*W[:,None]); grad=X.T@(y-p)
        step=np.linalg.pinv(H)@grad; b=b+step
        if np.max(np.abs(step))<1e-9: break
    eta=X@b; p=1/(1+np.exp(-eta)); u=(y-p)[:,None]*X
    A=np.linalg.pinv(X.T@(X*(p*(1-p))[:,None]))
    cl=pd.Series(clusters).reset_index(drop=True); meat=np.zeros((X.shape[1],)*2)
    for g,idx in cl.groupby(cl).groups.items():
        s=u[idx.values].sum(axis=0); meat+=np.outer(s,s)
    G=cl.nunique(); c=G/(G-1)
    V=c*A@meat@A; se=np.sqrt(np.diag(V)); t=b/se
    from math import erf,sqrt
    pv=[2*(1-0.5*(1+erf(abs(x)/sqrt(2)))) for x in t]
    return pd.DataFrame({'coef':b,'se':se,'z':t,'p':pv},index=names)

print("\n"+"="*78)
print("3. LOGISTIC REGRESSION (cluster-robust SEs by gene) — same specification")
for m,g in d.groupby('model'):
    g=g.reset_index(drop=True); X,names=design(g,False)
    r=logit_cluster(g.endorse.values.astype(float),X,g.gene.values,names)
    print(f"  {m:13s} bE = {r.loc['bE','coef']:+.3f}  bP = {r.loc['bP','coef']:+.3f}  "
          f"bEP = {r.loc['bEP','coef']:+.3f} (p={r.loc['bEP','p']:.4f})")
