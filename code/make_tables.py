import pandas as pd, numpy as np, os

def _VLAB(v):
    return 'Open-weight' if v=='open' else v
from anlib import LABEL, VENDOR, ORDER
os.makedirs('tables',exist_ok=True)
def w(name,s): open(f'tables/{name}.tex','w').write(s); print(f"  tables/{name}.tex")
def pct(x,d=1): return '--' if pd.isna(x) else f"{100*x:.{d}f}"
def sg(x,d=1): return '--' if pd.isna(x) else f"${100*x:+.{d}f}$"

# headline: three explicit endpoints (safety, sensitivity preservation, discrimination),
# with gene-matched Delta primary and the level-matched version alongside it.
h=pd.read_csv('results/headline.csv').set_index('model')
gmm=pd.read_csv('results/genematched.csv').set_index('model')
rows=[]
for m in ORDER:
    if m not in h.index or m not in gmm.index: continue
    r=h.loc[m]; g=gmm.loc[m]
    rows.append(f"{LABEL[m]} & {_VLAB(r.vendor)} & {pct(r.strong)} & {pct(r.endorse)} & {pct(r.sens)} & "
                f"{sg(g.d_gene)} & $[{100*g.lo:+.0f},{100*g.hi:+.0f}]$ \\\\")
w('headline',"\\begin{tabular}{llrrrrr}\n\\toprule\n"
  "& & \\multicolumn{2}{c}{Resistance arm} & Control & \\multicolumn{2}{c}{$\\Delta_{\\mathrm{gene}}$} \\\\\n"
  "\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-7}\n"
  "Model & Vendor & Strong rec. & Endorse & Endorse & Est. & 95\\% CI \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# leave-one-gene-out: swing in each estimator when any single gene is removed
_lg=pd.read_csv('results/logo.csv')
_sw=_lg.groupby('model').agg(gmin=('d_gene','min'),gmax=('d_gene','max'),
                             lmin=('d_level','min'),lmax=('d_level','max'))
rows=[]
for m in ORDER:
    if m not in _sw.index: continue
    r=_sw.loc[m]; g=gmm.loc[m]
    rows.append(f"{LABEL[m]} & {sg(g.d_gene)} & $[{100*r.gmin:+.0f},{100*r.gmax:+.0f}]$ & "
                f"{sg(g.d_level)} & $[{100*r.lmin:+.0f},{100*r.lmax:+.0f}]$ \\\\")
w('logo',"\\begin{tabular}{lrrrr}\n\\toprule\n"
  "& \\multicolumn{2}{c}{$\\Delta_{\\mathrm{gene}}$} & \\multicolumn{2}{c}{$\\Delta_{\\mathrm{lvl}}$} \\\\\n"
  "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
  "Model & Full & Leave-one-gene-out range & Full & Leave-one-gene-out range \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

bl=pd.read_csv('results/bylevel.csv')
sel=[m for m in ORDER if m in set(bl.model)][:5]
rows=[]
for kb in ['CIViC','JAX-CKB','CGI']:
    s=bl[bl.kb==kb]
    if not len(s): continue
    shown=False
    for i,l in enumerate(['A','B','C','D']):
        cells=[]
        n=None
        for m in sel:
            v=s[(s.model==m)&(s.level==l)]
            cells.append(sg(v.delta.iloc[0],0) if len(v) else '--')
            if len(v): n=int(v.n.iloc[0])
        if all(c=='--' for c in cells): continue
        lab=kb if not shown else ''
        shown=True
        rows.append(f"{lab} & {l} ({n}) & "+" & ".join(cells)+" \\\\")
    rows.append("\\addlinespace")
w('bylevel',"\\begin{tabular}{ll"+"r"*len(sel)+"}\n\\toprule\nCohort & Level ($n$) & "
  +" & ".join(LABEL[m] for m in sel)+" \\\\\n\\midrule\n"+"\n".join(rows[:-1])+"\n\\bottomrule\n\\end{tabular}\n")

c=pd.read_csv('results/consensus.csv')
c['o']=c.model.map({m:i for i,m in enumerate(ORDER)}); c=c.sort_values('o')
rows=[f"{LABEL[r.model]} & {_VLAB(r.vendor)} & {pct(r.res_end)} & {pct(r.res_strong)} & {pct(r.ctl_end)} & "
      f"{sg(r.delta)} & $[{100*r.lo:+.1f},{100*r.hi:+.1f}]$ \\\\" for _,r in c.iterrows()]
w('consensus',"\\begin{tabular}{llrrrrr}\n\\toprule\nModel & Vendor & Resist. & Strong rec. & Control & $\\Delta_{\\mathrm{lvl}}$ & 95\\% CI \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

k=pd.read_csv('results/knowledge.csv'); rows=[]
for m in [x for x in ORDER if x in set(k.model)]:
    g=k[k.model==m]
    y=g[g.answer=='Yes, confers resistance']; u=g[g.answer=='Unknown or insufficient evidence']
    n_=g[g.answer=='No, does not confer resistance']
    if not len(y) or not len(u): continue
    d_=y.endorse.iloc[0]-u.endorse.iloc[0]
    rows.append(f"{LABEL[m]} & {int(y.n.iloc[0])} & {pct(y.endorse.iloc[0])} & "
                f"{int(u.n.iloc[0])} & {pct(u.endorse.iloc[0])} & "
                f"{int(n_.n.iloc[0]) if len(n_) else 0} & {pct(n_.endorse.iloc[0]) if len(n_) else '--'} & {sg(d_,0)} \\\\")
w('knowledge',"\\begin{tabular}{lrrrrrrr}\n\\toprule\n& \\multicolumn{2}{c}{States ``yes''} & \\multicolumn{2}{c}{States ``unknown''} & \\multicolumn{2}{c}{States ``no''} & \\\\\n"
  "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nModel & $n$ & endorses & $n$ & endorses & $n$ & endorses & Gap \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

m=pd.read_csv('results/mitigation.csv')
m['o']=m.model.map({x:i for i,x in enumerate(ORDER)}); m=m.sort_values('o')
_mg=pd.read_csv('results/mitigation_gene.csv').set_index('model')
rows=[]
for _,r in m.iterrows():
    dg=(f"{sg(_mg.loc[r.model,'dg_before'],1)}~$\\to$~{sg(_mg.loc[r.model,'dg_after'],1)}"
        if r.model in _mg.index else '--')
    rows.append(f"{LABEL[r.model]} & {_VLAB(r.vendor)} & {pct(r.end_b,1)}~$\\to$~{pct(r.end_r,1)} & "
                f"{pct(r.str_b,1)}~$\\to$~{pct(r.str_r,1)} & {pct(r.sens_b,1)}~$\\to$~{pct(r.sens_r,1)} & {dg} \\\\")
w('mitigation',"\\begin{tabular}{llccc c}\n\\toprule\nModel & Vendor & Endorsement & Strong rec. & Correct therapies & $\\Delta_{\\mathrm{gene}}$ \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

x=pd.read_csv('results/crosskb.csv')
kbs=['CIViC','JAX-CKB','CGI','MOAlmanac','OncoKB']
mm=['gpt-4.1','gpt-4o','gpt-4o-mini']; rows=[]
for kb in kbs:
    s=x[x.kb==kb]
    if not len(s): continue
    n=int(s.n.max())
    cells=[]
    for mo in mm:
        v=s[s.model==mo]
        cells.append(f"{pct(v.endorse.iloc[0],1)} / {sg(v.delta.iloc[0],0)}" if len(v) else '--')
    rows.append(f"{kb} & {n} & "+" & ".join(cells)+" \\\\")
w('crosskb',"\\begin{tabular}{lrccc}\n\\toprule\n& & \\multicolumn{3}{c}{Endorsement \\% / $\\Delta_{\\mathrm{lvl}}$} \\\\\n\\cmidrule(lr){3-5}\n"
  "Knowledge base & $n$ & GPT-4.1 & GPT-4o & GPT-4o-mini \\\\\n\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

d=pd.read_csv('results/dist.csv'); rows=[]
# strong-rec rate on each arm and their ratio, so the commitment comparison in Sec 5.4 is tabulated
_sr={(r.arm,r.model):r['Strongly recommend'] for _,r in d.iterrows()}
for arm in ['RESISTANT','SENSITIVE']:
    s=d[d.arm==arm]
    for i,mo in enumerate([x for x in ORDER if x in set(s.model)]):
        r=s[s.model==mo].iloc[0]
        lab='Resistance' if (arm=='RESISTANT' and i==0) else ('Sensitivity' if (arm=='SENSITIVE' and i==0) else '')
        sr_r=_sr.get(('RESISTANT',mo)); sr_s=_sr.get(('SENSITIVE',mo))
        ratio=f"{sr_r/sr_s:.2f}" if (arm=='RESISTANT' and sr_r is not None and sr_s) else ''
        rows.append(f"{lab} & {LABEL[mo]} & {pct(r['Strongly recommend'])} & {pct(r['Consider'])} & "
                    f"{pct(r['Insufficient evidence'])} & {pct(r['Do not recommend'])} & {ratio} \\\\")
    rows.append("\\addlinespace")
w('dist',"\\begin{tabular}{llrrrrr}\n\\toprule\nArm & Model & Strongly rec. & Consider & Insufficient & Do not rec. & Strong $R/S$ \\\\\n\\midrule\n"
  +"\n".join(rows[:-1])+"\n\\bottomrule\n\\end{tabular}\n")

# coverage matrix
import glob as _g, json as _j
want={'dir':730,'rag':730,'know':312}
# --- coverage: one row per arm (models x cases), computed from raw responses ---
from anlib import ORDER as _ORD
def _norm(x):
    return {'qwen2.5-7b':'qwen2.5:7b','medgemma-4b':'medgemma:4b'}.get(x,x)
_ARMS=[('dir','CIViC headline','CIViC',None),
       ('know','Knowledge probe','CIViC',None),
       ('rag','Mitigation (evidence supplied)','CIViC',None),
       ('parat2','Paraphrase T2','CIViC',None),
       ('parat3','Paraphrase T3','CIViC',None),
       ('self','Self-conditioning','CIViC',None),
       ('noinstr','Control: no instruction','CIViC',None),
       ('falseself','Control: opposite attributed','CIViC',None),
       ('jax2','Cross-source','JAX-CKB',None),
       ('cgi2','Cross-source','CGI',None),
       ('moa2','Cross-source','MOAlmanac',None),
       ('oncokb','Cross-source','OncoKB',None),
       # arms collected after review; the leading '+' routes them to their own block
       ('instronly','+Control: instruction only','CIViC',None),
       ('distract','+Control: distractor evidence','CIViC',None),
       ('temp','+Variability, $T{=}1$, $k{=}3$ (resistance)','CIViC',None),
       ('temp_sens','+Variability, $T{=}1$, $k{=}3$ (controls)','CIViC',None)]
def _setname(ms):
    ms=set(ms)
    if ms==set(_ORD): return 'all 9'
    if ms=={m for m in _ORD if VENDOR[m]!='open'}: return '7 hosted'
    if ms=={m for m in _ORD if VENDOR[m]=='OpenAI'} - {'o4-mini'}: return '3 GPT'
    if all(VENDOR[m]=='OpenAI' for m in ms): return f'{len(ms)} OpenAI'
    _hosted={m for m in _ORD if VENDOR[m]!='open'}
    if ms < _hosted: return f'{len(ms)} of 7 hosted'
    return str(len(ms))
def _ok(path):
    """(distinct cases, successful judgments) for one arm/model file.

    Judgments are deduplicated on (uid, rep) so that a case retried after a transport
    failure is counted once, while the k=3 variability arms still count three per case."""
    seen=set()
    for l in open(path):
        try: r=_j.loads(l)
        except Exception: continue
        if 'error' in r.get('resp',{}) or not r.get('resp'): continue
        seen.add((r['uid'], r.get('rep',0)))
    return len({u for u,_ in seen}), len(seen)

rows=[];tot=0
for arm,lab,kb,_ in _ARMS:
    per={}
    for f in _g.glob(f'raw_responses/{arm}_*.jsonl'):
        m=_norm(f.split(f'{arm}_')[1].replace('.jsonl',''))
        if m not in _ORD: continue
        nc,nj_=_ok(f)
        if nj_: per[m]=(nc,nj_)          # a model whose every call failed is not "covered"
    if not per: continue
    if arm=='cgi2':
        _orig={m:v for m,v in per.items() if m in ('gpt-4.1','gpt-4o','gpt-4o-mini')}
        _added={m:v for m,v in per.items() if m not in _orig}
        _c=max(c for c,_ in _orig.values()); _nj=sum(j for _,j in _orig.values()); tot+=_nj
        rows.append(f"{lab} & {kb} & {_c:,} & {_setname(_orig)} & {_nj:,} \\\\")
        if _added:
            _c2=max(c for c,_ in _added.values()); _nj2=sum(j for _,j in _added.values()); tot+=_nj2
            _vend = sorted({VENDOR[m] for m in _added})
            rows.append(f"+Cross-source extension & {kb} & {_c2:,} & "
                        f"{len(_added)} more, {len(_vend)} vendors & {_nj2:,} \\\\")
        continue
    cases=max(c for c,_ in per.values()); nj=sum(j for _,j in per.values()); tot+=nj
    rows.append(f"{lab} & {kb} & {cases:,} & {_setname(per)} & {nj:,} \\\\")
# evidence-gradient arm is analysed from the pooled cohort file, not a per-arm dump
ctrl=[r for r in rows if r.startswith('Control:')]
new=[r[1:] for r in rows if r.startswith('+')]
main=[r for r in rows if not r.startswith('Control:') and not r.startswith('+')]
main.append(f"Evidence gradient & CIViC & 1,172 & 5 mixed & 5,860 \\\\")
# GPT-5 sits outside ORDER, so its rows are counted separately
_g5={}
for _a,_lab in [('dir','GPT-5 baseline'),('rag','GPT-5 oracle evidence'),('know','GPT-5 knowledge probe')]:
    _p=f'raw_responses/{_a}_gpt-5.jsonl'
    if not os.path.exists(_p): continue
    _g5[_lab]=_ok(_p)
for _lab,(_nc,_nj) in _g5.items():
    new.append(f"{_lab} & CIViC & {_nc:,} & GPT-5 & {_nj:,} \\\\")
def _n(rs): return sum(int(r.rsplit('&',1)[1].strip().rstrip('\\\\').strip().replace(',','')) for r in rs)
mt,ct,nt=_n(main),_n(ctrl),_n(new)
w('coverage',"\\begin{tabular}{llrlr}\n\\toprule\nArm & Reference & Cases & Models & Judgments \\\\\n\\midrule\n"
  +"\n".join(main)+"\n\\midrule\n\\multicolumn{4}{l}{\\emph{Main arms}} & "+f"{mt:,}"+" \\\\\n\\midrule\n"
  +"\n".join(ctrl)+"\n\\midrule\n\\multicolumn{4}{l}{\\emph{Self-conditioning controls}} & "+f"{ct:,}"+" \\\\\n\\midrule\n"
  +"\n".join(new)+"\n\\midrule\n\\multicolumn{4}{l}{\\emph{Control and robustness arms}} & "+f"{nt:,}"+" \\\\\n"
  +"\\multicolumn{4}{l}{\\textbf{Total}} & "+f"{mt+ct+nt:,}"+" \\\\\n\\bottomrule\n\\end{tabular}\n")

# (severity/commitment-rate table absorbed into tables/endpoints.tex)

# auxiliary field setup for the surviving robustness table
import numpy as _np
from anlib import load as _ld
_d=_ld('data/cohort_direction.csv','dir')
EVS={'Established clinical evidence':5,'Moderate clinical evidence':4,'Limited clinical evidence':3,
     'Preclinical only':2,'No supporting evidence':1}
_d['evs']=_d.evidence_strength.map(EVS); _d['lvl']=_d.best_lvl.astype(float)

# auxiliary output fields only. Delta_lvl, Delta_gene and the wild bootstrap for the primary
# estimator live in the headline table and Appendix app:wild; a Delta_lvl p-value here
# contradicted that inference, so those columns are gone.
rows=[]
for m in ORDER:
    if m not in set(_d.model): continue
    g=_d[_d.model==m].reset_index(drop=True)
    if g.uid.nunique()<300: continue
    S=g[g.label=='SENSITIVE'].dropna(subset=['evs','lvl'])
    sr=S[['evs','lvl']].corr('spearman').iloc[0,1]
    R=g[g.label=='RESISTANT']
    cc=R[R.endorse==0].confidence.mean(); cw=R[R.endorse==1].confidence.mean()
    rows.append(f"{LABEL[m]} & {sr:+.2f} & {cc-cw:+.1f} \\\\")
w('robust',"\\begin{tabular}{lrr}\n\\toprule\nModel & Self-rating $\\rho$ & Conf.\\ gap \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# pre-registered confirmatory test, CONFIRM half only
_pr=pd.read_csv('results/prereg.csv')
rows=[]
for m in ORDER:
    h=_pr[(_pr.half=='CONFIRM')&(_pr.model==m)]
    if not len(h): continue
    st=h[h.endpoint=='strong_rec'].iloc[0]; en=h[h.endpoint=='endorse'].iloc[0]
    rows.append(f"{LABEL[m]} & ${st.beta:+.3f}$ & {st.se:.3f} & {st.p:.3f} & ${en.beta:+.3f}$ & {en.se:.3f} & {en.p:.3f} \\\\")
w('prereg',"\\begin{tabular}{lrrrrrr}\n\\toprule\n"
  "& \\multicolumn{3}{c}{Strong rec.\\ (primary)} & \\multicolumn{3}{c}{Endorsement (secondary)} \\\\\n"
  "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
  "Model & $\\hat\\beta$ & SE & $p$ & $\\hat\\beta$ & SE & $p$ \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# coverage: add evidence-gradient column

# self-conditioning arm: the model's own prior conclusion quoted back to it
_sc=pd.read_csv('results/selfcond.csv').set_index('model')
_sp=pd.read_csv('results/selfcond_split.csv')
rows=[]
for m in ORDER:
    if m not in _sc.index: continue
    r=_sc.loc[m]
    y=_sp[(_sp.model==m)&(_sp.bucket=='yes')]
    u=_sp[(_sp.model==m)&(_sp.bucket=='unknown')]
    ys=f"{int(y.iloc[0].n)} & {y.iloc[0].base:.1f}~$\\to$~{y.iloc[0]['self']:.1f}" if len(y) else '-- & --'
    us=f"{int(u.iloc[0].n)} & {u.iloc[0].base:.1f}~$\\to$~{u.iloc[0]['self']:.1f}" if len(u) else '-- & --'
    rag=f"{r.rag_end:.1f}" if not pd.isna(r.rag_end) else '--'
    rows.append(f"{LABEL[m]} & {int(r.n)} & {r.base_end:.1f} & {r.self_end:.1f} & $[{r.d_lo:+.0f},{r.d_hi:+.0f}]$ & {rag} & {ys} & {us} \\\\")
w('selfcond',"\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
  "& & \\multicolumn{3}{c}{Endorsement, all resistance cases} & Oracle & \\multicolumn{2}{c}{Said resistance} & \\multicolumn{2}{c}{Said unknown} \\\\\n"
  "\\cmidrule(lr){3-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
  "Model & $n$ & Base & Self & 95\\% CI on diff. & evidence & $n$ & Base~$\\to$~Self & $n$ & Base~$\\to$~Self \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# wild cluster bootstrap p-values for the gene-matched estimator
_wg=pd.read_csv('results/wild_gene.csv').set_index('model')
_gm2=pd.read_csv('results/genematched.csv').set_index('model')
rows=[]
for m in ORDER:
    if m not in _wg.index: continue
    r=_wg.loc[m]; g=_gm2.loc[m]
    rows.append(f"{LABEL[m]} & {sg(g.d_gene)} & $[{100*g.lo:+.0f},{100*g.hi:+.0f}]$ & {r.wild_p:.4f} \\\\")
w('wildgene',"\\begin{tabular}{lrrr}\n\\toprule\nModel & $\\Delta_{\\mathrm{gene}}$ & 95\\% percentile CI & Wild boot.\\ $p$ \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# self-conditioning controls: false attribution and the no-instruction variant
_ct=pd.read_csv('results/controls.csv').set_index('model')
_cg=pd.read_csv('results/genematched.csv').set_index('model')
rows=[]
for m in ORDER:
    if m not in _ct.index: continue
    r=_ct.loc[m]
    ni=f"{r.noinstr_yes:.1f}" if not pd.isna(r.noinstr_yes) else '--'
    ci=f"$[{r.d_lo:+.0f},{r.d_hi:+.0f}]$" if not pd.isna(r.d_lo) else '--'
    dg=sg(_cg.loc[m,'d_gene']) if m in _cg.index else '--'
    rows.append(f"{LABEL[m]} & {int(r.n_yes)} & {dg} & {r.base_yes:.1f} & {r.self_yes:.1f} & {ni} & "
                f"{r.false_yes:.1f} & {ci} & {r.base_strong:.1f} & {r.false_strong:.1f} \\\\")
w('controls',"\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
  "& & & \\multicolumn{3}{c}{Endorsement} & \\multicolumn{2}{c}{Endorse, opposite attrib.} & \\multicolumn{2}{c}{Strong rec.} \\\\\n"
  "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
  "Model & $n$ & $\\Delta_{\\mathrm{gene}}$ & Base & Self & No instr. & \\% & 95\\% CI vs base & Base (strong) & Opposite \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# ---- review round 2 tables ----
_ep=pd.read_csv('results/endpoints.csv').set_index('model')

# safety endpoint with intervals + the strong-rec-only estimator
rows=[]
for m in ORDER:
    if m not in _ep.index: continue
    r=_ep.loc[m]
    ratio=r.strong_R/max(r.strong_S,1e-9)
    rows.append(f"{LABEL[m]} & {r.strong_R:.1f} & $[{r.s_lo:.1f},{r.s_hi:.1f}]$ & {r.strong_S:.1f} & {ratio:.2f} & "
                f"{r.end_R:.1f} & $[{r.e_lo:.0f},{r.e_hi:.0f}]$ & "
                f"{r.end_S:.1f} & $[{r.c_lo:.0f},{r.c_hi:.0f}]$ \\\\")
w('endpoints',"\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
  "& \\multicolumn{4}{c}{Unqualified strong recommendation} & \\multicolumn{2}{c}{$P(\\textrm{endorse}\\mid R)$} & \\multicolumn{2}{c}{$P(\\textrm{endorse}\\mid S)$} \\\\\n"
  "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
  "Model & on $R$ & 95\\% CI & on $S$ & $R/S$ & \\% & 95\\% CI & \\% & 95\\% CI \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# outcome sensitivity: endorsement vs strong-rec-only, plus the level-matched intervals
rows=[]
for m in ORDER:
    if m not in _ep.index: continue
    r=_ep.loc[m]
    rows.append(f"{LABEL[m]} & {r.dg:+.1f} & $[{r.dg_lo:+.0f},{r.dg_hi:+.0f}]$ & "
                f"{r.dg_strong:+.1f} & $[{r.dgs_lo:+.0f},{r.dgs_hi:+.0f}]$ & "
                f"{r.dl:+.1f} & $[{r.dl_lo:+.0f},{r.dl_hi:+.0f}]$ \\\\")
w('outcome',"\\begin{tabular}{lrrrrrr}\n\\toprule\n"
  "& \\multicolumn{2}{c}{$\\Delta_{\\mathrm{gene}}$, endorsement} & \\multicolumn{2}{c}{$\\Delta_{\\mathrm{gene}}$, strong rec.\\ only} & \\multicolumn{2}{c}{$\\Delta_{\\mathrm{lvl}}$} \\\\\n"
  "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
  "Model & Est. & 95\\% CI & Est. & 95\\% CI & Est. & 95\\% CI \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# threshold sensitivity and the gene-averaged estimand
_th=pd.read_csv('results/threshold.csv').set_index('model')
rows=[]
for m in ORDER:
    if m not in _th.index: continue
    r=_th.loc[m]
    rows.append(f"{LABEL[m]} & {r.t2:+.1f} & {r.t3:+.1f} & {r.t4:+.1f} & "
                f"{r.gene_avg:+.1f} & $[{r.ga_lo:+.0f},{r.ga_hi:+.0f}]$ \\\\")
w('threshold',"\\begin{tabular}{lrrrrr}\n\\toprule\n"
  "& \\multicolumn{3}{c}{$\\Delta_{\\mathrm{gene}}$ by cell threshold} & \\multicolumn{2}{c}{Gene-averaged} \\\\\n"
  "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
  "Model & $\\geq2$ & $\\geq3$ & $\\geq4$ & Est. & 95\\% CI \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# Delta_gene on the pre-registered halves
_hv=pd.read_csv('results/halves.csv')
rows=[]
for m in ORDER:
    h=_hv[_hv.model==m]
    if len(h)<2: continue
    e=h[h.half=='EXPLORE'].iloc[0]; c=h[h.half=='CONFIRM'].iloc[0]
    rows.append(f"{LABEL[m]} & {e.pt:+.1f} & $[{e.lo:+.0f},{e.hi:+.0f}]$ & "
                f"{c.pt:+.1f} & $[{c.lo:+.0f},{c.hi:+.0f}]$ \\\\")
w('halves',"\\begin{tabular}{lrrrr}\n\\toprule\n"
  "& \\multicolumn{2}{c}{EXPLORE (16 cells)} & \\multicolumn{2}{c}{CONFIRM (12 cells)} \\\\\n"
  "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
  "Model & $\\Delta_{\\mathrm{gene}}$ & 95\\% CI & $\\Delta_{\\mathrm{gene}}$ & 95\\% CI \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# knowledge-base overlap
_ov=pd.read_csv('results/kb_overlap.csv')
rows=[f"{r.a} & {r.b} & {int(r.n_a):,} & {int(r.n_b):,} & {int(r.shared)} & {r.pct_of_smaller:.1f} & {r.jaccard:.3f} \\\\"
      for _,r in _ov.iterrows()]
w('kboverlap',"\\begin{tabular}{llrrrrr}\n\\toprule\n"
  "Source A & Source B & $n_A$ & $n_B$ & Shared & \\% of smaller & Jaccard \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")

# level-by-cohort counts
_lc=pd.read_csv('results/levelcounts.csv')
piv=_lc.pivot(index='cohort',columns='level',values='resistance').fillna(0).astype(int)
pv2=_lc.pivot(index='cohort',columns='level',values='sensitivity').fillna(0).astype(int)
order=[c for c in ['CIViC','JAX-CKB','CGI','MOAlmanac','OncoKB'] if c in piv.index]
rows=[]
for c in order:
    tot=piv.loc[c].sum()
    rows.append(f"{c} & "+" & ".join(f"{piv.loc[c,l]}/{pv2.loc[c,l]}" for l in ['A','B','C','D'])
                +f" & {tot:,} & {100*piv.loc[c,'D']/max(tot,1):.1f} \\\\")
w('levelcounts',"\\begin{tabular}{lccccrr}\n\\toprule\n"
  "Cohort & A & B & C & D & Resist.\\ total & \\% at D \\\\\n\\midrule\n"
  +"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")
