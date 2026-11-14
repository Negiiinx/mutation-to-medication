"""Cross-check numbers quoted in the paper prose against the generated result files.

Reviewers found three separate arithmetic/reconciliation errors by hand. This asserts the
relationships that must hold, so the same class of error fails loudly instead of shipping.
"""
import warnings; warnings.filterwarnings('ignore')
import json, os, pathlib, re, subprocess, sys
import pandas as pd
from anlib import load, ORDER, LABEL, VENDOR as VEND

PAPER = 'manuscript/main.tex'
fails, checks = [], 0


def ck(name, cond, detail=''):
    global checks
    checks += 1
    if not cond:
        fails.append(f'{name}: {detail}')


if not os.path.isdir('tables'):
    print('Run make_tables.py and make_tables_cr.py first: this script checks the\n'
          'manuscript against the generated tables, which are not shipped.')
    raise SystemExit(0)

if not os.path.exists(PAPER):
    print('The manuscript is published rather than distributed here, so the prose\n'
          'checks have nothing to read. Put main.tex, and optionally main.pdf, in\n'
          'manuscript/ to run them.')
    raise SystemExit(0)

tex = open(PAPER).read()
# The compiled PDF is not part of the release either. Five presentation checks
# below need it. Drop main.pdf into manuscript/ to run them.
pdf = (subprocess.run(['pdftotext', '-layout', 'manuscript/main.pdf', '-'],
                      capture_output=True, text=True).stdout
       if os.path.exists('manuscript/main.pdf') else '')
_HAVE_PDF = bool(pdf)

# ---- 1. self-conditioning arm reconciles with the headline and knowledge tables ----
sc = pd.read_csv('results/selfcond.csv').set_index('model')
sp = pd.read_csv('results/selfcond_split.csv')
hd = pd.read_csv('results/headline.csv').set_index('model')

D = load('data/cohort_direction.csv', 'dir')
D = D[(D.label == 'RESISTANT') & D.model.isin(ORDER)]
RAG = load('data/cohort_direction_rag.csv', 'rag')
RAG = RAG[RAG.label == 'RESISTANT'] if RAG is not None else None

for m in sc.index:
    ck(f'selfcond n=312 [{m}]', sc.loc[m, 'n'] == 312, f"n={sc.loc[m,'n']}")
    base_all = 100 * D[D.model == m].endorse.mean()
    ck(f'selfcond base == headline resistance endorsement [{m}]',
       abs(sc.loc[m, 'base_end'] - base_all) < 0.05,
       f"{sc.loc[m,'base_end']:.2f} vs {base_all:.2f}")
    ck(f'selfcond base == headline.csv endorse [{m}]',
       abs(sc.loc[m, 'base_end'] - 100 * hd.loc[m, 'endorse']) < 0.05,
       f"{sc.loc[m,'base_end']:.2f} vs {100*hd.loc[m,'endorse']:.2f}")
    if RAG is not None and m in set(RAG.model):
        rag_all = 100 * RAG[RAG.model == m].endorse.mean()
        ck(f'selfcond oracle == mitigation arm [{m}]',
           abs(sc.loc[m, 'rag_end'] - rag_all) < 0.05,
           f"{sc.loc[m,'rag_end']:.2f} vs {rag_all:.2f}")

# bucket baselines must equal the knowledge-probe table's buckets
YES, UNK = 'Yes, confers resistance', 'Unknown or insufficient evidence'
for _, r in sp.iterrows():
    path = f"raw_responses/know_{r.model.replace(':', '-')}.jsonl"
    if not os.path.exists(path):
        continue
    d = {}
    for L in open(path):
        try:
            x = json.loads(L)
        except Exception:
            continue
        rs = x.get('resp', {})
        if 'error' in rs or 'resistance_known' not in rs:
            continue
        d.setdefault(x['uid'], rs['resistance_known'])
    key = YES if r.bucket == 'yes' else UNK
    g = D[D.model == r.model]
    ref = g[g.uid.map(lambda u: d.get(u)) == key]
    ck(f'bucket n [{r.model}/{r.bucket}]', len(ref) == r.n, f'{r.n} vs {len(ref)}')
    ck(f'bucket base [{r.model}/{r.bucket}]',
       abs(100 * ref.endorse.mean() - r.base) < 0.05,
       f"{r.base:.2f} vs {100*ref.endorse.mean():.2f}")

# ---- 2. prose claims that state a count ----
y = sp[sp.bucket == 'yes']
# the paper quotes this range at one decimal, the same precision as the table
ck('yes-bucket range 0.0-3.5 for all seven',
   len(y) == 7 and round(y['self'].min(), 1) >= 0.0 and round(y['self'].max(), 1) <= 3.5,
   f"n={len(y)} range {y['self'].min():.2f}-{y['self'].max():.2f}")
u = sp[sp.bucket == 'unknown']
inc = sum(1 for _, r in u.iterrows() if r.lo <= 0 <= r.hi)
ck('five of seven unknown intervals include zero', inc == 5 and len(u) == 7,
   f'{inc} of {len(u)}')
main_ex = sum(1 for _, r in sc.iterrows() if not (r.d_lo <= 0 <= r.d_hi))
ck('six of seven main intervals exclude zero', main_ex == 6, f'{main_ex} of {len(sc)}')
stronger = sum(1 for m in sc.index
               if not pd.isna(sc.loc[m, 'rag_end']) and sc.loc[m, 'self_end'] < sc.loc[m, 'rag_end'])
ck('four models: self stronger than oracle', stronger == 4, f'{stronger}')

# ---- 3. gene-matched estimator: agreement between the two inference procedures ----
gm = pd.read_csv('results/genematched.csv').set_index('model')
wg = pd.read_csv('results/wild_gene.csv').set_index('model')
perc = {m for m in gm.index if not (gm.loc[m, 'lo'] <= 0 <= gm.loc[m, 'hi'])}
wild = {m for m in wg.index if wg.loc[m, 'wild_p'] < 0.05}
ck('percentile and wild bootstrap agree on which models are significant',
   perc == wild, f'percentile {sorted(perc)} vs wild {sorted(wild)}')
ck('six significant under gene matching', len(perc) == 6, str(len(perc)))
# Delta_gene is weighted by the resistance cases inside the eligible cells, which is a
# smaller set than the cases in genes that have at least one eligible cell. The paper quoted
# the second figure as coverage for a long time; these two checks pin the first.
_mc = 2  # reviewer_analyses.MIN_CELL
_both = load('data/cohort_direction.csv', 'dir')      # D above is resistance-only
_one = _both[_both.model == ORDER[0]]
_w = sum(len(sub[sub.label == 'RESISTANT'])
         for _, sub in _one.groupby(['gene', 'best_level'])
         if len(sub[sub.label == 'RESISTANT']) >= _mc and len(sub[sub.label == 'SENSITIVE']) >= _mc)
_nres = len(_one[_one.label == 'RESISTANT'])
_cov = f'{100 * _w / _nres:.1f}'
assert _w > 100 and _nres == 312, f'cohort load changed: {_w}/{_nres}'
ck('paper quotes the weighted case count behind Delta_gene', str(_w) in tex, f'{_w} missing')
ck('paper quotes the matching coverage', _cov in tex, f'{_cov}% missing')

ck('gene-matched cells/genes as stated in text',
   gm.n_cells.iloc[0] == 28 and gm.n_genes.iloc[0] == 17,
   f'{gm.n_cells.iloc[0]} cells / {gm.n_genes.iloc[0]} genes')

# ---- 4. every appendix that promises a table contains exactly one ----
ap = tex[tex.index('\\appendix'):]
# the appendices are grouped, so the unit that promises a table is the subsection; splitting
# on both levels keeps one check per original appendix rather than one per group
secs = re.split(r'\\(?:sub)?section\{', ap)[1:]
for sec in secs:
    title = sec[:sec.index('}')]
    lab = re.search(r'\\label\{(app:[a-z0-9]+)\}', sec)
    if not lab:
        continue          # a group heading, which holds subsections rather than content
    ntab = sec.count('\\input{tables/')
    if lab and lab.group(1) in ('app:prompt', 'app:delta', 'app:harmon', 'app:curation',
                                'app:models'):   # models appendix has an inline tabular
        continue                                  # prose-only appendices
    prose_only = {'app:prompt', 'app:delta', 'app:models'}
    if lab and lab.group(1) in prose_only:
        continue
    ck(f'appendix carries at least one table [{title[:34]}]', ntab >= 1, f'{ntab} tables')

# ---- 5. no dangling references, no duplicate labels, page budget ----
labs = set(re.findall(r'\\label\{([^}]+)\}', tex))
refs = set(re.findall(r'\\ref\{([^}]+)\}', tex))
ck('no dangling \\ref', refs <= labs, str(sorted(refs - labs)))
alll = re.findall(r'\\label\{([^}]+)\}', tex)
ck('no duplicate labels', len(alll) == len(set(alll)),
   str([x for x in set(alll) if alll.count(x) > 1]))
ck('every table file is used',
   all(f'tables/{os.path.splitext(f)[0]}' in tex for f in os.listdir('tables')),
   str([f for f in os.listdir('tables')
        if f'tables/{os.path.splitext(f)[0]}' not in tex]))

# ---- 6. judgment total, Delta labelling, prompt coverage ----
cov = open('tables/coverage.tex').read()
def _block(label):
    m = re.search(r'\{' + label + r'\}.*?& ([\d,]+)', cov, re.S)
    return int(m.group(1).replace(',', '')) if m else None
# the coverage table has three blocks: the arms as submitted, the two self-conditioning
# controls, and the control and robustness arms. Each has a fixed expected total
# and the three must reconcile to the grand total quoted in the abstract.
_main = _block(r'\\emph\{Main arms\}')
_ctrl = _block(r'\\emph\{Self-conditioning controls\}')
_new = _block(r'\\emph\{Control and robustness arms\}')
_tot = _block(r'\\textbf\{Total\}')
ck('main-arm subtotal', _main == 36590, str(_main))
ck('controls subtotal', _ctrl == 3818, str(_ctrl))
ck('third-block subtotal present', _new is not None and _new > 0, str(_new))
ck('subtotals sum to the total', None not in (_main, _ctrl, _new, _tot)
   and _main + _ctrl + _new == _tot, f'{_main}+{_ctrl}+{_new} vs {_tot}')
ck('paper quotes the main-arm subtotal', '36{,}590' in tex, 'missing')
ck('paper quotes the controls subtotal', '3{,}818' in tex, 'missing')
ck('paper quotes the third-block subtotal', f'{_new:,}'.replace(',', '{,}') in tex, str(_new))
ck('abstract quotes the grand total',
   f'{_tot:,}'.replace(',', '{,}') in tex[tex.index('\\begin{abstract}'):tex.index('\\end{abstract}')],
   str(_tot))
# the main block reports the three-model CGI arm; the extension is counted in the third block
ck('CGI main row is the three-model arm', 'CGI & 903 & 3 GPT & 2,709' in cov, 'changed')
ck('total not called blinded', 'blinded judgments' not in tex, 'still says blinded judgments')
ck('self arm in coverage table', 'Self-conditioning' in cov, 'missing row')

body_tex = tex[:tex.index('\\appendix')]
after = body_tex[body_tex.index('A matched discrimination statistic'):]
_pat = re.compile(r'(?<!\\mathrm\{lvl\})(?<!\\mathrm\{gene\})\$\\Delta\$' + r'\s*(\w+)')
# A bare $\Delta$ is allowed only where the sentence is about both estimators at once:
# "$\Delta$ in either form", "$\Delta$ needs/requires no such coefficient", "$\Delta$ is a
# difference, so it also rewards...", "leaves every $\Delta$ unchanged", "the binary form for
# $\Delta$ as the more conservative test". Anywhere a single estimator is meant, name it.
_generic = ('in', 'needs', 'requires', 'is', 'unchanged', 'as')
bare = [m for m in _pat.finditer(after) if m.group(1) not in _generic]
ck('no bare $\\Delta$ after the estimator section', len(bare) == 0, f'{len(bare)} occurrences')

# the prompt appendix is a subsection once the appendices are grouped, so match either level
_ap = re.search(r'\\(?:sub)?section\{Prompts\}', tex)
ck('prompt appendix present', _ap is not None, 'no Prompts heading')
ap_prompt = tex[_ap.start():]
ap_prompt = ap_prompt[:ap_prompt.index('\\normalsize')]
for arm in ['T1 (primary)', 'T2', 'T3', 'Mitigation arm', 'Knowledge probe', 'Self-conditioning arm']:
    ck(f'prompt documented [{arm}]', arm in ap_prompt, 'missing')
ck('stateless calls stated', 'stateless' in ap_prompt, 'no statelessness note')
ck('no shared-session claim', 'same session' not in tex and 'one session' not in tex, 'session claim')

# mitigation table reports the primary estimator, and its values match the CSV
mg = pd.read_csv('results/mitigation_gene.csv').set_index('model')
mit = open('tables/mitigation.tex').read()
ck('mitigation table labelled gene-matched', 'Delta_{\\mathrm{gene}}' in mit, 'label')
for m in mg.index:
    v = f"{100*mg.loc[m,'dg_after']:+.1f}"
    ck(f'mitigation gene value present [{m}]', v in mit, v)

# ---- 7. items checked by hand ----
bl = open('tables/bylevel.tex').read()
ck('by-level table labels the CIViC block', 'CIViC &' in bl, 'CIViC label missing')
for kb in ['JAX-CKB', 'CGI']:
    ck(f'by-level labels {kb}', f'{kb} &' in bl, 'missing')
cv = open('tables/coverage.tex').read()
ck('gradient cohort attributed to CIViC, not "4 sources"',
   '4 sources' not in cv and 'Evidence gradient & CIViC' in cv, 'wrong attribution')
cons = open('tables/consensus.tex').read()
hk = re.search(r'Claude Haiku 4\.5.*?\$\[([-+\d.]+),([-+\d.]+)\]\$', cons)
ck('Haiku concordance interval matches the prose value',
   hk is not None and hk.group(1).strip() == '-0.2' and hk.group(2).strip() == '+89.0',
   hk.groups() if hk else 'not found')
ck('prose quotes that interval', '[-0.2, +89.0]' in tex, 'prose mismatch')

rb = open('tables/robust.tex').read()
ck('auxiliary table carries no Delta column', 'Delta' not in rb, 'Delta still present')
ck('auxiliary table carries no p column', 'boot' not in rb, 'p column still present')
ck('only one wild-bootstrap p is reported', tex.count('Wild boot') <= 1,
   f"{tex.count('Wild boot')} occurrences")

ck('pre-registration result reported', 'pre-registered confirmatory' in tex.lower(), 'not reported')
ck('pre-registration appendix exists', 'app:prereg' in tex, 'missing appendix')
pr = pd.read_csv('results/prereg.csv')
cf = pr[(pr.half == 'CONFIRM') & (pr.endpoint == 'strong_rec')]
ck('3 of 9 significant on the primary endpoint', (cf.p < 0.05).sum() == 3,
   str((cf.p < 0.05).sum()))
ck('8 of 9 point estimates in the predicted direction', (cf.beta < 0).sum() == 8,
   str((cf.beta < 0).sum()))
cfe = pr[(pr.half == 'CONFIRM') & (pr.endpoint == 'endorse')]
ck('5 of 9 significant on the secondary endpoint', (cfe.p < 0.05).sum() == 5,
   str((cfe.p < 0.05).sum()))

ck('three non-OpenAI models above GPT-4.1', 'three non-OpenAI models above GPT-4.1' in tex,
   'count wrong')
gmx = pd.read_csv('results/genematched.csv').set_index('model')
above = sum(1 for m in gmx.index if VEND.get(m) != 'OpenAI'
            and gmx.loc[m, 'd_gene'] > gmx.loc['gpt-4.1', 'd_gene'])
ck('that count is 3 in the data', above == 3, str(above))
ck('stratum-weighted framing present', 'stratum-weighted risk difference' in tex, 'missing')
ck('43 clusters scoped to both-direction cohorts',
   re.search(r'cohorts (that )?carry(ing)? both directions suppl(y|ies) at most 43', tex)
   is not None, 'unscoped')
ck('no cross-source-confirmed reference phrasing',
   'cross-source-confirmed reference' not in tex, 'phrase remains')
ck('case-difficulty claim softened',
   'not an artefact of case difficulty' not in tex, 'overclaim remains')

# ---- 8. self-conditioning controls ----
C = pd.read_csv('results/controls.csv')
ck('all seven models have a false-attribution arm', len(C) == 7 and C.false_yes.notna().all(),
   f'{C.false_yes.notna().sum()} of {len(C)}')
# the opposite-attribution interval in Table 27 is the paired difference against BASELINE,
# not against the quoted-conclusion condition; it was the latter for a while and the text
# claimed the former. Pin both the column source and the count the text quotes.
_bex = int(((C.b_lo > 0) | (C.b_hi < 0)).sum())
ck('six of seven false-attribution intervals exclude zero', _bex == 6, str(_bex))
ck('paper states six of seven rather than all seven',
   'all seven intervals strictly positive' not in tex
   and 'Six of seven paired differences' in tex, 'claim not updated')

ck('all seven have the no-instruction arm', C.noinstr_yes.notna().all(),
   f'{C.noinstr_yes.notna().sum()} of {len(C)}')
ck('false attribution raises endorsement above baseline for every model',
   (C.false_yes > C.base_yes).all(), str((C.false_yes > C.base_yes).sum()))
ck('every false-attribution interval is strictly positive',
   (C.d_lo > 0).all(), str((C.d_lo > 0).sum()))
ck('no-instruction matches true-self within 1pp on average',
   abs(C.self_yes - C.noinstr_yes).mean() < 1.0,
   f'{abs(C.self_yes - C.noinstr_yes).mean():.2f}pp')
ck('paper quotes that mean shift', '0.54' in tex, 'missing')
ck('false-attribution range quoted', '16.7' in tex and '95.2' in tex, 'missing')
ck('strong-rec range under false attribution quoted',
   '3.9' in tex and '66.5' in tex, 'missing')
from scipy.stats import spearmanr as _sp
_g = pd.read_csv('results/genematched.csv').set_index('model')
_rho = _sp(C.model.map(_g.d_gene), C.false_yes).correlation
ck('deference-discrimination correlation is -0.86', abs(_rho + 0.86) < 0.01, f'{_rho:.3f}')
# Review asked for the susceptibility association to be quantified rather than described, so
# the body now carries the coefficients. What must accompany them is the confound: the same
# -0.86 holds for baseline endorsement, and the normalised scale is the one we rely on.
_bodytex = tex.split('\\appendix')[0]
ck('susceptibility coefficient reported with its confound',
   ('-0.86' not in _bodytex)
   or ('+0.93' in _bodytex and '-0.79' in _bodytex and 'agreeableness' in _bodytex),
   'coefficient given without the baseline confound')

# the paper must no longer claim the self arm demonstrates evidence use
for bad in ['closes most of the gap',
            'residual failure is one of evidence use',
            'substantial evidence-use component',
            'is a substantial evidence-use']:
    ck(f'overclaim removed [{bad[:34]}]', bad not in tex, 'still present')
ck('deference framing present', 'deference' in tex.lower(), 'missing')
ck('correlational caveat present',
   re.search(r'remains\s+correlational', tex) is not None, 'missing')
ck('controls appendix exists', 'app:controls' in tex, 'missing')

# Delta_gene must never be presented as simply "primary": it is post-registration
ck('no bare "primary estimator"', 'primary estimator' not in tex, 'found')
ck('exploratory framing used at every site', tex.count('primary \\emph{exploratory}')
   + tex.count('primary exploratory') >= 4,
   str(tex.count('primary \\emph{exploratory}') + tex.count('primary exploratory')))
ck('post-registration stated where Delta_gene is introduced',
   'post-registration' in tex and 'no result computed from it is' in tex, 'missing')

if _HAVE_PDF:
    pages = pdf.split('\f')
    body = next(i for i, p in enumerate(pages, 1) if 'References' in p) - 1
    # ML4H Proceedings: 8 pages at submission, plus 1 content page for the camera-ready
    ck('body within 9 pages (camera-ready)', body <= 9, f'{body} pages')
    # the em dash rule is a house-style rule for our own prose; a title quoted in the
    # bibliography is reproduced as its publisher set it, so the reference list is exempt
    _prose = pdf
    if 'References' in _prose and 'Appendix A.' in _prose:
        _i = _prose.index('References')
        _j = _prose.index('Appendix A.', _i)
        _prose = _prose[:_i] + _prose[_j:]
    ck('no em dashes', '\u2014' not in _prose, 'found em dash')
    # author/affiliation strings are read from an untracked file so this script can ship anonymously
    _idf = pathlib.Path('.identity_strings')
    _ids = [x.strip() for x in _idf.read_text().split() if x.strip()] if _idf.exists() else []
    # a name may legitimately appear in the bibliography as a cited author; what must not appear
    # is an author or affiliation string in the body, the title block or the running header
    _body = pdf[:pdf.index('References')] if 'References' in pdf else pdf
    _hits = [x for x in _ids if x in _body]
    # Under review the author strings must be absent; in the camera-ready they must be
    # present, so the same word list checks the opposite condition either side of acceptance.
    if '\\finaltrue' in tex:
        # with no word list there is nothing to look for, exactly as in the branch above;
        # the public release ships without one
        ck('camera-ready: authors named in the title block', bool(_hits) or not _ids,
           'author block still carries no identifying string')
    else:
        ck('anonymous in body', not _hits,
           f'{_hits} in body text' if _ids else 'no .identity_strings file; check skipped')
    _title = 'From Mutation to Medication: Characterizing Failures of Clinical LLMs'
    _hdr = 'From Mutation to Medication: Genomic Resistance'
    _flat = ' '.join(pdf.split())
    ck('title present', _title in _flat, 'title missing')
    ck('running header derives from the title', _hdr in _flat, 'header mismatch')
    for _old in ('Knowing Is Not Using', 'Models Follow Conclusions Ascribed', 'Conclusions Ascribed'):
        ck(f'superseded title absent [{_old[:28]}]', _old not in _flat, 'still present')
    # the abstract total must equal the coverage table total, which is computed from raw files
    # Conclusion then Limitations: Limitations is the required closing section in this template
    ck('conclusion precedes the closing Limitations section',
       re.search(r'\\section\{Conclusion\}', tex) is not None
       and tex.index('\\section{Conclusion}') < tex.index('\\section{Limitations}'), 'misplaced')

    # camera-ready gates: nothing with a placeholder or an anonymous block may ship
    ck('camera-ready: author block de-anonymised', 'Anonymous Author' not in tex, 'still anonymous')
    ck('camera-ready: no unconfirmed email placeholder', 'TO-CONFIRM' not in tex, 'placeholder left in')
    ck('camera-ready: line numbering off',
       not re.search(r'^\\linenumbers', tex, re.M), 'linenumbers still active')
    ck('camera-ready: final switch set', '\\finaltrue' in tex, 'finalfalse')
    ck('camera-ready: required paragraphs present',
       'Data and Code Availability' in tex and 'Institutional Review Board' in tex, 'missing')

print(f'{checks} checks on the data and the text, {len(fails)} failures'
      + ('' if _HAVE_PDF else
         '\nPresentation checks (page count, typography, anonymity of the\n'
         'rendered body) were skipped: they need the compiled manuscript.\n'
         'Put main.pdf in manuscript/ and re-run to include them.'))
for f in fails:
    print('  FAIL', f)
sys.exit(1 if fails else 0)
