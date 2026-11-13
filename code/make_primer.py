from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.table import WD_TABLE_ALIGNMENT
NAVY=RGBColor(0x1F,0x3A,0x5F)
doc=Document(); st=doc.styles['Normal']; st.font.name='Calibri'; st.font.size=Pt(12.5)
st.paragraph_format.space_after=Pt(9)
for s in doc.sections:
    s.left_margin=s.right_margin=Inches(0.95); s.top_margin=Inches(0.8); s.bottom_margin=Inches(0.7)
def P(t,bold=False,size=12.5,after=9,before=0):
    p=doc.add_paragraph(); r=p.add_run(t); r.bold=bold; r.font.size=Pt(size)
    p.paragraph_format.space_after=Pt(after); p.paragraph_format.space_before=Pt(before)
def H1(t): P(t,True,17,after=8,before=22)
def H2(t): P(t,True,13.5,after=6,before=16)
def B(t):
    p=doc.add_paragraph(t,style='List Bullet'); p.paragraph_format.space_after=Pt(6)
    for r in p.runs: r.font.size=Pt(12.5)
def table(hdr,rows):
    t=doc.add_table(rows=1,cols=len(hdr)); t.style='Light List Accent 1'; t.alignment=WD_TABLE_ALIGNMENT.LEFT
    for i,h in enumerate(hdr):
        c=t.rows[0].cells[i]; c.text=''; r=c.paragraphs[0].add_run(h); r.bold=True; r.font.size=Pt(12)
    for row in rows:
        cs=t.add_row().cells
        for i,v in enumerate(row):
            cs[i].text=''; rr=cs[i].paragraphs[0].add_run(str(v)); rr.font.size=Pt(12)
    doc.add_paragraph().paragraph_format.space_after=Pt(4)

P('Understanding This Project From Scratch',True,20,after=6)
P('A plain-language guide to the evidence-discrimination study. No background assumed.',
  False,12.5,after=18)

H1('The project in one paragraph')
P('Doctors treating cancer increasingly choose drugs based on the specific genetic mutation in a '
  'patient’s tumour. Some of those drug–mutation pairings are backed by large clinical trials. Others '
  'are backed only by an experiment in a dish. AI language models are now being proposed to help '
  'doctors make these decisions. We tested whether these models treat strong and weak evidence '
  'differently — and found that they do, but only for genes that scientists have studied a lot. For '
  'genes that are less studied, the models recommend drugs at roughly the same rate whether the '
  'evidence is excellent or nearly nonexistent.')

H1('Part 1 — The clinical setting')
H2('Precision oncology, in brief')
P('Traditional chemotherapy attacks fast-dividing cells generally. Precision oncology instead looks '
  'for a specific molecular defect driving a particular tumour, then uses a drug designed to block '
  'that defect. If a lung tumour has a mutation in a gene called EGFR, a drug that blocks EGFR may '
  'shrink it dramatically. The same drug in a patient without that mutation does nothing useful.')
P('This is why tumours get genetically sequenced. The sequencing report says which genes are altered, '
  'and the treating team has to decide what, if anything, that implies for therapy.')

H2('Vocabulary you need')
table(['Term','What it means'],
 [['Gene','A stretch of DNA. EGFR, BRAF, ALK, NTRK1 are all gene names.'],
  ['Variant / alteration','A specific change in a gene. “EGFR L858R” means: in the EGFR gene, at position 858, the amino acid L was replaced by R.'],
  ['Fusion','Two genes accidentally joined together, creating a hybrid that behaves abnormally. Written like “EML4::ALK”.'],
  ['Targeted therapy','A drug built to block a specific molecular defect. Erlotinib, osimertinib, larotrectinib.'],
  ['Actionable','Shorthand for: this alteration gives us a real reason to choose a particular drug.'],
  ['Molecular tumour board','A recurring meeting where oncologists, pathologists and geneticists review sequencing results together and decide what to do.']])

H2('A concrete example')
P('A 62-year-old with metastatic lung cancer. Sequencing finds EGFR L858R. This is one of the '
  'best-established findings in oncology: multiple randomised trials show EGFR inhibitors work here. '
  'Recommending erlotinib is straightforwardly correct.')
P('Now change one thing. Sequencing instead finds EGFR A864T — a rare change in the same gene that '
  'almost nobody has studied. There are zero published papers on it. Is erlotinib still the right '
  'call? The honest answer is that nobody knows. The correct response is “insufficient evidence”, not '
  'a recommendation.')
P('That contrast — same gene, wildly different strength of evidence — is the engine of this whole '
  'study.', bold=True)

H1('Part 2 — Why evidence gets graded')
P('Not all scientific support is equal. Consider three ways someone might claim drug X works for '
  'mutation Y:')
B('A randomised controlled trial in 500 patients showed tumours shrank. Very strong.')
B('One doctor published a report about one patient who improved. Weak — that patient might have improved anyway.')
B('Cells in a dish carrying mutation Y died when exposed to drug X. Weakest — dishes are not people.')
P('Medicine formalises this into evidence hierarchies. We use one called CIViC, a public, expert-curated '
  'database of cancer variant interpretations. Every entry carries a level:')
table(['CIViC level','Meaning','Rough translation'],
 [['A','Validated association','Proven in trials, often FDA-approved'],
  ['B','Clinical evidence','Real patient data, but less definitive'],
  ['C','Case study','One or a few patients'],
  ['D','Preclinical','Cells or animals only'],
  ['E','Inferential','Indirect reasoning']])
P('Why this matters practically: giving a patient a drug on Level D evidence means exposing them to '
  'side effects, cost, and the opportunity cost of not using a treatment that works — on the strength '
  'of a cell-culture experiment. That is a real harm, not a bookkeeping error.')

H1('Part 3 — Where AI enters')
P('A large language model, or LLM, is a system trained on enormous quantities of text to predict what '
  'text comes next. GPT-4o is one. That simple training objective turns out to produce systems that '
  'can answer medical questions fluently, which is why hospitals and companies are now proposing them '
  'as decision support for exactly the tumour board setting described above.')
P('The catch is that fluency is not reliability. A model can produce a confident, well-written '
  'recommendation that is wrong. So a whole research field has grown up around evaluation: designing '
  'tests that reveal how these systems actually behave before anyone trusts them with patients. This '
  'project is an evaluation study.')

H1('Part 4 — The research question')
H2('The distinction that makes this a real question')
P('Most evaluations ask: how often is the model right? That is accuracy. Our question is different '
  'and subtler.')
P('Imagine two doctors. Both prescribe more readily when evidence is strong. But Doctor A changes her '
  'behaviour sharply between strong and weak evidence, while Doctor B barely changes at all — he '
  'prescribes at nearly the same rate either way. Both might have the same overall accuracy rate. But '
  'Doctor B is not really using the evidence. He is doing something else and getting lucky.')
P('Our question is whether a model behaves like Doctor A for some genes and Doctor B for others.', bold=True)
P('If so, that is invisible to accuracy metrics. Average performance can look fine while the model '
  'quietly abandons evidentiary reasoning in exactly the corners where doctors most need help.')

H2('Why literature volume is the variable of interest')
P('We measure how much has been written about each gene by counting PubMed articles. EGFR has about '
  '106,000. NTRK2 has about 800. The hypothesis is that a model’s ability to discriminate good evidence '
  'from bad depends on how much text about that gene it absorbed during training.')
P('Important caution: PubMed count measures how prominent a gene is in the literature. It does not '
  'directly measure what was in the model’s training data, which nobody outside the companies knows. '
  'We say “prominence” and treat training exposure as one plausible explanation, not a proven one.')

H1('Part 5 — How the experiment works')
H2('The confounding problem, and the fix')
P('Suppose we simply compared EGFR cases to NTRK2 cases. If they behaved differently, we would learn '
  'nothing — those genes differ in dozens of ways besides prominence. Different cancers, different drug '
  'classes, different biology. This is called confounding: the thing you care about is tangled with '
  'things you don’t.')
P('Our fix is the within-gene design. We keep only genes that contain both strongly-supported and '
  'weakly-supported cases. Then we look inside each gene. Within EGFR, prominence is a fixed number — '
  '106,000 papers for every case. Only the evidence level changes. So any difference in how the model '
  'responds cannot be caused by gene identity.', bold=True)
P('This left 534 cases across 27 genes.')

H2('The probe')
P('Each case became a short clinical vignette: a patient with a given cancer, a given alteration, and '
  'a proposed drug. The prompt never mentions the CIViC level or any article count — the model is '
  'blind to what we are measuring. The model returns a structured answer: a recommendation (Strongly '
  'recommend / Consider / Insufficient evidence / Do not recommend), its own assessment of evidence '
  'strength, and a confidence number.')
P('We ran all 534 cases through three models — gpt-4o-mini, gpt-4o, gpt-4.1 — giving about 1,600 '
  'judgments. Every one parsed correctly.')

H1('Part 6 — The statistics, in plain terms')
H2('Endorsement')
P('We collapse the four possible answers into one number: did the model endorse the drug or not? '
  '“Strongly recommend” and “Consider” count as endorsement. The other two do not. So endorsement '
  'rate is simply the fraction of cases where the model said yes.')

H2('Main effect versus interaction')
P('A main effect answers: does evidence level change endorsement overall? Ours is positive — stronger '
  'evidence does produce more endorsement. Good, the models are not random.')
P('An interaction answers a different question: does the size of that effect itself depend on something '
  'else? Here: is the evidence effect bigger in prominent genes than in obscure ones?', bold=True)
P('Back to the doctors. The main effect is “both doctors prescribe more with strong evidence.” The '
  'interaction is “Doctor A’s behaviour changes much more than Doctor B’s.” The interaction is our '
  'result. It is the whole paper.')

H2('p-values and clustering')
P('A p-value estimates how likely you would see a pattern this strong if nothing real were going on. '
  'Small means unlikely to be a fluke. Ours are 0.0004, 0.0009 and 0.047.')
P('Clustering handles a subtlety. We have 534 cases but only 27 genes, and cases within one gene are '
  'not independent — 79 of them are EGFR. Treating them as 534 independent observations would make our '
  'results look more certain than they are. Clustering by gene corrects for this. The honest caveat is '
  'that 27 clusters is on the low side; convention prefers 30 to 50, which is why we still owe a '
  'bootstrap check.')

H2('Ceiling effects')
P('For Level A evidence, every model endorses nearly 100% of the time in every gene. There is no room '
  'to go higher, so no differences can appear there. This means our effect lives entirely at the '
  'weak-evidence end. That is worth saying out loud, and it happens to be the clinically important '
  'end anyway.')

H1('Part 7 — What we found')
P('One. Models do track evidence, but only moderately. And they almost never say “insufficient '
  'evidence” — gpt-4o-mini uses it in about 5% of responses.')
P('Two. Preclinical-only evidence still draws an unqualified “strongly recommend” in 10 to 23 percent '
  'of cases, depending on the model.')
P('Three, the main result. The evidence-by-prominence interaction is positive and statistically '
  'significant in all three models. Concretely: the gap in endorsement between the best and worst '
  'evidence is about twice as large in well-studied genes as in poorly-studied ones.', bold=True)
P('Four. Gene by gene, EGFR tracks evidence at +0.60 and BRAF at +0.45. NTRK1 and NTRK2 come out at '
  'roughly zero — the model endorses at the same rate whether support is a trial or a dish.')
P('Why NTRK matters. Larotrectinib and entrectinib are approved for NTRK fusions across any tumour '
  'type. They are used precisely in rare situations where a doctor is most likely to want help. That '
  'is the setting where the model appears least able to tell good evidence from bad.')
P('The interpretation. Patients whose tumours carry alterations in less-studied genes may receive '
  'recommendations that are effectively evidence-blind. This is a fairness problem that no accuracy '
  'metric would surface.', bold=True)

H1('Part 8 — Why this might be wrong')
P('Taking the weaknesses seriously is what makes the finding credible. Three matter most.')
P('Only OpenAI models.', bold=True)
P('All three are from one company and likely share training data and methods. So the honest current '
  'claim is about GPT models, not about language models generally. Adding a different family is the '
  'top priority.')
P('Thin clustering.', bold=True)
P('27 genes is few. The p-values may be optimistic. A wild cluster bootstrap will tell us.')
P('Ground truth may be contaminated.', bold=True)
P('This is the deepest concern. CIViC is curated by humans who also study well-known genes more. So '
  'CIViC levels for famous genes may be more carefully assigned than for obscure ones. If so, part of '
  'our effect could reflect better labels rather than better model behaviour. A direct test: control '
  'for how many curated entries each gene has, and see whether the interaction survives.')

H1('If someone asks you…')
P('“Isn’t this just that models know less about rare things?”', bold=True)
P('No. That would be an accuracy gap. We are showing something else: the model applies a different '
  'evidentiary standard. It is not that it is wrong more often on rare genes — it is that its '
  'recommendations stop responding to evidence quality at all.')
P('“Why not just use accuracy?”', bold=True)
P('Because accuracy averages over exactly the distinction we care about. A model could score well '
  'overall while being evidence-blind in a subset of genes.')
P('“Why PubMed counts?”', bold=True)
P('They are public, reproducible, and a reasonable proxy for how much a topic has been discussed. We '
  'are careful to call it prominence, not training exposure.')
P('“Why within-gene?”', bold=True)
P('To remove confounding. Comparing across genes would confound prominence with all the other ways '
  'genes differ.')
P('“What would change your mind?”', bold=True)
P('If the interaction disappeared after controlling for CIViC curation density, or failed to replicate '
  'in a non-OpenAI model family, the claim would be in serious trouble.')

H1('Glossary')
table(['Term','Meaning'],
 [['Actionable','An alteration gives a real reason to pick a particular drug'],
  ['CIViC','Public expert-curated database of cancer variant interpretations, with graded evidence levels'],
  ['Endorsement','Model answered “Consider” or “Strongly recommend”'],
  ['Interaction','Whether the size of one effect depends on another variable — our main result'],
  ['LLM','Large language model; a text-prediction system such as GPT-4o'],
  ['Prominence','How much literature exists about a gene, measured by PubMed article count'],
  ['Within-gene design','Comparing cases inside the same gene so gene identity cannot explain differences'],
  ['Clustering','Statistical correction for non-independent observations grouped by gene'],
  ['Ceiling effect','No room for a measure to increase, hiding possible differences'],
  ['Confounding','When the variable you care about is entangled with others you did not control']])
doc.save('Project_Primer.docx'); print('primer written')
