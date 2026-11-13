import pandas as pd, re, json
trip = pd.read_csv('data/triples_raw.csv')

def genes_from_mp(mp):
    """Extract HGNC-like gene symbols from a CIViC molecular profile string."""
    s = str(mp)
    # fusions: A::B  (v:: means unspecified partner)
    if '::' in s:
        parts = re.split(r'::', s.split(' ')[0])
        gs = [p for p in parts if p and p.lower() != 'v']
        if gs: return gs
    # otherwise first token is the gene
    tok = s.split(' ')[0]
    tok = re.sub(r'[^A-Za-z0-9\-]', '', tok)
    return [tok] if tok else []

trip['genes'] = trip.molecular_profile.map(genes_from_mp)
trip['gene'] = trip.genes.map(lambda g: g[0] if g else None)
trip['n_genes'] = trip.genes.map(len)
trip = trip[trip.gene.notna() & (trip.gene.str.len()>1)]
print("triples with gene:", len(trip))
print("unique genes:", trip.gene.nunique(), "| unique therapies:", trip.therapy.nunique())
print("\nTop genes:", trip.gene.value_counts().head(15).to_dict())
print("\nMP->gene spot check:")
for mp in ['v::ALK Fusion','ETV6::NTRK3 Fusion','EGFR L858R','MET Exon 14 Skipping Mutation','BRCA1 Mutation','KRAS G12C']:
    print(f"  {mp!r:38s} -> {genes_from_mp(mp)}")
trip.to_csv('data/triples_genes.csv', index=False)
