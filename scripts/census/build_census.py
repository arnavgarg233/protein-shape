"""Main census: call the sheet topology of every AlphaFold DB domain cut at TED boundaries.

Stages (each resumable):

  filter     keep TED domains with >= 3 strands and mean pLDDT >= 70 whose chain is a Foldcomp
             afdb_rep_v4 representative, from the TED 365M domain summary on stdin:
               gzip -dc ted_365m_domain_boundaries_consensus_level.tsv.gz | \\
                 python scripts/census/build_census.py filter --out OUT/candidates.tsv --reps afdb_rep_v4.lookup
  call       cut each domain and call it under the three strand definitions (one JSON line per domain):
               python scripts/census/build_census.py call --cand OUT/candidates.tsv --out OUT/calls.jsonl \\
                 --source foldcomp --db afdb_rep_v4 --workers 32
             (--source api fetches single chains from the AFDB API, for small tests)
  summarise  counts per topology code:
               python scripts/census/build_census.py summarise --calls OUT/calls.jsonl --out OUT/
"""
import argparse
import json
import sys
import urllib.request
from multiprocessing import Pool
from pathlib import Path

from protein_shape.structure import call_domain

TED_COLS = ['ted_id', 'md5', 'consensus', 'chopping', 'nres', 'nseg', 'plddt', 'n_sse', 'n_helix',
            'n_strand', 'n_helix_strand', 'n_turn', 'proteome', 'cath_label', 'cath_level',
            'cath_method', 'packing', 'norm_rg', 'tax_common', 'tax_name', 'tax_lineage']


def chain_of(ted_id):
    return ted_id.rsplit('_TED', 1)[0]


def cmd_filter(a):
    reps = None
    if a.reps:
        reps = {line.split('\t')[1].strip().replace('.pdb', '') for line in open(a.reps)}
    keep = 0
    with open(a.out, 'w') as out:
        out.write('\t'.join(['ted_id', 'chain', 'chopping', 'n_strand', 'n_helix', 'plddt',
                             'cath_label', 'tax_lineage']) + '\n')
        for i, line in enumerate(sys.stdin):
            if a.max_lines and i >= a.max_lines:
                break
            f = line.rstrip('\n').split('\t')
            if len(f) < 21 or not f[9].isdigit():
                continue
            if int(f[9]) < 3 or float(f[6]) < 70:
                continue
            ch = chain_of(f[0])
            if reps is not None and ch not in reps:
                continue
            out.write('\t'.join([f[0], ch, f[3], f[9], f[8], f[6], f[13], f[20]]) + '\n')
            keep += 1
    print(f'kept {keep} domains', file=sys.stderr)


def fetch_api(chain):
    """PDB text for an AFDB chain id like AF-P12345-F1-model_v4 from the AFDB API."""
    acc = chain.split('-')[1]
    meta = json.load(urllib.request.urlopen(f'https://alphafold.ebi.ac.uk/api/prediction/{acc}', timeout=60))
    return urllib.request.urlopen(meta[0]['pdbUrl'], timeout=60).read().decode()


def call_api(row):
    try:
        text = fetch_api(row['chain'])
    except Exception as e:
        return [dict(row, ok=False, error=repr(e)[:200])]
    return [call_domain(row, text)]


def call_foldcomp_chunk(args):
    """Decompress one chunk of representative chains and call all their domains.

    foldcomp yields entries in the order of `ids`, so each entry is paired with its id and checked
    against the accession in its TITLE record before any domain is called.
    """
    import foldcomp
    db, ids, by_chain = args
    out = []
    with foldcomp.open(db, ids=ids) as handle:
        for dbname, (_, pdb_text) in zip(ids, handle):
            ch = dbname.replace('.pdb', '')
            # the title is the entry file name in afdb_rep_v4 and "(ACC)" in proteome databases
            title = ''.join(l[10:80] for l in pdb_text[:1000].splitlines() if l.startswith('TITLE')).replace(' ', '')
            if not (title.startswith(ch + '.') or f"({ch.split('-')[1]})" in title):
                out.extend(dict(r, ok=False, error='title/accession mismatch') for r in by_chain[ch])
                continue
            out.extend(call_domain(r, pdb_text) for r in by_chain[ch])
    return out


def cmd_call(a):
    import csv
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)['ted_id'] for l in open(a.out)}
    rows = [r for r in csv.DictReader(open(a.cand), delimiter='\t') if r['ted_id'] not in done]
    if a.limit:
        rows = rows[:a.limit]
    print(f'{len(rows)} domains to call ({len(done)} done)', file=sys.stderr)
    if a.source == 'foldcomp':
        in_db = {line.split('\t')[1].strip() for line in open(a.db + '.lookup')}
        by_chain, missing = {}, []
        for r in rows:
            name = next((x for x in (r['chain'], r['chain'] + '.pdb') if x in in_db), None)
            if name is None:
                missing.append(dict(r, ok=False, error='not in foldcomp db'))
            else:
                by_chain.setdefault(name, []).append(r)
        names = sorted(by_chain)
        tasks = [(a.db, names[i:i + a.chunk], {c.replace('.pdb', ''): by_chain[c] for c in names[i:i + a.chunk]})
                 for i in range(0, len(names), a.chunk)]
        if missing:
            with open(a.out, 'a') as out:
                out.writelines(json.dumps(m) + '\n' for m in missing)
        fn = call_foldcomp_chunk
    else:
        tasks, fn = rows, call_api
    n = 0
    with open(a.out, 'a') as out, Pool(a.workers) as p:
        for res in p.imap_unordered(fn, tasks):
            for r in res:
                out.write(json.dumps(r) + '\n')
            n += len(res)
            out.flush()
            print(f'{n}/{len(rows)}', file=sys.stderr, flush=True)


def cmd_summarise(a):
    import pandas as pd
    recs = [json.loads(l) for l in open(a.calls)]
    ok = [r for r in recs if r['ok']]
    rows = []
    for r in ok:
        agree = set.intersection(*(set(r['per_mode'][m]) for m in ('strict', 'lenient', 'lenient_noterm')))
        for code in r['topology_set']:
            rows.append(dict(code=code, n=len(code) // 2,
                             ted_id=r['ted_id'], agree=code in agree,
                             strict=code in r['per_mode']['strict'], lenient=code in r['per_mode']['lenient'],
                             cath=r['cath_label'], superkingdom=(r['tax_lineage'].split(',') + ['', ''])[1]))
    df = pd.DataFrame(rows)
    out = Path(a.out)
    df.to_csv(out / 'domain_topologies.csv', index=False)
    agg = df.groupby('code').agg(domains_in_set=('ted_id', 'nunique'),
                                 domains_agree=('agree', 'sum'),
                                 domains_strict=('strict', 'sum'),
                                 domains_lenient=('lenient', 'sum'),
                                 cath_superfamilies=('cath', lambda s: s[s != '-'].nunique()),
                                 superkingdoms=('superkingdom', 'nunique')).reset_index()
    agg.to_csv(out / 'topology_counts.csv', index=False)
    print(f'{len(recs)} domains, {len(ok)} called, {df.ted_id.nunique()} with a 3-7-strand open sheet')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('filter'); f.add_argument('--out', required=True); f.add_argument('--reps')
    f.add_argument('--max-lines', type=int, default=0)
    c = sub.add_parser('call'); c.add_argument('--cand', required=True); c.add_argument('--out', required=True)
    c.add_argument('--source', choices=['foldcomp', 'api'], default='foldcomp'); c.add_argument('--db')
    c.add_argument('--workers', type=int, default=2); c.add_argument('--limit', type=int, default=0)
    c.add_argument('--chunk', type=int, default=500, help='representative chains per foldcomp task')
    s = sub.add_parser('summarise'); s.add_argument('--calls', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args()
    {'filter': cmd_filter, 'call': cmd_call, 'summarise': cmd_summarise}[a.cmd](a)


if __name__ == '__main__':
    main()
