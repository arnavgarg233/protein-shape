"""Select natural ferredoxin-type domains for the folding test with a fixed rule: one domain per ECOD H-group and
class (OPP pair, nested opposite-face pair, reference), 60 to 160 residues, then nearest-neighbour matching on
length and strand count (2 nested and 4 reference domains per OPP domain).

    python scripts/folding/fd1_select.py --geom DIR --rcsb DIR --out DIR

--geom holds sheets.csv and domain_status.csv (ECOD develop210 F99 sheets called in lenient mode), --rcsb holds
PDBID.cif.gz files. Writes candidates.csv, selection.csv, summary.json and structures/NAME.pdb under --out.
"""
import argparse
import json
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

from protein_shape import s2x
from protein_shape import structure as T
from protein_shape.folding import entanglement as E
from protein_shape.frames import both_orientations, pair_faces
from protein_shape.paths import OCC_MAIN

NSTR = (5, 6, 7)   # census strand counts


def feats(t):
    f = s2x.features(t)
    npar, same, opp = pair_faces(t)
    return dict(C2z=f['C2z_crossing'], S2xn=f['S2x_nested'], npar=npar, SAME=same, OPP=opp, both=both_orientations(t))


def frame_violations():
    f = pd.read_csv(OCC_MAIN, usecols=['topology', 'n', 'violations'])
    f = f[f.n.isin(NSTR)]
    return dict(zip(f.topology, f.violations))


def cut(row, RCSB):
    """Cut, trim, check; returns dict of properties and the trimmed gemmi structure (chain A, 1..L)."""
    path = RCSB / f'{row.pdb.upper()}.cif.gz'
    full = gemmi.read_structure(str(path))
    res_A = full.resolution if full.resolution and full.resolution > 0 else np.inf
    method = gemmi.cif.read(str(path)).sole_block().find_value('_exptl.method') or ''
    st, labels = T.select_residues(path, None, T.parse_range(row.range), 1)
    for r in st[0][0]:
        if r.name == 'MSE':
            r.name, r.het_flag = 'MET', 'A'
            for a in r:
                if a.name == 'SE':
                    a.name, a.element = 'SD', gemmi.Element('S')
    ss = T.parse_dssp(T.run_dssp(st))['ss']
    if len(ss) != len(st[0][0]):
        return dict(ok=False, why='dssp_len'), None
    idx = [i for i, s in enumerate(ss) if s in 'HGIE']
    a, b = idx[0], idx[-1]
    ch = st[0][0]
    for i in range(len(ch) - 1, b, -1):
        del ch[i]
    for i in range(a - 1, -1, -1):
        del ch[i]
    for k, r in enumerate(ch):
        r.seqid = gemmi.SeqId(k + 1, ' ')
    ca = np.array([r.find_atom('CA', '*').pos.tolist() for r in ch])
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
    L = len(ca)
    st.setup_entities()
    return dict(ok=True, resolution=res_A, method=method, L=L, trim_n=a, trim_c=len(ss) - 1 - b,
                max_ca_step=float(d.max()), chain_break=bool((d > 4.2).any()), first_label=labels[a],
                last_label=labels[b]), st


def contact_stats(pdbfile, L):
    labels, ca, heavy, names, sg = E.load_residues(pdbfile, 'A', f'A:1-{L}', 1)
    ij = np.array(E.native_contacts(heavy, 4, 4.5))
    sep = ij[:, 1] - ij[:, 0]
    return dict(n_contacts=len(ij), contacts_per_res=len(ij) / L, RCO=float(sep.mean() / L), ACO=float(sep.mean()))


def main(GEOM, RCSB, OUT):
    STRUCT = OUT / 'structures'
    OUT.mkdir(parents=True, exist_ok=True)
    STRUCT.mkdir(exist_ok=True)
    sh = pd.read_csv(GEOM / 'sheets.csv', dtype={'ecod_h': str, 'ecod_fid': str})
    ds = pd.read_csv(GEOM / 'domain_status.csv')[['domain', 'range', 'nres', 'n_sheets']]
    F = pd.DataFrame([feats(t) for t in sh.topology])
    sh = pd.concat([sh, F], axis=1).merge(ds, on='domain', how='left')
    sh = sh[(sh.C2z >= 1) & (sh.npar >= 2) & sh.n_strands.isin(NSTR)].copy()
    sh['cls'] = np.select([(sh.OPP >= 1) & (sh.SAME == 0),
                           (sh.S2xn >= 1) & (sh.OPP == 0) & (sh.SAME == 0),
                           (sh.OPP == 0) & (sh.SAME == 0) & sh.both], ['OPP', 'NEST', 'REF'], 'other')
    sh = sh[sh.cls != 'other']
    nsheet = pd.read_csv(GEOM / 'sheets.csv', usecols=['domain']).domain.value_counts()
    sh['sheets_in_domain'] = sh.domain.map(nsheet)
    viol = frame_violations()
    sh['violations'] = sh.topology.map(viol)
    sh['contiguous'] = ~sh.range.str.contains(',')
    pre = sh[(sh.sheets_in_domain == 1) & sh.contiguous & (sh.nres >= 55) & (sh.nres <= 175)].copy()
    rows = []
    for r in pre.itertuples():
        try:
            info, st = cut(r, RCSB)
        except Exception as e:  # noqa: BLE001
            info, st = dict(ok=False, why=type(e).__name__ + str(e)[:60]), None
        rec = dict(domain=r.domain, pdb=r.pdb, ecod_h=str(r.ecod_h), arch=r.arch, cls=r.cls, topology=r.topology,
                   n_strands=r.n_strands, npar=r.npar, OPP=r.OPP, S2x_nested=r.S2xn, C2z=r.C2z, weak_strand=r.weak_strand,
                   violations=r.violations, range=r.range, nres_ecod=r.nres, **info)
        if info.get('ok'):
            f = STRUCT / f'{r.domain}.pdb'
            st.write_pdb(str(f))
            rec.update(contact_stats(f, info['L']))
        rows.append(rec)
    c = pd.DataFrame(rows)
    c['eligible'] = (c.ok & (c.L >= 60) & (c.L <= 160) & ~c.chain_break.fillna(True).astype(bool)
                     & (c.violations == 0) & ~((c.resolution > 4.0) & np.isfinite(c.resolution)))   # NMR and entries without a resolution are allowed
    c.to_csv(OUT / 'candidates.csv', index=False)
    e = c[c.eligible].copy()
    medL = e[e.cls == 'OPP'].L.median()
    e['dmed'] = (e.L - medL).abs()
    reps = (e.sort_values(['resolution', 'dmed', 'domain']).groupby(['cls', 'ecod_h'], sort=False).head(1))
    opp = reps[reps.cls == 'OPP'].sort_values('domain')
    sel = [dict(r._asdict(), match_to='') for r in opp.itertuples(index=False)]
    for cls, k in (('NEST', 2), ('REF', 4)):
        pool = reps[reps.cls == cls].copy()
        for _ in range(k):
            for o in opp.itertuples():
                if pool.empty:
                    break
                d = (pool.L - o.L).abs() / 20 + (pool.n_strands - o.n_strands).abs()
                j = d.sort_values(kind='stable').index[0]
                sel.append(dict(pool.loc[j].to_dict(), match_to=o.domain))
                pool = pool.drop(j)
    s = pd.DataFrame(sel)
    s['name'] = s.domain
    s.to_csv(OUT / 'selection.csv', index=False)
    summ = {r.name: dict(pdb=r.name, nres=int(r.L), ss=[['E', 1, int(r.L)]]) for r in s.itertuples()}
    json.dump(summ, open(OUT / 'summary.json', 'w'), indent=1)
    print('candidates by class (eligible / all):')
    print(c.groupby('cls').agg(all=('domain', 'size'), eligible=('eligible', 'sum'),
                               H_eligible=('ecod_h', lambda x: x[c.loc[x.index, 'eligible']].nunique())).to_string())
    print(s[['cls', 'name', 'ecod_h', 'L', 'n_strands', 'resolution', 'RCO', 'contacts_per_res', 'match_to']].to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--geom', required=True, help='directory with sheets.csv and domain_status.csv')
    ap.add_argument('--rcsb', required=True, help='directory with PDBID.cif.gz files')
    ap.add_argument('--out', default='results/folding/fd1')
    a = ap.parse_args()
    main(Path(a.geom), Path(a.rcsb), Path(a.out))
