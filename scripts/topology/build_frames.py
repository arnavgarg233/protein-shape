"""Enumerated topology frames data/frame{n}.csv: one row per n-strand topology with its descriptors,
the Minami rule flags, ECOD occupancy from the Minami et al. Source Data and, for pure parallel sheets,
the Murata clash-free table. Needs the external files under data/external/ (see data/manifest.json).

    python scripts/topology/build_frames.py [--sizes 4 5 6] [--external-dir DIR] [--out-dir DIR]
    python scripts/topology/build_frames.py --sizes 7
"""
import argparse
from pathlib import Path

import pandas as pd

from protein_shape import minami_data as MD
from protein_shape import topology as T
from protein_shape.minami_data import THRESH, load_sheets, nf_labels
from protein_shape.paths import DATA


def build_frame(n, summ, dom, nf, murata):
    sheets = dom[dom.n == n].copy()
    sheets['H'] = sheets.fid.str.split('.').str[:2].str.join('.')
    g = sheets.groupby('topology')
    occ = pd.DataFrame({'n_sheets': g.size(), 'n_domains': g.domain.nunique(),
                        'n_Hgroups': g.H.nunique(), 'n_Fgroups': g.fid.nunique(),
                        'freq_recomputed': g.w.sum()})
    nf_by_topology = {v: k for k, v in nf.items()}
    rows = []
    for r in summ[n].itertuples():
        order, signs = T.parse(r.topology)
        row = {'topology': r.topology, 'koga_code': T.koga(order, signs),
               'strand_order': '|'.join(map(str, order)), 'orientation': ''.join(signs),
               'orient_by_strand': ''.join(signs[order.index(i)] for i in range(1, n + 1))}
        row.update(T.descriptors(order, signs))
        row.update({'jp': r.jp, 'ov': r.ov, 'en': r.en, 'violations': r.jp + r.ov + r.en,
                    'frustrated': int(r.jp + r.ov + r.en > 0)})
        row.update(T.minami_rule_counts(order, signs))
        row.update({'freq': r.freq, 'cls': r.cls, 'observed': int(r.freq >= THRESH)})
        o = occ.loc[r.topology] if r.topology in occ.index else None
        for k in ['n_sheets', 'n_domains', 'n_Hgroups', 'n_Fgroups']:
            row[k] = int(o[k]) if o is not None else 0
        row['freq_recomputed'] = float(o.freq_recomputed) if o is not None else 0.0
        row['nf_label'] = nf_by_topology.get(r.topology, 'NA') if n == 4 else 'NA'
        row['category'] = ('frustrated' if row['frustrated'] else 'FF') + \
                          ('_observed' if row['observed'] else '_unobserved')
        m = murata.get(''.join(map(str, order))) if set(signs) == {'+'} else None
        row['murata_clash_free'] = ('NA' if set(signs) != {'+'} else int(m is not None))
        for k, col in [('murata_OFHT', 'OFHT'), ('murata_Nj', 'number_of_jumps'),
                       ('murata_NFabP', 'number_of_FabP')]:
            row[k] = ('NA' if m is None else float(m[col]) if k == 'murata_OFHT' else int(m[col]))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sizes', type=int, nargs='+', default=[4, 5, 6])
    ap.add_argument('--external-dir', default=str(MD.EXT), help='holds minami/, koga/ and murata_repo/')
    ap.add_argument('--out-dir', default=str(DATA))
    a = ap.parse_args()
    MD.EXT = Path(a.external_dir)
    MD.KOGA = MD.EXT / 'koga' / 'demos_src' / 'demos_explore_novel_abfolds'
    seven = 7 in a.sizes
    summ, dom = load_sheets(max(a.sizes))
    nf = nf_labels()
    cf = pd.read_csv(MD.EXT / 'murata_repo' / 'clash_free_topologies.csv', index_col=0,
                     dtype={'topology': str}).set_index('topology')
    murata = {t: r for t, r in cf.iterrows()}
    for n in a.sizes:
        df = build_frame(n, summ, dom, nf, murata)
        assert (df.freq - df.freq_recomputed).abs().max() < 5.1e-4
        out = Path(a.out_dir) / f'frame{n}.csv'
        df.to_csv(out, index=False, na_rep='NA')
        print(f'{out}: {len(df)} rows, {df.shape[1]} columns')
        if seven:
            continue
        print(df.category.value_counts().to_dict())
        if n == 4:
            print('NF rows:', df.nf_label.ne('NA').sum(),
                  'frustrated', int(df.frustrated.sum()), 'FF', int((1 - df.frustrated).sum()))


if __name__ == '__main__':
    main()
