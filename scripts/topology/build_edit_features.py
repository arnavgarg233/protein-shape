"""Edit-graph features of four- to seven-strand topologies (parents, circular permutations, flips and the
occupancy of their parents), written to data/edit_features{n}.csv. Each size reads occupancy of smaller
sheets only.

    python scripts/topology/build_edit_features.py [--dir DIR] [--external-dir DIR]        # 4, 5 and 6 strands
    python scripts/topology/build_edit_features.py --seven [--dir DIR]                     # 7 strands

DIR (default data/) holds the frame{n}.csv inputs and receives edit_features{n}.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

from protein_shape import edits as E
from protein_shape import minami_data as MD
from protein_shape.minami_data import THRESH, load_sheets
from protein_shape.paths import DATA


def build(n, g, lower_freq, same_level_freq=None):
    obs_lower = [t for t, f in lower_freq.items() if f >= THRESH]
    d_all = g.distances(obs_lower)
    d_nf = g.distances(obs_lower, types=('terminal_N', 'terminal_C', 'cp'))
    rows = []
    for t in g.nodes[n]:
        par = g.parents[t]
        pf = [lower_freq[p] for p in par]
        cp = sorted({u for et, u in E.circular_permutations(t) if u != t})
        fl = sorted({u for et, u in E.flips(t) if u != t})
        r = {'topology': t, 'parents': ';'.join(par), 'n_parents': len(par), 'n_obs_parents': sum(f >= THRESH for f in pf),
             'parent_freq_sum': sum(pf), 'parent_freq_max': max(pf) if pf else 0.0,
             'parent_freq_mean': sum(pf) / len(pf) if pf else 0.0,
             'n_del_ops': g.n_del_ops[t], 'n_splitting': g.n_split[t],
             'n_children': len(g.children[t]), 'n_cp_neighbours': len(cp),
             'n_flip_neighbours': len(fl),
             'in_degree': len(par) + len(cp) + len(fl), 'out_degree': len(g.children[t]) + len(cp) + len(fl),
             'reachable_from_obs_parent': int(any(f >= THRESH for f in pf)),
             'dist_to_obs_lower_all': d_all.get(t, -1), 'dist_to_obs_lower_noflip': d_nf.get(t, -1)}
        if same_level_freq is not None:
            cf = [same_level_freq[u] for u in cp]
            r.update({'cp_freq_sum': sum(cf), 'cp_freq_mean': sum(cf) / len(cf) if cf else 0.0,
                      'cp_freq_max': max(cf) if cf else 0.0, 'n_cp_obs': sum(f >= THRESH for f in cf)})
        rows.append(r)
    return pd.DataFrame(rows)


def main_7():
    """edit_features7.csv: parents are six-strand topologies with occupancy from frame6.csv only."""
    f6 = pd.read_csv(DATA / 'frame6.csv').set_index('topology').freq.to_dict()
    g = E.EditGraph(3, 7)
    e7 = build(7, g, f6)
    e7.to_csv(DATA / 'edit_features7.csv', index=False)
    print('edit_features7.csv', e7.shape)


def main():
    summ, _ = load_sheets()
    f3 = dict(zip(summ[3].topology, summ[3].freq))
    f4 = pd.read_csv(DATA / 'frame4.csv').set_index('topology').freq.to_dict()
    g = E.EditGraph(3, 6)
    e4 = build(4, g, f3, same_level_freq=f4)
    e5 = build(5, g, f4)
    f5 = pd.read_csv(DATA / 'frame5.csv').set_index('topology').freq.to_dict()   # parent occupancy only
    e6 = build(6, g, f5)
    e4.to_csv(DATA / 'edit_features4.csv', index=False)
    e5.to_csv(DATA / 'edit_features5.csv', index=False)
    e6.to_csv(DATA / 'edit_features6.csv', index=False)
    print(e4.describe().T.to_string())
    print(e5.describe().T.to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seven', action='store_true')
    ap.add_argument('--dir', default=str(DATA))
    ap.add_argument('--external-dir', default=str(MD.EXT), help='holds minami/fig3.xlsx')
    a = ap.parse_args()
    DATA, MD.EXT = Path(a.dir), Path(a.external_dir)
    main_7() if a.seven else main()
