"""Score five- to seven-strand topologies against census occupancy (used by at least one superfamily),
beside ECOD occupancy, with the models fitted on four strands; also among ECOD-unobserved topologies.

    python scripts/composition/score_census.py [--input-dir DIR] [--external-dir DIR] [--occupancy CSV] [--out OUT.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from protein_shape import geometry_score as H
from protein_shape import minami_data
from protein_shape.paths import DATA, OCC_MAIN, RESULTS

SCORED = ('M1_rules', 'M2_rules_geom', 'M6_plus_pp_cross_same')


def auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input-dir', default=str(DATA), help='frame, edit-feature and grammar-feature tables')
    ap.add_argument('--external-dir', default=str(DATA / 'external'), help='holds minami/fig3.xlsx')
    ap.add_argument('--occupancy', default=str(OCC_MAIN))
    ap.add_argument('--out', default=str(RESULTS / 'composition' / 'score_census.json'))
    a = ap.parse_args()
    H.INPUT_DIR, minami_data.EXT = Path(a.input_dir), Path(a.external_dir)
    occ = pd.read_csv(a.occupancy)
    X4, y4 = H.features(4)
    cont4, obs4 = np.log1p(y4.n_Hgroups), (y4.freq >= 0.25).astype(int)
    out = {}
    for n in (5, 6, 7):
        Xn, yn = H.features(n)
        o = Xn[['topology']].merge(occ[occ.n == n][['topology', 'L_H', 'hybrid']], on='topology', how='left').fillna(0)
        afdb = (o.L_H >= 1).astype(int).values
        ecod = (yn.freq >= 0.25).astype(int).values
        res = dict(n=int(len(o)), afdb_used=int(afdb.sum()), ecod_observed=int(ecod.sum()),
                   afdb_used_not_ecod=int(((afdb == 1) & (ecod == 0)).sum()))
        new = ecod == 0
        for name in SCORED:
            s = H.fit_predict(X4, cont4, obs4, Xn, H.MODELS[name])
            res[name] = dict(auroc_afdb=auc(afdb, s), auroc_ecod=auc(ecod, s),
                             auroc_afdb_among_ecod_unobserved=auc(afdb[new], s[new]))
        out[n] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1, default=lambda v: round(float(v), 3)))


if __name__ == '__main__':
    main()
