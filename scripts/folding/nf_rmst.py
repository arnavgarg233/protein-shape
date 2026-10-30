"""Restricted mean folding time (cap 3 M steps) of NF1 relative to the other designs, with trajectory-bootstrap 95%
intervals, and residuals of log10 RMST about lines in RCO and chain length over the six designs without a knot.

    python scripts/folding/nf_rmst.py [--dir DIR] [--tag rel0.92]

DIR (default data/folding/nf) holds design_info.csv and summary_NF*_TAG.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from protein_shape.folding.survival import km_rmst
from protein_shape.paths import DATA

CAP = 3000000
SIX = ['NF1', 'NF7', 'NF2', 'NF4', 'NF5', 'NF6']


def fit_res(y, X):
    A = np.column_stack([np.ones(len(y)), X]); coef = np.linalg.lstsq(A, y, rcond=None)[0]; return y - A @ coef


def main(D, tag):
    info = pd.read_csv(D / 'design_info.csv').set_index('nf')
    rng = np.random.default_rng(0)
    data = {}
    for nf in SIX:
        s = pd.read_csv(D / f'summary_{nf}_{tag}.csv')
        data[nf] = (np.where(s.folded_ent, s.t_fold_ent, s.steps_run).astype(float), s.folded_ent.astype(int).values)
    rco = np.array([info.loc[n, 'RCO'] for n in SIX]); logL = np.log10([info.loc[n, 'L'] for n in SIX])
    out = {'RCO': [], 'logL': [], 'RCO+logL': [], 'RCO_ctrlfit': []}

    def stat(d):
        y = np.log10([km_rmst(*d[n], CAP) for n in SIX]); r = {}
        r['RCO'] = fit_res(y, rco[:, None])[:2]; r['logL'] = fit_res(y, logL[:, None])[:2]
        r['RCO+logL'] = fit_res(y, np.column_stack([rco, logL]))[:2]
        A = np.column_stack([np.ones(4), rco[2:]]); c = np.linalg.lstsq(A, y[2:], rcond=None)[0]
        r['RCO_ctrlfit'] = y[:2] - (c[0] + c[1] * rco[:2]); return r, y
    obs, yobs = stat(data)
    print('log10 RMST (first passage to an entanglement-correct fold, cap 3e6):', dict(zip(SIX, yobs.round(2))))
    for k in out:
        print(f'{k}: residuals NF1, NF7 = {obs[k].round(2)}')
    for b in range(2000):
        d = {}
        for n in SIX:
            t, e = data[n]; i = rng.integers(len(t), size=len(t)); d[n] = (t[i], e[i])
        r, _ = stat(d)
        for k in out:
            out[k].append(r[k])
    for k in out:
        a = np.array(out[k])
        print(f'{k}: bootstrap 95% CI residual NF1 [{np.percentile(a[:, 0], 2.5):.2f},{np.percentile(a[:, 0], 97.5):.2f}] '
              f'NF7 [{np.percentile(a[:, 1], 2.5):.2f},{np.percentile(a[:, 1], 97.5):.2f}]')
    # the ratio draws continue the same generator, so they depend on the residual bootstrap above
    print('\nRMST ratios (NF1 / X) with bootstrap 95% CI')
    res = {}
    for x in ['NF4', 'NF5', 'NF6', 'NF2', 'NF7']:
        rs = []
        for b in range(2000):
            i1 = rng.integers(20, size=20); i2 = rng.integers(20, size=20)
            rs.append(km_rmst(data['NF1'][0][i1], data['NF1'][1][i1], CAP) / km_rmst(data[x][0][i2], data[x][1][i2], CAP))
        r0 = km_rmst(*data['NF1'], CAP) / km_rmst(*data[x], CAP)
        res[x] = (r0, np.percentile(rs, 2.5), np.percentile(rs, 97.5))
        print(f'NF1/{x}: {r0:.2f} [{np.percentile(rs, 2.5):.2f},{np.percentile(rs, 97.5):.2f}]')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dir', default=str(DATA / 'folding' / 'nf'))
    ap.add_argument('--tag', default='rel0.92')
    a = ap.parse_args()
    main(Path(a.dir), a.tag)
