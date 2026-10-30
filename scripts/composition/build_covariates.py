"""Rebuild data/composition_covariates.csv.gz: for each of the 2,961 frustration-free five- to seven-strand
topologies with at least two parallel crossovers, the sheet descriptors of the composition model and the
four-strand geometry score M2.

    python scripts/composition/build_covariates.py [--input-dir DIR] [--external-dir DIR] [--out OUT.csv.gz]
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from protein_shape import geometry_score as H
from protein_shape import minami_data
from protein_shape.paths import COMPOSITION_COVARIATES, DATA, OCC_MAIN

SIZES = (5, 6, 7)
COLS = ['topology', 'n', 'sum_jump', 'max_jump', 'n_jumps', 'rco_sheet', 'par_frac', 'M2']


def m2_features(n):
    """The M2 columns of geometry_score.features(n) for frustration-free topologies only (faster at 7)."""
    f = pd.read_csv(H.INPUT_DIR / f'frame{n}.csv')
    f = f[f.violations == 0]
    pairs = f.n_parallel_pairs + f.n_antiparallel_pairs
    return pd.DataFrame({'topology': f.topology, 'jp': f.jp, 'ov': f.ov, 'en': f.en,
                         'all_anti': (f.n_parallel_pairs == 0).astype(int),
                         'par_frac': f.n_parallel_pairs / pairs.where(pairs > 0, 1),
                         'jump_per_strand': f.sum_jump / n, 'rco': f.rco_sheet,
                         'sum_jump': f.sum_jump, 'rco_sheet': f.rco_sheet})


def m2_scores():
    X4, y4 = H.features(4)
    cont4, obs4 = np.log1p(y4.n_Hgroups), (y4.freq >= 0.25).astype(int)
    cols = H.MODELS['M2_rules_geom']
    out = []
    for n in SIZES:
        Xn = m2_features(n)
        out.append(pd.DataFrame({'topology': Xn.topology, 'n': n, 'M2': H.fit_predict(X4, cont4, obs4, Xn, cols),
                                 'sum_jump': Xn.sum_jump, 'rco_sheet': Xn.rco_sheet}))
    X5, _ = H.features(5)  # the shortcut must agree with the full feature code
    full = pd.Series(H.fit_predict(X4, cont4, obs4, X5, cols), index=X5.topology)
    assert np.allclose(full[out[0].topology].to_numpy(), out[0].M2.to_numpy())
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input-dir', default=str(DATA), help='frame, edit-feature and grammar-feature tables')
    ap.add_argument('--external-dir', default=str(DATA / 'external'), help='holds minami/fig3.xlsx')
    ap.add_argument('--occupancy', default=str(OCC_MAIN))
    ap.add_argument('--out', default=str(COMPOSITION_COVARIATES))
    a = ap.parse_args()
    H.INPUT_DIR, minami_data.EXT = Path(a.input_dir), Path(a.external_dir)
    o = pd.read_csv(a.occupancy)
    df = o[(o.violations == 0) & (o.npar >= 2)].reset_index(drop=True)
    frames = []
    for n in SIZES:
        fr = pd.read_csv(H.INPUT_DIR / f'frame{n}.csv',
                         usecols=['topology', 'n_parallel_pairs', 'n_antiparallel_pairs',
                                  'sum_jump', 'max_jump', 'n_jumps', 'rco_sheet'])
        pr = fr.n_parallel_pairs + fr.n_antiparallel_pairs
        fr['par_frac'] = fr.n_parallel_pairs / pr.where(pr > 0, 1)
        frames.append(fr.drop(columns=['n_parallel_pairs', 'n_antiparallel_pairs']))
    df = df.merge(pd.concat(frames, ignore_index=True), on='topology', how='left', validate='1:1')
    df = df.merge(m2_scores()[['topology', 'n', 'M2']], on=['topology', 'n'], how='left', validate='1:1')
    assert len(df) == 2961 and df[COLS].notna().all().all()
    with open(a.out, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as fh:
        fh.write(df[COLS].to_csv(index=False).encode())
    print(f'{a.out}: {len(df)} rows')


if __name__ == '__main__':
    main()
