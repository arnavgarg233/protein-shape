"""Power of the ferredoxin-domain folding contrast for the selected proteins, by simulation of the protein-level model
log10 RMST = line in RCO and log10 L + delta (OPP) + between-protein scatter + within-protein sampling error, over
a grid of the two variance components; reports the smallest delta with 80% power.

    python scripts/folding/fd1_power.py --dir DIR      (reads DIR/selection.csv, writes DIR/power.csv)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def contrast(X_ref, y_ref, X_t, y_t):
    """Mean residual of target proteins about the REF-fitted line; t interval half-width uses REF residual sd."""
    beta, *_ = np.linalg.lstsq(X_ref, y_ref, rcond=None)
    r = y_ref - X_ref @ beta
    df = len(y_ref) - X_ref.shape[1]
    s = np.sqrt((r ** 2).sum() / df)
    xbar = X_t.mean(0)
    XtXi = np.linalg.inv(X_ref.T @ X_ref)
    se = s * np.sqrt(1 / len(y_t) + xbar @ XtXi @ xbar)
    est = (y_t - X_t @ beta).mean()
    return est, se, df


def design(d):
    return np.c_[np.ones(len(d)), d.RCO.to_numpy(), np.log10(d.L.to_numpy())]


def power(Xr, Xo, sd_b, se_w, delta, nsim=4000, rng=None):
    hit = 0
    for _ in range(nsim):
        yr = rng.normal(0, sd_b, len(Xr)) + rng.normal(0, se_w, len(Xr))
        yo = delta + rng.normal(0, sd_b, len(Xo)) + rng.normal(0, se_w, len(Xo))
        est, se, df = contrast(Xr, yr, Xo, yo)
        hit += est - stats.t.ppf(0.975, df) * se > 0
    return hit / nsim


def main(D):
    rng = np.random.default_rng(122)
    s = pd.read_csv(D / 'selection.csv')
    Xr, Xo = design(s[s.cls == 'REF']), design(s[s.cls == 'OPP'])
    Xn = design(s[s.cls == 'NEST'])
    print(f'nOPP {len(Xo)}  nNEST {len(Xn)}  nREF {len(Xr)}')
    grid = np.round(np.arange(0.02, 0.81, 0.02), 2)
    rows = []
    for sd_b in (0.10, 0.152, 0.20, 0.279, 0.35):
        for se_w in (0.064, 0.10):
            for tgt, Xt in (('OPP vs REF', Xo), ('NEST vs REF', Xn)):
                mde = None
                for dlt in grid:
                    if power(Xr, Xt, sd_b, se_w, dlt, 2000, rng) >= 0.80:
                        mde = dlt
                        break
                rows.append(dict(contrast=tgt, sd_b=sd_b, se_w=se_w, MDE_log10=mde,
                                 MDE_ratio=None if mde is None else round(10 ** mde, 2)))
    out = pd.DataFrame(rows)
    out.to_csv(D / 'power.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dir', default='results/folding/fd1')
    main(Path(ap.parse_args().dir))
