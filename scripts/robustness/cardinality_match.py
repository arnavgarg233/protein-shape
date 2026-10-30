"""Cardinality matching (Zubizarreta 2014): the largest set of exposed-reference pairs whose covariates
are balanced to a stated standardised difference, solved as a mixed-integer program.

For covariate k with pooled sd s_k and tolerance d, |mean_exposed - mean_control| <= d * s_k over the
matched set is |sum_ij x_ij (v_k(i) - v_k(j))| <= d * s_k * sum_ij x_ij, two linear rows in x.

Run: python scripts/robustness/cardinality_match.py [tolerance] [controls_per_exposed]
"""
import argparse

import numpy as np
from protein_shape import composition as C
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

TOL, KNN = 0.10, 30


def frame():
    df = C.build()
    df['used'] = (df.L_H >= 1).astype(int)
    df['cnt'] = df.L_H.astype(float)
    return df


def propensity(df, fitm, expm, feats, seed=C.SEED):
    """Logistic propensity for membership of the exposed arm, out of fold by 10-fold grouped CV."""
    sub = np.flatnonzero(fitm | expm)
    X = df.loc[df.index[sub], feats].to_numpy(float)
    t = expm[sub].astype(int)
    g = df.rev_group.to_numpy()[sub]
    oof = np.full(len(t), np.nan)
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, t, g):
        oof[te] = C.make(1.0).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
    full = C.make(1.0).fit(X, t)
    ins = full.predict_proba(X)[:, 1]
    return dict(sub=sub, t=t, oof=oof, insample=ins, model=full,
                auc_oof=float(roc_auc_score(t, oof)), auc_insample=float(roc_auc_score(t, ins)))


def main():
    global TOL, KNN
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('tolerance', nargs='?', type=float, default=TOL, help='maximum |standardised difference|')
    ap.add_argument('controls_per_exposed', nargs='?', type=int, default=KNN, help='candidate controls per exposed row')
    a = ap.parse_args()
    TOL, KNN = a.tolerance, a.controls_per_exposed
    df = frame()
    feats = list(C.FEATURES_A)
    pr = propensity(df, (df.OPP.to_numpy() == 0), C.exposed_mask(df), feats)
    sub, t, oof = pr['sub'], pr['t'], pr['oof']
    lg = np.log(np.clip(oof, 1e-9, 1 - 1e-9)) - np.log(np.clip(1 - oof, 1e-9, 1 - 1e-9))
    rows_e, rows_c = sub[t == 1], sub[t == 0]
    V = df[feats].to_numpy(float)
    Ve, Vc = V[rows_e], V[rows_c]
    sd = np.sqrt((Ve.var(0, ddof=1) + Vc.var(0, ddof=1)) / 2)
    sd[sd == 0] = 1.0

    # Candidate pairs: the KNN nearest controls per exposed row on the propensity logit. Restricting
    # candidates can only lower the achievable count, so the result is a lower bound.
    nn = np.argsort(np.abs(lg[t == 1][:, None] - lg[t == 0][None, :]), axis=1)[:, :KNN]
    ei = np.repeat(np.arange(len(rows_e)), KNN)
    ci = nn.ravel()
    n = len(ei)
    print(f'exposed {len(rows_e)}  controls {len(rows_c)}  candidate pairs {n:,}  tol {TOL}  knn {KNN}')

    cons = []
    # each exposed row used at most once
    cons.append(LinearConstraint(coo_matrix((np.ones(n), (ei, np.arange(n))),
                                            shape=(len(rows_e), n)), -np.inf, 1))
    # each control used at most once
    uc, inv = np.unique(ci, return_inverse=True)
    cons.append(LinearConstraint(coo_matrix((np.ones(n), (inv, np.arange(n))),
                                            shape=(len(uc), n)), -np.inf, 1))
    # balance: |sum x (ve - vc)| <= TOL * sd * sum x, as two linear rows per covariate
    for k, f in enumerate(feats):
        d = Ve[ei, k] - Vc[ci, k]
        cons.append(LinearConstraint(coo_matrix(d - TOL * sd[k]), -np.inf, 0))
        cons.append(LinearConstraint(coo_matrix(-d - TOL * sd[k]), -np.inf, 0))

    res = milp(c=-np.ones(n), constraints=cons, integrality=np.ones(n),
               bounds=Bounds(0, 1), options=dict(time_limit=600, mip_rel_gap=0.0))
    if not res.success:
        print('solver did not finish cleanly:', res.message)
        if res.x is None:
            return
    x = np.round(res.x).astype(bool)
    pe, pc = rows_e[ei[x]], rows_c[ci[x]]
    print(f'\nbalanced matched pairs at |SMD| <= {TOL}: {x.sum()} of {len(rows_e)}')
    worst = 0.0
    for k, f in enumerate(feats):
        s = (V[pe, k].mean() - V[pc, k].mean()) / sd[k]
        worst = max(worst, abs(s))
        if abs(s) > TOL + 1e-9:
            print(f'   violated {f} {s:+.3f}')
    print(f'worst |SMD| on the matched set: {worst:.3f}  ({len(feats)} covariates)')

    y, L = df.used.to_numpy(), df.L_H.to_numpy()
    ye, yc = int(y[pe].sum()), int(y[pc].sum())
    tot = ye + yc
    print(f'\nexposed used {ye}/{len(pe)}   control used {yc}/{len(pc)}')
    print(f'graded L_H: exposed {int(L[pe].sum())}  control {int(L[pc].sum())}')
    if tot:
        p = stats.binomtest(ye, tot, 0.5, alternative='less').pvalue
        floor = stats.binomtest(0, tot, 0.5, alternative='less').pvalue
        print(f'binary rate ratio {ye / max(yc, 1):.3f}   exact conditional p {p:.4f}   floor {floor:.4f}')
        disc_e = int(((y[pe] == 1) & (y[pc] == 0)).sum())
        disc_c = int(((y[pe] == 0) & (y[pc] == 1)).sum())
        if disc_e + disc_c:
            pm = stats.binomtest(disc_e, disc_e + disc_c, 0.5, alternative='less').pvalue
            print(f'discordant pairs {disc_e + disc_c} ({disc_e} exposed-only, {disc_c} control-only), '
                  f'McNemar p {pm:.4f}')


if __name__ == '__main__':
    main()
