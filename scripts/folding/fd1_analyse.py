"""Folding test on natural ferredoxin-type domains: per-protein restricted mean folding times (cap 10 M steps), the
residual of log10 RMST about a line in RCO and log10 L fitted on reference proteins, and the OPP, nested and reference
contrasts with t intervals, exact permutation p values and a two-level bootstrap.

    python scripts/folding/fd1_analyse.py [--dir DIR] [--out DIR]

DIR (default data/folding/fd1) holds selection.csv and, per protein, remd_NAME.json, calib_NAME_rel0.92.json and
summary_NAME_rel0.92.csv; protein_table.csv and analysis.json go to --out (default results/folding/fd1).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from protein_shape import s2x
from protein_shape.paths import DATA, OCC_MAIN, RESULTS

TAG = 'rel0.92'
CAP = 10_000_000
CAP2 = 3_000_000
NTRAJ = 20


def rmst(t, cap):
    return float(np.minimum(np.nan_to_num(t, nan=np.inf), cap).mean())


def km_median(t, cap):
    """Kaplan-Meier median of first-passage times censored at cap (no ties issue: times are distinct step counts)."""
    t = np.sort(np.minimum(np.nan_to_num(t, nan=np.inf), cap))
    n = len(t)
    surv = 1.0
    for k, x in enumerate(t):
        if x >= cap:
            return np.inf
        surv *= 1 - 1 / (n - k)
        if surv <= 0.5:
            return float(x)
    return np.inf


def design(d):
    return np.c_[np.ones(len(d)), d.RCO.to_numpy(), np.log10(d.L.to_numpy())]


def ref_fit(d):
    X, y = design(d), d.y.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = len(y) - X.shape[1]
    s = np.sqrt(((y - X @ beta) ** 2).sum() / df)
    return beta, s, df, np.linalg.inv(X.T @ X)


def resid_contrast(ref, a, b=None):
    """Mean residual of class a (minus class b if given) about the REF line, with t interval (REF residual sd, df nREF-3)."""
    beta, s, df, XtXi = ref_fit(ref)
    ra = a.y.to_numpy() - design(a) @ beta
    if b is None:
        x = design(a).mean(0)
        est, var = ra.mean(), 1 / len(a) + x @ XtXi @ x
    else:
        rb = b.y.to_numpy() - design(b) @ beta
        x = design(a).mean(0) - design(b).mean(0)
        est, var = ra.mean() - rb.mean(), 1 / len(a) + 1 / len(b) + x @ XtXi @ x
    se = s * np.sqrt(var)
    q = stats.t.ppf(0.975, df)
    return dict(est_log10=est, se=se, df=df, lo=est - q * se, hi=est + q * se, ratio=10 ** est,
                ratio_lo=10 ** (est - q * se), ratio_hi=10 ** (est + q * se), p_t_one_sided_slower=float(1 - stats.t.cdf(est / se, df)))


def perm_p(pt, a_cls, b_cls):
    """Exact one-sided permutation p (a slower). OPP vs REF: relabel within OPP+REF, refit line on relabelled REF.
    a vs NEST: relabel within a+NEST about the fixed REF line."""
    ref = pt[pt.cls == 'REF']
    if b_cls == 'REF':
        pool = pt[pt.cls.isin([a_cls, 'REF'])].reset_index(drop=True)
        na = (pool.cls == a_cls).sum()
        obs = resid_contrast(ref, pool[pool.cls == a_cls])['est_log10']
        vals = []
        for idx in itertools.combinations(range(len(pool)), na):
            m = np.zeros(len(pool), bool)
            m[list(idx)] = True
            vals.append(resid_contrast(pool[~m], pool[m])['est_log10'])
    else:
        pool = pt[pt.cls.isin([a_cls, b_cls])].reset_index(drop=True)
        na = (pool.cls == a_cls).sum()
        beta = ref_fit(ref)[0]
        r = pool.y.to_numpy() - design(pool) @ beta
        obs = r[pool.cls == a_cls].mean() - r[pool.cls == b_cls].mean()
        vals = []
        for idx in itertools.combinations(range(len(pool)), na):
            m = np.zeros(len(pool), bool)
            m[list(idx)] = True
            vals.append(r[m].mean() - r[~m].mean())
    vals = np.array(vals)
    return float((vals >= obs - 1e-12).mean()), len(vals)


def boot_ci(pt, trajs, a_cls, b_cls, nboot=4000, seed=122):
    """Two-level bootstrap (proteins within class, trajectories within protein; REF line refitted each time)."""
    rng = np.random.default_rng(seed)
    out = []
    groups = {c: pt[pt.cls == c].reset_index(drop=True) for c in ('OPP', 'NEST', 'REF')}
    for _ in range(nboot):
        g = {}
        for c, d in groups.items():
            d = d.iloc[rng.integers(len(d), size=len(d))].copy()
            d['y'] = [np.log10(rmst(rng.choice(trajs[nm], len(trajs[nm])), CAP)) for nm in d.name]
            g[c] = d
        try:
            r = resid_contrast(g['REF'], g[a_cls], None if b_cls == 'REF' else g[b_cls])['est_log10']
        except np.linalg.LinAlgError:
            continue
        out.append(r)
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def strand_order_counts(sel):
    """Census cell of each selected sheet: frustration-free topologies with the same strand count and parallel-crossover
    count that also carry a ferredoxin-type pair, and how many of them are used (L_H > 0)."""
    f = pd.read_csv(OCC_MAIN)
    f = f[(f.violations == 0) & (f.n.isin(sel.n_strands.unique()))].copy()
    f['fd'] = [int(s2x.features(t)['C2z_crossing'] > 0) for t in f.topology]
    f = f[f.fd == 1]
    rows = []
    for r in sel.itertuples():
        cell = f[(f.n == r.n_strands) & (f.stratum == min(r.npar, 6))]
        rows.append(dict(name=r.name, cell_n=int(r.n_strands), cell_npar=int(min(r.npar, 6)), cell_topologies=len(cell),
                         cell_used=int((cell.L_H > 0).sum()), own_in_frame=bool((f.topology == r.topology).any()),
                         own_used=bool(((f.topology == r.topology) & (f.L_H > 0)).any())))
    return pd.DataFrame(rows)


def main(FD, OUT):
    sel = pd.read_csv(FD / 'selection.csv')
    rows, trajs, problems = [], {}, []
    for r in sel.itertuples():
        f = FD / f'summary_{r.name}_{TAG}.csv'
        if not f.exists():
            problems.append(f'{r.name}: no summary')
            continue
        s = pd.read_csv(f)
        if len(s) != NTRAJ or s.seed.min() != 1000 or int(s.steps_run.max()) > CAP:
            problems.append(f'{r.name}: n={len(s)} seed0={s.seed.min()}')
        cal = json.load(open(FD / f'calib_{r.name}_{TAG}.json'))
        rem = json.load(open(FD / f'remd_{r.name}.json'))
        t = s.t_fold_Q.to_numpy(float)
        trajs[r.name] = t
        Ts = rem['Ts']
        rows.append(dict(name=r.name, cls=r.cls, ecod_h=r.ecod_h, L=r.L, n_strands=r.n_strands, npar=r.npar,
                         topology=r.topology, resolution=r.resolution, RCO=r.RCO, ACO=r.ACO, contacts_per_res=r.contacts_per_res,
                         Tf=rem['Tf_cv'], Tf_q=rem['Tf_q'], Tf_at_edge=bool(min(abs(rem['Tf_cv'] - Ts[0]), abs(rem['Tf_cv'] - Ts[-1])) < 0.03),
                         remd_accept=rem['accept'], Qthr=cal['Qthr'], Q_native=cal['Q_native_mean'],
                         n_unfolded_starts=cal['n_unfolded_starts'],
                         frac_folded_10M=float(np.isfinite(t).mean()), frac_folded_3M=float((np.nan_to_num(t, nan=np.inf) <= CAP2).mean()),
                         km_median=km_median(t, CAP), rmst_10M=rmst(t, CAP), rmst_3M=rmst(t, CAP2),
                         ent_mismatch=int((s.t_fold_Q != s.t_fold_ent).sum())))
    pt = pd.DataFrame(rows)
    pt['y'] = np.log10(pt.rmst_10M)
    so = strand_order_counts(sel)
    pt = pt.merge(so, on='name', how='left')
    res = dict(integrity=problems)
    # censoring by class
    res['censoring'] = pt.groupby('cls').frac_folded_10M.agg(['mean', 'min']).to_dict()
    # instrument gate (class-blind): log10 RMST rises with absolute contact order
    rho, p = stats.spearmanr(pt.ACO, pt.y)
    res['gate'] = dict(spearman_ACO=float(rho), p_one_sided=float(p / 2 if rho > 0 else 1 - p / 2),
                       passed=bool(rho > 0 and p / 2 < 0.05))
    rL, pL = stats.spearmanr(pt.L, pt.y)
    rR, pR = stats.spearmanr(pt.RCO, pt.y)
    res['gate_descriptive'] = dict(spearman_L=float(rL), p_L=float(pL), spearman_RCO=float(rR), p_RCO=float(pR))
    beta, s, df, _ = ref_fit(pt[pt.cls == 'REF'])
    pt['resid'] = pt.y - design(pt) @ beta
    res['ref_fit'] = dict(b0=beta[0], b_RCO=beta[1], b_logL=beta[2], resid_sd=s, df=df)
    # contrasts
    ref = pt[pt.cls == 'REF']
    C = {}
    for name, a, b in (('OPP_vs_REF', 'OPP', 'REF'), ('OPP_vs_NEST', 'OPP', 'NEST'), ('NEST_vs_REF', 'NEST', 'REF')):
        c = resid_contrast(ref, pt[pt.cls == a], None if b == 'REF' else pt[pt.cls == b])
        c['perm_p_one_sided_slower'], c['n_perm'] = perm_p(pt, a, b)
        c['boot_ci_log10'] = boot_ci(pt, trajs, a, b)
        C[name] = c
    # secondary: fraction folded by the cap (protein-level means; Welch t)
    for name, a, b in (('OPP_vs_REF', 'OPP', 'REF'), ('OPP_vs_NEST', 'OPP', 'NEST')):
        x, z = pt[pt.cls == a].frac_folded_10M, pt[pt.cls == b].frac_folded_10M
        tt = stats.ttest_ind(x, z, equal_var=False)
        C[name]['frac_folded_diff'] = float(x.mean() - z.mean())
        C[name]['frac_folded_welch_p'] = float(tt.pvalue)
        ci = tt.confidence_interval() if hasattr(tt, 'confidence_interval') else None
        C[name]['frac_folded_ci'] = [float(ci.low), float(ci.high)] if ci is not None else None
    res['contrasts'] = C
    res['class_summary'] = pt.groupby('cls')[['L', 'n_strands', 'RCO', 'ACO', 'contacts_per_res', 'Tf', 'frac_folded_10M',
                                              'rmst_10M', 'resid', 'cell_topologies', 'cell_used']].mean().to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    pt.to_csv(OUT / 'protein_table.csv', index=False)
    json.dump(res, open(OUT / 'analysis.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    print(pt[['cls', 'name', 'ecod_h', 'L', 'n_strands', 'RCO', 'Tf', 'Tf_at_edge', 'Qthr', 'frac_folded_3M', 'frac_folded_10M',
              'km_median', 'rmst_10M', 'resid', 'cell_topologies', 'cell_used']].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in res.items() if k != 'class_summary'}, indent=1, default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dir', default=str(DATA / 'folding' / 'fd1'))
    ap.add_argument('--out', default=str(RESULTS / 'folding' / 'fd1'))
    a = ap.parse_args()
    main(Path(a.dir), Path(a.out))
