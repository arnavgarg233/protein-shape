"""Topology frames for the census: pair-face counts per topology, the occupancy frame of the five- to
seven-strand topologies, and the stratified van Elteren and Mantel-Haenszel tests used on it."""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm, rankdata

from . import grammar
from . import topology as TP
from .paths import DATA

MODES = ('strict', 'lenient', 'lenient_noterm')
SIZES = (5, 6, 7)
N_PERM = 20000
SEED = 122
FRAME_DIR = DATA  # frame{n}.csv, written by scripts/topology/build_frames.py; frame4.csv is shipped


def pair_faces(t):
    """(n_parallel_crossovers, same-face crossing same-direction pairs, opposite-face pairs)."""
    par = [c for c in grammar.parallel_connections(*TP.parse(t)) if c['typ'] == 'P']
    same = opp = 0
    for i, a in enumerate(par):
        for b in par[i + 1:]:
            if grammar.span_relation(a, b) == 'crossing' and (a['pos2'] > a['pos1']) == (b['pos2'] > b['pos1']):
                if grammar._rh_face(a) == grammar._rh_face(b):
                    same += 1
                else:
                    opp += 1
    return len(par), same, opp


def cmh(tables):
    """Mantel-Haenszel OR and one-sided (OR < 1) p with continuity correction; tables = [(a,b,c,d)]."""
    num = den = A = EA = VA = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if (a + b) == 0 or (c + d) == 0 or n < 2:
            continue
        num += a * d / n
        den += b * c / n
        EA += (a + b) * (a + c) / n
        VA += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
        A += a
    orr = num / den if den else float('inf')
    z = (A - EA + 0.5) / np.sqrt(VA) if VA else float('nan')  # continuity correction toward null (lower tail)
    return orr, float(norm.cdf(z))


def stratum(k):
    return min(max(k, 2), 6)


def test_group(df, mask_exposed, mask_ref, obs):
    tabs, rows = [], []
    for s, g in df.groupby('stratum'):
        e, r, o = mask_exposed[g.index], mask_ref[g.index], obs[g.index]
        t = (int((e & o).sum()), int((e & ~o).sum()), int((r & o).sum()), int((r & ~o).sum()))
        rows.append(dict(stratum=int(s), exp_obs=t[0], exp_unobs=t[1], ref_obs=t[2], ref_unobs=t[3]))
        tabs.append(t)
    orr, p = cmh(tabs)
    tot = np.array(tabs).sum(0)
    fe = fisher_exact([[tot[0], tot[1]], [tot[2], tot[3]]], alternative='less')
    return dict(cmh_or=orr, cmh_p_one_sided=p, fisher_or=fe.statistic, fisher_p_less=fe.pvalue,
                exposed=f'{tot[0]}/{tot[0] + tot[1]}', reference=f'{tot[2]}/{tot[2] + tot[3]}', strata=rows)


def both_orientations(t):
    return len({c['sign1'] for c in grammar.parallel_connections(*TP.parse(t)) if c['typ'] == 'P'}) == 2


def frame(occ, frame_dir=FRAME_DIR):
    """Five- to seven-strand topologies with pair-face counts, strata and the occupancy measures in `occ`."""
    rows = []
    for n in SIZES:
        f = pd.read_csv(f'{frame_dir}/frame{n}.csv', usecols=['topology', 'koga_code', 'violations', 'freq'])
        pf = f.topology.map(pair_faces)
        f['npar'] = [x[0] for x in pf]
        f['SAME'] = [x[1] for x in pf]
        f['OPP'] = [x[2] for x in pf]
        f['both'] = f.topology.map(both_orientations)
        f['n'] = n
        rows.append(f)
    f = pd.concat(rows, ignore_index=True)
    f['stratum'] = f.npar.clip(upper=6)
    f['ecod_obs'] = f.freq.fillna(0) >= 0.25
    for k, d in occ.items():
        f[k] = f.koga_code.map(lambda c: d.get(c, 0))
    return f


def van_elteren(df, a_mask, b_mask, y, by, n_perm=N_PERM):
    """One-sided test that y is lower in group a than in group b, stratified by `by`.

    Weights 1/(N_s + 1), midranks, tie-corrected variance. Returns the asymptotic p (continuity
    corrected) and a stratified permutation p (labels permuted within strata; decision statistic).
    """
    rng = np.random.default_rng(SEED)
    num = var = wsum = 0.0
    perm = np.zeros(n_perm)
    aucs, rows = [], []
    for key, g in df.groupby(by):
        ya = y[g.index][a_mask[g.index]].to_numpy(float)
        yb = y[g.index][b_mask[g.index]].to_numpy(float)
        m, k = len(ya), len(yb)
        if m == 0 or k == 0:
            continue
        N = m + k
        r = rankdata(np.concatenate([ya, yb]))
        W = r[:m].sum()
        _, t = np.unique(np.concatenate([ya, yb]), return_counts=True)
        v = m * k / 12.0 * ((N + 1) - (t ** 3 - t).sum() / (N * (N - 1)))
        w = 1.0 / (N + 1)
        E = m * (N + 1) / 2.0
        num += w * (W - E)
        var += w * w * v
        wsum += w
        for i in range(0, n_perm, 2000):
            b = min(2000, n_perm - i)
            idx = np.argsort(rng.random((b, N)), axis=1)[:, :m]
            perm[i:i + b] += w * (r[idx].sum(axis=1) - E)
        auc = (W - m * (m + 1) / 2.0) / (m * k)  # P(a > b) + 0.5 P(tie)
        aucs.append((auc, w * m * k))
        rows.append(dict(stratum=str(key), n_a=m, n_b=k, mean_a=float(ya.mean()), mean_b=float(yb.mean()),
                         frac_pos_a=float((ya > 0).mean()), frac_pos_b=float((yb > 0).mean()), auc=float(auc)))
    if not var:
        return dict(z=float('nan'), p_asymptotic=float('nan'), p_perm=float('nan'), strata=rows)
    z = (num + 0.5 * wsum) / np.sqrt(var)
    p_perm = (1 + int((perm <= num + 1e-12).sum())) / (n_perm + 1) if n_perm else float('nan')
    auc_w = sum(a * w for a, w in aucs) / sum(w for _, w in aucs)
    return dict(z=float(z), p_asymptotic=float(norm.cdf(z)), p_perm=float(p_perm),
                weighted_auc_a_gt_b=float(auc_w), strata=rows)


def binary(df, a_mask, b_mask, obs, by):
    tabs = []
    for _, g in df.groupby(by):
        e, r, o = a_mask[g.index], b_mask[g.index], obs[g.index]
        tabs.append((int((e & o).sum()), int((e & ~o).sum()), int((r & o).sum()), int((r & ~o).sum())))
    orr, p = cmh(tabs)
    t = np.array(tabs).sum(0)
    fe = fisher_exact([[t[0], t[1]], [t[2], t[3]]], alternative='less')
    return dict(cmh_or=orr, cmh_p_one_sided=p, fisher_p_less=fe.pvalue,
                exposed=f'{t[0]}/{t[0] + t[1]}', reference=f'{t[2]}/{t[2] + t[3]}')
