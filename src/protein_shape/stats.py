"""Matched-comparison estimators: Mantel-Haenszel risk and odds ratios with Greenland-Robins intervals,
a chain-reversal-class bootstrap, a superfamily-cluster bootstrap and a stratified CMH-score permutation
test that relabels whole chain-reversal classes.

The bootstrap and permutation loops draw the same random numbers in the same order as a direct
per-draw implementation, so results are identical for a given generator state.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from . import s2x
from .edits import reverse

SEED, N_BOOT, N_PERM = 122, 4000, 10000
BY3, BY2 = ['n', 'stratum', 'fd'], ['n', 'stratum']


def load(path):
    """Frustration-free topologies with at least two parallel crossovers, with pair-relation counts,
    the ferredoxin indicator fd, the outcome used = (L_H > 0) and the chain-reversal class."""
    f = pd.read_csv(path)
    f = f[(f.violations == 0) & (f.npar >= 2)].reset_index(drop=True)
    feats = pd.DataFrame([s2x.features(t) for t in f.topology])
    for c in feats.columns:
        f['f_' + c] = feats[c].to_numpy()
    f['fd'] = (f.f_C2z_crossing > 0).astype(int)
    f['used'] = (f.L_H > 0).astype(int)
    rev = {t: reverse(t) for t in f.topology}
    f['rclass'] = [min(t, rev[t]) for t in f.topology]
    return f


def arms(f):
    """Exposed (OPP pair, no same-face crossing pair), matched reference and nested arms."""
    A = (f.OPP >= 1) & (f.SAME == 0)
    R = (f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)
    N = (f.f_S2x_nested >= 1) & (f.OPP == 0) & (f.SAME == 0)
    return A, R, N


def cells(f, a, b, by):
    """(a_idx, b_idx) per stratum in which both arms are present, in sorted stratum order."""
    out = []
    for _, g in f[a | b].groupby(by):
        ia, ib = g.index[a[g.index]], g.index[b[g.index]]
        if len(ia) and len(ib):
            out.append((ia, ib))
    return out


def mh(f, cl):
    """MH risk ratio, Greenland-Robins variance of its log, MH odds ratio and RBG variance of its log."""
    y = f.used.to_numpy()
    num_rr = den_rr = num_or = den_or = 0.0
    P = PS_QR = QS = 0.0
    gr_num = 0.0
    for ia, ib in cl:
        a, n1 = y[ia].sum(), len(ia)
        c, n0 = y[ib].sum(), len(ib)
        b, d = n1 - a, n0 - c
        N = n1 + n0
        num_rr += a * n0 / N
        den_rr += c * n1 / N
        gr_num += (n1 * n0 * (a + c) - a * c * N) / N ** 2
        r, s = a * d / N, b * c / N
        p, q = (a + d) / N, (b + c) / N
        num_or += r
        den_or += s
        P += p * r
        PS_QR += p * s + q * r
        QS += q * s
    rr = num_rr / den_rr if den_rr else float('nan')
    var_lrr = gr_num / (num_rr * den_rr) if num_rr and den_rr else float('nan')
    orr = num_or / den_or if den_or else float('nan')
    var_lor = (P / (2 * num_or ** 2) + PS_QR / (2 * num_or * den_or) + QS / (2 * den_or ** 2)
               if num_or and den_or else float('nan'))
    return rr, var_lrr, orr, var_lor


def ci(est, var):
    if not np.isfinite(est) or not np.isfinite(var) or est <= 0:
        return [float('nan'), float('nan')]
    h = 1.96 * np.sqrt(var)
    return [float(np.exp(np.log(est) - h)), float(np.exp(np.log(est) + h))]


def boot_rr(f, a, b, by, rng, n_boot=None):
    """Percentile 95% interval of the MH risk ratio over n_boot resamples of chain-reversal classes."""
    n_boot = N_BOOT if n_boot is None else n_boot
    classes = f.groupby('rclass').indices
    keys = list(classes)
    K = len(keys)
    cls_of = np.empty(len(f), int)
    for k, key in enumerate(keys):
        cls_of[classes[key]] = k
    am, bm = np.asarray(a, bool), np.asarray(b, bool)
    sub = f[am | bm]
    strata = sorted(sub.groupby(by).groups)
    sid = {s: i for i, s in enumerate(strata)}
    rows = np.flatnonzero(am | bm)
    key_of = sub[by].itertuples(index=False, name=None) if len(by) > 1 else sub[by[0]]
    srow = np.array([sid[k] for k in key_of])
    y = f.used.to_numpy().astype(float)
    S = len(strata)
    Ia = np.zeros((len(rows), S)); Ia[np.flatnonzero(am[rows]), srow[am[rows]]] = 1
    Ib = np.zeros((len(rows), S)); Ib[np.flatnonzero(bm[rows]), srow[bm[rows]]] = 1
    Ya, Yb = Ia * y[rows, None], Ib * y[rows, None]
    counts = np.empty((n_boot, K))
    for i in range(n_boot):
        counts[i] = np.bincount(rng.integers(0, K, K), minlength=K)
    mult = counts[:, cls_of[rows]]
    n1, a_, n0, c_ = mult @ Ia, mult @ Ya, mult @ Ib, mult @ Yb
    N = n1 + n0
    ok = (n1 > 0) & (n0 > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        tn = np.where(ok, a_ * n0 / N, 0.0)
        td = np.where(ok, c_ * n1 / N, 0.0)
    num = np.zeros(n_boot); den = np.zeros(n_boot)
    for j in range(S):           # stratum order, as a running sum
        num = num + tn[:, j]
        den = den + td[:, j]
    with np.errstate(divide='ignore', invalid='ignore'):
        rr = np.where(den != 0, num / np.where(den != 0, den, 1), np.nan)
    out = rr[np.isfinite(rr)]     # a draw with no used carriers gives RR = 0, a real value
    if not len(out):
        return [float('nan'), float('nan')], 0
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], int(len(out))


def perm_p(f, a, b, by, rng, return_null=False, n_perm=None):
    """Permutation test of the CMH score (used in arm a minus expectation, summed over strata).

    Within each stratum, the arm labels of whole chain-reversal classes are exchanged among classes of
    the same size, so arm sizes hold. Returns the one-sided (lower) p and the two-sided p, twice the
    smaller tail, since the null is not centred at zero.
    """
    n_perm = N_PERM if n_perm is None else n_perm
    cl = cells(f, a, b, by)
    y = f.used.to_numpy()
    obs = sum(y[ia].sum() - len(ia) * (y[ia].sum() + y[ib].sum()) / (len(ia) + len(ib)) for ia, ib in cl)
    plan = []
    for ia, ib in cl:
        members = np.concatenate([ia, ib])
        lab = np.r_[np.ones(len(ia), bool), np.zeros(len(ib), bool)]
        grp = defaultdict(list)
        for m, l in zip(members, lab):
            grp[f.rclass.iat[m]].append((m, l))
        units = list(grp.values())
        by_size = defaultdict(list)
        for k, u in enumerate(units):
            by_size[len(u)].append(k)
        groups = []
        for sz, ks in by_size.items():
            pat = np.array([[l for _, l in units[k]] for k in ks], float)
            yy = np.array([[y[m] for m, _ in units[k]] for k in ks], float)
            groups.append((len(ks), pat @ yy.T))   # [src, dst]: used members of dst under src's labels
        n1, n0 = len(ia), len(ib)
        tot = y[ia].sum() + y[ib].sum()
        plan.append((len(units), groups, n1 * tot / (n1 + n0)))
    draws = [[[None] * len(g) for _, g, _ in plan] for _ in range(n_perm)]
    for w in range(n_perm):
        for c, (nu, groups, _) in enumerate(plan):
            rng.permutation(nu)
            for g, (m, _) in enumerate(groups):
                draws[w][c][g] = rng.permutation(m)
    stats = np.zeros(n_perm)
    for c, (_, groups, const) in enumerate(plan):
        a_ = np.zeros(n_perm)
        for g, (m, D) in enumerate(groups):
            dst = np.array([draws[w][c][g] for w in range(n_perm)]).reshape(n_perm, m)
            a_ += D[np.arange(m)[None, :], dst].sum(1)
        stats = stats + (a_ - const)
    lower = (np.sum(stats <= obs + 1e-9) + 1) / (n_perm + 1)
    upper = (np.sum(stats >= obs - 1e-9) + 1) / (n_perm + 1)
    two = min(1.0, 2 * min(lower, upper))
    if return_null:
        return float(lower), float(two), stats, float(obs)
    return float(lower), float(two)


def counts(f, a, b, by):
    cl = cells(f, a, b, by)
    y = f.used.to_numpy()
    au = int(sum(y[ia].sum() for ia, _ in cl)); an = int(sum(len(ia) for ia, _ in cl))
    bu = int(sum(y[ib].sum() for _, ib in cl)); bn = int(sum(len(ib) for _, ib in cl))
    return cl, f'{au}/{an}', f'{bu}/{bn}'


def contrast(f, a, b, by, rng, name, verbose=False):
    """Counts, MH risk and odds ratios with intervals, class bootstrap and permutation p for one contrast."""
    cl, sa, sb = counts(f, a, b, by)
    rr, vrr, orr, vor = mh(f, cl)
    bci, nb = boot_rr(f, a, b, by, rng)
    p1, p2 = perm_p(f, a, b, by, rng)
    r = dict(contrast=name, arm_a=sa, arm_b=sb, strata=len(cl),
             mh_rr=rr, mh_rr_ci=ci(rr, vrr), mh_rr_boot_ci=bci, n_boot_ok=nb,
             mh_or=orr, mh_or_ci=ci(orr, vor), p_perm_lower=p1, p_perm_two=p2)
    if verbose:
        print(f"{name:52s} {sa:>8s} vs {sb:>8s}  RR {rr:.2f} [{r['mh_rr_ci'][0]:.2f}, {r['mh_rr_ci'][1]:.2f}] "
              f"boot [{bci[0]:.2f}, {bci[1]:.2f}]  p {p1:.4f}", flush=True)
    return r


def deterministic(f, a, b, by):
    """Counts, MH risk ratio and Greenland-Robins interval only."""
    cl, sa, sb = counts(f, a, b, by)
    rr, v, _, _ = mh(f, cl)
    return dict(arm_a=sa, arm_b=sb, mh_rr=rr, mh_rr_ci=ci(rr, v), strata=len(cl))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def clusters(f, rows, sup):
    """Connected components of `rows` linked by chain-reversal class and, if sup is given, by a shared
    supporting superfamily. Superfamilies are visited in sorted order so the component labels, and with
    them the bootstrap draws, do not depend on the hash seed."""
    parent = {i: i for i in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for i in rows:
        keys = [('r', f.rclass.iat[i])] + ([('s', s) for s in sorted(sup[i])] if sup is not None else [])
        for k in keys:
            if k in owner:
                parent[find(i)] = find(owner[k])
            else:
                owner[k] = i
    roots = np.array([find(i) for i in rows])
    _, lab = np.unique(roots, return_inverse=True)
    return lab


def cluster_boot(f, a, b, by, sup, rng, B=4000):
    """Percentile interval of the MH risk ratio when whole clusters (see `clusters`) are resampled."""
    cl = cells(f, a, b, by)
    rows = np.concatenate([np.r_[ia, ib] for ia, ib in cl])
    cell = np.concatenate([np.full(len(ia) + len(ib), k) for k, (ia, ib) in enumerate(cl)])
    isa = np.concatenate([np.r_[np.ones(len(ia), bool), np.zeros(len(ib), bool)] for ia, ib in cl])
    y = f.used.to_numpy()[rows].astype(float)
    lab = clusters(f, rows, sup)
    K, C = lab.max() + 1, len(cl)
    Ma = np.zeros((len(rows), C)); Ma[np.flatnonzero(isa), cell[isa]] = 1
    Mb = np.zeros((len(rows), C)); Mb[np.flatnonzero(~isa), cell[~isa]] = 1
    out = []
    for lo in range(0, B, 500):
        W = rng.multinomial(K, np.full(K, 1 / K), size=min(500, B - lo)).astype(float)[:, lab]
        n1, a_, n0, c_ = W @ Ma, (W * y) @ Ma, W @ Mb, (W * y) @ Mb
        N = n1 + n0
        ok = (n1 > 0) & (n0 > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            num = np.where(ok, a_ * n0 / np.where(ok, N, 1), 0).sum(1)
            den = np.where(ok, c_ * n1 / np.where(ok, N, 1), 0).sum(1)
        out.append(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan))
    rr = np.concatenate(out)
    rr = rr[np.isfinite(rr)]
    used = y > 0
    size = np.bincount(lab, minlength=K)
    used_in = np.bincount(lab, used, minlength=K)
    bigu = int(np.argmax(used_in))
    return dict(ci=[float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))], n_ok=int(len(rr)),
                topologies=int(len(rows)), used=int(used.sum()), clusters=int(K),
                clusters_with_used=int((used_in > 0).sum()),
                largest_cluster_topologies=int(size.max()),
                max_used_in_one_cluster=int(used_in[bigu]))
