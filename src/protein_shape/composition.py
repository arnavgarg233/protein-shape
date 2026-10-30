"""Composition model: expected superfamily use of a topology from its make-up, fitted on topologies that
lack the pair class under test, with permutation worlds that move whole chain-reversal classes."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import s2x
from .edits import reverse
from .paths import COMPOSITION_COVARIATES, OCC_MAIN

SEED = 122
SIZES = (5, 6, 7)
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
C_FIXED = 0.1

# Relation x span counts; S2x_crossing (the exposure) is left out by construction.
REL_SPAN = ['T_crossing', 'T_nested', 'T_disjoint',
            'C2y_crossing', 'C2y_nested', 'C2y_disjoint',
            'C2z_crossing', 'C2z_nested', 'C2z_disjoint',
            'S2x_nested', 'S2x_disjoint']
ENTANGLED = ['mixed_sigma', 'two_faces', 'n_classes']
BASE = ['n', 'npar', 'par_frac', 'sum_jump', 'max_jump', 'n_jumps', 'rco_sheet', 'M2']
FEATURES_A = BASE + REL_SPAN + ENTANGLED
FEATURES_B = BASE + REL_SPAN


def build(occ_path=OCC_MAIN, cov_path=COMPOSITION_COVARIATES):
    """The 2,961 frustration-free five- to seven-strand topologies with at least two parallel crossovers,
    with pair-relation counts, sheet descriptors, the four-strand geometry score M2 and arm labels."""
    o = pd.read_csv(occ_path)
    df = o[(o.violations == 0) & (o.npar >= 2)].reset_index(drop=True)
    assert len(df) == 2961 and set(df.n) == set(SIZES), (len(df), sorted(set(df.n)))
    f = pd.DataFrame([s2x.features(t) for t in df.topology])
    assert (f.S2x_crossing == df.OPP).all() and (f.T_crossing == df.SAME).all()
    for c in f.columns:
        df[c] = f[c].to_numpy()
    cov = pd.read_csv(cov_path, float_precision='round_trip')
    df = df.merge(cov, on=['topology', 'n'], how='left', validate='1:1')
    assert df[FEATURES_A].notna().all().all(), 'missing covariate'
    df['fd'] = (df.C2z_crossing > 0).astype(int)
    nc = (df.OPP == 0) & (df.SAME == 0)
    arm = pd.Series('other', index=df.index)
    arm[nc & (df.mixed_sigma == 1) & (df.two_faces == 0)] = 'R1 one-face'
    arm[nc & (df.mixed_sigma == 0) & (df.two_faces == 1)] = 'R2 one-sigma two-face'
    arm[nc & (df.mixed_sigma == 1) & (df.two_faces == 1)] = 'E noncross-S2x'
    arm[(df.SAME >= 1) & (df.OPP == 0)] = 'SAME same-face crossing'
    arm[(df.OPP >= 1) & (df.SAME == 0)] = 'OPP'
    df['arm'] = arm
    assert ((df.arm == 'E noncross-S2x') == (nc & (df.S2x_noncross >= 1))).all()
    # cross-validation group: the chain-reversal class
    df['rev_group'] = [min(t, reverse(t)) for t in df.topology]
    return df


def make(C):
    return Pipeline([('s', StandardScaler()),
                     ('m', LogisticRegression(C=C, max_iter=5000, solver='lbfgs'))])


def pick_C(X, y, groups, seed=SEED):
    """L2 penalty by inner five-fold grouped cross-validation on mean log loss."""
    from sklearn.metrics import log_loss
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    scores = []
    for C in C_GRID:
        ll = []
        for tr, te in cv.split(X, y, groups):
            p = make(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            ll.append(log_loss(y[te], p, labels=[0, 1]))
        scores.append(float(np.mean(ll)))
    return C_GRID[int(np.argmin(scores))], dict(zip(map(str, C_GRID), scores))


def poisson_binomial_lower(k, p):
    """P(K <= k) for independent Bernoulli(p_i), by dynamic programming."""
    d = np.zeros(len(p) + 1)
    d[0] = 1.0
    for pi in p:
        d[1:] = d[1:] * (1 - pi) + d[:-1] * pi
        d[0] *= (1 - pi)
    return float(d[:k + 1].sum()), float(d.sum())


def exposed_mask(df):
    """Topologies with an OPP pair and no same-face crossing pair."""
    return (df.OPP.to_numpy() >= 1) & (df.SAME.to_numpy() == 0)


def fam_logistic(Xfit, yfit, Xall, C=C_FIXED):
    return make(C).fit(Xfit, yfit).predict_proba(Xall)[:, 1]


def _addc(X):
    return np.hstack([np.ones((len(X), 1)), X])


def fam_poisson(Xfit, cfit, Xall):
    g = sm.GLM(cfit, _addc(Xfit), family=sm.families.Poisson()).fit()
    return np.asarray(g.predict(_addc(Xall)))


def mc_se(p, n):
    return float(np.sqrt(max(p, 1.0 / n) * (1 - max(p, 1.0 / n)) / n))


def perm_p_with_se(null_vals, obs, n):
    """One-sided p in the smaller tail, raw and with the (r + 1) / (n + 1) correction."""
    nv = np.asarray(null_vals, float)
    lo = float((nv <= obs).sum())
    hi = float((nv >= obs).sum())
    r = min(lo, hi)
    p_raw = r / n
    p_add1 = (r + 1.0) / (n + 1.0)
    return dict(p_raw=round(p_raw, 6), p_add1=round(p_add1, 6),
                mc_se=round(mc_se(p_raw, n), 6), n_worlds=int(n),
                tail='lower' if lo <= hi else 'upper',
                null_q=[round(float(x), 4) for x in np.percentile(nv, [1, 5, 50, 95, 99])])


class Worlds:
    """Null worlds that relabel whole chain-reversal classes as exposed, either freely or within blocks of
    equal strand count and crossover count (stratified), so the pseudo-exposure keeps the real size
    profile. Within a block the draw is the first t entries of one shared permutation of the block."""

    def __init__(self, df):
        g = df.rev_group.to_numpy()
        codes, gidx = np.unique(g, return_inverse=True)
        self.ng = len(codes)
        self.gidx = gidx
        order = np.argsort(gidx, kind='stable')
        s = np.searchsorted(gidx[order], np.arange(self.ng), 'left')
        e = np.searchsorted(gidx[order], np.arange(self.ng), 'right')
        self.order, self.gstart, self.gend = order, s, e
        self.members = [order[s[i]:e[i]] for i in range(self.ng)]
        n = df.n.to_numpy()
        npar = df.npar.to_numpy()
        gn = np.array([n[m[0]] for m in self.members])
        gnp = np.array([int(npar[m[0]]) for m in self.members])
        self.key = gn * 100 + np.clip(gnp, 0, 9)
        self.ublocks = [np.flatnonzero(self.key == k) for k in np.unique(self.key)]
        self.nrow = len(df)

    def group_exposed(self, sel_real):
        out = np.zeros(self.ng, bool)
        np.logical_or.at(out, self.gidx, sel_real)
        return out

    def _rows_of(self, groups):
        st, en = self.gstart[groups], self.gend[groups]
        cnts = en - st
        tot = int(cnts.sum())
        if tot == 0:
            return np.empty(0, np.int64)
        off = np.zeros(len(cnts) + 1, np.int64)
        np.cumsum(cnts, out=off[1:])
        pos = np.arange(tot) - np.repeat(off[:-1], cnts)
        return self.order[np.repeat(st, cnts) + pos]

    def draw_many(self, gex_list, rp, strat):
        """One world applied to several exposures, so their statistics are jointly distributed."""
        if strat:
            shuf = [ii[rp.permutation(len(ii))] for ii in self.ublocks]
            out = []
            for gex in gex_list:
                tgt = [sh[:int(gex[ii].sum())] for ii, sh in zip(self.ublocks, shuf)
                       if int(gex[ii].sum())]
                out.append(np.concatenate(tgt) if tgt else np.empty(0, np.int64))
        else:
            perm = rp.permutation(self.ng)
            out = [perm[np.flatnonzero(gex)] for gex in gex_list]
        masks = []
        for tgt in out:
            m = np.zeros(self.nrow, bool)
            m[self._rows_of(tgt)] = True
            masks.append(m)
        return masks

    def draw(self, gex, rp, strat):
        s = np.zeros(self.nrow, bool)
        if strat:
            tgt = []
            for ii in self.ublocks:
                t = int(gex[ii].sum())
                if t:
                    tgt.append(ii[rp.permutation(len(ii))[:t]])
            tgt = np.concatenate(tgt) if tgt else np.empty(0, np.int64)
        else:
            perm = rp.permutation(self.ng)
            tgt = perm[np.flatnonzero(gex)]
        s[self._rows_of(tgt)] = True
        return s
