"""Recompute Table 1 and Supplementary Table 2 (and the contrasts they draw on) from the shipped
occupancy and credit tables, and check every value against results/tables/expected.json.

    python scripts/tables.py [--nulls 10000] [--no-check]

--nulls runs the composition-model permutation worlds (10,000 in the paper; about 4 min on 6 cores).
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from protein_shape import composition as C
from protein_shape import paths
from protein_shape import stats as E
from protein_shape.support import KINDS, guarded_use, load_credits, supporters

SEED = E.SEED
CREDITS = ('agree', 'lenient')
BY3, BY2 = E.BY3, E.BY2
FAMILY_CELLS = [(rel, span) for rel in ['T', 'C2y', 'C2z', 'S2x'] for span in ['crossing', 'nested', 'disjoint']]


def lenient_from(cache, f):
    """Lenient-credit outcome from a per-code credit table (codes absent from it count as unused)."""
    ch, sf = cache
    return np.array([int(guarded_use(ch['lenient'].get(c, {}), sf) > 0) for c in f.koga_code])


# ------------------------------------------------------------------------------ matched contrasts

def effect_sizes(fm, fr):
    """Three headline contrasts in both censuses and the twelve relation x span cells (agreement rule)."""
    rng = np.random.default_rng(SEED)
    res = {'meta': dict(seed=SEED, n_boot=E.N_BOOT, n_perm=E.N_PERM, outcome='used = L_H > 0',
                        strata='n x npar-stratum x fd')}
    for tag, f in (('main', fm), ('replication', fr)):
        A, R, Nn = E.arms(f)
        s7 = f.n == 7
        res[tag] = dict(
            opp_vs_reference=E.contrast(f, A, R, BY3, rng, 'OPP vs matched reference, pooled 5-7'),
            opp_vs_reference_7=E.contrast(f, A & s7, R & s7, BY3, rng, 'OPP vs matched reference, 7 strands'),
            opp_vs_nested=E.contrast(f, A, Nn, BY3, rng, 'OPP vs nested opposite-face, pooled'))
        if tag == 'main':
            fam = []
            for rel, span in FAMILY_CELLS:
                col = f'f_{rel}_{span}'
                car, non = f[col] >= 1, f[col] == 0
                by = BY2 if (rel, span) == ('C2z', 'crossing') else BY3
                fam.append(E.contrast(f, car, non, by, rng, f'family {rel}/{span}: carriers vs non-carriers'))
            adj = E.holm([r['p_perm_two'] for r in fam])
            for r, h in zip(fam, adj):
                r['p_holm_two'] = float(h)
            res['family12'] = fam
    return res


def fd_split(fm, fr, cache):
    """Headline contrasts split by ferredoxin-type core, with n x npar strata inside each split."""
    rng = np.random.default_rng(SEED)
    out = {}
    for tag, f0 in (('main', fm), ('replication', fr)):
        f = f0.copy()
        A, R, N = E.arms(f)
        S = (f.SAME >= 1) & (f.OPP == 0)
        for cr in (['agree', 'lenient'] if tag == 'main' else ['agree']):
            if cr == 'lenient':
                f['used'] = lenient_from(cache, f)
            res = {}
            for fd in (0, 1):
                m = f.fd == fd
                res[f'fd{fd}_opp_vs_ref'] = E.contrast(f, A & m, R & m, BY2, rng, f'fd{fd} OPP vs reference')
                if (N & m).sum() > 20:
                    res[f'fd{fd}_opp_vs_nested'] = E.contrast(f, A & m, N & m, BY2, rng, f'fd{fd} OPP vs nested')
                    # the nested arm lies inside the reference arm, so compare it with the rest of the reference
                    res[f'fd{fd}_nested_vs_ref'] = E.contrast(f, N & m, R & ~N & m, BY2, rng,
                                                              f'fd{fd} nested vs reference without nested')
            if cr == 'agree':
                res['face_opp_vs_same'] = E.contrast(f, A, S, BY3, rng, 'face: OPP vs SAME (n x npar x fd)')
                res['same_vs_ref'] = E.contrast(f, S, R, BY3, rng, 'SAME vs reference (n x npar x fd)')
            out[f'{tag}_{cr}'] = res
    return out


def fd_interaction(fds):
    """Ratio of the fd1 and fd0 risk ratios, Wald interval from the two Greenland-Robins variances."""
    def se(r):
        lo, hi = r['mh_rr_ci']
        return (math.log(hi) - math.log(lo)) / 3.92
    out = {}
    for key, r in fds.items():
        a, b = r['fd1_opp_vs_ref'], r['fd0_opp_vs_ref']
        lr = math.log(a['mh_rr']) - math.log(b['mh_rr'])
        s = math.sqrt(se(a) ** 2 + se(b) ** 2)
        out[key] = dict(rr_fd1=a['mh_rr'], rr_fd0=b['mh_rr'], ratio=math.exp(lr),
                        ci=[math.exp(lr - 1.96 * s), math.exp(lr + 1.96 * s)],
                        p_two=math.erfc(abs(lr / s) / math.sqrt(2)))
    return out


def consensus_free(fm, cache):
    """The three headline contrasts under agreement, lenient and any-mode credit."""
    ch, sfs = cache
    f = fm.copy()
    for kind in KINDS:
        f['LH_' + kind] = [guarded_use(ch[kind].get(c, {}), sfs) for c in f.koga_code]
    A, R, Nn = E.arms(f)
    assert (f.LH_agree[A | R | Nn] == f.L_H[A | R | Nn]).all()
    s7 = f.n == 7
    rng = np.random.default_rng(SEED)
    out = {}
    for kind in KINDS:
        f['used'] = (f['LH_' + kind] > 0).astype(int)
        out[kind] = dict(
            opp_vs_reference=E.contrast(f, A, R, BY3, rng, f'{kind}: OPP vs matched reference, pooled'),
            opp_vs_reference_7=E.contrast(f, A & s7, R & s7, BY3, rng, f'{kind}: OPP vs matched reference, 7'),
            opp_vs_nested=E.contrast(f, A, Nn, BY3, rng, f'{kind}: OPP vs nested opposite-face'))
    return out


def sum_jump(t):
    order = [int(x) for x in t.split(':')[0].split('|')]
    pos = {s: i for i, s in enumerate(order)}
    return sum(abs(pos[k + 1] - pos[k]) for k in range(1, len(order)))


def r3_checks(fm, cache):
    """Either-face arm (OPP or SAME pair) against the reference; summed connection length in the strata;
    ECOD occupancy as the outcome."""
    rng = np.random.default_rng(SEED)
    f = fm.copy()
    A, R, _ = E.arms(f)
    X = (f.OPP + f.SAME) >= 1
    out = {}
    agree = f.used.copy()
    for cr, y in (('agree', agree), ('lenient', lenient_from(cache, f))):
        f['used'] = y
        out[f'interleaved_any_{cr}'] = E.contrast(f, X, R, BY3, rng, f'{cr}: OPP or SAME vs reference')
        for fd in (0, 1):
            m = f.fd == fd
            out[f'interleaved_any_fd{fd}_{cr}'] = E.contrast(f, X & m, R & m, BY2, rng,
                                                             f'{cr}: OPP or SAME vs ref, fd{fd}')
    f['used'] = agree
    f['sj'] = np.minimum([sum_jump(t) for t in f.topology], 12)
    out['opp_vs_ref_sj'] = E.contrast(f, A, R, BY3 + ['sj'], rng, 'OPP vs ref, strata + summed jump')
    m = f.fd == 1
    out['opp_vs_ref_sj_fd1'] = E.contrast(f, A & m, R & m, ['n', 'stratum', 'sj'], rng, 'fd1 OPP vs ref + summed jump')
    f['used'] = (f.ecod_obs.astype(str) == 'True').astype(int)
    out['ecod_opp_vs_ref'] = E.contrast(f, A, R, BY3, rng, 'ECOD: OPP vs reference')
    out['ecod_opp_vs_ref_fd1'] = E.contrast(f, A & m, R & m, BY2, rng, 'ECOD: fd1 OPP vs reference')
    return out


# ----------------------------------------------------------------------------- composition model

G = {}


def _null_job(args):
    kind, strat, nulls = args
    d = G[kind]
    W, gex, rp = G['W'], G['gex'], np.random.default_rng(C.SEED + 7)
    X, Xg, y, cnt, Cpen = d['X'], d['Xg'], d['y'], d['cnt'], d['C']
    nb_, ng_ = np.empty(nulls), np.empty(nulls)
    for w in range(nulls):
        sm_ = W.draw(gex, rp, strat)
        fm_ = ~sm_
        p2 = C.fam_logistic(X[fm_], y[fm_], X, C=Cpen)
        m2 = C.fam_poisson(Xg[fm_], cnt[fm_], Xg)
        nb_[w] = y[sm_].sum() / p2[sm_].sum()
        ng_[w] = cnt[sm_].sum() / m2[sm_].sum()
    return kind, strat, nb_, ng_


def pct(a, q=(2.5, 97.5)):
    lo, hi = np.nanpercentile(a, q)
    return float(lo), float(hi)


def composition(credits_main, nulls=0):
    """Observed over expected use of OPP topologies under the composition model, by credit rule, with a
    row bootstrap, a superfamily-clustered bootstrap and the ferredoxin split."""
    cc, sf_size = credits_main
    df = C.build()
    sup = {k: [sorted(supporters(cc[k].get(c, {}), sf_size)) for c in df.koga_code] for k in KINDS}
    LH = {k: np.array([len(s) for s in sup[k]], float) for k in KINDS}
    assert (LH['agree'] == df.L_H.to_numpy(float)).all()
    sx, tx = df.S2x_crossing.to_numpy(), df.T_crossing.to_numpy()
    fitm = (sx == 0) & (tx == 0)
    expm = C.exposed_mask(df)
    FE = [f for f in C.FEATURES_A if f != 'T_crossing']
    X = df[FE].to_numpy(float)
    keep = [c for c in FE if df[c].to_numpy()[fitm].std() > 1e-9]
    Xg = sm.add_constant(df[keep].astype(float), has_constant='add').to_numpy()[:, 1:]
    grp = df.rev_group.to_numpy()
    fd = df.fd.to_numpy()
    res = {}
    for k in KINDS:
        cnt = LH[k]
        y = (cnt >= 1).astype(int)
        Cpen = C.pick_C(X[fitm], y[fitm], grp[fitm])[0]
        pa = C.fam_logistic(X[fitm], y[fitm], X, C=Cpen)
        mu = C.fam_poisson(Xg[fitm], cnt[fitm], Xg)
        G[k] = dict(X=X, Xg=Xg, y=y, cnt=cnt, C=Cpen, mu=mu)
        res[k] = dict(
            fit_n=int(fitm.sum()), fit_used=int(y[fitm].sum()), fit_LH=int(cnt[fitm].sum()),
            n_covariates=len(keep), penalty_C=float(Cpen),
            arm_n=int(expm.sum()), arm_used=int(y[expm].sum()), arm_LH=int(cnt[expm].sum()),
            exp_binary=float(pa[expm].sum()), exp_graded=float(mu[expm].sum()),
            oe_binary=float(y[expm].sum() / pa[expm].sum()), oe_graded=float(cnt[expm].sum() / mu[expm].sum()))
    if nulls:
        W = C.Worlds(df)
        G['W'], G['gex'] = W, W.group_exposed(expm)
        jobs = [(k, s, nulls) for k in KINDS for s in (False, True)]
        with mp.get_context('fork').Pool(min(len(jobs), max(1, (os.cpu_count() or 2) - 2))) as pool:
            for k, strat, nb_, ng_ in pool.map(_null_job, jobs, chunksize=1):
                tag = 'stratified' if strat else 'free'
                res[k][f'{tag}_binary'] = C.perm_p_with_se(nb_, res[k]['oe_binary'], nulls)
                res[k][f'{tag}_graded'] = C.perm_p_with_se(ng_, res[k]['oe_graded'], nulls)
    opp = expm
    all_sf = sorted(sf_size)
    sf_index = {s_: i for i, s_ in enumerate(all_sf)}
    for k in KINDS:
        cnt, mu, y, r = LH[k], G[k]['mu'], G[k]['y'], res[k]
        rng = np.random.default_rng(C.SEED)
        rowci = {}
        for lab, sel in (('fd1', opp & (fd == 1)), ('fd0', opp & (fd == 0)), ('all', opp)):
            idx = np.flatnonzero(sel)
            bs = []
            for _ in range(20000):
                kk = rng.choice(idx, size=len(idx), replace=True)
                ee = mu[kk].sum()
                bs.append(cnt[kk].sum() / ee if ee > 0 else np.nan)
            rowci[lab] = dict(n=int(sel.sum()), obs=float(cnt[sel].sum()), exp=float(mu[sel].sum()),
                              oe=float(cnt[sel].sum() / mu[sel].sum()), ci=pct(bs))
        r['row_bootstrap'] = rowci
        rng = np.random.default_rng(C.SEED)

        def credited(sel):
            v = np.zeros(len(all_sf))
            for row in np.flatnonzero(sel):
                for s_ in sup[k][row]:
                    v[sf_index[s_]] += 1
            return v
        ref_v, E_ref = credited(fitm), mu[fitm].sum()
        famci = {}
        for lab, sel in (('fd1', opp & (fd == 1)), ('all', opp)):
            v, Ex = credited(sel), mu[sel].sum()
            bs = []
            for _ in range(4000):
                kk = rng.integers(0, len(all_sf), len(all_sf))
                w = np.bincount(kk, minlength=len(all_sf)).astype(float)
                den = (ref_v * w).sum() / E_ref
                bs.append((v * w).sum() / Ex / den if den > 0 else np.nan)
            famci[lab] = dict(oe=float(v.sum() / Ex / (ref_v.sum() / E_ref)), ci=pct(bs),
                              n_superfamilies_credited=int((v > 0).sum()))
        r['family_clustered'] = famci
        FEc = [f for f in FE if df.loc[fitm, f].std() > 1e-9]
        Xc = df[FEc].to_numpy(float)
        Cc = C.pick_C(Xc[fitm], y[fitm], grp[fitm])[0]
        pb = C.make(Cc).fit(Xc[fitm], y[fitm]).predict_proba(Xc)[:, 1]
        Xgc = sm.add_constant(df[FEc].astype(float), has_constant='add')
        muc = np.asarray(sm.GLM(cnt[fitm], Xgc[fitm], family=sm.families.Poisson()).fit().predict(Xgc))
        rng = np.random.default_rng(C.SEED)
        fdres = {}
        for tag, obs, exp in (('binary', y.astype(float), pb), ('graded', cnt, muc)):
            store, cell = {}, {}
            for fv, lab in ((1, 'fd1'), (0, 'fd0')):
                sel = opp & (fd == fv)
                idx = np.flatnonzero(sel)
                bs = []
                for _ in range(20000):
                    kk = rng.choice(idx, len(idx), True)
                    ee = exp[kk].sum()
                    bs.append(obs[kk].sum() / ee if ee > 0 else np.nan)
                store[lab] = np.array(bs)
                cell[lab] = dict(n=int(sel.sum()), obs=float(obs[sel].sum()), exp=float(exp[sel].sum()),
                                 oe=float(obs[sel].sum() / exp[sel].sum()), ci=pct(bs))
            q = store['fd1'] / np.where(store['fd0'] > 0, store['fd0'], np.nan)
            cell['ratio_fd1_fd0'] = dict(median=float(np.nanmedian(q)),
                                         point=float(cell['fd1']['oe'] / cell['fd0']['oe']),
                                         ci=pct(q), p_two_sided=float(2 * min(np.nanmean(q >= 1), np.nanmean(q <= 1))))
            fdres[tag] = cell
        r['fd_stratum'] = fdres
    return dict(nulls=nulls, results=res)


# --------------------------------------------------------------------- lenient credit (primary)

def _family_job(f, a, b, by, seed):
    rng = np.random.default_rng(seed)
    r = E.deterministic(f, a, b, by)
    p1, p2, null, obs = E.perm_p(f, a, b, by, rng, return_null=True)
    pu = float((np.sum(null >= obs - 1e-9) + 1) / (len(null) + 1))
    r.update(p_perm_lower=p1, p_perm_upper=pu, p_perm_two=p2, p_two_centred=min(1.0, 2 * min(p1, pu)),
             score_obs=obs, null_mean=float(null.mean()), null_sd=float(null.std()))
    return r


def kn(f, m):
    return [int(f.used[m].sum()), int(m.sum())]


def lenient_primary(fm, fr, cache, cred_main, cred_rep, cfe, fds, es, ccf):
    """Every main-text contrast under lenient credit (primary) and the agreement rule (secondary)."""
    res = dict(meta=dict(seed=SEED, n_boot=E.N_BOOT, n_perm=E.N_PERM, n_cluster_boot=4000), checks={})
    sup, frames = {}, {}
    for tag, f0, (cnt, size) in (('main', fm, cred_main), ('rep', fr, cred_rep)):
        f = f0.copy()
        sup[tag] = {}
        for k in CREDITS:
            f['LH_' + k] = [guarded_use(cnt[k].get(c, {}), size) for c in f.koga_code]
            sup[tag][k] = [supporters(cnt[k].get(c, {}), size) for c in f.koga_code]
        frames[tag] = f
    fm, fr = frames['main'], frames['rep']
    res['checks']['agreement credit reproduces main L_H'] = bool((fm.LH_agree == fm.L_H).all())
    res['checks']['agreement credit reproduces replication L_H'] = bool((fr.LH_agree == fr.L_H).all())
    ch, sfc = cache
    res['checks']['lenient credit equals class-recall cache'] = all(
        guarded_use(ch['lenient'][c], sfc) == fm.LH_lenient.iat[i] for i, c in enumerate(fm.koga_code)
        if c in ch['lenient'])

    Am, Rm, Nm = E.arms(fm)
    Arr, Rrr, Nrr = E.arms(fr)
    s7m, s7r = fm.n == 7, fr.n == 7
    spec = {
        'opp_ref_pooled': ('main', Am, Rm, BY3, 'OPP vs reference, pooled 5-7'),
        'opp_ref_7': ('main', Am & s7m, Rm & s7m, BY3, 'OPP vs reference, 7 strands'),
        'opp_ref_fd1': ('main', Am & (fm.fd == 1), Rm & (fm.fd == 1), BY2, 'OPP vs reference, fd1'),
        'opp_ref_fd0': ('main', Am & (fm.fd == 0), Rm & (fm.fd == 0), BY2, 'OPP vs reference, fd0'),
        'opp_nested_pooled': ('main', Am, Nm, BY3, 'OPP vs nested, pooled'),
        'opp_nested_fd1': ('main', Am & (fm.fd == 1), Nm & (fm.fd == 1), BY2, 'OPP vs nested, fd1'),
        'rep_opp_ref_pooled': ('rep', Arr, Rrr, BY3, 'replication: OPP vs reference, pooled'),
        'rep_opp_ref_7': ('rep', Arr & s7r, Rrr & s7r, BY3, 'replication: OPP vs reference, 7 strands'),
        'rep_opp_ref_fd1': ('rep', Arr & (fr.fd == 1), Rrr & (fr.fd == 1), BY2, 'replication: OPP vs reference, fd1'),
        'rep_opp_ref_fd0': ('rep', Arr & (fr.fd == 0), Rrr & (fr.fd == 0), BY2, 'replication: OPP vs reference, fd0'),
        'rep_opp_nested_pooled': ('rep', Arr, Nrr, BY3, 'replication: OPP vs nested, pooled'),
    }
    # contrasts already computed (with their own random streams) by the functions above
    stored = {
        ('opp_ref_pooled', 'agree'): cfe['agree']['opp_vs_reference'],
        ('opp_ref_pooled', 'lenient'): cfe['lenient']['opp_vs_reference'],
        ('opp_ref_7', 'agree'): cfe['agree']['opp_vs_reference_7'],
        ('opp_ref_7', 'lenient'): cfe['lenient']['opp_vs_reference_7'],
        ('opp_nested_pooled', 'agree'): cfe['agree']['opp_vs_nested'],
        ('opp_nested_pooled', 'lenient'): cfe['lenient']['opp_vs_nested'],
        ('opp_ref_fd1', 'agree'): fds['main_agree']['fd1_opp_vs_ref'],
        ('opp_ref_fd0', 'agree'): fds['main_agree']['fd0_opp_vs_ref'],
        ('opp_nested_fd1', 'agree'): fds['main_agree']['fd1_opp_vs_nested'],
        ('opp_ref_fd1', 'lenient'): fds['main_lenient']['fd1_opp_vs_ref'],
        ('opp_ref_fd0', 'lenient'): fds['main_lenient']['fd0_opp_vs_ref'],
        ('opp_nested_fd1', 'lenient'): fds['main_lenient']['fd1_opp_vs_nested'],
        ('rep_opp_ref_pooled', 'agree'): es['replication']['opp_vs_reference'],
        ('rep_opp_ref_7', 'agree'): es['replication']['opp_vs_reference_7'],
        ('rep_opp_nested_pooled', 'agree'): es['replication']['opp_vs_nested'],
        ('rep_opp_ref_fd1', 'agree'): fds['replication_agree']['fd1_opp_vs_ref'],
        ('rep_opp_ref_fd0', 'agree'): fds['replication_agree']['fd0_opp_vs_ref'],
    }
    lite = {}
    for tag, f in frames.items():
        for cr in CREDITS:
            g = f[['rclass', 'n', 'stratum', 'fd']].copy()
            g['used'] = (f['LH_' + cr] > 0).astype(int)
            lite[tag, cr] = g
    out, same = {}, []
    for i, (key, (tag, a, b, by, lab)) in enumerate(spec.items()):
        for j, cr in enumerate(CREDITS):
            f = lite[tag, cr]
            d = E.deterministic(f, a, b, by)
            if (key, cr) in stored:
                s = stored[key, cr]
                same.append((d['arm_a'], d['arm_b']) == (s['arm_a'], s['arm_b']) and
                            np.allclose([d['mh_rr']] + d['mh_rr_ci'], [s['mh_rr']] + s['mh_rr_ci'], rtol=1e-9))
                out[key, cr] = dict(s, source='shared')
            else:
                rng = np.random.default_rng([SEED, 1, i, j])
                out[key, cr] = E.contrast(f, a, b, by, rng, f'{cr}: {lab}')
    res['checks']['shared contrasts agree with the full credit tables'] = all(same)

    # ten testable relation x span cells (T/nested and C2z/nested have no carriers)
    fam_keys = [c for c in FAMILY_CELLS if c not in (('T', 'nested'), ('C2z', 'nested'))]
    fam = {cr: {} for cr in CREDITS}
    for k, (rel, span) in enumerate(fam_keys):
        col = f'f_{rel}_{span}'
        car, non = fm[col] >= 1, fm[col] == 0
        by = BY2 if (rel, span) == ('C2z', 'crossing') else BY3
        for j, cr in enumerate(CREDITS):
            fam[cr][f"{rel}/{span}"] = _family_job(lite["main", cr], pd.Series(car.to_numpy()), pd.Series(non.to_numpy()), by, [SEED, 2, k, j])
    names = [f'{r}/{s}' for r, s in fam_keys]
    family = {}
    for cr in CREDITS:
        hc = E.holm([fam[cr][n]['p_two_centred'] for n in names])
        for i, n in enumerate(names):
            family.setdefault(n, {})[cr] = dict(fam[cr][n], p_holm10_centred=float(hc[i]))

    rng = np.random.default_rng([SEED, 3])
    cboot = {}
    for key in ('opp_ref_pooled', 'opp_ref_fd1', 'opp_nested_pooled'):
        tag, a, b, by, lab = spec[key]
        for cr in CREDITS:
            f = lite[tag, cr]
            rc = E.cluster_boot(f, pd.Series(a), pd.Series(b), by, None, rng)
            sf = E.cluster_boot(f, pd.Series(a), pd.Series(b), by, sup['main'][cr], rng)
            cboot[key, cr] = dict(reversal_only=rc, superfamily=sf)

    figd = {}
    for cr in CREDITS:
        f = lite['main', cr]
        a_ = {n: dict(opp=kn(f, Am & (fm.n == n)), ref=kn(f, Rm & (fm.n == n))) for n in (5, 6, 7)}
        b_ = [dict(npar_stratum=int(st), opp=kn(f, Am & s7m & (fm.stratum == st)),
                   ref=kn(f, Rm & s7m & (fm.stratum == st))) for st in sorted(fm.stratum[Am & s7m].unique())]
        cd = {f'fd{k}': {n: dict(opp=kn(f, Am & (fm.n == n) & (fm.fd == k)), ref=kn(f, Rm & (fm.n == n) & (fm.fd == k)))
                         for n in (5, 6, 7)} for k in (1, 0)}
        cl = E.cells(f, Am, Nm, BY3)
        y = f.used.to_numpy()
        g_ = dict(interleaved=[int(sum(y[ia].sum() for ia, _ in cl)), int(sum(len(ia) for ia, _ in cl))],
                  nested=[int(sum(y[ib].sum() for _, ib in cl)), int(sum(len(ib) for _, ib in cl))])
        figd[cr] = dict(fig3_a=a_, fig3_b_7strand=b_, fig3_cd=cd, fig4c_bottom=g_)

    oe = {}
    for cr in CREDITS:
        r = ccf['results'][cr]
        oe[cr] = dict(fd1=r['family_clustered']['fd1'], all=r['family_clustered']['all'],
                      fd0_row=r['row_bootstrap']['fd0'], arm_LH=r['arm_LH'], fd1_obs=r['row_bootstrap']['fd1']['obs'])
    res['checks']['composition-model credit totals equal the credit tables'] = all(
        oe[cr]['arm_LH'] == int(fm['LH_' + cr][Am].sum()) and oe[cr]['fd1_obs'] == int(fm['LH_' + cr][Am & (fm.fd == 1)].sum())
        for cr in CREDITS)
    res.update(figure_data=figd, contrasts={f'{k}|{cr}': v for (k, cr), v in out.items()}, family10=family,
               cluster_boot={f'{k}|{cr}': v for (k, cr), v in cboot.items()}, oe=oe)
    return res


# ---------------------------------------------------------------------------- tables and checks

SUPP2_ROWS = [
    ('opp_ref_pooled', 'OPP vs matched reference, 5-7 strands'),
    ('opp_ref_7', 'OPP vs matched reference, 7 strands'),
    ('opp_ref_fd1', 'OPP vs reference, ferredoxin-type sheets'),
    ('opp_ref_fd0', 'OPP vs reference, other sheets'),
    ('opp_nested_pooled', 'OPP vs nested opposite-face'),
    ('opp_nested_fd1', 'OPP vs nested, ferredoxin-type sheets'),
    ('rep_opp_ref_pooled', 'Replication census, OPP vs reference'),
    ('rep_opp_ref_7', 'Replication census, 7 strands'),
    ('rep_opp_ref_fd1', 'Replication census, ferredoxin-type sheets'),
    ('rep_opp_ref_fd0', 'Replication census, other sheets'),
    ('rep_opp_nested_pooled', 'Replication census, OPP vs nested'),
]


def flatten(lp, r3):
    """Every value printed in Table 1 and Supplementary Table 2, as a flat {name: value} dict."""
    v = {}

    def est(name, x, lo_hi):
        v[name] = x
        v[name + '/lo'], v[name + '/hi'] = lo_hi
    c, cb, oe = lp['contrasts'], lp['cluster_boot'], lp['oe']
    for key, _ in SUPP2_ROWS:
        for cr in CREDITS:
            r = c[f'{key}|{cr}']
            p = f'supp2/{key}/{cr}'
            v[p + '/arm_a'], v[p + '/arm_b'] = r['arm_a'], r['arm_b']
            est(p + '/rr', r['mh_rr'], r['mh_rr_ci'])
            v[p + '/boot/lo'], v[p + '/boot/hi'] = r['mh_rr_boot_ci']
            est(p + '/or', r['mh_or'], r['mh_or_ci'])
            v[p + '/p'] = r['p_perm_lower']
            if f'{key}|{cr}' in cb:
                v[p + '/cluster/lo'], v[p + '/cluster/hi'] = cb[f'{key}|{cr}']['superfamily']['ci']
    cols = (('fd1', 'opp_ref_fd1'), ('fd0', 'opp_ref_fd0'), ('all', 'opp_ref_pooled'))
    for col, key in cols:
        r = c[f'{key}|lenient']
        est(f'table1/matched/{col}', r['mh_rr'], r['mh_rr_ci'])
        r = c[f'rep_{key}|lenient']
        est(f'table1/replication/{col}', r['mh_rr'], r['mh_rr_ci'])
        if f'{key}|lenient' in cb:
            est(f'table1/matched_clustered/{col}', c[f'{key}|lenient']['mh_rr'], cb[f'{key}|lenient']['superfamily']['ci'])
    for col, key in (('fd1', 'interleaved_any_fd1_lenient'), ('fd0', 'interleaved_any_fd0_lenient'),
                     ('all', 'interleaved_any_lenient')):
        est(f'table1/either_face/{col}', r3[key]['mh_rr'], r3[key]['mh_rr_ci'])
    L = oe['lenient']
    est('table1/composition/fd1', L['fd1']['oe'], L['fd1']['ci'])
    est('table1/composition/fd0', L['fd0_row']['oe'], L['fd0_row']['ci'])
    est('table1/composition/all', L['all']['oe'], L['all']['ci'])
    return {k: (float(x) if not isinstance(x, str) else x) for k, x in v.items()}


def fmt_table1(v):
    def cell(p):
        if p + '/lo' not in v:
            return 'not computed'
        return f"{v[p]:.2f} ({v[p + '/lo']:.2f} to {v[p + '/hi']:.2f})"
    rows = [('Matched references', 'matched'), ('Matched references, superfamily clusters resampled', 'matched_clustered'),
            ('Replication census', 'replication'), ('Either face (OPP or same-face interleaved pair)', 'either_face'),
            ('Composition model, observed over expected superfamilies', 'composition')]
    out = ['| Analysis | Ferredoxin-type sheets | Other sheets | All sheets |', '|---|---|---|---|']
    for lab, key in rows:
        out.append(f'| {lab} | ' + ' | '.join(cell(f'table1/{key}/{c}') for c in ('fd1', 'fd0', 'all')) + ' |')
    return '\n'.join(out)


def fmt_p(p):
    return f'{p:.4f}' if p < 0.01 else f'{p:.2g}' if p < 0.1 else f'{p:.2f}'


def fmt_supp2(v):
    out = ['| Contrast | Credit | Used | Risk ratio (95% CI) | Bootstrap 95% | Cluster bootstrap 95% | Odds ratio (95% CI) | P |',
           '|---|---|---|---|---|---|---|---|']
    for key, lab in SUPP2_ROWS:
        for cr, crl in (('lenient', 'lenient'), ('agree', 'agreement')):
            p = f'supp2/{key}/{cr}'
            clu = f"{v[p + '/cluster/lo']:.2f}-{v[p + '/cluster/hi']:.2f}" if p + '/cluster/lo' in v else '-'
            out.append(f"| {lab if cr == 'lenient' else ''} | {crl} | {v[p + '/arm_a']} vs {v[p + '/arm_b']} | "
                       f"{v[p + '/rr']:.2f} ({v[p + '/rr/lo']:.2f}-{v[p + '/rr/hi']:.2f}) | "
                       f"{v[p + '/boot/lo']:.2f}-{v[p + '/boot/hi']:.2f} | {clu} | "
                       f"{v[p + '/or']:.2f} ({v[p + '/or/lo']:.2f}-{v[p + '/or/hi']:.2f}) | {fmt_p(v[p + '/p'])} |")
    return '\n'.join(out)


def check(v, expected, tol=0.005):
    bad = []
    for k, e in expected.items():
        if k not in v:
            bad.append(f'{k}: missing')
        elif isinstance(e, str):
            if v[k] != e:
                bad.append(f'{k}: {v[k]} != {e}')
        elif not abs(v[k] - e) <= tol:
            bad.append(f'{k}: {v[k]:.4f} vs expected {e:.4f}')
    extra = sorted(set(v) - set(expected))
    return bad, extra


def dump(name, obj):
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    (paths.TABLES / name).write_text(json.dumps(obj, indent=1, default=float) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--nulls', type=int, default=0, help='composition-model permutation worlds (0 skips them)')
    ap.add_argument('--no-check', action='store_true')
    a = ap.parse_args()
    fm, fr = E.load(paths.OCC_MAIN), E.load(paths.OCC_REPLICATION)
    cache = load_credits(paths.CLASS_RECALL_CENSUS)
    cred_main, cred_rep = load_credits(paths.CREDITS_MAIN), load_credits(paths.CREDITS_REPLICATION)

    es = effect_sizes(fm, fr)
    fds = fd_split(fm, fr, cache)
    cfe = consensus_free(fm, cache)
    r3 = r3_checks(fm, cred_main)
    ccf = composition(cred_main, a.nulls)
    lp = lenient_primary(fm, fr, cache, cred_main, cred_rep, cfe, fds, es, ccf)
    for name, obj in (('effect_sizes.json', es), ('fd_split.json', fds), ('fd_interaction.json', fd_interaction(fds)),
                      ('consensus_free_effects.json', cfe), ('r3_checks.json', r3), ('composition.json', ccf),
                      ('lenient_primary.json', lp)):
        dump(name, obj)
    v = flatten(lp, r3)
    dump('table_values.json', v)
    md = '# Table 1\n\n' + fmt_table1(v) + '\n\n# Supplementary Table 2\n\n' + fmt_supp2(v) + '\n'
    (paths.TABLES / 'tables.md').write_text(md)
    print(md)
    checks = lp['checks']
    for k, ok in checks.items():
        print(f'check: {k}: {"yes" if ok else "NO"}')
    if a.no_check:
        return
    expected = json.loads((paths.TABLES / 'expected.json').read_text())
    bad, extra = check(v, expected)
    n_cnt = sum(isinstance(e, str) for e in expected.values())
    print(f'{len(expected)} values checked against expected.json ({n_cnt} counts exact, '
          f'{len(expected) - n_cnt} estimates within 0.005): {len(expected) - len(bad)} match')
    for b in bad:
        print('  MISMATCH', b)
    if bad or extra or not all(checks.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
