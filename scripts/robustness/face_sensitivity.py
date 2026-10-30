"""Can the right-handed face convention create or inflate the OPP depletion? Measures face misclassification
in solved structures, propagates it through the census contrasts by probabilistic bias analysis, and reads faces
directly from coordinates of PDB sheets and AlphaFold models that carry used topologies.

    python scripts/robustness/face_sensitivity.py --pairs PAIRS.csv --sheets SHEETS.csv \
        [--audit-structures DIR --afdb-models DIR --targets T.jsonl --models M.jsonl]

PAIRS.csv and SHEETS.csv are the crossover-pair and sheet tables of the ECOD crystal-structure geometry run
(geom_loops.py); the optional AlphaFold inputs are the census audit models and the class_recall.py models.
"""
import argparse
import gzip
import json
import os
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from protein_shape import paths  # noqa: E402
from protein_shape import stats as E  # noqa: E402
from protein_shape.frames import both_orientations  # noqa: E402
from protein_shape.support import guarded_use, load_credits  # noqa: E402

SEED, NB, WORKERS, HMIN = 122, 2000, 3, 1.0
PCL = ('OPP', 'SAME', 'NEST', 'OTHER')
ARM4 = ('E', 'N', 'Rx', 'O')
ARM3 = ('E', 'R', 'O')
# set from the command line in main()
PAIRS = SHEETS = AUDIT = AFCR = TARGETS = MODELS = None
OUT = paths.RESULTS / 'robustness' / 'face_sensitivity'
LINES = []


def say(s=''):
    print(s, flush=True)
    LINES.append(s)


# ------------------------------------------------------------------------------- classes and arms

def rel_from(fa, fb, same_dir):
    sf = np.asarray(fa) == np.asarray(fb)
    return np.where(np.asarray(same_dir) == 1, np.where(sf, 'T', 'S2x'), np.where(sf, 'C2z', 'C2y'))


def pclass(rel, span):
    rel, span = np.asarray(rel), np.asarray(span)
    return np.select([(rel == 'S2x') & (span == 'crossing'), (rel == 'T') & (span == 'crossing'),
                      (rel == 'S2x') & (span == 'nested')], ['OPP', 'SAME', 'NEST'], 'OTHER')


def arm4(opp, same, nest, both):
    opp, same, nest, both = (np.asarray(x) for x in (opp, same, nest, both))
    ref = (opp == 0) & (same == 0) & both
    return np.select([(opp >= 1) & (same == 0), ref & (nest >= 1), ref], ['E', 'N', 'Rx'], 'O')


def to3(a4):
    return np.where(np.isin(a4, ['N', 'Rx']), 'R', a4)


def pair_table():
    P = pd.read_csv(PAIRS)
    rc = rel_from(P.face_frozen_a, P.face_frozen_b, P.same_dir)
    g_conv = bool((rc == P.relation.to_numpy()).all())
    und = (P.face_emp_a == 0) | (P.face_emp_b == 0)
    re = rel_from(P.face_emp_a, P.face_emp_b, P.same_dir)
    g_emp = bool((re[~und] == P.relation_emp.to_numpy()[~und]).all())
    P['cc'] = pclass(P.relation, P.span)
    P['kc'] = np.where(und, 'UND', pclass(re, P.span))
    # variant C: a coordinate face counts only when the loop sits >= HMIN from the surface
    fa = np.where(P.h_surf_centroid_a.abs() >= HMIN, P.face_emp_a, P.face_frozen_a)
    fb = np.where(P.h_surf_centroid_b.abs() >= HMIN, P.face_emp_b, P.face_frozen_b)
    P['kcC'] = pclass(rel_from(fa, fb, P.same_dir), P.span)
    return P, dict(conv_relation_reproduced=g_conv, emp_relation_reproduced=g_emp, undetermined_pairs=int(und.sum()))


def sheet_table(P, frame_topos):
    S = pd.read_csv(SHEETS, usecols=['sheet_id', 'ecod_h', 'n_strands', 'ferredoxin_core', 'topology'])
    S = S[S.sheet_id.isin(set(P.sheet_id))].set_index('sheet_id')
    bo = {t: both_orientations(t) for t in S.topology.unique()}
    S['both'] = S.topology.map(bo)
    for tag, col in (('c', 'cc'), ('k', 'kc'), ('kC', 'kcC')):
        cnt = pd.crosstab(P.sheet_id, P[col]).reindex(columns=list(PCL) + ['UND'], fill_value=0)
        cnt = cnt.reindex(S.index, fill_value=0)
        S[tag + '4'] = arm4(cnt.OPP, cnt.SAME, cnt.NEST, S.both)
        if tag == 'k':
            S['und'] = cnt.UND.to_numpy() > 0
    S['fd'] = S.ferredoxin_core.astype(int)
    S['in_frame'] = S.topology.isin(frame_topos)
    return S.reset_index()


# ------------------------------------------------------------------------------- bootstrap helpers

def hboot_counts(h, cell, K, rng, B=NB):
    """Point counts (K,) and B H-group-bootstrap count vectors (B, K)."""
    hu, hi = np.unique(h, return_inverse=True)
    M = np.zeros((len(hu), K))
    np.add.at(M, (hi, cell), 1)
    W = rng.multinomial(len(hu), np.full(len(hu), 1 / len(hu)), size=B).astype(float)
    return M.sum(0), W @ M


def rowrates(C, shape):
    """Counts (..., K) -> row-normalised rates over the last axis of `shape`."""
    X = C.reshape(C.shape[:-1] + shape)
    with np.errstate(invalid='ignore', divide='ignore'):
        return X / X.sum(-1, keepdims=True)


def pct(a):
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [np.nan, np.nan]


def fmt_r(pt, ci, scale=100, d=1):
    return f"{pt * scale:.{d}f} [{ci[0] * scale:.{d}f}, {ci[1] * scale:.{d}f}]"


# ------------------------------------------------------------------------------- task 1

def confusion(P, S, rng, res):
    say('\n' + '=' * 110)
    say('1. MISCLASSIFICATION IN SOLVED STRUCTURES (% of row, H-group cluster bootstrap 95% CI)')
    say('=' * 110)
    rows, cols = list(PCL), list(PCL) + ['UND']
    out = {}
    for fdtag, m in (('all', np.ones(len(P), bool)), ('fd0', P.ferredoxin_core.to_numpy() == 0),
                     ('fd1', P.ferredoxin_core.to_numpy() == 1)):
        Q = P[m]
        cell = Q.cc.map(rows.index).to_numpy() * len(cols) + Q.kc.map(cols.index).to_numpy()
        pt, bt = hboot_counts(Q.ecod_h.to_numpy(), cell, len(rows) * len(cols), rng)
        R0, RB = rowrates(pt, (len(rows), len(cols))), rowrates(bt, (len(rows), len(cols)))
        n = pt.reshape(len(rows), len(cols)).sum(1)
        say(f'\nPairs, {fdtag} sheets ({len(Q)} pairs, {Q.ecod_h.nunique()} H-groups). Rows conventional, '
            f'columns coordinate.')
        say(f"  {'conv':6s} {'n':>6s}  " + '  '.join(f'{c:>20s}' for c in cols))
        d = {}
        for i, r in enumerate(rows):
            say(f'  {r:6s} {int(n[i]):6d}  ' + '  '.join(f'{fmt_r(R0[i, j], pct(RB[:, i, j])):>20s}'
                                                       for j in range(len(cols))))
            d[r] = dict(n=int(n[i]), **{c: dict(rate=float(R0[i, j]), ci=pct(RB[:, i, j]), count=int(
                pt.reshape(len(rows), len(cols))[i, j])) for j, c in enumerate(cols)})
        out[fdtag] = dict(table=d, boot=RB)
    # key rates and differentials
    say('\nKey pair rates (%), fd1 minus fd0 difference with bootstrap CI (H-groups resampled within each fd):')
    keyp = {'OPP lost (OPP -> not OPP)': ('OPP', lambda R: 1 - R[0, 0]),
            'SAME -> OPP': ('SAME', lambda R: R[1, 0]),
            'OTHER -> OPP': ('OTHER', lambda R: R[3, 0]),
            'NEST lost (NEST -> not NEST)': ('NEST', lambda R: 1 - R[2, 2])}
    kp = {}
    for name, (_, fn) in keyp.items():
        vals = {t: (fn(rowrates(np.array([[out[t]['table'][r][c]['count'] for c in cols] for r in rows]).ravel(),
                                (len(rows), len(cols)))), fn(np.moveaxis(out[t]['boot'], 0, -1)))
                for t in ('all', 'fd0', 'fd1')}
        diff = vals['fd1'][1] - vals['fd0'][1]
        kp[name] = {t: dict(rate=float(v[0]), ci=pct(v[1])) for t, v in vals.items()}
        kp[name]['fd1_minus_fd0'] = dict(diff=float(vals['fd1'][0] - vals['fd0'][0]), ci=pct(diff))
        say(f"  {name:30s} all {fmt_r(vals['all'][0], pct(vals['all'][1]))}  fd0 {fmt_r(vals['fd0'][0], pct(vals['fd0'][1]))}"
            f"  fd1 {fmt_r(vals['fd1'][0], pct(vals['fd1'][1]))}  diff {fmt_r(vals['fd1'][0] - vals['fd0'][0], pct(diff))}")
    res['pairs'] = {t: v['table'] for t, v in out.items()}
    ch = P.relation.to_numpy() != P.relation_emp.to_numpy()
    touch = ch & ((P.cc != 'OTHER') | (P.kc != 'OTHER')).to_numpy()
    sd = (P.span == 'crossing') & (P.same_dir == 1)
    inv = bool((sd.groupby(P.sheet_id).sum() == (P.assign(x=P.kc.isin(['OPP', 'SAME'])).groupby('sheet_id').x.sum())).all())
    say(f"\nRelation changes by coordinates in {int(ch.sum())} of {len(P)} pairs ({ch.mean() * 100:.1f}%); "
        f"{int(touch.sum())} of them ({touch.sum() / ch.sum() * 100:.1f}%) involve a census class (OPP, SAME or NEST); "
        f"the rest are T<->S2x disjoint/nested-other and C2y<->C2z swaps.")
    say('Structural invariance: a face error swaps T<->S2x and C2y<->C2z but leaves span and direction, so the number')
    say('of same-direction crossing pairs (OPP + SAME) of a sheet cannot change. Reference (0 such pairs) can never')
    say('become exposed, and exposed can only become SAME-carrying (other). Checked on every PDB sheet: '
        f'{inv}.')
    res['relation_changes'] = dict(pairs=int(ch.sum()), census_class=int(touch.sum()), invariance_holds=inv)
    res['pair_key_rates'] = kp

    # sheet level
    res['sheets'] = {}
    for sset, msk in (('frame_5to7', S.in_frame.to_numpy()), ('all_4to8', np.ones(len(S), bool))):
        T = S[msk & ~S.und.to_numpy()]
        say(f"\nSheets, {sset}: {len(T)} sheets ({T.ecod_h.nunique()} H-groups); {int((msk & S.und.to_numpy()).sum())} "
            f"sheets with an undetermined face dropped. Rows conventional arm, columns coordinate arm (% of row).")
        dd = {}
        for fdtag, m in (('all', np.ones(len(T), bool)), ('fd0', T.fd.to_numpy() == 0), ('fd1', T.fd.to_numpy() == 1)):
            Q = T[m]
            cell = Q.c4.map(ARM4.index).to_numpy() * 4 + Q.k4.map(ARM4.index).to_numpy()
            pt, bt = hboot_counts(Q.ecod_h.to_numpy(), cell, 16, rng)
            R0, RB = rowrates(pt, (4, 4)), rowrates(bt, (4, 4))
            C0 = pt.reshape(4, 4)
            say(f"  {fdtag} ({len(Q)} sheets, {Q.ecod_h.nunique()} H-groups)")
            say(f"    {'conv':5s} {'n':>5s}  " + '  '.join(f'{c:>20s}' for c in ARM4) + '   leaves arm (E/R/O level)')
            d = {}
            for i, r in enumerate(ARM4):
                if C0[i].sum() == 0:
                    continue
                i3 = [j for j, c in enumerate(ARM4) if to3(np.array([c]))[0] == to3(np.array([r]))[0]]
                leave = 1 - R0[i, i3].sum()
                leaveB = 1 - RB[:, i, i3].sum(-1)
                say(f'    {r:5s} {int(C0[i].sum()):5d}  ' + '  '.join(
                    f'{fmt_r(R0[i, j], pct(RB[:, i, j])):>20s}' for j in range(4)) + f'   {fmt_r(leave, pct(leaveB))}')
                d[r] = dict(n=int(C0[i].sum()), counts={c: int(C0[i, j]) for j, c in enumerate(ARM4)},
                            rates={c: dict(rate=float(R0[i, j]), ci=pct(RB[:, i, j])) for j, c in enumerate(ARM4)},
                            leaves_arm3=dict(rate=float(leave), ci=pct(leaveB)))
            dd[fdtag] = d
        # fd flip of the sheet by coordinates (C2y <-> C2z changes the ferredoxin-core flag)
        res['sheets'][sset] = dd
    return res


def fd_flip(P, S, res):
    k = P[P.span == 'crossing']
    c2z = k.assign(z=(k.relation_emp == 'C2z')).groupby('sheet_id').z.any()
    T = S[S.in_frame & ~S.und].set_index('sheet_id')
    fdk = c2z.reindex(T.index, fill_value=False).astype(int)
    tab = pd.crosstab(T.fd, fdk)
    say('\nFerredoxin-core flag (C2z crossing pair present) by convention (rows) and by coordinates (columns), '
        'frame sheets:')
    say('  ' + tab.to_string().replace('\n', '\n  '))
    res['fd_flag_conv_vs_coord'] = {f'conv{int(i)}': {f'coord{int(j)}': int(tab.loc[i, j]) for j in tab.columns}
                                    for i in tab.index}


# ------------------------------------------------------------------------------- task 2

def frame_lenient():
    f = E.load(paths.OCC_MAIN)
    cnt, size = load_credits(paths.CREDITS_MAIN)
    f['used'] = [int(guarded_use(cnt['lenient'].get(c, {}), size) > 0) for c in f.koga_code]
    C = json.load(gzip.open(paths.CLASS_RECALL_CENSUS, 'rt'))
    cached = C['chains']['lenient']
    g2 = all(int(guarded_use(cached[c], C['sf_size']) > 0) == f.used.iat[i]
             for i, c in enumerate(f.koga_code) if c in cached)
    A = (f.OPP >= 1) & (f.SAME == 0)
    R = (f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)
    f['arm'] = np.where(A, 'E', np.where(R, 'R', 'O'))
    f['sid'] = f.groupby(['n', 'stratum', 'fd']).ngroup().to_numpy()
    return f, g2


def mh_counts(sid, used, a, b, K):
    """Per stratum (a_used, n1, c_used, n0) for strata with both arms; used may be fractional weights."""
    n1 = np.bincount(sid, weights=a, minlength=K)
    au = np.bincount(sid, weights=a * used, minlength=K)
    n0 = np.bincount(sid, weights=b, minlength=K)
    cu = np.bincount(sid, weights=b * used, minlength=K)
    return au, n1, cu, n0


def mh_rr(au, n1, cu, n0, keep=None):
    ok = (n1 > 0) & (n0 > 0) if keep is None else keep
    N = n1 + n0
    num = (au * n0 / np.where(ok, N, 1))[ok].sum()
    den = (cu * n1 / np.where(ok, N, 1))[ok].sum()
    return num / den if den > 0 else np.nan


def contrasts(f, arm, used=None):
    used = f.used.to_numpy().astype(float) if used is None else used
    K = f.sid.max() + 1
    out = {}
    for key, m in (('pooled', np.ones(len(f), bool)), ('fd1', f.fd.to_numpy() == 1), ('fd0', f.fd.to_numpy() == 0)):
        a = ((arm == 'E') & m).astype(float)
        b = ((arm == 'R') & m).astype(float)
        out[key] = mh_rr(*mh_counts(f.sid.to_numpy(), used, a, b, K))
    return out


def transitions(S, col, rng):
    """Point (2,3,3) and bootstrap (B,2,3,3) transition matrices conv arm3 -> coord arm3 by fd, frame sheets."""
    T = S[S.in_frame & ~S.und]
    c3, k3 = to3(T.c4.to_numpy()), to3(T[col].to_numpy())
    cell = T.fd.to_numpy() * 9 + np.array([ARM3.index(x) for x in c3]) * 3 + np.array([ARM3.index(x) for x in k3])
    pt, bt = hboot_counts(T.ecod_h.to_numpy(), cell, 18, rng)
    return pt.reshape(2, 3, 3), bt.reshape(-1, 2, 3, 3)


def draw_arms(arm0, fd, move, P_b, rng, own=None):
    """New arm per topology: rows in `move` drawn from P_b[fd, conv arm]; `own` maps index -> probability row."""
    new = arm0.copy()
    idx = np.flatnonzero(move)
    a = np.array([ARM3.index(x) for x in arm0[idx]])
    probs = P_b[fd[idx], a]                                     # (m, 3)
    tot = probs.sum(1, keepdims=True)
    probs = np.where(tot > 0, probs / np.where(tot > 0, tot, 1), np.eye(3)[a])
    if own is not None:
        for j, i in enumerate(idx):
            if i in own:
                probs[j] = own[i]
    u = rng.random(len(idx))
    k = (u[:, None] > np.cumsum(probs, 1)[:, :-1]).sum(1)
    new[idx] = np.array(ARM3)[k]
    return new, int((tot[:, 0] == 0).sum())


def bias_analysis(f, S, rng, res):
    say('\n' + '=' * 110)
    say('2. PROBABILISTIC BIAS ANALYSIS ON THE CENSUS (lenient credit; 2000 draws)')
    say('=' * 110)
    arm0, fd = f.arm.to_numpy(), f.fd.to_numpy()
    used = f.used.to_numpy().astype(bool)
    base = contrasts(f, arm0)
    say(f"Observed lenient MH RR: pooled {base['pooled']:.3f}, fd1 {base['fd1']:.3f}, fd0 {base['fd0']:.3f}")
    res['observed_rr'] = base
    se = {}
    for k, m in (('pooled', np.ones(len(f), bool)), ('fd1', fd == 1)):
        by = ['n', 'stratum', 'fd'] if k == 'pooled' else ['n', 'stratum']
        cl = E.cells(f, pd.Series((arm0 == 'E') & m), pd.Series((arm0 == 'R') & m), by)
        rr, v, _, _ = E.mh(f, cl)
        se[k] = float(np.sqrt(v))
        say(f'  {k}: Greenland-Robins 95% CI {E.ci(rr, v)[0]:.2f}-{E.ci(rr, v)[1]:.2f}')
    res['observed_gr_se'] = se
    # PDB sheets of the same string, for row B
    T = S[S.in_frame & ~S.und]
    own = {}
    tk = T.groupby('topology').k4.apply(lambda s: np.array([(to3(s.to_numpy()) == a).mean() for a in ARM3]))
    for i in np.flatnonzero(used):
        t = f.topology.iat[i]
        if t in tk.index:
            own[i] = tk[t]
    say(f'Used topologies: {int(used.sum())}; with PDB sheets of the same string (row B): {len(own)}')
    rows = {}
    for tag, col, move_all, use_own in (('A_main', 'k4', False, False), ('B_own_pdb_sheets', 'k4', False, True),
                                        ('C_confident_faces', 'kC4', False, False),
                                        ('D_all_topologies', 'k4', True, False)):
        P0, PB = transitions(S, col, rng)
        draws = {k: [] for k in ('pooled', 'fd1', 'fd0')}
        flows = []
        empty = 0
        for b in range(NB):
            move = np.ones(len(f), bool) if move_all else used
            new, e = draw_arms(arm0, fd, move, PB[b], rng, own if use_own else None)
            empty += e
            r = contrasts(f, new)
            for k in draws:
                draws[k].append(r[k])
            m1 = fd == 1
            flows.append([int((used & m1 & (arm0 == 'E') & (new != 'E')).sum()),
                          int((used & m1 & (arm0 != 'E') & (new == 'E')).sum()),
                          int((used & m1 & (arm0 == 'R') & (new != 'R')).sum()),
                          int((used & m1 & (arm0 != 'R') & (new == 'R')).sum())])
        fl = np.array(flows)
        d = {k: dict(median=float(np.nanmedian(v)), lo=float(np.nanpercentile(v, 2.5)),
                     hi=float(np.nanpercentile(v, 97.5)), share_ge_1=float(np.nanmean(np.array(v) >= 1)))
             for k, v in draws.items()}
        for k in ('pooled', 'fd1'):                 # bias + random error: log RR draw + N(0, GR SE of observed)
            tot = np.log(np.array(draws[k])) + rng.normal(0, se[k], NB)
            d[k]['total_lo'], d[k]['total_hi'] = float(np.exp(np.nanpercentile(tot, 2.5))), float(
                np.exp(np.nanpercentile(tot, 97.5)))
        d['fd1_used_flows_median'] = dict(E_out=float(np.median(fl[:, 0])), E_in=float(np.median(fl[:, 1])),
                                          R_out=float(np.median(fl[:, 2])), R_in=float(np.median(fl[:, 3])))
        d['rows_without_sheets_in_draws'] = empty
        d['P_point'] = {f'fd{z}': {a: (P0[z, i] / max(P0[z, i].sum(), 1)).round(4).tolist() for i, a in enumerate(ARM3)}
                        for z in (0, 1)}
        d['P_n'] = {f'fd{z}': {a: int(P0[z, i].sum()) for i, a in enumerate(ARM3)} for z in (0, 1)}
        rows[tag] = d
        say(f"\n  {tag}: corrected MH RR median [2.5, 97.5%]  pooled {d['pooled']['median']:.2f} "
            f"[{d['pooled']['lo']:.2f}, {d['pooled']['hi']:.2f}]  fd1 {d['fd1']['median']:.2f} "
            f"[{d['fd1']['lo']:.2f}, {d['fd1']['hi']:.2f}]  fd0 {d['fd0']['median']:.2f} "
            f"[{d['fd0']['lo']:.2f}, {d['fd0']['hi']:.2f}]   draws with fd1 RR >= 1: {d['fd1']['share_ge_1']:.4f}")
        say(f"     with random error added (Greenland-Robins SE of the observed RR): pooled [{d['pooled']['total_lo']:.2f}, "
            f"{d['pooled']['total_hi']:.2f}], fd1 [{d['fd1']['total_lo']:.2f}, {d['fd1']['total_hi']:.2f}]")
        f1 = d['fd1_used_flows_median']
        say(f"     fd1 used topologies per draw (median): E out {f1['E_out']:.0f}, into E {f1['E_in']:.0f}; "
            f"R out {f1['R_out']:.0f}, into R {f1['R_in']:.0f}")
        say('     point transition rows (E/R/O -> E/R/O), sheets per row: ' + '; '.join(
            f"fd{z} {a}->({', '.join(f'{x:.3f}' for x in d['P_point'][f'fd{z}'][a])}) n={d['P_n'][f'fd{z}'][a]}"
            for z in (0, 1) for a in ARM3))
    res['bias'] = rows
    return base


def worst_case(f, S, res):
    say('\n' + '-' * 110)
    say('Worst case (expected counts, fd1, strata as observed). A face error cannot move a topology between the')
    say('reference and exposed arms (structural invariance, section 1), so the R -> E row below is hypothetical:')
    say('it asks how much error such a route would need if it existed.')
    m1 = f.fd.to_numpy() == 1
    K = f.sid.max() + 1
    sid, used, arm = f.sid.to_numpy(), f.used.to_numpy().astype(float), f.arm.to_numpy()
    au, n1, cu, n0 = mh_counts(sid, used, ((arm == 'E') & m1).astype(float), ((arm == 'R') & m1).astype(float), K)
    keep = (n1 > 0) & (n0 > 0)
    ou = np.bincount(sid, weights=((arm == 'O') & m1) * used, minlength=K)
    w = np.where(keep, n1, 0) / np.where(keep, n1, 0).sum()

    def solve(fn):
        if fn(1.0) < 1:
            return np.nan
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fn(mid) < 1 else (lo, mid)
        return hi

    PB = res['_PB_main']
    bt = res['sheets']['frame_5to7']['fd1']
    meas = {}
    # (i) hypothetical R -> E
    qR = solve(lambda q: mh_rr(au + q * cu, n1 + q * cu, (1 - q) * cu, n0 - q * cu, keep))
    nE = sum(bt[r]['counts']['E'] for r in ('N', 'Rx') if r in bt)
    nT = sum(bt[r]['n'] for r in ('N', 'Rx') if r in bt)
    rb = PB[:, 1, 1, 0] / PB[:, 1, 1].sum(-1)
    meas['R_to_E'] = dict(q_needed=float(qR), used=float(cu[keep].sum()), measured=nE / nT, measured_ci=pct(rb),
                          sheets=nT)
    say(f"  used fd1 reference -> exposed: q needed {qR * 100:.1f}% of {cu[keep].sum():.0f} used reference topologies "
        f"(= {qR * cu[keep].sum():.1f} topologies); measured {nE}/{nT} fd1 reference sheets = {nE / nT * 100:.1f}% "
        f"[{pct(rb)[0] * 100:.1f}, {pct(rb)[1] * 100:.1f}] (zero by construction)")
    # (ii) the only real inflow: used fd1 'other' (SAME-carrying) -> exposed
    rr_all_o = mh_rr(au + ou, n1 + ou, cu, n0, keep)
    nE = bt['O']['counts']['E'] if 'O' in bt else 0
    nT = bt['O']['n'] if 'O' in bt else 0
    rb = PB[:, 1, 2, 0] / PB[:, 1, 2].sum(-1)
    meas['O_to_E'] = dict(used=float(ou[keep].sum()), used_all_strata=int(((arm == 'O') & m1 & (used > 0)).sum()),
                          rr_if_all_move=float(rr_all_o), measured=nE / nT if nT else np.nan, measured_ci=pct(rb),
                          sheets=nT)
    say(f"  used fd1 other (SAME-carrying or one-orientation) -> exposed: {int(meas['O_to_E']['used_all_strata'])} "
        f"used topologies exist ({ou[keep].sum():.0f} in the shared strata); if every one moved, fd1 RR "
        f"{rr_all_o:.2f}; measured {nE}/{nT} fd1 'other' sheets move to E "
        f"({(nE / nT * 100) if nT else float('nan'):.1f}%, one H-group)")
    # (iii) extra used exposed topologies needed, spread over strata in proportion to exposed-arm size
    x = solve(lambda q: mh_rr(au + q * 200 * w, n1, cu, n0, keep)) * 200
    meas['extra_used_exposed_needed'] = float(x)
    meas['observed_used_exposed'] = int(au[keep].sum())
    say(f"  extra used fd1 exposed topologies needed for RR = 1 (no denominator change): {x:.1f}, against "
        f"{int(au[keep].sum())} observed; the face-error route can supply at most "
        f"{int(meas['O_to_E']['used_all_strata'])}")
    res['worst_case_fd1'] = meas


# ------------------------------------------------------------------------------- task 3

def topo_check_pdb(f, S, res):
    say('\n' + '=' * 110)
    say('3. DIRECT COORDINATE CHECK ON USED TOPOLOGIES')
    say('=' * 110)
    T = S[S.in_frame & ~S.und].copy()
    T['k3'] = to3(T.k4.to_numpy())
    g = T.groupby('topology')
    u = f[f.used == 1]
    say('(a) PDB sheets with the same string. Topology "changes arm" when more than half of its')
    say('    sheets fall in another E/R/O arm by coordinates.')
    out = {}
    for (fdv, a), d in u.groupby(['fd', 'arm']):
        have = [t for t in d.topology if t in g.groups]
        ch = [t for t in have if (g.get_group(t).k3 != a).mean() > 0.5]
        to = pd.Series([g.get_group(t).k3.mode().iat[0] for t in ch]).value_counts().to_dict() if ch else {}
        ns = int(sum(len(g.get_group(t)) for t in have))
        nsc = int(sum((g.get_group(t).k3 != a).sum() for t in have))
        out[f'fd{fdv}_{a}'] = dict(used=len(d), with_pdb=len(have), sheets=ns, sheets_changing=nsc,
                                   topologies_changing=len(ch), changing_to=to, changing=ch)
        say(f'  fd{fdv} {a}: {len(d)} used, {len(have)} with PDB sheets ({ns} sheets, {nsc} in another arm); '
            f'topologies changing arm: {len(ch)} {to}')
    res['direct_pdb'] = out


def afdb_jobs(f):
    used = dict(zip(f.koga_code[f.used == 1], f.arm[f.used == 1]))
    jobs = []
    if AUDIT is None or AFCR is None or TARGETS is None or MODELS is None:
        return jobs, used
    from protein_shape.structure import chopping_to_range
    sel = pd.read_csv(AUDIT / 'selected_domains.tsv', sep='\t')
    for r in sel.itertuples(index=False):
        if r.code in used and (AUDIT / f'{r.chain}.pdb').exists():
            jobs.append(dict(src='audit', id=r.ted_id, path=str(AUDIT / f'{r.chain}.pdb'),
                             rng=chopping_to_range(r.chopping), codes=[r.code]))
    tg = {json.loads(line)['domain']: json.loads(line) for line in Path(TARGETS).open()}
    for line in Path(MODELS).open():
        m = json.loads(line)
        if m.get('status') != 'ok':
            continue
        codes = [c for c in m['per_mode']['lenient'] if c in used]
        p = AFCR / f"{m['acc']}.pdb.gz"
        if codes and p.exists():
            seg = tg[m['domain']]['uni_segments']
            jobs.append(dict(src='classrecall', id=m['domain'], path=str(p),
                             rng=','.join(f'A:{lo}-{hi}' for lo, hi in seg), codes=codes))
    return jobs, used


def afdb_one(job):
    import geom_loops as G  # crystal-geometry module (loop faces from coordinates), on the script path
    from protein_shape import structure as T
    out = []
    try:
        if job['path'].endswith('.gz'):
            tmp = tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=False)
            tmp.write(gzip.open(job['path'], 'rt').read())
            tmp.close()
            path = tmp.name
        else:
            path = job['path']
        try:
            st, labels = T.select_residues(path, None, T.parse_range(job['rng']), 1)
            dssp = T.parse_dssp(T.run_dssp(st))
            ca = np.array([r.find_atom('CA', '*').pos.tolist() for r in st[0][0]])
            plddt = np.array([r.find_atom('CA', '*').b_iso for r in st[0][0]])
        finally:
            if path != job['path']:
                os.unlink(path)
        if len(dssp['ss']) != len(labels):
            raise RuntimeError('DSSP residue count mismatch')
        resd = T.topology_from_dssp(dssp, labels, ca, strand_mode='lenient')
        found = T.strands_for_mode(dssp, 'lenient')
        base = dict(domain=job['id'], pdb='', fid='', h='', arch='', manual_rep='')
        for code in job['codes']:
            recs = [s for s in resd['sheets'] if s['code'] == code and not s['closed'] and not s['bifurcated']]
            if not recs:
                out.append(dict(src=job['src'], id=job['id'], code=code, status='code_not_a_full_sheet'))
                continue
            sh, xs, ps, _ = G.analyse_sheet(base, recs[0], labels, ca, dssp, found,
                                            {'strict': set(), 'lenient_noterm': set()})
            fe = [x['face_emp'] for x in xs]
            pdf = pd.DataFrame(ps)
            cc = pclass(pdf.relation, pdf.span)
            und = (pdf.face_emp_a == 0) | (pdf.face_emp_b == 0)
            kc = pclass(rel_from(pdf.face_emp_a, pdf.face_emp_b, pdf.same_dir), pdf.span)
            bo = both_orientations(recs[0]['topology'])
            c4 = arm4(*(int((cc == k).sum()) for k in ('OPP', 'SAME', 'NEST')), bo)
            k4 = arm4(*(int((kc == k).sum()) for k in ('OPP', 'SAME', 'NEST')), bo)
            out.append(dict(src=job['src'], id=job['id'], code=code, status='ok', topology=recs[0]['topology'],
                            c3=str(to3(np.array([c4]))[0]), k3=str(to3(np.array([k4]))[0]), und=bool(und.any()),
                            n_xo=len(xs), n_disagree=int(sum(1 for x in xs if not x['agree'])),
                            n_left=int(sum(1 for x in xs if x['handed_geom'] == 'left')),
                            plddt=round(float(plddt.mean()), 1), zero_face=int(sum(1 for v in fe if v == 0))))
    except Exception as e:  # noqa: BLE001
        out = [dict(src=job['src'], id=job['id'], code=c, status='error', error=(type(e).__name__ + ':' + str(e))[:120])
               for c in job['codes']]
    return out


def topo_check_afdb(f, res, cache=None):
    jobs, used = afdb_jobs(f)
    if not jobs:
        say('\n(b) AlphaFold models: no model inputs given, skipped.')
        return None
    t0 = time.time()
    if cache and Path(cache).exists():
        recs = json.loads(Path(cache).read_text())
    else:
        with Pool(WORKERS) as pool:
            recs = [r for rr in pool.imap_unordered(afdb_one, jobs, chunksize=4) for r in rr]
        if cache:
            Path(cache).write_text(json.dumps(recs))
    D = pd.DataFrame(recs)
    say(f'\n(b) AlphaFold models on disk: {len(jobs)} models ({sum(j["src"] == "audit" for j in jobs)} census audit, '
        f'{sum(j["src"] == "classrecall" for j in jobs)} class-recall), {len(D)} model x topology records, '
        f'{time.time() - t0:.0f} s.')
    say('    status: ' + ', '.join(f'{k} {v}' for k, v in D.status.value_counts().items()))
    ok = D[(D.status == 'ok')].copy()
    ok = ok[~ok.und.astype(bool)]
    ok['arm'] = ok.code.map(used)
    ok['fd'] = ok.code.map(dict(zip(f.koga_code, f.fd)))
    g_arm = bool((ok.c3 == ok.arm).all())
    say(f'    check: conventional arm of every measured sheet equals the frame arm of its code: {g_arm}')
    say(f"    crossovers measured {int(ok.n_xo.sum())}; loop face disagrees with the convention in "
        f"{int(ok.n_disagree.sum())} ({ok.n_disagree.sum() / ok.n_xo.sum() * 100:.1f}%); left-handed {int(ok.n_left.sum())}")
    out = dict(models=len(jobs), records=len(D), status=D.status.value_counts().to_dict(), gate_arm=g_arm)
    for (fdv, a), d in ok.groupby(['fd', 'arm']):
        per = d.groupby('code').apply(lambda x: (x.k3 != a).mean())
        ch = per[per > 0.5]
        to = d[d.code.isin(ch.index)].groupby('code').k3.agg(lambda s: s[s != a].mode().iat[0]).value_counts().to_dict()
        out[f'fd{int(fdv)}_{a}'] = dict(topologies=int(len(per)), models=int(len(d)),
                                        models_changing=int((d.k3 != a).sum()), topologies_changing=int(len(ch)),
                                        changing_to=to, changing=list(ch.index),
                                        n_used_in_arm=int(((f.used == 1) & (f.fd == fdv) & (f.arm == a)).sum()))
        say(f"  fd{int(fdv)} {a}: {len(per)} of {out[f'fd{int(fdv)}_{a}']['n_used_in_arm']} used topologies measured "
            f"({len(d)} model sheets, {int((d.k3 != a).sum())} in another arm); topologies changing arm: {len(ch)} {to}")
    res['direct_afdb'] = out
    return ok


# ------------------------------------------------------------------------------- main

def main():
    global PAIRS, SHEETS, AUDIT, AFCR, TARGETS, MODELS
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pairs', required=True, help='crossover-pair table of the crystal geometry run')
    ap.add_argument('--sheets', required=True, help='sheet table of the crystal geometry run')
    ap.add_argument('--audit-structures', help='directory of census audit AlphaFold models (with selected_domains.tsv)')
    ap.add_argument('--afdb-models', help='directory of AlphaFold models downloaded by class_recall.py')
    ap.add_argument('--targets', help='class_recall.py targets (JSON lines)')
    ap.add_argument('--models', help='class_recall.py model-side calls (JSON lines)')
    ap.add_argument('--afdb-cache', default=None, help='optional JSON cache of the AlphaFold measurements')
    a = ap.parse_args()
    PAIRS, SHEETS, TARGETS, MODELS = a.pairs, a.sheets, a.targets, a.models
    AUDIT = Path(a.audit_structures) if a.audit_structures else None
    AFCR = Path(a.afdb_models) if a.afdb_models else None
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    res = dict(meta=dict(seed=SEED, n_boot=NB, h_min_A=HMIN))
    say('=' * 110)
    say('Does right-hand face misclassification create or inflate the OPP-pair depletion? (seed 122)')
    say('=' * 110)
    f, g2 = frame_lenient()
    P, gates = pair_table()
    S = sheet_table(P, set(f.topology))
    # gate: conventional arm of in-frame PDB sheets equals the frame arm of the same string
    fa = dict(zip(f.topology, f.arm))
    T = S[S.in_frame]
    g_sheet = bool((to3(T.c4.to_numpy()) == T.topology.map(fa).to_numpy()).all())
    obs = contrasts(f, f.arm.to_numpy())
    cl = E.cells(f, pd.Series(f.arm == 'E') & (f.fd == 1), pd.Series(f.arm == 'R') & (f.fd == 1), ['n', 'stratum'])
    rr_e = E.mh(f, cl)[0]
    g_mh = bool(np.isclose(rr_e, obs['fd1']) and np.isclose(obs['fd1'], 0.40, atol=0.005)
                and np.isclose(obs['pooled'], 0.66, atol=0.005))
    gates.update(lenient_equals_class_recall_cache=g2, sheet_conv_arm_equals_frame_arm=g_sheet,
                 mh_reproduces_effect_sizes_and_lenient_primary=g_mh)
    res['gates'] = gates
    say('Checks: ' + ', '.join(f'{k} {v}' for k, v in gates.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not all(v for k, v in gates.items() if k != 'undetermined_pairs'):
        say('check failed - stopping')
        OUT.with_suffix('.txt').write_text('\n'.join(LINES) + '\n')
        sys.exit(1)
    say(f'Frame {len(f)} topologies, {int(f.used.sum())} used (lenient). PDB: {len(P)} pairs, {len(S)} sheets with '
        f'>= 2 parallel crossovers, {int(S.in_frame.sum())} of them 5-7-strand frame topologies.')

    confusion(P, S, rng, res)
    fd_flip(P, S, res)
    res['_PB_main'] = transitions(S, 'k4', np.random.default_rng(SEED + 1))[1]
    bias_analysis(f, S, rng, res)
    worst_case(f, S, res)
    topo_check_pdb(f, S, res)
    topo_check_afdb(f, res, a.afdb_cache)
    res.pop('_PB_main')
    res['runtime_s'] = round(time.time() - t0, 1)
    OUT.with_suffix('.json').write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    OUT.with_suffix('.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
