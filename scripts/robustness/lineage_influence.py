"""Leave-one-lineage-out influence on the matched OPP-versus-reference risk ratios: each CATH superfamily
or T group is removed in turn, and supporters of used ferredoxin-type topologies are removed cumulatively.

Run: python scripts/robustness/lineage_influence.py [--cath-names cath-names.txt]
"""
import argparse
import gzip
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from protein_shape import paths
from protein_shape import stats as E

OUT = paths.RESULTS / 'robustness' / 'lineage_influence'
CONTRASTS = ['pooled', 'fd1', 'fd0']
CREDITS = ['agree', 'lenient']
TOP = 5


def load_names(path):
    """Superfamily names from the CATH names file (optional; names are only labels in the report)."""
    nm = {}
    if path is None:
        return nm
    for line in open(path):
        if line.startswith('#') or ':' not in line:
            continue
        head, name = line.rstrip('\n').split(':', 1)
        nm[head.split()[0]] = name.strip()
    return nm


def tgroup(sf):
    return '.'.join(sf.split('.')[:3])


class Ctx:
    """Rows, arms, and the passing-superfamily set of every row under one credit rule."""

    def __init__(self, f, A, R, P):
        self.P = P
        self.n = len(P)
        self.cl = {'pooled': E.cells(f, A, R, ['n', 'stratum', 'fd']),
                   'fd1': E.cells(f, A & (f.fd == 1), R & (f.fd == 1), ['n', 'stratum']),
                   'fd0': E.cells(f, A & (f.fd == 0), R & (f.fd == 0), ['n', 'stratum'])}
        self.A, self.R, self.fd = A.to_numpy(), R.to_numpy(), f.fd.to_numpy()
        self.rows_of = defaultdict(list)
        for i, p in enumerate(P):
            for s in p:
                self.rows_of[s].append(i)
        self.base = np.array([int(len(p) > 0) for p in P])

    def used(self, removed):
        y = self.base.copy()
        for s in removed:
            for i in self.rows_of.get(s, ()):
                if y[i]:
                    y[i] = int(bool(self.P[i] - removed))
        return y

    def est(self, removed):
        y = self.used(removed)
        d = pd.DataFrame({'used': y})
        out = {}
        for w in CONTRASTS:
            rr, vrr = E.mh(d, self.cl[w])[:2]
            out[w] = (float(rr), float(vrr))
        return out, y

    def counts(self, y, which):
        cl = self.cl[which]
        return (int(sum(y[ia].sum() for ia, _ in cl)), int(sum(len(ia) for ia, _ in cl)),
                int(sum(y[ib].sum() for _, ib in cl)), int(sum(len(ib) for _, ib in cl)))


def coarsen(P, fn):
    return [frozenset(fn(s) for s in p) for p in P]


def influence_table(ctx, name_fn):
    """Remove each unit (superfamily or T group) that is the sole passing unit of some row, in turn.
    A unit that is never a sole supporter cannot change any row, so it leaves the estimate at the full value."""
    units = sorted(ctx.rows_of)
    full, _ = ctx.est(frozenset())
    res = {w: [] for w in CONTRASTS}
    for u in units:
        if not any(len(ctx.P[i]) == 1 for i in ctx.rows_of[u]):
            continue
        e, y = ctx.est(frozenset([u]))
        for w in CONTRASTS:
            rr = e[w][0]
            c = ctx.counts(y, w)
            res[w].append(dict(unit=u, name=name_fn(u), rr=rr, dlog=float(np.log(rr / full[w][0])),
                               used_opp=c[0], n_opp=c[1], used_ref=c[2], n_ref=c[3]))
    out = dict(n_units_supporting=len(units), n_units_evaluated=len(res['pooled']),
               full={w: full[w][0] for w in CONTRASTS})
    for w in CONTRASTS:
        rows = res[w]
        rrs = [r['rr'] for r in rows] + [full[w][0]]
        by_inf = sorted(rows, key=lambda r: -abs(r['dlog']))
        out[w] = dict(n_units_that_change_estimate=sum(1 for r in rows if abs(r['dlog']) > 1e-12),
                      rr_min=float(min(rrs)), rr_max=float(max(rrs)),
                      top=by_inf[:TOP], most_influential=by_inf[0],
                      toward_one=max(rows, key=lambda r: r['rr']) if full[w][0] < 1 else min(rows, key=lambda r: r['rr']))
    return out


def cumulative(ctx, order, which, name_fn):
    """Remove units in the given order, cumulatively; stop when the Greenland-Robins interval includes 1."""
    steps, removed = [], set()
    e0, _ = ctx.est(frozenset())
    steps.append(dict(k=0, unit=None, name=None, rr=e0[which][0], ci=E.ci(*e0[which])))
    for k, u in enumerate(order, 1):
        removed.add(u)
        e, y = ctx.est(frozenset(removed))
        rr, var = e[which]
        ci = E.ci(rr, var)
        c = ctx.counts(y, which)
        steps.append(dict(k=k, unit=u, name=name_fn(u), rr=rr, ci=ci,
                          used_opp=f'{c[0]}/{c[1]}', used_ref=f'{c[2]}/{c[3]}'))
        if not np.isfinite(rr) or rr == 0.0:
            steps[-1]['note'] = 'degenerate (no used OPP topologies left)'
            break
        if ci[1] >= 1.0:
            break
    reached = bool(np.isfinite(steps[-1]['ci'][1]) and steps[-1]['ci'][1] >= 1.0)
    return dict(candidates=len(order), steps=steps, interval_includes_1=reached,
                units_removed_when_reached=steps[-1]['k'] if reached else None)


def greedy(ctx, cands, which):
    """Remove, at every step, the unit whose removal moves the RR furthest up; stop when the interval includes 1."""
    removed, order = set(), []
    cands = set(cands)
    while cands:
        best, best_rr = None, -1.0
        for u in sorted(cands):
            rr = ctx.est(frozenset(removed | {u}))[0][which][0]
            if rr > best_rr:
                best, best_rr = u, rr
        order.append(best)
        removed.add(best)
        cands.discard(best)
        hi = E.ci(*ctx.est(frozenset(removed))[0][which])[1]
        if np.isfinite(hi) and hi >= 1.0:
            break
    return order


def fmt_ci(ci):
    return f'[{ci[0]:.2f}, {ci[1]:.2f}]'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cath-names', help='CATH cath-names.txt, for labelling superfamilies')
    a = ap.parse_args()
    names = load_names(a.cath_names)
    sfname = lambda u: names.get(u, '')  # noqa: E731
    f = E.load(paths.OCC_MAIN)
    A = (f.OPP >= 1) & (f.SAME == 0)
    R = (f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)
    C = json.load(gzip.open(paths.CLASS_RECALL_CENSUS))
    sf_size = C['sf_size']

    def passing(kind):
        ch = C['chains'][kind]
        return [frozenset(s for s, k in ch.get(c, {}).items() if k >= max(2, 0.01 * sf_size[s]))
                for c in f.koga_code]

    ctxs = {cr: Ctx(f, A, R, passing(cr)) for cr in CREDITS}

    # the full-data estimates must equal the stored matched contrasts
    arm = (A | R).to_numpy()
    assert (ctxs['agree'].base[arm] == f.used.to_numpy()[arm]).all(), 'agree chains do not reproduce L_H'
    full = {cr: ctxs[cr].est(frozenset())[0] for cr in CREDITS}
    chk = [('fd1 agree', full['agree']['fd1'][0], 0.259, ('fd_split', 'main_agree', 'fd1_opp_vs_ref')),
           ('fd1 lenient', full['lenient']['fd1'][0], 0.398, ('fd_split', 'main_lenient', 'fd1_opp_vs_ref')),
           ('pooled agree', full['agree']['pooled'][0], 0.527, ('effect_sizes', 'main', 'opp_vs_reference')),
           ('fd0 agree', full['agree']['fd0'][0], None, ('fd_split', 'main_agree', 'fd0_opp_vs_ref')),
           ('fd0 lenient', full['lenient']['fd0'][0], None, ('fd_split', 'main_lenient', 'fd0_opp_vs_ref'))]
    for nm_, got, rounded, (fn, k1, k2) in chk:
        stored = paths.TABLES / f'{fn}.json'
        if stored.exists():
            ref = json.load(open(stored))[k1][k2]['mh_rr']
            assert abs(got - ref) < 1e-9, (nm_, got, ref)
        if rounded is not None:
            assert round(got, 3) == rounded, (nm_, got, rounded)
    print('full-data estimates:', [(c[0], round(c[1], 3)) for c in chk], flush=True)

    res = dict(meta=dict(census='main', outcome='used = guarded superfamily occupancy > 0',
                         removal='global (OPP and reference arms), set difference on passing superfamilies',
                         full_estimates={cr: {w: full[cr][w][0] for w in CONTRASTS} for cr in CREDITS},
                         full_ci={cr: {w: E.ci(*full[cr][w]) for w in CONTRASTS} for cr in CREDITS}))
    for cr in CREDITS:
        c = ctxs[cr]
        idx = [i for i in range(c.n) if arm[i]]
        res['meta'][cr] = dict(arm_rows_used=int(sum(c.base[i] for i in idx)),
                               arm_rows_used_with_one_passing_unit=int(sum(1 for i in idx if len(c.P[i]) == 1)),
                               superfamilies_supporting_arm_rows=len({s for i in idx for s in c.P[i]}))

    # ---- 1. superfamily level; 2. T level ----
    for level, fn, namefn in [('H_superfamily', None, sfname),
                              ('T_topology', tgroup, lambda t: names.get(t, ''))]:
        res[level] = {}
        for cr in CREDITS:
            ctx = ctxs[cr] if fn is None else Ctx(f, A, R, coarsen(ctxs[cr].P, fn))
            res[level][cr] = influence_table(ctx, namefn)
            print(level, cr, 'done', flush=True)

    # ---- 3. cumulative removal, fd1 ----
    res['cumulative_fd1'] = {}
    for cr in CREDITS:
        ctx = ctxs[cr]
        fd1 = ctx.fd == 1
        sup_opp = sorted({s for i in range(ctx.n) if ctx.A[i] and fd1[i] and ctx.base[i] for s in ctx.P[i]})
        sup_ref = sorted({s for i in range(ctx.n) if ctx.R[i] and fd1[i] and ctx.base[i] for s in ctx.P[i]})
        single = {s: ctx.est(frozenset([s]))[0]['fd1'][0] for s in set(sup_opp) | set(sup_ref)}
        f1 = full[cr]['fd1'][0]
        ord_a = sorted(sup_opp, key=lambda s: -abs(np.log(single[s] / f1)))
        ord_b = sorted(sup_ref, key=lambda s: -single[s])
        ord_c = greedy(ctx, sup_ref, 'fd1')
        res['cumulative_fd1'][cr] = dict(
            n_supporting_used_opp=len(sup_opp), n_supporting_used_ref=len(sup_ref),
            a_opp_supporters_static=cumulative(ctx, ord_a, 'fd1', sfname),
            b_ref_supporters_static=cumulative(ctx, ord_b, 'fd1', sfname),
            c_ref_supporters_greedy=cumulative(ctx, ord_c, 'fd1', sfname))
        print('cumulative', cr, 'done', flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT.with_suffix('.json'), 'w'), indent=1)
    write_txt(res)


def write_txt(res):
    L = []
    p = L.append
    m = res['meta']
    p('Lineage influence on the OPP depletion. Point estimates (MH risk ratio, OPP arm vs matched reference).')
    p('Removing a superfamily deletes it from the passing set of every topology (both arms); a topology flips to')
    p('unused only if it was its sole passing superfamily.')
    p('')
    p('Full-data MH risk ratio (Greenland-Robins 95% interval):')
    for cr in CREDITS:
        for w in CONTRASTS:
            p(f"  {cr:8s} {w:7s} {m['full_estimates'][cr][w]:.3f} {fmt_ci(m['full_ci'][cr][w])}")
    p('')
    for cr in CREDITS:
        x = m[cr]
        p(f"Support structure, {cr} credit: {x['arm_rows_used']} used topologies in the two arms, "
          f"{x['arm_rows_used_with_one_passing_unit']} of them with exactly one passing superfamily; "
          f"{x['superfamilies_supporting_arm_rows']} superfamilies support at least one.")
    p('')
    lab = {'H_superfamily': 'CATH H-LEVEL SUPERFAMILY: remove each in turn',
           'T_topology': 'CATH T-LEVEL GROUP (coarser): remove each in turn, with all its H superfamilies'}
    for level in ('H_superfamily', 'T_topology'):
        p('=' * 100)
        p(lab[level])
        p('=' * 100)
        for cr in CREDITS:
            d = res[level][cr]
            p(f"-- {cr} credit ({d['n_units_supporting']} units support an arm topology; "
              f"{d['n_units_evaluated']} are the sole support of some topology and were re-estimated, the rest leave the estimate unchanged)")
            for w in CONTRASTS:
                x = d[w]
                mi, tw = x['most_influential'], x['toward_one']
                p(f"  {w:7s} full {d['full'][w]:.3f}  range [{x['rr_min']:.3f}, {x['rr_max']:.3f}]  "
                  f"({x['n_units_that_change_estimate']} units change it)")
                p(f"          most influential unit removed -> {mi['rr']:.3f} ({mi['unit']} {mi['name'][:45]})")
                p(f"          largest move toward 1 -> {tw['rr']:.3f} ({tw['unit']} {tw['name'][:45]})")
                for r in x['top']:
                    p(f"      {r['unit']:14s} RR {r['rr']:.3f}  OPP used {r['used_opp']}/{r['n_opp']}, "
                      f"ref used {r['used_ref']}/{r['n_ref']}  {r['name'][:55]}")
            p('')
    p('=' * 100)
    p('CUMULATIVE REMOVAL, fd1 OPP vs reference (Greenland-Robins interval at each step; stop when it includes 1)')
    p('=' * 100)
    desc = {'a_opp_supporters_static': 'a. superfamilies supporting used fd1 OPP topologies, static order of single-removal influence',
            'b_ref_supporters_static': 'b. superfamilies supporting used fd1 REFERENCE topologies, static order (single-removal RR, highest first)',
            'c_ref_supporters_greedy': 'c. superfamilies supporting used fd1 REFERENCE topologies, greedy order (re-ranked at each step)'}
    for cr in CREDITS:
        d = res['cumulative_fd1'][cr]
        p(f"-- {cr} credit: {d['n_supporting_used_opp']} superfamilies support used fd1 OPP topologies, "
          f"{d['n_supporting_used_ref']} support used fd1 reference topologies")
        for key in desc:
            x = d[key]
            end = ('interval includes 1 after removing ' + str(x['units_removed_when_reached'])
                   if x['interval_includes_1'] else 'interval never includes 1')
            p(f"  {desc[key]}")
            p(f"    {end}  ({x['candidates']} candidates)")
            for s in x['steps']:
                lab_ = '' if s['unit'] is None else f"  -{s['unit']} {s['name'][:40]}"
                oc = '' if s['unit'] is None else f"  OPP {s['used_opp']} ref {s['used_ref']}"
                p(f"    k={s['k']:3d} RR {s['rr']:.3f} {fmt_ci(s['ci'])}{oc}{lab_}{'  ' + s['note'] if 'note' in s else ''}")
        p('')
    open(OUT.with_suffix('.txt'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
