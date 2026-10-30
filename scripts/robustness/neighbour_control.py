"""Single-edit neighbour control: each exposed topology is compared with matched topologies one edit away
(strand flip, transposition or relocation) that lack the OPP and same-face pairs, with a within-set
permutation p, a superfamily-cluster bootstrap, and the size and power of the test.

Run: python scripts/robustness/neighbour_control.py
"""
import argparse
import gzip
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from protein_shape import edits as ED
from protein_shape import paths
from protein_shape import stats as ES
from protein_shape import topology as T
from protein_shape.support import guarded_use, load_credits, supporters

SEED = 122
N_PERM, N_BOOT = 20000, 4000
N_SIM, SIM_PERM, SIM_BOOT = 1000, 999, 1000
EDIT_SETS = ('E_all', 'E_swap', 'E_adj')
OUT = paths.RESULTS / 'robustness' / 'neighbour_control'
LINES = []


def say(s=''):
    print(s, flush=True)
    LINES.append(s)


# --------------------------------------------------------------- neighbours ----

def edit_images(t, kind):
    o, s = T.parse(t)
    n = len(o)
    out = set()
    for p in range(n):
        sg = list(s)
        sg[p] = '-' if sg[p] == '+' else '+'
        out.add((o, tuple(sg)))
    for p in range(n):
        for q in range(p + 1, n):
            if kind == 'E_adj' and q != p + 1:
                continue
            oo, ss = list(o), list(s)
            oo[p], oo[q] = oo[q], oo[p]
            ss[p], ss[q] = ss[q], ss[p]
            out.add((tuple(oo), tuple(ss)))
    if kind == 'E_all':
        for p in range(n):
            for q in range(n):
                if q != p:
                    oo, ss = list(o), list(s)
                    a, b = oo.pop(p), ss.pop(p)
                    oo.insert(q, a)
                    ss.insert(q, b)
                    out.add((tuple(oo), tuple(ss)))
    return out


def neighbour_sets(f, kind):
    idx = {t: i for i, t in enumerate(f.topology)}
    A = ((f.OPP >= 1) & (f.SAME == 0)).to_numpy()
    npar, fd, opp, same, both = (f[c].to_numpy() for c in ('npar', 'fd', 'OPP', 'SAME', 'both'))
    sets = {}
    for i in np.flatnonzero(A):
        hits = set()
        for oo, ss in edit_images(f.topology.iat[i], kind):
            h = {T.minami(a, b) for a, b in ED.orbit(oo, ss)} & idx.keys()
            assert len(h) <= 1
            hits |= h
        nb = sorted(j for j in (idx[u] for u in hits) if j != i and npar[j] == npar[i] and fd[j] == fd[i]
                    and opp[j] == 0 and same[j] == 0 and both[j])
        sets[i] = nb
    return sets


# --------------------------------------------------------------- estimators ----

class Design:
    """Matched sets as a padded member matrix; column 0 is the exposed topology."""

    def __init__(self, sets, sup):
        self.exp = np.array([i for i in sets if sets[i]])
        self.nbr = [sets[i] for i in self.exp]
        self.k = np.array([len(v) for v in self.nbr])
        K = self.k.max() + 1
        self.mem = np.full((len(self.exp), K), -1)
        for r, (i, v) in enumerate(zip(self.exp, self.nbr)):
            self.mem[r, :len(v) + 1] = [i] + v
        self.mask = self.mem >= 0
        self.size = self.k + 1
        self.cluster = self._clusters(sup)

    def _clusters(self, sup):
        m = len(self.exp)
        parent = list(range(m))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        owner = {}
        for r in range(m):
            keys = [('t', int(t)) for t in self.mem[r][self.mask[r]]]
            keys += [('s', s) for t in self.mem[r][self.mask[r]] for s in sup[t]]
            for key in keys:
                if key in owner:
                    parent[find(r)] = find(owner[key])
                else:
                    owner[key] = r
        roots = np.array([find(r) for r in range(m)])
        _, lab = np.unique(roots, return_inverse=True)
        return lab

    def Y(self, y):
        return np.where(self.mask, y[np.maximum(self.mem, 0)], 0.0)


def stat(Y, pos, k):
    """e, c per set for exposed position pos (array, one per set) on member outcomes Y."""
    rows = np.arange(len(k))
    e = Y[rows, pos]
    c = (Y.sum(1) - e) / k
    return e, c


def perm_null(Y, k, size, B, rng):
    """B x 1 statistic T = sum(e - c) under uniform within-set reassignment of the exposed label."""
    pos = (rng.random((B, len(k))) * size).astype(int)
    e = np.take_along_axis(Y, pos.T, 1).T           # B x m
    tot = Y.sum(1)
    return (e - (tot - e) / k).sum(1)


def boot(e, c, cl, B, rng):
    ncl = cl.max() + 1
    Ec, Cc = np.bincount(cl, e, ncl), np.bincount(cl, c, ncl)
    nset = np.bincount(cl, minlength=ncl).astype(float)
    W = rng.multinomial(ncl, np.full(ncl, 1 / ncl), size=B).astype(float)
    num, den, ns = W @ Ec, W @ Cc, W @ nset
    with np.errstate(divide='ignore', invalid='ignore'):
        R = num / den
    D = (num - den) / ns
    return R, D


def pct(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [float('nan')] * 2


def analyse(d, y, sel, rng, graded=None):
    Y = d.Y(y)[sel]
    k, size, cl = d.k[sel], d.size[sel], d.cluster[sel]
    _, cl = np.unique(cl, return_inverse=True)
    e, c = stat(Y, np.zeros(len(k), int), k)
    R = e.sum() / c.sum() if c.sum() else float('nan')
    D = float((e - c).mean())
    Tobs = (e - c).sum()
    null = perm_null(Y, k, size, N_PERM, rng)
    p1 = (np.sum(null <= Tobs + 1e-9) + 1) / (N_PERM + 1)
    p2 = (np.sum(np.abs(null - null.mean()) >= abs(Tobs - null.mean()) - 1e-9) + 1) / (N_PERM + 1)
    Rb, Db = boot(e, c, cl, N_BOOT, rng)
    # cluster concentration: share of used set members that sit in the largest cluster
    used_mem = np.bincount(cl, (Y > 0).sum(1), cl.max() + 1)
    sets_cl = np.bincount(cl)
    r = dict(sets=int(len(k)), exposed_used=f'{int(e.sum())}/{len(k)}', neighbours_weighted_used=float(c.sum()),
             distinct_neighbours=int(len(set(d.mem[sel][:, 1:][d.mask[sel][:, 1:]].tolist()))),
             mean_k=float(k.mean()), R=float(R), R_ci=pct(Rb), D=D, D_ci=pct(Db), p_lower=float(p1),
             p_two=float(p2), clusters=int(cl.max() + 1), largest_cluster_sets=int(sets_cl.max()),
             largest_cluster_used_share=float(used_mem.max() / used_mem.sum()) if used_mem.sum() else 0.0,
             discordant_exposed_only=int(((e > 0) & (c == 0)).sum()),
             discordant_neighbours_only=int(((e == 0) & (c > 0)).sum()))
    if graded is not None:
        r['graded'] = graded_sf(d, sel, graded, rng)
    return r


def graded_sf(d, sel, sup, rng):
    """Graded credits with a superfamily multiplicity bootstrap."""
    v, u = defaultdict(float), defaultdict(float)
    for r in np.flatnonzero(sel):
        for s in sup[d.exp[r]]:
            v[s] += 1
        for j in d.nbr[r]:
            for s in sup[j]:
                u[s] += 1 / d.k[r]
    keys = sorted(set(v) | set(u))
    V, U = np.array([v[s] for s in keys]), np.array([u[s] for s in keys])
    R = V.sum() / U.sum() if U.sum() else float('nan')
    nsf = graded_sf.n_sf
    # multinomial draw over all labelled superfamilies; only the columns that carry credit matter
    W = np.stack([rng.multinomial(nsf, np.full(nsf, 1 / nsf))[:len(keys)] for _ in range(N_BOOT)]).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        Rb = (W @ V) / (W @ U)
    return dict(exposed_credits=float(V.sum()), neighbour_credits=float(U.sum()), R=float(R), R_ci=pct(Rb))


def size_power(d, y, sel, rng):
    Y0 = d.Y(y)[sel]
    k, size, cl = d.k[sel], d.size[sel], d.cluster[sel]
    _, cl = np.unique(cl, return_inverse=True)
    m = len(k)
    rej = {'null_perm_1s': 0, 'null_ci_below': 0, 'null_ci_either': 0, 'null_perm_2s': 0,
           'pow_perm_1s': 0, 'pow_ci_below': 0}
    rows = np.arange(m)
    true_R = []
    for _ in range(N_SIM):
        pos = (rng.random(m) * size).astype(int)
        for tag, inject in (('null', False), ('pow', True)):
            Y = Y0.copy()
            if inject:
                keep = rng.random(m) < 0.5
                Y[rows, pos] = Y[rows, pos] * keep
            e, c = stat(Y, pos, k)
            # bring the pseudo-exposed to column 0 so the permutation helper sees the same layout
            Tobs = (e - c).sum()
            null = perm_null(Y, k, size, SIM_PERM, rng)
            p1 = (np.sum(null <= Tobs + 1e-9) + 1) / (SIM_PERM + 1)
            Rb, _ = boot(e, c, cl, SIM_BOOT, rng)
            lo, hi = pct(Rb)
            if tag == 'null':
                p2 = (np.sum(np.abs(null - null.mean()) >= abs(Tobs - null.mean()) - 1e-9) + 1) / (SIM_PERM + 1)
                rej['null_perm_1s'] += p1 <= 0.05
                rej['null_perm_2s'] += p2 <= 0.05
                rej['null_ci_below'] += hi < 1
                rej['null_ci_either'] += (hi < 1) or (lo > 1)
            else:
                rej['pow_perm_1s'] += p1 <= 0.05
                rej['pow_ci_below'] += hi < 1
                if c.sum():
                    true_R.append(e.sum() / c.sum())
    out = {k_: v / N_SIM for k_, v in rej.items()}
    out['injected_R_mean'] = float(np.mean(true_R)) if true_R else float('nan')
    out['mc_se_at_0.05'] = float(np.sqrt(0.05 * 0.95 / N_SIM))
    return out


# --------------------------------------------------------------------- main ----

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    f = ES.load(paths.OCC_MAIN)
    cnt, sf_size = load_credits(paths.CREDITS_MAIN)
    graded_sf.n_sf = len(sf_size)
    say('=' * 100)
    say('Single-edit topology-neighbour control')
    say('=' * 100)
    say(f'credits loaded {time.time() - t0:.0f}s; {len(sf_size)} labelled superfamilies; frame {len(f)} topologies')

    sup = {}
    used = {}
    for kind in ('agree', 'lenient'):
        sup[kind] = [supporters(cnt[kind].get(c, {}), sf_size) for c in f.koga_code]
        used[kind] = np.array([len(s) > 0 for s in sup[kind]], float)
    gate_a = bool((used['agree'] == (f.L_H.to_numpy() > 0)).all())
    C = json.load(gzip.open(paths.CLASS_RECALL_CENSUS, 'rt'))
    cached = C['chains']['lenient']
    chk = [(int(guarded_use(cached[c], C['sf_size']) > 0) == int(used['lenient'][i]))
           for i, c in enumerate(f.koga_code) if c in cached]
    gate_l = all(chk)
    say(f'check: agreement credit equals L_H>0 on all {len(f)} rows: {gate_a}')
    say(f'check: lenient credit equals class_recall_census on {len(chk)} cached codes: {gate_l}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not (gate_a and gate_l):
        say('credit check failed - stopping')
        open(OUT.with_suffix('.txt'), 'w').write('\n'.join(LINES) + '\n')
        return
    say(f"used in frame: agreement {int(used['agree'].sum())}, lenient {int(used['lenient'].sum())}")

    A = ((f.OPP >= 1) & (f.SAME == 0)).to_numpy()
    fd = f.fd.to_numpy()
    res = dict(meta=dict(seed=SEED, n_perm=N_PERM, n_boot=N_BOOT, n_sim=N_SIM, sim_perm=SIM_PERM,
                         sim_boot=SIM_BOOT, gate_agree=gate_a, gate_lenient=gate_l))
    designs = {}
    say('\nCoverage: exposed topologies with >= 1 eligible neighbour')
    for es in EDIT_SETS:
        sets = neighbour_sets(f, es)
        designs[es] = sets
        cov = {fv: (sum(1 for i in sets if fd[i] == fv and sets[i]), sum(1 for i in sets if fd[i] == fv))
               for fv in (0, 1)}
        say(f'  {es:7s} fd0 {cov[0][0]}/{cov[0][1]}  fd1 {cov[1][0]}/{cov[1][1]}')
        res.setdefault('coverage', {})[es] = {f'fd{k_}': list(v) for k_, v in cov.items()}

    # covered vs uncovered exposed, and the matched-reference contrast on the covered subset (E_all)
    sets = designs['E_all']
    covered = np.zeros(len(f), bool)
    covered[[i for i in sets if sets[i]]] = True
    Rref = ((f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)).to_numpy()
    say('\nCovered vs uncovered exposed (E_all), and the matched-reference MH ratio on the covered subset')
    for cr in ('agree', 'lenient'):
        f['used'] = used[cr].astype(int)
        for lab, m in (('pooled', np.ones(len(f), bool)), ('fd1', fd == 1), ('fd0', fd == 0)):
            cv, un = A & covered & m, A & ~covered & m
            by = ['n', 'stratum', 'fd'] if lab == 'pooled' else ['n', 'stratum']
            cl = ES.cells(f, pd.Series(cv), pd.Series(Rref & m), by)
            rr, v, _, _ = ES.mh(f, cl)
            ci = ES.ci(rr, v)
            r = dict(covered=f'{int(used[cr][cv].sum())}/{int(cv.sum())}',
                     uncovered=f'{int(used[cr][un].sum())}/{int(un.sum())}', ref_mh_rr_covered=rr, ref_ci=ci)
            res.setdefault('covered_check', {}).setdefault(cr, {})[lab] = r
            say(f"  {cr:8s} {lab:6s} covered used {r['covered']:>8s}  uncovered {r['uncovered']:>8s}  "
                f"reference MH RR on covered {rr:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")

    say('\nNeighbour contrast. R = exposed used / neighbour-mean used (sum over sets); D = mean(e - c).')
    say('Intervals: bootstrap over clusters of sets linked by a shared topology or supporting superfamily.')
    say('p: within-set label permutation (one-sided lower; two-sided).')
    hdr = (f"{'edits':7s} {'credit':8s} {'stratum':7s} {'sets':>5s} {'exp used':>9s} {'nbr used':>8s} "
           f"{'R':>6s} {'95% CI':>15s} {'D':>7s} {'95% CI':>17s} {'p1':>8s} {'p2':>8s} {'clus':>5s} {'maxU':>5s}")
    say(hdr)
    say('-' * len(hdr))
    for es in EDIT_SETS:
        for cr in ('agree', 'lenient'):
            d = Design(designs[es], sup[cr])
            dfd = fd[d.exp]
            for lab, sel in (('pooled', np.ones(len(d.exp), bool)), ('fd1', dfd == 1), ('fd0', dfd == 0)):
                r = analyse(d, used[cr], sel, rng, graded=sup[cr] if es == 'E_all' else None)
                res.setdefault('contrast', {}).setdefault(es, {}).setdefault(cr, {})[lab] = r
                say(f"{es:7s} {cr:8s} {lab:7s} {r['sets']:>5d} {r['exposed_used']:>9s} "
                    f"{r['neighbours_weighted_used']:>8.2f} {r['R']:>6.2f} "
                    f"{'[%.2f, %.2f]' % tuple(r['R_ci']):>15s} {r['D']:>7.3f} "
                    f"{'[%.3f, %.3f]' % tuple(r['D_ci']):>17s} {r['p_lower']:>8.4f} {r['p_two']:>8.4f} "
                    f"{r['clusters']:>5d} {r['largest_cluster_used_share']:>5.2f}")
    say('(maxU = share of used set members in the largest cluster.)')

    say('\nGraded outcome (supporting-superfamily credits), E_all, superfamily multiplicity bootstrap')
    for cr in ('agree', 'lenient'):
        for lab in ('pooled', 'fd1', 'fd0'):
            g = res['contrast']['E_all'][cr][lab]['graded']
            say(f"  {cr:8s} {lab:6s} credits exposed {g['exposed_credits']:.0f} vs neighbour-mean "
                f"{g['neighbour_credits']:.2f}  R {g['R']:.2f} [{g['R_ci'][0]:.2f}, {g['R_ci'][1]:.2f}]")

    say(f'\nSize and power, E_all ({N_SIM} pseudo-datasets per cell; exposed label moved uniformly within')
    say(f'sets; {SIM_PERM} permutations and {SIM_BOOT} cluster-bootstrap draws each; MC s.e. at 0.05 = '
        f'{np.sqrt(0.05 * 0.95 / N_SIM):.3f}). Power: used pseudo-exposed set to 0 with probability 0.5.')
    say(f"{'credit':8s} {'stratum':7s} {'perm1s':>7s} {'perm2s':>7s} {'CI<1':>6s} {'CI!1':>6s} | "
        f"{'pow perm':>8s} {'pow CI':>7s} {'inj R':>6s}")
    for cr in ('agree', 'lenient'):
        d = Design(designs['E_all'], sup[cr])
        dfd = fd[d.exp]
        for lab, sel in (('pooled', np.ones(len(d.exp), bool)), ('fd1', dfd == 1), ('fd0', dfd == 0)):
            sp = size_power(d, used[cr], sel, rng)
            res.setdefault('size_power', {}).setdefault(cr, {})[lab] = sp
            say(f"{cr:8s} {lab:7s} {sp['null_perm_1s']:>7.3f} {sp['null_perm_2s']:>7.3f} "
                f"{sp['null_ci_below']:>6.3f} {sp['null_ci_either']:>6.3f} | {sp['pow_perm_1s']:>8.3f} "
                f"{sp['pow_ci_below']:>7.3f} {sp['injected_R_mean']:>6.2f}")

    say(f'\nrun time {(time.time() - t0) / 60:.1f} min')
    json.dump(res, open(OUT.with_suffix('.json'), 'w'), indent=1)
    open(OUT.with_suffix('.txt'), 'w').write('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__.split('\n')[0]).parse_args()
    main()
