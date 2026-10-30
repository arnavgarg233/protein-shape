"""Internal duplication as an account of the ferredoxin-sheet OPP depletion: topologies buildable from two
symmetry-related copies of a half-unit, contrasts within and across that class, and internal two-fold
symmetry of AlphaFold models measured by TM-align of N- against C-terminal parts.

    python scripts/robustness/duplication_alt.py topo
    python scripts/robustness/duplication_alt.py sample --calls CALLS.jsonl   # main census call file
    python scripts/robustness/duplication_alt.py struct                       # fetches AFDB models; needs tmtools
    python scripts/robustness/duplication_alt.py report
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict

import numpy as np
from protein_shape import grammar, paths, s2x
from protein_shape import stats as E
from protein_shape import topology as TP

OUT = paths.RESULTS / 'robustness'
OUT_JSON = OUT / 'duplication_alt.json'
WORK = OUT / 'duplication_work'   # sampled domains, downloaded models and symmetry scores
CALLS = None
OPS = ('T', 'C2y', 'C2z', 'S2x')
SIGMA = {'T': 1, 'C2y': 1, 'C2z': -1, 'S2x': -1}


# ----------------------------------------------------------------------------- topology level
def strands(t):
    order, signs = TP.parse(t)
    n = len(order)
    pos, sg = [0] * (n + 1), [0] * (n + 1)
    for k, (s, x) in enumerate(zip(order, signs)):
        pos[s], sg[s] = k + 1, (1 if x == '+' else -1)
    return n, pos, sg


def repeats(t):
    """All repeats (a, b, m, op) of topology t: segments a..a+m-1 and b..b+m-1 (b >= a+m, m >= 2)."""
    n, pos, sg = strands(t)
    out = []
    for m in range(2, n // 2 + 1):
        for a in range(1, n - 2 * m + 2):
            for b in range(a + m, n - m + 2):
                A = [(pos[a + j], sg[a + j]) for j in range(m)]
                B = [(pos[b + j], sg[b + j]) for j in range(m)]
                rel = B[0][1] * A[0][1]
                if any(B[j][1] * A[j][1] != rel for j in range(m)):
                    continue
                k = B[0][0] - A[0][0]
                c = B[0][0] + A[0][0]
                if all(B[j][0] - A[j][0] == k for j in range(m)):
                    out.append((a, b, m, 'T' if rel == 1 else 'S2x'))
                if all(B[j][0] + A[j][0] == c for j in range(m)):
                    out.append((a, b, m, 'C2y' if rel == 1 else 'C2z'))
    return out


def has_cross(t, a, m):
    """A parallel crossover between consecutive strands inside segment a..a+m-1."""
    return any(c['idx'] + 1 >= a and c['idx'] + 2 <= a + m - 1 for c in s2x.crossovers(t))


def dup_flags(t):
    n = len(TP.parse(t)[0])
    reps = repeats(t)
    full = [r for r in reps if 2 * r[2] >= n - 1]
    core = [r for r in reps if r[1] - (r[0] + r[2]) <= 2 and has_cross(t, r[0], r[2]) and has_cross(t, r[1], r[2])]
    out = dict(D_full=int(bool(full)), D_core=int(bool(core)), D_any=int(bool(reps)),
               D_full_C2z=int(any(r[3] == 'C2z' for r in full)), D_core_C2z=int(any(r[3] == 'C2z' for r in core)),
               D_any_C2z=int(any(r[3] == 'C2z' for r in reps)),
               full_ops=''.join(sorted({r[3][:3] + ',' for r in full})).rstrip(','))
    out['reps_full'] = full
    out['reps_core'] = core
    return out


def opp_location(t, reps):
    """For each OPP pair (S2x relation, crossing spans), where it sits relative to the first repeat:
    'within' one copy, 'across' copies, or 'extra' (involves a strand outside both copies)."""
    par = s2x.crossovers(t)
    a, b, m, _ = reps[0]
    segA, segB = set(range(a, a + m)), set(range(b, b + m))

    def home(c):  # crossover joins strands idx+1 -> idx+2 (sequence numbers)
        s1, s2 = c['idx'] + 1, c['idx'] + 2
        if {s1, s2} <= segA:
            return 'A'
        if {s1, s2} <= segB:
            return 'B'
        return 'X'
    locs = []
    for i, x in enumerate(par):
        for y in par[i + 1:]:
            if s2x.relation(x, y) == 'S2x' and grammar.span_relation(x, y) == 'crossing':
                hx, hy = home(x), home(y)
                locs.append('extra' if 'X' in (hx, hy) else ('within' if hx == hy else 'across'))
    return locs


def selftest():
    # ferredoxin beta-alpha-beta-beta-alpha-beta, sheet order 4132: C2z repeat of halves 1-2 / 3-4
    f = TP.minami(*TP.parse('4132-+-+'))
    reps = repeats(f)
    assert (1, 3, 2, 'C2z') in reps, reps
    # Rossmann-like 321456 all parallel: T repeat 1-3 / 4-6 is absent (3,2,1 -> 4,5,6 is C2y), check C2y
    r = repeats('3|2|1|4|5|6:++++++')
    assert (1, 4, 3, 'C2y') in r, r
    # every repeat maps each in-copy crossover onto a crossover of relation g
    for n in (5, 6, 7):
        for o, sg in list(TP.enumerate_koga(n))[::97]:
            t = TP.minami(o, sg)
            par = {c['idx']: c for c in s2x.crossovers(t)}
            for a, b, m, op in repeats(t):
                for j in range(m - 1):
                    ia, ib = a + j - 1, b + j - 1
                    if ia in par:
                        assert ib in par and s2x.relation(par[ia], par[ib]) == op, (t, a, b, m, op)
    print('selftest ok')


def mh_ratio_diff(r1, r2):
    """Ratio of two independent MH RRs with log-scale interval from their GR variances."""
    if not (r1['mh_rr'] > 0 and r2['mh_rr'] > 0):
        return dict(ratio=float('nan'), ci=[float('nan')] * 2)
    v1 = ((np.log(r1['mh_rr_ci'][1]) - np.log(r1['mh_rr'])) / 1.96) ** 2
    v2 = ((np.log(r2['mh_rr_ci'][1]) - np.log(r2['mh_rr'])) / 1.96) ** 2
    est = r1['mh_rr'] / r2['mh_rr']
    from scipy.stats import norm
    z = np.log(est) / np.sqrt(v1 + v2)
    return dict(ratio=float(est), ci=[float(np.exp(np.log(est) - 1.96 * np.sqrt(v1 + v2))),
                                      float(np.exp(np.log(est) + 1.96 * np.sqrt(v1 + v2)))],
                p_two=float(2 * norm.sf(abs(z))))


def cmd_topo():
    selftest()
    E.N_PERM = 4000
    rng = np.random.default_rng(E.SEED)
    res = {'meta': dict(arms='exposed vs matched reference', strata='n x npar', outcome='used = L_H > 0', n_perm=E.N_PERM,
                        n_boot=E.N_BOOT, seed=E.SEED)}
    for tag, name in [('main', paths.OCC_MAIN), ('replication', paths.OCC_REPLICATION)]:
        f = E.load(name)
        fl = {t: dup_flags(t) for t in f.topology}
        for k in ('D_full', 'D_core', 'D_any', 'D_full_C2z', 'D_core_C2z', 'D_any_C2z'):
            f[k] = [fl[t][k] for t in f.topology]
        A = (f.OPP >= 1) & (f.SAME == 0)
        R = (f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)
        fd1 = f.fd == 1
        out = {}
        if tag == 'main':
            # counts (all fd1 arm topologies, and the cells used by the MH comparison)
            cnt = {}
            for arm, msk in [('opp', A & fd1), ('reference', R & fd1), ('fd1_all', fd1), ('fd0_opp', A & ~fd1),
                             ('fd0_reference', R & ~fd1)]:
                g = f[msk]
                cnt[arm] = {k: dict(n_compatible=int(g[k].sum()), n=int(len(g)),
                                    used_compatible=int(g.used[g[k] == 1].sum()),
                                    used_noncompatible=int(g.used[g[k] == 0].sum()),
                                    by_n={int(n): [int(x[k].sum()), int(len(x))] for n, x in g.groupby('n')})
                            for k in ('D_full', 'D_core', 'D_any', 'D_full_C2z', 'D_core_C2z', 'D_any_C2z')}
            out['counts'] = cnt
            # where the OPP pair sits in D_full-compatible OPP topologies
            loc = Counter()
            ops = Counter()
            for t in f.topology[A & fd1 & (f.D_full == 1)]:
                reps = fl[t]['reps_full']
                ops[reps[0][3]] += 1
                loc[tuple(sorted(Counter(opp_location(t, reps)).items()))] += 1
            out['opp_D_full_pair_location'] = {str(k): v for k, v in loc.items()}
            out['opp_D_full_first_repeat_op'] = dict(ops)
            ops_ref = Counter(fl[t]['reps_full'][0][3] for t in f.topology[R & fd1 & (f.D_full == 1)])
            out['ref_D_full_first_repeat_op'] = dict(ops_ref)
            # examples
            out['examples_opp_D_full_used'] = [dict(topology=t, reps=fl[t]['reps_full'])
                                               for t in f.topology[A & fd1 & (f.D_full == 1) & (f.used == 1)]]
        print(f'--- {tag}')
        defs = ('D_full', 'D_core') if tag == 'main' else ('D_full',)
        for k in defs:
            d = f[k] == 1
            r1 = E.contrast(f, A & fd1 & d, R & fd1 & d, ['n', 'stratum'], rng, f'fd1 OPP vs ref | {k}=1')
            r0 = E.contrast(f, A & fd1 & ~d, R & fd1 & ~d, ['n', 'stratum'], rng, f'fd1 OPP vs ref | {k}=0')
            adj = E.contrast(f, A & fd1, R & fd1, ['n', 'stratum', k], rng, f'fd1 OPP vs ref, strata n x npar x {k}')
            dref = E.contrast(f, R & fd1 & d, R & fd1 & ~d, ['n', 'stratum'], rng, f'fd1 reference: {k}=1 vs 0')
            dopp = (E.contrast(f, A & fd1 & d, A & fd1 & ~d, ['n', 'stratum'], rng, f'fd1 OPP: {k}=1 vs 0')
                    if (A & fd1 & d).sum() else None)
            out[k] = dict(within_D1=r1, within_D0=r0, ratio_D1_over_D0=mh_ratio_diff(r1, r0), adjusted=adj,
                          ref_D1_vs_D0=dref, opp_D1_vs_D0=dopp)
        base = E.contrast(f, A & fd1, R & fd1, ['n', 'stratum'], rng, 'fd1 OPP vs ref')
        out['baseline'] = base
        res[tag] = out
        if tag == 'main':
            res['_flags'] = {t: {k: v for k, v in fl[t].items() if not k.startswith('reps')}
                             for t in f.topology[fd1 & (A | R)]}
    OUT.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(OUT_JSON)) if OUT_JSON.exists() else {}
    prev['topology'] = res
    json.dump(prev, open(OUT_JSON, 'w'), indent=1, default=str)


# ----------------------------------------------------------------------------- structure level
MODES = ('strict', 'lenient', 'lenient_noterm')
K_PER_UNIT = 3
K_CAND = 25  # candidates kept per unit: many census v4 models are no longer served by AFDB (HTTP 403)


def targets():
    """Used fd1 OPP / reference codes and their guarded superfamilies (agree credit)."""
    C = json.load(gzip.open(paths.CLASS_RECALL_CENSUS))
    f = E.load(paths.OCC_MAIN)
    A = (f.OPP >= 1) & (f.SAME == 0)
    R = (f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)
    g = f[(f.fd == 1) & (A | R)].copy()
    g['arm'] = np.where(A[g.index], 'opp', 'reference')
    sfs, mism = {}, 0
    for code, lh in zip(g.koga_code, g.L_H):
        ch = C['chains']['agree'].get(code, {})
        s = sorted(x for x, k in ch.items() if k >= max(2, 0.01 * C['sf_size'][x]))
        mism += len(s) != lh
        if s:
            sfs[code] = s
    return g, sfs, mism


def cmd_sample():
    g, sfs, mism = targets()
    print(f'used fd1 arm codes {len(sfs)}; guarded-sf count mismatches vs L_H: {mism}', flush=True)
    want = {(c, s) for c, ss in sfs.items() for s in ss}
    best = defaultdict(list)  # (code, sf) -> [(plddt, chain, ted_id, chopping)]
    n = 0
    for line in open(CALLS):
        n += 1
        if n % 200000 == 0:
            print(n, flush=True)
        if '"ok": true' not in line:
            continue
        r = json.loads(line)
        sf = r['cath_label']
        agree = set.intersection(*(set(r['per_mode'][m]) for m in MODES))
        for c in agree:
            if (c, sf) in want:
                L = best[(c, sf)]
                if any(x[1] == r['chain'] for x in L):
                    continue
                L.append((float(r['plddt']), r['chain'], r['ted_id'], r['chopping']))
                L.sort(reverse=True)
                del L[K_CAND:]
    arm = dict(zip(g.koga_code, g.arm))
    out = [dict(code=c, sf=s, arm=arm[c], plddt=p, chain=ch, ted_id=t, chopping=cp)
           for (c, s), L in best.items() for p, ch, t, cp in L]
    WORK.mkdir(parents=True, exist_ok=True)
    json.dump(dict(units=sorted(map(list, want)), domains=out, mismatch=mism), open(WORK / 'sample.json', 'w'))
    print(f'units {len(want)}, units found {len(best)}, domains {len(out)}', flush=True)


AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
       'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
       'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}


def fetch_ca(chain, chopping):
    import urllib.request
    mdir = WORK / 'models'
    mdir.mkdir(parents=True, exist_ok=True)
    acc = chain.split('-')[1]
    fn = mdir / f'{acc}.pdb.gz'
    if not fn.exists():
        txt = None
        for v in ('6', '4'):
            try:
                with urllib.request.urlopen(f'https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{v}.pdb',
                                            timeout=60) as h:
                    txt = h.read()
                break
            except Exception:
                continue
        if txt is None:
            return None
        with gzip.open(fn, 'wb') as fh:
            fh.write(txt)
    segs = [tuple(int(x) for x in s.split('-')) for s in chopping.split('_')]
    xyz, seq = [], []
    for ln in gzip.open(fn, 'rt'):
        if ln.startswith('ATOM') and ln[12:16].strip() == 'CA':
            rn = int(ln[22:26])
            if any(a <= rn <= b for a, b in segs):
                xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
                seq.append(AA3.get(ln[17:20], 'X'))
    return np.array(xyz), ''.join(seq)


def sym_score(X1, s1, X2=None, s2=None):
    """Max over split fractions 0.30..0.70 of TM(N part, C part); returns (by longer, by shorter, frac).
    With a second domain, the N part comes from the first and the C part from the second (null)."""
    from tmtools import tm_align
    same = X2 is None
    if same:
        X2, s2 = X1, s1
    L1, L2 = len(X1), len(X2)
    best = (0.0, 0.0, 0.0)
    for fr in np.arange(0.30, 0.7001, 2.0 / max(L1, 40)):
        a, b = int(round(fr * L1)), int(round(fr * L2))
        P, Q = X1[:a], X2[b:]
        if len(P) < 20 or len(Q) < 20:
            continue
        r = tm_align(P, Q, s1[:a], s2[b:])
        lo = r.tm_norm_chain1 if len(P) >= len(Q) else r.tm_norm_chain2
        hi = r.tm_norm_chain2 if len(P) >= len(Q) else r.tm_norm_chain1
        if lo > best[0]:
            best = (float(lo), float(hi), float(fr))
    return best


def _work(d):
    try:
        got = fetch_ca(d['chain'], d['chopping'])
        if got is None or len(got[0]) < 50:
            return dict(d, status='no_model' if got is None else 'short')
        X, s = got
        sc = sym_score(X, s)
        return dict(d, status='ok', L=len(X), sym=sc[0], sym_short=sc[1], split=sc[2])
    except Exception as e:  # noqa: BLE001
        return dict(d, status=f'err {e}')


def cmd_struct():
    from concurrent.futures import ProcessPoolExecutor
    S = json.load(open(WORK / 'sample.json'))
    units = defaultdict(list)
    for d in S['domains']:
        units[(d['code'], d['sf'])].append(d)
    with ProcessPoolExecutor(3) as ex:
        res = [r for rs in ex.map(_unit, list(units.values()), chunksize=2) for r in rs]
    print(Counter(r['status'] for r in res), flush=True)
    # null: N part of one domain vs C part of another, 300 random pairs of ok domains
    ok = [r for r in res if r['status'] == 'ok']
    rng = np.random.default_rng(122)
    pairs = [tuple(rng.choice(len(ok), 2, replace=False)) for _ in range(300)]
    with ProcessPoolExecutor(3) as ex:
        null = list(ex.map(_null, [(ok[i], ok[j]) for i, j in pairs], chunksize=10))
    json.dump(dict(domains=res, null=null), open(WORK / 'struct.json', 'w'))
    print('null median', np.median(null), 'p95', np.percentile(null, 95), flush=True)


def _unit(cands):
    out, n_ok = [], 0
    for d in sorted(cands, key=lambda x: -x['plddt']):
        r = _work(d)
        out.append(r)
        n_ok += r['status'] == 'ok'
        if n_ok == K_PER_UNIT:
            break
    return out


def _null(pair):
    a, b = pair
    Xa, sa = fetch_ca(a['chain'], a['chopping'])
    Xb, sb = fetch_ca(b['chain'], b['chopping'])
    return sym_score(Xa, sa, Xb, sb)[0]


def fmt(r):
    if r is None:
        return 'n/a'
    return (f"{r['arm_a']} vs {r['arm_b']}  RR {r['mh_rr']:.2f} [{r['mh_rr_ci'][0]:.2f}, {r['mh_rr_ci'][1]:.2f}] "
            f"boot [{r['mh_rr_boot_ci'][0]:.2f}, {r['mh_rr_boot_ci'][1]:.2f}]  p1 {r['p_perm_lower']:.4f}  strata {r['strata']}")


def cmd_report():
    from scipy.stats import fisher_exact, mannwhitneyu
    J = json.load(open(OUT_JSON))
    T = J['topology']
    L = []
    p = L.append
    p('Internal duplication as an alternative account of the fd1 OPP depletion')
    p('Arms as the matched comparison: MH risk ratio, Greenland-Robins 95% interval, chain-reversal-class')
    p('bootstrap, one-sided within-cell permutation p (4,000), strata n x npar.')
    p('')
    p('=' * 100)
    p('1. TOPOLOGY LEVEL')
    p('=' * 100)
    p('Repeat: two contiguous sequence segments of m >= 2 strands related by a sheet symmetry g (T, S2x: p -> p+k;')
    p('C2y, C2z: p -> c-p; strand orientation flipped for S2x and C2z). D_full: 2m >= n-1 (two copies of a half-unit,')
    p('at most one extra strand). D_core: copies separated by <= 2 strands, each with its own parallel crossover.')
    p('D_any: any repeat. Suffix _C2z: the repeat operation is the ferredoxin pseudo-twofold.')
    p('')
    p('1a. Counts (all frame topologies in the arm; used = L_H > 0)')
    p(f"{'definition':12s} {'arm':15s} {'compatible/n':>14s} {'frac':>6s}  used|D=1  used|D=0   by n (5/6/7)")
    for k in ('D_full', 'D_full_C2z', 'D_core', 'D_core_C2z', 'D_any', 'D_any_C2z'):
        for arm in ('opp', 'reference', 'fd0_opp', 'fd0_reference'):
            c = T['main']['counts'][arm][k]
            bn = ' '.join(f"{v[0]}/{v[1]}" for _, v in sorted(c['by_n'].items()))
            p(f"{k:12s} {arm:15s} {c['n_compatible']:>6d}/{c['n']:<7d} {c['n_compatible'] / c['n']:6.3f}  "
              f"{c['used_compatible']:>3d}/{c['n_compatible']:<4d}  {c['used_noncompatible']:>3d}/{c['n'] - c['n_compatible']:<4d}  {bn}")
        p('')
    p(f"Repeat operation of D_full OPP topologies (first repeat found): {T['main']['opp_D_full_first_repeat_op']}")
    p(f"Repeat operation of D_full reference topologies: {T['main']['ref_D_full_first_repeat_op']}")
    p(f"Where the OPP pair sits in D_full OPP topologies (counts of topologies by pattern): "
      f"{T['main']['opp_D_full_pair_location']}")
    p(f"Used D_full OPP topologies: {[e['topology'] for e in T['main']['examples_opp_D_full_used']]}")
    p('')
    p('1b. Contrasts, fd1 OPP vs matched reference')
    p(f"  baseline                           {fmt(T['main']['baseline'])}")
    for k in ('D_full', 'D_core'):
        r = T['main'][k]
        p(f"  [{k}] within {k}=1               {fmt(r['within_D1'])}")
        p(f"  [{k}] within {k}=0               {fmt(r['within_D0'])}")
        q = r['ratio_D1_over_D0']
        p(f"  [{k}] ratio of RRs (D1/D0)        {q['ratio']:.2f} [{q['ci'][0]:.2f}, {q['ci'][1]:.2f}]  p_two {q.get('p_two', float('nan')):.3f}")
        p(f"  [{k}] adjusted, strata n x npar x {k}  {fmt(r['adjusted'])}")
        p(f"  [{k}] reference arm, D=1 vs D=0    {fmt(r['ref_D1_vs_D0'])}")
        p(f"  [{k}] OPP arm, D=1 vs D=0          {fmt(r['opp_D1_vs_D0'])}")
        p('')
    # supplementary: the ferredoxin-specific repeat, D_full with g = C2z
    f = E.load(paths.OCC_MAIN)
    fl = T['_flags']
    A = (f.OPP >= 1) & (f.SAME == 0)
    R = (f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)
    fd1 = f.fd == 1
    f['Dz'] = [fl.get(t, {}).get('D_full_C2z', 0) for t in f.topology]
    d = f.Dz == 1
    rng = np.random.default_rng(E.SEED)
    E.N_PERM = 4000
    sup = dict(within_D1=E.contrast(f, A & fd1 & d, R & fd1 & d, ['n', 'stratum'], rng, 'Dz=1'),
               within_D0=E.contrast(f, A & fd1 & ~d, R & fd1 & ~d, ['n', 'stratum'], rng, 'Dz=0'),
               adjusted=E.contrast(f, A & fd1, R & fd1, ['n', 'stratum', 'Dz'], rng, 'adj Dz'),
               ref=E.contrast(f, R & fd1 & d, R & fd1 & ~d, ['n', 'stratum'], rng, 'ref Dz'))
    p('  Supplementary: ferredoxin-specific repeat, D_full_C2z')
    p(f"  [D_full_C2z] within =1              {fmt(sup['within_D1'])}")
    p(f"  [D_full_C2z] within =0              {fmt(sup['within_D0'])}")
    p(f"  [D_full_C2z] adjusted               {fmt(sup['adjusted'])}")
    p(f"  [D_full_C2z] reference arm, 1 vs 0  {fmt(sup['ref'])}")
    p('')
    J['topology']['main']['D_full_C2z_supplementary'] = sup
    p('  Replication census')
    p(f"  baseline                            {fmt(T['replication']['baseline'])}")
    r = T['replication']['D_full']
    p(f"  [D_full] within D_full=1            {fmt(r['within_D1'])}")
    p(f"  [D_full] within D_full=0            {fmt(r['within_D0'])}")
    p(f"  [D_full] adjusted                   {fmt(r['adjusted'])}")
    p(f"  [D_full] reference arm, D=1 vs D=0  {fmt(r['ref_D1_vs_D0'])}")
    p('')

    # ------------------------------------------------------------- structure level
    sp = WORK / 'struct.json'
    if sp.exists():
        St = json.load(open(sp))
        g, sfs, mism = targets()
        flags = T['_flags']
        top = dict(zip(g.koga_code, g.topology))
        ok = [d for d in St['domains'] if d['status'] == 'ok']
        null = np.array(St['null'])
        p('=' * 100)
        p('2. STRUCTURE LEVEL (AlphaFold models of census domains, agree-credited, guarded superfamilies)')
        p('=' * 100)
        p(f"Used fd1 arm topologies {len(sfs)} (guarded superfamily counts equal L_H; mismatches {mism}); "
          f"(topology, superfamily) usage units {sum(len(v) for v in sfs.values())}; up to {K_PER_UNIT} domains per unit by pLDDT.")
        p(f"Domains sampled {len(St['domains'])}; scored {len(ok)}; status {dict(Counter(d['status'][:8] for d in St['domains']))}")
        p('Score: max over split fractions 0.30-0.70 of TM-score(N part, C part), TM-align (tmtools 0.3.0),')
        p('normalised by the longer part. Null: N part of one sampled domain vs C part of another (300 random pairs).')
        p(f"Null: median {np.median(null):.3f}, 95th pct {np.percentile(null, 95):.3f}, 99th pct {np.percentile(null, 99):.3f}, max {null.max():.3f}")
        sc = {a: np.array([d['sym'] for d in ok if d['arm'] == a]) for a in ('opp', 'reference')}
        for a, v in sc.items():
            p(f"  {a:10s} domains {len(v):4d}  median {np.median(v):.3f}  frac >= 0.5 {np.mean(v >= 0.5):.3f}  frac >= 0.45 {np.mean(v >= 0.45):.3f}")
        p(f"  Mann-Whitney OPP vs reference (domain level, two-sided) p = {mannwhitneyu(sc['opp'], sc['reference']).pvalue:.3g}")
        # concordance with the topology definition
        dfull = np.array([flags[top[d['code']]]['D_full'] for d in ok])
        dc2z = np.array([flags[top[d['code']]]['D_full_C2z'] for d in ok])
        v = np.array([d['sym'] for d in ok])
        p(f"  Concordance: D_full=1 domains median {np.median(v[dfull == 1]):.3f} (n {int(dfull.sum())}, frac>=0.5 "
          f"{np.mean(v[dfull == 1] >= 0.5):.3f}); D_full=0 median {np.median(v[dfull == 0]):.3f} (n {int((dfull == 0).sum())}, "
          f"frac>=0.5 {np.mean(v[dfull == 0] >= 0.5):.3f}); MW p {mannwhitneyu(v[dfull == 1], v[dfull == 0]).pvalue:.3g}")
        if dc2z.sum():
            p(f"  D_full_C2z=1 domains median {np.median(v[dc2z == 1]):.3f} (n {int(dc2z.sum())}, frac>=0.5 {np.mean(v[dc2z == 1] >= 0.5):.3f})")
        out_struct = dict(null=dict(median=float(np.median(null)), p95=float(np.percentile(null, 95))))
        armof = dict(zip(g.koga_code, g.arm))
        allunits = {(c, sf) for c, ss in sfs.items() for sf in ss}
        scored = {(d['code'], d['sf']) for d in ok}
        for a in ('opp', 'reference'):
            tot = sum(1 for c, _ in allunits if armof[c] == a)
            got = sum(1 for c, _ in scored if armof[c] == a)
            p(f"  coverage {a:10s}: usage units with >= 1 scored model {got}/{tot}")
        f = E.load(paths.OCC_MAIN)
        A = (f.OPP >= 1) & (f.SAME == 0)
        R = (f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)
        fd1 = f.fd == 1
        rng = np.random.default_rng(E.SEED)
        E.N_PERM = 4000
        thrs = [('primary', 0.5), ('sensitivity', 0.45),
                ('null 99th pct', float(np.percentile(null, 99))),
                ('null 95th pct', float(np.percentile(null, 95)))]
        for lab, thr in thrs:
            unit = defaultdict(list)
            for d in ok:
                unit[(d['code'], d['sf'])].append(d['sym'] >= thr)
            usym = {u: bool(np.mean(x) > 0.5) for u, x in unit.items()}
            tab = {}
            for a in ('opp', 'reference'):
                us = [v for (c, _), v in usym.items() if armof[c] == a]
                tab[a] = (int(sum(us)), len(us))
            orr, pf = fisher_exact([[tab['opp'][0], tab['opp'][1] - tab['opp'][0]],
                                    [tab['reference'][0], tab['reference'][1] - tab['reference'][0]]])
            dfr = {a: float(np.mean([d['sym'] >= thr for d in ok if d['arm'] == a])) for a in ('opp', 'reference')}
            p('')
            p(f"  threshold {thr:.3f} ({lab}): domains symmetric OPP {dfr['opp']:.3f}, reference {dfr['reference']:.3f}; "
              f"usage units symmetric (majority of models) OPP {tab['opp'][0]}/{tab['opp'][1]}, reference "
              f"{tab['reference'][0]}/{tab['reference'][1]}; Fisher OR {orr:.2f}, p {pf:.3g}")
            rec = dict(threshold=thr, label=lab, domain_frac=dfr, units=tab, fisher_or=orr, fisher_p=pf)
            # restricted use: only non-symmetric superfamilies count. Unscored units: (a) counted as
            # non-symmetric, (b) counted as symmetric (removes every unverified superfamily)
            for var, unk in (('unscored_as_nonsym', False), ('unscored_as_sym', True)):
                ns = {c: sum(1 for sf in ss if not usym.get((c, sf), unk)) for c, ss in sfs.items()}
                only = [c for c in sfs if ns[c] == 0]
                f['used'] = [int(ns.get(c, 0) > 0) for c in f.koga_code]
                rr = E.contrast(f, A & fd1, R & fd1, ['n', 'stratum'], rng, f'{var} {thr:.3f}')
                p(f"    restricted use ({var}): topologies losing use {len(only)} (OPP {sum(armof[c] == 'opp' for c in only)}); "
                  f"fd1 OPP vs reference {fmt(rr)}")
                rec[var] = dict(topologies_losing_use=len(only), contrast=rr)
            out_struct[f'thr_{thr:.3f}'] = rec
        J['structure'] = out_struct
        J['structure_domains'] = St['domains']
        json.dump(J, open(OUT_JSON, 'w'), indent=1, default=str)
    (OUT / 'duplication_alt.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L[-30:]))


def main():
    global CALLS
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', nargs='?', default='topo', choices=['topo', 'selftest', 'sample', 'struct', 'report'])
    ap.add_argument('--calls', help='main census call file, JSON lines (sample)')
    a = ap.parse_args()
    CALLS = a.calls
    {'topo': cmd_topo, 'selftest': selftest, 'sample': cmd_sample, 'struct': cmd_struct, 'report': cmd_report}[a.cmd]()


if __name__ == '__main__':
    main()
