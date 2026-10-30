"""Evolutionary accessibility: the number of used one-strand-smaller topologies that reach a topology by one
terminal hairpin or crossover addition, and the matched OPP-versus-reference contrasts with it held fixed.

Run: python scripts/robustness/accessibility.py
"""
import argparse
import gzip
import json
import multiprocessing as mp
import os
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from protein_shape import edits, paths  # noqa: E402
from protein_shape import stats as E  # noqa: E402
from protein_shape import topology as TP  # noqa: E402
from protein_shape.support import guarded_use, load_credits  # noqa: E402

OUT = paths.RESULTS / 'robustness' / 'accessibility'
WORKERS = 2
BY2, BY3 = ['n', 'stratum'], ['n', 'stratum', 'fd']
BINS = ['0', '1', '2', '3+']
LINES = []


def say(s=''):
    print(s, flush=True)
    LINES.append(s)


# --------------------------------------------------------------------------------------- steps

def steps(t):
    """(kind, parent Minami string) for each terminal strand of t at a sheet edge; kind a = hairpin, b = crossover,
    arch = antiparallel connection that jumps over at least one strand (not in the primary definition)."""
    order, signs = TP.parse(t)
    n = len(order)
    out = []
    for et, parent in edits.deletions(t):
        s, c = (1, 2) if et == 'terminal_N' else (n, n - 1)
        ps, pc = order.index(s), order.index(c)
        kind = 'b' if signs[ps] == signs[pc] else ('a' if abs(ps - pc) == 1 else 'arch')
        out.append((kind, parent))
    return out


def code(t):
    return TP.koga(*TP.parse(t))


# --------------------------------------------------------------------------------------- estimators

def det(f, a, b, by, y=None):
    """Counts, MH risk ratio and Greenland-Robins interval (the deterministic part of effect_sizes.contrast)."""
    if y is not None:
        f = f.copy()
        f['used'] = y
    cl = E.cells(f, a, b, by)
    yy = f.used.to_numpy()
    au = int(sum(yy[ia].sum() for ia, _ in cl))
    an = int(sum(len(ia) for ia, _ in cl))
    bu = int(sum(yy[ib].sum() for _, ib in cl))
    bn = int(sum(len(ib) for _, ib in cl))
    rr, v, _, _ = E.mh(f, cl)
    return dict(arm_a=f'{au}/{an}', arm_b=f'{bu}/{bn}', strata=len(cl), mh_rr=rr, mh_rr_ci=E.ci(rr, v))


def _job(args):
    key, f, a, b, by, seed, name = args
    rng = np.random.default_rng(seed)
    return key, E.contrast(f, pd.Series(a), pd.Series(b), by, rng, name)


def fmt(r, full=True):
    s = f"{r['arm_a']:>9s} vs {r['arm_b']:>9s}  strata {r['strata']:>3d}  RR {r['mh_rr']:.2f} " \
        f"[{r['mh_rr_ci'][0]:.2f}, {r['mh_rr_ci'][1]:.2f}]"
    if full and 'mh_rr_boot_ci' in r:
        s += f"  boot [{r['mh_rr_boot_ci'][0]:.2f}, {r['mh_rr_boot_ci'][1]:.2f}]  p1 {r['p_perm_lower']:.4f}"
    return s


def verdict(base, adj):
    """VANISHES / ATTENUATED / PERSISTS, from the log RR of the adjusted row against the baseline."""
    lb, la = np.log(base['mh_rr']), np.log(adj['mh_rr'])
    incl1 = adj['mh_rr_ci'][0] <= 1 <= adj['mh_rr_ci'][1]
    frac = la / lb
    if incl1 and abs(la) <= 0.5 * abs(lb):
        return 'VANISHES', frac
    if abs(la) < 0.75 * abs(lb):
        return 'ATTENUATED', frac
    return 'PERSISTS', frac


# --------------------------------------------------------------------------------------- main

def main():
    t0 = time.time()
    say('=' * 110)
    say('Is the OPP depletion in ferredoxin-type sheets explained by evolutionary accessibility? (seed 122, main census)')
    say('=' * 110)

    # ---- frame, steps, gates on the step definition
    f0 = E.load(paths.OCC_MAIN)
    raw = pd.read_csv(paths.OCC_MAIN, usecols=['topology', 'koga_code', 'n', 'L_H'])
    f4 = pd.read_csv(paths.DATA / 'frame4.csv')
    g_str = all(code(t) == k for t, k in zip(raw.topology, raw.koga_code)) and \
        all(code(t) == k for t, k in zip(f4.topology, f4.koga_code))
    say(f'G0 Koga code from the Minami string reproduces koga_code on all {len(raw)} occupancy rows and {len(f4)} 4-strand rows: {g_str}')
    st = {t: steps(t) for t in f0.topology}
    rt_ok = True
    for t, lst in st.items():
        for _, p in lst:
            rt_ok &= t in {u for _, u in edits.additions(p)}
    say(f'G0b every counted step round-trips (T is among edits.additions(P)) on all {sum(len(v) for v in st.values())} '
        f'terminal deletions of the {len(f0)} frame topologies: {rt_ok}')
    parent_codes = {t: [(k, code(p)) for k, p in lst] for t, lst in st.items()}
    codes = set(raw.koga_code[raw.n <= 7]) | set(f4.koga_code)

    # ---- credits
    cnt, size = load_credits(paths.CREDITS_MAIN)
    say(f'credits loaded, {time.time() - t0:.0f}s: {len(size)} labelled superfamilies')
    used = {cr: {c: guarded_use(cnt[cr].get(c, {}), size) > 0 for c in codes} for cr in ('agree', 'lenient')}
    g1 = all(int(used['agree'][c]) == int(lh > 0) for c, lh in zip(raw.koga_code, raw.L_H))
    say(f'G1 agreement credit reproduces L_H > 0 on all {len(raw)} n=5-7 rows of the occupancy file: {g1}')
    C = json.load(gzip.open(paths.CLASS_RECALL_CENSUS, 'rt'))
    cached = C['chains']['lenient']
    g2 = all((guarded_use(cached[c], C['sf_size']) > 0) == used['lenient'][c] for c in cached)
    say(f'G2 lenient credit equals class_recall_census on all {len(cached)} cached codes: {g2}')
    n4 = f4.koga_code
    say(f'4-strand parents used under census credit: lenient {sum(used["lenient"][c] for c in n4)}/96, '
        f'agreement {sum(used["agree"][c] for c in n4)}/96; frame4.csv observed {int(f4.observed.sum())}/96')
    obs4 = dict(zip(f4.koga_code, f4.observed.astype(bool)))
    say('Used (n-1)-strand parents, lenient credit: ' + ', '.join(
        f'n={n}: {sum(used["lenient"][c] for c in raw.koga_code[raw.n == n - 1])}/{(raw.n == n - 1).sum()}'
        for n in (6, 7)) + ', n=5 parents (4 strands): see above')

    # ---- frames per credit
    fr = {}
    for cr in ('lenient', 'agree'):
        f = f0.copy()
        f['used'] = [int(used[cr][c]) for c in f.koga_code]
        for tag, kinds, p4 in [('acc', ('a', 'b'), False), ('accS1', ('a', 'b', 'arch'), False), ('accS2', ('a', 'b'), True)]:
            vals = []
            for t, n in zip(f.topology, f.n):
                ps = {c for k, c in parent_codes[t] if k in kinds}
                if p4 and n == 5:
                    vals.append(sum(obs4[c] for c in ps))
                else:
                    vals.append(sum(used[cr][c] for c in ps))
            f[tag] = vals
            f[tag + 'b'] = np.minimum(f[tag], 3)
        fr[cr] = f
    f = fr['lenient']
    A = (f.OPP >= 1) & (f.SAME == 0)
    R = (f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)
    say(f'frame {len(f)} topologies; exposed {int(A.sum())}, reference {int(R.sum())} (arm rules of fd_split.arms)')
    say('kinds among all terminal edge deletions of the frame: ' + ', '.join(
        f'{k} {sum(1 for lst in parent_codes.values() for kk, _ in lst if kk == k)}' for k in ('a', 'b', 'arch')))

    res = dict(gates=dict(G0=g_str, G0b=rt_ok, G1=g1, G2=g2))

    # ---- (i) distribution per arm
    for cr in ('lenient', 'agree'):
        f = fr[cr]
        say()
        say('=' * 110)
        say(f'(i) ACCESSIBILITY DISTRIBUTION PER ARM, {cr} credit (primary: lenient)' if cr == 'lenient'
            else f'(i) ACCESSIBILITY DISTRIBUTION PER ARM, {cr} credit (secondary)')
        say('=' * 110)
        say(f"{'stratum':>8s} {'arm':>10s} {'N':>5s} {'mean':>6s} {'median':>6s} {'acc=0':>6s} {'1':>5s} {'2':>5s} {'3+':>5s} "
            f"{'share>=1':>9s} {'used':>9s}")
        dist = {}
        for fdv in (1, 0):
            for nm, m in (('exposed', A), ('reference', R)):
                g = f[m & (f.fd == fdv)]
                b = [int((g.accb == i).sum()) for i in range(4)]
                dist[f'fd{fdv}_{nm}'] = dict(N=len(g), mean=float(g.acc.mean()), median=float(g.acc.median()), bins=b,
                                             share_ge1=float((g.acc >= 1).mean()), used=int(g.used.sum()))
                say(f"{'fd' + str(fdv):>8s} {nm:>10s} {len(g):5d} {g.acc.mean():6.2f} {g.acc.median():6.1f} "
                    f"{b[0]:6d} {b[1]:5d} {b[2]:5d} {b[3]:5d} {(g.acc >= 1).mean():9.2f} {int(g.used.sum()):4d}/{len(g):<4d}")
        res[f'dist_{cr}'] = dist
        ge1 = (f.acc >= 1).astype(int).to_numpy()
        rr = {}
        for lab, by, mask in (('fd1', BY2, f.fd == 1), ('fd0', BY2, f.fd == 0), ('pooled', BY3, f.fd >= 0)):
            rr[lab] = det(f, A & mask, R & mask, by, y=ge1)
            say(f"share with acc>=1, exposed vs reference, MH RR within {'x'.join(by)} {lab:>6s}: {fmt(rr[lab], False)}")
        res[f'share_ge1_rr_{cr}'] = rr
        for i, nn in enumerate((5, 6, 7)):
            parts = []
            for fdv in (1, 0):
                for nm, m in (('exp', A), ('ref', R)):
                    g = f[m & (f.fd == fdv) & (f.n == nn)]
                    parts.append(f'fd{fdv} {nm} n={len(g)} mean {g.acc.mean() if len(g) else float("nan"):.2f}')
            say(f'  n={nn}: ' + '; '.join(parts))

    # ---- (ii) covariate live among references
    say()
    say('=' * 110)
    say('(ii) IS THE COVARIATE LIVE? Use among REFERENCE topologies by accessibility bin')
    say('=' * 110)
    jobs, k = [], 0

    def add(key, f, a, b, by, name):
        nonlocal k
        k += 1
        jobs.append((key, f, a.to_numpy(), b.to_numpy(), by, E.SEED + k, name))

    for cr in ('lenient', 'agree'):
        f = fr[cr]
        say(f'{cr} credit: use rate among reference topologies by accessibility bin')
        tab = {}
        for fdv in ('fd1', 'fd0', 'all'):
            m = R & (f.fd == int(fdv[2])) if fdv != 'all' else R
            row = []
            for i in range(4):
                g = f[m & (f.accb == i)]
                row.append(f'{int(g.used.sum()):4d}/{len(g):<4d} ({g.used.mean() if len(g) else float("nan"):.2f})')
            tab[fdv] = [int(f[m & (f.accb == i)].used.sum()) for i in range(4)] + [int((m & (f.accb == i)).sum()) for i in range(4)]
            say(f"  {fdv:>4s}: acc=0 {row[0]}  1 {row[1]}  2 {row[2]}  3+ {row[3]}")
        res[f'ref_rate_by_bin_{cr}'] = tab
        ge1, eq0, ge3 = (f.acc >= 1), (f.acc == 0), (f.acc >= 3)
        live = {}
        for lab, mask, by in (('fd1', f.fd == 1, BY2), ('fd0', f.fd == 0, BY2), ('pooled', f.fd >= 0, BY3)):
            live[f'{lab}_ge1_vs_0'] = det(f, R & ge1 & mask, R & eq0 & mask, by)
            live[f'{lab}_3plus_vs_0'] = det(f, R & ge3 & mask, R & eq0 & mask, by)
            say(f"  reference acc>=1 vs acc=0, {lab:>6s}: {fmt(live[f'{lab}_ge1_vs_0'], False)}")
            say(f"  reference acc 3+ vs acc=0, {lab:>6s}: {fmt(live[f'{lab}_3plus_vs_0'], False)}")
        res[f'live_{cr}'] = live
        add(f'live_pooled_{cr}', f, R & ge1, R & eq0, BY3, f'{cr} reference acc>=1 vs 0, pooled')

    # ---- (iii) contrasts: deterministic baselines, sensitivities, full jobs
    sd_path = paths.TABLES / 'fd_split.json'
    sd = json.load(open(sd_path)) if sd_path.exists() else None
    det_rows = {}
    for cr in ('lenient', 'agree'):
        f = fr[cr]
        rows = {}
        for lab, mask, by in (('fd1', f.fd == 1, BY2), ('fd0', f.fd == 0, BY2), ('pooled', f.fd >= 0, BY3)):
            a, b = A & mask, R & mask
            rows[f'{lab}_base'] = det(f, a, b, by)
            rows[f'{lab}_accbin'] = det(f, a, b, by + ['accb'])
            rows[f'{lab}_ge1'] = det(f, a & (f.acc >= 1), b & (f.acc >= 1), by + ['accb'])
            rows[f'{lab}_S1_accbin'] = det(f, a, b, by + ['accS1b'])
            rows[f'{lab}_S1_ge1'] = det(f, a & (f.accS1 >= 1), b & (f.accS1 >= 1), by + ['accS1b'])
            rows[f'{lab}_S2_accbin'] = det(f, a, b, by + ['accS2b'])
            rows[f'{lab}_S2_ge1'] = det(f, a & (f.accS2 >= 1), b & (f.accS2 >= 1), by + ['accS2b'])
            if lab != 'fd0':
                for tag, aa, bb, byy in (('base', a, b, by), ('accbin', a, b, by + ['accb']),
                                         ('ge1', a & (f.acc >= 1), b & (f.acc >= 1), by + ['accb'])):
                    add(f'{lab}_{tag}_{cr}', f, aa, bb, byy, f'{cr} {lab} OPP vs reference {tag}')
        det_rows[cr] = rows
        if sd is None:
            continue
        key = f'main_{cr}'
        sp = sd[key]['fd1_opp_vs_ref']
        g3 = sp['arm_a'] == rows['fd1_base']['arm_a'] and sp['arm_b'] == rows['fd1_base']['arm_b'] and \
            abs(sp['mh_rr'] - rows['fd1_base']['mh_rr']) < 1e-9
        g3b = sd[key]['fd0_opp_vs_ref']['arm_a'] == rows['fd0_base']['arm_a'] and \
            abs(sd[key]['fd0_opp_vs_ref']['mh_rr'] - rows['fd0_base']['mh_rr']) < 1e-9
        say(f'G3 {cr}: baseline fd1 / fd0 contrasts reproduce fd_split.json counts and MH RR: {g3} / {g3b}')
        res['gates'][f'G3_{cr}'] = bool(g3 and g3b)
    res['det'] = det_rows

    say()
    say(f'running {len(jobs)} full contrasts (bootstrap 4000, permutation 10000) on {WORKERS} workers, {time.time() - t0:.0f}s so far')
    with mp.get_context('fork').Pool(WORKERS) as pool:
        full = dict(pool.imap_unordered(_job, jobs, chunksize=1))
    res['full'] = full

    # ---- report (iii)
    say()
    say('=' * 110)
    say('(iii) OPP vs MATCHED REFERENCE, with accessibility bin added to the strata')
    say('MH risk ratio, Greenland-Robins 95% interval, chain-reversal-class bootstrap 95% interval, one-sided permutation p')
    say('=' * 110)
    verd = {}
    for cr in ('lenient', 'agree'):
        say()
        say(f'--- {cr} credit' + (' (primary)' if cr == 'lenient' else ' (secondary)'))
        for lab, strata in (('fd1', 'n x stratum'), ('pooled', 'n x stratum x fd')):
            say(f'  {lab}  (strata {strata}; "+acc" adds the accessibility bin 0/1/2/3+)')
            for tag, desc in (('base', 'baseline'), ('accbin', '+ accessibility bin'),
                              ('ge1', 'restricted to acc >= 1, + bin')):
                say(f"    {desc:>30s}: {fmt(full[f'{lab}_{tag}_{cr}'])}")
            for tag in ('accbin', 'ge1'):
                v, fr_ = verdict(full[f'{lab}_base_{cr}'], full[f'{lab}_{tag}_{cr}'])
                verd[f'{lab}_{tag}_{cr}'] = dict(verdict=v, log_rr_fraction_retained=float(fr_))
                say(f"    decision rule, {tag:>6s} row vs baseline: {v} (log RR retains {fr_:.2f} of the baseline)")
        f_ = det_rows[cr]
        say('  fd0 (deterministic part only)')
        for tag, desc in (('base', 'baseline'), ('accbin', '+ accessibility bin'), ('ge1', 'restricted to acc >= 1')):
            say(f"    {desc:>30s}: {fmt(f_['fd0_' + tag], False)}")
        say('  sensitivities, deterministic part only')
        for lab in ('fd1', 'pooled'):
            for tag, desc in (('S1_accbin', 'S1 arches counted, + bin'), ('S1_ge1', 'S1 restricted to acc >= 1'),
                              ('S2_accbin', 'S2 frame4 observed (n=5), + bin'), ('S2_ge1', 'S2 restricted to acc >= 1')):
                say(f"    {lab:>6s} {desc:>34s}: {fmt(f_[lab + '_' + tag], False)}")
    res['verdicts'] = verd
    lv = full['live_pooled_lenient']
    say()
    say(f"Covariate check (lenient, pooled, reference acc>=1 vs 0): {fmt(lv)}")

    # ---- reading
    say()
    say('=' * 110)
    say('READING (decision rule, lenient primary)')
    say('=' * 110)
    vf, vp = verd['fd1_accbin_lenient'], verd['pooled_accbin_lenient']
    rf, rp = verd['fd1_ge1_lenient'], verd['pooled_ge1_lenient']
    b1, bp = full['fd1_base_lenient'], full['pooled_base_lenient']
    a1, ap = full['fd1_accbin_lenient'], full['pooled_accbin_lenient']
    e1, ep = full['fd1_ge1_lenient'], full['pooled_ge1_lenient']
    live_ok = lv['mh_rr_ci'][0] > 1
    dl = res['dist_lenient']
    say(f"Covariate live among references: {'yes' if live_ok else 'no'} (reference acc>=1 vs 0 RR {lv['mh_rr']:.2f} "
        f"[{lv['mh_rr_ci'][0]:.2f}, {lv['mh_rr_ci'][1]:.2f}]).")
    say(f"Exposed vs reference share with acc>=1 in fd1: {dl['fd1_exposed']['share_ge1']:.2f} vs {dl['fd1_reference']['share_ge1']:.2f}; "
        f"mean acc {dl['fd1_exposed']['mean']:.2f} vs {dl['fd1_reference']['mean']:.2f}.")
    say(f"fd1 depletion {b1['mh_rr']:.2f} [{b1['mh_rr_ci'][0]:.2f}, {b1['mh_rr_ci'][1]:.2f}] -> {a1['mh_rr']:.2f} "
        f"[{a1['mh_rr_ci'][0]:.2f}, {a1['mh_rr_ci'][1]:.2f}] with accessibility held fixed ({vf['verdict']}); "
        f"restricted to acc>=1 {e1['mh_rr']:.2f} [{e1['mh_rr_ci'][0]:.2f}, {e1['mh_rr_ci'][1]:.2f}] ({rf['verdict']}).")
    say(f"Pooled depletion {bp['mh_rr']:.2f} [{bp['mh_rr_ci'][0]:.2f}, {bp['mh_rr_ci'][1]:.2f}] -> {ap['mh_rr']:.2f} "
        f"[{ap['mh_rr_ci'][0]:.2f}, {ap['mh_rr_ci'][1]:.2f}] ({vp['verdict']}); restricted to acc>=1 {ep['mh_rr']:.2f} "
        f"[{ep['mh_rr_ci'][0]:.2f}, {ep['mh_rr_ci'][1]:.2f}] ({rp['verdict']}).")
    ag = full['fd1_accbin_agree']
    say(f"Agreement credit, fd1, accessibility held fixed: {ag['mh_rr']:.2f} [{ag['mh_rr_ci'][0]:.2f}, {ag['mh_rr_ci'][1]:.2f}] "
        f"({verd['fd1_accbin_agree']['verdict']}).")

    overall = vf['verdict']
    res['verdict_overall'] = overall

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT.with_suffix('.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    open(OUT.with_suffix('.txt'), 'w').write('\n'.join(LINES) + '\n')
    print(f'done {time.time() - t0:.0f}s')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__.split('\n')[0]).parse_args()
    main()
