"""Lineage overlap between the main and replication censuses, and the replication contrast recomputed on
support from superfamily-topology pairs absent from the main census, calibrated on random halves of the
main census. Superfamilies are matched by exact CATH H-level label.

    python scripts/robustness/replication_lineage.py build --main-calls MAIN.jsonl --rep-calls REP.jsonl
    python scripts/robustness/replication_lineage.py run

build streams both census call files into pair tables (results/robustness/replication_lineage_pairs.pkl.gz).
"""
import argparse
import gzip
import json
import pickle
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from protein_shape import paths
from protein_shape import stats as E

OUT = paths.RESULTS / 'robustness'
MODES = ('strict', 'lenient', 'lenient_noterm')


def half(chain):
    return zlib.crc32(chain.encode()) & 1


def stream(path, codes):
    """pair[(code, sf)] -> set of chains; size[sf] -> set of called chains; per census."""
    pair = defaultdict(set)
    size = defaultdict(set)
    n = 0
    for line in open(path):
        r = json.loads(line)
        if not r.get('ok'):
            continue
        lab = r['cath_label']
        sf = lab if isinstance(lab, str) and lab.count('.') == 3 else None
        if not sf:
            continue
        n += 1
        size[sf].add(r['chain'])
        for c in set.intersection(*(set(r['per_mode'][m]) for m in MODES)):
            if c in codes:
                pair[(c, sf)].add(r['chain'])
    return pair, size, n


def build(files, cache):
    codes = set()
    for name in (paths.OCC_MAIN, paths.OCC_REPLICATION):
        f = pd.read_csv(name)
        f = f[(f.violations == 0) & (f.npar >= 2)]
        codes |= set(f.koga_code)
    out = {}
    for k, p in files.items():
        pair, size, n = stream(p, codes)
        d = dict(n_labelled_records=n,
                 size={s: len(c) for s, c in size.items()},
                 pair={k2: len(v) for k2, v in pair.items()})
        if k == 'main':
            d['size_h'] = [{s: sum(half(x) == h for x in c) for s, c in size.items()} for h in (0, 1)]
            d['pair_h'] = [{k2: sum(half(x) == h for x in v) for k2, v in pair.items()} for h in (0, 1)]
        out[k] = d
        print(k, n, len(pair), len(size), flush=True)
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, 'wb') as fh:
        pickle.dump(out, fh)


# --------------------------------------------------------------------------- analysis

def guarded(pair, size, allow=None):
    """code -> set of superfamilies that support it under max(2, 1% of called chains), optionally only
    pairs for which allow(code, sf) is True.  The guard threshold always uses the full size."""
    out = defaultdict(set)
    for (c, s), k in pair.items():
        if k >= max(2, 0.01 * size[s]) and (allow is None or allow(c, s)):
            out[c].add(s)
    return out


def arms(f):
    A = (f.OPP >= 1) & (f.SAME == 0)
    R = (f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)
    return A, R


def contrasts(f, used, rng, tag, nboot=None, nperm=None):
    """Pooled (n x npar x fd), fd1 and fd0 (n x npar) OPP-vs-reference, same estimators as fd_split."""
    if nboot:
        E.N_BOOT = nboot
    if nperm:
        E.N_PERM = nperm
    g = f.copy()
    g['used'] = used
    A, R = arms(g)
    out = {}
    out['pooled'] = E.contrast(g, A, R, ['n', 'stratum', 'fd'], rng, f'{tag} pooled')
    for fd in (1, 0):
        m = g.fd == fd
        out[f'fd{fd}'] = E.contrast(g, A & m, R & m, ['n', 'stratum'], rng, f'{tag} fd{fd}')
    return out


def run(cache):
    with gzip.open(cache, 'rb') as fh:
        T = pickle.load(fh)
    M, P = T['main'], T['rep']
    res = {'meta': dict(seed=E.SEED, n_boot=E.N_BOOT, n_perm=E.N_PERM,
                        main_labelled_records=M['n_labelled_records'], rep_labelled_records=P['n_labelled_records'],
                        main_superfamilies=len(M['size']), rep_superfamilies=len(P['size']),
                        label_level='CATH H (4-part string), exact match in both censuses')}
    f = E.load(paths.OCC_REPLICATION)
    fm = E.load(paths.OCC_MAIN)

    # reproduce both censuses' L_H from the pair tables
    gm, gp = guarded(M['pair'], M['size']), guarded(P['pair'], P['size'])
    mm = int((fm.koga_code.map(lambda c: len(gm.get(c, ()))) != fm.L_H).sum())
    pm = int((f.koga_code.map(lambda c: len(gp.get(c, ()))) != f.L_H).sum())
    res['reproduction'] = dict(main_L_H_mismatches=mm, rep_L_H_mismatches=pm, main_rows=len(fm), rep_rows=len(f))
    print('L_H mismatches main/rep', mm, pm, flush=True)

    sf_main = set(M['size'])
    main_any = {k for k, v in M['pair'].items() if v >= 1}
    main_T = {(c, '.'.join(s.split('.')[:3])) for c, s in main_any}
    main_guard = {(c, s) for c, ss in gm.items() for s in ss}
    defs = {
        'pair_absent_main': lambda c, s: (c, s) not in main_any,
        'pair_not_main_guarded': lambda c, s: (c, s) not in main_guard,
        'T_level_absent_main': lambda c, s: (c, '.'.join(s.split('.')[:3])) not in main_T,
        'superfamily_absent_main': lambda c, s: s not in sf_main,
    }

    # ---- 1. overlap
    A, R = arms(f)
    ov = {}
    for arm, mask in (('exposed', A), ('reference', R)):
        for fdn, fdm in (('all', f.fd >= 0), ('fd1', f.fd == 1), ('fd0', f.fd == 0)):
            sel = f[mask & fdm]
            tot = anyp = gdp = newT = newsf = 0
            used_t = used_with_new = used_all_new = used_not_in_main = 0
            for c in sel.koga_code:
                ss = gp.get(c, set())
                if not ss:
                    continue
                used_t += 1
                n_any = sum((c, s) in main_any for s in ss)
                n_g = sum((c, s) in main_guard for s in ss)
                n_Tn = sum(defs['T_level_absent_main'](c, s) for s in ss)
                n_sfn = sum(defs['superfamily_absent_main'](c, s) for s in ss)
                tot += len(ss)
                anyp += n_any
                gdp += n_g
                newT += n_Tn
                newsf += n_sfn
                used_with_new += (len(ss) - n_any) > 0
                used_all_new += n_any == 0
                used_not_in_main += len(gm.get(c, ())) == 0
            ov[f'{arm}_{fdn}'] = dict(
                topologies=len(sel), used_in_rep=used_t, rep_support_pairs=tot,
                pairs_present_in_main=anyp, pairs_main_guarded=gdp,
                pairs_absent_from_main=tot - anyp, frac_new_pair=(tot - anyp) / tot if tot else None,
                frac_not_main_guarded=(tot - gdp) / tot if tot else None,
                frac_new_T_level=newT / tot if tot else None,
                frac_superfamily_absent=newsf / tot if tot else None,
                used_topologies_with_any_new_pair=used_with_new,
                used_topologies_with_all_pairs_in_main=used_t - used_with_new,
                used_topologies_with_no_main_guarded_support=used_not_in_main)
    res['overlap'] = ov
    for k, v in ov.items():
        print(f"{k:14s} topo {v['topologies']:4d} used {v['used_in_rep']:4d} pairs {v['rep_support_pairs']:5d} "
              f"new {v['frac_new_pair'] if v['frac_new_pair'] is None else round(v['frac_new_pair'], 3)} "
              f"allInMain {v['used_topologies_with_all_pairs_in_main']}", flush=True)

    # ---- 2. new-lineage contrasts on replication support
    rng = np.random.default_rng(E.SEED)
    res['replication_full'] = contrasts(f, (f.L_H > 0).astype(int), rng, 'rep full')
    res['new_lineage'] = {}
    for nm, allow in defs.items():
        g = guarded(P['pair'], P['size'], allow)
        used = f.koga_code.map(lambda c: int(len(g.get(c, ())) > 0))
        res['new_lineage'][nm] = dict(
            used_topologies_exposed=int(used[A].sum()), used_topologies_reference=int(used[R].sum()),
            contrasts=contrasts(f, used, rng, f'rep {nm}'))

    # ---- calibration: the same filter applied to the main census split into random halves by chain
    E.N_BOOT, E.N_PERM = 1000, 5000
    cal = {}
    for a, b in ((0, 1), (1, 0)):
        pa = {k for k, v in M['pair_h'][a].items() if v >= 1}
        sb = M['size_h'][b]
        full = guarded(M['pair_h'][b], sb)
        new = guarded(M['pair_h'][b], sb, lambda c, s: (c, s) not in pa)
        u_full = fm.koga_code.map(lambda c: int(len(full.get(c, ())) > 0))
        u_new = fm.koga_code.map(lambda c: int(len(new.get(c, ())) > 0))
        tot = sum(len(v) for c, v in full.items() if c in set(fm.koga_code))
        nw = sum(len(v) for c, v in new.items() if c in set(fm.koga_code))
        cal[f'A{a}_B{b}'] = dict(frac_new_pair=1 - nw / tot if False else nw / tot,
                                 half_full=contrasts(fm, u_full, rng, f'half{b} full'),
                                 half_new=contrasts(fm, u_new, rng, f'half{b} pairs absent from half{a}'))
    res['split_half_calibration'] = cal
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'replication_lineage.json', 'w'), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['build', 'run'])
    ap.add_argument('--main-calls', help='main census call file, JSON lines (build)')
    ap.add_argument('--rep-calls', help='replication census call file, JSON lines (build)')
    ap.add_argument('--cache', default=str(OUT / 'replication_lineage_pairs.pkl.gz'), help='pair tables')
    a = ap.parse_args()
    if a.cmd == 'build':
        build({'main': a.main_calls, 'rep': a.rep_calls}, a.cache)
    else:
        run(a.cache)


if __name__ == '__main__':
    main()
