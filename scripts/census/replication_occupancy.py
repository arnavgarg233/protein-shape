"""Occupancy table and stratified tests of the replication census (ESMFold models of AFESM clusters
without an AlphaFold DB member), under the agreement rule, lenient credit and higher pLDDT thresholds.

    python scripts/census/replication_occupancy.py calls_merged.jsonl [--frame-dir DIR] [--out R.json]

The shipped data/occupancy_replication.csv.gz is the table this writes (gzip -n).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from protein_shape import s2x
from protein_shape.frames import FRAME_DIR, N_PERM, binary, frame, van_elteren
from protein_shape.paths import RESULTS

ALL_MODES = ('strict', 'lenient', 'lenient_noterm')


def occupancy(path, modes=ALL_MODES, plddt_min=None):
    """L_H and the other occupancy measures under a chosen set of strand modes.

    With all three modes a domain supports a topology when every mode calls it (the agreement rule);
    modes=('lenient',) is lenient credit. plddt_min keeps domains with at least that AFESM domain pLDDT.
    CATH labels stay strings, since a float parse would merge labels such as 3.30.70.1 and 3.30.70.10.
    """
    sf_chains = defaultdict(lambda: defaultdict(set))
    sf_size = defaultdict(set)
    hybrid, clusters, tlevel, uset = (defaultdict(set) for _ in range(4))
    n = n_ok = 0
    for line in open(path):
        r = json.loads(line)
        n += 1
        if not r.get('ok'):
            continue
        if plddt_min is not None and float(r['plddt']) < plddt_min:
            continue
        n_ok += 1
        lab = r['cath_label']
        if not isinstance(lab, str):
            raise TypeError(f'cath_label must be a string, got {type(lab)}: {lab!r}')
        lab = lab if lab not in ('', '-') else None
        sf = lab if lab and lab.count('.') == 3 else None
        if sf:
            sf_size[sf].add(r['chain'])
        key = lab or 'unl:' + r['chain']
        agree = set.intersection(*(set(r['per_mode'][m]) for m in modes))
        for c in agree:
            hybrid[c].add(key)
            clusters[c].add(r['chain'])
            if lab:
                tlevel[c].add('.'.join(lab.split('.')[:3]))
            if sf:
                sf_chains[c][sf].add(r['chain'])
        for c in r['topology_set']:
            uset[c].add(key)

    def guarded(c, cls=None):
        return sum(1 for s, ch in sf_chains.get(c, {}).items()
                   if len(ch) >= max(2, 0.01 * len(sf_size[s])) and (cls is None or s.startswith(cls)))

    codes = set(sf_chains) | set(hybrid) | set(uset)
    occ = {
        'L_H': {c: guarded(c) for c in codes},
        'L_H_ab': {c: guarded(c, '3.') for c in codes},
        'L_H_noguard': {c: len(sf_chains.get(c, {})) for c in codes},
        'L_T': {c: len(tlevel.get(c, ())) for c in codes},
        'hybrid': {c: len(hybrid.get(c, ())) for c in codes},
        'clusters': {c: len(clusters.get(c, ())) for c in codes},
        'set_hybrid': {c: len(uset.get(c, ())) for c in codes},
    }
    return occ, n, n_ok, len(sf_size), sf_chains


def population(occ, frame_dir=FRAME_DIR):
    """Frustration-free five- to seven-strand topologies with at least two parallel crossovers, and arm masks."""
    f = frame(occ, frame_dir)
    ff = f[(f.violations == 0) & (f.npar >= 2)].reset_index(drop=True)
    ff['fd'] = [int(s2x.features(t)['C2z_crossing'] > 0) for t in ff.topology]
    m = dict(exp=(ff.OPP >= 1) & (ff.SAME == 0),
             ref=(ff.OPP == 0) & (ff.SAME == 0) & ff.both,
             same=(ff.SAME >= 1) & (ff.OPP == 0),
             bref=(ff.OPP == 0) & (ff.SAME == 0),
             fd1=ff.fd == 1, fd0=ff.fd == 0, s7=ff.n == 7)
    return f, ff, m


# --------------------------------------------------------------------------- the statistic

def statistics(ff, m, y, nperm=N_PERM, ecod_obs=None):
    """Stratified van Elteren tests on occupancy vector y over the rows of ff.

    H7_1: OPP arm against reference in ferredoxin-type sheets (strata n x npar); H7_2: the same over all
    sheets (strata n x npar x fd); H7_3: the fd1 against fd0 interaction, z = (z_fd1 - z_fd0) / sqrt(2).
    """
    ys = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, float), index=ff.index)
    out = {}

    def ve(a, b, by, np_=nperm):
        return van_elteren(ff, a, b, ys, by, n_perm=np_)

    by, by7, byfd = ['n', 'stratum'], ['stratum'], ['n', 'stratum', 'fd']

    r1 = ve(m['exp'] & m['fd1'], m['ref'] & m['fd1'], by)
    out['H7_1_fd1_pooled57'] = r1
    r2 = ve(m['exp'], m['ref'], byfd)
    out['H7_2_C4_n_npar_fd'] = r2
    r0 = ve(m['exp'] & m['fd0'], m['ref'] & m['fd0'], by, np_=0)
    out['NC_fd0_pooled57'] = r0
    out['H7_3_interaction'] = dict(z=float((r1['z'] - r0['z']) / 2 ** 0.5),
                                   p_asymptotic=float(norm.cdf((r1['z'] - r0['z']) / 2 ** 0.5)))
    out['PC1_same_vs_broad'] = ve(m['same'], m['bref'], by)
    b = binary(ff, m['exp'] & m['fd1'], m['ref'] & m['fd1'], ys > 0, by)
    out['S_fd1_binary_CMH'] = dict(p_asymptotic=b['cmh_p_one_sided'] if b['cmh_or'] < 1 else 1.0,
                                   p_perm=float('nan'), **b)
    out['R_T2R1_stat_n_npar'] = ve(m['exp'], m['ref'], by)
    out['R_T2R1_stat_n7'] = ve(m['exp'] & m['s7'], m['ref'] & m['s7'], by7)
    out['R_n7_fd'] = ve(m['exp'] & m['s7'], m['ref'] & m['s7'], ['stratum', 'fd'], np_=0)
    out['R_fd1_n7'] = ve(m['exp'] & m['fd1'] & m['s7'], m['ref'] & m['fd1'] & m['s7'], by7, np_=0)
    eo = ff.ecod_obs.to_numpy() if ecod_obs is None else ecod_obs
    try:
        out['PC3_auroc'] = dict(auroc=float(roc_auc_score(eo, ys.to_numpy())))
    except ValueError:
        out['PC3_auroc'] = dict(auroc=float('nan'))
    return out


def events(ff, m, y):
    """Used / total in strata where both arms exist, for the three decisive contrasts."""
    ys = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, float), index=ff.index)
    spec = (('H7_1_fd1_pooled57', m['exp'] & m['fd1'], m['ref'] & m['fd1'], ['n', 'stratum']),
            ('H7_2_C4_n_npar_fd', m['exp'], m['ref'], ['n', 'stratum', 'fd']),
            ('PC1_same_vs_broad', m['same'], m['bref'], ['n', 'stratum']))
    out = {}
    for nm, a, b, by in spec:
        ok = [i for _, g in ff.groupby(by) if a[g.index].any() and b[g.index].any() for i in g.index]
        ok = pd.Index(ok)
        out[nm] = dict(exposed=f'{int(((ys[ok] > 0) & a[ok]).sum())}/{int(a[ok].sum())}',
                       reference=f'{int(((ys[ok] > 0) & b[ok]).sum())}/{int(b[ok].sum())}')
    return out


# --------------------------------------------------------------------------- the run

def label_bias(path, ff, m, modes=ALL_MODES):
    """H-label rate among domains supporting an exposed topology against a reference topology; FLAG when
    the exposed rate is more than 10 points lower, which would mean topology-dependent labelling."""
    expc = set(ff.koga_code[m['exp'].to_numpy()])
    refc = set(ff.koga_code[m['ref'].to_numpy()])
    cnt = {'exposed': [0, 0], 'reference': [0, 0]}
    for line in open(path):
        r = json.loads(line)
        if not r.get('ok'):
            continue
        agree = set.intersection(*(set(r['per_mode'][mm]) for mm in modes))
        lab = r['cath_label']
        has = isinstance(lab, str) and lab not in ('', '-') and lab.count('.') == 3
        for nm, codes in (('exposed', expc), ('reference', refc)):
            if agree & codes:
                cnt[nm][0] += has
                cnt[nm][1] += 1
    out = {nm: dict(labelled=v[0], called=v[1], rate=(v[0] / v[1] if v[1] else None))
           for nm, v in cnt.items()}
    if out['exposed']['rate'] is not None and out['reference']['rate'] is not None:
        out['exposed_minus_reference'] = out['exposed']['rate'] - out['reference']['rate']
        out['FLAG'] = out['exposed_minus_reference'] < -0.10
    return out


def cmd_run(a):
    out = {}
    occ, n_dom, n_ok, n_sf, _ = occupancy(a.calls)
    f, ff, m = population(occ, a.frame_dir)
    out['census'] = dict(domains=n_dom, domains_called=n_ok, superfamilies_H=n_sf,
                         codes_5to7_with_support=int((f.hybrid > 0).sum()))
    out['PRIMARY'] = statistics(ff, m, ff.L_H, nperm=a.nperm)
    out['events'] = events(ff, m, ff.L_H)

    # violating against frustration-free at 7 strands, all 7-strand topologies (asymptotic)
    f7 = f[f.n == 7].reset_index(drop=True)
    out['PC2_violating_vs_ff_7strand'] = van_elteren(
        f7, f7.violations > 0, f7.violations == 0, f7.L_H, ['n'], n_perm=0)
    out['label_bias'] = label_bias(a.calls, ff, m)

    # lenient credit
    occL, *_ = occupancy(a.calls, modes=('lenient',))
    fL, ffL, mL = population(occL, a.frame_dir)
    out['ALT_lenient_only'] = statistics(ffL, mL, ffL.L_H, nperm=a.nperm)
    out['ALT_lenient_only_events'] = events(ffL, mL, ffL.L_H)

    # higher structure-confidence thresholds
    for lo in (80, 90):
        occP, nP, nokP, nsfP, _ = occupancy(a.calls, plddt_min=lo)
        fP, ffP, mP = population(occP, a.frame_dir)
        out[f'ALT_plddt_ge{lo}'] = dict(domains_called=nokP, superfamilies_H=nsfP,
                                        **statistics(ffP, mP, ffP.L_H, nperm=a.nperm))

    # H7_1 under two other support rules; every labelled domain here carries an H-level label, so L_T
    # is the no-guard rule at the coarser CATH level on the same domains
    for nm, col in (('H7_1_noguard', 'L_H_noguard'), ('H7_1_coarser_cath_level', 'L_T')):
        out[nm] = van_elteren(ff, m['exp'] & m['fd1'], m['ref'] & m['fd1'], ff[col],
                                ['n', 'stratum'], n_perm=a.nperm)
        out[nm + '_events'] = events(ff, m, ff[col])['H7_1_fd1_pooled57']
    # other occupancy measures
    for nm, col in (('S_hybrid_with_unlabelled', 'hybrid'), ('S_clusters', 'clusters')):
        out[nm] = van_elteren(ff, m['exp'] & m['fd1'], m['ref'] & m['fd1'], ff[col],
                                ['n', 'stratum'], n_perm=a.nperm)

    Path(a.out_table).parent.mkdir(parents=True, exist_ok=True)
    ff.to_csv(a.out_table, index=False)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    brief = {}
    for k, v in out.items():
        brief[k] = ({kk: vv for kk, vv in v.items() if kk != 'strata'}
                    if isinstance(v, dict) else v)
    print(json.dumps(brief, indent=1, default=lambda v: round(float(v), 5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('calls')
    ap.add_argument('--frame-dir', default=str(FRAME_DIR), help='directory with frame5.csv to frame7.csv')
    ap.add_argument('--out', default=str(RESULTS / 'census' / 'census_tests_replication.json'))
    ap.add_argument('--out-table', default=str(RESULTS / 'census' / 'occupancy_replication.csv'))
    ap.add_argument('--nperm', type=int, default=N_PERM)
    cmd_run(ap.parse_args())


if __name__ == '__main__':
    main()
