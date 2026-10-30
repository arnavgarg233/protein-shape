"""Recovery of OPP-carrying sheet topologies by the census pipeline, by class: consensus survival on crystal
sheets, recovery of crystal topologies by AlphaFold models of the same residues, and use credited with and
without the three-mode consensus.

    python scripts/robustness/class_recall.py select  --sheets SHEETS.csv --paired-sample SAMPLE.jsonl
    python scripts/robustness/class_recall.py call    (fetches AFDB models; network, resumable)
    python scripts/robustness/class_recall.py census  --calls CALLS.jsonl
    python scripts/robustness/class_recall.py analyse --sheets SHEETS.csv

SHEETS.csv is the ECOD crystal-sheet table (one row per open sheet with topology, ecod_h, weak_strand,
code_in_strict, code_in_noterm); SAMPLE.jsonl the paired crystal/AFDB sample (one domain per accession with
UniProt segments); CALLS.jsonl the main census call file.
"""
import argparse
import gzip
import json
import os
import random
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from protein_shape import grammar, paths, s2x
from protein_shape import topology as TP
from protein_shape.support import guarded_use

OUT = paths.RESULTS / 'robustness'
# set from the command line in main()
SHEETS = PAIRED_SAMPLE = CALLS = None
CACHE = OUT / 'afdb_models'
TARGETS = OUT / 'class_recall_targets.jsonl'
MODEL = OUT / 'class_recall_model.jsonl'
CENSUS = paths.CLASS_RECALL_CENSUS
OUT_TXT = OUT / 'class_recall.txt'
OUT_JSON = OUT / 'class_recall.json'
SEED = 122
OTHER_FRAC = 0.25            # fraction of 'other'-only accessions also measured (weighted back up)
SIZES = (3, 4, 5, 6, 7)      # open-sheet sizes of the census call
MODES = ('strict', 'lenient', 'lenient_noterm')
PLDDT_MIN = 70.0
API = 'https://alphafold.ebi.ac.uk/api/prediction/{}'
UA = {'User-Agent': 'protein-shape-classrecall/1.0 (structural bioinformatics; non-commercial research)'}
PAUSE = 0.34


# ---------------------------------------------------------------------------------------------
# classes
# ---------------------------------------------------------------------------------------------

def code_to_topology(code):
    n = sum(ch.isdigit() for ch in code)
    return '|'.join(code[:n]) + ':' + code[n:]


@lru_cache(maxsize=None)
def tclass(t):
    """(class, frustration_free, npar, stratum, fd) for a topology string; class in exposed/reference/nested/other.
    'nested' is the nested sub-arm of the reference arm (it is also reference for the census contrast)."""
    from protein_shape import structure as T
    f = s2x.features(t)
    par = [c for c in grammar.parallel_connections(*TP.parse(t)) if c['typ'] == 'P']
    both = len({c['sign1'] for c in par}) == 2
    opp, same, npar = f['S2x_crossing'], f['T_crossing'], f['npar']
    if opp >= 1 and same == 0:
        cls = 'exposed'
    elif opp == 0 and same == 0 and npar >= 2 and both:
        cls = 'nested' if f['S2x_nested'] >= 1 else 'reference'
    else:
        cls = 'other'
    fl = T.frustration_flags(t)
    ff = int(fl['jp'] + fl['ov'] + fl['en'] == 0)
    return cls, ff, int(npar), int(min(npar, 6)), int(f['C2z_crossing'] > 0)


def is_ref(cls):
    return cls in ('reference', 'nested')


# ---------------------------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------------------------

def crystal_sheets():
    import pandas as pd
    s = pd.read_csv(SHEETS, dtype={'ecod_h': str, 'ecod_fid': str})
    s = s[s.n_strands.between(5, 7)].copy()
    cl = [tclass(t) for t in s.topology]
    s['cls'] = [c[0] for c in cl]
    s['ff'] = [c[1] for c in cl]
    s['npar'] = [c[2] for c in cl]
    s['stratum'] = [c[3] for c in cl]
    s['fd'] = [c[4] for c in cl]
    s['cons'] = ((s.code_in_strict == 1) & (s.code_in_noterm == 1)).astype(int)
    return s


def cmd_select(a):
    s = crystal_sheets()
    sample = [json.loads(line) for line in Path(PAIRED_SAMPLE).open()]
    by_dom = {d: g for d, g in s.groupby('domain')}
    rng = random.Random(SEED)
    rows, n_sel = [], defaultdict(int)
    for r in sample:                                   # sample order, seeded draw
        g = by_dom.get(r['domain'])
        classes = sorted(set(g.cls)) if g is not None else []
        key = any(c != 'other' for c in classes)
        u = rng.random()                               # drawn for every row so the draw is order-stable
        if key:
            w, sel = 1.0, True
        elif classes:
            w, sel = 1.0 / OTHER_FRAC, u < OTHER_FRAC
        else:
            w, sel = 1.0 / OTHER_FRAC, u < OTHER_FRAC  # no 5-7-strand crystal sheet: reverse-direction pool
        n_sel[(key, bool(classes), sel)] += 1
        if sel:
            rows.append(dict(r, crystal_classes=classes, stratum_key=bool(key), weight=w))
    TARGETS.parent.mkdir(parents=True, exist_ok=True)
    with TARGETS.open('w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    for k, v in sorted(n_sel.items()):
        print(f'key_class={k[0]!s:5s} has_5to7={k[1]!s:5s} selected={k[2]!s:5s} {v}')
    print(f'{len(rows)} targets -> {TARGETS}')


# ---------------------------------------------------------------------------------------------
# call (model side, census pipeline)
# ---------------------------------------------------------------------------------------------

def get(url, timeout=60, tries=6):
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                raise
            last = e
        except Exception as e:                                   # noqa: BLE001
            last = e
        time.sleep(3 * (i + 1) ** 2)
    raise last


def model_text(acc):
    """Cached AFDB model (gzip) for an accession; fetched once from the public API."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f'{acc}.pdb.gz'
    if p.exists() and p.stat().st_size > 0:
        return gzip.open(p, 'rt').read(), json.loads((CACHE / f'{acc}.meta.json').read_text())['ver'], False
    meta = json.loads(get(API.format(acc)).decode())
    if not meta:
        raise RuntimeError('accession not in AFDB')
    url = meta[0]['pdbUrl']
    text = get(url).decode()
    ver = url.split('_v')[-1].split('.')[0]
    tmp = p.with_suffix('.tmp')
    with gzip.open(tmp, 'wt') as fh:
        fh.write(text)
    os.replace(tmp, p)
    (CACHE / f'{acc}.meta.json').write_text(json.dumps(dict(url=url, ver=ver)))
    return text, ver, True


def call_model(row, text):
    """Census call on the model cut to the paired residues."""
    from protein_shape import structure as T
    rng = ','.join(f'A:{lo}-{hi}' for lo, hi in row['uni_segments'])
    with tempfile.NamedTemporaryFile('w', suffix='.pdb', delete=True) as tmp:
        tmp.write(text)
        tmp.flush()
        st, labels = T.select_residues(tmp.name, None, T.parse_range(rng), 1)
        dssp = T.parse_dssp(T.run_dssp(st))
        ca = np.array([r.find_atom('CA', '*').pos.tolist() for r in st[0][0]])
        plddt = np.array([r.find_atom('CA', '*').b_iso for r in st[0][0]])
        res = T.topology_from_dssp(dssp, labels, ca, strand_mode='lenient')
        ts = T.topology_set(dssp, labels, ca, sizes=SIZES)
    sheets = [dict(n=s['n_strands'], code=s['code'], closed=s['closed'], bifurcated=s['bifurcated'])
              for s in res['sheets']]
    return dict(nres=len(labels), plddt=round(float(plddt.mean()), 2), n_strand=res['n_strands'],
                sheets=sheets, per_mode=ts['per_mode'], topology_set=ts['topology_set'])


def cmd_call(a):
    rows = [json.loads(line) for line in TARGETS.open()]
    done = set()
    if MODEL.exists():
        done = {json.loads(line)['domain'] for line in MODEL.open()}
    rows = [r for r in rows if r['domain'] not in done]
    if a.limit:
        rows = rows[:a.limit]
    t0, n_ok = time.time(), 0
    with MODEL.open('a') as fh:
        for i, row in enumerate(rows, 1):
            out = dict(domain=row['domain'], acc=row['acc'], pdb=row['pdb'])
            fetched = False
            try:
                text, ver, fetched = model_text(row['acc'])
                out.update(call_model(row, text), afdb_version=ver, status='ok')
                n_ok += 1
            except Exception as e:                               # noqa: BLE001
                out.update(status='fetch_error' if 'HTTP' in type(e).__name__ else 'error',
                           error=(type(e).__name__ + ':' + str(e))[:160])
            fh.write(json.dumps(out) + '\n')
            fh.flush()
            if fetched:
                time.sleep(PAUSE)
            if i % 200 == 0:
                print(f'{i}/{len(rows)} ok {n_ok} {(time.time() - t0) / i:.2f} s/domain', flush=True)
    print(f'done: {len(rows)} attempted, {n_ok} ok, {(time.time() - t0) / 60:.1f} min', flush=True)


def code_n(code):
    return sum(ch.isdigit() for ch in code)


def cmd_census(a):
    """One streaming pass over the census calls.

    surv[(cluster, cls, ff)] = [lenient 5-7 codes, of which in the consensus]; cluster = TED superfamily
    label, or the chain for unlabelled domains. For every frustration-free arm topology, per superfamily,
    the chains that build it under the consensus, in lenient mode, and in any mode."""
    path = CALLS
    out_path = OUT / 'class_recall_census.json.gz'
    surv = defaultdict(lambda: [0, 0])
    sf_size = defaultdict(set)
    chains = {k: defaultdict(lambda: defaultdict(set)) for k in ('agree', 'lenient', 'union')}
    n = n_ok = 0
    t0 = time.time()
    for line in open(path):
        r = json.loads(line)
        n += 1
        if not r['ok']:
            continue
        n_ok += 1
        lab = r['cath_label'] if r['cath_label'] not in ('', '-') else None
        sf = lab if lab and lab.count('.') == 3 else None
        if sf:
            sf_size[sf].add(r['chain'])
        pm = {m: set(r['per_mode'][m]) for m in MODES}
        agree = set.intersection(*pm.values())
        clu = sf or ('unl:' + r['chain'])
        for c in pm['lenient']:
            if 5 <= code_n(c) <= 7:
                cls, ff = tclass(code_to_topology(c))[:2]
                s = surv[(clu, cls, ff)]
                s[0] += 1
                s[1] += c in agree
        if not sf:
            continue
        for kind, cs in (('agree', agree), ('lenient', pm['lenient']), ('union', set(r['topology_set']))):
            for c in cs:
                if 5 <= code_n(c) <= 7:
                    cls, ff = tclass(code_to_topology(c))[:2]
                    if ff and cls != 'other':
                        chains[kind][c][sf].add(r['chain'])
        if n % 200000 == 0:
            print(f'{n} lines {(time.time() - t0) / 60:.1f} min', flush=True)
    out = dict(n_lines=n, n_ok=n_ok,
               sf_size={s: len(v) for s, v in sf_size.items()},
               surv=[[k[0], k[1], k[2], v[0], v[1]] for k, v in surv.items()],
               chains={kind: {c: {s: len(ch) for s, ch in d.items()} for c, d in dd.items()}
                       for kind, dd in chains.items()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, 'wt') as fh:
        json.dump(out, fh)
    print(f'done {n} lines, {n_ok} ok, {(time.time() - t0) / 60:.1f} min -> {out_path}', flush=True)


# ---------------------------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------------------------

B_BOOT = 4000
ALPHA = 0.05


def topology_to_code(t):
    o, s = t.split(':')
    return o.replace('|', '') + s


def cluster_boot(clu, grp, y, w=None, B=B_BOOT, seed=SEED, strata=None):
    """Ratio of weighted means, group 'e' over group 'r', with a cluster bootstrap over `clu`.

    If `strata` is given, the 'r' rate is directly standardised to the 'e' stratum distribution
    (strata the 'e' group lacks are ignored; 'e' strata without 'r' support are dropped from both)."""
    clu, grp, y = np.asarray(clu), np.asarray(grp), np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    st = np.zeros(len(y), int) if strata is None else np.unique(np.asarray(strata), return_inverse=True)[1]
    uc, ci = np.unique(clu, return_inverse=True)
    G, S = len(uc), st.max() + 1
    T = {}
    for g in ('e', 'r'):
        m = grp == g
        n = np.zeros((G, S))
        k = np.zeros((G, S))
        np.add.at(n, (ci[m], st[m]), w[m])
        np.add.at(k, (ci[m], st[m]), w[m] * y[m])
        T[g] = (n, k)

    def ratio(cnt):
        ne, ke = cnt @ T['e'][0], cnt @ T['e'][1]
        nr, kr = cnt @ T['r'][0], cnt @ T['r'][1]
        ok = (ne > 0) & (nr > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            re_ = np.where(ok, ke / ne, 0)
            rr = np.where(ok, kr / nr, 0)
            wt = np.where(ok, ne, 0)
            pe = (wt * re_).sum(-1) / wt.sum(-1)
            pr = (wt * rr).sum(-1) / wt.sum(-1)
            return pe, pr, pe / pr

    pe, pr, rat = ratio(np.ones(G))
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)
    bpe, bpr, brat = ratio(cnt)
    good = np.isfinite(brat)
    lo, hi = np.percentile(brat[good], [2.5, 97.5]) if good.any() else (np.nan, np.nan)
    ne = T['e'][0].sum()
    nr = T['r'][0].sum()
    return dict(rate_e=float(pe), rate_r=float(pr), ratio=float(rat), lo=float(lo), hi=float(hi),
                n_e=float(ne), n_r=float(nr), clusters_e=int((T['e'][0].sum(1) > 0).sum()),
                clusters_r=int((T['r'][0].sum(1) > 0).sum()), boot_ok=int(good.sum()))


def fmt_ratio(d):
    return (f"exposed {d['rate_e']:.3f} (n {d['n_e']:.0f}, {d['clusters_e']} clusters) / reference "
            f"{d['rate_r']:.3f} (n {d['n_r']:.0f}, {d['clusters_r']} clusters) = ratio {d['ratio']:.3f} "
            f"[{d['lo']:.3f}, {d['hi']:.3f}]")


def injection_power(clu, grp, y, ratios, sims=200, B=1000, seed=SEED, strata=None):
    """Positive control and minimum detectable effect: thin the observed exposed successes so that the
    expected exposed rate is r * (observed reference rate), keep the real clustering, and count how often
    the 95% cluster-bootstrap interval for the ratio excludes 1."""
    clu, grp, y = np.asarray(clu), np.asarray(grp), np.asarray(y, float)
    base = cluster_boot(clu, grp, y, B=200, strata=strata)
    rng = np.random.default_rng(seed + 1)
    out = {}
    for r in ratios:
        keep = r * base['rate_r'] / base['rate_e'] if base['rate_e'] > 0 else 0
        if keep > 1:
            out[r] = None                     # cannot inject upward by thinning
            continue
        hit, est = 0, []
        for s in range(sims):
            yy = y.copy()
            m = (grp == 'e') & (yy == 1)
            yy[m] = (rng.random(m.sum()) < keep).astype(float)
            d = cluster_boot(clu, grp, yy, B=B, seed=seed + 100 + s, strata=strata)
            hit += d['hi'] < 1
            est.append(d['ratio'])
        out[r] = dict(power=hit / sims, mean_est=float(np.mean(est)))
    return out


def mde(power):
    """Largest injected ratio (closest to 1) with power >= 0.8, scanning from 1 downward."""
    best = None
    for r in sorted((k for k, v in power.items() if v), reverse=True):
        if power[r]['power'] >= 0.8:
            best = r
            break
    return best


def cmd_analyse(a):
    import pandas as pd
    R = {}
    L = []

    def p(s=''):
        L.append(s)

    p('Recovery of OPP-carrying sheet topologies by the census pipeline, by class (seed 122)')
    p('Classes on 5-7-strand topologies: exposed = OPP>=1 & SAME==0; reference = OPP==0 & SAME==0 & npar>=2 &')
    p('both orientations ("nested" is the nested-pair sub-arm of reference and is counted in')
    p('reference unless shown separately).  Ratios are exposed / reference, 95% percentile intervals from')
    p(f'{B_BOOT} cluster-bootstrap draws (ECOD H-group as a string for solved structures, TED superfamily in the')
    p('census).  The observed census depletion to be explained: 26/643 vs 79/903 used, use-rate ratio 0.462.')
    p()

    # ---------------- A: crystal side ----------------
    s = crystal_sheets()
    s['g'] = np.where(s.cls == 'exposed', 'e', np.where(s.cls.isin(['reference', 'nested']), 'r', 'o'))
    p('=' * 100)
    p('A. CRYSTAL SIDE (no AlphaFold): does the 3-mode consensus drop OPP-carrying sheets more often?')
    p('   ECOD develop210 F99 domains, lenient call on the crystal; consensus = the lenient topology is also')
    p('   called in strict and in lenient_noterm mode (the census credit rule).')
    p('=' * 100)
    tab = s.groupby(['cls', 'n_strands']).agg(sheets=('cons', 'size'), hgroups=('ecod_h', 'nunique'),
                                              consensus=('cons', 'mean'), weak_strand=('weak_strand', 'mean'))
    p(tab.round(3).to_string())
    A = {}
    for lab, sub in (('all', s), ('frustration_free', s[s.ff == 1])):
        m = sub.g != 'o'
        A[lab] = dict(crude=cluster_boot(sub.ecod_h[m], sub.g[m], sub.cons[m]),
                      n_std=cluster_boot(sub.ecod_h[m], sub.g[m], sub.cons[m], strata=sub.n_strands[m]))
        p(f'\n  consensus survival, {lab}:')
        p(f'    crude                 {fmt_ratio(A[lab]["crude"])}')
        p(f'    strand-count standard {fmt_ratio(A[lab]["n_std"])}')
    # mechanism: weak strands
    mech = s[s.g != 'o'].groupby(['g', 'weak_strand']).cons.agg(['size', 'mean'])
    p('\n  consensus survival by whether the lenient sheet contains a weak (2-residue) strand:')
    p(mech.round(3).to_string())
    A['weak_strand_frac'] = {g: float(s[s.g == g].weak_strand.mean()) for g in ('e', 'r', 'o')}
    A['weak_strand_ratio'] = cluster_boot(s.ecod_h[s.g != 'o'], s.g[s.g != 'o'], s.weak_strand[s.g != 'o'])
    p(f"  weak-strand frequency: {fmt_ratio(A['weak_strand_ratio'])}")
    top = (s[s.g == 'e'].groupby('ecod_h').agg(sheets=('cons', 'size'), consensus=('cons', 'mean'))
           .sort_values('sheets', ascending=False).head(5))
    p('  largest exposed H-groups:  ' + '; '.join(f'{h} {int(r.sheets)} sheets, consensus {r.consensus:.2f}'
                                               for h, r in top.iterrows()))
    loo = s[(s.g != 'o') & ~s.ecod_h.isin(top.index[:2])]
    A['without_top2_hgroups'] = cluster_boot(loo.ecod_h, loo.g, loo.cons)
    p(f"  without the two largest exposed H-groups: {fmt_ratio(A['without_top2_hgroups'])}")
    R['A_crystal_consensus'] = A

    # ---------------- B: paired side ----------------
    p()
    p('=' * 100)
    p('B. PAIRED CRYSTAL <-> ALPHAFOLD: the AlphaFold model of the same protein, cut to the same UniProt')
    p('   residues, run through the census call.  Unit = crystal 5-7-strand sheet of the paired sample.')
    p('=' * 100)
    tg = {json.loads(line)['domain']: json.loads(line) for line in TARGETS.open()}
    md = {}
    for line in MODEL.open():
        r = json.loads(line)
        md[r['domain']] = r
    status = defaultdict(int)
    for d in tg:
        status[md.get(d, {}).get('status', 'missing')] += 1
    p(f'  targets {len(tg)}; model-side status: ' + ', '.join(f'{k} {v}' for k, v in sorted(status.items())))
    vers = defaultdict(int)
    for r in md.values():
        vers[r.get('afdb_version')] += 1
    p('  AFDB model versions: ' + ', '.join(f'v{k} {v}' for k, v in sorted(vers.items(), key=str)))
    ps = s[s.domain.isin(tg)].copy()
    rows = []
    for r in ps.itertuples():
        m = md.get(r.domain)
        if not m or m['status'] != 'ok':
            continue
        X = topology_to_code(r.topology)
        pm = {k: set(v) for k, v in m['per_mode'].items()}
        agree = set.intersection(*pm.values())
        gate = m['plddt'] >= PLDDT_MIN
        same_agree = {c for c in agree if 5 <= code_n(c) <= 7}
        same_len = {c for c in pm['lenient'] if 5 <= code_n(c) <= 7}

        def cls_of(c):
            k = tclass(code_to_topology(c))[0]
            return 'r' if k in ('reference', 'nested') else ('e' if k == 'exposed' else 'o')

        g = r.g
        rows.append(dict(
            domain=r.domain, h=r.ecod_h, g=g, cls=r.cls, ff=r.ff, n=r.n_strands, cryst_cons=r.cons,
            weak=r.weak_strand, plddt=m['plddt'], w=tg[r.domain]['weight'], gate=int(gate),
            rec=int(gate and X in agree), rec_len=int(gate and X in pm['lenient']),
            in_len=int(X in pm['lenient']), surv=int(X in agree),
            modes_agree=int(gate and len({frozenset(v) for v in pm.values()}) == 1 and bool(pm['lenient'])),
            pres=int(gate and any(cls_of(c) == g for c in same_agree)),
            pres_len=int(gate and any(cls_of(c) == g for c in same_len)),
            model_any_agree_5to7=int(bool(same_agree))))
    P = pd.DataFrame(rows)
    p('  measured sheets: ' + ', '.join(f'{g} {int((P.g == g).sum())} ({P[P.g == g].h.nunique()} H-groups)'
                                          for g in ('e', 'r', 'o')))
    p(f"  mean model pLDDT of the cut: exposed {P[P.g == 'e'].plddt.mean():.1f}, reference "
      f"{P[P.g == 'r'].plddt.mean():.1f}; gate (pLDDT >= 70) passed by "
      f"{P[P.g == 'e'].gate.mean():.3f} / {P[P.g == 'r'].gate.mean():.3f}")
    Bres = {}
    outcomes = [
        ('rec', '1. recovery: model consensus contains the identical canonical topology'),
        ('rec_len', '   recovery in lenient mode alone (no consensus)'),
        ('modes_agree', '2. all three model strand modes return the same non-empty code set'),
        ('pres', '3. class preservation: a model consensus 5-7 topology of the same class'),
        ('pres_len', '   class preservation in lenient mode alone'),
    ]
    for sub_lab, sub in (('all', P), ('crystal_consensus_only', P[P.cryst_cons == 1]),
                         ('frustration_free', P[P.ff == 1])):
        q = sub[sub.g != 'o']
        p(f'\n  [{sub_lab}]')
        Bres[sub_lab] = {}
        for col, lab in outcomes:
            d = cluster_boot(q.h, q.g, q[col])
            Bres[sub_lab][col] = d
            p(f'  {lab}')
            p(f'      {fmt_ratio(d)}')
        qq = q[q.in_len == 1]
        d = cluster_boot(qq.h, qq.g, qq.surv)
        Bres[sub_lab]['surv_given_lenient'] = d
        p('  2b. consensus survival of the crystal topology, given the model lenient call has it')
        p(f'      {fmt_ratio(d)}')
        d = cluster_boot(q.h, q.g, q.rec, strata=q.n)
        Bres[sub_lab]['rec_n_std'] = d
        p('  1s. recovery, reference standardised to the exposed strand counts')
        p(f'      {fmt_ratio(d)}')
    q = P[P.g != 'o']
    top = (q[q.g == 'e'].groupby('h').agg(sheets=('rec', 'size'), rec=('rec', 'mean'), rec_len=('rec_len', 'mean'),
                                           weak=('weak', 'mean')).sort_values('sheets', ascending=False))
    p('\n  exposed paired sheets by H-group (largest five): ' + '; '.join(
        f'{h} {int(r.sheets)} sheets, recovery {r.rec:.2f}, lenient {r.rec_len:.2f}, weak strand {r.weak:.2f}'
        for h, r in top.head(5).iterrows()))
    loo = q[~q.h.isin(top.index[:2])]
    Bres['without_top2_hgroups'] = {c: cluster_boot(loo.h, loo.g, loo[c]) for c in ('rec', 'rec_len')}
    p(f"  recovery without the two largest exposed H-groups: {fmt_ratio(Bres['without_top2_hgroups']['rec'])}")
    p(f"  lenient recovery without them:                     {fmt_ratio(Bres['without_top2_hgroups']['rec_len'])}")
    for lab, m in (('crystal sheet has a weak strand', q.weak == 1), ('crystal sheet has no weak strand', q.weak == 0)):
        d = cluster_boot(q.h[m], q.g[m], q.rec[m])
        Bres['rec_weak' if 'a weak' in lab else 'rec_noweak'] = d
        p(f'  recovery, {lab}: {fmt_ratio(d)}')
    other = P[P.g == 'o']
    p(f"\n  'other' class (sampled at {OTHER_FRAC}): recovery {other.rec.mean():.3f} (n {len(other)}), "
      f"lenient {other.rec_len.mean():.3f}")
    R['B_paired'] = Bres

    # positive controls on the instrument: harder sheets must show lower recovery
    p('\n  POSITIVE CONTROLS (can this measurement see a recovery difference?)')
    PC = {}
    allk = P.copy()
    allk['g_pl'] = np.where(allk.plddt < 85, 'e', np.where(allk.plddt >= 90, 'r', 'x'))
    q = allk[allk.g_pl != 'x']
    PC['plddt_lt85_vs_ge90'] = cluster_boot(q.h, q.g_pl, q.rec)
    p(f"    recovery, model pLDDT < 85 (as 'exposed') vs >= 90:  {fmt_ratio(PC['plddt_lt85_vs_ge90'])}")
    allk['g_n'] = np.where(allk.n == 7, 'e', np.where(allk.n == 5, 'r', 'x'))
    q = allk[allk.g_n != 'x']
    PC['n7_vs_n5'] = cluster_boot(q.h, q.g_n, q.rec)
    p(f"    recovery, 7-strand (as 'exposed') vs 5-strand:       {fmt_ratio(PC['n7_vs_n5'])}")
    allk['g_w'] = np.where(allk.weak == 1, 'e', 'r')
    PC['weak_vs_not'] = cluster_boot(allk.h, allk.g_w, allk.rec)
    p(f"    recovery, crystal sheet with a weak strand vs none:  {fmt_ratio(PC['weak_vs_not'])}")
    q = P[P.g != 'o']
    grid = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4]
    PC['injection_rec'] = {str(k): v for k, v in injection_power(q.h, q.g, q.rec_len, grid).items()}
    PC['mde_rec'] = mde({float(k): v for k, v in PC['injection_rec'].items()})
    p('    injected class-specific drop in exposed LENIENT recovery (the AlphaFold-versus-crystal step, whose')
    p('    observed ratio is near 1; thinning the real data with the real clustering,')
    p('    200 simulations x 1000 bootstrap draws; power = P(95% interval for the ratio excludes 1)):')
    for k, v in PC['injection_rec'].items():
        p(f'      true ratio {float(k):.2f}: ' + ('not reachable by thinning' if v is None else
                                                 f"power {v['power']:.2f}, mean estimate {v['mean_est']:.3f}"))
    p(f"    minimum detectable recovery ratio at 80% power: {PC['mde_rec']}")
    R['B_controls'] = PC

    # reverse direction: model OPP calls that are not OPP in the crystal
    p('\n  4. REVERSE DIRECTION: among model-side consensus 5-7-strand calls of a class, how often the crystal')
    p('     domain carries the identical topology / any sheet of the same class (weighted for the')
    p(f'     {OTHER_FRAC} sampling of domains without an exposed or reference crystal sheet)')
    cs_by_dom = {d: set(g.topology) for d, g in s.groupby('domain')}
    cls_by_dom = {d: set(g.g) for d, g in s.groupby('domain')}
    rev = []
    for d, m in md.items():
        if m.get('status') != 'ok' or m['plddt'] < PLDDT_MIN:
            continue
        agree = set.intersection(*(set(v) for v in m['per_mode'].values()))
        for c in agree:
            if not 5 <= code_n(c) <= 7:
                continue
            t = code_to_topology(c)
            k = tclass(t)[0]
            g = 'r' if k in ('reference', 'nested') else ('e' if k == 'exposed' else 'o')
            rev.append(dict(domain=d, g=g, w=tg[d]['weight'], same_topo=int(t in cs_by_dom.get(d, set())),
                            same_cls=int(g in cls_by_dom.get(d, set()))))
    V = pd.DataFrame(rev)
    Rv = {}
    for g in ('e', 'r'):
        v = V[V.g == g]
        Rv[g] = dict(n=int(len(v)), w=float(v.w.sum()),
                     identical=float((v.w * v.same_topo).sum() / v.w.sum()),
                     same_class=float((v.w * v.same_cls).sum() / v.w.sum()))
        p(f"     model {'exposed' if g == 'e' else 'reference'} calls: n {len(v)} (weighted {v.w.sum():.0f}); "
          f"crystal identical {Rv[g]['identical']:.3f}; crystal same class {Rv[g]['same_class']:.3f}; "
          f"false-positive class rate {1 - Rv[g]['same_class']:.3f}")
    R['B_reverse'] = Rv

    # ---------------- C: census side ----------------
    p()
    p('=' * 100)
    p('C. CENSUS SIDE (no crystal): the consensus step inside the census itself, and use without it')
    p('=' * 100)
    C = json.load(gzip.open(CENSUS, 'rt'))
    sv = pd.DataFrame(C['surv'], columns=['clu', 'cls', 'ff', 'n', 'k'])
    sv['g'] = np.where(sv.cls == 'exposed', 'e', np.where(sv.cls.isin(['reference', 'nested']), 'r', 'o'))
    CC = {}
    for lab, sub in (('all', sv), ('frustration_free', sv[sv.ff == 1])):
        q = sub[sub.g != 'o']
        # expand cluster aggregates to the bootstrap's (cluster, group, y, weight) form
        clu = np.concatenate([q.clu, q.clu])
        grp = np.concatenate([q.g, q.g])
        y = np.concatenate([np.ones(len(q)), np.zeros(len(q))])
        w = np.concatenate([q.k, q.n - q.k])
        d = cluster_boot(clu, grp, y, w=w, B=2000)
        CC[lab] = d
        p(f'  P(lenient 5-7 code survives the consensus), {lab}:  {fmt_ratio(d)}')
    R['C_census_consensus'] = CC

    f = pd.read_csv(paths.OCC_MAIN)
    ff = f[(f.violations == 0) & (f.npar >= 2)].reset_index(drop=True)
    ff['fd'] = [tclass(t)[4] for t in ff.topology]
    exp = (ff.OPP >= 1) & (ff.SAME == 0)
    ref = (ff.OPP == 0) & (ff.SAME == 0) & ff.both
    sfs = C['sf_size']
    for kind in ('agree', 'lenient', 'union'):
        ch = C['chains'][kind]
        ff['LH_' + kind] = [guarded_use(ch.get(c, {}), sfs) for c in ff.koga_code]
    arm = (exp | ref).to_numpy()
    repro = bool((ff.LH_agree[arm] == ff.L_H[arm]).all())
    p(f'  recomputed consensus L_H reproduces L_H on all {int(arm.sum())} exposed + reference rows: {repro}')
    key = list(zip(ff.n, ff.stratum, ff.fd))
    cells_e = {k for k, e in zip(key, exp) if e}
    cells_r = {k for k, r_ in zip(key, ref) if r_}
    shared = np.array([k in cells_e and k in cells_r for k in key])
    from protein_shape.frames import van_elteren
    CU = {}
    for kind in ('agree', 'lenient', 'union'):
        y = ff['LH_' + kind]
        ue, ne = int(((y > 0) & exp & shared).sum()), int((exp & shared).sum())
        ur, nr = int(((y > 0) & ref & shared).sum()), int((ref & shared).sum())
        ve = van_elteren(ff, exp & shared, ref & shared, y, ['n', 'stratum', 'fd'])
        CU[kind] = dict(used_e=ue, n_e=ne, used_r=ur, n_r=nr, use_ratio=(ue / ne) / (ur / nr),
                        p_perm=ve['p_perm'], z=ve['z'])
        p(f'  use credited by {kind:8s}: exposed {ue}/{ne} vs reference {ur}/{nr}, use ratio '
          f"{CU[kind]['use_ratio']:.3f}, van Elteren one-sided p_perm {ve['p_perm']:.5f}")
    R['C_use_without_consensus'] = CU
    for kind in ('lenient', 'union'):
        sh = np.log(CU[kind]['use_ratio'] / CU['agree']['use_ratio']) / np.log(1 / CU['agree']['use_ratio'])
        CU[kind]['share_of_log_depletion_removed'] = float(sh)
        p(f'  dropping the consensus ({kind}) moves the use ratio from {CU["agree"]["use_ratio"]:.3f} to '
          f'{CU[kind]["use_ratio"]:.3f}: {sh:.2f} of the log-depletion, an upper bound since it also credits '
          f'calls the consensus rejects as errors')

    # how much of the 0.462 could a recovery deficit of size rho produce?
    p('\n  Thin the reference arm\'s census chains by rho (binomial, per superfamily, guard')
    p('  denominators unchanged) and recompute reference use; the depletion a pure recovery artifact of size')
    p('  rho would produce is ref_use(rho)/ref_use(1); the observed depletion is 0.462.')
    ch = C['chains']['agree']
    refcodes = [c for c, r_, sh in zip(ff.koga_code, ref, shared) if r_ and sh]
    ue0 = CU['agree']['used_e'] / CU['agree']['n_e']
    base_r = CU['agree']['used_r'] / CU['agree']['n_r']
    rng = np.random.default_rng(SEED + 7)
    TH = {}
    rec_main = Bres['all']['rec']
    surv_main = A['all']['crude']
    cand = [rec_main['ratio'], rec_main['lo'], surv_main['ratio'], surv_main['lo'],
            CC['frustration_free']['ratio'], CC['frustration_free']['lo']]
    rhos = sorted({1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3} |
                  {round(x, 3) for x in cand if np.isfinite(x) and 0 < x < 1}, reverse=True)
    for rho in rhos:
        uses = []
        for _ in range(200):
            u = 0
            for c in refcodes:
                for sf, k in ch.get(c, {}).items():
                    if rng.binomial(k, rho) >= max(2, 0.01 * sfs[sf]):
                        u += 1
                        break
            uses.append(u / len(refcodes))
        ru = float(np.mean(uses))
        dep = ru / base_r
        expl = np.log(dep) / np.log(ue0 / base_r) if dep < 1 else 0.0
        TH[str(rho)] = dict(ref_use=ru, artifact_depletion=dep, share_of_log_depletion=float(expl))
        p(f'    rho {rho:.3f}: reference use {ru:.4f} (observed {base_r:.4f}); artifact-only depletion '
          f'{dep:.3f}; share of the observed log-depletion {expl:.2f}')
    R['C_thinning'] = TH
    # the AlphaFold step on its own, applied to use credited without the consensus
    chl = C['chains']['lenient']
    ue_l = CU['lenient']['used_e'] / CU['lenient']['n_e']
    base_l = CU['lenient']['used_r'] / CU['lenient']['n_r']
    p('\n  The AlphaFold-versus-crystal step alone (lenient recovery), applied to lenient-credited use, whose')
    p(f'  observed depletion is {ue_l / base_l:.3f}:')
    THL = {}
    for rho in sorted({round(Bres['all']['rec_len']['ratio'], 3), round(Bres['all']['rec_len']['lo'], 3),
                       round(Bres['frustration_free']['rec_len']['lo'], 3)}, reverse=True):
        if not 0 < rho < 1:
            continue
        uses = []
        for _ in range(200):
            u = 0
            for c in refcodes:
                for sf, k in chl.get(c, {}).items():
                    if rng.binomial(k, rho) >= max(2, 0.01 * sfs[sf]):
                        u += 1
                        break
            uses.append(u / len(refcodes))
        dep = float(np.mean(uses)) / base_l
        expl = np.log(dep) / np.log(ue_l / base_l) if dep < 1 else 0.0
        THL[str(rho)] = dict(artifact_depletion=dep, share_of_lenient_log_depletion=float(expl),
                             residual_use_ratio=float((ue_l / base_l) / dep))
        p(f'    rho {rho:.3f}: artifact-only depletion {dep:.3f}; share {expl:.2f}; residual use ratio '
          f'{(ue_l / base_l) / dep:.3f}')
    R['C_thinning_lenient'] = THL
    R['meta'] = dict(seed=SEED, B=B_BOOT, other_frac=OTHER_FRAC, plddt_min=PLDDT_MIN)

    # ---------------- summary ----------------
    def ci(d):
        return f"{d['ratio']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"
    b, a_, cu, cc = Bres['all'], A['all']['crude'], CU, CC['frustration_free']
    thl = THL[max(THL, key=lambda k: -float(k))] if THL else None   # the smallest rho = lower bound
    p()
    p('=' * 100)
    p('SUMMARY')
    p('=' * 100)
    p(f"  AlphaFold step (model vs crystal, lenient call): recovery ratio {ci(b['rec_len'])}, class preservation")
    p(f"  {ci(b['pres_len'])}; minimum detectable ratio at 80% power {PC['mde_rec']}.  The predicted models do not")
    p('  lose OPP sheets preferentially.')
    p(f"  Consensus step: class-specific.  Paired census-rule recovery {ci(b['rec'])}; crystal-only consensus")
    p(f"  survival {ci(a_)}; inside the census {ci(cc)} (frustration-free) and {ci(CC['all'])} (all).  The")
    p(f"  mechanism is the 2-residue (weak) strand: {A['weak_strand_frac']['e']:.2f} of exposed crystal sheets carry one")
    p(f"  against {A['weak_strand_frac']['r']:.2f} of reference sheets, the strict mode cannot see it, and without one")
    p(f"  recovery is class-neutral ({ci(Bres['rec_noweak'])}).")
    p(f"  Removing the consensus altogether leaves exposed {cu['lenient']['used_e']}/{cu['lenient']['n_e']} vs reference "
      f"{cu['lenient']['used_r']}/{cu['lenient']['n_r']} used,")
    p(f"  use ratio {cu['lenient']['use_ratio']:.3f} (p_perm {cu['lenient']['p_perm']:.4f}); any-mode credit "
      f"{cu['union']['use_ratio']:.3f} (p_perm {cu['union']['p_perm']:.4f}).")
    if thl:
        p(f"  Charging the AlphaFold step at its lower bound on top of that leaves a use ratio of "
          f"{thl['residual_use_ratio']:.3f}.")
    p('  The measurement artifact is real but too small to explain the depletion.  The')
    p(f"  consensus inflates the depletion from about {cu['lenient']['use_ratio']:.2f} to 0.46 (at most "
      f"{cu['lenient']['share_of_log_depletion_removed']:.0%} of the log-depletion);")
    p('  the rest survives with no consensus filter and an AlphaFold step that is class-neutral.')
    p('  Not tested here: TED versus ECOD domain cutting (the paired cut is ECOD), and census proteins below the')
    p(f"  paired set's confidence (paired model pLDDT ~93; the pLDDT<85 control gave {ci(PC['plddt_lt85_vs_ge90'])}).")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(R, indent=1, default=float))
    OUT_TXT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L[-40:]))


def main():
    global SHEETS, PAIRED_SAMPLE, CALLS, CENSUS, TARGETS, MODEL, CACHE
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['select', 'call', 'census', 'analyse'])
    ap.add_argument('--limit', type=int)
    ap.add_argument('--sheets', help='ECOD crystal-sheet table (select, analyse)')
    ap.add_argument('--paired-sample', help='paired crystal/AFDB sample, JSON lines (select)')
    ap.add_argument('--calls', help='main census call file, JSON lines (census)')
    ap.add_argument('--census', default=str(CENSUS), help='per-code chain counts (analyse)')
    ap.add_argument('--targets', default=str(TARGETS), help='selected targets (written by select)')
    ap.add_argument('--models', default=str(MODEL), help='model-side calls (written by call)')
    ap.add_argument('--cache', default=str(CACHE), help='directory for downloaded AFDB models')
    a = ap.parse_args()
    SHEETS, PAIRED_SAMPLE, CALLS = a.sheets, a.paired_sample, a.calls
    CENSUS, TARGETS, MODEL, CACHE = Path(a.census), Path(a.targets), Path(a.models), Path(a.cache)
    {'select': cmd_select, 'call': cmd_call, 'census': cmd_census, 'analyse': cmd_analyse}[a.cmd](a)


if __name__ == '__main__':
    main()
