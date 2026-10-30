"""Superfamily support: how many superfamilies use each topology under a given strand-credit rule."""
import gzip
import json
from collections import defaultdict

MODES = ('strict', 'lenient', 'lenient_noterm')
KINDS = ('agree', 'lenient', 'union')


def guarded_use(chains_code, sf_size):
    """Number of superfamilies in which at least max(2, 1% of the superfamily's chains) call the topology."""
    return sum(1 for s, k in chains_code.items() if k >= max(2, 0.01 * sf_size[s]))


def supporters(chains_code, sf_size):
    """The superfamilies that pass the support rule for one topology."""
    return {s for s, k in chains_code.items() if k >= max(2, 0.01 * sf_size[s])}


def credit_sets(per_mode):
    """Topology codes credited to one domain under each rule."""
    pm = {m: set(per_mode[m]) for m in MODES}
    return dict(agree=set.intersection(*pm.values()), lenient=pm['lenient'], union=set.union(*pm.values()))


def credits(path, codes=None, kinds=KINDS):
    """Stream a census call file (JSON lines).

    Returns cnt[kind][code][superfamily] = number of distinct chains crediting the code, and
    sf_size[superfamily] = number of called chains carrying that CATH H-level label.
    """
    ch = {k: defaultdict(lambda: defaultdict(set)) for k in kinds}
    sf_size = defaultdict(set)
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get('ok'):
                continue
            lab = r['cath_label']
            if lab in ('', '-', None) or lab.count('.') != 3:
                continue
            sf_size[lab].add(r['chain'])
            sets = credit_sets(r['per_mode'])
            for kind in kinds:
                cs = sets[kind] if codes is None else sets[kind] & codes
                for c in cs:
                    ch[kind][c][lab].add(r['chain'])
    size = {s: len(v) for s, v in sf_size.items()}
    cnt = {k: {c: {s: len(v) for s, v in d.items()} for c, d in dd.items()} for k, dd in ch.items()}
    return cnt, size


def load_credits(path):
    """Read a credit table written by scripts/census/extract_credits.py: (chains, sf_size)."""
    d = json.load(gzip.open(path, 'rt'))
    return d['chains'], d['sf_size']


def occupancy(path):
    """Occupancy per topology code from a census call file, under the agreement rule.

    L_H is the guarded superfamily count used throughout; the other measures are the alternative
    definitions reported in Supplementary Note 1.
    """
    sf_chains = defaultdict(lambda: defaultdict(set))
    sf_size = defaultdict(set)
    hybrid, clusters, tlevel, uset = (defaultdict(set) for _ in range(4))
    n = n_ok = 0
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
        key = lab or 'unl:' + r['chain']
        agree = set.intersection(*(set(r['per_mode'][m]) for m in MODES))
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
    return occ, n, n_ok, len(sf_size)
