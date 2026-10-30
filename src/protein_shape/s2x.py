"""Symmetry relation (T, C2y, C2z, S2x) and span relation (crossing, nested, disjoint) of every pair
of parallel crossovers, from the topology string alone. S2x + crossing is the OPP pair, T + crossing the
same-face interleaved pair and C2z + crossing the ferredoxin-type pair.
"""
from . import grammar
from . import topology as TP

RELATIONS = ('T', 'C2y', 'C2z', 'S2x')
SPANS = ('crossing', 'nested', 'disjoint')


def crossovers(t):
    """Parallel crossovers of topology t (frozen grammar.parallel_connections, typ 'P'), in sequence order."""
    return [c for c in grammar.parallel_connections(*TP.parse(t)) if c['typ'] == 'P']


def relation(a, b):
    same_d = (a['pos2'] > a['pos1']) == (b['pos2'] > b['pos1'])
    same_f = grammar._rh_face(a) == grammar._rh_face(b)
    return ('T' if same_f else 'S2x') if same_d else ('C2z' if same_f else 'C2y')


def link(t, a, b):
    """How two crossovers (a before b in sequence) are joined: 'hairpin' (one antiparallel jump-0
    connection), 'arch' (antiparallel connections only, otherwise) or 'crossover' (another parallel
    crossover in between)."""
    cs = grammar.parallel_connections(*TP.parse(t))[a['idx'] + 1:b['idx']]
    if any(c['typ'] == 'P' for c in cs):
        return 'crossover'
    return 'hairpin' if len(cs) == 1 and cs[0]['jump'] == 0 else 'arch'


def pairs(t):
    """(relation, span, link) for every pair of parallel crossovers."""
    par = crossovers(t)
    out = []
    for i, a in enumerate(par):
        for b in par[i + 1:]:
            span = grammar.span_relation(a, b)
            assert span in SPANS, (t, span)  # two crossovers cannot share both sheet positions
            out.append((relation(a, b), span, link(t, a, b)))
    return out


def features(t):
    """Counts per relation x span, the orientation/face cell and the S2x bridging flag."""
    par = crossovers(t)
    out = {f'{r}_{s}': 0 for r in RELATIONS for s in SPANS}
    out.update(npar=len(par), S2x_hairpin=0, S2x_arch=0, S2x_crossover=0)
    for r, s, lk in pairs(t):
        out[f'{r}_{s}'] += 1
        if r == 'S2x' and s != 'crossing':
            out[f'S2x_{lk}'] += 1
    out['S2x'] = out['S2x_crossing'] + out['S2x_nested'] + out['S2x_disjoint']
    out['S2x_noncross'] = out['S2x_nested'] + out['S2x_disjoint']
    cls = {(c['sign1'], grammar._rh_face(c)) for c in par}
    out['mixed_sigma'] = int(len({s for s, _ in cls}) == 2)
    out['two_faces'] = int(len({f for _, f in cls}) == 2)
    antipodal = any((-s, -f) in cls for s, f in cls)
    out['bridged'] = int(antipodal and len(cls) >= 3)
    out['n_classes'] = len(cls)
    return out
