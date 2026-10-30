"""Connections of a topology string and the relation between two connection spans; also the
connection-grammar violation count, which needs a support table of connection classes.
"""
import itertools as it
import json

from . import topology as T
from .paths import DATA

MODEL_PATH = DATA / 'grammar_model.json'
_MODEL = None


def conn_key(typ, jump, hand=None):
    return f"{typ}{min(jump, 2)}" + (f"/{hand}" if typ == 'P' and hand else '')


def span_relation(a, b):
    amin, amax = min(a['pos1'], a['pos2']), max(a['pos1'], a['pos2'])
    bmin, bmax = min(b['pos1'], b['pos2']), max(b['pos1'], b['pos2'])
    if amax <= bmin or bmax <= amin:
        return 'disjoint'
    if (amin, amax) == (bmin, bmax):
        return 'same'
    if (amin < bmin and bmax < amax) or (bmin < amin and amax < bmax):
        return 'nested'
    return 'crossing'


def pair_key(a, b, same_face):
    """a, b: dicts with typ, jump, pos1, pos2. Order-independent."""
    ka, kb = sorted([(a['typ'], min(a['jump'], 2), a['pos1'], a['pos2']), (b['typ'], min(b['jump'], 2), b['pos1'], b['pos2'])])
    da = 1 if a['pos2'] > a['pos1'] else -1
    db = 1 if b['pos2'] > b['pos1'] else -1
    return '|'.join([f"{ka[0]}{ka[1]}", f"{kb[0]}{kb[1]}", span_relation(a, b),
                     'dir_same' if da == db else 'dir_opp', 'face_same' if same_face else 'face_opp'])


def load_model(path=MODEL_PATH):
    global _MODEL
    if _MODEL is None:
        raw = json.load(open(path))
        _MODEL = {k: {c: [tuple(x) for x in v] for c, v in raw[k].items()} for k in ('conn', 'pair')}
    return _MODEL


def support(kind, key, exclude=None, model=None):
    """Number of distinct H-groups using the class, ignoring domains of topology `exclude`."""
    m = (model or load_model())[kind].get(key, [])
    return len({h for t, h in m if t != exclude})


def _rh_face(c):
    """+1 front / -1 back for a right-handed parallel connection (Koga geom)."""
    return 1 if (c['pos2'] - c['pos1']) * c['sign1'] > 0 else -1


def parallel_connections(order, signs):
    _, cons = T.connections(order, signs)
    out = []
    for c in cons:
        d = dict(typ='P' if c['type'] == 'over' else 'A', jump=c['jump'], pos1=c['pos1'], pos2=c['pos2'],
                 sign1=c['sign1'], idx=c['i1'])
        out.append(d)
    return out


def carriers(order, signs):
    _, cons = T.connections(order, signs)
    out = []
    for c in cons:
        d = dict(typ='P' if c['type'] == 'over' else 'A', jump=c['jump'], pos1=c['pos1'], pos2=c['pos2'],
                 sign1=c['sign1'], idx=c['i1'])
        if d['typ'] == 'P' or d['jump'] >= 1:
            out.append(d)
    return out


def grammar_violations(topology, min_h=2, exclude=None, model=None, detail=False):
    """Minimum number of unsupported connection + pair classes over handedness / free-face assignments.

    exclude: topology string whose own domains are removed from the support (default: `topology`).
    """
    order, signs = T.parse(topology)
    topo = T.minami(order, signs)
    exclude = topo if exclude is None else exclude
    car = carriers(order, signs)
    best = None
    for opt in it.product('RL', repeat=len(car)):
        # for a parallel carrier opt is its handedness; for an antiparallel jump it is its free face (R=+1, L=-1)
        bad = []
        face = {}
        for c, o in zip(car, opt):
            if c['typ'] == 'P':
                key = conn_key('P', c['jump'], o)
                face[c['idx']] = _rh_face(c) * (1 if o == 'R' else -1)
            else:
                key = conn_key('A', c['jump'])
                face[c['idx']] = 1 if o == 'R' else -1
            if support('conn', key, exclude, model) < min_h:
                bad.append(('conn', key))
        for a, b in it.combinations(car, 2):
            key = pair_key(a, b, face[a['idx']] == face[b['idx']])
            if support('pair', key, exclude, model) < min_h:
                bad.append(('pair', key))
        if best is None or len(bad) < len(best[0]):
            best = (bad, opt)
    out = dict(min_violations=len(best[0]), conn_violations=sum(1 for k, _ in best[0] if k == 'conn'),
               pair_violations=sum(1 for k, _ in best[0] if k == 'pair'))
    if detail:
        out.update(violations=best[0], assignment=''.join(best[1]))
    return out


def grammar_min_support(topology, exclude=None, model=None):
    """Continuous version: the best assignment maximises the support (distinct H-groups) of its rarest class.

    Returns (min_support, rarest_class, assignment); min_support is None when the topology has no carrier
    (every connection is a hairpin), in which case the grammar says nothing.
    """
    order, signs = T.parse(topology)
    exclude = T.minami(order, signs) if exclude is None else exclude
    car = carriers(order, signs)
    if not car:
        return None, '', ''
    best = None
    for opt in it.product('RL', repeat=len(car)):
        face, sup, names = {}, [], []
        for c, o in zip(car, opt):
            if c['typ'] == 'P':
                k = conn_key('P', c['jump'], o)
                face[c['idx']] = _rh_face(c) * (1 if o == 'R' else -1)
            else:
                k = conn_key('A', c['jump'])
                face[c['idx']] = 1 if o == 'R' else -1
            sup.append(support('conn', k, exclude, model)); names.append(k)
        for a, b in it.combinations(car, 2):
            k = pair_key(a, b, face[a['idx']] == face[b['idx']])
            sup.append(support('pair', k, exclude, model)); names.append(k)
        score = (min(sup), sum(sup))
        if best is None or score > best[0]:
            i = sup.index(min(sup))
            best = (score, names[i], ''.join(opt))
    return best[0][0], best[1], best[2]
