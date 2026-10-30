"""Canonical topology strings and the edit graph over open sheets (terminal additions and deletions,
circular permutations and terminal flips), plus chain reversal.
"""
from functools import lru_cache

from . import topology as T

EDGE_TYPES = ('terminal_N', 'terminal_C', 'cp', 'flip_N', 'flip_C')


@lru_cache(maxsize=None)
def topology_set(n):
    """All canonical Minami strings of n strands (Koga generator list)."""
    return frozenset(T.minami(o, s) for o, s in T.enumerate_koga(n))


def _flip(signs):
    return tuple('-' if s == '+' else '+' for s in signs)


def _normalise(order, signs):
    return (order, signs if signs[0] == '+' else _flip(signs))


def orbit(order, signs):
    """The two sign-normalised images (the global sign flip is folded into normalisation)."""
    order, signs = tuple(order), tuple(signs)
    return [_normalise(order, signs), _normalise(order[::-1], signs[::-1])]


def canon(order, signs):
    """Minami string of the orbit of (order, signs); raises if the list is not a transversal."""
    n = len(order)
    hits = {T.minami(o, s) for o, s in orbit(order, signs)} & topology_set(n)
    if len(hits) != 1:
        raise ValueError(f'orbit of {order}{signs} hits {len(hits)} list entries')
    return next(iter(hits))


def canon_str(s):
    return canon(*T.parse(s))


def reverse(t):
    """Chain reversal (strand k -> n+1-k, every direction flipped), as a canonical Minami string."""
    order, signs = T.parse(t)
    n = len(order)
    return canon(tuple(n + 1 - x for x in order), tuple('-' if s == '+' else '+' for s in signs))


# ---------------------------------------------------------------- edits ----

def _renumber_without(order, strand):
    """Remove `strand` from the sheet order and close the numbering gap."""
    return tuple(x - 1 if x > strand else x for x in order if x != strand)


def deletions(topo):
    """Valid terminal deletions: list of (edge_type, result) for strand 1 and strand n."""
    order, signs = T.parse(topo)
    n = len(order)
    out = []
    if n <= 3:
        return out            # 2-strand sheets are outside the frames
    for strand, et in ((1, 'terminal_N'), (n, 'terminal_C')):
        pos = order.index(strand)
        if pos in (0, n - 1):
            out.append((et, canon(_renumber_without(order, strand), signs[:pos] + signs[pos + 1:])))
    return out


def splitting_deletions(topo):
    """Terminal strands (1 and n) that sit in the sheet interior: removal would split the sheet."""
    order, _ = T.parse(topo)
    n = len(order)
    return [et for strand, et in ((1, 'terminal_N'), (n, 'terminal_C'))
            if order.index(strand) not in (0, n - 1)]


def additions(topo):
    """New N- or C-terminal strand at either sheet edge with either orientation (8 raw edits)."""
    order, signs = T.parse(topo)
    n = len(order)
    out = []
    for et in ('terminal_N', 'terminal_C'):
        base = tuple(x + 1 for x in order) if et == 'terminal_N' else order
        new = 1 if et == 'terminal_N' else n + 1
        for side in ('left', 'right'):
            for sg in '+-':
                if side == 'left':
                    out.append((et, canon((new,) + base, (sg,) + signs)))
                else:
                    out.append((et, canon(base + (new,), signs + (sg,))))
    return out


def circular_permutations(topo):
    """Cut after strand k (k = 1..n-1): strands k+1..n become 1..n-k, strands 1..k become n-k+1..n."""
    order, signs = T.parse(topo)
    n = len(order)
    out = []
    for k in range(1, n):
        new = tuple(x - k if x > k else x + n - k for x in order)
        out.append(('cp', canon(new, signs)))
    return out


def flips(topo):
    """Reverse the orientation of strand 1 or strand n (a hairpin-swap proxy)."""
    order, signs = T.parse(topo)
    n = len(order)
    out = []
    for strand, et in ((1, 'flip_N'), (n, 'flip_C')):
        pos = order.index(strand)
        sg = list(signs)
        sg[pos] = '-' if sg[pos] == '+' else '+'
        out.append((et, canon(order, tuple(sg))))
    return out


def same_level_edges(topo):
    return circular_permutations(topo) + flips(topo)


# ---------------------------------------------------------------- graph ----

class EditGraph:
    """Undirected typed graph over topologies with n_min..n_max strands.

    edges[(a, b)] = set of edge types (a, b ordered pairs, both directions stored).
    The edge label of a terminal add/delete is the same from both ends.
    """

    def __init__(self, n_min=3, n_max=6):
        self.n_min, self.n_max = n_min, n_max
        self.nodes = {n: sorted(topology_set(n)) for n in range(n_min, n_max + 1)}
        self.level = {t: n for n, ts in self.nodes.items() for t in ts}
        self.adj = {t: {} for t in self.level}       # t -> {nbr: set(types)}
        self.parents = {t: [] for t in self.level}   # distinct (n-1) neighbours via deletion
        self.children = {t: set() for t in self.level}
        self.n_del_ops = {t: 0 for t in self.level}
        self.n_split = {t: 0 for t in self.level}
        for t, n in self.level.items():
            for et, u in deletions(t):
                self.n_del_ops[t] += 1
                if n - 1 >= n_min:
                    self._add(t, u, et)
                    if u not in self.parents[t]:
                        self.parents[t].append(u)
                    self.children[u].add(t)
            self.n_split[t] = len(splitting_deletions(t))
            if n < n_max:
                for et, u in additions(t):
                    self._add(t, u, et)
                    self.children[t].add(u)
                    if t not in self.parents[u]:
                        self.parents[u].append(t)
            for et, u in same_level_edges(t):
                if u != t:
                    self._add(t, u, et)

    def _add(self, a, b, et):
        self.adj[a].setdefault(b, set()).add(et)
        self.adj[b].setdefault(a, set()).add(et)

    def neighbours(self, t, types=None):
        return [u for u, ts in self.adj[t].items() if types is None or ts & set(types)]

    def distances(self, sources, max_d=None, types=None):
        """Multi-source BFS over edges whose types intersect `types` (None = all)."""
        dist = {s: 0 for s in sources}
        frontier = list(sources)
        d = 0
        while frontier and (max_d is None or d < max_d):
            d += 1
            nxt = []
            for a in frontier:
                for b in self.neighbours(a, types):
                    if b not in dist:
                        dist[b] = d
                        nxt.append(b)
            frontier = nxt
        return dist

    def edge_types(self, a, b):
        return self.adj[a].get(b, set())
