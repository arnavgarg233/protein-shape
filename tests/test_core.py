from collections import defaultdict

import numpy as np
import pandas as pd

from protein_shape import paths
from protein_shape import stats as E
from protein_shape.edits import topology_set


def test_enumeration_counts():
    assert [len(topology_set(n)) for n in (4, 5, 6)] == [96, 960, 11520]


def test_matched_arms():
    f = E.load(paths.OCC_MAIN)
    A, R, N = E.arms(f)
    assert len(f) == 2961 and (A.sum(), R.sum(), N.sum()) == (643, 1090, 90)


def _toy(seed=0, n=120):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame(dict(n=rng.integers(5, 7, n), stratum=rng.integers(2, 4, n), fd=0,
                          used=rng.integers(0, 2, n), rclass=[f'c{i // 2}' for i in range(n)]))
    a = pd.Series(rng.random(n) < 0.4)
    return f, a, ~a


def _perm_direct(f, a, b, by, rng, n_perm):
    """Per-draw reference implementation of the class-relabelling CMH-score null."""
    cl = E.cells(f, a, b, by)
    y = f.used.to_numpy()
    stats = []
    for _ in range(n_perm):
        num = 0.0
        for ia, ib in cl:
            members = np.concatenate([ia, ib])
            lab = np.r_[np.ones(len(ia), bool), np.zeros(len(ib), bool)]
            grp = defaultdict(list)
            for m, l in zip(members, lab):
                grp[f.rclass.iat[m]].append((m, l))
            units = list(grp.values())
            rng.permutation(len(units))
            by_size = defaultdict(list)
            for k, u in enumerate(units):
                by_size[len(u)].append(k)
            new = {}
            for ks in by_size.values():
                for src, dst in zip(ks, rng.permutation(ks)):
                    new[dst] = [l for _, l in units[src]]
            ja, jb = [], []
            for k, u in enumerate(units):
                for (m, _), l in zip(u, new[k]):
                    (ja if l else jb).append(m)
            a_, c_ = y[ja].sum(), y[jb].sum()
            num += a_ - len(ja) * (a_ + c_) / (len(ja) + len(jb))
        stats.append(num)
    return np.array(stats)


def test_permutation_matches_direct_implementation():
    f, a, b = _toy()
    by = ['n', 'stratum']
    _, _, fast, _ = E.perm_p(f, a, b, by, np.random.default_rng(5), return_null=True, n_perm=200)
    slow = _perm_direct(f, a, b, by, np.random.default_rng(5), 200)
    assert np.array_equal(fast, slow)


def test_mh_risk_ratio_single_stratum():
    f = pd.DataFrame(dict(used=[1, 1, 0, 0, 1, 0, 0, 0], g=0))
    a = pd.Series([True] * 4 + [False] * 4)
    rr = E.mh(f, E.cells(f, a, ~a, ['g']))[0]
    assert rr == 2.0
