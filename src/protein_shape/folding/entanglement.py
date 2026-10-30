"""Native contacts and partial Gaussian linking numbers of a C-alpha trace, the entanglement measure of
Nissley et al. (2022) and Rana et al. (2024), as used by the Go model."""
import numpy as np
from scipy.spatial import cKDTree

from .. import structure as ET

CONTACT_A = 4.5


def load_residues(path, chain=None, rng=None, model=1):
    """(labels, ca[N,3], heavy-atom coordinates per residue, residue names, cysteine SG by index)."""
    st, labels = ET.select_residues(str(path), chain, ET.parse_range(rng) if rng else None, model)
    ca, heavy, names, sg = [], [], [], {}
    for k, r in enumerate(st[0][0]):
        ca.append(r.find_atom('CA', '*').pos.tolist())
        heavy.append(np.array([a.pos.tolist() for a in r if a.element.name != 'H']))
        names.append(r.name)
        a = r.find_atom('SG', '*')
        if a is not None and r.name == 'CYS':
            sg[k] = np.array(a.pos.tolist())
    return labels, np.array(ca), heavy, names, sg


def native_contacts(heavy, min_sep=4, cutoff=CONTACT_A):
    """Residue pairs (i<j, 0-based) with a heavy-atom pair within cutoff and j-i >= min_sep."""
    pts = np.vstack(heavy)
    owner = np.concatenate([np.full(len(h), k) for k, h in enumerate(heavy)])
    pairs = cKDTree(pts).query_pairs(cutoff, output_type='ndarray')
    ij = np.unique(np.sort(np.stack([owner[pairs[:, 0]], owner[pairs[:, 1]]], 1), axis=1), axis=0)
    ij = ij[ij[:, 1] - ij[:, 0] >= min_sep]
    return [tuple(x) for x in ij]


def gauss_kernel(ca):
    """G[m,n] for segments m,n = 0..N-2 (segment m joins C-alpha m and m+1)."""
    R = 0.5 * (ca[:-1] + ca[1:])
    dR = ca[1:] - ca[:-1]
    d = R[:, None, :] - R[None, :, :]
    r3 = np.linalg.norm(d, axis=2) ** 3
    np.fill_diagonal(r3, np.inf)
    cr = np.cross(dR[:, None, :], dR[None, :, :])
    return np.einsum('mnk,mnk->mn', d, cr) / r3 / (4 * np.pi)


class Gauss:
    """Prefix sums of the Gauss kernel, so any block of segment pairs sums in constant time."""

    def __init__(self, ca):
        self.N = len(ca)
        self.G = gauss_kernel(ca)
        self.P = np.zeros((self.N, self.N))
        self.P[1:, 1:] = self.G.cumsum(0).cumsum(1)

    def block(self, m0, m1, n0, n1):
        """Sum of G over segments m0..m1, n0..n1 (inclusive, 0-based, empty -> 0)."""
        m1, n1 = min(m1, self.N - 2), min(n1, self.N - 2)
        m0, n0 = max(m0, 0), max(n0, 0)
        if m1 < m0 or n1 < n0:
            return 0.0
        P = self.P
        return P[m1 + 1, n1 + 1] - P[m0, n1 + 1] - P[m1 + 1, n0] + P[m0, n0]

    def partial(self, i, j, skip=5, flank=6):
        """(g_N, g_C) for a loop closed by 0-based residues i<j."""
        gn = self.block(skip, i - flank, i, j - 1)
        gc = self.block(i, j - 1, j + flank, self.N - 2 - skip)
        return gn, gc
