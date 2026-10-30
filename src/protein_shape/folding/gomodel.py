"""C-alpha structure-based (Go) model of Clementi, Nymeyer and Onuchic (J. Mol. Biol. 298, 937, 2000; Eq. A1),
with native-contact (Q) and entanglement observables. Units: eps = 1 kJ/mol, nm, ps, amu (100 amu per bead).

E = sum Kr (r - r0)^2 + sum Ktheta (theta - theta0)^2 + sum [Kphi1 (1 - cos(phi - phi0)) + Kphi3 (1 - cos 3(phi - phi0))]
    + sum_native eps [5 (s/r)^12 - 6 (s/r)^10] + sum_nonnative eps (4 A / r)^12,
with Kr = 100 eps, Ktheta = 20 eps, Kphi1 = eps, Kphi3 = 0.5 eps, native pairs from heavy atoms within 4.5 A and
|i - j| >= 4, and pairs |i - j| <= 3 excluded from the repulsion. A contact is formed when r <= 1.2 s.
"""
from pathlib import Path

import numpy as np

from . import entanglement as E

KB = 0.0083144626           # kJ/mol/K (eps = 1 kJ/mol)
SKIP, FLANK = 5, 6          # terminal and loop-flank exclusions of Rana et al. (2024)
GAMMA = 1.2                 # contact-formation cutoff, as a multiple of the native distance


def load_design(pdb, core_end, model=1, struct_dir='.', ca_cut=None):
    """C-alpha coordinates (A), native contacts (0-based i<j) and residue names for residues 1..core_end of
    chain A of {struct_dir}/{pdb}.pdb. With ca_cut (A), contacts use a C-alpha distance cutoff instead of the
    heavy-atom rule, for backbone-only structures."""
    path = Path(struct_dir) / f'{pdb}.pdb'
    labels, ca, heavy, names, sg = E.load_residues(path, 'A', f'A:1-{core_end}', model)
    if ca_cut:
        c = np.asarray(ca, float)
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        ii, jj = np.where((d <= float(ca_cut)) & np.triu(np.ones_like(d, bool), 4))
        contacts = [(int(i), int(j)) for i, j in zip(ii, jj)]
    else:
        contacts = E.native_contacts(heavy, 4, 4.5)
    return ca, contacts, names, labels


def dihedral(p):
    """Dihedral angles for all consecutive quadruples, IUPAC sign (the OpenMM convention)."""
    b0 = p[:-3] - p[1:-2]
    b1 = p[2:-1] - p[1:-2]
    b2 = p[3:] - p[2:-1]
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - (b0 * b1).sum(1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(1, keepdims=True) * b1
    x = (v * w).sum(1)
    y = (np.cross(b1, v) * w).sum(1)
    return np.arctan2(y, x)


def build_system(ca_A, contacts, mass=100.0):
    """The same model as an OpenMM System, for checking the C integrator. Returns system, positions (nm), sigma (nm)."""
    import openmm as mm
    pos = np.asarray(ca_A) / 10.0
    n = len(pos)
    sysm = mm.System()
    for _ in range(n):
        sysm.addParticle(mass)
    eps = 1.0
    # OpenMM harmonic terms carry a factor 1/2, so k = 2 K; Kr = 100 eps/A^2 = 1e4 eps/nm^2
    f = mm.HarmonicBondForce()
    for i in range(n - 1):
        f.addBond(i, i + 1, float(np.linalg.norm(pos[i + 1] - pos[i])), 2 * 100.0 * 100.0 * eps)
    f.setForceGroup(0); sysm.addForce(f)
    f = mm.HarmonicAngleForce()
    for i in range(n - 2):
        a, b = pos[i] - pos[i + 1], pos[i + 2] - pos[i + 1]
        th = np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
        f.addAngle(i, i + 1, i + 2, float(th), 2 * 20.0 * eps)
    f.setForceGroup(0); sysm.addForce(f)
    f = mm.PeriodicTorsionForce()
    phi = dihedral(pos)
    for i, p0 in enumerate(phi):
        # eps*(1 - cos(n (phi - phi0))) = eps*(1 + cos(n phi - (n phi0 + pi)))
        f.addTorsion(i, i + 1, i + 2, i + 3, 1, float((p0 + np.pi) % (2 * np.pi)), 1.0 * eps)
        f.addTorsion(i, i + 1, i + 2, i + 3, 3, float((3 * p0 + np.pi) % (2 * np.pi)), 0.5 * eps)
    f.setForceGroup(0); sysm.addForce(f)
    cf = mm.CustomBondForce('eps*(5*(s/r)^12-6*(s/r)^10)')
    cf.addGlobalParameter('eps', eps)
    cf.addPerBondParameter('s')
    sigma = []
    for i, j in contacts:
        s = float(np.linalg.norm(pos[i] - pos[j]))
        sigma.append(s)
        cf.addBond(int(i), int(j), [s])
    cf.setForceGroup(1); sysm.addForce(cf)
    rf = mm.CustomNonbondedForce('eps*(0.4/r)^12')
    rf.addGlobalParameter('eps', eps)
    rf.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
    for _ in range(n):
        rf.addParticle([])
    nat = set((int(i), int(j)) for i, j in contacts)
    for i in range(n):
        for j in range(i + 1, n):
            if j - i <= 3 or (i, j) in nat:
                rf.addExclusion(i, j)
    rf.setForceGroup(2); sysm.addForce(rf)
    return sysm, pos, np.array(sigma)


class Observables:
    """Q and partial linking numbers of all native contacts for a C-alpha trace (nm or A)."""

    def __init__(self, ca_A, contacts, sigma_nm):
        self.ca0 = np.asarray(ca_A)
        self.ij = np.array(contacts)
        self.s = sigma_nm
        self.N = len(ca_A)
        self.gN0, self.gC0 = self.partial_all(self.ca0)
        self.tot0 = np.maximum(np.abs(self.gN0), np.abs(self.gC0))
        # key loop: the native contact with the largest |g|; it counts as entangled only if |g| >= 0.6
        self.key = int(np.argmax(self.tot0))
        self.key_ij = tuple(int(x) for x in self.ij[self.key])
        self.key_tail = 'N' if abs(self.gN0[self.key]) >= abs(self.gC0[self.key]) else 'C'
        self.key_g0 = float(self.gN0[self.key] if self.key_tail == 'N' else self.gC0[self.key])
        self.entangled = bool(self.tot0.max() >= 0.6)

    def Q(self, ca_nm):
        d = np.linalg.norm(ca_nm[self.ij[:, 0]] - ca_nm[self.ij[:, 1]], axis=1)
        return float((d <= GAMMA * self.s).mean()), d <= GAMMA * self.s

    def partial_all(self, ca):
        """(gN, gC) for every native contact; ca in any length unit."""
        g = E.Gauss(np.asarray(ca))
        P, N = g.P, g.N
        i, j = self.ij[:, 0], self.ij[:, 1]

        def blk(m0, m1, n0, n1):
            m1 = np.minimum(m1, N - 2); n1 = np.minimum(n1, N - 2)
            m0 = np.maximum(m0, 0); n0 = np.maximum(n0, 0)
            ok = (m1 >= m0) & (n1 >= n0)
            v = P[np.where(ok, m1 + 1, 0), np.where(ok, n1 + 1, 0)] - P[np.where(ok, m0, 0), np.where(ok, n1 + 1, 0)] \
                - P[np.where(ok, m1 + 1, 0), np.where(ok, n0, 0)] + P[np.where(ok, m0, 0), np.where(ok, n0, 0)]
            return np.where(ok, v, 0.0)
        gn = blk(np.full_like(i, SKIP), i - FLANK, i, j - 1)
        gc = blk(i, j - 1, j + FLANK, np.full_like(i, N - 2 - SKIP))
        return gn, gc

    def ent_state(self, ca_nm):
        """(key g in its native tail, fraction of native contacts whose g_N or g_C moved by more than 0.5)."""
        gn, gc = self.partial_all(ca_nm)
        changed = (np.abs(gn - self.gN0) > 0.5) | (np.abs(gc - self.gC0) > 0.5)
        gk = gn[self.key] if self.key_tail == 'N' else gc[self.key]
        return float(gk), float(changed.mean())
