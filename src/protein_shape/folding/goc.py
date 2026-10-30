"""ctypes wrapper for gocore.c, the Langevin integrator of the Go model. The shared library is compiled on first
use into build/gocore/ at the repository root (or $PROTEIN_SHAPE_BUILD); a C compiler (cc) is required."""
import ctypes as C
import os
import subprocess
from pathlib import Path

import numpy as np

from . import gomodel as G
from ..paths import REPO

SRC = Path(__file__).with_name('gocore.c')


def _library():
    out = Path(os.environ.get('PROTEIN_SHAPE_BUILD', REPO / 'build')) / 'gocore' / 'libgocore.so'
    if not out.exists() or out.stat().st_mtime < SRC.stat().st_mtime:
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cc', '-O3', '-shared', '-fPIC', '-o', str(out), str(SRC), '-lm'], check=True)
    return C.CDLL(str(out))


_lib = _library()
_P = C.POINTER(C.c_double)
_lib.model_new.restype = C.c_void_p
_lib.forces.argtypes = [C.c_void_p, _P, _P, _P]
_lib.run.argtypes = [C.c_void_p, _P, _P, C.c_int, C.c_double, C.c_double, C.c_double, _P]
_lib.seed_rng.argtypes = [C.c_uint64]


def _d(a): return np.ascontiguousarray(a, dtype=np.float64)
def _p(a): return a.ctypes.data_as(_P)


class CoreModel:
    def __init__(self, ca_A, contacts, mass=100.0):
        pos = np.asarray(ca_A) / 10.0
        n = len(pos)
        self.n = n
        r0 = _d(np.linalg.norm(pos[1:] - pos[:-1], axis=1))
        a, b = pos[:-2] - pos[1:-1], pos[2:] - pos[1:-1]
        th0 = _d(np.arccos((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)))
        phi0 = _d(G.dihedral(pos))
        ci = np.array([c[0] for c in contacts], dtype=np.int32); cj = np.array([c[1] for c in contacts], dtype=np.int32)
        cs = _d([np.linalg.norm(pos[i] - pos[j]) for i, j in contacts])
        ex = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if j - i <= 3:
                    ex[i, j] = ex[j, i] = 1
        for i, j in contacts:
            ex[i, j] = ex[j, i] = 1
        self._keep = (r0, th0, phi0, ci, cj, cs, ex)
        Ip = C.POINTER(C.c_int); Cp = C.POINTER(C.c_char)
        self.m = C.c_void_p(_lib.model_new(n, C.c_double(mass), _p(r0), C.c_double(2 * 100.0 * 100.0), _p(th0),
                                           C.c_double(2 * 20.0), _p(phi0), C.c_double(1.0), C.c_double(0.5), len(ci),
                                           ci.ctypes.data_as(Ip), cj.ctypes.data_as(Ip), _p(cs),
                                           ex.ctypes.data_as(Cp), C.c_double(0.4)))
        self.pos0 = pos
        self.sigma = cs

    def forces(self, x):
        x = _d(x).reshape(-1); f = np.zeros_like(x); e = np.zeros(3)
        _lib.forces(self.m, _p(x), _p(f), _p(e))
        return f.reshape(-1, 3), e

    def run(self, x, v, nsteps, T_star, gamma=0.5, dt=0.02):
        """Advance x, v (contiguous float64 [n, 3]) in place; returns the last (bonded, contact, repulsion) energies."""
        e = np.zeros(3)
        _lib.run(self.m, _p(x), _p(v), int(nsteps), C.c_double(T_star), C.c_double(gamma), C.c_double(dt), _p(e))
        return e

    @staticmethod
    def seed(s): _lib.seed_rng(int(s))


def design(summary, name, struct_dir):
    """Build the model of one structure listed in a summary JSON ({name: {pdb, nres, ss: [[type, start, end], ...]}}),
    keeping residues 1 to the end of the last secondary-structure element."""
    s = summary[name]
    core_end = max(b for _, a, b in s['ss'])
    ca, con, names, labels = G.load_design(s['pdb'], core_end, struct_dir=struct_dir)
    cm = CoreModel(ca, con)
    ob = G.Observables(ca, con, cm.sigma)
    return s, core_end, ca, con, cm, ob
