"""Temperature replica exchange for one structure, with multi-histogram (MBAR-form) reweighting to find the folding
temperature Tf as the heat-capacity peak; also saves unfolded start structures from the top replica.

    python scripts/folding/remd.py NAME --summary SUMMARY.json --structures DIR --out DIR [--steps 600000 --nrep 12 --tlo 0.60 --thi 1.50]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from protein_shape.folding import goc


def wham(E, T_samples_index, Tgrid_sim, Tq):
    """MBAR-form self-consistent estimate; returns f_k and a function of weights at arbitrary T. E: all samples; Tgrid_sim: simulated T*."""
    beta = 1.0 / np.asarray(Tgrid_sim)
    K = len(beta)
    Nk = np.bincount(T_samples_index, minlength=K).astype(float)
    Es = E - E.min()
    f = np.zeros(K)
    for _ in range(5000):
        a = -beta[None, :] * Es[:, None] + (np.log(Nk) + f)[None, :]       # [M,K] log of Nk exp(f_k - beta_k E_n)
        lden = np.logaddexp.reduce(a, axis=1)                              # [M]
        fn = -np.logaddexp.reduce(-beta[None, :] * Es[:, None] - lden[:, None], axis=0)
        fn -= fn[0]
        if np.abs(fn - f).max() < 1e-9:
            f = fn
            break
        f = fn
    a = -beta[None, :] * Es[:, None] + (np.log(Nk) + f)[None, :]
    lden = np.logaddexp.reduce(a, axis=1)
    out = []
    for T in Tq:
        lw = -Es / T - lden
        lw -= np.logaddexp.reduce(lw)
        out.append(np.exp(lw))
    return out


def main(nf, summary, struct_dir, OUT, nsteps, nrep=12, Tlo=0.60, Thi=1.50, interval=500, seed=1):
    s, core_end, ca, con, cm, ob = goc.design(summary, nf, struct_dir)
    Ts = np.linspace(Tlo, Thi, nrep)
    rng = np.random.default_rng(seed)
    cm.seed(seed)
    X = [np.ascontiguousarray(cm.pos0.copy()) for _ in range(nrep)]
    V = [rng.normal(0, np.sqrt(Ts[r] / 100.0), cm.pos0.shape) for r in range(nrep)]
    perm = list(range(nrep))                       # perm[slot] = temperature index currently held by replica slot
    Es, Qs, Ti, snaps = [], [], [], []
    nsw, nacc = 0, 0
    t0 = time.time()
    nround = nsteps // interval
    for rd in range(nround):
        e = np.zeros((nrep, 3))
        for r in range(nrep):
            e[r] = cm.run(X[r], V[r], interval, Ts[perm[r]])
        for r in range(nrep):
            if rd >= nround // 10:                                     # discard first 10 % as equilibration
                Es.append(e[r].sum()); Qs.append(ob.Q(X[r])[0]); Ti.append(perm[r])
        if rd % 10 == 0 and rd >= nround // 10:
            r = perm.index(nrep - 1)
            snaps.append(X[r].copy())
        # exchanges between neighbouring temperatures (alternate even/odd)
        by_t = {perm[r]: r for r in range(nrep)}
        for k in range(rd % 2, nrep - 1, 2):
            a, b = by_t[k], by_t[k + 1]
            d = (1 / Ts[k] - 1 / Ts[k + 1]) * (e[a].sum() - e[b].sum())
            nsw += 1
            if d >= 0 or rng.random() < np.exp(d):
                nacc += 1
                perm[a], perm[b] = k + 1, k
                V[a] *= np.sqrt(Ts[k + 1] / Ts[k]); V[b] *= np.sqrt(Ts[k] / Ts[k + 1])
                e[a], e[b] = e[b].copy(), e[a].copy()
        if rd % 200 == 0:
            print(f'{nf} round {rd}/{nround} elapsed {time.time() - t0:.0f}s acc {nacc / max(nsw, 1):.2f}', flush=True)
    Es, Qs, Ti = np.array(Es), np.array(Qs), np.array(Ti)
    Tq = np.linspace(Tlo + 0.02, Thi - 0.02, 160)
    W = wham(Es, Ti, Ts, Tq)
    cv, qm = [], []
    for T, w in zip(Tq, W):
        m1 = (w * Es).sum(); var = (w * (Es - m1) ** 2).sum()
        cv.append(var / T ** 2); qm.append((w * Qs).sum())
    cv, qm = np.array(cv), np.array(qm)
    Tf_cv = float(Tq[np.argmax(cv)])
    qlo, qhi = qm[-1], qm[0]
    Tf_q = float(Tq[np.argmin(np.abs(qm - 0.5 * (qlo + qhi)))])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'remd_{nf}.npz', Ts=Ts, Tq=Tq, cv=cv, qm=qm, E=Es, Q=Qs, Ti=Ti, snaps=np.array(snaps))
    res = dict(nf=nf, n_res=len(ca), n_contacts=len(con), core_end=core_end, Tf_cv=Tf_cv, Tf_q=Tf_q, accept=nacc / max(nsw, 1),
               steps_per_replica=nsteps, Ts=Ts.tolist(), Q_lo=float(qlo), Q_hi=float(qhi), wall_s=time.time() - t0)
    json.dump(res, open(OUT / f'remd_{nf}.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'Ts'}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('name')
    ap.add_argument('--summary', required=True, help='structure summary JSON')
    ap.add_argument('--structures', required=True, help='directory holding NAME.pdb files')
    ap.add_argument('--out', default='results/folding')
    ap.add_argument('--steps', type=int, default=600000, help='steps per replica')
    ap.add_argument('--nrep', type=int, default=12)
    ap.add_argument('--tlo', type=float, default=0.60)
    ap.add_argument('--thi', type=float, default=1.50)
    a = ap.parse_args()
    main(a.name, json.load(open(a.summary)), a.structures, Path(a.out), a.steps, nrep=a.nrep, Tlo=a.tlo, Thi=a.thi)
