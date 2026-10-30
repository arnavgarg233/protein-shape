"""Independent folding trajectories of one structure at T = TREL * Tf, each from a different unfolded REMD snapshot
(Q < 0.25) with fresh velocities, run until folded (Q >= Qthr with native entanglement) or MAXSTEPS.

    python scripts/folding/fold_traj.py NAME TREL NTRAJ MAXSTEPS --summary SUMMARY.json --structures DIR --out DIR [--seed0 1000]

Reads remd_NAME.{json,npz} from --out and writes calib_NAME_relX.json, summary_NAME_relX.csv (one row per trajectory)
and ts_NAME_relX.npz (Q, key g in both tails, changed-entanglement fraction and Q around the key loop every REC steps).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from protein_shape.folding import goc

REC = 1000
GAMMA_FRIC = 0.5
DT = 0.02


def observe(ob, close_idx, x):
    """Q, g_key (own tail), g_key (other tail), G_all, Q_close at positions x (nm)."""
    q, formed = ob.Q(x)
    gn, gc = ob.partial_all(x * 10.0)
    k = ob.key
    own = gn[k] if ob.key_tail == 'N' else gc[k]
    oth = gc[k] if ob.key_tail == 'N' else gn[k]
    changed = (np.abs(gn - ob.gN0) > 0.5) | (np.abs(gc - ob.gC0) > 0.5)
    qc = float(formed[close_idx].mean()) if len(close_idx) else np.nan
    return q, float(own), float(oth), float(changed.mean()), qc


def main(nf, summary, struct_dir, OUT, trel, ntraj, maxsteps, seed0=1000):
    s, core_end, ca, con, cm, ob = goc.design(summary, nf, struct_dir)
    info = json.load(open(OUT / f'remd_{nf}.json'))
    Tf = info['Tf_cv']
    T = trel * Tf
    tag = f'rel{trel:.2f}'
    snaps = np.load(OUT / f'remd_{nf}.npz')['snaps']
    qs = np.array([ob.Q(x)[0] for x in snaps])
    unf = snaps[qs < 0.25]
    i0, j0 = ob.key_ij
    cidx = np.array([c for c, (a, b) in enumerate(ob.ij) if abs(a - i0) <= 5 and abs(b - j0) <= 5])
    # calibration: a run started from the native structure at T
    cm.seed(seed0 - 1)
    x = np.ascontiguousarray(cm.pos0.copy()); v = np.random.default_rng(seed0).normal(0, np.sqrt(T / 100.0), x.shape)
    cm.run(x, v, 20000, T, GAMMA_FRIC, DT)
    cal = np.array([observe(ob, cidx, (cm.run(x, v, REC, T, GAMMA_FRIC, DT), x)[1]) for _ in range(300)])
    Qnat = float(cal[:, 0].mean())
    Qthr = float(np.floor(0.9 * Qnat * 100) / 100)
    Gthr = float(max(0.05, np.quantile(cal[:, 3], 0.99)))
    calib = dict(nf=nf, T_star=T, Tf=Tf, trel=trel, Q_native_mean=Qnat, Q_native_p05=float(np.quantile(cal[:, 0], 0.05)), Qthr=Qthr,
                 Gthr=Gthr, key_ij=list(ob.key_ij), key_tail=ob.key_tail, key_g0=ob.key_g0, entangled=ob.entangled,
                 gkey_native_sd=float(cal[:, 1].std()), n_unfolded_starts=int(len(unf)), n_close_contacts=int(len(cidx)))
    json.dump(calib, open(OUT / f'calib_{nf}_{tag}.json', 'w'), indent=1)
    print(nf, 'calib', calib, flush=True)

    ts, rows = {}, []
    t0 = time.time()
    for k in range(ntraj):
        seed = seed0 + k
        cm.seed(seed)
        rng = np.random.default_rng(seed)
        x = np.ascontiguousarray(unf[rng.integers(len(unf))].copy())
        v = rng.normal(0, np.sqrt(T / 100.0), x.shape)
        rec = []
        t_q = t_ent = None
        for step in range(maxsteps // REC):
            cm.run(x, v, REC, T, GAMMA_FRIC, DT)
            o = observe(ob, cidx, x)
            rec.append(o)
            q, gk, _, gall, _ = o
            ent_ok = (abs(gk - ob.key_g0) <= 0.5 if ob.entangled else True) and gall <= Gthr
            if t_q is None and q >= Qthr:
                t_q = (step + 1) * REC
            if q >= Qthr and ent_ok:
                t_ent = (step + 1) * REC
                break
        rec = np.array(rec, dtype=np.float32)
        ts[f'traj{k}'] = rec
        fin = rec[-10:]
        rows.append(dict(nf=nf, traj=k, seed=seed, trel=trel, T_star=T, Tf=Tf, Qthr=Qthr, Gthr=Gthr,
                         t_fold_Q=t_q if t_q is not None else np.nan, t_fold_ent=t_ent if t_ent is not None else np.nan,
                         steps_run=len(rec) * REC, folded_Q=t_q is not None, folded_ent=t_ent is not None,
                         Q_final=float(fin[:, 0].mean()), gkey_final=float(fin[:, 1].mean()), Gall_final=float(fin[:, 3].mean()),
                         Q_max=float(rec[:, 0].max())))
        print(f'{nf} traj {k}: t_Q={t_q} t_ent={t_ent} steps={len(rec) * REC} Qfin={fin[:, 0].mean():.2f} '
              f'elapsed {time.time() - t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(OUT / f'summary_{nf}_{tag}.csv', index=False)
        np.savez_compressed(OUT / f'ts_{nf}_{tag}.npz', **ts)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('name')
    ap.add_argument('trel', type=float)
    ap.add_argument('ntraj', type=int)
    ap.add_argument('maxsteps', type=int)
    ap.add_argument('--summary', required=True, help='structure summary JSON')
    ap.add_argument('--structures', required=True, help='directory holding NAME.pdb files')
    ap.add_argument('--out', default='results/folding')
    ap.add_argument('--seed0', type=int, default=1000)
    a = ap.parse_args()
    main(a.name, json.load(open(a.summary)), a.structures, Path(a.out), a.trel, a.ntraj, a.maxsteps, a.seed0)
