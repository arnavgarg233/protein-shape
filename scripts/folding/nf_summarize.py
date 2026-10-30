"""Per-design folding summaries for the NF designs: fraction folded, Kaplan-Meier medians, restricted mean times,
closure-before-threading episodes and end states, plus log-rank tests of NF1 against the other designs.

    python scripts/folding/nf_summarize.py --dir DIR [--tag rel0.92]

DIR holds design_info.csv and the fold_traj.py outputs (summary_, calib_ and ts_ files); writes
design_summary_TAG.csv and traj_events_TAG.csv there.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from protein_shape.folding.survival import km_median, km_rmst, logrank

REC = 1000
CAP = 3000000   # common trajectory cap (steps) for every design
NFS = ['NF1', 'NF7', 'NF2', 'NF4', 'NF5', 'NF6', 'NF8']


def episodes(q, qclose, gown, g0, qthr):
    """Backtracking descriptors for one trajectory (arrays per record), evaluated before the first fold record.

    q_backtrack : number of episodes where Q had reached >= 0.5 and then dropped by >= 0.25 from its running maximum (running max re-armed after each).
    closed_unthreaded_open : episodes where the loop closure (Q_close >= 0.6) was present with the thread NOT in place (|g - g0| > 0.5), later
                             followed by loss of the closure (Q_close < 0.3), i.e. loop closed first, then re-opened.
    first_closure_unthreaded : at the first record with Q_close >= 0.6, was the thread out of place?
    dwell_closed_unthreaded : records spent closed-and-unthreaded.
    """
    ev_q, runmax = 0, q[0]
    for x in q:
        runmax = max(runmax, x)
        if runmax >= 0.5 and x <= runmax - 0.25:
            ev_q += 1; runmax = x
    closed = qclose >= 0.6
    thr_ok = np.abs(gown - g0) <= 0.5
    cu = closed & ~thr_ok
    ev_c, state = 0, False
    for k in range(len(q)):
        if cu[k]:
            state = True
        elif state and qclose[k] < 0.3:
            ev_c += 1; state = False
    first = np.argmax(closed) if closed.any() else -1
    return dict(q_backtrack=ev_q, closed_unthreaded_open=ev_c, ever_closed=bool(closed.any()),
                first_closure_unthreaded=(bool(~thr_ok[first]) if first >= 0 else np.nan), dwell_closed_unthreaded=int(cu.sum()),
                dwell_closed_threaded=int((closed & thr_ok).sum()))


def main(D, tag):
    info = pd.read_csv(D / 'design_info.csv').set_index('nf')
    rows, evrows, allsum = [], [], []
    for nf in NFS:
        f = D / f'summary_{nf}_{tag}.csv'
        if not f.exists():
            continue
        sm = pd.read_csv(f)
        cal = json.load(open(D / f'calib_{nf}_{tag}.json'))
        ts = np.load(D / f'ts_{nf}_{tag}.npz')
        maxsteps = CAP
        n = len(sm)
        tq = sm.t_fold_Q.fillna(np.inf).values
        te = sm.t_fold_ent.fillna(np.inf).values
        cens = float(sm.steps_run.max())
        t_obs_q = np.where(np.isfinite(tq), tq, sm.steps_run.values)
        t_obs_e = np.where(np.isfinite(te), te, sm.steps_run.values)
        ev_q = np.isfinite(tq).astype(int); ev_e = np.isfinite(te).astype(int)
        # backtracking per trajectory
        for k in range(n):
            a = ts[f'traj{k}']
            kq = int(sm.loc[k, 't_fold_ent'] / REC) if np.isfinite(te[k]) else len(a)
            kq = max(kq, 1)
            e = episodes(a[:kq, 0], a[:kq, 4], a[:kq, 1], cal['key_g0'], cal['Qthr'])
            e.update(nf=nf, traj=k, t_q=tq[k], t_ent=te[k])
            # end state classification
            ent_bad = (abs(sm.loc[k, 'gkey_final'] - cal['key_g0']) > 0.5 if cal['entangled'] else False) or sm.loc[k, 'Gall_final'] > cal['Gthr']
            if ev_e[k]:
                st = 'folded'
            elif sm.loc[k, 'Q_final'] >= 0.7 and ent_bad:
                st = 'near-native, wrong entanglement'
            elif sm.loc[k, 'Q_final'] >= cal['Qthr'] and not ent_bad:
                st = 'near-native (entanglement ok)'
            elif sm.loc[k, 'Q_final'] >= 0.7:
                st = 'high-Q unfolded'
            else:
                st = 'partly folded / unfolded'
            e['end_state'] = st
            # time from first Q >= Qthr to the entanglement-correct fold
            e['t_q_to_ent'] = (te[k] - tq[k]) if np.isfinite(te[k]) and np.isfinite(tq[k]) else np.nan
            evrows.append(e)
        evdf = pd.DataFrame([r for r in evrows if r['nf'] == nf])
        rows.append(dict(nf=nf, L=info.loc[nf, 'L'], contacts=info.loc[nf, 'n_contacts'], RCO=info.loc[nf, 'RCO'], Tf=cal['Tf'],
                         T_run=cal['T_star'], Qthr=cal['Qthr'], entangled=cal['entangled'], n_traj=n, max_steps=maxsteps,
                         frac_folded_Q=ev_q.mean(), frac_folded_ent=ev_e.mean(),
                         med_t_Q_KM=km_median(t_obs_q, ev_q), med_t_ent_KM=km_median(t_obs_e, ev_e),
                         rmst_ent=km_rmst(t_obs_e, ev_e, maxsteps), rmst_Q=km_rmst(t_obs_q, ev_q, maxsteps),
                         q_backtrack_mean=evdf.q_backtrack.mean(), frac_q_backtrack=(evdf.q_backtrack > 0).mean(),
                         closed_unthreaded_open_mean=evdf.closed_unthreaded_open.mean(), frac_cuo=(evdf.closed_unthreaded_open > 0).mean(),
                         frac_first_closure_unthreaded=evdf.first_closure_unthreaded.astype(float).mean(),
                         frac_near_native_wrong_ent=(evdf.end_state == 'near-native, wrong entanglement').mean(),
                         mean_t_q_to_ent=evdf.t_q_to_ent.mean()))
        allsum.append(sm)
    df = pd.DataFrame(rows)
    df.to_csv(D / f'design_summary_{tag}.csv', index=False)
    pd.DataFrame(evrows).to_csv(D / f'traj_events_{tag}.csv', index=False)
    pd.set_option('display.width', 250)
    print(df.round(3).to_string())
    # log-rank of NF1 and NF7 against the other designs (first passage to an entanglement-correct fold)
    S = {nf: pd.read_csv(D / f'summary_{nf}_{tag}.csv') for nf in df.nf}
    def tev(nf):
        s = S[nf]; t = np.where(s.folded_ent, s.t_fold_ent, s.steps_run); return t, s.folded_ent.astype(int).values
    for a in ['NF1', 'NF7']:
        if a not in S:
            continue
        for b in ['NF2', 'NF4', 'NF5', 'NF6']:
            if b in S:
                z, p = logrank(*tev(a), *tev(b))
                print(f'log-rank {a} vs {b}: z={z:.2f} p={p:.3g}')
        ctrl = [b for b in ['NF2', 'NF4', 'NF5', 'NF6'] if b in S]
        if ctrl:
            t2 = np.concatenate([tev(b)[0] for b in ctrl]); e2 = np.concatenate([tev(b)[1] for b in ctrl])
            z, p = logrank(*tev(a), t2, e2); print(f'log-rank {a} vs pooled NF2/4/5/6: z={z:.2f} p={p:.3g}')
    # RCO regression on 6 non-knot designs
    six = df[df.nf.isin(['NF1', 'NF7', 'NF2', 'NF4', 'NF5', 'NF6'])].copy()
    if len(six) == 6:
        for col in ['med_t_ent_KM', 'rmst_ent']:
            y = np.log10(six[col].replace(np.inf, np.nan))
            ok = y.notna()
            if ok.sum() >= 4:
                r, p = stats.pearsonr(six.RCO[ok], y[ok]); rho, ps = stats.spearmanr(six.RCO[ok], y[ok])
                sl, ic = np.polyfit(six.RCO[ok], y[ok], 1)
                res = y - (sl * six.RCO + ic)
                print(f'{col}: log10 t vs RCO (n={ok.sum()}): pearson r={r:.2f} p={p:.2f}; spearman {rho:.2f} p={ps:.2f}; slope={sl:.2f}')
                print('   residuals (log10 steps):', dict(zip(six.nf, res.round(2))))
                ctr = six[six.nf.isin(['NF2', 'NF4', 'NF5', 'NF6']) & ok]
                if len(ctr) >= 3:
                    sl2, ic2 = np.polyfit(ctr.RCO, np.log10(ctr[col]), 1)
                    pred = {n_: float(np.log10(six.set_index('nf').loc[n_, col]) - (sl2 * six.set_index('nf').loc[n_, 'RCO'] + ic2)) for n_ in ['NF1', 'NF7']}
                    print('   control-only fit (NF2/4/5/6) residuals for NF1, NF7:', {k: round(v, 2) for k, v in pred.items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--dir', required=True)
    ap.add_argument('--tag', default='rel0.92')
    a = ap.parse_args()
    main(Path(a.dir), a.tag)
