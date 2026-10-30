"""Precision-recall evidence for the occupancy scores M1 (rules) and M2 (rules and geometry) fitted on four
strands: average precision, enrichment over the base rate, precision at k and recall in the top 1% and 5%,
with bootstrap intervals, against ECOD and census occupancy at five to seven strands.

    python scripts/composition/precision_recall.py [--input-dir DIR] [--external-dir DIR] [--out-dir DIR]

Run the four score_*.py scripts first; their AUROCs are checked before anything new is computed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score

from protein_shape import geometry_score as H
from protein_shape import minami_data
from protein_shape.paths import DATA, OCC_MAIN, RESULTS

NS = (5, 6, 7)
MODELS = ('M1_rules', 'M2_rules_geom')
OUTCOMES = ('ecod', 'census')
NBOOT, SEED = 2000, 20261028
METRICS = ('base_rate', 'auroc', 'ap', 'enrichment', 'p_at_npos', 'p_at_100', 'p_at_1000',
           'recall_top1pct', 'recall_top5pct')


def load(occ_path):
    occ = pd.read_csv(occ_path)
    X4, y4 = H.features(4)
    cont4, obs4 = np.log1p(y4.n_Hgroups), (y4.freq >= 0.25).astype(int)
    data = {}
    for n in NS:
        Xn, yn = H.features(n)
        o = Xn[['topology']].merge(occ[occ.n == n][['topology', 'L_H']], on='topology', how='left').fillna(0)
        data[n] = dict(
            topology=Xn.topology.values,
            y={'ecod': (yn.freq >= 0.25).astype(int).values, 'census': (o.L_H >= 1).astype(int).values},
            s={m: H.fit_predict(X4, cont4, obs4, Xn, H.MODELS[m]) for m in MODELS})
    return data


def gate(data, res_dir):
    """The regenerated AUROCs must equal those written by the score_*.py scripts to three decimals."""
    J = lambda p: json.loads((res_dir / p).read_text())  # noqa: E731
    pub = {5: J('score_5_strand.json')['test_5strand']['n_Hgroups'],
           6: J('score_6_strand.json')['n_Hgroups'],
           7: J('score_7_strand.json')['models_7strand']['n_Hgroups']}
    geo = J('score_census.json')
    rows, ok = [], True
    for n in NS:
        for m in MODELS:
            for oc, ref in (('ecod', pub[n][m]['auroc']), ('census', geo[str(n)][m]['auroc_afdb'])):
                new = float(roc_auc_score(data[n]['y'][oc], data[n]['s'][m]))
                good = round(new, 3) == round(ref, 3)
                ok &= good
                rows.append(dict(n=n, model=m, outcome=oc, published=ref, regenerated=new,
                                 abs_diff=abs(new - ref), match3dp=bool(good)))
    return ok, rows


class Ranked:
    """Tie-aware ranking: scores grouped into tie blocks (descending); weights = bootstrap multiplicities."""

    def __init__(self, s, y):
        _, self.g = np.unique(-s, return_inverse=True)
        self.y, self.ng = y, self.g.max() + 1

    def metrics(self, w):
        g, y = self.g, self.y
        T = np.bincount(g, weights=w, minlength=self.ng)
        P = np.bincount(g, weights=w * y, minlength=self.ng)
        cT, cP = np.cumsum(T), np.cumsum(P)
        N, Pw = cT[-1], cP[-1]
        neg = T - P
        out = dict(base_rate=Pw / N)
        # AUROC with ties counted half
        negbelow = neg.sum() - np.cumsum(neg)
        out['auroc'] = float((P * negbelow + 0.5 * P * neg).sum() / (Pw * (N - Pw)))
        keep = P > 0
        out['ap'] = float((P[keep] / Pw * cP[keep] / cT[keep]).sum())
        out['enrichment'] = out['ap'] / out['base_rate']

        def hits(k):  # expected true positives in the top k, ties broken at random
            j = int(np.searchsorted(cT, k))
            before_T = cT[j - 1] if j else 0.0
            before_P = cP[j - 1] if j else 0.0
            return before_P + P[j] * (k - before_T) / T[j]
        out['p_at_npos'] = hits(Pw) / Pw
        out['p_at_100'] = hits(100) / 100 if N >= 100 else np.nan
        out['p_at_1000'] = hits(1000) / 1000 if N >= 1000 else np.nan
        k1, k5 = int(np.ceil(0.01 * N)), int(np.ceil(0.05 * N))
        out['recall_top1pct'] = hits(k1) / Pw
        out['recall_top5pct'] = hits(k5) / Pw
        return out


def summarize(draws, point):
    res = {}
    for k in point:
        v = np.array([d[k] for d in draws], float)
        res[k] = dict(estimate=float(point[k]), lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)))
    return res


def compute(data):
    rng = np.random.default_rng(SEED)
    res = {}
    for n in NS:
        d = data[n]
        N = len(d['topology'])
        ranks = {(m, oc): Ranked(d['s'][m], d['y'][oc]) for m in MODELS for oc in OUTCOMES}
        ones = np.ones(N)
        point = {key: r.metrics(ones) for key, r in ranks.items()}
        boots = {key: [] for key in ranks}
        for _ in range(NBOOT):
            w = rng.multinomial(N, np.full(N, 1.0 / N)).astype(float)
            for key, r in ranks.items():
                boots[key].append(r.metrics(w))
        res[n] = dict(n_topologies=N)
        for oc in OUTCOMES:
            ent = dict(n_positives=int(d['y'][oc].sum()))
            for m in MODELS:
                ent[m] = summarize(boots[(m, oc)], point[(m, oc)])
            diff_pt = {k: point[('M2_rules_geom', oc)][k] - point[('M1_rules', oc)][k] for k in METRICS}
            diff_dr = [{k: b2[k] - b1[k] for k in METRICS}
                       for b1, b2 in zip(boots[('M1_rules', oc)], boots[('M2_rules_geom', oc)])]
            ent['M2_minus_M1'] = summarize(diff_dr, diff_pt)
            res[n][oc] = ent
    return res


def fmt(e, p=3):
    if np.isnan(e['estimate']):
        return 'n/a'
    return f"{e['estimate']:.{p}f} [{e['lo']:.{p}f}, {e['hi']:.{p}f}]"


def table(res):
    lab = dict(base_rate='base rate', auroc='AUROC', ap='AP', enrichment='AP / base rate',
               p_at_npos='precision@k=#pos', p_at_100='precision@100', p_at_1000='precision@1000',
               recall_top1pct='recall@top1%', recall_top5pct='recall@top5%')
    L = []
    for oc, name in (('ecod', 'ECOD occupancy (freq >= 0.25)'), ('census', 'Census occupancy (AFDB L_H >= 1)')):
        for n in NS:
            r = res[n][oc]
            L.append(f"\n{name}, {n}-strand: {r['n_positives']}/{res[n]['n_topologies']} used "
                     f"(base rate {r['n_positives'] / res[n]['n_topologies']:.4f})")
            L.append(f"  {'metric':20s}{'M1_rules':26s}{'M2_rules_geom':26s}M2 - M1")
            for k in METRICS:
                p = 4 if k in ('base_rate',) else (2 if k == 'enrichment' else 3)
                L.append(f"  {lab[k]:20s}{fmt(r['M1_rules'][k], p):26s}{fmt(r['M2_rules_geom'][k], p):26s}"
                         f"{fmt(r['M2_minus_M1'][k], p)}")
    return '\n'.join(L)


def figure(data, res, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    col = {'M1_rules': '#E69F00', 'M2_rules_geom': '#0072B2'}
    lab = {'M1_rules': 'M1 rules', 'M2_rules_geom': 'M2 rules + geometry'}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0), sharey='row')
    for i, (oc, name) in enumerate((('ecod', 'ECOD'), ('census', 'AFDB census'))):
        for j, n in enumerate(NS):
            ax, y = axs[i, j], data[n]['y'][oc]
            for m in MODELS:
                p, r, _ = precision_recall_curve(y, data[n]['s'][m])
                ax.step(r, p, where='post', color=col[m], lw=1.1,
                        label=f"{lab[m]} (AP {res[n][oc][m]['ap']['estimate']:.2f})")
            base = y.mean()
            ax.axhline(base, color='0.4', ls=':', lw=0.9, label=f'base rate {base:.4f}')
            ax.set_title(f'{name}, {n}-strand\n{int(y.sum())} of {len(y)} topologies used')
            ax.set_xlim(0, 1)
            ax.set_yscale('log')
            ax.set_ylim(base / 2, 60)
            ax.set_yticks([t for t in (1e-4, 1e-3, 1e-2, 1e-1, 1) if t > base / 2])
            ax.set_yticklabels([f'{t:g}' for t in (1e-4, 1e-3, 1e-2, 1e-1, 1) if t > base / 2])
            ax.minorticks_off()
            if i == 1:
                ax.set_xlabel('Recall')
            if j == 0:
                ax.set_ylabel('Precision (log scale)')
            ax.legend(loc='upper right', frameon=False, handlelength=1.6, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out / 'precision_recall.pdf')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input-dir', default=str(DATA), help='frame, edit-feature and grammar-feature tables')
    ap.add_argument('--external-dir', default=str(DATA / 'external'), help='holds minami/fig3.xlsx')
    ap.add_argument('--occupancy', default=str(OCC_MAIN))
    ap.add_argument('--out-dir', default=str(RESULTS / 'composition'))
    a = ap.parse_args()
    H.INPUT_DIR, minami_data.EXT = Path(a.input_dir), Path(a.external_dir)
    OUT = Path(a.out_dir)
    data = load(a.occupancy)
    ok, rows = gate(data, OUT)
    print('GATE', 'PASS' if ok else 'FAIL')
    for r in rows:
        print(f"  n={r['n']} {r['model']:14s} {r['outcome']:6s} pub {r['published']:.6f} new {r['regenerated']:.6f} "
              f"{'ok' if r['match3dp'] else 'MISMATCH'}")
    if not ok:
        sys.exit(1)
    res = compute(data)
    (OUT / 'precision_recall.json').write_text(json.dumps(dict(
        seed=SEED, n_boot=NBOOT, gate=rows, results={str(k): v for k, v in res.items()}), indent=2))
    sc = pd.concat([pd.DataFrame(dict(n=n, topology=data[n]['topology'], ecod_used=data[n]['y']['ecod'],
                                      census_used=data[n]['y']['census'],
                                      M1_rules=data[n]['s']['M1_rules'], M2_rules_geom=data[n]['s']['M2_rules_geom']))
                    for n in NS])
    sc.to_csv(OUT / 'precision_recall_scores.csv.gz', index=False)
    figure(data, res, OUT)
    print(table(res))


if __name__ == '__main__':
    main()
