"""Occupancy table of the main census: for every five- to seven-strand topology, the number of TED
superfamilies that use it (L_H) and the alternative occupancy measures, plus the stratified van Elteren
tests of the OPP arm against its reference under the agreement rule.

    python scripts/census/build_occupancy.py calls.jsonl [--frame-dir DIR] [--out-table T.csv] [--out-json R.json]

The shipped data/occupancy_main.csv.gz is the table this writes (gzip -n).
"""
import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score

from protein_shape.frames import FRAME_DIR, binary, frame, van_elteren
from protein_shape.paths import RESULTS
from protein_shape.support import occupancy


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('calls', help='census call file (JSON lines)')
    ap.add_argument('--frame-dir', default=str(FRAME_DIR), help='directory with frame5.csv to frame7.csv')
    ap.add_argument('--out-table', default=str(RESULTS / 'census' / 'occupancy_main.csv'))
    ap.add_argument('--out-json', default=str(RESULTS / 'census' / 'census_tests_main.json'))
    a = ap.parse_args()
    occ, n_dom, n_ok, n_sf = occupancy(a.calls)
    f = frame(occ, a.frame_dir)
    out = dict(domains=n_dom, domains_called=n_ok, superfamilies_H=n_sf,
               codes_5to7_with_support=int((f.hybrid > 0).sum()))
    ff = f[(f.violations == 0) & (f.npar >= 2)].copy()
    ff.index = range(len(ff))
    exp = (ff.OPP >= 1) & (ff.SAME == 0)
    ref = (ff.OPP == 0) & (ff.SAME == 0) & ff.both
    ref_broad = (ff.OPP == 0) & (ff.SAME == 0)
    same = (ff.SAME >= 1) & (ff.OPP == 0)
    s7 = ff.n == 7
    by7, by = ['stratum'], ['n', 'stratum']

    out['PRIMARY_T2R1_7strand'] = van_elteren(ff, exp & s7, ref & s7, ff.L_H, by7)
    out['PC1_same_face_7strand'] = van_elteren(ff, same & s7, ref_broad & s7, ff.L_H, by7)
    f7 = f[f.n == 7].copy()
    f7.index = range(len(f7))
    out['PC2_violating_vs_ff_7strand'] = van_elteren(f7, f7.violations > 0, f7.violations == 0, f7.L_H, ['n'],
                                                      n_perm=0)  # 161k topologies: asymptotic
    ff7 = f7[f7.violations == 0]
    out['PC3_auroc_LH_vs_ecod_observed_7strand'] = float(roc_auc_score(ff7.ecod_obs, ff7.L_H))

    out['S1_binary_LH_ge_1_7strand'] = binary(ff, exp & s7, ref & s7, ff.L_H >= 1, by7)
    out['S2_pooled_5to7'] = van_elteren(ff, exp, ref, ff.L_H, by)
    for n in (5, 6):
        out[f'S3_{n}strand'] = van_elteren(ff, exp & (ff.n == n), ref & (ff.n == n), ff.L_H, by7)
    out['S4_broad_reference_7strand'] = van_elteren(ff, exp & s7, ref_broad & s7, ff.L_H, by7)
    for name, col in (('S5_alpha_beta', 'L_H_ab'), ('S6_noguard', 'L_H_noguard'), ('S7_T_level', 'L_T'),
                      ('S8_hybrid_with_unlabelled', 'hybrid'), ('S9_clusters', 'clusters'),
                      ('S10_topology_set_upper', 'set_hybrid')):
        out[name + '_7strand'] = van_elteren(ff, exp & s7, ref & s7, ff[col], by7)

    viol = f[f.violations > 0]
    for name, d in (('noise_floor_violating', viol), ('coverage_ff', f[f.violations == 0])):
        out[name] = {int(n): dict(n_topologies=int(len(g)), frac_LH_ge_1=float((g.L_H >= 1).mean()),
                                  frac_hybrid_ge_1=float((g.hybrid >= 1).mean())) for n, g in d.groupby('n')}
    Path(a.out_table).parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(a.out_table, index=False)
    Path(a.out_json).write_text(json.dumps(out, indent=2, default=float))
    brief = {k: ({kk: vv for kk, vv in v.items() if kk != 'strata'} if isinstance(v, dict) else v)
             for k, v in out.items()}
    print(json.dumps(brief, indent=1, default=lambda v: round(float(v), 4)))


if __name__ == '__main__':
    main()
