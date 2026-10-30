"""Score the 11,520 six-strand topologies with the models fitted on four strands (Fig. 2). The models
that need grammar features are not scored, since those features exist for four and five strands only.

    python scripts/composition/score_6_strand.py [--input-dir DIR] [--external-dir DIR] [--out OUT.json]
"""
import argparse
import json
from pathlib import Path

from protein_shape import geometry_score as H
from protein_shape import minami_data
from protein_shape.paths import DATA, RESULTS

SCORED = ['M1_rules', 'M2_rules_geom', 'M3_plus_inflow', 'M5a_plus_both_faces', 'M6_plus_pp_cross_same']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input-dir', default=str(DATA), help='frame, edit-feature and grammar-feature tables')
    ap.add_argument('--external-dir', default=str(DATA / 'external'), help='holds minami/fig3.xlsx')
    ap.add_argument('--out', default=str(RESULTS / 'composition' / 'score_6_strand.json'))
    a = ap.parse_args()
    H.INPUT_DIR, minami_data.EXT = Path(a.input_dir), Path(a.external_dir)
    H.MODELS = {k: v for k, v in H.MODELS.items() if k in SCORED}
    X4, y4 = H.features(4)
    X6, y6 = H.features(6)
    out = {o: H.test(X4, y4, X6, y6, o) for o in ('n_Hgroups', 'nH_ab')}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    for o, res in out.items():
        print(o)
        for k, v in res.items():
            if k == 'T1':
                print('  T1', {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
            else:
                print(f"  {k:26s} auroc {v['auroc']:.3f}  spearman_obs {v['spearman_observed']:.3f}  "
                      f"n={v['n']} obs={v['n_observed']}")


if __name__ == '__main__':
    main()
