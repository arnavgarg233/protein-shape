"""Fit the occupancy scores on the 96 four-strand topologies and score the 960 five-strand topologies
against ECOD occupancy (Fig. 2); also reports leave-one-circular-permutation-class-out CV on four strands.

    python scripts/composition/score_5_strand.py [--input-dir DIR] [--external-dir DIR] [--out OUT.json]
"""
import argparse
import json
from pathlib import Path

from protein_shape import geometry_score as H
from protein_shape import minami_data
from protein_shape.paths import DATA, RESULTS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input-dir', default=str(DATA), help='frame, edit-feature and grammar-feature tables')
    ap.add_argument('--external-dir', default=str(DATA / 'external'), help='holds minami/fig3.xlsx')
    ap.add_argument('--out', default=str(RESULTS / 'composition' / 'score_5_strand.json'))
    a = ap.parse_args()
    H.INPUT_DIR, minami_data.EXT = Path(a.input_dir), Path(a.external_dir)
    X4, y4 = H.features(4)
    X5, y5 = H.features(5)
    out = {'dev_cv': {o: H.dev_cv(X4, y4, o) for o in ('n_Hgroups', 'nH_ab')},
           'test_5strand': {o: H.test(X4, y4, X5, y5, o) for o in ('n_Hgroups', 'nH_ab')}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=lambda v: round(float(v), 3)))


if __name__ == '__main__':
    main()
