"""Length, native-contact count and relative contact order (RCO) of each NF design as simulated.

    python scripts/folding/nf_design_info.py --summary SUMMARY.json --structures DIR --out DIR/design_info.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

from protein_shape.folding import gomodel as G


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--summary', required=True, help='structure summary JSON')
    ap.add_argument('--structures', required=True, help='directory holding PDB.pdb files')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    S = json.load(open(a.summary))
    rows = []
    for nf, s in S.items():
        ce = max(b for _, a_, b in s['ss'])
        ca, con, _, lab = G.load_design(s['pdb'], ce, struct_dir=a.structures)
        ij = np.array(con)
        L = len(ca)
        rows.append(dict(nf=nf, pdb=s['pdb'], core_end=ce, full_len=s['nres'], L=L, n_contacts=len(con),
                         contacts_per_res=len(con) / L, RCO=float(np.abs(ij[:, 1] - ij[:, 0]).mean() / L),
                         ACO=float(np.abs(ij[:, 1] - ij[:, 0]).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.round(3).to_string())


if __name__ == '__main__':
    main()
