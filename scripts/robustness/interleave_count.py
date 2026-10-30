"""Matched OPP-versus-reference contrasts with the number of interleaved crossover pairs (capped at 4)
added to the strata, under lenient and agreement credit.

Run: python scripts/robustness/interleave_count.py
"""
import argparse
import gzip
import json

import numpy as np
from protein_shape import paths
from protein_shape import stats as E
from protein_shape.support import guarded_use

OUT = paths.RESULTS / 'robustness'


def main():
    rng = np.random.default_rng(E.SEED)
    C = json.load(gzip.open(paths.CLASS_RECALL_CENSUS))
    f = E.load(paths.OCC_MAIN)
    A = (f.OPP >= 1) & (f.SAME == 0)
    R = (f.OPP == 0) & (f.SAME == 0) & f.both.astype(bool)
    f['ninter'] = np.minimum(sum(f[f'f_{r}_crossing'] for r in ('T', 'C2y', 'C2z', 'S2x')), 4)
    f['ninter_nofd'] = np.minimum(sum(f[f'f_{r}_crossing'] for r in ('T', 'C2y', 'S2x')), 4)
    agree = f.used.copy()
    lenient = np.array([int(guarded_use(C['chains']['lenient'].get(c, {}), C['sf_size']) > 0) for c in f.koga_code])
    m = f.fd == 1
    for tag in ('ninter', 'ninter_nofd'):
        print(tag, 'fd1 arm means: OPP', round(f[tag][A & m].mean(), 2), 'ref', round(f[tag][R & m].mean(), 2))
    out = {}
    for cr, y in (('lenient', lenient), ('agree', agree)):
        f['used'] = y
        for tag in ('ninter', 'ninter_nofd'):
            out[f'pooled_{tag}_{cr}'] = E.contrast(f, A, R, ['n', 'stratum', 'fd', tag], rng,
                                                   f'{cr}: pooled + {tag}', verbose=True)
            out[f'fd1_{tag}_{cr}'] = E.contrast(f, A & m, R & m, ['n', 'stratum', tag], rng,
                                                f'{cr}: fd1 + {tag}', verbose=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / 'interleave_count.json', 'w'), indent=1)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__.split('\n')[0]).parse_args()
    main()
