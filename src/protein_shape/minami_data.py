"""Readers for the Minami et al. (2023) Source Data workbook and the Koga-laboratory NF design labels.
These external files are not redistributed; see data/manifest.json.
"""
import re

import openpyxl
import pandas as pd

from . import topology as T
from .paths import DATA

EXT = DATA / 'external'
KOGA = EXT / 'koga' / 'demos_src' / 'demos_explore_novel_abfolds'
THRESH = 0.25  # topologies with observation frequency below 1/4 count as unobserved


def load_sheets(max_n=6):
    """Return ({n: summary frame for 3..max_n strands (default 6)}, per-sheet domain frame).

    Summary columns: freq, cls, jp, ov, en, topology, n. Domain frame: one row per
    independent open sheet (domain, n, topology, independent, w, fid, n_dom_F, n_F_in_H, arch).
    CAUTION: the two count columns are named the wrong way round. Checked on the data: `n_dom_F` is
    constant per H-group (X.H) = number of F-groups in the H-group; `n_F_in_H` is constant per F-group
    = number of domains in the F-group; w = 1 / n_dom_F / n_F_in_H (symmetric, so w is unaffected).
    """
    wb = openpyxl.load_workbook(EXT / 'minami' / 'fig3.xlsx', read_only=True)
    summ = {}
    for n in range(3, max_n + 1):
        rows = list(wb[f'{n}beta.summary'].iter_rows(values_only=True))
        df = pd.DataFrame(rows[5:], columns=rows[4])
        df = df.rename(columns={'frequency': 'freq', 'class': 'cls', 'pattern': 'topology'})
        df['n'] = n
        summ[n] = df
    rows = list(wb['ECOD99_lt8beta_independent_cura'].iter_rows(values_only=True))
    dom = pd.DataFrame(rows, columns=['domain', 'n', 'topology', 'independent', 'w', 'fid',
                                      'n_dom_F', 'n_F_in_H', 'arch'])
    return summ, dom


def nf_labels():
    """{'NF1': '1|3|2|4:+-+-', ...} from the Koga demo3 output (design file names NFk-*)."""
    nf = {}
    for line in open(KOGA / 'Demos' / 'demo3' / 'expected_result2.dat'):
        m = re.match(r'\s*4\s+(\d{4}[+-]{4})\s+independent=\S+\s+n_open_sheet=\S+\s*:\s*(NF\d)-', line)
        if m:
            nf[m.group(2)] = T.minami(*T.parse(m.group(1)))
    return nf
