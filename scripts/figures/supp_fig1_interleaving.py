"""Supplementary Figure 1: observed over expected occupancy of every interleaved connection-pair cell, against
composition alone and with the published rules added, and the measured size of each cell test.
"""
import json
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter

from figstyle import C, DOUBLE, FIGURE_INPUTS, OPP_C, REF_C, finish, panel_label

D = json.loads((FIGURE_INPUTS / 'interleaving_general.json').read_text())

# ---- the eight interleaved cells with any enumerated opportunity, in panel order
ORDER = ['PP|interleaved|same_dir|face_same', 'PP|interleaved|same_dir|face_opp',
         'PP|interleaved|dir_opp|face_same', 'PP|interleaved|dir_opp|face_opp',
         'JP|interleaved|same_dir|face_free', 'JP|interleaved|dir_opp|face_free',
         'JJ|interleaved|same_dir|face_free', 'JJ|interleaved|dir_opp|face_free']
LIVE = [c for c in D['cells'] if c['relation'] == 'interleaved' and c['opportunity'] > 0]
assert [c['cell'] for c in LIVE] == ORDER, [c['cell'] for c in LIVE]
assert all(c['opportunity'] == c['opportunity_route2'] for c in LIVE)
assert D['enum']['n_topologies'] == 173760 and D['enum']['routes_agree']
MRR = D['minami_rule_relations']
assert all(v == 173760 for v in MRR['reproduces_published_counts'].values())
assert MRR['ov_pp_equals_PP_nested_strict_face_same'] == 173760

POP_ALL = 'all 5-7-strand topologies'
POP_AB = 'alpha/beta (at least one parallel crossover)'
COMP, RULES = 'composition', 'composition+published rules'


def read(features, population, cell):
    return D['readings'][f'{features} | {population} | per-cell | {cell}']


def size(cell, null):
    return D['size'][f'{cell} | all | {null}']


a_comp = [read(COMP, POP_ALL, c)['ratio'] for c in ORDER]
a_rule = [read(RULES, POP_ALL, c)['ratio'] for c in ORDER]
b_comp = [read(COMP, POP_AB, c)['ratio'] for c in ORDER]
b_rule = [read(RULES, POP_AB, c)['ratio'] for c in ORDER]
band = {n: [(size(c, n)['null_binary_q'][0], size(c, n)['null_binary_q'][2]) for c in ORDER]
        for n in ('free', 'stratified')}
c_size = {n: [size(c, n)['size_exact_pb'] for c in ORDER] for n in ('free', 'stratified')}

# The null band and the sizes are the reading with the published rules in the model, not composition alone.
for i, c in enumerate(ORDER):
    for n in ('free', 'stratified'):
        assert round(size(c, n)['observed_binary'], 3) == a_rule[i], (c, n)
        assert size(c, n)['n_null'] == 400

# Measured size of each cell's reading rule and whether it holds its nominal size, asserted against the
# input file below.
TXT = {'PP|interleaved|same_dir|face_same': (0.050, 0.292, False),
       'PP|interleaved|same_dir|face_opp': (0.062, 0.025, True),
       'PP|interleaved|dir_opp|face_same': (0.072, 0.025, True),
       'PP|interleaved|dir_opp|face_opp': (0.052, 0.352, False),
       'JP|interleaved|same_dir|face_free': (0.128, 0.107, False),
       'JP|interleaved|dir_opp|face_free': (0.155, 0.150, False),
       'JJ|interleaved|same_dir|face_free': (0.070, 0.050, True),
       'JJ|interleaved|dir_opp|face_free': (0.062, 0.065, True)}
for c, (sf, ss, _) in TXT.items():
    assert round(size(c, 'free')['size_exact_pb'], 3) == sf, c
    assert round(size(c, 'stratified')['size_exact_pb'], 3) == ss, c
AT_SIZE = [TXT[c][2] for c in ORDER]
assert sum(AT_SIZE) == 4

X = [0.0, 1.0, 2.0, 3.0, 4.5, 5.5, 7.0, 8.0]
XLIM = (-0.6, 8.6)
DX, MS, HALF = 0.15, 3.8, 0.42
COL = [OPP_C if c == 'PP|interleaved|same_dir|face_opp' else REF_C for c in ORDER]
TICK = ['same dir.\nsame face', 'same dir.\nopp. face', 'opp. dir.\nsame face', 'opp. dir.\nopp. face',
        'same dir.', 'opp. dir.', 'same dir.', 'opp. dir.']
GROUPS = [(0, 3, 'two parallel crossovers',
           f"published-rule firings: {MRR['by_relation']['ov_pp']['nested_strict']:,} nested, "
           f"{MRR['by_relation']['ov_pp']['interleaved']:,} interleaved"),
          (4, 5, 'crossover + antiparallel', f"{MRR['by_relation']['ov_pa']['interleaved']:,} interleaved"),
          (6, 7, 'two antiparallel', f"{MRR['by_relation']['ov_aa']['interleaved']:,} interleaved")]
BAND_F, BAND_S = '#EAEAEA', '#9A9A9A'

fig = plt.figure(figsize=(DOUBLE, 4.3))
gs = fig.add_gridspec(3, 1, left=0.078, right=0.988, bottom=0.165, top=0.915, height_ratios=[1, 1, 0.62],
                      hspace=0.42)
A, B, Cc = (fig.add_subplot(gs[i, 0]) for i in range(3))

# ---- a, b: observed / expected under the two models
for ax, comp, rule, ttl in ((A, a_comp, a_rule, 'Interleaved cells: all 5–7-strand topologies'),
                            (B, b_comp, b_rule, 'Sheets with at least one parallel crossover')):
    if ax is A:
        for (lo, hi), x in zip(band['free'], X):
            ax.add_patch(Rectangle((x - HALF, lo), 2 * HALF, hi - lo, facecolor=BAND_F, edgecolor='none',
                                   zorder=0, gid='keepout'))
        for (lo, hi), x in zip(band['stratified'], X):
            ax.add_patch(Rectangle((x - HALF, lo), 2 * HALF, hi - lo, facecolor='none', edgecolor=BAND_S,
                                   lw=1.0, zorder=1, gid='keepout'))
    ax.axhline(1.0, color=C['grey'], lw=1.0, ls=(0, (3, 2)), zorder=1)
    for x, y, col, at in zip(X, comp, COL, AT_SIZE):
        ax.plot([x - DX], [y], marker='s', ms=MS, color=col, mfc=col if at else 'white', ls='none', zorder=3)
    for x, y, col, at in zip(X, rule, COL, AT_SIZE):
        ax.plot([x + DX], [y], marker='o', ms=MS, color=col, mfc=col if at else 'white', ls='none', zorder=3)
    ax.set_yscale('log')
    ax.set_ylim(0.30, 3.8)
    ax.set_yticks([0.5, 1.0, 2.0])
    ax.set_yticklabels(['0.5', '1', '2'])
    ax.yaxis.set_minor_locator(FixedLocator([0.4, 0.6, 0.7, 0.8, 0.9, 1.5, 3.0]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel('Observed / expected')
    ax.set_title(ttl, pad=13)        # both a and b now carry a legend above the axes

A.legend(handles=[Line2D([], [], marker='s', ms=MS, color='black', ls='none', label='composition only'),
                  Line2D([], [], marker='o', ms=MS, color='black', ls='none', label='with published rules'),
                  Patch(facecolor=BAND_F, edgecolor='none', label='permutation null, free (5–95%)'),
                  Patch(facecolor='none', edgecolor=BAND_S, lw=1.0, label='stratified (5–95%)')],
         loc='lower left', bbox_to_anchor=(-0.005, 0.99), ncol=4, frameon=False, fontsize=7, handlelength=1.4,
         columnspacing=1.0, handletextpad=0.4, borderaxespad=0)
B.legend(handles=[Line2D([], [], marker='o', ms=MS, color=C['grey'], ls='none',
                         label='at nominal size under both nulls'),
                  Line2D([], [], marker='o', ms=MS, color=C['grey'], mfc='white', ls='none',
                         label='not at nominal size')],
         loc='lower left', bbox_to_anchor=(-0.005, 0.99), ncol=2, frameon=False, fontsize=7,
         handlelength=1.0, columnspacing=1.0, handletextpad=0.4, borderaxespad=0)

# ---- c: measured size of each reading rule
Cc.axhline(0.05, color=C['grey'], lw=1.0, ls=(0, (3, 2)), zorder=1)
for marker, null, dx in (('o', 'free', -DX), ('D', 'stratified', DX)):
    for x, y, col in zip(X, c_size[null], COL):
        Cc.plot([x + dx], [y], marker=marker, ms=MS, color=col, ls='none', zorder=3)
Cc.set_ylim(0, 0.385)
Cc.set_yticks([0, 0.1, 0.2, 0.3])
Cc.set_yticklabels(['0', '0.1', '0.2', '0.3'])
Cc.set_ylabel('Measured size', labelpad=1)   # tighter, so it clears the panel letter above it
Cc.set_title('Measured size of each reading rule (dashed: nominal 0.05)', pad=10)
Cc.legend(handles=[Line2D([], [], marker='o', ms=MS, color=C['grey'], ls='none', label='free null'),
                   Line2D([], [], marker='D', ms=MS, color=C['grey'], ls='none', label='stratified null')],
          loc='upper right', frameon=True, framealpha=1.0, edgecolor='#CCCCCC', fancybox=False, fontsize=7,
          handlelength=1.0, borderpad=0.35, borderaxespad=0.3)

for ax in (A, B, Cc):
    ax.set_xlim(*XLIM)
    ax.set_xticks(X)
    ax.tick_params(axis='x', length=0)
A.tick_params(labelbottom=False)
B.tick_params(labelbottom=False)
Cc.set_xticklabels(TICK, fontsize=7)

# ---- shared x furniture: one bracket per connection class, under the cell labels
PH = Cc.get_position().height * fig.get_figheight()      # panel-c height in inches
xa = Cc.get_xaxis_transform()
for i, j, name, firings in GROUPS:
    Cc.plot([X[i] - HALF, X[j] + HALF], [-0.255 / PH] * 2, transform=xa, color='black', lw=1.0, clip_on=False)
    xc = (X[i] + X[j]) / 2
    Cc.text(xc, -0.285 / PH, name, transform=xa, ha='center', va='top', fontsize=7.5, clip_on=False)
    Cc.text(xc, -0.425 / PH, firings, transform=xa, ha='center', va='top', fontsize=7,
            color='#404040', clip_on=False)

for ax, s in ((A, 'a'), (B, 'b'), (Cc, 'c')):
    lab = panel_label(ax, s, x=-0.055, y=1.05)
    if ax is A:            # panel a's title sits above its legend
        lab.set_y(1.0)
        lab.set_transform(ax.title.get_transform())
        lab.set_va('baseline')


def check_marker_gaps(figure, panels):
    """No two plotted markers of one panel may be closer than the sum of their radii (a hidden reading)."""
    figure.set_dpi(300)
    figure.canvas.draw()
    rad = MS * figure.dpi / 72.0 / 2.0
    for ax, pts in panels:
        xy = [ax.transData.transform((x, y)) for x, y in pts]
        for (p, q) in combinations(xy, 2):
            if ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 < 2 * rad:
                raise AssertionError(f'markers overlap in {ax.get_title(loc="left")!r}: {p} {q}')


check_marker_gaps(fig, [
    (A, [(x - DX, y) for x, y in zip(X, a_comp)] + [(x + DX, y) for x, y in zip(X, a_rule)]),
    (B, [(x - DX, y) for x, y in zip(X, b_comp)] + [(x + DX, y) for x, y in zip(X, b_rule)]),
    (Cc, [(x - DX, y) for x, y in zip(X, c_size['free'])] + [(x + DX, y) for x, y in zip(X, c_size['stratified'])]),
])

finish(fig, 'supp_fig1_interleaving')
print('plotted', dict(cells=ORDER, a_comp=a_comp, a_rule=a_rule, b_comp=b_comp, b_rule=b_rule,
                      band_free=band['free'], band_strat=band['stratified'], size=c_size, at_size=AT_SIZE))
