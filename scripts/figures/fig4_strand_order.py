"""Figure 4: renders of a solved interleaved and a solved nested opposite-face pair (cytidine deaminase, PDB 1AF2;
SV40 large T antigen origin-binding domain, PDB 2FUF), and for each arm the fraction of clean solved pairs that
cross in projection and the fraction of topologies used in the strata both arms share (lenient credit).
"""
import json

import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from figstyle import C, DOUBLE, FIGURE_INPUTS, LABEL_FS, OPP_C, PLOT_FS, REF_C, TABLES, finish, panel_label, wilson

ASSETS = FIGURE_INPUTS / 'ribbons'
G = json.loads((FIGURE_INPUTS / 'crossing_geometry.json').read_text())
D = json.loads((FIGURE_INPUTS / 'opp_vs_nested_strata.json').read_text())
LP = json.loads((TABLES / 'lenient_primary.json').read_text())['figure_data']

# ---- c, top: crossing in projection over the clean solved pairs
ROWS = {r['span']: r for r in G['main_table_clean'] if r['facerel'] == 'opp_face' and r['dirrel'] == 'same_dir'}
cross = {arm: (ROWS[arm]['odd'], ROWS[arm]['pairs']) for arm in ('crossing', 'nested')}
assert cross == {'crossing': (29, 29), 'nested': (52, 55)}, cross

# ---- c, bottom: use by any superfamily within the strata the two arms share
SH = [s for s in D['design']['strata'] if s['n_B'] > 0]
use = {'crossing': (sum(s['used_A'] for s in SH), sum(s['n_A'] for s in SH)),
       'nested': (sum(s['used_B'] for s in SH), sum(s['n_B'] for s in SH))}
assert (use['crossing'][1], use['nested'][1]) == (D['design']['in_shared_cells']['A'],
                                                  D['design']['in_shared_cells']['B']) == (561, 90)
assert (use['crossing'][0], use['nested'][0]) == (15, 13), use
assert LP['agree']['fig4c_bottom'] == {'interleaved': list(use['crossing']), 'nested': list(use['nested'])}
# plotted: lenient credit, same topologies
use = {'crossing': tuple(LP['lenient']['fig4c_bottom']['interleaved']),
       'nested': tuple(LP['lenient']['fig4c_bottom']['nested'])}
assert (use['crossing'][1], use['nested'][1]) == (561, 90), use

CARRIERS = [('a', 'interleaved', 'Interleaved pair', 'cytidine deaminase, PDB 1AF2'),
            ('b', 'nested', 'Nested pair', 'SV40 large T antigen origin-binding domain, PDB 2FUF')]
VIEWS = ['faceon', 'edgeon']


def cropped(key, view):
    """Render cropped to its alpha bounding box plus 3 per cent, its crop origin, and angstroms per pixel."""
    img = mpimg.imread(str(ASSETS / f'ribbon_{key}_{view}.png'))
    meta = json.loads((ASSETS / f'ribbon_{key}_positions.json').read_text())
    ys, xs = np.nonzero(img[:, :, 3] > 0)
    m = int(round(0.03 * max(xs.max() - xs.min(), ys.max() - ys.min())))
    x0, y0 = max(xs.min() - m, 0), max(ys.min() - m, 0)
    x1, y1 = min(xs.max() + m + 1, img.shape[1]), min(ys.max() + m + 1, img.shape[0])
    return img[y0:y1, x0:x1], (x0, y0), meta['angstrom_per_px'][view], meta['strand_centre_px'][view]


R = {(k, v): cropped(k, v) for _, k, _, _ in CARRIERS for v in VIEWS}

# ---- layout, in inches
# One scale for all four renders, 0.047 in per angstrom. Cells are one width and each row is as tall
# as its tallest crop.
SCALE = 0.047
W_IN = DOUBLE
CELL_W = 1.02 * SCALE * max(im.shape[1] * apx for im, _, apx, _ in R.values())
ROW_H = [1.02 * SCALE * max(R[key, v][0].shape[0] * R[key, v][2] for v in VIEWS) for _, key, _, _ in CARRIERS]
TOP, ROWGAP = 0.48, 0.30
H_IN = TOP + ROW_H[0] + ROWGAP + ROW_H[1] + 0.06
X0 = [0.22, 0.22 + CELL_W + 0.14]              # left edge of the face-on and edge-on columns
Y0 = [H_IN - TOP - ROW_H[0], H_IN - TOP - ROW_H[0] - ROWGAP - ROW_H[1]]   # bottom edge of rows a and b
for (key, view), (im, _, apx, _) in R.items():    # nothing is clipped: every crop fits its cell at SCALE
    i = [k for _, k, _, _ in CARRIERS].index(key)
    assert im.shape[1] * apx * SCALE <= CELL_W and im.shape[0] * apx * SCALE <= ROW_H[i], (key, view)

fig = plt.figure(figsize=(W_IN, H_IN))


def axes_in(x, y, w, h):
    return fig.add_axes([x / W_IN, y / H_IN, w / W_IN, h / H_IN])


for j, title in enumerate(['Face-on, sheet positions numbered', 'Edge-on']):
    fig.text((X0[j] + CELL_W / 2) / W_IN, (H_IN - 0.06) / H_IN, title, ha='center', va='top', fontsize=7.5)

for i, (letter, key, name, protein) in enumerate(CARRIERS):
    for j, view in enumerate(VIEWS):
        im, (cx, cy), apx, centres = R[key, view]
        ax = axes_in(X0[j], Y0[i], CELL_W, ROW_H[i])
        hw, hh = CELL_W / SCALE / 2, ROW_H[i] / SCALE / 2        # half-cell in angstroms
        w, h = im.shape[1] * apx, im.shape[0] * apx
        ax.imshow(im, extent=(-w / 2, w / 2, -h / 2, h / 2), interpolation='antialiased', zorder=2)
        ax.set_xlim(-hw, hw)
        ax.set_ylim(-hh, hh)
        ax.axis('off')
        if view == 'faceon':
            for p, (px, py) in sorted(centres.items()):
                x = (px - cx) * apx - w / 2
                y = h / 2 - (py - cy) * apx
                ax.text(x, y, p, ha='center', va='center', fontsize=7.5, fontweight='bold', zorder=4,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
            lab = panel_label(ax, letter, x=-0.04, y=1.03)
            ax.text(0.0, 1.03, f'{name}: {protein}', transform=ax.transAxes, ha='left', va='bottom', fontsize=LABEL_FS)

# ---- c: two proportions per arm, each on its own linear axis
CX, CW, CH = 4.78, 2.02, 1.0
BARS = [('crossing', 'interleaved', OPP_C, 1.0), ('nested', 'nested', REF_C, 0.0)]
PANELS = [(cross, 'Cross in projection', 'clean solved pairs', 'Fraction of pairs', 1.0,
           [0, 0.5, 1.0], ['0', '0.5', '1']),
          (use, 'Used by any superfamily', 'topologies in shared strata', 'Fraction of topologies', 0.30,
           [0, 0.1, 0.2, 0.3], ['0', '0.1', '0.2', '0.3'])]
for k, (data, ttl, sub, xlab, xmax, ticks, ticklabels) in enumerate(PANELS):
    ax = axes_in(CX, Y0[k] + (ROW_H[k] - CH) / 2 - 0.05, CW, CH)
    for arm, label, col, y in BARS:
        n_k, n = data[arm]
        f = n_k / n
        lo, hi = wilson(n_k, n)
        ax.barh(y, f, height=0.56, color=col, edgecolor='none', zorder=2)
        ax.plot([lo, hi], [y, y], color='black', lw=0.8, zorder=3, solid_capstyle='butt')
        for e in (lo, hi):
            ax.plot([e, e], [y - 0.13, y + 0.13], color='black', lw=0.8, zorder=3)
        ax.text(max(hi, f) + 0.04 * xmax, y, f'{n_k}/{n}', ha='left', va='center', fontsize=LABEL_FS)
    ax.set_xlim(0, xmax * 1.32)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.spines['bottom'].set_bounds(0, xmax)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels(['interleaved', 'nested'])
    for t, (_, _, col, _) in zip(ax.get_yticklabels(), BARS):
        t.set_color(col)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(labelsize=PLOT_FS)
    ax.set_xlabel(xlab, labelpad=2, fontsize=PLOT_FS)
    ax.set_title(f'{ttl}\n{sub}', fontsize=PLOT_FS, pad=4)
    if k == 0:
        panel_label(ax, 'c', x=-0.70 / CW, y=1.0).set_transform(ax.title.get_transform())

finish(fig, 'fig4_strand_order')
