"""Figure 1: reading a topology string, the four relations of two parallel crossovers, interleaved and nested
pairs, and a solved sheet that builds the OPP pair. Every cartoon is drawn from its topology string.
"""
from matplotlib.patches import Circle, Ellipse, FancyArrow, FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

import topodraw as TD
from figstyle import C, DOUBLE, FIGURE_INPUTS, REL_COLOR, finish

S = 0.27                     # inches per data unit (same in x and y)
PANEL_D_H = 5.5              # data units of new space BELOW y = 0 for panel d
GAP = 0.9                    # data units between row c and panel d, split by a rule
Y_RULE = -0.25 - GAP / 2
W_IN, H_IN = DOUBLE, 5.15 + (PANEL_D_H + GAP) * S
W, H = W_IN / S, 5.15 / S    # H still measures the schematic rows, so panels a-c do not move

fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(-PANEL_D_H - GAP, H)   # negative y is panel d; a to c keep their coordinates
ax.axis('off')

DY = 3.3                     # face-on centre to edge-on centre
TOP_F, BOT_F = 14.75, 5.1    # face-on centres of the two rows
SUB = 1.95                   # subtitle height above a face-on centre
ROWH = 0.5                   # text line spacing (data units)


def pair_colours(topo, rel, span=None):
    """Colour the crossovers of every pair with relation `rel` (and span) by the relation colour."""
    hl = {}
    for a, b, r, s in TD.pairs(topo):
        if r == rel and (span is None or s == span):
            hl[a] = hl[b] = REL_COLOR[rel]
    assert hl, (topo, rel, span)
    return hl


def centre(topo, x0):
    return x0 + (len(TD.sheet(topo)[0]) - 1) / 2


def cartoon(topo, x0, yF, hl=None, termini=False):
    e1 = TD.draw_face(ax, topo, ox=x0, oy=yF, hl=hl, termini=termini)
    e2 = TD.draw_edge(ax, topo, ox=x0, oy=yF - DY, hl=hl)
    TD.keepout(ax, e1)
    TD.keepout(ax, e2)
    return e1, e2


def lens(x, y, w=0.30, h=0.17, z=9):
    ax.add_patch(Ellipse((x, y), w, h, fc='black', ec='white', lw=1.0, zorder=z))


def helix_xy(topo, idx, x0, yE, height=0.8):
    for c in TD.crossovers(topo):
        if c['idx'] == idx:
            return (x0 + (c['pos1'] + c['pos2']) / 2 - 1, yE + c['face'] * height)


def heading(x, y, letter, text):
    ax.text(x, y, letter, fontsize=8, fontweight='bold', va='center')
    ax.text(x + 0.6, y, text, fontsize=7.5, fontweight='bold', va='center')


# ---------------- a: reading a topology string ----------------
heading(0.15, 18.5, 'a', 'Topology string')
FLAV = '21345+++++'
xa = 1.75
ax.text(centre(FLAV, xa), TOP_F + SUB + 0.1, TD.pretty(FLAV), fontsize=8.5, ha='center', va='center')
e1, e2 = cartoon(FLAV, xa, TOP_F, termini=True)
ax.text(xa - 0.95, TOP_F, 'face-on', fontsize=7, rotation=90, ha='center', va='center')
ax.text(xa - 0.95, TOP_F - DY, 'edge-on', fontsize=7, rotation=90, ha='center', va='center')
xr = xa + 4.7
for sgn, lab in ((1, 'face +'), (-1, 'face −')):
    ax.add_patch(FancyArrowPatch((xr, TOP_F - DY + sgn * 0.25), (xr, TOP_F - DY + sgn * 1.05),
                                 arrowstyle='-|>', mutation_scale=5, color='black', lw=1.0))
    ax.text(xr + 0.2, TOP_F - DY + sgn * 0.75, lab, fontsize=7, va='center', ha='left')
ya = TOP_F - DY - 1.5
ax.text(centre(FLAV, xa), ya, 'flavodoxin-like sheet', fontsize=7, ha='center', va='center')
ax.text(centre(FLAV, xa), ya - ROWH, 'digits: strands in sheet order', fontsize=7, ha='center', va='center',
        color=C['grey'])
ax.text(centre(FLAV, xa), ya - 2 * ROWH, 'signs: strand directions', fontsize=7, ha='center', va='center',
        color=C['grey'])

# ---------------- b: four relations ----------------
heading(8.6, 18.5, 'b', 'Relations of two parallel crossovers')
REL = [  # relation, example topology, symmetry, lateral direction, face, strands
    ('T', '1234+++-', 'translation', 'same', 'same', 'parallel'),
    ('C2y', '2134+++-', 'twofold along strands', 'opposite', 'opposite', 'parallel'),
    ('C2z', '2134++--', 'twofold normal to sheet', 'opposite', 'same', 'antiparallel'),
    ('S2x', '1234++--', 'screw axis across strands', 'same', 'opposite', 'antiparallel'),
]
LAB = {'C2y': 'C2y', 'C2z': 'C2z', 'S2x': 'S2x', 'T': 'T'}
xb0, step = 9.6, 4.05
for k, (rel, topo, sym, d, f, st) in enumerate(REL):
    x0 = xb0 + k * step
    xc = centre(topo, x0)
    col = REL_COLOR[rel]
    ax.text(xc, TOP_F + SUB + 0.55, LAB[rel], fontsize=7.5, fontweight='bold', color=col, ha='center',
            va='center')
    ax.text(xc, TOP_F + SUB, sym, fontsize=7, ha='center', va='center')
    hl = pair_colours(topo, rel)
    e1, e2 = cartoon(topo, x0, TOP_F, hl=hl)
    yE = TOP_F - DY
    ia, ib = sorted(hl)
    (hx1, hy1), (hx2, hy2) = helix_xy(topo, ia, x0, yE), helix_xy(topo, ib, x0, yE)
    mx, my = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    if rel == 'T':
        ax.add_patch(FancyArrowPatch((hx1 + 0.05, hy1 + 0.45), (hx2 - 0.05, hy2 + 0.45), arrowstyle='-|>',
                                     mutation_scale=5, color='black', lw=1.0, zorder=8))
    elif rel == 'C2y':
        lens(mx, my)
    elif rel == 'C2z':
        ax.plot([mx, mx], [yE - 1.2, yE + 1.2], color='black', lw=1.0, ls=(0, (4, 1.5, 1, 1.5)), zorder=8)
        lens(mx, yE + 1.2, w=0.17, h=0.30)
    else:
        ax.plot([x0 - 0.45, x0 + 3.35], [yE, yE], color='black', lw=1.0, ls=(0, (4, 1.5, 1, 1.5)), zorder=8)
        ax.add_patch(FancyArrowPatch((x0 + 3.2, yE), (x0 + 3.6, yE), arrowstyle='-|>', mutation_scale=5,
                                     color='black', lw=1.0, zorder=8))
    yt = TOP_F - DY - 1.5
    ax.text(xc, yt, f'direction {d}', fontsize=7, ha='center', va='center')
    ax.text(xc, yt - ROWH, f'face {f}', fontsize=7, ha='center', va='center')
    ax.text(xc, yt - 2 * ROWH, f'strands {st}', fontsize=7, ha='center', va='center')

# ---------------- c: how the strand intervals relate ----------------
heading(0.15, 8.45, 'c', 'Interleaved pairs, and the nested opposite-face comparator')
CORES = [  # title, subtitle, topology, relation, span, x0
    ('SAME pair', 'T, interleaved', '13425+-+++', 'T', 'crossing', 1.3),
    ('OPP pair (NF1 core)', 'S2x, interleaved', '1324+-+-', 'S2x', 'crossing', 7.1),
    ('OPP pair, nested', 'S2x, nested', '1342+--+', 'S2x', 'nested', 11.9),
    ('ferredoxin core', 'C2z, interleaved', '4132+-+-', 'C2z', 'crossing', 17.2),
]
for title, sub, topo, rel, span, x0 in CORES:
    xc = centre(topo, x0)
    col = REL_COLOR[rel]
    ax.text(xc, BOT_F + SUB + 0.55, title, fontsize=7.5, fontweight='bold', color=col, ha='center', va='center')
    ax.text(xc, BOT_F + SUB, sub, fontsize=7, ha='center', va='center')
    hl = pair_colours(topo, rel, span)
    e1, e2 = cartoon(topo, x0, BOT_F, hl=hl)
    ax.text(xc, BOT_F - DY - 1.5, TD.pretty(topo), fontsize=8.5, ha='center', va='center')

# --- d: a solved structure that builds the arrangement ----------------------------------------
# ECOD e3hmrA1 (PDB 3hmr A:43-122), 1|5|3|4|2:+-+-+, seen edge-on so the two connections of the
# S2x pair lie on opposite sides of the sheet (render: ribbons/ribbon_carrier.py).
import matplotlib.image as mpimg  # noqa: E402

_img = mpimg.imread(str(FIGURE_INPUTS / 'ribbons' / 'ribbon_3hmr_edgeon.png'))
_ih, _iw = _img.shape[0], _img.shape[1]
_dh = PANEL_D_H - 1.15
_dw = _dh * _iw / _ih
_x0, _y0 = 1.25, -PANEL_D_H - GAP + 0.35
ax.plot([0.15, W - 0.15], [Y_RULE, Y_RULE], color=C['lgrey'], lw=1.0, solid_capstyle='butt')
ax.imshow(_img, extent=(_x0, _x0 + _dw, _y0, _y0 + _dh), zorder=2, interpolation='antialiased')

ax.text(0.35, -0.55 - GAP, 'd', fontsize=9, fontweight='bold', va='top', ha='left')
ax.text(_x0 + _dw + 0.9, _y0 + _dh - 0.15,
        'A sheet that builds it', fontsize=7.5, va='top', ha='left', fontweight='bold')
_lines = [
    ('3hmr, residues 43 to 122; five strands, order 1|5|3|4|2', C['black']),
    ('the pair, on opposite faces of the sheet:', C['black']),
    ('    residues 65 to 86, sheet position 5 to 3', C['vermillion']),
    ('    residues 101 to 109, position 4 to 2', C['blue']),
    ('seen edge-on, so one passes above the sheet', C['black']),
    ('and the other below it', C['black']),
]
for _j, (_t, _c) in enumerate(_lines):
    ax.text(_x0 + _dw + 0.9, _y0 + _dh - 1.00 - _j * 0.56, _t, fontsize=7, va='top', ha='left',
            color=_c)

# key (in panel d's row, to the right of the structure and its notes)
kx, ky = 17.6, -0.95 - GAP
ax.text(kx, ky, 'Key', fontsize=7.5, fontweight='bold', va='center')
items = [
    ('arrow', 'strand; number = order in chain'),
    ('front', 'crossover helix on face +'),
    ('back', 'crossover helix on face −'),
    ('arc', 'antiparallel connection'),
    ('away', 'strand end-on, + (away from viewer)'),
    ('toward', 'strand end-on, − (toward viewer)'),
    ('disc', 'helix end-on; arrow = lateral direction'),
]
for j, (kind, text) in enumerate(items):
    y = ky - 0.85 - j * 0.58
    gx = kx + 0.3
    if kind == 'arrow':
        ax.add_patch(FancyArrow(gx - 0.32, y, 0.64, 0, width=0.22, head_width=0.36, head_length=0.22,
                                length_includes_head=True, fc=TD.STRAND_FILL, ec=TD.STRAND_EDGE, lw=1.0))
    elif kind in ('front', 'back'):
        ax.add_patch(FancyBboxPatch((gx - 0.36, y - 0.1), 0.72, 0.2, boxstyle='round,pad=0,rounding_size=0.1',
                                    fc=C['grey'], ec=C['grey'], lw=1.0, alpha=1 if kind == 'front' else 0.35,
                                    ls='-' if kind == 'front' else (0, (2.2, 1.4))))
    elif kind == 'arc':
        TD._bez(ax, (gx - 0.35, y - 0.12), (gx - 0.35, y + 0.22), (gx + 0.35, y + 0.22), (gx + 0.35, y - 0.12),
                color=TD.LOOP, lw=1.0)
    elif kind in ('away', 'toward'):
        TD._strand_symbol(ax, gx, y, 1 if kind == 'away' else -1)
    else:
        ax.add_patch(Circle((gx, y), 0.2, fc=C['grey'], ec='none'))
    ax.text(kx + 0.95, y, text, fontsize=7, va='center', ha='left')


def check_rule_sides():
    """Render-time check: everything of rows a to c is above the rule, everything of panel d below it."""
    fig.set_dpi(300)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    yr = ax.transData.transform((0, Y_RULE))[1]
    errs = []
    for art in ax.texts + ax.images:
        bb = art.get_window_extent(r)
        above = (art.get_position()[1] if hasattr(art, 'get_position') else art.get_extent()[2]) > Y_RULE
        if (above and bb.y0 <= yr) or (not above and bb.y1 >= yr):
            errs.append(getattr(art, 'get_text', lambda: 'panel d image')())
    assert not errs, f'fig1_concept: items cross the panel d rule: {errs}'


check_rule_sides()
finish(fig, 'fig1_concept')
