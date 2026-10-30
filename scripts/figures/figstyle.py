"""Shared style, render-time layout guard and deterministic saving for the figures.

Figures are drawn at print width; finish() checks text sizes, overlaps and panel boundaries, saves PDF and PNG
with fixed metadata, and audits the saved PDF for rendered type size and font (typeaudit.py).
"""
import hashlib
import math
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.text import Text  # noqa: E402

from protein_shape.paths import FIGURE_INPUTS, FIGURES, TABLES  # noqa: E402,F401

OUT = FIGURES
MIN_PT = 7.0          # authored floor
# Figures are authored at 7.0 in and placed at 6.27 in, so the type as read is PLACED_SCALE of
# its authored size; MIN_PT_PLACED is the floor as read.
SINGLE, DOUBLE = 3.4, 7.0
PLACED_SCALE = 6.27 / DOUBLE
MIN_PT_PLACED = 6.0
# One axes box for the data plots of Figs 2 and 3, so both are read at the same plot and type size.
BOX_W, BOX_H = 2.40, 1.30
PLOT_FS, LABEL_FS = 8, 7.5            # ticks, axis labels, titles, legend; point labels and counts


def boxed_axes(fig, w_in, h_in, x_in, y_in):
    """Axes of exactly BOX_W x BOX_H inches with its lower-left corner at (x_in, y_in) inches."""
    ax = fig.add_axes([x_in / w_in, y_in / h_in, BOX_W / w_in, BOX_H / h_in])
    ax.tick_params(labelsize=PLOT_FS)
    return ax


def assert_box(fig, axs):
    """Render-time check: every data axes is BOX_W x BOX_H inches."""
    for ax in axs:
        b = ax.get_position()
        w, h = b.width * fig.get_figwidth(), b.height * fig.get_figheight()
        assert abs(w - BOX_W) < 1e-6 and abs(h - BOX_H) < 1e-6, (w, h)

# Okabe-Ito colour-blind-safe palette
C = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442',
         blue='#0072B2', vermillion='#D55E00', purple='#CC79A7', grey='#7F7F7F', lgrey='#BDBDBD')
REL_COLOR = {'T': C['orange'], 'C2y': C['purple'], 'C2z': C['green'], 'S2x': C['vermillion']}
OPP_C, REF_C = C['vermillion'], C['blue']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 1.0,
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'lines.linewidth': 1.0,
    'lines.markersize': 4, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'protein-shape',
    'legend.frameon': False, 'axes.titlelocation': 'left', 'figure.dpi': 100,
    'savefig.dpi': 600, 'mathtext.default': 'regular',
})


def panel_label(ax, s, x=-0.02, y=1.0, fig=None):
    """Bold lower-case panel letter at the top-left of the axes box (axes coords)."""
    return ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='right')


def frac_label(k, n):
    return f'{k}/{n}'


def parse_frac(s):
    k, n = s.split('/')
    return int(k), int(n)


def _texts(fig):
    hidden = set()
    for ax in fig.axes:   # tick texts of axes drawn with axis('off') are never rendered
        if not ax.axison:
            hidden |= {id(t) for t in ax.xaxis.findobj(Text) + ax.yaxis.findobj(Text)}
    out = []
    for t in fig.findobj(Text):
        if id(t) in hidden:
            continue
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        out.append(t)
    return out


def _owner(t, fig):
    if t.axes is not None:
        return t.axes
    for ax in fig.axes:  # tick labels and axis labels carry .axes; legend texts too
        if t in ax.findobj(Text):
            return ax
    return None


def guard(fig, name, dpi=300):
    fig.set_dpi(dpi)        # measure at print-like resolution: low-dpi hinting underestimates text widths
    tol = dpi / 100.0       # 1 px at 100 dpi
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw, fh = fig.bbox.width, fig.bbox.height
    texts = _texts(fig)
    errs = []
    boxes = []
    for t in texts:
        if t.get_fontsize() < MIN_PT - 1e-9:
            errs.append(f'font {t.get_fontsize():.2f} pt < {MIN_PT}: {t.get_text()!r}')
        bb = t.get_window_extent(r)
        if bb.x0 < tol or bb.y0 < tol or bb.x1 > fw - tol or bb.y1 > fh - tol:
            errs.append(f'text outside figure: {t.get_text()!r} {bb.bounds}')
        boxes.append((t, bb, _owner(t, fig)))
    # text-text overlap
    for i in range(len(boxes)):
        ti, bi, _ = boxes[i]
        if ti.get_gid() == 'nocheck':
            continue
        for j in range(i + 1, len(boxes)):
            tj, bj, _ = boxes[j]
            if tj.get_gid() == 'nocheck':
                continue
            if bi.overlaps(bj):
                ix = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                iy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                if ix > 0.5 * tol and iy > 0.5 * tol:
                    errs.append(f'text overlap: {ti.get_text()!r} x {tj.get_text()!r}')
    # furniture of one panel inside another panel's box
    axes = [a for a in fig.axes if a.get_visible()]
    for t, bb, own in boxes:
        for a in axes:
            if a is own or own is None:
                continue
            ab = a.get_window_extent(r)
            ix = min(bb.x1, ab.x1) - max(bb.x0, ab.x0)
            iy = min(bb.y1, ab.y1) - max(bb.y0, ab.y0)
            if ix > 0.5 * tol and iy > 0.5 * tol:
                errs.append(f'text {t.get_text()!r} enters another panel box')
    # checked texts must stay out of cartoon keep-out boxes (gid 'keepout') in their own axes
    for t, bb, own in boxes:
        if t.get_gid() == 'nocheck' or own is None:
            continue
        for p in own.patches:
            if p.get_gid() != 'keepout':
                continue
            pb = p.get_window_extent(r)
            ix = min(bb.x1, pb.x1) - max(bb.x0, pb.x0)
            iy = min(bb.y1, pb.y1) - max(bb.y0, pb.y0)
            if ix > 0.5 * tol and iy > 0.5 * tol:
                errs.append(f'text {t.get_text()!r} enters a cartoon')
    # point annotations must not cover data markers of their own axes
    for t, bb, own in boxes:
        if t.get_gid() != 'label' or own is None:
            continue
        for ln in own.get_lines():
            if ln.get_marker() in (None, 'None', '', ' ') or not ln.get_visible():
                continue
            xy = ln.get_transform().transform(ln.get_xydata())
            rad = ln.get_markersize() * fig.dpi / 72.0 / 2.0
            for (px, py) in xy:
                if (bb.x0 - rad + 0.5 < px < bb.x1 + rad - 0.5) and (bb.y0 - rad + 0.5 < py < bb.y1 + rad - 0.5):
                    errs.append(f'label {t.get_text()!r} covers a marker')
                    break
    # an in-axes legend must not cover plotted lines or markers of its axes
    for a in fig.axes:
        lg = a.get_legend()
        if lg is None or not a.get_visible():
            continue
        lb = lg.get_window_extent(r)
        segs, marks = _obstacles(a)
        if any(_seg_hits_box(p, q, lb) for p, q in segs) or \
                any(lb.x0 - rd < q[0] < lb.x1 + rd and lb.y0 - rd < q[1] < lb.y1 + rd for q, rd in marks):
            errs.append(f'legend covers data in panel {a.get_title(loc="left")!r}')
    # point annotations must not cross a plotted line of their own axes
    for t, bb, own in boxes:
        if t.get_gid() != 'label' or own is None:
            continue
        segs, _ = _obstacles(own)
        if any(_seg_hits_box(p, q, bb) for p, q in segs):
            errs.append(f'label {t.get_text()!r} crosses a line')
    if errs:
        raise AssertionError(f'{name}: render-time guard failed:\n  ' + '\n  '.join(dict.fromkeys(errs)))
    return dict(n_texts=len(texts), min_pt=min(t.get_fontsize() for t in texts),
                words=sum(len(t.get_text().split()) for t in texts),
                width_in=fig.get_figwidth(), height_in=fig.get_figheight())


def type_audit(pdf):
    """Post-save audit of a saved PDF: rendered span sizes >= MIN_PT and Arial only. Raises AssertionError."""
    import typeaudit
    n, mn, fonts, errs = typeaudit.audit(pdf)
    print(f'type audit {"FAIL" if errs else "PASS"} {pdf.name}: {n} spans, min rendered {mn:.2f} pt, fonts {fonts}')
    if errs:
        raise AssertionError(f'{pdf.name}: type audit failed:\n  ' + '\n  '.join(errs))


def finish(fig, name):
    """Guard, save PDF and PNG deterministically, then audit the saved PDF. Returns guard stats."""
    try:
        stats = guard(fig, name)
    except AssertionError:
        if os.environ.get('FIG_DRAFT'):
            fig.savefig(Path(os.environ['FIG_DRAFT']) / f'{name}_DRAFT.png', dpi=200)
        raise
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None, 'Creator': None,
                                                'Producer': None})
    fig.savefig(OUT / f'{name}.png', dpi=600, metadata={'Software': None})
    plt.close(fig)
    type_audit(OUT / f'{name}.pdf')
    stats['words_per_sq_in'] = round(stats['words'] / (stats['width_in'] * stats['height_in']), 1)
    print(name, stats)
    return stats


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


os.environ.setdefault('SOURCE_DATE_EPOCH', '0')
warnings.filterwarnings('error', message='.*missing from font.*')   # a missing glyph fails the build


# ---------- fraction-used axes (log scale with an explicit zero row) ----------
ZERO, YMAX = 0.0016, 1.25


def logzero(ax, label='Fraction of topologies used'):
    """Log y axis for fractions; exact zeros sit on a separate '0' row below an axis break."""
    ax.set_yscale('log')
    ax.set_ylim(ZERO / 1.45, YMAX)
    ax.set_yticks([ZERO, 0.01, 0.1, 1.0])
    ax.set_yticklabels(['0', '1%', '10%', '100%'])
    ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(
        [v * 10 ** e for e in (-3, -2, -1) for v in range(2, 10) if v * 10 ** e > 0.0024]))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    yb = (ZERO * 0.0028) ** 0.5
    kw = dict(transform=ax.get_yaxis_transform(), color='black', lw=1.0, clip_on=False)
    for dy in (0.93, 1.07):
        ax.plot([-0.025, 0.025], [yb * dy / 1.06, yb * dy * 1.06], **kw)
    ax.plot([0, 0], [yb * 0.93, yb * 1.07], transform=ax.get_yaxis_transform(), color='white', lw=1.2,
            clip_on=False, zorder=5)
    if label:
        ax.set_ylabel(label)


def yval(k, n):
    return k / n if k > 0 else ZERO


def wilson(k, n, z=1.959964):
    """95% Wilson interval for k of n. Display only: it is a function of the two plotted integers."""
    c = (k + z * z / 2) / (n + z * z)
    h = z / (n + z * z) * math.sqrt(k * (n - k) / n + z * z / 4)
    return max(c - h, 0.0), min(c + h, 1.0)


def frac_series(ax, xs, kn, color, marker, label, pref, ci=True, **kw):
    """Points at k/n (zeros on the zero row), joined by a thin line, over 95% Wilson intervals of the
    plotted counts. Returns label items for place_labels."""
    ys = [yval(k, n) for k, n in kn]
    if ci:
        for x, (k, n) in zip(xs, kn):
            lo, hi = wilson(k, n)
            ax.plot([x, x], [max(lo, ZERO), max(hi, ZERO * 1.02)], color=color, lw=0.8, alpha=0.5,
                    solid_capstyle='butt', zorder=2, clip_on=False)
    kw = dict(dict(lw=1.0, ms=3.8), **kw)
    ax.plot(xs, ys, color=color, marker=marker, label=label, zorder=3, clip_on=False, **kw)
    return [(x, y, f'{k}/{n}', color, pref) for x, y, (k, n) in zip(xs, ys, kn)]


# ---------- automatic placement of point labels away from lines, markers and other labels ----------
def _seg_hits_box(p, q, bb):
    """True if segment p-q (display coords) intersects box bb."""
    (x0, y0), (x1, y1) = p, q
    if max(x0, x1) < bb.x0 or min(x0, x1) > bb.x1 or max(y0, y1) < bb.y0 or min(y0, y1) > bb.y1:
        return False
    for t in [i / 24 for i in range(25)]:
        x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        if bb.x0 <= x <= bb.x1 and bb.y0 <= y <= bb.y1:
            return True
    return False


def _obstacles(ax):
    segs, marks = [], []
    for ln in ax.get_lines():
        if not ln.get_visible():
            continue
        xy = ln.get_transform().transform(ln.get_xydata())
        if ln.get_linestyle() not in ('None', '', ' ') and ln.get_linewidth() > 0:
            segs += list(zip(xy[:-1], xy[1:]))
        if ln.get_marker() not in (None, 'None', '', ' '):
            rad = ln.get_markersize() * ax.figure.dpi / 72.0 / 2.0
            marks += [(p, rad) for p in xy]
    return segs, marks


_RING = {'above': [(0, 6), (5, 5), (-5, 5), (6, 0), (-6, 0), (0, -6), (5, -5), (-5, -5)],
         'below': [(0, -6), (-5, -5), (5, -5), (-6, 0), (6, 0), (0, 6), (-5, 5), (5, 5)]}
CANDIDATES = {k: v + [(int(dx * 1.8), int(dy * 1.8)) for dx, dy in v] for k, v in _RING.items()}


LEGEND = dict(loc='upper right', frameon=True, framealpha=1.0, edgecolor='#CCCCCC', fancybox=False, handlelength=1.6,
              borderpad=0.35, borderaxespad=0.3)


def place_labels(ax, items, fs=7):
    """items: list of (x, y, text, colour, preference 'above'/'below'). Picks the first candidate offset whose
    text box avoids every line segment, marker and earlier label of the axes and stays inside the axes."""
    fig = ax.figure
    fig.set_dpi(300)        # same resolution as the guard
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    segs, marks = _obstacles(ax)
    abox = ax.get_window_extent(r)
    placed = []
    if ax.get_legend() is not None:
        placed.append(ax.get_legend().get_window_extent(r))
    for x, y, s, col, pref in items:
        best = None
        for dx, dy in CANDIDATES[pref]:
            ha = 'center' if dx == 0 else ('left' if dx > 0 else 'right')
            va = 'center' if dy == 0 else ('bottom' if dy > 0 else 'top')
            t = ax.annotate(s, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=fs,
                            color=col, gid='label')
            bb = t.get_window_extent(r).expanded(1.06, 1.12)
            ok = (abox.x0 <= bb.x0 and bb.x1 <= abox.x1 + 40 and abox.y0 - 2 <= bb.y0 and bb.y1 <= abox.y1)
            ok = ok and not any(_seg_hits_box(p, q, bb) for p, q in segs)
            ok = ok and not any(bb.x0 - rd < p[0] < bb.x1 + rd and bb.y0 - rd < p[1] < bb.y1 + rd for p, rd in marks)
            ok = ok and not any(bb.overlaps(o) for o in placed)
            if ok:
                best = t
                placed.append(bb)
                break
            t.remove()
        if best is None:
            raise AssertionError(f'no free position for label {s!r} at ({x}, {y})')
