"""Sheet cartoons drawn from topology strings: a face-on view (strands as arrows, crossover helices in front of
or behind the sheet) and an edge-on view (strands end-on, crossover helices above or below the sheet).

Both views share one frame: x across the strands (sheet position 1..n), y along the strands, face +1 toward the
viewer of the face-on view and up in the edge-on view, which looks along +y.
"""
from matplotlib.patches import Circle, FancyArrow, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

from figstyle import C
from protein_shape import grammar, s2x
from protein_shape import topology as TP

STRAND_FILL, STRAND_EDGE, CORE_FILL = '#D9D9D9', '#606060', '#A6A6A6'
LOOP = '#555555'
L = 1.0          # half strand length (face-on)
HELIX_HALF = 0.5


def sheet(topo):
    order, signs = TP.parse(topo)
    b, cons = TP.connections(order, signs)
    return order, signs, b, cons


def crossovers(topo):
    """Parallel crossovers (s2x convention) with their face."""
    out = []
    for c in s2x.crossovers(topo):
        c = dict(c)
        c['face'] = grammar._rh_face(c)
        out.append(c)
    return out


def pairs(topo):
    """(idx_a, idx_b, relation, span) for every pair of parallel crossovers."""
    par = s2x.crossovers(topo)
    out = []
    for i, a in enumerate(par):
        for b in par[i + 1:]:
            out.append((a['idx'], b['idx'], s2x.relation(a, b), grammar.span_relation(a, b)))
    return out


def koga_to_minami(topo):
    o, s = TP.parse(topo)
    return TP.minami(o, s)


def pretty(topo):
    """Minami string with true minus signs for print."""
    return koga_to_minami(topo).replace('-', '−').replace(':', ' : ')


def _bez(ax, p0, c0, c1, p1, **kw):
    path = MPath([p0, c0, c1, p1], [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, capstyle='round', **kw))
    # extreme y of a cubic whose inner control points share one height: 0.25 * end + 0.75 * control
    return 0.25 * p0[1] + 0.75 * c0[1], 0.25 * p1[1] + 0.75 * c1[1]


def keepout(ax, ext, pad=0.05):
    """Invisible rectangle over a cartoon; the render-time guard fails if a checked text enters it."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((ext['xmin'] - pad, ext['ymin'] - pad), ext['xmax'] - ext['xmin'] + 2 * pad,
                           ext['ymax'] - ext['ymin'] + 2 * pad, fill=False, lw=0, alpha=0, gid='keepout'))


def draw_face(ax, topo, ox=0.0, oy=0.0, hl=None, core=(), numbers=True, termini=False, lw=1.0,
              number_size=7.5):
    """Face-on cartoon. hl: {crossover idx: colour}; core: sequence indices (1-based) of strands to shade."""
    hl = hl or {}
    order, signs, b, cons = sheet(topo)
    faces = {c['idx']: c['face'] for c in crossovers(topo)}
    xs = [ox + st['pos'] - 1 for st in b]
    for k, st in enumerate(b):
        s = st['sign']
        fill = CORE_FILL if (k + 1) in core else STRAND_FILL
        ax.add_patch(FancyArrow(xs[k], oy - s * L, 0, 2 * s * L, width=0.36, head_width=0.58, head_length=0.36,
                                length_includes_head=True, fc=fill, ec=STRAND_EDGE, lw=1.0, zorder=3))
        if numbers:
            ax.text(xs[k], oy - s * 0.79, str(k + 1), ha='center', va='center', fontsize=number_size, zorder=9,
                    gid='nocheck', bbox=dict(boxstyle='round,pad=0.12,rounding_size=0.3', fc=fill, ec='none'))
    if termini:
        s0, s1 = b[0]['sign'], b[-1]['sign']
        ax.text(xs[0], oy - s0 * (L + 0.22), 'N', ha='center', va='center', fontsize=7, gid='nocheck',
                color=C['grey'])
        ax.text(xs[-1], oy + s1 * (L + 0.22), 'C', ha='center', va='center', fontsize=7, gid='nocheck',
                color=C['grey'])
    used_h = []
    ys = [oy - L - 0.36, oy + L + 0.36]   # arrowheads
    for c in cons:
        x1, x2 = ox + c['pos1'] - 1, ox + c['pos2'] - 1
        s = c['sign1']
        if c['type'] == 'turn':
            e = oy + s * L
            h = s * (0.32 + 0.16 * c['jump'])
            ys += _bez(ax, (x1, e), (x1, e + h), (x2, e + h), (x2, e), color=LOOP, lw=max(1.0, lw * 0.8), zorder=2)
            continue
        f = faces[c['i1']]
        col = hl.get(c['i1'], C['grey'])
        xh = (x1 + x2) / 2
        while any(abs(xh - u) < 0.3 for u in used_h):
            xh += 0.3
        used_h.append(xh)
        front = f > 0
        z = 6 if front else 1
        ls = '-' if front else (0, (2.2, 1.4))
        lift = s * (L + 0.45)
        ys += _bez(ax, (x1, oy + s * L), (x1, oy + lift), (xh, oy + lift), (xh, oy + s * HELIX_HALF),
                   color=col, lw=lw, ls=ls, zorder=z)
        ys += _bez(ax, (xh, oy - s * HELIX_HALF), (xh, oy - lift), (x2, oy - lift), (x2, oy - s * L),
                   color=col, lw=lw, ls=ls, zorder=z)
        ax.add_patch(FancyBboxPatch((xh - 0.15, oy - HELIX_HALF), 0.30, 2 * HELIX_HALF,
                                    boxstyle='round,pad=0,rounding_size=0.15', fc=col,
                                    alpha=1.0 if front else 0.35, ec=col, lw=1.0, ls=ls, zorder=z + 1))
    return dict(xmin=ox - 0.3, xmax=ox + len(b) - 0.7, ymin=min(ys), ymax=max(ys))


def _strand_symbol(ax, x, y, sign, r=0.16, z=4):
    ax.add_patch(Circle((x, y), r, fc='white', ec='black', lw=1.0, zorder=z))
    if sign > 0:   # pointing away from the viewer (+y): cross
        d = r * 0.62
        ax.plot([x - d, x + d], [y - d, y + d], color='black', lw=1.0, zorder=z + 1, solid_capstyle='butt')
        ax.plot([x - d, x + d], [y + d, y - d], color='black', lw=1.0, zorder=z + 1, solid_capstyle='butt')
    else:          # toward the viewer: dot
        ax.add_patch(Circle((x, y), r * 0.28, fc='black', ec='none', zorder=z + 1))


def draw_edge(ax, topo, ox=0.0, oy=0.0, hl=None, only=None, arrows=True, lw=1.0, height=0.8):
    """Edge-on view (crossovers only). only: crossover idx to draw (default all)."""
    hl = hl or {}
    order, signs, b, cons = sheet(topo)
    n = len(b)
    ax.plot([ox - 0.55, ox + n - 0.45], [oy, oy], color=C['lgrey'], lw=1.6, zorder=1, solid_capstyle='round')
    for k in range(n):
        _strand_symbol(ax, ox + k, oy, int(signs[k] + '1'))
    placed = []
    ys = [oy - 0.17, oy + 0.17]
    for c in crossovers(topo):
        if only is not None and c['idx'] not in only:
            continue
        x1, x2 = ox + c['pos1'] - 1, ox + c['pos2'] - 1
        f = c['face']
        xh = (x1 + x2) / 2
        lev = 0
        while any(abs(xh - u) < 0.55 and fu == f and lu == lev for u, fu, lu in placed):
            lev += 1
        placed.append((xh, f, lev))
        yh = oy + f * (height + 0.5 * lev)
        col = hl.get(c['idx'], C['grey'])
        r = 0.22
        for xa in (x1, x2):
            ax.plot([xa, xh], [oy + f * 0.16, yh], color=col, lw=lw, zorder=2, solid_capstyle='round')
        if arrows:
            # arrowhead on the helix -> second strand leg shows the lateral direction d
            t0, t1 = 0.35, 0.80
            p0 = (xh + (x2 - xh) * t0, yh + (oy + f * 0.16 - yh) * t0)
            p1 = (xh + (x2 - xh) * t1, yh + (oy + f * 0.16 - yh) * t1)
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=5.5, color=col, lw=0,
                                         zorder=3, shrinkA=0, shrinkB=0))
        ax.add_patch(Circle((xh, yh), r, fc=col, ec='none', zorder=3))
        ys += [yh - r, yh + r]
    return dict(xmin=ox - 0.55, xmax=ox + n - 0.45, ymin=min(ys), ymax=max(ys))
