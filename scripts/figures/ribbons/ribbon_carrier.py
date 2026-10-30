"""PyMOL ribbon render of ECOD domain e3hmrA1 (PDB 3HMR chain A, residues 43-122) seen edge-on, with the two
connections of its OPP pair (residues 65-86 and 101-109) highlighted, for Figure 1d.

Run:  pymol -cq scripts/figures/ribbons/ribbon_carrier.py -- <3hmr.cif> <outdir>
"""
import sys

from pymol import cmd

C_SHEET = '0x9A9A9A'
C_REST = '0xF0F0F0'
C_A = '0xD55E00'           # vermillion
C_B = '0x0072B2'           # blue
C_HELIX = '0xDDDDDD'

DOMAIN = ('A', 43, 122)
STRANDS = [(43, 52), (61, 64), (87, 91), (96, 100), (110, 122)]
CONN_A = (65, 86)          # sheet position 5 -> 3
CONN_B = (101, 109)        # sheet position 4 -> 2


def main(path, outdir):
    cmd.set('ray_opaque_background', 0)
    cmd.set('ray_trace_mode', 0)
    cmd.set('antialias', 2)
    cmd.set('cartoon_fancy_helices', 1)
    cmd.set('cartoon_transparency', 0.0)
    cmd.set('orthoscopic', 1)
    cmd.set('depth_cue', 0)
    cmd.set('ray_shadow', 0)
    cmd.set('specular', 0.15)

    cmd.load(path, 'raw')
    ch, lo, hi = DOMAIN
    cmd.create('dom', f'raw and polymer and chain {ch} and resi {lo}-{hi}')
    cmd.delete('raw')
    cmd.hide('everything')
    cmd.show('cartoon', 'dom')
    cmd.color(C_REST, 'dom')
    for a, b in STRANDS:
        cmd.color(C_SHEET, f'dom and resi {a}-{b}')
    cmd.color(C_A, 'dom and resi %d-%d' % CONN_A)
    cmd.color(C_B, 'dom and resi %d-%d' % CONN_B)

    # View from the sheet's principal axes: PCA on the strand C-alphas gives e1 and e2 in the sheet
    # plane and e3 along the normal; looking down e2 is edge-on, so the two connections appear on
    # opposite faces.
    import numpy as np
    sel = ' or '.join(f'(dom and resi {a}-{b} and name CA)' for a, b in STRANDS)
    xyz = np.array(cmd.get_coords(sel), float)
    cen = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz - cen)
    e1, e2, e3 = vt[0], vt[1], vt[2]

    def view_along(axis_rows):
        R = np.array(axis_rows, float)
        if np.linalg.det(R) < 0:
            R[1] = -R[1]
        ext = (xyz - cen) @ R.T
        rad = float(np.abs(ext[:, :2]).max()) * 1.35 + 2.0
        return tuple(R.flatten()) + (0.0, 0.0, -rad) + tuple(cen) + (rad * 0.5, rad * 1.5, -20.0)

    views = {'faceon': view_along([e1, e2, e3]), 'edgeon': view_along([e1, e3, e2])}
    for name, v in views.items():
        cmd.set_view(v)
        cmd.zoom('dom', 1.0, complete=1)
        cmd.png(f'{outdir}/ribbon_{name}.png', width=2000, height=1500, dpi=600, ray=1)
    view = views['edgeon']

    with open(f'{outdir}/view.txt', 'w') as fh:
        fh.write(repr(view) + '\n')
    print('wrote', outdir, flush=True)


if __name__ == '__main__' or True:
    args = sys.argv[1:]
    if '--' in args:
        args = args[args.index('--') + 1:]
    if len(args) >= 2:
        main(args[0], args[1])
