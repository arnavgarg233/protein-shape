"""PyMOL ribbon renders, face-on and edge-on, of the interleaved pair in cytidine deaminase (ECOD e1af2A1, PDB 1AF2)
and the nested pair in the SV40 large T antigen origin-binding domain (ECOD e2fufA1, PDB 2FUF) for Figure 4a,b.
It also writes the projected pixel centre of each strand and the angstroms per pixel of each image.

Run:  pymol -cq scripts/figures/ribbons/ribbon_pair.py -- <interleaved|nested> <structure.cif> <outdir>
"""
import json
import math
import sys

import numpy as np
from pymol import cmd

C_SHEET = '0x9A9A9A'
C_REST = '0xF0F0F0'

CARRIERS = {
    'interleaved': dict(domain=('A', 1, 150), topology='1|3|2|4|5',
                        strands=[(32, 33), (74, 79), (84, 88), (119, 124), (146, 148)],
                        conn_a=(34, 73), conn_b=(89, 118), c_a='0xD55E00', c_b='0xE69F00'),
    'nested': dict(domain=('A', 131, 256), topology='4|1|3|2|5',
                   strands=[(155, 163), (183, 189), (192, 204), (221, 225), (242, 246)],
                   conn_a=(164, 182), conn_b=(226, 241), c_a='0x0072B2', c_b='0x56B4E9'),
}
W, H = 2000, 1500


def main(key, path, outdir):
    car = CARRIERS[key]
    cmd.set('ray_opaque_background', 0)
    cmd.set('ray_trace_mode', 0)
    cmd.set('max_threads', 1)
    cmd.set('antialias', 2)
    cmd.set('cartoon_fancy_helices', 1)
    cmd.set('cartoon_transparency', 0.0)
    cmd.set('orthoscopic', 1)
    cmd.set('depth_cue', 0)
    cmd.set('ray_shadow', 0)
    cmd.set('specular', 0.15)

    cmd.load(path, 'raw')
    ch, lo, hi = car['domain']
    cmd.create('dom', f'raw and polymer and chain {ch} and resi {lo}-{hi}')
    cmd.delete('raw')
    cmd.hide('everything')
    cmd.show('cartoon', 'dom')
    cmd.color(C_REST, 'dom')
    for a, b in car['strands']:
        cmd.color(C_SHEET, f'dom and resi {a}-{b}')
    cmd.color(car['c_a'], 'dom and resi %d-%d' % car['conn_a'])
    cmd.color(car['c_b'], 'dom and resi %d-%d' % car['conn_b'])

    ca = lambda a, b: np.array(cmd.get_coords(f'dom and resi {a}-{b} and name CA'), float)
    xyz = np.vstack([ca(a, b) for a, b in car['strands']])
    cen = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz - cen)
    n = vt[2]
    # strand centres, indexed by sequence order; sheet position of strand i from the topology string
    sc = np.array([ca(a, b).mean(0) for a, b in car['strands']])
    pos = {int(s): p + 1 for p, s in enumerate(car['topology'].split('|'))}
    by_pos = sc[[pos_s - 1 for pos_s in sorted(pos, key=pos.get)]]
    inpl = (by_pos - cen) - np.outer((by_pos - cen) @ n, n)
    u = np.linalg.svd(inpl - inpl.mean(0))[2][0]
    u -= (u @ n) * n
    u /= np.linalg.norm(u)
    if (by_pos[-1] - by_pos[0]) @ u < 0:
        u = -u
    side = (ca(*car['conn_a']).mean(0) - cen) @ n
    if side < 0:
        n = -n
    v = np.cross(u, n)

    def view_along(rows):
        R = np.array(rows, float)
        ext = (xyz - cen) @ R.T
        rad = float(np.abs(ext[:, :2]).max()) * 1.35 + 2.0
        return tuple(R.T.flatten()) + (0.0, 0.0, -rad) + tuple(cen) + (rad * 0.5, rad * 1.5, -20.0)

    views = {'faceon': view_along([u, np.cross(n, u), n]), 'edgeon': view_along([u, n, v])}
    proj, apx = {}, {}
    for name, vw in views.items():
        cmd.set_view(vw)
        cmd.zoom('dom', 1.0, complete=1)
        cmd.png(f'{outdir}/ribbon_{key}_{name}.png', width=W, height=H, dpi=600, ray=1)
        view = cmd.get_view()
        M = np.array(view[:9]).reshape(3, 3)
        cam = (sc - np.array(view[12:15])) @ M + np.array(view[9:12])
        half_h = abs(view[11]) * math.tan(math.radians(cmd.get_setting_float('field_of_view')) / 2)
        half_w = half_h * W / H
        apx[name] = 2 * half_h / H
        proj[name] = {str(pos[i + 1]): [W / 2 + c[0] / half_w * W / 2, H / 2 - c[1] / half_h * H / 2]
                      for i, c in enumerate(cam)}
    with open(f'{outdir}/ribbon_{key}_positions.json', 'w') as fh:
        json.dump(dict(carrier=key, width=W, height=H, strand_centre_px=proj, angstrom_per_px=apx), fh, indent=1, sort_keys=True)
    print('wrote', key, outdir, flush=True)


if __name__ == '__main__' or True:
    args = sys.argv[1:]
    if '--' in args:
        args = args[args.index('--') + 1:]
    if len(args) >= 3:
        main(args[0], args[1], args[2])
