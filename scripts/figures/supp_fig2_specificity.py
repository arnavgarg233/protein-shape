"""Supplementary Figure 2: the OPP pair against related features of sheet topology: core type, the frustrated
alpha-beta pair of Murata and co-workers, chain self-entanglement at matched jump length, and crossing against
non-crossing opposite-face pairs.
"""
import ast
import json
from collections import defaultdict

import matplotlib.pyplot as plt

import topodraw as TD
from figstyle import (C, DOUBLE, FIGURE_INPUTS, LEGEND, OPP_C, REF_C, ZERO, finish, frac_series, logzero, panel_label,
                      parse_frac, place_labels, wilson, yval)

J = lambda p: json.loads((FIGURE_INPUTS / p).read_text())  # noqa: E731
core, fabp, t3 = J('core_type.json'), J('fabp_general.json'), J('frustration_free_tests.json')
NS = (5, 6, 7)

# ---- data a
nf1 = [(core[f'only_NF1_core_{n}strand']['used'], core[f'only_NF1_core_{n}strand']['n_exposed']) for n in NS]
alt = [(core[f'any_3142_core_{n}strand']['used'], core[f'any_3142_core_{n}strand']['n_exposed']) for n in NS]
ref = [(core[f'all_OPP_{n}strand']['ref_used'], core[f'all_OPP_{n}strand']['ref']) for n in NS]
for i, n in enumerate(NS):
    a = core[f'all_OPP_{n}strand']
    assert nf1[i][0] + alt[i][0] == a['used'] and nf1[i][1] + alt[i][1] == a['n_exposed']
    assert core[f'only_NF1_core_{n}strand']['ref'] == a['ref'] == core[f'any_3142_core_{n}strand']['ref']

# ---- data c
agg = defaultdict(lambda: [0, 0, 0, 0])
for s in t3['PRIMARY_T3_7strand']['strata']:
    npar, sj = (int(v) for v in s['stratum'].strip("(),'").split('_'))
    ka, kb = s['frac_pos_a'] * s['n_a'], s['frac_pos_b'] * s['n_b']
    assert abs(ka - round(ka)) < 1e-9 and abs(kb - round(kb)) < 1e-9
    g = agg[sj]
    g[0] += round(ka); g[1] += s['n_a']; g[2] += round(kb); g[3] += s['n_b']  # noqa: E702
SJ = sorted(agg)
forced = [(agg[j][0], agg[j][1]) for j in SJ]
t3ref = [(agg[j][2], agg[j][3]) for j in SJ]
fk, fn = parse_frac(t3['PRIMARY_binary_LH_ge_1']['exposed'])
assert (sum(k for k, _ in forced), sum(n for _, n in forced)) == (fk, fn)

# ---- data d
arms = {}
for line in (FIGURE_INPUTS / 'relation_arms.txt').read_text().splitlines():
    head, _, rest = line.partition(' ')
    if head in ('OPP', 'E', 'R1', 'R2') and rest.startswith('{'):
        arms[head] = ast.literal_eval(rest)
assert arms['OPP'] == {'fd1': '9/406', 'fd0': '17/237'}
chk = J('ferredoxin_check.json')['pooled_fd_core_used']
assert (chk['exp_fd1'], chk['exp_fd0']) == (arms['OPP']['fd1'], arms['OPP']['fd0'])

fig = plt.figure(figsize=(DOUBLE, 3.85))
gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, bottom=0.12, top=0.89, wspace=0.22, hspace=0.62)
A, B, Cc, D = (fig.add_subplot(gs[i, j]) for i in (0, 1) for j in (0, 1))

# a
LAB = frac_series(A, [n - 0.08 for n in NS], nf1, OPP_C, 'o', 'OPP, NF1 cores only', 'below') + \
    frac_series(A, [n + 0.08 for n in NS], alt, C['purple'], '^', 'OPP, with 3|1|4|2 core', 'below',
                mfc='white') + \
    frac_series(A, NS, ref, REF_C, 's', 'reference', 'above')
A.set_title('Core type of the OPP pair', pad=13)
A.set_xticks(NS)
A.set_xlim(4.35, 7.65)
A.set_xlabel('Strands in sheet')
logzero(A, 'Fraction used')
A.legend(loc='lower left', bbox_to_anchor=(-0.01, 0.99), ncol=3, frameon=False, handlelength=1.6,
         columnspacing=0.8, handletextpad=0.4, borderaxespad=0)
place_labels(A, LAB)

# b  schematic (data-unit axes with equal aspect)
B.set_title('Not a frustrated \u03b1\u03b2-unit pair (Murata 2024)', pad=13)
B.set_xlim(-0.8, 10.3)
B.set_ylim(-3.55, 2.05)
B.set_aspect('equal')
B.axis('off')
for topo, x0, lab1, lab2, lab3, rel in (
        ('132+++', 0.9, 'Murata F\u03b1\u03b2P', '\u03b1\u03b2-strands parallel', 'unsatisfiable', 'C2y'),
        ('1324+-+-', 5.9, 'NF1 core (OPP)', '\u03b1\u03b2-strands antiparallel', 'satisfiable', 'S2x')):
    pr = [p for p in TD.pairs(topo) if p[2] == rel]
    assert len(pr) == 1, (topo, pr)
    col = {'C2y': C['purple'], 'S2x': OPP_C}[rel]
    hl = {pr[0][0]: col, pr[0][1]: col}
    ab = {i + 2 for i in hl}       # alpha-beta strand = second strand of each crossover (1-based)
    TD.keepout(B, TD.draw_face(B, topo, ox=x0, oy=0.0, hl=hl, core=ab))
    xc = x0 + (len(TD.sheet(topo)[0]) - 1) / 2
    # name and topology string on separate lines, so the two schematics' labels do not collide
    B.text(xc, -1.95, lab1, ha='center', va='center', fontsize=7)
    B.text(xc, -2.45, TD.pretty(topo), ha='center', va='center', fontsize=7)
    B.text(xc, -2.95, lab2, ha='center', va='center', fontsize=7)
    B.text(xc, -3.45, lab3, ha='center', va='center', fontsize=7, color=col, fontweight='bold')
gen = fabp['a_overlap']
val = fabp['V1_murata_167']
B.text(4.75, 1.8, f"generalised F\u03b1\u03b2P in OPP topologies: {gen['exposed_with_generalised_FabP']} of "
       f"{gen['exposed']}", ha='center', va='center', fontsize=7)

# c
LAB = frac_series(Cc, SJ, forced, OPP_C, 'D', 'forced self-entangled', 'below', ls='none') + \
    frac_series(Cc, SJ, t3ref, REF_C, 's', 'reference, same strata', 'above', ls='none')
Cc.set_title('7 strands: self-entanglement at matched jump length', pad=13)
Cc.set_xticks(SJ)
Cc.set_xticklabels([str(j) if j < 8 else '≥8' for j in SJ])
Cc.set_xlim(min(SJ) - 0.6, max(SJ) + 0.6)
Cc.set_xlabel('Summed jump length of connections (strands)')
logzero(Cc, 'Fraction used')
Cc.legend(loc='lower left', bbox_to_anchor=(-0.01, 0.99), ncol=2, frameon=False, handlelength=1.6,
          columnspacing=0.8, handletextpad=0.4, borderaxespad=0)
place_labels(Cc, LAB)

# d
ARMS = [('OPP', 'OPP\n(crossing S2x)', OPP_C), ('E', 'non-crossing\nS2x', C['vermillion']),
        ('R1', 'mixed orient.,\none face', REF_C), ('R2', 'one orient.,\ntwo faces', C['sky'])]
LAB = []
for i, (key, lab, col) in enumerate(ARMS):
    for fd, dx, mfc in (('fd0', -0.13, 'white'), ('fd1', 0.13, col)):
        k, n = parse_frac(arms[key][fd])
        if n == 0:
            continue
        lo, hi = wilson(k, n)
        D.plot([i + dx] * 2, [max(lo, ZERO), max(hi, ZERO * 1.02)], color=col, lw=0.8, alpha=0.5,
               solid_capstyle='butt', zorder=2, clip_on=False)
        D.plot([i + dx], [yval(k, n)], marker='o', ms=3.8, color=col, mfc=mfc, ls='none', zorder=3,
               clip_on=False)
        LAB.append((i + dx, yval(k, n), f'{k}/{n}', col, 'below' if fd == 'fd0' else 'above'))
D.plot([], [], marker='o', ms=3.8, color=C['grey'], mfc='white', ls='none', label='no ferredoxin-type pair')
D.plot([], [], marker='o', ms=3.8, color=C['grey'], ls='none', label='ferredoxin-type pair present')
D.set_xticks(range(len(ARMS)))
D.set_xticklabels([a[1] for a in ARMS], fontsize=7)
D.set_xlim(-0.55, len(ARMS) - 0.45)
D.set_title('5–7 strands: crossing vs non-crossing S2x', pad=4)
logzero(D, None)
D.legend(**dict(LEGEND, handlelength=1.0, loc='lower right'))
place_labels(D, LAB)

for ax, s, x in ((A, 'a', -0.10), (B, 'b', -0.03), (Cc, 'c', -0.10), (D, 'd', -0.05)):
    lab = panel_label(ax, s, x=x, y=1.03)
    if ax in (A, B, Cc):   # titles sit above the legend of panels a and c
        lab.set_y(1.0)
        lab.set_transform(ax.title.get_transform())
        lab.set_va('baseline')

finish(fig, 'supp_fig2_specificity')
print('plotted', dict(a_nf1=nf1, a_alt=alt, a_ref=ref, b=dict(gen=gen['exposed_with_generalised_FabP'],
      exposed=gen['exposed'], murata_match=val['M24_unsat_iff_NFabP_gt0'], murata_n=val['n']),
      c=dict(sum_jump=SJ, forced=forced, ref=t3ref), d=arms))
