"""Figure 3: fraction of topologies used with and without the OPP pair, by strand count, by parallel crossover
count at seven strands, and split by whether the sheet carries a ferredoxin-type pair (lenient credit).
"""
import json

import matplotlib.pyplot as plt

from figstyle import BOX_H, BOX_W, FIGURE_INPUTS, LABEL_FS, LEGEND, OPP_C, PLOT_FS, REF_C, TABLES, assert_box, boxed_axes, \
    finish, frac_series, logzero, panel_label, parse_frac, place_labels

J = lambda p: json.loads((FIGURE_INPUTS / p).read_text())  # noqa: E731
core, t2r1, fdn = J('core_type.json'), J('census_primary_tests.json'), J('ferredoxin_by_n.json')
LP = json.loads((TABLES / 'lenient_primary.json').read_text())['figure_data']
NS = (5, 6, 7)


def series(cr):
    """Panel counts under one credit: a (opp, ref), b strata, c/d by ferredoxin-type pair."""
    d = LP[cr]
    a = ([tuple(d['fig3_a'][str(n)]['opp']) for n in NS], [tuple(d['fig3_a'][str(n)]['ref']) for n in NS])
    b = [(r['npar_stratum'], tuple(r['opp']), tuple(r['ref'])) for r in d['fig3_b_7strand']]
    cd = {k: ([tuple(d['fig3_cd'][f'fd{k}'][str(n)]['opp']) for n in NS],
              [tuple(d['fig3_cd'][f'fd{k}'][str(n)]['ref']) for n in NS]) for k in (0, 1)}
    return a, b, cd


def counts(strata):
    out = []
    for s in strata:
        ka, kb = s['frac_pos_a'] * s['n_a'], s['frac_pos_b'] * s['n_b']
        assert abs(ka - round(ka)) < 1e-9 and abs(kb - round(kb)) < 1e-9
        out.append((int(s['stratum'].strip('(),')), (round(ka), s['n_a']), (round(kb), s['n_b'])))
    return out


# a: per strand count
opp_n = [(core[f'all_OPP_{n}strand']['used'], core[f'all_OPP_{n}strand']['n_exposed']) for n in NS]
ref_n = [(core[f'all_OPP_{n}strand']['ref_used'], core[f'all_OPP_{n}strand']['ref']) for n in NS]
strata = {5: t2r1['S3_5strand']['strata'], 6: t2r1['S3_6strand']['strata'], 7: t2r1['PRIMARY_T2R1_7strand']['strata']}
for i, n in enumerate(NS):  # strata sums equal the per-n totals
    c = counts(strata[n])
    assert (sum(x[1][0] for x in c), sum(x[1][1] for x in c)) == opp_n[i]
    assert (sum(x[2][0] for x in c), sum(x[2][1] for x in c)) == ref_n[i]
    fd = [parse_frac(fdn[f'n{n}_fd{k}'][arm]) for k in (0, 1) for arm in ('exp', 'ref')]
    assert (fd[0][0] + fd[2][0], fd[0][1] + fd[2][1]) == opp_n[i]
    assert (fd[1][0] + fd[3][0], fd[1][1] + fd[3][1]) == ref_n[i]
# b: 7-strand primary strata
c7 = counts(strata[7])
assert ''.join(f'{k}/{n}' for k, n in opp_n[2:]) == t2r1['S1_binary_LH_ge_1_7strand']['exposed']
assert ''.join(f'{k}/{n}' for k, n in ref_n[2:]) == t2r1['S1_binary_LH_ge_1_7strand']['reference']
# the agreement-rule counts equal the census result files above
fd_census = {k: ([parse_frac(fdn[f'n{n}_fd{k}']['exp']) for n in NS], [parse_frac(fdn[f'n{n}_fd{k}']['ref']) for n in NS])
             for k in (0, 1)}
assert series('agree') == ((opp_n, ref_n), c7, fd_census)
# plotted: lenient credit. Same topologies in every arm and panel; strata and fd halves sum to the per-n totals.
(opp_n, ref_n), c7, fdl = series('lenient')
assert [n for _, n in opp_n] == [n for _, n in series('agree')[0][0]]
assert [n for _, n in ref_n] == [n for _, n in series('agree')[0][1]]
assert (sum(o[0] for _, o, _ in c7), sum(o[1] for _, o, _ in c7)) == opp_n[2]
assert (sum(r[0] for _, _, r in c7), sum(r[1] for _, _, r in c7)) == ref_n[2]
for i in range(3):
    for j, tot in ((0, opp_n), (1, ref_n)):
        assert (fdl[0][j][i][0] + fdl[1][j][i][0], fdl[0][j][i][1] + fdl[1][j][i][1]) == tot[i]

# Same BOX_W x BOX_H axes and type sizes as Fig. 2; a, b on top, c, d (the two halves of a) below.
L_IN, GAP_IN, R_IN, B_IN, MID_IN, T_IN = 0.66, 0.55, 0.12, 0.44, 0.72, 0.30
W_IN = L_IN + 2 * BOX_W + GAP_IN + R_IN
H_IN = B_IN + 2 * BOX_H + MID_IN + T_IN
fig = plt.figure(figsize=(W_IN, H_IN))
XL, XR, YB, YT = L_IN, L_IN + BOX_W + GAP_IN, B_IN, B_IN + BOX_H + MID_IN
A, B = boxed_axes(fig, W_IN, H_IN, XL, YT), boxed_axes(fig, W_IN, H_IN, XR, YT)
Cc, D = boxed_axes(fig, W_IN, H_IN, XL, YB), boxed_axes(fig, W_IN, H_IN, XR, YB)

LAB = {}
LAB['A'] = frac_series(A, NS, opp_n, OPP_C, 'o', 'OPP pair', 'below') + \
    frac_series(A, NS, ref_n, REF_C, 's', 'reference', 'above')
A.set_title('All OPP topologies', pad=4, fontsize=PLOT_FS)

xs = [x for x, _, _ in c7]
LAB['B'] = frac_series(B, xs, [o for _, o, _ in c7], OPP_C, 'o', 'OPP pair', 'below') + \
    frac_series(B, xs, [r for _, _, r in c7], REF_C, 's', 'reference', 'above')
B.set_title('7 strands, by number of parallel crossovers', pad=4, fontsize=PLOT_FS)
B.set_xticks(xs)
B.set_xlim(min(xs) - 0.75, max(xs) + 0.75)
B.set_xlabel('Parallel crossovers in sheet', fontsize=PLOT_FS, labelpad=2)

for ax, key, k, title in ((Cc, 'C', 1, 'Ferredoxin-type (C2z-crossing) pair present'),
                          (D, 'D', 0, 'No ferredoxin-type pair')):
    LAB[key] = frac_series(ax, NS, fdl[k][0], OPP_C, 'o', 'OPP pair', 'below') + \
        frac_series(ax, NS, fdl[k][1], REF_C, 's', 'reference', 'above')
    ax.set_title(title, pad=4, fontsize=PLOT_FS)

for ax in (A, Cc, D):
    ax.set_xticks(NS)
    ax.set_xlim(4.35, 7.65)
    ax.set_xlabel('Strands in sheet', fontsize=PLOT_FS, labelpad=2)
for ax in (A, B, Cc, D):
    logzero(ax, label='Fraction used' if ax in (A, Cc) else None)
    ax.yaxis.label.set_fontsize(PLOT_FS)
    ax.tick_params(labelsize=PLOT_FS)
A.legend(**dict(LEGEND, fontsize=PLOT_FS))
for ax, key in ((A, 'A'), (B, 'B'), (Cc, 'C'), (D, 'D')):
    place_labels(ax, LAB[key], fs=LABEL_FS)
for ax, s in zip((A, B, Cc, D), 'abcd'):
    panel_label(ax, s, x=-0.55 / BOX_W if ax in (A, Cc) else -0.30 / BOX_W, y=1.03)
assert_box(fig, (A, B, Cc, D))

finish(fig, 'fig3_rule')
print('plotted (lenient credit)', dict(a_opp=opp_n, a_ref=ref_n, b=c7, c=fdl[1], d=fdl[0]))
