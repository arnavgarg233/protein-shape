"""Figure 2: AUROC of the rules-only (M1) and rules-plus-geometry (M2) scores, fitted on the 96 four-strand
topologies, against ECOD and AFDB-census use of five- to seven-strand topologies. Values are read, not recomputed.
"""
import json

import matplotlib.pyplot as plt

from figstyle import BOX_H, BOX_W, C, FIGURE_INPUTS, LABEL_FS, PLOT_FS, assert_box, boxed_axes, finish, panel_label

J = lambda p: json.loads((FIGURE_INPUTS / p).read_text())  # noqa: E731
hold, fin6, t2, geo = J('score_5_strand.json'), J('score_6_strand.json'), J('score_7_strand.json'), \
    J('score_census.json')
ecod_src = {5: hold['test_5strand']['n_Hgroups'], 6: fin6['n_Hgroups'], 7: t2['models_7strand']['n_Hgroups']}
NS = (5, 6, 7)
MODELS = (('M1_rules', 'M1 rules', C['sky'], 's'), ('M2_rules_geom', 'M2 rules + geometry', C['blue'], 'o'))
ecod = {m: [ecod_src[n][m]['auroc'] for n in NS] for m, *_ in MODELS}
for m, *_ in MODELS:
    for i, n in enumerate(NS):
        assert abs(geo[str(n)][m]['auroc_ecod'] - ecod[m][i]) < 1e-12, (m, n)
afdb = {m: [geo[str(n)][m]['auroc_afdb'] for n in NS] for m, *_ in MODELS}
afdb_new = [geo[str(n)]['M2_rules_geom']['auroc_afdb_among_ecod_unobserved'] for n in NS]
n_used_ecod = [ecod_src[n]['M2_rules_geom']['n_observed'] for n in NS]
n_used_afdb = [geo[str(n)]['afdb_used'] for n in NS]
n_top = [geo[str(n)]['n'] for n in NS]
assert n_used_ecod == [geo[str(n)]['ecod_observed'] for n in NS]

# Two axes of the BOX_W x BOX_H box that Fig. 3 uses, with the same type sizes.
L_IN, GAP_IN, R_IN, B_IN, T_IN = 0.66, 0.18, 0.10, 0.70, 0.30
W_IN = L_IN + 2 * BOX_W + GAP_IN + R_IN
H_IN = B_IN + BOX_H + T_IN
fig = plt.figure(figsize=(W_IN, H_IN))
a0 = boxed_axes(fig, W_IN, H_IN, L_IN, B_IN)
a1 = boxed_axes(fig, W_IN, H_IN, L_IN + BOX_W + GAP_IN, B_IN)
a1.sharey(a0)
a1.tick_params(labelleft=False)
axs = [a0, a1]
for ax, title, vals, nused in ((axs[0], 'ECOD (experimental)', ecod, n_used_ecod),
                               (axs[1], 'AFDB census', afdb, n_used_afdb)):
    for m, lab, col, mk in MODELS:
        ax.plot(NS, vals[m], marker=mk, color=col, ms=3.6, lw=1.0, label=lab, clip_on=False, zorder=3)
    ax.set_title(title, fontsize=PLOT_FS, pad=3)
    ax.set_xticks(NS)
    ax.set_xlim(4.6, 7.4)
    ax.set_xlabel('Strands in sheet', fontsize=PLOT_FS, labelpad=2)
    ax.text(6, 0.737, 'topologies used', ha='center', va='bottom', fontsize=LABEL_FS, color=C['grey'])
    for n, k in zip(NS, nused):
        ax.text(n, 0.712, f'{k}', ha='center', va='bottom', fontsize=LABEL_FS, color=C['grey'])
axs[1].plot(NS, afdb_new, marker='o', mfc='white', color=C['blue'], ls=(0, (2.5, 1.5)), ms=3.6, lw=1.0,
            label='M2, ECOD-unobserved only', clip_on=False, zorder=3)
axs[0].set_ylim(0.70, 0.95)
axs[0].set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
axs[0].set_ylabel('AUROC (used vs unused)', fontsize=PLOT_FS)
h0, l0 = axs[0].get_legend_handles_labels()
h1, l1 = axs[1].get_legend_handles_labels()
fig.legend(h0 + h1[2:], l0 + l1[2:], loc='lower center', ncol=3,
           bbox_to_anchor=((L_IN + BOX_W + GAP_IN / 2) / W_IN, 0.0), handlelength=1.8,
           columnspacing=1.0, handletextpad=0.4, fontsize=PLOT_FS, borderaxespad=0.15)
panel_label(axs[0], 'a', x=-0.55 / BOX_W, y=1.03)
panel_label(axs[1], 'b', x=-0.03, y=1.03)
assert_box(fig, axs)

finish(fig, 'fig2_prediction')
print('plotted', dict(ecod=ecod, afdb=afdb, afdb_new=afdb_new, n_used_ecod=n_used_ecod, n_used_afdb=n_used_afdb,
                      n_topologies=n_top))
