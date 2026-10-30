"""Occupancy score of a topology fitted on the 96 four-strand topologies: M1 uses the three Minami rule
flags, M2 adds simple sheet geometry. Scores are applied without refitting to larger sheets."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

from . import edits, grammar
from . import topology as TP
from .minami_data import load_sheets
from .paths import DATA

# frame{n}.csv, edit_features{n}.csv, grammar_features5.csv and grammar_features34.csv; only frame4.csv is
# shipped, the rest are written by scripts/topology/ (scripts may point this elsewhere)
INPUT_DIR = DATA

AB = ('a+b', 'a/b')


def ab_occupancy(dom, n):
    """α/β-architecture weighted occupancy and H-group count per topology for n-strand sheets."""
    d = dom[(dom.n == n) & dom.arch.str.lower().str.startswith(AB)].copy()
    d['H'] = d.fid.astype(str).str.split('.').str[:2].str.join('.')
    return d.groupby('topology').agg(freq_ab=('w', 'sum'), nH_ab=('H', 'nunique'))


def cp_class(topology):
    """Canonical label shared by all circular permutations of a topology."""
    seen, todo = {topology}, [topology]
    while todo:
        for _, nb in edits.circular_permutations(todo.pop()):
            if nb not in seen:
                seen.add(nb)
                todo.append(nb)
    return min(seen)


def pp_cross_same(topology):
    """Pairs of parallel crossovers whose spans cross and run the same direction."""
    order, signs = TP.parse(topology)
    par = [c for c in grammar.parallel_connections(order, signs) if c['typ'] == 'P']
    k = 0
    for i, a in enumerate(par):
        for b in par[i + 1:]:
            same_dir = (a['pos2'] > a['pos1']) == (b['pos2'] > b['pos1'])
            k += grammar.span_relation(a, b) == 'crossing' and same_dir
    return k


def grammar_features(n):
    if n == 5:
        g = pd.read_csv(INPUT_DIR / 'grammar_features5.csv').rename(
            columns={'grammar_g1': 'g1', 'grammar_min_support': 'min_support'})
    else:
        g = pd.read_csv(INPUT_DIR / 'grammar_features34.csv').query('n == @n')
    g = g[['topology', 'g1', 'min_support']].copy()
    return g


def features(n):
    """Model covariates X and ECOD occupancy y of every n-strand topology."""
    f = pd.read_csv(INPUT_DIR / f'frame{n}.csv')
    e = pd.read_csv(INPUT_DIR / f'edit_features{n}.csv')[['topology', 'parent_freq_max']]
    f = f.merge(e, on='topology', how='left')
    pairs = f.n_parallel_pairs + f.n_antiparallel_pairs
    X = pd.DataFrame({
        'topology': f.topology,
        'jp': f.jp, 'ov': f.ov, 'en': f.en,
        'all_anti': (f.n_parallel_pairs == 0).astype(int),
        'par_frac': f.n_parallel_pairs / pairs.where(pairs > 0, 1),
        'jump_per_strand': f.sum_jump / n,
        'rco': f.rco_sheet,
        'parent_max': np.log1p(f.parent_freq_max.fillna(0)),
        'cp_class': f.topology.map(cp_class),
        'x_both': ((f.n_crossover_front > 0) & (f.n_crossover_back > 0)).astype(int),
    })
    g = grammar_features(n)
    X = X.merge(g, on='topology', how='left')
    cap = 200.0
    X['log_min_support'] = np.log1p(X.min_support.fillna(cap).clip(upper=cap))
    X['g1'] = X.g1.fillna(0)
    X['pp_cross_same'] = X.topology.map(pp_cross_same)
    _, dom = load_sheets()
    y = f[['topology', 'freq', 'n_Hgroups']].merge(ab_occupancy(dom, n), on='topology', how='left').fillna(0)
    return X, y


MODELS = {
    'M1_rules': ['jp', 'ov', 'en'],
    'M2_rules_geom': ['jp', 'ov', 'en', 'all_anti', 'par_frac', 'jump_per_strand', 'rco'],
    'M3_plus_inflow': ['jp', 'ov', 'en', 'all_anti', 'par_frac', 'jump_per_strand', 'rco', 'parent_max'],
    'M5a_plus_both_faces': ['jp', 'ov', 'en', 'all_anti', 'par_frac', 'jump_per_strand', 'rco', 'x_both'],
    'M5_plus_grammar_support': ['jp', 'ov', 'en', 'all_anti', 'par_frac', 'jump_per_strand', 'rco', 'log_min_support'],
    'M5b_plus_grammar_g1': ['jp', 'ov', 'en', 'all_anti', 'par_frac', 'jump_per_strand', 'rco', 'g1'],
    'M6_plus_pp_cross_same': ['jp', 'ov', 'en', 'all_anti', 'par_frac', 'jump_per_strand', 'rco', 'pp_cross_same'],
}


def cp_loo(X, target):
    """Leave-one-out mean of the target over the other members of each CP class."""
    s = pd.Series(target.values, index=X.index)
    g = s.groupby(X.cp_class)
    tot, cnt = g.transform('sum'), g.transform('count')
    loo = (tot - s) / (cnt - 1)
    return loo.where(cnt > 1, s.mean())


def scores(score, y_obs, y_cont):
    out = {'auroc': roc_auc_score(y_obs, score) if 0 < y_obs.sum() < len(y_obs) else np.nan}
    m = y_obs.astype(bool)
    out['spearman_observed'] = spearmanr(score[m], y_cont[m]).statistic if m.sum() > 3 else np.nan
    out['n'], out['n_observed'] = int(len(y_obs)), int(m.sum())
    return out


def fit_predict(Xtr, ytr_cont, ytr_obs, Xte, cols):
    """Hurdle-style score: P(observed) from logistic + E[log1p count | observed] from linear fit."""
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr[cols], ytr_obs)
    m = ytr_obs.astype(bool)
    reg = LinearRegression().fit(Xtr.loc[m, cols], ytr_cont[m])
    return clf.predict_proba(Xte[cols])[:, 1] * np.clip(reg.predict(Xte[cols]), 0, None)


def dev_cv(X, y, outcome):
    """Leave-one-CP-class-out CV on the 4-strand set (CP siblings never split across folds)."""
    cont = np.log1p(y[outcome])
    obs = (y['freq_ab' if outcome.endswith('_ab') else 'freq'] >= 0.25).astype(int)
    res = {}
    for name, cols in MODELS.items():
        pred = np.zeros(len(X))
        for c in X.cp_class.unique():
            te = X.cp_class == c
            pred[te.values] = fit_predict(X[~te], cont[~te], obs[~te], X[te], cols)
        res[name] = scores(pred, obs.values, cont.values)
    res['M4_cp_loo'] = scores(cp_loo(X, cont).values, obs.values, cont.values)
    return res


def test(X4, y4, X5, y5, outcome):
    cont4, cont5 = np.log1p(y4[outcome]), np.log1p(y5[outcome])
    key = 'freq_ab' if outcome.endswith('_ab') else 'freq'
    obs4, obs5 = (y4[key] >= 0.25).astype(int), (y5[key] >= 0.25).astype(int)
    res = {name: scores(fit_predict(X4, cont4, obs4, X5, cols), obs5.values, cont5.values)
           for name, cols in MODELS.items()}
    res['M4_cp_loo_within_test'] = scores(cp_loo(X5, cont5).values, obs5.values, cont5.values)
    res['T1'] = t1(X5, y5, key)
    return res


def t1(X, y, key):
    """Frustration-free topologies: observed rate with vs without a same-direction crossing crossover pair."""
    from scipy.stats import fisher_exact
    ff = (X.jp + X.ov + X.en) == 0
    obs = (y[key] >= 0.25).values
    has = (X.pp_cross_same > 0).values
    a, b = int((ff & has & obs).sum()), int((ff & has & ~obs).sum())
    c, d = int((ff & ~has & obs).sum()), int((ff & ~has & ~obs).sum())
    p = fisher_exact([[a, b], [c, d]], alternative='less').pvalue if (a + b) and (c + d) else float('nan')
    return dict(with_pair_observed=a, with_pair_unobserved=b, without_observed=c, without_unobserved=d,
                rate_with=a / max(a + b, 1), rate_without=c / max(c + d, 1), fisher_p_less=p)
