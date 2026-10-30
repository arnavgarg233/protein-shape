"""Kaplan-Meier median, restricted mean survival time and log-rank test for right-censored folding times."""
import numpy as np
from scipy import stats


def km_median(times, events):
    """Kaplan-Meier median (inf if survival never reaches 0.5)."""
    order = np.argsort(times)
    t, e = np.asarray(times)[order], np.asarray(events)[order]
    S, n = 1.0, len(t)
    for k in range(n):
        if e[k]:
            S *= 1 - 1.0 / (n - k)
            if S <= 0.5:
                return float(t[k])
    return np.inf


def km_rmst(times, events, tmax):
    """Restricted mean survival time up to tmax, the area under the Kaplan-Meier curve."""
    t = np.minimum(times, tmax); e = np.where(np.asarray(times) > tmax, 0, events)
    order = np.argsort(t); t, e = t[order], e[order]
    S, last, area, n = 1.0, 0.0, 0.0, len(t)
    for k in range(n):
        area += S * (t[k] - last); last = t[k]
        if e[k]:
            S *= 1 - 1.0 / (n - k)
    area += S * (tmax - last)
    return float(area)


def logrank(t1, e1, t2, e2):
    """Two-sample log-rank z and two-sided p."""
    t = np.concatenate([t1, t2]); e = np.concatenate([e1, e2]); g = np.concatenate([np.zeros(len(t1)), np.ones(len(t2))])
    O = Ex = V = 0.0
    for u in np.unique(t[e == 1]):
        at = t >= u; n = at.sum(); n1 = (at & (g == 0)).sum(); d = ((t == u) & (e == 1)).sum(); d1 = ((t == u) & (e == 1) & (g == 0)).sum()
        O += d1; Ex += d * n1 / n
        if n > 1:
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    z = (O - Ex) / np.sqrt(V) if V > 0 else 0.0
    return float(z), float(2 * stats.norm.sf(abs(z)))
