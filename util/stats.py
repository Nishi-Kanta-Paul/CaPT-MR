import numpy as np
from scipy import stats


def t_critical(df, families=1, level=0.95):
    return float(stats.t.ppf(1.0 - (1.0 - level) / (2.0 * families), df))


def marginal(by_seed):
    values = np.asarray([v for v in by_seed.values() if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return {'mean': float('nan'), 'half': float('nan'), 'sd': float('nan'), 'n': 0}
    if values.size == 1:
        return {'mean': float(values[0]), 'half': float('nan'), 'sd': float('nan'), 'n': 1}
    sd = float(values.std(ddof=1))
    return {'mean': float(values.mean()), 'half': t_critical(values.size - 1) * sd / np.sqrt(values.size), 'sd': sd, 'n': int(values.size)}


def paired(a, b, families=1):
    seeds = sorted(set(a) & set(b))
    deltas = np.asarray([a[s] - b[s] for s in seeds], dtype=float)
    out = {'seeds': seeds, 'deltas': deltas.tolist(), 'n': int(deltas.size), 'families': families,
           'mean': float(deltas.mean()) if deltas.size else float('nan'), 'positive': int((deltas > 0).sum())}
    if deltas.size < 2:
        out.update({'lo': float('nan'), 'hi': float('nan'), 'slo': float('nan'), 'shi': float('nan')})
        return out
    se = deltas.std(ddof=1) / np.sqrt(deltas.size)
    ordinary, simultaneous = t_critical(deltas.size - 1) * se, t_critical(deltas.size - 1, families) * se
    out.update({'lo': out['mean'] - ordinary, 'hi': out['mean'] + ordinary, 'slo': out['mean'] - simultaneous, 'shi': out['mean'] + simultaneous})
    return out


def best_shared(contrasts):
    contrasts = [c for c in contrasts if c is not None and c['n']]
    if not contrasts:
        return None
    return {'mean': min(c['mean'] for c in contrasts), 'slo': min(c['slo'] for c in contrasts), 'shi': min(c['shi'] for c in contrasts),
            'n': min(c['n'] for c in contrasts), 'families': contrasts[0]['families']}


def seed_summary(by_seed):
    return {'mean': float(np.mean(list(by_seed.values()))) if by_seed else float('nan'),
            'sd': float(np.std(list(by_seed.values()), ddof=1)) if len(by_seed) > 1 else float('nan'), 'n': len(by_seed)}
