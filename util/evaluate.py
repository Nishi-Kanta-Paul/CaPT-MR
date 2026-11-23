import numpy as np
import pandas as pd

from .utils import log, save_json, load_json_safe, run_tag
from .datasets import patch_overlap
from .engine import Run, SETTINGS, PRIMARY_METHODS, CONFIRM_METHODS, BACKBONE_METHODS, DIAG_METHODS, experiment_runs
from .perturbations import FAMILIES, severity_levels
from .stats import marginal, paired, best_shared, seed_summary


SHARED_RULES = ('C2-max', 'C2-mean', 'C2-weighted')
STRONG = ('Bi-FRN', 'TDM', 'SFMR')
F_SETTINGS, F_SHARED, F_DIRECTIONAL, F_STRONG, F_BACKBONE, F_TWO, F_ROBUST = 6, 18, 12, 18, 6, 2, 3


def key(*parts):
    return '|'.join(str(p) for p in parts)


def load_records(args, ws):
    rows, preds = [], {}
    for run in experiment_runs(args):
        tag = run_tag(run.method, dict(run.variant))
        for pool in run.pools:
            for seed in args.seeds:
                path = ws.record(run, pool, seed)
                if not path.exists():
                    continue
                with np.load(path) as data:
                    rows.append({'backbone': run.backbone, 'dataset': run.dataset, 'shot': run.shot, 'method': tag, 'pool': pool, 'seed': seed,
                                 'acc': 100.0 * float(data['acc'].mean())})
                    if pool == 'test' and run.backbone == 'conv64' and not run.variant:
                        preds[(run.dataset, run.shot, run.method, seed)] = data['pred']
    return pd.DataFrame(rows), preds


class Accuracies:
    def __init__(self, frame):
        self.frame = frame
        self.index = {}
        for r in frame.itertuples():
            self.index.setdefault((r.backbone, r.dataset, r.shot, r.method, r.pool), {})[r.seed] = r.acc

    def get(self, method, dataset, shot, pool='test', backbone='conv64'):
        return self.index.get((backbone, dataset, shot, method, pool), {})


def contrast(acc, a, b, dataset, shot, families, pool='test', backbone='conv64'):
    x, y = acc.get(a, dataset, shot, pool, backbone), acc.get(b, dataset, shot, pool, backbone)
    return paired(x, y, families) if x and y else None


def main_tables(acc):
    out = {'main': {}, 'components': {}, 'shared': {}, 'directional': {}, 'strong': {}, 'c2_test': {}}
    for dataset, shot in SETTINGS:
        s = key(dataset, shot)
        for method in PRIMARY_METHODS:
            values = acc.get(method, dataset, shot)
            if values:
                out['main'][key(method, s)] = marginal(values)
        shared = {rule: contrast(acc, 'A3', rule, dataset, shot, F_SHARED) for rule in SHARED_RULES}
        out['shared'][s] = shared
        out['components'][s] = {'c2max_a2': contrast(acc, 'C2-max', 'A2', dataset, shot, F_SETTINGS),
                                'a3_best_shared': best_shared(list(shared.values())),
                                'mutual_query': contrast(acc, 'A3', 'A3-query', dataset, shot, F_DIRECTIONAL)}
        out['directional'][s] = {'mutual_query': contrast(acc, 'A3', 'A3-query', dataset, shot, F_DIRECTIONAL),
                                 'mutual_support': contrast(acc, 'A3', 'A3-support', dataset, shot, F_DIRECTIONAL)}
        out['strong'][s] = {ref: contrast(acc, 'A3', ref, dataset, shot, F_STRONG) for ref in STRONG}
        out['c2_test'][s] = {'c2_a2': contrast(acc, 'C2-max', 'A2', dataset, shot, F_SETTINGS),
                             'a3_c2': contrast(acc, 'A3', 'C2-max', dataset, shot, F_SETTINGS),
                             'a3_a2': contrast(acc, 'A3', 'A2', dataset, shot, F_SETTINGS)}
    return out


def confirm_table(acc):
    out = {}
    for method in CONFIRM_METHODS:
        values = acc.get(method, 'cub', 1, 'conf')
        if values:
            out[method] = {'marginal': marginal(values), 'versus': paired(acc.get('A3', 'cub', 1, 'conf'), values, 1) if method != 'A3' else None}
    return out


def backbone_tables(acc):
    out = {'marginal': {}, 'contrasts': {}, 'bifrn': {}, 'interaction': {}}
    for shot in (1, 5):
        for method in BACKBONE_METHODS:
            values = acc.get(method, 'cub', shot, backbone='resnet12')
            if values:
                out['marginal'][key(method, shot)] = marginal(values)
        out['contrasts'][str(shot)] = {
            'a3_c2mean': contrast(acc, 'A3', 'C2-mean', 'cub', shot, F_BACKBONE, backbone='resnet12'),
            'a3_sfmr': contrast(acc, 'A3', 'SFMR', 'cub', shot, F_BACKBONE, backbone='resnet12'),
            'mutual_query': contrast(acc, 'A3', 'A3-query', 'cub', shot, F_BACKBONE, backbone='resnet12')}
        out['bifrn'][str(shot)] = contrast(acc, 'A3', 'Bi-FRN', 'cub', shot, F_TWO, backbone='resnet12')
        gains = {}
        for backbone in ('conv64', 'resnet12'):
            a, b = acc.get('A3', 'cub', shot, backbone=backbone), acc.get('C2-mean', 'cub', shot, backbone=backbone)
            gains[backbone] = {s: a[s] - b[s] for s in set(a) & set(b)}
        if gains['conv64'] and gains['resnet12']:
            out['interaction'][str(shot)] = {'conv64_gain': float(np.mean(list(gains['conv64'].values()))),
                                             'resnet12_gain': float(np.mean(list(gains['resnet12'].values()))),
                                             'I': paired(gains['resnet12'], gains['conv64'], F_TWO)}
    return out


def transitions_table(preds, args):
    out = {}
    for dataset, shot in SETTINGS:
        labels = np.repeat(np.arange(args.n_way), args.test_queries)
        for ref in ('A2', 'C2-mean'):
            corrected, new, total = 0, 0, 0
            for seed in args.seeds:
                a, b = preds.get((dataset, shot, 'A3', seed)), preds.get((dataset, shot, ref, seed))
                if a is None or b is None:
                    continue
                ca, cb = a == labels[None], b == labels[None]
                corrected += int((ca & ~cb).sum())
                new += int((~ca & cb).sum())
                total += ca.size
            if total:
                out[key(dataset, shot, ref)] = {'corrected': 100.0 * corrected / total, 'new_errors': 100.0 * new / total,
                                                'net': 100.0 * (corrected - new) / total}
    return out


def resources(ws):
    path = ws.profile / 'timings.csv'
    if not path.exists():
        return {}
    timings = pd.read_csv(path)
    out = {'latency': {}, 'differences': {}, 'stages': {}, 'models': {}}
    for (backbone, shot, method), part in timings.groupby(['backbone', 'shot', 'method']):
        ms = part['ms']
        out['latency'][key(backbone, shot, method)] = {
            'median': float(ms.median()), 'q1': float(ms.quantile(0.25)), 'q3': float(ms.quantile(0.75)), 'p95': float(ms.quantile(0.95)),
            'observations': int(len(ms)), 'peak_mb': float(part['peak_mb'].median()), 'sweeps_median': float(part['sweeps_median'].median()),
            'sweeps_p95': float(part['sweeps_p95'].median()), 'sweeps_max': float(part['sweeps_max'].max()),
            'cap_percent': 100.0 * float(part['cap_rate'].mean())}
    index = ['backbone', 'shot', 'seed', 'pass', 'episode']
    for a, b in (('A3', 'C2-mean'), ('A3', 'Bi-FRN'), ('A3', 'SFMR')):
        merged = timings[timings['method'] == a].merge(timings[timings['method'] == b], on=index, suffixes=('_a', '_b'))
        for (backbone, shot), part in merged.groupby(['backbone', 'shot']):
            delta = part['ms_a'] - part['ms_b']
            out['differences'][key(backbone, shot, a, b)] = {'median': float(delta.median()), 'q1': float(delta.quantile(0.25)),
                                                             'q3': float(delta.quantile(0.75)), 'p95': float(delta.quantile(0.95))}
    stages_path = ws.profile / 'stages.csv'
    if stages_path.exists():
        stages = pd.read_csv(stages_path).fillna(0.0)
        for (backbone, shot, method), part in stages.groupby(['backbone', 'shot', 'method']):
            head = part[[c for c in ('costs', 'gate', 'solver', 'recon') if c in part]].sum(1)
            out['stages'][key(backbone, shot, method)] = {
                'encoder': float(part['encoder'].median()), 'costs_gate': float((part.get('costs', 0) + part.get('gate', 0)).median()),
                'solver': float(part['solver'].median()), 'recon': float(part['recon'].median()), 'head': float(head.median())}
    models_path = ws.profile / 'models.csv'
    if models_path.exists():
        for r in pd.read_csv(models_path).itertuples():
            out['models'][key(r.backbone, r.shot, r.method)] = {'params': int(r.params), 'train_ms': float(r.train_ms)}
    return out


def latency(res, backbone, shot, method, field='median'):
    entry = res.get('latency', {}).get(key(backbone, shot, method))
    return entry[field] if entry else float('nan')


def practical_tables(main, res, args):
    rows = []
    for margin in args.margin_grid:
        count_mean = count_lower = count_budget = 0
        for dataset, shot in SETTINGS:
            c = main['c2_test'][key(dataset, shot)]['a3_c2']
            if c is None:
                continue
            within = budget_ok(res, shot, ['C2-max'], args)
            count_mean += int(c['mean'] >= margin)
            count_lower += int(c['slo'] > margin)
            count_budget += int(c['slo'] > margin and within)
        rows.append({'margin': margin, 'mean_at_least': count_mean, 'lower_above': count_lower, 'within_budget': count_budget})
    region = {}
    for dataset, shot in SETTINGS:
        s = key(dataset, shot)
        best = main['components'][s]['a3_best_shared']
        useful = bool(best is not None and best['slo'] > args.practical_margin and budget_ok(res, shot, SHARED_RULES, args))
        region[s] = {'best_shared': best, 'useful': useful, 'mutual_query': main['components'][s]['mutual_query'],
                     'dominance': {ref: dominance(main, res, dataset, shot, ref) for ref in ('Bi-FRN', 'SFMR')}}
    return rows, region


def budget_ok(res, shot, rules, args):
    for rule in rules:
        a3, other = latency(res, 'conv64', shot, 'A3'), latency(res, 'conv64', shot, rule)
        mem_a3, mem_other = latency(res, 'conv64', shot, 'A3', 'peak_mb'), latency(res, 'conv64', shot, rule, 'peak_mb')
        if not np.isfinite(a3) or not np.isfinite(other):
            return False
        if 100.0 * (a3 - other) / other > args.latency_budget:
            return False
        if np.isfinite(mem_a3) and np.isfinite(mem_other) and 100.0 * (mem_a3 - mem_other) / mem_other > args.memory_budget:
            return False
    return True


def dominance(main, res, dataset, shot, ref):
    a, b = main['main'].get(key('A3', dataset, shot)), main['main'].get(key(ref, dataset, shot))
    if a is None or b is None:
        return None
    t_a, t_b = latency(res, 'conv64', shot, 'A3'), latency(res, 'conv64', shot, ref)
    m_a, m_b = latency(res, 'conv64', shot, 'A3', 'peak_mb'), latency(res, 'conv64', shot, ref, 'peak_mb')
    if not np.isfinite(t_a) or not np.isfinite(t_b):
        return None
    better = [a['mean'] >= b['mean'], t_a <= t_b, (m_a <= m_b) if np.isfinite(m_a) and np.isfinite(m_b) else True]
    worse = [a['mean'] <= b['mean'], t_a >= t_b, (m_a >= m_b) if np.isfinite(m_a) and np.isfinite(m_b) else True]
    strict_better = a['mean'] > b['mean'] or t_a < t_b or (np.isfinite(m_a) and np.isfinite(m_b) and m_a < m_b)
    strict_worse = a['mean'] < b['mean'] or t_a > t_b or (np.isfinite(m_a) and np.isfinite(m_b) and m_a > m_b)
    if all(better) and strict_better:
        return 'A3 dominates'
    if all(worse) and strict_worse:
        return 'Baseline dominates'
    return 'Trade-off'


def robustness(args, ws):
    clean_path = ws.perturb / 'clean.npz'
    if not clean_path.exists():
        return {}
    clean = dict(np.load(clean_path))
    levels = severity_levels(args)
    drops = {}
    for family in FAMILIES:
        per = {}
        for h in range(len(levels[family])):
            path = ws.perturb / f'{family}-{h + 1}.npz'
            if not path.exists():
                continue
            for name, values in np.load(path).items():
                per.setdefault(name, []).append(100.0 * (clean[name] - values).mean())
        for name, values in per.items():
            method, seed = name.split('__seed')
            drops.setdefault(method, {}).setdefault(family, {})[int(seed)] = float(np.mean(values))
    out = {'clean': {}, 'drops': {}, 'contrasts': {}}
    for name, values in clean.items():
        method, seed = name.split('__seed')
        out['clean'].setdefault(method, {})[int(seed)] = 100.0 * float(values.mean())
    out['clean'] = {m: seed_summary(v) for m, v in out['clean'].items()}
    for method, families in drops.items():
        entry = {f: seed_summary(v) for f, v in families.items()}
        seeds = set.intersection(*[set(v) for v in families.values()])
        entry['mean'] = seed_summary({s: np.mean([families[f][s] for f in families]) for s in seeds})
        entry['mean_no_background'] = seed_summary({s: np.mean([families[f][s] for f in families if f != 'background']) for s in seeds})
        out['drops'][method] = entry
    if 'background' in drops.get('A3', {}):
        bg = {m: drops[m]['background'] for m in ('A2', 'C2-max', 'A3') if 'background' in drops.get(m, {})}
        for a, b in (('A2', 'A3'), ('A2', 'C2-max'), ('C2-max', 'A3')):
            if a in bg and b in bg:
                out['contrasts'][f'{a}-{b}'] = paired(bg[a], bg[b], F_ROBUST)
    return out


def manyway(args, ws):
    out = {}
    for ways in args.ways:
        for method in args.followup_methods:
            accs, ms = {}, []
            for seed in args.seeds:
                path = ws.manyway / f'{method}_N{ways}_seed{seed}.npz'
                if path.exists():
                    with np.load(path) as data:
                        accs[seed] = 100.0 * float(data['acc'].mean())
                        ms.append(float(np.median(data['ms'])) / ways)
            if accs:
                out[key(method, ways)] = {**seed_summary(accs), 'ms_per_query': float(np.median(ms)), 'by_seed': accs}
        a, c = out.get(key('A3', ways)), out.get(key('C2-max', ways))
        if a and c:
            out[key('A3-C2', ways)] = paired(a['by_seed'], c['by_seed'], 1)
    return out


def coverage_and_kappa(args, ws, acc):
    out = {'coverage': {}, 'kappa_frozen': {}, 'kappa_retrained': {}}
    for rho in args.coverage_grid:
        for method in ('A2', 'A3'):
            tag = method if abs(rho - args.rho) < 1e-12 else run_tag(method, {'rho': rho})
            values = acc.get(tag, 'cub', 1, 'tune')
            if values:
                out['coverage'][key(method, rho)] = seed_summary(values)
    labels = np.repeat(np.arange(args.n_way), args.test_queries)
    for kappa in args.kappa_grid:
        values = {}
        for seed in args.seeds:
            path = ws.record(Run('conv64', 'cub', 1, 'A3', (), ()), 'tune', seed)
            if path.exists():
                with np.load(path) as data:
                    if 'residual' in data:
                        values[seed] = 100.0 * float(((data['residual'] + kappa * data['dispersion']).argmin(-1) == labels[None]).mean())
        if values:
            out['kappa_frozen'][str(kappa)] = seed_summary(values)
    for name, method, variant in (('A3_kappa', 'A3', {'kappa': args.kappa_retrain}), ('A2_kappa', 'A2', {'kappa': args.kappa_retrain}),
                                  ('A3_small_entropy', 'A3', {'entropy': args.small_entropy})):
        values = acc.get(run_tag(method, variant), 'cub', 1, 'tune')
        if values:
            out['kappa_retrained'][name] = {**seed_summary(values), 'variant': variant}
    return out


def transport_diagnostics(args, ws):
    path = ws.split_path('cub')
    if not path.exists():
        return {}
    boxes = np.load(ws.cache_paths('cub', args.image_size)[1])
    out, records = {}, {}
    for method in DIAG_METHODS:
        for seed in args.seeds:
            rec = ws.record(Run('conv64', 'cub', 1, method, (), ()), 'test', seed)
            if rec.exists():
                with np.load(rec) as data:
                    if 'row_mass' in data and data['row_mass'].shape[-1]:
                        records[(method, seed)] = {k: data[k] for k in ('row_mass', 'dest', 'margin', 'query')}
    for method in DIAG_METHODS:
        seeds = [s for s in args.seeds if (method, s) in records]
        if not seeds:
            continue
        box_mass, eff_rows, margins, rerouted, participating = [], [], [], 0, 0
        for seed in seeds:
            r = records[(method, seed)]
            rows = r['row_mass'].astype(np.float64)
            grid = int(round(np.sqrt(rows.shape[-1])))
            overlap = patch_overlap(boxes[r['query'].reshape(-1)], args.image_size, grid).reshape(rows.shape)
            total = rows.sum(-1)
            box_mass.append(float(np.nanmean((rows * overlap).sum(-1) / total)))
            p = rows / total[..., None]
            eff_rows.append(float(np.mean(np.exp(-(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)).sum(-1)))))
            margins.append(r['margin'].reshape(-1))
            ref = records.get(('A2', seed))
            if ref is not None and method not in ('A2', 'A1'):
                threshold = args.participation_threshold / rows.shape[-1]
                both = (rows >= threshold) & (ref['row_mass'].astype(np.float64) >= threshold)
                rerouted += int((both & (r['dest'] != ref['dest'])).sum())
                participating += int(both.sum())
        out[method] = {'box_mass': float(np.mean(box_mass)), 'effective_rows': float(np.mean(eff_rows)),
                       'rerouting': rerouted / participating if participating else float('nan'),
                       'margin_x100': 100.0 * float(np.median(np.concatenate(margins))), 'seeds': len(seeds)}
    return out


def stage_evaluate(args, ws):
    frame, preds = load_records(args, ws)
    frame.to_csv(ws.metrics / 'seed_accuracies.csv', index=False)
    acc = Accuracies(frame)
    main = main_tables(acc)
    res = resources(ws)
    practical, region = practical_tables(main, res, args)
    results = {'config': {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, list, bool, type(None)))},
               'computed': {'verify': load_json_safe(ws.verify / 'verify_summary.json'), 'probe': load_json_safe(ws.probe / 'probe_summary.json')},
               **main, 'confirm': confirm_table(acc), 'backbone': backbone_tables(acc), 'transitions': transitions_table(preds, args),
               'resources': res, 'practical': practical, 'operating_region': region, 'robust': robustness(args, ws),
               'manyway': manyway(args, ws), **coverage_and_kappa(args, ws, acc), 'diagnostics': transport_diagnostics(args, ws),
               'dataset_summary': load_json_safe(ws.splits / 'dataset_summary.json')}
    rows = []
    for dataset, shot in SETTINGS:
        c = main['c2_test'][key(dataset, shot)]['a3_c2']
        if c is not None:
            rows += [{'dataset': dataset, 'shot': shot, 'seed': s, 'a3_minus_c2max_pp': d} for s, d in zip(c['seeds'], c['deltas'])]
    pd.DataFrame(rows).to_csv(ws.metrics / 'a3_minus_c2max_seed_differences.csv', index=False)
    if res.get('latency'):
        pd.DataFrame([{'entry': k, **v} for k, v in res['latency'].items()]).to_csv(ws.metrics / 'latency_summary.csv', index=False)
    flat = []
    flatten('', results, flat)
    pd.DataFrame(flat).to_csv(ws.metrics / 'summary.csv', index=False)
    save_json(results, ws.metrics / 'results.json')
    log(f'evaluate: {len(frame)} seed-level accuracies, {len(flat)} statistics written to {ws.metrics}')
    return results


def flatten(prefix, node, rows):
    if isinstance(node, dict):
        if 'mean' in node:
            rows.append({'statistic': prefix, **{k: v for k, v in node.items() if not isinstance(v, (list, dict))}})
            return
        for k, v in node.items():
            flatten(f'{prefix}/{k}' if prefix else str(k), v, rows)
