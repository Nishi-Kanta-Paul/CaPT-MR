import os
import time
import platform
import tracemalloc

import numpy as np
import pandas as pd

from .utils import log, save_json, stable_hash, quantiles
from .procedural import DescriptorGenerator, bank_costs
from .reference_ot import solve, reconstruction_terms, rival_costs, row_bias, pool_margins, residuals, gate_config, KERNELS, SHARED_GATES


EASY_EPISODES = 40
HARD_CALIBRATION_EPISODES = 10
HARD_EPISODES = 20
AMBIGUITY_CALIBRATION_EPISODES = 10
HARD_METHODS = ('A2', 'A3', 'C2-max', 'C2-mean', 'C2-weighted', 'Anchored')
AMBIGUITY_METHODS = ('A2', 'C2-max', 'C2-mean', 'C2-weighted', 'A3')
METHOD_GATES = {'A2': 'none', 'A3': 'candidate', 'C2-max': 'shared-max', 'C2-mean': 'shared-mean', 'C2-weighted': 'shared-weighted',
                'Anchored': 'anchored'}
TRANSITIONS = (('A3', 'A2'), ('C2-max', 'A2'), ('C2-mean', 'A2'), ('C2-weighted', 'A2'), ('Anchored', 'A2'), ('A3', 'C2-max'),
               ('A3', 'C2-mean'), ('A3', 'C2-weighted'))
EASY_SOLVER = {'tol': 1e-8, 'feas_tol': 1e-8, 'max_iter': 40000}
PROBE_SOLVER = {'tol': 1e-6, 'feas_tol': 1e-7, 'max_iter': 500, 'fallback_iter': 5000}
REFERENCE_SOLVER = {'tol': 1e-8, 'feas_tol': 1e-9, 'max_iter': 40000}
KERNEL_PASSES = 200
KERNEL_TRACES = 5


def calibrate_offsets(args, seed):
    bank = DescriptorGenerator(seed, args.hard_separation, args.hard_noise).bank(HARD_CALIBRATION_EPISODES)
    costs, _ = bank_costs(bank)
    d_pos, d_neg = rival_costs(costs, args.t_m)
    margin = d_neg - d_pos
    return {kind: float(np.median(pool_margins(margin, kind, args.weight_temp))) for kind in SHARED_GATES}


def solve_bank(costs, support, query, bias, args, solver, kind='mutual', keep_plans=False):
    q, n, k, l, m = costs.shape
    flat_bias = np.broadcast_to(bias[:, :, None, :], (q, n, k, l)).reshape(-1, l)
    plans, info = solve(costs.reshape(-1, l, m), args.rho, args.entropy, bias=flat_bias, **solver)
    vq = np.broadcast_to(query[:, None, None], (q, n, k, l, query.shape[-1])).reshape(-1, l, query.shape[-1])
    vs = support.reshape(-1, m, support.shape[-1])
    distance = residuals(plans, vq, vs, args.rho, kind).reshape(q, n, k).mean(-1)
    return distance, info, (plans, vq, vs) if keep_plans else None


def bank_biases(costs, args, cfg, methods):
    d_pos, d_neg = rival_costs(costs, args.t_m)
    return {method: row_bias(METHOD_GATES.get(method, method), d_pos, d_neg, cfg) for method in methods}


def query_records(distance, labels, episode, method):
    q = len(labels)
    true = distance[np.arange(q), labels]
    wrong = distance.copy()
    wrong[np.arange(q), labels] = np.inf
    best_wrong = wrong.min(1)
    pred = distance.argmin(1)
    return pd.DataFrame({'method': method, 'query': np.arange(q), 'episode': episode, 'label': labels, 'pred': pred,
                         'true_residual': true, 'best_wrong_residual': best_wrong, 'margin': best_wrong - true,
                         'correct': (pred == labels).astype(int)})


def summarize_rule(frame):
    margin = frame['margin'].to_numpy()
    return {'accuracy': 100.0 * frame['correct'].mean(), 'errors': int((1 - frame['correct']).sum()), 'queries': len(frame),
            'median_margin': float(np.median(margin)), 'margin_q05': float(np.quantile(margin, 0.05)), 'margin_q95': float(np.quantile(margin, 0.95))}


def easy_probe(args):
    bank = DescriptorGenerator(args.probe_seed, args.easy_separation, args.easy_noise).bank(EASY_EPISODES)
    costs, support = bank_costs(bank)
    biases = bank_biases(costs, args, gate_config(args), ('A2', 'A3'))
    out = {}
    for method in ('A2', 'A3'):
        distance, info, cache = solve_bank(costs, support, bank['query'], biases[method], args, EASY_SOLVER, keep_plans=True)
        out[method] = (distance, info, cache[0])
    q, n = costs.shape[:2]
    true = np.arange(n)[None] == bank['labels'][:, None]
    lowered = out['A3'][0] < out['A2'][0]
    rows_a2, rows_a3 = out['A2'][2].sum(2).reshape(q, n, -1), out['A3'][2].sum(2).reshape(q, n, -1)
    realloc = 0.5 * np.abs(rows_a3 - rows_a2).sum(-1) / args.rho
    sweeps = np.concatenate([out[m][1]['sweeps'] for m in ('A2', 'A3')])
    return {
        'episodes': EASY_EPISODES, 'queries': int(q), 'candidate_problems': int(q * n), 'solves': int(2 * q * n),
        'separation': args.easy_separation, 'noise': args.easy_noise, 'seed': args.probe_seed,
        'wrong_lowered_fraction': float(lowered[~true].mean()), 'true_lowered_fraction': float(lowered[true].mean()),
        'wrong_median_reallocated': float(np.median(realloc[~true])),
        'accuracy_A2': float(100.0 * (out['A2'][0].argmin(1) == bank['labels']).mean()),
        'accuracy_A3': float(100.0 * (out['A3'][0].argmin(1) == bank['labels']).mean()),
        'max_sweeps': int(sweeps.max()), 'cap_hits': int(sum(out[m][1]['cap_hit'].sum() for m in ('A2', 'A3'))),
    }


def hard_bank(args, tag, n_episodes):
    return DescriptorGenerator(stable_hash('hard', tag, args.probe_seed), args.hard_separation, args.hard_noise).bank(n_episodes)


def run_rules(bank, args, cfg, methods, keep=()):
    costs, support = bank_costs(bank)
    biases = bank_biases(costs, args, cfg, methods)
    frames, infos, cache = [], {}, {}
    for method in methods:
        distance, info, plans = solve_bank(costs, support, bank['query'], biases[method], args, PROBE_SOLVER, keep_plans=method in keep)
        frames.append(query_records(distance, bank['labels'], bank['episode'], method))
        infos[method] = info
        if plans is not None:
            cache[method] = plans
    return pd.concat(frames, ignore_index=True), infos, cache, biases


def anchored_bound(args, bank, biases):
    costs, _ = bank_costs(bank)
    q, n, k, l, m = costs.shape
    u0 = biases['C2-max'].reshape(-1, l)
    u1 = biases['A3'].reshape(-1, l)
    alpha = args.anchor_alpha
    t0, _ = solve(costs[:, :, 0].reshape(-1, l, m), args.rho, args.entropy, bias=u0, **PROBE_SOLVER)
    ta, _ = solve(costs[:, :, 0].reshape(-1, l, m), args.rho, args.entropy, bias=(1 - alpha) * u0 + alpha * u1, **PROBE_SOLVER)
    osc = np.ptp(u1 - u0, axis=1)
    change = np.abs(ta - t0).sum((1, 2))
    linear = alpha * args.rho / (2 * args.entropy) * osc
    coverage = np.sqrt(args.rho * min(args.rho, 1 - args.rho) * alpha * osc / args.entropy)
    return pd.DataFrame({'pair': np.arange(len(change)), 'l1_change': change, 'linear_bound': linear, 'coverage_bound': coverage,
                         'excess': change - np.minimum(linear, coverage)})


def transitions(frame):
    rows = []
    by = {m: part.sort_values('query').reset_index(drop=True) for m, part in frame.groupby('method')}
    for rule, ref in TRANSITIONS:
        a, b = by[rule], by[ref]
        ref_wrong = b['correct'] == 0
        delta = a['margin'] - b['margin']
        rows.append({'rule': rule, 'reference': ref, 'corrected': int(((a['correct'] == 1) & ref_wrong).sum()), 'reference_errors': int(ref_wrong.sum()),
                     'new_errors': int(((a['correct'] == 0) & ~ref_wrong).sum()), 'reference_correct': int((~ref_wrong).sum()),
                     'margin_rises': int((delta > 0).sum()), 'queries': len(a), 'median_margin_change': float(delta.median())})
    return pd.DataFrame(rows)


def directions(args, bank, cache, saved):
    plans, vq, vs = cache['A3']
    q, n = len(bank['labels']), bank['support'].shape[1]
    rows, frames = [], []
    for kind in ('query', 'support', 'mutual'):
        distance = KERNELS[kind](plans, vq, vs, args.rho).reshape(q, n, -1).mean(-1)
        frame = query_records(distance, bank['labels'], bank['episode'], kind)
        frames.append(frame)
        rows.append({'score': kind, **summarize_rule(frame)})
    mutual = frames[-1].sort_values('query')['true_residual'].to_numpy()
    reference = saved.sort_values('query')['true_residual'].to_numpy()
    return pd.DataFrame(rows), float(np.abs(mutual - reference).max())


def ambiguity(args, offsets):
    cfg = gate_config(args, offsets)
    calibration = hard_bank(args, 'ambiguity-calibration', AMBIGUITY_CALIBRATION_EPISODES)
    cal_gap = gaps(calibration, args, cfg)
    lower, upper = float(np.quantile(cal_gap, 0.25)), float(np.quantile(cal_gap, 0.75))
    banks = {'Existing': hard_bank(args, 'evaluation', HARD_EPISODES), 'New': hard_bank(args, 'ambiguity-new', HARD_EPISODES)}
    accuracy_rows, effect_rows, query_frames = [], [], []
    alpha = 0.05 / (2 * 30)
    for name, bank in banks.items():
        frame, _, _, _ = run_rules(bank, args, cfg, AMBIGUITY_METHODS)
        gap = gaps(bank, args, cfg)
        band = np.where(gap <= lower, 'Low gap', np.where(gap <= upper, 'Middle gap', 'High gap'))
        frame['band'] = band[frame['query'].to_numpy()]
        frame['bank'] = name
        query_frames.append(frame)
        episodes = np.unique(bank['episode'])
        rng = np.random.default_rng(stable_hash('ambiguity-bootstrap', name, args.probe_seed))
        weights = rng.multinomial(len(episodes), np.full(len(episodes), 1.0 / len(episodes)), size=args.n_boot).astype(float)
        boot = {}
        for b in ('Low gap', 'Middle gap', 'High gap', 'All'):
            for method in AMBIGUITY_METHODS:
                part = frame[frame['method'] == method]
                sel = part if b == 'All' else part[part['band'] == b]
                accuracy_rows.append({'bank': name, 'band': b, 'queries': len(sel), 'method': method, 'accuracy': 100.0 * sel['correct'].mean()})
                correct = np.bincount(sel['episode'], weights=sel['correct'], minlength=len(episodes))
                count = np.bincount(sel['episode'], minlength=len(episodes)).astype(float)
                with np.errstate(invalid='ignore', divide='ignore'):
                    boot[(b, method)] = (100.0 * (weights @ correct) / (weights @ count), 100.0 * correct.sum() / max(count.sum(), 1))
        for rule in ('C2-max', 'C2-mean', 'C2-weighted'):
            for b in ('Low gap', 'Middle gap', 'High gap', 'All'):
                diff = boot[(b, 'A3')][0] - boot[(b, rule)][0]
                point = boot[(b, 'A3')][1] - boot[(b, rule)][1]
                effect_rows.append(effect_row(name, b, rule, point, diff, alpha))
            low = boot[('Low gap', 'A3')][0] - boot[('Low gap', rule)][0]
            high = boot[('High gap', 'A3')][0] - boot[('High gap', rule)][0]
            point = (boot[('Low gap', 'A3')][1] - boot[('Low gap', rule)][1]) - (boot[('High gap', 'A3')][1] - boot[('High gap', rule)][1])
            effect_rows.append(effect_row(name, 'Low minus high', rule, point, low - high, alpha))
    meta = {'lower_boundary': lower, 'upper_boundary': upper, 'calibration_queries': int(len(cal_gap)), 'n_boot': args.n_boot,
            'contrasts': len(effect_rows), 'offsets': offsets}
    return pd.DataFrame(accuracy_rows), pd.DataFrame(effect_rows), pd.concat(query_frames, ignore_index=True), meta


def effect_row(bank, band, rule, point, replicates, alpha):
    replicates = replicates[np.isfinite(replicates)]
    lo, hi = (np.quantile(replicates, alpha), np.quantile(replicates, 1 - alpha)) if replicates.size else (np.nan, np.nan)
    return {'bank': bank, 'contrast': band, 'rule': rule, 'difference': point, 'lo': lo, 'hi': hi, 'valid_resamples': int(replicates.size)}


def gaps(bank, args, cfg):
    costs, support = bank_costs(bank)
    distance, _, _ = solve_bank(costs, support, bank['query'], np.zeros(costs.shape[:2] + costs.shape[-2:-1]), args, PROBE_SOLVER)
    ordered = np.sort(distance, axis=1)
    return ordered[:, 1] - ordered[:, 0]


def convergence(args):
    full = DescriptorGenerator(stable_hash('hard', 'convergence', args.probe_seed), args.hard_separation, args.hard_noise).bank(1, k_shot=5)
    rows, cache = [], {}
    cfg = gate_config(args)
    for shots in (1, 5):
        bank = dict(full, support=full['support'][:, :, :shots])
        costs, support = bank_costs(bank)
        biases = bank_biases(costs, args, cfg, ('A2', 'C2-mean', 'A3'))
        q, n, k, l, m = costs.shape
        for method in ('A2', 'C2-mean', 'A3'):
            flat_bias = np.broadcast_to(biases[method][:, :, None, :], (q, n, k, l)).reshape(-1, l)
            plans, info = solve(costs.reshape(-1, l, m), args.rho, args.entropy, bias=flat_bias, **PROBE_SOLVER)
            reference, _ = solve(costs.reshape(-1, l, m), args.rho, args.entropy, bias=flat_bias, **REFERENCE_SOLVER)
            sweeps = info['sweeps']
            rows.append({'method': method, 'setting': f'{shots}-shot', 'solves': len(sweeps), 'sweeps_median': float(np.median(sweeps)),
                         'sweeps_p95': float(np.quantile(sweeps, 0.95)), 'sweeps_max': int(sweeps.max()), 'cap_hits': int(info['cap_hit'].sum()),
                         'max_feasibility': float(info['feasibility'].max()),
                         'max_reference_l1': float((np.abs(plans - reference).sum((1, 2)) / args.rho).max())})
            if method == 'A3':
                vq = np.broadcast_to(bank['query'][:, None, None], (q, n, k, l, bank['query'].shape[-1])).reshape(-1, l, bank['query'].shape[-1])
                cache[f'{shots}-shot'] = (plans, np.ascontiguousarray(vq), np.ascontiguousarray(support.reshape(-1, m, support.shape[-1])))
    return pd.DataFrame(rows), cache


def kernel_timing(args, cache):
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
    except ImportError:
        limiter = None
    rows, agreement = [], 0.0
    for setting, (plans, vq, vs) in cache.items():
        reference = reconstruction_terms(plans, vq, vs, args.rho)
        times = {kind: [] for kind in KERNELS}
        order = list(KERNELS)
        for p in range(KERNEL_PASSES):
            for kind in order[p % 3:] + order[:p % 3]:
                start = time.perf_counter()
                KERNELS[kind](plans, vq, vs, args.rho)
                times[kind].append(1000.0 * (time.perf_counter() - start))
        for kind, kernel in KERNELS.items():
            value = kernel(plans, vq, vs, args.rho)
            target = {'query': reference['Rq'], 'support': reference['Rs'], 'mutual': reference['R']}[kind]
            agreement = max(agreement, float(np.abs(value - target).max()))
            peaks = []
            for _ in range(KERNEL_TRACES):
                tracemalloc.start()
                kernel(plans, vq, vs, args.rho)
                peaks.append(tracemalloc.get_traced_memory()[1] / 1024.0)
                tracemalloc.stop()
            q = quantiles(times[kind], (0.25, 0.5, 0.75, 0.95))
            rows.append({'setting': setting, 'scorer': kind, 'couplings': len(plans), 'median_ms': q['q50'], 'q1_ms': q['q25'],
                         'q3_ms': q['q75'], 'p95_ms': q['q95'], 'traced_kib': float(np.median(peaks))})
    if limiter is not None:
        limiter.unregister()
    meta = {'processor': platform.processor() or platform.machine(), 'platform': platform.platform(), 'python': platform.python_version(),
            'numpy': np.__version__, 'cpu_count': os.cpu_count(), 'blas_threads': 1 if limiter is not None else None,
            'passes': KERNEL_PASSES, 'max_agreement_error': agreement}
    return pd.DataFrame(rows), meta


def stage_probe(args, ws):
    out = ws.probe
    if (out / 'probe_summary.json').exists() and not args.overwrite:
        log('probe: outputs present')
        return
    log('probe: easy fixed-feature probe')
    easy = easy_probe(args)
    log('probe: harder fixed-feature probe')
    offsets = calibrate_offsets(args, stable_hash('hard', 'calibration', args.probe_seed))
    cfg = gate_config(args, offsets)
    bank = hard_bank(args, 'evaluation', HARD_EPISODES)
    frame, infos, cache, biases = run_rules(bank, args, cfg, HARD_METHODS, keep=('A3',))
    frame.to_csv(out / 'harder_queries.csv', index=False)
    summary = pd.DataFrame([{'rule': m, **summarize_rule(frame[frame['method'] == m])} for m in HARD_METHODS])
    summary.to_csv(out / 'harder_summary.csv', index=False)
    sweeps = np.concatenate([infos[m]['sweeps'] for m in HARD_METHODS])
    fallback = int(sum(infos[m]['fallback'].sum() for m in HARD_METHODS))
    bound = anchored_bound(args, bank, biases)
    bound.to_csv(out / 'anchored_bound.csv', index=False)
    transitions(frame).to_csv(out / 'transitions.csv', index=False)
    direction_table, direction_error = directions(args, bank, cache, frame[frame['method'] == 'A3'])
    direction_table.to_csv(out / 'directions.csv', index=False)
    log('probe: calibration-defined ambiguity subgroups')
    amb_acc, amb_eff, amb_queries, amb_meta = ambiguity(args, offsets)
    amb_acc.to_csv(out / 'ambiguity_accuracy.csv', index=False)
    amb_eff.to_csv(out / 'ambiguity_effects.csv', index=False)
    amb_queries.to_csv(out / 'ambiguity_queries.csv', index=False)
    log('probe: matched convergence sample and residual-kernel timing')
    conv, kernel_cache = convergence(args)
    conv.to_csv(out / 'convergence.csv', index=False)
    kernels, kernel_meta = kernel_timing(args, kernel_cache)
    kernels.to_csv(out / 'directional_kernel.csv', index=False)
    save_json({'easy': easy, 'harder': {'separation': args.hard_separation, 'noise': args.hard_noise, 'episodes': HARD_EPISODES,
                                        'queries': int(len(bank['labels'])), 'solves': int(len(sweeps)), 'max_sweeps': int(sweeps.max()),
                                        'fallback_solves': fallback, 'offsets': offsets,
                                        'anchored_bound_max_excess': float(bound['excess'].max()), 'anchored_pairs': len(bound),
                                        'directions_regenerated_max_error': direction_error},
               'ambiguity': amb_meta, 'kernel': kernel_meta}, out / 'probe_summary.json')
    log(f'probe: harder accuracies ' + ', '.join(f'{r.rule} {r.accuracy:.1f}%' for r in summary.itertuples()))
