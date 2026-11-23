import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .utils import log, save_json, stable_hash
from .procedural import DescriptorGenerator, bank_costs, unit
from .reference_ot import solve, solve_lp, reconstruction_terms, rival_costs, row_bias, gate_values, gate_config, softmin, uniform


IDENTITY_INSTANCES = 200
FULL_MASS_PROBLEMS = 100
SENSITIVITY_PAIRS = 150
SENSITIVITY_RHOS = (0.2, 0.6, 0.95, 0.999, 1.0)
SENSITIVITY_GRIDS = ((25, 25), (7, 9))
SENSITIVITY_EPS = (0.05, 0.1)
ALLOCATION_EPISODES = 60
TIED_RHOS = (0.5, 1.0)
TIED_EPS = (0.1, 0.01, 0.001)
REFERENCE = {'tol': 1e-8, 'feas_tol': 1e-9, 'max_iter': 40000}
STRICT = {'tol': 1e-14, 'feas_tol': 1e-14, 'max_iter': 40000}


def candidate_problems(args, tag, n_episodes, kind='candidate'):
    bank = DescriptorGenerator(stable_hash('verify', tag, args.probe_seed), args.easy_separation, args.easy_noise).bank(n_episodes)
    costs, support = bank_costs(bank)
    d_pos, d_neg = rival_costs(costs, args.t_m)
    cfg = gate_config(args)
    gate = gate_values('candidate', d_pos, d_neg, cfg)
    bias = row_bias(kind, d_pos, d_neg, cfg)
    q, n = costs.shape[:2]
    return {
        'costs': costs[:, :, 0].reshape(q * n, *costs.shape[-2:]), 'bias': bias.reshape(q * n, -1), 'gate': gate.reshape(q * n, -1),
        'vq': np.repeat(bank['query'], n, axis=0), 'vs': support[:, :, 0].reshape(q * n, *support.shape[-2:]),
        'true': (np.arange(n)[None] == bank['labels'][:, None]).reshape(-1),
        'object': np.repeat(bank['query_object'], n, axis=0), 'query': np.repeat(np.arange(q), n), 'candidate': np.tile(np.arange(n), q),
    }


def identity_checks(args):
    problems = candidate_problems(args, 'identity', -(-IDENTITY_INSTANCES // 25))
    rng = np.random.default_rng(stable_hash('identity-plans', args.probe_seed))
    rows = []
    for i in range(IDENTITY_INSTANCES):
        kind = ('arbitrary', 'entropic', 'exact')[i % 3]
        cost, vq, vs = problems['costs'][i], problems['vq'][i], problems['vs'][i]
        biased = cost + problems['bias'][i][:, None]
        if kind == 'arbitrary':
            plan = rng.exponential(size=cost.shape)
            plan[rng.random(cost.shape[0]) < 0.2] = 0.0
            plan *= args.rho / plan.sum()
        elif kind == 'entropic':
            plan, _ = solve(biased, args.rho, args.entropy, **REFERENCE)
        else:
            plan = solve_lp(biased, args.rho)
        mass = plan.sum()
        t = reconstruction_terms(plan, vq, vs, mass)
        rows.append({'instance': i, 'kind': kind, 'mass': mass,
                     'err_query': abs(t['P'][0] - t['Rq'][0] - t['Vq'][0]), 'err_support': abs(t['P'][0] - t['Rs'][0] - t['Vs'][0]),
                     'err_mutual': abs(t['P'][0] - t['R'][0] - t['V'][0]), 'err_cost': abs(t['P'][0] - 2.0 * (plan * cost).sum() / mass),
                     'P': t['P'][0], 'R': t['R'][0], 'V': t['V'][0], 'min_V': min(t['Vq'][0], t['Vs'][0], t['V'][0])})
    return pd.DataFrame(rows)


def full_mass_checks(args):
    problems = candidate_problems(args, 'full-mass', FULL_MASS_PROBLEMS // 25)
    plain, _ = solve(problems['costs'], 1.0, args.entropy, **STRICT)
    gated, _ = solve(problems['costs'], 1.0, args.entropy, bias=problems['bias'], **STRICT)
    return pd.DataFrame({'problem': np.arange(len(plain)), 'max_abs_diff': np.abs(plain - gated).max((1, 2))})


def capacities(rng, n, nonuniform):
    if not nonuniform:
        return uniform(n)
    w = rng.uniform(0.5, 1.5, n)
    return w / w.sum()


def sensitivity_checks(args):
    rng = np.random.default_rng(stable_hash('sensitivity', args.probe_seed))
    specs = []
    for p in range(SENSITIVITY_PAIRS):
        l, m = SENSITIVITY_GRIDS[(p // 5) % 2]
        spec = {'pair': p, 'rho': SENSITIVITY_RHOS[p % 5], 'rows': l, 'cols': m, 'nonuniform': (p // 10) % 2 == 1,
                'eps': SENSITIVITY_EPS[(p // 20) % 2]}
        x, y = unit(rng.standard_normal((l, 64))), unit(rng.standard_normal((m, 64)))
        spec.update({'cost': 1.0 - x @ y.T, 'a': capacities(rng, l, spec['nonuniform']), 'b': capacities(rng, m, spec['nonuniform']),
                     'u': rng.uniform(0, args.lambda_g, l), 'v': rng.uniform(0, args.lambda_g, l), 'shift': rng.uniform(-1, 1)})
        specs.append(spec)
    rows = []
    groups = {}
    for spec in specs:
        groups.setdefault((spec['rows'], spec['cols'], spec['rho'], spec['eps']), []).append(spec)
    for (l, m, rho, eps), group in groups.items():
        cost = np.stack([s['cost'] for s in group])
        a, b = np.stack([s['a'] for s in group]), np.stack([s['b'] for s in group])
        u, v = np.stack([s['u'] for s in group]), np.stack([s['v'] for s in group])
        shift = np.array([s['shift'] for s in group])[:, None]
        tu, iu = solve(cost, rho, eps, a=a, b=b, bias=u, **STRICT)
        tv, iv = solve(cost, rho, eps, a=a, b=b, bias=v, **STRICT)
        tc, ic = solve(cost, rho, eps, a=a, b=b, bias=u + shift, **STRICT)
        for i, s in enumerate(group):
            ru, rv = tu[i].sum(1), tv[i].sum(1)
            l1 = np.abs(tu[i] - tv[i]).sum()
            jeffreys = ((tu[i] - tv[i]) * np.log(tu[i] / tv[i])).sum()
            inner = (s['u'] - s['v']) @ (rv - ru)
            osc = np.ptp(s['u'] - s['v'])
            bound = min(rho, 1.0 - rho) * osc
            rows.append({'pair': s['pair'], 'rho': rho, 'eps': eps, 'grid': f'{l}x{m}', 'nonuniform': s['nonuniform'],
                         'l1': l1, 'jeffreys': jeffreys, 'inner': inner, 'bound': bound,
                         'excess_1': eps / rho * l1 ** 2 - eps * jeffreys, 'excess_2': eps * jeffreys - inner, 'excess_3': inner - bound,
                         'coverage_bound': np.sqrt(rho * min(rho, 1.0 - rho) * osc / eps), 'constant_shift_diff': np.abs(tc[i] - tu[i]).max(),
                         'sweeps_max': max(iu['sweeps'][i], iv['sweeps'][i], ic['sweeps'][i]),
                         'cap_hits': int(iu['cap_hit'][i]) + int(iv['cap_hit'][i]) + int(ic['cap_hit'][i])})
    return pd.DataFrame(rows).sort_values('pair').reset_index(drop=True)


def tied_cost_checks():
    rows = []
    e1, e2 = np.eye(64)[0], np.eye(64)[1]
    vq, vs = np.stack([e1, -e1]), np.stack([e2, -e2])
    cost = 1.0 - vq @ vs.T
    for rho in TIED_RHOS:
        for eps in TIED_EPS:
            plan, info = solve(cost, rho, eps, tol=1e-14, feas_tol=1e-14, max_iter=40000)
            t = reconstruction_terms(plan, vq, vs, rho)
            rows.append({'rho': rho, 'eps': eps, 'max_entry_error': np.abs(plan - rho / 4).max(), 'P': t['P'][0], 'R': t['R'][0],
                         'V': t['V'][0], 'sweeps': int(info['sweeps'])})
    return pd.DataFrame(rows)


def pool_composition_checks(args):
    rng = np.random.default_rng(stable_hash('pool', args.probe_seed))
    pool = rng.uniform(0.3, 0.9, 20)
    base = softmin(pool, args.t_m)
    replicated = softmin(np.concatenate([pool, pool]), args.t_m)
    easier, harder = rng.uniform(0.0, 0.3, 10), rng.uniform(0.9, 1.5, 10)
    out = {'softmin': base, 'replicated': replicated, 'replication_error': abs(replicated - base)}
    for name, extra in (('easier', easier), ('harder', harder)):
        union = softmin(np.concatenate([pool, extra]), args.t_m)
        d_new = softmin(extra, args.t_m)
        formula = -args.t_m * np.log((len(pool) * np.exp(-base / args.t_m) + len(extra) * np.exp(-d_new / args.t_m)) / (len(pool) + len(extra)))
        out[f'{name}_union'] = union
        out[f'{name}_formula_error'] = abs(union - formula)
        out[f'{name}_between'] = bool(min(base, d_new) - 1e-12 <= union <= max(base, d_new) + 1e-12)
    out['easier_decreases'] = bool(out['easier_union'] < base)
    out['harder_increases'] = bool(out['harder_union'] > base)
    return out


def allocation_checks(args):
    problems = candidate_problems(args, 'allocation', ALLOCATION_EPISODES)
    plain, info_plain = solve(problems['costs'], args.rho, args.entropy, **REFERENCE)
    gated, info_gated = solve(problems['costs'], args.rho, args.entropy, bias=problems['bias'], **REFERENCE)
    threshold = args.participation_threshold / problems['costs'].shape[1]
    rows = []
    for i in range(len(plain)):
        r0, r1 = plain[i].sum(1), gated[i].sum(1)
        change = np.abs(r1 - r0).sum() / args.rho
        both = (r0 >= threshold) & (r1 >= threshold)
        moved = plain[i].argmax(1) != gated[i].argmax(1)
        rho_s = spearmanr(problems['gate'][i], r1 - r0).correlation if np.ptp(r1 - r0) > 0 else np.nan
        obj = problems['object'][i]
        rows.append({'problem': i, 'query': problems['query'][i], 'candidate': problems['candidate'][i], 'true': bool(problems['true'][i]),
                     'row_change': change, 'reallocated': 0.5 * change, 'spearman': rho_s, 'participating': int(both.sum()),
                     'rerouted': int((both & moved).sum()), 'object_share_plain': r0[obj].sum() / args.rho,
                     'object_share_gated': r1[obj].sum() / args.rho,
                     'sweeps_plain': int(info_plain['sweeps'][i]), 'sweeps_gated': int(info_gated['sweeps'][i])})
    return pd.DataFrame(rows), problems, plain, gated


def stopping_checks(args, problems, plain_ref, gated_ref):
    bench = {'tol': args.ot_tol, 'max_iter': args.ot_max_iter, 'patience': args.ot_patience}
    plan_only, info_plan = solve(problems['costs'], args.rho, args.entropy, bias=problems['bias'], feas_tol=np.inf, stop='plan', **bench)
    plain, info_plain = solve(problems['costs'], args.rho, args.entropy, feas_tol=args.ot_feas_tol, **bench)
    gated, info_gated = solve(problems['costs'], args.rho, args.entropy, bias=problems['bias'], feas_tol=args.ot_feas_tol, **bench)
    return pd.DataFrame({
        'problem': np.arange(len(plain)), 'plan_only_feasibility': info_plan['feasibility'], 'plan_only_sweeps': info_plan['sweeps'],
        'plain_sweeps': info_plain['sweeps'], 'plain_cap_hit': info_plain['cap_hit'], 'plain_feasibility': info_plain['feasibility'],
        'gated_sweeps': info_gated['sweeps'], 'gated_cap_hit': info_gated['cap_hit'], 'gated_feasibility': info_gated['feasibility'],
        'plain_reference_l1': np.abs(plain - plain_ref).sum((1, 2)) / args.rho,
        'gated_reference_l1': np.abs(gated - gated_ref).sum((1, 2)) / args.rho,
    })


def toy_pair(args, allocation, problems, plain, gated):
    true = allocation[allocation['true']].copy()
    true['gain'] = true['object_share_gated'] - true['object_share_plain']
    i = int(true.iloc[(true['gain'] - true['gain'].median()).abs().argsort().iloc[0]]['problem'])
    full_plain, _ = solve(problems['costs'][i], 1.0, args.entropy, **STRICT)
    full_gated, _ = solve(problems['costs'][i], 1.0, args.entropy, bias=problems['bias'][i], **STRICT)
    return {'problem': i, 'gate': problems['gate'][i], 'object': problems['object'][i], 'plan_plain': plain[i], 'plan_gated': gated[i],
            'full_plain': full_plain, 'full_gated': full_gated, 'rho': args.rho}


def stage_verify(args, ws):
    out = ws.verify
    if (out / 'verify_summary.json').exists() and not args.overwrite:
        log('verify: outputs present')
        return
    log('verify: residual-dispersion identity')
    identity = identity_checks(args)
    identity.to_csv(out / 'identity_checks.csv', index=False)
    log('verify: full-mass neutrality')
    full = full_mass_checks(args)
    full.to_csv(out / 'full_mass_neutrality.csv', index=False)
    log('verify: coverage-dependent sensitivity chain')
    sens = sensitivity_checks(args)
    sens.to_csv(out / 'sensitivity_checks.csv', index=False)
    tied = tied_cost_checks()
    tied.to_csv(out / 'tied_cost.csv', index=False)
    pools = pool_composition_checks(args)
    log('verify: allocation behaviour and stopping rule')
    allocation, problems, plain, gated = allocation_checks(args)
    allocation.to_csv(out / 'allocation.csv', index=False)
    stopping = stopping_checks(args, problems, plain, gated)
    stopping.to_csv(out / 'stopping_rule.csv', index=False)
    toy = toy_pair(args, allocation, problems, plain, gated)
    np.savez(out / 'toy_pair.npz', **toy)
    true = allocation[allocation['true']]
    exact = identity[identity['kind'] == 'exact']
    full_rows = sens[sens['rho'] >= 1.0]
    summary = {
        'config': {'rho': args.rho, 'entropy': args.entropy, 'lambda_g': args.lambda_g, 'eta': args.eta, 't_m': args.t_m, 'rows': 25, 'channels': 64},
        'identity': {'instances': len(identity), 'max_identity_error': float(identity[['err_query', 'err_support', 'err_mutual']].to_numpy().max()),
                     'max_cost_error': float(identity['err_cost'].max()), 'max_vertex_dispersion': float(exact['V'].abs().max()),
                     'min_dispersion': float(identity['min_V'].min())},
        'full_mass': {'problems': len(full), 'max_abs_diff': float(full['max_abs_diff'].max())},
        'sensitivity': {'pairs': len(sens), 'solves': 3 * len(sens), 'max_positive_excess': float(np.clip(sens[['excess_1', 'excess_2', 'excess_3']].to_numpy(), 0, None).max()),
                        'max_constant_shift_diff': float(sens['constant_shift_diff'].max()), 'full_mass_pairs': len(full_rows),
                        'full_mass_max_l1': float(full_rows['l1'].max()), 'cap_hits': int(sens['cap_hits'].sum()), 'max_sweeps': int(sens['sweeps_max'].max())},
        'tied_cost': {'max_entry_error': float(tied['max_entry_error'].max()), 'P': tied['P'].tolist(), 'R': tied['R'].tolist(), 'V': tied['V'].tolist()},
        'pool_composition': pools,
        'allocation': {'problems': len(allocation), 'median_row_change': float(allocation['row_change'].median()),
                       'median_reallocated': float(allocation['reallocated'].median()), 'median_spearman': float(allocation['spearman'].median()),
                       'rerouted_fraction': float(allocation['rerouted'].sum() / max(allocation['participating'].sum(), 1)),
                       'object_share_plain': float(true['object_share_plain'].mean()), 'object_share_gated': float(true['object_share_gated'].mean())},
        'stopping': {'plan_only_violation_fraction': float((stopping['plan_only_feasibility'] > args.ot_feas_tol).mean()),
                     'plan_only_violation_p95': float(stopping['plan_only_feasibility'].quantile(0.95)),
                     'plan_only_violation_max': float(stopping['plan_only_feasibility'].max()),
                     'plain_sweeps_median': float(stopping['plain_sweeps'].median()), 'plain_sweeps_p95': float(stopping['plain_sweeps'].quantile(0.95)),
                     'gated_sweeps_median': float(stopping['gated_sweeps'].median()), 'gated_sweeps_p95': float(stopping['gated_sweeps'].quantile(0.95)),
                     'cap_hits': int(stopping['plain_cap_hit'].sum() + stopping['gated_cap_hit'].sum()),
                     'all_feasible': bool((stopping[['plain_feasibility', 'gated_feasibility']].to_numpy() <= args.ot_feas_tol).all()),
                     'gated_reference_l1_median': float(stopping['gated_reference_l1'].median())},
        'toy_problem': toy['problem'],
    }
    save_json(summary, out / 'verify_summary.json')
    log(f'verify: identity error {summary["identity"]["max_identity_error"]:.2e}, full-mass {summary["full_mass"]["max_abs_diff"]:.2e}, '
        f'sensitivity excess {summary["sensitivity"]["max_positive_excess"]:.2e}, row change {summary["allocation"]["median_row_change"]:.3f}')
