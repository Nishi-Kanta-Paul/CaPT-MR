import numpy as np
from scipy.special import logsumexp, expit


SHARED_GATES = ('shared-max', 'shared-mean', 'shared-weighted')


def uniform(n):
    return np.full(n, 1.0 / n)


def feasibility(plan, a, b, rho, balanced):
    rows, cols = plan.sum(2), plan.sum(1)
    if balanced:
        return np.maximum(np.abs(rows - a).max(1), np.abs(cols - b).max(1))
    over = np.maximum(np.clip(rows - a, 0, None).max(1), np.clip(cols - b, 0, None).max(1))
    return np.maximum(over, np.abs(plan.sum((1, 2)) - rho) / rho)


def iterate(kernel, a, b, rho, tol, feas_tol, max_iter, patience, stop):
    count, l, m = kernel.shape
    balanced = rho >= 1.0
    mass = min(rho, 1.0)
    x, y, w = np.ones((count, l)), np.ones((count, m)), np.ones(count)
    plan = np.zeros_like(kernel)
    sweeps = np.zeros(count, dtype=int)
    streak = np.zeros(count, dtype=int)
    done = np.zeros(count, dtype=bool)
    feas = np.full(count, np.inf)
    active = np.arange(count)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for it in range(1, max_iter + 1):
            k = kernel[active]
            if balanced:
                xa = a[active] / np.matmul(k, y[active][..., None])[..., 0]
                ya = b[active] / np.matmul(xa[:, None, :], k)[:, 0]
                wa = w[active]
            else:
                xa = np.minimum(1.0, a[active] / (w[active, None] * np.matmul(k, y[active][..., None])[..., 0]))
                ya = np.minimum(1.0, b[active] / (w[active, None] * np.matmul(xa[:, None, :], k)[:, 0]))
                wa = rho / np.matmul(np.matmul(xa[:, None, :], k), ya[:, :, None])[:, 0, 0]
            x[active], y[active], w[active] = xa, ya, wa
            new = wa[:, None, None] * xa[:, :, None] * k * ya[:, None, :]
            change = np.abs(new - plan[active]).sum((1, 2)) / mass
            fa = feasibility(new, a[active], b[active], mass, balanced)
            ok = change <= tol
            if stop == 'combined':
                ok &= fa <= feas_tol
            streak[active] = np.where(ok, streak[active] + 1, 0)
            plan[active], sweeps[active], feas[active] = new, it, fa
            finished = streak[active] >= patience
            done[active[finished]] = True
            active = active[~finished]
            if active.size == 0:
                break
    return plan, sweeps, done, feas


def solve(cost, rho, eps, a=None, b=None, bias=None, tol=1e-8, feas_tol=1e-9, max_iter=40000, patience=2, fallback_iter=0,
          stop='combined'):
    cost = np.asarray(cost, dtype=np.float64)
    single = cost.ndim == 2
    if single:
        cost = cost[None]
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        cost = cost + (bias[None] if bias.ndim == 1 else bias)[:, :, None]
    count, l, m = cost.shape
    a = np.ascontiguousarray(np.broadcast_to(uniform(l) if a is None else a, (count, l)), dtype=np.float64)
    b = np.ascontiguousarray(np.broadcast_to(uniform(m) if b is None else b, (count, m)), dtype=np.float64)
    kernel = np.exp(-(cost - cost.min(axis=(1, 2), keepdims=True)) / eps)
    plan, sweeps, done, feas = iterate(kernel, a, b, rho, tol, feas_tol, max_iter, patience, stop)
    cap_hit = ~done
    fallback = np.zeros(count, dtype=bool)
    if fallback_iter and cap_hit.any():
        idx = np.flatnonzero(cap_hit)
        p2, s2, d2, f2 = iterate(kernel[idx], a[idx], b[idx], rho, tol, feas_tol, fallback_iter, patience, stop)
        plan[idx], sweeps[idx], done[idx], feas[idx] = p2, s2, d2, f2
        fallback[idx] = True
    info = {'sweeps': sweeps, 'converged': done, 'cap_hit': cap_hit, 'fallback': fallback, 'feasibility': feas}
    if single:
        return plan[0], {k: v[0] for k, v in info.items()}
    return plan, info


def solve_lp(cost, rho, a=None, b=None):
    from scipy.optimize import linprog
    l, m = cost.shape
    a = uniform(l) if a is None else a
    b = uniform(m) if b is None else b
    rows = np.kron(np.eye(l), np.ones((1, m)))
    cols = np.kron(np.ones((1, l)), np.eye(m))
    result = linprog(cost.ravel(), A_ub=np.vstack([rows, cols]), b_ub=np.concatenate([a, b]), A_eq=np.ones((1, l * m)),
                     b_eq=[rho], bounds=(0, None), method='highs-ds')
    if not result.success:
        raise RuntimeError(result.message)
    return np.clip(result.x.reshape(l, m), 0, None)


def sq_distances(x, y):
    return ((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(-1)


def barycentres(plan, vq, vs):
    rows, cols = plan.sum(2), plan.sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        vq_hat = np.where(rows[..., None] > 0, np.matmul(plan, vs) / rows[..., None], 0.0)
        vs_hat = np.where(cols[..., None] > 0, np.matmul(plan.transpose(0, 2, 1), vq) / cols[..., None], 0.0)
    return rows, cols, vq_hat, vs_hat


def reconstruction_terms(plan, vq, vs, rho):
    plan, vq, vs = np.asarray(plan, dtype=np.float64), np.asarray(vq, dtype=np.float64), np.asarray(vs, dtype=np.float64)
    if plan.ndim == 2:
        plan = plan[None]
    if vq.ndim == 2:
        vq, vs = vq[None], vs[None]
    rows, cols, vq_hat, vs_hat = barycentres(plan, vq, vs)
    rq = (rows * ((vq - vq_hat) ** 2).sum(-1)).sum(1) / rho
    rs = (cols * ((vs - vs_hat) ** 2).sum(-1)).sum(1) / rho
    vq_term = (plan * sq_distances(vq_hat, vs)).sum((1, 2)) / rho
    vs_term = (plan * sq_distances(vq, vs_hat)).sum((1, 2)) / rho
    pairwise = (plan * sq_distances(vq, vs)).sum((1, 2)) / rho
    return {'Rq': rq, 'Rs': rs, 'R': 0.5 * (rq + rs), 'Vq': vq_term, 'Vs': vs_term, 'V': 0.5 * (vq_term + vs_term), 'P': pairwise}


def query_kernel(plan, vq, vs, rho):
    rows = plan.sum(2)
    recon = np.matmul(plan, vs) / rows[..., None]
    return (rows * ((vq - recon) ** 2).sum(-1)).sum(1) / rho


def support_kernel(plan, vq, vs, rho):
    cols = plan.sum(1)
    recon = np.matmul(plan.transpose(0, 2, 1), vq) / cols[..., None]
    return (cols * ((vs - recon) ** 2).sum(-1)).sum(1) / rho


def mutual_kernel(plan, vq, vs, rho):
    return 0.5 * (query_kernel(plan, vq, vs, rho) + support_kernel(plan, vq, vs, rho))


KERNELS = {'query': query_kernel, 'support': support_kernel, 'mutual': mutual_kernel}


def residuals(plan, vq, vs, rho, kind='mutual'):
    return KERNELS[kind](plan, vq, vs, rho)


def softmin(values, t, axis=-1):
    values = np.asarray(values, dtype=np.float64)
    return -t * (logsumexp(-values / t, axis=axis) - np.log(values.shape[axis]))


def rival_costs(costs, t_m):
    *lead, n, k, l, m = costs.shape
    z = -np.swapaxes(costs, -3, -2).reshape(*lead, n, l, k * m) / t_m
    pooled = logsumexp(z, axis=-1)
    prefix = np.logaddexp.accumulate(pooled, axis=-2)
    suffix = np.flip(np.logaddexp.accumulate(np.flip(pooled, -2), axis=-2), -2)
    empty = np.full_like(pooled[..., :1, :], -np.inf)
    excluded = np.logaddexp(np.concatenate([empty, prefix[..., :-1, :]], axis=-2), np.concatenate([suffix[..., 1:, :], empty], axis=-2))
    d_pos = -t_m * (pooled - np.log(k * m))
    d_neg = -t_m * (excluded - np.log((n - 1) * k * m))
    return d_pos, d_neg


def gate_config(args, offsets=None):
    return {'lambda_g': args.lambda_g, 'eta': args.eta, 't_m': args.t_m, 'candidate_offset': args.candidate_offset,
            'candidate_temp': args.candidate_temp, 'shared_temp': args.shared_temp, 'weight_temp': args.weight_temp,
            'anchor_alpha': args.anchor_alpha, 'shared_offset': offsets or {k: args.shared_offset for k in SHARED_GATES}}


def pool_margins(margin, kind, t_w):
    if kind == 'shared-max':
        return margin.max(axis=-2)
    if kind == 'shared-mean':
        return margin.mean(axis=-2)
    z = margin / t_w
    weights = np.exp(z - logsumexp(z, axis=-2, keepdims=True))
    return (weights * margin).sum(axis=-2)


def gate_values(kind, d_pos, d_neg, cfg):
    margin = d_neg - d_pos
    if kind == 'candidate':
        return expit(margin / cfg['eta'])
    if kind == 'candidate-only':
        return expit((cfg['candidate_offset'] - d_pos) / cfg['candidate_temp'])
    pooled = pool_margins(margin, kind, cfg['weight_temp'])
    return np.broadcast_to(expit((pooled - cfg['shared_offset'][kind]) / cfg['shared_temp'])[..., None, :], margin.shape)


def row_bias(kind, d_pos, d_neg, cfg):
    if kind == 'none':
        return np.zeros_like(d_pos)
    if kind == 'anchored':
        shared = row_bias('shared-max', d_pos, d_neg, cfg)
        specific = row_bias('candidate', d_pos, d_neg, cfg)
        return (1.0 - cfg['anchor_alpha']) * shared + cfg['anchor_alpha'] * specific
    return cfg['lambda_g'] * (1.0 - gate_values(kind, d_pos, d_neg, cfg))
