import math

import torch


SHARED_GATES = ('shared-max', 'shared-mean', 'shared-weighted')


def descriptors(fmap, floor):
    flat = fmap.flatten(-2).transpose(-1, -2)
    return flat / flat.norm(dim=-1, keepdim=True).clamp_min(floor)


def pairwise_costs(hq, hs):
    return 1.0 - torch.einsum('qic,nkjc->qnkij', hq, hs)


def softmin(values, t, dim=-1):
    return -t * (torch.logsumexp(-values / t, dim=dim) - math.log(values.shape[dim]))


def rival_costs(costs, t_m):
    q, n, k, l, m = costs.shape
    pooled = torch.logsumexp((-costs / t_m).permute(0, 1, 3, 2, 4).reshape(q, n, l, k * m), dim=-1)
    prefix = torch.logcumsumexp(pooled, dim=1)
    suffix = torch.logcumsumexp(pooled.flip(1), dim=1).flip(1)
    empty = torch.full_like(pooled[:, :1], float('-inf'))
    excluded = torch.logaddexp(torch.cat([empty, prefix[:, :-1]], dim=1), torch.cat([suffix[:, 1:], empty], dim=1))
    d_pos = -t_m * (pooled - math.log(k * m))
    d_neg = -t_m * (excluded - math.log((n - 1) * k * m))
    return d_pos, d_neg


def pool_margins(margin, kind, t_w):
    if kind == 'shared-max':
        return margin.max(dim=1).values
    if kind == 'shared-mean':
        return margin.mean(dim=1)
    weights = torch.softmax(margin / t_w, dim=1)
    return (weights * margin).sum(dim=1)


def gate_values(kind, d_pos, d_neg, cfg):
    margin = d_neg - d_pos
    if kind in ('candidate', 'permuted'):
        gate = torch.sigmoid(margin / cfg['eta'])
        return gate[..., cfg['permutation']] if kind == 'permuted' else gate
    if kind == 'candidate-only':
        return torch.sigmoid((cfg['candidate_offset'] - d_pos) / cfg['candidate_temp'])
    if kind in SHARED_GATES:
        pooled = pool_margins(margin, kind, cfg['weight_temp'])
        return torch.sigmoid((pooled - cfg['shared_offset']) / cfg['shared_temp']).unsqueeze(1).expand_as(margin)
    raise ValueError(f'unknown gate {kind}')


def row_bias(kind, d_pos, d_neg, cfg):
    if kind == 'none':
        return torch.zeros_like(d_pos)
    if kind == 'anchored':
        shared = row_bias('shared-max', d_pos, d_neg, cfg)
        specific = row_bias('candidate', d_pos, d_neg, cfg)
        return (1.0 - cfg['anchor_alpha']) * shared + cfg['anchor_alpha'] * specific
    return cfg['lambda_g'] * (1.0 - gate_values(kind, d_pos, d_neg, cfg).detach())


def feasibility(plan, log_a, log_b, rho, balanced):
    rows, cols = plan.sum(-1), plan.sum(-2)
    a, b = log_a.exp(), log_b.exp()
    if balanced:
        return torch.maximum((rows - a).abs().amax(-1), (cols - b).abs().amax(-1))
    over = torch.maximum((rows - a).clamp_min(0).amax(-1), (cols - b).clamp_min(0).amax(-1))
    return torch.maximum(over, (plan.sum((-2, -1)) - rho).abs() / rho)


def transport(cost, rho, eps, tol=1e-4, feas_tol=1e-5, max_iter=500, patience=2, fallback_iter=0, log_a=None, log_b=None,
              stop='combined'):
    *batch, l, m = cost.shape
    log_a = (cost.new_full((l,), -math.log(l)) if log_a is None else log_a).expand(*batch, l)
    log_b = (cost.new_full((m,), -math.log(m)) if log_b is None else log_b).expand(*batch, m)
    balanced = rho >= 1.0
    log_rho = math.log(min(rho, 1.0))
    z = -cost / eps
    alpha, beta, gamma = cost.new_zeros(*batch, l), cost.new_zeros(*batch, m), cost.new_zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool, device=cost.device)
    cap_hit = torch.zeros(batch, dtype=torch.bool, device=cost.device)
    streak = torch.zeros(batch, dtype=torch.long, device=cost.device)
    sweeps = torch.zeros(batch, dtype=torch.long, device=cost.device)
    limit = max(max_iter, fallback_iter) if fallback_iter and not torch.is_grad_enabled() else max_iter
    plan, feas = None, None
    for it in range(1, limit + 1):
        if balanced:
            a_new = log_a - torch.logsumexp(z + beta.unsqueeze(-2), dim=-1)
            b_new = log_b - torch.logsumexp(z + a_new.unsqueeze(-1), dim=-2)
            g_new = gamma
        else:
            shift = gamma[..., None, None]
            a_new = (log_a - torch.logsumexp(z + beta.unsqueeze(-2) + shift, dim=-1)).clamp_max(0.0)
            b_new = (log_b - torch.logsumexp(z + a_new.unsqueeze(-1) + shift, dim=-2)).clamp_max(0.0)
            g_new = log_rho - torch.logsumexp((z + a_new.unsqueeze(-1) + b_new.unsqueeze(-2)).flatten(-2), dim=-1)
        if plan is None:
            alpha, beta, gamma = a_new, b_new, g_new
        else:
            alpha = torch.where(done.unsqueeze(-1), alpha, a_new)
            beta = torch.where(done.unsqueeze(-1), beta, b_new)
            gamma = torch.where(done, gamma, g_new)
        current = (z + alpha.unsqueeze(-1) + beta.unsqueeze(-2) + gamma[..., None, None]).exp()
        with torch.no_grad():
            if plan is None:
                change = torch.full(batch, float('inf'), device=cost.device)
            else:
                change = (current - plan).abs().sum((-2, -1)) / min(rho, 1.0)
            feas = feasibility(current, log_a, log_b, min(rho, 1.0), balanced)
            ok = change <= tol
            if stop == 'combined':
                ok = ok & (feas <= feas_tol)
            streak = torch.where(ok, streak + 1, torch.zeros_like(streak))
            sweeps = torch.where(done, sweeps, torch.full_like(sweeps, it))
            done = done | (streak >= patience)
            if it == max_iter:
                cap_hit = ~done
        plan = current
        if bool(done.all()):
            break
    return plan, {'sweeps': sweeps, 'converged': done, 'cap_hit': cap_hit, 'feasibility': feas}


def query_residual(plan, vq, vs, rho, floor=1e-12):
    rows = plan.sum(-1)
    recon = torch.einsum('qnkij,nkjc->qnkic', plan, vs) / rows.unsqueeze(-1).clamp_min(floor)
    return (rows * (vq[:, None, None] - recon).pow(2).sum(-1)).sum(-1) / rho


def support_residual(plan, vq, vs, rho, floor=1e-12):
    cols = plan.sum(-2)
    recon = torch.einsum('qnkij,qic->qnkjc', plan, vq) / cols.unsqueeze(-1).clamp_min(floor)
    return (cols * (vs[None] - recon).pow(2).sum(-1)).sum(-1) / rho


def pairwise_discrepancy(plan, vq, vs, dots, rho):
    rows, cols = plan.sum(-1), plan.sum(-2)
    nq, ns = vq.pow(2).sum(-1), vs.pow(2).sum(-1)
    return ((rows * nq[:, None, None]).sum(-1) + (cols * ns[None]).sum(-1) - 2.0 * (plan * dots).sum((-2, -1))) / rho


def residual_score(plan, hq, hs, rho, kind, floor=1e-12):
    if kind == 'query':
        return query_residual(plan, hq, hs, rho, floor)
    if kind == 'support':
        return support_residual(plan, hq, hs, rho, floor)
    return 0.5 * (query_residual(plan, hq, hs, rho, floor) + support_residual(plan, hq, hs, rho, floor))
