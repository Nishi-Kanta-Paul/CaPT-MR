import torch
import torch.nn as nn

from .transport import (descriptors, pairwise_costs, rival_costs, row_bias, transport, residual_score, query_residual,
                        support_residual, pairwise_discrepancy)


ALLOCATION = {
    'A0': {'family': 'attention', 'gate': 'none', 'score': 'mutual'},
    'A1': {'family': 'transport', 'gate': 'none', 'score': 'mutual', 'rho': 1.0},
    'A2': {'family': 'transport', 'gate': 'none', 'score': 'mutual'},
    'A3': {'family': 'transport', 'gate': 'candidate', 'score': 'mutual'},
    'A3-query': {'family': 'transport', 'gate': 'candidate', 'score': 'query'},
    'A3-support': {'family': 'transport', 'gate': 'candidate', 'score': 'support'},
    'C1': {'family': 'transport', 'gate': 'candidate-only', 'score': 'mutual'},
    'C2-max': {'family': 'transport', 'gate': 'shared-max', 'score': 'mutual'},
    'C2-mean': {'family': 'transport', 'gate': 'shared-mean', 'score': 'mutual'},
    'C2-weighted': {'family': 'transport', 'gate': 'shared-weighted', 'score': 'mutual'},
    'C3': {'family': 'toprow', 'gate': 'none', 'score': 'mutual'},
    'C4': {'family': 'transport', 'gate': 'permuted', 'score': 'mutual'},
    'Anchored': {'family': 'transport', 'gate': 'anchored', 'score': 'mutual'},
}


def mark(timer, name):
    if timer is not None:
        timer.mark(name)


class AllocationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.family = cfg['family']
        self.gate = cfg['gate']
        self.score_kind = cfg['score']

    def permutation(self, length, device):
        generator = torch.Generator().manual_seed(int(self.cfg['permutation_seed']))
        return torch.randperm(length, generator=generator).to(device)

    def solve(self, costs, d_pos, d_neg, timer):
        cfg = self.cfg
        gate_cfg = dict(cfg)
        if self.gate == 'permuted':
            gate_cfg['permutation'] = self.permutation(costs.shape[-2], costs.device)
        biased = costs
        if self.gate != 'none':
            biased = costs + row_bias(self.gate, d_pos, d_neg, gate_cfg)[:, :, None, :, None]
        mark(timer, 'gate')
        plan, info = transport(biased, cfg['rho'], cfg['entropy'], cfg['ot_tol'], cfg['ot_feas_tol'], cfg['ot_max_iter'],
                               cfg['ot_patience'], cfg['ot_fallback_iter'])
        mark(timer, 'solver')
        return plan, info

    def toprow(self, costs, d_pos, d_neg, timer):
        length = costs.shape[-2]
        keep = int(round(self.cfg['rho'] * length))
        with torch.no_grad():
            top = (d_neg - d_pos).topk(keep, dim=-1).indices
            mask = torch.zeros_like(d_pos).scatter_(-1, top, 1.0)
        mark(timer, 'gate')
        plan = mask[:, :, None, :, None] * torch.softmax(-costs / self.cfg['toprow_temp'], dim=-1) / length
        mark(timer, 'solver')
        return plan, keep / length

    def forward(self, fq, fs, need_plans=False, need_terms=False, timer=None):
        cfg = self.cfg
        hq, hs = descriptors(fq, cfg['floor']), descriptors(fs, cfg['floor'])
        costs = pairwise_costs(hq, hs)
        d_pos = d_neg = None
        if self.gate != 'none' or self.family == 'toprow':
            with torch.no_grad():
                d_pos, d_neg = rival_costs(costs, cfg['t_m'])
        mark(timer, 'costs')
        info, plan = None, None
        if self.family == 'attention':
            length = hq.shape[1]
            plan_q = torch.softmax(-costs / cfg['attn_temp'], dim=-1) / length
            plan_s = torch.softmax(-costs / cfg['attn_temp'], dim=-2) / length
            mark(timer, 'solver')
            rq, rs = query_residual(plan_q, hq, hs, 1.0), support_residual(plan_s, hq, hs, 1.0)
            residual = {'query': rq, 'support': rs, 'mutual': 0.5 * (rq + rs)}[self.score_kind]
            dispersion = torch.zeros_like(residual)
            plan = plan_q
        else:
            if self.family == 'toprow':
                plan, mass = self.toprow(costs, d_pos, d_neg, timer)
            else:
                plan, info = self.solve(costs, d_pos, d_neg, timer)
                mass = min(cfg['rho'], 1.0)
            residual = residual_score(plan, hq, hs, mass, self.score_kind)
            dispersion = None
            if cfg['kappa'] > 0 or need_terms:
                dispersion = pairwise_discrepancy(plan, hq, hs, 1.0 - costs, mass) - residual
        distance = residual if cfg['kappa'] == 0 else residual + cfg['kappa'] * dispersion
        mark(timer, 'recon')
        out = {'score': -distance.mean(2), 'info': info}
        if need_terms:
            out['residual'] = residual.mean(2)
            out['dispersion'] = dispersion.mean(2)
        if need_plans:
            out['plan'] = plan
        return out
