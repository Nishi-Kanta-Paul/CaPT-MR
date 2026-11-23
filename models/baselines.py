import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transport import descriptors, transport


def flat(fmap):
    return fmap.flatten(-2).transpose(-1, -2)


def ridge_reconstruct(x, pool, lam, scale):
    m, c = pool.shape[-2:]
    eye = torch.eye(min(m, c), device=pool.device, dtype=pool.dtype)
    if m <= c:
        gram = pool @ pool.transpose(-1, -2) + lam * eye
        return scale * (x @ pool.transpose(-1, -2)) @ torch.linalg.solve(gram, pool)
    hat = pool.transpose(-1, -2) @ pool
    return scale * x @ torch.linalg.solve(hat + lam * eye, hat)


def query_errors(xq, xs, alpha, beta):
    q, l, c = xq.shape
    n, m, _ = xs.shape
    lam = m / c * alpha.exp() + 1e-6
    target = xq.reshape(1, q * l, c)
    recon = ridge_reconstruct(target.expand(n, -1, -1), xs, lam, beta.exp())
    return (recon - target).pow(2).sum(-1).view(n, q, l).permute(1, 0, 2)


def support_errors(xq, xs, alpha, beta):
    q, l, c = xq.shape
    n, m, _ = xs.shape
    lam = l / c * alpha.exp() + 1e-6
    target = xs.reshape(1, n * m, c)
    recon = ridge_reconstruct(target.expand(q, -1, -1), xq, lam, beta.exp())
    return (recon - target).pow(2).sum(-1).view(q, n, m)


class ProtoNetHead(nn.Module):
    def forward(self, fq, fs, **kwargs):
        query, proto = fq.mean((-2, -1)), fs.mean((1, -2, -1))
        return {'score': -(query[:, None] - proto[None]).pow(2).mean(-1), 'info': None}


class DN4Head(nn.Module):
    def __init__(self, neighbours, floor):
        super().__init__()
        self.neighbours, self.floor = neighbours, floor

    def forward(self, fq, fs, **kwargs):
        hq = descriptors(fq, self.floor)
        hs = descriptors(fs, self.floor).flatten(1, 2)
        sims = torch.einsum('qic,njc->qnij', hq, hs)
        return {'score': sims.topk(min(self.neighbours, sims.shape[-1]), dim=-1).values.sum(-1).mean(-1), 'info': None}


class DMN4Head(nn.Module):
    def __init__(self, floor):
        super().__init__()
        self.floor = floor

    def forward(self, fq, fs, **kwargs):
        hq = descriptors(fq, self.floor)
        hs = descriptors(fs, self.floor)
        n, k, l, c = hs.shape
        q = hq.shape[0]
        sims = torch.einsum('qic,mc->qim', hq, hs.reshape(n * k * l, c))
        with torch.no_grad():
            nearest = sims.argmax(-1)
            back = sims.argmax(1)
            mutual = (back.gather(1, nearest) == torch.arange(hq.shape[1], device=hq.device)).float()
        class_best = sims.view(q, -1, n, k * l).amax(-1)
        score = (class_best * mutual.unsqueeze(-1)).sum(1) / mutual.sum(1).clamp_min(1.0).unsqueeze(-1)
        return {'score': score, 'info': None}


class DeepEMDHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)

    def forward(self, fq, fs, need_plans=False, timer=None, **kwargs):
        cfg = self.cfg
        hq = descriptors(fq, cfg['floor'])
        hs = descriptors(fs, cfg['floor']).flatten(1, 2)
        sims = torch.einsum('qic,njc->qnij', hq, hs)
        weight_q = F.relu(torch.einsum('qic,nc->qni', hq, hs.mean(1))) + 1e-3
        weight_s = F.relu(torch.einsum('njc,qc->qnj', hs, hq.mean(1))) + 1e-3
        log_a = (weight_q / weight_q.sum(-1, keepdim=True)).log()
        log_b = (weight_s / weight_s.sum(-1, keepdim=True)).log()
        if timer is not None:
            timer.mark('costs')
        plan, info = transport(1.0 - sims, 1.0, cfg['emd_entropy'], cfg['ot_tol'], cfg['ot_feas_tol'], cfg['ot_max_iter'],
                               cfg['ot_patience'], cfg['ot_fallback_iter'], log_a=log_a, log_b=log_b)
        if timer is not None:
            timer.mark('solver')
        out = {'score': (plan * sims).sum((-2, -1)), 'info': info}
        if need_plans:
            out['plan'] = plan
        return out


class FRNHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))

    def features(self, fmap):
        return flat(fmap) / math.sqrt(self.channels)

    def class_pools(self, fs):
        n, k = fs.shape[:2]
        return self.features(fs).reshape(n, -1, self.channels)

    def forward(self, fq, fs, **kwargs):
        distance = query_errors(self.features(fq), self.class_pools(fs), self.alpha, self.beta).mean(-1)
        return {'score': -distance, 'info': None}


class BiFRNHead(FRNHead):
    def __init__(self, channels):
        super().__init__(channels)
        self.align = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))
        self.alpha_s = nn.Parameter(torch.zeros(()))
        self.beta_s = nn.Parameter(torch.zeros(()))
        self.log_weights = nn.Parameter(torch.full((2,), math.log(0.5)))

    def refine(self, fmap):
        return F.relu(fmap + self.align(fmap))

    def forward(self, fq, fs, **kwargs):
        fq = self.refine(fq)
        fs = self.refine(fs.flatten(0, 1)).view_as(fs)
        xq, xs = self.features(fq), self.class_pools(fs)
        dq = query_errors(xq, xs, self.alpha, self.beta).mean(-1)
        ds = support_errors(xq, xs, self.alpha_s, self.beta_s).mean(-1)
        weights = self.log_weights.exp()
        return {'score': -(weights[0] * dq + weights[1] * ds), 'info': None}


class BSNetHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relation = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 8), nn.ReLU(inplace=True), nn.Linear(8, 1),
        )

    def forward(self, fq, fs, **kwargs):
        proto = fs.mean(1)
        q, n = fq.shape[0], proto.shape[0]
        pairs = torch.cat([fq[:, None].expand(q, n, *fq.shape[1:]), proto[None].expand(q, n, *proto.shape[1:])], dim=2)
        relation = torch.sigmoid(self.relation(pairs.flatten(0, 1))).view(q, n)
        cosine = F.cosine_similarity(fq.mean((-2, -1))[:, None], proto.mean((-2, -1))[None], dim=-1)
        return {'score': 0.5 * (relation + cosine), 'info': None}


class TDMHead(FRNHead):
    def __init__(self, channels, hidden=16):
        super().__init__(channels)
        self.support_attention = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
        self.query_attention = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def channel_weights(self, xq, xs):
        n = xs.shape[0]
        proto = xs.mean(1)
        intra = (xs - proto[:, None]).pow(2).mean(1)
        inter = (proto[:, None] - proto[None]).pow(2).sum(1) / max(n - 1, 1)
        support = torch.sigmoid(self.support_attention(torch.stack([intra.add(1e-6).log(), inter.add(1e-6).log()], -1)).squeeze(-1))
        stats = torch.stack([xq.mean(1), xq.var(1, unbiased=False).add(1e-6).log()], -1)
        query = torch.sigmoid(self.query_attention(stats).squeeze(-1))
        return 0.5 * (support[None] + query[:, None])

    def forward(self, fq, fs, **kwargs):
        xq, xs = self.features(fq), self.class_pools(fs)
        q, l, c = xq.shape
        n, m, _ = xs.shape
        weights = self.channel_weights(xq, xs)
        xq_w = (xq[:, None] * weights[:, :, None]).reshape(q * n, l, c)
        xs_w = (xs[None] * weights[:, :, None]).reshape(q * n, m, c)
        recon = ridge_reconstruct(xq_w, xs_w, m / c * self.alpha.exp() + 1e-6, self.beta.exp())
        return {'score': -(recon - xq_w).pow(2).sum(-1).mean(-1).view(q, n), 'info': None}


class SFMRHead(nn.Module):
    def __init__(self, channels, radius):
        super().__init__()
        self.channels, self.radius = channels, radius
        self.saliency = nn.Conv2d(channels, 1, 1)
        self.alpha = nn.Parameter(torch.zeros(3, 2))
        self.beta = nn.Parameter(torch.zeros(3, 2))
        self.stream_logits = nn.Parameter(torch.zeros(3))

    def streams(self, fmap):
        h, w = fmap.shape[-2:]
        fy = torch.fft.fftfreq(h, device=fmap.device)
        fx = torch.fft.fftfreq(w, device=fmap.device)
        radius = (fy[:, None].pow(2) + fx[None].pow(2)).sqrt() / math.sqrt(0.5)
        mask = (radius <= self.radius).to(fmap.dtype)
        low = torch.fft.ifft2(torch.fft.fft2(fmap) * mask).real
        return (fmap, low, fmap - low)

    def forward(self, fq, fs, **kwargs):
        n, k = fs.shape[:2]
        flat_s = fs.flatten(0, 1)
        saliency_q = torch.softmax(self.saliency(fq).flatten(1), dim=-1)
        saliency_s = torch.softmax(self.saliency(flat_s).flatten(1).view(n, -1), dim=-1)
        weights = torch.softmax(self.stream_logits, dim=0)
        total = 0.0
        for i, (sq, ss) in enumerate(zip(self.streams(fq), self.streams(flat_s))):
            xq = flat(sq) / math.sqrt(self.channels)
            xs = (flat(ss) / math.sqrt(self.channels)).reshape(n, -1, self.channels)
            dq = (query_errors(xq, xs, self.alpha[i, 0], self.beta[i, 0]) * saliency_q[:, None]).sum(-1)
            ds = (support_errors(xq, xs, self.alpha[i, 1], self.beta[i, 1]) * saliency_s[None]).sum(-1)
            total = total + weights[i] * 0.5 * (dq + ds)
        return {'score': -total, 'info': None}
