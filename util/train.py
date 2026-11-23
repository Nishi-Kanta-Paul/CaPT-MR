import time
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .episodes import episode_tensors, query_labels
from .utils import log, synchronize


class StageTimer:
    def __init__(self, device):
        self.device = device
        self.times = {}
        self.last = None

    def start(self):
        synchronize(self.device)
        self.last = time.perf_counter()

    def mark(self, name):
        synchronize(self.device)
        now = time.perf_counter()
        self.times[name] = self.times.get(name, 0.0) + 1000.0 * (now - self.last)
        self.last = now


def make_optimizer(model, args, episodes):
    decay = [p for name, p in model.named_parameters() if name != 'log_tau']
    groups = [{'params': decay, 'weight_decay': args.weight_decay}, {'params': [model.log_tau], 'weight_decay': 0.0}]
    optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum)
    milestones = sorted({max(1, int(round(f * episodes))) for f in args.lr_milestones})
    return optimizer, torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.lr_gamma)


def solver_summary(info):
    if info is None:
        return np.nan, np.nan, np.nan, np.nan
    sweeps = info['sweeps'].float().flatten()
    return (float(sweeps.median()), float(torch.quantile(sweeps, 0.95)), float(sweeps.max()), float(info['cap_hit'].float().mean()))


@torch.no_grad()
def evaluate_bank(model, images, bank, device, diagnostics=False, terms=False, query_images=None):
    model.eval()
    n_way, k_shot, n_query = bank.shape
    labels = query_labels(n_way, n_query, device)
    n_episodes, n_queries = len(bank), n_way * n_query
    out = {'acc': np.zeros(n_episodes, dtype=np.float32), 'pred': np.zeros((n_episodes, n_queries), dtype=np.int16),
           'sweeps': np.full((n_episodes, 4), np.nan, dtype=np.float32)}
    if diagnostics:
        out.update({'row_mass': np.zeros((n_episodes, n_queries, 0), dtype=np.float16), 'dest': None,
                    'margin': np.zeros((n_episodes, n_queries), dtype=np.float32)})
    if terms:
        out.update({'residual': np.zeros((n_episodes, n_queries, n_way), dtype=np.float32),
                    'dispersion': np.zeros((n_episodes, n_queries, n_way), dtype=np.float32)})
    for e in range(n_episodes):
        support, query = bank.episode(e)
        s, q = episode_tensors(images, support, query, device, query_images=None if query_images is None else query_images(e))
        result = model(s, q, need_plans=diagnostics, need_terms=terms)
        pred = result['logits'].argmax(-1)
        out['acc'][e] = float((pred == labels).float().mean())
        out['pred'][e] = pred.cpu().numpy()
        out['sweeps'][e] = solver_summary(result.get('info'))
        if diagnostics and 'plan' in result:
            plan = result['plan'][torch.arange(n_queries, device=device), labels, 0]
            rows = plan.sum(-1).cpu().numpy()
            if out['dest'] is None:
                out['row_mass'] = np.zeros((n_episodes, n_queries, rows.shape[-1]), dtype=np.float16)
                out['dest'] = np.zeros((n_episodes, n_queries, rows.shape[-1]), dtype=np.int8)
            out['row_mass'][e] = rows
            out['dest'][e] = plan.argmax(-1).cpu().numpy()
            distance = -result['score']
            true = distance[torch.arange(n_queries, device=device), labels]
            wrong = distance.clone()
            wrong[torch.arange(n_queries, device=device), labels] = float('inf')
            out['margin'][e] = (wrong.min(-1).values - true).cpu().numpy()
        if terms:
            out['residual'][e] = result['residual'].cpu().numpy()
            out['dispersion'][e] = result['dispersion'].cpu().numpy()
    if diagnostics and out['dest'] is None:
        out.pop('dest')
    return out


def train_model(model, images, stream, select_bank, args, device, name):
    optimizer, scheduler = make_optimizer(model, args, args.train_episodes)
    labels = query_labels(stream.n_way, stream.n_query, device)
    best = {'acc': -1.0, 'episode': 0, 'state': None}
    history, window = [], []
    start = time.time()
    for t in range(args.train_episodes):
        model.train()
        support, query = stream.episode(t)
        s, q = episode_tensors(images, support, query, device, stream.generator(t), args.jitter)
        result = model(s, q)
        loss = F.cross_entropy(result['logits'], labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        info = result.get('info')
        window.append((float(loss.detach()), float((result['logits'].argmax(-1) == labels).float().mean()),
                       float(info['cap_hit'].float().mean()) if info is not None else np.nan))
        if (t + 1) % args.val_every == 0 or t + 1 == args.train_episodes:
            val = 100.0 * float(evaluate_bank(model, images, select_bank, device)['acc'].mean())
            loss_m, acc_m = float(np.mean([w[0] for w in window])), float(np.mean([w[1] for w in window]))
            caps = [w[2] for w in window if np.isfinite(w[2])]
            cap_m = float(np.mean(caps)) if caps else float('nan')
            history.append({'episode': t + 1, 'train_loss': loss_m, 'train_acc': 100.0 * acc_m, 'cap_hit_rate': cap_m, 'val_acc': val,
                            'tau': float(model.log_tau.exp()), 'lr': optimizer.param_groups[0]['lr'], 'seconds': time.time() - start})
            window = []
            if val > best['acc']:
                best = {'acc': val, 'episode': t + 1, 'state': copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            log(f'{name}: episode {t + 1}/{args.train_episodes} loss {loss_m:.4f} train {100 * acc_m:.2f}% val {val:.2f}% '
                f'(best {best["acc"]:.2f}% at {best["episode"]})')
    return best, history
