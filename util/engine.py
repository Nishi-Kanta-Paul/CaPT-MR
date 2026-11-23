import os
import ast
import time
import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import log, ensure_dir, save_json, load_json_safe, stable_hash, resolve_device, synchronize, free_memory, set_seed, run_tag
from .datasets import DATASETS, load_dataset, dataset_summary, build_cache
from .episodes import EpisodeBank, EpisodeStream, make_bank, episode_tensors, query_labels, normalize
from .perturbations import conditions, apply_condition, parse_condition
from .train import StageTimer, evaluate_bank, train_model, make_optimizer


SETTINGS = tuple((d, k) for d in DATASETS for k in (1, 5))
PRIMARY_METHODS = ('ProtoNet', 'DN4', 'DMN4', 'DeepEMD', 'FRN', 'Bi-FRN', 'BSNet', 'TDM', 'SFMR', 'A2', 'C2-max', 'C2-mean',
                   'C2-weighted', 'A3', 'A3-query', 'A3-support')
CONFIRM_METHODS = ('A0', 'A1', 'A2', 'A3', 'C1', 'C2-max', 'C3', 'C4')
BACKBONE_METHODS = ('Bi-FRN', 'A2', 'C2-max', 'C2-mean', 'SFMR', 'A3-query', 'A3-support', 'A3')
DIAG_METHODS = ('A1', 'A2', 'C2-max', 'A3')
FOLLOWUP_METHODS = ('FRN', 'Bi-FRN', 'A2', 'C2-max', 'A3')
PROFILE_METHODS = ('ProtoNet', 'FRN', 'Bi-FRN', 'TDM', 'SFMR', 'DeepEMD', 'A1', 'A2', 'C2-max', 'C2-mean', 'C2-weighted', 'A3-query',
                   'A3-support', 'A3')
STAGE_METHODS = ('A2', 'C2-max', 'A3')
EXPERIMENTS = ('primary', 'confirm', 'coverage', 'kappa', 'backbone')
TUNED_METHODS = ('A0', 'A3', 'C1', 'C2-max', 'C2-mean', 'C2-weighted', 'C3')
TUNE_ALIASES = {'A3-query': 'A3', 'A3-support': 'A3', 'C4': 'A3', 'Anchored': 'A3'}
DATASET_DIRS = {'cub': 'CUB_200_2011', 'dogs': 'StanfordDogs', 'cars': 'StanfordCars'}

Run = namedtuple('Run', ['backbone', 'dataset', 'shot', 'method', 'variant', 'pools'])


class Workspace:
    def __init__(self, root):
        self.root = ensure_dir(root)
        self.splits = ensure_dir(self.root / 'splits')
        self.cache = ensure_dir(self.root / 'cache')
        self.manifests = ensure_dir(self.root / 'manifests')
        self.verify = ensure_dir(self.root / 'verify')
        self.probe = ensure_dir(self.root / 'probe')
        self.tune = ensure_dir(self.root / 'tune')
        self.checkpoints = ensure_dir(self.root / 'checkpoints')
        self.train_logs = ensure_dir(self.root / 'train_logs')
        self.records = ensure_dir(self.root / 'records')
        self.manyway = ensure_dir(self.root / 'manyway')
        self.perturb = ensure_dir(self.root / 'perturb')
        self.profile = ensure_dir(self.root / 'profile')
        self.qualitative = ensure_dir(self.root / 'qualitative')
        self.metrics = ensure_dir(self.root / 'metrics')
        self.report = ensure_dir(self.root / 'report')

    def split_path(self, dataset):
        return self.splits / f'{dataset}.csv'

    def cache_paths(self, dataset, size):
        return self.cache / f'{dataset}_{size}_images.npy', self.cache / f'{dataset}_{size}_boxes.npy'

    def bank_path(self, dataset, pool, n_way, k_shot, n_query, n_episodes):
        return self.manifests / f'{dataset}_{pool}_{n_way}way{k_shot}shot{n_query}q_{n_episodes}.npz'

    def run_dir(self, run):
        return self.root / run.backbone / run.dataset / f'{run.shot}shot' / run_tag(run.method, dict(run.variant))

    def checkpoint(self, run, seed):
        return self.checkpoints / self.run_dir(run).relative_to(self.root) / f'seed{seed}.pt'

    def train_log(self, run, seed):
        return self.train_logs / self.run_dir(run).relative_to(self.root) / f'seed{seed}.csv'

    def record(self, run, pool, seed):
        return self.records / self.run_dir(run).relative_to(self.root) / pool / f'seed{seed}.npz'


def dataset_root(args, dataset):
    explicit = getattr(args, f'{dataset}_path')
    return explicit or os.path.join(args.data_root, DATASET_DIRS[dataset])


def load_split(ws, dataset):
    return pd.read_csv(ws.split_path(dataset))


def pool_rows(frame, pool):
    return np.flatnonzero(frame['pool'].to_numpy() == pool)


def get_bank(ws, args, frame, dataset, pool, n_way, k_shot, n_query, n_episodes):
    path = ws.bank_path(dataset, pool, n_way, k_shot, n_query, n_episodes)
    if path.exists():
        return EpisodeBank.load(path)
    source = 'tune' if pool == 'select' else pool
    seed = stable_hash('manifest', dataset, pool, n_way, k_shot, n_query, n_episodes, args.seed)
    bank = make_bank(frame['class_id'].to_numpy(), pool_rows(frame, source), n_way, k_shot, n_query, n_episodes, seed)
    bank.save(path)
    return bank


def eval_spec(args, pool):
    if pool == 'select':
        return args.n_way, args.test_queries, args.val_episodes
    return args.n_way, args.test_queries, args.test_episodes


def pool_bank(ws, args, frame, dataset, pool, shot):
    n_way, n_query, n_episodes = eval_spec(args, pool)
    return get_bank(ws, args, frame, dataset, pool, n_way, shot, n_query, n_episodes)


class DataCache:
    def __init__(self, ws, args, device):
        self.ws, self.args, self.device = ws, args, device
        self.items = {}

    def get(self, dataset):
        if dataset not in self.items:
            self.items.clear()
            free_memory()
            images_path, boxes_path = self.ws.cache_paths(dataset, self.args.image_size)
            images = np.load(images_path)
            self.items[dataset] = {'frame': load_split(self.ws, dataset), 'images': torch.from_numpy(images).to(self.device),
                                   'numpy': images, 'boxes': np.load(boxes_path)}
        return self.items[dataset]


def experiment_runs(args):
    runs = {}
    datasets = [d for d in args.datasets if d in DATASETS]
    cub_one = 'cub' in datasets and 1 in args.shots

    def add(backbone, dataset, shot, method, variant=None, pools=()):
        key = (backbone, dataset, shot, method, tuple(sorted((variant or {}).items())))
        runs.setdefault(key, set()).update(pools)

    if 'primary' in args.experiments:
        for dataset in datasets:
            for shot in args.shots:
                for method in PRIMARY_METHODS:
                    add('conv64', dataset, shot, method, pools=('test',))
        if cub_one:
            for method in ('A2', 'C2-max', 'A3'):
                add('conv64', 'cub', 1, method, pools=('conf',))
            for method in ('A2', 'A3'):
                add('conv64', 'cub', 1, method, pools=('tune',))
    if 'confirm' in args.experiments and cub_one:
        for method in CONFIRM_METHODS:
            add('conv64', 'cub', 1, method, pools=('conf',))
        add('conv64', 'cub', 1, 'A1', pools=('test',))
    if 'coverage' in args.experiments and cub_one:
        for rho in args.coverage_grid:
            for method in ('A2', 'A3'):
                add('conv64', 'cub', 1, method, None if abs(rho - args.rho) < 1e-12 else {'rho': rho}, pools=('tune',))
    if 'kappa' in args.experiments and cub_one:
        add('conv64', 'cub', 1, 'A3', {'kappa': args.kappa_retrain}, pools=('tune',))
        add('conv64', 'cub', 1, 'A2', {'kappa': args.kappa_retrain}, pools=('tune',))
        add('conv64', 'cub', 1, 'A3', {'entropy': args.small_entropy}, pools=('tune',))
    if 'backbone' in args.experiments and 'cub' in datasets:
        for shot in args.shots:
            for method in BACKBONE_METHODS:
                add('resnet12', 'cub', shot, method, pools=('test',))
    out = [Run(b, d, k, m, v, tuple(sorted(p))) for (b, d, k, m, v), p in runs.items()]
    if args.methods:
        out = [r for r in out if r.method in args.methods]
    return sorted(out, key=lambda r: (r.backbone, DATASETS.index(r.dataset), r.shot, r.method, r.variant))


def base_config(args):
    keys = ('rho', 'entropy', 'lambda_g', 'eta', 't_m', 'kappa', 'shared_offset', 'shared_temp', 'weight_temp', 'candidate_offset',
            'candidate_temp', 'attn_temp', 'toprow_temp', 'anchor_alpha', 'permutation_seed', 'emd_entropy', 'dn4_neighbours',
            'sfmr_radius', 'tau_init', 'ot_tol', 'ot_feas_tol', 'ot_max_iter', 'ot_patience', 'ot_fallback_iter')
    cfg = {k: getattr(args, k) for k in keys}
    cfg['floor'] = args.descriptor_floor
    return cfg


def tuned_params(ws, method):
    selected = load_json_safe(ws.tune / 'selected.json')
    params = dict(selected.get('A2', {}).get('params', {}))
    params.update(selected.get(TUNE_ALIASES.get(method, method), {}).get('params', {}))
    return params


def method_config(args, ws, method, variant=()):
    cfg = base_config(args)
    if args.use_tuned:
        cfg.update(tuned_params(ws, method))
    cfg.update(dict(variant))
    return cfg


def build_run_model(args, ws, run, device):
    from models import build_model
    return build_model(run.method, run.backbone, method_config(args, ws, run.method, run.variant)).to(device)


def load_run_model(args, ws, run, seed, device):
    path = ws.checkpoint(run, seed)
    if not path.exists():
        return None
    model = build_run_model(args, ws, run, device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)['state'])
    return model.eval()


def stage_split(args, ws):
    summary = {}
    for dataset in args.datasets:
        root = dataset_root(args, dataset)
        if not os.path.exists(root):
            log(f'split: {dataset} root {root} not found, skipping')
            continue
        frame = load_dataset(dataset, root, args.split_seed, args.split_dir)
        frame.to_csv(ws.split_path(dataset), index=False)
        summary[dataset] = dataset_summary(frame)
        images_path, boxes_path = ws.cache_paths(dataset, args.image_size)
        if args.overwrite or not images_path.exists() or not boxes_path.exists():
            build_cache(frame, args.image_size, images_path, boxes_path)
        pools = ['test', 'tune', 'select'] + (['conf'] if dataset == 'cub' else [])
        for shot in args.shots:
            for pool in pools:
                pool_bank(ws, args, frame, dataset, pool, shot)
        log(f'split: {dataset}: {summary[dataset]["n_images"]} images, ' +
            ', '.join(f'{p} {v["n_classes"]} classes' for p, v in summary[dataset]['pools'].items()))
    save_json(summary, ws.splits / 'dataset_summary.json')


def search_space(method, args):
    if method == 'A2':
        return {'entropy': args.entropy_grid}
    if method == 'A0':
        return {'attn_temp': args.attn_temp_grid}
    if method == 'C3':
        return {'t_m': args.tm_grid, 'toprow_temp': args.attn_temp_grid}
    space = {'lambda_g': args.lambda_grid, 't_m': args.tm_grid}
    if method == 'A3':
        space['eta'] = args.eta_grid
    elif method == 'C1':
        space.update({'candidate_offset': args.candidate_offset_grid, 'candidate_temp': args.eta_grid})
    else:
        space.update({'shared_offset': args.offset_grid, 'shared_temp': args.shared_temp_grid})
        if method == 'C2-weighted':
            space['weight_temp'] = args.weight_temp_grid
    return space


def tune_trials(method, args):
    space = search_space(method, args)
    combos = [dict(zip(space, values)) for values in itertools.product(*space.values())]
    if method == 'A2' or len(combos) <= args.tune_trials:
        return combos
    rng = np.random.default_rng(stable_hash('tune-trials', method, args.seed))
    return [combos[i] for i in sorted(rng.choice(len(combos), args.tune_trials, replace=False))]


def stage_tune(args, ws):
    from models import build_model
    device = resolve_device(args.device)
    data = DataCache(ws, args, device).get(args.tune_dataset)
    frame = data['frame']
    shot = args.tune_shot
    select = pool_bank(ws, args, frame, args.tune_dataset, 'select', shot)
    stream = EpisodeStream(frame['class_id'].to_numpy(), pool_rows(frame, 'train'), args.n_way, shot, args.train_queries, args.tune_seed,
                           f'{args.tune_dataset}-{shot}')
    trials_path = ws.tune / 'trials.csv'
    done = pd.read_csv(trials_path) if trials_path.exists() and not args.overwrite else pd.DataFrame()
    rows = done.to_dict('records')
    selected = {}
    episodes = args.train_episodes
    args.train_episodes = args.tune_episodes
    for method in ['A2'] + [m for m in args.tune_methods if m != 'A2']:
        base = dict(selected.get('A2', {}).get('params', {})) if method != 'A2' else {}
        for i, params in enumerate(tune_trials(method, args)):
            key = f'{method}|{sorted({**base, **params}.items())}'
            previous = [r for r in rows if r['key'] == key]
            if not previous:
                cfg = {**base_config(args), **base, **params}
                set_seed(args.tune_seed)
                model = build_model(method, 'conv64', cfg).to(device)
                best, _ = train_model(model, data['images'], stream, select, args, device, f'tune {method} trial {i + 1}')
                rows.append({'key': key, 'method': method, 'trial': i + 1, 'params': str({**base, **params}), 'val_acc': best['acc'],
                             'best_episode': best['episode']})
                pd.DataFrame(rows).to_csv(trials_path, index=False)
                del model
                free_memory()
        trials = [r for r in rows if r['method'] == method]
        winner = max(trials, key=lambda r: r['val_acc'])
        selected[method] = {'params': ast.literal_eval(winner['params']), 'val_acc': winner['val_acc'], 'trials': len(trials)}
        log(f'tune: {method} selected {selected[method]["params"]} at {winner["val_acc"]:.2f}%')
    args.train_episodes = episodes
    save_json(selected, ws.tune / 'selected.json')


def stage_train(args, ws):
    device = resolve_device(args.device)
    cache = DataCache(ws, args, device)
    for run in experiment_runs(args):
        pending = [s for s in args.seeds if args.overwrite or not ws.checkpoint(run, s).exists()]
        if not pending:
            continue
        data = cache.get(run.dataset)
        frame = data['frame']
        select = pool_bank(ws, args, frame, run.dataset, 'select', run.shot)
        for seed in pending:
            name = f'{run.backbone}/{run.dataset}/{run.shot}-shot/{run_tag(run.method, dict(run.variant))}/seed{seed}'
            stream = EpisodeStream(frame['class_id'].to_numpy(), pool_rows(frame, 'train'), args.n_way, run.shot, args.train_queries, seed,
                                   f'{run.dataset}-{run.shot}')
            set_seed(seed)
            model = build_run_model(args, ws, run, device)
            best, history = train_model(model, data['images'], stream, select, args, device, name)
            path = ws.checkpoint(run, seed)
            ensure_dir(path.parent)
            torch.save({'state': best['state'], 'method': run.method, 'backbone': run.backbone, 'variant': dict(run.variant),
                        'config': method_config(args, ws, run.method, run.variant), 'best_val': best['acc'], 'best_episode': best['episode'],
                        'seed': seed}, path)
            ensure_dir(ws.train_log(run, seed).parent)
            pd.DataFrame(history).to_csv(ws.train_log(run, seed), index=False)
            del model
            free_memory()


def stage_infer(args, ws):
    from models import ALLOCATION
    device = resolve_device(args.device)
    cache = DataCache(ws, args, device)
    for run in experiment_runs(args):
        for pool in run.pools:
            pending = [s for s in args.seeds if ws.checkpoint(run, s).exists() and (args.overwrite or not ws.record(run, pool, s).exists())]
            if not pending:
                continue
            data = cache.get(run.dataset)
            bank = pool_bank(ws, args, data['frame'], run.dataset, pool, run.shot)
            diagnostics = (run.backbone == 'conv64' and run.dataset == 'cub' and run.shot == 1 and pool == 'test' and run.method in DIAG_METHODS
                           and not run.variant)
            terms = pool == 'tune' and run.method in ALLOCATION
            for seed in tqdm(pending, desc=f'infer {run.backbone}/{run.dataset}/{run.shot}/{run_tag(run.method, dict(run.variant))}/{pool}'):
                model = load_run_model(args, ws, run, seed, device)
                out = evaluate_bank(model, data['images'], bank, device, diagnostics=diagnostics, terms=terms)
                path = ws.record(run, pool, seed)
                ensure_dir(path.parent)
                np.savez_compressed(path, query=bank.query.reshape(len(bank), -1), **out)
                del model
                free_memory()


@torch.no_grad()
def timed_bank(model, images, bank, device):
    model.eval()
    n_way, _, n_query = bank.shape
    labels = query_labels(n_way, n_query, device)
    acc, ms = np.zeros(len(bank), dtype=np.float32), np.zeros(len(bank), dtype=np.float32)
    for e in range(len(bank)):
        support, query = bank.episode(e)
        s, q = episode_tensors(images, support, query, device)
        synchronize(device)
        start = time.perf_counter()
        logits = model(s, q)['logits']
        synchronize(device)
        ms[e] = 1000.0 * (time.perf_counter() - start)
        acc[e] = float((logits.argmax(-1) == labels).float().mean())
    return acc, ms


def followup_run(method, shot=1):
    return Run('conv64', 'cub', shot, method, (), ('test',))


def stage_manyway(args, ws):
    device = resolve_device(args.device)
    data = DataCache(ws, args, device).get('cub')
    for ways in args.ways:
        bank = get_bank(ws, args, data['frame'], 'cub', 'test', ways, 1, 1, args.manyway_episodes)
        for method in args.followup_methods:
            for seed in args.seeds:
                path = ws.manyway / f'{method}_N{ways}_seed{seed}.npz'
                if path.exists() and not args.overwrite:
                    continue
                model = load_run_model(args, ws, followup_run(method), seed, device)
                if model is None:
                    continue
                acc, ms = timed_bank(model, data['images'], bank, device)
                np.savez_compressed(path, acc=acc, ms=ms, ways=ways)
                del model
        log(f'manyway: {ways}-way done')


def perturbed_queries(data, bank, e, condition, args):
    _, query = bank.episode(e)
    flat = query.reshape(-1)
    family, _ = parse_condition(condition)
    images, boxes, frame = data['numpy'], data['boxes'], data['frame']
    out = np.empty((len(flat),) + images.shape[1:], dtype=np.uint8)
    episode_classes = set(int(c) for c in bank.classes[e])
    donors = None
    if family == 'background':
        donors = np.flatnonzero(~frame['class_id'].isin(episode_classes).to_numpy())
    for j, idx in enumerate(flat):
        rng = np.random.default_rng(stable_hash('perturb', condition, e, j, args.seed))
        donor = int(rng.choice(donors)) if donors is not None else None
        out[j] = apply_condition(images[idx], condition, args, rng=rng, box=boxes[idx],
                                 donor=images[donor] if donor is not None else None, donor_box=boxes[donor] if donor is not None else None)
    return torch.from_numpy(out)


def stage_perturb(args, ws):
    device = resolve_device(args.device)
    data = DataCache(ws, args, device).get('cub')
    bank = pool_bank(ws, args, data['frame'], 'cub', 'test', 1)
    models = {}
    for method in args.followup_methods:
        for seed in args.seeds:
            model = load_run_model(args, ws, followup_run(method), seed, device)
            if model is not None:
                models[(method, seed)] = model
    if not models:
        log('perturb: no CUB 1-shot checkpoints found')
        return
    labels = query_labels(args.n_way, args.test_queries, device)
    for condition in conditions(args):
        path = ws.perturb / f'{condition}.npz'
        if path.exists() and not args.overwrite:
            continue
        acc = {key: np.zeros(len(bank), dtype=np.float32) for key in models}
        for e in tqdm(range(len(bank)), desc=f'perturb {condition}'):
            support, query = bank.episode(e)
            perturbed = None if condition == 'clean' else perturbed_queries(data, bank, e, condition, args)
            s, q = episode_tensors(data['images'], support, query, device, query_images=perturbed)
            with torch.no_grad():
                for key, model in models.items():
                    acc[key][e] = float((model(s, q)['logits'].argmax(-1) == labels).float().mean())
        np.savez_compressed(path, **{f'{m}__seed{s}': v for (m, s), v in acc.items()})


def profile_inputs(ws, args, data, shot, device):
    bank = pool_bank(ws, args, data['frame'], 'cub', 'test', shot)
    return [episode_tensors(data['images'], *bank.episode(e), device) for e in range(min(args.profile_episodes, len(bank)))]


def peak_reset(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device):
    return torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else float('nan')


def train_step_ms(model, args, data, ws, shot, device, episodes):
    from models import count_parameters
    frame = data['frame']
    stream = EpisodeStream(frame['class_id'].to_numpy(), pool_rows(frame, 'train'), args.n_way, shot, args.train_queries, 0, 'profile')
    optimizer, _ = make_optimizer(model, args, args.train_episodes)
    labels = query_labels(args.n_way, args.train_queries, device)
    times = []
    for t in range(episodes):
        model.train()
        s, q = episode_tensors(data['images'], *stream.episode(t), device, stream.generator(t), args.jitter)
        synchronize(device)
        start = time.perf_counter()
        loss = F.cross_entropy(model(s, q)['logits'], labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        synchronize(device)
        times.append(1000.0 * (time.perf_counter() - start))
    return float(np.median(times)), count_parameters(model)


def stage_profile(args, ws):
    from models import ALLOCATION
    device = resolve_device(args.device)
    data = DataCache(ws, args, device).get('cub')
    timing_rows, stage_rows, model_rows = [], [], []
    for backbone, methods in (('conv64', PROFILE_METHODS), ('resnet12', BACKBONE_METHODS)):
        methods = [m for m in methods if not args.methods or m in args.methods]
        for shot in args.shots:
            inputs = profile_inputs(ws, args, data, shot, device)
            for seed in args.seeds:
                models = {}
                for method in methods:
                    model = load_run_model(args, ws, Run(backbone, 'cub', shot, method, (), ('test',)), seed, device)
                    if model is not None:
                        models[method] = model
                if not models:
                    continue
                with torch.no_grad():
                    for method, model in models.items():
                        for w in range(args.profile_warmup):
                            model(*inputs[w % len(inputs)])
                    order = list(models)
                    for p in range(args.profile_passes):
                        for e, (s, q) in enumerate(inputs):
                            rotated = order[(p + e) % len(order):] + order[:(p + e) % len(order)]
                            for method in rotated:
                                peak_reset(device)
                                synchronize(device)
                                start = time.perf_counter()
                                out = models[method](s, q)
                                synchronize(device)
                                ms = 1000.0 * (time.perf_counter() - start)
                                info = out.get('info')
                                sweeps = info['sweeps'].flatten().float() if info is not None else None
                                timing_rows.append({'backbone': backbone, 'shot': shot, 'method': method, 'seed': seed, 'pass': p, 'episode': e,
                                                    'ms': ms, 'peak_mb': peak_mb(device),
                                                    'sweeps_median': float(sweeps.median()) if sweeps is not None else np.nan,
                                                    'sweeps_p95': float(torch.quantile(sweeps, 0.95)) if sweeps is not None else np.nan,
                                                    'sweeps_max': float(sweeps.max()) if sweeps is not None else np.nan,
                                                    'cap_rate': float(info['cap_hit'].float().mean()) if info is not None else np.nan})
                    for method in [m for m in STAGE_METHODS if m in models]:
                        for e, (s, q) in enumerate(inputs):
                            timer = StageTimer(device)
                            timer.start()
                            models[method](s, q, timer=timer)
                            stage_rows.append({'backbone': backbone, 'shot': shot, 'method': method, 'seed': seed, 'episode': e, **timer.times})
                if seed == args.seeds[0]:
                    for method, model in models.items():
                        train_ms, params = train_step_ms(model, args, data, ws, shot, device, len(inputs))
                        model_rows.append({'backbone': backbone, 'shot': shot, 'method': method, 'params': params, 'train_ms': train_ms})
                del models
                free_memory()
            log(f'profile: {backbone} {shot}-shot done')
    pd.DataFrame(timing_rows).to_csv(ws.profile / 'timings.csv', index=False)
    pd.DataFrame(stage_rows).to_csv(ws.profile / 'stages.csv', index=False)
    pd.DataFrame(model_rows).to_csv(ws.profile / 'models.csv', index=False)
    save_json({'device': str(device), 'cuda': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
               'torch': torch.__version__, 'warmup': args.profile_warmup, 'passes': args.profile_passes, 'episodes': args.profile_episodes,
               'seeds': args.seeds}, ws.profile / 'profile_meta.json')


def resnet50_features(size, device):
    import torchvision
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    body = torch.nn.Sequential(*list(model.children())[:-2]).to(device).eval()
    return body


def load_square(path, size):
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((size, size), Image.BILINEAR))


def stage_qualitative(args, ws):
    from models import AllocationHead, ALLOCATION
    device = resolve_device(args.device)
    body = resnet50_features(args.qual_size, device)
    heads = {m: AllocationHead({**base_config(args), **ALLOCATION[m]}).to(device) for m in ('A2', 'A3')}
    rows, arrays = [], {}
    for dataset in args.datasets:
        if not ws.split_path(dataset).exists():
            continue
        frame = load_split(ws, dataset)
        bank = get_bank(ws, args, frame, dataset, 'test', args.n_way, 1, 1, args.qual_episodes)
        cache = {}

        def features(indices):
            missing = [i for i in indices if i not in cache]
            if missing:
                batch = np.stack([load_square(frame['path'].iloc[i], args.qual_size) for i in missing])
                x = normalize(torch.from_numpy(batch).to(device).permute(0, 3, 1, 2).float() / 255.0)
                with torch.no_grad():
                    for i, f in zip(missing, body(x)):
                        cache[i] = f
            return torch.stack([cache[i] for i in indices])

        for e in tqdm(range(len(bank)), desc=f'qualitative {dataset}'):
            support, query = bank.episode(e)
            fs = features(support.reshape(-1).tolist()).unsqueeze(1)
            fq = features(query.reshape(-1).tolist())
            with torch.no_grad():
                outs = {m: head(fq, fs, need_plans=True) for m, head in heads.items()}
            n = fq.shape[0]
            for j in range(n):
                part = {}
                for m, out in outs.items():
                    plan = out['plan'][j, j, 0]
                    part[m] = (plan.sum(-1) * plan.shape[0]).cpu().numpy()
                    rows_m = int(out['score'][j].argmax())
                    part[f'{m}_pred'] = rows_m
                key = f'{dataset}_{e}_{j}'
                arrays[f'{key}_A2'], arrays[f'{key}_A3'] = part['A2'], part['A3']
                rows.append({'dataset': dataset, 'episode': e, 'query_pos': j, 'key': key, 'query_index': int(query.reshape(-1)[j]),
                             'support_index': int(support[j, 0]), 'label': j, 'pred_A2': part['A2_pred'], 'pred_A3': part['A3_pred'],
                             'change': float(np.abs(part['A3'] - part['A2']).sum()), 'classes': ','.join(str(int(c)) for c in bank.classes[e])})
    frame = pd.DataFrame(rows)
    frame.to_csv(ws.qualitative / 'episodes.csv', index=False)
    np.savez_compressed(ws.qualitative / 'participation.npz', **arrays)
    picks = []
    for dataset in args.datasets:
        part = frame[(frame['dataset'] == dataset) & (frame['pred_A2'] == frame['label']) & (frame['pred_A3'] == frame['label'])]
        if not part.empty:
            picks.append({**part.loc[part['change'].idxmax()].to_dict(), 'row': 'correct'})
    failure = frame[(frame['dataset'] == 'cub') & (frame['pred_A2'] == frame['label']) & (frame['pred_A3'] != frame['label'])]
    if not failure.empty:
        picks.append({**failure.loc[failure['change'].idxmax()].to_dict(), 'row': 'failure'})
    save_json({'picks': picks, 'size': args.qual_size, 'encoder': 'ResNet-50 (ImageNet-1k V2)'}, ws.qualitative / 'selection.json')
    log(f'qualitative: {len(picks)} examples selected from {len(frame)} queries')
