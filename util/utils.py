import os
import gc
import json
import random
import hashlib
import datetime
from pathlib import Path

import numpy as np
import torch


def log(msg):
    stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{stamp}] {msg}', flush=True)


def set_seed(seed):
    seed = int(seed) % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_hash(*parts):
    key = '|'.join(str(p) for p in parts)
    return int(hashlib.sha256(key.encode('utf-8')).hexdigest()[:15], 16)


def resolve_device(name):
    if name.startswith('cuda') and not torch.cuda.is_available():
        log('CUDA is not available, using CPU')
        return torch.device('cpu')
    return torch.device(name)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'Cannot serialize {type(obj)}')


def clean_floats(obj):
    if isinstance(obj, dict):
        return {str(k): clean_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_floats(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, 'w') as f:
        json.dump(clean_floats(obj), f, indent=2, default=to_serializable)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_json_safe(path):
    return load_json(path) if os.path.exists(path) else {}


def quantiles(values, qs=(0.25, 0.5, 0.75, 0.95)):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f'q{int(round(q * 100))}': float('nan') for q in qs}
    return {f'q{int(round(q * 100))}': float(np.quantile(values, q)) for q in qs}


def fmt_value(value, digits):
    return f'{float(value):g}' if digits is None else f'{float(value):.{digits}f}'


def run_tag(method, variant):
    if not variant:
        return method
    return method + '_' + '_'.join(f'{k}{fmt_value(v, None)}' for k, v in sorted(variant.items()))
