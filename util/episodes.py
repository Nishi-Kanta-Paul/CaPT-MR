import numpy as np
import torch

from .datasets import MEAN, STD
from .utils import stable_hash


class EpisodeBank:
    def __init__(self, classes, support, query):
        self.classes = np.asarray(classes, dtype=np.int32)
        self.support = np.asarray(support, dtype=np.int32)
        self.query = np.asarray(query, dtype=np.int32)

    def __len__(self):
        return len(self.classes)

    @property
    def shape(self):
        return self.support.shape[1], self.support.shape[2], self.query.shape[2]

    def episode(self, e):
        return self.support[e], self.query[e]

    def save(self, path):
        np.savez_compressed(path, classes=self.classes, support=self.support, query=self.query)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        return cls(data['classes'], data['support'], data['query'])


def class_index(labels, rows):
    return {int(c): rows[labels[rows] == c] for c in np.unique(labels[rows])}


def sample_episode(rng, by_class, n_way, k_shot, n_query):
    classes = rng.choice(sorted(by_class), n_way, replace=False)
    support = np.zeros((n_way, k_shot), dtype=np.int32)
    query = np.zeros((n_way, n_query), dtype=np.int32)
    for i, c in enumerate(classes):
        pick = rng.choice(by_class[int(c)], k_shot + n_query, replace=False)
        support[i], query[i] = pick[:k_shot], pick[k_shot:]
    return classes, support, query


def make_bank(labels, rows, n_way, k_shot, n_query, n_episodes, seed):
    by_class = class_index(labels, rows)
    small = [c for c, idx in by_class.items() if len(idx) < k_shot + n_query]
    if small:
        raise ValueError(f'{len(small)} classes have fewer than {k_shot + n_query} images')
    if len(by_class) < n_way:
        raise ValueError(f'pool has {len(by_class)} classes, fewer than {n_way} ways')
    rng = np.random.default_rng(seed)
    episodes = [sample_episode(rng, by_class, n_way, k_shot, n_query) for _ in range(n_episodes)]
    return EpisodeBank(*[np.stack(part) for part in zip(*episodes)])


class EpisodeStream:
    def __init__(self, labels, rows, n_way, k_shot, n_query, seed, tag):
        self.by_class = class_index(labels, rows)
        self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query
        self.seed, self.tag = seed, tag

    def episode(self, t):
        rng = np.random.default_rng(stable_hash('train-episode', self.tag, self.seed, t))
        _, support, query = sample_episode(rng, self.by_class, self.n_way, self.k_shot, self.n_query)
        return support, query

    def generator(self, t):
        return torch.Generator().manual_seed(stable_hash('augment', self.tag, self.seed, t) % (2 ** 63 - 1))


def to_float(images, device):
    return images.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x):
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def grayscale(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def augment(x, generator, jitter):
    n = x.shape[0]
    flip = (torch.rand(n, generator=generator) < 0.5).to(x.device)
    factors = (1.0 + (torch.rand(n, 3, generator=generator) * 2.0 - 1.0) * jitter).to(x.device)
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(-1), x)
    x = (x * factors[:, 0].view(n, 1, 1, 1)).clamp(0, 1)
    mean = grayscale(x).mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * factors[:, 1].view(n, 1, 1, 1) + mean).clamp(0, 1)
    gray = grayscale(x)
    return ((x - gray) * factors[:, 2].view(n, 1, 1, 1) + gray).clamp(0, 1)


def episode_tensors(images, support, query, device, generator=None, jitter=0.0, query_images=None):
    n_way, k_shot = support.shape
    idx = torch.as_tensor(np.concatenate([support.reshape(-1), query.reshape(-1)]), dtype=torch.long)
    batch = to_float(images[idx], device)
    if query_images is not None:
        batch[n_way * k_shot:] = to_float(query_images, device)
    if generator is not None:
        batch = augment(batch, generator, jitter)
    batch = normalize(batch)
    s = batch[:n_way * k_shot].view(n_way, k_shot, *batch.shape[1:])
    return s, batch[n_way * k_shot:]


def query_labels(n_way, n_query, device):
    return torch.arange(n_way, device=device).repeat_interleave(n_query)
