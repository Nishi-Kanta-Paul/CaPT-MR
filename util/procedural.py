import numpy as np


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class DescriptorGenerator:
    def __init__(self, seed, separation, noise, rows=25, dim=64, background_rows=10, background_prototypes=4):
        self.rng = np.random.default_rng(seed)
        self.separation, self.noise = separation, noise
        self.rows, self.dim = rows, dim
        self.background_rows = background_rows
        self.parts = rows - background_rows
        self.background_prototypes = background_prototypes

    def image(self, class_parts, background):
        order = self.rng.permutation(self.rows)
        object_rows, background_rows = order[:self.parts], order[self.parts:]
        prototypes = np.empty((self.rows, self.dim))
        prototypes[object_rows] = class_parts[self.rng.permutation(self.parts)]
        prototypes[background_rows] = background[self.rng.integers(len(background), size=len(background_rows))]
        noise = self.noise * self.rng.standard_normal((self.rows, self.dim)) / np.sqrt(self.dim)
        mask = np.zeros(self.rows, dtype=bool)
        mask[object_rows] = True
        return unit(prototypes + noise), mask

    def episode(self, n_way=5, k_shot=1, n_query=1):
        background = unit(self.rng.standard_normal((self.background_prototypes, self.dim)))
        base = unit(self.rng.standard_normal((self.parts, self.dim)))
        offsets = unit(self.rng.standard_normal((n_way, self.parts, self.dim)))
        class_parts = unit(base[None] + self.separation * offsets)
        support = np.zeros((n_way, k_shot, self.rows, self.dim))
        support_object = np.zeros((n_way, k_shot, self.rows), dtype=bool)
        query = np.zeros((n_way * n_query, self.rows, self.dim))
        query_object = np.zeros((n_way * n_query, self.rows), dtype=bool)
        for c in range(n_way):
            for k in range(k_shot):
                support[c, k], support_object[c, k] = self.image(class_parts[c], background)
            for q in range(n_query):
                query[c * n_query + q], query_object[c * n_query + q] = self.image(class_parts[c], background)
        labels = np.repeat(np.arange(n_way), n_query)
        return {'support': support, 'support_object': support_object, 'query': query, 'query_object': query_object, 'labels': labels}

    def bank(self, n_episodes, n_way=5, k_shot=1, n_query=1):
        episodes = [self.episode(n_way, k_shot, n_query) for _ in range(n_episodes)]
        return {
            'support': np.stack([e['support'] for e in episodes]),
            'query': np.concatenate([e['query'] for e in episodes]),
            'query_object': np.concatenate([e['query_object'] for e in episodes]),
            'labels': np.concatenate([e['labels'] for e in episodes]),
            'episode': np.repeat(np.arange(n_episodes), n_way * n_query),
        }


def bank_costs(bank):
    support = bank['support'][bank['episode']]
    return 1.0 - np.einsum('qlc,qnkmc->qnklm', bank['query'], support), support
