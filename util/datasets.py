import os
import glob
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .utils import log, ensure_dir


DATASETS = ('cub', 'dogs', 'cars')
DATASET_NAMES = {'cub': 'CUB-200-2011', 'dogs': 'Stanford Dogs', 'cars': 'Stanford Cars'}
CLASS_COUNTS = {'cub': (100, 50, 50), 'dogs': (70, 20, 30), 'cars': (130, 17, 49)}
COLUMNS = ['dataset', 'image_id', 'path', 'class_id', 'class_name', 'box_x0', 'box_y0', 'box_x1', 'box_y1']
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def find_file(root, name):
    hits = sorted(glob.glob(os.path.join(root, '**', name), recursive=True))
    if not hits:
        raise FileNotFoundError(f'{name} not found under {root}')
    return hits[0]


def find_dir(root, name):
    if os.path.basename(os.path.normpath(root)) == name:
        return root
    hits = sorted(p for p in glob.glob(os.path.join(root, '**', name), recursive=True) if os.path.isdir(p))
    if not hits:
        raise FileNotFoundError(f'{name}/ not found under {root}')
    return hits[0]


def load_cub(root):
    base = os.path.dirname(find_file(root, 'images.txt'))
    images = pd.read_csv(os.path.join(base, 'images.txt'), sep=' ', names=['image_id', 'rel'])
    labels = pd.read_csv(os.path.join(base, 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
    classes = pd.read_csv(os.path.join(base, 'classes.txt'), sep=' ', names=['class_id', 'class_name'])
    boxes = pd.read_csv(os.path.join(base, 'bounding_boxes.txt'), sep=' ', names=['image_id', 'x', 'y', 'w', 'h'])
    frame = images.merge(labels, on='image_id').merge(classes, on='class_id').merge(boxes, on='image_id')
    return pd.DataFrame({
        'dataset': 'cub',
        'image_id': 'cub:' + frame['image_id'].astype(str),
        'path': [os.path.join(base, 'images', r) for r in frame['rel']],
        'class_id': frame['class_id'].astype(int) - 1,
        'class_name': frame['class_name'].str.split('.', n=1).str[-1].str.replace('_', ' '),
        'box_x0': frame['x'].astype(float), 'box_y0': frame['y'].astype(float),
        'box_x1': (frame['x'] + frame['w']).astype(float), 'box_y1': (frame['y'] + frame['h']).astype(float),
    })[COLUMNS]


def dogs_box(path):
    if not os.path.exists(path):
        return (np.nan,) * 4
    obj = ET.parse(path).getroot().find('object')
    if obj is None:
        return (np.nan,) * 4
    box = obj.find('bndbox')
    return tuple(float(box.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax'))


def load_dogs(root):
    image_root = find_dir(root, 'Images')
    annotation_root = os.path.join(os.path.dirname(image_root), 'Annotation')
    rows = []
    synsets = sorted(d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d)))
    for class_id, synset in enumerate(synsets):
        name = synset.split('-', 1)[-1].replace('_', ' ')
        for file in sorted(os.listdir(os.path.join(image_root, synset))):
            stem = os.path.splitext(file)[0]
            box = dogs_box(os.path.join(annotation_root, synset, stem))
            rows.append({'dataset': 'dogs', 'image_id': f'dogs:{stem}', 'path': os.path.join(image_root, synset, file),
                         'class_id': class_id, 'class_name': name, 'box_x0': box[0], 'box_y0': box[1],
                         'box_x1': box[2], 'box_y1': box[3]})
    return pd.DataFrame(rows)[COLUMNS]


def mat_records(path):
    from scipy.io import loadmat
    data = loadmat(path, squeeze_me=True)['annotations']
    return [{name: data[name][i] for name in data.dtype.names} for i in range(data.shape[0])]


def load_cars(root):
    from scipy.io import loadmat
    names = [str(n) for n in loadmat(find_file(root, 'cars_meta.mat'), squeeze_me=True)['class_names']]
    rows = []
    for split, annos in (('train', 'cars_train_annos.mat'), ('test', 'cars_test_annos_withlabels.mat')):
        image_root = find_dir(root, f'cars_{split}')
        for r in mat_records(find_file(root, annos)):
            fname = str(r['fname'])
            class_id = int(r['class']) - 1
            rows.append({'dataset': 'cars', 'image_id': f'cars:{split}:{fname}', 'path': os.path.join(image_root, fname),
                         'class_id': class_id, 'class_name': names[class_id],
                         'box_x0': float(r['bbox_x1']), 'box_y0': float(r['bbox_y1']),
                         'box_x1': float(r['bbox_x2']), 'box_y1': float(r['bbox_y2'])})
    return pd.DataFrame(rows)[COLUMNS]


LOADERS = {'cub': load_cub, 'dogs': load_dogs, 'cars': load_cars}


def class_split(dataset, class_ids, split_seed, split_dir=''):
    classes = np.array(sorted(set(int(c) for c in class_ids)))
    if split_dir:
        table = pd.read_csv(os.path.join(split_dir, f'{dataset}_split.csv'))
        return dict(zip(table['class_id'].astype(int), table['split'].astype(str)))
    n_train, n_val, n_test = CLASS_COUNTS[dataset]
    if len(classes) != n_train + n_val + n_test:
        raise ValueError(f'{dataset}: expected {n_train + n_val + n_test} classes, found {len(classes)}')
    if dataset == 'cub':
        return {int(c): 'train' if c % 2 == 0 else ('val' if c % 4 == 1 else 'test') for c in classes}
    order = np.random.default_rng(split_seed).permutation(classes)
    split = {int(c): 'train' for c in order[:n_train]}
    split.update({int(c): 'val' for c in order[n_train:n_train + n_val]})
    split.update({int(c): 'test' for c in order[n_train + n_val:]})
    return split


def assign_pools(frame, dataset, split_seed):
    frame['pool'] = frame['split']
    val_classes = np.array(sorted(frame.loc[frame['split'] == 'val', 'class_id'].unique()))
    if dataset == 'cub':
        order = np.random.default_rng(split_seed + 1).permutation(val_classes)
        tune = set(int(c) for c in order[:len(order) // 2])
        frame.loc[frame['split'] == 'val', 'pool'] = ['tune' if c in tune else 'conf' for c in frame.loc[frame['split'] == 'val', 'class_id']]
    else:
        frame.loc[frame['split'] == 'val', 'pool'] = 'tune'
    return frame


def load_dataset(dataset, root, split_seed, split_dir=''):
    frame = LOADERS[dataset](root)
    split = class_split(dataset, frame['class_id'], split_seed, split_dir)
    frame['split'] = frame['class_id'].map(split)
    frame = assign_pools(frame, dataset, split_seed)
    return frame.sort_values(['class_id', 'image_id']).reset_index(drop=True)


def dataset_summary(frame):
    out = {'n_images': int(len(frame)), 'n_classes': int(frame['class_id'].nunique()),
           'missing_boxes': int(frame['box_x0'].isna().sum()), 'pools': {}}
    for pool, part in frame.groupby('pool'):
        counts = part.groupby('class_id').size()
        out['pools'][pool] = {'n_classes': int(counts.size), 'n_images': int(len(part)),
                              'min_per_class': int(counts.min()), 'max_per_class': int(counts.max())}
    return out


def build_cache(frame, size, out_images, out_boxes):
    images = np.zeros((len(frame), size, size, 3), dtype=np.uint8)
    boxes = np.full((len(frame), 4), np.nan, dtype=np.float32)
    for i, row in enumerate(tqdm(frame.to_dict('records'), desc=f'cache {frame["dataset"].iloc[0]}')):
        with Image.open(row['path']) as image:
            image = image.convert('RGB')
            width, height = image.size
            images[i] = np.asarray(image.resize((size, size), Image.BILINEAR))
        if np.isfinite(row['box_x0']):
            sx, sy = size / width, size / height
            boxes[i] = (row['box_x0'] * sx, row['box_y0'] * sy, row['box_x1'] * sx, row['box_y1'] * sy)
    ensure_dir(os.path.dirname(out_images))
    np.save(out_images, images)
    np.save(out_boxes, boxes)
    log(f'cache: {len(frame)} images at {size}x{size} written to {out_images}')


def patch_overlap(boxes, size, grid):
    cell = size / grid
    edges = np.arange(grid + 1) * cell
    lo, hi = edges[:-1], edges[1:]
    ox = np.clip(np.minimum(hi[None, :], boxes[:, 2:3]) - np.maximum(lo[None, :], boxes[:, 0:1]), 0, None) / cell
    oy = np.clip(np.minimum(hi[None, :], boxes[:, 3:4]) - np.maximum(lo[None, :], boxes[:, 1:2]), 0, None) / cell
    return (oy[:, :, None] * ox[:, None, :]).reshape(len(boxes), grid * grid)
