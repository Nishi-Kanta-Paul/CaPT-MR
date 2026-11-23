import io

import numpy as np
from PIL import Image
from scipy import ndimage


FAMILIES = ('blur', 'jpeg', 'brightness', 'contrast', 'noise', 'background')
FAMILY_NAMES = {'blur': 'Gaussian blur', 'jpeg': 'JPEG', 'brightness': 'Brightness', 'contrast': 'Contrast',
                'noise': 'Gaussian noise', 'background': 'Background replacement'}


def severity_levels(args):
    return {'blur': args.blur_sigmas, 'jpeg': args.jpeg_qualities, 'brightness': args.brightness_shifts,
            'contrast': args.contrast_factors, 'noise': args.noise_sigmas, 'background': args.background_weights}


def conditions(args):
    names = ['clean']
    for family, levels in severity_levels(args).items():
        names += [f'{family}-{h + 1}' for h in range(len(levels))]
    return names


def parse_condition(name):
    if name == 'clean':
        return 'clean', 0
    family, level = name.rsplit('-', 1)
    return family, int(level)


def gaussian_blur(image, sigma):
    return np.clip(ndimage.gaussian_filter(image.astype(np.float32), sigma=(sigma, sigma, 0), mode='reflect'), 0, 255)


def jpeg(image, quality):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='JPEG', quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert('RGB'), dtype=np.float32)


def brightness(image, shift):
    return np.clip(image.astype(np.float32) + 255.0 * shift, 0, 255)


def contrast(image, factor):
    x = image.astype(np.float32)
    mean = x.mean()
    return np.clip((x - mean) * factor + mean, 0, 255)


def gaussian_noise(image, sigma, rng):
    return np.clip(image.astype(np.float32) + 255.0 * sigma * rng.standard_normal(image.shape), 0, 255)


def outside_strip(box, size):
    if not np.all(np.isfinite(box)):
        return (0, 0, size, size)
    x0, y0, x1, y1 = (int(round(v)) for v in np.clip(box, 0, size))
    strips = [(0, 0, x0, size), (x1, 0, size, size), (0, 0, size, y0), (0, y1, size, size)]
    areas = [(s[2] - s[0]) * (s[3] - s[1]) for s in strips]
    best = strips[int(np.argmax(areas))]
    if areas[int(np.argmax(areas))] <= 0:
        return (0, 0, size, size)
    return best


def background_replacement(image, box, donor, donor_box, weight):
    size = image.shape[0]
    x0, y0, x1, y1 = outside_strip(donor_box, size)
    texture = np.asarray(Image.fromarray(donor[y0:y1, x0:x1]).resize((size, size), Image.BILINEAR), dtype=np.float32)
    outside = np.ones((size, size), dtype=bool)
    if np.all(np.isfinite(box)):
        bx0, by0, bx1, by1 = (int(round(v)) for v in np.clip(box, 0, size))
        outside[by0:by1, bx0:bx1] = False
    out = image.astype(np.float32)
    out[outside] = (1.0 - weight) * out[outside] + weight * texture[outside]
    return np.clip(out, 0, 255)


def apply_condition(image, name, args, rng=None, box=None, donor=None, donor_box=None):
    family, level = parse_condition(name)
    if family == 'clean':
        return image
    value = severity_levels(args)[family][level - 1]
    if family == 'blur':
        out = gaussian_blur(image, value)
    elif family == 'jpeg':
        out = jpeg(image, value)
    elif family == 'brightness':
        out = brightness(image, value)
    elif family == 'contrast':
        out = contrast(image, value)
    elif family == 'noise':
        out = gaussian_noise(image, value, rng)
    else:
        out = background_replacement(image, box, donor, donor_box, value)
    return np.round(out).astype(np.uint8)
