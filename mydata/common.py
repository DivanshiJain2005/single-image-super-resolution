import random
import numpy as np
import torch


def get_patch(lr, hr, patch_size=48, scale=2):
    """Crop LR-HR patch for training"""
    ih, iw = lr.shape[:2]          # LR image height/width
    tp = patch_size               # LR patch size
    ip = tp * scale               # HR patch size

    # random top-left corner
    ix = np.random.randint(0, iw - tp + 1)
    iy = np.random.randint(0, ih - tp + 1)

    # crop patches
    lr_patch = lr[iy:iy + tp, ix:ix + tp, ...]
    hr_patch = hr[iy * scale:iy * scale + ip,
                  ix * scale:ix * scale + ip, ...]

    return lr_patch, hr_patch


def _set_channel(img, n_colors=3):
    """Ensure image has correct number of channels (1 or 3)"""
    img = np.array(img)

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    if n_colors == 1 and img.shape[2] == 3:
        img = np.expand_dims(
            0.299 * img[:, :, 0] +
            0.587 * img[:, :, 1] +
            0.114 * img[:, :, 2],
            axis=2
        )

    elif n_colors == 3 and img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=2)

    return img


def set_channel(images, n_colors):
    return [_set_channel(img, n_colors) for img in images]


def _np2Tensor(img, rgb_range=255):
    """Convert HWC NumPy image to CHW torch Tensor"""
    img = np.array(img, dtype=np.float32)
    tensor = torch.from_numpy(
        np.ascontiguousarray(img.transpose((2, 0, 1)))
    )
    return tensor.div(rgb_range)


def np2Tensor(images, rgb_range=255):
    return [_np2Tensor(img, rgb_range) for img in images]


def _augment(img):
    img = np.array(img)

    # horizontal flip
    if random.random() < 0.5:
        img = img[:, ::-1, :]

    # vertical flip
    if random.random() < 0.5:
        img = img[::-1, :, :]

    # rotation
    if random.random() < 0.5:
        img = img.transpose(1, 0, 2)

    return img


def augment(images):
    """Apply random augmentation to list of images"""
    return [_augment(img) for img in images]


def add_noise(img, noise_type=None, sigma=5):
    """Add noise to image"""
    if noise_type is None or noise_type.lower() == 'none':
        return img

    img = np.array(img, dtype=np.float32)

    if noise_type.lower() == 'gaussian':
        noise = np.random.normal(0, sigma, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    return img
