"""The patch set `data_preprocess.ipynb` writes, read back for the model notebooks.

Each patch is a short clip: `frames_per_patch` consecutive frames of one spot of the
screen, the crop held still while the content moves under it - what one tile sees at
inference. The LR side is the 64 px core plus `halo` LR pixels of real context on every
side (72 px), the way a tile is fed; the HR side is the 128 px target of the core alone.
So a model's output is `halo * scale` pixels too wide on each side, and `trim` cuts it to
the target: the loss and every score are taken where a tile's output is kept, and the
zero padding at a patch edge never reaches either.

The upscalers here are single frame, so a sample is one frame of a patch: a random one
when training - each epoch sees another - and a fixed one for validation, spread over the
clip by index, so the scores are repeatable and cover the keyframe as often as the clip
holds it. The clips are kept whole for what does need them: a flicker test, a model that
reads the previous frame, frame generation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def trim(images, pixels, channels_last=False):
    """`images` with `pixels` cut from every edge: (..., H, W), or (..., H, W, C)."""
    if not pixels:
        return images
    if channels_last:
        return images[..., pixels:images.shape[-3] - pixels, pixels:images.shape[-2] - pixels, :]
    return images[..., pixels:images.shape[-2] - pixels, pixels:images.shape[-1] - pixels]


def _arrays(directory, split, manifest):
    info = manifest["splits"][split]
    return (np.load(directory / info["lr"], mmap_mode="r"),
            np.load(directory / info["hr"], mmap_mode="r"))


def _context(manifest, halo):
    """How much of the stored halo to cut away to leave `halo` (None: keep all of it)."""
    stored = manifest["halo"]
    halo = stored if halo is None else halo
    if not 0 <= halo <= stored:
        raise ValueError(f"halo {halo} is outside the 0..{stored} the set was made with")
    return stored - halo


def load(directory, split, manifest, halo=None, count=None):
    """One frame of each patch of `split` as uint8 arrays - the first `count` patches, or
    all: LR (N, h, w, 3) with `halo` LR pixels of context (default: all there are), HR
    (N, 128, 128, 3). The frame is the validation one, so two calls agree."""
    lr, hr = _arrays(directory, split, manifest)
    lr, hr = lr[:count], hr[:count]
    cut = _context(manifest, halo)
    pick = np.arange(len(lr)) % lr.shape[1]
    lr = np.stack([lr[i, t] for i, t in enumerate(pick)]) if len(lr) else lr[:, 0]
    hr = np.stack([hr[i, t] for i, t in enumerate(pick)]) if len(hr) else hr[:, 0]
    return trim(lr, cut, channels_last=True), hr


class PatchPairs(Dataset):
    """Patch pairs as float tensors in [0, 1], with the flip/rotate augmentation."""

    def __init__(self, directory, split, manifest, augment, halo=None):
        info = manifest["splits"][split]
        self.split = split
        self.frames = info["frames"]
        self.lr, self.hr = _arrays(directory, split, manifest)
        self.cut = _context(manifest, halo)
        self.halo = manifest["halo"] - self.cut
        self.augment = augment

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, index):
        frames = self.lr.shape[1]
        t = int(torch.randint(0, frames, ())) if self.augment else index % frames
        lr = trim(np.asarray(self.lr[index, t]), self.cut, channels_last=True)
        hr = np.asarray(self.hr[index, t])

        if self.augment:
            turns = int(torch.randint(0, 4, ()))
            lr, hr = np.rot90(lr, turns), np.rot90(hr, turns)
            if int(torch.randint(0, 2, ())):
                lr, hr = lr[:, ::-1], hr[:, ::-1]

        to_tensor = lambda a: torch.from_numpy(
            np.ascontiguousarray(a.transpose(2, 0, 1))).float().div_(255.0)
        return to_tensor(lr), to_tensor(hr)
