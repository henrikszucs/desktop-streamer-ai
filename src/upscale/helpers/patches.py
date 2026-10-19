"""Read back the patch set that `0_data_preprocess.ipynb` writes.

Each patch is a short clip (`frames_per_patch` frames) of one spot of the screen:

- LR: the 64 px core plus `halo` pixels of real context on every side (72 px at halo 4),
  the same way a tile is fed to the model in the browser;
- HR: the 128 px target for the core only.

So a model's output is `halo * scale` pixels too wide on every side, and `trim` cuts it
back before the loss and the scores.

The models upscale one frame at a time, so one sample = one frame of a patch: a random
frame while training, a fixed one for validation (so scores are repeatable).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def trim(images, pixels, channels_last=False):
    """Cut `pixels` from every edge of (..., H, W) images, or (..., H, W, C) ones."""
    if not pixels:
        return images
    if channels_last:
        return images[..., pixels:images.shape[-3] - pixels, pixels:images.shape[-2] - pixels, :]
    return images[..., pixels:images.shape[-2] - pixels, pixels:images.shape[-1] - pixels]


def _arrays(directory, split, manifest):
    """The LR and HR arrays of a split, memory mapped (read from disk only when used)."""
    info = manifest["splits"][split]
    return (np.load(directory / info["lr"], mmap_mode="r"),
            np.load(directory / info["hr"], mmap_mode="r"))


def _cut(manifest, halo):
    """How many of the stored halo pixels to cut away to leave `halo` (None = keep all)."""
    stored = manifest["halo"]
    halo = stored if halo is None else halo
    if not 0 <= halo <= stored:
        raise ValueError(f"halo {halo} is outside the 0..{stored} the set was made with")
    return stored - halo


def load(directory, split, manifest, halo=None, count=None):
    """One frame of each patch as uint8 arrays: LR (N, h, w, 3) and HR (N, 128, 128, 3).

    The frame is the same one validation uses, so two calls give the same pictures.
    """
    lr, hr = _arrays(directory, split, manifest)
    lr, hr = lr[:count], hr[:count]
    frame = np.arange(len(lr)) % lr.shape[1]
    lr = np.stack([lr[i, t] for i, t in enumerate(frame)]) if len(lr) else lr[:, 0]
    hr = np.stack([hr[i, t] for i, t in enumerate(frame)]) if len(hr) else hr[:, 0]
    return trim(lr, _cut(manifest, halo), channels_last=True), hr


class PatchPairs(Dataset):
    """(LR, HR) float tensors in [0, 1]. With `augment`, a random frame, flip and rotation.

    With a `condition` - a function of a source's manifest entry that returns a 1-D array,
    such as `degrade.condition` of its degradation - a sample is ((LR, condition), HR)
    instead, the condition shaped (K, 1, 1) so a batch of them broadcasts over a picture.
    """

    def __init__(self, directory, split, manifest, augment, halo=None, condition=None):
        self.split = split
        self.sources = manifest["splits"][split]["sources"]  # each source's file, motion and degradation
        self.lr, self.hr = _arrays(directory, split, manifest)
        self.cut = _cut(manifest, halo)
        self.halo = manifest["halo"] - self.cut
        self.augment = augment
        self.conditions = None
        if condition is not None:
            # one row per patch: its source's condition
            rows = np.zeros((len(self.lr), len(condition(self.sources[0]))), np.float32)
            for source in self.sources:
                first, end = source["patches"]
                rows[first:end] = condition(source)
            self.conditions = torch.from_numpy(rows)[:, :, None, None]

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, index):
        frames = self.lr.shape[1]
        t = int(torch.randint(0, frames, ())) if self.augment else index % frames
        lr = trim(np.asarray(self.lr[index, t]), self.cut, channels_last=True)
        hr = np.asarray(self.hr[index, t])

        if self.augment:
            # the same flip and rotation on both, or the pair stops matching
            turns = int(torch.randint(0, 4, ()))
            lr, hr = np.rot90(lr, turns), np.rot90(hr, turns)
            if int(torch.randint(0, 2, ())):
                lr, hr = lr[:, ::-1], hr[:, ::-1]

        if self.conditions is not None:
            return (_to_tensor(lr), self.conditions[index]), _to_tensor(hr)
        return _to_tensor(lr), _to_tensor(hr)


def _to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
