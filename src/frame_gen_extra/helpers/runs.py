"""Read back the runs that `00_data_preprocess.ipynb` writes.

A run is `frames_per_run` consecutive frames of one spot of the stream, as the client
decoded them: the 64 px core plus `halo` pixels of real context on every side, the way a
tile is fed to a model in the browser. The stored halo (16) is wider than the client's (4)
so a notebook can measure what more context is worth; `halo` cuts it down.

A sample is `context` frames in a row and the frame `ahead` frames after the newest of them:
from frames t-context+1..t, predict frame t+ahead. While training t is random; for
validation it is fixed per run (spread over the run by its index), so scores are repeatable.
A model's output is the size of its input, and `trim` cuts the halo off before the loss and
the scores.

Most of a desktop stands still, and repeating the newest frame is perfect there, so a score
over every sample flatters everything. `changed_share` is the one definition of a pixel
that changed, and `groups` splits the samples into the ones worth reporting.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

CHANGED_LEVEL = 8        # a pixel changed if its luma moved by more than this (of 255) - over codec noise
MOVING_SHARE = 0.005     # a sample is "changed" if this share of its core's pixels changed (20 of 64x64)
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def trim(images, pixels, channels_last=False):
    """Cut `pixels` from every edge of (..., H, W) images, or (..., H, W, C) ones."""
    if not pixels:
        return images
    if channels_last:
        return images[..., pixels:images.shape[-3] - pixels, pixels:images.shape[-2] - pixels, :]
    return images[..., pixels:images.shape[-2] - pixels, pixels:images.shape[-1] - pixels]


def changed_share(a, b, level=CHANGED_LEVEL):
    """The share of pixels whose luma differs by more than `level` between uint8 (..., H, W, 3)
    pictures, one value per picture."""
    difference = (a.astype(np.float32) - b.astype(np.float32)) @ LUMA
    return (np.abs(difference) > level).mean(axis=(-2, -1))


def _frames(directory, split, manifest):
    """A split's runs, memory mapped (read from disk only when used)."""
    return np.load(directory / manifest["splits"][split]["frames"], mmap_mode="r")


def _cut(manifest, halo):
    """How many of the stored halo pixels to cut away to leave `halo` (None = keep all)."""
    stored = manifest["halo"]
    halo = stored if halo is None else halo
    if not 0 <= halo <= stored:
        raise ValueError(f"halo {halo} is outside the 0..{stored} the set was made with")
    return stored - halo


def newest(count, frames, context, ahead):
    """The index of the newest input frame of each of `count` validation samples."""
    choices = frames - context - ahead + 1
    if choices < 1:
        raise ValueError(f"{context} frames of context and {ahead} ahead do not fit a run of {frames}")
    return context - 1 + np.arange(count) % choices


def load(directory, split, manifest, context=2, ahead=1, halo=None, count=None):
    """The validation samples as uint8 arrays: inputs (N, context, h, w, 3), oldest first, and
    targets (N, h, w, 3). The same frames every call."""
    runs = _frames(directory, split, manifest)[:count]
    cut = _cut(manifest, halo)
    t = newest(len(runs), runs.shape[1], context, ahead)
    inputs = np.stack([runs[i, n - context + 1:n + 1] for i, n in enumerate(t)])
    targets = np.stack([runs[i, n + ahead] for i, n in enumerate(t)])
    return trim(inputs, cut, channels_last=True), trim(targets, cut, channels_last=True)


def groups(manifest, split, current, target, halo):
    """Boolean masks over a split's samples, by name: `all`; `changed` and `unchanged` (by
    `MOVING_SHARE`, measured on the core between the newest input and the target); `cursor`
    (runs cut around the cursor); and one per kind of source (page, desktop, photo, video)."""
    sources = manifest["splits"][split]["sources"]
    count = len(current)
    moving = changed_share(trim(current, halo, True), trim(target, halo, True)) >= MOVING_SHARE
    masks = {"all": np.ones(count, bool), "changed": moving, "unchanged": ~moving,
             "cursor": np.zeros(count, bool)}
    for source in sources:
        first, end = source["runs"]
        end = min(end, count)
        if first >= end:
            continue
        kind = masks.setdefault(source["kind"], np.zeros(count, bool))
        kind[first:end] = True
        tags = [spot[2] for spot in source["spots"]][:end - first]
        masks["cursor"][first:end] |= np.array(tags) == "cursor"
    return masks


class Runs(Dataset):
    """((context frames...), target) float tensors in [0, 1]. With `augment`, a random newest
    frame and a random horizontal flip.

    No rotation: a desktop scrolls vertically far more than sideways, and a model should
    learn that rather than have it averaged away.
    """

    def __init__(self, directory, split, manifest, augment, context=2, ahead=1, halo=None):
        self.split = split
        self.sources = manifest["splits"][split]["sources"]
        self.runs = _frames(directory, split, manifest)
        self.cut = _cut(manifest, halo)
        self.halo = manifest["halo"] - self.cut
        self.context, self.ahead = context, ahead
        self.augment = augment
        self.fixed = newest(len(self.runs), self.runs.shape[1], context, ahead)

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, index):
        frames = self.runs.shape[1]
        if self.augment:
            t = int(torch.randint(self.context - 1, frames - self.ahead, ()))
        else:
            t = int(self.fixed[index])
        clip = trim(np.asarray(self.runs[index, t - self.context + 1:t + self.ahead + 1]), self.cut,
                    channels_last=True)
        if self.augment and int(torch.randint(0, 2, ())):
            clip = clip[:, :, ::-1]
        inputs = tuple(_to_tensor(frame) for frame in clip[:self.context])
        return inputs, _to_tensor(clip[-1])


def _to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
