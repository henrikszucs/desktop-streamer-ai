"""Read back the runs that `00_data_preprocess.ipynb` writes.

A run is `frames_per_run` consecutive frames of one spot of the stream, as the client
decoded them: the 64 px core plus `halo` pixels of real context on every side, the way a
tile is fed to a model in the browser. The stored halo (16) is wider than the client's (4)
so a notebook can measure what more context is worth; `halo` cuts it down.

A sample is `context` frames `step` apart and the frame `ahead` frames after the newest of
them: from frames t-(context-1)*step, ..., t-step, t, predict frame t+ahead. While training t
is random; for validation it is fixed per run (spread over the run by its index), so scores are
repeatable.

The set is made for 2x frame generation (`manifest["received_every"]` = 2): the stream sends
every second frame of the display rate, and the client makes the one between. Only frames
the client received can be inputs - every second one, from the first of a run - so `step`
defaults to that, and t is always a received frame. The default sample is then the client's:
received frames N-2 and N in, the unsent frame N+1 out. A set without the field (every frame
received) behaves as before: `step` 1, consecutive frames.
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


def received_every(manifest):
    """How many display frames apart the stream's frames are: 2 for 2x frame generation."""
    return int(manifest.get("received_every", 1))


def choices(frames, context, ahead, step, every=1):
    """Every index a sample's newest input can have in a run of `frames`: a received frame,
    with all its context received and its target inside the run."""
    if step % every:
        raise ValueError(f"inputs {step} frames apart are not all received (every {every}th is)")
    found = [t for t in range(frames) if t % every == 0 and t - (context - 1) * step >= 0 and t + ahead < frames]
    if not found:
        raise ValueError(f"{context} frames {step} apart and {ahead} ahead do not fit a run of {frames}")
    return np.array(found)


def newest(count, frames, context, ahead, step=1, every=1):
    """The index of the newest input frame of each of `count` validation samples."""
    options = choices(frames, context, ahead, step, every)
    return options[np.arange(count) % len(options)]


def load(directory, split, manifest, context=2, ahead=1, halo=None, count=None, step=None):
    """The validation samples as uint8 arrays: inputs (N, context, h, w, 3), oldest first, and
    targets (N, h, w, 3). The same frames every call. `step` defaults to the set's
    `received_every`."""
    runs = _frames(directory, split, manifest)[:count]
    cut = _cut(manifest, halo)
    every = received_every(manifest)
    step = every if step is None else step
    t = newest(len(runs), runs.shape[1], context, ahead, step, every)
    reach = (context - 1) * step
    inputs = np.stack([runs[i, n - reach:n + 1:step] for i, n in enumerate(t)])
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

    def __init__(self, directory, split, manifest, augment, context=2, ahead=1, halo=None, step=None):
        self.split = split
        self.sources = manifest["splits"][split]["sources"]
        self.runs = _frames(directory, split, manifest)
        self.cut = _cut(manifest, halo)
        self.halo = manifest["halo"] - self.cut
        every = received_every(manifest)
        self.context, self.ahead = context, ahead
        self.step = every if step is None else step
        self.augment = augment
        self.options = choices(self.runs.shape[1], context, ahead, self.step, every)
        self.fixed = newest(len(self.runs), self.runs.shape[1], context, ahead, self.step, every)

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, index):
        if self.augment:
            t = int(self.options[int(torch.randint(0, len(self.options), ()))])
        else:
            t = int(self.fixed[index])
        reach = (self.context - 1) * self.step
        picked = [*range(t - reach, t + 1, self.step), t + self.ahead]
        clip = trim(np.asarray(self.runs[index, picked]), self.cut, channels_last=True)
        if self.augment and int(torch.randint(0, 2, ())):
            clip = clip[:, :, ::-1]
        inputs = tuple(_to_tensor(frame) for frame in clip[:self.context])
        return inputs, _to_tensor(clip[-1])


def _to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
