"""Per tile: what the published search measured, and the score of every picture it can make
(notebooks 3 and 4).

Extrapolation's `helpers/decide.py`, for interpolation and for the graph: every client tile of a
split's clips, rebuilt whole (`clips`), at every received frame N+2 a live capture could give
(`runs.valid`), with

- for each of the `searches` given - the published one and wider ones - the six numbers
  `models.Decided.features` measures from its own costs (the same code the graph runs, so a
  decision trained on this table is the graph's) and the wPSNR, over the region the client
  keeps, of every picture `Decided` can choose from it (the search, the mean, either frame held,
  either frame moved alone);
- the wPSNR of any other `methods` given - any model of (previous, current);
- `changed` (N+1 differs from N or from N+2 over the kept region, as `clips.groups`), `moving`
  (the first search's gate open: something moved between N and N+2), and where the tile came
  from.

The rebuilt clips are what takes the time, so every search is scored in the one pass.

So any rule, fitted or not, is scored by looking up the picture it picked. The scores are of the
pictures against N+1, which only training and scoring see; nothing here picks with them.
"""

from collections import defaultdict

import numpy as np
import torch

from . import clips, metrics, models, runs
from .training import DEVICE, to_tensor
from .webexport import TILE_INPUT


# The searches notebook 3 scores, by the reach they cover between N and N+2 in stream pixels: the
# published one (notebook 1), the same at twice the radius - 289 offsets for 81 - and the same
# reach at a quarter of the resolution, 81 offsets again but each 4 px apart.
SEARCHES = {"±8": {"radius": 4, "scale": 2, "stride": 2},
            "±16": {"radius": 8, "scale": 2, "stride": 2},
            "±16 at ¼": {"radius": 4, "scale": 4, "stride": 1}}


# What each costs: GPU ms for a 1080p frame (36 tiles) as `models.Interpolate`, in Chrome's
# WebGPU on the RTX 3090 (notebook 3, "The reach"). 289 offsets cost 2.7 times 81.
BROWSER_MS = {"±8": 3.3, "±16": 8.9, "±16 at ¼": 2.9}


def search(name):
    """The `SEARCHES` entry as a `models.TileSearch`, with the published sharpness, share and gate."""
    return models.TileSearch(**SEARCHES[name], sharpness=10.0, fraction=0.5, still=0.1)


def reach(train, cap=50.0, published="±8"):
    """The search the decision builds on: of those no slower in the browser than the published one,
    the one whose picture scores best on the training split's moving tiles (`train`, a `load` table)."""
    affordable = [name for name in SEARCHES if BROWSER_MS[name] <= BROWSER_MS[published]]
    return max(affordable, key=lambda name: np.minimum(train["scores"][name]["search"], cap)[train["moving"]].mean())


def sample_wpsnr(prediction, target):
    """(N,) wPSNR of each picture - `metrics.wpsnr` before its mean over the batch."""
    yuv_prediction, yuv_target = metrics._yuv(prediction, target)
    mse = metrics._mse(yuv_prediction, yuv_target)
    weight = torch.tensor(metrics.WEIGHTS, device=mse.device, dtype=mse.dtype)
    return metrics._db((mse * weight).sum(dim=1) / weight.sum())


@torch.no_grad()
def load(manifest, split, searches, methods=None, chosen=None, batch_size=32, progress=None):
    """Every client tile of a split's clips at every valid received frame N+2 (N+2 >= 2):
    {"features": {search: (S, 6)}, "scores": {search: {choice: (S,) wPSNR}}, "methods": {name:
    (S,) wPSNR}, "changed", "moving", "source", "frame", "tile", "file"}. `searches` is {name:
    `models.TileSearch`}; `chosen` limits it to those sources (indices into the split's)."""
    received = runs.received_every(manifest)
    deciders = {name: models.Decided(search).to(DEVICE).eval() for name, search in searches.items()}
    methods = {name: method.to(DEVICE).eval() for name, method in (methods or {}).items()}
    listed = manifest["splits"][split]["sources"]
    out, scores, features = defaultdict(list), defaultdict(list), defaultdict(list)
    for count, index in enumerate(range(len(listed)) if chosen is None else chosen):
        source = listed[index]
        stream = clips.rebuild(source, manifest)
        layout, size = clips.plan(stream.shape[2], stream.shape[1])
        if size != TILE_INPUT:
            continue
        tiles = [(n, t) for n in range(received, len(stream), received) if runs.valid(source, n, received)
                 for t in range(len(layout))]
        for first in range(0, len(tiles), batch_size):
            batch = tiles[first:first + batch_size]
            windows = np.stack([stream[[n - received, n, n - received // 2], layout[t][1]:layout[t][1] + size[0],
                                       layout[t][0]:layout[t][0] + size[1]] for n, t in batch])
            previous, current, target = to_tensor(windows).unbind(1)
            pictures = {}
            for search_name, decider in deciders.items():
                made, found = decider.candidates(previous, current)
                pictures.update({(search_name, choice): picture for choice, picture in made.items()})
                features[search_name].append(found.cpu().numpy())
            pictures.update({name: method(previous, current) for name, method in methods.items()})
            boxes = np.array([(layout[t][3], layout[t][2], layout[t][5], layout[t][4]) for _, t in batch])
            made = {name: np.zeros(len(batch)) for name in pictures}
            changed = np.zeros(len(batch), bool)
            for box in {tuple(b) for b in boxes}:
                same = np.flatnonzero((boxes == box).all(1))
                truth = clips.kept(target[same], box)
                for name, picture in pictures.items():
                    made[name][same] = sample_wpsnr(clips.kept(picture[same].float(), box), truth).cpu().numpy()
                y, x, h, w = box
                region = windows[same][:, :, y:y + h, x:x + w]
                changed[same] = np.maximum(runs.changed_share(region[:, 0], region[:, 2]),
                                           runs.changed_share(region[:, 1], region[:, 2])) >= runs.MOVING_SHARE
            out["changed"].extend(changed.tolist())
            for name, values in made.items():
                scores[name].extend(values.tolist())
            for n, t in batch:
                out["source"].append(index)
                out["frame"].append(n)
                out["tile"].append(t)
                out["file"].append(source["file"])
        if progress:
            progress(count, index)
    table = {key: np.asarray(value) for key, value in out.items()}
    table["features"] = {name: np.concatenate(features[name]) for name in searches}
    table["scores"] = {name: {choice: np.asarray(scores[name, choice]) for choice in models.Decided.CHOICES}
                       for name in searches}
    table["methods"] = {name: np.asarray(scores[name]) for name in methods}
    first = next(iter(searches))
    table["moving"] = np.exp(table["features"][first][:, 0]) - 0.01 >= searches[first].still
    return table
