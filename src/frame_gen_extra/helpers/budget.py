"""Where the tile search's error comes from.

The search assumes one rigid motion per tile, going on at the same speed, and that every
change is a motion. Each scene records what really happened on every frame (`scenes.Scene`),
so every client tile at a generated frame N+1 can be labelled by what touched it between the
received frame N-2 and N+1 (`events`), and the error of repeat and of the search summed
there (`errors`).

The error is the weighted Y'CbCr squared error the score uses (6:1:1), summed over the
region the client keeps from a tile - so errors add up across tiles and a budget can be
read off them, which wPSNR averages cannot.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from . import clips, metrics, models, runs
from .training import DEVICE, to_tensor
from .webexport import TILE_INPUT

STEADY = 2      # window px: a motion that goes on within this of the same speed is steady
WEIGHTS = torch.tensor(metrics.WEIGHTS) / sum(metrics.WEIGHTS)

# a tile's label is the first of these that touched it - the order of how little the frames
# before say about it
LABELS = ("menu opens or closes at N+1", "menu opened or closed before N",
          "drag: starts", "drag: stops", "drag: changes speed", "drag: goes on",
          "region: starts", "region: stops", "region: changes speed", "region: goes on",
          "scroll: starts", "scroll: stops", "scroll: changes speed", "scroll: goes on",
          "screen", "video", "nothing recorded")


def _state(before, now, after):
    """How a position (a number or an (x, y)) moved over N-2, N, N+1: None if it did not."""
    before, now, after = (np.atleast_1d(np.asarray(p, dtype=float)) for p in (before, now, after))
    earlier, next_ = now - before, after - now
    moved_before, moves = bool(np.any(earlier)), bool(np.any(next_))
    if not moved_before and not moves:
        return None
    if not moved_before:
        return "starts"
    if not moves:
        return "stops"
    return "goes on" if np.abs(next_ - earlier / 2).max() <= STEADY else "changes speed"


def _overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def events(source, n, step, rect):
    """Every label (of `LABELS`) of what touched `rect` - (x, y, w, h) in window pixels - over
    received frames N-2 (= n - step) and N (= n) and the generated N+1."""
    scene = source.get("scene")
    if not scene:
        # a recording: nothing is recorded of what moved in it, only what it is
        return [source["kind"]] if source["kind"] in ("screen", "video") else ["nothing recorded"]
    times = (n - step, n, n + 1)
    found = []
    popup = scene.get("popup")
    if popup and _overlaps(rect, (*popup["at"], *popup["source"][2:])):
        first, end = popup["frames"]
        shown = [first <= t < end for t in times]
        if shown[1] != shown[2]:
            found.append("menu opens or closes at N+1")
        elif shown[0] != shown[1]:
            found.append("menu opened or closed before N")
    drag = scene.get("drag")
    if drag:
        size = drag["source"][2:]
        if any(_overlaps(rect, (*drag["at"][t], *size)) for t in times):
            state = _state(*(drag["at"][t] for t in times))
            if state:
                found.append(f"drag: {state}")
    region = scene.get("region")
    if region and _overlaps(rect, region["box"]):
        state = _state(*(region["offset"][t] for t in times))
        if state:
            found.append(f"region: {state}")
    state = _state(*(scene["scroll"][t] for t in times))
    if state:
        found.append(f"scroll: {state}")
    return found or ["nothing recorded"]


def primary(found):
    """The one label a tile is counted under: the first of `LABELS` it has."""
    return min(found, key=LABELS.index)


# ---------------------------------------------------------------- the tile cost's minima

def minima(cost, count, size):
    """(N, count): each tile's next-cheapest local minima of its (N, offsets) costs, after the
    cheapest - repeated when there are fewer."""
    grid = cost.view(-1, 1, size, size)
    lowest = -F.max_pool2d(-grid, 3, stride=1, padding=1)
    ranked = torch.where(grid == lowest, grid, torch.inf).flatten(1).argsort(dim=1)
    found = (grid == lowest).flatten(1).sum(1, keepdim=True)
    picks = torch.arange(1, count + 1, device=cost.device).expand(len(cost), -1)
    return ranked.gather(1, torch.minimum(picks, found - 1))


def squared(pictures, target):
    """(N, K, H, W): the weighted Y'CbCr squared error of each of K (N, K, 3, H, W) pictures."""
    yuv = metrics.rgb_to_yuv(pictures.clamp(0, 1).flatten(0, 1)).view_as(pictures)
    weights = WEIGHTS.to(target.device).view(1, 1, 3, 1, 1)
    return ((yuv - metrics.rgb_to_yuv(target)[:, None]) ** 2 * weights).sum(2)


@torch.no_grad()
def errors(search, previous, current, target, box, keep=False):
    """{name: (N,) summed squared error over the kept `box`} for repeat and the published search.
    With `keep`, also {name: (N, 3, H, W)} - each one's picture."""
    y, x, h, w = box
    searched = models.warp(current, search(previous, current).expand(-1, -1, *current.shape[-2:]))
    pictures = torch.stack([current, searched], 1)
    total = squared(pictures, target)[:, :, y:y + h, x:x + w].sum(dim=(-2, -1))
    found = {"repeat": total[:, 0], "search": total[:, 1]}
    if not keep:
        return found
    return found, {"repeat": current, "search": searched}


@torch.no_grad()
def load(manifest, split, search, every=1, depth=1, count=None, batch_size=24):
    """Every client tile of a split's clips at every `every`th received frame N from N = depth
    * received: {"error": {name: (S,)}, "pixels": (S,), "labels": [[label, ...]], "label": (S,)
    the primary one, "changed", "moving", "source", "frame", "tile", "box"}."""
    received = runs.received_every(manifest)
    scale = manifest["scale"]
    parts, summed = defaultdict(list), defaultdict(list)
    for index, source in enumerate(manifest["splits"][split]["sources"][:count]):
        stream = clips.rebuild(source, manifest)
        layout, size = clips.plan(stream.shape[2], stream.shape[1])
        if size != TILE_INPUT:
            continue
        chosen = [(n, t) for n in range(max(depth, 1) * received, len(stream) - 1, received * every)
                  if runs.valid(source, n, received) for t in range(len(layout))]
        for first in range(0, len(chosen), batch_size):
            batch = chosen[first:first + batch_size]
            windows = np.stack([stream[[n - received, n, n + 1], layout[t][1]:layout[t][1] + size[0],
                                       layout[t][0]:layout[t][0] + size[1]] for n, t in batch])
            boxes = [(layout[t][3], layout[t][2], layout[t][5], layout[t][4]) for _, t in batch]
            previous, current, target = to_tensor(windows).unbind(1)
            still = search.costs(previous, current)[:, search.no_motion] < search.still
            found = {}
            for box in set(boxes):
                same = torch.as_tensor([b == box for b in boxes], device=DEVICE)
                part = errors(search, previous[same], current[same], target[same], box)
                for name, values in part.items():
                    found.setdefault(name, torch.zeros(len(batch), device=DEVICE))[same] = values
            for name, values in found.items():
                summed[name].append(values.cpu().numpy())
            for i, ((n, t), box) in enumerate(zip(batch, boxes)):
                y, x, h, w = box
                ix, iy = layout[t][:2]
                rect = (scale * (ix + x), scale * (iy + y), scale * w, scale * h)
                labels = events(source, n, received, rect)
                share = runs.changed_share(windows[i:i + 1, 1, y:y + h, x:x + w], windows[i:i + 1, 2, y:y + h, x:x + w])
                parts["labels"].append(labels)
                parts["label"].append(primary(labels))
                parts["changed"].append(bool(share[0] >= runs.MOVING_SHARE))
                parts["moving"].append(not bool(still[i]))
                parts["pixels"].append(h * w)
                parts["source"].append(index)
                parts["frame"].append(n)
                parts["tile"].append(t)
                parts["box"].append(box)
    table = {key: (value if key == "labels" else np.asarray(value)) for key, value in parts.items()}
    table["error"] = {name: np.concatenate(values) for name, values in summed.items()}
    return table


def wpsnr(error, pixels):
    """(S,) wPSNR from summed weighted squared errors over `pixels` each, as `metrics.wpsnr`."""
    return 10 * np.log10(1 / np.maximum(error / pixels, 1e-12))
