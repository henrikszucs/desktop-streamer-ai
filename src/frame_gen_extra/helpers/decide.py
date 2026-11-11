"""Per tile: what the published search measured, and whether carrying its motion beat repeat (notebook 13).

The tile search moves every tile its gate lets through - its cost at no motion at least 0.1 luma
levels - by half the motion it found. On real screens much of what passes that gate is codec
noise over a still picture, where repeat is better (notebook 11). A decision per tile between
the search's N+1 and N cannot blur: it picks one of two sharp pictures. This module gives every
tile of a split's clips the numbers such a decision could use - all from the search's own work on
the two received frames, and the stream's bitrate, which the client knows - and the score of both
choices, so any rule, fitted or not, is scored by looking up which picture it picked.

`FEATURES` (per tile, over the window):

- `log zero cost`, `log best cost`: the search's cost at no motion and at its best offset (luma
  levels, half resolution);
- `explained`: best over zero cost - how much of the change the best offset accounts for;
- `peak`: the largest of the search's softmax weights - one clear motion, or a spread;
- `gap`: the next local minimum's cost over the best's - how ambiguous the match is;
- `log motion`: how far the found motion goes, stream pixels;
- `residual share`: mean |N − N−2 moved by the motion| over mean |N − N−2|, at full resolution;
- `unexplained cells`: the share of 16 px cells where the motion leaves more than 0.6 of the
  change unexplained (notebook 10's `u`);
- `changed pixels`, `change strength`: the share of pixels whose luma moved by more than 8 levels
  from N−2 to N, and their mean change - a real change, or codec noise;
- `log bpp`, `deblock`: the stream's bits per pixel and loop filter (no codec counts as 2 bpp).
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from . import clips, history, models, runs
from .metrics import RGB_TO_YUV
from .training import DEVICE, to_tensor
from .webexport import TILE_INPUT

FEATURES = ("log zero cost", "log best cost", "explained", "peak", "gap", "log motion", "residual share",
            "unexplained cells", "changed pixels", "change strength", "log bpp", "deblock")


@torch.no_grad()
def features(search, previous, current, bpp, deblock, cell=16):
    """(N, len(FEATURES)) for tiles (N, 3, H, W) of N-2 and N; `bpp`, `deblock` per tile."""
    cost = search.costs(previous, current)                                   # (N, offsets), luma levels
    zero, best = cost[:, search.no_motion], cost.min(1).values
    peak = torch.softmax(-search.sharpness * cost, dim=1).max(1).values
    size = 2 * search.radius + 1
    grid = cost.view(-1, 1, size, size)
    local = (grid == -F.max_pool2d(-grid, 3, stride=1, padding=1)).flatten(1)
    ranked = torch.where(local, cost, torch.full_like(cost, math.inf)).sort(1).values
    second = torch.where(torch.isfinite(ranked[:, 1]), ranked[:, 1], cost.max(1).values)
    motion, _, _ = search.measure(previous, current)                          # (N, 2), how far the content moved
    luma = RGB_TO_YUV[0].to(current.device, current.dtype).view(1, 3, 1, 1) * 255.0
    now, before = (current * luma).sum(1, keepdim=True), (previous * luma).sum(1, keepdim=True)
    height, width = current.shape[-2:]
    aligned = models.warp(before, (-motion)[:, :, None, None].expand(-1, -1, height, width).contiguous())
    residual, change = (now - aligned).abs(), (now - before).abs()
    u = (F.avg_pool2d(residual, cell, ceil_mode=True) / (F.avg_pool2d(change, cell, ceil_mode=True) + 1.0)).clamp(0, 1)
    moved = change > runs.CHANGED_LEVEL
    strength = (change * moved).sum(dim=(1, 2, 3)) / moved.sum(dim=(1, 2, 3)).clamp_min(1)
    bpp = torch.as_tensor(bpp, device=current.device, dtype=current.dtype)
    return torch.stack([torch.log(zero + 0.01), torch.log(best + 0.01), best / (zero + 0.01), peak,
                        (second - best) / (best + 0.01), torch.log1p(motion.norm(dim=1)),
                        residual.mean(dim=(1, 2, 3)) / (change.mean(dim=(1, 2, 3)) + 0.1),
                        (u > 0.6).float().mean(dim=(1, 2, 3)), moved.float().mean(dim=(1, 2, 3)), strength / 32.0,
                        torch.log2(bpp), torch.as_tensor(deblock, device=current.device, dtype=current.dtype)], 1)


@torch.no_grad()
def load(manifest, split, search, start=1, chosen=None, batch_size=64, progress=None):
    """Every client tile of a split's clips at every received frame N from `start` received frames
    on: {"features": (S, F), "repeat", "search": (S,) wPSNR over the kept region, "moving" (the
    search's gate open), "changed" (N+1 differs from N), "source", "frame", "tile", "file"}."""
    received = runs.received_every(manifest)
    model = models.Motion(search=search, learned=False).to(DEVICE).eval()
    out = defaultdict(list)
    recorded = manifest["splits"][split]["sources"]
    for count, index in enumerate(range(len(recorded)) if chosen is None else chosen):
        source = recorded[index]
        stream = clips.rebuild(source, manifest)
        layout, size = clips.plan(stream.shape[2], stream.shape[1])
        if size != TILE_INPUT:
            continue
        degradation = source["degradation"]
        bpp = degradation["bpp"] if degradation["bpp"] is not None else 2.0
        deblock = float(bool(degradation.get("deblock")))
        chosen_tiles = [(n, t) for n in range(start * received, len(stream) - 1, received) for t in range(len(layout))]
        for first in range(0, len(chosen_tiles), batch_size):
            batch = chosen_tiles[first:first + batch_size]
            windows = np.stack([stream[[n - received, n, n + 1], layout[t][1]:layout[t][1] + size[0],
                                       layout[t][0]:layout[t][0] + size[1]] for n, t in batch])
            previous, current, target = to_tensor(windows).unbind(1)
            found = features(search, previous, current, [bpp] * len(batch), [deblock] * len(batch))
            made = model(previous, current)
            boxes = [(layout[t][3], layout[t][2], layout[t][5], layout[t][4]) for _, t in batch]
            for i, ((n, t), box) in enumerate(zip(batch, boxes)):
                truth = clips.kept(target[i:i + 1], box)
                out["repeat"].append(float(history.sample_wpsnr(clips.kept(current[i:i + 1], box), truth)[0]))
                out["search"].append(float(history.sample_wpsnr(clips.kept(made[i:i + 1], box), truth)[0]))
                y, x, h, w = box
                out["changed"].append(bool(runs.changed_share(windows[i, 1, y:y + h, x:x + w],
                                                              windows[i, 2, y:y + h, x:x + w]) >= runs.MOVING_SHARE))
                out["source"].append(index)
                out["frame"].append(n)
                out["tile"].append(t)
                out["file"].append(source["file"])
            out["features"].append(found.cpu().numpy())
        if progress:
            progress(count, index)
    table = {key: (np.concatenate(value) if key == "features" else np.asarray(value)) for key, value in out.items()}
    table["moving"] = np.exp(table["features"][:, 0]) - 0.01 >= search.still
    return table
