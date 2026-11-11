"""Each tile's motion over the clip so far, and what carrying a share of the newest one scores.

The published search finds one motion per tile from the two received frames N-2 and N, and
carries half of it to N+1. A client that keeps that motion between frames also has the
search's answer for the pairs before - N-4 -> N-2, N-6 -> N-4 - at no extra cost: they are
the motions it found one and two received frames ago. Whether a motion goes on, slows down
or stops may show in them; it cannot show in one pair.

So a sample here is one client tile at received frame N (whole clips rebuilt from the
manifest, cut as the client cuts them - `clips.plan`), and it holds:

- **the history**: for the `depth` pairs ending at N, N-2, ... the search's motion (the
  displacement of the content, x and y in stream pixels), its cost at no motion and its least
  cost, in luma levels - index 0 the newest pair;
- **the scores of carrying**: N moved by `c` times the newest motion, for every `c` of `GRID`,
  scored against N+1 (wPSNR per tile, over the region the client keeps from it). `c` = 0 is
  repeat, `c` = 1/2 the published search's rule. Any method that only decides how much of the
  newest motion to carry - a rule, a fitted table, a small network, an oracle - is then a
  choice of `c` per tile, scored by looking it up, with nothing warped again;
- what `clips.groups` gives a tile: changed or not, its kind, and the pointer.

The published search itself, gate and all, is run beside it on every tile, and its score is
kept (`published`) to hold the table to: it must equal the table at `c` = 1/2, or repeat where
the gate says the tile is still.
"""

from collections import defaultdict

import numpy as np
import torch

from . import clips, metrics, models, runs
from .training import DEVICE, to_tensor
from .webexport import TILE_INPUT

GRID = np.linspace(0.0, 1.0, 17)   # shares of the newest motion carried to N+1, in sixteenths


@torch.no_grad()
def pairs(stream, search, layout, size, received, batch_size=64):
    """The search on every tile of every received pair of a clip: (motion (P, T, 2), still (P, T),
    best (P, T)) for P pairs ending at received frames n = received, 2 received, ... and T tiles.
    `motion` is how far the content moved from n - received to n, (x, y) in stream pixels -
    any search with `measure` (`models.TileSearch`, `models.PyramidSearch`)."""
    ends = list(range(received, len(stream), received))
    windows = np.stack([stream[[n - received, n], iy:iy + size[0], ix:ix + size[1]]
                        for n in ends for ix, iy, *_ in layout])
    found = []
    for first in range(0, len(windows), batch_size):
        previous, current = to_tensor(windows[first:first + batch_size]).unbind(1)
        found.append(search.measure(previous, current))
    motion, still, best = (torch.cat(parts) for parts in zip(*found))
    shape = (len(ends), len(layout))
    return (motion.view(*shape, 2).cpu().numpy(), still.view(shape).cpu().numpy(), best.view(shape).cpu().numpy())


def sample_wpsnr(prediction, target):
    """(N,) wPSNR of each picture - `metrics.wpsnr` before its mean over the batch."""
    yuv_prediction, yuv_target = metrics._yuv(prediction, target)
    mse = metrics._mse(yuv_prediction, yuv_target)
    weight = torch.tensor(metrics.WEIGHTS, device=mse.device, dtype=mse.dtype)
    return metrics._db((mse * weight).sum(dim=1) / weight.sum())


def carried(current, motion, share, whole=False, mode="bilinear"):
    """`current` (N, 3, H, W) with its content moved on by `share` (N,) of `motion` (N, 2) -
    by whole pixels with `whole`, else read between them by `mode`."""
    offsets = -share[:, None] * motion
    if whole:
        offsets = torch.round(offsets)
    offsets = offsets[:, :, None, None].expand(-1, -1, *current.shape[-2:])
    return models.warp(current, offsets.contiguous(), mode)


@torch.no_grad()
def load_many(manifest, split, searches, depth=3, every=1, grid=GRID, count=None, batch_size=48, scored=None):
    """`load` for several searches over the same frames - x264 does not encode a clip the same
    way twice, so searches compared on separate rebuilds would see different pictures.
    `searches` is {name: search}; `scored` the names whose N+1 is scored (all by default), the
    others only measured. Returns {"tiles": what every search shares - "changed", "source",
    "frame", "tile", "box", "origin", "skipped"} and per name {"motion", "still", "best"} and,
    if scored, {"scores", "published"} - as `load` has them."""
    received = runs.received_every(manifest)
    scored = set(searches if scored is None else scored)
    published = {name: models.Motion(search=search, learned=False).to(DEVICE).eval()
                 for name, search in searches.items() if name in scored}
    shares = torch.as_tensor(grid, device=DEVICE, dtype=torch.float32)
    tiles, skipped = defaultdict(list), []
    per = {name: defaultdict(list) for name in searches}
    for index, source in enumerate(manifest["splits"][split]["sources"][:count]):
        stream = clips.rebuild(source, manifest)
        layout, size = clips.plan(stream.shape[2], stream.shape[1])
        if size != TILE_INPUT:
            skipped.append(source["file"])
            continue
        measured = {name: pairs(stream, search, layout, size, received) for name, search in searches.items()}
        chosen = [(n, t) for n in range(depth * received, len(stream) - 1, received * every)
                  for t in range(len(layout))]
        for first in range(0, len(chosen), batch_size):
            batch = chosen[first:first + batch_size]
            windows = np.stack([stream[[n - received, n, n + 1], layout[t][1]:layout[t][1] + size[0],
                                       layout[t][0]:layout[t][0] + size[1]] for n, t in batch])
            boxes = [(layout[t][3], layout[t][2], layout[t][5], layout[t][4]) for _, t in batch]
            # pair p ends at received frame (p + 1) * received; the newest first
            history = [[n // received - 1 - k for k in range(depth)] for n, _ in batch]
            previous, current, target = to_tensor(windows).unbind(1)
            offset = len(tiles["source"])
            for name in scored:
                motion = measured[name][0]
                newest = torch.as_tensor(np.stack([motion[h[0], t] for h, (_, t) in zip(history, batch)]), device=DEVICE)
                search = searches[name]
                how = {"whole": getattr(search, "whole", False), "mode": getattr(search, "interpolation", "bilinear")}
                made = {j: carried(current, newest, share.expand(len(batch)), **how) for j, share in enumerate(shares)}
                made["published"] = published[name](previous, current)
                scores = torch.empty(len(batch), len(grid), device=DEVICE)
                own = torch.empty(len(batch), device=DEVICE)
                for box in set(boxes):
                    same = torch.as_tensor([b == box for b in boxes], device=DEVICE)
                    truth = clips.kept(target[same], box)
                    for j in range(len(grid)):
                        scores[same, j] = sample_wpsnr(clips.kept(made[j][same], box), truth)
                    own[same] = sample_wpsnr(clips.kept(made["published"][same], box), truth)
                per[name]["scores"].append(scores.cpu().numpy())
                per[name]["published"].append(own.cpu().numpy())
            for i, ((n, t), box, h) in enumerate(zip(batch, boxes, history)):
                y, x, hh, ww = box
                share = runs.changed_share(windows[i:i + 1, 1, y:y + hh, x:x + ww], windows[i:i + 1, 2, y:y + hh, x:x + ww])
                tiles["changed"].append(bool(share[0] >= runs.MOVING_SHARE))
                tiles["source"].append(index)
                tiles["frame"].append(n)
                tiles["tile"].append(t)
                tiles["box"].append(box)
                tiles["origin"].append(layout[t][:2])
                for name, (motion, still, best) in measured.items():
                    per[name]["motion"].append(motion[h, t])
                    per[name]["still"].append(still[h, t])
                    per[name]["best"].append(best[h, t])
    found = {"tiles": {key: np.asarray(value) for key, value in tiles.items()}}
    found["tiles"]["skipped"] = skipped
    for name, parts in per.items():
        found[name] = {key: (np.concatenate(value) if key in ("scores", "published") else np.asarray(value))
                       for key, value in parts.items()}
    return found


def load(manifest, split, search, depth=3, every=1, grid=GRID, count=None, batch_size=48, wide=None):
    """Every client tile of a split's clips at every `every`th received frame N with `depth`
    pairs of history: {"motion": (S, depth, 2), "still": (S, depth), "best": (S, depth),
    "scores": (S, len(grid)), "published": (S,), "changed": (S,), "source", "frame", "tile",
    "box": (S, 4), "origin": (S, 2)}, and "skipped": the sources whose frames are smaller than
    a tile. With `wide`, a second search (a further reach), also "wide": (S, depth, 2), its
    motions of the same pairs - for reading the history, not for carrying. No pictures are
    kept: a split's whole train set fits in memory this way."""
    searches = {"search": search, **({"wide": wide} if wide is not None else {})}
    many = load_many(manifest, split, searches, depth, every, grid, count, batch_size, scored=["search"])
    found = {**many["tiles"], **many["search"]}
    if wide is not None:
        found["wide"] = many["wide"]["motion"]
    return found


def groups(manifest, split, table):
    """Boolean masks over the samples, as `clips.groups` makes them over tiles: `all`,
    `changed`, `unchanged`, one per kind of source, and `pointer` - the pointer's tip in the
    kept region at N+1."""
    recorded = manifest["splits"][split]["sources"]
    count = len(table["source"])
    masks = {"all": np.ones(count, bool), "changed": table["changed"].copy(), "unchanged": ~table["changed"],
             "pointer": np.zeros(count, bool)}
    scale = manifest["scale"]
    for i in range(count):
        source = recorded[table["source"][i]]
        masks.setdefault(source["kind"], np.zeros(count, bool))[i] = True
        cursor = (source.get("scene") or {}).get("cursor")
        if cursor:
            tip_x, tip_y = np.asarray(cursor[table["frame"][i] + 1]) / scale
            ix, iy = table["origin"][i]
            y, x, h, w = table["box"][i]
            masks["pointer"][i] = ix + x <= tip_x < ix + x + w and iy + y <= tip_y < iy + y + h
    return masks


def save(path, table):
    """A `load` result to one .npz, to read back with `restore`."""
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in table.items()})


def restore(path):
    with np.load(path) as stored:
        table = {key: stored[key] for key in stored.files}
    table["skipped"] = list(table["skipped"])
    return table


def along(table, k):
    """(S,) how far pair `k`'s motion went along the newest one, as a share of the newest: 1 for
    a steady motion, more for one that is slowing down, less for one speeding up or starting,
    NaN where the newest motion is under a tenth of a pixel. Read from the wide search's
    motions where the table has them."""
    motions = table.get("wide", table["motion"])
    newest, older = motions[:, 0], motions[:, k]
    length = (newest ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(length > 0.01, (older * newest).sum(1) / length, np.nan)


