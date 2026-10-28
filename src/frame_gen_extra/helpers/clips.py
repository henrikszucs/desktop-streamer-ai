"""Whole frames, cut the way the client cuts them.

The run set holds 80-96 px spots, but the client finds one motion per 328x188 tile, and a
tile can hold a scrolled page beside a still panel. So this rebuilds a split's whole clips
from what the manifest recorded of each source - the file, the scene (`scenes.Scene`), the
degradation, the offset of its run - and cuts their frames into the client's tiles.
`rebuild` gives the same frames the runs were cut from, to the codec's run-to-run
difference (about one luma level).

`plan` is desktop-streamer's `planTiles` (`src/client/web/src/room/stream-enhance.js`):
tiles of `TILE_STEP` plus `TILE_HALO` on every side, all the same size. A tile at an edge of
the frame is not padded: its window is slid back into the frame, and the region kept from it
moves inside the window instead. A frame smaller than a tile is one tile of its own size.

A sample is one tile of the client's 2x generation: received frames N-2 and N in, N+1 out,
at every `every`th received frame of a clip. With `raw`, each sample also gets the luma of the
clean frames N and N+1 at the source's resolution - the frames before the stream's shrink and
codec, which is what the host has when it measures a hint (notebook 5). Its score is over the region the client keeps
from it (`boxes`), not the whole window, and `evaluate` takes the model or any function of
(previous, current, target) - the target, for an oracle.
"""

from collections import defaultdict

import numpy as np
import torch

from . import degrade, metrics, runs, scenes, sources
from .training import DEVICE, to_tensor
from .webexport import TILE_HALO, TILE_STEP


def rebuild(source, manifest, raw=False):
    """The decoded stream frames (T, h, w, 3) of one source, as `00_data_preprocess` made them.
    With `raw`, (clean, stream): the frames before the stream as well, `scale` times the size."""
    path = sources.RAW / source["dataset"] / source["file"]
    if source["kind"] == "video":
        frames, _ = sources.load_sequence(path)
    else:
        scene = scenes.Scene(**source["scene"])
        frames = scenes.render(sources.load_image(path), scene, cursor=manifest.get("cursor_drawn", True))
    clean, stream = degrade.degrade(frames, degrade.Degradation(**source["degradation"]))
    return (clean, stream) if raw else stream


def plan(width, height, step=TILE_STEP, halo=TILE_HALO):
    """The client's tiles of a frame: [(ix, iy, ox, oy, w, h)] - the window's top left in the
    frame, the kept region's top left in the window, and the kept region's size. Every window
    is `size` (height, width), returned second."""
    step_h, step_w = step
    tile_w, tile_h = min(width, step_w + 2 * halo), min(height, step_h + 2 * halo)
    tiles = []
    for y in range(0, height, step_h):
        h = min(step_h, height - y)
        iy = min(max(0, y - halo), height - tile_h)
        for x in range(0, width, step_w):
            w = min(step_w, width - x)
            ix = min(max(0, x - halo), width - tile_w)
            tiles.append((ix, iy, x - ix, y - iy, w, h))
    return tiles, (tile_h, tile_w)


def load(manifest, split="val", every=2, count=None, raw=False):
    """Every client tile of a split's clips at every `every`th received frame N (N >= 2):
    {"inputs": (S, 2, h, w, 3) uint8 - frames N-2 and N - "targets": (S, h, w, 3) - N+1 -
    "boxes": (S, 4) the kept (y, x, h, w) in the window, "source", "frame", "origin": the
    window's (x, y) in the frame}. With `raw`, "raw": (S, 2, scale h, scale w) uint8, the luma
    of the clean frames N and N+1 under the same window. Sources whose tiles are not the
    client's full size are left out and named in "skipped"."""
    received = runs.received_every(manifest)
    full = (TILE_STEP[0] + 2 * TILE_HALO, TILE_STEP[1] + 2 * TILE_HALO)
    parts, skipped = defaultdict(list), []
    for index, source in enumerate(manifest["splits"][split]["sources"][:count]):
        stream = rebuild(source, manifest, raw=raw)
        if raw:
            clean, stream = stream
            scale = clean.shape[1] // stream.shape[1]
            weights = metrics.RGB_TO_YUV[0].numpy()
            clean = np.stack([(frame.astype(np.float32) @ weights).round().clip(0, 255).astype(np.uint8)
                              for frame in clean])
        tiles, size = plan(stream.shape[2], stream.shape[1])
        if size != full:
            skipped.append(source["file"])
            continue
        for n in range(received, len(stream) - 1, received * every):
            for ix, iy, ox, oy, w, h in tiles:
                window = stream[:, iy:iy + size[0], ix:ix + size[1]]
                parts["inputs"].append(window[[n - received, n]])
                parts["targets"].append(window[n + 1])
                parts["boxes"].append((oy, ox, h, w))
                parts["source"].append(index)
                parts["frame"].append(n)
                parts["origin"].append((ix, iy))
                if raw:
                    parts["raw"].append(clean[[n, n + 1], scale * iy:scale * (iy + size[0]),
                                              scale * ix:scale * (ix + size[1])])
    tiles = {key: np.asarray(value) for key, value in parts.items()}
    tiles["skipped"] = skipped
    return tiles


def kept(images, box, channels_last=False):
    """The region `box` (y, x, h, w) of (..., H, W) images, or of (..., H, W, C) ones."""
    y, x, h, w = box
    if channels_last:
        return images[..., y:y + h, x:x + w, :]
    return images[..., y:y + h, x:x + w]


def groups(manifest, split, tiles):
    """Boolean masks over the tiles, as `runs.groups` makes them over runs: `all`, `changed`
    and `unchanged` (by `runs.MOVING_SHARE` over the kept region, N to N+1), one per kind of
    source, and `pointer` - the pointer's tip in the kept region at N+1, where windows are
    dragged and menus open."""
    recorded = manifest["splits"][split]["sources"]
    count = len(tiles["source"])
    moving = np.zeros(count, bool)
    for box in {tuple(b) for b in tiles["boxes"]}:
        chosen = np.flatnonzero((tiles["boxes"] == box).all(1))
        current = kept(tiles["inputs"][chosen, 1], box, True)
        moving[chosen] = runs.changed_share(current, kept(tiles["targets"][chosen], box, True)) >= runs.MOVING_SHARE
    masks = {"all": np.ones(count, bool), "changed": moving, "unchanged": ~moving, "pointer": np.zeros(count, bool)}
    scale = manifest["scale"]
    for i in range(count):
        source = recorded[tiles["source"][i]]
        masks.setdefault(source["kind"], np.zeros(count, bool))[i] = True
        cursor = (source.get("scene") or {}).get("cursor")
        if cursor:
            tip_x, tip_y = np.asarray(cursor[tiles["frame"][i] + 1]) / scale
            ix, iy = tiles["origin"][i]
            y, x, h, w = tiles["boxes"][i]
            masks["pointer"][i] = ix + x <= tip_x < ix + x + w and iy + y <= tip_y < iy + y + h
    return masks


@torch.no_grad()
def evaluate(model, inputs, targets, masks, boxes, batch_size=64, side=None):
    """Every score in `metrics.report` per group, as `training.evaluate`, each sample scored on
    its own `boxes` region. `model` is called as model(previous, current) - with `side`, an
    array of one entry per sample (a hint), as model(previous, current, side) - and, if it has
    an `oracle` attribute set, with the target last."""
    if isinstance(model, torch.nn.Module):
        model.to(DEVICE).eval()
    oracle = getattr(model, "oracle", False)
    totals = {name: {} for name in masks}
    counts = dict.fromkeys(masks, 0)
    for first in range(0, len(inputs), batch_size):
        chunk = slice(first, first + batch_size)
        previous, current = to_tensor(inputs[chunk]).unbind(1)
        target = to_tensor(targets[chunk])
        extra = [] if side is None else [torch.as_tensor(np.asarray(side[chunk]), device=DEVICE).float()]
        prediction = model(previous, current, *extra, *([target] if oracle else [])).float()
        for box in {tuple(b) for b in boxes[chunk]}:
            same = torch.from_numpy((boxes[chunk] == box).all(1)).to(DEVICE)
            for name, mask in masks.items():
                chosen = same & torch.from_numpy(mask[chunk]).to(DEVICE)
                n = int(chosen.sum())
                if not n:
                    continue
                scores = metrics.report(kept(prediction[chosen], box), kept(target[chosen], box))
                for key, value in scores.items():
                    totals[name][key] = totals[name].get(key, 0.0) + value * n
                counts[name] += n
    return {name: {key: value / counts[name] for key, value in scores.items()}
            for name, scores in totals.items() if counts[name]}
