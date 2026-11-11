"""3-D recursive search (3DRS): a motion per block from a handful of candidates (notebook 9).

de Haan, Biezen, Huijgen and Ojo, "True-motion estimation with 3-D recursive search block
matching" (IEEE TCSVT, 1993) - the block motion estimator behind Philips' Natural Motion TVs.
A full search per block tests every offset and keeps the cheapest, which on a desktop is often
not the true motion: a flat block matches anywhere, a line of text matches the line above.
Notebook 4's per-block searches lost to one motion per tile for that. 3DRS tests only what the
field already holds, because real motion is smooth in space and in time:

- **spatial**: the vectors of the block's four neighbours, and its own;
- **temporal**: the block's own vector in the previous pair's field - memory across frames;
- **update**: the block's or a neighbour's vector plus a small step from a fixed rotating set
  (`UPDATES`), so the field can follow a change of speed;
- **zero**, and **seeds**: vectors from elsewhere - for the whole tile, the tile search's motion
  (`"tile"`), the next two minima of its cost (`"minima"`, notebook 4's candidates) and the
  pointer's motion (`"pointer"`); per 32 px block, the motion the published search's costs give
  there (`"blocks"`, also notebook 4's). Started from zero alone, 3DRS cannot find a window
  that starts moving several pixels a pair: no small update matches better on the way there,
  and a TV's field converges over frames a desktop does not give it.

Each candidate costs the mean absolute luma difference between the block in N and N-2 read
back along it, plus a penalty by kind (`PENALTIES`, in luma levels): none for a neighbour's
vector, most for an update. So a new vector wins only where it matches clearly better, and the
field stays smooth. Of equal costs the first candidate wins, and the seeds come before zero: a
flat block matches every vector, and on a scroll it has to take the tile's motion - stopped,
it cuts off the text that should move into it.

The original scans the blocks in order, each taking what its left and upper neighbours found a
moment ago. Here every block updates at once from the field of the pass before, `passes` times
(Jacobi-style) - the form a GPU, and an ONNX graph, can run. Motion spreads a block a pass
instead of a row a pass; the temporal and seed candidates make up for it.

Vectors are whole stream pixels while searching. The final field gets an equiangular fit below
a pixel per block (as `models.PyramidSearch`), and what is carried to N+1 is rounded to whole
pixels again: measured finely, carried whole (notebook 8). A block that took the tile's motion
carries the tile search's measure of it instead of its own fit (`snap`), so a scroll stays one
motion.

Convention, as `history` keeps a tile's motion: a vector (x, y) is how far the content moved
from N-2 to N, so N(p) = N-2(p - v), and N+1(p) = N(p - v/2).

`full_search` is the control: the same blocks and costs, every offset within a radius, no
candidates and no penalties but a tie-break towards small vectors. `load` runs any of these
over every client tile of a split's clips, in order, and scores N+1 beside the stateless
methods (repeat, the tile searches, notebook 4's blocks, the motion oracle).
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from . import budget, clips, history, models, runs
from .metrics import RGB_TO_YUV
from .training import DEVICE, to_tensor
from .webexport import TILE_INPUT

# luma levels (of 255) added to a candidate's mean absolute difference, by kind
PENALTIES = {"spatial": 0.0, "temporal": 0.125, "seed": 0.125, "zero": 0.125, "update": 0.5}
# (x, y) steps added to a vector to make an update candidate, taken in turn by block and pass
UPDATES = ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2), (2, 0), (-2, 0),
           (1, 1), (-1, -1), (1, -1), (-1, 1), (0, 4), (0, -4), (4, 0), (-4, 0))
MIN_VALID = 0.25    # a candidate whose source lies inside the window for less of the block costs inf
TIE_BREAK = 0.025   # full search: luma levels per pixel of a vector, so equal matches keep the smaller


def luma(frames):
    """(N, 1, H, W) luma in levels of 255, of (N, 3, H, W) RGB in [0, 1]."""
    weights = RGB_TO_YUV[0].to(frames.device, frames.dtype).view(1, 3, 1, 1)
    return 255.0 * (frames * weights).sum(1, keepdim=True)


def per_pixel(field, height, width, block):
    """(N, 2, H, W): each block's vector of an (N, 2, h, w) field spread over its pixels."""
    return field.repeat_interleave(block, 2).repeat_interleave(block, 3)[..., :height, :width]


def costs(before, now, fields, block):
    """(N, K, h, w): the cost of each of K candidate fields (N, K, 2, h, w) in whole pixels - the
    mean |N(p) - N-2(p - v)| over the block's pixels whose source lies inside the window, in luma
    levels; inf where under `MIN_VALID` of the block's pixels have one. `before`, `now` are
    (N, 1, H, W) luma."""
    n, k = fields.shape[:2]
    height, width = now.shape[-2:]
    v = per_pixel(fields.flatten(0, 1), height, width, block).round().long()
    xs = torch.arange(width, device=now.device).view(1, 1, width)
    ys = torch.arange(height, device=now.device).view(1, height, 1)
    sx, sy = xs - v[:, 0], ys - v[:, 1]
    valid = ((sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)).to(now.dtype)
    index = (sy.clamp(0, height - 1) * width + sx.clamp(0, width - 1)).flatten(1)
    source = before.flatten(1).repeat_interleave(k, 0).gather(1, index).view(n * k, height, width)
    difference = (now[:, 0].repeat_interleave(k, 0) - source).abs() * valid
    total = F.avg_pool2d(difference[:, None], block, ceil_mode=True)[:, 0]
    count = F.avg_pool2d(valid[:, None], block, ceil_mode=True)[:, 0]
    share = count / F.avg_pool2d(torch.ones_like(valid[:1, None]), block, ceil_mode=True)[:, 0]
    cost = torch.where(share >= MIN_VALID, total / count.clamp_min(1e-6), torch.full_like(total, math.inf))
    return cost.view(n, k, *cost.shape[-2:])


def refine(before, now, field, block):
    """(N, 2, h, w): a whole-pixel `field` moved below a pixel per block, on each axis by an
    equiangular fit through its cost and the costs a pixel either side - two lines of one slope,
    the shape a mean absolute difference has near its minimum."""
    steps = torch.tensor([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], device=field.device, dtype=field.dtype)
    centre, left, right, up, down = costs(before, now, field[:, None] + steps.view(1, 5, 2, 1, 1), block).unbind(1)

    def vertex(lower, upper):
        curve = 2 * (torch.maximum(lower, upper) - centre)
        fine = torch.isfinite(lower) & torch.isfinite(upper) & torch.isfinite(centre) & (curve > 1e-6)
        step = (lower - upper) / curve.clamp_min(1e-6)
        return torch.where(fine, step.clamp(-0.5, 0.5), torch.zeros_like(step))

    return field + torch.stack([vertex(left, right), vertex(up, down)], 1)


def _neighbour(field, dy, dx):
    """The field of the block (dy, dx) away from each block, the edge repeated."""
    h, w = field.shape[-2:]
    padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
    return padded[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


class RecursiveSearch:
    """3DRS over a batch of tiles (see the module docstring).

    `block` pixels a side, `passes` updates of the whole field, the penalties of `PENALTIES`
    times `scale`. `temporal` and `seeds` say which candidates it takes: the previous pair's
    field, and the names of the tile-wide vectors (`"tile"`, `"pointer"`) it is given.
    `project` is how its field is carried to N+1: read each pixel's vector where its content
    was in N (p - v/2) instead of at p - at a motion edge, the vector of the content arriving.
    """

    def __init__(self, block=8, passes=3, scale=1.0, temporal=True, seeds=("tile",), subpixel=True,
                 project=False, snap=True, penalties=None):
        self.block, self.passes, self.scale = block, passes, scale
        self.temporal, self.seeds, self.subpixel, self.project = temporal, tuple(seeds), subpixel, project
        self.snap = snap
        self.penalties = {kind: scale * value for kind, value in (penalties or PENALTIES).items()}

    def __call__(self, previous, current, last=None, seeds=()):
        """`previous`, `current` (N, 3, H, W) in [0, 1]; `last` the previous pair's whole-pixel
        field (N, 2, h, w) or None; `seeds` a list of vectors (x, y) in stream pixels, each (N, 2)
        for the whole tile or (N, 2, H, W) per pixel (taken at each block's centre). Returns
        (field, fine): (N, 2, h, w) each - whole pixels, the state to keep; below a pixel, what to
        carry."""
        before, now = luma(previous), luma(current)
        n, _, height, width = now.shape
        b, penalty = self.block, self.penalties
        h, w = math.ceil(height / b), math.ceil(width / b)

        def choose(candidates, penalties):
            stack = torch.stack(candidates, 1)
            added = torch.tensor(penalties, device=now.device, dtype=now.dtype).view(1, -1, 1, 1)
            best = (costs(before, now, stack, b) + added).argmin(1)
            return stack.gather(1, best[:, None, None].expand(-1, 1, 2, -1, -1))[:, 0]

        rows = (torch.arange(h, device=now.device) * b + b // 2).clamp_max(height - 1)
        cols = (torch.arange(w, device=now.device) * b + b // 2).clamp_max(width - 1)
        # the order breaks ties - the first of equal costs wins: on a flat block every vector
        # matches, and it should take the tile's motion, not stop while its neighbours scroll
        fixed, fixed_penalties = [], []
        for seed in seeds:
            seed = seed.to(now.dtype)
            seed = seed[:, :, rows][:, :, :, cols] if seed.dim() == 4 else seed[:, :, None, None].expand(-1, -1, h, w)
            fixed.append(seed.round())
            fixed_penalties.append(penalty["seed"])
        if last is not None and self.temporal:
            fixed.append(last)
            fixed_penalties.append(penalty["temporal"])
        fixed.append(now.new_zeros(n, 2, h, w))
        fixed_penalties.append(penalty["zero"])
        field = choose(fixed, fixed_penalties)

        steps = torch.tensor(UPDATES, device=now.device, dtype=now.dtype)
        by = torch.arange(h, device=now.device).view(h, 1)
        bx = torch.arange(w, device=now.device).view(1, w)
        for k in range(self.passes):
            around = [_neighbour(field, dy, dx) for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0))]
            first = steps[(bx + 2 * by + 3 * k) % len(UPDATES)].permute(2, 0, 1)
            second = steps[(bx + 2 * by + 3 * k + 7) % len(UPDATES)].permute(2, 0, 1)
            candidates = [field, *around, field + first, around[k % 2] + second, around[2 + k % 2] + first, *fixed]
            penalties = [penalty["spatial"]] * 5 + [penalty["update"]] * 3 + fixed_penalties
            field = choose(candidates, penalties)
        fine = refine(before, now, field, b) if self.subpixel else field
        if self.snap and seeds and seeds[0].dim() == 2:
            # a block that took the tile's motion carries the tile search's measure of it, not a
            # fit of its own on 64 noisy pixels - so a scroll stays one motion, and no two
            # neighbours round half of it to different whole pixels
            tile = seeds[0].to(now.dtype)[:, :, None, None].expand(-1, -1, h, w)
            fine = torch.where((field == tile.round()).all(1, keepdim=True), tile, fine)
        return field, fine


def full_search(previous, current, block=8, radius=16, subpixel=True):
    """(field, fine) as `RecursiveSearch` gives them: every whole-pixel vector within `radius`
    tried on every block, the cheapest kept (`TIE_BREAK` towards small ones) - the control."""
    before, now = luma(previous), luma(current)
    n, _, height, width = now.shape
    h, w = math.ceil(height / block), math.ceil(width / block)
    best = now.new_full((n, h, w), math.inf)
    field = now.new_zeros(n, 2, h, w)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            vector = now.new_tensor([dx, dy]).view(1, 1, 2, 1, 1).expand(n, 1, 2, h, w)
            cost = costs(before, now, vector, block)[:, 0] + TIE_BREAK * (abs(dx) + abs(dy))
            better = cost < best
            best = torch.where(better, cost, best)
            field = torch.where(better[:, None], vector[:, 0], field)
    return field, (refine(before, now, field, block) if subpixel else field)


def carried(current, fine, block, share=0.5, project=False):
    """`current` (N, 3, H, W) with each block's content moved on by `share` of its vector, rounded
    to whole pixels. With `project`, each pixel takes the vector found where its content was in
    N - p minus that share of its own vector - instead of the vector of the block it is in."""
    height, width = current.shape[-2:]
    v = per_pixel(fine, height, width, block)
    if project:
        xs = torch.arange(width, device=v.device, dtype=v.dtype).view(1, 1, width)
        ys = torch.arange(height, device=v.device, dtype=v.dtype).view(1, height, 1)
        sx = (xs - share * v[:, 0]).round().clamp(0, width - 1).long()
        sy = (ys - share * v[:, 1]).round().clamp(0, height - 1).long()
        index = (sy * width + sx).flatten(1)[:, None].expand(-1, 2, -1)
        v = v.flatten(2).gather(2, index).view_as(v)
    return models.warp(current, torch.round(-share * v).contiguous())


# ---------------------------------------------------------------- over whole clips


class Clip:
    """One source's decoded stream cut into the client's tiles, with what every method shares:
    for each received pair, the published search's cost of no motion (its gate) and the seeds,
    (x, y) in stream pixels - {"tile": (P, T, 1, 2) the tile search's motion, "minima": (P, T, 2, 2)
    the published search's next two minima, "pointer": (P, T, 1, 2), "blocks": (P, T, 2, H, W)
    per pixel, the published search's motion of each 32 px block}."""

    def __init__(self, source, manifest, published, tile_search, block=32):
        self.received = runs.received_every(manifest)
        self.stream = clips.rebuild(source, manifest)
        self.layout, self.size = clips.plan(self.stream.shape[2], self.stream.shape[1])
        self.ends = list(range(self.received, len(self.stream), self.received))
        self.frames = to_tensor(self.stream)
        _, self.still, _ = history.pairs(self.stream, published, self.layout, self.size, self.received)
        tile, _, _ = history.pairs(self.stream, tile_search, self.layout, self.size, self.received)
        minima, blocks = [], []
        for p in range(len(self.ends)):
            previous, current = self.windows([p]).unbind(1)
            order = budget.minima(published.costs(previous, current), 2, 2 * published.radius + 1)
            minima.append(-published.table[order] / published.fraction)
            blocks.append(-published.blocks(previous, current, block) / published.fraction)
        scale = manifest["scale"]
        cursor = (source.get("scene") or {}).get("cursor")
        if cursor:
            cursor = np.asarray(cursor, np.float32) / scale
            pointer = np.stack([cursor[n] - cursor[n - self.received] for n in self.ends])
        else:
            pointer = np.zeros((len(self.ends), 2), np.float32)
        pointer = np.repeat(pointer[:, None, None], len(self.layout), 1)
        self.seeds = {"tile": torch.as_tensor(tile, device=DEVICE)[:, :, None], "minima": torch.stack(minima),
                      "pointer": torch.as_tensor(pointer, device=DEVICE), "blocks": torch.stack(blocks)}

    def windows(self, pairs, frames=(-1, 0)):
        """(P T, F, 3, h, w): the tiles of each pair index in `pairs`, the frames `frames` taken
        relative to the pair's end n - -1 for n - received, 0 for n, 1 for n + 1."""
        h, w = self.size
        picked = []
        for p in pairs:
            n = self.ends[p]
            at = [n + (f * self.received if f < 0 else f) for f in frames]
            for ix, iy, *_ in self.layout:
                picked.append(self.frames[at, :, iy:iy + h, ix:ix + w])
        return torch.stack(picked)

    def track(self, search):
        """(P, T, 2, h, w): `search`'s fine field for every pair in order, each pair taking the
        field the pair before left, gated by the published search's cost of no motion."""
        last, out = None, []
        for p in range(len(self.ends)):
            previous, current = self.windows([p]).unbind(1)
            seeds = []
            for name in search.seeds:
                found = self.seeds[name][p]
                seeds += [found] if found.dim() == 4 else list(found.unbind(1))
            last, fine = search(previous, current, last, seeds)
            out.append(fine)
        return torch.stack(out)

    def searched(self, block, radius, batch=32):
        """(P, T, 2, h, w): `full_search` on every pair - no state, so many pairs at a time."""
        out = []
        for first in range(0, len(self.ends), batch):
            pairs = range(first, min(first + batch, len(self.ends)))
            previous, current = self.windows(pairs).unbind(1)
            out.append(full_search(previous, current, block, radius)[1])
        fine = torch.cat(out)
        return fine.view(len(self.ends), len(self.layout), *fine.shape[1:])


@torch.no_grad()
def load(manifest, split, fields, pictures, published, tile_search, still=0.1, start=3, chosen=None,
         batch_size=48, progress=None):
    """Every client tile of a split's clips at every received frame N from `start` received
    frames on, N+1 made by every method and scored over the region the client keeps.

    `fields` is {name: RecursiveSearch, or ("full", block, radius)} - run over each clip in
    order, carried by `carried`, gated by the published search's still tiles. `pictures` is
    {name: f(previous, current, target)} - stateless methods, and oracles. `chosen` the indices
    of the split's sources to use (all by default).

    Returns {"tiles": {"changed", "moving" (the published gate open), "label", "family"
    (see `family`), "source", "frame", "tile", "box"}, "wpsnr": {name: (S,)}, "error": {name: (S,)}
    - the weighted squared error summed over the kept region, which adds up across tiles."""
    received = runs.received_every(manifest)
    recorded = manifest["splits"][split]["sources"]
    scale = manifest["scale"]
    tiles, wpsnr, error = defaultdict(list), defaultdict(list), defaultdict(list)
    for count, index in enumerate(range(len(recorded)) if chosen is None else chosen):
        source = recorded[index]
        clip = Clip(source, manifest, published, tile_search)
        if clip.size != TILE_INPUT:
            continue
        found = {}
        for name, method in fields.items():
            if isinstance(method, tuple):
                found[name] = (clip.searched(*method[1:]), method[1], False)
            else:
                found[name] = (clip.track(method), method.block, method.project)
        moving = torch.as_tensor(clip.still >= still, device=DEVICE)
        chosen_tiles = [(p, t) for p in range(start - 1, len(clip.ends)) if clip.ends[p] + 1 < len(clip.stream)
                        and runs.valid(source, clip.ends[p], clip.received) for t in range(len(clip.layout))]
        for first in range(0, len(chosen_tiles), batch_size):
            batch = chosen_tiles[first:first + batch_size]
            pairs = [p for p, _ in batch]
            windows = torch.stack([clip.windows([p], (-1, 0, 1))[t] for p, t in batch])
            previous, current, target = windows.unbind(1)
            gate = torch.stack([moving[p, t] for p, t in batch])
            made = {name: f(previous, current, target) for name, f in pictures.items()}
            for name, (fine, block, project) in found.items():
                field = torch.stack([fine[p, t] for p, t in batch]) * gate.view(-1, 1, 1, 1)
                made[name] = carried(current, field, block, project=project)
            boxes = [(clip.layout[t][3], clip.layout[t][2], clip.layout[t][5], clip.layout[t][4]) for _, t in batch]
            scores = {name: torch.empty(len(batch), device=DEVICE) for name in made}
            sums = {name: torch.empty(len(batch), device=DEVICE) for name in made}
            for box in set(boxes):
                same = torch.as_tensor([b == box for b in boxes], device=DEVICE)
                truth = clips.kept(target[same], box)
                for name, picture in made.items():
                    part = clips.kept(picture[same], box)
                    scores[name][same] = history.sample_wpsnr(part, truth)
                    sums[name][same] = budget.squared(part[:, None], truth)[:, 0].sum(dim=(-2, -1))
            for name in made:
                wpsnr[name].append(scores[name].cpu().numpy())
                error[name].append(sums[name].cpu().numpy())
            for i, ((p, t), box) in enumerate(zip(batch, boxes)):
                n = clip.ends[p]
                y, x, h, w = box
                ix, iy = clip.layout[t][:2]
                label = budget.primary(budget.events(source, n, received, (scale * (ix + x), scale * (iy + y),
                                                                          scale * w, scale * h)))
                frames = clip.stream[[n, n + 1], iy + y:iy + y + h, ix + x:ix + x + w]
                tiles["changed"].append(bool(runs.changed_share(frames[0], frames[1]) >= runs.MOVING_SHARE))
                tiles["moving"].append(bool(gate[i]))
                tiles["label"].append(label)
                tiles["family"].append(family(label))
                tiles["source"].append(index)
                tiles["frame"].append(n)
                tiles["tile"].append(t)
                tiles["box"].append(box)
        if progress:
            progress(count, index)
    return {"tiles": {key: np.asarray(value) for key, value in tiles.items()},
            "wpsnr": {name: np.concatenate(value) for name, value in wpsnr.items()},
            "error": {name: np.concatenate(value) for name, value in error.items()}}


FAMILIES = ("whole tile moves", "part of the tile moves", "unannounced", "screen", "video", "other")


def family(label):
    """Notebook 8's family of one of `budget.LABELS`."""
    if label.startswith("scroll") and not label.endswith(": starts"):
        return "whole tile moves"
    if label.startswith(("drag", "region")) and not label.endswith(": starts"):
        return "part of the tile moves"
    if label == "menu opens or closes at N+1" or label.endswith(": starts"):
        return "unannounced"
    return label if label in ("screen", "video") else "other"
