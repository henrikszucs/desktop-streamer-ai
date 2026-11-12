"""Interpolators, with the client's contract.

Every model is called as `model(previous, current)` - (N, 3, H, W) in [0, 1] each, the two
inputs `src/mock/make_mock_models.py` gives `interpolate.onnx`: received frames N and N+2 -
and returns the frame between them, N+1, the same size. The halo is the model's business: the
scores cut it off.

- `Hold`: N again - what a 30 fps stream shown at 60 fps is with no interpolation.
- `Blend(weight)`: `(1 - weight) * previous + weight * current`. `weight=½` is the mock's mean
  of the two frames: right for a fade and a still picture, and wherever something moves it
  draws it twice at half strength - a ghost at both ends of the motion.
- `Shift(radius, block)`: motion instead of brightness. The whole-pixel offset d that carries
  `previous` onto `current` best is found by search (one per tile, or one per block of
  `current`), and N+1 is N moved d/2 forwards and N+2 moved d/2 back, averaged: content
  half way, drawn once. `side` keeps one of the two instead of their mean.
- `Bilateral(radius, block)`: the classic frame rate converter's search, centred on the frame
  being made: for each block of N+1, the d whose two ends - N at -d/2 and N+2 at +d/2 - agree
  best. It needs no motion field to be split between frames, and leaves no hole.

In a graph (notebook 1 exports them):

- `TileSearch`: extrapolation's notebook 3 search, a copy - the cost of every offset between
  the two frames, averaged over the tile, a softmax, one motion - nothing learned.
- `Interpolate(search)`: N and N+2 moved half that motion towards each other by `warp`
  (`GridSample`), averaged.
- `TilePick(search)`: the same, and per pixel a choice between it and the mean of the two
  frames unmoved, by which of them agrees with itself better there: |N moved - N+2 moved|
  over a small window. Both ends known is what makes this possible - a pixel where the tile's
  motion is wrong (the still desktop under a dragged window) shows it as disagreement, at the
  very place being drawn. Extrapolation has no such evidence (its notebook 4).

Notebook 2:

- `Refine(search, parts)`: the tile search and a small network that corrects it per pixel -
  where each frame is read, how the two reads mix, and pixels of its own. Starts as the search.
- `Commit(search)`: the tile search, reading N alone where its two reads disagree - a change
  drawn once, a frame late, instead of twice at half strength.

Notebook 4:

- `Decided(search, decision)`: per tile, one of six pictures the search's motion makes - the
  search, the mean, either frame held, either frame moved alone - picked by a small network on
  six numbers of the search's own costs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import RGB_TO_YUV


def _luma(picture):
    weights = RGB_TO_YUV[0].to(picture.device, picture.dtype).view(1, 3, 1, 1)
    return (picture * weights).sum(1, keepdim=True)


class Hold(nn.Module):
    def forward(self, previous, current):
        return previous


class Blend(nn.Module):
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight

    def forward(self, previous, current):
        return (1.0 - self.weight) * previous + self.weight * current


class Shift(nn.Module):
    """One whole-pixel motion per tile (`block=None`) or per block of `current`, by search,
    split half and half between the two frames."""

    # the cost of an offset grows this much per pixel of it, so where every offset matches
    # equally well (a flat block, a still one) the block stays where it is
    PENALTY = 1e-4

    def __init__(self, radius=8, block=None, side="both"):
        super().__init__()
        self.radius, self.block, self.side = radius, block, side
        self.offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]

    def offsets_of(self, previous, current):
        """The (N, h, w) index into `offsets` of the motion from `previous` to `current`, one per
        block of the inner picture (the radius cut off every side, where every offset is known)."""
        r = self.radius
        now, before = _luma(current), _luma(previous)
        height, width = now.shape[-2] - 2 * r, now.shape[-1] - 2 * r
        inner = now[..., r:r + height, r:r + width]
        costs = []
        for dy, dx in self.offsets:
            difference = (inner - before[..., r - dy:r - dy + height, r - dx:r - dx + width]).abs()
            if self.block:
                cost = F.avg_pool2d(difference, self.block, ceil_mode=True)
            else:
                cost = difference.mean(dim=(-2, -1), keepdim=True)
            costs.append(cost + self.PENALTY * (abs(dy) + abs(dx)))
        return torch.cat(costs, dim=1).argmin(dim=1)

    def motion(self, previous, current):
        """(N, 2, H, W): the motion (x, y) from N to N+2 at every pixel, each block's spread over
        its pixels and out over the ring the radius cut off."""
        r = self.radius
        height, width = current.shape[-2:]
        chosen = self.offsets_of(previous, current)[:, None].float()
        inner = (height - 2 * r, width - 2 * r)
        if self.block:
            chosen = F.interpolate(chosen, scale_factor=self.block, mode="nearest")[..., :inner[0], :inner[1]]
        chosen = F.pad(chosen.expand(-1, 1, *inner), (r, r, r, r), mode="replicate").long()[:, 0]
        table = torch.tensor([(dx, dy) for dy, dx in self.offsets], device=current.device, dtype=current.dtype)
        return table[chosen].permute(0, 3, 1, 2)

    def forward(self, previous, current):
        return meet(previous, current, 0.5 * self.motion(previous, current), self.side)


class Bilateral(nn.Module):
    """Per block of N+1, the motion d (whole stream pixels, within `radius`) for which N read
    at -d/2 and N+2 read at +d/2 agree best in luma; N+1 is their mean there."""

    PENALTY = 1e-4

    def __init__(self, radius=8, block=16):
        super().__init__()
        self.radius, self.block = radius, block
        self.offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]

    def forward(self, previous, current):
        n, _, height, width = current.shape
        best_cost, best = None, None
        for dy, dx in self.offsets:
            half = torch.tensor([-dx / 2, -dy / 2], device=current.device, dtype=current.dtype).view(1, 2, 1, 1)
            half = half.expand(n, 2, height, width)
            before, after = warp(previous, half), warp(current, -half)
            cost = F.avg_pool2d((_luma(before) - _luma(after)).abs(), self.block, ceil_mode=True)
            cost = F.interpolate(cost + self.PENALTY * (abs(dy) + abs(dx)), scale_factor=self.block,
                                 mode="nearest")[..., :height, :width]
            made = 0.5 * (before + after)
            if best is None:
                best_cost, best = cost, made
            else:
                better = cost < best_cost
                best_cost = torch.where(better, cost, best_cost)
                best = torch.where(better, made, best)
        return best


class TileSearch(nn.Module):
    """One motion for the whole tile, found by search and built from fixed convolutions - a
    copy of `../../frame_gen_extra/helpers/models.py`'s.

    Luma of each frame at 1/`scale` of the resolution (a fixed strided `Conv`). Then
    `previous` moved by every offset d within `radius` there - one fixed `Conv` from 1 channel
    to one per offset, each kernel a single 1 - and `current` at the kernels' centre, both
    without padding, so only positions where every offset reads inside the picture count.
    |current - previous moved by d| is a `Sub` and an `Abs`, and its mean over the tile
    (`ReduceMean`) is the cost of each offset. A softmax of the costs times `sharpness` weighs
    the offsets, and their weighted mean is the motion - between the whole-pixel offsets where
    two of them match about as well. The cost is in luma levels (of 255).

    `stride` samples the cost on a coarser grid. `fraction` is the share of the found motion
    the offset carries: ½ here, since N+1 is half way from N to N+2. `still` gates it: a tile
    whose cost at no motion - the mean luma difference between the two frames - is under
    `still` luma levels gets no motion at all.

    `forward` returns (N, 2, 1, 1): the (x, y) offset a pixel of N+1 reads N+2 from -
    `fraction` of the motion from N to N+2 - and N is read at its negative.
    """

    def __init__(self, radius=4, scale=2, sharpness=10.0, stride=2, fraction=0.5, still=0.1):
        super().__init__()
        self.radius, self.scale, self.sharpness, self.stride = radius, scale, sharpness, stride
        self.fraction, self.still = fraction, still
        offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
        self.no_motion = offsets.index((0, 0))
        size = 2 * radius + 1
        luma = (RGB_TO_YUV[0].view(1, 3, 1, 1) / scale ** 2).expand(1, 3, scale, scale)
        moved = torch.zeros(len(offsets), 1, size, size)
        for i, (dy, dx) in enumerate(offsets):
            moved[i, 0, radius - dy, radius - dx] = 1.0          # previous(x - d)
        centre = torch.zeros(1, 1, size, size)
        centre[0, 0, radius, radius] = 1.0
        self.luma = nn.Conv2d(3, 1, scale, stride=scale, bias=False)
        self.moved = nn.Conv2d(1, len(offsets), size, stride=stride, bias=False)
        self.centre = nn.Conv2d(1, 1, size, stride=stride, bias=False)
        for conv, weight in ((self.luma, luma), (self.moved, moved), (self.centre, centre)):
            conv.weight.data.copy_(weight)
            conv.weight.requires_grad_(False)
        # (x, y) of each offset in full-resolution pixels, times the share of the motion to
        # carry: where N+2 is read from (N at its negative) - a constant, so the graph pays
        # nothing for it
        self.register_buffer("table", scale * fraction * torch.tensor([(dx, dy) for dy, dx in offsets],
                                                                       dtype=torch.float32))

    def difference(self, previous, current):
        """(N, offsets, h, w): |current - previous moved by d| at every position sampled."""
        return (self.centre(self.luma(current)) - self.moved(self.luma(previous))).abs()

    def choose(self, cost):
        """The offset (N, 2) that costs (N, offsets) in luma levels pick, gated by `still`."""
        weights = torch.softmax(-self.sharpness * cost, dim=1)
        motion = weights @ self.table
        if self.still is not None:
            moved = cost[:, self.no_motion:self.no_motion + 1] >= self.still
            motion = motion * moved.to(motion.dtype)
        return motion

    def costs(self, previous, current):
        """(N, offsets): the tile's cost of every offset, in luma levels."""
        return 255.0 * self.difference(previous, current).mean(dim=(2, 3))

    def forward(self, previous, current):
        return self.choose(self.costs(previous, current))[:, :, None, None]


class Interpolate(nn.Module):
    """N and N+2 moved towards each other by `search`'s motion, and averaged (`side` keeps
    one of them)."""

    def __init__(self, search, side="both"):
        super().__init__()
        self.search, self.side = search, side

    def forward(self, previous, current):
        offsets = self.search(previous, current).expand(-1, -1, *current.shape[-2:])
        return meet(previous, current, offsets, self.side)


class TilePick(nn.Module):
    """Per pixel, the tile's motion or none - whichever two frames agree on better there.

    Two candidates for N+1: `moved`, the two frames moved half the tile's motion towards each
    other and averaged (`Interpolate`), and `still`, their mean where they are. Each one's
    disagreement is |N - N+2| in luma between its two reads, averaged over a `window` x
    `window` box of the pixels inside the tile - two fixed `AveragePool`s, a column and then a
    row, whose pads they do not count: the same average as one box, since both its sum and its
    count of pixels inside the tile split by rows and columns, for 2·`window` reads a pixel
    instead of `window`². Not padded by repeating the edge: the exporter builds `F.pad`'s pads at
    run time, and onnxruntime-web's WebGPU provider gets them wrong - an output twice the tile's
    size. Where the motion is right
    the two reads are the same content and agree; where it is wrong they do not. The pick is
    soft - a sigmoid of the difference of the two disagreements times `sharpness` per luma
    level - so a pixel where both agree as well gets both, and the seam between a moved
    window and the still desktop is not a hard edge.

    `Sub`, `Abs`, `AveragePool`, `Sigmoid`, `Mul` and `Add` beyond `Interpolate`'s.
    """

    def __init__(self, search, window=5, sharpness=1.0):
        super().__init__()
        self.search, self.window, self.sharpness = search, window, sharpness

    def disagreement(self, a, b):
        difference = (255.0 * (_luma(a) - _luma(b))).abs()
        pad = self.window // 2
        column = F.avg_pool2d(difference, (self.window, 1), stride=1, padding=(pad, 0), count_include_pad=False)
        return F.avg_pool2d(column, (1, self.window), stride=1, padding=(0, pad), count_include_pad=False)

    def forward(self, previous, current):
        offsets = self.search(previous, current).expand(-1, -1, *current.shape[-2:])
        before, after = warp(previous, -offsets), warp(current, offsets)
        moved = 0.5 * (before + after)
        still = 0.5 * (previous + current)
        # > 0 where the motion explains the pixel better than no motion does
        gain = self.disagreement(previous, current) - self.disagreement(before, after)
        weight = torch.sigmoid(self.sharpness * gain)
        return still + weight * (moved - still)


class Commit(nn.Module):
    """The tile search, committing to N where its two reads disagree (notebook 2).

    N and N+2 moved half way towards each other give two reads of every pixel of N+1. Where
    they agree - |difference| in luma under `level` levels, over a `window` box - the result is
    their mean, as `Interpolate`'s; where they do not, N's read alone. A change that happens
    in one frame - a menu, the end of a wheel notch - is in N+1 as it is in N or as in N+2,
    and two frames cannot say which: the mean draws it at half strength in both places, this
    draws it once, a display frame early or late - and on the score, exactly right half the
    time and as wrong as holding the frame the other half. Soft: a sigmoid of
    `level` - disagreement at `sharpness` per level.
    """

    def __init__(self, search, window=3, level=2.0, sharpness=1.0):
        super().__init__()
        self.search, self.window, self.level, self.sharpness = search, window, level, sharpness

    def forward(self, previous, current):
        offsets = self.search(previous, current).expand(-1, -1, *current.shape[-2:])
        before, after = warp(previous, -offsets), warp(current, offsets)
        pad = self.window // 2
        difference = (255.0 * (_luma(before) - _luma(after))).abs()
        disagreement = F.avg_pool2d(F.pad(difference, (pad, pad, pad, pad), mode="replicate"), self.window, stride=1)
        agree = torch.sigmoid(self.sharpness * (self.level - disagreement))
        return before + agree * 0.5 * (after - before)


class Decided(nn.Module):
    """Per tile, one of the pictures the tile search's motion makes, as a small network decides
    from the search's own costs (notebook 4) - extrapolation's notebook 13 decision, with more
    to choose from, since both ends of the motion are known.

    Every candidate is N read at -s·d and N+2 read at +s·d, mixed a : 1 - a - d the search's
    offset (half its motion from N to N+2), s and a two numbers a tile (`CHOICES`):

    | choice | s | a | |
    | --- | --- | --- | --- |
    | search | 1 | ½ | `Interpolate`: both moved half way, averaged |
    | mean | 0 | ½ | the mock |
    | hold N | 0 | 1 | N shown again |
    | hold N+2 | 0 | 0 | N+2 shown early |
    | N moved | 1 | 1 | N moved half way alone: a change drawn once, a frame late |
    | N+2 moved | 1 | 0 | N+2 moved back half way alone: a change drawn once, a frame early |

    So the graph keeps the search's two `GridSample`s and a mix, scaled per tile: the decision
    adds nothing at the tile's resolution. A tile the search's gate calls still has d = 0.

    `features` gives the six numbers extrapolation's decision reads from the search's 81 costs
    and its motion (`FEATURES`, as `../../frame_gen_extra/helpers/models.py`'s `Decided` computes
    them; here the costs compare N and N+2), standardised by `mean` and `std` and handed to
    `decision`, any module from (N, 6) to one logit per choice (N, len(choices)): the largest
    wins. `choices` names which of `CHOICES` it picks among. Without a decision the module still
    measures (`measure`) and makes every candidate (`candidates`), for a table to train one on.

    Built for the browser's WebGPU provider as extrapolation's is (its `helpers/webexport.py`):
    no bool reshaped, no `MaxPool` - the next local minimum by one fixed `MatMul` that gathers
    every offset's neighbours - and the winner a float one-hot, so the graph also runs under
    graph capture.
    """

    FEATURES = ("log zero cost", "log best cost", "explained", "peak", "gap", "log motion")
    CHOICES = {"search": (1.0, 0.5), "mean": (0.0, 0.5), "hold N": (0.0, 1.0), "hold N+2": (0.0, 0.0),
               "N moved": (1.0, 1.0), "N+2 moved": (1.0, 0.0)}
    FAR = 1e4                                     # past any cost (255 luma levels at most)

    def __init__(self, search, decision=None, mean=None, std=None, choices=tuple(CHOICES)):
        super().__init__()
        self.search, self.decision, self.choices = search, decision, tuple(choices)
        width = len(self.FEATURES)
        self.register_buffer("mean", torch.zeros(1, width) if mean is None else
                             torch.as_tensor(mean, dtype=torch.float32).view(1, -1))
        self.register_buffer("std", torch.ones(1, width) if std is None else
                             torch.as_tensor(std, dtype=torch.float32).view(1, -1))
        self.register_buffer("table", torch.tensor([self.CHOICES[name] for name in self.choices]))
        # cost (N, offsets) @ around + outside = every offset's eight neighbours, (N, 8 * offsets)
        size = 2 * search.radius + 1
        steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        around = torch.zeros(size * size, len(steps) * size * size)
        outside = torch.zeros(1, len(steps) * size * size)
        for k, (dy, dx) in enumerate(steps):
            for y in range(size):
                for x in range(size):
                    column = (k * size + y) * size + x
                    if 0 <= y + dy < size and 0 <= x + dx < size:
                        around[(y + dy) * size + x + dx, column] = 1.0
                    else:
                        outside[0, column] = self.FAR
        self.register_buffer("around", around)
        self.register_buffer("outside", outside)
        self.grid = (len(steps), size * size)

    def measure(self, previous, current):
        """(features (N, 6), the search's offset d (N, 2), gated as it gates it)."""
        search = self.search
        cost = search.costs(previous, current)                                  # (N, offsets), luma levels
        zero, best = cost[:, search.no_motion], cost.amin(1)
        weights = torch.softmax(-search.sharpness * cost, dim=1)
        far = self.FAR
        neighbours = (cost @ self.around + self.outside).view(-1, *self.grid)
        local = neighbours.amin(1) >= cost
        minima = torch.where(local, cost, far)
        tied = (minima == best[:, None]).to(cost.dtype).sum(1) >= 2
        above = torch.where(minima > best[:, None], minima, far).amin(1)
        second = torch.where(tied, best, above)
        second = torch.where(second < far, second, cost.amax(1))
        offset = weights @ search.table                                         # N+2 read at +offset, N at -offset
        motion = offset / search.fraction                                       # N to N+2
        if search.still is not None:
            offset = offset * (zero >= search.still).to(offset.dtype)[:, None]
        # one `Concat` of six: WebGPU allows 8 storage buffers a kernel by default
        found = torch.stack([torch.log(zero + 0.01), torch.log(best + 0.01), best / (zero + 0.01), weights.amax(1),
                             torch.log(1 + (second - best) / (best + 0.01)),
                             torch.log(1 + (motion * motion).sum(1).sqrt())], 1)
        return found, offset

    def features(self, previous, current):
        return self.measure(previous, current)[0]

    def make(self, previous, current, offset, share, weight):
        """The picture of one choice a tile: N at -share·d, N+2 at +share·d, mixed weight : 1 - weight;
        `share` and `weight` (N, 1)."""
        height, width = current.shape[-2:]
        reach = (offset * share)[:, :, None, None].expand(-1, -1, height, width)
        weight = weight[:, :, None, None]
        return weight * warp(previous, -reach) + (1 - weight) * warp(current, reach)

    def candidates(self, previous, current):
        """{choice: its picture (N, 3, H, W)} for every one of `choices`, and the features."""
        found, offset = self.measure(previous, current)
        ones = offset[:, :1] * 0 + 1
        return {name: self.make(previous, current, offset, ones * self.CHOICES[name][0], ones * self.CHOICES[name][1])
                for name in self.choices}, found

    def logits(self, previous, current):
        return self.decision((self.features(previous, current) - self.mean) / self.std)

    def forward(self, previous, current):
        found, offset = self.measure(previous, current)
        logits = self.decision((found - self.mean) / self.std)
        # the winner as a float one-hot: a bool tensor reshaped would run on the provider's CPU
        winner = (logits >= logits.amax(1, keepdim=True)).to(logits.dtype)
        picked = (winner @ self.table) / winner.sum(1, keepdim=True)
        return self.make(previous, current, offset, picked[:, :1], picked[:, 1:])


def _zero(conv):
    """Start a convolution at zero, so the model it ends starts as what it refines."""
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class Refine(nn.Module):
    """The tile search, and a small network that corrects it per pixel (notebook 2).

    The search's motion moves N and N+2 half way towards each other (`before`, `after`), as in
    `Interpolate`. The network sees N, N+2, `before` and `after` - 12 channels, so it sees where
    the two reads disagree - and makes, per output pixel:

    - `flow`: a correction (x, y, in pixels) of where N is read and another of where N+2 is
      read - two, so N+1 need not be half way between them (a wheel scroll's easing);
    - `mask`: the share of `before` in the output, `sigmoid` of one channel - so a pixel that
      only one frame shows (uncovered by a dragged window, a menu that opened) can come from
      that frame alone;
    - `residual`: three channels added to the result - pixels the network draws itself.

    Each part can be left out (`parts`). Its body is `TwoScale`'s: two stride-2 3x3
    convolutions (`widths`), `blocks` 3x3 convolutions at a quarter of the resolution, and
    `DepthToSpace` back up with the half-resolution features beside them; the last
    convolution makes 4x the channels at half resolution and a `DepthToSpace` brings them to
    full. **It starts at zero**: no correction, a mask of ½, no residual - untrained, the model
    is the tile search exactly, and every gain training makes is over it.
    """

    SIZES = {"flow": 4, "mask": 1, "residual": 3}

    def __init__(self, search, parts=("flow", "mask"), widths=(16, 32), blocks=3, slope=0.1):
        super().__init__()
        self.search, self.parts = search, tuple(parts)
        self.widths, self.blocks = tuple(widths), blocks
        unknown = set(self.parts) - set(self.SIZES)
        if unknown or not self.parts:
            raise ValueError(f"parts must be some of {list(self.SIZES)}, not {self.parts}")
        self.outputs = sum(self.SIZES[part] for part in self.parts)
        act = nn.LeakyReLU(slope)
        self.down = nn.ModuleList()
        channels = 12
        for width in self.widths:
            self.down.append(nn.Sequential(nn.Conv2d(channels, width, 3, stride=2, padding=1), act))
            channels = width
        self.body = nn.Sequential(*[layer for _ in range(blocks)
                                    for layer in (nn.Conv2d(channels, channels, 3, padding=1), act)])
        self.up = nn.ModuleList()
        for level in range(len(self.widths) - 1, 0, -1):
            out = 4 * self.outputs if level == 1 else self.widths[level - 1]
            conv = nn.Conv2d(channels // 4 + self.widths[level - 1], out, 3, padding=1)
            self.up.append(_zero(conv) if level == 1 else nn.Sequential(conv, act))
            channels = out
        self.shuffle = nn.PixelShuffle(2)

    def network(self, *frames):
        """{part: (N, channels, H, W)} - the network's output, split into its parts."""
        x = torch.cat(frames, dim=1)
        skips = []
        for down in self.down:
            x = down(x)
            skips.append(x)
        x = self.body(x)
        skips.pop()
        for up in self.up:
            x = up(torch.cat([self.shuffle(x), skips.pop()], dim=1))
        return dict(zip(self.parts, self.shuffle(x).split([self.SIZES[part] for part in self.parts], dim=1)))

    def forward(self, previous, current):
        offsets = self.search(previous, current).expand(-1, -1, *current.shape[-2:])
        before, after = warp(previous, -offsets), warp(current, offsets)
        out = self.network(previous, current, before, after)
        if "flow" in out:
            before = warp(previous, out["flow"][:, :2] - offsets)
            after = warp(current, out["flow"][:, 2:] + offsets)
        if "mask" in out:
            made = after + torch.sigmoid(out["mask"]) * (before - after)
        else:
            made = 0.5 * (before + after)
        if "residual" in out:
            made = made + out["residual"]
        return made


def meet(previous, current, offsets, side="both"):
    """N+1 from N read at -`offsets` and N+2 read at +`offsets` ((N, 2, H, W) in pixels, the
    (x, y) N+2 is read from): their mean, or one of them (`side` "previous" or "current")."""
    if side == "previous":
        return warp(previous, -offsets)
    if side == "current":
        return warp(current, offsets)
    return 0.5 * (warp(previous, -offsets) + warp(current, offsets))


def warp(picture, offsets):
    """`picture` read at every pixel plus its (x, y) offset in pixels, bilinearly, the edge
    repeated beyond the picture."""
    n, _, height, width = picture.shape
    xs = torch.arange(width, device=picture.device, dtype=picture.dtype).view(1, 1, width)
    ys = torch.arange(height, device=picture.device, dtype=picture.dtype).view(1, height, 1)
    # pixel centres in [-1, 1], the convention of align_corners=False
    grid_x = (xs + offsets[:, 0] + 0.5) * (2.0 / width) - 1.0
    grid_y = (ys + offsets[:, 1] + 0.5) * (2.0 / height) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(picture, grid, mode="bilinear", padding_mode="border", align_corners=False)
