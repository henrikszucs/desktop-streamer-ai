"""Extrapolators, with the client's contract.

Every model is called as `model(previous, current)` - (N, 3, H, W) in [0, 1] each, the two
inputs `src/mock/make_mock_models.py` gives `extrapolate.onnx` - and returns the frame it
expects next, the same size. The halo is the model's business: the scores cut it off.

- `Repeat`: the newest frame again. What a client shows through a stall today.
- `Blend(step)`: `current + step * (current - previous)`, clipped. `step=1` is the mock's
  linear extrapolation, which is right only where brightness changes at a constant rate - a
  fade - and wrong wherever something moves: a moved edge is drawn twice, once dark and once
  light.
- `Shift(radius, block)`: motion instead of brightness. Every block of `current` is matched
  to `previous` at every whole-pixel offset within `radius`, and the best offset is applied
  once more. `block=None` finds one offset for the whole picture, which is right for a
  scroll and nothing else. A block can only follow motion the input shows, so `radius` must
  not exceed the halo: content that moves further came from outside the tile. `fraction`
  carries the motion only that share of the way: ½ when the inputs are two received frames
  N-2 and N and the output is N+1, as in 2x frame generation.

The learned ones add a correction to `current`, and their last convolution starts at zero,
so untrained they are `Repeat` exactly - as upscale's models start as bicubic:

- `TwoScale`: the two frames side by side, taken down to a quarter of the resolution by
  strided convolutions, where a few 3x3 convolutions reach far enough to follow a scroll,
  and brought back up by `DepthToSpace` with the half-resolution features beside them. The
  picture comes out of a `DepthToSpace` from 12 channels, so no convolution at full
  resolution has 3 output channels (the WebGPU provider has no fast path for that).
- `Plain`: the control - the same kind of body at full resolution, no downsampling, at about
  the same operations a pixel. If it does as well, the quarter-resolution path is not needed.

Every operator in both is `Conv`, `LeakyRelu`, `Concat`, `DepthToSpace` or `Add`.

Motion instead of pixels (notebook 3): `Motion` moves `current` by an offset per pixel with
`warp` (`GridSample`), so a correct offset copies content exactly instead of having to draw
it. The offset comes from `TileSearch` - notebook 1's tile shift built from fixed
convolutions, one motion a tile, found by search and not learned - and from a learned
correction on top, either of which can be left out. `Gate` wraps any model with the rule a
client can apply per tile: repeat where nothing moved.

Where the search is unsure (notebook 10): `Masked` adds a `TwoScale` network's correction to
the search's N+1, through a gate opened by how much of the change the search's motion leaves
unexplained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import RGB_TO_YUV


class Repeat(nn.Module):
    def forward(self, previous, current):
        return current


class Blend(nn.Module):
    def __init__(self, step=1.0):
        super().__init__()
        self.step = step

    def forward(self, previous, current):
        return (current + self.step * (current - previous)).clamp(0.0, 1.0)


class Shift(nn.Module):
    # the cost of an offset grows this much per pixel of it, so where every offset matches
    # equally well (a flat block, a still one) the block stays where it is
    PENALTY = 1e-4

    def __init__(self, radius=4, block=16, fraction=1.0):
        super().__init__()
        self.radius = radius
        self.block = block
        self.fraction = fraction
        self.offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]

    def offsets_of(self, previous, current):
        """The (N, h, w) index into `offsets` of the motion from `previous` to `current`, one per
        block of the inner picture (the radius cut off every side, where every offset is known)."""
        r = self.radius
        luma = RGB_TO_YUV[0].to(current.device, current.dtype).view(1, 3, 1, 1)
        now = (current * luma).sum(1, keepdim=True)
        before = (previous * luma).sum(1, keepdim=True)
        height, width = now.shape[-2] - 2 * r, now.shape[-1] - 2 * r
        inner = now[..., r:r + height, r:r + width]
        costs = []
        for dy, dx in self.offsets:
            # before(y - dy, x - dx) over the inner picture, which never reaches past the edge
            difference = (inner - before[..., r - dy:r - dy + height, r - dx:r - dx + width]).abs()
            if self.block:
                cost = F.avg_pool2d(difference, self.block, ceil_mode=True)
            else:
                cost = difference.mean(dim=(-2, -1), keepdim=True)
            costs.append(cost + self.PENALTY * (abs(dy) + abs(dx)))
        return torch.cat(costs, dim=1).argmin(dim=1)

    def _motion(self, previous, current):
        """(dy, dx) per pixel, each (N, H, W): each block's choice spread over its pixels, and
        out over the ring the radius cut off."""
        r = self.radius
        height, width = current.shape[-2:]
        chosen = self.offsets_of(previous, current)[:, None].float()
        inner = (height - 2 * r, width - 2 * r)
        if self.block:
            chosen = F.interpolate(chosen, scale_factor=self.block, mode="nearest")[..., :inner[0], :inner[1]]
        chosen = F.pad(chosen.expand(-1, 1, *inner), (r, r, r, r), mode="replicate").long()[:, 0]
        table = torch.tensor(self.offsets, device=current.device)
        return table[chosen].unbind(-1)

    def field(self, previous, current):
        """The same motion as `Motion.offsets` gives it, (N, 2, H, W): the (x, y) each output
        pixel is read from, relative to itself - so `warp(current, field)` is this model's
        output. A teacher for a network that has to find the motion itself."""
        dy, dx = self._motion(previous, current)
        return -torch.stack([dx, dy], dim=1).to(current.dtype)

    def forward(self, previous, current):
        if self.fraction != 1.0:
            return warp(current, self.fraction * self.field(previous, current))
        r = self.radius
        n, channels, height, width = current.shape
        dy, dx = self._motion(previous, current)
        # out(y, x) = current(y - dy, x - dx), read from a copy with its edges repeated
        rows = torch.arange(height, device=current.device).view(1, -1, 1) - dy + r
        cols = torch.arange(width, device=current.device).view(1, 1, -1) - dx + r
        padded = F.pad(current, (r, r, r, r), mode="replicate")
        index = (rows * (width + 2 * r) + cols).view(n, 1, -1).expand(-1, channels, -1)
        return padded.flatten(2).gather(2, index).view(n, channels, height, width)


def _zero(conv):
    """Start a convolution at zero, so the model it ends starts as `Repeat`."""
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class TwoScale(nn.Module):
    """current + a correction made at 1/2 and 1/4 of the resolution (see the module docstring).

    `widths` are the channels after each stride-2 convolution, one per level (each a multiple
    of 4); `blocks` 3x3 convolutions run at the deepest level. An input side must be a
    multiple of 2 ** len(widths).
    """

    def __init__(self, widths=(16, 32), blocks=3, slope=0.1, inputs=6):
        super().__init__()
        self.widths, self.blocks = tuple(widths), blocks
        act = nn.LeakyReLU(slope)
        self.down = nn.ModuleList()
        channels = inputs
        for width in self.widths:
            self.down.append(nn.Sequential(nn.Conv2d(channels, width, 3, stride=2, padding=1), act))
            channels = width
        self.body = nn.Sequential(*[layer for _ in range(blocks)
                                    for layer in (nn.Conv2d(channels, channels, 3, padding=1), act)])
        # back up one level at a time: DepthToSpace, the encoder's features of that level
        # beside it, and a 3x3 convolution to 4x the channels the next DepthToSpace needs:
        # the last one makes 12, which the final DepthToSpace turns into the 3 colours
        if len(self.widths) < 2:
            raise ValueError("TwoScale needs at least two levels: the last DepthToSpace makes the picture")
        self.up = nn.ModuleList()
        for level in range(len(self.widths) - 1, 0, -1):
            out = 12 if level == 1 else self.widths[level - 1]
            conv = nn.Conv2d(channels // 4 + self.widths[level - 1], out, 3, padding=1)
            self.up.append(_zero(conv) if level == 1 else nn.Sequential(conv, act))
            channels = out
        self.shuffle = nn.PixelShuffle(2)

    def correction(self, x):
        """(N, 3, H, W): what the network adds, from its stacked inputs `x`."""
        skips = []
        for down in self.down:
            x = down(x)
            skips.append(x)
        x = self.body(x)
        skips.pop()   # the deepest level is the body's own input
        for up in self.up:
            x = up(torch.cat([self.shuffle(x), skips.pop()], dim=1))
        return self.shuffle(x)

    def forward(self, previous, current):
        return current + self.correction(torch.cat([previous, current], dim=1))


class Plain(nn.Module):
    """current + a correction made at full resolution: a 3x3 stem, `blocks` 3x3 convolutions
    of `width` channels, and a 3x3 convolution to the 3 colours."""

    def __init__(self, width=12, blocks=3, slope=0.1):
        super().__init__()
        self.width, self.blocks = width, blocks
        act = nn.LeakyReLU(slope)
        layers = [nn.Conv2d(6, width, 3, padding=1), act]
        for _ in range(blocks):
            layers += [nn.Conv2d(width, width, 3, padding=1), act]
        self.body = nn.Sequential(*layers, _zero(nn.Conv2d(width, 3, 3, padding=1)))

    def forward(self, previous, current):
        return current + self.body(torch.cat([previous, current], dim=1))


class TileSearch(nn.Module):
    """One motion for the whole tile, found by search and built from fixed convolutions:
    notebook 1's tile shift in a form a graph can hold.

    Luma of each frame at 1/`scale` of the resolution (a fixed strided `Conv`). Then
    `previous` moved by every offset d within `radius` there - one fixed `Conv` from 1 channel
    to one per offset, each kernel a single 1 - and `current` at the kernels' centre (the same
    size of `Conv` with only its centre tap), both without padding, so only positions where
    every offset reads inside the picture count. |current - previous moved by d| is a `Sub` and
    an `Abs`, and its mean over the tile (`ReduceMean`) is the cost of each offset. A softmax of
    the costs times `sharpness` weighs the offsets, and their weighted mean is the motion -
    between the whole-pixel offsets where two of them match about as well. The cost is in luma
    levels (of 255), so `sharpness` 10 makes one level of difference worth a factor of e^10.

    `stride` samples the cost on a coarser grid: a mean over the tile needs a few thousand
    positions, not every one, and the work falls with its square. `fraction` is the share of
    the found motion to carry: 1 for consecutive frames, ½ for 2x frame generation (received
    frames N-2 and N in, N+1 out). `still` gates it: a tile whose cost at no motion - the mean
    luma difference between the two frames - is under `still` luma levels gets no motion at
    all, so the warp returns `current` as it is.

    Nothing in it learns. Outside the operators upscale measured as fast: `Sub`, `Abs`,
    `ReduceMean`, `Softmax` and a `MatMul` of 81 numbers a tile at the defaults - all on the GPU
    in the browser (notebook 3, "Speed").
    """

    def __init__(self, radius=4, scale=2, sharpness=10.0, stride=1, fraction=1.0, still=None):
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
        # (x, y) of each offset in full-resolution pixels, as "read from" (minus the motion),
        # times the share of the motion to carry - a constant, so the graph pays nothing for it
        self.register_buffer("table", -scale * fraction * torch.tensor([(dx, dy) for dy, dx in offsets],
                                                                       dtype=torch.float32))

    def difference(self, previous, current):
        """(N, offsets, h, w): |current - previous moved by d| at every position the search
        samples - 1/`scale` of the resolution, every `stride`th, `radius` in from each edge."""
        return self.compare(self.luma(previous), self.luma(current))

    def compare(self, previous, current):
        """`difference` for luma already at 1/`scale` of the resolution, (N, 1, h, w) each."""
        return (self.centre(current) - self.moved(previous)).abs()

    def choose(self, cost):
        """The motion (N, 2) that costs (N, offsets) in luma levels pick, gated by `still`."""
        weights = torch.softmax(-self.sharpness * cost, dim=1)
        motion = weights @ self.table
        if self.still is not None:
            # the cost of no motion is how much the tile changed: under `still`, nothing moved,
            # and a zero motion hands `current` back as it is - a gate for two numbers a tile
            moved = cost[:, self.no_motion:self.no_motion + 1] >= self.still
            motion = motion * moved.to(motion.dtype)
        return motion

    def forward(self, previous, current):
        """(N, 2, 1, 1): the (x, y) offset every pixel of the tile is read from."""
        cost = 255.0 * self.difference(previous, current).mean(dim=(2, 3))
        return self.choose(cost)[:, :, None, None]

    def costs(self, previous, current):
        """(N, offsets): the tile's cost of every offset, in luma levels."""
        return 255.0 * self.difference(previous, current).mean(dim=(2, 3))

    def measure(self, previous, current):
        """(motion (N, 2), still (N,), best (N,)): the whole motion from `previous` to
        `current` - how far the content moved, (x, y) in pixels, whatever `fraction` and
        ungated - the cost of no motion, and the least cost, in luma levels."""
        cost = self.costs(previous, current)
        weights = torch.softmax(-self.sharpness * cost, dim=1)
        motion = -(weights @ self.table) / self.fraction
        return motion, cost[:, self.no_motion], cost.min(1).values

    def blocks(self, previous, current, block):
        """(N, 2, H, W): one motion per `block` x `block` pixels instead of one per tile - the same
        costs, averaged over each block, each block choosing (and gated) on its own. The ring the
        search cannot sample takes the nearest block's. For scoring in torch, not in the graph."""
        cells = max(1, block // (self.scale * self.stride))
        cost = 255.0 * F.avg_pool2d(self.difference(previous, current), cells, ceil_mode=True)
        n, k, rows, cols = cost.shape
        motion = self.choose(cost.permute(0, 2, 3, 1).reshape(-1, k)).view(n, rows, cols, 2).permute(0, 3, 1, 2)
        # the full-resolution pixel each block starts at: the first sampled position is `radius`
        # half-resolution pixels in
        margin = self.scale * self.radius
        height, width = current.shape[-2:]
        y = ((torch.arange(height, device=current.device) - margin) // block).clamp(0, rows - 1)
        x = ((torch.arange(width, device=current.device) - margin) // block).clamp(0, cols - 1)
        return motion[:, :, y][:, :, :, x]


class PyramidSearch(nn.Module):
    """`TileSearch` coarse to fine: one motion a tile, found further and finer for about the
    same work (notebook 8).

    `levels` is a list of (scale, radius, stride). The first searches every offset within
    `radius` at 1/`scale` of the resolution, as `TileSearch` does - (4, 4, 1) reaches ±16
    stream pixels at a quarter of the published search's positions. Each next level moves
    `previous` by the motion found so far (`warp`, `GridSample`) and searches only what is
    left, ±`radius` at its own scale: a few offsets at half and at full resolution.

    The last level ends with `subpixel`: "softmax" weighs its offsets as `TileSearch` does;
    "parabola" fits one through the cost of the cheapest offset and its two neighbours on each
    axis - the vertex of the fit, at most half a pixel from it; "equiangular" fits two lines of one slope instead, the shape a mean
    absolute difference has near its minimum; "none" takes the cheapest whole offset. `fraction` and `still` are
    `TileSearch`'s: the share carried to N+1, and the gate on the first level's cost of no
    motion. Each level moves only the luma of `previous`, at its own resolution. Every level's
    operators are `TileSearch`'s, plus `GridSample` and a few elementwise ones; the sub-pixel
    fits add `ArgMin` and `Gather`.

    What is carried to N+1 need not be as fine as what is measured: `whole` rounds the carried
    offset to whole pixels, so N is copied, never resampled - a decoded frame read between its
    pixels is blurred, bilinearly or bicubically, more than half a pixel of position is worth
    (notebook 8). `interpolation` is the warp's mode otherwise.
    """

    def __init__(self, levels=((4, 4, 1), (2, 1, 2), (1, 2, 4)), sharpness=10.0, fraction=1.0, still=None,
                 subpixel="softmax", whole=False, interpolation="bilinear"):
        super().__init__()
        self.levels, self.sharpness, self.fraction, self.still = tuple(levels), sharpness, fraction, still
        self.subpixel, self.whole, self.interpolation = subpixel, whole, interpolation
        self.searches = nn.ModuleList([TileSearch(radius, scale, sharpness, stride) for scale, radius, stride in levels])

    def _step(self, search, cost, last):
        """(N, 2): the motion a level's costs give, in full-resolution pixels."""
        if not last or self.subpixel == "softmax":
            return -(torch.softmax(-self.sharpness * cost, dim=1) @ search.table)
        size = 2 * search.radius + 1
        best = cost.argmin(1)
        dy, dx = best // size - search.radius, best % size - search.radius
        whole = torch.stack([dx, dy], 1).to(cost.dtype)
        if self.subpixel == "none":
            return search.scale * whole

        def vertex(lower, upper, centre, inside):
            # no step where the cheapest offset is on the searched square's edge: one neighbour is missing
            if self.subpixel == "parabola":
                curve = lower - 2 * centre + upper
            else:   # "equiangular": two lines of one slope through the three, the steeper side's
                curve = 2 * (torch.maximum(lower, upper) - centre)
            step = (lower - upper) / (2 * curve.clamp_min(1e-6)) if self.subpixel == "parabola" \
                else (lower - upper) / curve.clamp_min(1e-6)
            return torch.where(inside & (curve > 1e-6), step.clamp(-0.5, 0.5), torch.zeros_like(step))

        def at(oy, ox):
            y = (dy + oy).clamp(-search.radius, search.radius) + search.radius
            x = (dx + ox).clamp(-search.radius, search.radius) + search.radius
            return cost.gather(1, (y * size + x)[:, None])[:, 0]

        centre = at(0, 0)
        fine = torch.stack([vertex(at(0, -1), at(0, 1), centre, dx.abs() < search.radius),
                            vertex(at(-1, 0), at(1, 0), centre, dy.abs() < search.radius)], 1)
        return search.scale * (whole + fine)

    def measure(self, previous, current):
        """As `TileSearch.measure`: (motion (N, 2), still (N,), best (N,))."""
        motion = current[:, :2, 0, 0] * 0               # a zero per tile that keeps N symbolic in a graph
        for i, search in enumerate(self.searches):
            before, now = search.luma(previous), search.luma(current)
            if i:
                # the luma of N-2 moved by the motion found so far, in this level's pixels
                shift = (-motion / search.scale)[:, :, None, None].expand(-1, -1, *before.shape[-2:])
                before = warp(before, shift)
            cost = 255.0 * search.compare(before, now).mean(dim=(2, 3))
            if i == 0:
                still = cost[:, search.no_motion]
            motion = motion + self._step(search, cost, i == len(self.searches) - 1)
        return motion, still, cost.min(1).values

    def forward(self, previous, current):
        """(N, 2, 1, 1): the (x, y) offset every pixel of the tile is read from, as `TileSearch`'s."""
        motion, still, _ = self.measure(previous, current)
        offsets = -self.fraction * motion
        if self.whole:
            offsets = torch.round(offsets)
        if self.still is not None:
            offsets = offsets * (still >= self.still).to(offsets.dtype)[:, None]
        return offsets[:, :, None, None]


class Motion(nn.Module):
    """`current` moved by an offset per pixel: the copy is exact, and only *where from* is
    found.

    The offset is the sum of two parts, either of which can be left out:

    - `search`: one motion for the whole tile (`TileSearch`), which gets a scroll or a pan
      right and nothing smaller;
    - the network: `TwoScale`'s encoder, and a head of 8 channels at half the resolution
      that a `DepthToSpace` turns into 2 at full - a correction (x, y, in pixels) for every
      output pixel. The head starts at zero, so untrained the model is its search alone, or
      `Repeat` without one. `learned=False` leaves the network out.

    `GridSample` reads `current` there, between pixels bilinearly, the edge repeated beyond
    the picture. It is outside the operator set upscale measured as fast, and so are the few
    elementwise operators that build its grid: the graph needs its own browser measurement.
    """

    def __init__(self, widths=(16, 32), blocks=3, search=None, learned=True, slope=0.1):
        super().__init__()
        self.widths, self.blocks, self.learned = tuple(widths), blocks, learned
        self.search = search
        if not learned:
            return
        if len(self.widths) < 2:
            raise ValueError("Motion needs at least two levels: the last DepthToSpace makes the offsets")
        act = nn.LeakyReLU(slope)
        self.down = nn.ModuleList()
        channels = 6
        for width in self.widths:
            self.down.append(nn.Sequential(nn.Conv2d(channels, width, 3, stride=2, padding=1), act))
            channels = width
        self.body = nn.Sequential(*[layer for _ in range(blocks)
                                    for layer in (nn.Conv2d(channels, channels, 3, padding=1), act)])
        self.up = nn.ModuleList()
        for level in range(len(self.widths) - 1, 0, -1):
            out = 8 if level == 1 else self.widths[level - 1]
            conv = nn.Conv2d(channels // 4 + self.widths[level - 1], out, 3, padding=1)
            self.up.append(_zero(conv) if level == 1 else nn.Sequential(conv, act))
            channels = out
        self.shuffle = nn.PixelShuffle(2)

    def correction(self, previous, current):
        """(N, 2, H, W): the network's part of the offsets."""
        x = torch.cat([previous, current], dim=1)
        skips = []
        for down in self.down:
            x = down(x)
            skips.append(x)
        x = self.body(x)
        skips.pop()
        for up in self.up:
            x = up(torch.cat([self.shuffle(x), skips.pop()], dim=1))
        return self.shuffle(x)

    def offsets(self, previous, current):
        """(N, 2, H, W): for every output pixel, the (x, y) offset in pixels it is read from."""
        # built from the parts there are, never from a zero picture: an exporter folds that
        # into a constant the size of the whole input, batch and all
        height, width = current.shape[-2:]
        if self.search is not None and self.learned:
            return self.search(previous, current) + self.correction(previous, current)
        if self.search is not None:
            return self.search(previous, current).expand(-1, -1, height, width)
        if self.learned:
            return self.correction(previous, current)
        return current[:, :2] * 0

    def forward(self, previous, current):
        return warp(current, self.offsets(previous, current), getattr(self.search, "interpolation", "bilinear"))


class Gate(nn.Module):
    """`model` where something moved between `previous` and `current`, and repeat where nothing
    did - the rule a client can apply per tile before it runs anything. "Moved" is the run set's
    own definition (`runs.CHANGED_LEVEL`, `runs.MOVING_SHARE`), over the whole input.

    In the graph it is a few elementwise operators (`Sub`, `Mul`, `ReduceSum`, `Abs`, a
    comparison, `ReduceMean`, `Where`) that return `current` for a tile where nothing moved, so
    the client needs no code for it. A client that compared the tiles itself could also skip
    running the model on them.
    """

    LUMA = (0.299, 0.587, 0.114)

    def __init__(self, model, level=8, share=0.005):
        super().__init__()
        self.model, self.level, self.share = model, level / 255, share
        self.register_buffer("luma", torch.tensor(self.LUMA).view(1, 3, 1, 1))

    def forward(self, previous, current):
        changed = (((current - previous) * self.luma).sum(1).abs() > self.level).float().mean(dim=(1, 2))
        moved = (changed >= self.share).view(-1, 1, 1, 1)
        return torch.where(moved, self.model(previous, current), current)


class Masked(nn.Module):
    """The tile search's N+1, and a network's correction added where the search is unsure
    (notebook 10): out = search's N+1 + g * r.

    - **The search** (`TileSearch`, the published one): one motion a tile, half of it carried;
      a still tile comes back as it was, correction and all.
    - **The confidence**: N-2 moved by the search's whole motion should be N. Where it is not, the
      motion does not explain the change: per `cell` px, `u` = the mean |N - N-2 moved| over the
      mean |N - N-2| (+ one luma level, so a still cell is not unsure), clipped to 0..1 - 0 where
      the search explains everything, 1 where it explains nothing.
    - **The gate**: g = sigmoid(a u + b) per pixel, `a` and `b` learned. `gated=False` is the
      control: g = 1 everywhere.
    - **The network** (`TwoScale`'s body): N, the search's N+1, the residual |N - N-2 moved| and
      `u`, and the tile's motion as two flat channels - 10 inputs - to a correction of 3. Its
      last convolution starts at zero, so untrained the model is the search exactly, with the
      gate at ½ (a = b = 0): both get a gradient from the first step.
    """

    def __init__(self, search, widths=(16, 32), blocks=3, cell=16, gated=True):
        super().__init__()
        self.search, self.cell, self.gated = search, cell, gated
        self.network = TwoScale(widths, blocks, inputs=10)
        self.gate = nn.Parameter(torch.zeros(2))     # a, b
        self.register_buffer("luma", RGB_TO_YUV[0].view(1, 3, 1, 1) * 255.0)

    def parts(self, previous, current):
        """{"search": the search's N+1, "u": (N, 1, H, W), "gate": g, "correction": r}."""
        height, width = current.shape[-2:]
        search = self.search
        cost = search.costs(previous, current)
        offsets = search.choose(cost)[:, :, None, None]                  # read from: -fraction * motion
        moving = (cost[:, search.no_motion] >= search.still).to(current.dtype).view(-1, 1, 1, 1)
        made = warp(current, offsets.expand(-1, -1, height, width))
        aligned = warp(previous, (offsets / search.fraction).expand(-1, -1, height, width))
        now = (current * self.luma).sum(1, keepdim=True)
        residual = (now - (aligned * self.luma).sum(1, keepdim=True)).abs()
        change = (now - (previous * self.luma).sum(1, keepdim=True)).abs()
        unexplained = F.avg_pool2d(residual, self.cell, ceil_mode=True)
        changed = F.avg_pool2d(change, self.cell, ceil_mode=True)
        u = (unexplained / (changed + 1.0)).clamp(0, 1)
        u = u.repeat_interleave(self.cell, 2).repeat_interleave(self.cell, 3)[..., :height, :width]
        motion = (-offsets / search.fraction / 8.0).expand(-1, -1, height, width)
        inputs = torch.cat([current, made, residual / 32.0, u, motion], dim=1)
        r = self.network.correction(inputs) * moving
        gate = torch.sigmoid(self.gate[0] * u + self.gate[1]) if self.gated else torch.ones_like(u)
        return {"search": made, "u": u, "gate": gate, "correction": r}

    def forward(self, previous, current):
        found = self.parts(previous, current)
        return found["search"] + found["gate"] * found["correction"]


def warp(picture, offsets, mode="bilinear"):
    """`picture` read at every pixel plus its (x, y) offset in pixels - bilinearly, or by
    `mode` ("bicubic", "nearest") - the edge repeated beyond the picture."""
    n, _, height, width = picture.shape
    xs = torch.arange(width, device=picture.device, dtype=picture.dtype).view(1, 1, width)
    ys = torch.arange(height, device=picture.device, dtype=picture.dtype).view(1, height, 1)
    # pixel centres in [-1, 1], the convention of align_corners=False
    grid_x = (xs + offsets[:, 0] + 0.5) * (2.0 / width) - 1.0
    grid_y = (ys + offsets[:, 1] + 0.5) * (2.0 / height) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(picture, grid, mode=mode, padding_mode="border", align_corners=False)
