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
  not exceed the halo: content that moves further came from outside the tile.

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

    def __init__(self, radius=4, block=16):
        super().__init__()
        self.radius = radius
        self.block = block
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

    def forward(self, previous, current):
        r = self.radius
        n, channels, height, width = current.shape
        chosen = self.offsets_of(previous, current)[:, None].float()
        # each block's choice spread over its pixels, and out over the ring the radius cut off
        inner = (height - 2 * r, width - 2 * r)
        if self.block:
            chosen = F.interpolate(chosen, scale_factor=self.block, mode="nearest")[..., :inner[0], :inner[1]]
        chosen = F.pad(chosen.expand(-1, 1, *inner), (r, r, r, r), mode="replicate").long()[:, 0]
        table = torch.tensor(self.offsets, device=current.device)
        dy, dx = table[chosen].unbind(-1)
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

    def __init__(self, widths=(16, 32), blocks=3, slope=0.1):
        super().__init__()
        self.widths, self.blocks = tuple(widths), blocks
        act = nn.LeakyReLU(slope)
        self.down = nn.ModuleList()
        channels = 6
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

    def forward(self, previous, current):
        x = torch.cat([previous, current], dim=1)
        skips = []
        for down in self.down:
            x = down(x)
            skips.append(x)
        x = self.body(x)
        skips.pop()   # the deepest level is the body's own input
        for up in self.up:
            x = up(torch.cat([self.shuffle(x), skips.pop()], dim=1))
        return current + self.shuffle(x)


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
