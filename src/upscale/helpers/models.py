"""The upscaler models the notebooks share.

Every model here has the same three parts:

    input (LR) ──► skip: bicubic x2, written as a frozen convolution ──┐
              └──► body: a few convolutions that predict a correction ─┴─► add ─► PixelShuffle ─► output (HR)

- The **skip** is plain bicubic upscaling. It is a fixed `Conv2d` instead of
  `F.interpolate`, because `F.interpolate` exports to ONNX as `Resize`, which is slow
  in the browser. A convolution is fast everywhere.
- The **body** learns only the difference between bicubic and the real picture. Its last
  layer starts at zero, so an untrained model is exactly bicubic.
- The **PixelShuffle** turns `3 * scale²` low resolution channels into one image `scale`
  times larger. Both branches are added before it, at low resolution, which is cheaper.

The **halo** of a model is how many input pixels around a tile it needs to see: half the
first kernel, plus one for every 3x3 convolution after it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bicubic(x, scale=2):
    """Plain bicubic upscaling, the baseline every model is compared to."""
    return F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)


class Resize(nn.Module):
    """A plain filter ("nearest", "bilinear" or "bicubic") as a module, so it can be scored
    and exported like a model. It exports to ONNX as one `Resize` operator."""

    def __init__(self, scale=2, mode="bicubic"):
        super().__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        if self.mode == "nearest":
            return F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)


def bicubic_kernel(scale=2, size=5):
    """Bicubic upscaling as `scale²` small kernels, one per output sub-pixel.

    Bicubic is linear, so we can read its weights straight out of `F.interpolate`: put a
    single 1 into an empty image and see where it lands in the output.
    """
    weights = torch.zeros(scale * scale, size, size)
    centre = size // 2
    for y in range(size):
        for x in range(size):
            impulse = torch.zeros(1, 1, size, size)
            impulse[0, 0, y, x] = 1.0
            out = bicubic(impulse, scale)[0, 0]
            for dy in range(scale):
                for dx in range(scale):
                    weights[dy * scale + dx, y, x] = out[centre * scale + dy, centre * scale + dx]
    return weights


class BicubicSkip(nn.Module):
    """Bicubic upscaling as one frozen convolution (no `Resize` in the ONNX graph)."""

    def __init__(self, scale=2, size=5):
        super().__init__()
        self.conv = nn.Conv2d(3, 3 * scale * scale, size, padding=size // 2, bias=False)
        phases = bicubic_kernel(scale, size)
        weight = torch.zeros_like(self.conv.weight)
        for channel in range(3):  # each colour only reads itself
            for phase in range(scale * scale):
                weight[channel * scale * scale + phase, channel] = phases[phase]
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


def zero_start(conv):
    """Set a convolution to zero, so the model starts out as exactly bicubic."""
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class Upscaler(nn.Module):
    """The shared frame: skip + residual, then one PixelShuffle.

    A subclass builds its own layers and implements `residual(x)`, which must return
    `3 * scale²` channels at the input's size.
    """

    def __init__(self, scale=2, skip=5):
        super().__init__()
        self.scale = scale
        self.skip = BicubicSkip(scale, skip)
        self.shuffle = nn.PixelShuffle(scale)
        self.skip_size = skip

    def halo_for(self, stem, num_3x3):
        """Half the first kernel, one per 3x3 after it, and never less than the skip needs."""
        return max(stem // 2 + num_3x3, self.skip_size // 2)

    def residual(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.shuffle(self.skip(x) + self.residual(x))


class PlainUpscaler(Upscaler):
    """The model that ships: a `stem`x`stem` convolution, 3x3s after it, a 3x3 tail.

    `widths=(32, 16)` is the "32-16" model the notebooks talk about.
    """

    def __init__(self, widths=(32, 16), stem=5, skip=5, scale=2):
        super().__init__(scale, skip)
        layers, channels = [], 3
        for index, width in enumerate(widths):
            kernel = stem if index == 0 else 3
            layers += [nn.Conv2d(channels, width, kernel, padding=kernel // 2), nn.ReLU(inplace=True)]
            channels = width
        self.widths, self.stem = tuple(widths), stem
        self.body = nn.Sequential(*layers)
        self.tail = zero_start(nn.Conv2d(channels, 3 * scale * scale, 3, padding=1))
        self.halo = self.halo_for(stem, len(widths))

    def residual(self, x):
        return self.tail(self.body(x))


class RepConv(nn.Module):
    """A convolution trained as three branches that add up to one convolution.

    While training: a k x k, a 1x1, and a wide 1x1 followed by a k x k. All three are
    linear, so after training `collapse()` merges them into a single k x k with the same
    output. More capacity to learn with, same cost to run.
    """

    def __init__(self, in_channels, out_channels, kernel=3, expand=4):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2)
        self.point = nn.Conv2d(in_channels, out_channels, 1)
        # no bias on the first 1x1: the k x k after it would pad that bias with zeros at
        # the border, and the merged kernel could not do the same
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expand, 1, bias=False),
            nn.Conv2d(in_channels * expand, out_channels, kernel, padding=kernel // 2))

    def forward(self, x):
        return self.conv(x) + self.point(x) + self.expand(x)

    @torch.no_grad()
    def collapse(self):
        """The single convolution the three branches add up to."""
        merged = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, self.kernel,
                           padding=self.kernel // 2)
        centre = self.kernel // 2
        weight = self.conv.weight.clone()
        # a 1x1 is a k x k that only has a centre tap
        weight[:, :, centre, centre] += self.point.weight[:, :, 0, 0]
        # a 1x1 only mixes channels, so it is a matrix (mid x in). A 1x1 then a k x k is one
        # k x k whose weights are the second conv's weights times that matrix.
        first, second = self.expand[0].weight[:, :, 0, 0], self.expand[1].weight
        weight += torch.einsum("omuv,mi->oiuv", second, first)
        merged.weight.copy_(weight)
        merged.bias.copy_(self.conv.bias + self.point.bias + self.expand[1].bias)
        return merged


class ReparamUpscaler(Upscaler):
    """`PlainUpscaler` with every body convolution trained as a `RepConv`."""

    def __init__(self, widths=(32, 16), stem=5, expand=4, skip=5, scale=2):
        super().__init__(scale, skip)
        layers, channels = [], 3
        for index, width in enumerate(widths):
            kernel = stem if index == 0 else 3
            layers += [RepConv(channels, width, kernel, expand), nn.ReLU(inplace=True)]
            channels = width
        self.widths, self.stem = tuple(widths), stem
        self.body = nn.Sequential(*layers)
        self.tail = zero_start(nn.Conv2d(channels, 3 * scale * scale, 3, padding=1))
        self.halo = self.halo_for(stem, len(widths))

    def residual(self, x):
        return self.tail(self.body(x))

    @torch.no_grad()
    def collapsed(self):
        """The same trained model as a `PlainUpscaler` - what gets exported."""
        plain = PlainUpscaler(self.widths, self.stem, self.skip_size, self.scale)
        reps = [m for m in self.body if isinstance(m, RepConv)]
        convs = [m for m in plain.body if isinstance(m, nn.Conv2d)]
        for rep, conv in zip(reps, convs):
            merged = rep.collapse()
            conv.weight.copy_(merged.weight)
            conv.bias.copy_(merged.bias)
        plain.tail.load_state_dict(self.tail.state_dict())
        return plain.to(next(self.parameters()).device).eval()


# BT.709 luma: Y = 0.2126 R + 0.7152 G + 0.0722 B
LUMA_709 = (0.2126, 0.7152, 0.0722)


class ResidualBlock(nn.Module):
    """x + 1x1(ReLU(3x3(x))). The 3x3 reaches one pixel further, the 1x1 none, so a block
    costs one pixel of halo; the block's input added back keeps a deep body easy to train."""

    def __init__(self, width):
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=1)
        self.point = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return x + self.point(torch.relu(self.conv(x)))


class LumaUpscaler(Upscaler):
    """Bicubic for all three colours, plus a learned correction of brightness only.

    The eye resolves fine detail in brightness (luma) far better than in colour, so the
    network spends everything on luma. It predicts one correction per output sub-pixel and
    adds the same value to R, G and B: in BT.709 that moves Y by exactly that value and
    leaves Cb and Cr as bicubic made them. The three copies are a `Concat`.

    The body: `widths` plain convolutions (the first `stem` x `stem`, then 3x3s), then
    `blocks` `ResidualBlock`s at the last width, then a 3x3 tail that starts at zero.
    `luma_in=True` feeds the body Y alone (a frozen 1x1 convolution) instead of RGB.
    """

    def __init__(self, widths=(24,), blocks=2, stem=3, skip=5, scale=2, luma_in=False):
        super().__init__(scale, skip)
        self.widths, self.blocks, self.stem, self.luma_in = tuple(widths), blocks, stem, luma_in
        channels = 3
        layers = []
        if luma_in:
            to_luma = nn.Conv2d(3, 1, 1, bias=False)
            with torch.no_grad():
                to_luma.weight.copy_(torch.tensor(LUMA_709)[None, :, None, None])
            to_luma.weight.requires_grad_(False)
            layers.append(to_luma)
            channels = 1
        for index, width in enumerate(widths):
            kernel = stem if index == 0 else 3
            layers += [nn.Conv2d(channels, width, kernel, padding=kernel // 2), nn.ReLU(inplace=True)]
            channels = width
        layers += [ResidualBlock(channels) for _ in range(blocks)]
        self.body = nn.Sequential(*layers)
        self.tail = zero_start(nn.Conv2d(channels, scale * scale, 3, padding=1))
        # the 3x3s after the stem, one per block, and the tail
        self.halo = self.halo_for(stem, len(widths) - 1 + blocks + 1)

    def residual(self, x):
        luma = self.tail(self.body(x))
        return torch.cat([luma, luma, luma], dim=1)   # channel c * scale² + phase, for every colour c


def count_parameters(model, learned_only=False):
    """How many numbers the model holds. `learned_only=True` leaves out frozen ones, such as
    the bicubic skip (900 weights at x2)."""
    return sum(p.numel() for p in model.parameters() if not learned_only or p.requires_grad)


def check_skip_is_bicubic(scale=2, size=5):
    """The conv skip must give the same picture as `F.interpolate` (away from the border,
    where a convolution pads with zeros). Returns the largest difference."""
    probe = torch.rand(1, 3, 24, 24)
    edge = 2 * size
    ours = nn.PixelShuffle(scale)(BicubicSkip(scale, size)(probe))
    difference = (ours - bicubic(probe, scale))[..., edge:-edge, edge:-edge].abs().max().item()
    assert difference < 1e-5, f"the conv skip is not bicubic ({difference:.1e})"
    return difference
