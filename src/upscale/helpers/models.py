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

import copy

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


def activation(slope=0.0):
    """ReLU, or with a `slope` a leaky ReLU: a negative value passes at `slope` times its size,
    so a channel that is negative everywhere still learns instead of switching off for good
    (notebook 16). Both are inside the WebGPU op budget (`Relu`, `LeakyRelu`)."""
    return nn.LeakyReLU(slope, inplace=True) if slope else nn.ReLU(inplace=True)


class ResidualBlock(nn.Module):
    """x + 1x1(ReLU(3x3(x))). The 3x3 reaches one pixel further, the 1x1 none, so a block
    costs one pixel of halo; the block's input added back keeps a deep body easy to train."""

    def __init__(self, width, slope=0.0):
        super().__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=1)
        self.point = nn.Conv2d(width, width, 1)
        self.act = activation(slope)

    def forward(self, x):
        return x + self.point(self.act(self.conv(x)))


class LumaUpscaler(Upscaler):
    """Bicubic for all three colours, plus a learned correction of brightness only.

    The eye resolves fine detail in brightness (luma) far better than in colour, so the
    network spends everything on luma. It predicts one correction per output sub-pixel and
    adds the same value to R, G and B: in BT.709 that moves Y by exactly that value and
    leaves Cb and Cr as bicubic made them. The three copies are a `Concat`.

    The body: `widths` plain convolutions (the first `stem` x `stem`, then 3x3s), then
    `blocks` `ResidualBlock`s at the last width, then a 3x3 tail that starts at zero.
    `luma_in=True` feeds the body Y alone (a frozen 1x1 convolution) instead of RGB. A
    `slope` makes every ReLU leaky (see `activation`).
    """

    def __init__(self, widths=(24,), blocks=2, stem=3, skip=5, scale=2, luma_in=False, slope=0.0):
        super().__init__(scale, skip)
        self.widths, self.blocks, self.stem, self.luma_in = tuple(widths), blocks, stem, luma_in
        self.slope = slope
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
            layers += [nn.Conv2d(channels, width, kernel, padding=kernel // 2), activation(slope)]
            channels = width
        layers += [ResidualBlock(channels, slope) for _ in range(blocks)]
        self.body = nn.Sequential(*layers)
        self.tail = zero_start(nn.Conv2d(channels, scale * scale, 3, padding=1))
        # the 3x3s after the stem, one per block, and the tail
        self.halo = self.halo_for(stem, len(widths) - 1 + blocks + 1)

    def residual(self, x):
        luma = self.tail(self.body(x))
        return torch.cat([luma, luma, luma], dim=1)   # channel c * scale² + phase, for every colour c


class ToldLuma(Upscaler):
    """`LumaUpscaler` given a condition (N, K, 1, 1) as a second input, which shifts each
    channel after the stem and inside each residual block (notebook 8). For one stream the
    condition is a constant, so `baked(condition)` folds it into the biases and returns a
    plain `LumaUpscaler`: the graph that ships."""

    def __init__(self, conditions, width=24, blocks=2, stem=3, skip=5, scale=2, slope=0.0):
        super().__init__(scale, skip)
        self.width, self.blocks, self.stem, self.slope = width, blocks, stem, slope
        # the same layers as LumaUpscaler((width,), blocks, stem), in the same order, so the
        # same seed starts both alike and `baked` can copy them across
        self.body = nn.Sequential(nn.Conv2d(3, width, stem, padding=stem // 2), activation(slope),
                                  *[ResidualBlock(width, slope) for _ in range(blocks)])
        self.tail = zero_start(nn.Conv2d(width, scale * scale, 3, padding=1))
        self.halo = self.halo_for(stem, blocks + 1)
        self.shifts = nn.ModuleList([nn.Conv2d(conditions, width, 1, bias=False) for _ in range(blocks + 1)])
        for shift in self.shifts:
            nn.init.zeros_(shift.weight)   # untrained, the condition changes nothing

    def residual(self, x, condition):
        h = self.body[1](self.body[0](x) + self.shifts[0](condition))
        for block, shift in zip(self.body[2:], self.shifts[1:]):
            h = h + block.point(block.act(block.conv(h) + shift(condition)))
        luma = self.tail(h)
        return torch.cat([luma, luma, luma], dim=1)

    def forward(self, x, condition):
        return self.shuffle(self.skip(x) + self.residual(x, condition))

    @torch.no_grad()
    def baked(self, condition):
        """The same model for one condition, as a plain LumaUpscaler: each shift folded into
        the bias of the convolution it follows."""
        device = next(self.parameters()).device
        plain = LumaUpscaler((self.width,), self.blocks, self.stem, self.skip_size, self.scale,
                             slope=self.slope).to(device)
        plain.load_state_dict({k: v for k, v in self.state_dict().items() if not k.startswith("shifts.")})
        c = torch.as_tensor(condition, dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
        convs = [plain.body[0], *[block.conv for block in plain.body[2:]]]
        for conv, shift in zip(convs, self.shifts):
            conv.bias += shift(c).flatten()
        return plain.eval()


class QuarterBicubicSkip(nn.Module):
    """`BicubicSkip` for 2x2 LR pixels at once: one 6x6 stride-2 convolution, 3 -> 3 * 16, whose
    `PixelShuffle(4)` is bicubic x2. Output cell i covers LR pixels 2i and 2i+1, whose 5x5 bicubic
    windows together span LR 2i-2 .. 2i+3: six taps, two of padding at each end - the same zeros
    `BicubicSkip` pads with, so the two agree everywhere, the border included. Channel
    c * 16 + 4 * sy + sx is colour c at HR row sy, column sx of the cell's 4x4 output."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 48, 6, stride=2, padding=2, bias=False)
        phases = bicubic_kernel(2, 5)              # (phase dy * 2 + dx, 5, 5) around one LR pixel
        weight = torch.zeros_like(self.conv.weight)
        for channel in range(3):
            for sy in range(4):
                for sx in range(4):
                    # HR (sy, sx) of the cell is LR pixel (sy // 2, sx // 2) of it at phase (sy % 2, sx % 2)
                    rows, cols = slice(sy // 2, sy // 2 + 5), slice(sx // 2, sx // 2 + 5)
                    weight[channel * 16 + sy * 4 + sx, channel, rows, cols] = phases[(sy % 2) * 2 + sx % 2]
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


def truncated_quarter_skip(trainable=False):
    """A 4x4 stride-2 convolution, 3 -> 48: `QuarterBicubicSkip` without its outer ring of taps
    (LR 2i-2 and 2i+3), each output renormalised to sum to 1. The taps it drops weigh 0.023 in
    bicubic's 1-D kernel, and it reads 16 LR pixels a colour instead of 36 - in the browser it
    costs half as much as the exact skip. Frozen, its colour is bicubic's to 0.01 dB of U and V
    PSNR (notebook 24). `trainable=True` leaves it to train with the model; under a loss that
    is mostly luma SSIM that costs about 1 dB of chroma and buys nothing."""
    conv = nn.Conv2d(3, 48, 4, stride=2, padding=1, bias=False)
    with torch.no_grad():
        weight = QuarterBicubicSkip().conv.weight[:, :, 1:5, 1:5].clone()
        weight /= weight.sum(dim=(1, 2, 3), keepdim=True)
        conv.weight.copy_(weight)
    conv.weight.requires_grad_(trainable)
    return conv


class FastLuma(nn.Module):
    """A told luma corrector small enough for 60 fps at a 1280x720 stream on a GTX 1660 Ti
    (notebook 24). Colour is bicubic; the network adds one correction per output pixel to
    R, G and B alike, as `LumaUpscaler` does, and is told the frame's condition as
    `ToldLuma` is: a shift per channel after the stem and inside each 3x3 block, which
    folds into those convolutions' biases.

    `layout="quarter"` runs the body at a quarter of the pixels: a `stem` x `stem` stride-2
    convolution (4: reaches 1 LR px past its 2x2 cell; 2: none) makes `width` channels for each
    2x2 LR cell, `blocks` residual blocks (`ResidualBlock`: a 3x3, then a 1x1, added back) each
    reach 2 LR px further, `points` 1x1 residual blocks reach no further, and a 1x1 tail writes
    the 16 HR luma values of the cell. The skip is `QuarterBicubicSkip` and one
    `PixelShuffle(4)` makes the picture.

    `layout="lr"` is the same at the LR pixels: a `stem` x `stem` stem, the blocks, a 1x1 tail
    writing 4 luma phases, `BicubicSkip` and `PixelShuffle(2)`.

    The quarter layout's skip (`skip`): "bicubic", the exact `QuarterBicubicSkip`; "truncated",
    `truncated_quarter_skip` frozen - half the cost, and the same colour; "linear", the same 4x4
    trained with the rest, which trades colour away (notebook 24).

    Called without a condition, it runs untold (shifts left out): what `baked` returns."""

    def __init__(self, conditions=3, width=32, blocks=1, points=0, stem=4, layout="quarter", slope=0.1,
                 skip="bicubic"):
        super().__init__()
        self.conditions, self.width, self.blocks, self.points = conditions, width, blocks, points
        self.stem_size, self.layout, self.slope, self.skip_kind = stem, layout, slope, skip
        if skip not in ("bicubic", "truncated", "linear") or (skip != "bicubic" and layout != "quarter"):
            raise ValueError(f"no skip {skip!r} for layout {layout!r}")
        if layout == "quarter":
            self.stem = nn.Conv2d(3, width, stem, stride=2, padding=(stem - 2) // 2)
            self.skip = (QuarterBicubicSkip() if skip == "bicubic"
                         else truncated_quarter_skip(trainable=(skip == "linear")))
            self.cell, self.shuffle = 4, nn.PixelShuffle(4)
            # LR px past the cell: the stem's, then 2 for every 3x3 at this resolution - and
            # never less than the skip's (2 for bicubic, 1 for the 4x4)
            self.halo = max((stem - 2) // 2 + 2 * blocks, 2 if skip == "bicubic" else 1)
        elif layout == "lr":
            self.stem = nn.Conv2d(3, width, stem, padding=stem // 2)
            self.skip, self.cell, self.shuffle = BicubicSkip(2, 5), 2, nn.PixelShuffle(2)
            self.halo = max(stem // 2 + blocks, 2)
        else:
            raise ValueError(f"no layout {layout!r}")
        self.act = activation(slope)
        self.body = nn.ModuleList([ResidualBlock(width, slope) for _ in range(blocks)])
        self.point_blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(width, width, 1), activation(slope),
                                                         nn.Conv2d(width, width, 1)) for _ in range(points)])
        self.tail = zero_start(nn.Conv2d(width, self.cell * self.cell, 1))
        self.shifts = nn.ModuleList([nn.Conv2d(conditions, width, 1, bias=False) for _ in range(blocks + 1)])
        for shift in self.shifts:
            nn.init.zeros_(shift.weight)   # untrained, the condition changes nothing

    def shifted(self):
        """The convolutions a condition shifts, in the order of `shifts`."""
        return [self.stem, *[block.conv for block in self.body]]

    def residual(self, x, condition=None):
        def told(index, value):
            return value if condition is None else value + self.shifts[index](condition)

        h = self.act(told(0, self.stem(x)))
        for index, block in enumerate(self.body, start=1):
            h = h + block.point(block.act(told(index, block.conv(h))))
        for point in self.point_blocks:
            h = h + point(h)
        luma = self.tail(h)
        return torch.cat([luma, luma, luma], dim=1)   # colour c * cell² + sub-pixel, for every colour

    def forward(self, x, condition=None):
        return self.shuffle(self.skip(x) + self.residual(x, condition))

    @torch.no_grad()
    def baked(self, condition):
        """The same model for one condition, called with the picture alone: each shift folded
        into the bias of the convolution it follows."""
        plain = copy.deepcopy(self).eval()
        device = next(plain.parameters()).device
        c = torch.as_tensor(condition, dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
        for conv, shift in zip(plain.shifted(), plain.shifts):
            conv.bias += shift(c).flatten()
            nn.init.zeros_(shift.weight)
        return plain


class FastLumaShipped(nn.Module):
    """`FastLuma` as the client runs it: the picture and the frame's three numbers in, the picture
    out (notebook 24). As notebook 23's `Shipped` does for `ToldLuma`, each shift becomes a 1x1
    convolution over a 1x1 picture holding the condition, with the bias of the convolution it
    shifts as its own bias, so what comes out is that convolution's bias for this frame and
    nothing per pixel is added. The tail's rows are written out once per colour, so the graph
    adds the luma correction to R, G and B without a `Concat`."""

    def __init__(self, model):
        super().__init__()
        model = copy.deepcopy(model).cpu().eval()
        self.skip, self.shuffle, self.stem, self.act = model.skip, model.shuffle, model.stem, model.act
        self.body, self.point_blocks = model.body, model.point_blocks
        self.tail = nn.Conv2d(model.width, 3 * model.cell * model.cell, 1)
        with torch.no_grad():
            self.tail.weight.copy_(model.tail.weight.repeat(3, 1, 1, 1))
            self.tail.bias.copy_(model.tail.bias.repeat(3))
        self.biases = nn.ModuleList()
        for conv, shift in zip(model.shifted(), model.shifts):
            line = nn.Conv2d(shift.in_channels, shift.out_channels, 1)
            with torch.no_grad():
                line.weight.copy_(shift.weight)
                line.bias.copy_(conv.bias)
            self.biases.append(line)

    def forward(self, input, condition):
        c = condition.reshape(1, -1, 1, 1)
        bias = [line(c).reshape(-1) for line in self.biases]
        stem = self.stem
        h = self.act(F.conv2d(input, stem.weight, bias[0], stride=stem.stride, padding=stem.padding))
        for block, b in zip(self.body, bias[1:]):
            h = h + block.point(block.act(F.conv2d(h, block.conv.weight, b, padding=block.conv.padding)))
        for point in self.point_blocks:
            h = h + point(h)
        return self.shuffle(self.skip(input) + self.tail(h))


def check_quarter_skip_is_bicubic():
    """`QuarterBicubicSkip` must give exactly `BicubicSkip`'s picture, border included. Returns
    the largest difference."""
    probe = torch.rand(2, 3, 20, 28)
    ours = nn.PixelShuffle(4)(QuarterBicubicSkip()(probe))
    theirs = nn.PixelShuffle(2)(BicubicSkip(2, 5)(probe))
    difference = (ours - theirs).abs().max().item()
    assert difference < 1e-5, f"the quarter skip is not bicubic ({difference:.1e})"
    return difference


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
