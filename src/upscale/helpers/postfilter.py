"""Filters a host fits for its client, from the full-resolution frame only the host has.

The client sorts every LR pixel into a class by the direction and strength of its edges; per
class, the host fits a small filter by least squares against its HR frame and sends the
coefficients with the stream. A post-filter changes the luma of an upscaled picture, and an HR
pixel takes its LR pixel's class. Notebook 13 measures it on the patch set, notebook 14 on whole
streamed frames.

    levels = postfilter.activity_levels(luma_of_training_frames, 5)
    pixel_class = postfilter.upsampled(postfilter.classes(postfilter.luma(lr), levels), scale)
    fit = postfilter.Fit(groups=1, classes=postfilter.count(levels), outputs=1)
    fit.add(postfilter.windows(postfilter.luma(picture)), postfilter.luma(hr).flatten(2),
            pixel_class.flatten(1), group)
    picture = postfilter.post_filter(picture, fit.solve(), pixel_class, group)

- **Classes**, from what the client has: at every pixel, how much luma bends across, down and
  along both diagonals, averaged over its 3x3 neighbourhood (1-D Laplacians, as VVC's adaptive
  loop filter measures them). A direction counts if it bends at least twice as much as the one
  at right angles to it (ALF's rule), else none; activity is the bend across and down together,
  cut at `levels`. `DIRECTIONS` x (len(levels) + 1) classes.
- **A filter** makes a value from the `taps` x `taps` luma around a pixel, plus an offset.
- **A fit** adds up the normal equations per (group, class) - a group is whatever gets filters
  of its own, a source or a frame - and solves them, optionally pulled toward a prior.
- **Bits**: a filter is sent as its difference from the one the client already has, in steps
  of its own size, and the step.
"""

import torch
import torch.nn.functional as F

from .models import LUMA_709

DIRECTIONS = 5   # none, across, down, and the two diagonals


def _laplacians():
    """Across, down and both diagonals: 2 x the pixel minus its two neighbours, as (4, 1, 3, 3)."""
    kernels = torch.zeros(4, 1, 3, 3)
    for k, ((y0, x0), (y1, x1)) in enumerate((((1, 0), (1, 2)), ((0, 1), (2, 1)), ((0, 0), (2, 2)), ((0, 2), (2, 0)))):
        kernels[k, 0, 1, 1] = 2
        kernels[k, 0, y0, x0] = kernels[k, 0, y1, x1] = -1
    return kernels


LAPLACIANS = _laplacians()


def luma(rgb):
    """(N, 3, h, w) RGB -> BT.709 luma (N, 1, h, w)."""
    weights = torch.tensor(LUMA_709, device=rgb.device, dtype=rgb.dtype)
    return (rgb * weights[None, :, None, None]).sum(dim=1, keepdim=True)


def bends(y):
    """How much luma pictures (N, 1, h, w) bend across, down and along both diagonals at every
    pixel, averaged over its 3x3 neighbourhood, the edge repeated outward: (N, 4, h, w)."""
    padded = F.pad(y, (2, 2, 2, 2), mode="replicate")
    return F.avg_pool2d(F.conv2d(padded, LAPLACIANS.to(y)).abs(), 3, stride=1)


def activity_levels(y, count, sample=1_000_000):
    """`count` - 1 thresholds that cut the activity of luma pictures `y` into `count` equal shares."""
    bend = bends(y)
    activity = (bend[:, 0] + bend[:, 1]).flatten()
    if len(activity) > sample:   # torch.quantile takes a limited number of values
        activity = activity[torch.randperm(len(activity), device=activity.device)[:sample]]
    return torch.quantile(activity, torch.linspace(0, 1, count + 1, device=y.device)[1:-1])


def count(levels):
    """How many classes `classes` sorts into with these activity levels."""
    return DIRECTIONS * (len(levels) + 1)


def classes(y, levels):
    """The class of every pixel of LR luma pictures (N, 1, h, w): (N, h, w), 0 .. count(levels) - 1."""
    across, down, diagonal, other = bends(y).unbind(dim=1)
    straight = torch.maximum(across, down) / (torch.minimum(across, down) + 1e-4)
    slanted = torch.maximum(diagonal, other) / (torch.minimum(diagonal, other) + 1e-4)
    direction = torch.zeros_like(across, dtype=torch.long)   # 0: none dominates
    is_straight = (straight > 2) & (straight >= slanted)
    is_slanted = (slanted > 2) & (slanted > straight)
    direction[is_straight] = torch.where(across > down, 1, 2)[is_straight]
    direction[is_slanted] = torch.where(diagonal > other, 3, 4)[is_slanted]
    return direction * (len(levels) + 1) + torch.bucketize(across + down, levels)


def upsampled(pixel_class, scale):
    """Every HR pixel takes its LR pixel's class: (N, h, w) -> (N, h * scale, w * scale)."""
    return pixel_class.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)


def windows(y, taps=5, pad=True):
    """The `taps` x `taps` luma around every pixel of `y` (N, 1, h, w), and a 1 for the offset:
    (N, taps * taps + 1, P). With `pad` every pixel, the edge repeated outward; without, only
    the pixels the window fits around."""
    if pad:
        y = F.pad(y, (taps // 2,) * 4, mode="replicate")
    window = F.unfold(y, taps)
    return torch.cat([window, torch.ones_like(window[:, :1])], dim=1)


class Fit:
    """The normal equations of a least-squares filter per (group, class): the sums of x x^T and
    of x y^T over every pixel added, in float64."""

    def __init__(self, groups, classes, outputs, taps=5, device=None):
        size = taps * taps + 1
        self.classes = classes
        self.xx = torch.zeros(groups, classes, size, size, dtype=torch.float64, device=device)
        self.xy = torch.zeros(groups, classes, size, outputs, dtype=torch.float64, device=device)

    def add(self, window, target, pixel_class, group):
        """window (N, taps² + 1, P), target (N, outputs, P), pixel_class (N, P), group (N,)."""
        size = self.xx.shape[-1]
        x = window.double().permute(0, 2, 1).reshape(-1, size)
        y = target.double().permute(0, 2, 1).reshape(-1, target.shape[1])
        key = (group[:, None] * self.classes + pixel_class).flatten()
        xx, xy = self.xx.view(-1, size, size), self.xy.view(-1, size, self.xy.shape[-1])
        for part in range(0, len(key), 1 << 18):   # in parts: every pixel's x x^T is size² numbers
            rows = slice(part, part + (1 << 18))
            xx.index_add_(0, key[rows], x[rows, :, None] * x[rows, None, :])
            xy.index_add_(0, key[rows], x[rows, :, None] * y[rows, None, :])

    def pixels(self):
        """How many pixels each (group, class) saw: the offset's 1 times itself, added up."""
        return self.xx[..., -1, -1]

    def solve(self, one_class=False, prior=None, ridge=0.0):
        """The filters, (groups, classes, taps² + 1, outputs). `one_class` fits one filter for
        every class; a `prior` (broadcast to that shape) pulls each filter toward it with the
        weight of `ridge` pixels."""
        xx, xy = self.xx, self.xy
        if one_class:
            xx, xy = xx.sum(dim=1, keepdim=True).expand_as(xx), xy.sum(dim=1, keepdim=True).expand_as(xy)
        right = xy if prior is None else xy + ridge * prior
        eye = torch.eye(xx.shape[-1], dtype=xx.dtype, device=xx.device)
        return torch.linalg.solve(xx + (ridge + 1e-6) * eye, right)


def quantise(filters, base, bits=10):
    """Filters as the client receives them: each one's difference from `base`, in `bits`-bit
    steps of its own size."""
    difference = filters - base
    step = difference.abs().amax(dim=-2, keepdim=True).clamp_min(1e-12) / (2 ** (bits - 1) - 1)
    return base + torch.round(difference / step) * step


def update_bits(sent, outputs, taps=5, coefficient_bits=10, step_bits=16):
    """Bits of one update per group, from which classes are sent (groups, classes): a flag per
    class, and for every class sent, `outputs` filters of taps² + 1 coefficients and a step."""
    per_filter = (taps * taps + 1) * coefficient_bits + step_bits
    return sent.sum(dim=-1) * outputs * per_filter + sent.shape[-1]


def through(filters, window, pixel_class, group):
    """Every pixel through its group's filter for its class: (N, outputs, P)."""
    return torch.einsum("njp,npjk->nkp", window, filters[group[:, None], pixel_class])


def post_filter(picture, filters, pixel_class, group, taps=5):
    """`picture` (N, 3, H, W) with its luma changed by one-output filters, per pixel class
    (N, H, W) and group (N,) - the same change on R, G and B, so colour stays as it was."""
    y = luma(picture)
    change = through(filters, windows(y, taps), pixel_class.flatten(1), group) - y.flatten(2)
    return picture + change.view_as(y)
