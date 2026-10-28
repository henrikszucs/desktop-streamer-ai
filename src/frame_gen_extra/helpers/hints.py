"""The host's hint: how each client tile moves from frame N, which the client has, to N+1,
which only the host captured (notebook 5).

The host captures at the display rate and sends every second frame. Before it drops the one
between, it can measure its motion and send that instead - two numbers a tile - so the client
moves N by what really happened rather than by a guess from N-2 and N.

The measurement works on luma of the clean frames, before the stream's shrink and codec,
at the source's resolution or any power of two below it, as a `pyramid` (level 0 the source,
each next one half the size). `measure` searches every whole-pixel offset within `radius` at
one level, then refines the best by ±1 pixel at each finer level down to `stop`. Started at
the stream's own level with a wide radius it is an exhaustive search; started three levels
up with a radius of 4 it reaches as far (±32 source pixels) for a few hundredth of the work,
which is what a host can afford 30 times a second on every tile.

A motion is (x, y), where the content of `before` is in `after`, in pixels of level 0. The
stream is `scale` times smaller, so a motion at level 0 is a 1/`scale` stream pixel step.
"""

import torch
import torch.nn.functional as F

PENALTY = 1e-4      # the cost of an offset grows this much per pixel of it: a flat tile stays put


def pyramid(luma, levels):
    """[level 0, 1, ...] of (N, 1, H, W) luma in [0, 1], each half the size of the one before."""
    found = [luma]
    for _ in range(levels):
        found.append(F.avg_pool2d(found[-1], 2))
    return found


def _costs(before, after, offsets, margin, sample):
    """(N, K): mean |after(x) - before(x - d)| over the picture `margin` pixels in from every
    edge, for each of the K whole-pixel (x, y) offsets d (N, K, 2) - every `sample`th pixel."""
    n, k = offsets.shape[:2]
    height, width = after.shape[-2:]
    inner_y = torch.arange(margin, height - margin, sample, device=after.device)
    inner_x = torch.arange(margin, width - margin, sample, device=after.device)
    target = after[:, 0][:, inner_y][:, :, inner_x]                                    # (N, h, w)
    ys = (inner_y.view(1, 1, -1, 1) - offsets[..., 1].view(n, k, 1, 1)).clamp(0, height - 1)
    xs = (inner_x.view(1, 1, 1, -1) - offsets[..., 0].view(n, k, 1, 1)).clamp(0, width - 1)
    flat = before[:, 0].flatten(1)                                                    # (N, H*W)
    moved = flat.gather(1, (ys * width + xs).flatten(1)).view(n, k, len(inner_y), len(inner_x))
    penalty = PENALTY * offsets.abs().sum(-1)
    return (moved - target[:, None]).abs().mean(dim=(2, 3)) + penalty


def measure(before, after, start, radius, stop=0, sample=1, chunk=81, subpixel=False, coarse=1):
    """(N, 2) float: each picture's motion (x, y) from `before` to `after`, pyramids as
    `pyramid` makes them, in level-0 pixels. Every offset within `radius` at level `start`,
    then ±1 at each finer level down to `stop`. `sample` scores every `sample`th pixel only, and
    `coarse` every `coarse`th at `start`.
    `subpixel` fits a parabola through the costs either side of the best offset at `stop`, on
    each axis, and rounds the motion to half a pixel - the precision a host measuring at the
    stream's own resolution can give, where one measuring at twice it has the pixels."""
    device = after[start].device
    span = torch.arange(-radius, radius + 1, device=device)
    grid = torch.stack(torch.meshgrid(span, span, indexing="xy"), -1).view(-1, 2)      # (K, 2) x, y
    n = after[start].shape[0]
    costs = torch.cat([_costs(before[start], after[start], grid[i:i + chunk].expand(n, -1, -1), radius, coarse)
                       for i in range(0, len(grid), chunk)], dim=1)
    motion = grid[costs.argmin(1)]                                                     # (N, 2)
    near = torch.stack(torch.meshgrid(span.new_tensor([-1, 0, 1]), span.new_tensor([-1, 0, 1]),
                                      indexing="xy"), -1).view(-1, 2)
    for level in range(start - 1, stop - 1, -1):
        motion = 2 * motion
        tried = motion[:, None] + near[None]                                           # (N, 9, 2)
        reach = (radius + 1) * 2 ** (start - level)
        costs = _costs(before[level], after[level], tried, reach, sample)
        motion = tried[torch.arange(n, device=device), costs.argmin(1)]
    motion = motion.float()
    if subpixel:
        # the costs one pixel either side of the best on each axis, and the vertex of the
        # parabola through the three - a shift of at most half a pixel, rounded to a half
        reach = (radius + 1) * 2 ** (start - stop)
        tried = motion.long()[:, None] + near[None]
        costs = _costs(before[stop], after[stop], tried, reach, sample).view(n, 3, 3)    # [y, x]
        for axis, (low, mid, high) in enumerate(((costs[:, 1, 0], costs[:, 1, 1], costs[:, 1, 2]),
                                                 (costs[:, 0, 1], costs[:, 1, 1], costs[:, 2, 1]))):
            curve = low - 2 * mid + high
            shift = torch.where(curve > 0, (low - high) / (2 * curve).clamp_min(1e-12), torch.zeros_like(curve))
            motion[:, axis] += (shift.clamp(-0.5, 0.5) * 2).round() / 2
    return motion * 2 ** stop
