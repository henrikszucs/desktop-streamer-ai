"""Pictures of motion, shared by the notebooks: what a method moved, where it went wrong, and
how a motion goes on over time.

Offsets are as `models.warp` takes them - an (x, y) per pixel saying where that pixel of the
output is *read from* - so the arrows drawn are their negative: where the content goes.
Changes and errors are in luma levels (of 255), on one colour scale everywhere, so two
pictures can be compared by eye.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from .metrics import RGB_TO_YUV

LEVELS = 48            # a change or an error this large, in luma levels, is full colour
TRUE = "lime"          # the arrows of the motion that really happened
FOUND = "yellow"       # the arrows of the motion a method found


def _numpy(picture):
    """(H, W, C) or (H, W) numpy from a tensor (1, C, H, W), (C, H, W) or (H, W)."""
    if isinstance(picture, torch.Tensor):
        picture = picture.detach().float().cpu()
        if picture.dim() == 4:
            picture = picture[0]
        picture = picture.permute(1, 2, 0).numpy() if picture.dim() == 3 else picture.numpy()
    return np.asarray(picture)


def luma(picture):
    """(H, W) luma in levels of 255 of an RGB picture in [0, 1]."""
    return _numpy(picture).clip(0, 1) @ RGB_TO_YUV[0].numpy() * 255


def _frame(ax, title, box):
    if box is not None:
        y, x, h, w = box
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), w, h, fill=False, color="red", lw=0.8))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def picture(ax, image, title="", box=None):
    """An RGB picture in [0, 1]. `box` (y, x, h, w) outlines the part that is scored."""
    ax.imshow(_numpy(image).clip(0, 1), interpolation="nearest")
    _frame(ax, title, box)


def change(ax, before, after, title="", box=None, levels=LEVELS):
    """`after` - `before` in luma: red where it got brighter, blue where darker, white where
    nothing changed. Before = N and after = N+1 is what a method has to draw."""
    ax.imshow(luma(after) - luma(before), cmap="RdBu_r", vmin=-levels, vmax=levels, interpolation="nearest")
    _frame(ax, title, box)


def error(ax, prediction, target, title="", box=None, levels=LEVELS):
    """|prediction - target| in luma: black where it is right, bright where wrong."""
    ax.imshow(np.abs(luma(prediction) - luma(target)), cmap="magma", vmin=0, vmax=levels, interpolation="nearest")
    _frame(ax, title, box)


def arrows(ax, offsets, every=8, gain=2.0, colour=FOUND):
    """The motion of `offsets` ((1, 2, H, W) or (2, H, W), read-from) as an arrow every `every`
    pixels, from where the content is to where it goes, `gain` times its length. A dot is a
    pixel that stays."""
    field = _numpy(offsets)                       # (H, W, 2)
    height, width = field.shape[:2]
    ys, xs = np.mgrid[every // 2:height:every, every // 2:width:every]
    u, v = -gain * field[ys, xs, 0], -gain * field[ys, xs, 1]
    moving = np.hypot(u, v) > 0.1 * gain
    ax.quiver(xs[moving], ys[moving], u[moving], v[moving], color=colour, angles="xy", scale_units="xy",
              scale=1, width=0.01, headwidth=4, headlength=4, headaxislength=3.5)
    ax.scatter(xs[~moving], ys[~moving], s=1.5, color=colour)


def motion(ax, image, offsets, title="", box=None, every=8, gain=2.0, colour=FOUND):
    """A picture with a motion drawn over it."""
    picture(ax, image, title, box)
    arrows(ax, offsets, every, gain, colour)


def offset_map(ax, values, radius, scale, title="", cmap="viridis", mark=None):
    """One value per offset of a search - `values` (offsets,), in the order `TileSearch` and
    `Shift` list them, (dy, dx) row by row - as a map of the motion each stands for in
    full-resolution pixels between the received frames. `mark` (dx, dy) is drawn as a cross."""
    size = 2 * radius + 1
    grid = _numpy(torch.as_tensor(values).reshape(size, size))
    reach = scale * radius + scale / 2
    image = ax.imshow(grid, cmap=cmap, extent=(-reach, reach, reach, -reach), interpolation="nearest")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if mark is not None:
        ax.plot(*mark, "x", color="red", markersize=8, mew=2)
    ax.set_xlabel("x px", fontsize=7)
    ax.set_ylabel("y px", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8)


def track(ax, positions, every, fraction, title=""):
    """How a motion goes on: `positions` (frames,) of one moving thing at every display frame.
    Every `every`th is received (a dot); every frame after a received one is made, and two
    guesses at it are drawn - repeat (the received frame again) and the motion carried on
    `fraction` of the way from the two received before it (the search's rule) - each with a
    line to where the content really was."""
    positions = np.asarray(positions, dtype=float)
    frames = np.arange(len(positions))
    ax.plot(frames, positions, color="0.6", lw=1, label="where it is (every display frame)")
    received = frames[::every]
    ax.plot(received, positions[received], "o", color="black", ms=3.5, label="received")
    made = [n for n in received if n >= every and n + 1 < len(positions)]
    for n in made:
        carried = positions[n] + fraction * (positions[n] - positions[n - every])
        ax.plot([n + 1, n + 1], [carried, positions[n + 1]], color="red", lw=0.8)
        ax.plot(n + 1, carried, "x", color="red", ms=5)
        ax.plot(n + 1, positions[n], "_", color="tab:blue", ms=7, mew=1.5)
    ax.plot([], [], "x", color="red", label="motion carried on (the search)")
    ax.plot([], [], "_", color="tab:blue", ms=7, mew=1.5, label="repeat")
    ax.set_xlabel("display frame", fontsize=8)
    ax.set_ylabel("position (stream px)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)


@torch.no_grad()
def moving_picks(inputs, targets, masks, kinds, halo, least=2.0, seed=0, batch_size=256):
    """[(kind, index)]: one changed sample of each kind whose content really moves, chosen at
    random among those - a tenth of its pixels at least `least` px from N to N+1, by a block
    search between N and the true N+1 (which sees the answer: for choosing what to show only).
    A kind with none falls back to any changed sample."""
    from . import models, training

    reference = models.Shift(radius=halo, block=16, fraction=1.0).to(training.DEVICE)
    rng = np.random.default_rng(seed)
    picks = []
    for kind in kinds:
        changed = np.flatnonzero(masks[f"{kind}, changed"])
        if not len(changed):
            continue
        reach = []
        for first in range(0, len(changed), batch_size):
            index = changed[first:first + batch_size]
            current = training.to_tensor(inputs[index, -1])
            field = reference.field(current, training.to_tensor(targets[index]))
            reach.append(torch.quantile(field.norm(dim=1).flatten(1), 0.9, dim=1).cpu().numpy())
        moving = changed[np.concatenate(reach) >= least]
        picks.append((kind, int(rng.choice(moving if len(moving) else changed))))
    return picks


def legend(ax, names, cmap="tab10", **kwargs):
    """A legend of colour patches for categories drawn with `cmap` indices 0, 1, ..."""
    colours = plt.get_cmap(cmap)
    ax.legend(handles=[Patch(color=colours(i), label=name) for i, name in enumerate(names)], **kwargs)
