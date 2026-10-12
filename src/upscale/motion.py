"""A still image turned into the second of screen it would be in a stream.

A photo held still is one picture the encoder sees once: a keyframe, then frames that only
sharpen it. A screen moves - a page scrolls, a window is dragged, a game or a video pans
and zooms - and most of what the client decodes is P-frames predicting that motion, with
artifacts a keyframe never has. So an image is shown through a **window** (the screen,
1280x720 by default) that moves over it, and the clip is what `degrade.degrade` encodes.

Three motions, drawn per image:

- **slide**: the window moves a whole number of pixels a frame, mostly up or down - a
  scroll - else sideways or both. Every frame is a plain crop of the image, so the target
  is never resampled;
- **zoom**: the window takes a region from 1x to `MAX_ZOOM`x its own size and drifts,
  resampled with Lanczos. The region is never smaller than the window, so the image is
  only ever shrunk and the target stays as sharp as the source;
- **still**: the window does not move - a static screen settling after its keyframe.

An image too small for the window gives the window its own size, and one with no room to
move is still. ffmpeg's `zoompan` would do the same job, but it puts its crop on whole
pixels, so a slow zoom shakes - motion the model would learn and the screen never has.
"""

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

KINDS = ("slide", "zoom", "still")
WEIGHTS = (0.4, 0.4, 0.2)

WINDOW = (1280, 720)
MAX_ZOOM = 1.15
MAX_SPEED = 16  # HR pixels a frame - a fast scroll at 30 fps
UNIT = 4        # window sides are multiples of this, so a x2 LR frame has even sides


@dataclass(frozen=True)
class Motion:
    """Where the window is on the first and the last frame, as (x, y, width) boxes over the
    image; the frames between move linearly. Recorded per sample, like the degradation."""

    kind: str
    window: tuple[int, int]
    frames: int
    start: tuple[float, float, float]
    end: tuple[float, float, float]

    def as_dict(self):
        return asdict(self)

    def boxes(self):
        """(x, y, width, height) of the region shown on each frame."""
        aspect = self.window[1] / self.window[0]
        for t in np.linspace(0.0, 1.0, self.frames):
            x, y, width = (a + (b - a) * t for a, b in zip(self.start, self.end))
            yield x, y, width, width * aspect


def window_for(size, window=WINDOW):
    """The window an image of `size` (width, height) is seen through."""
    return tuple(min(w, s) // UNIT * UNIT for w, s in zip(window, size))


def sample(rng, size, frames=30, window=WINDOW):
    """A random motion of the window over an image of `size` (width, height)."""
    width, height = size
    win = window_for(size, window)
    room = (width - win[0], height - win[1])
    kind = str(rng.choice(KINDS, p=WEIGHTS))
    steps = max(frames - 1, 1)

    if kind == "slide":
        # mostly a scroll; an axis with no room for a pixel a frame cannot move
        axes = [(0, 1), (1, 0), (1, 1)][int(rng.choice(3, p=(0.6, 0.2, 0.2)))]
        top = [min(MAX_SPEED, room[i] // steps) if axes[i] else 0 for i in range(2)]
        if not any(top):
            kind = "still"
        else:
            speed = [int(rng.integers(1, t + 1)) * int(rng.choice((-1, 1))) if t else 0 for t in top]
            travel = [s * steps for s in speed]
            x0, y0 = (int(rng.integers(max(0, -d), room[i] - max(0, d) + 1)) for i, d in enumerate(travel))
            return Motion(kind, win, frames, (x0, y0, win[0]), (x0 + travel[0], y0 + travel[1], win[0]))

    if kind == "zoom":
        reach = min(MAX_ZOOM, width / win[0], height / win[1])
        if reach < 1.02:
            kind = "still"
        else:
            zooms = [1.0, float(rng.uniform(1.0 + (reach - 1.0) / 3, reach))]
            rng.shuffle(zooms)  # in or out
            boxes = []
            drift = MAX_SPEED * steps / 2
            for zoom in zooms:
                w, h = win[0] * zoom, win[1] * zoom
                boxes.append([width - w, height - h, w])  # room left for this box
            x0 = float(rng.uniform(0, boxes[0][0]))
            y0 = float(rng.uniform(0, boxes[0][1]))
            x1 = float(np.clip(x0 + rng.uniform(-drift, drift), 0, boxes[1][0]))
            y1 = float(np.clip(y0 + rng.uniform(-drift, drift), 0, boxes[1][1]))
            return Motion(kind, win, frames, (x0, y0, boxes[0][2]), (x1, y1, boxes[1][2]))

    x0, y0 = (int(rng.integers(0, r + 1)) for r in room)
    return Motion("still", win, frames, (x0, y0, win[0]), (x0, y0, win[0]))


def render(image, motion):
    """The clip `motion` shows of `image` (H, W, 3) uint8, as (T, h, w, 3) uint8 frames."""
    image = np.asarray(image, dtype=np.uint8)
    w, h = motion.window
    if motion.kind != "zoom":
        # whole pixels, so the frames are the image itself, cut
        return np.stack([image[round(y):round(y) + h, round(x):round(x) + w]
                         for x, y, _, _ in motion.boxes()])
    source = Image.fromarray(image)
    return np.stack([np.asarray(source.resize((w, h), Image.LANCZOS, box=(x, y, x + bw, y + bh)))
                     for x, y, bw, bh in motion.boxes()])
