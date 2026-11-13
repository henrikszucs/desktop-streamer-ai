"""Turn a still image into a short moving clip, like a screen in a stream.

A screen moves: pages scroll, windows are dragged, videos pan and zoom. Most frames a
client decodes predict that motion from the previous frame, and their artifacts differ
from a single still frame's. So an image is shown through a **window** (1280x720 by
default) that moves over it, and the clip is what `degrade.degrade` encodes.

Three kinds of motion, picked at random per image:

- **slide**: the window moves a whole number of pixels per frame (mostly a scroll);
- **zoom**: the window grows or shrinks a little (up to `MAX_ZOOM`) and drifts. The image
  is only ever shrunk, never enlarged, so the HR target stays sharp. Rare: a desktop
  seldom zooms, and H.264 cannot predict a zoom, so it is the most damaged motion at
  any bitrate;
- **still**: the window does not move.
"""

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

KINDS = ("slide", "zoom", "still")
WEIGHTS = (0.6, 0.1, 0.3)

WINDOW = (1280, 720)
MAX_ZOOM = 1.15
MAX_SPEED = 16  # pixels per frame - a fast scroll at 30 fps
UNIT = 4        # window sides are multiples of this, so a x2 LR frame has even sides


@dataclass(frozen=True)
class Motion:
    """The window's (x, y, width) on the first and last frame; frames between move linearly."""

    kind: str
    window: tuple[int, int]
    frames: int
    start: tuple[float, float, float]
    end: tuple[float, float, float]

    def as_dict(self):
        return asdict(self)

    def boxes(self):
        """(x, y, width, height) of the image region shown on each frame."""
        aspect = self.window[1] / self.window[0]
        for t in np.linspace(0.0, 1.0, self.frames):
            x, y, width = (a + (b - a) * t for a, b in zip(self.start, self.end))
            yield x, y, width, width * aspect


def window_for(size, window=WINDOW):
    """The window size for an image of `size` (width, height) - smaller if the image is."""
    return tuple(min(w, s) // UNIT * UNIT for w, s in zip(window, size))


def _slide(rng, size, win, frames):
    room = (size[0] - win[0], size[1] - win[1])
    steps = max(frames - 1, 1)
    axes = [(0, 1), (1, 0), (1, 1)][int(rng.choice(3, p=(0.6, 0.2, 0.2)))]  # mostly vertical
    top_speed = [min(MAX_SPEED, room[i] // steps) if axes[i] else 0 for i in range(2)]
    if not any(top_speed):
        return None
    speed = [int(rng.integers(1, s + 1)) * int(rng.choice((-1, 1))) if s else 0 for s in top_speed]
    travel = [s * steps for s in speed]
    x0, y0 = (int(rng.integers(max(0, -d), room[i] - max(0, d) + 1)) for i, d in enumerate(travel))
    return Motion("slide", win, frames, (x0, y0, win[0]), (x0 + travel[0], y0 + travel[1], win[0]))


def _zoom(rng, size, win, frames):
    width, height = size
    reach = min(MAX_ZOOM, width / win[0], height / win[1])
    if reach < 1.02:
        return None
    zooms = [1.0, float(rng.uniform(1.0 + (reach - 1.0) / 3, reach))]
    rng.shuffle(zooms)  # zoom in or out
    drift = MAX_SPEED * max(frames - 1, 1) / 2
    room = [(width - win[0] * z, height - win[1] * z, win[0] * z) for z in zooms]
    x0 = float(rng.uniform(0, room[0][0]))
    y0 = float(rng.uniform(0, room[0][1]))
    x1 = float(np.clip(x0 + rng.uniform(-drift, drift), 0, room[1][0]))
    y1 = float(np.clip(y0 + rng.uniform(-drift, drift), 0, room[1][1]))
    return Motion("zoom", win, frames, (x0, y0, room[0][2]), (x1, y1, room[1][2]))


def sample(rng, size, frames=30, window=WINDOW):
    """A random motion of the window over an image of `size` (width, height)."""
    win = window_for(size, window)
    kind = str(rng.choice(KINDS, p=WEIGHTS))
    motion = None
    if kind == "slide":
        motion = _slide(rng, size, win, frames)
    elif kind == "zoom":
        motion = _zoom(rng, size, win, frames)
    if motion is not None:
        return motion
    # still - also used when the image has no room to slide or zoom
    x0, y0 = (int(rng.integers(0, r + 1)) for r in (size[0] - win[0], size[1] - win[1]))
    return Motion("still", win, frames, (x0, y0, win[0]), (x0, y0, win[0]))


def render(image, motion):
    """The clip `motion` shows of `image` (H, W, 3), as (T, h, w, 3) uint8 frames."""
    image = np.asarray(image, dtype=np.uint8)
    w, h = motion.window
    if motion.kind != "zoom":
        # whole pixel steps: every frame is a plain crop of the image
        return np.stack([image[round(y):round(y) + h, round(x):round(x) + w]
                         for x, y, _, _ in motion.boxes()])
    source = Image.fromarray(image)
    return np.stack([np.asarray(source.resize((w, h), Image.LANCZOS, box=(x, y, x + bw, y + bh)))
                     for x, y, bw, bh in motion.boxes()])
