"""Turn a still picture into a short clip of a desktop being used.

`../upscale/helpers/motion.py` moves a window over an image at one constant speed, which is
all an upscaler needs: moving frames, so the codec damages them the way it damages a stream.
An extrapolator learns the motion itself, and a constant speed is the one motion that a
straight line through the last two frames already gets right. A desktop's motion starts and
stops, eases in and out, moves a small part of the screen while the rest holds still, and
now and then does something nothing before it predicted.

So a clip is a **background** and up to four things on top of it:

- the background is a window (1280x720 by default) over the image, `still` or scrolling:
  by `wheel` notches (a jump eased over a few frames, as a browser's smooth scroll does), a
  touchpad `flick` (fast, then decaying) or a `steady` drag of the scrollbar. A page scrolls
  vertically, a photo pans any way, a desktop does not scroll;
- `region`: a rectangle inside the window scrolls by wheel notches while the rest stays,
  like a document in one of several windows;
- `drag`: a window - a piece of the image from elsewhere - is carried along by the cursor;
- `popup`: a menu - another piece of the image - appears at the cursor, and sometimes goes
  again. Nothing in the frames before it says it is coming: it is the case an extrapolator
  can only get wrong, and the data has to contain it so a score can show how wrong;
- the **cursor**, an arrow pointer: it rests, moves to a new place along a slightly curved,
  eased path, and rests again. It drives the drag and the menu, and `render` can leave it
  out of the frames: desktop-streamer's capture does, and its client draws the pointer itself.

What each picture gets is drawn from `PLANS` by its kind: a `page` (a tall document,
`datasets/screens.py`), a `desktop` (a whole screen) or a `photo`. Positions are whole pixels
at the window's resolution, as a desktop draws them; the stream's downscale makes them
fractional.

Every speed and duration here is per frame at `BASE_FPS` (30), the rate it was tuned at. A
clip at another rate (`fps=60` for 2x frame generation) is drawn on that time line and then
read at its own frame times - the same motion in pixels per second, so half as far a frame at
60 fps, and a menu that opens at the same moment.
"""

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

WINDOW = (1280, 720)
UNIT = 4                     # window sides are multiples of this, so a x2 stream frame has even sides

BACKGROUNDS = ("still", "wheel", "flick", "steady")
EVENTS = ("region", "drag", "popup")

# per kind of picture: the chance of each background, and of each event and the cursor
PLANS = {
    "page": {"background": {"still": 0.2, "wheel": 0.4, "flick": 0.25, "steady": 0.15},
             "region": 0.0, "drag": 0.1, "popup": 0.2, "cursor": 0.8},
    "desktop": {"background": {"still": 1.0},
                "region": 0.4, "drag": 0.35, "popup": 0.3, "cursor": 0.9},
    "photo": {"background": {"still": 0.3, "flick": 0.3, "steady": 0.4},
              "region": 0.0, "drag": 0.1, "popup": 0.1, "cursor": 0.5},
}

NOTCH = (40, 120)            # pixels one wheel notch scrolls
MAX_SPEED = 16               # pixels per frame of a steady scroll - a fast one at 30 fps
FLICK_SPEED = (8, 48)        # pixels per frame a flick starts at
CURSOR_SCALES = (1.0, 1.25, 1.5, 2.0)
BASE_FPS = 30                # every speed and duration above is per frame at this rate
BORDER = (80, 80, 80)        # the 1 px frame drawn around a dragged window or a menu


@dataclass
class Scene:
    """Everything a clip shows, frame by frame, so it can be drawn again and scored by kind.

    `origin` is the window's top left on the image at frame 0 and `scroll` its offset from
    there on every frame. The events are None when absent; positions are window pixels.
    """

    kind: str
    window: tuple[int, int]
    frames: int
    origin: tuple[int, int]
    background: str
    scroll: list = field(default_factory=list)       # [(dx, dy)] per frame
    region: dict | None = None                       # {"box": (x, y, w, h), "offset": [dy per frame]}
    drag: dict | None = None                         # {"source": (x, y, w, h), "at": [(x, y) per frame]}
    popup: dict | None = None                        # {"source": (x, y, w, h), "at": (x, y), "frames": (first, end)}
    cursor: list | None = None                       # [(x, y) per frame] of the pointer's tip
    cursor_scale: float = 1.0

    def as_dict(self):
        return asdict(self)

    def events(self):
        """The names of what moves in this clip, background first."""
        names = [] if self.background == "still" else [self.background]
        names += [name for name in EVENTS if getattr(self, name) is not None]
        return names + (["cursor"] if self.cursor is not None else [])


def window_for(size, window=WINDOW):
    """The window size for an image of `size` (width, height) - smaller if the image is."""
    return tuple(min(w, s) // UNIT * UNIT for w, s in zip(window, size))


def ease(t):
    """Minimum-jerk easing from 0 to 1 - how a hand, and a smooth scroll, starts and stops."""
    t = np.clip(t, 0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t * t)


# ---------------------------------------------------------------- how far things travel

def _wheel(rng, frames):
    """Offsets (one per frame) of a smooth-scrolled wheel: one to three notches at a time,
    each eased over a few frames, with pauses between them."""
    offset = np.zeros(frames)
    notch = int(rng.integers(*NOTCH))
    t = int(rng.integers(0, 8))
    while t < frames:
        glide = int(rng.integers(3, 7))
        offset += notch * int(rng.integers(1, 4)) * ease((np.arange(frames) - t) / glide)
        t += glide + int(rng.integers(2, 12))
    return offset


def _flick(rng, frames):
    """Offsets of a touchpad flick: released at full speed, then decaying exponentially."""
    start = int(rng.integers(0, frames // 2))
    speed, decay = rng.uniform(*FLICK_SPEED), rng.uniform(4, 15)
    elapsed = np.clip(np.arange(frames) - start, 0, None)
    return speed * decay * (1 - np.exp(-elapsed / decay))


def _steady(rng, frames):
    """Offsets of a constant speed that may start late and stop early."""
    start = int(rng.integers(0, frames // 3))
    stop = int(rng.integers(start + frames // 3, frames + 1))
    return int(rng.integers(1, MAX_SPEED + 1)) * np.clip(np.arange(frames) - start, 0, stop - start)


PROFILES = {"wheel": _wheel, "flick": _flick, "steady": _steady}


def _fit(offset, room):
    """Scale a travel down until it reaches no further than `room` pixels, and round it to
    whole pixels."""
    reach = np.abs(offset).max(initial=0)
    if reach > room:
        offset = offset * (room / reach)
    return np.round(offset).astype(int)


# ---------------------------------------------------------------- the pieces of a scene

def _background(rng, kind, plan, size, win, frames):
    """The window's origin on the image and its (dx, dy) on every frame."""
    names, weights = zip(*plan["background"].items())
    name = str(rng.choice(names, p=np.array(weights) / sum(weights)))
    room = (size[0] - win[0], size[1] - win[1])
    if name == "still" or not any(room):
        origin = tuple(int(rng.integers(0, r + 1)) for r in room)
        return origin, "still", [(0, 0)] * frames

    travel = PROFILES[name](rng, frames) * rng.choice((-1, 1))
    if kind == "page":
        direction = (0.0, 1.0)
    else:   # a photo pans: mostly along one axis, sometimes both
        angle = float(rng.choice((0, np.pi / 2, rng.uniform(0, 2 * np.pi)), p=(0.3, 0.4, 0.3)))
        direction = (np.cos(angle), np.sin(angle))
    # the travel shortened to what the image has room for, then a start that keeps all of it on the image
    moves, origin = [], []
    for axis in range(2):
        along = _fit(travel * direction[axis], room[axis])
        lowest, highest = int(along.min(initial=0)), int(along.max(initial=0))
        moves.append(along)
        origin.append(int(rng.integers(-lowest, room[axis] - highest + 1)))
    if not np.any(moves[0]) and not np.any(moves[1]):
        return tuple(origin), "still", [(0, 0)] * frames
    return tuple(origin), name, [(int(x), int(y)) for x, y in zip(*moves)]


def _cursor(rng, win, frames):
    """The pointer's tip on every frame: rest, move along a bent eased path, rest again."""
    width, height = win
    position = np.array([rng.uniform(0, width), rng.uniform(0, height)])
    path = []
    while len(path) < frames:
        path += [position] * int(rng.integers(0, 10))
        target = np.array([rng.uniform(0, width), rng.uniform(0, height)])
        span = target - position
        bend = (position + target) / 2 + np.array([-span[1], span[0]]) * rng.uniform(-0.25, 0.25)
        steps = int(rng.integers(4, 16))
        for s in ease(np.arange(1, steps + 1) / steps):
            # a quadratic Bezier through the bend point
            path.append((1 - s) ** 2 * position + 2 * (1 - s) * s * bend + s ** 2 * target)
        position = target
    points = np.round(np.array(path[:frames])).astype(int)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return [(int(x), int(y)) for x, y in points]


def _piece(rng, size, width, height):
    """A (x, y, w, h) rectangle of the image of about `width` x `height` - a window's content."""
    w = int(min(rng.integers(*width), size[0]))
    h = int(min(rng.integers(*height), size[1]))
    return int(rng.integers(0, size[0] - w + 1)), int(rng.integers(0, size[1] - h + 1)), w, h


def _region(rng, size, win, origin, frames):
    """A rectangle of the window whose content scrolls by wheel notches, room permitting."""
    w = int(rng.integers(min(200, win[0] // 2), max(min(800, win[0]), 201)))
    h = int(rng.integers(min(150, win[1] // 2), max(min(500, win[1]), 151)))
    x, y = int(rng.integers(0, win[0] - w + 1)), int(rng.integers(0, win[1] - h + 1))
    top = origin[1] + y                                # the region's first image row at rest
    sign = int(rng.choice((-1, 1)))
    offset = _fit(_wheel(rng, frames) * sign, top if sign < 0 else size[1] - top - h)
    if not np.any(offset):
        return None
    return {"box": (x, y, w, h), "offset": [int(v) for v in offset]}


def _retime(values, times):
    """Per-frame positions read at fractional frame `times`: linearly between frames, rounded
    to whole pixels. `values` is a list of numbers or of (x, y) pairs."""
    array = np.asarray(values, dtype=np.float64)
    index = np.arange(len(array))
    if array.ndim == 1:
        return [int(v) for v in np.round(np.interp(times, index, array))]
    return [tuple(int(v) for v in point)
            for point in np.round(np.stack([np.interp(times, index, array[:, k]) for k in range(array.shape[1])], 1))]


def sample(rng, size, kind, frames=30, window=WINDOW, cursor_scale=None, fps=BASE_FPS):
    """A random scene over an image of `size` (width, height) whose kind is "page", "desktop"
    or "photo", `frames` long at `fps`."""
    if fps == BASE_FPS:
        return _sample(rng, size, kind, frames, window, cursor_scale)
    # drawn at the base rate over the same span of time, then read at this rate's frame times
    rate = fps / BASE_FPS
    base = _sample(rng, size, kind, int(np.ceil((frames - 1) / rate)) + 1, window, cursor_scale)
    times = np.arange(frames) / rate
    scene = Scene(kind=base.kind, window=base.window, frames=frames, origin=base.origin,
                  background=base.background, scroll=_retime(base.scroll, times),
                  cursor_scale=base.cursor_scale)
    if base.region:
        scene.region = {"box": base.region["box"], "offset": _retime(base.region["offset"], times)}
    if base.cursor is not None:
        scene.cursor = _retime(base.cursor, times)
    if base.drag:
        scene.drag = {"source": base.drag["source"], "at": _retime(base.drag["at"], times)}
    if base.popup:
        first, end = base.popup["frames"]
        scene.popup = {**base.popup, "frames": (int(round(first * rate)), min(frames, int(round(end * rate))))}
    return scene


def _sample(rng, size, kind, frames, window, cursor_scale):
    plan = PLANS[kind]
    win = window_for(size, window)
    origin, background, scroll = _background(rng, kind, plan, size, win, frames)
    scene = Scene(kind=kind, window=win, frames=frames, origin=origin, background=background,
                  scroll=scroll, cursor_scale=float(cursor_scale or rng.choice(CURSOR_SCALES)))

    if rng.random() < plan["region"]:
        scene.region = _region(rng, size, win, origin, frames)
    dragging = rng.random() < plan["drag"]
    popping = rng.random() < plan["popup"]
    # a drag or a menu needs a hand on the mouse, so either brings the cursor
    if dragging or popping or rng.random() < plan["cursor"]:
        scene.cursor = _cursor(rng, win, frames)
    if dragging:
        source = _piece(rng, size, (200, 600), (150, 450))
        grip = (int(rng.integers(10, max(11, source[2] - 10))), int(rng.integers(4, 16)))   # on its title bar
        scene.drag = {"source": source, "at": [(x - grip[0], y - grip[1]) for x, y in scene.cursor]}
    if popping:
        first = int(rng.integers(2, frames - 1))
        end = int(rng.integers(first + 4, frames + 1)) if rng.random() < 0.3 and first + 4 <= frames else frames
        x, y = scene.cursor[first]
        scene.popup = {"source": _piece(rng, size, (120, 300), (80, 360)), "at": (x + 2, y + 2),
                       "frames": (first, end)}
    return scene


# ---------------------------------------------------------------- drawing it

@lru_cache(maxsize=None)
def pointer(scale):
    """The arrow pointer as (h, w, 3) colours and an (h, w, 1) alpha, its tip at (0, 0)."""
    outline = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
    points = [(x * scale, y * scale) for x, y in outline]
    size = (int(13 * scale) + 2, int(21 * scale) + 2)
    sprite = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(sprite).polygon(points, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255),
                                   width=max(1, round(scale)))
    pixels = np.asarray(sprite).astype(np.float32)
    return pixels[..., :3], pixels[..., 3:] / 255.0


def _paste(frame, piece, x, y, alpha=None):
    """Draw `piece` onto `frame` with its top left at (x, y), cut to the frame's edges."""
    height, width = frame.shape[:2]
    top, left = max(0, -y), max(0, -x)
    bottom, right = min(piece.shape[0], height - y), min(piece.shape[1], width - x)
    if bottom <= top or right <= left:
        return
    target = frame[y + top:y + bottom, x + left:x + right]
    part = piece[top:bottom, left:right]
    if alpha is None:
        target[:] = part
    else:
        a = alpha[top:bottom, left:right]
        target[:] = np.round(part * a + target * (1 - a)).astype(np.uint8)


def _framed(image, source):
    """A piece of the image with a 1 px border, as a window or a menu is drawn."""
    x, y, w, h = source
    piece = image[y:y + h, x:x + w].copy()
    piece[[0, -1]] = BORDER
    piece[:, [0, -1]] = BORDER
    return piece


def render(image, scene, cursor=True):
    """The clip `scene` shows of `image` (H, W, 3), as (T, h, w, 3) uint8 frames. `cursor=False`
    leaves the pointer out - desktop-streamer's capture does, and its client draws the pointer
    itself - while it still drags windows and opens menus where it is."""
    image = np.asarray(image, dtype=np.uint8)
    w, h = scene.window
    x0, y0 = scene.origin
    dragged = _framed(image, scene.drag["source"]) if scene.drag else None
    menu = _framed(image, scene.popup["source"]) if scene.popup else None
    colours, alpha = pointer(scene.cursor_scale)

    frames = []
    for t in range(scene.frames):
        dx, dy = scene.scroll[t]
        frame = image[y0 + dy:y0 + dy + h, x0 + dx:x0 + dx + w].copy()
        if scene.region:
            bx, by, bw, bh = scene.region["box"]
            shift = scene.region["offset"][t]
            frame[by:by + bh, bx:bx + bw] = image[y0 + by + shift:y0 + by + shift + bh, x0 + bx:x0 + bx + bw]
        if dragged is not None:
            _paste(frame, dragged, *scene.drag["at"][t])
        if menu is not None and scene.popup["frames"][0] <= t < scene.popup["frames"][1]:
            _paste(frame, menu, *scene.popup["at"])
        if cursor and scene.cursor is not None:
            _paste(frame, colours, *scene.cursor[t], alpha=alpha)
        frames.append(frame)
    return np.stack(frames)
