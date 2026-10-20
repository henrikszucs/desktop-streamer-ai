"""Tile geometry: how a frame is cut into tiles, and what each tile costs.

The browser never upscales a whole frame at once. It cuts the frame into tiles, runs the
model on each one, and puts the centres back together. Three sizes describe a tile, and
mixing them up is the usual bug:

- **step**: the low resolution pixels a tile contributes to the picture;
- **input**: the step plus the halo on both sides - what the model is fed;
- **kept**: the high resolution centre of the output that is actually used.

Shapes are (height, width) everywhere, except a frame, which is (width, height) like
"1920x1080".
"""

import math

# The tile a client should run, in output pixels: 640x360 (a 320x180 step). It divides
# every 16:9 resolution exactly - 720p is 2x2 tiles, 1080p 3x3, 1440p 4x4, 4K 6x6.
# Chosen in 05_upscale_geometry.ipynb.
TARGET_KEPT_PX = 640 * 360


def as_size(size):
    """A size as (height, width). A single number means a square."""
    if isinstance(size, (tuple, list)):
        height, width = size
        return int(height), int(width)
    return int(size), int(size)


def geometry(step, halo, scale=2, batch=1):
    """All the sizes of one run, from the step (in LR pixels) it keeps."""
    step_h, step_w = as_size(step)
    input_h, input_w = step_h + 2 * halo, step_w + 2 * halo
    kept = batch * step_h * step_w * scale * scale
    computed = batch * input_h * input_w * scale * scale
    return {
        "step": (step_h, step_w),
        "halo": halo,
        "scale": scale,
        "batch": batch,
        "input": (batch, 3, input_h, input_w),
        "output": (batch, 3, input_h * scale, input_w * scale),
        "kept_hr": (step_h * scale, step_w * scale),
        "kept": kept,                    # output pixels used
        "computed": computed,            # output pixels computed
        "halo_tax": computed / kept,     # extra work the halo costs, e.g. 1.13 = 13% more
    }


def tiling(frame, kept_hr, batch=1):
    """How many tiles of `kept_hr` cover a (width, height) frame, and how much is wasted.

    Every tile has the same shape, so a frame the tile does not divide gets a last row or
    column that hangs over the edge. `tiling_tax` is that waste: 1.0 means none.
    """
    width, height = frame
    kept_h, kept_w = kept_hr
    cols, rows = math.ceil(width / kept_w), math.ceil(height / kept_h)
    tiles = cols * rows
    runs = math.ceil(tiles / batch)
    return {
        "cols": cols,
        "rows": rows,
        "tiles": tiles,
        "runs": runs,
        "tiles_run": runs * batch,
        "delivered": width * height,
        "tiling_tax": runs * batch * kept_w * kept_h / (width * height),
        "exact": cols * kept_w == width and rows * kept_h == height and tiles % batch == 0,
    }


def exact_tilings(frame, scale=2, halo=4, max_aspect=2.5):
    """Every tile size that divides the frame exactly, smallest first.

    Tiles may be rectangles. 1920x1080 has no useful square tile (it would have to divide
    gcd(1920, 1080) = 120), but plenty of rectangles: 384x270, 640x360, 640x540...
    Very thin tiles (more than `max_aspect` to 1) are skipped: they are nearly all halo.
    """
    width, height = frame
    candidates = []
    for kept_w in range(scale, width + 1, scale):
        if width % kept_w:
            continue
        for kept_h in range(scale, height + 1, scale):
            if height % kept_h or max(kept_w / kept_h, kept_h / kept_w) > max_aspect:
                continue
            shape = geometry((kept_h // scale, kept_w // scale), halo, scale)
            candidates.append({**shape, "kept_px": kept_w * kept_h,
                               "tiles": (width // kept_w) * (height // kept_h)})
    candidates.sort(key=lambda row: row["kept_px"])
    return candidates


def plan_tiling(frame, scale=2, halo=4, target_kept=TARGET_KEPT_PX, batch=1,
                max_aspect=2.5, tolerance=4.0):
    """The tile to run a frame at: the exact tiling closest to `target_kept` pixels.

    If no exact tiling is within `tolerance` times the target area, a square tile near the
    target is used instead and the frame edge is wasted (`padded` is then True).
    """
    exact = [row for row in exact_tilings(frame, scale, halo, max_aspect)
             if 1 / tolerance <= row["kept_px"] / target_kept <= tolerance]
    if exact:
        # Two costs, added as logs so they weigh the same: how far the tile's area is from
        # the target (too big or too small alike), and the extra work its halo costs (a
        # square tile has the least halo for its area).
        def cost(row):
            return abs(math.log(row["kept_px"] / target_kept)) + math.log(row["halo_tax"])

        best = min(exact, key=cost)
        return {**best, **tiling(frame, best["kept_hr"], batch), "padded": False}

    side = max(scale, int(round(math.sqrt(target_kept) / scale)) * scale)
    shape = geometry(side // scale, halo, scale, batch)
    return {**shape, "kept_px": side * side, **tiling(frame, shape["kept_hr"], batch), "padded": True}
