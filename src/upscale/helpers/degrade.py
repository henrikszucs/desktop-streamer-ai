"""Make the low resolution (LR) picture a client really receives, from a clean frame.

The stream does not send a clean downscale. The host shrinks the screen, converts it to
4:2:0 colour and encodes it with H.264, so the client gets blocky, smeared frames. A model
trained only on clean downscales never learns to handle that. So we run training frames
through the same steps with ffmpeg:

1. **crop** so both sides are a multiple of 2 x scale (LR pixels line up with HR pixels);
2. **scale down** with a filter a host uses (`bilinear`, `bicubic` or `area`);
3. **convert to 4:2:0** (BT.709, TV range);
4. **encode with H.264** like the host does (libx264 `ultrafast`, constant bitrate);
5. **decode** back to RGB.

The bitrate is given in **bits per pixel per frame** (`bpp`), so it means the same at any
size. The host's 8 Mbit/s default at 960x540, 30 fps is about 0.46 bpp; its 200 kbit/s
floor is about 0.013. `bpp=None` skips the codec. `deblock=True` turns on x264's loop
filter, like the hardware encoders use.

Pass frames as (T, H, W, 3) to encode a moving clip, or one image (H, W, 3).
"""

import math
import subprocess
from dataclasses import asdict, dataclass

import numpy as np

FILTERS = ("bilinear", "bicubic", "area")

STREAM_BPP = 0.46            # 8 Mbit/s * 0.9 at 960x540, 30 fps
BPP_RANGE = (0.013, 0.46)    # from the 200 kbit/s floor to the default


@dataclass(frozen=True)
class Degradation:
    """One way through the stream. Saved per source, so results can be split by it."""

    scale: int = 2
    filter: str = "bicubic"
    bpp: float | None = STREAM_BPP  # None = no codec
    fps: float = 30
    deblock: bool = False

    def as_dict(self):
        return asdict(self)


def sample(rng, scale=2, fps=30, bpp_range=BPP_RANGE, clean=0.1, deblock=0.5):
    """A random degradation: any filter, a random bitrate (log-uniform, so low and high
    are equally common), the loop filter for a `deblock` share, no codec for a `clean` share."""
    low, high = (math.log(b) for b in bpp_range)
    filter = str(rng.choice(FILTERS))
    bpp = float(math.exp(rng.uniform(low, high)))
    filtered = bool(rng.random() < deblock)
    if rng.random() < clean:
        return Degradation(scale=scale, filter=filter, bpp=None, fps=fps)
    return Degradation(scale=scale, filter=filter, bpp=bpp, fps=fps, deblock=filtered)


def crop(frames, scale):
    """Cut the bottom/right edge so both sides are multiples of 2 x scale."""
    unit = 2 * scale
    height, width = frames.shape[-3:-1]
    return frames[..., :height - height % unit, :width - width % unit, :]


# ffmpeg's fast defaults darken colours slightly; these flags keep them exact
EXACT = "accurate_rnd+full_chroma_int"
# BT.709 TV range back to RGB (also used when reading videos in `sources`)
FROM_YUV = f"scale=flags={EXACT}:in_color_matrix=bt709:in_range=tv,format=rgb24"


def ffmpeg(args, data=None):
    """Run ffmpeg with `data` on stdin and return its stdout."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args, "pipe:1"],
                            input=data, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def degrade(hr, degradation=Degradation()):
    """Return (HR, LR): HR is `hr` cropped, LR is what the stream turns it into."""
    d = degradation
    hr = crop(np.asarray(hr, dtype=np.uint8), d.scale)
    frames = hr[None] if hr.ndim == 3 else hr
    count, height, width = frames.shape[:3]
    lr_width, lr_height = width // d.scale, height // d.scale

    raw_input = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(d.fps), "-i", "pipe:0"]
    downscale = (f"scale={lr_width}:{lr_height}:flags={d.filter}+{EXACT}"
                 f":out_color_matrix=bt709:out_range=tv,format=yuv420p")

    if d.bpp is None:
        out = ffmpeg([*raw_input, "-vf", f"{downscale},{FROM_YUV}", "-f", "rawvideo"], frames.tobytes())
    else:
        kbps = d.bpp * lr_width * lr_height * d.fps / 1000
        keyframe_every = str(round(d.fps))
        h264 = ffmpeg([*raw_input, "-vf", downscale,
                       "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-profile:v", "high",
                       "-b:v", f"{kbps:.0f}k", "-maxrate", f"{kbps:.0f}k", "-bufsize", f"{2 * kbps / d.fps:.0f}k",
                       "-g", keyframe_every, "-keyint_min", keyframe_every, "-bf", "0",
                       *(["-deblock", "0:0"] if d.deblock else []), "-f", "h264"], frames.tobytes())
        # passthrough: exactly one frame out per frame in
        out = ffmpeg(["-f", "h264", "-i", "pipe:0", "-vf", FROM_YUV, "-fps_mode", "passthrough",
                      "-f", "rawvideo"], h264)

    lr = np.frombuffer(out, dtype=np.uint8).reshape(-1, lr_height, lr_width, 3)
    if len(lr) != count:
        raise RuntimeError(f"ffmpeg returned {len(lr)} frames for {count}")
    return hr, (lr[0] if hr.ndim == 3 else lr)
