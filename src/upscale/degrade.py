"""The low resolution input an upscaler meets in the stream, made from a clean frame.

A bicubic downscale is not what reaches the client. desktop-streamer's host
(`src/client/web/src/room/stream-ffmpeg.js`) scales the captured screen to the height the peer
asked for, converts it to 4:2:0 and encodes it as H.264 at a constant bitrate, and the
client upscales what the decoder gives back - blocking, ringing, smeared chroma and all. A
model trained on clean bicubic pairs has never seen any of that and sharpens it. So this
runs the frame through the same chain, with ffmpeg doing what ffmpeg does on the host:

1. **crop** to a multiple of twice the scale, top left, so the LR frame has even sides
   (4:2:0 and every hardware encoder need them) and every LR pixel sits on exactly
   `scale` x `scale` HR pixels - the alignment the patch pairs rely on;
2. **scale** down in RGB with one of the filters a host uses: `bilinear` stands in for the
   `scale_d3d11` of the NVENC and AMF lines, `bicubic` is ffmpeg's own `scale`, which the
   libx264 lines run, and `area` covers a box-averaging scaler;
3. **4:2:0** in BT.709, TV range;
4. **H.264** with the host's libx264 line - `ultrafast`, `zerolatency`, High profile, no B
   frames, a keyframe every second, a constant bitrate with a buffer of two frames;
5. **decode** and convert back to RGB with the same matrix.

The bitrate is given as **bits per pixel per frame** of the encoded picture, not in kbit/s,
so one number means the same quality at any size and frame rate. The host's default - 8
Mbit/s, 90% of it for the picture - is 0.46 at 960x540 and 30 fps, the frame a 1080p
screen is sent as when the client upscales x2; its 200 kbit/s floor is 0.013. `sample`
draws from that range, log-uniform, so a low bandwidth line is as common as a fast one.

`deblock` turns x264's loop filter back on, which `ultrafast` leaves off. The libx264 line
shows hard edges on the block grid; the NVENC and AMF lines a host with a GPU takes always
filter them, and their picture sits between the two. Drawing it per sample covers both
kinds of host with the one encoder that runs everywhere.

Frames (T, H, W, 3) are encoded as the video they are, motion and all, and come back one
for one, the first of every second a keyframe squeezed into a two-frame buffer. A single
image (H, W, 3) is one frame, so it comes back as that keyframe - the worst picture a
stream shows. To see a still image as a stream does, motion and all, make a clip of it
first (`motion.py`).

`bpp=None` skips the codec and leaves the downscale and the 4:2:0 round trip - the clean
pair the notebooks trained on until now, minus its full resolution chroma, and what a fast
line looks like once the codec has nothing left to throw away. `scale=1` skips the
downscale instead, which is the stream degradation for a model that does not upscale.
"""

import math
import subprocess
from dataclasses import asdict, dataclass

import numpy as np

FILTERS = ("bilinear", "bicubic", "area")

# 8 Mbit/s * 0.9 over 960x540 at 30 fps, and the 200 kbit/s floor over the same frame
STREAM_BPP = 0.46
BPP_RANGE = (0.013, 0.46)


@dataclass(frozen=True)
class Degradation:
    """One way through the stream - recorded per sample so results can be split by it."""

    scale: int = 2
    filter: str = "bicubic"
    bpp: float | None = STREAM_BPP  # None: no codec
    fps: float = 30
    deblock: bool = False  # x264's loop filter, as the hardware encoders run it

    def as_dict(self):
        return asdict(self)


def sample(rng, scale=2, fps=30, bpp_range=BPP_RANGE, clean=0.1, deblock=0.5):
    """A random degradation: any filter, a log-uniform bitrate, the loop filter on for a
    `deblock` share - or, for a `clean` share, no codec at all. `rng` is a
    `numpy.random.Generator`."""
    low, high = (math.log(b) for b in bpp_range)
    filter = str(rng.choice(FILTERS))
    bpp = float(math.exp(rng.uniform(low, high)))
    filtered = bool(rng.random() < deblock)
    if rng.random() < clean:
        return Degradation(scale=scale, filter=filter, bpp=None, fps=fps)
    return Degradation(scale=scale, filter=filter, bpp=bpp, fps=fps, deblock=filtered)


def crop(frames, scale):
    """`frames` cut down, top left, to sides that are multiples of 2 * `scale`."""
    unit = 2 * scale
    height, width = frames.shape[-3:-1]
    return frames[..., :height - height % unit, :width - width % unit, :]


# swscale's fast defaults round and interpolate chroma loosely enough to darken a frame by
# over a level, which a model would learn to undo; the accurate flags take it to ~0.1
EXACT = "accurate_rnd+full_chroma_int"
# BT.709 TV range back to RGB - also how `sources` decodes, so both agree on every colour
FROM_YUV = f"scale=flags={EXACT}:in_color_matrix=bt709:in_range=tv,format=rgb24"


def ffmpeg(args, data=None):
    """Run ffmpeg with `data` on its stdin, and return what it writes to its stdout."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args, "pipe:1"],
                            input=data, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def degrade(hr, degradation=Degradation()):
    """(HR, LR) from a uint8 RGB image (H, W, 3) or frames (T, H, W, 3): HR is `hr` cropped
    onto the scale grid, LR is what the stream turns it into. Both keep the input's layout."""
    d = degradation
    hr = crop(np.asarray(hr, dtype=np.uint8), d.scale)
    frames = hr[None] if hr.ndim == 3 else hr
    count, height, width = frames.shape[:3]
    lr_width, lr_height = width // d.scale, height // d.scale

    raw = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(d.fps), "-i", "pipe:0"]
    down = f"scale={lr_width}:{lr_height}:flags={d.filter}+{EXACT}:out_color_matrix=bt709:out_range=tv,format=yuv420p"
    if d.bpp is None:
        out = ffmpeg([*raw, "-vf", f"{down},{FROM_YUV}", "-f", "rawvideo"], frames.tobytes())
    else:
        kbps = d.bpp * lr_width * lr_height * d.fps / 1000
        gop = str(round(d.fps))
        stream = ffmpeg([*raw, "-vf", down,
                          "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-profile:v", "high",
                          "-b:v", f"{kbps:.0f}k", "-maxrate", f"{kbps:.0f}k", "-bufsize", f"{2 * kbps / d.fps:.0f}k",
                          "-g", gop, "-keyint_min", gop, "-bf", "0",
                          *(["-deblock", "0:0"] if d.deblock else []), "-f", "h264"], frames.tobytes())
        # one frame out per frame in: a raw H.264 stream has no timestamps, and the decoder
        # would otherwise re-time it and duplicate frames at some rates (12 fps doubles)
        out = ffmpeg(["-f", "h264", "-i", "pipe:0", "-vf", FROM_YUV, "-fps_mode", "passthrough", "-f", "rawvideo"], stream)

    lr = np.frombuffer(out, dtype=np.uint8).reshape(-1, lr_height, lr_width, 3)
    if len(lr) != count:
        raise RuntimeError(f"ffmpeg returned {len(lr)} frames for {count}")
    return hr, (lr[0] if hr.ndim == 3 else lr)
