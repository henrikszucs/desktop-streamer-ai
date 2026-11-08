"""Make the low resolution (LR) picture a client really receives, from a clean frame.

The stream does not send a clean downscale. The host shrinks the screen, converts it to
4:2:0 colour and encodes it with H.264, so the client gets blocky, smeared frames. A model
trained only on clean downscales never learns to handle that. So we run training frames
through the same steps with ffmpeg:

1. **crop** so both sides are a multiple of 2 x scale (LR pixels line up with HR pixels);
2. **scale down** with a filter a host uses (`bilinear`, `bicubic` or `area`);
3. **convert to 4:2:0** (BT.709, TV range);
4. **encode with H.264** like the host does: its libx264 line (`ultrafast`, constant bitrate, no loop
   filter), or its NVENC line (`p1`, `ull`, the loop filter on) with `encoder="nvenc"`;
5. **decode** back to RGB.

The bitrate is given in **bits per pixel per frame** (`bpp`), so it means the same at any
size. The host's 8 Mbit/s default at 960x540, 30 fps is about 0.46 bpp; its 200 kbit/s
floor is about 0.013. `bpp=None` skips the codec. `deblock=True` turns on x264's loop
filter, like the hardware encoders use.

Training draws from `BPP_RANGE`, 0.2 to 1.0, centred on the default: a client streams a
small picture at a high bitrate (8 Mbit/s at 640x360 is 1.0 bpp). Below about 0.1 bpp
most busy blocks are destroyed, which no stream a user watches looks like (summary.md,
"How much the stream costs the LR").

Pass frames as (T, H, W, 3) to encode a moving clip, or one image (H, W, 3).
"""

import math
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass

import numpy as np

FILTERS = ("bilinear", "bicubic", "area")
ENCODERS = ("x264", "nvenc")   # the host's software line and its NVIDIA line (desktop-streamer, stream-ffmpeg.js)

STREAM_BPP = 0.46            # 8 Mbit/s * 0.9 at 960x540, 30 fps
BPP_RANGE = (0.2, 1.0)       # 3.5 to 17 Mbit/s at 960x540, 1.5 to 7.7 at 640x360


@dataclass(frozen=True)
class Degradation:
    """One way through the stream. Saved per source, so results can be split by it."""

    scale: float = 2
    filter: str = "bicubic"
    bpp: float | None = STREAM_BPP  # None = no codec
    fps: float = 30
    deblock: bool = False           # x264's loop filter; NVENC always runs its own
    encoder: str = "x264"           # one of ENCODERS

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


# What a client can be told about how its frames were made, in two groups:
# - "bpp": the bitrate, which the client always knows - it asked for the bandwidth, the
#   height and the frame rate itself;
# - "path": the downscale filter and the loop filter, which it knows only if the host says
#   which encoder line it runs (libx264 is swscale's bicubic and no loop filter).
CONDITIONS = {"bpp": 2, "path": len(FILTERS) + 1}


def condition(degradation, parts=tuple(CONDITIONS)):
    """A degradation (a `Degradation` or the dictionary the manifest stores) as numbers a
    model can be given: for "bpp", log2 of the bitrate against the host's default and a 1
    for no codec; for "path", the filter one-hot and a 1 for the loop filter."""
    d = degradation if isinstance(degradation, dict) else degradation.as_dict()
    values = []
    for part in parts:
        if part == "bpp":
            values += [0.0, 1.0] if d["bpp"] is None else [math.log2(d["bpp"] / STREAM_BPP), 0.0]
        elif part == "path":
            values += [float(d["filter"] == name) for name in FILTERS] + [float(bool(d["deblock"]))]
        else:
            raise ValueError(f"unknown condition part {part!r}, not one of {list(CONDITIONS)}")
    return np.array(values, np.float32)


def crop(frames, scale):
    """Cut the bottom/right edge so both sides are multiples of 2 x scale (rounded up, for a
    scale that is not a whole number; then an LR pixel does not line up with whole HR pixels)."""
    unit = math.ceil(2 * scale)
    height, width = frames.shape[-3:-1]
    return frames[..., :height - height % unit, :width - width % unit, :]


def stream_size(width, height, scale):
    """The stream's size (width, height) for a frame `crop` has cut: both sides even, for 4:2:0."""
    return int(width / scale) // 2 * 2, int(height / scale) // 2 * 2


# ffmpeg's fast defaults darken colours slightly; these flags keep them exact
EXACT = "accurate_rnd+full_chroma_int"
# BT.709 TV range back to RGB (also used when reading videos in `sources`)
FROM_YUV = f"scale=flags={EXACT}:in_color_matrix=bt709:in_range=tv,format=rgb24"


def run_ffmpeg(args, data=None, loglevel="error"):
    """Run ffmpeg with `data` on stdin and return the finished process."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", loglevel, *args, "pipe:1"],
                            input=data, capture_output=True)
    if result.returncode != 0:
        tail = result.stderr.decode(errors="replace").strip().splitlines()[-5:]
        raise RuntimeError("ffmpeg failed: " + "\n".join(tail))
    return result


def ffmpeg(args, data=None):
    """Run ffmpeg with `data` on stdin and return its stdout."""
    return run_ffmpeg(args, data).stdout


# x264's report on one coded frame, which ffmpeg prints at its debug log level
X264_FRAME = re.compile(r"frame=\s*(\d+) QP=([\d.]+) NAL=\d+ Slice:(\w) Poc:\d+\s+"
                        r"I:(\d+)\s+P:(\d+)\s+SKIP:(\d+)\s+size=(\d+) bytes")


def coded_frames(log):
    """What x264 reports about each frame, from ffmpeg's debug log (bytes), in frame order:
    `qp` is the mean over the frame's macroblocks, `type` "I" or "P", `intra` / `predicted` /
    `skipped` how many macroblocks were coded each way, `bits` the frame's coded size."""
    found = (X264_FRAME.search(line) for line in log.decode(errors="replace").splitlines())
    rows = [{"frame": int(m[1]), "qp": float(m[2]), "type": m[3], "intra": int(m[4]), "predicted": int(m[5]),
             "skipped": int(m[6]), "bits": 8 * int(m[7])} for m in found if m]
    return sorted(rows, key=lambda row: row["frame"])


# a slice header field as ffmpeg's trace_headers bitstream filter prints it
TRACE_FIELD = re.compile(r"\] \d+\s+(\w+)\s+\S+ = (-?\d+)")
SLICE_TYPES = "PBI"


def coded_slices(log, macroblocks):
    """What each coded picture's slice headers say, from trace_headers (text), in decode order: `slice_qp` the
    QP its slices start at (26 + pic_init_qp_minus26 + slice_qp_delta, the mean over its macroblocks when it has
    several slices), `type` "I" or "P", `deblocked` whether the loop filter runs. It is what a client can read off
    the stream (desktop-streamer, stream-h264.js), where x264's `qp` is the mean the encoder ended up at."""
    pictures, init, current = [], 0, None
    for line in log.splitlines():
        m = TRACE_FIELD.search(line)
        if not m:
            continue
        name, value = m[1], int(m[2])
        if name == "pic_init_qp_minus26":
            init = value
        elif name == "first_mb_in_slice":
            if value == 0:
                pictures.append([])
            current = {"first": value, "deblock_idc": 0}
            pictures[-1].append(current)
        elif current is not None and name == "slice_type":
            current["type"] = SLICE_TYPES[value % 5] if value % 5 < 3 else "?"
        elif current is not None and name == "slice_qp_delta":
            current["qp"] = 26 + init + value
        elif current is not None and name == "disable_deblocking_filter_idc":
            current["deblock_idc"] = value
    rows = []
    for t, slices in enumerate(pictures):
        starts = [s["first"] for s in slices] + [macroblocks]
        qp = sum(s["qp"] * (end - start) for s, start, end in zip(slices, starts, starts[1:])) / macroblocks
        rows.append({"frame": t, "slice_qp": qp, "type": "I" if all(s["type"] == "I" for s in slices) else "P",
                     "deblocked": any(s["deblock_idc"] != 1 for s in slices)})
    return rows


def _encoder(lr_width, lr_height, d):
    """The host's line for `d.encoder`, for a stream of this size: a constant bitrate with a buffer of two
    frames (100 kbit at least, as the host asks), a keyframe a second, no B frames."""
    kbps = d.bpp * lr_width * lr_height * d.fps / 1000
    keyframe_every = str(round(d.fps))
    rate = ["-b:v", f"{kbps:.0f}k", "-maxrate", f"{kbps:.0f}k", "-bufsize", f"{max(100, 2 * kbps / d.fps):.0f}k",
            "-g", keyframe_every, "-keyint_min", keyframe_every, "-bf", "0"]
    if d.encoder == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull", "-rc", "cbr", "-zerolatency", "1",
                "-delay", "0", "-forced-idr", "1", "-profile:v", "high", *rate, "-f", "h264"]
    if d.encoder != "x264":
        raise ValueError(f"unknown encoder {d.encoder!r}, not one of {list(ENCODERS)}")
    return ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-profile:v", "high", *rate,
            *(["-deblock", "0:0"] if d.deblock else []), "-f", "h264"]


def _report(log, count, d, lr_width, lr_height):
    """Each frame's slice header report (`coded_slices`), with x264's own (`coded_frames`) merged in for x264."""
    log = log.decode(errors="replace")
    report = coded_slices(log, (lr_width // 16) * ((lr_height + 15) // 16))
    if len(report) != count:
        raise RuntimeError(f"{d.encoder} coded {len(report)} pictures for {count} frames")
    if d.encoder == "x264":
        x264 = coded_frames(log.encode())
        if len(x264) != count:
            raise RuntimeError(f"x264 reported {len(x264)} frames for {count}")
        report = [{**row, **own} for row, own in zip(report, x264)]
    return report


def _encode(frames, d, with_frames):
    """`frames` (T, H, W, 3) through the downscale and the encoder: the H.264 stream, and the report on each
    frame if `with_frames` (else [])."""
    height, width = frames.shape[1:3]
    lr_width, lr_height = stream_size(width, height, d.scale)
    flags = _encoder(lr_width, lr_height, d)
    if with_frames:
        flags = [*flags[:-2], "-bsf:v", "trace_headers", *flags[-2:]]
    encoded = run_ffmpeg([*_raw_input(width, height, d.fps), "-vf", _downscale(lr_width, lr_height, d.filter), *flags],
                         frames.tobytes(), loglevel="debug" if with_frames else "error")
    return encoded.stdout, _report(encoded.stderr, len(frames), d, lr_width, lr_height) if with_frames else []


def _raw_input(width, height, fps):
    return ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0"]


def _downscale(lr_width, lr_height, filter):
    return (f"scale={lr_width}:{lr_height}:flags={filter}+{EXACT}"
            f":out_color_matrix=bt709:out_range=tv,format=yuv420p")


def encode_reports(hr, degradations):
    """[(the report on each frame, `_report`; the stream's bits)] of (T, H, W, 3) through
    each of `degradations`, which differ only in bitrate. Not decoded back: for measuring the encoder.

    The frames are downscaled once and x264 reads them from a file. Piping a clip through
    `subprocess` moves 4 KB at a time under the GIL, which is most of the time when many run at once."""
    first = degradations[0]
    frames = crop(np.asarray(hr, dtype=np.uint8), first.scale)
    count, height, width = frames.shape[:3]
    lr_width, lr_height = stream_size(width, height, first.scale)
    # in memory where there is a RAM disk: this is hundreds of MB, written and read again at once
    with tempfile.TemporaryDirectory(dir="/dev/shm" if os.path.isdir("/dev/shm") else None) as folder:
        rgb, yuv = os.path.join(folder, "rgb"), os.path.join(folder, "yuv")
        frames.tofile(rgb)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *_raw_input(width, height, first.fps)[:-1], rgb,
                        "-vf", _downscale(lr_width, lr_height, first.filter), "-f", "rawvideo", yuv], check=True, capture_output=True)
        os.unlink(rgb)
        out = []
        for d in degradations:
            flags = _encoder(lr_width, lr_height, d)
            encoded = run_ffmpeg(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{lr_width}x{lr_height}", "-r", str(d.fps),
                                  "-i", yuv, *flags[:-2], "-bsf:v", "trace_headers", *flags[-2:]], loglevel="debug")
            out.append((_report(encoded.stderr, count, d, lr_width, lr_height), 8 * len(encoded.stdout)))
    return out


def degrade(hr, degradation=Degradation(), with_bits=False, with_frames=False):
    """Return (HR, LR): HR is `hr` cropped, LR is what the stream turns it into. With
    `with_bits`, (HR, LR, bits): the size of the H.264 stream it was sent as (0 with no codec).
    With `with_frames`, (HR, LR, bits, frames): also the report on each frame, `_report`: the slice
    headers' QP, type and loop filter, and for x264 its own report too (empty with no codec)."""
    d = degradation
    hr = crop(np.asarray(hr, dtype=np.uint8), d.scale)
    frames = hr[None] if hr.ndim == 3 else hr
    count, height, width = frames.shape[:3]
    lr_width, lr_height = stream_size(width, height, d.scale)

    report = []
    if d.bpp is None:
        h264 = b""
        out = ffmpeg([*_raw_input(width, height, d.fps), "-vf", f"{_downscale(lr_width, lr_height, d.filter)},{FROM_YUV}",
                      "-f", "rawvideo"], frames.tobytes())
    else:
        h264, report = _encode(frames, d, with_frames)
        # passthrough: exactly one frame out per frame in
        out = ffmpeg(["-f", "h264", "-i", "pipe:0", "-vf", FROM_YUV, "-fps_mode", "passthrough",
                      "-f", "rawvideo"], h264)

    lr = np.frombuffer(out, dtype=np.uint8).reshape(-1, lr_height, lr_width, 3)
    if len(lr) != count:
        raise RuntimeError(f"ffmpeg returned {len(lr)} frames for {count}")
    lr = lr[0] if hr.ndim == 3 else lr
    if with_frames:
        return hr, lr, 8 * len(h264), report
    return (hr, lr, 8 * len(h264)) if with_bits else (hr, lr)
