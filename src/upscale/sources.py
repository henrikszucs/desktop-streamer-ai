"""Where the upscaler's frames come from: the public datasets `datasets/download.py` fetched,
read back through the lists it wrote.

Every dataset folder holds a `train.txt` and a `test.txt` - one file per line, relative to
that folder - and the split in them is the one to keep: the publisher's where there is one,
by video for the YouTube sequences, so the sequences of one video never land on both sides.
Three formats come out of them:

- images - DIV2K, Flickr2K, Set5..Urban100 - PNG, read whole by `load_image`;
- sequences - yt8m-videos - lossless H.264 `.mkv` of a few dozen frames, `load_sequence`;
- raw clips - UVG, once its `.7z` are unpacked - headerless 1080p 4:2:0, `load_yuv`.

All three hand back uint8 RGB, (H, W, 3) for an image and (T, H, W, 3) for frames: the
layout `degrade.degrade` takes. Video is decoded by ffmpeg and converted with
`degrade.FROM_YUV` - BT.709 in TV range, what YouTube and UVG are both coded in, with
swscale's accurate rounding - so a frame read here and pushed through the stream comes
back in the colours it went in with. (PyAV's conversion, the obvious alternative, darkens
every frame by about a level and has no switch for the accurate path.)

The datasets folder is `DATASETS_DIR` if that is set - the downloader's own rule - else
`datasets/` at the repo root. Downloads are in its `raw/` (`RAW`); what this problem makes
of them goes beside it, in `upscale/`.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from degrade import FROM_YUV, ffmpeg

ROOT = Path(os.environ.get("DATASETS_DIR", Path(__file__).resolve().parents[2] / "datasets"))
RAW = ROOT / "raw"

IMAGES = ("div2k", "flickr2k", "sr-benchmark")
SEQUENCES = ("yt8m-videos",)
CLIPS = ("uvg",)

UVG_SIZE = (1920, 1080)


def files(name, split="train", root=RAW):
    """The data files of one dataset's `split` ("train" or "test"), as absolute paths."""
    folder = Path(root) / name
    listing = folder / f"{split}.txt"
    if not listing.exists():
        raise FileNotFoundError(f"{listing} is missing - run: uv run datasets/download.py {name}")
    return [folder / line for line in listing.read_text(encoding="utf-8").splitlines() if line]


def load_image(path):
    """One image as (H, W, 3) uint8 RGB."""
    return np.asarray(Image.open(path).convert("RGB"))


def load_sequence(path):
    """Every frame of one `.mkv` sequence as (T, H, W, 3) uint8 RGB, and its frame rate."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
                            "-show_entries", "stream=width,height,avg_frame_rate", str(path)],
                           capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    numerator, denominator = (int(x) for x in stream["avg_frame_rate"].split("/"))
    out = ffmpeg(["-i", str(path), "-map", "0:v:0", "-vf", FROM_YUV, "-f", "rawvideo"])
    frames = np.frombuffer(out, dtype=np.uint8).reshape(-1, stream["height"], stream["width"], 3)
    return frames, numerator / (denominator or 1)


def yuv_length(path, size=UVG_SIZE):
    """How many frames a raw 4:2:0 file holds."""
    width, height = size
    return Path(path).stat().st_size // (width * height * 3 // 2)


def load_yuv(path, start=0, count=None, step=1, size=UVG_SIZE):
    """Frames `start`, `start + step`, ... of a raw 8-bit 4:2:0 file as (T, H, W, 3) uint8 RGB:
    `count` of them, or all there are. A `step` of 2 or 4 turns UVG's 120 fps into 60 or 30."""
    width, height = size
    frame_bytes = width * height * 3 // 2
    data = np.memmap(path, dtype=np.uint8, mode="r")
    indices = range(start, data.size // frame_bytes, step)
    if count is not None:
        indices = indices[:count]
    planes = b"".join(data[i * frame_bytes:(i + 1) * frame_bytes].tobytes() for i in indices)
    out = ffmpeg(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-i", "pipe:0",
                  "-vf", FROM_YUV, "-f", "rawvideo"], planes)
    return np.frombuffer(out, dtype=np.uint8).reshape(-1, height, width, 3)
