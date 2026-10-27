"""Read the public datasets that `datasets/download.py` fetched.

A copy of `../../upscale/helpers/sources.py`: each problem folder owns its helpers, so a
change made for one problem cannot quietly change another's data.

Every dataset folder has a `train.txt` and a `test.txt`, one file per line. We keep that
split: for the YouTube sequences it is by video, so one video is never on both sides.
Two kinds of file:

- images (DIV2K, Flickr2K, Set5..Urban100, and the desktop screens `datasets/screens.py`
  draws): PNG, read by `load_image`;
- sequences (yt8m-videos): short lossless `.mkv` clips, read by `load_sequence`.

Both return uint8 RGB: (H, W, 3) for an image, (T, H, W, 3) for frames. Video is converted
with the same ffmpeg settings as `degrade`, so colours do not shift on the way.

The datasets folder is `DATASETS_DIR` if set, else `datasets/` at the repo root. Downloads
are in `raw/` (`RAW`).
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .degrade import FROM_YUV, ffmpeg

ROOT = Path(os.environ.get("DATASETS_DIR", Path(__file__).resolve().parents[3] / "datasets"))
RAW = ROOT / "raw"

IMAGES = ("div2k", "flickr2k", "sr-benchmark", "screens")
SEQUENCES = ("yt8m-videos",)


def files(name, split="train", root=RAW):
    """The files of one dataset's `split` ("train" or "test"), as absolute paths."""
    folder = Path(root) / name
    listing = folder / f"{split}.txt"
    if not listing.exists():
        raise FileNotFoundError(f"{listing} is missing - run: uv run datasets/download.py {name}")
    return [folder / line for line in listing.read_text(encoding="utf-8").splitlines() if line]


def load_image(path):
    """One image as (H, W, 3) uint8 RGB."""
    return np.asarray(Image.open(path).convert("RGB"))


def load_sequence(path):
    """Every frame of one `.mkv` as (T, H, W, 3) uint8 RGB, and its frame rate."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
                            "-show_entries", "stream=width,height,avg_frame_rate", str(path)],
                           capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    numerator, denominator = (int(x) for x in stream["avg_frame_rate"].split("/"))
    out = ffmpeg(["-i", str(path), "-map", "0:v:0", "-vf", FROM_YUV, "-f", "rawvideo"])
    frames = np.frombuffer(out, dtype=np.uint8).reshape(-1, stream["height"], stream["width"], 3)
    return frames, numerator / (denominator or 1)
