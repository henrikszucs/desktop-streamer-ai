"""Real screen recordings as a test split, in the shape `clips` and `recursive.load` read.

The run set's desktops, pages and photos are drawn (`scenes.py`): a still image moved by whole
pixels, most of it as one rigid window, every change of speed on a received frame. Notebook 11
checks the methods on what a screen really shows instead: YouTube-8M sequences of the computer
categories (`datasets/yt8m-mix.txt`) that are screen captures - picked by eye from a frame of
each, since the categories also hold filmed monitors, people and kitchen blenders.

Most screen videos are 24-30 fps. Received every second frame, that is a 12-15 fps stream whose
generated frame is the real one between - the same task as the run set's, at half the rate, so
twice the motion between received frames. The few at 50-60 fps are the task itself, but they
were in the run set, and the trained models have seen them.

`manifest` returns {"splits": {name: {"sources": [...]}}, "scale", "received_every", "fps"} with
one video source per clip (`file` as `sources.files` lists it, its frame rate, and a degradation
drawn as the run set draws them - `degrade.sample`, seeded per clip), which `clips.rebuild` turns
into the decoded stream.
"""

import json
import subprocess

import numpy as np

from . import degrade, sources


def frame_rate(path):
    """A video file's frame rate."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
                            "-show_entries", "stream=avg_frame_rate", str(path)], capture_output=True, text=True, check=True)
    numerator, denominator = (int(x) for x in json.loads(probe.stdout)["streams"][0]["avg_frame_rate"].split("/"))
    return numerator / (denominator or 1)


def manifest(groups, dataset="yt8m-videos", scale=2, received_every=2, seed=0):
    """`groups` is {split name: ["Category/video_id", ...]}; every clip of each video is a source."""
    rng = np.random.default_rng(seed)
    root = sources.RAW / dataset
    listed = [line for split in ("train", "test") for line in (root / f"{split}.txt").read_text().splitlines() if line]
    splits = {}
    for name, videos in groups.items():
        found = []
        for video in videos:
            for file in sorted(line for line in listed if line.startswith(video + "/")):
                fps = frame_rate(root / file)
                found.append({"dataset": dataset, "file": file, "kind": "video", "video": video, "fps": fps,
                              "degradation": degrade.sample(rng, scale=scale, fps=round(fps)).as_dict()})
        splits[name] = {"sources": found}
    return {"splits": splits, "scale": scale, "received_every": received_every, "cursor_drawn": False}
