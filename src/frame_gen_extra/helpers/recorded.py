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


# screen captures of desktops and applications, 24-30 fps (notebook 11, picked by eye)
SCREEN_VIDEOS = (
    "Cinema_4D/3N2nRlKlbgk", "Cinema_4D/3OnjlqkFClw", "Cinema_4D/6y8iPU4xTZs",
    "Cinema_4D/HuPwUlHWi7M", "Cinema_4D/Wb7GrS8csC8", "Cinema_4D/WqMNDw7zj24",
    "Cinema_4D/YMpf7sjLOsw", "Cinema_4D/YUF6yUFzZR8", "Cinema_4D/ek28bIHBxLE",
    "Cinema_4D/fI49YQ-c7Gg", "Cinema_4D/m077KW7jD_w", "Cinema_4D/zpNnUUQBgMo",
    "Computer/ldH3aBSYGW4", "Digital_painting/CyaXHJjh1E0", "Digital_painting/FpmtImp2jQE",
    "Digital_painting/anv74Y5n2BI", "Digital_painting/kbA1I_1LdLM", "Digital_painting/usj75M0UOIg",
    "Font/NBRptKep6vU", "Font/_pqbCCuxIeM", "Font/uEiWevVqYw0",
    "Map/VFotXCaAxbE", "Map/kOPj4JVxkaE", "Map/rSSVApBDxdk",
    "Map/tb0C1CO4ang", "Map/w1sWRlMQD3U", "Microsoft_Windows/0ElkyEf7bVU",
    "Microsoft_Windows/5XN1agV-pHg", "Microsoft_Windows/5l4GAZxhjkY", "Microsoft_Windows/65KCQsbVjao",
    "Microsoft_Windows/A5HBnSAZEtY", "Microsoft_Windows/EjpH5f9xv90", "Microsoft_Windows/OJzjrFRwouQ",
    "Microsoft_Windows/Tmd9TMyFdEM", "Microsoft_Windows/WqnDsSZJaRk", "Microsoft_Windows/ZyCTAVT_e6M",
    "Microsoft_Windows/gscBIcLm_Gw", "Microsoft_Windows/kVhdOWxQNF8", "Microsoft_Windows/muZ87U-unZE",
    "Microsoft_Windows/rqRovhzewks", "Microsoft_Windows/yAqsK_3gimg", "Microsoft_Windows/yw9xxpdfGbo",
    "Personal_computer/6IkVjaV-PxU", "Personal_computer/wMw8j9qoY24", "Screencast/1_iVs-exIEk",
    "Screencast/1cqdCQZ_rYw", "Screencast/5IWA6qNMiA8", "Screencast/BQldp1LelSY",
    "Screencast/gyogNNYHSuo", "Screencast/i_Vl80pJjdg", "Screencast/jH6J-n6zt4c",
    "Screencast/nAhUUivmD-Q", "Screencast/pykywx452N8", "Screencast/tGRfnsJnSTY",
    "Screenshot/_j1FKT3sdfw", "Sprite_computer_graphics/haAw7_HBc24", "Text_messaging/eYEFLv9do0k",
    "Web_page/AZITFc78-sA", "Web_page/GD9yAOho4zc", "Web_page/QdZj6wYAtZs",
    "Web_page/lrIYoUJqu44", "Web_page/q6CBH_TKEuE", "Website/15qRSwJ6uso",
    "Website/Hh3ms730O1o", "Website/IBA-dN_pSH8", "Website/IdZ8_S0ptJg",
    "Website/VVxYUNZ8u8U", "Website/aP-pZIQs6yE", "Website/md5YK_CxD3s",
    "Website/wCSTWjZ31iA", "Website/xNmwZ1rholY", "Computer/DyS-XDmfjmM",
    "Digital_painting/YUwbMpIbz-w", "Font/JxOaKvheZms", "Microsoft_Windows/1u954QeRdFE",
    "Microsoft_Windows/jKMgt_ugvH0", "Personal_computer/aZeEqvaJzus", "Personal_computer/zqCPKnzmcYA",
    "Sprite_computer_graphics/gmaQQmRPeeM", "Text_messaging/bkoMyKY_cAY", "Web_page/MmNPG0Gn6lc",
)

# screen captures of Synthesia, a piano-roll app: bars falling at a constant speed, 24-30 fps
PIANO_ROLL_VIDEOS = (
    "Synthesia/2exjc7vLx0c", "Synthesia/3vA2Waiv5uw", "Synthesia/DNtD6V8-PL0",
    "Synthesia/EsAzTy5B_gc", "Synthesia/OLtdrTX15jE", "Synthesia/h_1iiOMovYc",
    "Synthesia/nTcyWr7ahS0", "Synthesia/v8ySoSABhuo", "Synthesia/6_iEpiSccyM",
    "Synthesia/6fz3zq3zfrk",
)

# the screen captures at 50-60 fps
SCREEN_VIDEOS_60 = (
    "Digital_painting/vG2G88APxas", "Map/VSn5N7cNiGg", "Microsoft_Windows/XlvlIq-TpDI",
    "Microsoft_Windows/veL8195FELU",
)
SCREENS = SCREEN_VIDEOS + PIANO_ROLL_VIDEOS + SCREEN_VIDEOS_60


def split_by_video(videos, share, seed):
    """Per category ("Category/video_id"), a seeded `share` of its videos - at least one where it
    has two or more - and the rest: (chosen, rest). A video's clips are never on both sides."""
    rng = np.random.default_rng(seed)
    chosen, rest = [], []
    for category in sorted({v.split("/")[0] for v in videos}):
        items = sorted(v for v in videos if v.startswith(category + "/"))
        rng.shuffle(items)
        count = int(round(share * len(items))) if len(items) > 1 else 0
        chosen += items[:count]
        rest += items[count:]
    return chosen, rest


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
