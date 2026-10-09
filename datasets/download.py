"""Fetch the public benchmark datasets, high resolution only, into `<dest>/<name>/`.

The datasets folder - `DATASETS_DIR` if that environment variable is set, else this one -
holds `raw/`, where the downloads go, and beside it one folder per problem for what that
problem's preprocessing makes of them (`upscale/`, later `frame_gen_intra/`...). So `<dest>`
is `--dest` if given, else `<datasets folder>/raw`. All of it is gitignored but this script
and the YouTube mix. Pointing it at an external drive is the expected case for the large
ones (`setx DATASETS_DIR E:\\datasets` makes that the default in new terminals).

A dataset is a list of files on its publisher's server: each is downloaded to a `.part` file
(resumed with a Range request if the connection drops), checked against the length the
server announced, unpacked into the dataset's folder, and then deleted unless
`--keep-archives` is given. A `.done` marker per file makes a second run skip what the first
finished, so the command can be re-run after any failure.

Every run ends by writing `train.txt` and `test.txt` into the dataset's folder: one data
file per line, as a path relative to that folder with forward slashes, so the lists stay
valid wherever the folder is moved. The split is the publisher's where there is one (DIV2K
train / valid, the YouTube-8M partition) and the dataset's role where there is not
(Flickr2K is all train; Set5..Urban100 and UVG are all test). `--index` rewrites the lists
from what is on disk without downloading - after unpacking UVG's `.7z`, for instance.

    uv run datasets/download.py --list
    uv run datasets/download.py div2k sr-benchmark
    uv run datasets/download.py div2k flickr2k --dest E:/datasets
    uv run datasets/download.py yt8m --yt8m-shard 1,100
    uv run datasets/download.py uvg --index
    uv run datasets/download.py --categories game
    uv run datasets/download.py yt8m-videos --category "Microsoft Windows=300" --category Minecraft=100
    uv run datasets/download.py yt8m-videos --total 1000
    uv run datasets/download.py yt8m-videos --total 1000 --sequences 5 --sequence-length 32

**No low-resolution data is kept.** How a frame is degraded - the scale, the filter, the
compression - differs between tasks, so each task makes its own inputs from the HR images.
Where a publisher has an HR-only file, only that is fetched; where the LR copies come in
the same archive (Flickr2K), they are downloaded but skipped when unpacking.

YouTube-8M is the exception in both senders and content: its files are TFRecords of
precomputed per-video (or per-second) feature vectors and labels, **not pixels** - there is
nothing in it an upscaler or a frame generator can train on directly. It is fetched the way
Google's own downloader does it (a download plan of file names and MD5s, then a mirror), and
sharded, because the frame-level partition is over a terabyte.

`yt8m-videos` is what YouTube-8M is good for here: its labels, used as an index of YouTube.
`--categories [TEXT]` lists the 3862 categories (name, vertical, train video count), and each
`--category NAME=N` asks for N videos from one, by name or index - the mix is yours. Without
`--category`, the `WEIGHT CATEGORY` lines of `--mix` (default `raw/yt8m-videos/mix.txt`, a
remote-desktop mix: half computer, then film and TV, then everything else) share `--total`
videos. Google
publishes each category's videos as 4-character pseudo-IDs and a lookup from those to
YouTube IDs; the videos are drawn from that list in a fixed shuffled order, resolved, and
fetched with yt-dlp into `yt8m-videos/<Category>/`: video only, SDR, and the best stream there
is - highest resolution (up to `--max-height` if given), then fps, then bitrate. Most
YouTube-8M videos top out at H.264 1080p: they predate YouTube's VP9 and AV1 re-encodes.
Raising N on a re-run continues the same sample. A video already fetched for another
category is not fetched again, so overlapping categories (Minecraft is also Video game) do
not double up. Much of the 2016-2018 list is now deleted or private: those are recorded in
`unavailable.txt` and skipped (delete it to try them again). YouTube-8M keeps only videos of
120-500 s, so a whole video is roughly 50-150 MB at 1080p. `--sequences N` keeps only N runs
of `--sequence-length` frames per video, spread evenly over its middle 90%: the video-only
stream is downloaded, ffmpeg (on PATH) cuts the sequences, and the stream is deleted. (Seeking
into the stream over HTTP would fetch less, but YouTube throttles plain range requests to
about playback speed - 100x slower than yt-dlp's chunked download.) Each sequence lands in
`<Category>/<id>/<k>.mkv`, lossless H.264 (yuv420p, as YouTube sent it): the decoded frames
bit for bit, about 25x smaller than PNGs, at the price of decoding while training. With
`--sequence-format png` it is a folder `<Category>/<id>/<k>/` of PNGs instead. A sequence can
span a scene cut.
Videos whose best stream is under `--min-height` (default 720 - many are 240p or 360p) are
skipped and recorded in `low_resolution.json` with their height. yt-dlp needs a JavaScript
runtime for YouTube, else it gets 403s: the `deno` dependency is one. The categories are video-level labels -
a "Microsoft Windows" video may be someone filming a monitor - so filter before training.
The test list is a fixed 10% of video IDs. CC BY 4.0 covers Google's labels, not the
videos: those stay their uploaders', and YouTube's terms do not allow downloading - keep
them private and never redistribute them.

The `.7z` archives (UVG) are left packed: the standard library cannot read them. Unpack with
7-Zip. Each `.yuv` inside is headerless 8-bit 4:2:0 at 1920x1080, 120 fps.

None of these is screen content - they are photographs and camera video, and a desktop
stream is text, UI edges and flat colour - so they are a check against published results
and general pretraining, not a substitute for real desktop captures. They are research
datasets: DIV2K is for academic use, UVG is CC BY-NC, YouTube-8M is CC BY 4.0.
"""

# internal
import argparse
import csv
import hashlib
import http.client
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

DIV2K = "https://data.vision.ee.ethz.ch/cvl/DIV2K"
EDSR = "https://cv.snu.ac.kr/research/EDSR"
HF = "https://huggingface.co/datasets/eugenesiow"
UVG = "https://ultravideo.fi/video"
UVG_CLIPS = ("Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySetGo", "ShakeNDry", "YachtRide")
YT8M = "http://data.yt8m.org/2/j"
YT8M_VOCABULARY = "https://research.google.com/youtube8m/csv/2/vocabulary.csv"

# name -> (what it is, approximate download, file URLs)
DATASETS = {
    "div2k": (
        "DIV2K 2K images: 800 train + 100 valid (NTIRE 2017)",
        "3.9 GB",
        [f"{DIV2K}/DIV2K_train_HR.zip", f"{DIV2K}/DIV2K_valid_HR.zip"],
    ),
    "flickr2k": (
        "Flickr2K, 2650 2K images - DIV2K's usual training companion (DF2K); the tar's LR is skipped",
        "21.5 GB",
        [f"{EDSR}/Flickr2K.tar"],
    ),
    "sr-benchmark": (
        "Set5, Set14, BSD100, Urban100 - the classic SR test sets",
        "170 MB",
        [f"{HF}/{name}/resolve/main/data/{name}_HR.tar.gz" for name in ("Set5", "Set14", "BSD100", "Urban100")],
    ),
    "uvg": (
        "UVG 1080p 120 fps 8-bit YUV420 test clips, 7 sequences, left as .7z",
        "5.3 GB",
        [f"{UVG}/{clip}_1920x1080_120fps_420_8bit_YUV_RAW.7z" for clip in UVG_CLIPS],
    ),
    "yt8m": (
        "YouTube-8M features + labels (TFRecord, no pixels), sharded - see --yt8m-*",
        "depends on shard",
        [],
    ),
    "yt8m-videos": (
        "YouTube videos from the YouTube-8M categories you pick, via yt-dlp - see --category",
        "~100 MB a video",
        [],
    ),
}

# name -> which list a file goes in, from its path relative to the dataset folder
SPLITS = {
    "div2k": lambda rel: "test" if rel.startswith("DIV2K_valid_HR/") else "train",
    "flickr2k": lambda rel: "train",
    "sr-benchmark": lambda rel: "test",
    "uvg": lambda rel: "test",
    "yt8m": lambda rel: "train" if rel.split("/", 1)[0].endswith("_train") else "test",
    # by video ID - <Category>/<id>.webm or <Category>/<id>/<k>/... - so a video's sequences stay together
    "yt8m-videos": lambda rel: "test" if int(hashlib.md5(rel.split("/")[1].split(".")[0].encode()).hexdigest(), 16) % 10 == 0 else "train",
}

# what a list holds: the pictures, the unpacked UVG clips, the YouTube-8M records, the videos
VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv"}
DATA_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".yuv", ".tfrecord"} | VIDEO_SUFFIXES

# a path component that is, or is delimited as, "LR": Flickr2K_LR_bicubic/, LR_bicubic/, X4_LR
LOW_RES = re.compile(r"(^|[/_])LR([/_]|$)")


def fetch(url: str, dest: Path, attempts: int = 5) -> None:
    """Download `url` to `dest`, resuming a `.part` left by an earlier run or attempt: a
    dropped connection is retried with backoff, a 4xx answer is not."""
    for attempt in range(1, attempts + 1):
        try:
            return fetch_once(url, dest)
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == attempts:
                raise
            reason = e
        except (OSError, http.client.HTTPException) as e:
            if attempt == attempts:
                raise
            reason = e
        print(f"\n  {dest.name}: {reason} - retrying in {2 ** attempt} s ({attempt}/{attempts - 1})")
        time.sleep(2 ** attempt)


def fetch_once(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "desktop-streamer-ai"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if have and response.status != 206:
            have = 0  # the server ignored the range: start over
        total = have + int(response.headers.get("Content-Length", 0))
        with part.open("ab" if have else "wb") as out:
            done = have
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {dest.name}  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    if total and part.stat().st_size != total:
        raise IOError(f"{dest.name}: got {part.stat().st_size} of {total} bytes - re-run to resume")
    part.replace(dest)


def unpack(archive: Path, into: Path) -> bool:
    """Extract a zip or tar into `into`, minus any LR copies; False if we cannot read it."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(into, members=[n for n in z.namelist() if not LOW_RES.search(n)])
    elif archive.name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive) as t:
            t.extractall(into, members=[m for m in t if not LOW_RES.search(m.name)], filter="data")
    else:
        return False
    return True


def get(folder: Path, urls: list[str], keep_archives: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for url in urls:
        archive = folder / url.rsplit("/", 1)[1]
        marker = folder / f".{archive.name}.done"
        if marker.exists():
            print(f"  {archive.name}  already done")
            continue
        if not archive.exists():
            fetch(url, archive)
        if unpack(archive, folder) and not keep_archives:
            archive.unlink()
        marker.touch()


def md5sum(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            md5.update(chunk)
    return md5.hexdigest()


def case_safe(name: str) -> str:
    """Google's file names differ only in case - trainbX, trainBX, trainbx - and overwrite each
    other on Windows and macOS; a hash of the exact name keeps them apart: trainbX-3fa2c1."""
    stem, ext = name.rsplit(".", 1)
    return f"{stem}-{hashlib.md5(name.encode()).hexdigest()[:6]}.{ext}"


def get_yt8m(folder: Path, partition: str, mirror: str, shard: str) -> None:
    """Google's download.py, minus the Python 2: a plan of files and MD5s, then a mirror."""
    shard_id, num_shards = (int(x) for x in shard.split(","))
    if not 1 <= shard_id <= num_shards:
        sys.exit("--yt8m-shard must be X,Y with 1 <= X <= Y")
    release, level, split = partition.split("/")
    folder = folder / partition.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=True)

    plan_path = folder / "download_plan.json"
    if not plan_path.exists():
        fetch(f"http://data.yt8m.org/{release}/download_plans/{level}_{split}.json", plan_path)
    plan = json.loads(plan_path.read_text())["files"]

    files = [f for f in plan if int(hashlib.md5(f.encode()).hexdigest(), 16) % num_shards == shard_id - 1]
    print(f"  {len(files)} of {len(plan)} files in shard {shard_id}/{num_shards}")
    for f in files:
        out = folder / case_safe(f)
        if out.exists() and md5sum(out) == plan[f]:
            continue
        old = folder / f  # where an earlier version of this script put it
        if old.exists() and md5sum(old) == plan[f]:
            old.replace(out)
            continue
        fetch(f"http://{mirror}.data.yt8m.org/{partition}/{f}", out)
        if md5sum(out) != plan[f]:
            out.unlink()
            print(f"  {f}: MD5 mismatch, removed - re-run to fetch it again")


def read_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "desktop-streamer-ai"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def yt8m_vocabulary(folder: Path) -> list[dict]:
    """YouTube-8M's categories, one dict per vocabulary.csv row, fetched once into `folder`."""
    path = folder / "vocabulary.csv"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        fetch(YT8M_VOCABULARY, path)
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def list_categories(folder: Path, text: str) -> None:
    """Print the categories whose name or vertical contains `text`, most videos first."""
    rows = [r for r in yt8m_vocabulary(folder)
            if text.lower() in " ".join((r["Name"], r["Vertical1"], r["Vertical2"], r["Vertical3"])).lower()]
    rows.sort(key=lambda r: -int(r["TrainVideoCount"]))
    print(f"  {'index':>5} {'videos':>9}  name  (vertical)")
    for r in rows:
        verticals = ", ".join(v for v in (r["Vertical1"], r["Vertical2"], r["Vertical3"]) if v)
        print(f"  {r['Index']:>5} {int(r['TrainVideoCount']):>9,}  {r['Name']}  ({verticals})")
    print(f"  {len(rows)} categories")


def parse_categories(vocabulary: list[dict], specs: list[str]) -> list[tuple[dict, int]]:
    """`NAME=N` or `INDEX=N` -> (vocabulary row, N); exits on a name it does not know."""
    by_key = {r["Name"].lower(): r for r in vocabulary} | {r["Index"]: r for r in vocabulary}
    wants = []
    for spec in specs:
        key, _, count = spec.rpartition("=")
        if not key or not count.isdigit():
            sys.exit(f"--category {spec!r}: expected NAME=N, e.g. \"Microsoft Windows=300\"")
        if key.strip().lower() not in by_key:
            sys.exit(f"--category {spec!r}: no such category - see --categories")
        wants.append((by_key[key.strip().lower()], int(count)))
    return wants


def read_mix(path: Path, total: int) -> list[str]:
    """A mix file's `WEIGHT NAME` lines, scaled to `total` videos, as `NAME=N` specs."""
    weights = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            weight, name = line.split(maxsplit=1)
            weights.append((name, float(weight)))
    scale = total / sum(w for _, w in weights)
    counts = [int(w * scale) for _, w in weights]
    # largest remainders get the videos the rounding down left over, so the sum is `total`
    for i in sorted(range(len(weights)), key=lambda i: counts[i] - weights[i][1] * scale)[:total - sum(counts)]:
        counts[i] += 1
    return [f"{name}={n}" for (name, _), n in zip(weights, counts) if n]


class QuietLogger:
    """yt-dlp prints every failure itself; the reason is reported once, by the caller."""

    def debug(self, msg: str) -> None:
        pass

    info = warning = error = debug


def fetched(folder: Path, pattern: str) -> set[str]:
    """The YouTube IDs fetched under `folder`: whole videos `<pattern>.webm`, and sequence
    folders `<pattern>/` that have their `.done` marker."""
    return ({p.stem for p in folder.glob(pattern) if p.suffix in VIDEO_SUFFIXES}
            | {p.parent.name for p in folder.glob(f"{pattern}/.done")})


def get_sequences(source: Path, info: dict, out: Path, sequences: int, length: int, as_mkv: bool) -> int:
    """Cut `sequences` runs of `length` frames from `source`, evenly spread over the middle 90%
    of the video: `out/<k>/0000.png...`, or `out/<k>.mkv` as lossless H.264 (the decoded
    frames bit for bit, far smaller than PNGs). Returns how many sequences it got."""
    fps, duration = info.get("fps") or 30, info.get("duration") or 0
    first, last = 0.05 * duration, 0.95 * duration - length / fps
    if last < first:
        return 0
    got = 0
    for k in range(sequences):
        start = first + (last - first) * (k + 0.5) / sequences
        final = out / (f"{k:02d}.mkv" if as_mkv else f"{k:02d}")
        part = out / f"{final.name}.part"
        shutil.rmtree(part, ignore_errors=True)
        if as_mkv:
            target, encode = part, ["-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-f", "matroska"]
        else:
            part.mkdir()
            target, encode = part / "%04d.png", []
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-ss", f"{start:.3f}", "-i", str(source),
                   "-map", "0:v:0", "-frames:v", str(length), *encode, str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        frames = length if as_mkv else len(list(part.glob("*.png")))
        if result.returncode != 0 or frames != length:
            reason = (result.stderr.strip().splitlines() or [f"{frames} frames"])[-1]
            print(f"    sequence {k} at {start:.0f}s failed: {reason}")
            if part.is_dir():
                shutil.rmtree(part)
            else:
                part.unlink(missing_ok=True)
            continue
        part.replace(final)
        got += 1
    return got


def download_video(ydl, video: str, min_height: int, attempts: int = 3) -> tuple[dict | None, str]:
    """Fetch one video's chosen stream with `ydl`: (info, "") once it is on disk, (info, why)
    when its best stream is under `min_height`, (None, why) when YouTube will not give it.
    A 403 is retried with a freshly extracted stream URL - most are one-offs."""
    # external - only this dataset needs it
    import yt_dlp

    for attempt in range(1, attempts + 1):
        try:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video}", download=False)
            if (info.get("height") or 0) < min_height:
                return info, f"{info.get('height')}p is under --min-height"
            ydl.process_info(info)
            return info, ""
        except yt_dlp.utils.DownloadError as e:
            reason = next((l for l in str(e).splitlines() if l.strip()), "no reason given").removeprefix("ERROR: ")
            if "403" not in reason or attempt == attempts:
                return None, reason
            time.sleep(5 * attempt)


def get_yt8m_videos(folder: Path, specs: list[str], min_height: int, max_height: int, sequences: int,
                    length: int, as_mkv: bool) -> None:
    # external - only this dataset needs it
    import yt_dlp

    if sequences and not shutil.which("ffmpeg"):
        sys.exit("--sequences needs ffmpeg on PATH (winget install ffmpeg)")
    folder.mkdir(parents=True, exist_ok=True)
    wants = parse_categories(yt8m_vocabulary(folder), specs)

    # pseudo-ID -> YouTube ID ("" if the lookup has none), so a re-run resolves nothing twice
    ids_path = folder / "youtube_ids.json"
    ids = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    gone_path = folder / "unavailable.txt"
    gone = set(gone_path.read_text().split()) if gone_path.exists() else set()
    # YouTube ID -> its best height, for those under --min-height: a lower limit tries them again
    low_path = folder / "low_resolution.json"
    low = json.loads(low_path.read_text()) if low_path.exists() else {}
    have = fetched(folder, "*/*")
    refused = 0  # 403s in a row
    cap = f"[height<={max_height}]" if max_height else ""

    try:
        for row, count in wants:
            sub = folder / re.sub(r"[^\w-]+", "_", row["Name"]).strip("_")
            sub.mkdir(exist_ok=True)
            done = len(fetched(sub, "*"))
            print(f"  {row['Name']}: {done} of {count} already here")
            if done >= count:
                continue

            pseudo_ids = re.findall(r'"([^"]+)"', read_text(f"{YT8M}/v/{row['KnowledgeGraphId'][3:]}.js"))[1:]
            random.Random(row["KnowledgeGraphId"]).shuffle(pseudo_ids)  # same order every run
            options = {
                # video only (no merge), SDR (HDR decodes to washed-out 8-bit unless tone-mapped);
                # then the most pixels, the most frames, the most bits - an HLS copy of the same
                # stream often carries ~25% more than the DASH one yt-dlp would otherwise prefer
                "format": f"bv[dynamic_range=SDR]{cap}/bv{cap}/b{cap}",
                "format_sort": ["res", "fps", "tbr"],
                "concurrent_fragment_downloads": 4,  # HLS comes in fragments
                # a sequence's source stream sits in its video's folder until the cuts are made
                "outtmpl": str(sub / "%(id)s" / "source.%(ext)s" if sequences else sub / "%(id)s.%(ext)s"),
                "logger": QuietLogger(),
                "noprogress": True,
                "retries": 3,
            }
            with yt_dlp.YoutubeDL(options) as ydl:
                for pseudo in pseudo_ids:
                    if done >= count:
                        break
                    if pseudo not in ids:
                        try:
                            ids[pseudo] = re.findall(r'"([^"]+)"', read_text(f"{YT8M}/i/{pseudo[:2]}/{pseudo}.js"))[1]
                        except (OSError, IndexError):
                            ids[pseudo] = ""
                    video = ids[pseudo]
                    if not video or video in gone or video in have or low.get(video, min_height) < min_height:
                        continue
                    info, reason = download_video(ydl, video, min_height)
                    if info and reason:
                        low[video] = info.get("height") or 0
                        low_path.write_text(json.dumps(low))
                        print(f"  {video}  skipped: {reason}")
                        continue
                    if not info:
                        shutil.rmtree(sub / video, ignore_errors=True)  # a sequence source's .part
                        if "confirm you" in reason or "429" in reason:
                            sys.exit(f"  YouTube is refusing requests ({reason}) - wait, then re-run")
                        print(f"  {video}  skipped: {reason}")
                        # a 403 that outlasts the retries is still no proof the video is gone - a
                        # re-run tries it again - but a run of them means a block
                        if "403" in reason:
                            refused += 1
                            if refused == 5:
                                sys.exit(f"  YouTube refused {refused} videos in a row - wait, then re-run")
                            continue
                        gone.add(video)
                        with gone_path.open("a") as f:
                            f.write(video + "\n")
                        continue
                    refused = 0
                    if sequences:
                        source = Path(ydl.prepare_filename(info))
                        got = get_sequences(source, info, sub / video, sequences, length, as_mkv)
                        source.unlink(missing_ok=True)
                        if not got:
                            print(f"  {video}  skipped: no sequence could be cut")
                            continue
                        (sub / video / ".done").touch()
                        shape = f"{got} x {length} frames at {info.get('fps')} fps"
                    else:
                        shape = info.get("ext")
                    have.add(video)
                    done += 1
                    print(f"  [{done}/{count}] {video}  {info.get('height')}p  {shape}")
            if done < count:
                print(f"  {row['Name']}: only {done} of {count} - the category has no more available videos")
    finally:
        ids_path.write_text(json.dumps(ids))


def write_lists(folder: Path, split) -> None:
    """Write `train.txt` and `test.txt` into `folder`: one data file per line, sorted, as a
    path relative to `folder` with forward slashes. Both are always written, empty or not."""
    lists = {"train": [], "test": []}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in DATA_SUFFIXES:
            rel = path.relative_to(folder).as_posix()
            lists[split(rel)].append(rel)
    for which, files in lists.items():
        files.sort()
        (folder / f"{which}.txt").write_text("".join(f"{f}\n" for f in files), encoding="utf-8")
    print(f"  train.txt {len(lists['train'])} files, test.txt {len(lists['test'])} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", help="datasets to fetch (see --list)")
    parser.add_argument("--list", action="store_true", help="list the datasets and exit")
    parser.add_argument("--dest", type=Path, default=Path(os.environ.get("DATASETS_DIR", HERE)) / "raw",
                        help="where the dataset folders go (default $DATASETS_DIR/raw, else raw/ in this folder)")
    parser.add_argument("--keep-archives", action="store_true", help="keep zips/tars after unpacking")
    parser.add_argument("--index", action="store_true",
                        help="only rewrite train.txt / test.txt from what is on disk, download nothing")
    parser.add_argument("--yt8m-partition", default="2/video/train",
                        help="<release>/<video|frame>/<train|validate|test> (default 2/video/train)")
    parser.add_argument("--yt8m-mirror", default="us", choices=("us", "eu", "asia"))
    parser.add_argument("--yt8m-shard", default="1,100",
                        help="X,Y: fetch the X-th of Y deterministic shards (default 1,100)")
    parser.add_argument("--categories", nargs="?", const="", metavar="TEXT",
                        help="list the YouTube-8M categories (name or vertical containing TEXT) and exit")
    parser.add_argument("--category", action="append", default=[], metavar="NAME=N",
                        help="yt8m-videos: N videos from this category (name or index); repeat for a mix")
    parser.add_argument("--mix", type=Path, default=HERE / "raw" / "yt8m-videos" / "mix.txt",
                        help="yt8m-videos without --category: a file of WEIGHT CATEGORY lines "
                             "(default raw/yt8m-videos/mix.txt beside this script)")
    parser.add_argument("--total", type=int, default=100,
                        help="yt8m-videos with --mix: how many videos the weights share (default 100)")
    parser.add_argument("--min-height", type=int, default=720,
                        help="yt8m-videos: skip videos whose best stream is lower (default 720)")
    parser.add_argument("--max-height", type=int, default=0,
                        help="yt8m-videos: highest resolution to fetch (default 0: the best there is)")
    parser.add_argument("--sequences", type=int, default=0, metavar="N",
                        help="yt8m-videos: cut N sequences from each video instead of keeping it whole")
    parser.add_argument("--sequence-length", type=int, default=32, metavar="FRAMES",
                        help="yt8m-videos with --sequences: frames per sequence (default 32)")
    parser.add_argument("--sequence-format", choices=("mkv", "png"), default="mkv",
                        help="yt8m-videos with --sequences: one lossless H.264 .mkv per sequence, or a "
                             "folder of PNG frames (default mkv)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up live in a log file too

    if args.categories is not None:
        list_categories(args.dest.resolve() / "yt8m-videos", args.categories)
        return
    if args.list or not args.names:
        for name, (what, size, _) in DATASETS.items():
            print(f"  {name:<13} {size:>16}  {what}")
        return
    unknown = [n for n in args.names if n not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset(s): {', '.join(unknown)} - see --list")

    dest = args.dest.resolve()
    if args.index:
        for name in args.names:
            if not (dest / name).is_dir():
                sys.exit(f"{dest / name} does not exist - download it first")
            print(f"{name}:")
            write_lists(dest / name, SPLITS[name])
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"into {dest}, {shutil.disk_usage(dest).free / 1e9:,.0f} GB free")
    for name in args.names:
        print(f"{name}: {DATASETS[name][0]}")
        if name == "yt8m":
            get_yt8m(dest / name, args.yt8m_partition, args.yt8m_mirror, args.yt8m_shard)
        elif name == "yt8m-videos":
            get_yt8m_videos(dest / name, args.category or read_mix(args.mix, args.total), args.min_height,
                            args.max_height, args.sequences, args.sequence_length, args.sequence_format == "mkv")
        else:
            get(dest / name, DATASETS[name][2], args.keep_archives)
        write_lists(dest / name, SPLITS[name])


if __name__ == "__main__":
    main()
