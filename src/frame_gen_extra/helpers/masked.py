"""Whole client tiles to train on, and the loop that trains notebook 10's masked network.

Notebooks 2 and 3 trained on the run set's 80 px spots, where one motion per spot is already
local. Notebook 10 asks where in a client tile (328x188) the tile search is unsure, so its
network trains on those tiles: per clip, `per_clip` tiles where the published search's gate is
open - something moved between N-2 and N - at random received frames, frames N-2, N and N+1
each. A still tile needs no training: the gate hands it back as it was, correction and all.
`build` writes them once into the data folder (`tiles_{split}.npy`, uint8 (S, 3, H, W, 3), and a
`.json` beside it with each tile's source, frame, tile, kept box and label); `load` reads them.

`train` is Adam on `metrics.luma_chroma_l1` over the window less the client's halo, as
`training.train`, but it keeps the epoch that scores best on the **moving** tiles capped at
50 dB (notebooks 6 to 9's column) - so a network that learns to hold back where a motion stops
gets the credit the changed tiles would not give it. The untrained model, the search itself,
counts as the first best.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import budget, clips, history, metrics, runs, training
from .training import DEVICE, to_tensor
from .webexport import TILE_HALO, TILE_INPUT

CAP = 50.0


def build(manifest, split, published, per_clip, seed=0, path=None, count=None):
    """Cut `per_clip` moving tiles from every clip of a split; save and return (frames, meta)."""
    path = Path(path or training.DATA_DIR / f"tiles_{split}")
    received = runs.received_every(manifest)
    scale = manifest["scale"]
    rng = np.random.default_rng(seed)
    frames, meta = [], defaultdict(list)
    for index, source in enumerate(manifest["splits"][split]["sources"][:count]):
        stream = clips.rebuild(source, manifest)
        layout, size = clips.plan(stream.shape[2], stream.shape[1])
        if size != TILE_INPUT:
            continue
        _, still, _ = history.pairs(stream, published, layout, size, received)
        ends = list(range(received, len(stream), received))
        options = [(p, t) for p, n in enumerate(ends) if n + 1 < len(stream) for t in range(len(layout))
                   if still[p, t] >= published.still]
        for k in sorted(rng.choice(len(options), min(per_clip, len(options)), replace=False)):
            p, t = options[k]
            n = ends[p]
            ix, iy, ox, oy, w, h = layout[t]
            frames.append(stream[[n - received, n, n + 1], iy:iy + size[0], ix:ix + size[1]])
            label = budget.primary(budget.events(source, n, received, (scale * (ix + ox), scale * (iy + oy),
                                                                        scale * w, scale * h)))
            meta["source"].append(index)
            meta["frame"].append(n)
            meta["tile"].append(t)
            meta["box"].append((oy, ox, h, w))
            meta["label"].append(label)
    frames = np.stack(frames)
    np.save(path.with_suffix(".npy"), frames)
    path.with_suffix(".json").write_text(json.dumps(meta))
    return frames, {key: np.asarray(value) for key, value in meta.items()}


def load(split, path=None):
    """(frames, meta) as `build` saved them, or None if they are not there."""
    path = Path(path or training.DATA_DIR / f"tiles_{split}")
    if not path.with_suffix(".npy").exists():
        return None
    meta = json.loads(path.with_suffix(".json").read_text())
    return np.load(path.with_suffix(".npy")), {key: np.asarray(value) for key, value in meta.items()}


class Tiles(Dataset):
    """(3, 3, H, W) float tensors - frames N-2, N, N+1 - mirrored left to right half the time."""

    def __init__(self, frames, flip=True):
        self.frames, self.flip = frames, flip

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.frames[i])).permute(0, 3, 1, 2).float().div_(255.0)
        if self.flip and torch.rand(()) < 0.5:
            x = x.flip(-1)
        return x


@torch.no_grad()
def scores(model, frames, meta, batch_size=32):
    """(S,) each tile's wPSNR over its kept box, by `model(previous, current)`."""
    model.eval()
    out = np.empty(len(frames), np.float32)
    boxes = [tuple(b) for b in meta["box"]]
    for first in range(0, len(frames), batch_size):
        previous, current, target = to_tensor(frames[first:first + batch_size]).unbind(1)
        made = model(previous, current)
        for i in range(len(target)):
            box = boxes[first + i]
            out[first + i] = history.sample_wpsnr(clips.kept(made[i:i + 1], box), clips.kept(target[i:i + 1], box))[0]
    return out


def train(model, frames, validation, checkpoint, learning_rate=1e-3, batch_size=16, patience=4, max_epochs=200,
          min_learning_rate=None, log=print):
    """Train until the moving validation tiles (capped) stop improving; resumable from
    `checkpoint`. `validation` is (frames, meta). Returns (history, best) and leaves the best
    epoch's weights in `model`."""
    checkpoint = Path(checkpoint)
    min_learning_rate = min_learning_rate or learning_rate / 100
    model.to(DEVICE)
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        progress = state["progress"]
        log(f"Loaded {checkpoint.name}: {progress['epoch']} epochs, best {progress['best']['epoch']} "
            f"({progress['best']['score']:.3f} dB){', finished' if progress['stopped'] else ''}.")
    else:
        start = np.minimum(scores(model, *validation), CAP).mean()
        progress = {"epoch": 0, "history": [], "waited": 0, "stopped": None, "seconds": 0.0, "start": float(start),
                    "best": {"epoch": 0, "score": float(start), "state": training.copy_weights(model)}}
        log(f"start (the search, or the weights it was given): {start:.3f} dB on the moving validation tiles, capped")
    loader = DataLoader(Tiles(frames), batch_size=batch_size, shuffle=True, drop_last=True)
    while progress["stopped"] is None:
        progress["epoch"] += 1
        started, total, seen = time.time(), 0.0, 0
        rate = optimiser.param_groups[0]["lr"]
        model.train()
        for batch in loader:
            previous, current, target = batch.to(DEVICE).unbind(1)
            made = runs.trim(model(previous, current), TILE_HALO)
            loss = metrics.luma_chroma_l1(made, runs.trim(target, TILE_HALO))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
            seen += len(batch)
        score = float(np.minimum(scores(model, *validation), CAP).mean())
        seconds = time.time() - started
        progress["seconds"] += seconds
        entry = {"loss": total / seen, "score": score, "learning_rate": rate}
        if hasattr(model, "gate"):
            entry["gate"] = model.gate.detach().cpu().tolist()
        progress["history"].append(entry)
        improved = score > progress["best"]["score"] + 0.001
        if improved:
            progress["best"] = {"epoch": progress["epoch"], "score": score, "state": training.copy_weights(model)}
            progress["waited"] = 0
        else:
            progress["waited"] += 1
        if np.isnan(entry["loss"]):
            progress["stopped"] = "the loss became NaN"
        elif progress["waited"] >= patience and rate <= min_learning_rate:
            progress["stopped"] = "converged"
        elif progress["waited"] >= patience:
            training.set_learning_rate(optimiser, max(rate / 2, min_learning_rate))
            progress["waited"] = 0
        if progress["stopped"] is None and progress["epoch"] >= max_epochs:
            progress["stopped"] = f"reached max_epochs ({max_epochs})"
        log(f"epoch {progress['epoch']}: loss {entry['loss']:.5f}, {score:.3f} dB{' (best)' if improved else ''}, "
            f"lr {rate:.1e}{', gate a, b ' + ', '.join(f'{g:+.2f}' for g in entry['gate']) if 'gate' in entry else ''}, "
            f"{seconds:.0f}s")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(), "progress": progress}, checkpoint)
    model.load_state_dict(progress["best"]["state"])
    log(f"best epoch {progress['best']['epoch']}: {progress['best']['score']:.3f} dB against the start's "
        f"{progress['start']:.3f} ({progress['best']['score'] - progress['start']:+.3f}); {progress['epoch']} epochs, "
        f"{progress['seconds'] / 60:.1f} min ({progress['stopped']})")
    return progress["history"], progress["best"]
