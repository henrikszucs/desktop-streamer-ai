"""Loading the run set and scoring a model on it - shared by every model notebook.

A typical notebook does:

    manifest = training.load_manifest()
    inputs, targets = runs.load(DATA_DIR, "val", manifest, halo=HALO)
    masks = runs.groups(manifest, "val", inputs[:, -1], targets, HALO)
    scores = training.evaluate(model, inputs, targets, masks, trim=HALO)

and to train:

    train_loader = training.loader(DATA_DIR, manifest, batch_size=64, halo=HALO)
    validation = (inputs, targets, masks)
    history, best = training.train(model, train_loader, validation, HALO, checkpoint=path)

`train` is upscale's: it halves the learning rate whenever the validation score stalls, and
stops once even the smallest learning rate no longer helps. The difference is what it
follows - one score of one group of samples (`select`, by default SSIM-Y on the changed
ones), since over every sample repeating the newest frame is already close to perfect.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics, runs, sources

DATA_DIR = sources.ROOT / "frame_gen_extra"     # what 00_data_preprocess.ipynb writes

# Local output, not committed: trained weights and sample pictures.
FOLDER = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = FOLDER / "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_manifest(data_dir=DATA_DIR):
    """The manifest.json that 00_data_preprocess.ipynb wrote next to the runs."""
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise RuntimeError(f"no dataset in {data_dir} - run 00_data_preprocess.ipynb first")
    return json.loads(path.read_text(encoding="utf-8"))


def describe(manifest):
    """Print the device and what the run set is."""
    gpu = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else ""
    print("device ", DEVICE, gpu)
    print("stream ", f"x{manifest['scale']} down from a {manifest['image_clip']['window'][0]}x"
          f"{manifest['image_clip']['window'][1]} window, {manifest['image_clip']['fps']} fps")
    print("run    ", f"{manifest['input']} px ({manifest['core']} px + a halo of {manifest['halo']}), "
          f"{manifest['frames_per_run']} frames each")


def to_tensor(frames):
    """uint8 (..., H, W, 3) numpy frames as float (..., 3, H, W) on DEVICE in [0, 1]."""
    array = np.ascontiguousarray(np.moveaxis(np.asarray(frames), -1, -3))
    return torch.from_numpy(array).to(DEVICE).float().div_(255.0)


@torch.no_grad()
def predict(model, inputs, batch_size=256):
    """A model's frames for uint8 inputs (N, context, h, w, 3), as a float (N, 3, h, w) tensor
    on the CPU."""
    model.to(DEVICE).eval()
    out = []
    for first in range(0, len(inputs), batch_size):
        batch = to_tensor(inputs[first:first + batch_size])
        out.append(model(*batch.unbind(1)).float().cpu())
    return torch.cat(out)


def evaluate(model, inputs, targets, masks, trim, batch_size=256):
    """Every score in `metrics.report` per group of samples: {group: scores}. `masks` is
    `runs.groups`; `trim` is the halo cut off before scoring."""
    totals = {name: {} for name in masks}
    counts = dict.fromkeys(masks, 0)
    for first in range(0, len(inputs), batch_size):
        chunk = slice(first, first + batch_size)
        prediction = runs.trim(predict(model, inputs[chunk], batch_size), trim).to(DEVICE)
        target = runs.trim(to_tensor(targets[chunk]), trim)
        for name, mask in masks.items():
            chosen = torch.from_numpy(mask[chunk]).to(DEVICE)
            n = int(chosen.sum())
            if not n:
                continue
            for key, value in metrics.report(prediction[chosen], target[chosen]).items():
                totals[name][key] = totals[name].get(key, 0.0) + value * n
            counts[name] += n
    return {name: {key: value / counts[name] for key, value in scores.items()}
            for name, scores in totals.items() if counts[name]}


def loader(data_dir, manifest, batch_size=64, context=2, ahead=1, halo=None):
    """The training loader: a random newest frame and a random horizontal flip per run."""
    train_set = runs.Runs(data_dir, "train", manifest, augment=True, context=context, ahead=ahead, halo=halo)
    # num_workers stays 0: the array is memory mapped, and worker processes would copy it
    print(f"train {len(train_set):6} runs from {len(train_set.sources)} sources")
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)


def train_one_epoch(model, loader, trim, loss_fn, optimiser, show_progress=False):
    """One pass over every training run. Returns the average loss.

    With `show_progress`, a Keras-style progress bar is redrawn in place while it runs.
    """
    model.train()
    total_loss, seen = 0.0, 0
    started = shown = time.time()
    for step, (inputs, target) in enumerate(loader, start=1):
        inputs, target = [part.to(DEVICE) for part in inputs], target.to(DEVICE)
        prediction = runs.trim(model(*inputs), trim)
        loss = loss_fn(prediction, runs.trim(target, trim))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(target)
        seen += len(target)

        # redraw the bar twice a second ("\r" goes back to the start of the line)
        if show_progress and time.time() - shown > 0.5:
            bar = progress_bar(step, len(loader), time.time() - started)
            print(f"\r{bar} - loss: {total_loss / seen:.4f}", end="", flush=True)
            shown = time.time()
    return total_loss / seen


def progress_bar(step, steps, seconds, width=30):
    """Keras-style progress, e.g. "120/348 [==========>...................] - 3s 26ms/step"."""
    done = width * step // steps
    bar = "=" * done + (">" if done < width else "") + "." * (width - done - 1)
    return f"{step}/{steps} [{bar}] - {format_time(seconds)} {seconds / step * 1000:.0f}ms/step"


def format_time(seconds):
    """A duration as "42s" or "12.5 min"."""
    return f"{seconds:.0f}s" if seconds < 120 else f"{seconds / 60:.1f} min"


def save_training(path, progress, model, optimiser):
    """Everything needed to carry on a training later, in one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {**progress, "model": model.state_dict(), "optimiser": optimiser.state_dict()}
    # write a temporary file first, so a crash while saving never leaves half a checkpoint
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_training(path, model, optimiser):
    """Put a saved training back into `model` and `optimiser`. Returns its progress."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state.pop("model"))
    optimiser.load_state_dict(state.pop("optimiser"))   # this holds the learning rate too
    return state


def set_learning_rate(optimiser, rate):
    """Change the step size of every parameter the optimiser trains."""
    for group in optimiser.param_groups:
        group["lr"] = rate


# The least gain that counts as an improvement: 0.001 dB, or about as fine a step of SSIM (0..1)
MIN_IMPROVEMENT = {"ssim": 1e-5}
# What every epoch prints, Keras style: (group, score) -> name
SHOWN = {("changed", "ssim"): "val_changed_ssim_y", ("changed", "wpsnr"): "val_changed_wpsnr",
         ("unchanged", "wpsnr"): "val_unchanged_wpsnr"}


def train(model, train_loader, validation, trim, max_epochs=500, learning_rate=1e-3,
          loss_fn=metrics.luma_chroma_l1, select=("changed", "ssim"), patience=6, min_learning_rate=None,
          min_improvement=None, checkpoint=None, verbose=1):
    """Train `model` with Adam until it stops getting better, scoring it after every epoch.

    `validation` is (inputs, targets, masks) as `runs.load` and `runs.groups` make them, and
    `select` the (group, score) that decides. Otherwise it is upscale's `train`:

    - **Best epoch**: an epoch improves on the best if it beats it by more than
      `min_improvement` (0.001 dB if not given, 0.00001 for SSIM). The best epoch is loaded
      back into `model` at the end. The untrained model - `Repeat` - counts as the first best.
    - **Learning rate on plateau**: after `patience` epochs without improvement, the learning
      rate is halved, and the smaller steps get another `patience` epochs to help.
    - **Stop when it has converged**: once the learning rate is down to `min_learning_rate`
      (`learning_rate` / 100 if not given) and `patience` more epochs bring no improvement,
      training stops. `max_epochs` is a safety limit - or, set low, a smoke test. Training
      also stops at once if the loss becomes NaN.
    - **Checkpoint**: with a `checkpoint` path, the whole training is saved there after every
      epoch. Run the same call again and it carries on where it stopped, or, if it had
      finished, just loads the result. A larger `max_epochs` than the saved run's carries on
      a run that stopped at its limit. Delete the file to train from the start.

    Returns the per-epoch history (every score of every group) and the best epoch
    ({"epoch", "score", "state"}).
    """
    group, key = select
    if min_learning_rate is None:
        min_learning_rate = learning_rate / 100
    if min_improvement is None:
        min_improvement = MIN_IMPROVEMENT.get(key, 0.001)
    metric = f"val_{group}_{key}"
    model.to(DEVICE)
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    if checkpoint is not None and Path(checkpoint).exists():
        progress = load_training(checkpoint, model, optimiser)
        if progress["stopped"] and progress["stopped"].startswith("reached max_epochs") and progress["epoch"] < max_epochs:
            progress["stopped"] = None   # a smoke test carried on into a longer run
        if verbose:
            best = progress["best"]
            print(f"Loaded the training saved in {checkpoint}: {progress['epoch']} epochs done in "
                  f"{format_time(progress['seconds'])}, best epoch {best['epoch']} ({metric}: "
                  f"{best['score']:.4f}), learning rate now {optimiser.param_groups[0]['lr']:.1e}.")
            if progress["stopped"]:
                print(f"It had finished ({progress['stopped']}), so nothing is trained again - "
                      f"delete the file to train from the start.")
    else:
        start = evaluate(model, *validation, trim)[group][key]
        progress = {"epoch": 0, "history": [], "waited": 0, "stopped": None, "start": start,
                    "seconds": 0.0, "best": {"epoch": 0, "score": start, "state": copy_weights(model)}}

    if verbose and progress["stopped"] is None:
        print(f"Training until the model stops improving, at most {max_epochs} epochs. "
              f"Untrained model (repeat): {metric}: {progress['start']:.4f}.")
        if checkpoint is not None:
            print(f"The training is saved to {checkpoint} after every epoch.")

    while progress["stopped"] is None:
        progress["epoch"] += 1
        epoch = progress["epoch"]
        if verbose:
            print(f"Epoch {epoch}/{max_epochs}")

        epoch_started = time.time()
        rate = optimiser.param_groups[0]["lr"]
        loss = train_one_epoch(model, train_loader, trim, loss_fn, optimiser, show_progress=verbose == 1)
        scores = evaluate(model, *validation, trim)
        seconds = time.time() - epoch_started
        progress["history"].append({"scores": scores, "loss": loss, "learning_rate": rate})
        progress["seconds"] += seconds

        best_before = progress["best"]["score"]
        improved = scores[group][key] > best_before + min_improvement
        if improved:
            progress["best"] = {"epoch": epoch, "score": scores[group][key], "state": copy_weights(model)}
            progress["waited"] = 0
        else:
            progress["waited"] += 1
        waited = progress["waited"]

        smaller_rate = None
        if math.isnan(loss):
            progress["stopped"] = "the loss became NaN"
        elif progress["waited"] >= patience and rate <= min_learning_rate:
            progress["stopped"] = f"converged - no improvement in {patience} epochs, even at the smallest learning rate"
        elif progress["waited"] >= patience:
            smaller_rate = max(rate / 2, min_learning_rate)
            set_learning_rate(optimiser, smaller_rate)
            progress["waited"] = 0
        if progress["stopped"] is None and epoch >= max_epochs:
            progress["stopped"] = f"reached max_epochs ({max_epochs}) - it may still get better"

        if checkpoint is not None:
            save_training(checkpoint, progress, model, optimiser)

        if verbose:
            steps = len(train_loader)
            shown = " - ".join(f"{name}: {scores[g][k]:.4f}" for (g, k), name in SHOWN.items() if g in scores)
            over_the_bar = "\r" if verbose == 1 else ""
            print(f"{over_the_bar}{progress_bar(steps, steps, seconds)} - loss: {loss:.4f} - {shown} - "
                  f"learning_rate: {rate:.1e}")
            if improved:
                print(f"Epoch {epoch}: {metric} improved from {best_before:.4f} to {scores[group][key]:.4f}")
            else:
                print(f"Epoch {epoch}: {metric} did not improve from {best_before:.4f} "
                      f"({waited} of {patience} epochs without improvement)")
            if smaller_rate is not None:
                print(f"Epoch {epoch}: no improvement in {patience} epochs - reducing learning rate "
                      f"from {rate:.1e} to {smaller_rate:.1e}")
            if progress["stopped"]:
                print(f"Epoch {epoch}: stopping ({progress['stopped']})")

    best = progress["best"]
    model.load_state_dict(best["state"])
    if verbose:
        print(f"Restoring model weights from the end of the best epoch: {best['epoch']} ({metric}: "
              f"{best['score']:.4f}, {best['score'] - progress['start']:+.4f} over repeat). "
              f"Training took {progress['epoch']} epochs, {format_time(progress['seconds'])}.")
    if best["epoch"] == 0:
        print(f"warning: no epoch beat the untrained model ({metric}: {best['score']:.4f}) - "
              f"with these settings it has learned nothing")
    return progress["history"], best


def copy_weights(model):
    """A copy of the model's weights on the CPU, which further training leaves alone."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def plot_history(histories, baselines, group="changed", key="ssim"):
    """Training loss and one validation score by epoch, for {name: history}, with
    {name: scores} baselines as dashed lines."""
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 3.5))
    for name, history in histories.items():
        epochs = range(1, len(history) + 1)
        left.plot(epochs, [h["loss"] for h in history], "o-", label=name)
        right.plot(epochs, [h["scores"][group][key] for h in history], "o-", label=name)
    for (name, scores), style in zip(baselines.items(), ("--", ":", "-.")):
        right.axhline(scores[group][key], color="0.5", linestyle=style, label=name)
    left.set_title("training loss")
    right.set_title(f"validation {key}, {group} samples")
    for ax in (left, right):
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def save_checkpoint(path, model, **info):
    """Save the weights together with whatever is needed to rebuild the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **info,
                "trained": time.strftime("%Y-%m-%dT%H:%M:%S")}, path)
    print(f"saved {path.resolve()} ({path.stat().st_size / 1e3:.0f} KB)")


def load_checkpoint(path, device=DEVICE):
    return torch.load(path, map_location=device, weights_only=True)


@torch.no_grad()
def assert_same_model(a, b, size=64):
    """Two models must give the same output on the same random input (e.g. after save and
    load), in full float32 - cuDNN's default TF32 rounds to about 1e-3."""
    a, b = a.to(DEVICE).eval(), b.to(DEVICE).eval()
    probe = torch.rand(2, 2, 3, size, size, device=DEVICE)
    with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, allow_tf32=False):
        difference = float((a(*probe) - b(*probe)).abs().max())
    assert difference < 1e-5, f"the two models differ ({difference:.1e})"
    return difference
