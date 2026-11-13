"""Loading the patch set, training a model, and scoring it - shared by every model notebook.

A typical notebook does:

    manifest = training.load_manifest(DATA_DIR)
    train_loader, val_loader = training.loaders(DATA_DIR, manifest, batch_size=32)
    trim = training.trimmer(manifest)
    history, best = training.train(model, train_loader, val_loader, trim)
    scores = training.evaluate(model, val_loader, trim)

`train` runs until the model stops getting better: it halves the learning rate whenever the
validation score stalls, and stops once even the smallest learning rate no longer helps. It
can save itself to a checkpoint every epoch. `train_runs` trains several fresh models
(other seeds or settings) and keeps the best one.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import metrics, patches, sources
from .patches import PatchPairs

DATA_DIR = sources.ROOT / "upscale"     # what 00_data_preprocess.ipynb writes

# Local output, not committed: trained weights and sample pictures.
UPSCALE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = UPSCALE_DIR / "checkpoints"
SAMPLE_DIR = UPSCALE_DIR / "data" / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_manifest(data_dir=DATA_DIR):
    """The manifest.json that 00_data_preprocess.ipynb wrote next to the patches."""
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise RuntimeError(f"no dataset in {data_dir} - run 00_data_preprocess.ipynb first")
    return json.loads(path.read_text(encoding="utf-8"))


def describe(manifest):
    """Print the device and what the patch set is."""
    gpu = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else ""
    print("device ", DEVICE, gpu)
    print("scale  ", manifest["scale"])
    print("patch  ", f"{manifest['input_lr']} px LR in ({manifest['patch_lr']} px + a halo of "
          f"{manifest['halo']}) -> {manifest['patch_hr']} px HR out, {manifest['frames_per_patch']} frames each")


def loaders(data_dir, manifest, batch_size=32, val_batch_size=None, condition=None):
    """Training loader (shuffled, flipped and rotated) and validation loader (fixed).

    With a `condition` (see `PatchPairs`), every batch's input is (LR, condition), and
    `train` and `evaluate` call the model with both."""
    train_set = PatchPairs(data_dir, "train", manifest, augment=True, condition=condition)
    val_set = PatchPairs(data_dir, "val", manifest, augment=False, condition=condition)
    # num_workers stays 0: the arrays are memory mapped, and worker processes would copy them
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size or batch_size, shuffle=False)
    print(f"train {len(train_set):6} patches from {len(train_set.sources)} sources")
    print(f"val   {len(val_set):6} patches from {len(val_set.sources)} sources")
    return train_loader, val_loader


def trimmer(manifest):
    """Make a `trim(images)` function that cuts the halo off a model's output.

    The LR patch has `halo` extra pixels on every side, so the output is `halo * scale`
    pixels too wide on every side. We only score the centre, like a tile keeps only its centre.
    `train` and `evaluate` take this function, so they work at any halo.
    """
    pixels = manifest["halo"] * manifest["scale"]

    def trim(images):
        return patches.trim(images, pixels)

    return trim


def to_device(inputs):
    """A batch's input on DEVICE: one tensor, or a list of them (an image and its condition)."""
    if isinstance(inputs, (list, tuple)):
        return [part.to(DEVICE) for part in inputs]
    return inputs.to(DEVICE)


def forward(model, inputs):
    """Run a model on a batch's input, however many tensors it is."""
    return model(*inputs) if isinstance(inputs, list) else model(inputs)


@torch.no_grad()
def evaluate(model, loader, trim):
    """Every score in `metrics.report`, averaged over a loader."""
    model.eval()
    totals, count = {}, 0
    for lr, hr in loader:
        lr, hr = to_device(lr), hr.to(DEVICE)
        scores = metrics.report(trim(forward(model, lr)), hr)
        for key, value in scores.items():
            totals[key] = totals.get(key, 0.0) + value * len(hr)
        count += len(hr)
    return {key: value / count for key, value in totals.items()}


def copy_weights(model):
    """A copy of the model's weights on the CPU, which further training leaves alone."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_one_epoch(model, loader, trim, loss_fn, optimiser, show_progress=False, teacher=None):
    """One pass over every training patch. Returns the average loss.

    With `show_progress`, a Keras-style progress bar is redrawn in place while it runs. With a
    `teacher` (a model that is not trained), the loss is `loss_fn(prediction, target, taught)`,
    where `taught` is the teacher's picture of the same batch.
    """
    model.train()
    total_loss, seen = 0.0, 0
    started = shown = time.time()
    for step, (lr_batch, hr_batch) in enumerate(loader, start=1):
        lr_batch, hr_batch = to_device(lr_batch), hr_batch.to(DEVICE)
        prediction = trim(forward(model, lr_batch))
        if teacher is None:
            loss = loss_fn(prediction, hr_batch)
        else:
            with torch.no_grad():
                taught = trim(forward(teacher, lr_batch))
            loss = loss_fn(prediction, hr_batch, taught)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(hr_batch)
        seen += len(hr_batch)

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


# How each validation score is named in the printout, Keras style ("val_" = validation).
SCORE_NAMES = {"wpsnr": "val_wpsnr", "y": "val_psnr_y", "u": "val_psnr_u", "v": "val_psnr_v",
               "rgb": "val_psnr_rgb", "ssim": "val_ssim_y"}
# The least gain that counts as an improvement: 0.001 dB, or about as fine a step of SSIM (0..1)
MIN_IMPROVEMENT = {"ssim": 1e-5}


def train(model, train_loader, val_loader, trim, max_epochs=500, learning_rate=1e-3,
          loss_fn=F.l1_loss, select="wpsnr", patience=6, min_learning_rate=None,
          min_improvement=None, checkpoint=None, verbose=1, teacher=None):
    """Train `model` with Adam until it stops getting better, scoring it after every epoch.

    How long it trains is not fixed: it follows the `select` score on the validation split.

    - **Best epoch**: an epoch improves on the best if it beats it by more than
      `min_improvement` (0.001 dB if not given, 0.00001 for SSIM). The best epoch is loaded
      back into `model` at the end. The untrained model counts as the first best.
    - **Learning rate on plateau**: after `patience` epochs without improvement, the
      learning rate is halved, and the smaller steps get another `patience` epochs to help.
    - **Stop when it has converged**: once the learning rate is down to `min_learning_rate`
      (`learning_rate` / 100 if not given) and `patience` more epochs bring no improvement,
      even small steps no longer help, so training stops. `max_epochs` is only a safety
      limit. Training also stops at once if the loss becomes NaN. If no epoch ever beat the
      untrained model, it warns: the model has learned nothing.
    - **Checkpoint**: with a `checkpoint` path, the whole training is saved there after
      every epoch. Run the same call again and it carries on where it stopped, or, if it had
      finished, just loads the result. Delete the file to train from the start.
    - **Teacher**: with a `teacher`, `loss_fn` is also given the teacher's picture of every
      training batch (see `train_one_epoch`); validation still scores against the truth.

    The printout is like Keras: `verbose=1` shows a progress bar while an epoch trains, then
    its time, loss, every validation score and the learning rate, and what the epoch meant
    (improved or not, learning rate halved, stopping). `verbose=2` is the same without the
    progress bar, `verbose=0` prints only warnings.

    Returns the per-epoch history and the best epoch ({"epoch", "score", "state"}).
    """
    if min_learning_rate is None:
        min_learning_rate = learning_rate / 100
    if min_improvement is None:
        min_improvement = MIN_IMPROVEMENT.get(select, 0.001)
    metric = SCORE_NAMES[select]
    model.to(DEVICE)
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    # `progress` is the training so far: the untrained score, every epoch's scores, the best
    # epoch, how many epochs it has waited for an improvement at this learning rate, the
    # seconds spent, and why it stopped (None while it is still going)
    if checkpoint is not None and Path(checkpoint).exists():
        progress = load_training(checkpoint, model, optimiser)
        if verbose:
            best = progress["best"]
            print(f"Loaded the training saved in {checkpoint}: {progress['epoch']} epochs done in "
                  f"{format_time(progress['seconds'])}, best epoch {best['epoch']} ({metric}: "
                  f"{best['score']:.4f}), learning rate now {optimiser.param_groups[0]['lr']:.1e}.")
            if progress["stopped"]:
                print(f"It had finished ({progress['stopped']}), so nothing is trained again - "
                      f"delete the file to train from the start.")
    else:
        start = evaluate(model, val_loader, trim)[select]
        progress = {"epoch": 0, "history": [], "waited": 0, "stopped": None, "start": start,
                    "seconds": 0.0, "best": {"epoch": 0, "score": start, "state": copy_weights(model)}}

    if verbose and progress["stopped"] is None:
        print(f"Training until the model stops improving, at most {max_epochs} epochs. "
              f"Untrained model: {metric}: {progress['start']:.4f}.")
        print(f"Learning rate {optimiser.param_groups[0]['lr']:.1e}, halved after {patience} epochs "
              f"without improvement; training stops when {min_learning_rate:.1e} does not help either.")
        if checkpoint is not None:
            print(f"The training is saved to {checkpoint} after every epoch.")

    while progress["stopped"] is None:
        progress["epoch"] += 1
        epoch = progress["epoch"]
        if verbose:
            print(f"Epoch {epoch}/{max_epochs}")

        epoch_started = time.time()
        rate = optimiser.param_groups[0]["lr"]
        loss = train_one_epoch(model, train_loader, trim, loss_fn, optimiser, show_progress=verbose == 1,
                               teacher=teacher)
        scores = evaluate(model, val_loader, trim)
        seconds = time.time() - epoch_started
        scores["loss"] = loss
        scores["learning_rate"] = rate
        progress["history"].append(scores)
        progress["seconds"] += seconds

        best_before = progress["best"]["score"]
        improved = scores[select] > best_before + min_improvement
        if improved:
            progress["best"] = {"epoch": epoch, "score": scores[select], "state": copy_weights(model)}
            progress["waited"] = 0
        else:
            progress["waited"] += 1
        waited = progress["waited"]   # for the printout: a halving below starts it again at 0

        smaller_rate = None
        if math.isnan(loss):
            progress["stopped"] = "the loss became NaN"
        elif progress["waited"] >= patience and rate <= min_learning_rate:
            progress["stopped"] = (f"converged - no improvement in {patience} epochs, "
                                   f"even at the smallest learning rate")
        elif progress["waited"] >= patience:
            # a plateau: take smaller steps, and give them `patience` epochs to help
            smaller_rate = max(rate / 2, min_learning_rate)
            set_learning_rate(optimiser, smaller_rate)
            progress["waited"] = 0
        if progress["stopped"] is None and epoch >= max_epochs:
            progress["stopped"] = f"reached max_epochs ({max_epochs}) - it may still get better"

        if checkpoint is not None:
            save_training(checkpoint, progress, model, optimiser)

        if verbose:
            # e.g. "348/348 [==============================] - 9s 26ms/step - loss: 0.0341 -
            #       val_wpsnr: 30.9120 - ... - learning_rate: 1.0e-04"
            steps = len(train_loader)
            validation = " - ".join(f"{name}: {scores[key]:.4f}" for key, name in SCORE_NAMES.items())
            over_the_bar = "\r" if verbose == 1 else ""
            print(f"{over_the_bar}{progress_bar(steps, steps, seconds)} - loss: {loss:.4f} - "
                  f"{validation} - learning_rate: {rate:.1e}")
            if improved:
                print(f"Epoch {epoch}: {metric} improved from {best_before:.4f} to {scores[select]:.4f}")
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
              f"{best['score']:.4f}, {best['score'] - progress['start']:+.4f} over the untrained "
              f"model). Training took {progress['epoch']} epochs, {format_time(progress['seconds'])}.")
    if best["epoch"] == 0:
        print(f"warning: no epoch beat the untrained model ({metric}: {best['score']:.4f}) - "
              f"with these settings it has learned nothing")
    return progress["history"], best


def train_runs(build_model, runs, train_loader, val_loader, trim, checkpoint_dir=None, **settings):
    """Train a fresh model for every run and keep the one with the best validation score.

    One training is partly luck: the same recipe with another seed can land a few tenths of
    a dB away (notebook 4). Training several times and keeping the best gets closer to what
    the model can really do.

    - `build_model()` returns a new, untrained model.
    - `runs` is a list with one dictionary per run. Each has a "seed", and may change any
      other `train` setting for that run, e.g. {"seed": 1, "learning_rate": 3e-4}.
    - `settings` are the `train` settings every run shares, e.g. `loss_fn`.
    - With a `checkpoint_dir`, each run saves its training there (see `train`), in a file
      named after the run's own settings. Delete the folder to train every run again.

    Returns the best run and the list of all runs. A run is a dictionary: "run" (its entry
    in `runs`), "model" (trained, at its best epoch), "history" and "best" (from `train`).
    """
    print(f"{len(runs)} runs, each training a fresh model: {', '.join(run_name(run) for run in runs)}")
    results = []
    for number, run in enumerate(runs, start=1):
        run_settings = {**settings, **run}
        seed = run_settings.pop("seed")
        print(f"\n{'=' * 100}\nrun {number} of {len(runs)}: {run_name(run)}\n{'=' * 100}")

        torch.manual_seed(seed)
        model = build_model().to(DEVICE)
        checkpoint = None
        if checkpoint_dir is not None:
            checkpoint = Path(checkpoint_dir) / f"{run_name(run)}.pt"

        history, best = train(model, train_loader, val_loader, trim, checkpoint=checkpoint, **run_settings)
        results.append({"run": run, "model": model, "history": history, "best": best})

    winner = max(results, key=lambda result: result["best"]["score"])
    print_runs(results, winner, settings.get("select", "wpsnr"))
    return winner, results


def run_name(run):
    """A short name for a run's settings, e.g. "seed1_learning_rate0.0003". It is also the
    run's checkpoint file name, so a function setting is written as its name."""
    parts = []
    for key, value in run.items():
        value = getattr(value, "__name__", value)
        parts.append(f"{key}{value}")
    return "_".join(parts)


def print_runs(results, winner, select="wpsnr"):
    """One line per run, and how far apart the runs landed."""
    width = max(len(run_name(result["run"])) for result in results) + 2
    print(f"\n{'run':<{width}}{'epochs':>8}{'best epoch':>12}{select:>9}")
    for result in results:
        kept = "   <- kept" if result is winner else ""
        print(f"{run_name(result['run']):<{width}}{len(result['history']):>8}{result['best']['epoch']:>12}"
              f"{result['best']['score']:9.2f}{kept}")
    scores = [result["best"]["score"] for result in results]
    print(f"the runs are {max(scores) - min(scores):.2f} dB apart: a smaller difference between "
          f"two models is luck")


def plot_history(history, baseline, key="wpsnr", loss_name="L1 loss"):
    """Training loss, validation score against bicubic, and the learning rate, by epoch."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(history) + 1)
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(14, 3.5))
    left.plot(epochs, [h["loss"] for h in history])
    left.set_title(f"training {loss_name}")
    middle.plot(epochs, [h[key] for h in history], label="model")
    middle.axhline(baseline[key], color="0.5", linestyle="--", label="bicubic")
    middle.set_title(f"validation {key}" + ("" if key == "ssim" else " (dB)"))
    middle.legend()
    right.plot(epochs, [h["learning_rate"] for h in history])
    right.set_yscale("log")
    right.set_title("learning rate")
    for ax in (left, middle, right):
        ax.set_xlabel("epoch")
    fig.tight_layout()
    plt.show()


def plot_runs(results, baseline, key="wpsnr"):
    """The validation score and the learning rate of every run from `train_runs`, by epoch."""
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 3.5))
    for result in results:
        epochs = range(1, len(result["history"]) + 1)
        label = run_name(result["run"])
        left.plot(epochs, [h[key] for h in result["history"]], label=label)
        right.plot(epochs, [h["learning_rate"] for h in result["history"]], label=label)
    left.axhline(baseline[key], color="0.5", linestyle="--", label="bicubic")
    left.set_title(f"validation {key} (dB)")
    right.set_yscale("log")
    right.set_title("learning rate")
    for ax in (left, right):
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
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


def full_precision():
    """A `with` block in which a GPU computes convolutions in full float32. By default cuDNN
    uses TF32, which rounds to about 1e-3: fine for training, too coarse to check that two
    models compute the same thing (a reparameterized model and its merged form differ by
    9e-4 in TF32 and by 1e-6 in float32)."""
    return torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, allow_tf32=False)


@torch.no_grad()
def assert_same_model(a, b, size=72):
    """Two models must give the same output on random input (e.g. after save and load, or
    a reparameterized model and its merged form), in full float32."""
    a, b = a.to(DEVICE).eval(), b.to(DEVICE).eval()
    probe = torch.rand(2, 3, size, size, device=DEVICE)
    with full_precision():
        difference = float((a(probe) - b(probe)).abs().max())
    assert difference < 1e-5, f"the two models differ ({difference:.1e})"
    return difference


def to_tensor(image):
    """uint8 (H, W, 3) array or PIL image -> float (1, 3, H, W) tensor in [0, 1] on DEVICE."""
    array = np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1))
    return torch.from_numpy(array)[None].float().div(255).to(DEVICE)


def to_image(tensor):
    """(1, 3, H, W) tensor in [0, 1] -> uint8 (H, W, 3) array."""
    return (tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def show_patches(columns):
    """A grid of image batches: one column per entry of `columns` ({name: (N, 3, H, W)})."""
    import matplotlib.pyplot as plt

    rows = len(next(iter(columns.values())))
    fig, axes = plt.subplots(rows, len(columns), figsize=(2.8 * len(columns), 2.8 * rows), squeeze=False)
    for column, (name, images) in enumerate(columns.items()):
        for row in range(rows):
            axes[row][column].imshow(images[row].clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
                                     interpolation="nearest")
            axes[row][column].axis("off")
        axes[0][column].set_title(name, fontsize=10)
    fig.tight_layout()
    plt.show()
