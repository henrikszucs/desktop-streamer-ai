"""Loading the patch set, training a model, and scoring it - shared by every model notebook.

A typical notebook does:

    manifest = training.load_manifest(DATA_DIR)
    train_loader, val_loader = training.loaders(DATA_DIR, manifest, batch_size=32)
    trim = training.trimmer(manifest)
    history, best = training.train(model, train_loader, val_loader, trim, epochs=100)
    scores = training.evaluate(model, val_loader, trim)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import metrics, patches, sources
from .patches import PatchPairs

DATA_DIR = sources.ROOT / "upscale"     # what 0_data_preprocess.ipynb writes

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
    """The manifest.json that 0_data_preprocess.ipynb wrote next to the patches."""
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        raise RuntimeError(f"no dataset in {data_dir} - run 0_data_preprocess.ipynb first")
    return json.loads(path.read_text(encoding="utf-8"))


def describe(manifest):
    """Print the device and what the patch set is."""
    gpu = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else ""
    print("device ", DEVICE, gpu)
    print("scale  ", manifest["scale"])
    print("patch  ", f"{manifest['input_lr']} px LR in ({manifest['patch_lr']} px + a halo of "
          f"{manifest['halo']}) -> {manifest['patch_hr']} px HR out, {manifest['frames_per_patch']} frames each")


def loaders(data_dir, manifest, batch_size=32, val_batch_size=None):
    """Training loader (shuffled, flipped and rotated) and validation loader (fixed)."""
    train_set = PatchPairs(data_dir, "train", manifest, augment=True)
    val_set = PatchPairs(data_dir, "val", manifest, augment=False)
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


@torch.no_grad()
def evaluate(model, loader, trim):
    """Every score in `metrics.report`, averaged over a loader."""
    model.eval()
    totals, count = {}, 0
    for lr, hr in loader:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        scores = metrics.report(trim(model(lr)), hr)
        for key, value in scores.items():
            totals[key] = totals.get(key, 0.0) + value * len(lr)
        count += len(lr)
    return {key: value / count for key, value in totals.items()}


def train(model, train_loader, val_loader, trim, epochs, learning_rate=1e-3, loss_fn=F.l1_loss,
          select="wpsnr", print_every=10):
    """Adam with a cosine learning rate, scored on the validation split after every epoch.

    The epoch with the best `select` score is kept and loaded back into `model` at the end.
    Returns the per-epoch history and the best epoch.
    """
    model.to(DEVICE)
    optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    history, best = [], {"epoch": -1, "score": float("-inf"), "state": None}
    started = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(DEVICE), hr_batch.to(DEVICE)
            loss = loss_fn(trim(model(lr_batch)), hr_batch)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(lr_batch)
            seen += len(lr_batch)
        schedule.step()

        scores = evaluate(model, val_loader, trim)
        scores["loss"] = total_loss / seen
        history.append(scores)
        if scores[select] > best["score"]:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = {"epoch": epoch + 1, "score": scores[select], "state": state}

        if print_every and (epoch < 3 or (epoch + 1) % print_every == 0 or epoch + 1 == epochs):
            print(f"epoch {epoch + 1:3}/{epochs}  loss {scores['loss']:.4f}  "
                  f"{select} {scores[select]:.2f} dB  {time.time() - started:5.0f}s")

    if best["state"] is None:
        raise RuntimeError(f"no epoch had a valid {select} score - the training diverged "
                           "(NaN); try a lower learning rate")
    model.load_state_dict(best["state"])
    if print_every:
        print(f"best epoch {best['epoch']} ({select} {best['score']:.2f} dB), restored")
    return history, best


def plot_history(history, baseline, key="wpsnr", loss_name="L1 loss"):
    """Training loss on the left, validation score against bicubic on the right."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(history) + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.5))
    left.plot(epochs, [h["loss"] for h in history])
    left.set_title(f"training {loss_name}")
    left.set_xlabel("epoch")
    right.plot(epochs, [h[key] for h in history], label="model")
    right.axhline(baseline[key], color="0.5", linestyle="--", label="bicubic")
    right.set_title(f"validation {key} (dB)")
    right.set_xlabel("epoch")
    right.legend()
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
def assert_same_model(a, b, size=72):
    """Two models must give the same output on random input (e.g. after save and load)."""
    a, b = a.to(DEVICE).eval(), b.to(DEVICE).eval()
    probe = torch.rand(2, 3, size, size, device=DEVICE)
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
