"""Scores that judge a picture the way people do, beside `metrics.py`'s SSIM-Y and PSNR (notebook 26).

- **JND** - the perceptual JND metric of the `perceptual` repo, checked out beside this one
  (`PERCEPTUAL_DIR` moves it): visible error in just-noticeable differences, its predicted
  BT.500 MOS, and the same thing as a differentiable training loss (`jnd_loss`).
- **LPIPS** (Zhang et al. 2018, AlexNet, v0.1) - a distance between deep features whose
  per-channel weights were fitted to people's 2AFC choices between distorted patches. Lower is
  closer. It is built here from its two public weight files (torchvision's AlexNet and the
  LPIPS linear layers), downloaded once into `checkpoints/lpips/`, so it needs no package.
- **GMSD** (Xue et al. 2014) - how unevenly the gradient magnitude is kept across the picture,
  on luma at half resolution. Lower is better. Fitted to nothing.

None of them is anything a model in this folder trains on, except JND for notebook 26's JND
models: a judge that is also a loss favours the model trained on it.
"""

import os
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import metrics
from .training import CHECKPOINT_DIR, UPSCALE_DIR

PERCEPTUAL_DIR = Path(os.environ.get("PERCEPTUAL_DIR", UPSCALE_DIR.parents[2] / "perceptual"))
# the metric's settings, in the perceptual repo's config format ({"loss": {...}}); they differ from
# the spec's defaults in masking exponent 1.2 (0.7), tiles pooled at beta 16 (4), and the BT.500
# annoyance curve's rho 0.551 (0.6) and d50 2.83 (5)
JND_SETTINGS = Path(__file__).with_name("jnd_settings.json")


def _perceptual():
    """The `perceptual_jnd_loss` module of the perceptual repo."""
    if not (PERCEPTUAL_DIR / "perceptual_jnd_loss.py").exists():
        raise RuntimeError(f"no perceptual repo at {PERCEPTUAL_DIR} - check it out there or set PERCEPTUAL_DIR")
    if str(PERCEPTUAL_DIR) not in sys.path:
        sys.path.insert(0, str(PERCEPTUAL_DIR))
    import perceptual_jnd_loss
    return perceptual_jnd_loss


def jnd_loss(**overrides):
    """`PerceptualJNDLoss` with `JND_SETTINGS` (45 px a degree, 64 px tiles), as a training loss:
    d / (d + 1) per picture, the tile-mean colour as CIELAB delta E 76 (smooth)."""
    return _perceptual().PerceptualJNDLoss.from_config(JND_SETTINGS, **{"objective": "bounded", **overrides})


def jnd_judge(**overrides):
    """The same metric for scoring: tile-mean colour as CIEDE2000, as the NumPy reference does.
    `.evaluate(prediction, target)` gives per picture "jnd" (d) and "mos" (BT.500, 1-5)."""
    return _perceptual().PerceptualJNDLoss.from_config(
        JND_SETTINGS, **{"objective": "jnd", "dc_metric": "de2000", **overrides}).eval()


# ---------------------------------------------------------------- LPIPS
LPIPS_DIR = CHECKPOINT_DIR / "lpips"
ALEXNET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
LINEAR_URL = "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/alex.pth"


def _fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
    return path


class LPIPS(nn.Module):
    """LPIPS v0.1 on AlexNet, as `lpips.LPIPS(net="alex")` computes it: each picture scaled to
    [-1, 1] and normalised, the five ReLU outputs of AlexNet's features, each unit-normalised
    over channels, their squared difference weighted per channel and averaged over the picture,
    summed over the five. `forward(prediction, target)` -> (N,), pictures in [0, 1]."""

    SHIFT = (-0.030, -0.088, -0.188)
    SCALE = (0.458, 0.448, 0.450)
    TAPS = (1, 4, 7, 9, 11)          # the ReLUs after conv 1..5

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, 2), nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, 2), nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        alexnet = torch.load(_fetch(ALEXNET_URL, LPIPS_DIR / "alexnet-owt-7be5be79.pth"), map_location="cpu",
                             weights_only=True)
        self.features.load_state_dict({k[len("features."):]: v for k, v in alexnet.items() if k.startswith("features.")})
        linear = torch.load(_fetch(LINEAR_URL, LPIPS_DIR / "lpips_v0.1_alex.pth"), map_location="cpu", weights_only=True)
        self.weights = nn.ParameterList([nn.Parameter(linear[f"lin{i}.model.1.weight"], requires_grad=False)
                                         for i in range(len(self.TAPS))])
        self.register_buffer("shift", torch.tensor(self.SHIFT)[None, :, None, None])
        self.register_buffer("scale", torch.tensor(self.SCALE)[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def _taps(self, x):
        h, out = (2 * x - 1 - self.shift) / self.scale, []
        for index, layer in enumerate(self.features):
            h = layer(h)
            if index in self.TAPS:
                out.append(h / (h.pow(2).sum(1, keepdim=True).sqrt() + 1e-10))
        return out

    @torch.no_grad()
    def forward(self, prediction, target):
        total = 0
        for a, b, w in zip(self._taps(prediction.clamp(0, 1)), self._taps(target), self.weights):
            total = total + F.conv2d((a - b) ** 2, w).mean(dim=(1, 2, 3))
        return total


# ---------------------------------------------------------------- GMSD
_PREWITT = torch.tensor([[1.0, 0.0, -1.0]] * 3) / 3


@torch.no_grad()
def gmsd(prediction, target, c=170 / 255 ** 2):
    """GMSD per picture, (N,): luma (BT.709) averaged 2x2, Prewitt gradient magnitudes, their
    similarity (2 g1 g2 + c) / (g1² + g2² + c) at every pixel, and its standard deviation."""
    def magnitude(y):
        kx = _PREWITT.to(y)[None, None]
        return torch.sqrt(F.conv2d(y, kx) ** 2 + F.conv2d(y, kx.transpose(-1, -2)) ** 2 + 1e-12)

    luma = [F.avg_pool2d(metrics.rgb_to_yuv(x)[:, :1], 2) for x in (prediction.clamp(0, 1), target)]
    gp, gt = (magnitude(y) for y in luma)
    similarity = (2 * gp * gt + c) / (gp ** 2 + gt ** 2 + c)
    return similarity.flatten(1).std(dim=1, unbiased=False)
