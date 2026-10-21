"""How an upscaled frame is scored.

Plain PSNR over RGB counts every colour channel the same. The eye does not: it sees detail
in brightness (luma, Y) much better than in colour (chroma, U and V). That is why video
codecs keep only a quarter of the colour samples and nobody notices.

So we convert to Y'CbCr (BT.709, full range) and weight the errors 6:1:1 - a colour error
counts a sixth of a brightness error of the same size. That is `wpsnr`, the main score.

`report` returns every score at once:

- `wpsnr` - the weighted PSNR above (dB, higher is better);
- `y`, `u`, `v` - PSNR of each channel on its own;
- `rgb` - plain RGB PSNR, for comparing with older numbers (a different scale - do not
  subtract it from `wpsnr`);
- `ssim` - structural similarity of luma (0..1). PSNR rewards blur; SSIM does less so.
"""

import torch
import torch.nn.functional as F

# RGB -> Y'CbCr, BT.709 full range. U and V span [-0.5, 0.5], the same width as Y.
RGB_TO_YUV = torch.tensor([
    [0.2126, 0.7152, 0.0722],
    [-0.2126 / 1.8556, -0.7152 / 1.8556, 0.9278 / 1.8556],
    [0.7874 / 1.5748, -0.7152 / 1.5748, -0.0722 / 1.5748],
])
# computed instead of copied from a table, so a round trip is exact
YUV_TO_RGB = torch.linalg.inv(RGB_TO_YUV.double()).float()

WEIGHTS = (6.0, 1.0, 1.0)   # Y : U : V
CHANNELS = ("y", "u", "v")


def rgb_to_yuv(x):
    """(N, 3, H, W) RGB in [0, 1] -> Y in [0, 1], U and V in [-0.5, 0.5]."""
    return torch.einsum("ij,njhw->nihw", RGB_TO_YUV.to(x.device, x.dtype), x)


def yuv_to_rgb(x):
    return torch.einsum("ij,njhw->nihw", YUV_TO_RGB.to(x.device, x.dtype), x)


def _yuv(prediction, target):
    # values outside [0, 1] cannot be displayed, so clamp before scoring
    return rgb_to_yuv(prediction.clamp(0, 1)), rgb_to_yuv(target)


def _mse(prediction, target, subsample=1):
    """Mean squared error per image and channel. `subsample=2` averages 2x2 first, which
    measures chroma at the resolution 4:2:0 video actually sends."""
    if subsample > 1:
        prediction = F.avg_pool2d(prediction, subsample, ceil_mode=True)
        target = F.avg_pool2d(target, subsample, ceil_mode=True)
    return ((prediction - target) ** 2).mean(dim=(2, 3))


def _db(mse):
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def psnr(prediction, target):
    """Plain RGB PSNR, per image, then averaged."""
    mse = F.mse_loss(prediction.clamp(0, 1), target, reduction="none").mean(dim=(1, 2, 3))
    return _db(mse).mean().item()


def channel_psnr(prediction, target, chroma_subsample=1):
    """PSNR of Y, U and V separately."""
    yuv_prediction, yuv_target = _yuv(prediction, target)
    luma = _db(_mse(yuv_prediction[:, :1], yuv_target[:, :1])).mean(dim=0)
    chroma = _db(_mse(yuv_prediction[:, 1:], yuv_target[:, 1:], chroma_subsample)).mean(dim=0)
    return {"y": luma[0].item(), "u": chroma[0].item(), "v": chroma[1].item()}


def wpsnr(prediction, target, weights=WEIGHTS, chroma_subsample=1):
    """The main score: weight the Y, U, V errors 6:1:1, then convert to dB."""
    yuv_prediction, yuv_target = _yuv(prediction, target)
    luma = _mse(yuv_prediction[:, :1], yuv_target[:, :1])
    chroma = _mse(yuv_prediction[:, 1:], yuv_target[:, 1:], chroma_subsample)
    weight = torch.tensor(weights, device=luma.device, dtype=luma.dtype)
    mse = (torch.cat([luma, chroma], dim=1) * weight).sum(dim=1) / weight.sum()
    return _db(mse).mean().item()


def wpsnr_db(prediction, target, weights=WEIGHTS, chroma_subsample=1):
    """The convention in published papers: average the three dB numbers 6:1:1 instead.
    One very accurate channel lifts this one more than `wpsnr`."""
    scores = channel_psnr(prediction, target, chroma_subsample)
    return sum(w * scores[c] for w, c in zip(weights, CHANNELS)) / sum(weights)


def ssim_y(prediction, target, size=11, sigma=1.5):
    """SSIM on luma with the standard 11x11 Gaussian window, no padding. 1.0 is identical."""
    yuv_prediction, yuv_target = _yuv(prediction, target)
    return _ssim(yuv_prediction[:, :1], yuv_target[:, :1], size, sigma).mean().item()


def _ssim(a, b, size=11, sigma=1.5):
    """SSIM of (N, 1, H, W) images, one per image, as a tensor a loss can follow."""
    coords = torch.arange(size, device=a.device, dtype=a.dtype) - (size - 1) / 2
    window = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = window / window.sum()
    window = (window[:, None] * window[None, :])[None, None]

    def blur(x):
        return F.conv2d(x, window)

    mean_a, mean_b = blur(a), blur(b)
    variance_a = blur(a * a) - mean_a ** 2
    variance_b = blur(b * b) - mean_b ** 2
    covariance = blur(a * b) - mean_a * mean_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
    denominator = (mean_a ** 2 + mean_b ** 2 + c1) * (variance_a + variance_b + c2)
    return (numerator / denominator).mean(dim=(1, 2, 3))


def luma_chroma_l1(prediction, target, weights=WEIGHTS):
    """A training loss that matches `wpsnr`: L1 in Y'CbCr, weighted 6:1:1.

    Not clamped (unlike the scores): clamping would stop the gradient for exactly the
    pixels that are out of range and need pulling back.
    """
    difference = (rgb_to_yuv(prediction) - rgb_to_yuv(target)).abs().mean(dim=(0, 2, 3))
    weight = torch.tensor(weights, device=difference.device, dtype=difference.dtype)
    return (difference * weight).sum() / weight.sum()


def ssim_l1_loss(prediction, target, alpha=0.84):
    """A training loss that rewards structure, not only pixel values: `alpha` x (1 - SSIM on
    luma) + (1 - `alpha`) x L1 on RGB.

    SSIM compares the edges and contrast in every 11x11 window, so a blurred edge costs more
    than under L1; L1 keeps the colours and the flat areas right, which SSIM barely sees.
    The mix and its 0.84 are from Zhao et al. (2017), who found it beat either alone - with
    multi-scale SSIM, which needs a larger picture than these 128 px patches. Not clamped,
    like `luma_chroma_l1`.
    """
    structure = 1 - _ssim(rgb_to_yuv(prediction)[:, :1], rgb_to_yuv(target)[:, :1]).mean()
    return alpha * structure + (1 - alpha) * F.l1_loss(prediction, target)


def per_image(prediction, target):
    """Luma SSIM and luma PSNR of every image on its own, as (N,) tensors - for choosing
    between models image by image. Their means are `report`'s "ssim" and "y"."""
    yuv_prediction, yuv_target = _yuv(prediction, target)
    luma_prediction, luma_target = yuv_prediction[:, :1], yuv_target[:, :1]
    return {"ssim": _ssim(luma_prediction, luma_target), "y": _db(_mse(luma_prediction, luma_target))[:, 0]}


def report(prediction, target, chroma_subsample=1):
    """Every score for one batch, in one dictionary."""
    return {"wpsnr": wpsnr(prediction, target, chroma_subsample=chroma_subsample),
            **channel_psnr(prediction, target, chroma_subsample),
            "rgb": psnr(prediction, target),
            "ssim": ssim_y(prediction, target)}


HEADER = f"{'wpsnr':>9}{'psnr-y':>9}{'psnr-u':>9}{'psnr-v':>9}{'psnr-rgb':>10}{'ssim-y':>9}"


def row(scores):
    """One `report` as a line under `HEADER`."""
    return (f"{scores['wpsnr']:9.2f}{scores['y']:9.2f}{scores['u']:9.2f}"
            f"{scores['v']:9.2f}{scores['rgb']:10.2f}{scores['ssim']:9.4f}")


def print_table(results, extra=None):
    """Print {name: report} as a table. `extra(name, scores)` may add a last column."""
    width = max(16, max(len(name) for name in results) + 2)
    print(f"{'':<{width}}{HEADER}")
    for name, scores in results.items():
        print(f"{name:<{width}}{row(scores)}{extra(name, scores) if extra else ''}")
