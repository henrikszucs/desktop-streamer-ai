"""How a reconstructed frame is scored, weighted the way an eye is.

PSNR over RGB counts a red error and a green error the same, and both the same as a
brightness error. An eye does not. Luminance carries the detail it resolves - edges, text,
the structure it judges sharpness by - and chrominance carries far less: colour-opponent
vision falls off at a fraction of the spatial frequency luminance vision reaches, which is
why every video codec in use subsamples chroma to a quarter of the samples (4:2:0) and
nobody notices, and why nobody would dare do that to luma.

So the measurement here works in Y'CbCr rather than RGB, reports the three channels apart,
and weights them 6:1:1 when it puts them back together - the weighting JVET uses, for the
same reason. Two consequences worth stating before any number is read:

- A model that spends capacity on colour accuracy it cannot be seen to have is now
  measured as spending it, and a model that blurs an edge to keep a hue is now punished
  for it. That is the point of the change.
- These dB are not the RGB dB. Y carries most of the energy and most of the error, so a
  weighted score sits close to the luma score and away from the RGB one. `report` returns
  the RGB number too, so older tables stay readable beside newer ones, but the two
  columns are different measurements and must not be subtracted from each other.

Two weightings exist and they are not the same function:

- `wpsnr` weights the errors, then takes one logarithm: a chroma error counts one sixth
  of a luma error of the same size. That is the perceptual claim stated directly, and it
  is what this repo reports.
- `wpsnr_db` averages the three dB figures 6:1:1, which is the convention published
  numbers use. Averaging logarithms is a geometric mean of the errors, so a channel that
  is very accurate pulls the whole score up in a way weighting the errors does not allow.
  It is here for comparability with the literature, not as the headline.

`ssim_y` sits beside both because PSNR rewards blur: an upscaler that hedges between two
plausible edges beats one that commits to the wrong one, on any pure error metric. SSIM
compares local structure instead, on luma alone - the only channel whose structure an eye
is really reading.

BT.709 primaries, full range (0-255 mapped to [0, 1], no 16-235 headroom): these frames
come from a desktop capture path, which is what a screen is authored in.
"""

# third-party
import torch
import torch.nn.functional as F

# BT.709 full range. Y = 0.2126 R + 0.7152 G + 0.0722 B, and the two chroma channels are
# the blue and red differences normalised so each spans [-0.5, 0.5] - the same width as Y,
# which is what lets one peak value serve all three in a PSNR.
RGB_TO_YUV = torch.tensor([
    [0.2126, 0.7152, 0.0722],
    [-0.2126 / 1.8556, -0.7152 / 1.8556, 0.9278 / 1.8556],
    [0.7874 / 1.5748, -0.7152 / 1.5748, -0.0722 / 1.5748],
])

# Inverted rather than quoted: the published inverse coefficients are rounded to four
# decimals, which puts 2e-05 of drift into a round trip that ought to be free. Two of
# these matrices become fixed 1x1 convolutions in a model, so the round trip is a real
# path through a graph and not just a property of this file.
YUV_TO_RGB = torch.linalg.inv(RGB_TO_YUV.double()).float()

# Luma against each chroma channel. 6:1:1 is the JVET weighting; the ratio is the one
# thing here worth arguing about, and it is a constant so that arguing with it is one edit.
WEIGHTS = (6.0, 1.0, 1.0)

CHANNELS = ("y", "u", "v")


def rgb_to_yuv(x):
    """(N, 3, H, W) in [0, 1] -> Y in [0, 1], U and V in [-0.5, 0.5]."""
    return torch.einsum("ij,njhw->nihw", RGB_TO_YUV.to(x.device, x.dtype), x)


def yuv_to_rgb(x):
    """The inverse, exactly - the same matrix, inverted once and written down."""
    return torch.einsum("ij,njhw->nihw", YUV_TO_RGB.to(x.device, x.dtype), x)


def _prepare(prediction, target):
    """Clamp the prediction to the display range, then measure in Y'CbCr.

    Clamping first is deliberate: a value outside [0, 1] cannot be shown, so an error that
    exists only before the clamp is an error nobody sees, and counting it would score a
    model for what its own output clipping already fixes.
    """
    return rgb_to_yuv(prediction.clamp(0, 1)), rgb_to_yuv(target)


def _mse(prediction, target, subsample=1):
    """Per-image, per-channel MSE. `subsample` box-averages the channels first.

    Passing 2 measures chroma the way it is transmitted - a 2x2 average, 4:2:0 - so an
    error finer than the chroma sampling grid is not counted at full weight when it would
    not survive the encoder that follows anyway.
    """
    if subsample > 1:
        prediction = F.avg_pool2d(prediction, subsample, ceil_mode=True)
        target = F.avg_pool2d(target, subsample, ceil_mode=True)
    return ((prediction - target) ** 2).mean(dim=(2, 3))


def _db(mse):
    """MSE -> dB at peak 1.0, floored so an exact match is a number rather than an error."""
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def psnr(prediction, target):
    """Plain RGB PSNR, per image then averaged - the measurement everything older used."""
    mse = F.mse_loss(prediction.clamp(0, 1), target, reduction="none").mean(dim=(1, 2, 3))
    return _db(mse).mean().item()


def channel_psnr(prediction, target, chroma_subsample=1):
    """PSNR of Y, U and V separately, each averaged over the batch."""
    yuv_prediction, yuv_target = _prepare(prediction, target)
    luma = _db(_mse(yuv_prediction[:, :1], yuv_target[:, :1])).mean(dim=0)
    chroma = _db(_mse(yuv_prediction[:, 1:], yuv_target[:, 1:], chroma_subsample)).mean(dim=0)
    return {"y": luma[0].item(), "u": chroma[0].item(), "v": chroma[1].item()}


def psnr_y(prediction, target):
    """Luma only - what most super-resolution work reports, and where the detail lives."""
    return channel_psnr(prediction, target)["y"]


def wpsnr(prediction, target, weights=WEIGHTS, chroma_subsample=1):
    """The headline: weight the errors 6:1:1, then take one logarithm.

    A chroma error counts a sixth of a luma error of the same size, which is the eye
    claim written as arithmetic. Because the weighting happens before the log, a model
    cannot buy the score back with chroma it happened to get very right.
    """
    yuv_prediction, yuv_target = _prepare(prediction, target)
    luma = _mse(yuv_prediction[:, :1], yuv_target[:, :1])
    chroma = _mse(yuv_prediction[:, 1:], yuv_target[:, 1:], chroma_subsample)
    weight = torch.tensor(weights, device=luma.device, dtype=luma.dtype)
    mse = (torch.cat([luma, chroma], dim=1) * weight).sum(dim=1) / weight.sum()
    return _db(mse).mean().item()


def wpsnr_db(prediction, target, weights=WEIGHTS, chroma_subsample=1):
    """The published convention: average the three dB figures 6:1:1 rather than the errors."""
    scores = channel_psnr(prediction, target, chroma_subsample)
    return sum(w * scores[c] for w, c in zip(weights, CHANNELS)) / sum(weights)


def _gaussian(size, sigma, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    window = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = window / window.sum()
    return (window[:, None] * window[None, :])[None, None]


def ssim_y(prediction, target, size=11, sigma=1.5):
    """SSIM on luma - structure rather than error, because PSNR rewards blur.

    The standard 11x11 Gaussian window, the standard stabilising constants, and no
    padding, so every statistic is taken over real pixels. A number in [0, 1].
    """
    yuv_prediction, yuv_target = _prepare(prediction, target)
    a, b = yuv_prediction[:, :1], yuv_target[:, :1]
    window = _gaussian(size, sigma, a.device, a.dtype)

    blur = lambda x: F.conv2d(x, window)
    mean_a, mean_b = blur(a), blur(b)
    variance_a = blur(a * a) - mean_a ** 2
    variance_b = blur(b * b) - mean_b ** 2
    covariance = blur(a * b) - mean_a * mean_b

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
    denominator = (mean_a ** 2 + mean_b ** 2 + c1) * (variance_a + variance_b + c2)
    return (numerator / denominator).mean(dim=(1, 2, 3)).mean().item()


def luma_chroma_l1(prediction, target, weights=WEIGHTS):
    """L1 in Y'CbCr under the metric's own weighting - a loss that optimises what is scored.

    Weighted on the error rather than on the channel, so it stays an L1: the gradient of a
    chroma pixel is a sixth of the gradient of a luma pixel and nothing else changes.
    Unclamped, unlike the metric - clamping would zero the gradient of exactly the pixels
    that have left the range and most need pulling back into it.
    """
    difference = (rgb_to_yuv(prediction) - rgb_to_yuv(target)).abs().mean(dim=(0, 2, 3))
    weight = torch.tensor(weights, device=difference.device, dtype=difference.dtype)
    return (difference * weight).sum() / weight.sum()


def report(prediction, target, chroma_subsample=1):
    """Every number this module has, for one batch, in one dictionary."""
    return {"wpsnr": wpsnr(prediction, target, chroma_subsample=chroma_subsample),
            **channel_psnr(prediction, target, chroma_subsample),
            "rgb": psnr(prediction, target),
            "ssim": ssim_y(prediction, target)}


COLUMNS = ("wpsnr", "y", "u", "v", "rgb", "ssim")
HEADER = f"{'wpsnr':>9}{'psnr-y':>9}{'psnr-u':>9}{'psnr-v':>9}{'psnr-rgb':>10}{'ssim-y':>9}"


def row(scores):
    """One `report` as a line under `HEADER`."""
    return (f"{scores['wpsnr']:9.2f}{scores['y']:9.2f}{scores['u']:9.2f}"
            f"{scores['v']:9.2f}{scores['rgb']:10.2f}{scores['ssim']:9.4f}")
