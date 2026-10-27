"""How fast a model runs, and how much arithmetic it does.

The timing is `../../upscale/helpers/speed.py`'s, copied; the operation count is redone for a
model of two input pictures.
"""

import time

import torch

# One frame at 60 fps and 30 fps, in milliseconds.
FRAME_60FPS_MS = 1000 / 60
FRAME_30FPS_MS = 1000 / 30


def time_ms(run, seconds=0.5, warmup=2):
    """Average milliseconds per call of `run()`.

    It is called a few times first (warm-up), then as often as fits in `seconds`. On a GPU
    we wait for the work to finish before reading the clock.
    """
    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(warmup):
        run()
    sync()
    calls, started = 0, time.perf_counter()
    while time.perf_counter() - started < seconds or calls < 3:
        run()
        calls += 1
    sync()
    return (time.perf_counter() - started) / calls * 1000


def frame_budget(ms):
    """Which frame rate a per-frame time fits into."""
    if ms <= FRAME_60FPS_MS:
        return "60 fps"
    if ms <= FRAME_30FPS_MS:
        return "30 fps"
    return "under 30 fps"


@torch.no_grad()
def operations_per_pixel(model, size=64):
    """Operations per input pixel (a multiply-add is 2, a bias 1), counted from the
    convolutions a two-input model runs on a `size` x `size` picture."""
    total = 0

    def count(module, inputs, output):
        nonlocal total
        kernel = module.kernel_size[0] * module.kernel_size[1]
        outputs = output.shape[1] * output.shape[2] * output.shape[3]
        total += (2 * (module.in_channels // module.groups) * kernel + (module.bias is not None)) * outputs

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    device = next(model.parameters()).device
    picture = torch.zeros(1, 3, size, size, device=device)
    model(picture, picture)
    for hook in hooks:
        hook.remove()
    return total / (size * size)
