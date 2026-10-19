"""How fast a model runs, and how much arithmetic it does."""

import time

import torch
import torch.nn as nn

# One frame at 60 fps and 30 fps, in milliseconds.
FRAME_60FPS_MS = 1000 / 60
FRAME_30FPS_MS = 1000 / 30

# Devices the client should run on, for the runtime estimate (`roofline_ms`): peak float32
# TFLOPS and memory bandwidth in GB/s, from the spec sheets.
HARDWARE = {
    "Intel UHD 620 (laptop iGPU)": (0.42, 38),
    "Intel Iris Xe, 96 EU (laptop iGPU)": (2.0, 68),
    "Apple M1, 8-core GPU": (2.6, 68),
    "NVIDIA GTX 1650": (3.0, 128),
    "NVIDIA RTX 3060": (12.7, 360),
    "NVIDIA RTX 3090": (35.6, 936),
}
THIS_GPU = "NVIDIA RTX 3090"   # this computer's GPU, which calibrates the estimate


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


def roofline_ms(work, tflops, gbps):
    """The least time a device can take on `work` ({"operations", "bytes"}): the slower of
    its arithmetic and its memory traffic. Returns (ms, "compute" or "memory")."""
    compute = work["operations"] / (tflops * 1e12)
    memory = work["bytes"] / (gbps * 1e9)
    return max(compute, memory) * 1000, "compute" if compute >= memory else "memory"


@torch.no_grad()
def macs_per_lr_pixel(model, size=32):
    """Multiply-accumulates per input pixel, counted from the convolutions that run."""
    total = 0

    def count(module, inputs, output):
        nonlocal total
        kernel = module.kernel_size[0] * module.kernel_size[1]
        pixels = output.shape[2] * output.shape[3]
        total += (module.in_channels // module.groups) * module.out_channels * kernel * pixels

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, nn.Conv2d)]
    device = next(model.parameters()).device
    model(torch.zeros(1, 3, size, size, device=device))
    for hook in hooks:
        hook.remove()
    return total / (size * size)
