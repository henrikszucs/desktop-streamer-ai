# The 60 fps reference

The upscaler has to keep up with **60 frames a second on a GTX 1660 Ti with 20% to spare**.
That card at 80% of its speed is the reference: the weakest GPU a 60 fps stream should be
upscaled on. Every number here was measured on that card, a laptop GTX 1660 Ti (TU116,
80 W cap, 1.7-1.9 GHz under load, 6 GB) in Chrome 157 with ONNX Runtime Web 1.29.0, on
2026-11-12.

## The budget

A frame at 60 fps is 16.67 ms. A card 20% slower takes 1.25× as long, so on the 1660 Ti a
frame has to cost at most:

> **13.3 ms per shown frame**, from the decoded frame arriving to its upscaled picture done
> on the GPU: the enhancer's own `enhance.ms` reading, with upscale on.

The client spends some of that around the model: cutting the frame into tiles, drawing the
2× picture, handing it to the drawer, and the wait for the GPU. That measured **2.2-4.1 ms**
over the graph alone. With the larger figure set aside:

> **The model gets 9.2 ms a frame**, timed the way the benchmark page times it: the graph
> alone, float16 with float32 at both ends, batch = the stream's tile count, on the 1660 Ti.

| Stream | Tiles | Output | Model budget per tile | The told 24×2 model (fp16) per tile | Speed-up it needs |
|---|---|---|---|---|---|
| 640×360 | 4 | 1280×720 | 2.30 ms | 2.10 ms | **fits** (12.2 ms a frame in the client) |
| 854×480 | 9 | 1708×960 | 1.02 ms | 2.08 ms | 2.0× |
| 1280×720 | 16 | 2560×1440 | 0.58 ms | 2.16 ms | 3.8× |
| 1920×1080 | 36 | 3840×2160 | 0.26 ms | 2.07 ms | 8.1× |

A tile is 328×188 in (a 320×180 step and a 4 px halo), 656×376 out.

## What the told model does today

Float16 (`dist/upscale/client/upscale_fp16.onnx`, which the client runs on a GPU with
`shader-f16`):

| Stream | Tiles | Graph alone (benchmark page) | In the client (`enhance.ms`) | Client over the graph | Kernels in the client | GPU clock |
|---|---|---|---|---|---|---|
| 640×360 | 4 | 8.4 ms | **12.2 ms** | 3.8 ms | 7.4 ms | 1860 MHz |
| 854×480 | 9 | 18.8 ms | 21.6 ms | 2.9 ms | 19.2 ms | 1800 MHz |
| 1280×720 | 16 | 34.6 ms | 36.8 ms | 2.2 ms | 31.3 ms | 1785 MHz |
| 1920×1080 | 36 | 74.5 ms | 78.6 ms | 4.1 ms | 71.8 ms | 1710 MHz (power-capped) |

Only a 640×360 stream upscaled to 1280×720 reaches 60 fps on the reference, with 1.1 ms to
spare.

## A model built for it (notebook 24) - predicted, not measured on the 1660 Ti

Notebook 24's told `q24` (`dist/upscale/client/upscale_fast.onnx` and `upscale_fast_fp16.onnx`)
is a drop-in for the told 24×2: the same inputs, output and 4 px halo. Its body runs at a
quarter of the pixels: a 4×4 stride-2 stem, one residual block, and a 1×1 tail writing each
2×2 cell's 16 luma values. Its colour comes from a frozen 4×4 stride-2 skip: bicubic without
its smallest outer taps, within 0.01 dB of bicubic's colour. Its 1660 Ti time below is
**predicted from the RTX 3090** (see the next section), because it has not run on the card
yet:

| Stream | Tiles | Graph alone, float16 | 3090, measured | Predicted on the 1660 Ti |
|---|---|---|---|---|
| 1280×720 | 16 | `upscale_fast_q24_frozen_skip_fp16_tile320x180_b16.onnx` | 1.35-1.41 ms (told 24×2: 6.65-6.95 in the same sessions) | **7.1-7.5 ms** (7.3-7.7 with the largest ratios; 7.0 scaled by the told graph's measured 34.6) |

That is inside the 9.2 ms budget, and inside the 8.3 ms that notebook 24 held it to while it
is only predicted. On the 9,628 validation patches it scores 0.9212 SSIM-Y and 34.47 dB luma,
with bicubic's colour. The told 24×2 scores 0.9289 and 35.07, and bicubic 0.9045 and 33.55,
so q24 keeps 69% of the told model's SSIM-Y gain. To confirm it, run "Checking a model
against it" below with the b16 graph above.

## What a layer costs

Float16, per 328×188 tile, at full clock: single-layer graphs with random weights, 16 tiles a
run, input and output resident on the GPU, the GPU time of the layer's own kernel. The 3×3
24→24 layer measured the same inside the told model (0.43 ms).

| Layer | ms per tile |
|---|---|
| 3×3 conv, 3 in (24 or 16 out) | 0.10-0.11 |
| 3×3 conv, 4 in | 0.13 |
| 3×3 conv, 8 in | 0.17-0.18 |
| 3×3 conv, 12 in | 0.21-0.22 |
| 3×3 conv, 16 in | 0.28-0.32 |
| 3×3 conv, 24 in | 0.39-0.44 |
| 3×3 conv, 32 in, 32 out | 0.57 |
| 1×1 conv, 16-24 channels | 0.07-0.08 |
| 3×3 depthwise, 16-24 channels | 0.09-0.10 |
| 5×5 conv, 3 in, 12 out (the bicubic skip) | 0.17 |
| `Add`, 24 channels / 12 channels | 0.034 / 0.019 |
| `Concat` to 12 channels, `DepthToSpace` ×2, the output `Transpose`, the two `Cast`s | 0.15 together |

- **A 3×3 convolution costs what its input width says, not its output width.** 24→4,
  24→12 and 24→24 all take about 0.4 ms. ORT's `Conv2DMatMul` is bound by reading the
  input patches (9 × input channels per pixel), so a cheap layer is a narrow input, and
  widening is done with a 1×1.
- **A whole graph costs its kernels plus about 12%**: dispatches and the gaps between
  them. The told model's kernels are 30.4 ms at 16 tiles; the benchmark page times 34.6.
- **The fixed part of the output is 0.15 ms a tile**, and the bicubic skip 0.17 more.

In kernel time, after the 12%:
- **1280×720 (0.58 ms a tile, so 0.52 of kernels):** the output path and skip take 0.32,
  which leaves 0.20 for the network. That is a 3×3 stem from the picture and one 1×1. With
  bicubic drawn in the client's shader instead of the graph (fix B3 in
  `bottlenecks_gtx1660ti.md`), it is 0.37: a stem, one 3×3 layer at 12 channels in, and a
  1×1 tail.
- **854×480 (1.02 ms a tile, so 0.91 of kernels):** about 0.59 for the network, or a stem,
  two 3×3 layers at 12 channels in and a 1×1 tail.
- **640×360:** the told 24×2 model fits as it is.

## Predicting the 1660 Ti from the RTX 3090

Notebook 24 measured every layer of the table above again on the RTX 3090 (2026-11-13,
Chrome 157, ORT Web 1.29.0, the same single-layer graphs, the layer's own kernels from the
benchmark page's `gpuByKernel`). The two cards do not differ by one factor:

| Kernel | 1660 Ti over 3090 | Rows behind it |
|---|---|---|
| `Conv2DMatMul` (3×3, 5×5) reading up to 24 channels | 4.0-4.8 | 3/4/8/12/16/24 in, the 5×5 skip |
| `Conv2DMatMul` reading 32 channels | 6.2 | 3×3 32→32: the 3090 pays 5% over 24 in, the 1660 Ti 37% |
| `MatMul` (a 1×1) | 6.8-7.0 | 1×1 16-24 |
| `GroupedConv` (depthwise) | 1.7-2.4 | depthwise 16-24 |
| `Add` | 3.3-3.6 | 12 and 24 channels |
| `Concat`, `DepthToSpace`, `Transpose`, `Cast` | 4.6 | the told graph's output path |

On the 3090 a graph's page time is its kernels plus 1%; on the 1660 Ti, plus 13%. So a graph's
1660 Ti page time is predicted as **1.13 × the sum of its 3090 kernels, each times its kind's
ratio**. With the typical ratios this puts the told 24×2 at 16 tiles at 36.4 ms; with the
largest ratio of each kind (and 6.2 for any convolution reading 28 or more channels), at
37.7 ms. It measured 34.6, so both estimates come out 5-9% high.

A prediction is not a measurement: a kernel the table does not have (a stride-2
convolution, a 3×3 over a quarter of the pixels) borrows its kind's ratio. Notebook 24 holds
a model to 8.3 ms predicted, 10% under the 9.2 ms, until it is measured on the card.

## Checking a model against it

1. Export it as notebook 23 does, then `webexport.to_float16(..., keep_io_types=True)`.
2. Run the benchmark page on the GTX 1660 Ti with a fixed-shape copy at the stream's batch,
   as Run 8 of `bottlenecks_gtx1660ti.md` makes them, using
   `?auto=<id>&backend=webgpu&data=gpu&mode=reuse&seconds=5`. The graph's time must be at most
   **9.2 ms**.
3. Check it in the client: `enhance.ms` at most **13.3 ms**.

On the RTX 3090 the same graph runs about 4× faster per tile in float32 (0.646 against
2.84 ms). That ratio is from one model and is not this reference. Use it only to discard a
candidate early, and measure on the 1660 Ti before calling one fast enough.

## How it was measured

- **The host sat on neither GPU.** The host was a second Chrome sharing a tab, started with
  `--disable-gpu`, so its rendering, capture and H.264 encoding ran on the CPU and the
  NVIDIA card did only the viewer's work. A host on the same GPU slowed the viewer's model
  by 13% at 30 fps and 20-30% at 60 fps, depending on how much its page draws. A real
  viewer does not pay that.
- **The source was sized with DevTools emulation** (1920×1080, or 640×360 for the
  smallest stream), so each resolution setting gave an exact 16:9 stream. A software host
  sends 20-45 frames a second rather than 60. That does not change what one frame costs,
  as long as the card stays at full clock, and the clock was checked during every run. A
  640×360 stream at 26-34 fps left the card at 1 GHz and read 12.6-17.8 ms; the same stream
  from a 640×360 source, at 40-47 fps, kept it at 1.86 GHz and read 12.2.
- **The graph alone was timed on the benchmark page** (`src/upscale/benchmark/`), with
  `mode=reuse`, GPU-resident tensors and 5 s per graph, on the same card with the host off
  it.
- **Kernel times inside the client** came from ORT's own profiling, switched on in the
  stream worker over the DevTools protocol. `bottlenecks_gtx1660ti.md` has the method and
  the earlier runs.
- **Not covered:** 60 fps made by frame generation from a 30 fps stream. Each 33.3 ms
  interval then holds two upscales and one interpolation, which leaves the same 13.3 ms
  per shown picture, less the interpolation's share. That is not measured here.
