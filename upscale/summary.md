# Upscale — summary

Reconstructing a higher resolution frame from the one the stream delivered, so a client can
be sent fewer pixels than it displays.

> **Status: a baseline and one learned model.** A static baseline (resampling filters, no
> model) and a small learned upscaler, both exported to ONNX and both measurable in the
> browser through the benchmark page. All measured on synthetic frames, all on CPU. Nothing
> in `src/` calls any of this yet.

**One notebook is one model.** A notebook owns its architecture, its training, its numbers
and its export cell, and publishes straight into `benchmark/www/models/` — so a model is
added or dropped by adding or deleting one file, and the page shows whatever is published.

## Notebooks

Run in order, from this folder:

- `data_preprocess.ipynb` — source frames from `data/raw/` (or synthesised desktop-like
  frames when it is empty) → bicubic ×2 degradation → aligned LR/HR patch pairs, flat
  patches dropped by luma variance, split by *source frame* rather than by patch → four
  `uint8` `.npy` arrays and a `manifest.json` in `data/processed/`.
- `upscale_dummy.ipynb` — **static methods only.** Seven of them: nearest, box, bilinear,
  hamming, bicubic, Lanczos, and bicubic followed by an unsharp mask. No training; it
  measures quality and runtime, hands out the bar, and exports the three ONNX can express
  as `Resize` so the bar exists in the browser too.
- `upscale_nn.ipynb` — **the PyTorch attempt.** Trains `ResidualUpscaler` on the patch
  pairs, measures it the same way, writes `checkpoints/upscale_nn.pt`, and exports it —
  `Resize` and all, which is the point of having it on the page.
Each model notebook owns its own export cell and writes its `.onnx` into
`benchmark/www/models/` through `webexport.py`. That is all publishing is: the benchmark
server lists whatever graphs are in that folder, per request, so there is no manifest for a
re-export to leave stale. Four models reach the page: the three static filters and the
bicubic-residual model.

`data/`, `checkpoints/` and `benchmark/www/models/` are gitignored. The export cells need
`onnx` and `onnxruntime`, which are project dependencies.

Every number below is CPU, 12 threads. The installed torch is a CPU build (`2.13.0+cpu`)
even though `pyproject.toml` declares a CUDA index, so nothing here has been measured on a
GPU. Runtimes are forward pass only — no decode, no copy, no PNG encoding.

## Static methods (`upscale_dummy.ipynb`)

### Quality — 76 validation patches, 64px → 128px

| Method | PSNR | vs nearest |
| --- | --- | --- |
| bicubic + unsharp | **23.62 dB** | +1.35 dB |
| lanczos | 23.41 dB | +1.14 dB |
| bicubic | 23.33 dB | +1.06 dB |
| bilinear | 22.74 dB | +0.47 dB |
| hamming | 22.66 dB | +0.39 dB |
| nearest | 22.27 dB | — |
| box | 22.27 dB | +0.00 dB |

Box ties nearest exactly, which is right rather than broken: box averaging over an *integer*
upscale has one source pixel per output pixel, so it is replication. Sharpening winning on
PSNR is the result to distrust — this dataset's degradation is a clean bicubic blur with
nothing else in it, and an unsharp mask partly inverts exactly that. On real captures it
would sharpen the codec artifacts too.

### Runtime — batch of one

| Output | nearest | bilinear | bicubic (torch) | bicubic (PIL) | lanczos (PIL) | +unsharp (PIL) |
| --- | --- | --- | --- | --- | --- | --- |
| 960×540 | 0.25 ms | 0.95 ms | 3.40 ms | 3.42 ms | 4.64 ms | 15.92 ms |
| 1280×720 | 0.42 ms | 1.66 ms | 6.02 ms | 6.33 ms | 8.63 ms | 28.64 ms |
| 1920×1080 | 0.97 ms | 3.77 ms | 13.01 ms | 14.27 ms | 19.19 ms | 64.71 ms |
| 2560×1440 | 1.91 ms | 6.67 ms | 22.60 ms | 25.63 ms | 34.24 ms | 116.43 ms |

Every plain filter clears 60 fps up to 1080p on the CPU alone. The sharpening is the
expensive part — the unsharp mask costs about 4.5× the bicubic upscale it follows, which
puts the best-scoring static method outside the 30 fps budget at 1080p.

### Runtime — batch sweep, 320×180 → 640×360

| Batch | nearest ms/frame | bilinear ms/frame | bicubic ms/frame |
| --- | --- | --- | --- |
| 1 | 0.12 | 0.43 | 1.56 |
| 4 | 0.11 | 0.43 | 1.46 |
| 16 | 0.12 | 0.43 | 1.41 |

Flat. Batching a static filter buys nothing measurable (0.98–1.11× by batch 16) — there is
no per-call overhead worth amortising.

### The `Resize` kernel is not the filter

Every runtime number above is `torch`/PIL. The benchmark page runs the *exported graph*
through ONNX Runtime instead, and for cubic those are not the same kernel — on one 132×132
tile, in one process, on the same input:

| Tile 132×132 → 264×264 | nearest | bilinear | bicubic |
| --- | --- | --- | --- |
| `F.interpolate` (torch) | 0.08 ms | 0.45 ms | 1.11 ms |
| ONNX Runtime, CPU provider | 0.04 ms | 0.34 ms | **48.10 ms** |
| ×40 tiles, one 1080p frame | 1.6 ms | 13.6 ms | **1924 ms** |

Nearest and bilinear are fine — faster than torch, in fact. Cubic is about **65× slower**,
and it is a property of that one kernel rather than of the extra taps:

- a flat **~230 ns per output pixel**, scaling linearly from a 32×32 tile to 264×264, where
  the same runtime's `linear` mode costs ~1 ns per output pixel;
- **single-threaded** whatever `intra_op_num_threads` is set to, while every other number
  here is 12 threads;
- unmoved by `coordinate_transformation_mode`, by `exclude_outside`, and by whether the
  input shape is pinned or dynamic — all four measure the same.

Four taps against sixteen is 4×, not 250×. So this is one unoptimised implementation, and
the static bicubic baseline is the shape that lands on it.

Two things follow. **A browser number for `static bicubic` is not a number for bicubic** —
it is a number for that kernel on that provider, and the WebGPU one is a different kernel
again. And **nothing would ship this graph**: a browser upscaling a frame with a static
filter uses `drawImage` on a canvas or a texture sampler, which does bilinear and bicubic
in the fixed-function path for free. The `Resize` graphs exist so that one runtime measures
both the bar and the models, which is the only way those two numbers are comparable — they
are the cost of the bar *through ONNX Runtime*, not the cost of the bar.

## Learned model (`upscale_nn.ipynb`)

**`ResidualUpscaler`** — 26,796 parameters (0.11 MB float32), fixed at ×2, ESPCN-shaped:
two convolutions at *low* resolution, then one `PixelShuffle`. Upscaling last is what keeps
it cheap enough to matter for a live stream. It predicts a residual on top of bicubic and
its last convolution starts at zero, so an untrained model *is* the bicubic baseline and
every reported gain is honest.

Trained on 12 synthetic frames (268 train / 76 validation patches), 12 epochs, L1 loss,
Adam with a cosine schedule. About 9 s on CPU.

### Quality

| Measure | Bicubic | Model | Gain |
| --- | --- | --- | --- |
| Validation patches, PSNR | 23.41 dB | **24.17 dB** | +0.76 dB |
| Whole frame, PSNR | 23.88 dB | **24.29 dB** | +0.41 dB |

### Runtime — batch of one

| Output | ms/frame | fps | Frame budget |
| --- | --- | --- | --- |
| 960×540 | 24.1 | 41.5 | clears 30 fps |
| 1280×720 | 44.7 | 22.4 | misses 30 fps |
| 1920×1080 | 102.6 | 9.7 | misses 30 fps |
| 2560×1440 | 199.1 | 5.0 | misses 30 fps |

### Runtime — batch sweep, 320×180 → 640×360

| Batch | ms/batch | ms/frame | fps | vs batch 1 |
| --- | --- | --- | --- | --- |
| 1 | 10.0 | 10.00 | 100.0 | 1.00× |
| 4 | 43.4 | 10.86 | 92.1 | 0.92× |
| 16 | 195.6 | 12.23 | 81.8 | 0.82× |

Batching costs here rather than paying: a CPU already spreading one frame across twelve
threads has no idle width for a second, so a batch adds memory traffic and the latency of
waiting for it. Worth rechecking on a GPU, where the opposite is the usual result.

## Tiling geometry

Inference on the page runs on fixed tiles, not on frames: one pinned input size, one
session, one reused buffer, whatever resolution the stream turns out to be. A tile has
**three** sizes, and confusing them is the usual bug — here for a model with a 6 pixel halo:

| | LR | HR |
| --- | --- | --- |
| **step** — what the tile contributes to the output | 128 | 256 |
| **model input** — step plus a halo each side | 140 | 280 |
| **kept** — the centre, after the halo is cropped | 128 | **256** |

**The halo is the whole game.** A convolution at a tile edge has no neighbours, pads with
zeros, and invents an edge; assembled, those invented edges line up into a visible grid.
Feed each tile a margin of real pixels and throw it away afterwards, and the margin needed
is exactly the receptive field — **one pixel per 3×3 convolution in the path**. Measured on
a depth-4 network, tiled against whole-frame, the error drops to floating point noise
(2.4e-07) the moment the halo reaches the receptive field, and not one pixel before. So
tiling is not an approximation: with the right halo it is *bit-identical* to upscaling the
whole frame, no seam blending required. That makes the halo a property of the architecture
rather than a tuning knob, and each notebook states its own — 0/1/2 for the static filters,
4 for `upscale_nn`.

(A residual error remains at the true frame border at any halo. That is not the tiling: a
convolution at the real border pads with zeros after *every* layer, while a pre-padded halo
turns those zeros into bias-driven values at the first. Whole-frame and tiled inference just
handle the outer border differently, and it never reaches the interior.)

`webexport.TILE_STEP` is **128 in → 256 out**. A sweep from 64 to 320 found the cost per
1080p frame flat and unstable enough not to decide the design, so 128 wins on everything
else: 84% of computed pixels kept at halo 6, a power of two once the pixels live in textures
and workgroups, bounded memory independent of stream resolution, and 40 tiles per 1080p
frame so the main thread yields 40 times instead of blocking once. That sweep was run
against a model that has since been removed, so the timings behind it are gone; the geometry
argument is not.

## The browser benchmark (`benchmark/`)

A page that times a tile through a model in the browser it will actually run in. Speed only
— it never looks at the picture.

```
uv run upscale/benchmark/main.py   # from model/, then open http://127.0.0.1:8000
```

`main.py` serves `www/` and answers `GET /api/models`; the page cannot run from `file://`
because ONNX Runtime Web fetches its own `.wasm` at runtime. **The dropdown is built from the
folder, not from a file describing it.** Every request walks `www/models/*.onnx` and reads
each graph — shapes, operators, parameter count, size, and the halo, which is whatever the
input carries beyond its step — parsing a model again only when its mtime or size has
changed. The label and the desktop `cpu_ms` are the only two things a graph cannot state, and
they ride in its `metadata_props`, so a model is one self-describing file: drop one in and it
is in the dropdown on the next reload, delete one and it is gone. A graph with no metadata
still lists, under a label derived from its filename.

Pick a model, pick a backend (auto / WebGPU / WASM), pick where the tile lives, press the
button. Three details it gets right, each of which it got wrong first:

- **The tile stays on the GPU.** A WebGPU run fed a JS array uploads the tile and reads the
  result back *every run* — 209 KB up and 836 KB down for a 132 tile, through a fence that
  stalls the pipeline. That cost is identical whatever the graph does, so it hides the model
  completely. GPU-resident I/O (`Tensor.fromGpuBuffer` in, `preferredOutputLocation:
  "gpu-buffer"` out) is what makes the number a number about the model. The page keeps the
  round trip as an option, because the size of the difference is worth seeing.
- **A sample is a batch, and the batch is sized from a probe.** `performance.now()` is
  deliberately coarse in a browser and one 140×140 tile finishes under its resolution, so
  runs are timed in batches. The size cannot be a constant: settling a batch takes one
  fence, and **one fence costs ~3.4 ms of latency here whatever it is waiting for**. Spread
  over a fixed 20 runs that is 0.17 ms added to every run — twice the cost of the cheapest
  model on the page. Each batch is now sized to fill ~200 ms, which puts its fence under 2%.
- **The backend column is the one that resolved, not the one requested.** Asking for
  `["webgpu", "wasm"]` lets the runtime fall back silently, and a row that cannot say which
  backend produced it is not a result — so "auto" tries WebGPU alone first and reports what
  it got. The data column resolves the same way: WASM has no GPU buffers.

Four models reach the dropdown, and the tile geometry differs per model because the halo
does — nearest needs none, bilinear 1, bicubic 2, the residual network 4:

| Source | Models | Model input | Halo |
| --- | --- | --- | --- |
| `upscale_dummy` | static nearest / bilinear / bicubic | 128 / 130 / 132 | 0 / 1 / 2 |
| `upscale_nn` | nn bicubic residual | 136 | 4 |

### Results — NVIDIA Ampere, Chromium, ms per tile

| Model | WebGPU, GPU-resident | WebGPU, CPU round trip | WASM | 1080p frame (best) |
| --- | --- | --- | --- | --- |
| static nearest | **0.068** | 3.82 | 0.20 | 2.7 ms |
| static bilinear | **0.083** | 3.65 | 0.63 | 3.3 ms |
| static bicubic | **0.066** | 3.75 | 20.49 | 2.6 ms |
| nn bicubic residual | **0.510** | 3.80 | 44.88 | 20.4 ms |

**The middle column is the old harness, and it is the answer to "how can a resize cost what
a whole network costs".** It cannot: all four models land within 5% of 3.7 ms because none
of those numbers is a model. They are the fence, measured four times. Take the copy out and
the network costs **7.5× the filters**, which is the ordering the page exists to show. The
earlier figures in this file — static bicubic 3.49 ms, the network 3.83 ms, "the adapter is
dispatch-bound" — were that artifact, not a software adapter as recorded.

Two things the corrected numbers say:

- **The filters are below the floor of the harness.** Dispatch alone — `session.run` with
  the fence deliberately skipped — measures 0.046 ms per run, so most of the 0.066–0.083 ms
  is JavaScript issuing the call, not the GPU running it. A one-node `Resize` on this card
  is too cheap to time one tile at a time. It also means ~2 ms of every 1080p frame is pure
  per-call overhead at 40 tiles, which is an argument for a larger tile in the browser than
  the CPU sweep chose.
- **The cubic kernel problem is a CPU problem only.** Bicubic is the *fastest* model on
  WebGPU and costs 20.5 ms on WASM — 33× the bilinear beside it, the same pathology the
  desktop CPU provider shows at 48 ms. The GPU implementation of that operator is fine; the
  scalar one is not.

The `cpu_ms` each notebook writes into its model is an ONNX Runtime CPU number for all four,
which is why a 48 ms desktop cubic sits in the dropdown beside a 0.066 ms WebGPU cubic
without either being wrong: same graph, three different kernels.

`www/models/` is gitignored: run a notebook's export cell to fill it.

## Where that leaves it

| | Best static | `upscale_nn` |
| --- | --- | --- |
| PSNR, validation patches | 23.62 dB | **24.17 dB** |
| ms/frame at 1280×720, torch on CPU | **6.02 ms** | 44.7 ms |
| 1080p frame, WebGPU | **2.6 ms** | 20.4 ms |
| 1080p frame, WASM | 820 ms | 1795 ms |

The learned model buys 0.55 dB over the best static method (bicubic + unsharp, 23.62 dB)
and 0.76 dB over plain bicubic, for 44.7 ms against bicubic's 6.02 ms at 720p — about seven
times the cost, which is not yet a trade worth making for a live stream. It also carries a
`Resize`, which the export cells flag as outside the op budget — the operator a WebGPU
provider is least likely to run natively, where a fallback would copy the tensor out of GPU
memory and back once per frame.

**The browser measurement does not support that fear on this hardware.** The WebGPU provider
runs `Resize` natively and runs it well: the static bicubic that is *nothing but* a `Resize`
is the fastest model on the page, and the network carrying one clears 20 ms for a 1080p
frame. Where the operator does bite is every scalar path — 20.5 ms per tile on WASM against
0.63 ms for bilinear, and 48 ms on the desktop CPU provider. The budget is still the right
rule to design to, and it was pointed at the wrong runtime.

So the bar still stands: **a learned model earns its place by beating 23.62 dB inside a
frame budget the static filters leave almost entirely unspent.** Nothing here does that yet.

Read the numbers with the dataset in mind: 12 synthetic frames of drawn windows and text is
a narrow, repetitive distribution. **None of them say anything about real content.** They
are measured on synthetic frames the preprocessing notebook drew itself, against the exact
bicubic degradation it applied.

## Next

- **A model shaped for the browser.** One notebook, one model: convolutions at low
  resolution, the skip connection as a frozen 1×1 convolution feeding `DepthToSpace` rather
  than a `Resize`, and nothing outside the `Conv`/`LeakyRelu`/`Add`/`DepthToSpace` budget.
  That is the shape the `Resize` result above argues for.
- **A bigger tile in the browser.** The 128 step was chosen on CPU timings. On a GPU the
  per-call overhead is ~0.05 ms of JavaScript against a filter that runs in less, so 40
  tiles spend ~2 ms per frame issuing calls. Re-run the tile sweep through the page.
- The same runtime tables on a GPU for training and for the desktop client. The browser
  side is now measured on real hardware; torch and onnxruntime here are still CPU builds.
- Real data: screen captures instead of synthetic frames, and a codec round trip in the
  degradation — a model trained on clean bicubic pairs has never seen blocking or ringing,
  and the unsharp result above is probably an artifact of their absence.
- Tiled inference, so a frame larger than memory can be upscaled.
- A perceptual metric beside PSNR, which rewards blur.
- Then a model worth the name, with both baselines kept as the numbers to beat.

Sibling problems: `../frame_gen_intra/summary.md`, `../frame_gen_extra/summary.md`.
