# Upscale — summary

Reconstructing a higher resolution frame from the one the stream delivered, so a client can
be sent fewer pixels than it displays.

> **Status: a baseline, three learned models, a sweep over six ways of building one, and
> the geometry to run one at.**
> `upscale_web` clears the frame budget: **0.19 ms per tile, 7 ms for a 1080p frame** on an
> NVIDIA Ampere card in Chromium, at **+6.9 dB over bicubic**. The measurement has since
> changed — quality is scored in Y'CbCr with luma weighted six times either chroma channel,
> because that is the ratio an eye reads them at — and the strategy sweep that uses it found
> that **the noise floor of this dataset is 0.61 dB**, which is wider than most of the
> differences anything here has claimed. All still synthetic frames, so the dB is a number
> about this dataset and the ms is a number about that card. Nothing in `src/` calls any of
> it yet.
>
> **The shape it runs at is now measured too, and the shipped one is wrong.** A tile is
> **640×360 out, from a 320×180 step and a 328×188 input** — 6.8 ms a 1080p frame against
> 11.7 ms at the published 128 step, and the same tile divides 720p, 1440p and 4K exactly.
> Everything about geometry is charged in **nanoseconds per output pixel**, because a run
> is a tile and a rate of runs cannot compare two tile sizes.

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
- `upscale_web.ipynb` — **the model shaped for the browser.** Same data, same metric; every
  design decision made against a measurement taken in the browser rather than in torch.
  Trains `WebUpscaler`, publishes it at three precisions and two tile sizes, and carries
  the ladder of what each optimisation cost and bought.
- `upscale_strategies.ipynb` — **six architectures under one recipe**, scored by the
  eye-weighted metric: the reference body under two losses, a colour-split network in two
  variants, a reparameterized one that collapses for export, a recursive one, a dense
  cascade and a channel-attention gate. It also measures what a difference has to be worth
  before it is one, which is the most useful thing in it.
- `upscale_geometry.ipynb` — **not a model: the shape one runs at.** Tile size, batch and
  scale factor, swept on the CPU and then measured in the browser on the card, charged in
  **nanoseconds per output pixel** because a run is a tile and a rate of runs cannot compare
  two tile sizes. It is what chose 640×360, and what `webexport.plan_tiling` implements.

`metrics.py` beside them is **how a frame is scored** — Y'CbCr, luma weighted 6:1:1, the
channels reported apart, SSIM on luma, and the same weighting available as a loss. It is
shared, and every notebook that measures quality should measure through it; the three
older ones still report RGB PSNR, so their tables and the newer ones are not comparable
column for column.

Each model notebook owns its own export cell and writes its `.onnx` into
`benchmark/www/models/` through `webexport.py`. That is all publishing is: the benchmark
server lists whatever graphs are in that folder, per request, so there is no manifest for a
re-export to leave stale. Twelve graphs reach the page when every notebook has been run:
the three static filters, the bicubic-residual model, the four `upscale_web` candidates,
and the two `upscale_strategies` publishes at two precisions each.

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

> **These numbers are of an undertrained model, and the notebook still produces them.**
> 268 patches at batch 32 is 8 steps an epoch, so 12 epochs is **96 optimizer steps**. The
> same architecture on the same data at 100 epochs reaches **31.25 dB (+7.84 dB over
> bicubic)** against the +0.76 dB below — measured in the sweep that chose the shape of
> `upscale_web`. Nothing about the *runtime* tables changes, and nothing about the
> conclusion that `Resize` is expensive changes; but every quality figure in this section,
> and the "not yet a trade worth making" it fed, was a fact about a training budget rather
> than about the model. Re-running that notebook at a real budget is the first item under
> Next.

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

## The browser-shaped model (`upscale_web.ipynb`)

**`WebUpscaler`** — 9,696 parameters (8,796 of them learned), fixed at ×2, halo 4. The same
ESPCN shape as `upscale_nn` with a narrower body (32-16 against 64-32) and one structural
change: **the skip is a convolution instead of a `Resize`**. Trained 100 epochs, L1, Adam
with a cosine schedule, about 30 s on CPU. **30.28 dB on the validation patches, +6.87 dB
over bicubic**, and 0.19 ms per tile in the browser.

### Every optimisation, and what it was worth

| | change | quality | speed |
| --- | --- | --- | --- |
| 1 | `Resize` → 5×5 `Conv` + the `DepthToSpace` already there | identical (2.4e-07) | **55.6 → 1.9 ms** per tile, ORT CPU |
| 2 | add the residual at LR, before the shuffle | identical | a quarter of the elements, one node fewer |
| 3 | body 64-32 → 32-16 | −1.2 dB | 27.6k → 9.6k MACs per LR pixel |
| 4 | float32 → float16 | −0.00 dB | 0.260 → 0.190 ms, and half the file |
| 5 | GPU-resident tensors instead of a CPU round trip | — | **3.24 → 0.30 ms** |
| 6 | reuse the output buffer instead of allocating per run | — | 0.300 → 0.250 ms |
| 7 | graph capture | — | 0.250 → 0.208 ms |
| 8 | tile step 128 → 256 | identical | 8 → 5 ms per 1080p frame |
| 9 | skip tiles that did not change (desktop content) | identical | 40 tiles → the ones that moved |
| ✗ | float32 → int8 (static QDQ, calibrated) | −0.10 dB | **0.19 → 26 ms**: Q/DQ is off the WebGPU path |
| ✗ | depthwise-separable body | −0.5 dB | no measurable saving |

**1 is the one that mattered, and it is arithmetic rather than approximation.** Bicubic ×2
is linear and shift-invariant with an integer stride, which is a convolution: each of the
four output phases is a fixed 4-tap filter, so the operator can be written down as weights
in a 5×5 `Conv` feeding the `DepthToSpace` that was already in the graph. The notebook does
not derive them from the cubic formula — it pushes unit impulses through `F.interpolate`
and reads its weights straight out of it, then asserts the result matches to 2.4e-07. What
that buys is not a faster kernel, it is *leaving* a kernel: the graph drops inside the
WebGPU op budget, and the 48 ms cubic `Resize` documented above stops being in the model.

The only place the two differ is the outermost ring, where `Resize` extends the edge and a
convolution pads with zeros. On a 64 px patch that ring is 12% of the pixels and it shows —
an untrained model measures 0.9 dB *below* the bicubic it is meant to start at, while the
interior is identical to 1e-6. It is the same border the halo already discards, so tiled
inference never sees it, and training absorbs the rest.

**4 and ✗ are the same experiment with opposite results, and the reason is the op set.**
float16 is the same graph in a cheaper type: 27% faster on a card with `shader-f16`, no
measurable dB. int8 is a cheaper type reached through *two extra operators* the WebGPU
provider does not implement, and it costs 130×. A quantized session also refuses graph
capture, so the trick that would claw some back is unavailable. In a browser the op set
decides the speed and the arithmetic barely gets a say.

**9 is the largest one available and it is not in the model at all.** A tile whose input
has not changed has an output that has not changed, so the model does not have to run — the
previous result is still correct. On a synthesised desktop sequence (a dragged window, a
moving cursor, a block of video playing) 44% of tiles change per frame at step 128, which
turns 7.6 ms into 3.3 ms. A real session with a typing pause in it sits much further to the
cheap end, and a full-screen video sits at the other. That is a client-side strategy rather
than a model property, and nothing in `src/` implements it yet.

### Where the quality went

The shape ladder, all at 100 epochs on the same data, same seed, same recipe — read to
±0.15 dB, which is the noise 76 validation patches support:

| body | val PSNR | vs bicubic | params | MACs / LR px |
| --- | --- | --- | --- | --- |
| 64-32 (`upscale_nn`'s body, conv skip) | 31.62 dB | +8.21 | 27,696 | 27,588 |
| 32-32-32 | 31.47 dB | +8.06 | 25,296 | 25,188 |
| 48-24 | 31.08 dB | +7.67 | 17,544 | 17,460 |
| **32-16 (shipped)** | **30.43 dB** | **+7.02** | **9,696** | **9,636** |
| 24-12 | 29.51 dB | +6.10 | 6,636 | 6,588 |
| 16-8 | 28.53 dB | +5.12 | 4,152 | 4,116 |

**Read that ladder against 0.61 dB, not against 0.15.** The ±0.15 above came from one pair
of runs; three runs of one strategy in `upscale_strategies.ipynb` land 0.61 dB apart, which
covers several of these rungs. The choice of shape survives it — a third of the MACs for
1.2 dB is a gap four times the noise — but the rows nearest each other were never
distinguishable.

A third of the MACs of the wide body for 1.2 dB, and it is the point where a 1080p frame
fits the 60 fps budget with the decode still to pay for. Two shapes that looked promising
and were not: a depthwise-separable body loses 0.5 dB at the same cost (the pointwise
mixing it saves is the part that was working), and a strided stem running the body at half
resolution loses 1 dB for a saving a narrower body buys more cheaply.

## How quality is measured (`metrics.py`)

Every dB above this line is **RGB PSNR**, and every dB below it is not. The instrument
changed, because the old one was answering about the wrong thing.

RGB PSNR counts a red error, a green error and a brightness error the same. An eye does
not. Luminance carries the detail it resolves — edges, text, the structure it judges
sharpness by — and chrominance carries far less, which is why every codec in use throws
away three quarters of the chroma samples (4:2:0) and nobody notices, and why nobody does
that to luma. So `metrics.py` converts to Y'CbCr (BT.709, full range), reports the three
channels apart, and weights them **6:1:1** — the ratio JVET uses, for the same reason.

The demonstration costs one cell. One validation patch, damaged twice with the *same*
error energy, once in luma and once in chroma:

| | psnr-y | psnr-u | psnr-v | **weighted** | psnr-rgb | ssim-y |
| --- | --- | --- | --- | --- | --- | --- |
| luma damaged | 26.09 | 55.89 | 50.77 | **27.34** | 26.12 | 0.562 |
| chroma damaged | 58.18 | 26.46 | 62.13 | **35.47** | 25.84 | 0.999 |

**RGB PSNR ranks the invisible damage as the worse of the two** (25.84 against 26.12).
The weighted score puts 8.1 dB between them, in the direction an eye would, and luma SSIM
agrees at 0.999 against 0.562.

What the module provides, and the distinctions that matter:

- `wpsnr` — weight the **errors** 6:1:1, then take one logarithm. A chroma error counts a
  sixth of a luma error of the same size: the perceptual claim written as arithmetic, and
  the number this folder reports.
- `wpsnr_db` — average the three **dB** figures 6:1:1, which is the published convention.
  Averaging logarithms is a geometric mean of the errors, so one very accurate channel
  lifts the whole score: it reads the chroma damage above at 54.71 dB where weighting the
  errors reads 35.47. Comparability, not the headline.
- `channel_psnr` / `psnr_y` — the channels apart, because the combined number hides which
  one moved. `chroma_subsample=2` scores chroma after a 2×2 average, which is how it is
  actually transmitted.
- `ssim_y` — structure on luma, standard 11×11 Gaussian window. PSNR rewards blur; an
  upscaler that hedges between two plausible edges beats one that commits to the wrong
  one on any pure error metric, and SSIM is the column that does not fall for it.
- `luma_chroma_l1` — the same weighting as a training loss, so the objective can be made
  to match the metric. Whether that is worth doing is measured below, and the answer is no.
- `psnr` — plain RGB, kept so the older tables stay readable.

**The two columns are different measurements and must not be subtracted.** Y carries most
of the error, so a weighted score sits near the luma score and away from the RGB one: the
32-16 model reads 31.81 dB weighted and 30.15 dB RGB on the same patches, in the same run.

## Six strategies (`upscale_strategies.ipynb`)

`upscale_web` asked how fast one architecture could be made. This notebook asks whether it
is the right architecture, and scores every answer with the metric above. Six ideas, eight
training runs, one recipe — 100 epochs, Adam at 1e-3 on a cosine schedule, the same seed
before every build, best epoch kept by weighted PSNR — and all of them on `upscale_web`'s
frame: frozen bicubic skip as a `Conv`, residual at LR, one `DepthToSpace`.

| strategy | weighted | vs bicubic | vs control | psnr-y | ssim-y | params | MACs/LR px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bicubic | 24.90 | — | — | 23.81 | 0.877 | 0 | 0 |
| **reparameterized** | **32.32** | **+7.42** | **+0.51** | 31.50 | 0.946 | 42,564 → 9,696 | 42,408 → **9,636** |
| 32-16 rgb l1 (control) | 31.81 | +6.91 | — | 30.93 | 0.944 | 9,696 | 9,636 |
| 32-16 luma l1 | 31.62 | +6.72 | −0.19 | 30.73 | 0.945 | 9,696 | 9,636 |
| yuv split | 31.48 | +6.59 | −0.32 | 30.57 | 0.941 | 8,288 | 8,216 |
| dense cascade | 31.46 | +6.56 | −0.35 | 30.56 | 0.942 | 10,424 | 10,374 |
| yuv split, wide | 31.34 | +6.44 | −0.47 | 30.43 | 0.942 | 9,888 | 9,813 |
| recursive ×3 | 30.95 | +6.05 | −0.86 | 30.01 | 0.938 | 6,176 | 10,740 |
| attention (se) | 30.71 | +5.82 | −1.10 | 29.77 | 0.939 | 10,248 | 9,636 |

### The result is the ruler, not the ranking

**The same strategy, trained three times at three seeds, lands 0.61 dB apart** — 31.42,
31.81, 32.03, nothing changed but the draw. That is the width of a coin toss on this
split, and it is wider than most of the table:

- **The winner's lead is inside it.** Reparameterization beats the control by 0.51 dB, so
  it is *unbeaten* rather than proven. It is still the one to take, because what it wins
  is free: the three branches collapse into the reference model's exact graph — `Conv`×4,
  `Add`, `DepthToSpace`, verified at 3.0e-07 — so a gain that might be luck costs nothing
  if it is. Every other strategy asks to be paid for.
- **Six rows are beaten by more than the noise**, and those are decided: both colour
  splits, the dense cascade, recursion, attention, and the luma-weighted control all lose
  to the reference shape by more than a draw.

**This corrects a claim in the section above.** `upscale_web` reads its width ladder to
±0.15 dB, on the strength of one pair of runs; three runs say the spread is four times
that. It does not overturn the shape it chose — a third of the MACs for 1.2 dB is still
the trade it was — but the rungs closest together in that ladder were never
distinguishable, and neither was the 0.15 dB it attributed to a re-draw.

### What each idea was worth

- **Matching the loss to the metric does nothing measurable.** Luma-weighted L1 lands 0.19
  dB *below* RGB L1, a third of the noise band. The reason is worth keeping: RGB L1 is
  already mostly a luma loss. Y is 0.21R + 0.72G + 0.07B and the three channels of a
  desktop frame move together, so an RGB error is a luma error wearing a hat, and
  reweighting it 6:1:1 moves the gradient less than it looks like it should. **The metric
  had to change; the loss did not have to follow.**
- **Splitting the colour does exactly what it was built to do, and the trade is bad.**
  Both variants own the chroma columns — 38.92/38.56 dB for the narrow one, 39.16 dB of
  `psnr-v` for the one whose luma branch sees all three channels, the best chroma in the
  table — and both give up more luma than that chroma is worth at six to one. A strategy
  derived from the metric, which the metric then rejects.
- **Reparameterization is the only free lunch.** 4.4× the arithmetic while training
  (42,408 MACs/LR px), the reference graph when it runs, because a sum of parallel
  convolutions is one convolution. The collapse is asserted, not assumed. One detail
  decides whether it is exact: the expansion's 1×1 carries **no bias**, or the 3×3 after
  it pads with zeros where the composed kernel would pad with that bias, and the two
  differ in a one-pixel ring — the same border error tiling would tile.
- **Attention loses three ways at once.** −1.10 dB, `Sigmoid` and `Mul` outside the WebGPU
  op budget, and a pooled gate that makes every output pixel depend on the whole tile — so
  two tilings of one frame disagree and the halo stops being a complete account of the
  tiling error. The notebook measures that rather than arguing it: its receptive-field
  probe reports `whole tile` where every other row reports a number.
- **Recursion is the cheapest to download and the most expensive to run.** 6,176
  parameters, 10,740 MACs — depth without parameters is the trade a *download* cares about
  and an inference budget does not, and it lost 0.86 dB here.

### Two bugs the notebook's own checks caught

Both are the kind that ship silently, and both were found by cells that measure a claim
instead of restating it:

- **A halo probe.** Every architecture declares its halo; the notebook fills a copy's
  weights with a positive constant and traces one output pixel's gradient back to the
  input to see how far it really reaches. `recursive` and `dense` declared one pixel short
  — the tail's 3×3 was not counted — which is exactly the error that tiles with a visible
  seam.
- **An op-budget check on every export.** The colour split read its luma channel as
  `x[:, :1]`, which exports as `Slice`: outside the budget, to save nine multiply-
  accumulates a pixel. It is a one-row 1×1 convolution now.

Published to the page: the collapsed reparameterized model and the control, at float32 and
float16 — one graph per architecture, since the top rows are otherwise the same shape twice.
Both are 0.93–1.07 ms per tile on the desktop CPU runtime, which is a CPU number: `upscale_web`
has already shown that CPU and WebGPU disagree about which operators are expensive.

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

`webexport.TILE_STEP` is **128 in → 256 out**. That was chosen on CPU timings; the browser
disagrees, and there is now a sweep on real hardware to say so — one `session.run` per tile
is JavaScript issuing commands, and at step 64 that overhead *is* the frame:

| step | ms / tile | tiles / 1080p frame | ms / frame |
| --- | --- | --- | --- |
| 64 | 0.155 | 144 | 21 |
| 128 | 0.202 | 40 | 8 |
| 192 | 0.311 | 18 | 5 |
| 256 | 0.452 | 12 | **5** |
| 320 | 0.657 | 8 | **4** |

Per-tile cost grows far more slowly than tile area, which is the signature of a fixed
per-call cost, and the halo agrees: at halo 4 a step-128 tile computes 13% more pixels than
it keeps, a step-256 tile only 6%. So `upscale_web` publishes both geometries — step 128
because it is the only one comparable with everything else on the page, and step 256
because it is what a client should run. The constant stays 128 until the static baselines
are re-exported beside it.

**That sweep asked the question of square steps only, and the answer was not among them.**
`upscale_geometry.ipynb` below sweeps the batch and the rectangle as well, charges everything
per output pixel, and lands on **640×360 out, from a 320×180 step** — 6.8 ms a 1080p frame
against this table's 8 ms at 128, and the same tile divides 720p, 1440p and 4K exactly too.
The `ms / frame` column here is also missing what it costs to *not* divide the frame: a 256
step computes 52% more of a 1080p output than that output has.

The rest of the original argument for 128 still holds: 84% of computed pixels kept at halo 6, a power of two once the pixels live in textures
and workgroups, bounded memory independent of stream resolution, and 40 tiles per 1080p
frame so the main thread yields 40 times instead of blocking once. That sweep was run
against a model that has since been removed, so the timings behind it are gone; the geometry
argument is not.

## The geometry a client runs at (`upscale_geometry.ipynb`)

Everything above is published at one geometry — a 128 LR step, a 136×136 input, 256×256
kept — chosen on CPU timings before there was a browser to choose it on. This notebook asks
what it should be, and measures the answer on the card.

**The metric is nanoseconds per output pixel, and frames per second is not one.** A run of
these graphs is a tile, so a rate of runs prices the geometry a graph was exported at rather
than the model in it: at ×2 a 256 step is worth four times a 128 step before either one has
been run. There are three pixel counts and they differ — **computed** (what the GPU ran),
**kept** (what the tile contributes) and **delivered** (the frame) — and two taxes between
them: the **halo tax**, `computed / kept`, which falls as the tile grows, and the **tiling
tax**, what a step wastes off the edge of a frame it does not divide, which rises. Only the
first is visible in a measurement of one tile.

### The answer: 640×360 out, 320×180 step, 328×188 in

Measured on the page — Ampere, fp16, WebGPU, GPU-resident, graph capture, one pass each, so
read differences under 10% as noise:

| geometry | keeps | ms / run | ns / kept px | ns / frame px | runs | 1080p frame | exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| whole frame | 1920×1080 | 5.703 | 2.75 | **2.75** | 1 | **5.7 ms** | yes |
| **320×180 step** | **640×360** | 0.755 | 3.28 | **3.28** | 9 | **6.8 ms** | **yes** |
| 320×270 step | 640×540 | 1.175 | 3.40 | 3.40 | 6 | 7.1 ms | yes |
| 192 step | 384×384 | 0.516 | 3.50 | 3.73 | 15 | 7.7 ms | no |
| 192×135 step | 384×270 | 0.420 | 4.05 | 4.05 | 20 | 8.4 ms | yes |
| 128 step, batch 16 | 256×256 | 3.040 | 2.90 | 4.40 | 3 | 9.1 ms | no |
| 384 step | 768×768 | 1.721 | 2.92 | 4.98 | 6 | 10.3 ms | no |
| 256 step | 512×512 | 0.893 | 3.41 | 5.17 | 12 | 10.7 ms | no |
| **128 step (shipped)** | 256×256 | 0.293 | 4.47 | **5.65** | 40 | **11.7 ms** | no |
| 64 step | 128×128 | 0.186 | 11.35 | 12.11 | 135 | 25.1 ms | no |

**The step everything is published at costs 72% more per delivered pixel than the best
tiling, and 105% more than running the frame whole.** A third of what a 128 tile costs is
the fixed cost of making a run at all, paid forty times a frame.

**A tile does not have to be square, and the good ones are not.** 1920×1080 has no square
tile that divides it — one would have to divide gcd(1920, 1080) = 120, a 60 pixel step,
small enough to be nearly all overhead — so every square step pays an edge: 26% at 128, 52%
at 256. Allow a rectangle and the waste goes to zero, and **640×360 divides every 16:9
resolution**: 720p is 2×2 of it, 1080p 3×3, 1440p 4×4, 4K 6×6. One exported graph, one
session, every stream. `webexport.plan_tiling` returns it for all four, and falls back to a
padded square for a frame that admits nothing near it — 1366×768 pays 32%.

Every exact row above has `ns / kept px` equal to its `ns / frame px`. That is what a tiling
tax of exactly 1.0 looks like from the far end, and it is the cleanest evidence that the two
columns measure what they claim to.

### What else the sweep said

- **The cost curve has no interior minimum.** Both taxes fall as the tile grows, so on cost
  alone the cheapest way to upscale a frame is to upscale it in one run — which the
  whole-frame row confirms at 2.75 ns. What bounds a tile from above is not speed: it is
  what can be skipped, what fits in a WebGPU buffer, and what blocks the main thread.
  **640×360 costs 19% more than the one-run frame and leaves 9 units that can be skipped**,
  and tiling beats a whole-frame run whenever fewer than about 80% of tiles have changed —
  against the 44% a desktop sequence measured. Hold both graphs and pick by content.
- **Batching amortises a dispatch without coarsening what can be skipped.** 16 tiles of step
  128 in one run cost 0.190 ms a tile against 0.293 run singly — 1.54×, where the fit
  predicted 2.1×, so some of what looks like per-run cost is per-tile work a batch pays
  anyway. It costs one halo per tile and nothing at the frame edge, which is what a bigger
  tile cannot say. On the CPU provider a batch buys nothing at all, because there is no
  dispatch there worth amortising.
- **Scale is the cheapest lever and the least measured.** The body runs at LR whatever the
  scale, so per *output* pixel ×2 is 2,409 MACs, ×3 is 1,436 and ×4 is 1,095 — and the
  stream carries a quarter of the pixels at ×4. Measured at the same step: 5.65, 3.23 and
  2.68 ns per delivered pixel. The quality of a ×3 or ×4 model is not measured anywhere
  here, because the dataset is a ×2 degradation.
- **Tiling stays exact at every geometry.** Tiled and whole-frame reconstruction are
  bit-identical at halo 4 and visibly wrong at halo 3, so all of this is a speed question
  and none of it is a quality one.
- **A mid-low card is a 30 fps card at 1080p.** The same Ampere card read 2.1× faster in the
  sweep further up this file than in this one, months apart, so absolute figures are a band
  and only ratios survive it: inside that band an Iris Xe or an M1 is over the whole 1080p60
  budget with the decode still to pay for, a GTX 1650 is at or over it, an RTX 3050 is
  borderline, and a mid card is inside. 720p60 and 1080p30 are the honest low-end targets.
  Every card but the one measured is a nominal TFLOPS figure and a ratio — the notebook
  publishes the family so the same sweep can be taken on a real one.

`TILE_STEP` stays 128: it is what the page *compares* models at, and moving it would
invalidate every published number without measuring anything. `TARGET_KEPT_PX` is the new
constant beside it, and it is 640×360.

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

Pick a model, pick a backend (auto / WebGPU / WASM), pick where the tile lives, pick how
the run is made, press the button. Five details it gets right, each of which it got wrong
first:

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
  it got. The data and runtime columns resolve the same way: WASM has no GPU buffers, and
  graph capture needs both.
- **The input type comes out of the graph.** The server reads the input's element type and
  the page allocates to match, so a float16 export is fed halves (converted in JS, which has
  no half type) and nothing anywhere has to be told which model is which. A graph whose
  input type the page cannot allocate is refused from the list with the reason, rather than
  listed and fed the wrong bytes.
- **The session owns its buffers, and the run is checked before it is timed.** Graph capture
  records a command buffer against the exact buffers of the run it recorded, so freeing them
  while keeping the session replays commands pointing at destroyed memory. That does not
  fail loudly: the queue raises a validation error, no work happens, and the timing comes
  back **ten times too good** — 0.067 ms for a model that costs 0.3. So the session and its
  two buffers are one object with one lifetime, and every result now reads its output back
  once and reports the mean absolute value in a `check` column. A run that computed nothing
  fails instead of reporting a record. The column doubles as a cross-check: every backend,
  precision and runtime mode is fed the *same* seeded tile, so two rows that disagree about
  the check were not running the same thing — fp32 0.4993, fp16 0.4992 (precision), int8
  0.4982 (quantization error), and a model that differs in the first decimal is a bug.

Twelve graphs reach the dropdown, and the tile geometry differs per model because the halo
does — nearest needs none, bilinear 1, bicubic 2, every network here 4:

| Source | Models | Model input | Halo |
| --- | --- | --- | --- |
| `upscale_dummy` | static nearest / bilinear / bicubic | 128 / 130 / 132 | 0 / 1 / 2 |
| `upscale_nn` | nn bicubic residual | 136 | 4 |
| `upscale_web` | web 32-16 in fp32 / fp16 / int8, and fp16 at step 256 | 136 / 264 | 4 |
| `upscale_strategies` | reparameterized (collapsed) and the 32-16 control, fp32 / fp16 | 136 | 4 |

The results table below predates the last two rows, which have not been measured in a
browser yet.

### Results — NVIDIA Ampere, Chromium, ms per tile

Medians of three interleaved rounds, not one run each: the same configuration drifts by up
to 10% between passes on this card, which is wider than some of the differences being
claimed, so the configurations are run in turn rather than in blocks.

| Model | WebGPU, GPU-resident | WebGPU, CPU round trip | WASM | 1080p frame (best) |
| --- | --- | --- | --- | --- |
| static nearest | **0.068** | 3.82 | 0.20 | 2.7 ms |
| static bilinear | **0.083** | 3.65 | 0.63 | 3.3 ms |
| static bicubic | **0.038** | 3.75 | 20.52 | 2 ms |
| nn bicubic residual | **0.447** | 3.80 | 44.94 | 18 ms |
| web 32-16 fp32 | **0.260** | — | 9.90 | 10 ms |
| **web 32-16 fp16** | **0.190** | 3.24 | 10.10 | **7 ms** |
| web 32-16 fp16, step 256 | **0.424** | — | 36.60 | **5 ms** |
| web 32-16 int8 | 26.2 | — | 10.37 | 1049 ms |

The three static rows and `nn` are with graph capture, which the earlier figures did not
have; the static nearest and bilinear rows are the older harness's and have not been
re-measured.

**The middle column is the old harness, and it is the answer to "how can a resize cost what
a whole network costs".** It cannot: all four models land within 5% of 3.7 ms because none
of those numbers is a model. They are the fence, measured four times. Take the copy out and
the network costs **7.5× the filters**, which is the ordering the page exists to show. The
earlier figures in this file — static bicubic 3.49 ms, the network 3.83 ms, "the adapter is
dispatch-bound" — were that artifact, not a software adapter as recorded.

Two things the corrected numbers say:

- **The filters are below the floor of the harness.** Dispatch alone — `session.run` with
  the fence deliberately skipped — measures 0.046 ms per run, so most of the 0.038–0.083 ms
  is JavaScript issuing the call, not the GPU running it. A one-node `Resize` on this card
  is too cheap to time one tile at a time. It also means ~2 ms of every 1080p frame is pure
  per-call overhead at 40 tiles — which the tile sweep above now prices directly, and which
  graph capture takes about 17% off.
- **The cubic kernel problem is a CPU problem only.** Bicubic is the *fastest* model on
  WebGPU and costs 20.5 ms on WASM — 33× the bilinear beside it, the same pathology the
  desktop CPU provider shows at 48 ms. The GPU implementation of that operator is fine; the
  scalar one is not.

The `cpu_ms` each notebook writes into its model is an ONNX Runtime CPU number for all of
them, which is why a 48 ms desktop cubic sits in the dropdown beside a 0.038 ms WebGPU cubic
without either being wrong: same graph, three different kernels. The same trap catches
float16 — the CPU provider has no half kernels for these ops, casts to float32 and runs the
same arithmetic, so `cpu_ms` shows fp16 as *slightly slower* while the browser shows it 27%
faster. A precision is not fast or slow on its own; a provider's kernels for it are.

`www/models/` is gitignored: run a notebook's export cell to fill it.

## Where that leaves it

| | Best static | `upscale_nn` | **`upscale_web`** | `upscale_strategies` |
| --- | --- | --- | --- | --- |
| PSNR (RGB), validation patches | 23.62 dB | 24.17 dB (12 epochs) / 31.25 dB (100) | **30.28 dB** | 30.15 dB |
| weighted PSNR, validation patches | — | — | — | **32.32 dB** (best of eight runs) |
| 1080p frame, WebGPU | 2 ms | 18 ms | **7 ms**, or 5 ms at step 256 | not measured yet |
| 1080p frame, WASM | 831 ms | 1798 ms | 396 ms | not measured yet |
| parameters | 0 | 26,796 | **9,696** | **9,696** (42,564 while training) |
| inside the op budget | no (`Resize`) | no (`Resize`) | **yes** | **yes** |

The frame figures in that table are all at the published 128 step, which
`upscale_geometry.ipynb` has since shown to be the most expensive geometry of the ones it
swept — the same model at 320×180 reads 6.8 ms where 128 reads 11.7 on the same card in the
same sitting.

**The bar is cleared.** It stood as: *beat 23.62 dB inside a frame budget the static filters
leave almost entirely unspent.* `upscale_web` scores +6.7 dB over the best static method and
takes 7 ms of a 16.7 ms frame at 60 fps — 5 ms at the tile size the browser sweep argues
for, and a fraction of that on a desktop stream where most tiles do not change. Two of the
three numbers that used to make this look impossible were not about the model: one was a
`Resize` kernel, and the other was a training budget of 96 optimizer steps.

What replaced the old fear is a sharper rule. The `Resize` worry was pointed at the wrong
runtime — the WebGPU provider runs that operator *well*, and the static bicubic that is
nothing but a `Resize` is the fastest model on the page — but the op budget was right for a
different reason than it was written for: **whether a provider implements an operator
natively is worth more than what the operator costs in theory**, which is why int8 through
two extra ops is 130× slower than float16 through none. Design to the budget; just expect
the surprises to be about which provider, not which operator.

Read every dB here with the dataset in mind: 12 synthetic frames of drawn windows and text
is a narrow, repetitive distribution, and the degradation is the exact bicubic downscale the
preprocessing notebook applied. Inverting a known linear operator on repetitive content is
the easiest problem in this family, which is why +7 dB is available at all. **None of these
numbers say anything about real content.** What they do support is the comparison between
the models, which were all measured the same way, and the runtime numbers, which do not
depend on the data at all.

## Next

- **More validation frames.** This is now the binding constraint on everything else: two
  held-out frames and 76 patches give a 0.61 dB noise floor, and most of the questions
  worth asking are about differences smaller than that. Nothing else on this list can be
  answered until it moves.
- **Re-run `upscale_nn.ipynb` at a real training budget, and re-measure it and
  `upscale_dummy.ipynb` through `metrics.py`.** Its 12 epochs are 96 optimizer steps, and
  every quality number attributed to it is a number about that; and both notebooks still
  report RGB PSNR, so neither can be compared with anything measured since.
- **Real data, which is now the only thing standing between this and a decision.** Screen
  captures instead of synthetic frames, and a codec round trip in the degradation: a model
  trained on clean bicubic pairs has never seen blocking or ringing, and a +7 dB that comes
  from inverting a known blur will not survive contact with one it has not seen.
- **Skip the tiles that did not change**, in whatever ends up driving inference in `src/`.
  It is the largest remaining factor on desktop content, it is exact rather than
  approximate, and the model needs no change for it.
- **Re-export everything at 320×180.** The step is decided — `upscale_geometry.ipynb`
  measured it — and every published graph is still at 128, including the one a client would
  run. `TILE_STEP` itself stays 128 while it is what the page compares models at.
- The same runtime tables on a GPU for training and for the desktop client. The browser
  side is now measured on real hardware; torch and onnxruntime here are still CPU builds.
- Tiled inference in the client, so a frame larger than memory can be upscaled.
- Then a model worth the name, with both baselines kept as the numbers to beat.

Sibling problems: `../frame_gen_intra/summary.md`, `../frame_gen_extra/summary.md`.
