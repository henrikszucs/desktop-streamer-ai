# Upscale — summary

Reconstructing a higher resolution frame from the one the stream delivered, so a client can
be sent fewer pixels than it displays.

> **Status: a baseline, three learned models, a sweep over six ways of building one, and
> the geometry to run one at.**
> `upscale_web` clears the frame budget: **0.19 ms per tile, 7 ms for a 1080p frame** on an
> NVIDIA Ampere card in Chromium, at **+6.9 dB over bicubic**. The measurement has since
> changed — quality is scored in Y'CbCr with luma weighted six times either chroma channel,
> because that is the ratio an eye reads them at — and the strategy sweep, re-run on the
> stream-degraded set, measures **a noise floor of 0.0004 SSIM-Y and 0.13 dB**, under which
> every 32-16-sized idea it tried is a tie. All still synthetic frames, so the dB is a number
> about this dataset and the ms is a number about that card. Nothing in desktop-streamer's `src/` calls any of
> it yet.
>
> **The shape it runs at is now measured too, and the shipped one is wrong.** A tile is
> **640×360 out, from a 320×180 step and a 328×188 input** — 6.8 ms a 1080p frame against
> 11.7 ms at the published 128 step, and the same tile divides 720p, 1440p and 4K exactly.
> Everything about geometry is charged in **nanoseconds per output pixel**, because a run
> is a tile and a rate of runs cannot compare two tile sizes.
>
> **Of eight ways to build the model, one pays, and it is the one that costs: residual
> blocks**, +0.0023 SSIM-Y over the 32-16 control (six times the seed spread) at 2.2× the
> memory traffic. Reparameterization, which notebook 6 was chosen for, buys nothing
> measurable on the stream-degraded set.
>
> **Screens were the missing data, and the host's encoder path is worth telling the model;
> the bitrate is not.** Notebook 7's model trained with desktop screens gains +0.062 SSIM-Y
> over bicubic on them, against +0.043 for the same model trained without, and +0.082 at
> 100% display scale. Told the downscale filter and the loop filter it gains another
> +0.0021 SSIM-Y overall (two seeds) and +0.42 dB luma on what the host sends today. Told
> the wrong filter it is worse than not told at all.

**One notebook is one model.** A notebook owns its architecture, its training, its numbers
and its export cell, and publishes straight into `dist/upscale/` at the repo root — so a
model is added or dropped by adding or deleting one file, and the page shows whatever is
published.

## Notebooks

Run in order, from this folder:

- `0_data_preprocess.ipynb` — source frames from the downloaded datasets (DIV2K, Flickr2K,
  YouTube sequences, read by `helpers/sources.py`), the desktop screens `datasets/screens.py`
  draws (400 train, 80 validation: code, documents, terminals, spreadsheets, chat, file
  explorers and settings in real fonts, light and dark, 100–200% display scale, some in
  ClearType; a third of the patches now) and `datasets/raw/captures/` (synthesised
  desktop-like frames when all of that is empty), every one made a clip: a YouTube
  sequence as it is, an image as a second of a 1280×720 window sliding, holding still or
  (a tenth of them) zooming over it (`helpers/motion.py`) → the stream's own ×2 degradation from `helpers/degrade.py` on the
  whole frame: downscale, 4:2:0, H.264 at a random constant 0.2–1.0 bpp with x264's loop filter
  on for half of them, no codec for a tenth, decode, drawn per source → patches at random
  even places, each held still for 8 frames, a 72 px LR input (64 px core + a halo of 4)
  over a 128 px HR target, flat ones dropped by luma variance and ones the codec destroyed
  dropped by `MIN_LR_PSNR` (LR core under 24 dB luma PSNR against a clean downscale; 3% of
  the busy spots at 0.2–1.0 bpp), split by *source* (an image, or a whole video)
  rather than by patch → four `uint8` `.npy` arrays of
  `(patches, frames, h, w, 3)` and a `manifest.json` in `datasets/upscale/` that records
  each source's motion and degradation. `helpers/patches.py` reads it back for the model notebooks,
  one frame a sample, and they trim the halo off the output before the loss and every
  score. The results below were measured on the earlier clean bicubic pairs of 64 px
  patches without a halo, so none of them carries over: on a 40-source test set, the
  32-16 model trained 100 epochs gains +1.0 dB over bicubic without the codec and +0.16 dB
  with it.
- `1_upscale_dummy.ipynb` — **static methods only.** Seven of them: nearest, box, bilinear,
  hamming, bicubic, Lanczos, and bicubic followed by an unsharp mask. No training; it
  measures quality and runtime, hands out the bar, and exports the three ONNX can express
  as `Resize` so the bar exists in the browser too.
- `2_upscale_nn.ipynb` — **the PyTorch attempt.** Trains `PlainCNN`, an ESPCN-shaped
  network with no bicubic shortcut that makes the whole picture itself, on the patch
  pairs, measures it the same way, writes `checkpoints/upscale_nn.pt`, and exports it.
- `3_upscale_web.ipynb` — **the model shaped for the browser.** Same data, same metric; every
  design decision made against a measurement taken in the browser rather than in torch.
  Trains `PlainUpscaler` (`helpers/models.py`), publishes it at three precisions and two
  tile sizes, and carries the ladder of what each optimisation cost and bought.
- `4_upscale_strategies.ipynb` — **eight strategies under one recipe**, on the
  stream-degraded set, in notebook 3's order (data → models → speed → train → save →
  export): the 32-16 control, with a leaky ReLU, reparameterized, correcting luma only,
  both of those, notebook 7's residual blocks with and without reparameterization, and a
  colour branch. Every strategy's halo and starting point are measured, and its speed
  estimated, before it trains; every run is checkpointed, so a re-run only loads them.
  It also measures what a difference has to be worth before it is one — the control at
  three seeds — which is still the most useful thing in it. Publishes the best two graphs
  at the 320×180 step.
- `5_upscale_geometry.ipynb` — **not a model: the shape one runs at.** Tile size, batch and
  scale factor, swept on the CPU and then measured in the browser on the card, charged in
  **nanoseconds per output pixel** because a run is a tile and a rate of runs cannot compare
  two tile sizes. It is what chose 640×360, and what `tiles.plan_tiling` implements.
- `6_upscale_reparam.ipynb` — **the selected model on the stream-degraded set.** The
  reparameterized 32-16 the strategy sweep chose, trained on the new patch set as several
  runs (other seeds and learning rates) with the best on validation kept. Each run trains
  until it has converged - the learning rate halves whenever the score stalls, and the run
  stops once a hundredth of the starting rate brings nothing more - and saves a checkpoint
  every epoch. The sweep's 1e-3 does not learn on this data: 100 epochs at 1e-3 ended
  within 0.001 dB of bicubic, and a 15-epoch run at 1e-3 and an 8-epoch one at 3e-3 never
  beat it, while 1e-4 was +0.14 dB after 5 epochs and still rising - so the runs start at
  1e-4. The kept run is measured against bilinear, bicubic and Lanczos-3 overall and by
  bitrate and content, collapsed, checkpointed to `checkpoints/upscale_reparam.pt`, and
  published at the client's 320×180 step (float32, float16 and a whole-1080p-frame graph)
  with bilinear and bicubic beside it as the page's baselines. Not run yet:
  `dist/upscale/` was emptied of every graph made on the old data, so until it runs the
  page has no models. Its premise has not held up: the strategy sweep re-run on this data
  finds reparameterization worth nothing measurable, and explains the 1e-3 failure — the
  branches switch off 14 of the 16 narrow channels within 20 steps (see "Strategies").
- `7_upscale_luma.ipynb` — **a residual model that corrects brightness only.** Colour is
  bicubic and never touched; the network adds one luma correction per output pixel, the
  same value to R, G and B, with two residual blocks inside the 4-pixel halo. Same workflow
  and loss as notebooks 2 and 3; published at the client's 320×180 step (float32, float16,
  whole frame).
- `8_upscale_screens.ipynb` — **notebook 7's model on a set with desktop screens, blind and
  told how the stream was made.** Trains it blind at three seeds (the ruler), then told the
  bitrate (`degrade.condition`: what the client always knows, having asked for the
  bandwidth, height and frame rate), then told the host's encoder path too (the downscale
  filter and the loop filter). A told model adds one shift per channel after the stem and in
  each block, which folds exactly into the biases for a given stream, so it ships as the
  same single-input graph, one per bitrate. Scored on screens, photos and video apart, by
  display scale, bitrate and filter, and on what the host sends today; notebook 4's model,
  trained before there were screens, is scored on the same set. Screens are worth +0.019
  SSIM-Y on screens over notebook 4's model, the bitrate is worth nothing, and the path
  +0.0021 SSIM-Y overall — see "Desktop screens and a known stream". It publishes the told
  model baked for the host's path today at 0.2, 0.46 and 1.0 bpp, and only the 0.46 graph
  earns its place.
- `9_upscale_dct.ipynb` — **a model that only guesses gains.** Luma is the DCT of 8×8 blocks
  every 4 px, one learned gain per coefficient of the 16×16 spectrum, and a windowed IDCT
  whose halves overlap and add up to exactly 1; colour is bicubic. The 16-point DCT of a
  nearest-neighbour block is its 8-point DCT mirrored into four quadrants, so the larger
  spectrum is the small one copied, and with every gain at 1 the model is nearest neighbour
  (checked, 7e-7). The transforms are fixed convolutions; what learns is the gains: *fixed*
  (256 numbers, the same for every block), or *decided* per block by two 1×1 convolutions on
  its coefficients, starting from the fixed gains as a frozen bias. Also measures the ceiling
  — gains chosen from the truth — before training. All `Conv`, `Concat`, `DepthToSpace`,
  `Relu`, `Add`, plus one `Mul` (the gains), which is outside the measured budget. Counted
  from the graphs: 4,600 operations and 875 bytes per LR pixel for the fixed model, 6,104 and
  1,061 deciding through 32 channels, against 28,120 and 2,536 for notebook 7's. **Not run
  yet.** Its smoke run (one epoch each) puts the ceiling high — gains chosen from the truth
  reach 0.974 SSIM-Y on the inner 112 px, where bicubic is at 0.905 — and the models, one
  epoch in, far under it: on the whole cores 0.906 fixed and 0.913 deciding through 32,
  against 0.905 for bicubic and 0.929 for notebook 4's model.

The notebooks share their code through `helpers/`, so each one reads as a short list of
steps: `models.py` holds the bicubic skip and the models built on it (`PlainUpscaler`,
`ReparamUpscaler`), `training.py` the loaders, the training loop, scoring and checkpoints,
and `speed.py` the timing.

`helpers/metrics.py` is **how a frame is scored** — Y'CbCr, luma weighted 6:1:1, the
channels reported apart, SSIM on luma, and the same weighting available as a loss. Every
model notebook scores through it (`training.evaluate`). Notebooks 2 and 3 still keep their
best epoch by RGB PSNR, and their numbers recorded below are RGB PSNR, so compare those
with the `psnr-rgb` column, not with `wpsnr`.

Each model notebook owns its own export cell and writes its `.onnx` into `dist/upscale/`
at the repo root through `helpers/webexport.py`. That is all publishing is: the benchmark server
lists whatever graphs are in that folder, per request, so there is no manifest for a
re-export to leave stale. Twelve graphs reach the page when every notebook has been run:
the three static filters, the bicubic-residual model, the four `upscale_web` candidates,
and the two `upscale_strategies` publishes at two precisions each. (That count predates
notebooks 6 and 7, and the strategy sweep now publishes three graphs for each of its two.)

`data/` and `checkpoints/` are gitignored: they are what training leaves behind. `dist/` is
committed, because it holds only what is finished and published, and those models took
long to train. The export cells need
`onnx` and `onnxruntime`, which are project dependencies.

Every number below is CPU, 12 threads. The installed torch is a CPU build (`2.13.0+cpu`)
even though `pyproject.toml` declares a CUDA index, so nothing here has been measured on a
GPU. Runtimes are forward pass only — no decode, no copy, no PNG encoding.

## How much the stream costs the LR (`0_data_preprocess.ipynb`)

Every stored LR patch against the *ideal* downscale of its HR target — the same filter, in
RGB, no 4:2:0 and no codec — on the inner 56×56 of its core (RGB PSNR). Rebuilding a
second reference with 4:2:0 but no codec splits the loss into its stages.

**The codec is all of it.** On the set drawn from 0.013–0.46 bpp, 4:2:0 alone costs a
median 48 dB and the downscale filter about 1 dB either way; the codec is 96% of the
squared error. Half the coded frames were under 30 dB, a fifth under 26.

**Bitrate decides it, then motion.** 72 sources rebuilt exactly as the notebook makes them
and encoded at fixed bitrates; busy 64×64 LR blocks, share under 30 dB:

| bpp | Mbit/s, 640×360 | Mbit/s, 960×540 | still | slide | zoom | video | all |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.4 | 0.9 | 90% | 80% | 91% | 54% | 74% |
| 0.13 | 1.0 | 2.2 | 52% | 55% | 80% | 16% | 43% |
| 0.20 | 1.5 | 3.5 | 34% | 46% | 74% | 7% | 33% |
| 0.46 | 3.5 | 7.9 | 13% | 29% | 58% | 0% | 20% |
| 1.00 | 7.7 | 17.3 | 5% | 11% | 30% | 0% | 9% |
| 1.50 | 11.5 | 25.9 | 3% | 4% | 19% | 0% | 5% |

Real video is clean from 0.2 bpp. A zoom stays the worst motion at every bitrate, because
H.264 has no prediction for it. Keyframes, one a second into a two-frame buffer, are the
worst frames: a median 21 dB at 0.05 bpp, 35 at 0.46.

**So the set is drawn from 0.2–1.0 bpp** (log-uniform, centred on the host's 0.46
default — a client streams a small picture at a high bitrate) and **zoom is a tenth of the
image clips**, down from four tenths. Below about 0.1 bpp the LR is mostly destroyed, which
is not a stream anyone watches, and the model cannot learn from it.

The set rebuilt that way, against the one before (every stored frame, RGB PSNR against the
ideal downscale):

| | median | under 30 dB | under 26 dB | under 20 dB | coded frames smeared |
|---|---|---|---|---|---|
| 0.013–0.46 bpp, zoom 4 in 10 | 31.5 dB | 43% | 20% | 0.76% | 20% |
| **0.2–1.0 bpp, zoom 1 in 10** | **39.0 dB** | **12%** | **3.7%** | **0.04%** | **2.7%** |

The gate now drops 478 busy spots (3%) instead of 12,643. What is left of the damage is
mostly **slides** — 29% of the frames, a median of 33.7 dB and a quarter under 30 dB —
which is scrolling, a desktop's commonest motion, so it stays. Real video is at 41.9 dB
with 4% under 30. Every result further down was trained on an older set.

## Static methods (`1_upscale_dummy.ipynb`)

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

## Learned model (`2_upscale_nn.ipynb`)

**`PlainCNN`** — the notebook's model: a plain CNN with no bicubic shortcut, so it makes the
whole picture itself - convolutions at low resolution, then `PixelShuffle`; halo 4, nothing
outside the operator budget. Its random start is far below bicubic (8.3 dB RGB for
128-64). Six shapes, all with halo 4, each trained once at seed 0 on the stream-degraded
patch set until it converged (gain over bicubic on validation; operations per LR pixel as
`webexport.count_work` counts them):

| Shape (kernels) | Parameters | Operations per LR pixel | wpsnr | psnr-rgb |
|---|---|---|---|---|
| **16-24-24-32 (3, 3, 3, 1, output 3)**, the default | 13,404 | 26,796 | −0.10 | −0.10 |
| 16-24-24-32 (5, 3, 1, 1, output 3) | 9,564 | 19,116 | −0.15 | −0.15 |
| 64-32 (5, 3, output 3) | 26,796 | 53,580 | +0.19 | +0.18 |
| 128-64 (5, 3, output 3) | 90,444 | 180,876 | +0.22 | +0.22 |
| 64-64-64 (5, 3, 3, output 1×1) | 79,500 | 158,988 | +0.11 | +0.13 |
| 128-128-64 (5, 3, 3, output 1×1) | 231,884 | 463,756 | +0.20 | +0.20 |

The default, 16-24-24-32 (few channels first, more deeper in), is there for speed: about 7×
less work than 128-64, and memory-bound on most target GPUs. The notebook's speed section
estimates about 36 ms per 1080p frame on an Iris Xe or an M1 and 19 ms on a GTX 1650,
against 121 and 81 ms for 128-64. But as a plain CNN it stays **0.1 dB below bicubic**, and
about 1 dB below it in each chroma channel: with 13k parameters it cannot both rebuild the
picture and improve it. 64-32 is the smallest shape here that beats bicubic.

A 1×1 output convolution hurts, and more channels help a little; the three best are within
about one run's noise. The 64-32 body *with* the bicubic shortcut (`ResidualUpscaler`,
below), trained the same way, reached +0.27 dB RGB and +0.33 dB wpsnr: learning a
correction to bicubic is easier than making the picture from scratch.

The default 16-24-24-32 body *with* the shortcut (bicubic as a frozen 5×5 convolution added
before the `PixelShuffle`, the output layer starting at zero; about 12% more work than the
plain one, still inside the operator budget) reached **+0.27 dB RGB and +0.32 dB wpsnr**
with the learning rate starting at 1e-3: as good as the 64-32 residual, and better than the
plain 128-64 at a seventh of its work. Started at 1e-4, the same model found only +0.04 dB
RGB - the opposite of notebook 6's reparameterized model, which needs the smaller start.

**The loss is now SSIM on luma + L1, and the best epoch is picked by luma SSIM.** L1 and
PSNR judge each pixel alone, so a blurred edge is their safest bet; SSIM compares edges and
contrast in 11×11 windows. Three losses on the default body, on the 0.2–1.0 bpp set, two
seeds each (the seeds agree to 0.0007 SSIM), change against bicubic:

| Loss | ssim-y | psnr-y | psnr-u / -v | wpsnr | minutes |
|---|---|---|---|---|---|
| L1 | +0.0091 | **+0.78** | **−1.3** | **+0.73** | 16 |
| **0.84 × (1 − SSIM-Y) + 0.16 × L1** (`metrics.ssim_l1_loss`) | **+0.0127** | +0.64 | −3.4 | +0.40 | 33 |
| the same with SSIM on each RGB channel (Zhao et al.'s form) | +0.0119 | +0.42 | −2.1 | +0.33 | 36 |

The SSIM loss buys 40% more SSIM gain for 0.14 dB of luma PSNR, and keeps visibly more
fine texture side by side; it is better than L1 on 81% of the validation patches. The
price is chroma, which it barely trains: 2 dB under L1, though no colour shift shows in the
same pictures. Every loss leaves chroma *below* bicubic, because a plain CNN has to rebuild
the colour that bicubic gets for free — the shortcut above is the fix, not the loss.

The rest of this section is about that earlier `ResidualUpscaler`.

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

## The browser-shaped model (`3_upscale_web.ipynb`)

**`PlainUpscaler`** (`helpers/models.py`; called `WebUpscaler` when these numbers were taken) — 9,696 parameters (8,796 of them learned), fixed at ×2, halo 4. The same
ESPCN shape as `upscale_nn` with a narrower body (32-16 against 64-32) and one structural
change: **the skip is a convolution instead of a `Resize`**. Trained 100 epochs, L1, Adam
with a cosine schedule, about 30 s on CPU. **30.28 dB on the validation patches, +6.87 dB
over bicubic**, and 0.19 ms per tile in the browser.

**On the 0.2–1.0 bpp set, in notebook 2's workflow** (SSIM-Y + L1, best epoch by luma SSIM,
150 epochs, 11 min on an RTX 3090), against bicubic and against notebook 2's plain CNN
under the same loss (the mean of two seeds):

| | ssim-y | psnr-y | psnr-u / -v | wpsnr | operations per LR pixel | ms per 1080p frame, RTX 3090 |
|---|---|---|---|---|---|---|
| bicubic | 0.9135 | 36.38 | 47.06 / 47.27 | 37.30 | | |
| plain CNN 16-24-24-32 (notebook 2) | 0.9263 | 37.03 | 43.6 / 44.0 | 37.71 | 26,796 | 2.3 |
| **residual 32-16** | **0.9265** | **37.13** | 44.6 / 45.4 | **37.96** | **19,392** | **1.7** |

The residual model is as good as the plain one with 28% less work, and keeps a dB more
chroma. The gain over bicubic is still small — +0.013 SSIM, +0.75 dB luma — and chroma
still ends 2 dB *under* bicubic although it starts exactly there: the loss is 84% luma
SSIM, and the correction the body learns for luma moves the colour with it. On the drawn
desktop frame, with no codec, chroma matches bicubic and the model gains +1.5 dB wpsnr.
The estimate puts a 1080p frame at 25 ms on an Iris Xe or an M1 (30 fps) and 13 ms on a
GTX 1650 (60 fps). float16 scores the same as float32; int8 loses 0.3 dB.

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
than a model property, and nothing in desktop-streamer's `src/` implements it yet.

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
of runs; three runs of one strategy in `4_upscale_strategies.ipynb` land 0.61 dB apart, which
covers several of these rungs. (That was the synthetic set; on the stream-degraded one, three
seeds of the 32-16 land 0.13 dB apart.) The choice of shape survives it — a third of the MACs for
1.2 dB is a gap four times the noise — but the rows nearest each other were never
distinguishable.

A third of the MACs of the wide body for 1.2 dB, and it is the point where a 1080p frame
fits the 60 fps budget with the decode still to pay for. Two shapes that looked promising
and were not: a depthwise-separable body loses 0.5 dB at the same cost (the pointwise
mixing it saves is the part that was working), and a strided stem running the body at half
resolution loses 1 dB for a saving a narrower body buys more cheaply.

## The luma residual model (`7_upscale_luma.ipynb`)

The eye resolves detail in brightness far better than in colour, so **`LumaUpscaler`**
(`helpers/models.py`) spends nothing on colour: the bicubic skip makes the picture, and the
network predicts one correction per output sub-pixel that is added to R, G and B alike —
in BT.709 that moves Y by exactly that value and leaves Cb and Cr as bicubic made them
(checked: 1e-7 with random weights). The body is a 3×3 stem and two **residual blocks**
(3×3 → ReLU → 1×1, plus the block's input): a 1×1 reaches no further, so a block costs one
pixel of halo, and the model gets deeper inside the same halo of 4. All `Conv`, `Relu`,
`Add`, `Concat`, `DepthToSpace`.

Five designs under notebook 3's recipe (SSIM-Y + L1, best epoch by luma SSIM), two seeds
each; the seeds agree to 0.0001–0.0014 SSIM. Milliseconds are the roofline estimate for a
1080p frame at notebook 3's measured share of the peak:

| | ssim-y | psnr-y | psnr-u / -v | wpsnr | ops / bytes per LR px | Iris Xe | GTX 1650 |
|---|---|---|---|---|---|---|---|
| bicubic | 0.9135 | 36.38 | 47.06 / 47.27 | 37.30 | | | |
| residual 32-16, RGB correction (notebook 3) | 0.9260 | 37.10 | 44.6 / 45.4 | 37.94 | 19,392 / 1,128 | 25 ms | 13 ms |
| the same body, luma correction | 0.9260 | 37.12 | **47.1 / 47.3** | 37.98 | 17,080 / 1,192 | 26 ms | 14 ms |
| **stem 24 + 2 residual blocks, luma** | **0.9284** | **37.31** | **47.1 / 47.3** | **38.16** | 28,120 / 2,536 | 56 ms | 30 ms |
| stem 32 + 2 residual blocks, luma | 0.9289 | 37.34 | 47.1 / 47.3 | 38.19 | 47,128 / 3,240 | 72 ms | 38 ms |
| stem 24 + 2 blocks, the body reads Y only | 0.9280 | 37.31 | 47.1 / 47.3 | 38.15 | 27,262 / 2,544 | 56 ms | 30 ms |

- **Correcting luma only is free and fixes the colour.** The same body keeps every bit of
  its luma gain and its chroma stays at bicubic instead of 2 dB under it — the drift of
  notebooks 2 and 3 is gone by construction, not by tuning the loss.
- **Residual blocks buy luma, and memory pays.** +0.0024 SSIM and +0.2 dB luma over the
  same-cost body, at 2.2× the bytes a frame moves — and every target but the RTX cards is
  memory-bound, so 2.2× the time there. 32 channels add 0.0005 more for another 30%.
- **Reading Y alone buys nothing**: the stem is a small part of the work, and R, G, B
  carry nothing the network can use against it either way.

The notebook trains the 24-channel model: **0.9282 SSIM-Y (+0.0147), +0.91 dB luma, colour
at bicubic**, 108 epochs in 9 min. On the drawn desktop frame (no codec) +2.0 dB luma and
0.9648 SSIM against 0.9437. 3.3 ms a 1080p frame on the RTX 3090; the estimate is 9 ms on
an RTX 3060 (60 fps), 26 ms on a GTX 1650 (30 fps) and 49 ms on an Iris Xe or an M1 —
under 30 fps there, where notebook 3's model is at 25 ms. Those devices want the same-cost
luma body instead, or tiles that did not change skipped.

## How quality is measured (`helpers/metrics.py`)

Every dB above this line is **RGB PSNR**, and every dB below it is not. The instrument
changed, because the old one was answering about the wrong thing.

RGB PSNR counts a red error, a green error and a brightness error the same. An eye does
not. Luminance carries the detail it resolves — edges, text, the structure it judges
sharpness by — and chrominance carries far less, which is why every codec in use throws
away three quarters of the chroma samples (4:2:0) and nobody notices, and why nobody does
that to luma. So `helpers/metrics.py` converts to Y'CbCr (BT.709, full range), reports the three
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
- `ssim_l1_loss` — 0.84 × (1 − SSIM on luma) + 0.16 × L1: a loss that pays for structure,
  not only pixel values. `2_upscale_nn.ipynb` trains with it (see "Learned model").
- `psnr` — plain RGB, kept so the older tables stay readable.

**The two columns are different measurements and must not be subtracted.** Y carries most
of the error, so a weighted score sits near the luma score and away from the RGB one: the
32-16 model reads 31.81 dB weighted and 30.15 dB RGB on the same patches, in the same run.

## Strategies (`4_upscale_strategies.ipynb`)

`upscale_web` asked how fast one architecture could be made, and `upscale_luma` found a
better one. This notebook asks which *ideas* are worth keeping. Eight strategies under one
recipe — notebook 3's: SSIM-Y + L1, the best epoch by luma SSIM, each trained until it has
converged, every one from the same seed and so on the same batches in the same order — on
the stream-degraded set (0.2–1.0 bpp), all on notebook 3's frame (frozen bicubic skip as a
`Conv`, residual at LR, one `DepthToSpace`), and all inside the client's 4-pixel halo and
the WebGPU op budget, which the notebook measures rather than trusts. Then the control
again at two more seeds, which is the ruler, and once from 3e-4. Eleven runs, 107 minutes
on the RTX 3090.

Against the mean of the control's three seeds (0.9263 SSIM-Y, 37.93 dB wpsnr). A verdict
needs a gap wider than the seeds' spread, **0.0004 SSIM-Y and 0.13 dB wpsnr**; anything
narrower is *luck*. Milliseconds are the roofline estimate for a 1080p frame in 9 tiles of
328×188, at the 38% of its peak the RTX 3090 reaches on the control:

| strategy | ssim-y | vs control | wpsnr | vs control | psnr-u / -v | ops / bytes per LR px | Iris Xe | GTX 1650 |
|---|---|---|---|---|---|---|---|---|
| bicubic | 0.9135 | | 37.30 | | 47.06 / 47.27 | | | |
| **luma 24 + 2 blocks** (notebook 7's model) | **0.9286** | **+0.0023** better | **38.20** | **+0.27** better | 47.10 / 47.30 | 28,120 / 2,536 | 54 ms | 29 ms |
| the same, reparameterized | 0.9285 | +0.0022 better | 38.18 | +0.25 better | 47.10 / 47.31 | 28,120 / 2,536 | 54 ms | 29 ms |
| 32-16, leaky ReLU | 0.9266 | +0.0003 luck | 38.04 | +0.11 luck | 44.67 / 45.50 | 19,392 / 1,128 | 24 ms | 13 ms |
| yuv split (an 8-4 colour branch) | 0.9265 | +0.0002 luck | 37.99 | +0.06 luck | 45.15 / 45.76 | 16,576 / 1,556 | 33 ms | 18 ms |
| 32-16, reparameterized, from 3e-4 | 0.9265 | +0.0002 luck | 38.01 | +0.08 luck | 44.64 / 45.43 | 19,392 / 1,128 | 24 ms | 13 ms |
| luma 32-16, reparameterized, from 3e-4 | 0.9265 | +0.0002 luck | 37.97 | +0.04 luck | 47.09 / 47.29 | 17,080 / 1,192 | 25 ms | 14 ms |
| luma 32-16 | 0.9264 | +0.0002 luck | 38.07 | +0.14 better | 47.09 / 47.29 | 17,080 / 1,192 | 25 ms | 14 ms |
| 32-16, the control, 3 seeds | 0.9261–0.9265 | | 37.88–38.01 | | 44.60 / 45.39 | 19,392 / 1,128 | 24 ms | 13 ms |
| the control from 3e-4 | 0.9260 | −0.0003 luck | 37.93 | +0.00 luck | 44.58 / 45.42 | 19,392 / 1,128 | 24 ms | 13 ms |

### What each idea was worth

- **One idea pays, and it is the one that costs.** Residual blocks are nearly six times
  the seed spread above the control in SSIM-Y — the only SSIM verdict that is not luck —
  and they move 2.2× the bytes: 54 ms a 1080p frame on an Iris Xe, under 30 fps, where every
  32-16-sized body is at 24–25. Every other idea, five of them, lands inside the control's
  band or at its edge. At this size the shape of the body inside 17–19k operations a pixel is
  not where the quality is.
- **Reparameterization is worth nothing measurable, three times over.** On the control's
  body +0.0002 SSIM-Y, on the luma body +0.0001 over its plain twin, on notebook 7's body
  −0.0001. The first sweep's +0.51 dB (below) was inside its own 0.61 dB noise and does not
  reproduce. It still has a price: 4.4× the arithmetic while training, and a learning rate
  it can survive. **At 1e-3 the 32-16 switches off 14 of the 16 channels of its narrow layer
  within 20 steps** — three branches moving at once take a larger step on the merged kernel
  than one convolution does, and a ReLU passes no gradient to a channel that is negative
  everywhere — and the model is bicubic again, for good. That is notebook 6's run at 1e-3
  that never beat bicubic. From 3e-4 it learns; residual blocks, which add their input back,
  keep it alive at 1e-3; and the control from 3e-4 lands where it does from 1e-3, so the
  rate is not what the branches were missing. **Notebook 6 is built on the first sweep's
  pick, and this sweep does not support it.**
- **Correcting luma only is free, and what it buys is colour, not structure.** `luma
  32-16` has the control's SSIM-Y (luck) and colour at bicubic instead of about 2 dB under
  it. By arithmetic that is worth about 0.1 dB of wpsnr — the control's chroma error is a
  few per cent of its weighted error — and the two luma 32-16 runs land at +0.14 and +0.04,
  the seed spread either side of it. Notebook 7's claim, repeated in a different sweep.
- **A colour branch cannot beat bicubic at colour here.** `yuv split` starts at bicubic's
  colour and ends 1.5–1.9 dB under it — better than the RGB bodies' 2 dB under, worse than
  leaving colour alone — and its two bodies cost a third more time than the luma frame. The
  loss is 84% luma SSIM and the best epoch is picked by luma SSIM, so nothing in the recipe
  asks colour to get better; under this recipe, bicubic is the best colour on the table.
- **Dying ReLUs are real, and cost nothing measurable.** At 1e-3 the control switches off
  6 of its 16 narrow channels in its first 60 steps. A leaky ReLU (slope 0.1, `LeakyRelu`,
  inside the budget) keeps them, and lands +0.0003 SSIM-Y and +0.11 dB wpsnr: both luck.
- **The ruler is almost five times narrower than the first sweep's** — 0.13 dB wpsnr
  against 0.61, from 2,369 validation patches of 149 sources instead of 76 patches of two
  frames — and the differences between bodies of one size are still inside it.

**What that means for the client: two tiers, and nothing between them earns its place.**
Notebook 7's `luma 24 + 2 blocks` where the device has the bandwidth (10 ms a 1080p frame
on an RTX 3060, 29 ms on a GTX 1650), and a 32-16-sized luma body where it does not (25 ms
on an Iris Xe or an M1, where the blocks take 54). Within the cheap tier the choice does
not move quality, so it should be made on what the browser measures, and on colour, which
the luma frame keeps at bicubic for nothing.

On the drawn desktop frame of notebooks 3 and 7 (no codec), `luma 24 + 2 blocks` reads
27.68 dB wpsnr and 0.9656 SSIM-Y against the control's 27.10 and 0.9609 and bicubic's 25.82
and 0.9437.

### What the checks caught

- **`training.assert_same_model` failed a correct merge on the GPU.** cuDNN runs float32
  convolutions in TF32 by default, which put 9e-4 between a reparameterized model and its
  merged form where the true difference is 1e-6. `training.full_precision()` now turns TF32
  off for the check — notebook 6's merge check had the same false alarm waiting for it.
- **Two strategies of the first sweep reach past the halo.** `recursive ×3` reaches 6
  pixels and `dense cascade` 5, against the 4 of the patches and of the client's tiles:
  trained, they would learn from zero padding, and tiled, they would seam. The notebook now
  asserts every strategy's measured halo against the patches', and every graph against the
  op budget; those two and `attention` are no longer in it.
- **One run of the control flatters everything.** The verdict first compared each
  strategy with seed 0 of the control, which happened to be the lowest of the three, and
  called two ties wins. It compares with the mean now. Every single-run comparison in this
  file has the same exposure.

Published to the page: `luma 24 + 2 blocks` and `32-16 leaky`, the best two graphs (the
reparameterized strategies merge into the same graphs as their plain twins, so they count
once), at the 320×180 step in float32, float16 and float16 as a whole 1080p frame
(`upscale_strategy_*`). The first is notebook 7's architecture with this run's weights, so
on the speed-only page it is notebook 7's graph twice; the second is a tie on quality, and
puts a `LeakyRelu` on the page, whose WebGPU cost is a browser question. Its float16 tile is
2.4× its float32 one on the desktop CPU runtime (7.08 against 2.92 ms), which is the CPU
provider having no half kernel for that operator, not a fact about the browser.

### The first sweep, on the synthetic set

Kept for the record: run before the stream-degraded set existed, on 12 synthetic frames
with a clean bicubic downscale, 76 validation patches, 100 epochs at 1e-3 on a cosine
schedule, best epoch by weighted PSNR, with L1 (RGB for the control, luma-weighted for the
rest). Its ranking does not carry over — reparameterization's lead was inside its noise, and
does not reproduce above — but its ruler, its loss result and its two bugs still stand.

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

#### The result is the ruler, not the ranking

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

#### What each idea was worth

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

#### Two bugs the notebook's own checks caught

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

## Desktop screens and a known stream (`8_upscale_screens.ipynb`)

Notebook 7's model (`luma 24 + 2 blocks`) on the patch set with desktop screens in both
splits (1,154 of the 3,523 validation patches), trained blind at three seeds, which is the
ruler, and then told how its frames were made: the bitrate, then the bitrate and the host's
encoder path (the downscale filter, and whether x264's loop filter was on). Same recipe as
notebooks 4 and 7; every run converged in 15–18 minutes on the RTX 3090. Notebook 4's model,
the same architecture and recipe trained on the set before it had screens, is scored on the
same set. A second seed of the path model and the two checks under "What a told graph
needs" were run afterwards from the notebook's saved runs; they are not in the notebook yet.

Against the mean of the blind model's three seeds (0.9348 SSIM-Y, 35.53 dB wpsnr), whose
spread is **0.0005 SSIM-Y and 0.07 dB wpsnr**:

| | ssim-y | vs blind | psnr-y | wpsnr | vs blind | psnr-u / -v |
|---|---|---|---|---|---|---|
| bicubic | 0.9049 | | 33.18 | 34.14 | | 50.67 / 51.46 |
| notebook 4's model, trained before screens | 0.9292 | −0.0056 worse | 34.45 | 35.33 | −0.20 worse | 50.70 / 51.49 |
| blind, 3 seeds | 0.9345–0.9350 | | 34.63–34.70 | 35.51–35.57 | | 50.71 / 51.49 |
| told the bitrate | 0.9349 | +0.0001 luck | 34.64 | 35.52 | −0.01 luck | 50.71 / 51.49 |
| **told the bitrate and path, 2 seeds** | **0.9369 / 0.9371** | **+0.0021 better** | **34.94 / 34.92** | **35.79 / 35.78** | **+0.25 better** | 50.72 / 51.50 |

### Screens were the missing data

Gain over bicubic, SSIM-Y / luma PSNR dB; blind is the mean of three seeds, told the mean
of two:

| | patches | bicubic ssim-y | before screens | blind | told bitrate + path |
|---|---|---|---|---|---|
| screens | 1,154 | 0.8851 | +0.043 / +1.75 | **+0.062 / +2.99** | +0.063 / +3.07 |
| screens at 100% display scale | 303 | 0.8504 | +0.052 / +1.17 | **+0.082 / +2.10** | +0.084 / +2.20 |
| screens at 125% | 278 | 0.8712 | +0.045 / +1.23 | +0.068 / +2.36 | +0.068 / +2.50 |
| screens at 150% | 387 | 0.8970 | +0.042 / +2.08 | +0.057 / +3.80 | +0.057 / +3.81 |
| screens at 200% | 186 | 0.9378 | +0.026 / +2.76 | +0.030 / +3.71 | +0.031 / +3.78 |
| photos | 784 | 0.8359 | +0.022 / +0.74 | +0.019 / +0.46 | **+0.024 / +0.73** |
| video | 1,585 | 0.9535 | +0.012 / +1.20 | +0.012 / +0.88 | **+0.014 / +1.30** |

- **Seeing screens is worth more than any idea in the strategy sweep.** +0.019 SSIM-Y and
  +1.24 dB luma on screens over the same model trained without them. It gains most at 100%
  display scale (+0.030 SSIM-Y), which is the smallest text and the commonest case for a
  1080p desktop streamed at half its height.
- **It is paid for on natural content.** Against the model trained without screens, photos
  lose 0.002 SSIM-Y and 0.28 dB, and video 0.32 dB at the same SSIM. The model is the same
  28k operations a pixel, and the screens take some of what it can learn.
- **The told model wins photos and video back, not screens.** Its gain over the blind model
  is +0.0045 SSIM-Y and +0.27 dB on photos and +0.0018 and +0.42 dB on video. On screens it
  is +0.0009 and +0.07 dB, inside the screens' own seed spread (0.0012 and 0.34 dB).

### What knowing the stream is worth

- **The bitrate is worth nothing.** +0.0001 SSIM-Y overall, and nothing in any bitrate
  bucket either. The model reads the bitrate off the picture itself.
- **The encoder path is worth +0.0021 SSIM-Y and +0.25 dB wpsnr**, four times the seed
  spread, and it holds at both seeds. Against the blind run at the same seed (same start,
  same batches, so the gap is the condition alone) it is +0.0018 and +0.0021 SSIM-Y, and
  +0.31 and +0.22 dB luma. By filter, bilinear gains SSIM (+0.0031, no luma PSNR) and area
  and bicubic gain luma PSNR (+0.42 and +0.34 dB).
- **On what the host sends today** (736 patches: swscale's bicubic, x264 without its loop
  filter) it is +0.0019 SSIM-Y, against a blind spread of 0.0010 there, and **+0.42 dB
  luma**, against 0.07. That row is the whole case for a told graph, because it is what a
  client receives.

### What a told graph needs

The told model ships baked: one stream's condition folded into the biases, one
single-input graph. So the question is what a baked graph does on a stream it was not baked
for. Both checks use the told model at both seeds, which agree to within 0.0015 SSIM-Y and
0.18 dB in every cell shown; the tables give their mean.

**The filter has to be right.** The graph baked for the host's path today, on each path a
stream can take, against the blind model and against being told the truth (gain over
bicubic, SSIM-Y / luma dB):

| true path | patches | blind | told the truth | baked for the host today |
|---|---|---|---|---|
| bilinear | 622 | +0.0349 / +3.02 | +0.0372 / +3.07 | +0.0291 / +2.57 |
| bilinear, loop filter | 442 | +0.0367 / +2.08 | +0.0408 / +1.97 | +0.0291 / +1.82 |
| **bicubic** (the host today) | 736 | +0.0303 / +1.05 | **+0.0323 / +1.48** | **+0.0323 / +1.48** |
| bicubic, loop filter | 440 | +0.0381 / +1.07 | +0.0380 / +1.26 | +0.0374 / +1.27 |
| area | 473 | +0.0229 / +1.02 | +0.0244 / +1.50 | +0.0234 / +1.22 |
| area, loop filter | 474 | +0.0172 / +0.70 | +0.0182 / +1.04 | +0.0144 / +0.70 |

- **A wrong filter is worse than no condition at all.** The host-today graph on a bilinear
  stream lands 0.006–0.008 SSIM-Y and 0.26–0.45 dB under the blind model. The other way
  round is worse. Told bilinear when the stream is bicubic, the model drops to +0.11 dB;
  with the loop filter also claimed, it drops to −0.33 dB, and told the same on an area
  stream, to −1.1 dB. That is under bicubic itself: the model sharpens for a blur that is
  not there.
- **The loop filter barely matters.** Within the right filter, getting it wrong costs at
  most 0.0018 SSIM-Y and 0.19 dB.
- **No single path beats the blind model on a mix of streams.** On all coded patches the
  best fixed path (bicubic with the loop filter) reads +0.0290 SSIM-Y, against +0.0301 for
  the blind model and +0.0319 for the told model given each stream's own path. A told graph
  is worth shipping only to a client that knows which path its host runs.

**One bitrate is enough.** On every coded patch, given its true path, told the true bitrate
or baked at the three bitrates the notebook publishes:

| bitrate | patches | blind | true bitrate | baked at 0.2 | baked at 0.46 | baked at 1.0 |
|---|---|---|---|---|---|---|
| 0.2–0.3 bpp | 831 | +0.0301 / +1.42 | +0.0308 / +1.60 | +0.0300 / +1.57 | **+0.0314 / +1.61** | +0.0281 / +1.45 |
| 0.3–0.5 bpp | 1,080 | +0.0302 / +1.39 | +0.0325 / +1.62 | +0.0315 / +1.50 | **+0.0324 / +1.63** | +0.0293 / +1.52 |
| 0.5–1.0 bpp | 1,276 | +0.0300 / +1.71 | +0.0322 / +2.01 | +0.0298 / +1.79 | **+0.0321 / +1.99** | +0.0316 / +1.97 |

The graph baked at the host's default 0.46 bpp is within 0.0006 SSIM-Y and 0.02 dB of the
true bitrate in every bucket. The 0.2 and 1.0 graphs are worse than it even at their own
bitrates. That is the same finding as "told the bitrate" above, seen from the other side:
the model reacts to the bitrate it is told, so a wrong one costs, but the right one gains
it nothing over the default.

**What this means for what ships:**

- **One told graph, not three.** `upscale_screens_24x2_bpp0.2_*` and `_bpp1_*` (four of the
  seven files) can only be worse than the 0.46 graph. `BAKE_BPP` should be the host's
  default alone, and a told model trained on the path alone is the simpler thing to try.
- **The blind model has to be on the page too, and it is not.** The notebook exports only
  its winner, which is the told model. The blind model is the safe graph for a client that
  does not know its host's path, and the right one on any bilinear stream. Its best seed
  by SSIM-Y is seed 0 (0.9350).
- **The told graph is tied to the host's ffmpeg line.** It is right only while the host
  downscales with swscale's bicubic. Fixing the Windows hardware lines (under Next) hands a
  Windows client the D3D11 video processor's filter, whose kernel is not measured and need
  not be any of the three the model was trained on, so no baked graph is known to be right
  for that stream. The host would have to say
  which path it runs, and the client would need the blind graph wherever the answer is not
  a path the model knows.

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
`5_upscale_geometry.ipynb` below sweeps the batch and the rectangle as well, charges everything
per output pixel, and lands on **640×360 out, from a 320×180 step** — 6.8 ms a 1080p frame
against this table's 8 ms at 128, and the same tile divides 720p, 1440p and 4K exactly too.
The `ms / frame` column here is also missing what it costs to *not* divide the frame: a 256
step computes 52% more of a 1080p output than that output has.

The rest of the original argument for 128 still holds: 84% of computed pixels kept at halo 6, a power of two once the pixels live in textures
and workgroups, bounded memory independent of stream resolution, and 40 tiles per 1080p
frame so the main thread yields 40 times instead of blocking once. That sweep was run
against a model that has since been removed, so the timings behind it are gone; the geometry
argument is not.

## The geometry a client runs at (`5_upscale_geometry.ipynb`)

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
session, every stream. `tiles.plan_tiling` returns it for all four, and falls back to a
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
uv run src/upscale/benchmark/main.py   # from the repo root, then open http://127.0.0.1:8000
```

`main.py` serves `www/`, serves `dist/upscale/` under `models/`, and answers
`GET /api/models`; the page cannot run from `file://` because ONNX Runtime Web fetches its
own `.wasm` at runtime. **The dropdown is built from the folder, not from a file describing
it.** Every request walks `dist/upscale/*.onnx` and reads
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
| `upscale_strategies` | luma 24 + 2 blocks and 32-16 leaky, fp32 / fp16, and fp16 as a whole 1080p frame | 328×188 / 968×548 | 4 |

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

`dist/` is committed, so the page has models on a fresh clone; a notebook's export cell
replaces the ones it owns.

## Where that leaves it

| | Best static | `upscale_nn` | **`upscale_web`** | `upscale_strategies` |
| --- | --- | --- | --- | --- |
| PSNR (RGB), validation patches | 23.62 dB | 24.17 dB (12 epochs) / 31.25 dB (100) | **30.28 dB** | 35.67 dB † |
| weighted PSNR, validation patches | — | — | — | **38.20 dB** † |
| 1080p frame, WebGPU | 2 ms | 18 ms | **7 ms**, or 5 ms at step 256 | not measured yet |
| 1080p frame, WASM | 831 ms | 1798 ms | 396 ms | not measured yet |
| parameters | 0 | 26,796 | **9,696** | 13,156 |
| inside the op budget | no (`Resize`) | no (`Resize`) | **yes** | **yes** |

† `upscale_strategies` is measured on the stream-degraded set (0.2–1.0 bpp H.264), where
bicubic reads 35.00 dB RGB and 37.30 dB weighted; the other columns are the old synthetic
set, so that column is not comparable with its neighbours.

The frame figures in that table are all at the published 128 step, which
`5_upscale_geometry.ipynb` has since shown to be the most expensive geometry of the ones it
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

- **Publish what notebook 8 found.** Export the blind screens model next to the told one,
  bake the told one at 0.46 bpp alone (drop the 0.2 and 1.0 graphs), and fold the second
  seed and the two checks (a wrong path, a fixed bitrate) into the notebook, which does not
  contain them yet. Then try a model told the path alone, without the bitrate.
- **More validation frames.** Moved: 2,369 patches from 149 sources put three seeds of
  the 32-16 0.13 dB wpsnr and 0.0004 SSIM-Y apart, against 0.61 dB on two frames. It is
  still the constraint on the questions left — every 32-16-sized idea in the strategy sweep
  is inside it — so a decision between bodies of one size needs more seeds, not more ideas.
- **Re-point `6_upscale_reparam.ipynb`.** It trains the model the first strategy sweep
  chose, and the sweep re-run on its own data finds reparameterization worth nothing
  measurable. The candidates are now two tiers: `luma 24 + 2 blocks` and a 32-16-sized luma
  body.
- **Fix the host's Windows hardware lines.** `stream-ffmpeg.js` in desktop-streamer asks
  `scale_d3d11` for `w=`/`h=`; the filter's options are `width`/`height` (checked with the
  2026-04 gyan.dev build in `C:\Users\henrik\ffmpeg`, which rejects `w` outright). So
  every scaled share on Windows should fail its NVENC and AMF lines and fall through to
  libx264, which is why notebook 8 calls swscale's bicubic with no loop filter "what the host
  sends today". Its tests only compare argument arrays, so they cannot see it. Fixing it
  changes the downscale filter a Windows client receives to the D3D11 video processor's,
  whose kernel is not measured yet (`hwupload` frames are refused by `scale_d3d11`; it needs
  `gfxcapture` or a hardware decode in front of it). It also leaves the path notebook 8's
  told graph is baked for, and a told graph given the wrong filter is worse than the blind
  one, so the host has to tell the client which path it runs.
- **Measure the two tiers in the browser, on a laptop iGPU.** The tier boundary — 54 ms
  against 25 ms a 1080p frame on an Iris Xe — is a roofline estimate calibrated on an RTX
  3090; whether the blocks model misses 30 fps there is the page's question.
- **Re-run `2_upscale_nn.ipynb` at a real training budget, and re-measure it and
  `1_upscale_dummy.ipynb` through `helpers/metrics.py`.** Its 12 epochs are 96 optimizer steps, and
  every quality number attributed to it is a number about that; and both notebooks still
  report RGB PSNR, so neither can be compared with anything measured since.
- **Real data, which is now the only thing standing between this and a decision.** Screen
  captures instead of synthetic frames, and a codec round trip in the degradation: a model
  trained on clean bicubic pairs has never seen blocking or ringing, and a +7 dB that comes
  from inverting a known blur will not survive contact with one it has not seen.
- **Skip the tiles that did not change**, in whatever ends up driving inference in desktop-streamer's `src/`.
  It is the largest remaining factor on desktop content, it is exact rather than
  approximate, and the model needs no change for it.
- **Re-export everything at 320×180.** The step is decided — `5_upscale_geometry.ipynb`
  measured it — and every published graph is still at 128, including the one a client would
  run. `TILE_STEP` itself stays 128 while it is what the page compares models at.
- The same runtime tables on a GPU for training and for the desktop client. The browser
  side is now measured on real hardware; torch and onnxruntime here are still CPU builds.
- Tiled inference in the client, so a frame larger than memory can be upscaled.
- Then a model worth the name, with both baselines kept as the numbers to beat.

Sibling problems: `../frame_gen_intra/summary.md`, `../frame_gen_extra/summary.md`.
