# Frame generation, extra — summary

Extrapolation: given the frames the stream has delivered so far, generate the next one
before it arrives, so a client can keep drawing through a stall and a dropped frame does
not read as a freeze.

> **Status: four notebooks, and one graph ready for the client: notebook 3's tile search,
> measured in the browser.** Nothing in desktop-streamer's `src/` calls any of it; the client
> still runs the mock graph.
>
> **The best exportable extrapolator is a search, not a network.** Notebook 3 rebuilds
> notebook 1's tile shift from fixed convolutions: the cost of every offset within ±8 px on
> half-resolution luma, sampled on every second position and averaged over the tile, and a
> softmax to pick one motion, applied with `GridSample`. Nothing in it learns. With the
> client's gate (repeat where nothing moved) it scores **31.20 dB on the changed samples
> (+3.99 over repeat, 0.8127 SSIM-Y) and 72.6 dB on the unchanged ones**, +1.34 dB over repeat
> on everything. At the client's halo of 4 it keeps 30.59 dB. **In Chrome's WebGPU on the
> RTX 3090 it costs 2.3 ms of GPU time a 1080p frame** (36 tiles in one run), every operator
> on the GPU, `GridSample` included: under notebook 2's network (3.6 ms) and a tenth of the
> first form of the search (13.9 ms). It is in `dist/frame_gen_extra/`, as one tile a run
> and as a whole 1080p frame a run.
>
> **The mock the client runs is worse than doing nothing.** Its linear extrapolation,
> `2 · current − previous`, scores **24.74 dB wPSNR against 27.21 dB for repeating the newest
> frame** on the 1,852 validation samples where something changed, and loses on every kind
> of source: a moved edge is drawn twice, once too dark and once too light. On the samples
> where nothing changed it loses 8.9 dB. It should not be switched on as it is.
>
> **Continuing the motion pays, where the motion is a scroll.** Matching `previous` to
> `current` at every whole-pixel offset and moving `current` once more gains **+5.06 dB
> (0.8230 SSIM-Y against 0.7312)** on the changed samples: +11.9 dB on scrolled pages, +9.5 dB
> on panned photos, +1.3 dB on video. On a desktop - a cursor, a dragged window, a menu,
> a region scrolling - nothing beats repeating the frame yet.
>
> **The client's 4-pixel halo costs a third of that.** Searched as far as the halo lets it,
> the same method gains +3.24 dB at a halo of 4, +4.79 dB at 8 and +5.06 dB at 16: 41% of
> the scrolls and pans move more than 4 stream pixels a frame.
>
> **No trained network beats it.** Asked for pixels (notebook 2) or for motion (notebook 3),
> a network trained on two frames hedges, and on a desktop that means blur.
>
> **A network asked for pixels learns to blur, not to move.** Notebook 2's two-scale network
> (5,775 operations a pixel, all in the browser's fast set) trains to **+1.14 dB (0.7628
> SSIM-Y)** on the changed samples - a quarter of what the untrained shift gets. It hedges:
> scrolled text and a moving cursor come out faded instead of moved. Its one win is the desktop
> and cursor samples, where it edges past both repeat and the shift.

**One notebook is one model**, as in `../upscale/`: a notebook owns its method, its numbers
and its export cell, and publishes into `dist/frame_gen_extra/` at the repo root.

## What makes it its own problem

This is the one a live remote desktop session can actually use: it adds no latency, because
it needs nothing that has not arrived yet. It is also the harder of the two. The answer is
bounded on one side only, so the model is guessing rather than filling in, and it is wrong
whenever the user does something unpredictable — which on a desktop is most of what they
do. A click that opens a menu is unforeseeable from the frames before it.

Being wrong here is visible in a way the other two problems are not: an extrapolated frame
is shown and then contradicted by the real one. That makes the failure mode — how the guess
is corrected when the true frame lands, and whether a wrong guess is worse than a repeated
frame — a bigger question than the model architecture.

The cursor is the case worth watching. It moves constantly and predictably, and dragging it
smoothly through a stall is most of the perceived benefit, so a model that gets nothing else
right but the pointer may still be worth shipping.

## Notebooks

Run in order, from this folder:

- `00_data_preprocess.ipynb` — the run set. Sources are the drawn desktop screens
  (`datasets/screens.py`, 400 train and 80 validation), 400 YouTube sequences and 125 DIV2K
  photos, and `datasets/raw/captures/` (synthesised desktop frames when all of that is
  empty). A YouTube sequence is a clip as it is. A still picture becomes a one-second clip of
  a 1280×720 window **animated as a desktop in use** (`helpers/scenes.py`): a page scrolls by
  wheel notches, a flick or a steady drag, a photo pans, a desktop holds still while a region
  inside it scrolls, a window is dragged along the cursor's path or a menu pops up at the
  cursor, and an arrow pointer rests and moves along eased, slightly bent paths. Then the
  stream's own ×2 degradation from `helpers/degrade.py` (the same as upscale's: random filter,
  H.264 at 0.2–1.0 bpp, loop filter for half, no codec for a tenth) → runs of 8 consecutive
  decoded frames of one spot, a 64 px core with a halo of 16, at even places. Two runs per
  clip are cut around the cursor; of the others, flat ones are dropped and ones where nothing
  moves are kept a fifth of the time. Split by source → one `uint8` `.npy` array of
  `(runs, 8, 96, 96, 3)` per split and a `manifest.json` in `datasets/frame_gen_extra/` that
  records each source's scene, degradation and the spot and tag of each run. 12,491 train
  and 3,226 validation runs, 3.5 GB, 16 minutes. `helpers/runs.py` reads it back.
- `01_extrapolate_dummy.ipynb` — **static methods only.** Repeat, linear (the mock), half step,
  and three motion shifts (one offset per tile, per 16 px block, per 8 px block). No
  training; it measures quality by group and by halo, how far content moves, and speed, and
  exports the three ONNX can express as plain arithmetic at the client's tile.
- `02_extrapolate_twoscale.ipynb` — **the first network**, built for the browser: `current`
  plus a correction made at 1/2 and 1/4 of the resolution (`models.TwoScale`), beside a
  control of the same cost with no downsampling (`models.Plain`). Trained at a halo of 8,
  scored at 8 and at the client's 4, checkpointed to `checkpoints/extrapolate_twoscale.pt`,
  and exported at the client's tile - into `checkpoints/export/` until `PUBLISH` is set
  once a model is worth shipping. Trained to convergence it gains +1.14 dB on the changed
  samples, a quarter of notebook 1's shift; see "The two-scale model".
- `03_extrapolate_motion.ipynb` — **motion, not pixels**: `current` moved by an offset per pixel
  with `GridSample` (`models.Motion`). The offset comes from a learned head (notebook 2's
  encoder), from `models.TileSearch` (notebook 1's tile shift built from fixed convolutions,
  nothing learned), or from both. Also scores every model through `models.Gate`, the client's
  rule of repeating tiles where nothing moved. Publishes the tile search (stride 2) to
  `dist/frame_gen_extra/`, one tile a run and a 1080p frame a run; see "Motion", which also
  has its browser timings.

The code the notebooks share lives in `helpers/`. `sources.py`, `degrade.py` and `metrics.py`
are copies of upscale's, so a change made for one problem cannot quietly change the other's
data or units. `scenes.py` is the desktop motion. `runs.py` reads the run set back and holds
the one definition of a changed sample. `models.py` holds the static methods with the
client's two-input contract. `training.py` handles loading and grouped scoring; the training
loop joins it with the first notebook that trains. `speed.py` is the timing. `webexport.py`
is the two-input export into `dist/frame_gen_extra/`.

## The data

**Input and target are both decoded frames.** The client extrapolates from what it decoded,
and the frame a guess stands in for is a decoded one too, so there is one array and no
clean target. Everything is at the stream's resolution (640×360 from a 1280×720 window): the
guess is made before upscaling.

**A sample is `context` frames and the one `ahead` after them**, cut from a run of 8, so a
later notebook can try more history (up to 7) or a longer stall (up to 6 ahead) without new
data. The client's contract is 2 and 1: `previous` and `current` in, the next frame out.
Training samples a random newest frame; validation fixes one per run, spread over the run
by its index.

**Scores are reported over the samples that changed.** A sample is *changed* if more than
0.5% of its core's pixels move by more than 8 luma levels from the newest input to the target
(`runs.CHANGED_LEVEL`, `runs.MOVING_SHARE`), which is above the codec's flicker on a still
picture. 1,852 of the 3,226 validation samples changed. Over all samples repeating is at 47.5
dB, and on the unchanged ones at 74.8 dB: a score that mixes them flatters everything.

How often the newest frame of a training sample is followed by a change, on a random 3,000
of the train runs:

| | runs | changed |
| --- | --- | --- |
| page | 761 | 61% |
| desktop | 669 | 39% |
| photo | 411 | 70% |
| video | 1,159 | 70% |

Of the 798 train clips, 417 have a cursor, 111 a menu, 106 a dragged window, 83 a scrolling
region, and 255 a scrolling background (99 wheel, 97 flick, 59 steady). The runs are 834 cut around the cursor, 9,464 that move and 2,193 still ones in train
(170, 2,540 and 516 in validation). Frames one apart differ by 9.3 luma levels on average
and frames two apart by 13.1, which checks that the codec round trip kept them in order.

## Static methods

Notebook 1, on the 3,226 validation samples at their fixed frame, scored on the 64 px core
(`metrics.report`: wPSNR is Y'CbCr weighted 6:1:1, SSIM on luma). The shift methods search
every offset within the halo (16).

| changed samples (1,852) | wPSNR | SSIM-Y | vs repeat |
| --- | --- | --- | --- |
| repeat | 27.21 | 0.7312 | |
| linear (the mock) | 24.74 | 0.6728 | −2.47 dB |
| half step | 26.13 | 0.7021 | −1.07 dB |
| **shift, tile** | **32.26** | **0.8230** | **+5.06 dB** |
| shift, 16 px blocks | 31.49 | 0.8166 | +4.29 dB |
| shift, 8 px blocks | 30.57 | 0.8111 | +3.37 dB |

On the 1,374 unchanged samples repeat is at 74.80 dB and every other method loses: shift by
2.3–4.2 dB (a scroll that stops is carried on one frame too far), linear by 8.9.

By kind of source, changed samples only (SSIM-Y, then wPSNR):

| | page (454) | desktop (167) | photo (292) | video (939) | cursor runs (118) |
| --- | --- | --- | --- | --- | --- |
| repeat | 0.6418 · 23.53 | **0.7817 · 25.81** | 0.5479 · 24.28 | 0.8225 · 30.14 | 0.7148 · 24.20 |
| linear | 0.5900 · 20.57 | 0.7384 · 22.98 | 0.4872 · 21.47 | 0.7590 · 28.08 | 0.6634 · 21.50 |
| shift, tile | **0.7794 · 35.46** | 0.7541 · 25.65 | **0.8462 · 33.78** | **0.8491 · 31.42** | **0.7710 · 25.53** |
| shift, 16 px blocks | 0.7733 · 33.25 | 0.7561 · 25.15 | 0.8338 · 32.57 | 0.8430 · 31.44 | 0.7659 · 25.41 |

What it says:

- **Brightness extrapolation is the wrong model for a screen.** Nothing on a desktop fades.
  It moves, appears or holds still, and `2 · current − previous` is wrong for all three.
  Half a step is less wrong and still worse than repeating.
- **A scroll is solved by a translation.** Pages and photos gain 10–12 dB from one offset per
  tile, and smaller blocks only add false matches (repeating text lines match each other).
- **A desktop is not.** Its changes are small and local - a cursor, a dragged window, a menu,
  one region scrolling - and every method here is at or under repeat on them. The cursor runs
  gain only because many are on scrolling pages. A block holding the arrow also holds the
  still background around it, so one offset cannot move the one without the other, and a
  menu's content is in neither input frame. This is where a learned model has to earn its
  place.
- **Integer offsets are a floor, not a ceiling.** The host moves content by whole pixels at
  1280×720, which the ×2 downscale makes half-pixels. A sub-pixel shift, or a model that learns
  one, should gain more than these numbers.

### The halo

The client's tile has a halo of 4. Content that moved further than that in one frame came
from outside the tile, and no tile-local method can draw it.

| changed samples | halo 4 | halo 8 | halo 16 |
| --- | --- | --- | --- |
| shift, tile | 0.7930 · 30.45 dB | 0.8180 · 32.00 dB | 0.8230 · 32.26 dB |
| shift, 16 px blocks | 0.7871 · 30.16 dB | 0.8057 · 31.13 dB | 0.8166 · 31.49 dB |
| repeat | 0.7312 · 27.21 dB | | |

Of the 746 changed page and photo samples, the offset the tile method finds is 0 for 29.6%
(the change was a cursor or a menu, or under a pixel), at most 4 px for 58.7%, at most 8 for
77.5% and at most 15 for 95.4%. A halo of 8 keeps 95% of the gain of 16. For upscaling,
`../upscale/` chose the tile's halo and every pixel of halo is recomputed work. An
extrapolator only needs it on its input, so a larger halo for this model alone is worth
measuring on the client.

### Speed

In torch on the RTX 3090, one stream frame: repeat is free, and linear and half step cost
0.08 ms at 960×540 and 0.3 ms at 1920×1080. The shift search is 1,089 offsets at a radius of
16, each a few small torch calls, so it takes 60–300 ms almost whatever the frame size. That
is launch overhead, not what a fused kernel would cost, and it is not exportable as it is.
On the desktop CPU through onnxruntime, at the client's 328×188 tile: repeat 0.03 ms, linear
0.08 ms, half step 0.30 ms.

### Published

`dist/frame_gen_extra/` holds three graphs, `extrapolate_static_{repeat,linear,half_step}_tile320x180.onnx`.
They have the mock's contract (`previous` and `current` in, `output` out, float32 NCHW) at
the client's tile, with label, step, halo and CPU time in their metadata. Repeat is one
`Identity`. The exporter drops an input the model never reads, so `webexport.export` puts
`previous` back, unused. Linear and half step use `Mul`, `Sub` and `Clip`, outside the
operator set upscale measured as fast. The mock's own docstring times them at 1–2 ms a 1080p
frame each.

Nothing serves them yet: `../upscale/benchmark/` lists only single-input upscalers.

## The two-scale model

Notebook 2. The two frames side by side, two stride-2 3×3 convolutions (16 and 32
channels), three 3×3 convolutions at a quarter of the resolution, `DepthToSpace` back to
half with the 16 channels beside it, a 3×3 convolution to 12 channels and a last
`DepthToSpace` to the 3 colours, added to `current`. 35,868 parameters and 5,775 operations
a stream pixel. Its last convolution starts at zero, so untrained it *is* repeat (checked:
the same scores to 1e-6). It reaches about ±20 px, enough for 95% of the scrolls. The control
(`Plain`, 8 channels, four 3×3 convolutions at full resolution) costs 5,947 operations a
pixel and reaches ±6.

Both graphs are `Conv`, `LeakyRelu`, `Concat`, `DepthToSpace` and `Add` only. The two-scale
graph runs a 328×188 tile in 1.2–2.8 ms on the desktop CPU, the control in 8.1 ms. In torch on
the RTX 3090 a 1920×1080 stream frame costs 3.8 ms and 11.8 ms. At the same operation
count, full resolution is three times slower.

Trained at a halo of 8 with `luma_chroma_l1`, Adam from 1e-3, keeping the epoch with the best
SSIM-Y on the changed samples, until it converged: 124 epochs (5 minutes) for the two-scale
network, best at epoch 100; 150 epochs (7.6 minutes) for the control, best at 144. Both sit
flat on repeat for their first ~15 epochs: with the last convolution at zero, nothing before
it gets a gradient until that convolution has grown.

| changed samples (1,852), halo 8 | wPSNR | SSIM-Y | vs repeat |
| --- | --- | --- | --- |
| repeat | 27.21 | 0.7312 | |
| **shift, tile** (notebook 1, radius 8) | **32.00** | **0.8180** | **+4.79 dB** |
| two-scale | 28.35 | 0.7628 | +1.14 dB |
| control | 28.03 | 0.7546 | +0.83 dB |

| changed, SSIM-Y · wPSNR | page (454) | desktop (167) | photo (292) | video (939) | cursor (118) |
| --- | --- | --- | --- | --- | --- |
| repeat | 0.6418 · 23.53 | 0.7817 · 25.81 | 0.5479 · 24.28 | 0.8225 · 30.14 | 0.7148 · 24.20 |
| shift, tile | **0.7733 · 34.71** | 0.7564 · 25.35 | **0.8220 · 33.37** | **0.8494 · 31.45** | **0.7553** · 25.23 |
| two-scale | 0.6668 · 24.86 | 0.7804 · 26.29 | 0.6638 · 26.92 | 0.8367 · 30.84 | 0.7331 · **25.42** |
| control | 0.6798 · 25.31 | **0.7879 · 26.59** | 0.6059 · 25.38 | 0.8310 · 30.44 | 0.7300 · 25.37 |

What it says:

- **The network learned to hedge, not to move.** On a scrolled page it gains 1.3 dB where a
  translation gains 11.2. Its pictures show why: the lines of a scrolled page fade where
  they were instead of appearing where they go, and a moving cursor turns into a faint
  ghost. Under L1, a network unsure where something goes draws a little of it everywhere it
  might go. Making a pixel of output equal a pixel 12 px away in the input is easy to
  describe and hard for 3×3 convolutions to learn from a zero start.
- **The quarter-resolution path earns a little.** +0.32 dB over the control overall, and
  much more on photos (+1.5 dB), where the motion is largest. The control does better on
  pages and desktops, where most of the change is near and small. Neither difference is
  worth an architecture on its own.
- **The halo is not what holds it back.** The same weights at the client's halo of 4 score
  28.29 dB against 28.35 at 8: the network never learned motion far enough for the halo to
  matter.
- **It damages still tiles, and a gate fixes that for free.** On the unchanged samples the
  networks fall from repeat's 74.8 dB to 60–61: faint noise everywhere. Running the model
  only where `previous` and `current` differ (`runs.changed_share` over the tile), and
  repeating elsewhere, brings the two-scale network back to 70.9 dB on the unchanged samples
  and leaves its changed score as it was. The gate fires on 1,650 of the 1,852 changed
  samples and 302 of the 1,374 unchanged ones, and the client would save the model's work on
  every tile it skips. Measured outside the notebook, with the trained checkpoint.

## Motion

Notebook 3. Notebook 2's network gained 1.3 dB on scrolled pages where a translation gains
11.2: asked for pixels, it has to *draw* moved content, and it hedges by blurring. So here the
output is an offset per pixel, and `current` is read there (`models.warp`, a `GridSample` with
the edge repeated). A correct offset copies the content exactly. Three ways to get the offset,
all trained and scored at a halo of 8 like notebook 2:

| changed samples (1,852), halo 8 | wPSNR | SSIM-Y | vs repeat | ops a pixel |
| --- | --- | --- | --- | --- |
| repeat | 27.21 | 0.7312 | | 0 |
| exact shift (notebook 1, not exportable) | 32.00 | 0.8180 | +4.79 dB | |
| two-scale network (notebook 2) | 28.35 | 0.7628 | +1.14 dB | 5,775 |
| learned motion | 28.02 | 0.7651 | +0.81 dB | 5,342 |
| tile search, stride 1 | 31.39 | 0.8148 | +4.19 dB | 1,880 |
| **tile search, stride 2** (published) | **31.20** | **0.8127** | **+3.99 dB** | **479** |
| search + correction | 31.20 | 0.8127 | +3.99 dB (untrained is best) | 5,821 |

- **Learned motion** - notebook 2's encoder with a head that makes the offset, starting at zero
  (repeat). It trains for 164 epochs (6.6 minutes, Adam from 1e-4, best epoch kept by wPSNR on
  the changed samples) to +0.81 dB, under notebook 2's network. It does not find motion. In
  the prototypes, neither a multi-scale loss nor notebook 1's shift as a teacher for its offsets
  changed that. Nor did a cost volume handed to it, or an untrained per-pixel soft-argmin over
  that cost volume (all at most 28.5 dB after 20 epochs, pages at most 25.8). **Per pixel, a desktop's motion is ambiguous:** text lines repeat and white matches white. What makes notebook 1's shift right is
  adding the matching cost over the whole tile before choosing.
- **Tile search** (`models.TileSearch`) - that, in a graph. Luma of each frame at half the
  resolution (a fixed 2×2 stride-2 `Conv`); `previous` moved by each of the 81 offsets within
  ±4 there (±8 px) - one fixed `Conv` from 1 channel to 81, each kernel a single 1 - and
  `current` at the kernels' centre, both with no padding, so only positions where every offset
  reads inside the tile count; `Sub` and `Abs` make |current − previous moved by d|. Then the
  mean over the tile (`ReduceMean`), a softmax of the 81 costs times a sharpness of 10 (one
  luma level of difference is a factor e^10), and the weighted mean of the offsets is the
  tile's motion. Nothing learns. Sharpness 3 lands between offsets and blurs scrolled pages
  (32.50 dB on them), 30 is close to an argmin (33.48), and 10 is best overall. At stride 1 it
  is 0.6 dB under the exact search at full resolution, the price of half resolution. Full
  resolution at a radius of 8 scored 31.94 in the prototypes, for four times the work.
  `stride` samples the cost on a coarser grid: the mean over a tile needs a few thousand
  positions, not every one. Stride 2 costs 0.19 dB here and stride 4 0.53. These 80 px samples
  have 16×16 positions at stride 2; the client's tile has 78×43, so the loss there should be
  smaller. Stride 2 is published.
- **Search + correction** - the tile search plus the learned head's correction per pixel,
  for what one motion a tile cannot follow. It starts as the search, and **no epoch beats
  that**: the correction drifts from the first epoch (31.09 dB after one, 30.04 at
  convergence, scrolled pages 33.4 → 28.4 in the prototypes) while desktops and cursors gain
  a few tenths. The same hedge as notebook 2's, through the warp this time. A fractional
  offset reads between pixels, and that is a blur. When two frames cannot tell whether a wheel
  notch is easing in or out, a slightly short and blurred motion is the better bet under L1.
  `train` keeps the untrained model, so its scores are the search's.

**By kind** (changed samples, SSIM-Y · wPSNR, stride 2): the tile search is 0.7630 · 33.05
on pages, 0.8137 · 31.83 on photos, 0.8462 · 31.17 on video. Like every motion method, it is at
or under repeat on desktops (0.7580 · 25.23 against 0.7817 · 25.81). On cursors it is mixed:
better structure but lower wPSNR (0.7539 · 24.98 against 0.7148 · 24.20).

**The gate** (`models.Gate`) repeats a sample where `previous` and `current` do not differ by
the run set's own definition of a change. It leaves every changed score as it was: the gate
lets the model run on 1,650 of the 1,852 changed samples. The other 202 are changes that start
after the newest frame - a scroll beginning, a menu opening - which no model could predict.
It lifts the unchanged ones: the tile search from 69.56 to 72.56 dB (stride 2), notebook 2's
network from 60.33 to 70.93. Over every sample the gated tile search is +1.34 dB over repeat,
the only exportable model here that beats repeat on all samples together.

**At the client's halo of 4**, the same search reaches past the halo and reads the edge
repeated there, and still scores 30.59 dB (0.8109, stride 2; 30.77 at stride 1), above the
exact shift limited to ±4 (30.45). Searching only ±4 px scores 30.06.

**Speed, in the browser.** `../upscale/benchmark/main.py --models <folder>` serves any folder
of graphs, and the page now fills every input a graph has, so it runs two-input models. Opened
as `?auto=<ids>`, it measures them with no one at the controls and posts each row to the
server. That is how these were taken: headless Chrome 156 on Windows, WebGPU, the RTX 3090
(the adapter reports NVIDIA Ampere, not a software fallback), GPU-resident tensors, the output
buffer reused, GPU time from timestamp queries. One 1080p stream frame is 36 tiles in one run:

| graph, one 1080p frame | GPU ms | wall ms |
| --- | --- | --- |
| linear (the mock) | 0.25 | 0.27 |
| two-scale network (notebook 2) | 3.6 | 3.7 |
| tile search, first form (a 162-channel 9×9 `Conv`, two taps of 81 used) | 13.9 | 15.0 |
| tile search + correction, first form | 16.8 | 18.2 |
| tile search, stride 1 | 5.7 | 5.9 |
| **tile search, stride 2** (published) | **2.2–2.3** | **2.3–2.4** |
| tile search, stride 4 | 1.4 | 1.4 |

Every operator runs on the GPU, `GridSample` included: GPU time and wall clock agree. One
tile a run costs 0.18 ms of GPU but 0.29 ms of wall clock, so 36 separate runs are 10.5 ms a
frame: the client should run a frame's tiles as one batch, as it already does. Graph capture
changes nothing at batch 36. The first form spent its time on a dense convolution whose kernels
used 2 taps of 81. Moving `previous` with a 1-to-81 `Conv` and subtracting is the same
arithmetic for a quarter of the work, and stride 2 quarters it again. The graph is 34 KB, and
on the desktop CPU it takes 1.5 ms a tile.

**What it is not measured on.** It finds one motion for the whole tile. The samples are 80 px;
the client's tile is 328×188, which can hold a scrolled page beside a still panel, and then one
motion is wrong for part of it. Whole frames are the measurement that settles it.

## Open questions

- **Is the cursor in the stream?** The data draws it into every frame that has one, as a
  screen capture with the pointer on does. If desktop-streamer sends the pointer apart and
  the client draws it, the client already knows where it is, and the case the problem
  statement calls the most valuable is solved without a model. The scenes record whether a
  clip has a cursor, so it can be dropped from the data by filtering.
- **How long is a stall?** Everything here is one frame ahead. The runs hold up to six; the
  cost of a guess grows with each, and so does the question of when to stop guessing and
  hold the frame.
- **What does a wrong guess cost against a repeated frame?** A repeat is a freeze; a wrong
  guess is a jump back when the real frame lands. The scores above treat both as pixel error.
- **The scenes are drawn, not recorded.** Their speeds, pauses and event rates are guesses
  about how people use a desktop. A few minutes of real screen recording would check them,
  and there is no reader for one yet.

## Next

1. **Put it in the client.** Copy `extrapolate_tilesearch_r4_s2_tile320x180_b36.onnx` (a 1080p
   frame of tiles in one run) or the one-tile graph over the mock's `extrapolate.onnx`, and add
   the gate: compare `previous` and `current` per tile and repeat the tiles where nothing moved
   without running the model. Then measure it where it matters, on a laptop iGPU. The 3090
   numbers are a floor, and an Intel UHD has about a fortieth of the compute.
2. **Score on whole frames.** The 80 px samples cannot show what one motion per 328×188 tile
   does to a frame that holds two motions, or what stride 2 costs on a tile of that size. A
   few hundred whole streamed clips, cut into the client's tiles, and scored by the same groups.
3. **Settle the cursor question before modelling desktops.** Every method is at or under
   repeat on desktops and cursors. If desktop-streamer draws the pointer in the client, the
   cursor needs no model, and what is left on a desktop is drags, menus and region scrolls.
4. **A third frame, if the network is to come back.** Both learned models hedge because two
   frames cannot tell a scroll that is easing in from one that is easing out. The runs hold up
   to 7 frames of context; the client's contract would need a third input. The test is the
   search + correction model with three frames: if its correction still drifts off the search,
   the problem is the loss, not the context.

Sibling problems: `../upscale/summary.md`, `../frame_gen_intra/summary.md`.
