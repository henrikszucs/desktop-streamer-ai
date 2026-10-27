# Frame generation, extra — summary

Extrapolation: given the frames the stream has delivered so far, generate the next one
before it arrives, so a client can keep drawing through a stall and a dropped frame does
not read as a freeze.

> **Status: three notebooks: the data, the baselines, and a first network trained only as
> far as a 2-epoch smoke test.** Nothing in desktop-streamer's `src/` calls any of it; the
> client still runs the mock graph.
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
  after a real run. So far only a 2-epoch smoke test; see "The two-scale model".

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
graph runs a 328×188 tile in 2.75 ms on the desktop CPU, the control in 8.1 ms. In torch on
the RTX 3090 a 1920×1080 stream frame costs 3.8 ms and 11.8 ms. At the same operation
count, full resolution is three times slower.

**Trained: only a smoke test.** Two epochs (390 steps of 64, 10 seconds each model) leave
both at repeat: 0.7313 SSIM-Y and 27.21 dB on the changed samples, where notebook 1's tile
shift is at 0.8180 and 32.00 dB at this halo. Overfitting one batch shows the recipe learns -
25.41 → 27.28 dB on its changed samples in 600 steps at 1e-3 (26.44 at 3e-4) - but it is flat
for the first ~300 steps: with the last convolution at zero, nothing before it gets a
gradient until that convolution has grown. Two epochs end on that plateau, so the smoke test
says the notebook runs end to end, and nothing yet about the architecture. On the unchanged
samples the networks fall from 74.8 to 63 dB: still 0.07% error, invisible, but it is the
number to watch, since repeat is perfect there.

A real run is raising `MAX_EPOCHS` and running again: the training carries on from its
checkpoint, at about 5 seconds an epoch.

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

A real run of notebook 2's two models, to convergence, then score them against the tile
shift by kind of source. If the two-scale network beats the control, the quarter-resolution
path earns its place. If neither gets near the shift, the network cannot find the motion
from a zero start in reasonable time. Then the thing to try is a fixed bank of shifted copies
of `current` - a depthwise `Conv` with one-hot kernels, which is inside the operator set -
for the network to choose between.

Sibling problems: `../upscale/summary.md`, `../frame_gen_intra/summary.md`.
