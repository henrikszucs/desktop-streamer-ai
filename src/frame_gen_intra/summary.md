# Frame generation, intra — summary

**2x frame generation by interpolation.** The host streams every second frame of the display
rate - 30 of 60 a second - and once received frame N+2 is in, the client draws the unsent
frame N+1 from N and N+2 and shows it before N+2. Received frames N and N+2 in, N+1 out. Both
ends of the motion are known, so the model fills in rather than guesses - at the price of
showing every frame half a received interval late: desktop-streamer's client shows the made
frame at 0 and N+2 at ½ of the interval (`schedule` in its `stream-enhance.js`).
Extrapolation (`../frame_gen_extra/`) adds no latency and guesses.

> **Status: three notebooks, and two graphs with the client's contract, not yet run in the
> client or the browser.** `dist/frame_gen_intra/interpolate2x_tilesearch_r4_s2.onnx` and
> `interpolate2x_tilepick_r4_s2.onnx` take `previous` (N) and `current` (N+2), each
> `[N, 3, H, W]` with `N`, `H`, `W` symbolic, and make `output`; both are checked against torch
> at 36 and 4 tiles of 328×188 and at one of 64×32. Neither has been timed in the browser.
>
> **Motion beats the mock by 9.3 dB.** Extrapolation's tile search, run on N and N+2 and
> applied half each way - N moved half the motion forwards, N+2 half back, averaged - scores
> **37.52 dB wPSNR on the 868 changed validation samples against 26.89 for holding N (+10.63)
> and 28.18 for the mock's mean (+9.34)**, 0.9096 SSIM-Y against 0.6663 and 0.6938. It beats
> holding N on 92% of them. Nothing in it learns.
>
> **A per-pixel pick barely adds to it.** Choosing per pixel between the tile's motion and no
> motion, by which of the two reads of the pixel agree better, gives +0.05 dB overall and
> +0.32 on desktops, and halves the worst loss against the mean (−3.1 dB on one sample
> against −7.2). It costs about 20% more on the CPU.
>
> **The exact search knows more than the graph**: a whole-pixel search over ±16 px reaches
> 38.03 dB, and moving N alone by it 38.60 - pages 40.63 against 38.29 for the two-sided mean.
> Both are torch references, not graphs.
>
> **A neural network does not make the search more precise** (notebook 2). Three networks on
> top of it - correcting where each frame is read, how the two reads mix, and drawing pixels
> of their own - start as the search and, trained to convergence, never beat it. The search's
> motion is precise enough: the best time for N+1 between N and N+2, picked with the answer in
> hand, adds 0.35 dB. What is missing is *when* a change happened - a menu, the end of a wheel
> notch is in N+1 exactly as in N or exactly as in N+2 - and an oracle that knows is 6.4 dB up;
> two frames do not say which, so nothing trained on them finds it.

**One notebook is one model**, as in `../upscale/` and `../frame_gen_extra/`: a notebook owns
its method, its numbers and its export cell, and publishes into `dist/frame_gen_intra/` at
the repo root.

## What makes it its own problem

Both endpoints are known, so the answer is bounded on either side. Whether a scroll stopped
between N and N+1 is still not in the frames - N+2 shows where it ended, not when - but a
window that appears, a menu that opens and a scroll that turns round are all in N+2, where an
extrapolator never sees them. Interpolation also has evidence for every choice *at the pixel
being drawn*: moving N and N+2 towards each other by a motion gives two reads of each pixel of
N+1, and where the motion is wrong they disagree. Extrapolation's notebook 4 found a picker
there had no such evidence.

The cost is latency. Holding every frame until its successor arrives adds half a received
interval at the client's schedule - 16.7 ms on a 30 fps stream - which a remote desktop
session feels and recorded playback does not.

The pointer is not part of it, as in extrapolation: the capture leaves it out and the client
draws it over the picture (`DRAW_CURSOR = False` in notebook 00).

## Notebooks

Run in order, from this folder:

- `00_data_preprocess.ipynb` — the run set. `../frame_gen_extra/00_data_preprocess.ipynb` with
  its output in `datasets/frame_gen_intra/`: the same sources, scenes, stream and seed, at 60 fps
  with every second frame received. Runs of 8 frames, received 0, 2, 4, 6; a sample is two
  received frames and the one between (N+2 = 2, 4, 6). 11,510 train and 1,730 validation runs,
  2.9 GB, 12 minutes. More YouTube sequences were on disk than when extrapolation's set was
  made (4,455 against 4,444), so the sets are close but not the same, and the two problems'
  numbers do not compare sample for sample.
- `01_interpolate_dummy.ipynb` — **static methods only.** Hold, the mock's mean, the exact
  shift and the bilateral block search as torch references, extrapolation's tile search turned
  two-sided, and the per-pixel pick; by kind, at the client's halo, in pictures, and the two
  graphs exported with the mock's mean beside them. Nothing is trained; a minute.
- `02_interpolate_refine.ipynb` — **a network on top of the search** (`models.Refine`), three
  ways and under three losses, trained to convergence (12 minutes in all); the oracle that
  shows what is left and why two frames cannot reach it; and `models.Commit`, a rule that
  reads N alone where the search's two reads disagree. Nothing is exported. See "A network on
  top".

The code the notebooks share lives in `helpers/`. `sources.py`, `degrade.py`, `metrics.py`,
`scenes.py`, `speed.py` and `training.py` are copies of `../frame_gen_extra/`'s (paths and
wording changed), so a change made for one problem cannot quietly change the other's data or
units. `runs.py` is extrapolation's turned round: the sample is (N, N+2) → N+1 (`ahead` −1),
training can swap the two frames in time, and a sample is *changed* when N+1 differs from
either received frame. `models.py` holds every method with the client's two-input contract.
`show.py` draws pictures and motion. `webexport.py` exports into `dist/frame_gen_intra/`.

## The data

**A sample is received frames N and N+2, and the target N+1.** Input and target are decoded
frames at the stream's resolution (640×360 from a 1280×720 window), as in extrapolation.

**Scores are reported over the samples that changed**: more than 0.5% of the core's pixels
differ by more than 8 luma levels between N+1 and N, *or* between N+1 and N+2 - so a sample
where N+1 equals N but N+2 differs, which holding N gets right and the mean gets wrong,
counts. 868 of the 1,730 validation samples changed; on the others holding N is at 86.3 dB.

## Static methods

Notebook 1. Every method gets N and N+2 and is scored on N+1 over the 64 px core (wPSNR,
Y'CbCr weighted 6:1:1; SSIM on luma).

| changed samples (868) | wPSNR | SSIM-Y | vs hold | vs mean |
| --- | --- | --- | --- | --- |
| hold N | 26.89 | 0.6663 | | −1.29 |
| hold N+2 | 27.34 | 0.6679 | +0.45 | −0.84 |
| mean (the mock) | 28.18 | 0.6938 | +1.29 | |
| shift, tile (±16, exact, torch) | 38.03 | **0.9226** | +11.14 | +9.84 |
| shift, tile, from N only | **38.60** | 0.9177 | **+11.72** | **+10.42** |
| shift, 16 px blocks | 37.56 | 0.9221 | +10.67 | +9.38 |
| bilateral, 16 px blocks (±8) | 37.30 | 0.9082 | +10.42 | +9.12 |
| bilateral, 8 px blocks | 36.52 | 0.8951 | +9.63 | +8.34 |
| **tile search** (graph) | **37.52** | **0.9096** | **+10.63** | **+9.34** |
| **tile search + pick** (graph) | **37.57** | **0.9119** | **+10.68** | **+9.39** |

Over every sample the tile search is +5.84 dB over hold; on the unchanged ones every motion
method is within 0.2 dB of the mean (87.2–87.4 dB), so neither graph damages a still tile.

By kind (changed samples, wPSNR):

| | page (375) | desktop (113) | photo (266) | video (114) | pointer (60) |
| --- | --- | --- | --- | --- | --- |
| hold N | 25.55 | 25.64 | 24.00 | 39.25 | 25.19 |
| mean (the mock) | 25.41 | 27.73 | 27.11 | 40.22 | 26.11 |
| shift, tile | 38.29 | 30.38 | 39.14 | 42.11 | 34.34 |
| shift, tile, from N only | **40.63** | 29.08 | 38.77 | 41.00 | **35.21** |
| shift, 16 px blocks | 37.19 | **31.19** | 38.97 | 41.83 | 34.26 |
| tile search | 37.28 | 30.15 | **39.18** | 41.76 | 33.77 |
| tile search + pick | 37.26 | 30.47 | 39.20 | 41.82 | 34.03 |

- **The mean is a cross-fade, and a screen does not fade.** It gains on photos and video,
  where a soft ghost reads as blur, and nothing on pages, where it draws every line of text
  twice at half strength.
- **Pages and photos are translations**, and one motion a tile lands them: +11.7 and +15.2 dB
  over hold.
- **Desktops are where one motion a tile is not enough** (+4.5): a dragged window or a menu
  over a still desktop is two motions in one tile. Per-block motion does best there (31.19),
  and the pick recovers a third of the gap (30.47).
- **Taking N+1 from N alone beats the two-sided mean on pages** (40.63 against 38.29). Over
  all kinds it is +0.57 dB, and it loses on desktops, photos and video. Notebook 2 finds why:
  not the easing of a scroll - N+1 is half way, near enough - but changes that happen in one
  frame, which in N+1 are as in N or as in N+2, never their mean.
- **The pick** (`models.TilePick`): per pixel, the tile's motion where its two reads of the
  pixel agree better than N and N+2 unmoved do, the mean where not, softly - a 15×15 box of
  luma disagreement, a sigmoid at sharpness 3 per luma level, chosen on 2,000 train runs (the
  best at the edge of the grid tried, 35.96 dB against the tile search's 35.90 there). It
  turns a tile's wrong motion back into the mean where it is wrong, not into the truth: the
  mean is the only other candidate.

**The halo.** The exact shift can only search as far as the halo and falls to 32.66 dB at the
client's halo of 4. The tile search reads past it (the edge repeated) and holds: 36.86 at 4,
37.20 at 8, 37.52 at 16; the pick 36.83, 37.30, 37.57. The client's tile needs no wider halo.

**Published**, into `dist/frame_gen_intra/`:

| graph | KB | desktop CPU, one tile | |
| --- | --- | --- | --- |
| `interpolate_static_mean_tile320x180.onnx` | 0.5 | 0.02 ms | the mock's mean, a baseline |
| `interpolate2x_tilesearch_r4_s2.onnx` | 40.1 | 2.97 ms | symbolic `N`, `H`, `W` - the client's |
| `interpolate2x_tilepick_r4_s2.onnx` | 45.6 | 3.52 ms | symbolic |
| `..._tile320x180_b36.onnx` of each | 39.7 / 45.2 | 107 / 132 ms for 36 | pinned to one 1080p frame, for the benchmark page |

Extrapolation's graph is the same search with one `GridSample` where these have two; it costs
2.34 ms of GPU a 1080p frame in Chrome's WebGPU on the RTX 3090. These are not measured yet.
Like it, they cannot run on WebGL (`GridSample`, `ConstantOfShape`).

## A network on top

Notebook 2. `models.Refine`: N, N+2 and the search's two reads (12 channels) into `TwoScale`'s
body (16 and 32 channels, three 3×3 convolutions at a quarter of the resolution), out at full
resolution as some of: a correction of where N and where N+2 are read (4 channels - N+1 need
not be half way), a mask mixing the two reads per pixel, and 3 channels of pixels added. The
last convolution starts at zero, so untrained it is the search. Trained at a halo of 8, Adam
from 1e-4 (1e-3 for pixels only), halved on a plateau, best epoch by wPSNR on the changed
samples, frames swapped in time half the time; each converged in about 48 epochs, 2.4 minutes.

| changed samples (868), halo 8 | wPSNR | SSIM-Y | still samples, after one epoch | best epoch |
| --- | --- | --- | --- | --- |
| tile search (the start) | 37.20 | 0.9102 | 87.24 | |
| flow + mask, MSE | 37.20 | 0.9102 | 57.40 | 0 (the start) |
| flow + mask, L1 | 37.20 | 0.9103 | 81.77 | 7 (+0.001 dB) |
| flow + mask, `log_wmse` | 37.20 | 0.9102 | 77.71 | 0 |
| residual, `log_wmse` | 37.20 | 0.9102 | | 0 |
| flow + mask + residual, `log_wmse` | 37.20 | 0.9102 | | 0 |

- **The loss decides how much harm, not whether it helps.** MSE is a mean over pixels, ruled by
  the samples with the largest errors, where a blur lowers it; the score is a mean of decibels
  per sample, ruled by the many with small errors, where the blur costs most. Under MSE the
  motion network loses 1.1 dB on changed samples and 30 on still ones after one epoch.
  `metrics.log_wmse` - the score itself, minus the per-sample wPSNR - harms less; nothing gains.

**The ceiling.** An oracle that sees N+1 picks, per sample or per 8×8 cell, among reads made
from the search's motion d: N+1 at a time t between N and N+2 (0.3 … 0.7), or from one side
only (N moved half way, N+2 moved back half way).

| wPSNR, changed | all | page | desktop | photo | video | no codec (118) | H.264 (750) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| search (both, t = ½) | 37.18 | 36.54 | 30.10 | 39.07 | 41.84 | 39.13 | 36.87 |
| N only | 37.64 | 38.84 | 28.66 | 38.26 | 41.18 | 42.89 | 36.81 |
| N+2 only | 37.75 | 37.63 | 29.84 | 39.74 | 41.31 | 42.58 | 36.98 |
| oracle per sample: t | 37.53 | 36.79 | 30.37 | 39.54 | 42.36 | 39.33 | 37.25 |
| oracle per sample: side | 43.58 | 45.05 | 32.85 | 44.74 | 46.67 | 52.36 | 42.20 |
| oracle per 8×8: t and side | 45.15 | 46.35 | 34.92 | 46.18 | 48.95 | 53.51 | 43.83 |

(H.264 column: the mean of the notebook's two bitrate groups, weighted by their 372 and 378
samples.)

- **Where on the motion N+1 lies is worth 0.35 dB**: half way, near enough. The search's
  motion is not what limits it.
- **Which frame to read it from is worth 6.4 dB, and more with no codec** - so it is not the
  60 fps encode predicting N+1 from N. A desktop changes in single frames: a menu opens, a
  wheel notch ends, a region redraws, and in N+1 such a pixel is exactly as in N or exactly as
  in N+2. The search's mean draws it at half strength in both places.
- **Nothing in N and N+2 says which**, so the network finds nothing: under any loss the best
  guess without that evidence is the hedge, which the mean already is. Extrapolation lacked
  *whether* a motion goes on; interpolation lacks *when* a change happened.

**A rule instead.** `models.Commit` reads N alone where the search's two reads disagree (a
box of luma disagreement under a level), their mean where they agree: a change drawn once, a
display frame late, instead of twice at half strength.

| changed samples | wPSNR | SSIM-Y | still | page | desktop | photo | video |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tile search | 37.20 | **0.9102** | **87.24** | 36.54 | **30.10** | **39.15** | **41.84** |
| N only, always | **37.92** | 0.8942 | 86.36 | **39.42** | 28.66 | 38.34 | 41.18 |
| commit, 3×3, under 2 levels | 37.79 | 0.8952 | 87.00 | 38.82 | 28.78 | 38.60 | 41.45 |
| commit, 5×5, under 4 levels | 37.55 | 0.8969 | 87.21 | 38.15 | 28.90 | 38.62 | 41.60 |

Committing gains on pages and loses everywhere else, and wPSNR and SSIM-Y disagree about it:
a mean of decibels per sample rewards being exactly right half the time over being half
wrong always. Whether a change a frame late beats a ghost is for the eye. Not exported.

## In desktop-streamer

Not tried yet. Copied over the client's `src/client/web/media/models/interpolate.onnx`, either
graph should take the mock's place - same input and output names, symbolic shapes - and the
client's tests (`tests/enhance.test.js`) are the check. What the client does with it:

- with **interpolate** on alone, the made frame is shown at 0 and N+2 at ½ of the interval -
  half way between N and N+2 in time, which is what the graphs make;
- with **interpolate and extrapolate** both on, the made frame is shown at 0, N+2 at ⅓ and the
  extrapolated one at ⅔: three frames an interval, and the interpolated one is then a third of
  the way from N to N+2 in time, not half. The graphs carry half the motion, so it lands too
  far - a graph with a third would fit;
- on **WebGL** it does not load.

## Open questions

- **When did a change happen?** Worth 6.4 dB on the score and not in N and N+2 (notebook 2).
  A host that sent one bit a tile - "N+1 is as N" or "as N+2" - would carry it, but it needs
  the frame it does not send, as extrapolation's shelved hint did. A third received frame
  (N−2) shows how a motion has been going, not when the next change comes - a weak lead.
- **Ghost or late?** The mean draws a one-frame change twice at half strength; committing
  draws it once, a frame late. The scores disagree about which is better; only looking at a
  60 fps stream decides.
- **More motions for the pick.** On desktops per-block motion beats the tile's (31.19 against
  30.15 dB): a pick among per-block motions, by the two reads' agreement, is untried.
- **Whole frames.** The 80 px samples flattered extrapolation by half (+11.1 dB on samples,
  +5.8 on the client's whole-frame tiles). Interpolation should lose less - N+2 shows where a
  scroll stopped - but that is to be measured, with extrapolation's `clips.py`.
- **Is half an interval of latency worth it?** The other side of extrapolation's open question:
  a guess that is right more often, shown later. Only a live session answers it.

## Next

1. **Time both graphs in the browser** (`../upscale/benchmark/main.py --models
   ../../dist/frame_gen_intra`), and on a laptop iGPU.
2. **Run notebook 01's search on whole frames in the client's tiles** - copy
   `../frame_gen_extra/helpers/clips.py` - for the number a client sees.
3. **Ghost or late**: the search and `Commit` side by side on a 60 fps screen.
4. **Put the better graph in desktop-streamer's client** and watch a scroll and a dragged
   window, beside extrapolation's.

Sibling problems: `../upscale/summary.md`, `../frame_gen_extra/summary.md`.
