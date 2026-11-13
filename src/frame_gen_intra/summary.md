# Frame generation, intra — summary

**2x frame generation by interpolation.** The host streams every second frame of the display
rate and once received frame N+2 is in, the client draws the unsent frame N+1 from N and N+2
and shows it before N+2. Received frames N and N+2 in, N+1 out. Both ends of the motion are
known, so the model fills in rather than guesses - at the price of showing every frame half a
received interval late: desktop-streamer's client shows the made frame at 0 and N+2 at ½ of the
interval (`schedule` in its `stream-enhance.js`). Extrapolation (`../frame_gen_extra/`) adds no
latency and guesses.

> **Status: five notebooks on the run set extrapolation uses - two thirds recorded - and two
> graphs that pass desktop-streamer's client tests in place of its mock.** In
> `dist/frame_gen_intra/`: `interpolate2x_tilesearch_r4_s2.onnx`, the tile search, and
> `interpolate2x_tilepick_r4_s2.onnx`, the search with a per-pixel pick. Each takes `previous` (N)
> and `current` (N+2), `[N, 3, H, W]` with `N`, `H`, `W` symbolic, and all 16 tests of
> `tests/enhance.test.js` pass with either as `interpolate.onnx`. Neither has run in a live session.
>
> **On the client's whole tiles the search is +2.40 dB over the mock** on the moving tiles - 40.55
> dB against 38.15 for the mean of the two frames (95% over videos +1.59 to +3.42) - and +3.67 on
> the changed ones (notebook 3). **In Chrome's WebGPU on the RTX 3090 it costs 3.3 ms of GPU a
> 1080p frame** (36 tiles in one run), every node on the GPU.
>
> **It is worth most where the picture moves as one, and little on real screens**: over the mean,
> photos +7.5 dB, drawn pages +6.8, video +0.9, recorded screens +0.5, drawn desktops nothing.
>
> **The per-pixel pick is the one thing that gains over the search on whole tiles**: +0.11 dB on
> the moving tiles (+0.04 to +0.19), on the recordings only - for 11.1 ms a 1080p frame, 3.4 times
> the search. Before this rebuild its graph did not run in the browser at all (see "Speed").
>
> **Nothing learned beats the search.** A network on top of it (notebook 2) never leaves its
> untrained start; a decision per tile among six pictures the search's motion makes (notebook 4),
> extrapolation's one learned gain, is +0.00 dB. What the search gets wrong is a change in one
> display frame, which N and N+2 show the same way whichever frame it happened in.
>
> **A wider search is worth +0.22 dB for 2.7 times the time** (±16 px, notebook 3) - most of it on
> drawn pages - so the published search stays at ±8.

**One notebook is one model**, as in `../upscale/` and `../frame_gen_extra/`: a notebook owns
its method, its numbers and its export cell, and publishes into `dist/frame_gen_intra/` at
the repo root.

## What makes it its own problem

Both endpoints are known, so the answer is bounded on either side. Whether a scroll stopped
between N and N+1 is still not in the frames - N+2 shows where it ended, not when - but a
window that appears, a menu that opens and a scroll that turns round are all in N+2, where an
extrapolator never sees them. Interpolation also has evidence for every choice *at the pixel
being drawn*: moving N and N+2 towards each other by a motion gives two reads of each pixel of
N+1, and where the motion is wrong they disagree. That is what the pick reads.

The cost is latency. Holding every frame until its successor arrives adds half a received
interval at the client's schedule - 16.7 ms on a 30 fps stream - which a remote desktop
session feels and recorded playback does not.

The pointer is not part of it, as in extrapolation: the capture leaves it out and the client
draws it over the picture (`DRAW_CURSOR = False` in notebook 00). Screen recordings may show one.

## Notebooks

Run in order, from this folder:

- `00_data_preprocess.ipynb` — the run set: `../frame_gen_extra/00_data_preprocess.ipynb` with its
  output in `datasets/frame_gen_intra/` - the same sources, scenes, stream, filter and seed, so the
  same runs to the codec's run-to-run difference. Screen captures from YouTube-8M's computer
  categories (`screen`), other video at 50-60 fps (`video`), drawn screens and photos animated as a
  desktop in use from a random phase (`page`, `desktop`, `photo`); recordings split by video, and a
  recorded sample kept only where its frames all change or none does (`runs.valid`). Train: 681
  sources, 10,532 runs (63% recorded); validation: 232 sources, 3,546 runs (77% recorded). 3.1 GB,
  13 minutes.
- `01_interpolate_dummy.ipynb` — **static methods only**, on 80 px samples: hold, the mock's mean,
  the exact shift and the bilateral block search as torch references, extrapolation's tile search
  turned two-sided, and the per-pixel pick; by kind, at the client's halo, in pictures, and the two
  graphs exported with the mock's mean beside them - and the search's 3x twin, at ⅔, for the
  client's both-on mode. Nothing is trained; a minute.
- `02_interpolate_refine.ipynb` — **a network on top of the search** (`models.Refine`), three ways
  and under three losses, trained to convergence (12 minutes in all), and `models.Commit`, a rule
  that reads N alone where the search's two reads disagree. Nothing is exported.
- `03_interpolate_tiles.ipynb` — **whole frames in the client's tiles**: the validation clips
  rebuilt whole and cut as the client cuts them (`helpers/clips.py`), every picture the search's
  motion makes and the pick scored per tile, by kind, with 95% ranges over videos; three reaches
  of the search, priced in the browser. Builds the per-tile tables of both splits
  (`helpers/decide.py`, 17 minutes, cached in `checkpoints/decide/`). Nothing is exported.
- `04_interpolate_decide.ipynb` — **a picture chosen per tile**: extrapolation's notebook 13
  decision with six pictures to choose from, learned from six numbers of the search's costs
  (`models.Decided`), scored, checked on real tiles and exported - into `checkpoints/export/`,
  since it does not beat the search.

The code the notebooks share lives in `helpers/`. `sources.py`, `degrade.py`, `metrics.py`,
`scenes.py`, `recorded.py`, `speed.py` and `training.py` are copies of `../frame_gen_extra/`'s
(paths and wording changed), so a change made for one problem cannot quietly change the other's
data or units. `runs.py` and `clips.py` are extrapolation's turned round: the sample is (N, N+2) →
N+1, training can swap the two frames in time, a sample or tile is *changed* when N+1 differs from
either received frame, and `runs.valid` is extrapolation's test unchanged. `models.py` holds every
method with the client's two-input contract. `decide.py` builds the per-tile tables and names the
searches compared (`SEARCHES`, `BROWSER_MS`, `reach`). `show.py` draws pictures and motion.
`webexport.py` exports into `dist/frame_gen_intra/`, and lists what the browser's WebGPU provider
will not take.

## The data

**A sample is received frames N and N+2, and the target N+1**, every second frame received
whatever the clip's rate. For a drawn scene (60 fps) and `video` (50-60 fps) the client receives
25-30 fps; for most screen captures (24-30 fps) it receives 12-15 fps and makes the real frame
between - the same task at half the rate, twice the motion between received frames. Inputs and
target are decoded frames at the stream's resolution: the whole clip goes through one H.264 encode
and the received frames are every second of it.

**Drawn scenes start at a random phase.** A scene's motion is tuned per frame at 30 fps and read at
60; read from the first frame, every stop and every menu would land on a received frame, and N+1
would always lie exactly half way between N and N+2 - the one case where the mean of the two motions
is right. The first set had no phase.

**Repeated frames.** A recorder that captures slower than its video's rate repeats frames; then N+1
is a copy of N or of N+2, and holding it is exact by construction. `runs.valid` keeps a received
frame only where the steps around it all change or none does; 29 of the screen clips and 83 of the
`video` clips have no valid run at all.

**Scores** on the 80 px samples are over the samples that changed - more than 0.5% of the core's
pixels differ by more than 8 luma levels between N+1 and N, *or* between N+1 and N+2: 1,293 of
3,546. On whole tiles (notebooks 3-4) over the **changed** tiles (6,972 of 19,374) and the
**moving** ones - the search's gate open, something moved between N and N+2 (10,692) - each tile
capped at 50 dB, with 95% ranges from a bootstrap over videos (clips for the drawn kinds).

## Static methods

Notebook 1, on the 80 px validation samples (wPSNR, Y'CbCr weighted 6:1:1; SSIM on luma):

| changed samples (1,293) | wPSNR | SSIM-Y | vs hold | vs mean |
| --- | --- | --- | --- | --- |
| hold N | 33.59 | 0.8029 | | +0.69 |
| hold N+2 | 32.07 | 0.7913 | −1.52 | −0.83 |
| mean (the mock) | 32.91 | 0.8165 | −0.69 | |
| shift, tile (±16, exact, torch) | 38.47 | **0.9152** | +4.87 | +5.56 |
| shift, tile, from N only | **38.81** | 0.9082 | **+5.22** | **+5.90** |
| shift, 16 px blocks | 38.02 | 0.9076 | +4.42 | +5.11 |
| bilateral, 16 px blocks (±8) | 37.00 | 0.8786 | +3.41 | +4.09 |
| **tile search** (graph) | **37.08** | **0.8783** | **+3.48** | **+4.17** |
| tile search + pick (graph) | 36.97 | 0.8810 | +3.37 | +4.06 |

By kind (changed samples, wPSNR):

| | page (242) | desktop (19) | photo (92) | screen (546) | video (394) | pointer (30) |
| --- | --- | --- | --- | --- | --- | --- |
| hold N | 25.18 | 34.46 | 24.28 | 39.55 | 32.63 | 22.34 |
| mean (the mock) | 23.86 | 27.44 | 27.02 | 37.73 | 33.42 | 24.58 |
| shift, tile | 38.64 | 30.16 | 42.13 | 39.94 | 35.87 | 43.00 |
| shift, tile, from N only | 38.75 | 37.72 | 42.55 | 41.45 | 34.36 | 42.08 |
| tile search | 33.60 | 27.14 | 39.64 | 39.76 | 35.37 | 35.26 |
| tile search + pick | 32.69 | 27.41 | 39.55 | 39.81 | 35.52 | 33.00 |

- **On a recording the mean is worse than holding a frame** (32.91 against 33.59 overall, 37.73
  against 39.55 on screens): it draws every changed line twice at half strength.
- **The exact shift's lead is reach**: it searches ±16 px, the graph ±8. On the pages and around
  the pointer, where drawn scrolls and drags move further than 8 px in two display frames, the gap
  is 5-8 dB; on the recordings 0.2-0.5. Notebook 3 prices the reach on whole tiles.
- **The pick** (`models.TilePick`): per pixel, the tile's motion where its two reads of the pixel
  agree better than N and N+2 unmoved do, the mean where not - a 15×15 box of luma disagreement over
  the pixels inside the tile, a sigmoid at sharpness 1 per luma level, chosen on 2,000 train runs.
  On these samples it is −0.11 dB; on whole tiles +0.11 (notebook 3).

**The halo.** The tile search reads past a small halo (the edge repeated) and holds at the client's
4: 36.72 dB, against 37.08 at 16. The exact shift falls to 35.25.

## A network on top

Notebook 2. `models.Refine`: N, N+2 and the search's two reads (12 channels) into `TwoScale`'s body
(16 and 32 channels, three 3×3 convolutions at a quarter of the resolution), out at full resolution
as some of: a correction of where N and where N+2 are read, a mask mixing the two reads per pixel,
and 3 channels of pixels added. The last convolution starts at zero, so untrained it is the search.
Trained at a halo of 8 to convergence (48 epochs each, 2.2-2.4 minutes), frames swapped in time half the
time.

| changed samples (1,293), halo 8 | wPSNR | still samples, after one epoch | best epoch |
| --- | --- | --- | --- |
| tile search (the start) | 36.94 | 79.67 | |
| flow + mask, MSE | 36.94 | 59.31 | 0 (the start) |
| flow + mask, L1 | 36.94 | 77.81 | 0 |
| flow + mask, `log_wmse` | 36.94 | 73.97 | 0 |
| residual, `log_wmse` | 36.94 | | 0 |
| flow + mask + residual, `log_wmse` | 36.94 | | 0 |

- **No network ever beats its start**, under any loss. Under MSE the motion network loses 0.7 dB
  on the changed samples and 20 on the still ones after one epoch: the loss is ruled by the samples
  with the largest errors, where a blur lowers it; the score by the many with small errors, where
  the blur costs most.
- **A change in one frame is what nothing here draws.** A menu that opens or a wheel notch that
  ends is in N+1 exactly as in N or as in N+2, and N and N+2 look the same whichever it was; unsure,
  a network hedges, and the hedge is the mean the search already draws.

**Committing instead.** `models.Commit` reads N alone where the search's two reads disagree. On this
set it loses 0.5 to 0.8 dB on the changed samples (N alone, always: −0.48), gaining on the drawn
desktops (+4.3 dB for N alone) and a little on screens, losing on pages, photos and video. On the
first set, whose drawn scenes put every stop on a received frame, it was +0.7. Not exported.

## Whole frames

Notebook 3: the validation clips rebuilt whole and cut into the client's 328×188 tiles, at every
valid received frame - 19,374 tiles, 6,972 changed, 10,692 moving.

| moving tiles, ≤ 50 dB | page | desktop | photo | screen | video | all | vs the mean | 95% over videos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hold N | 27.73 | 37.54 | 32.75 | 42.35 | 34.49 | 37.01 | −1.14 | −1.57 to −0.59 |
| mean (the mock) | 29.03 | 38.22 | 34.86 | 43.20 | 35.75 | 38.15 | | |
| N moved alone, ±8 | 35.10 | 37.35 | 40.83 | 42.92 | 34.94 | 39.50 | +1.35 | +0.44 to +2.40 |
| **tile search ±8** (published) | 35.81 | 38.20 | 42.35 | 43.70 | 36.69 | **40.55** | **+2.40** | +1.59 to +3.42 |
| tile search ±8 + pick (published) | 35.77 | 38.38 | 42.33 | **43.80** | **37.03** | **40.67** | **+2.51** | +1.70 to +3.53 |
| tile search ±16 | **36.87** | 38.19 | **42.65** | 43.67 | 36.79 | 40.77 | +2.62 | +1.78 to +3.70 |
| tile search ±16 at ¼ resolution | 34.69 | 38.19 | 39.50 | 43.65 | 36.62 | 40.01 | +1.85 | +1.16 to +2.69 |

On the changed tiles: hold N 30.70, mean 32.16, the search 35.83 (+3.67), the pick 36.01, ±16
36.18. On the unchanged tiles the searches and the pick are within 0.02 dB of the mean; holding a
frame or moving one alone loses 0.2.

- **The drawn scenes carry the gain**: photos +7.5 dB and pages +6.8 over the mean, against +0.9 on
  video and +0.5 on the recorded screens. A pan or a scroll is one motion a tile; a screen's change
  mostly is not.
- **One frame alone loses to both**: N moved half way alone is 1.05 dB under the search. On the
  score a change half drawn in two places beats one drawn whole in the wrong place.
- **The pick gains on the recordings** (screens +0.10, video +0.34 over the search) - the opposite
  of the 80 px samples, where the drawn kinds dominate.
- **The reach is chosen on the training split among the searches the client can afford** (no
  slower than the published one in the browser): ±8. ±16 is +0.22 dB on validation, almost all of
  it on pages (+1.06), for 2.7 times the time.

## A picture chosen per tile

Notebook 4. Every picture `models.Decided` makes is N read at −s·d and N+2 at +s·d, mixed a : 1−a,
d the search's offset: the search (s 1, a ½), the mean (0, ½), either frame held (0, 1 / 0, 0),
either frame moved alone (1, 1 / 1, 0) - so in a graph the decision scales the search's motion and
mix per tile and adds nothing at full resolution. Logistic regression and a network (6 → 16 → 16 →
choices) on the six numbers of extrapolation's decision, trained on the moving tiles of the training
split to raise the expected score of the picture picked, kept at the best step on a held-out tenth
of the training sources.

| moving validation tiles, ≤ 50 dB | all | vs the search | 95% over videos |
| --- | --- | --- | --- |
| tile search ±8 | 40.55 | | |
| learned, two choices (search or mean), logistic / network | 40.55 / 40.55 | −0.01 / −0.01 | −0.03 to +0.01 |
| learned, six choices, logistic / network | 40.55 / 40.55 | +0.00 / +0.00 | −0.02 to +0.02 |
| tile search + pick | 40.67 | +0.11 | +0.04 to +0.19 |

- **No decision beats the search.** Held out, the best is +0.006 dB, at its untrained start. The
  graph is right - the same choice as the table's rule on 920 real tiles, as torch to 1e-4 - and
  stays in `checkpoints/export/`.
- **What extrapolation's decision found is in N+2 already.** There it gained where a motion stopped
  or was not a motion, which N−2 and N do not announce; between N and N+2 the search sees where the
  motion went. What it gets wrong is a change in one display frame, which leaves nothing in its
  costs: the six-choice network takes the search on every moving tile.

## Speed

In Chrome 157's WebGPU on the RTX 3090 (`../upscale/benchmark/main.py --models dist/frame_gen_intra`,
GPU-resident tensors, the output buffer reused, GPU time from timestamp queries), one 1080p frame
as 36 tiles in one run:

| graph | kernels | GPU ms | wall ms | wall ms, graph capture |
| --- | --- | --- | --- | --- |
| tile search ±8 (published) | 47 | 3.2 | 3.3 | 3.1 |
| tile search ±8 + pick (published) | 77 | 10.9 | 11.1 | 10.9 |
| tile search ±16 | 47 | 8.9 | 9.0 | |
| tile search ±16 at ¼ resolution | 47 | 2.9 | 3.0 | |
| tile search ±8 + decision (`checkpoints/export/`) | 97 | 3.5 | 3.6 | 3.4 |

Every node of the 36-tile graphs runs on the WebGPU provider; the client's graphs, with symbolic
shapes, keep on its CPU only the int64 shape arithmetic extrapolation's keep there. Extrapolation's
search, one `GridSample` to these two, costs 2.5 ms.

**The pick did not run in the browser before this rebuild.** Its 15×15 box of disagreement was
padded with `F.pad(mode="replicate")`, which exports with pads computed at run time; the WebGPU
provider computed them wrong and returned a picture twice the tile's size, though onnxruntime on the
CPU matched torch. The box now averages over the pixels inside the tile, a column pass and a row
pass with fixed pads - the rule is in `helpers/webexport.py`. It costs what it does because of its
two dozen full-resolution elementwise passes, not the box.

Desktop CPU, one tile: the search 2.4 ms, the pick 3.7. Neither runs on WebGL (`GridSample`).

## In desktop-streamer

`interpolate2x_tilesearch_r4_s2.onnx` is the client's `interpolate.onnx` now, in place of the mock
(in desktop-streamer's working tree, not yet committed there): +2.40 dB over the mock for 3.3 ms;
the pick, which would fit as well, adds 0.11 for 7.8 ms more. Either passes all 16 tests of its
`tests/enhance.test.js` - 17 now - and loads in Chrome's WebGPU with only the int64 shape arithmetic
on the CPU. How the client places it:

- with **interpolate** on alone, the made frame is shown at 0 and N+2 at ½ of the interval -
  half way between N and N+2 in time, which is what the graphs make;
- with **interpolate and extrapolate** both on, three pictures an interval: the made frame at 0,
  N+2 at ⅓, the extrapolated one at ⅔ - so the interpolated frame is two thirds of the way from N
  to N+2, and the client runs `interpolate3x_tilesearch_r4_s2.onnx` as `interpolate3x.onnx` there:
  the same search, N read at −⅔ of the motion and N+2 at +⅓, mixed ⅓ : ⅔ by nearness (notebook 1,
  `models.Interpolate(at=...)`). The run set holds 2x samples only, so that placement is not
  scored;
- on **WebGL** it is not offered: the client greys the row "(needs WebGPU)".

`../mock/make_mock_models.py --interpolate` writes the mock back over both.

## Open questions

- **Ghost or late?** The search draws a change that happened in one frame twice at half strength;
  committing draws it once, a frame late. The score prefers the ghost on this set; only looking at a
  60 fps stream decides.
- **The reach for less.** ±16 is +0.22 dB on the moving tiles (pages +1.06) at 2.7 times the time;
  a coarse-to-fine search - ±16 at a quarter of the resolution, then ±2 at half around it - might
  keep the reach at about the published cost. Extrapolation's notebook 8 built one
  (`PyramidSearch`); there it did not pay.
- **The pick for less.** Its gain is on the recordings, and its cost is full-resolution elementwise
  work; the same disagreement measured at half resolution would cost a quarter.
- **Is half an interval of latency worth it?** The other side of extrapolation's open question: a
  guess that is right more often, shown later. Only a live session answers it.

## Next

1. **Watch it in a live session** - it is the client's `interpolate.onnx` - a scroll and a dragged
   window beside extrapolation's, and time it on a laptop iGPU.
2. **A cheaper reach and a cheaper pick** (open questions), each scored on notebook 3's tables.

## History

- **Until this rebuild** the set was the first recipe: drawn scenes read from their first frame and
  50-60 fps YouTube clips, none of the 24-30 fps screen captures (11,510 train runs). There the
  search was +9.3 dB over the mean on the changed samples and the pick +0.05; the drawn scenes put
  every stop on a received frame, which flattered every motion method. Archived in
  `datasets/frame_gen_intra/_synthetic_v1/` and `checkpoints/_synthetic_v1/`; its notebook outputs
  are in commit 6c12222.
- **Removed with it**: notebook 2's oracle - which picked the time of N+1 and the frame to read it
  from with N+1 in hand - and the open question of a host that sends one bit a tile, which needs the
  frame desktop-streamer's host never captures.
