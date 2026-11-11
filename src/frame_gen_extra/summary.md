# Frame generation, extra — summary

**2x frame generation without latency.** The host streams every second frame of the display
rate and the client draws the one in between from the two it received last, *before* the next
one arrives. Received frames N−2 and N in, the unsent frame N+1 out. Interpolation
(`../frame_gen_intra/`) would wait for N+2 and add a received interval of latency;
extrapolation adds none. desktop-streamer's client already schedules it this way: an
extrapolated frame is shown half a frame interval after the frame it came from.

> **Status: fourteen notebooks on a set that is mostly real video, and a graph that passes
> desktop-streamer's client tests in place of its mock.**
> `dist/frame_gen_extra/extrapolate2x_tilesearch_r4_s2.onnx` has the mock's contract exactly -
> `previous`, `current` and `output`, each `[N, 3, H, W]` with `N`, `H`, `W` symbolic - and all 14
> tests of desktop-streamer's `tests/enhance.test.js` pass with it as `extrapolate.onnx`. Not yet
> run in a live session.
>
> **The set is two thirds recorded** (notebook 0): screen captures from YouTube-8M's computer
> categories, picked by eye (kind `screen`, 93 videos, mostly 24-30 fps), other video at 50-60 fps
> (`video`, 49 videos), and beside them drawn scenes - a drawn screen or a photo animated as a
> desktop in use (`page`, `desktop`, `photo`). Recordings are split by video. A recorder that
> captures slower than its video's rate repeats frames, which a live capture does not; a
> recorded sample is kept only where the frames around it all change or none does.
>
> **The best generator is still a search, not a network.** The tile search (notebook 3) finds
> one motion per tile from fixed convolutions - the cost of every offset within ±8 px on
> half-resolution luma, a softmax, half of it applied to N - and nothing in it learns. On the
> changed validation samples it scores **32.53 dB against 28.96 for showing N twice (+3.57)**; on
> whole client tiles +2.69 (notebook 4). **In Chrome's WebGPU on the RTX 3090 it costs 2.34 ms of
> GPU a 1080p frame** (36 tiles in one run), a seventh of the 16.7 ms a generated frame has at 60
> fps.
>
> **It is worth most where the picture moves as one, and little on real screens.** On whole
> tiles: photos (pans) +7.4 dB over repeat, drawn pages +4.2, screens +1.4, video +0.8, drawn
> desktops +0.1 (notebook 4). On a screen recording most of what changes between two frames is
> not motion at all - a redraw, a highlight, text, a fade: even the true motion per 16 px block,
> found with N+1 in hand, takes back only a third of the search's error there (notebook 7).
>
> **The mock the client runs is worse than doing nothing**: its `2 · current − previous` aims at
> N+2 and scores 24.84 dB against repeat's 28.96 (notebook 1).
>
> **No network that draws pixels or motion beats the search.** Asked for pixels (notebook 2,
> +0.72 dB over repeat) or for motion (notebook 3, +0.72), a network on two frames hedges; a
> correction on top of the search learns nothing; behind the search's own confidence mask
> (notebook 10) it is +0.06 dB for 14 times the work, trained on drawn scenes, recordings or both
> alike (notebook 12).
>
> **What learning does add is a decision** (notebook 13): per tile, the search's N+1 or N - a
> 500-weight network on twelve numbers the search mostly computes - **+0.22 dB on the moving
> tiles** (95% over videos +0.05 to +0.48), a third of what the perfect choice would give.
>
> **Searching per block ties, and helps the recordings** (notebook 9): 3-D recursive search, a TV's
> block motion estimator, is +0.03 dB overall - +0.42 on video, +0.24 on screens, −0.5 to −0.8 on
> the drawn scenes, where it carries a dragged window past its stop - for 14 times the work.
>
> **The rest of the gap is what happens between N and N+1.** A host that measured it on the frame
> it does not send would be worth +3.2 dB (notebook 5), but desktop-streamer's host captures at
> the rate it streams. History (notebook 6) and a better-measured motion (notebook 8) add +0.1 to
> +0.2 dB, all of it on the drawn scrolls.

**One notebook is one model**, as in `../upscale/`: a notebook owns its method, its numbers
and its export cell, and publishes into `dist/frame_gen_extra/` at the repo root.

## What makes it its own problem

It adds no latency, because it needs nothing that has not arrived yet - which is what a live
remote desktop session needs. It is also the harder of the two frame generation problems:
the answer is bounded on one side only, so the model is guessing rather than filling in, and
it is wrong whenever the user does something unpredictable. A click that opens a menu is
unforeseeable from the frames before it.

In 2x generation, half the frames on screen are guesses, all the time - not only through a
stall. A wrong guess is shown and then contradicted by the next received frame, so errors
read as judder or flicker, and the bar is a guess that is right where it acts and leaves
alone what it cannot predict.

**The pointer is not part of it.** desktop-streamer's capture is told not to draw the mouse
pointer into the video; the host sends its shape and position separately and the client
draws it over the canvas (`cursor.js` there). So the pointer moves smoothly at the client's
own rate with no frame generation at all, and the frames a model sees have none - the drawn
scenes leave it out too (`DRAW_CURSOR = False` in notebook 0). Screen recordings may show one.

## Notebooks

Run in order, from this folder:

- `00_data_preprocess.ipynb` — the run set. **Recorded**: the YouTube-8M sequences of the computer
  categories that are screen captures (`helpers/recorded.py`, picked by eye: 93 videos, 89 of them
  at 24-30 fps) as kind `screen`, and the other sequences at 50 fps or more as `video` (49 videos);
  both split by video, a third of each category's videos to validation. **Drawn**: 240 desktop
  screens (`datasets/screens.py`) and 75 DIV2K photos, animated as a desktop in use
  (`helpers/scenes.py`) at 60 fps from a random phase - a page scrolls by wheel notches, a flick or
  a drag of the scrollbar, a photo pans, a desktop holds still while a region scrolls, a window is
  dragged or a menu pops up, the pointer itself not drawn. Each clip at its own rate through the
  stream's ×2 downscale and H.264 (`helpers/degrade.py`), cut into runs of 8 frames of one spot from
  a received frame - for a recording only where `runs.valid` keeps every sample - a 64 px core with
  a halo of 16. Train: 678 sources, 10,479 runs (63% recorded); validation: 232 sources, 3,523 runs
  (77% recorded). 3.1 GB, 13 minutes.
- `01_extrapolate_dummy.ipynb` — **static methods only**: repeat, linear (the mock), half step, and
  the motion shifts (one offset per tile, per 16 px block, per 8 px block). No training; by kind, by
  halo, how far content moves, speed, and the three that ONNX expresses as plain arithmetic
  exported.
- `02_extrapolate_twoscale.ipynb` — **the first network**: `current` plus a correction made at 1/2
  and 1/4 of the resolution (`models.TwoScale`), beside a control without downsampling.
  Checkpointed to `checkpoints/extrapolate_twoscale.pt`, exported into `checkpoints/export/`.
- `03_extrapolate_motion.ipynb` — **motion, not pixels**: `current` moved per pixel with
  `GridSample` (`models.Motion`), the motion from a learned head, from `models.TileSearch` (fixed
  convolutions, nothing learned), or both. The gates, and the published tile search exported to
  `dist/`.
- `04_extrapolate_oracle.ipynb` — **the ceiling of choosing**: the validation clips rebuilt whole and
  cut into the client's tiles (`clips.plan`), the search per tile and per block, an oracle over
  fixed candidate motions per cell, and a picker that needs no training.
- `05_extrapolate_hint.ipynb` — **a hint from the host**: each tile's motion from N to N+1 measured
  on the clean frames, as cheaply as a host can afford, scored per tile and per block, its bits,
  late hints; the hinted graph exported to `checkpoints/export/` - shelved, no real host captures
  the frame it is measured on.
- `06_extrapolate_history.ipynb` — **the motion history**: each tile's search motion over the last
  three received pairs, N moved on by 17 shares of the newest motion, rules from physics and tables
  fitted on two, three and four frames. Cached in `checkpoints/history/`.
- `07_extrapolate_budget.ipynb` — **where the search's error comes from**: every validation tile
  labelled by what its scene recorded (or `screen` / `video` for a recording), the search's summed
  error split by label and family, and what three oracles take back.
- `08_extrapolate_pyramid.ipynb` — **measuring the motion better**: `models.PyramidSearch` coarse to
  fine with three sub-pixel fits and three ways to carry, beside the scroll the scenes recorded,
  and with notebook 6's stop decision fitted on its own history.
- `09_extrapolate_recursive.ipynb` — **a motion per block, as a TV finds it**: 3-D recursive search
  (`helpers/recursive.py`) seeded by the tile searches, tuned on train clips, run in order over every
  validation clip beside the searches, notebook 4's blocks, a full search per block and the motion
  oracle.
- `10_extrapolate_masked.ipynb` — **a network where the search is unsure**: `models.Masked`, the
  search's N+1 plus a two-scale network's correction through a gate opened by the search's own
  confidence per cell; trained on whole client tiles (`helpers/masked.py`), beside the gate held
  open and a fallback to N with no network.
- `11_extrapolate_recorded.ipynb` — **real against drawn**: every method's validation score by kind
  and by frame rate, and of each kind's changed tiles how much is motion, and how often the tile
  search finds it.
- `12_extrapolate_masked_real.ipynb` — **where the masked network's training comes from**: trained
  on the drawn tiles only, on the recorded only, and on both (notebook 10), scored by kind.
- `13_extrapolate_decide.ipynb` — **carry or repeat, decided per tile**: twelve features of the
  search's work and the stream's bitrate (`helpers/decide.py`), the search's gate raised, and
  logistic regression and a small network trained on every kind, the drawn and the recorded.

The code the notebooks share lives in `helpers/`. `sources.py`, `degrade.py` and `metrics.py`
are copies of upscale's. `scenes.py` is the drawn desktop motion. `recorded.py` lists the screen
captures and splits recordings by video. `runs.py` reads the run set back, holds the one
definition of a changed sample and, in `valid`, of a recorded sample a live capture could give.
`clips.py` rebuilds a split's whole clips from the manifest and cuts them as the client's
`planTiles` does. `models.py` holds every method with the client's two-input contract.
`training.py` handles loading, grouped scoring and upscale's training loop. `speed.py` is the
timing. `show.py` draws motion for the notebooks. `hints.py` is the host's measurement of a
hint. `budget.py` labels tiles by what moved and scores notebook 7's oracles. `history.py` keeps
each tile's search motion over a clip. `recursive.py` is 3-D recursive search and the loop that
runs any method over whole clips in order. `masked.py` cuts moving whole tiles and trains the
masked network. `decide.py` gives every tile the search's measurements and the score of carrying
its motion and of repeating. `webexport.py` is the export into `dist/frame_gen_extra/`.

## The methods at a glance

Every method gets received frames N−2 and N and makes N+1: change the brightness of each pixel,
draw it with a network, or copy pixels of N from where a motion points. Gains over repeat on the
changed validation samples (80 px spots), and over the published search on the moving whole
tiles (capped at 50 dB) for the later notebooks.

| method | notebook | how it makes N+1 | learns? | result |
| --- | --- | --- | --- | --- |
| repeat | 1 | N again | no | 28.96 dB on the changed samples |
| linear (the mock) | 1 | N + (N − (N−2)) per pixel: lands on N+2 | no | −4.12 dB vs repeat |
| half step | 1 | N + ½ (N − (N−2)) | no | −2.02 dB |
| shift, tile / 16 px blocks | 1 | the whole-pixel offset that best carries N−2 onto N, half of it applied to N | no: a search | +3.85 / +3.53 dB |
| two-scale network | 2 | a network draws a correction added to N | yes | +0.72 dB |
| learned motion | 3 | a network outputs an offset per pixel; N copied from there | yes | +0.72 dB |
| **tile search** (published) | 3 | 81 motions' costs averaged over the tile, a softmax, one motion, half applied | **no** | **+3.57 dB** (+2.69 on whole tiles) |
| search + correction | 3 | the tile search plus a learned correction per pixel | the correction | no epoch beat the search |
| match picker per cell | 4 | per cell, the candidate motion that best carries N−2 onto N | no | +0.30 dB over the search, changed whole tiles |
| hint (shelved) | 5 | the motion N → N+1 the host measured on the frame it did not send | no | +3.16 dB over the search, changed whole tiles |
| history table, three frames | 6 | per bin of how far the motion went a pair earlier, the share of it to carry | 12 numbers | +0.14 dB over the search, moving tiles |
| pyramid + stop decision | 8 | a coarse-to-fine sub-pixel search, carried whole, with a history table | a table | +0.15 dB |
| 3-D recursive search | 9 | a motion per 16 px block from a few penalised candidates | no | +0.03 dB (+0.42 video, +0.24 screens) |
| masked network | 10 | the search's N+1 plus a correction through a gate opened where the search is unsure | yes | +0.06 dB |
| carry-or-repeat decision | 13 | per tile, the search's N+1 or N, chosen by a small network on the search's measurements | 500 weights | **+0.22 dB** |

## The data

**A sample is received frames N−2 and N, and the target N+1** - every second frame received,
whatever the clip's rate. For a drawn scene (60 fps) and for `video` (50-60 fps) the client
receives 25-30 fps and makes the frame between; for most screen captures (24-30 fps) it receives
12-15 fps and makes the real frame between - the same task at half the rate, twice the motion
between received frames. Inputs and target are decoded frames at the stream's resolution: the
whole clip goes through one H.264 encode (`-bf 0`, zero latency: nothing of N+1 reaches the
received frames) and the received frames are every second of it.

**Repeated frames.** A recorder that captures slower than its video's rate repeats frames; then
N+1 is a copy of N - repeat exact by construction - or already of N+2. Each recording keeps the mean
|luma change| between its consecutive source frames, and `runs.valid` keeps a received frame N only
where the steps from N−2 to N+2 all change (over 0.25 luma levels) or none does. It drops much:
29 of the 465 screen clips and 83 of the 245 `video` clips have no run of valid samples at all -
many "60 fps" uploads are slower video doubled.

**Drawn scenes start at a random phase.** A scene's motion is tuned per frame at 30 fps and read at
60; read from the first frame, every change of speed and every menu would land on a received frame
and the frame between would lie exactly halfway. The phase puts half of them on the frame to make.

**Scores are reported over the samples that changed** - more than 0.5% of the core's pixels move by
more than 8 luma levels from N to N+1 - and for whole tiles also over the **moving** ones (the
search's gate open: something moved between N−2 and N), each capped at 50 dB: "changed" cannot
judge a stop, since a tile whose motion stopped has N+1 = N. On the validation set 1,129 of the
3,523 samples changed: page 197, desktop 25, photo 145, screen 410, video 352. On the unchanged
ones repeat is at 80.6 dB; an average over every sample or tile (60+ dB) is mostly those.

## Static methods

Notebook 1, on the 80 px validation samples:

| changed samples (1,129) | wPSNR | SSIM-Y | vs repeat |
| --- | --- | --- | --- |
| repeat | 28.96 | 0.7740 | |
| linear (the mock) | 24.84 | 0.6926 | −4.12 dB |
| half step | 26.94 | 0.7326 | −2.02 dB |
| **shift, tile** | **32.80** | **0.8534** | **+3.85 dB** |
| shift, 16 px blocks | 32.49 | 0.8495 | +3.53 dB |
| shift, 8 px blocks | 31.92 | 0.8425 | +2.97 dB |

By kind (wPSNR, changed): repeat → tile shift - photos 22.98 → 35.43, pages 23.13 → 30.11, screens
32.37 → 34.01, video 30.77 → 32.11, desktops 28.08 → 28.76. A scroll or a pan is one motion, and
the shift copies it; a screen or a video changes in ways it does not follow. At the client's halo
of 4 the tile shift scores 31.52 (32.69 at 8): a motion further than the halo came from outside the
tile. Of the changed scrolls and pans, 93% move 15 px or less between received frames.

## The two-scale model

Notebook 2. Two stride-2 convolutions, three at a quarter of the resolution, `DepthToSpace` back up:
35,868 parameters, 5,775 operations a pixel, 3.6 ms of GPU a 1080p frame in the browser. It starts
as repeat and gains **+0.72 dB** on the changed samples (29.68), at most +1.7 on photos and +1.0 on
screens; the control without downsampling learns nothing (its best epoch is the 4th, +0.00). It
hedges: unsure where content goes, it draws a little of it everywhere, and on the unchanged samples
it falls from repeat's 80.56 to 68.42 (79.09 behind a gate).

## Motion

Notebook 3. The output is an offset per pixel and `current` is read there (`models.warp`), so a
correct offset copies the content exactly.

| changed samples (1,129), halo 8 | wPSNR | vs repeat | ops a pixel |
| --- | --- | --- | --- |
| repeat | 28.96 | | 0 |
| exact shift (notebook 1) | 32.69 | +3.73 dB | |
| two-scale network (notebook 2) | 29.68 | +0.72 dB | 5,775 |
| learned motion | 29.68 | +0.72 dB | 5,342 |
| **tile search, stride 2, own gate** (published) | **32.53** | **+3.57 dB** | **479** |
| search + correction | 32.53 | no epoch beat the search | 5,821 |

- **Learned motion** - notebook 2's encoder with a head that makes the offset, starting at zero -
  gains a fifth of the search. Per pixel a desktop's motion is ambiguous; what makes the search right
  is adding the matching cost over the whole tile before choosing.
- **Tile search** (`models.TileSearch`): luma at half resolution, `previous` moved by each of the 81
  offsets within ±4 there (±8 px) by one fixed `Conv`, |current − previous moved| averaged over the
  tile, a softmax of the costs times 10, their weighted mean the motion; `fraction` ½ carries half of
  it. Stride 2 scores as stride 1 (32.53 each); at the client's halo of 4, 32.30. Nothing learns.
- **The gates.** The search's own gate (`still` = 0.1): its cost at no motion is the mean luma
  difference over the tile, so under 0.1 levels the motion is zero and `current` comes back as it
  is - unchanged samples 79.44 dB against 75.02 ungated and `Gate`'s 79.49, the changed ones unmoved.
- **By kind** (wPSNR, changed), the published search: pages 29.45, desktops 31.26, photos 35.00,
  screens 34.04, video 31.55.

**Speed, in the browser** (measured once, on the first set; the graphs do not depend on the data):
headless Chrome, WebGPU on the RTX 3090, one 1080p frame as 36 tiles in one run - the mock 0.25 ms
of GPU, the two-scale network 3.6, the tile search at stride 1 5.7, **at stride 2 2.33** (the
published graph), at stride 4 1.4. Every operator on the GPU, `GridSample` included. On the desktop
CPU the graph takes 1.6 ms a tile.

**WebGL cannot run it.** ONNX Runtime's WebGL provider has no `ConstantOfShape` or `ReduceSum`, nor
`GridSample`; desktop-streamer falls back to WebGL on a browser without WebGPU, and there its
enhancer turns every enhancement off when this graph fails to load.

## The ceiling of choosing

Notebook 4, on the validation clips rebuilt whole and cut into the client's tiles: 10,241 tiles,
3,469 changed.

| changed whole tiles | wPSNR | vs repeat |
| --- | --- | --- |
| repeat | 29.15 | |
| linear (the mock) | 24.89 | −4.26 |
| **tile search (published)** | **31.84** | **+2.69** |
| tile search ±16 px | 31.70 | +2.55 |
| per 64 / 32 px block | 31.73 / 31.17 | +2.58 / +2.03 |
| match picker + blocks, per 32x32 (no training) | 32.14 | +2.99 |

By kind, repeat → search: photos 23.76 → 31.13, pages 25.47 → 29.66, screens 33.04 → 34.42, video
30.21 → 30.96, desktops 29.53 → 29.62. **The oracle** - per cell the candidate closest to N+1 among
repeat, the tile's motion, per-block motions and the cost's next two minima - is +2.42 dB over the
search per 4x4 cell, +0.41 choosing only repeat or the tile per tile; it is as large on screens
(36.89 against 34.42) and video (33.83 against 30.96) as on pages. **A picker without training** -
per cell the candidate that best carries N−2 onto N - is +0.30 dB over the search per 32x32 cell on
the changed tiles, gaining on screens and video (+0.5 per 16x16) and losing on pages and photos
(−0.4), whose content moves as one.

## A hint from the host

**Shelved: no real host can send it.** It needs the frame between two the stream sends, and
desktop-streamer's host captures at the rate it streams. Notebook 5 measures what it would be worth.

| changed whole tiles (3,460) | all | page | desktop | photo | screen | video |
| --- | --- | --- | --- | --- | --- | --- |
| repeat | 29.10 | 25.55 | 29.53 | 23.72 | 32.93 | 30.22 |
| tile search (published) | 31.79 | 29.68 | 29.62 | 31.09 | 34.30 | 30.98 |
| hint per tile, whole stream px | 33.91 | 35.69 | 30.69 | 32.18 | 35.08 | 32.33 |
| **hint per tile, half px, the host's** | **34.95** | **37.97** | 30.87 | 34.81 | 35.17 | 32.70 |
| hint per 32 px block, whole px | 35.55 | 35.96 | **35.58** | 32.81 | **36.86** | **35.13** |

- **It knows when a scroll stops**: +3.16 dB over the search on the changed tiles, +8.3 on pages. On
  a wheel-scrolled clip it averages 45.7 dB a frame, the search 41.1, repeat 42.9.
- **On recordings one motion per tile is worth little even known exactly** (screens +0.87); per 32 px
  block it is +2.6 on screens and +4.2 on video, at ten times the bits.
- **The host can afford it**: from an eighth of the source with ±1 refinements it finds the exhaustive
  search's motion on 92% of the changed tiles - 0.74 million absolute differences a tile. Two bytes a
  tile: 17 kbit/s at 1080p; per block 178 kbit/s. A quarter of the hints late: still 34.12.

## Motion history

Notebook 6. The search's motion for N−4 → N−2 is the one it found a received frame ago; with N−6 →
N−4 as well a tile has four frames of history. Every method decides only how much of the newest
motion to carry - a share from 0 (repeat) through ½ (the search) to 1 - scored on the **moving** tiles
capped at 50 dB, the fair column for a stop. `r` is how far the motion one pair earlier went along
the newest, as a share of it.

| moving validation tiles, ≤ 50 dB | wPSNR | vs the search |
| --- | --- | --- |
| repeat | 36.65 | −1.44 |
| published search (½) | 38.09 | |
| constant acceleration / exponential slowing, slowing only | 37.92 / 37.91 | −0.17 / −0.17 |
| stop after a slowdown, r ≥ 2 | 38.20 | +0.11 |
| table, two frames: speed × match (control) | 38.15 | +0.06 |
| **table, three frames: r** | **38.23** | **+0.14** |
| table, four frames: r × r before | 38.05 | −0.04 |
| oracle: repeat or ½, per tile | 38.73 | +0.64 |
| oracle: best of 17 shares, per tile | 39.51 | +1.42 |

The history helps the drawn scrolls (pages +0.63, desktops +0.60) and not the recordings (screens
−0.10, video +0.06); a fourth frame's evidence does not hold up off the training clips; physics
loses. Most stops are not announced in the frames before.

## Where the error comes from

Notebook 7. Every validation tile labelled by what touched it between N−2 and N+1 - for a drawn
scene by what it recorded, for a recording only `screen` or `video` - and the search's weighted
squared error split by family; three oracles, each seeing N+1: **share** (the best of 17 shares of
the tile's motion), **cell** (the best of notebook 4's candidates per 16x16), **motion** (the true
motion per 16 px block, searched between N and N+1).

| active tiles, by family | tiles | of the search's error | share takes | cell takes | motion takes |
| --- | --- | --- | --- | --- | --- |
| screen | 5,169 | 35.3% | 5% | 11% | 35% |
| drawn, the whole tile moves | 1,828 | 31.6% | 47% | 31% | 74% |
| video | 2,134 | 17.4% | 20% | 37% | 75% |
| drawn, part of the tile moves | 642 | 9.1% | 11% | 24% | 68% |
| unannounced (anything starting, a menu at N+1) | 222 | 6.4% | 2% | 11% | 81% |

- **On screens most of the error is not motion**: even the true motion per block takes back a third.
- **On video the motion is there, per block, not per tile.**
- **Drawn whole-tile motion is part measurement**: on steady scrolls the best share takes back 43% -
  the search is half a stream pixel off, and past its ±8 px reach for the fastest scrolls (37% of
  that label's error), where a ±16 px search is off by 1 px.

## Measuring the motion better

Notebook 8. `models.PyramidSearch`: every offset within ±4 at a quarter of the resolution (±16 stream
px), then ±1 at half and ±2 at full resolution, ending below a pixel by an equiangular fit.

| moving validation tiles, ≤ 50 dB | scroll measured off by (median) | wPSNR | vs the search | 95% over clips |
| --- | --- | --- | --- | --- |
| published: ±8 px | 0.61 px | 38.07 | | |
| pyramid ±16 px, equiangular, carried bilinearly | 0.09 px | 37.64 | −0.43 | −0.65 to −0.24 |
| pyramid, whole-pixel carry | 0.09 px | 37.99 | −0.08 | −0.22 to +0.06 |
| published + stop decision (notebook 6's table) | | 38.15 | +0.08 | +0.01 to +0.15 |
| **pyramid, whole-pixel carry + stop decision** | | **38.21** | **+0.15** | −0.04 to +0.34 |

Reading a decoded frame between its pixels blurs it, worst on screens (−1.0 dB): measure finely,
carry whole pixels. With its own stop decision the pyramid gains +0.86 dB where a drawn tile moves
whole, nothing on the recordings. 1.33 times the published search's luma differences, 1.9 times its
CPU time; the stop decision needs per-tile state the client does not keep.

## 3-D recursive search

Notebook 9. Per 16 px block, a few candidates - the neighbours' vectors and its own, its own from the
pair before, small updates, zero, and seeds from the tile searches and notebook 4's 32 px blocks -
each costing its mean absolute luma difference plus a penalty by kind (`helpers/recursive.py`). Every
block updates at once from the pass before, three passes.

| moving validation tiles, ≤ 50 dB | vs the search | 95% over clips | whole tile moves | part moves | screen | video |
| --- | --- | --- | --- | --- | --- | --- |
| published search | 38.10 | | 30.69 | 30.15 | 42.68 | 34.18 |
| full search per 8 px block, ±16 px | −1.68 | −2.02 to −1.36 | 27.39 | 28.92 | 41.29 | 33.73 |
| **3DRS** | **+0.03** | −0.20 to +0.25 | 30.23 | 29.35 | **42.92** | **34.60** |
| 3DRS + pointer seed | +0.03 | −0.20 to +0.26 | 30.20 | 29.65 | 42.92 | 34.60 |
| motion oracle (sees N+1) | +3.23 | | 36.20 | 36.05 | 44.35 | 38.89 |

On the changed tiles +0.33 dB; with the stop decided perfectly 39.10 against the search's 38.74. It
wins where a tile holds more than one motion (video, screens) and loses where a drawn motion stops.
53 candidates a pixel - 14 times the published search's luma differences.

## A network behind the mask

Notebook 10. `models.Masked` keeps the published search's N+1 and adds a correction only where the
search cannot account for what moved: per 16 px cell, `u` = the mean |N − N−2 moved by the search's
motion| over the mean |N − N−2| (+1 level); a gate g = sigmoid(a·u + b), learned; notebook 2's
two-scale network given N, the search's N+1, the residual, `u` and the motion.

- **The mask finds the error**: pixels with `u` over 0.6 are 19% of the moving tiles and hold 86% of
  the search's error. Repeat is as bad there (85%).
- **The gate learns it**: g = 0.13 where the search explains the change, 0.27 where it explains
  nothing.
- **What comes through is a hedge**: +0.06 dB on the moving tiles (95% over clips +0.00 to +0.13),
  −0.04 on the changed, −0.24 on all; the same with the gate held open, and +0.08 for a fallback to N
  with no network. 6,542 operations a pixel against the search's 479.

## Real against drawn

Notebook 11. The validation set by kind - 28 pages, 12 desktops and 15 photos drawn; 144 screen
clips of 30 videos (25-30 fps) and 33 video clips of 9 videos (50-60 fps) - and every method's score
split by it. 17% of the recordings' received frames are dropped for repeated frames.

| changed tiles | page | desktop | photo | screen | video |
| --- | --- | --- | --- | --- | --- |
| tiles | 1,310 | 184 | 955 | 1,652 | 1,452 |
| repeat | 24.69 | 29.30 | 23.49 | 32.39 | 30.13 |
| published search | 29.32 | 29.33 | 31.06 | 34.17 | 30.85 |
| search over repeat, 95% over videos (clips for drawn kinds) | +4.6 (+1.8 to +7.3) | +0.0 | +7.6 (+4.5 to +10.1) | +1.8 (+0.1 to +3.8) | +0.7 (−1.0 to +1.6) |
| 3DRS (notebook 9's run) | 29.72 | 29.69 | 30.11 | 34.75 | 31.74 |
| a whole-tile motion of 1 px or more (N → N+1) | 76% | 32% | 81% | 26% | 51% |
| of those, the search carries it (0.75-1.25x) | 35% | 2% | 58% | 48% | 36% |
| true motion, one per tile (sees N+1) | 35.74 | 30.59 | 33.20 | 34.88 | 32.51 |
| true motion, per 16 px block (sees N+1) | 35.79 | 39.31 | 34.81 | 37.79 | 36.32 |

- **One motion per tile is the drawn scenes' world**: most of their changed tiles move as one, and
  there the search gains 5-8 dB. A quarter of a screen's change is one motion, half of a video's.
- **The rest of a screen's change is not motion**: a window fading, a menu, text, a highlight.
- **The recordings' motion is per block**: the true block motion is 3-6 dB above the search on screens
  and video, against 0-1.6 for one motion per tile; 3DRS takes +0.6 to +0.9 of it on the changed
  tiles.
- With every frame from N−2 to N+2 new, the screens' gain (+1.8) is larger than on the first set's
  screen recordings, where uneven capture timing made the search carry half a motion (+0.4).

## Where the training comes from

Notebook 12. Notebook 10's masked network trained from scratch on the drawn tiles only (3,670) and on
the recorded only (6,186), beside notebook 10's own, trained on both (9,856):

| moving validation tiles, ≤ 50 dB, vs the search | drawn | 95% over clips | recorded | 95% over videos |
| --- | --- | --- | --- | --- |
| drawn only | +0.00 (no epoch beat the search) | | +0.00 | |
| recorded only | −0.36 | −0.60 to −0.16 | +0.07 | −0.06 to +0.20 |
| both (notebook 10) | −0.03 | −0.13 to +0.06 | +0.12 | −0.01 to +0.25 |

The drawn scenes teach it nothing the search does not do; the recordings teach it a correction that
blurs a scroll; both together teach it to stay quiet where the search explains the change. On every
source it is within a tenth of a dB of the search.

## A decision per tile

Notebook 13. Between the search's N+1 and N, which cannot blur: twelve features of the search's own
work (`helpers/decide.py`) - its cost at no motion and at its best, how much of the change the best
motion explains, how sharp and how ambiguous the match is, the motion, the residual after aligning
N−2, the share of cells it leaves unexplained, how many pixels changed and by how much - and the
stream's bitrate and loop filter.

| moving validation tiles, ≤ 50 dB | page | desktop | photo | screen | video | all | vs the search | 95% over videos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| repeat | 32.06 | 38.88 | 31.12 | 42.74 | 34.13 | 37.62 | −1.19 | −1.88 to −0.49 |
| search (gate 0.1) | 34.11 | 38.29 | 35.84 | 43.24 | 34.20 | 38.80 | | |
| still gate raised (best of 0.1-8 levels) | 34.11 | 38.29 | 35.84 | 43.24 | 34.20 | 38.80 | +0.00 | |
| learned, logistic, every kind | 34.41 | 38.73 | 36.07 | 43.26 | 34.63 | 38.99 | +0.19 | +0.04 to +0.43 |
| **learned, small network, every kind** | **34.57** | **38.74** | **36.14** | 43.25 | **34.64** | **39.03** | **+0.22** | +0.05 to +0.48 |
| learned, small network, drawn kinds only | 34.23 | 38.90 | 36.04 | 43.20 | 34.65 | 38.94 | +0.14 | −0.09 to +0.41 |
| learned, small network, recorded kinds only | 33.65 | 38.55 | 34.63 | 43.28 | 34.09 | 38.57 | −0.23 | −0.51 to +0.04 |
| the better of the two per tile (sees N+1) | 35.58 | 38.99 | 36.67 | 43.36 | 35.11 | 39.42 | +0.61 | +0.36 to +0.94 |

It leans on how much of the change the best motion explains, how clear the best match is against
the next, and how good it is. It gains where a motion stops or is not a motion (pages, desktops,
video) and nothing on screens, where the search and repeat are within half a dB. It needs every kind
to learn from. Twelve numbers the search mostly computes and a 500-weight network: the one learned
piece worth putting beside the search.

## In desktop-streamer

`extrapolate2x_tilesearch_r4_s2.onnx` copied over the client's
`src/client/web/media/models/extrapolate.onnx` passes all 14 tests of its
`tests/enhance.test.js`: two inputs, `[N, 3, H, W]` with the symbolic names it reads, and the
symbolic dimensions patched to a fixed shape for WebGL. It has been checked against torch at
36 and 4 tiles of 328×188 and at one tile of 64×32 (a frame smaller than a tile). What the
client does with it, as the code stands:

- with **extrapolate** on alone, a received frame is shown at 0 and the generated one at ½ of
  the frame interval - frame N+1 of the 2x stream, which is what the graph makes;
- with **interpolate and extrapolate** both on, the generated frame is shown at ⅔, a third of
  an interval after the received one. The graph still carries half the motion, so it lands
  half again too far;
- on **WebGL** it does not load (see "Motion").

`../mock/make_mock_models.py` still writes the mock `extrapolate.onnx`; running it again puts
the mock back.

## History

- **Until commit 61f83fb** the set was 30 fps with every frame received, and the task was filling a
  stall: consecutive frames N−1 and N in, N+1 out.
- **Until commit 51a2096** the set was drawn scenes and 50-60 fps YouTube video only (11,445 train
  runs): every scene a still image moved by whole pixels, most of it one rigid window, and every
  change of speed, stop and menu on a received frame. There the tile search was +11.09 dB over
  repeat on the changed samples and +5.4 on whole tiles - its own model of the world. Scored on real
  screen recordings (that commit's notebook 11) it gained +0.37 where N+1 was a new frame. The
  notebooks of that commit hold the drawn set's results; this set replaced it.

## Open questions

- **Does a guess read better than a repeated frame, live?** 2x generation shows a guess every other
  frame; the scores treat its errors as pixel error, not as judder. On real screens the search and
  repeat are within half a dB.
- **What do live 60 fps desktops look like?** The screen captures here are edited YouTube tutorials
  at 24-30 fps, recompressed by YouTube. A few minutes of real 60 fps desktop recording, read by
  `helpers/recorded.py` the same way, would say what a live session gains.
- **Can the client's own knowledge stand in for the hint?** When the peer drives, it knows its own
  wheel and drag input. A scroll stopping and a window following the pointer are the two cases two
  frames cannot predict.
- **Video is still thin at 60 fps**: 49 videos, 11 of them to validation, and 83 of the 245 clips
  dropped for repeated frames.

## Next

1. **Run the published search in a live session**, and time it on a laptop iGPU - the 3090 numbers
   are a floor. With a real 60 fps recording of the session kept as a test set.
2. **Put the carry-or-repeat decision in the graph** (notebook 13): its twelve features are the
   search's own numbers, and a 500-weight network is nothing beside it - +0.22 dB, and no state to
   keep between frames.
3. **Fix the two client cases the graph does not fit**: offer extrapolate only on WebGPU, and either
   drop the ⅔ placement when interpolate is also on or export a graph that carries a third of the
   motion for it.
4. **Skip still tiles in the client.** The graph returns them unchanged, but still runs the search on
   them; comparing the two received tiles first would save that work.

Sibling problems: `../upscale/summary.md`, `../frame_gen_intra/summary.md`.
