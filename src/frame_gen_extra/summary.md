# Frame generation, extra — summary

**2x frame generation without latency.** The host streams every second frame of the display
rate - 30 of 60 a second - and the client draws the one in between from the two it received
last, *before* the next one arrives. Received frames N−2 and N in, the unsent frame N+1 out.
Interpolation (`../frame_gen_intra/`) would wait for N+2 and add a received interval of
latency; extrapolation adds none. desktop-streamer's client already schedules it this way: an
extrapolated frame is shown half a frame interval after the frame it came from.

> **Status: fourteen notebooks, and a graph that passes desktop-streamer's client tests in place
> of its mock.** `dist/frame_gen_extra/extrapolate2x_tilesearch_r4_s2.onnx` has the mock's
> contract exactly - `previous`, `current` and `output`, each `[N, 3, H, W]` with `N`, `H`, `W`
> symbolic - and all 14 tests of desktop-streamer's `tests/enhance.test.js` pass with it as
> `extrapolate.onnx`. Not yet run in a live session.
>
> **The best generator is a search, not a network.** Notebook 3 rebuilds notebook 1's tile
> shift from fixed convolutions: the cost of every offset within ±8 px between the two
> received frames, on half-resolution luma sampled every second position and averaged over
> the tile, a softmax to pick one motion, and **half** of it applied to N with `GridSample`;
> a tile whose cost at no motion is under 0.1 luma levels gets no motion at all, so a still
> tile comes back as it was. Nothing in it learns. On the 749 validation samples where frame
> N+1 differs from N it scores **36.78 dB wPSNR against 25.69 for showing N twice (+11.09 dB,
> 0.8604 SSIM-Y against 0.7040)**, 36.42 at the client's halo of 4, and +4.36 dB over repeat on
> every sample. **In Chrome's WebGPU on the RTX 3090 it costs 2.34 ms of GPU a 1080p frame**
> (36 tiles in one run), every operator on the GPU: a seventh of the 16.7 ms a generated
> frame has at 60 fps.
>
> **The mock the client runs is worse than doing nothing.** Its `2 · current − previous` aims
> at N+2 and scores **21.94 dB against repeat's 25.69**; aimed at N+1 (half a step) it is
> 23.94. A moved edge is drawn twice, once too dark and once too light.
>
> **It works on everything a desktop stream shows**: +14.0 dB on scrolled pages, +12.6 on
> panned photos, +2.4 on video, +1.2 on desktops (a dragged window, a menu, a region
> scrolling), +6.2 around the pointer.
>
> **No trained network beats it.** Asked for pixels (notebook 2, +0.91 dB) or for motion
> (notebook 3, +2.01 dB), a network trained on two received frames hedges, and on a desktop
> that means blur. A learned correction on top of the search only drifts off it.
>
> **On whole frames it gains half as much, and a learned picker could add at most ~2 dB.**
> Cut into the client's 328×188 tiles (notebook 4), the search is +5.75 dB over repeat on the
> changed tiles, not +11.09, and nothing on desktops. An oracle that picks per 4×4 cell among
> six fixed candidate motions - repeat, the tile's, per-block ones, the cost's other minima -
> is +2.09 dB over the search; it sees N+1, so a real picker gets less. Picking by how well a
> candidate explains N−2 → N loses to the search outright.
>
> **The search is the two-frame answer.** Carrying less of the motion, or gating it per cell,
> does not beat it (notebook 4's numbers, "The ceiling of choosing"). A hint from the host -
> the motion of the frame it does not send - would be worth +5.95 dB (notebook 5), but it needs
> a host capturing at twice the stream's rate, and desktop-streamer's captures at the rate it
> streams. It was prototyped there and reverted; the graph that takes it is not published.
>
> **Two more received frames are worth about +0.3 dB** (notebook 6). Kept per tile, the
> search's motions of the pairs before - N−4 → N−2, N−6 → N−4 - show a motion slowing down,
> and a table of 24 shares fitted on them (none where the motion was 1.6 times as fast a pair
> earlier, half where it is steady) gains +0.33 dB over the search on the tiles that moved
> (95% over clips +0.20 to +0.49), +0.64 on pages. Rules from physics - constant
> acceleration, a flick's decay - lose. Most stops are not announced in the frames before.
>
> **Much of the search's error is measurement, not prediction** (notebook 7, an error budget
> by what each scene recorded). Of its error on the tiles that move, 55% is on tiles that move
> whole, 23% on tiles that move in part (a dragged window, a scrolled region), 18% on what
> starts without warning. On steady scrolls the search's motion is half a pixel off, and past
> ±8 px it misses fast ones by ~13 px - a wider, sub-pixel search fixes that with no network.
> The tiles that move in part are where a network belongs: notebook 4's candidates take a
> quarter of their error, the true motion per block 73%.
>
> **Measured better and stopped in time, +0.8 dB** (notebook 8, no training). A coarse-to-fine
> search (`models.PyramidSearch`: ±16 px at a quarter of the resolution, refined to full, an
> equiangular fit below a pixel) measures the scroll to 0.10 px instead of 0.50, and carrying
> whole pixels of it keeps N sharp. With notebook 6's stop decision on its history it gains
> +0.77 dB on the moving tiles (95% over clips +0.44 to +1.11): a stopping scroll 34.2 → 40.8
> dB. Twice the published graph's CPU time; the stop decision needs per-tile state the client
> does not keep yet.
>
> **A motion per block measures the part that moves, and loses for it** (notebook 9, no
> training). 3-D recursive search - a TV's block motion estimator: a few candidates per 16 px
> block, penalised when new to the neighbourhood - finds the region or the window one motion
> per tile misses, and carries it past the frame it stops: **−0.40 dB** on the moving tiles. With
> the stop decided perfectly it is +0.21 over the published search, +0.09 over the pyramid,
> and takes back 18% of the search's error where part moves (the true motion per block 73%),
> for 14 times the work. Part of a tile is a question of prediction before measurement.
>
> **A network behind the search's own mask hedges too** (notebook 10). How much of the change
> the search's motion leaves unexplained, per 16 px cell, puts 59% of its error in 18% of the
> pixels; a gate on it, learned, opens to 0.35 there and stays at 0.07 elsewhere. The network
> behind it is +0.06 dB on the moving tiles (95% over clips −0.00 to +0.12), −0.37 on all - the
> same with the gate held open, and the same as falling back to N there with no network - for
> 14 times the search's operations. Where the search is unsure, two frames do not say more.
>
> **On real screens the gains shrink to a few tenths of a dB** (notebook 11). The scenes above
> are drawn: a still image moved by whole pixels, mostly as one rigid window, every change on a
> received frame - the tile search's own model. On 81 real screen recordings from YouTube-8M's
> computer categories (24-30 fps, none seen in training), where N+1 is a new frame, the search
> is **+0.37 dB** over repeat on the changed tiles (95% over videos −0.02 to +0.77), 3DRS +0.74,
> the masked network +0.61 - against +5.41 for the search on the drawn set. Real screens change
> in ways a translation does not follow. Every number above is the drawn set's, and the headline
> gains are mostly its own.
>
> **Training the masked network on real screens does not help** (notebook 12). Split by video,
> 42 to train, 32 to test: from scratch it learns nothing (−0.05 dB against the search where N+1
> is new); fine-tuned from the drawn scenes it is quieter where nothing changes (+0.15 on the
> moving tiles) but no better where something does (+0.07, 95% over videos −0.38 to +0.32,
> against the drawn-scene network's +0.16). Most real training tiles are codec noise over a still
> screen. 3DRS is the one method clearly ahead there: +0.60 dB over the search.
>
> **A learned carry-or-repeat decision per tile learns the kind of content, not the tile**
> (notebook 13). Twelve numbers from the search's own work and a 500-weight network: on the real
> test videos it repeats, as good as repeat (+0.38 over the search on the moving tiles, the
> ceiling +0.50); on the drawn set it carries, as good as the search (−0.07). Per tile it finds
> nothing beyond that: +0.04 of the +0.13 left over repeat on real screens, none of the +0.83 over
> the search on the drawn set. A raised still gate takes +0.05 at most.

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
own rate with no frame generation at all, and the frames a model sees have none - the data
here leaves it out too (`DRAW_CURSOR = False` in notebook 00).

## Notebooks

Run in order, from this folder:

- `00_data_preprocess.ipynb` — the run set, at the display rate. Sources are the drawn desktop
  screens (`datasets/screens.py`, 400 train and 80 validation), DIV2K photos (100 and 25),
  the YouTube sequences at 50 fps or more (250 and 10 - of 4,444 and 555; a 30 fps video has
  no frame between two a 30 fps stream sends), and `datasets/raw/captures/` (synthesised
  desktop frames when all of that is empty). A still picture becomes a one-second, 60 fps
  clip of a 1280×720 window **animated as a desktop in use** (`helpers/scenes.py`): a page
  scrolls by wheel notches, a flick or a steady drag, a photo pans, a desktop holds still
  while a region inside it scrolls, a window is dragged along the pointer's path or a menu
  pops up where the pointer rests - the pointer itself not drawn. The motion is tuned per
  frame at 30 fps and read at 60, so the same pixels a second. Then the stream's own ×2
  degradation from `helpers/degrade.py` (the same as upscale's: random filter, H.264 at
  0.2–1.0 bpp, loop filter for half, no codec for a tenth) on the whole clip → runs of 8
  consecutive decoded frames of one spot, starting on a received frame (run frames 0, 2, 4,
  6 received, 1, 3, 5, 7 to be made), a 64 px core with a halo of 16. Two runs per clip are
  cut around the pointer, where windows are dragged and menus open; of the others, flat ones
  are dropped and still ones kept a fifth of the time. Split by source → one `uint8` `.npy`
  array of `(runs, 8, 96, 96, 3)` per split and a `manifest.json` in
  `datasets/frame_gen_extra/` (with `received_every: 2`, `fps: 60`, `cursor_drawn: false`, and
  each source's scene, degradation and runs). 11,445 train and 1,677 validation runs, 2.9 GB,
  10 minutes. `helpers/runs.py` reads it back.
- `01_extrapolate_dummy.ipynb` — **static methods only.** Repeat, linear (the mock), half step,
  and three motion shifts carrying half the motion (one offset per tile, per 16 px block, per
  8 px block). No training; quality by group and by halo, how far content moves, speed, and
  the three that ONNX expresses as plain arithmetic exported at the client's tile.
- `02_extrapolate_twoscale.ipynb` — **the first network**, built for the browser: `current`
  plus a correction made at 1/2 and 1/4 of the resolution (`models.TwoScale`), beside a
  control of the same cost with no downsampling (`models.Plain`). Trained at a halo of 8,
  scored at 8 and 4, checkpointed to `checkpoints/extrapolate_twoscale.pt`, exported into
  `checkpoints/export/` - it is not worth publishing. See "The two-scale model".
- `03_extrapolate_motion.ipynb` — **motion, not pixels**: `current` moved by an offset per pixel
  with `GridSample` (`models.Motion`). The offset comes from a learned head (notebook 2's
  encoder), from `models.TileSearch` (notebook 1's tile shift built from fixed convolutions,
  nothing learned), or from both. Scores the gates, and publishes the tile search: the
  client's graph and a fixed 36-tile copy for the benchmark page. See "Motion".
- `04_extrapolate_oracle.ipynb` — **the ceiling of choosing**, before training a picker: the
  validation clips rebuilt whole and cut into the client's tiles, the search per tile and per
  block, an oracle over fixed candidate motions per cell, and a picker that needs no training.
  Nothing is trained or exported; a few minutes. See "The ceiling of choosing".
- `05_extrapolate_hint.ipynb` — **a hint from the host**: the motion of each tile from N to the
  N+1 the host captured, measured on the clean frames - exhaustively and as cheaply as a host
  can afford - scored per tile and per block, its bits, late hints falling back to the
  search, and the hinted graph exported to `checkpoints/export/` - shelved: no real host captures
  the frame it is measured on. Nothing is trained. See "A hint from the host".
- `06_extrapolate_history.ipynb` — **the motion history**: the search's motion of every tile
  for the last three received pairs - what a client could keep between frames - and N moved
  on by each of 17 shares of the newest motion, scored against N+1, on all 753 train and 115
  validation clips rebuilt whole. The per-tile ceiling of choosing a share, rules from
  physics, and tables fitted on two frames (the control) against three and four. Nothing is
  trained or exported; the tables are cached in `checkpoints/history/`. See "Motion history".
- `07_extrapolate_budget.ipynb` — **where the search's error comes from**: every validation tile
  labelled by what its scene recorded between N−2 and N+1 (a menu, a drag, a region, the
  scroll - starting, stopping, changing speed or going on - video, nothing), the search's
  summed error split by label, and how much of it three oracles take back: the best share of
  the tile's motion, the best of notebook 4's candidates per cell, and the true motion per
  block. Nothing is trained; cached in `checkpoints/budget/`. See "Where the error comes from".
- `08_extrapolate_pyramid.ipynb` — **measuring the motion better**: the published search, one
  reaching ±16 px, and `models.PyramidSearch` coarse to fine with three sub-pixel fits and
  three ways to carry its motion (bilinear, bicubic, whole pixels), all on the same frames
  (`history.load_many`); each set beside the scroll the scenes recorded, scored, and given
  notebook 6's stop decision fitted on its own history. The whole-pixel pyramid exported with
  the client's contract to `checkpoints/export/` (not published). See "Measuring the motion better".
- `09_extrapolate_recursive.ipynb` — **a motion per block, as a TV finds it**: 3-D recursive
  search (`helpers/recursive.py`) seeded by the tile searches, tuned on 40 train clips, and run
  in order over every validation clip beside the tile searches, notebook 4's blocks, a full
  search per 8 px block and notebook 7's motion oracle - all on the same frames, scored raw and
  with the stop decided perfectly, by family and as notebook 7's error budget. Without its
  memory, with the pointer as a seed, and other variants. Nothing is trained or exported;
  cached in `checkpoints/recursive/`. See "3-D recursive search".
- `10_extrapolate_masked.ipynb` — **a network where the search is unsure**: `models.Masked`, the
  search's N+1 plus a two-scale network's correction through a gate opened by the search's own
  confidence per cell; trained on whole client tiles (11,125 moving tiles cut from the train
  clips into the data folder, `helpers/masked.py`), beside the same network with the gate held
  open and a fallback to N with no network, scored on every validation tile as notebook 9.
  Checkpoints in `checkpoints/masked_runs/`; nothing exported. See "A network behind the mask".
- `11_extrapolate_recorded.ipynb` — **real screens**: the YouTube-8M sequences of the computer
  categories that are screen captures, picked by eye (81 at 24-30 fps, 4 at 50-60 fps, 10 of
  Synthesia apart), through the stream's downscale and codec (`helpers/recorded.py`) and scored
  on every client tile as notebooks 9 and 10, beside the drawn validation set; how often one
  motion per tile fails there; and the clips' repeated frames, so the fair test - N+1 a new
  frame - can be read apart. Nothing is trained or exported; cached in `checkpoints/recorded/`.
  See "Real screens".
- `12_extrapolate_masked_real.ipynb` — **the masked network trained on real screens**: notebook
  11's screen videos split by video within each category (42 to train, 7 to choose epochs, 32 to
  test), the network trained from scratch and fine-tuned from notebook 10's weights, and both
  scored on every test tile beside repeat, the searches, 3DRS and notebook 10's network.
  Tiles and trainings in `checkpoints/masked_real/`; nothing exported. See "Real screens".
- `13_extrapolate_decide.ipynb` — **carry or repeat, decided per tile**: twelve features of the
  search's work and the stream's bitrate for every tile (`helpers/decide.py`), the search's
  still gate raised, and logistic regression and a small network trained on real screens, on the
  drawn clips and on both, scored on notebook 12's test videos and the drawn validation set
  against the per-tile ceiling. Nothing exported; cached in `checkpoints/decide/`. See "Real
  screens".

The code the notebooks share lives in `helpers/`. `sources.py`, `degrade.py` and `metrics.py`
are copies of upscale's, so a change made for one problem cannot quietly change the other's
data or units. `scenes.py` is the desktop motion. `runs.py` reads the run set back, picks the
received frames, and holds the one definition of a changed sample. `clips.py` rebuilds a
split's whole clips from the manifest and cuts them as the client's `planTiles` does.
`models.py` holds every method with the client's two-input contract. `training.py` handles
loading, grouped scoring and upscale's training loop. `speed.py` is the timing. `show.py`
draws motion for the notebooks: arrows of what moved, changes and errors on one colour
scale, a search's costs, and a motion followed over time. `hints.py` is the host's
measurement of a hint, as notebook 5 scores it. `budget.py` labels tiles by their scenes'
events and scores notebook 7's oracles. `history.py` keeps each tile's search
motion over a clip and scores carrying any share of the newest, as notebook 6 uses it.
`recursive.py` is 3-D recursive search and the loop that runs it over whole clips in order,
as notebook 9 scores it. `masked.py` cuts moving whole tiles to train on and trains notebook
10's network on them (notebook 12 on real screens). `recorded.py` makes a test split of real screen recordings in the
manifest's shape. `decide.py` gives every tile the search's own measurements and the score of
carrying its motion and of repeating, as notebook 13 fits a decision on them. `webexport.py` is the export
into `dist/frame_gen_extra/` - the two frames and any per-tile input such as the hint - at
fixed shapes or with `N`, `H` and `W` symbolic as the client needs.

## The methods at a glance

Every method gets received frames N−2 and N and makes N+1. Three ways to make it: change
the brightness of each pixel, draw it with a network, or copy pixels of N from where a
motion points (`GridSample`). Scores are wPSNR over repeat on the 749 changed samples.

| method | notebook | how it makes N+1 | learns? | vs repeat |
| --- | --- | --- | --- | --- |
| repeat | 1 | N again | no | |
| linear (the mock) | 1 | N + (N − (N−2)) per pixel: lands on N+2 | no | −3.75 dB |
| half step | 1 | N + ½ (N − (N−2)) per pixel | no | −1.76 dB |
| shift, tile / blocks | 1 | the whole-pixel offset that best carries N−2 onto N, searched over every offset, half of it applied to N | no: a search | +10.11 / +8.68 dB |
| two-scale network, control | 2 | a network draws a correction added to N | yes, all of it | +0.91 / +0.34 dB |
| learned motion | 3 | a network outputs an offset per pixel; N copied from there | yes, all of it | +2.01 dB |
| **tile search** (published) | 3 | the cost of 81 motions averaged over the tile, a softmax, one motion; half of it applied to N | **no**: fixed convolutions, numbers chosen by measuring | **+11.09 dB** |
| search + correction | 3 | the tile search plus a learned correction per pixel | the correction only, and no epoch of it beat none | +11.09 dB (untrained) |
| `Gate`, `still` | 3 | rules: repeat where the two frames match | no | (on unchanged tiles) |
| oracle | 4 | per cell, the candidate motion closest to N+1 | not a model: it reads the answer | +2.09 dB over the search, whole tiles |
| match picker | 4 | per cell, the candidate that best carries N−2 onto N | no | −0.13 to −0.87 dB under the search |
| history: stop after a slowdown | 6 | none of the newest motion if the pair before went twice as far, else half | no | +0.19 dB over the search, moving tiles |
| history: table, four frames | 6 | per bin of how fast the two pairs before went against the newest, the share that scored best on train | a table of 24 numbers | +0.33 dB over the search, moving tiles |
| hint (shelved) | 5 | the motion from N to N+1 the host measured on the frame it did not send, applied to N - needs a host capturing at twice the stream's rate | no: a search on the host | +5.95 dB over the search, whole tiles |
| 3-D recursive search | 9 | per 16 px block, the cheapest of a few candidates - neighbours, the pair before, small updates, zero, the tile searches' motions - plus a penalty by kind; half applied to N | no: a search, penalties chosen by measuring | −0.40 dB under the search, moving tiles (+0.21 with a perfect stop) |
| masked network | 10 | the tile search's N+1, plus a network's correction through a gate opened where the search's motion leaves the change unexplained | the network and the gate | +0.06 dB over the search, moving tiles; −0.37 on all |

Everything that learns made N+1 worse than the search that does not. A network trained on
two frames hedges - under L1, a little of every place content might have gone - and on a
desktop that is blur. The notebooks show it: notebook 1 follows a scroll over time and draws
each shift's motion against the true one; notebook 2 draws what each network *added* to N
next to what had to change; notebook 3 shows the search's 81 costs and softmax weights, and
every model's motion field; notebook 4 follows a wheel scroll frame by frame, where the
search wins while it runs and loses the frame it stops.

## The data

**A sample is received frames N−2 and N, and the target N+1.** `runs.load` and `runs.Runs` take
the spacing from the manifest (`received_every`), so the inputs are always frames the client
got; a run of 8 has three such samples (N = 2, 4, 6), and up to four received frames of
history (N−6 … N) for a later notebook. Training picks one at random; validation fixes one
per run, spread over the run by its index.

**Input and target are decoded frames, at the stream's resolution** (640×360 from a 1280×720
window): the client works from what it decoded, before upscaling. The whole 60 fps clip goes
through one H.264 encode and the received frames are every second of it - simpler than
encoding them alone, at the price of a codec that predicts over half the time a real 30 fps
stream does.

**Scores are reported over the samples that changed.** A sample is *changed* if more than
0.5% of its core's pixels move by more than 8 luma levels from N to N+1 (`runs.CHANGED_LEVEL`,
`runs.MOVING_SHARE`). 749 of the 1,677 validation samples changed. On the unchanged ones
repeating is at 84.8 dB: a score that mixes them flatters everything.

How often N+1 differs from N, on a random 3,000 of the train runs: page 54% (820 runs),
desktop 26% (702), photo 45% (414), video 58% (1,064). Of the 748 train clips, 416 have a
pointer, 102 a dragged window, 101 a menu, 69 a scrolling region, and 238 a scrolling
background (86 wheel, 85 flick, 67 steady). The runs are 832 cut around the pointer, 7,224
that move and 3,389 still ones in train (164, 898 and 615 in validation). Frames one apart
differ by 6.5 luma levels on average and frames two apart by 9.4: the codec round trip kept
them in order. Video is thin in validation - 99 changed samples from 10 sequences.

## Static methods

Notebook 1: every method gets N−2 and N, is scored on N+1 over the 64 px core
(`metrics.report`: wPSNR is Y'CbCr weighted 6:1:1, SSIM on luma), and the shifts search every
offset within the halo (16) and carry half of it.

| changed samples (749) | wPSNR | SSIM-Y | vs repeat |
| --- | --- | --- | --- |
| repeat N | 25.69 | 0.7040 | |
| linear (the mock, aims at N+2) | 21.94 | 0.6326 | −3.75 dB |
| half step (linear aimed at N+1) | 23.94 | 0.6692 | −1.76 dB |
| **shift, tile** | **35.80** | **0.8675** | **+10.11 dB** |
| shift, 16 px blocks | 34.37 | 0.8634 | +8.68 dB |
| shift, 8 px blocks | 33.02 | 0.8569 | +7.33 dB |

On the 928 unchanged samples repeat is at 84.77 dB; the tile shift loses 1.2, linear 7.7.
By kind (changed samples, SSIM-Y · wPSNR), the tile shift against repeat: pages 0.8108 · 35.68
against 0.6723 · 23.49, photos 0.9249 · 36.89 against 0.6462 · 25.04, video 0.9643 · 37.84
against 0.9277 · 34.98, desktops 0.7912 · 27.55 against 0.7668 · 25.94, around the pointer
0.8257 · 31.64 against 0.7170 · 26.15.

- **Brightness extrapolation is the wrong model for a screen.** Nothing on a desktop fades:
  it moves, appears or holds still.
- **A scroll, a pan and a dragged window are translations**, and one per tile is enough;
  smaller blocks only add false matches - repeating text lines match each other.
- **A menu cannot be predicted**: its content is in neither input frame.

**The halo.** Content moves two display frames between the received ones, and anything that
moved further than the halo came from outside the tile:

| changed samples | halo 4 | halo 8 | halo 16 |
| --- | --- | --- | --- |
| shift, tile | 0.8261 · 34.82 | 0.8657 · 35.96 | 0.8675 · 35.80 |
| shift, 16 px blocks | 0.8165 · 32.77 | 0.8561 · 34.15 | 0.8634 · 34.37 |

Between the received frames, the tile shift finds no motion on 15.2% of the changed page and
photo samples, at most 4 px on 59.4%, at most 8 on 86.0% and at most 15 on 97.3%. A search
past ±8 adds false matches about as fast as true ones.

**Published**: `extrapolate_static_{repeat,linear,half_step}_tile320x180.onnx`, the mock's
contract at the client's tile. Repeat is one `Identity` (the exporter drops the unused
`previous` input, and `webexport.export` puts it back); linear and half step are `Mul`, `Sub`,
`Add` and `Clip`.

## The two-scale model

Notebook 2. The two frames side by side, two stride-2 3×3 convolutions (16 and 32 channels),
three 3×3 convolutions at a quarter of the resolution, `DepthToSpace` back to half with the 16
channels beside it, a 3×3 convolution to 12 channels and a last `DepthToSpace` to the 3
colours, added to `current`: 35,868 parameters, 5,775 operations a stream pixel, all `Conv`,
`LeakyRelu`, `Concat`, `DepthToSpace` and `Add`. The last convolution starts at zero, so
untrained it *is* repeat. The control (`Plain`: 8 channels, four 3×3 convolutions at full
resolution) costs 5,947 operations a pixel. In torch on the RTX 3090 a 1080p stream frame
costs 1.9 and 5.4 ms.

Trained at a halo of 8 with `luma_chroma_l1`, Adam from 1e-3, keeping the epoch with the best
SSIM-Y on the changed samples, until it converged: 140 epochs (4.8 minutes, best 134) and 97
(4.3 minutes, best 91).

| changed samples (749), halo 8 | wPSNR | SSIM-Y | vs repeat |
| --- | --- | --- | --- |
| repeat | 25.69 | 0.7040 | |
| **shift, tile** (notebook 1, radius 8) | **35.96** | **0.8657** | **+10.27 dB** |
| two-scale | 26.60 | 0.7340 | +0.91 dB |
| control | 26.03 | 0.7144 | +0.34 dB |

- **It learns to hedge, not to move.** On scrolled pages +0.5 dB where the translation gains
  13.2, on photos +1.8 where it gains 11.3. Under L1, a network unsure where something goes
  draws a little of it everywhere it might go: scrolled lines fade instead of moving.
- **It wins nowhere**: on desktops 26.17 dB against the shift's 27.27, around the pointer
  26.92 against 30.78. With a pointer drawn into the frames (the data before), desktops were
  the one place it led - the pointer was what neither could predict.
- **The quarter-resolution path earns +0.57 dB** over the control, most of it on photos.
- **It damages still tiles**: 69.0 dB on the unchanged samples against repeat's 84.8. The
  gate brings it back to 83.7 (notebook 3).

## Motion

Notebook 3. Asked for pixels, a network has to *draw* moved content. Here the output is an
offset per pixel and `current` is read there (`models.warp`, a `GridSample` with the edge
repeated), so a correct offset copies the content exactly. Three ways to get the offset:

| changed samples (749), halo 8 | wPSNR | SSIM-Y | vs repeat | ops a pixel |
| --- | --- | --- | --- | --- |
| repeat | 25.69 | 0.7040 | | 0 |
| exact shift (notebook 1, not exportable) | 35.96 | 0.8657 | +10.27 dB | |
| two-scale network (notebook 2) | 26.60 | 0.7340 | +0.91 dB | 5,775 |
| learned motion | 27.70 | 0.7753 | +2.01 dB | 5,342 |
| tile search, stride 1 | 36.71 | 0.8622 | +11.02 dB | 1,880 |
| **tile search, stride 2, own gate** (published) | **36.78** | **0.8604** | **+11.09 dB** | **479** |
| search + correction | 36.78 | 0.8604 | +11.09 dB (untrained is best) | 5,821 |

- **Learned motion** - notebook 2's encoder with a head that makes the offset, starting at zero
  (repeat). 188 epochs (6.7 minutes, Adam from 1e-4, best epoch 158 by wPSNR on the changed
  samples) to +2.01 dB: better than asking for pixels, a fifth of the search. Per pixel, a
  desktop's motion is ambiguous - text lines repeat and white matches white - and what makes
  the search right is adding the matching cost over the whole tile before choosing. In
  prototypes on earlier data, neither a multi-scale loss, nor the search as a teacher, nor a
  cost volume handed to the network changed that.
- **Tile search** (`models.TileSearch`) - that, in a graph. Luma of each frame at half the
  resolution (a fixed 2×2 stride-2 `Conv`); `previous` moved by each of the 81 offsets within ±4
  there (±8 px) - one fixed `Conv` from 1 channel to 81, each kernel a single 1 - and `current`
  at the kernels' centre, both with no padding, so only positions where every offset reads
  inside the tile count; `Sub` and `Abs` make |current − previous moved by d|. The mean over
  the tile (`ReduceMean`), a softmax of the 81 costs times a sharpness of 10, and the weighted
  mean of the offsets is the motion between the received frames; `fraction` ½, a constant in
  the graph, carries half of it to N+1. The softmax lands between whole-pixel offsets where
  two match about as well, which is why the search beats the exact whole-pixel shift: half
  the motion between received frames is often half a pixel. `stride` 2 samples the cost on
  every second position and costs nothing here (36.78 against 36.71); stride 4 loses 0.9.
  Sharpness 3 lands between offsets too often and blurs (36.39), 30 is close to an argmin
  (36.69 at stride 1). Nothing learns.
- **Search + correction** - the search plus the learned head's correction per pixel. It starts
  as the search and **no epoch beats that**: the correction drifts off it from the first epoch.
  A fractional offset reads between pixels, which is a blur, and when two received frames
  cannot tell whether a scroll is easing in or out, a slightly short, blurred motion is the
  better bet under L1. Using a third frame for the change in speed does not help either: on
  earlier data, carrying any of the acceleration lost 0.5 dB or more.

**By kind** (changed samples, SSIM-Y · wPSNR), the published search against repeat: pages
0.8083 · 37.46 against 0.6723 · 23.49, photos 0.9157 · 37.61 against 0.6462 · 25.04, video
0.9574 · 37.40 against 0.9277 · 34.98, desktops 0.7633 · 27.16 against 0.7668 · 25.94, around
the pointer 0.7889 · 32.30 against 0.7170 · 26.15.

**The gates.** `models.Gate` repeats a tile where N−2 and N do not differ by the run set's own
definition of a change, comparing every pixel. The search has a cheaper one of its own
(`TileSearch(still=...)`): its cost at no motion *is* the mean luma difference between the two
frames over the tile, so under `still` luma levels the motion is set to zero, and a zero
motion hands `current` back as it is. Both leave the changed scores as they were and lift the
unchanged ones, from 81.12 dB ungated to 83.75 (`Gate`) and 83.69 (`still` = 0.1); over every
sample the search is +4.39 and +4.36 dB over repeat. In the browser `Gate` costs 2 ms of GPU
and 5.5 ms of wall clock a 1080p frame - as much as the search - and the search's own gate
nothing measurable, so that is the one published. (`Gate` brings notebook 2's network from
69.02 to 83.67 on the unchanged samples.) A client that compared the tiles itself could also
skip running the model on them.

**At the client's halo of 4**, the search reaches past the halo, reads the edge repeated there,
and still scores **36.42 dB** (0.8593) - against 34.82 for the exact shift limited to ±4, and
34.88 for the search limited to ±4. The client's tile needs no wider halo for it.

**Speed, in the browser.** `../upscale/benchmark/main.py --models <folder>` serves any folder
of graphs, the page fills every input a graph has, and `?auto=<ids>` measures with no one at
the controls and posts each row to the server (`--results`). Headless Chrome 156 on Windows,
WebGPU on the RTX 3090 (the adapter reports NVIDIA Ampere, not a software fallback),
GPU-resident tensors, the output buffer reused, GPU time from timestamp queries; one 1080p
stream frame is 36 tiles in one run:

| graph, one 1080p frame | GPU ms | wall ms |
| --- | --- | --- |
| linear (the mock) | 0.25 | 0.27 |
| two-scale network (notebook 2) | 3.6 | 3.7 |
| tile search, first form (a 162-channel 9×9 `Conv`, two taps of 81 used) | 13.9 | 15.0 |
| tile search, stride 1 | 5.7 | 5.9 |
| tile search, stride 2 | 2.46 | 2.45 |
| tile search, stride 2, with `Gate` | 4.46 | 7.92 |
| **tile search, stride 2, own gate** (published, `_b36`) | **2.33** | **2.42** |
| the same, the client's dynamic file pinned to 36 tiles | 2.34 | 2.41 |
| tile search, stride 4 | 1.4 | 1.4 |

Every operator runs on the GPU, `GridSample` included. One tile a run costs 0.18 ms of GPU but
0.29 ms of wall clock, so 36 separate runs are 10.4 ms a frame: run a frame's tiles as one
batch, as the client already does. Graph capture changes nothing at batch 36. The graph is 36
KB, and on the desktop CPU it takes 1.6 ms a tile.

**WebGL cannot run it.** ONNX Runtime's WebGL provider has no `ConstantOfShape` or
`ReduceSum` (both refused by name in the browser), nor `GridSample`. desktop-streamer falls back to WebGL on a
browser without WebGPU, and there its enhancer turns every enhancement off and shows the
error when this graph fails to load.

## The ceiling of choosing

Notebook 4. Every network so far hedged. The combination of learning and search left is a
small network that *picks*, per cell, one of a few motions the search proposes - a choice,
which cannot be averaged into a blur. Before training one, this measures what it could gain.

**Whole frames.** The 113 validation clips are made again from the manifest - file, scene,
degradation (`clips.rebuild`; within x264's run-to-run difference of the stored runs, about
3 levels at 0.2 bpp) - and cut into the client's tiles exactly as `planTiles` cuts them: 6,700
tiles of 328×188 at every second received frame, 2,480 changed, each scored on the region
the client keeps.

| wPSNR, changed | 80 px samples (749) | whole-frame tiles (2,480) | all tiles (6,700) |
| --- | --- | --- | --- |
| repeat | 25.69 | 27.09 | 62.84 |
| linear (the mock) | 21.94 (−3.75) | 23.22 (−3.86) | 56.99 (−5.84) |
| **tile search (published)** | **36.78 (+11.09)** | **32.84 (+5.75)** | **64.47 (+1.64)** |
| tile search, ±16 px | 36.00 (+10.31) | 32.80 (+5.71) | 64.45 (+1.61) |
| per 64 px block | 36.78 (+11.09) | 32.41 (+5.32) | 63.90 (+1.06) |
| per 32 px block | 35.49 (+9.80) | 31.46 (+4.37) | 63.08 (+0.25) |

On whole tiles the search gains half as much, and on desktops nothing (28.66 against
repeat's 28.68): a 320×180 region counts as changed when a dragged window or a menu moves a
corner of it, and one motion cannot move the corner without the rest. Pages drop most, 37.46
on samples to 32.42. Neither a longer reach nor smaller blocks help. A third of the page
frames that scroll move more than 8 px between received frames, but those are wheel notches
easing in and out, where half the motion from N−2 to N is not the motion to N+1. Smaller
blocks bring back the false matches notebook 1 found.

**The oracle** takes, per cell, the candidate closest to the true N+1, by the error the score
weighs. It sees the answer, so it bounds any picker over those candidates from above. On the
changed whole-frame tiles, over the published search:

| candidates | per pixel | per 4×4 | per 8×8 | per 16×16 | per tile |
| --- | --- | --- | --- | --- | --- |
| repeat, tile | +1.78 | +0.96 | +0.82 | +0.74 | +0.50 |
| + per-block motions (64, 32 px) | +2.72 | +1.66 | +1.45 | +1.27 | +0.84 |
| + the cost's next two minima | +3.77 | **+2.09** | +1.77 | +1.56 | +1.10 |

Per pixel it picks noise as well; per 4×4 is what a picker at a quarter of the resolution
decides. By kind (per 4×4, all six candidates): pages 35.10 against the search's 32.42,
desktops 30.08 against 28.66, photos 34.90 against 33.50, video 39.74 against 37.23, around
the pointer 31.76 against 29.31. Two kinds of choice make it up:

- **whether a motion goes on.** The tiles it gains most on are pages where a wheel notch
  ended between N and N+1: the search carries the motion on (18 dB) and the oracle repeats
  (40). Choosing only repeat or the tile, once per tile, is +0.50;
- **where each motion is.** A window dragged or a region scrolled over a still desktop,
  where it takes the per-block motions for the part that moves.

**A picker without training** - per cell, the candidate whose motion carried the whole way
best takes N−2 to N - loses to the search at every cell size: −0.13 dB per 32×32, −0.34 per
16×16, −0.87 per 8×8 (repeat, tile and blocks; with the minima, worse). Within a cell, two
frames are as ambiguous as within a pixel. Whether a scroll stops is not in them at all.

So a learned picker could gain at most ~2 dB on changed tiles, and a realistic one less. The
candidates are not the limit; the evidence for choosing is.

**Two static changes, measured afterwards, do not help either.** Carrying less than half the
motion - betting on the scroll stopping - loses where it goes on more than it wins where it stops
(32.84 dB at a half, 31.66 at 0.45, 29.57 at 0.3). A gate per cell between the tile's motion and
none, chosen by which explains N−2 → N better there, is at best the search (32.87 at 64 px,
32.66 at 16) and worse the more readily it picks none. The published search stays the
two-frame method.

## A hint from the host

**Shelved: no real host can send it.** It needs the frame between two the stream sends, and
desktop-streamer's host captures at the rate it streams. What follows is what that frame would
be worth, kept as the measure of the gap two frames leave.

Notebook 5. The host captures at the display rate and sends every second frame, so it has
N+1 - the frame the client makes - before dropping it. Measuring how each client tile moved
from N to N+1 there, and sending that, gives the client what notebook 4 found it lacks:
whether a scroll goes on.

**The measurement** (`hints.measure`) is notebook 1's shift between the clean frames N and
N+1 - before the stream's shrink and codec, as the host has them - for every client tile.
The host's form starts at an eighth of the source, ±4 pixels (±32 source, ±16 stream), and
refines by ±1 at each finer level down to the source: 0.74 million absolute differences a
tile, 0.8 billion a second for a 1080p stream's 36 tiles at 30 hints a second. It finds the
same motion as an exhaustive search of all 1,089 offsets on 94% of the changed tiles.

| wPSNR, changed whole-frame tiles (2,488) | all | page | desktop | photo | video | pointer |
| --- | --- | --- | --- | --- | --- | --- |
| repeat | 27.15 | 25.87 | 28.62 | 25.91 | 36.16 | 25.79 |
| tile search (published) | 32.85 | 32.46 | 28.60 | 33.49 | 37.19 | 29.44 |
| hint per tile, whole stream px | 37.32 | 40.54 | 28.85 | 35.60 | 37.46 | 35.11 |
| hint per tile, half px, exhaustive | 38.88 | 42.04 | 28.92 | 38.05 | 37.80 | 36.36 |
| **hint per tile, half px, the host's** | **38.80** | **41.88** | 28.90 | **38.01** | 37.78 | **36.44** |
| hint per 32 px block, whole px | 37.78 | 39.47 | **33.72** | 35.96 | **40.70** | 36.30 |
| hint per tile at the stream's resolution, parabola, ±2 from 1/8 (desktop-streamer's `hint.js`) | 38.58 | 41.40 | 29.00 | 37.95 | 37.89 | 36.20 |

- **It knows when a scroll stops.** Pages gain 9.4 dB over the search, around the pointer 7.0.
  On notebook 4's wheel-scrolled clip the hinted frames average 49.2 dB, the search's 38.8 and
  repeat's 41.5.
- **Half a stream pixel is worth 1.5 dB.** A host whose grey frames are the stream's own size
  gets it back with a parabola through the costs either side of the best offset (38.68 from
  ±4 at a quarter, 38.58 from ±2 at an eighth for half the work - the form desktop-streamer
  runs).
- **Per tile it still cannot follow a desktop** (+0.3). Per 32 px block it can (+5.1), and
  video gains too, but pages lose 2.4 to false matches and whole pixels, for ten times the
  bits. A block hint at half pixels, sent only where a block leaves its tile's motion, is
  untried.
- **Bits**: two signed bytes a tile, (x, y) in half stream pixels - 1.9 kbit/s at 640×360,
  17 kbit/s at 1080p. Per block, a bit per block and two bytes for each that moves apart from
  its tile: 174 kbit/s at 1080p.
- **Late hints** fall back to the search, tile by tile: with a quarter of them late the changed
  tiles score 37.52, with all of them late exactly the search.

**The hinted graph**, `checkpoints/export/extrapolate2x_hinted_tilesearch_r4_s2.onnx` (37.5
KB, not published): `previous` and `current` as before, and `hint`, `[N, 3]` float32 - per tile, in the
client's tile order, the motion (x, y) from N to N+1 in stream pixels and 1 if it arrived (0:
the published search's guess). Checked against torch at 36 and 4 tiles and at one of 64×32,
with half the hints missing. It runs the search either way, so it costs what the published
graph does plus a few elementwise operators; its browser time is not measured yet.

**Prototyped in desktop-streamer and reverted.** The host's ffmpeg line captured at twice the
stream's rate and split it - even frames to the encoder, every frame as grey down a third pipe -
and a worker measured each pair (12 ms a 1080p frame on one desktop core, the search at the
stream's resolution with a parabola, 38.58 dB). It ran against a real ffmpeg and found a scroll
on every tile. It was taken out because a real host cannot capture faster than it streams; a
raw ffmpeg output left to its own pacing also doubles its first frame, which would have put
every hint a frame off - worth knowing for anything that pairs two ffmpeg outputs.

## Motion history

Notebook 6. The client keeps nothing between frames today, but the search's motion for
N−4 → N−2 is the one it found a received frame ago: two numbers a tile to keep. With the
motion of N−6 → N−4 as well, a tile has four frames of history. Every method here only
decides how much of the newest motion to carry to N+1 - a share from 0 (repeat) through ½
(the search) to 1 - so every picture is still N moved by one motion, as sharp as the
search's. The run set was rebuilt for it (11,550 train runs, against the 11,445 above: the
helpers changed since), and the whole clips with it: 73,600 train tiles and 12,185
validation tiles at every received frame from N = 6, 6,700 of which move between N−2 and N.

**How it is scored.** *Changed* tiles - N+1 differs from N - cannot judge a stop: a tile
whose motion stopped has N+1 = N, so the changed tiles leave out every stop a method catches
and keep every one it misses. All tiles have them, but there a tile that barely changed sits
at 60-80 dB and a fraction of a pixel moves that by tens of invisible dB. So the column that
matters here is the **moving** tiles - the gate is open, something moved between N−2 and N -
each capped at 50 dB, and the tables are fitted on the train clips' moving tiles the same way.
A first run fitted on uncapped scores chose repeat wherever a motion was not steady and lost
to the search on the changed tiles.

`r` is how far the motion one pair earlier went along the newest, as a share of it: 1
steady, 2 for a motion that halved its speed, 0 for one that just started.

| wPSNR, validation tiles | moving, ≤ 50 dB | changed | page, moving | desktop, moving | pointer, moving |
| --- | --- | --- | --- | --- | --- |
| repeat | 33.41 (−3.14) | 26.24 | 32.06 | 35.03 | 31.45 |
| published search (½) | 36.55 | 31.68 | 34.56 | 34.59 | 33.85 |
| constant acceleration, slowing only | 36.44 (−0.10) | 31.18 | 34.68 | 34.83 | 33.87 |
| exponential slowing, slowing only | 36.39 (−0.16) | 31.26 | 34.49 | 34.71 | 33.79 |
| stop after a slowdown, r ≥ 2 | 36.74 (+0.19) | 31.68 | | | |
| table, two frames: speed × match (control) | 36.35 (−0.20) | 31.28 | 34.55 | 34.74 | 33.73 |
| table, three frames: r | 36.74 (+0.19) | 31.37 | 35.05 | 35.03 | 34.29 |
| **table, four frames: r × r before** | **36.88 (+0.33)** | **31.82** | **35.20** | 34.86 | **34.35** |
| oracle: repeat or ½, per tile | 37.47 (+0.92) | 32.19 | 36.06 | 35.07 | 34.90 |
| oracle: best of 17 shares, per tile | 38.92 (+2.37) | 34.28 | 37.88 | 35.17 | 36.36 |

- **The history holds the signal, in few tiles.** On the moving train tiles, a motion that
  went 1.6 times as far a pair earlier is best not carried at all, +3 to +6 dB over the half
  on those tiles; a steady one is best carried exactly half, by several dB. A motion that
  just started or reversed is best not carried either: there the search mostly found a
  change that was not a motion. The slowdowns are a few percent of the moving tiles.
- **Physics loses**, as acceleration did on the 30 fps set: a wheel notch eases out over a
  few frames and then stops, and a smooth extrapolation of its speed is wrong both ways.
  Stopping outright after a slowdown works, with nothing fitted.
- **The four-frame table** gains +0.33 dB on the moving tiles (95% over clips +0.20 to +0.49;
  better on 43% of the clips, worse on 20%), +0.14 on the changed ones. On notebook 4's wheel
  clip it catches two of the three stops (44.9 dB a frame against the search's 43.7); the
  third follows a move past any search's reach.
- **It is the history, not the fitting**: the two-frame control gains +0.22 on train and loses
  0.20 on validation; the history tables hold their train gain.
- **The reach does not limit it.** Read with the published search's own ±8 px instead of ±16,
  the history tables gain +0.14 to +0.30 dB, within the spread between clips - the client can
  read the history from the motions it keeps.
- **A third of choosing between repeat and the half, a seventh of the ceiling.** Most stops
  are not announced in the frames before - a notch ends, a drag is let go - and a tile holding
  a still part and a moving one wants a share between.

Not in the client: it needs the motions kept per tile between frames and a graph input for
them. At +0.3 dB it is not worth that change on its own; it would ride along with whatever
else gives the graph per-tile state.

## Where the error comes from

Notebook 7. Before a network goes on top of the search: what would it have to learn? Every
scene records what happened on each frame, so every validation tile at a generated frame is
labelled by what touched it between N−2 and N+1, and the search's error - the weighted squared
error the score uses, summed over the kept region, so it adds up across tiles - is split by
label. 7,728 of the 13,155 tiles are active (the gate is open, or N+1 differs from N); they
hold all but 0.04% of the search's error. Three oracles, each seeing N+1: **share**, the best
of 17 shares of the tile's motion; **cell**, the best of notebook 4's candidates per 16x16;
**motion**, the true motion per 16 px block, searched between N and N+1.

| active tiles, by family | tiles | of the search's error | search dB | share takes | cell takes | motion takes |
| --- | --- | --- | --- | --- | --- | --- |
| whole tile moves (background scroll goes on, changes speed, stops) | 3,529 | 55.0% | 33.95 | 48% | 31% | 76% |
| part of the tile moves (a drag, a scrolled region) | 1,084 | 23.4% | 33.14 | 13% | 25% | 73% |
| unannounced (anything starting, a menu at N+1) | 495 | 17.7% | 25.69 | 1% | 10% | 92% |
| video | 939 | 3.3% | 45.49 | 11% | 28% | 57% |
| a menu that opened before N | 45 | 0.5% | 52.48 | 31% | 37% | 86% |
| nothing recorded | 1,636 | 0.2% | 51.98 | 0% | 0% | −27% |

By label, the largest are the scroll changing speed (25.1%), going on (22.6%), starting
(15.0%), a drag changing speed (13.7%) and the scroll stopping (7.3%).

- **Measurement, not prediction.** On steady scrolls the best share takes back 47% - nothing
  stops there; the search's motion is wrong. Set beside the scroll the scene recorded, it is
  half a stream pixel off below 8 px, and above 12 px between received frames (37% of the
  steady scrolls' error) it is past the ±8 px reach and off by ~13 px; a ±16 px search
  measures those to half a pixel. Notebook 4 found ±16 px alone no better on whole tiles -
  the fast scrolls are the ones easing in and out - so the reach wants the stop decision of
  "Motion history" beside it. No network is needed for either: notebook 5's coarse-to-fine,
  sub-pixel host measurement, run on the client's two frames.
- **Part of the tile is where a network belongs.** The candidates take a quarter of that
  error, the true motion per block three quarters: the right motion is missing from the
  candidates, and finding which part of a tile moves, and how, takes context a per-block search
  lacks. A dragged window follows the pointer exactly, and the client has the pointer.
- **The unannounced sixth is out of any model's reach**: only the motion itself takes it
  back, which the client's own input would know and its frames never do.

## Measuring the motion better

Notebook 8. `models.PyramidSearch` is the published search coarse to fine: every offset within
±4 at a quarter of the resolution (±16 stream px), then the luma of N−2 moved by what was found
and ±1 searched at half the resolution, then ±2 at full resolution every 4th position, ending
below a pixel by a fit through the cheapest offset and its neighbours. All on the same frames
of the 115 validation clips (x264 does not encode a clip the same way twice, so searches are
measured in one pass - `history.load_many`).

| search | measured motion off by (scroll alone, median) | moving tiles, ≤ 50 dB | changed |
| --- | --- | --- | --- |
| published: ±8 px | 0.50 px; 9.6 px past 12 px | 36.59 | 31.70 |
| ±16 px, one level | 0.50 px | +0.10 | +0.19 |
| pyramid ±16 px, softmax / parabola / equiangular, carried bilinearly | 0.39 / 0.11 / 0.10 px | −0.31 / −0.29 / −0.41 | −0.28 / −0.18 / −0.20 |
| pyramid, equiangular, carried bicubically | 0.10 px | −0.44 | −0.24 |
| pyramid, equiangular, **whole-pixel carry** | 0.10 px | +0.04 | +0.27 |
| published + stop decision (notebook 6's table) | | +0.13 | −0.00 |
| **pyramid, whole-pixel carry + stop decision** | | **+0.77** (95% over clips +0.44 to +1.11) | **+0.85** |

- **Reading a decoded frame between its pixels blurs it more than the precision is worth**,
  bilinearly or bicubically. The published search's motions fall mostly on even pixels, so half
  of one is a whole pixel and N is copied exactly. Measure finely, carry whole pixels.
- **Precision pays through the stop decision**: with the motion to a tenth of a pixel, how far
  the motion before went against the newest reads a slowdown that half-pixel steps hid. On the
  moving tiles a stopping scroll goes from 34.2 to 40.8 dB, one changing speed from 23.9 to 26.6,
  a steady one only from 34.8 to 35.1.
- **Cost**: 1.33 times the published search's luma differences a tile; 2.0 times its ONNX
  Runtime CPU time for 36 tiles (three `GridSample`s, `ArgMin`, `Gather`). Not timed in a browser.
- **The stop decision needs per-tile state** - the motions of the two pairs before - and a graph
  input for it. The exported graph (`checkpoints/export/extrapolate2x_pyramid_r16_equiangular.onnx`,
  the client's contract, checked against torch) is the pyramid alone, +0.04 dB.

## 3-D recursive search

Notebook 9. Notebook 7 put a quarter of the search's error on tiles where only part moves,
and notebook 4's per-block searches lost to false matches there. TV frame-rate converters
measure a motion per block without them: **3-D recursive search** (de Haan et al., 1993,
`helpers/recursive.py`). Per block, a few candidates - the neighbours' vectors and its own, its
own from the pair before, three small updates of them, zero, and seeds: the pyramid's motion of
the tile, the next two minima of the published search's cost, its motion per 32 px block, and
as a variant the pointer's - each costing its mean absolute luma difference plus a penalty by
kind, so a vector new to the neighbourhood wins only where it matches clearly better. Every
block updates at once from the pass before, three passes: the form a GPU can run. Ties go to
the seeds before zero, and a block that took the tile's motion carries the pyramid's measure of
it; without either, a first run lost on every setting. Tuned on 40 train clips: 16 px blocks;
the size of the penalties hardly matters. Run in order over every validation clip, on the same
frames as everything it is set beside:

| moving tiles, ≤ 50 dB | raw | 95% over clips | perfect stop | part moves | part, perfect stop | error taken back where part moves, perfect stop |
| --- | --- | --- | --- | --- | --- | --- |
| repeat | 33.44 (−3.13) | | 33.44 | 29.77 | 29.77 | −12% |
| published search | 36.57 | | 37.49 | 29.94 | 30.55 | 6% |
| pyramid, whole-pixel (notebook 8) | 36.61 (+0.04) | −0.12 to +0.23 | 37.61 | 29.85 | 30.59 | 11% |
| 32 px blocks (notebook 4) | 35.28 (−1.29) | −1.60 to −0.99 | 37.08 | 27.70 | 30.66 | 9% |
| full search, 8 px blocks, ±16 px | 33.70 (−2.87) | −3.43 to −2.34 | 36.56 | 27.13 | 30.63 | 14% |
| **3DRS** | **36.16 (−0.40)** | −0.71 to −0.10 | **37.70** | 27.87 | 30.83 | **18%** |
| 3DRS + pointer | 36.04 (−0.53) | −0.85 to −0.19 | 37.70 | 27.95 | 31.00 | 23% |
| motion oracle (notebook 7) | 40.64 (+4.07) | | 40.73 | 35.97 | 36.02 | 73% |

*Perfect stop*: per tile, the better of the method's N+1 and repeat - what the field is worth
beside a stop decision that never misses; it sees N+1.

- **It measures the part that moves, and carries it past the stop.** A drag or a region that
  stops between N and N+1: 32.4 and 31.1 dB, against the tile search's 41.0 and 46.6 - the tile
  search missed the window and so repeats it. Most drags change speed (520 of the 1,034 moving
  tiles where part moves): 26.8 against 27.4. Where the motion goes on it wins: a region
  scrolling steadily 28.1 against 26.9, a drag 25.7 against 25.3 (26.4 with the pointer).
- **Smoothness is what makes a motion per block usable**: the same blocks searched exhaustively
  lose 2.87 dB, 3DRS 0.40. Its memory (the pair before as candidates) adds nothing (−0.34
  without it); the tile-wide seeds alone lose least (−0.31).
- **It costs 14 times the published search**: 53 candidates a pixel, 3.27 M luma differences a
  tile against 0.27 M, plus the two searches it is seeded by.

So better measurement per block is worth at most +0.1 to +0.2 dB beside a perfect stop decision,
and loses without one. The part of a tile that moves on its own needs to know when it stops -
a stop decision per block from the history, or the client's own input - before it needs to
know where it went.

## A network behind the mask

Notebook 10. Every network so far worked on the whole picture. `models.Masked` keeps the
published search's N+1 and adds a correction only where the search cannot account for what
moved: per 16 px cell, `u` = the mean |N − N−2 moved by the search's motion| over the mean
|N − N−2| (+1 level), clipped to 0..1; a gate g = sigmoid(a·u + b) per pixel, learned; and
notebook 2's two-scale network given N, the search's N+1, the residual, `u` and the motion.
The correction starts at zero, so untrained it is the search. Trained on whole client tiles -
16 moving tiles from each train clip (11,125), epochs kept by the moving validation tiles
capped at 50 dB - and scored on every validation tile as notebook 9:

| moving tiles, ≤ 50 dB | raw | 95% over clips | perfect stop | part moves | error taken back where part moves | changed | all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| published search | 36.57 | | 37.49 | 29.99 | | 31.71 | 62.83 |
| pyramid, whole-pixel (notebook 8) | 36.61 (+0.04) | −0.13 to +0.22 | 37.60 | 29.88 | −1% | 31.99 | 62.96 |
| fallback: N where u > 0.95, no network | 36.64 (+0.06) | −0.03 to +0.17 | 37.50 | 30.13 | 4% | 31.79 | 62.87 |
| **masked, gated** | **36.63 (+0.06)** | −0.00 to +0.12 | 37.53 | 30.09 | 9% | 31.66 | 62.45 |
| masked, gate held open | 36.63 (+0.06) | +0.01 to +0.12 | 37.52 | 30.08 | 8% | 31.64 | 62.48 |

- **The mask finds the error**: the pixels with `u` over 0.6 are 18% of the moving tiles and
  hold 59% of the search's error. Repeat is worse there (67%); the better of the two per cell
  takes back a tenth.
- **The gate learns it**: a = +1.91, b = −2.54 - g = 0.07 where the search explains the change,
  0.35 where it explains nothing.
- **What comes through is a hedge**: +0.06 dB on the moving tiles, a tenth of the summed error
  on the worst tiles, and a faint copy of moved content where the search was right - a region
  that stopped, its N+1 exact, drawn with a ghost of the text it would have scrolled to.
- **The gate adds nothing over the network alone**, and the network nothing over a fallback with
  no network. It costs 6,542 operations a pixel against the search's 479.

## Real screens

Notebook 11. The drawn scenes are close to the tile search's model of the world: a still image
moved by whole pixels; 44 of the 51 validation pages and 19 of the 25 photos scroll or pan the
whole window, with no toolbar or scrollbar standing still; and since a scene is timed at 30 fps
and read at 60, every change of speed, stop and menu lands on a received frame - no validation
tile has a menu opening at N+1. So the methods are scored again on real screen recordings: the
YouTube-8M sequences of the computer categories (`datasets/yt8m-mix.txt`) that are screen
captures, picked by eye from a frame of each - the categories also hold filmed monitors, people,
and kitchen blenders. 81 videos at 24-30 fps, never in the run set (it took 50 fps and up): the
client receives 12-15 fps and makes the real frame between, twice the motion of the 60 fps case.

Screen videos repeat frames: 232 of their 405 clips do not move, and of the 173 that do, 129 fill
a slower capture up to the video's rate - N+1 is then a copy of N (repeat exact) or of N+2. The
fair test is where N+1 is a new frame:

| changed tiles where N+1 is a new frame | real screens (2,352 tiles, 51 videos) | 95% over videos | Synthesia (2,163, 10 videos) | drawn validation set, all changed tiles |
| --- | --- | --- | --- | --- |
| repeat | 29.98 | | 31.87 | 26.30 |
| mock: linear | −3.25 | −3.60 to −2.82 | −2.84 | |
| published search | **+0.37** | −0.02 to +0.77 | +2.55 | **+5.41** |
| pyramid, whole-pixel (notebook 8) | +0.31 | −0.08 to +0.72 | +1.99 | +5.69 |
| 3DRS (notebook 9) | +0.74 | +0.13 to +1.38 | +5.56 | +5.68 |
| masked network (notebook 10) | +0.61 | +0.22 to +1.03 | +3.13 | +5.36 |

(gains over repeat, dB). Over every tile, repeated frames and all, the search is a little worse
than repeat on real screens: 32.93 dB on the changed tiles against 33.05, 43.45 on the moving ones
against 43.97.

- **Real screens change in ways a translation does not follow**: an annotation that comes and
  goes, a highlight, a fade, a zoom, a still watermark over a moving picture. Motion is worth a
  fifteenth of what the drawn set says.
- **Moving part of a tile matters more there**: 3DRS and the masked network lead the tile search,
  where on the drawn set they lost or tied; the cells one motion per tile cannot explain hold 70%
  of its error, against 59% drawn. At these sizes the order is not settled.
- **Synthesia is the drawn scenes' kind of motion** - bars falling at a constant speed over a still
  keyboard - and there the search gains +2.55 dB and 3DRS +5.56.
- The four 60 fps screen captures say nothing: 15 of their 20 clips do not move.

They are YouTube tutorials - edited, recompressed, at half the frame rate - not a live session.
But the drawn set's headline gains are mostly the drawn set's.

**Trained on them (notebook 12).** The 81 videos split by video within each category - 42 to
train, 7 to choose epochs, 32 to test - and notebook 10's masked network trained on the 5,245
moving tiles of the training videos, from scratch and fine-tuned from its drawn-scene weights:

| test videos, against the published search | moving tiles, ≤ 50 dB | 95% over videos | changed, N+1 a new frame (1,243 tiles, 18 videos) | 95% over videos |
| --- | --- | --- | --- | --- |
| repeat | +0.37 | +0.17 to +0.62 | −0.24 | −0.91 to +0.21 |
| 3DRS (notebook 9) | −0.73 | −1.03 to −0.47 | **+0.60** | +0.29 to +0.92 |
| masked, drawn scenes (notebook 10) | +0.05 | +0.00 to +0.10 | +0.16 | −0.10 to +0.30 |
| masked, real screens from scratch | +0.00 | +0.00 to +0.01 | −0.05 | −0.19 to +0.01 |
| masked, real screens fine-tuned | +0.15 | +0.08 to +0.23 | +0.07 | −0.38 to +0.32 |

From scratch it learns nothing; fine-tuned, it learns to stay quieter over still screens - which
the epoch-choosing score rewards - and no more where something changes. Most real training
tiles are a still screen stirred by the codec, and most real changes worth learning are not
motion. What real screens reward is drawing nothing where nothing moved and moving per block
where something did: 3DRS leads where N+1 is new and loses most over codec noise - a per-block
still gate in front of it is the obvious next step.

**A decision per tile (notebook 13).** Between the search's N+1 and N, which cannot blur:

| moving tiles, ≤ 50 dB, against the search | real test videos | 95% over videos | drawn validation set |
| --- | --- | --- | --- |
| repeat | +0.37 | +0.17 to +0.62 | −2.93 |
| still gate raised (best of 0.1-8 levels, chosen on real training videos) | +0.05 | −0.02 to +0.12 | −0.57 |
| learned, logistic, drawn clips | +0.41 | +0.23 to +0.65 | +0.01 |
| learned, small network, real + drawn | +0.38 | +0.20 to +0.60 | −0.07 |
| learned, trained on real screens only | +0.37 to +0.38 | | −1.80 to −2.20 |
| the better of the two per tile (sees N+1) | +0.50 | +0.31 to +0.75 | +0.83 |

The features (`helpers/decide.py`): the search's cost at no motion and at its best, how much of
the change the best motion explains, how sharp and how ambiguous the match is, the motion, the
residual after aligning N−2, the share of 16 px cells it leaves unexplained, how many pixels
changed and by how much, and the stream's bits per pixel and loop filter. The decision leans on
`explained` above all. It learns the kind of content - repeat what changes like a tutorial,
carry what scrolls - and nothing per tile: what would pick the right picture is what happens
between N and N+1.

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
- on **WebGL** it does not load (above).

`../mock/make_mock_models.py` still writes the mock `extrapolate.onnx`; running it again puts
the mock back.

## History

Until commit 61f83fb the set was 30 fps with every frame received, and the task was filling a
stall: consecutive frames N−1 and N in, N+1 out. The order of the methods was the same - the
mock −2.47 dB against repeat, the tile shift +4.79 dB at halo 8, the two-scale network +1.14,
learned motion +0.81, the tile search +3.99 at stride 2 (+1.34 gated over every sample).
The first 60 fps set drew the pointer into the frames: there the search scored +6.73 dB and
desktops were the one place a network led, because the pointer was what nothing could
predict - and the real stream never contains it. Setting `received_every` to 1 and
`IMAGE_FPS` to 30, or `DRAW_CURSOR` to True, in `00_data_preprocess.ipynb` builds those sets
again; every notebook takes its spacing from the manifest.

## Open questions

- **Does a guess read better than a repeated frame, live?** 2x generation shows a guess every
  other frame; the scores treat its errors as pixel error, not as judder.
- **Can the client's own knowledge stand in for the hint?** When the peer drives, it knows its
  own wheel and drag input, and the host already sends the pointer's position apart from the
  picture. A scroll stopping and a window following the pointer are the two cases two frames
  cannot predict; timing that input against frames that arrive later is unexplored.
- **What announces the stops the frames do not?** Two more received frames (notebook 6) catch
  the stops a slowdown announces, +0.3 dB; the oracle over repeat or the half is +0.9 and over
  every share +2.4. The rest is a notch ending or a drag let go without warning - the client's
  own input (above) - and tiles holding two motions. Measuring the two motions apart (notebook
  9) does not help until their stops are known: a block's motion carried past its stop loses
  more than one motion per tile does.
- **Whole frames are 640×360 here.** A 1080p stream has 36 tiles, each a smaller share of the
  screen, so one motion a tile should hold more often there.
- **The scenes are drawn, not recorded - and it shows.** On real screen recordings (notebook 11)
  the tile search gains +0.37 dB over repeat where the drawn set says +5.41. Those recordings are
  edited YouTube tutorials at 24-30 fps; a few minutes of real 60 fps desktop recording, read by
  `helpers/recorded.py` the same way, would say what a live session gains. And the drawn scenes
  could come closer: still chrome over a scrolling page, line-by-line scrolling without easing,
  typing, fades and zooms, and events between received frames.
- **Video is thin**: 250 training and 10 validation sequences at 50 fps or more. More would
  need `datasets/download.py` to fetch them - a download only its user should decide on.

## Next

1. **Run the published search in a live session.** It is in desktop-streamer's client,
   uncommitted. Turn on extrapolate in a room, watch a scroll and a dragged window at 60 fps,
   and time it on a laptop iGPU - the 3090 numbers are a floor, and an Intel UHD has about a
   fortieth of the compute. Notebook 11 makes this the first question: on real screens the
   search gains a few tenths of a dB over repeat, so whether a guessed frame reads better than a
   repeated one is for eyes to judge, and a real 60 fps recording of the session is the test set
   to keep.
2. **Give the graph per-tile state.** Notebook 8's pyramid with the stop decision is +0.8 dB,
   but the decision needs each tile's last two motions: a graph input `[N, 4]` the client fills
   from the graph's own previous outputs, and a second output with the newest motion. Time the
   pyramid in Chrome's WebGPU first - it is twice the published graph on CPU.
3. **A learned picker only if a 2-epoch smoke test earns it.** Notebook 4 bounds what choosing
   among motions could add at ~2 dB. Notebook 6 found +0.3 of it in the motion history, with a
   table a network on the same features could at most match; a network would need more to
   look at - per-block motions, the pointer - to find more. Notebook 9 bounds the per-block
   motions as input: 3DRS's field beside a perfect stop decision is +0.21 dB over the search,
   so what a picker learns has to be the stop, not the motion. Notebook 10 adds that a network
   drawing pixels where the search is unsure, through a learned mask, gains +0.06 dB: a picker
   is the form left, and only with something beyond two frames to look at.
4. **Fix the two client cases the graph does not fit**: offer extrapolate only on WebGPU, and
   either drop the ⅔ placement when interpolate is also on or export a graph that carries a
   third of the motion for it.
5. **Skip still tiles in the client.** The graph returns them unchanged, but still runs the
   search on them; comparing the two received tiles first would save that work, which is most
   of a desktop's tiles most of the time.

Sibling problems: `../upscale/summary.md`, `../frame_gen_intra/summary.md`.
