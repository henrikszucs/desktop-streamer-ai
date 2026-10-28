# Frame generation, extra — summary

**2x frame generation without latency.** The host streams every second frame of the display
rate - 30 of 60 a second - and the client draws the one in between from the two it received
last, *before* the next one arrives. Received frames N−2 and N in, the unsent frame N+1 out.
Interpolation (`../frame_gen_intra/`) would wait for N+2 and add a received interval of
latency; extrapolation adds none. desktop-streamer's client already schedules it this way: an
extrapolated frame is shown half a frame interval after the frame it came from.

> **Status: six notebooks, and a graph that passes desktop-streamer's client tests in place
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

The code the notebooks share lives in `helpers/`. `sources.py`, `degrade.py` and `metrics.py`
are copies of upscale's, so a change made for one problem cannot quietly change the other's
data or units. `scenes.py` is the desktop motion. `runs.py` reads the run set back, picks the
received frames, and holds the one definition of a changed sample. `clips.py` rebuilds a
split's whole clips from the manifest and cuts them as the client's `planTiles` does.
`models.py` holds every method with the client's two-input contract. `training.py` handles
loading, grouped scoring and upscale's training loop. `speed.py` is the timing. `show.py`
draws motion for the notebooks: arrows of what moved, changes and errors on one colour
scale, a search's costs, and a motion followed over time. `hints.py` is the host's
measurement of a hint, as notebook 5 scores it. `webexport.py` is the export
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
| hint (shelved) | 5 | the motion from N to N+1 the host measured on the frame it did not send, applied to N - needs a host capturing at twice the stream's rate | no: a search on the host | +5.95 dB over the search, whole tiles |

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
- **Is a third received frame the evidence a picker lacks?** Whether a scroll stops is half
  the oracle's gain, and two frames cannot show deceleration. Carrying the acceleration
  outright lost on the 30 fps set; as an input to a picker it is untested.
- **Whole frames are 640×360 here.** A 1080p stream has 36 tiles, each a smaller share of the
  screen, so one motion a tile should hold more often there.
- **The scenes are drawn, not recorded.** Their speeds, pauses and event rates are guesses
  about how people use a desktop, and 60 fps is reached by reading a 30 fps motion between its
  frames. A few minutes of real 60 fps screen recording would check them, and there is no
  reader for one yet.
- **Video is thin**: 250 training and 10 validation sequences at 50 fps or more. More would
  need `datasets/download.py` to fetch them - a download only its user should decide on.

## Next

1. **Run the published search in a live session.** It is in desktop-streamer's client,
   uncommitted. Turn on extrapolate in a room, watch a scroll and a dragged window at 60 fps,
   and time it on a laptop iGPU - the 3090 numbers are a floor, and an Intel UHD has about a
   fortieth of the compute.
2. **A learned picker only if a 2-epoch smoke test earns it.** Notebook 4 bounds what choosing
   among motions could add at ~2 dB, and nothing without the answer has found any of it.
3. **Fix the two client cases the graph does not fit**: offer extrapolate only on WebGPU, and
   either drop the ⅔ placement when interpolate is also on or export a graph that carries a
   third of the motion for it.
4. **Skip still tiles in the client.** The graph returns them unchanged, but still runs the
   search on them; comparing the two received tiles first would save that work, which is most
   of a desktop's tiles most of the time.

Sibling problems: `../upscale/summary.md`, `../frame_gen_intra/summary.md`.
