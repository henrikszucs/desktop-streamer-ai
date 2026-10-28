# Frame generation, extra — summary

**2x frame generation without latency.** The host streams every second frame of the display
rate - 30 of 60 a second - and the client draws the one in between from the two it received
last, *before* the next one arrives. Received frames N−2 and N in, the unsent frame N+1 out.
Interpolation (`../frame_gen_intra/`) would wait for N+2 and add a received interval of
latency; extrapolation adds none. desktop-streamer's client already schedules it this way: an
extrapolated frame is shown half a frame interval after the frame it came from.

> **Status: four notebooks, and a graph that passes desktop-streamer's client tests in place
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

The code the notebooks share lives in `helpers/`. `sources.py`, `degrade.py` and `metrics.py`
are copies of upscale's, so a change made for one problem cannot quietly change the other's
data or units. `scenes.py` is the desktop motion. `runs.py` reads the run set back, picks the
received frames, and holds the one definition of a changed sample. `models.py` holds every
method with the client's two-input contract. `training.py` handles loading, grouped scoring
and upscale's training loop. `speed.py` is the timing. `webexport.py` is the two-input export
into `dist/frame_gen_extra/`, at fixed shapes or with `N`, `H` and `W` symbolic as the client
needs.

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
- **One motion per 328×188 tile.** The 80 px samples cannot show what it does to a tile that
  holds two motions - a scrolled page beside a still panel.
- **The scenes are drawn, not recorded.** Their speeds, pauses and event rates are guesses
  about how people use a desktop, and 60 fps is reached by reading a 30 fps motion between its
  frames. A few minutes of real 60 fps screen recording would check them, and there is no
  reader for one yet.
- **Video is thin**: 250 training and 10 validation sequences at 50 fps or more. More would
  need `datasets/download.py` to fetch them - a download only its user should decide on.

## Next

1. **Run it in a live session.** The graph is in desktop-streamer's client, uncommitted. Turn
   on extrapolate in a room, watch a scroll and a dragged window at 60 fps, and time it on a
   laptop iGPU - the 3090 numbers are a floor, and an Intel UHD has about a fortieth of the
   compute.
2. **Fix the two client cases the graph does not fit**: offer extrapolate only on WebGPU, and
   either drop the ⅔ placement when interpolate is also on or export a graph that carries a
   third of the motion for it.
3. **Score on whole frames**, cut into the client's tiles, to see what one motion per tile does
   to a frame that holds two motions.
4. **Skip still tiles in the client.** The graph returns them unchanged, but still runs the
   search on them; comparing the two received tiles first would save that work, which is most
   of a desktop's tiles most of the time.

Sibling problems: `../upscale/summary.md`, `../frame_gen_intra/summary.md`.
