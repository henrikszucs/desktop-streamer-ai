# Upscale — summary

Reconstructing a higher resolution frame from the one the stream delivered, so a client can
be sent fewer pixels than it displays.

> **The speed target is 60 fps on a GTX 1660 Ti with 20% to spare**
> (`reference_60fps.md`): 13.3 ms a frame in the client on that card, which leaves the graph
> 9.2 ms on the benchmark page at the stream's tile count. That is 0.58 ms a tile for a
> 1280×720 stream. The told model (notebook 23, float16) meets it only for a 640×360 stream.
> Notebook 24's told `q24` (`client/upscale_fast.onnx`) is built for the 1280×720 stream:
> **7.6 ms predicted** (7.8 conservative) at 16 tiles. That figure is predicted from the RTX 3090
> and has not been measured on the card. On notebook 0's set it keeps 60% of the told model's
> SSIM-Y gain (notebook 24), with bicubic's colour.

> **2026-11-22: every notebook has run again on notebook 0's set**, one recipe, scored on one
> validation split, and their graphs are back in `dist/upscale/` (every trained graph was
> deleted on 2026-11-20 to train them again). "Notebook 0's set, notebook by notebook" has
> every notebook (17 describes the stream and uses no set). The sections after it are on older
> sets: their numbers do not compare with this set's, and their browser times are of the
> deleted graphs, which had the same shapes.
>
> **Status: twenty-five notebooks. desktop-streamer runs notebook 23's told 24×2**, its copy of
> `dist/upscale/client/upscale.onnx` (and the float16 twin on a GPU with `shader-f16`), told
> each frame's slice QP, loop filter and frame type. On notebook 0's set it is still the
> model to ship.
>
> **The best models are the luma 24 + 2-block body told its stream**: +0.0394 to +0.0405 SSIM-Y
> over bicubic's 0.8521 on the 24,286 validation patches, at 13,932 multiply-adds an LR pixel,
> with bicubic's colour. The six best rows - notebook 20's two models switched at QP 30,
> notebook 27's mix 0.1 and notebook 22's model, two seeds each - lie within 0.0011, where one
> model's seeds land 0.0002 to 0.0004 apart. The switched pair is first at both seeds, +0.0008
> over notebook 22's model, but it is two graphs and its picture jumps where a stream's QP
> crosses 30. Telling the body its stream pays most where the stream is unusual: +0.005 to
> +0.006 over the same body blind at slice QP 0–18 and 46–52, +0.0015 at 30–34 (notebook 29).
>
> **On screens compute buys what bits cannot** (notebook 15). Raising H.264 from the host's
> default 0.46 to 3.3 bits per LR pixel adds +0.0136 SSIM-Y on screens; the told 24×2 adds
> +0.040 at the same bits, a quality no bitrate reaches on 7 of the 8 screen clips. On photos
> and video it adds the most of any client at every bitrate (+0.0163 and +0.0137 at the
> default) and stays above plain H.264, where most blind models fall below it on photos from
> 0.7 bits up.
>
> **The 60 fps tier is notebook 24's told q24** (the box above): 60% of the told model's gain
> at 1,824 multiply-adds an LR pixel. At the same bits it adds the most quality for its
> compute on photos and video, and half of the told model's gain on screens.
>
> **The SSIM loss stays.** Trained on the JND loss, both bodies lose on every judge, but for
> the q24 on the MOS predicted from the loss it trained on (notebook 26). Added to the SSIM
> loss at λ = 0.1 it gains a little luma (+0.13 to +0.17 dB) and nothing beyond seed luck on
> SSIM-Y or LPIPS; at 0.3 and 1 it costs SSIM-Y (notebook 27). The q24 on mix 1 that an older
> set favoured is not exported.
>
> **A look-up table is as good on screens and still does not run fast.** Notebook 10's
> two-stage `SDY x2` table matches the told 24×2 on screens at the same bits, falls below
> plain H.264 on photos from 0.7 bits up, and is read with `Gather`, `Floor` and integer
> operators outside the browser's fast set. Read at one exact address (notebook 11) it keeps
> three quarters of its gain.
>
> **Operations per pixel do not predict browser time**, which is why every graph from
> notebook 4 on was measured in the browser ("Results at the client's tile", on the deleted
> graphs): what a provider runs natively, how many operators a graph has, and whether any
> node falls back to the CPU decide it. The tile is 640×360 out from a 320×180 step and a
> 328×188 input (notebook 5), and everything about geometry is charged in nanoseconds per
> output pixel.
>
> **Not measured on this set**: the browser time of its graphs, the q24 on the GTX 1660 Ti,
> and anyone's eyes on the JND losses' pictures.

**One notebook is one model.** A notebook owns its architecture, its training, its numbers
and its export cell, and publishes straight into `dist/upscale/` at the repo root — so a
model is added or dropped by adding or deleting one file, and the page shows whatever is
published.

## Notebooks

Run in order, from this folder:

- `00_data_preprocess.ipynb` — **the one dataset every notebook trains on and is scored on**
  (rebuilt 2026-11-21 on notebook 28's recipe, which it replaces): 1,396 clean, detailed
  YouTube sequences (up to three a video), 1,200 drawn screens of two sets and 460 DIV2K and
  Flickr2K photos, each a clip through the host's two lines (x264 and NVENC) at six bitrates;
  one frame of each slice QP bin and one I frame a source and line, 4 patches of each (72 px
  LR with a halo of 4 over 128 px HR), labelled with the slice QP, loop filter, frame type,
  line and kind. Three splits by source: **train** 140,795 patches, **select** 6,982 (held-out
  training sources, to choose best epochs) and **val** 24,286, only ever scored. Bicubic reads
  0.8521 SSIM-Y on val. Writes `datasets/upscale/`.
- `01_upscale_dummy.ipynb` — **the static filters, scored as every model is**
  (`training.val_scores` on the val split, recorded for notebook 29's table): bicubic 0.8521,
  Lanczos-3 +0.0007, PIL's bicubic −0.0008, bilinear −0.0087, nearest −0.0159 SSIM-Y.
  Exports nearest, bilinear and bicubic at the client's 320×180 step and as a whole frame.
  See "Static methods".
- `02_upscale_nn.ipynb` — **a first learned upscaler**: `PlainCNN`, ESPCN-shaped with no
  bicubic shortcut, at two shapes (16-24-24-32, and 64-32). Exports both.
- `03_upscale_web.ipynb` — **the residual 32-16** (`models.PlainUpscaler`): bicubic as a frozen
  5×5 convolution plus a learned correction that starts at zero. Trained at three seeds, keeping
  the best on the select split (one seed can stall at bicubic). Exports the kept run in float32
  and float16, and scores the float16 graph on the val split.
- `04_upscale_strategies.ipynb` — **three strategies under one recipe**: the 32-16 control
  (notebook 3's three seeds, the ruler for luck; a run that stalled at bicubic is left out of
  it), the same body correcting luma only, and luma with two residual blocks. Exports
  `luma 32-16`.
- `05_upscale_geometry.ipynb` — **not a model: the shape one runs at.** Tile size, batch and
  scale factor, swept on the CPU and then measured in the browser on the card, charged in
  **nanoseconds per output pixel** because a run is a tile and a rate of runs cannot compare
  two tile sizes. It is what chose 640×360, and what `tiles.plan_tiling` implements. Times
  notebook 3's model.
- `06_upscale_reparam.ipynb` — **the reparameterized 32-16**: notebook 3's body trained as three
  branches a convolution and merged into one, at a learning rate of 1e-4. Scored beside the
  control's three seeds; exports the merged graph.
- `07_upscale_luma.ipynb` — **the luma model** (notebook 4's `luma 24 + 2 blocks`, loaded from
  its run): colour is bicubic, and the network adds one correction per output pixel to R, G
  and B alike. Checks that the trained model leaves colour untouched and publishes it as
  `upscale_luma`.
- `08_upscale_screens.ipynb` — **a model told the bitrate and the line** (the loop filter):
  what the client knows of its stream without reading a frame, as `models.ToldLuma`'s shifts,
  against the blind model at three seeds. A told model bakes into the blind graph for one stream.
- `09_upscale_dct.ipynb` — **a model that only guesses gains** on the DCT of 8×8 blocks every
  4 px, mirrored into a 16×16 spectrum and windowed back: the *fixed* gains, and gains
  *decided* per block through 8 (three seeds) and 32 channels on top of them.
- `10_upscale_lut.ipynb` — **look-up tables instead of arithmetic** (SR-LUT, MuLUT): networks
  that read four luma pixels (S at three seeds, SDY, SDY×2), filled into tables at 9, 17 and 33
  points a tap, then the tables fine-tuned as the parameters. Exports the best table and `S`.
- `11_upscale_fastlut.ipynb` — **look-up tables read at one exact address**: a binned table
  (the centre's level and the three other taps' differences, `S` at three seeds and `SDY x2`)
  and a 3×3 pattern filter. Float32 only.
- `12_upscale_dirnet.ipynb` — **a directional upscaler**: four learned 1D interpolators a
  sub-pixel; B blends them the same everywhere, C chooses per pixel, D adds a learned
  correction. Exports all three (C and D in float32 only).
- `15_upscale_tradeoff.ipynb` — **bits, compute and quality in one currency.** For every
  published client, how many bits plain H.264 and bicubic would need for the same SSIM-Y, over
  the bits the client's stream used: its worth in bandwidth, on whole streamed clips of the val
  sources. See "Bits against compute".
- `16_upscale_distill.ipynb` — **distillation, a leaky ReLU, and an edge-weighted loss**: the
  luma 32-16 plain, leaky and distilled from notebook 8's told model; the told model with an
  edge-weighted L1 and with a leaky ReLU. Exports the leaky 32-16 as the cheap tier.
- `17_upscale_qp.ipynb` — **QP statistics**: x264's QP of every frame of the host's stream by
  bandwidth, stream size and content. Describes the stream; trains nothing.
- `18_upscale_qp_models.ipynb` — **blind, told the slice QP, or a specialist**: notebook 7's
  body told each patch's slice QP (two seeds), and a blind model for QP 38 and up, against the
  blind model's three seeds, by slice QP bin.
- `19_upscale_qp_told.ipynb` — **how exact the QP has to be, and how it ships**: notebook 18's
  told model given a wrong QP, and the check that a told model's biases are a line in the QP.
  Trains nothing.
- `20_upscale_qp_split.ipynb` — **a low and a high QP model**, each told the QP, switched at
  QP 30, against notebook 18's one told model; and what the switch does to the picture.
- `22_upscale_qp_slice_models.ipynb` — **told the slice QP, the loop filter and I or P**: what
  the client reads off every frame, against told the QP alone and blind, by line, frame type
  and QP.
- `23_upscale_qp_export.ipynb` — **notebook 22's model as the browser runs it**: `input` and a
  three-number `condition`, the biases computed from it, exported as
  `dist/upscale/client/upscale.onnx` and its float16 twin, checked against the trained model.
- `24_upscale_fast.ipynb` — **a told model for 60 fps at a 1280×720 stream on a GTX 1660
  Ti**, as a drop-in for notebook 23's graph. It measures every layer of the 1660 Ti's cost
  table again on the RTX 3090 to predict that card from this one, trains bodies that run at a
  quarter of the pixels, chooses one on the select split, and exports `client/upscale_fast.onnx`
  and its float16 twin. See "60 fps on a GTX 1660 Ti".
- `26_upscale_jnd_loss.ipynb` — **the JND loss against the SSIM loss** on the two best bodies,
  notebook 22's told 24×2 and notebook 24's q24: the perceptual repo's differentiable JND metric
  as the loss (`helpers/jnd_settings.json`), judged by SSIM-Y, PSNR, JND, LPIPS, GMSD and
  pictures of patches and whole frames. Exports nothing.
- `27_upscale_mixed_loss.ipynb` — **the SSIM loss and the JND loss together**: the SSIM loss +
  λ × notebook 26's JND loss at λ = 0.1, 0.3 and 1, on the same two bodies and seeds, against
  both losses alone. Exports nothing.
- `29_upscale_scoreboard.ipynb` — **every model on one set**: every notebook's recorded
  validation scores in one table against notebook 1's bicubic, by kind, line and slice QP.

Notebooks 13, 14, 21 (notebook 17's pairs relabelled), 25 (the static filters on notebook 21's
set) and 28 (the pairs notebook 0 now builds) were deleted on 2026-11-21.

Every model notebook trains with the one recipe in `training.fit` (Adam at 1e-3, batches of
32, `metrics.ssim_l1_loss`, epochs of 34,368 patches, the best epoch by SSIM-Y on the select
split, halving at plateaus until it converges) and records its models' per-patch scores on the
val split (`training.record`), so notebook 29 can set every model beside every other.

## Notebook 0's set, notebook by notebook

*Filled in as the notebooks run on notebook 0's set, from 2026-11-21 to 2026-11-22: every notebook (17 describes the stream and uses no set).
Every number in this section compares with every other; the sections after it are on older
sets.*

Each model on the 24,286 validation patches: SSIM-Y and its gain over `models.bicubic` with the
95% range over sources, luma PSNR, the mean gain in chroma PSNR (U and V) over bicubic, the
SSIM-Y gain on screens, and multiply-adds an LR pixel. A PSNR stops at 60 dB, libvmaf's 8-bit
ceiling (`metrics.PSNR_CEILING`).

| nb | model | SSIM-Y | over bicubic | 95% range | luma dB | chroma dB | screens | MACs / LR px |
|---|---|---|---|---|---|---|---|---|
| 20 | **two told models, switched at slice QP 30** | **0.8926** | **+0.0405** | +0.0381 to +0.0429 | **30.31** | +0.05 | **+0.0595** | 13,932 |
| 20 | two told models at QP 30, seed 1 | 0.8924 | +0.0403 | +0.0380 to +0.0428 | 30.30 | +0.05 | +0.0594 | 13,932 |
| 27 | told 24×2, the SSIM loss + 0.1 × the JND loss ("mix 0.1"), seed 1 | 0.8922 | +0.0401 | +0.0378 to +0.0424 | 30.34 | +0.04 | +0.0582 | 13,932 |
| 22 | told the slice QP, the loop filter and I or P | 0.8919 | +0.0398 | +0.0375 to +0.0422 | 30.20 | +0.05 | +0.0582 | 13,932 |
| 27 | told 24×2, mix 0.1 | 0.8918 | +0.0397 | +0.0375 to +0.0421 | 30.35 | +0.05 | +0.0578 | 13,932 |
| 22 | told the QP, the loop filter and I or P, seed 1 | 0.8915 | +0.0394 | +0.0371 to +0.0418 | 30.14 | +0.04 | +0.0574 | 13,932 |
| 18 | luma 24 + 2 blocks, told the slice QP | 0.8914 | +0.0393 | +0.0371 to +0.0417 | 30.21 | +0.05 | +0.0578 | 13,932 |
| 18 | told the slice QP, seed 1 | 0.8909 | +0.0388 | +0.0365 to +0.0412 | 30.17 | +0.04 | +0.0573 | 13,932 |
| 8 | luma 24 + 2 blocks, told the bitrate and line | 0.8905 | +0.0384 | +0.0361 to +0.0408 | 30.09 | +0.04 | +0.0570 | 13,932 |
| 27 | told 24×2, mix 0.3 (2 seeds) | 0.8896–0.8903 | +0.0375 to +0.0382 | | 30.28–30.34 | +0.04 | +0.0545 to +0.0556 | 13,932 |
| 16 | told the bitrate and line, leaky ReLU | 0.8900 | +0.0379 | +0.0357 to +0.0402 | 30.09 | +0.04 | +0.0559 | 13,932 |
| 16 | told the bitrate and line, edge-weighted loss | 0.8899 | +0.0378 | +0.0355 to +0.0402 | 30.07 | +0.04 | +0.0562 | 13,932 |
| 4, 7 | luma 24 + 2 blocks (`upscale_luma`) | 0.8886 | +0.0365 | +0.0342 to +0.0389 | 30.00 | +0.04 | +0.0555 | 13,932 |
| 27 | told 24×2, mix 1 (2 seeds) | 0.8869–0.8876 | +0.0348 to +0.0355 | | 30.22–30.28 | +0.03 | +0.0502 to +0.0515 | 13,932 |
| 2 | plain CNN 64-32 | 0.8872 | +0.0351 | +0.0328 to +0.0375 | 29.81 | −3.05 | +0.0540 | 26,688 |
| 2 | plain CNN 16-24-24-32 | 0.8867 | +0.0346 | +0.0323 to +0.0369 | 29.72 | −3.66 | +0.0535 | 13,296 |
| 10 | LUT SDY x2: two stages of tables, fine-tuned | 0.8858 | +0.0337 | +0.0313 to +0.0364 | 29.97 | +0.03 | +0.0554 | look-up |
| 10 | LUT SDY x2 network (what the tables are filled from) | 0.8858 | +0.0337 | +0.0312 to +0.0363 | 29.91 | +0.03 | +0.0548 | |
| 16 | luma 32-16, distilled from the told model | 0.8836 | +0.0315 | +0.0294 to +0.0336 | 29.79 | +0.03 | +0.0489 | 8,484 |
| 16 | luma 32-16, seed 1 | 0.8832 | +0.0311 | +0.0289 to +0.0333 | 29.68 | +0.03 | +0.0489 | 8,484 |
| 16 | luma 32-16 leaky, seed 1 (published, the cheap tier) | 0.8828 | +0.0307 | +0.0286 to +0.0328 | 29.71 | +0.03 | +0.0480 | 8,484 |
| 16 | luma 32-16 leaky, distilled | 0.8827 | +0.0306 | +0.0285 to +0.0326 | 29.73 | +0.03 | +0.0472 | 8,484 |
| 10 | LUT SDY network | 0.8827 | +0.0306 | +0.0284 to +0.0329 | 29.72 | +0.03 | +0.0497 | |
| 6 | reparam 32-16 | 0.8827 | +0.0307 | +0.0286 to +0.0328 | 29.64 | −2.35 | +0.0480 | 9,636 |
| 3 | residual 32-16, seed 2 (kept, `upscale_web`) | 0.8826 | +0.0306 | +0.0284 to +0.0328 | 29.54 | −2.70 | +0.0484 | 9,636 |
| 16 | luma 32-16 leaky | 0.8826 | +0.0305 | +0.0284 to +0.0326 | 29.71 | +0.03 | +0.0477 | 8,484 |
| 4 | luma 32-16 | 0.8822 | +0.0301 | +0.0280 to +0.0322 | 29.61 | +0.03 | +0.0476 | 8,484 |
| 12 | DirNet full: a blend chosen per pixel and a learned correction | 0.8811 | +0.0290 | +0.0269 to +0.0312 | 29.51 | +0.03 | +0.0470 | 5,687 |
| 3 | residual 32-16, seed 1 | 0.8804 | +0.0283 | +0.0262 to +0.0303 | 29.57 | −2.55 | +0.0454 | 9,636 |
| 10 | LUT SDY: three tables, fine-tuned | 0.8802 | +0.0281 | +0.0259 to +0.0304 | 29.70 | +0.03 | +0.0469 | look-up |
| 24 | q32: a quarter-pixel body 32 wide, its skip trained | 0.8785 | +0.0264 | +0.0250 to +0.0279 | 29.36 | −0.91 | +0.0367 | 3,072 |
| 11 | binned LUT SDY x2: one read, no interpolation | 0.8772 | +0.0251 | +0.0232 to +0.0270 | 29.24 | +0.02 | +0.0411 | look-up |
| 10 | LUT S network (3 seeds) | 0.8769–0.8771 | +0.0248 to +0.0250 | | 29.30–29.31 | +0.02 | +0.0425 to +0.0429 | |
| 18 | specialist: blind, trained on slice QP 38 and up only | 0.8765 | +0.0244 | +0.0220 to +0.0267 | 28.64 | +0.03 | +0.0410 | 13,932 |
| 24 | q24, the exact 6×6 skip | 0.8762 | +0.0241 | +0.0228 to +0.0254 | 29.32 | +0.02 | +0.0328 | 1,824 |
| 12 | DirNet adaptive: a blend chosen per pixel | 0.8758 | +0.0237 | +0.0221 to +0.0255 | 29.21 | +0.01 | +0.0382 | 4,343 |
| 24 | **q24, the 4×4 skip frozen (published, `upscale_fast`; 2 seeds' mean)** | 0.8758 | +0.0237 | +0.0225 to +0.0249 | 29.29 | +0.02 | +0.0322 | 1,824 |
| 27 | fast q24, mix 0.1 (2 seeds) | 0.8749–0.8758 | +0.0228 to +0.0237 | | 29.37–29.47 | +0.02 | +0.0313 to +0.0324 | 1,824 |
| 10 | LUT S: one table, fine-tuned | 0.8754 | +0.0233 | +0.0213 to +0.0254 | 29.31 | +0.02 | +0.0406 | look-up |
| 24 | q24, its skip trained (2 seeds' mean) | 0.8751 | +0.0230 | +0.0219 to +0.0242 | 29.26 | −1.00 | +0.0306 | 1,824 |
| 26 | told 24×2, trained on the JND loss (2 seeds) | 0.8749–0.8750 | +0.0228 to +0.0229 | | 29.75–29.77 | +0.02 | +0.0322 to +0.0327 | 13,932 |
| 27 | fast q24, mix 0.3 (2 seeds) | 0.8738–0.8746 | +0.0217 to +0.0225 | | 29.41–29.47 | +0.02 | +0.0294 to +0.0304 | 1,824 |
| 24 | q20, its skip trained (2 seeds' mean) | 0.8736 | +0.0215 | +0.0204 to +0.0226 | 29.21 | −0.98 | +0.0284 | 1,320 |
| 11 | binned LUT S (3 seeds) | 0.8724–0.8725 | +0.0203 to +0.0204 | | 29.01–29.02 | +0.02 | +0.0346 to +0.0347 | look-up |
| 27 | fast q24, mix 1 (2 seeds) | 0.8702–0.8722 | +0.0181 to +0.0201 | | 29.35–29.46 | +0.02 | +0.0236 to +0.0270 | 1,824 |
| 9 | DCT decide 32 | 0.8677 | +0.0156 | +0.0146 to +0.0165 | 28.95 | +0.01 | +0.0219 | 2,827 |
| 11 | 3×3 pattern filter | 0.8663 | +0.0142 | +0.0133 to +0.0151 | 29.00 | +0.01 | +0.0211 | look-up |
| 9 | DCT decide 8 (3 seeds) | 0.8630–0.8635 | +0.0109 to +0.0114 | | 28.82–28.86 | +0.01 | +0.0167 to +0.0176 | 2,386 |
| 26 | fast q24, trained on the JND loss (2 seeds) | 0.8626–0.8630 | +0.0105 to +0.0109 | | 29.12 | +0.01 | +0.0144 to +0.0152 | 1,824 |
| 9 | DCT fixed | 0.8540 | +0.0019 | +0.0017 to +0.0022 | 28.74 | +0.01 | +0.0035 | 2,239 |
| 12 | DirNet fixed: one learned blend, one 5×5 filter | 0.8530 | +0.0009 | +0.0007 to +0.0010 | 28.65 | +0.00 | +0.0017 | 900 |
| 1 | Lanczos-3, PIL | 0.8528 | +0.0007 | +0.0006 to +0.0009 | 28.70 | +0.00 | +0.0013 | |
| 3 | residual 32-16, seed 0 (stalled) | 0.8523 | +0.0002 | +0.0001 to +0.0003 | 28.58 | −0.05 | +0.0001 | 9,636 |
| 1 | **bicubic** | 0.8521 | | | 28.58 | | | |
| 1 | bicubic, PIL (a = −0.5) | 0.8513 | −0.0008 | −0.0010 to −0.0006 | 28.50 | +0.01 | −0.0017 | |
| 1 | bilinear | 0.8434 | −0.0087 | −0.0096 to −0.0078 | 27.97 | +0.01 | −0.0140 | |
| 1 | nearest | 0.8362 | −0.0159 | −0.0166 to −0.0151 | 27.47 | −0.13 | −0.0126 | |

- **Every CNN that learned gains +0.028 to +0.041 SSIM-Y over bicubic**, most on screens
  (+0.045 to +0.060) and least on video and photos (+0.013 to +0.026). No static filter is
  more than 0.001 above bicubic.
- **Residual blocks on luma lead**: luma 24 + 2 blocks is +0.0070 over the 32-16 control (the
  mean of its two runs that learned, 0.0023 apart), above notebook 2's 64-32 at half its
  multiply-adds.
- **Luma only costs nothing**: luma 32-16 lands within luck of the RGB 32-16, with 12% fewer
  multiply-adds and bicubic's colour. The RGB models lose 2.4–3.7 dB of chroma PSNR against
  bicubic; over half of that is on the 2,816 gray patches, where bicubic's colour is exact
  and the model adds a faint tint, and pooled over the split it is 0.15–0.5 dB.
- **One seed in three of the residual 32-16 stalls at bicubic** at the shared 1e-3: 11 of its
  narrow layer's 16 ReLUs never fire. Notebook 3 trains three seeds and keeps the best on the
  select split; notebook 4 leaves a stalled run out of its ruler for luck.
- **A leaky ReLU is insurance, not quality** (notebook 16). The luma 32-16 loses 5 and 1 of
  its 16 narrow channels at two seeds, which land 0.0010 apart; leaky, it loses none and its
  seeds land 0.0002 apart, at the same mean (0.8827) and the same cost. It is the published
  cheap tier. Distilling it from the told model adds nothing measurable, and the edge-weighted
  loss and a leaky ReLU each cost the told model 0.0005–0.0006.
- **Telling the model its stream pays**: told the bitrate and the encoder line, luma 24 + 2
  blocks gains +0.0020 SSIM-Y over its blind runs (three seeds, 0.0002 apart), most where a
  stream is unusual (+0.004 below slice QP 18 and above 42, +0.0005 from 26 to 34). For one
  stream the condition bakes into the biases: the same graph, the same cost.
- **The slice QP is the condition to tell** (notebook 18). Told it, the same body gains
  +0.0027 over its blind seeds (two seeds, 0.0005 apart), +0.0010 above the bitrate and line at
  the same seed: the best model so far. It pays most at both ends of the QP range (+0.005 below
  QP 18 and from 42 up, +0.0007 from 26 to 34). A blind specialist trained on QP 38 and up
  beats it there by +0.002 to +0.003 and loses everywhere else, below bicubic at QP 0–18;
  switched in at QP 38 it would add +0.0005 overall, for a second graph. The told model ships at
  the blind graph's cost, its biases a line in the QP that the client computes each frame
  (notebook 19); the QP has to be exact to a few steps: 3 off costs 0.0005, 6 off most of the
  gain, and 12 off lands below the blind model.
- **Two told models, switched at QP 30, beat one** (notebook 20): +0.0013 over the one told
  model (whose seeds land 0.0005 apart), in every QP bin but the one just past the switch, and
  +0.0405 over bicubic, the best so far. The switch is a jump, though: where a stream's QP
  crosses 30 the two pictures differ by 1.6 levels of luma on average and 21 at the 99th
  percentile, against 0.15 and 1.3 for one QP step of the one model: a flicker the scores do
  not see.
- **The loop filter and the frame type add little to the QP** (notebook 22): told all three,
  the body gains +0.0005 over the QP alone, no wider than either model's seeds land apart.
  What they add is on NVENC P frames at QP 26–38 (+0.001), where the QP alone gains nothing
  over the blind model. Notebook 23 exports its seed 0 as the client's graph
  (`dist/upscale/client/upscale.onnx`, and a float16 copy within 0.36 of an 8-bit step), which
  scores what the trained model does on every validation patch. On onnxruntime's CPU at its
  full optimization level the told graph runs half again slower than the blind one (144 against
  95 ms for a 960×540 frame): the three convolutions whose bias is an input stay out of the
  blocked layout. The browser does not run that pass.
- **A model for 60 fps on the GTX 1660 Ti keeps three fifths of the told model's gain** (notebook
  24). q24 works at a quarter of the pixels with a frozen 4×4 bicubic skip, is predicted at 7.6 ms
  (7.8 conservative) at 16 tiles against the 8.3 ms bar, and gains +0.0237 over bicubic against
  notebook 22's +0.0396: 71% of it on video, 56% on screens. Its skip has to stay frozen: trained,
  it gives up 1 dB of U and V for no SSIM-Y. It is a drop-in for the client's graph
  (`dist/upscale/client/upscale_fast.onnx`), not yet timed on the card itself.
- **On screens compute buys what bits cannot; on photos a blind model loses to the bits** (notebook
  15, 8 clips of each kind streamed at 0.1 to 3.3 bits per LR pixel). On screens, x264 at 3.3 bits
  adds only +0.0136 SSIM-Y over the host's default (0.46), while the told 24x2, the luma 24x2 and
  the `SDY x2` table each add +0.040 to +0.041 at the same bits, a quality no bitrate reaches on 7
  of the 8 clips. The told 24x2 adds the most on photos and video at every bitrate (+0.0163 and
  +0.0137 at the default, worth 1.29 and 1.56 times the bits). On photos at a target of 1.0 the
  blind DirNet C and D, the leaky 32-16 and the table score below the traditional stream with the
  same bits; both told models stay above it. The told fast q24 adds the most for its 5,244
  operations off screens (level with the luma 24x2 on photos at a fifth of the work), and half of
  the told 24x2's gain on screens.
- **Both bodies stay on the SSIM loss** (notebook 26, the JND loss at the 2026-11-19 settings).
  Trained on the JND loss, the told 24×2 loses on every judge, its own JND included (LPIPS 0.1599
  against 0.1515; closer on 30% of the patches); the fast q24 on every judge but the predicted MOS
  (LPIPS 0.1691 against 0.1663, closer on 43%). Both lose most on screens and are level only on
  photos. The JND loss brightens (+0.7 to +0.8 levels of luma), keeps less fine detail, and takes
  part of the fast q24's grid away (the step across its 2×2 cells 1.20 times the step elsewhere,
  against 1.28 under the SSIM loss and 1.13 in the truth). On notebook 21's video-only set the
  q24 had looked better on the JND loss; on this set, with screens, it does not. A rare batch
  makes the JND loss or its gradient not finite (20 to 32 batches a run, the same batches for both
  bodies); `training.fit` now skips such a batch.
- **Adding the JND loss to the SSIM loss is no clear win either** (notebook 27, the SSIM loss + λ
  × the JND loss at λ = 0.1, 0.3 and 1). Only mix 0.1 costs no SSIM-Y: for the told 24×2 it is
  level on SSIM-Y, LPIPS and JND and above on luma (+0.17 dB) and GMSD (0.0753 against 0.0759),
  and it centres the brightness; for the fast q24 it is +0.13 dB and 0.0007 better on LPIPS,
  inside its seeds, and takes part of the grid away (1.23 against 1.29; 1.13 in the truth).
  λ = 0.3 and 1 cost 0.0016 to 0.0046 SSIM-Y, and λ = 1 LPIPS too. No mix beats the SSIM loss by
  more than its seeds land apart on SSIM-Y or LPIPS; nothing is exported. On notebook 21's set
  mix 1 had been the q24's best; at these JND settings, on this set, it is its worst mix.
- **The scoreboard** (notebook 29, 78 models from 17 notebooks): every run of the 24 + 2-block
  body told its stream is above every blind model (+0.0375 to +0.0405 against +0.0362 to
  +0.0365), except mix 1 and the JND-loss runs. The six best rows (notebooks 20, 27 and 22, two
  seeds each) span 0.0011, with one model's seeds 0.0002 to 0.0004 apart; the pair switched at
  QP 30 is first at both seeds, +0.0008 over notebook 22's model. The told models gain the most
  at QP 38–42 and lead the blind body most at the ends of the QP range (+0.005 to +0.006, against
  +0.0015 at QP 30–34).
- **A DCT model that only decides gains** (notebook 9) is cheap and weak: the best fixed gains
  are +0.0019 over bicubic, gains decided block by block +0.011 (3,080 weights) to +0.0156
  (12,320), a fifth of notebook 7's multiply-adds for 0.021 less SSIM-Y. A CNN buys more for
  each multiply-add it adds.
- **A look-up table matches a CNN on screens only** (notebook 10). Filled at 17 points a tap, a table loses
  0.004 to 0.006 SSIM-Y against its network, and fine-tuning wins back +0.003 to +0.004: the two-stage `SDY x2`
  table scores what its network does, +0.0337 over bicubic for 2.5 MB in float16. On screens (+0.0554) it is
  inside notebook 7's three seeds; on video and photos it is 0.004 to 0.007 below, 0.0026 overall. Each step up the
  ladder pays (`SDY` +0.0048 over `S`, `SDY x2` +0.0056 over `SDY`). Reading a table needs `Gather`, `Floor` and
  integer arithmetic, outside the browser's fast set, and most of its memory holds neighbourhoods pictures do
  not have: in the `S` table, 89% of the reads land in 1.4% of the cells.
- **Read at one exact address, a table keeps three quarters of its gain** (notebook 11). The
  binned `SDY x2` gains +0.0251, 74% of notebook 10's interpolated one, on screens as everywhere; the
  binned `S` +0.0204 (three seeds, 0.0001 apart). A 3×3 pattern filter is level with `S` on video and
  photos and loses only on screens. Their graphs are a third to a quarter of notebook 10's CPU time,
  float32 only, and still read with `Gather`.
- **A directional upscaler gains by choosing per pixel** (notebook 12). One learned blend of
  its four 1D candidates is +0.0009 over bicubic, a blend chosen per pixel +0.0237, and with a
  learned correction +0.0290 at 5,687 multiply-adds: more for each one than the DCT, and 0.0011
  below notebook 4's luma 32-16 at two thirds of its arithmetic. The choice needs `Softmax`,
  `Mul`, `ReduceSum` and `Sub`, outside the browser's fast set, and flips at some edges in
  float16, so those two graphs are float32 only.
- **Reparameterization is luck** (0.8827 against the control's 0.8804–0.8826), and free at run
  time: it merges into notebook 3's graph.
- **The stream's QP** (notebook 17, x264 at x2 on 1,297 clips, 0.5 to 32 Mbit/s): each doubling
  of bandwidth takes about 6 off the median QP of settled P frames; screens reach QP 0 from
  16 Mbit/s, video and photos do not. What moves sets the QP more than what is shown: a still
  page is at 0 from 4 Mbit/s, a zooming screen at 14 at 16 Mbit/s. At 0.5 Mbit/s every group is
  at QP 32 to 45, and sending x2 instead of full size lowers the QP by 6 to 14 at 2 to 8 Mbit/s.
- **Speed is not measured on this set's graphs yet.** The notebooks' CPU timings were taken while
  trainings ran. Notebook 5 ran last on a quiet machine (2026-11-22); it times notebook 3's
  32-16 at every geometry and still answers 640×360 tiles, from its earlier browser timings.
  The browser benchmark of the new graphs is still to run.

## Every published model on notebook 28's set (the first `29_upscale_scoreboard.ipynb`)

*Measured before every trained graph was deleted from `dist/upscale/` on 2026-11-20: the table is
the record of those graphs. Notebook 29 now sets every model of notebook 0's set side by side, from the scores each
notebook records (the section above).*

The sections further down measured each model on the set of its day — notebook 0's patches
(notebooks 1–16, no longer on disk), notebook 17's (18–20) or notebook 21's (22–27) — so their
numbers do not compare across sections. Notebook 29 scores every model that was published, as
its graph, on notebook 28's validation split: video, drawn screens and photos, x264 and NVENC,
slice QP 0–51. The same code reproduces the numbers recorded on notebook 21's set (bicubic
0.9045, the told 24×2 0.9289, q24 0.9213 against 0.9212), and packing patches into a tile
changes no core of any graph. **Compare models by this table**; a section's own numbers say
what that notebook found on its set.

SSIM-Y gain over `models.bicubic` (0.8524 on this set), with its 95% range over sources, and
by kind:

| nb | model | SSIM-Y | over bicubic | 95% range | video | screens | photos |
|---|---|---|---|---|---|---|---|
| 23 | **told 24×2 (the client's)** | **0.8878** | **+0.0354** | +0.0337 to +0.0371 | **+0.0249** | +0.0474 | **+0.0257** |
| 8 | screens 24×2, told 0.2 bpp | 0.8826 | +0.0301 | +0.0280 to +0.0324 | +0.0146 | +0.0483 | +0.0149 |
| 8 | screens 24×2, told 0.46 bpp | 0.8820 | +0.0296 | +0.0274 to +0.0319 | +0.0132 | +0.0481 | +0.0153 |
| 10 | LUT SDY x2 17 | 0.8797 | +0.0273 | +0.0248 to +0.0299 | +0.0078 | **+0.0493** | +0.0105 |
| 8 | screens 24×2, told 1 bpp | 0.8789 | +0.0265 | +0.0242 to +0.0288 | +0.0095 | +0.0456 | +0.0121 |
| 16 | luma 32-16 leaky, distilled | 0.8770 | +0.0246 | +0.0224 to +0.0269 | +0.0077 | +0.0433 | +0.0112 |
| 4, 7 | luma 24×2 (notebook 4's run) | 0.8763 | +0.0239 | +0.0224 to +0.0254 | +0.0132 | +0.0350 | +0.0176 |
| 12 | DirNet full | 0.8758 | +0.0234 | +0.0213 to +0.0256 | +0.0075 | +0.0411 | +0.0103 |
| 11 | fast LUT SDY x2 | 0.8724 | +0.0200 | +0.0180 to +0.0220 | +0.0049 | +0.0369 | +0.0071 |
| 12 | DirNet adaptive | 0.8720 | +0.0196 | +0.0176 to +0.0216 | +0.0043 | +0.0365 | +0.0075 |
| 24 | told q24 fast (the client's fallback) | 0.8711 | +0.0186 | +0.0178 to +0.0195 | +0.0176 | +0.0212 | +0.0134 |
| 3 | web 32-16 | 0.8710 | +0.0185 | +0.0172 to +0.0198 | +0.0092 | +0.0282 | +0.0132 |
| 10 | LUT S 17 | 0.8709 | +0.0185 | +0.0165 to +0.0206 | +0.0026 | +0.0364 | +0.0050 |
| 4 | strategy luma 32-16 | 0.8706 | +0.0182 | +0.0169 to +0.0195 | +0.0087 | +0.0279 | +0.0129 |
| 3 | web 32-16, int8 | 0.8696 | +0.0172 | +0.0159 to +0.0185 | +0.0076 | +0.0269 | +0.0122 |
| 11 | fast LUT S | 0.8688 | +0.0164 | +0.0146 to +0.0182 | +0.0027 | +0.0318 | +0.0047 |
| 2 | plain CNN 16-24-24-32 | 0.8650 | +0.0125 | +0.0116 to +0.0134 | +0.0062 | +0.0194 | +0.0080 |
| 11 | fast LUT 3×3 pattern | 0.8634 | +0.0110 | +0.0100 to +0.0119 | +0.0039 | +0.0187 | +0.0055 |
| 9 | DCT h32 | 0.8620 | +0.0096 | +0.0085 to +0.0107 | +0.0009 | +0.0185 | +0.0047 |
| 9 | DCT fixed | 0.8541 | +0.0016 | +0.0009 to +0.0024 | −0.0034 | +0.0065 | −0.0000 |
| 12 | DirNet fixed | 0.8534 | +0.0009 | +0.0003 to +0.0015 | −0.0031 | +0.0048 | −0.0004 |
| 1 | bicubic (`Resize` graph) | 0.8524 | +0.0000 | | | | |
| 6 | reparam 32-16 | 0.8524 | −0.0000 | | | | |
| 1 | bilinear | 0.8437 | −0.0087 | −0.0096 to −0.0078 | −0.0019 | −0.0138 | −0.0106 |
| 1 | nearest | 0.8365 | −0.0159 | −0.0167 to −0.0151 | −0.0163 | −0.0126 | −0.0252 |

- **The client's told 24×2 is the best model**, clear of every other range, and the only one
  besides q24 whose gain holds as the slice QP rises (+0.031 to +0.040 in every bin). A model
  not told the QP keeps at QP 46–52 about half its gain at 0–18, or less: notebooks 2–16
  trained on x264 at 0.2–1 bpp, which seldom reaches those QPs.
- **On screens the leaders are within 0.004 of each other**: notebook 10's LUT SDY x2
  (+0.0493), notebook 8's three graphs (+0.0456 to +0.0483) and the told 24×2 (+0.0474); no
  range by kind was computed to separate them. On video and photos the told 24×2 leads the
  next model by +0.007 to +0.008.
- **q24 is mid-table** (+0.0186), level with notebook 3's web 32-16, and weakest on clean
  pictures: +0.0139 at QP 0–18, +0.0212 on screens.
- **Four graphs do little or nothing**: notebook 6's reparam graph is bicubic to four places in
  every bin; notebook 9's DCT fixed and notebook 12's DirNet fixed gain under +0.002 and fall
  below bicubic from QP 30–34 up. Notebook 1's bicubic `Resize` graph scores what
  `models.bicubic` does.
- A new model is added to this table by adding its graph to the notebook's `GRAPHS`; graphs
  already scored are read back from `scoreboard/`.

The notebooks share their code through `helpers/`, so each one reads as a short list of
steps: `models.py` holds the bicubic skip and the models built on it (`PlainUpscaler`,
`ReparamUpscaler`, `LumaUpscaler`, `ToldLuma`, `FastLuma`; `slope` makes any of the last
three leaky),
`training.py` the loaders, the training loop (`teacher=` to distil), scoring and checkpoints,
and `speed.py` the timing.

`helpers/metrics.py` is **how a frame is scored** — Y'CbCr, luma weighted 6:1:1, the
channels reported apart, SSIM on luma, and the same weighting available as a loss. Every
model notebook scores through it (`training.evaluate`). Notebooks 2 and 3 still keep their
best epoch by RGB PSNR, and their numbers recorded below are RGB PSNR, so compare those
with the `psnr-rgb` column, not with `wpsnr`.

Each model notebook owns its own export cell and writes its `.onnx` into `dist/upscale/`
at the repo root through `helpers/webexport.py`. That is all publishing is: the benchmark server
lists whatever graphs are in that folder, per request, so there is no manifest for a
re-export to leave stale. 7 graphs are on the page today, notebook 1's three filters at the
128 and 320×180 steps: every trained graph was deleted on 2026-11-20 (64 were on the page
before, notebooks 1–3's eight and every model from notebook 4 on at the 320×180 step, most at
float32, float16 and float16 as a whole 1080p frame).

`data/` and `checkpoints/` are gitignored: they are what training leaves behind. `dist/` is
committed, because it holds only what is finished and published, and those models took
long to train. The export cells need
`onnx` and `onnxruntime`, which are project dependencies.

Notebooks 1–3 ran on Windows, where torch from PyPI is a CPU build (`2.13.0+cpu`), so their
runtimes are CPU, 12 threads. Every notebook from 4 on trained in WSL, where the same
requirement installs a CUDA build (`2.13.0+cu130`), on an RTX 3090. `onnxruntime` is the CPU
package everywhere, so every `cpu_ms` written into a graph is a CPU number; GPU numbers for a
graph are the browser's ("The browser benchmark"). Runtimes are forward pass only — no
decode, no copy, no PNG encoding.

## How much the stream costs the LR (the first `00_data_preprocess.ipynb`)

*Measured on notebook 0's first set, before it was rebuilt on 2026-11-21; the notebook no longer
makes this measurement.*

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

## Static methods (`01_upscale_dummy.ipynb`, `25_upscale_static.ipynb`)

### Quality on the one set — notebook 1

On notebook 0's 24,286 validation patches through `training.val_scores`, as every model is
scored; the gain's range is a 95% bootstrap over the sources:

| Filter | SSIM-Y | over bicubic | luma dB | video | screens | photos |
| --- | --- | --- | --- | --- | --- | --- |
| Lanczos-3 (PIL, float) | 0.8528 | +0.0007 (+0.0006 to +0.0009) | 28.70 | −0.0000 | +0.0013 | +0.0007 |
| **bicubic** (`models.bicubic`, a = −0.75) | **0.8521** | | **28.58** | | | |
| bicubic, PIL (a = −0.5) | 0.8513 | −0.0008 | 28.50 | +0.0005 | −0.0017 | −0.0013 |
| bilinear | 0.8434 | −0.0087 | 27.97 | −0.0018 | −0.0140 | −0.0102 |
| nearest | 0.8362 | −0.0159 | 27.47 | −0.0162 | −0.0126 | −0.0248 |

- **Lanczos-3 is the best filter by +0.0007**, on clean frames (+0.0023 at QP 0–18), and below
  bicubic from QP 38 up, where its sharper kernel brings out the codec's blocks. ONNX cannot
  express it; bicubic stays the bar.
- **Softer filters gain where the codec damaged most**: bilinear is +0.0017 at QP 46–52 and
  −0.0175 at 0–18. A blur hides blocks.

### Quality, scored as the models are — notebook 25 (notebook 21's set)

Notebook 25 scores the filters on notebook 21's 9,628 validation patches through
`training.evaluate`, the patches and code of notebooks 22–24. Bicubic is `models.bicubic`
(a = −0.75), the skip every model is built on:

| Filter | SSIM-Y | luma dB | over bicubic |
| --- | --- | --- | --- |
| Lanczos-3 (PIL, float) | 0.9047 | 33.69 | +0.0002 / +0.14 |
| notebook 1's pipeline: PIL bicubic + unsharp 1 px, 80% | 0.9044 | 33.66 | −0.0002 / +0.10 |
| bicubic + unsharp 1.5 px, 10% (tuned on the training split) | 0.9042 | 33.60 | −0.0004 / +0.05 |
| **bicubic** | **0.9045** | **33.55** | |
| bicubic, PIL (a = −0.5) | 0.9048 | 33.50 | +0.0003 / −0.05 |
| bicubic + unsharp 1 px, 80% | 0.8991 | 33.36 | −0.0055 / −0.19 |
| bilinear | 0.9016 | 32.94 | −0.0030 / −0.61 |
| Hamming (PIL, float) | 0.8963 | 32.58 | −0.0082 / −0.97 |
| nearest | 0.8883 | 32.01 | −0.0162 / −1.54 |

- **No unsharp mask beats bicubic on SSIM-Y.** Every radius and amount tried on the
  training split scores under plain bicubic. The best by luma PSNR gains +0.05 dB.
- **The mask costs most where the stream has no loop filter**: −0.0104 SSIM-Y on x264's P
  frames at 80%, against −0.0005 on NVENC's deblocked ones.
- **Lanczos-3 is the best static filter**, at +0.14 dB, and neither ONNX `Resize` nor a GPU
  sampler has it. Bicubic stays the bar.

Notebook 21's targets are video only; screens and photos are not in it.

### Quality — notebook 1's first run, 76 validation patches, 64px → 128px

The table below is notebook 1's first run, in RGB PSNR, on the earlier clean bicubic pairs;
the runtimes after it are from that run too. The notebook no longer measures either.

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

## Learned model (`02_upscale_nn.ipynb`)

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

## The browser-shaped model (`03_upscale_web.ipynb`)

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
of runs; three runs of one strategy in `04_upscale_strategies.ipynb` land 0.61 dB apart, which
covers several of these rungs. (That was the synthetic set; on the stream-degraded one, three
seeds of the 32-16 land 0.13 dB apart.) The choice of shape survives it — a third of the MACs for
1.2 dB is a gap four times the noise — but the rows nearest each other were never
distinguishable.

A third of the MACs of the wide body for 1.2 dB, and it is the point where a 1080p frame
fits the 60 fps budget with the decode still to pay for. Two shapes that looked promising
and were not: a depthwise-separable body loses 0.5 dB at the same cost (the pointwise
mixing it saves is the part that was working), and a strided stem running the body at half
resolution loses 1 dB for a saving a narrower body buys more cheaply.

## The luma residual model (`07_upscale_luma.ipynb`)

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

The notebook now loads notebook 4's run of the same model instead of training its own, so
`upscale_luma_24x2` carries that run's weights (116 epochs, best at 98): on the current
validation set (3,523 patches) **0.9292 SSIM-Y, +0.0243 over bicubic**, and 27.68 dB wpsnr
and 0.9656 SSIM on the drawn desktop frame.

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
  not only pixel values. `02_upscale_nn.ipynb` trains with it (see "Learned model").
- `psnr` — plain RGB, kept so the older tables stay readable.

**The two columns are different measurements and must not be subtracted.** Y carries most
of the error, so a weighted score sits near the luma score and away from the RGB one: the
32-16 model reads 31.81 dB weighted and 30.15 dB RGB on the same patches, in the same run.

## Strategies (`04_upscale_strategies.ipynb`)

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
  That was on this set, before screens. On notebook 8's set it is not free: notebook 16
  finds the luma 32-16's narrow layer losing 2 to 10 of its 16 channels by seed, which is
  its whole 0.0035 seed spread, and a leaky ReLU worth +0.0023 SSIM-Y over the ReLU pair.
- **The ruler is almost five times narrower than the first sweep's** — 0.13 dB wpsnr
  against 0.61, from 2,369 validation patches of 149 sources instead of 76 patches of two
  frames — and the differences between bodies of one size are still inside it.

**What that means for the client: two tiers, and nothing between them earns its place.**
Notebook 7's `luma 24 + 2 blocks` where the device has the bandwidth (10 ms a 1080p frame
on an RTX 3060, 29 ms on a GTX 1650), and a 32-16-sized luma body where it does not (25 ms
on an Iris Xe or an M1, where the blocks take 54). Within the cheap tier the choice does
not move quality, so it should be made on what the browser measures, and on colour, which
the luma frame keeps at bicubic for nothing. (On the screens set it does move it: notebook
16's leaky luma 32-16 is now the cheap tier.)

So the notebook now trains only those two tiers and the control: `luma 24 + 2 blocks`,
`luma 32-16` and `32-16` at three seeds, five runs (about 45 minutes) instead of eleven. The
table above is the record of the other five, and the notebook publishes `luma 32-16` as its
second graph where it published `32-16 leaky`. That graph's architecture (`PlainUpscaler` with
a leaky ReLU) is no longer in the code, so its graphs were deleted from `dist/upscale/` on
2026-11-20; the numbers below that mention it are the record of what it scored.

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
  trained, they would learn from zero padding, and tiled, they would seam. The notebook
  measured every strategy's halo while it defined strategies of its own; it now builds only
  `helpers/models.py`'s, asserts their declared halo against the patches' and every graph
  against the op budget. Those two and `attention` are no longer in it.
- **One run of the control flatters everything.** The verdict first compared each
  strategy with seed 0 of the control, which happened to be the lowest of the three, and
  called two ties wins. It compares with the mean now. Every single-run comparison in this
  file has the same exposure.

Published to the page (`32-16 leaky`'s graphs since deleted, see above): `luma 24 + 2 blocks`
and `32-16 leaky`, the best two graphs (the
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

## Desktop screens and a known stream (`08_upscale_screens.ipynb`)

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

## Fewer operations: gains, tables, directions (notebooks 9–12)

Four notebooks asked whether notebook 7's body (28,120 operations an LR pixel) could be beaten
or matched for much less arithmetic. All four keep colour bicubic and correct luma only; all
train to convergence on the screens patch set with notebook 8's recipe; all are measured on
the same 3,523 validation patches. Gain over bicubic, SSIM-Y / luma PSNR dB:

| | ops / LR px | ssim-y | psnr-y | screens | photos | video |
|---|---|---|---|---|---|---|
| bicubic | 0 | 0.9049 | 33.18 | | | |
| DirNet B: one 5×5 filter (12) | 1,800 | 0.9096 | 33.35 | +0.0095 / +0.30 | +0.0071 / +0.14 | −0.0001 / +0.11 |
| DCT, fixed gains (9) | 4,600 | 0.9107 | 33.45 | +0.0114 / +0.38 | +0.0085 / +0.24 | +0.0002 / +0.21 |
| DCT, gains decided through 32 (9) | 6,104 | 0.9172 | 33.45 | +0.0227 / +0.45 | +0.0129 / +0.14 | +0.0044 / +0.20 |
| 3×3 pattern filter (11) | 2,032 | 0.9188 | 33.62 | +0.0259 / +0.61 | +0.0114 / +0.21 | +0.0063 / +0.45 |
| `S`, one exact address (11, 3 seeds) | 2,295 | 0.9238 | 33.44 | +0.0427 / +1.33 | +0.0113 / −0.06 | +0.0052 / −0.35 |
| `SDY x2`, one exact address (11) | 7,389 | 0.9267 | 33.73 | +0.0480 / +1.81 | +0.0138 / +0.07 | +0.0066 / −0.12 |
| `S` table, fine-tuned (10) | 3,850 | 0.9251 | 33.83 | +0.0453 / +2.08 | +0.0116 / +0.02 | +0.0060 / −0.08 |
| DirNet C: a blend chosen per pixel (12) | 8,878 | 0.9277 | 33.80 | +0.0481 / +1.46 | +0.0150 / +0.21 | +0.0081 / +0.22 |
| DirNet D: C and a learned correction (12) | 11,598 | 0.9303 | 34.22 | +0.0526 / +2.11 | +0.0162 / +0.29 | +0.0100 / +0.63 |
| **`SDY x2` table, fine-tuned (10)** | 16,724 | **0.9343** | 34.63 | **+0.0643 / +3.30** | +0.0167 / +0.38 | +0.0102 / +0.64 |
| notebook 4's model, trained before screens | 28,120 | 0.9292 | 34.45 | +0.0425 / +1.75 | +0.0217 / +0.74 | +0.0122 / +1.20 |
| notebook 8's model, blind (3 seeds) | 28,120 | 0.9348 | 34.67 | +0.062 / +2.99 | +0.019 / +0.46 | +0.012 / +0.88 |
| notebook 8's model, told the path (2 seeds) | 28,120 | **0.9370** | **34.93** | +0.063 / +3.07 | **+0.024 / +0.73** | **+0.014 / +1.30** |

The rulers: `decide 8` at three seeds 0.0008 SSIM-Y apart (notebook 9), the `S` network at
three seeds 0.0003 (notebook 10, which notebook 12 borrows).

- **Gains on a DCT are not enough.** The ceiling is high — gains chosen from the truth reach
  0.974 SSIM-Y on the inner 112 px, against 0.905 for bicubic — but a fixed gain table is
  barely better than a fixed filter, and deciding per block buys +0.0065 over it. Notebook 4's
  model is 0.012 SSIM-Y ahead of the best of them. Dropped.
- **Tables are the best model on screens in this project.** Two stages of three kernels,
  `SDY x2`, gain +0.064 SSIM-Y and +3.3 dB on screens, and filling its network into a table at
  17 points costs 0.0023 SSIM-Y that fine-tuning wins back (+0.0024). On photos and video they are
  behind any model that computes. Most of a table is memory nothing reads: 99% of the `S`
  table's reads land in 15% of its cells, and 85% within one step of its diagonal - the
  observation notebook 11 is built on.
- **Choosing a direction per pixel is what DirNet is worth.** C over B is +0.018 SSIM-Y; the
  learned correction (D over C) adds +0.0026. D beats notebook 4's model at 41% of its
  operations - by 0.010 SSIM-Y on screens and 0.0011 overall - and loses to it on photos
  and video. The predictor did not learn which way an edge runs: on screens it spreads its
  choice over the row, the column and the other diagonal (31%, 26% and 30%) much as it does
  on photos and video, which is what sent notebook 13 to ask the host.
- **None of them beats notebook 8's own model overall**, which notebooks 10–14 do not compare
  against: they compare against notebook 4's, trained before the set had screens. Notebook
  8's blind model ties the `SDY x2` table on screens (+0.062 against +0.064) and is ahead on
  photos and video; told the host's path, it is the best model here on every kind but
  screens, where it is within 0.0013 of the table. It costs what notebook 4's does - and in
  the browser that is 3.9 ms a 1080p frame against the table's 75 (see "The browser
  benchmark").

## Bits against compute (`15_upscale_tradeoff.ipynb`)

For every published client, on 24 whole 720p clips (8 of each kind) streamed the way the host
sent then: how many bits plain H.264 with bicubic would need for the same SSIM-Y, over the
bits the client's stream really used - its **worth in bandwidth** - and the SSIM-Y it adds
over plain H.264 at the same bits. The plain stream is measured up to 3.3 bits per LR pixel so
the best clients have something to be matched against. At the host's default 0.46 bpp, with
what each costs in the browser (from "Results at the client's tile"):

| client | 1080p, browser | screens: worth, same bits | photos | video |
|---|---|---|---|---|
| DirNet B | 0.8 ms | ×1.06, +0.0043 | ×1.07, +0.0027 | ×0.89, −0.0022 |
| DirNet C | 7.1 ms | ×1.51, +0.0359 | ×1.16, +0.0127 | ×1.28, +0.0086 |
| DirNet D | 7.7 ms | ×1.68, +0.0367 | ×1.20, +0.0147 | ×1.52, +0.0108 |
| **luma 32-16 leaky** (16) | **2.8 ms** | ×1.81, +0.0402 | ×1.21, +0.0157 | ×1.51, +0.0110 |
| notebook 4 | 3.9 ms | ×1.43, +0.0292 | **×1.43, +0.0237** | ×1.47, +0.0129 |
| **notebook 8** (told the host's path) | **3.9 ms** | **×2.13, +0.0436** | ×1.22, +0.0163 | **×1.60, +0.0151** |
| `SDY x2` table | 75 ms | ×2.56, +0.0479 | ×1.21, +0.0154 | ×1.48, +0.0111 |

Worth is the geometric mean over the clips some bitrate reaches; on screens 7 of the 8 clips
are reached by **no** bitrate at all - H.264 codes a still screen at the same size whatever
it is given, so no bandwidth buys what a model adds there, and the same-bits column is the one
to read.

- **Notebook 8's model is the client to ship.** At the cost of notebook 4's it adds +0.0436
  SSIM-Y on screens at the same bits, where notebook 4's adds +0.0292, and it is the best
  client on video. Only the 75 ms table beats it on screens, by 0.0043.
- **The leaky luma 32-16 is the cheap tier.** In 70% of notebook 8's browser time it keeps
  all but 0.0034 of its lead on screens (+0.0402) and 0.0041 on video. It is ahead of
  notebook 4's model on screens (behind on photos and video) and level with or ahead of
  DirNet D everywhere, for less time than either.
- **It gives up photos, and so does everything trained with screens.** On whole photo clips
  notebook 8's model is behind notebook 4's (+0.0163 against +0.0237), which the patch scores
  did not show (there it was ahead, +0.024 against +0.022). But the leaky 32-16, DirNet D and
  the table - blind, and trained on the same screens set - land at +0.0147–0.0157 too, so it
  is the data mix, not the told graph. The clips are 8 sources a kind, so this is a direction
  to check rather than a measurement; for a desktop stream, screens and video are what matter.
- **The traditional stream saturates early on screens**: bicubic gains 0.0147 SSIM-Y from
  0.46 to 3.3 bpp on screens, where notebook 8 adds 0.0436 at 0.46. Spending compute beats
  spending bits on a desktop stream by a wide margin.

The notebook ran for 97 minutes, most of it - by the graphs' own CPU times - notebook 10's
table on the ONNX Runtime CPU provider. On the re-run that added notebook 8, the same-bits
numbers of the clients it had before moved by at most 0.0004 SSIM-Y - the codec is not
deterministic, and the clips are re-encoded each run - but a worth, a ratio read off a flat
curve, can move far more. The table above is the re-run that added notebook 16's leaky 32-16
(109 minutes): the same-bits numbers moved by at most 0.0002.

## Distillation, edges and a leaky ReLU (`16_upscale_distill.ipynb`)

Two ways to train bodies already here better without changing what the client runs:
**distillation** for the cheap luma 32-16 - half of its loss L1 against notebook 8's model's
picture instead of the truth - and an **edge-weighted loss** for notebook 8's told model - the
L1 weighted by the truth's luma gradient (Sobel, normalised to a mean of 1). Each was
smoke-tested for two epochs from a trained model, then trained to convergence with notebook
8's recipe on the screens patch set, at two seeds where one was not enough, and scored on the
3,523 validation patches. The runs turned up a third thing, and it is the one that pays.

| | ssim-y | psnr-y | seeds apart |
|---|---|---|---|
| luma 32-16, ReLU, seeds 0 / 1 | 0.9270 / 0.9305 | 34.03 / 34.27 | 0.0035 |
| luma 32-16, ReLU, distilled | 0.9239 | 33.94 | one seed |
| **luma 32-16, leaky ReLU, seeds 0 / 1** | **0.9309 / 0.9312** | 34.36 / 34.34 | **0.0003** |
| luma 32-16, leaky, distilled | 0.9312 | 34.43 | one seed |
| notebook 8 told, seeds 0 / 1 | 0.9369 / 0.9371 | 34.94 / 34.92 | 0.0002 |
| told, edge-weighted, seeds 0 / 1 | 0.9374 / 0.9364 | 34.97 / 34.92 | 0.0010 |
| told, leaky ReLU, seeds 0 / 1 | 0.9367 / 0.9375 | 34.94 / 35.06 | 0.0008 |

- **The 32-16's seed lottery is dying ReLUs.** Counted over the validation set, its 16-wide
  layer has lost 8 channels at seed 0, 2 at seed 1 and 10 in the distilled run. A leaky ReLU
  (slope 0.1) cannot lose one: the two seeds land 0.0003 apart, 0.0023 above the ReLU pair's
  mean, and above its better seed on every content but screens, where it is level. Notebook
  4 measured dying as free, before there were screens; on this set it costs.
- **`LeakyRelu` is free in the browser.** The leaky graph runs 0.430 ms a float32 tile,
  0.356 a float16 one and **2.75 ms a 1080p frame** as one float16 graph, against 0.430,
  0.354 and 2.73 for notebook 4's ReLU luma 32-16 in the same rounds (2026-10-27, the
  harness of "Results at the client's tile", which reproduced that table's other rows).
- **Distillation buys luma dB, not SSIM-Y.** The leaky student scores the plain seeds'
  SSIM-Y and +0.08 dB luma against a ruler of 0.02, on every content. It is one seed, and
  the recipe picks its epoch by SSIM-Y, which distillation does not move, so it is not
  published. The ReLU student lost 10 of its 16 narrow channels and says nothing about the
  teacher.
- **The edge-weighted loss moves the error, it does not shrink it.** Both seeds keep a
  direction in dB - screens +0.16 and +0.10, video −0.04 and −0.09 - but their SSIM-Y mean
  (0.9369) is the plain pair's, and they are five times as far apart. Notebook 8's loss
  stays.
- **A leaky ReLU does nothing for notebook 8's model**: its residual blocks add their input
  back, so nothing died. The mean is 0.9371 against 0.9370, four times as far apart. Seed 1
  is the best single model in the project (+0.13 dB luma, ahead on every content), but picking it
  would be picking a seed, not a body.
- **Two epochs from a trained model say which way a loss pushes, not where it ends.** The
  smoke tests found the edge loss's direction and missed that it does not pay.

Published: the leaky luma 32-16 at seed 1 (`upscale_luma_32-16_leaky_*`, float32, float16
and a float16 frame graph; 17,080 operations and 1,192 bytes an LR pixel, inside the op
budget). On notebook 15's whole clips it is the cheap tier (see "Bits against compute").
`models.ToldLuma` moved into `helpers/models.py` for this notebook; notebook 8 still defines
its own copy.

## 60 fps on a GTX 1660 Ti (`24_upscale_fast.ipynb`)

Notebook 23's told 24×2 takes 34.6 ms for a 1280×720 stream's 16 tiles on the GTX 1660 Ti.
The reference allows the graph 9.2 ms (`reference_60fps.md`). Notebook 24 makes a drop-in
that fits: the same `input`, `condition` and `output`, the same 4 px halo, and a float16 twin.
It was built on the RTX 3090 alone. The 1660 Ti is predicted by timing every layer of that
card's cost table again on the 3090 and charging each kernel its kind's ratio (4.0-4.8 for a
3×3 reading up to 24 channels, 6.2 at 32, about 7 for a 1×1, 3.5 for an `Add`, 4.6 for the
output path). That puts the told graph at 36.0-37.3 ms against its measured 34.6. A model is
held to 8.3 ms predicted, 10% under the 9.2.

What made room:

- **Nothing fits at the LR pixels.** The bicubic skip (0.17 ms a tile on the 1660 Ti) and the
  output path at full size (0.15) take two thirds of the budget, so a single 3×3 stem and a 1×1
  tail already predict 9.8 ms. A body at **a quarter of the pixels** fits: a 4×4 stride-2 stem
  makes one feature vector for each 2×2 LR cell, one residual block (3×3, then 1×1) works
  there, a 1×1 tail writes the cell's 16 luma values, and one `DepthToSpace` by 4 makes the
  picture. It reaches 3 LR px, inside the client's 4.
- **The skip shrinks.** Bicubic for a 2×2 cell is one 6×6 stride-2 convolution, 3 → 48
  (`models.QuarterBicubicSkip`, exact). Without its outer ring of taps, which weigh 0.023 in
  bicubic's kernel, it is a 4×4 at half the cost (`models.truncated_quarter_skip`). Frozen,
  its colour is bicubic's to 0.01 dB of U and V PSNR. The tail's rows are written once per
  colour instead of a `Concat`.

On notebook 21's 9,628 validation patches; a model with two seeds is their mean (0.0001-0.0003
apart). The network adds the same value to R, G and B, so a model's colour (U, V) is its
skip's. The 1660 Ti figures are predicted, central / conservative, from the candidate graphs
timed in one session, except the told model's:

| | ssim-y | luma dB | U / V dB | over bicubic | MAC / LR px | 1660 Ti, 16 tiles |
|---|---|---|---|---|---|---|
| bicubic | 0.9045 | 33.55 | 43.99 / 43.93 | | | |
| told 24×2 (23, ships) | 0.9289 | 35.07 | 44.03 / 43.97 | +0.0243 / +1.52 | ~13,000 | 34.6 ms, measured |
| **q24, frozen 4×4 skip (published)** | **0.9212** | **34.47** | **44.00 / 43.94** | **+0.0167 / +0.92** | 1,824 | **7.6 / 7.8** |
| q24, exact 6×6 skip | 0.9213 | 34.51 | 44.00 / 43.95 | +0.0167 / +0.96 | 1,824 | 8.7 / 8.9 |
| q24, 4×4 skip trained | 0.9214 | 34.43 | 43.04 / 43.23 | +0.0168 / +0.87 | 1,824 | 7.6 / 7.8 |
| q32, 4×4 skip trained | 0.9229 | 34.55 | 43.03 / 43.26 | +0.0184 / +1.00 | 3,072 | 8.2 / 9.1 |
| q20, 4×4 skip trained | 0.9197 | 34.34 | 43.06 / 43.22 | +0.0152 / +0.78 | 1,320 | 7.2 / 7.3 |

- **The published graph fits, by prediction.** Timed again beside the told 24×2 (1.35
  against 6.65 ms on the 3090), it predicts 7.1 ms central and 7.3 conservative, and 7.0 ms
  scaled by the told graph's measured 34.6. With the 2.2-4.1 ms the client spends around a
  model, that is 9-12 ms a frame against 13.3.
- **It keeps about two thirds of the told 24×2's gain** for a fifth of its 3090 time: 69% of
  the SSIM-Y and 61% of the luma dB, and all of bicubic's colour. It keeps three quarters on
  x264's frames and three fifths on NVENC's.
- **The skip has to stay frozen.** Trained with the rest, under a loss that is 84% luma
  SSIM, it gives up about 1 dB of U and V - colour worse than bicubic's - and buys nothing:
  the same SSIM-Y, 0.04 dB less luma. Frozen, it is within 0.04 dB of luma of the exact 6×6
  skip, with the same colour, for the 1.1 ms that would take q24 over the bar.
- **Four channels of width are worth about 0.0016 SSIM-Y.** q32 is inside the 9.2 ms by the
  central prediction and outside it by the conservative one; only the card can say which.
  Smaller does not pay: about 3.1 ms of q24's time is the contract's fixed part (casts,
  layout transposes, `DepthToSpace`, the skip), and q20 saves 0.2 ms for 0.0017 SSIM-Y.
- Notebook 23's checks all pass:
  - the graph computes the model to 1e-6 at every condition;
  - the float16 twin is within 0.16/255;
  - a 1280×720 frame cut into the client's tiles matches the whole frame to 6e-7;
  - graph capture accepts it, so no node runs on the CPU.

**Not yet measured on the GTX 1660 Ti, and its pictures not yet looked at** (the body works on
2×2 cells, and a faint pattern on that grid is something SSIM-Y need not show). The b16 graph
`upscale_fast_q24_frozen_skip_fp16_tile320x180_b16.onnx` is ready for Run 8 of
`bottlenecks_gtx1660ti.md`.

## The JND loss (`26_upscale_jnd_loss.ipynb`)

The perceptual repo beside this one (`perceptual_jnd_loss.py`) makes its JND metric a loss:
visible error in just-noticeable differences after contrast sensitivity, a brightness gain
and masking, d / (d + 1) a picture. Its settings are in `helpers/jnd_settings.json`, in the
repo's config format; the judge reads the same file. Notebook 26 trains the two best bodies on it
with notebook 22's recipe and seeds, keeps the epoch with the best validation JND, and sets them
against their SSIM-loss runs. `helpers/judges.py` holds the judges neither loss trained on: LPIPS
(AlexNet, v0.1, fitted to people's choices; built from its public weights, checked against the
`lpips` package to 8e-5) and GMSD.

Two rounds so far: the spec's defaults (2026-11-17), and settings given on 2026-11-19 that differ
in masking exponent 1.2 (0.7), tiles pooled at β = 16 (4), and the d-to-MOS curve. On the 9,628
validation patches, each the mean of two seeds (the JND column changes with the settings, so it
is left out):

| | SSIM-Y | luma dB | LPIPS | GMSD | seam between LR pixels |
|---|---|---|---|---|---|
| bicubic | 0.9045 | 33.55 | 0.1556 | 0.0712 | 1.063 |
| told 24×2, SSIM loss (ships) | **0.9288** | **35.08** | **0.1208** | **0.0567** | 1.210 |
| told 24×2, JND loss, spec defaults | 0.9224 | 34.97 | 0.1231 | 0.0587 | **1.160** |
| told 24×2, JND loss, 11-19 settings | 0.9209 | 34.79 | 0.1249 | 0.0599 | 1.176 |
| q24, SSIM loss (published) | **0.9212** | 34.47 | 0.1289 | **0.0600** | 1.292 |
| q24, JND loss, spec defaults | 0.9159 | **34.48** | **0.1261** | 0.0604 | **1.171** |
| q24, JND loss, 11-19 settings | 0.9147 | 34.31 | 0.1279 | 0.0616 | 1.193 |

- **The told 24×2 stays on the SSIM loss**: under either setting the JND loss loses on every
  judge, its own JND included, and at QP 30 and over it leaves lighter strokes on mottled
  backgrounds.
- **The q24 looks better on the JND loss**: with the spec's defaults, LPIPS closer on 64% of the
  patches (seeds 0.0007 apart), crisper text with less ghosting, and most of the pixel grid the
  SSIM loss leaves gone (the step between two LR pixels' outputs is 1.29 times the step inside
  one, 1.17 under the JND loss, 1.04 in the truth) - the pattern on the q24's cells that
  notebook 24 had not looked at. It costs 0.0053 SSIM-Y and no luma dB.
- **The 11-19 settings train worse models than the defaults**, both bodies, every judge above:
  LPIPS 0.0018 worse, less fine detail kept (0.55-0.57 of the truth's against 0.59-0.60). The q24
  still edges its SSIM-loss run (closer on 55% of the patches). Likely, untested: at β = 16 a
  patch's d is nearly its worst tile's, so one of its 2×2 tiles takes most of the gradient.
- People: one viewer's blind A/B of 20 whole-frame crops (the Upscaler Eye Test page) preferred
  the spec-default JND loss in 11 pairs, the SSIM loss in 5, neither in 4 (6-3 for the told
  24×2, 5-2 for the q24). Round 2, the 11-19 models on the same windows, waits for votes.
  Nothing is exported; the JND q24 to export is round 1's
  (`checkpoints/jnd_runs/_spec_defaults/fast_q24_seed1.pt`), through notebook 24's export cell.

## The SSIM and JND losses together (`27_upscale_mixed_loss.ipynb`)

The two losses fail differently: the SSIM loss places edges and keeps brightness but leaves a
grid of the upscaler's pixels; the JND loss takes most of the grid away but lets edges and
brightness drift. Notebook 27 trains the sum, `ssim_l1_loss` + λ × the JND loss at the spec's
defaults, from random weights with notebook 26's recipe. On a trained model the JND term's
gradient is 2.6-2.9 times the SSIM loss's, so λ = 0.1, 0.3 and 1 give it a quarter of, the same
as, and three times the SSIM loss's pull. On the validation patches, each the mean of two seeds
(at most 0.0009 apart on LPIPS):

| | SSIM-Y | luma dB | LPIPS | GMSD | seam between LR pixels |
|---|---|---|---|---|---|
| told 24×2, SSIM loss (ships) | **0.9288** | 35.08 | **0.1208** | **0.0567** | 1.207 |
| told 24×2, mix 0.1 | 0.9283 | 35.18 | 0.1209 | 0.0568 | 1.196 |
| told 24×2, mix 0.3 | 0.9278 | **35.24** | 0.1218 | 0.0568 | 1.174 |
| told 24×2, JND loss | 0.9224 | 34.97 | 0.1231 | 0.0587 | **1.159** |
| q24, SSIM loss (published) | **0.9212** | 34.47 | 0.1289 | 0.0600 | 1.291 |
| q24, mix 0.1 | **0.9212** | 34.59 | 0.1275 | 0.0595 | 1.264 |
| q24, mix 1 | 0.9196 | **34.64** | 0.1269 | **0.0594** | 1.208 |
| q24, JND loss | 0.9159 | 34.48 | **0.1261** | 0.0604 | **1.170** |

- **The q24 is better on the sum than on either loss alone.** Mix 1 beats the SSIM loss on
  every judge but SSIM-Y (LPIPS closer on 59% of the patches, 65% of the screens; GMSD the best
  of all; +0.17 dB; the least error at edges; a third less grid) for 0.0016 SSIM-Y; mix 0.1
  gets about two thirds of the LPIPS gain with no SSIM-Y lost. Its text is a little darker and
  smoother, nearer the truth's weight.
- **The told 24×2 gains no LPIPS or GMSD from it**, only 0.10-0.16 dB of luma, centred
  brightness and a little less grid; it stays on the SSIM loss.
- Every mix scores more luma dB than either loss alone. Nothing is exported; the q24 to export
  next is mix 1 (`checkpoints/mix_runs/fast_q24_mix_1_seed*.pt`), through notebook 24's export
  cell, once the eye agrees.

## A larger, more diverse set (`28_upscale_qp_diverse_pairs.ipynb`)

Notebooks 22–27 train on notebook 21's pairs: YouTube video only (394 targets, one sequence of
each training video and two of each test video measured), 8 patches a frame, no screens and no
photos. Notebook 28 makes them again with notebook 21's lines, labels, bins and patch shape,
and wider content: every sequence on disk measured with notebook 17's test, up to three
targets a video; 1,000 drawn screens from two sets (the second, `raw/screens-s1/`, drawn for
it with `screens.py --seed 1 --name screens-s1`); 400 DIV2K and Flickr2K photos; images as
32-frame clips of a window sliding, zooming or still (`helpers/motion.py`); 4 patches a
frame; and a `kind` label per patch, for scoring only. It writes `qp_diverse_pairs/` in
`qp_slice_pairs/`'s format, so a notebook reads it by changing `PAIRS_DIR`.

| | notebook 21 | notebook 28 |
| --- | --- | --- |
| training patches | 34,364 | **147,813** (video 46%, screens 37%, photos 16%) |
| validation patches | 9,628 | **24,216** (video 39%, screens 46%, photos 15%) |
| sources, train / validation | 306 / 88 | 2,603 / 426 |
| videos, train / validation | 306 / 54 | 540 / 72 |
| slice QP 42+, training | 3,190 | 17,732 |
| NVENC P frames at 46–52, train / validation | 430 / 64 | 2,630 / 422 |
| bicubic SSIM-Y, validation | 0.9045 | 0.8524 (video 0.9034, screens 0.8487, photos 0.7322) |

- **A score on this set is not comparable with one on notebook 21's**: it is harder by its
  content. Its video alone reads about what notebook 21's set does under bicubic (0.9034
  against 0.9045).
- **The label still means what it did**: within every kind and line, bicubic falls as the
  slice QP rises, up to 42–46. NVENC's frames are nearer the target than x264's at the same
  QP from 18–26 to 38–42 in every kind, as notebook 21 found on video.
- Screens and photos fill the top bins that the video leaves thin: notebook 17's targets are
  cheap to code, and a detailed picture sliding at 0.25 Mbit/s is not.
- 80 minutes to build (12 sources at once, through `degrade.degrade_many`: notebook 21's
  `degrade` holds the GIL while it pipes a clip, so in threads it ran slower than one at a
  time). Trains nothing yet.

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
`05_upscale_geometry.ipynb` below sweeps the batch and the rectangle as well, charges everything
per output pixel, and lands on **640×360 out, from a 320×180 step** — 6.8 ms a 1080p frame
against this table's 8 ms at 128, and the same tile divides 720p, 1440p and 4K exactly too.
The `ms / frame` column here is also missing what it costs to *not* divide the frame: a 256
step computes 52% more of a 1080p output than that output has.

The rest of the original argument for 128 still holds: 84% of computed pixels kept at halo 6, a power of two once the pixels live in textures
and workgroups, bounded memory independent of stream resolution, and 40 tiles per 1080p
frame so the main thread yields 40 times instead of blocking once. That sweep was run
against a model that has since been removed, so the timings behind it are gone; the geometry
argument is not.

## The geometry a client runs at (`05_upscale_geometry.ipynb`)

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
| `upscale_strategies` | luma 24 + 2 blocks and luma 32-16, fp32 / fp16, and fp16 as a whole 1080p frame | 328×188 / 968×548 | 4 |

The first results table below is the older graphs at the 128 step; "Results at the client's
tile" measures every graph at the 320×180 step, which is everything from notebook 4 on.

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

### Results at the client's tile — NVIDIA Ampere, Chrome 155, ms per tile

Every 320×180-step graph in `dist/upscale/`, measured 2026-10-26: WebGPU, GPU-resident,
graph capture, medians of three interleaved rounds (every configuration within 3% of itself
across rounds), driven headless over the DevTools protocol so the page ran exactly as it does
by hand. A tile is 640×360 out, nine of them a 1080p frame; a frame graph is the whole frame
in one run. Notebook 11's rows were first measured untrained, since a table's speed does not
depend on what is in it, and again once trained: the same to 1%.

| Model | ops / LR px | fp32 tile | fp16 tile | fp16 frame graph | **1080p frame** |
| --- | --- | --- | --- | --- | --- |
| static bilinear / bicubic (6) | 0 | 0.040 / 0.040 | — | 0.098 / 0.159 | 0.1 / 0.2 ms |
| DirNet B, one 5×5 filter (12) | 1,800 | 0.118 | 0.102 | 0.756 | 0.8 ms |
| DCT, fixed / decided through 32 (9) | 4,600 / 6,104 | 0.239 / 0.266 | 0.211 / 0.234 | 1.39 / 1.52 | 1.4 / 1.5 ms |
| 32-16: reparam, luma, leaky (4, 6) | 17,080–19,392 | 0.42–0.43 | 0.34–0.35 | 2.62–2.73 | 2.6–2.7 ms |
| luma 32-16 leaky (16) § | 17,080 | 0.430 | 0.356 | 2.75 | **2.8 ms** |
| luma 24 + 2 blocks: notebooks 4, 7, 8 | 28,120 | 0.646 | 0.500 | 3.91 | **3.9 ms** |
| DirNet C (12) | 8,878 | 0.963 | 0.864 | 7.05 | 7.1 ms |
| DirNet D (12) | 11,598 | 1.069 | 0.950 | 7.72 | 7.7 ms |
| `S` table, interpolated (10) | 3,850 | 1.720 | 7.36 ‡ | 44.0 ‡ | 15.5 ms |
| `SDY x2` table, interpolated (10) | 16,724 | **8.350** | 30.76 ‡ | 264 ‡ | **75 ms** |
| `S`, one exact address (11) | 2,295 | 0.323 † | | 2.27 † | 2.3 ms |
| 3×3 pattern filter (11) | 2,032 | 0.522 † | | 3.94 † | 3.9 ms |
| **`SDY x2`, one exact address (11)** | 7,389 | **0.966 †** | | 7.33 † | **7.3 ms** |

‡ refuses graph capture, so timed without it: some of its nodes run on the CPU, and the
round trips are most of the time (the GPU's own share is 1.5 ms of the `S` tile's 7.4).
† float32 only (a float16 tap can round into the next row), and after the change to
notebook 11 described below; as first written its graphs took 5.7, 8.3 and 25.0 ms a tile.
§ measured 2026-10-27 the same way, beside notebook 4's luma 32-16 (0.430, 0.354, 2.73) and
notebook 8's body, which read as above.

- **The table that wins on screens is the one graph on the page that cannot run.** `SDY x2`
  costs 75 ms a 1080p frame, 19 times notebook 8's model (3.9 ms) for +0.0013 SSIM-Y on
  screens and less everywhere else. Notebook 15 charges it 16,724 operations - fewer than
  notebook 8's 28,120 - because it counts arithmetic, and a table's cost is not arithmetic:
  it is dozens of small operators writing full tensors and two `Gather`s reading a 5 MB table at
  random. **Operations per pixel do not rank these graphs.** DirNet C, at a third of notebook
  8's operations, is 1.5–1.8 times as slow: its blend is five `DepthToSpace`s, a `Softmax`,
  a `ReduceSum` and products of full-size tensors, where the residual body is seven
  convolutions.
- **Exact-address tables fix it, once the graph stays on the GPU.** Reading one entry
  instead of interpolating takes `SDY x2`'s GPU time from 8.1 to 0.91 ms a tile. As first
  written, though, notebook 11's graphs were 3 times *slower* than notebook 10's, because
  three of their operators have no WebGPU kernel and ran on the CPU. Found with one-operator
  probe graphs, each run under graph capture (which a session refuses outright when any
  node leaves the GPU): **integer `ReduceSum`, `Round`, integer `Mul` and anything int64**
  are off the GPU; `Cast` to int32, int32 `Add`, `Floor`, and `Gather` with int32 indices
  are on it. Notebook 11 now sums an address with three `Add`s, rounds with
  `floor(x + 0.5)` and adds its pattern digits in float32 (exact at these sizes), and checks
  the same rows by hand as before. Its `SDY x2` runs a 1080p frame in 7.3 ms, in the same
  tier as DirNet C - but it scores +0.048 SSIM-Y on screens, not the interpolated table's
  +0.064, and under notebook 8's model (see "Notebooks").
- **float16 is not a speed-up for a table.** Both of notebook 10's float16 graphs leave the
  GPU (its integer arithmetic on halves), and notebook 11 cannot use float16 at all, since
  a tap can round into the next row. Every other model here is 10–23% faster in fp16.
- **A frame graph saves the per-run overhead**: 3.9 ms against 9 × 0.50 for notebook 8's
  body, and 0.1 against 0.4 ms for the static filters - which on a frame are all dispatch.

`dist/` is committed, so the page has models on a fresh clone; a notebook's export cell
replaces the ones it owns.

## Where that leaves it

On notebook 0's set. SSIM-Y on the 24,286 validation patches, at the published seed; screens,
photos and video are SSIM-Y over plain H.264 with bicubic at the bits the client's stream used,
at the host's default 0.46 bits per LR pixel (notebook 15, 8 whole clips of each kind).

| | SSIM-Y | over bicubic | multiply-adds / LR px | screens | photos | video | in `dist/upscale/` |
|---|---|---|---|---|---|---|---|
| bicubic (the GPU's sampler) | 0.8521 | | 0 | | | | nothing to run |
| **told 24×2 (notebooks 22, 23)** | **0.8919** | **+0.0398** | 13,932 | +0.0397 | **+0.0163** | **+0.0137** | `client/upscale.onnx`: what desktop-streamer runs |
| **told q24, its skip frozen (24)** | 0.8759 | +0.0238 | 1,824 | +0.0194 | +0.0079 | +0.0080 | `client/upscale_fast.onnx`: the 60 fps tier |
| luma 24 + 2 blocks, blind (7) | 0.8886 | +0.0365 | 13,932 | +0.0400 | +0.0081 | +0.0110 | `upscale_luma` |
| `SDY x2` table, fine-tuned (10) | 0.8858 | +0.0337 | look-up, 2.5 MB | **+0.0413** | +0.0028 | +0.0101 | `upscale_lut_sdy_x2_17`: slow in a browser |
| luma 32-16, leaky ReLU (16) | 0.8828 | +0.0307 | 8,484 | +0.0350 | +0.0039 | +0.0091 | `upscale_luma_32-16_leaky` |
| DirNet full (12) | 0.8811 | +0.0290 | 5,687 | +0.0353 | +0.0023 | +0.0087 | `upscale_dirnet_full`: float32 only |

**The told 24×2 stays.** On patches it is 0.0008 under the best rows (notebook 20's switched
pair), at the same bits it adds the most of any client on photos and video and within 0.0016 of
the most on screens, and it stays above plain H.264 at every bitrate measured. The earlier graph
of its shape ran a 1280×720 stream's 16 tiles in 6.9 ms on the RTX 3090 and in 34.6 ms on the
GTX 1660 Ti: too slow there at that stream size.

**Where it cannot run, the told q24**, predicted at 7.6 ms on the GTX 1660 Ti: 60% of the gain,
bicubic's colour, and the most quality for its compute off screens. The blind leaky 32-16 and
DirNet full gain more on screens at three to five times its multiply-adds, and less on photos
and video.

**No table and no new loss.** The `SDY x2` table matches the told model on screens but trails
on photos and video and runs outside the browser's fast set; the JND loss and its mixes win
nothing worth a change (notebooks 26, 27). Two told models switched at QP 30 (notebook 20) are
+0.0008 over notebook 22's one at both seeds, for a second graph and a jump in the picture
where a stream's QP crosses 30.

The rule the browser keeps teaching is the one notebook 3 started: **whether a provider runs
an operator natively is worth more than what the operator costs in theory.** int8 through two
extra ops was 130× slower than float16; a table interpolated through dozens of small operators
is 19× slower than a network with more arithmetic; a table whose graph has one integer
`ReduceSum` in it runs on the CPU. Count operations to design, and measure in the browser to
decide.

## Next

- **Measure the told q24 on the GTX 1660 Ti, and look at its pictures**: Run 8 of
  `bottlenecks_gtx1660ti.md` with `upscale_fast_q24_frozen_skip_fp16_tile320x180_b16.onnx`
  (at most 9.2 ms), then the client with `upscale_fast.onnx` / `upscale_fast_fp16.onnx` in
  place of `upscale.onnx` / `upscale_fp16.onnx` (`enhance.ms` at most 13.3). If the card has
  the room, train q32 with the frozen skip: with their skips trained, q32 is +0.0034 over q24
  (notebook 24), predicted at 8.2 ms central and 9.1 conservative.
- **Run the browser benchmark on this set's graphs** (`uv run src/upscale/benchmark/main.py`):
  every browser time in this file is of the graphs deleted on 2026-11-20.
- **Let the client choose its upscaler by what the GPU affords**, rather than replace the told
  24×2 with the q24 everywhere: the RTX 3090 ran the told graph's shape at 16 tiles in 6.9 ms.
- **Skip the tiles that did not change**, in the client: exact, free, and the largest factor
  on a desktop stream. Tiling beats one run of the whole frame while fewer than about 80% of
  the tiles change (notebook 5).
- **Measure the Windows host's downscale.** On Windows desktop-streamer now has the capture
  scale a smaller picture (`gfxcapture` with `resize_mode=scale:scale_mode=bicubic`, in
  `stream-ffmpeg.js`) rather than swscale. The models are trained on swscale's bilinear,
  bicubic and area (`degrade.FILTERS`); how close the capture's scaler is to them is not
  measured.
- **Look at notebooks 26 and 27's frames by eye** (`data/samples/26_jnd/one_set/frames/` and
  `data/samples/27_mix/one_set/frames/`) before closing the JND loss: the scores say the SSIM
  loss stays, and no one has looked.
- **If the JND loss is used again, find the patches it cannot take**: 20 to 35 batches a run
  gave a loss or gradient that is not finite, the same batches whatever the model;
  `training.fit` skips them.

Sibling problems: `../frame_gen_intra/summary.md`, `../frame_gen_extra/summary.md`.
