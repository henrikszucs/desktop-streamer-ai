# desktop-streamer-ai

The video work for Desktop Streamer: upscaling a received frame, and generating frames
between or after the ones that arrive. A `uv`-managed Python project, separate from the Node
application, [desktop-streamer](https://github.com/henrikszucs/desktop-streamer) — nothing in
its `src/` calls it yet. It used to be that repo's `model/` folder, and its history came along.

Two top-level folders. **`src/`** is the source: the code, the notebooks that train and
measure, and the data and checkpoints they leave behind. **`dist/`** is the output: the
finished models, one subfolder per problem, written by the notebooks' export cells and
nothing else. Beside them, **`datasets/`** holds the data, never committed: the public
benchmarks every problem can draw on in `datasets/raw/`, downloaded by
[`datasets/download.py`](datasets/download.py) - and desktop screens, drawn by
[`datasets/screens.py`](datasets/screens.py) - and beside it one folder per problem for what
that problem's preprocessing makes of them (`datasets/upscale/`, `datasets/frame_gen_extra/`).

Three problems, one folder each under `src/`, each with its own `summary.md` - and beside
them `src/mock/`, which is not a problem but the three stand-ins the client runs today (see
below):

| Folder | Problem | State |
| --- | --- | --- |
| [`src/upscale/`](src/upscale/summary.md) | more pixels out than in | fifteen notebooks: static baselines, nine ways to build a model, host hints, and the bits-against-compute trade-off; nothing shipped to the client yet |
| [`src/frame_gen_intra/`](src/frame_gen_intra/summary.md) | interpolate between two frames | not started |
| [`src/frame_gen_extra/`](src/frame_gen_extra/summary.md) | 2x frame generation without latency: the frame between two received ones, made before the next arrives | six notebooks: 60 fps runs of desktop-like motion, static baselines, two trained networks that hedge, and a tile search built from fixed convolutions (+11.1 dB over repeating the frame on 80 px samples, +5.8 on whole frames in the client's tiles, 2.3 ms a 1080p frame in Chrome's WebGPU) that passes desktop-streamer's client tests in place of its mock; an oracle bounds what a learned picker could add at ~2 dB; a motion hint from the host would add +6.0 dB but needs a host capturing at twice the stream's rate, and is shelved |

## The mock graphs the client runs

`src/mock/make_mock_models.py` writes the three graphs desktop-streamer's
`src/client/web/media/models/` holds - `upscale.onnx`, `interpolate.onnx`,
`extrapolate.onnx` - which the client's enhancement menu runs through ONNX Runtime Web
(`src/client/web/src/room/stream-enhance.js` there). They are
stand-ins, not models: the smallest ONNX graph with the shape of the real thing and real GPU
work in between, computing something that leaves the picture right (an identity convolution
and a bilinear ×2; the mean of two frames; their linear extrapolation), so the client
pipeline could be built and timed before any trained weights exist. The client never hands a
graph a whole frame: it cuts it into 328×188 tiles (a 320×180 step with a 4 pixel halo, the
geometry `src/upscale/helpers/tiles.py` measured), runs every tile of a frame as one batch, and
merges the kept centres back - so a model has to be right on a tile of that size, read no
further than the halo, and take a batch. The generator's docstring holds what the runtime's
WebGPU provider charges for the operators the mocks could have been built from, which is
worth reading before choosing a trained model's. A trained model replaces
one by being exported under the same name with the same contract: float32 NCHW in [0, 1]
with `N`, `H` and `W` symbolic - `input` `[N, 3, H, W]` for the upscaler, `previous` and
`current` `[N, 3, H, W]` each for the two-frame ones (two inputs, never one stacked six
channel tensor: the client would pay a copy and the graph a `Slice` for it), `output`
`[N, 3, H, W]` or twice it for the upscaler.

```
uv run src/mock/make_mock_models.py                 # into ../desktop-streamer, checked out beside this repo
uv run src/mock/make_mock_models.py <models folder> # or anywhere else
```

## Setup

`uv` handles the interpreter and the dependencies; Python 3.14 is pinned in
`pyproject.toml`, so nothing needs to be installed first.

```
uv sync
```

The CUDA index declared in `pyproject.toml` is not wired to the `torch` requirement, so
torch comes from PyPI, and what that is depends on the platform: on Linux (and WSL) the
PyPI wheel is a **CUDA build** (`2.13.0+cu130`), which is what every notebook from 4 on
trained with, on an RTX 3090; on Windows it is a **CPU build**, which is what notebooks 1–3
ran on. `onnxruntime` is the CPU package either way, so every `cpu_ms` a notebook writes
into a graph is a CPU number. GPU numbers for a graph come from the benchmark page.

## Running the benchmark server

The upscaling models are meant to run in a browser through ONNX Runtime Web, so the
measurement that decides anything is a browser measurement. `src/upscale/benchmark/` is a small
FastAPI app that serves the page which takes it.

**1. Publish the models.** **One notebook is one model.** Publishing one is writing its
`.onnx` into `dist/upscale/` and nothing else — **every model notebook ends
with its own export cell** that does exactly that. There is no staging folder and no
manifest: the server builds the dropdown out of the graphs in that folder on every request,
so dropping a file in or deleting one is the whole operation, no restart and no re-export of
anything else. Run whichever notebooks you want on the page:

- `src/upscale/00_data_preprocess.ipynb` — builds the patch dataset into `datasets/upscale/`,
  and has to run first, after `uv run datasets/screens.py` has drawn the desktop screens it
  mixes in with the photos and video. With no dataset downloaded and nothing in
  `datasets/raw/captures/` it trains on synthesised desktop-like frames, so it works before
  any real data exists. It is not a model and exports nothing.
- `src/upscale/01_upscale_dummy.ipynb` — the static baseline: the three filters ONNX can express
  as a `Resize` node, nearest, bilinear and bicubic. The bar a learned model must beat.
- `src/upscale/02_upscale_nn.ipynb` — a plain CNN that makes the whole picture itself, with
  no bicubic shortcut.
- `src/upscale/03_upscale_web.ipynb` — the same model shaped for the browser, published at three
  precisions and two tile sizes.
- `src/upscale/04_upscale_strategies.ipynb` — the strategies worth keeping (the 32-16 control,
  its luma-only twin and notebook 7's residual blocks) under notebook 3's recipe on the
  stream-degraded data, judged against the spread of the control at three seeds, publishing
  the best two at the 320×180 step.
- `src/upscale/05_upscale_geometry.ipynb` — not a model: the *shape* one runs at. Sweeps tile size,
  batch and scale factor, charges everything in nanoseconds per output pixel, and publishes
  a family of geometries for the page to measure on whatever GPU is in front of it.
- `src/upscale/06_upscale_reparam.ipynb` — the reparameterized 32-16 the first strategy sweep
  picked, trained on the stream-degraded data and published at the 320×180 step a client
  runs, with bilinear and bicubic beside it - the page's 320×180 baselines. Notebook 4 no
  longer supports that pick.
- `src/upscale/07_upscale_luma.ipynb` — a residual model that corrects brightness only (colour
  stays bicubic), with residual blocks, published at the same 320×180 step.
- `src/upscale/08_upscale_screens.ipynb` — notebook 7's model trained with desktop screens in
  the data, blind and told how the stream was made (its bitrate, and the host's encoder path),
  scored on screens, photos and video apart; published at the 320×180 step.
- `src/upscale/09_upscale_dct.ipynb` — a model that only guesses gains: the DCT of overlapping
  luma blocks, one learned gain per coefficient (fixed, or decided per block by a tiny
  network), and a fixed windowed IDCT; colour stays bicubic. Published at the 320×180 step.
- `src/upscale/10_upscale_lut.ipynb` — look-up tables instead of arithmetic (SR-LUT, MuLUT): a
  network that reads four luma pixels trains as itself, answers every question it can be asked
  into a table of megabytes, and is thrown away; the model reads five corners of a table cell
  per look-up, in four rotations, through one to three tables and one or two stages. Colour
  stays bicubic. Published at the 320×180 step.
- `src/upscale/11_upscale_fastlut.ipynb` — look-up tables read at one exact address, with no
  interpolation: a table addressed by the centre and binned differences from it, and a 3×3
  sign pattern that looks up a filter. Exports float32 only, at the 320×180 step.
- `src/upscale/12_upscale_dirnet.ipynb` — a directional upscaler: four learned 1D
  interpolators per sub-pixel (row, column, two diagonals), chosen per pixel by a tiny
  network. Three variants, from one fixed 5×5 filter to a blend with a learned correction;
  published at the 320×180 step.
- `src/upscale/13_upscale_hints.ipynb` — not a model: what a few bits from the host's
  full-resolution frame are worth - a content label to route on, or filters the host fits on
  the client's output. Trains nothing and exports nothing.
- `src/upscale/14_upscale_postfilter.ipynb` — notebook 13's post-filter on whole streamed
  clips, refitted on a schedule, against a plain stream at the same total bits. Exports
  nothing.
- `src/upscale/15_upscale_tradeoff.ipynb` — every published client and the hint in one
  currency, bandwidth: what bitrate plain bicubic would need for the same quality. Exports
  nothing.

The code the notebooks share lives in `src/upscale/helpers/`: `models.py` (the bicubic skip
and the models built on it), `training.py` (loading the patches, the training loop, scoring,
checkpoints), `speed.py` (timing), `postfilter.py` (the host-fitted post-filter of
notebooks 13–15), `metrics.py` (**how a frame is scored**, in Y'CbCr with
luma weighted six times either chroma channel, because that is the ratio an eye reads them
at), `tiles.py` (the tile geometry: how a frame is cut into tiles and what each costs) and
`webexport.py` (the export call, and the one place the tile step and the WebGPU operator
budget are written down — `main.py` imports it rather than keeping a second copy).

Almost everything the dropdown shows is read back out of the graph: the input and output
shapes, the operators, the parameter count, the size on disk, and the halo, which is
whatever the model input carries beyond the step it contributes. The two things a graph
cannot state — the label a human chose, and the desktop CPU time the notebook measured — are
written into the model's own `metadata_props`, so a model is one file that can be copied or
renamed without losing anything. A model exported by something else, with no metadata at
all, still lists correctly: the id and label fall back to the filename.

`dist/` is committed: the models in it took long to train, so a fresh clone has them and
the page lists them without running a notebook. Re-running an export cell overwrites the
file it owns, and the change shows up in git like any other. The server serves `dist/upscale/` under the page's `models/` path, so the page reads the finished
models where they are rather than from a copy.

The dependency that runs notebooks is `ipykernel`, not a frontend: open them in VS Code and
pick `.venv` as the kernel, or point whatever Jupyter you already have at that
interpreter. Add `jupyterlab` to the project if you want one of your own.

**2. Start the server.** It needs `fastapi[standard]` — the `[standard]` extra is what
pulls in uvicorn, without which there is nothing to serve with.

```
uv run src/upscale/benchmark/main.py
```

Then open <http://127.0.0.1:8000>. Options:

| Flag | Default | |
| --- | --- | --- |
| `-p`, `--port` | `8000` | |
| `--host` | `127.0.0.1` | `0.0.0.0` to measure from another machine on the network |
| `--models` | `dist/upscale/` | another folder of graphs - `dist/frame_gen_extra/` for the extrapolators, whose two inputs (`previous`, `current`) the page fills like the one an upscaler has |
| `--results` | none | append every measurement the page posts to this file, one JSON line each |

Opened as `/?auto=<id>,<id>,...` (or `?auto=all`), with `backend`, `data`, `mode` and `seconds`
in the query as the dropdowns would set them, the page measures those models one after
another with no one at the controls, posts each row - or the error that stopped it - to
the server, and posts `{"done": true}` last. That is how a headless browser takes a
measurement. Chrome on Windows, from WSL, against a server on port 8765:

```
"C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --enable-unsafe-webgpu
    --user-data-dir=%TEMP%\bench-profile "http://localhost:8765/?auto=all&backend=webgpu&data=gpu&mode=reuse"
```

Each posted row carries the WebGPU adapter it ran on: check that it names the real GPU and not
a software fallback.

`[standard]` also installs the `fastapi` CLI, so `uv run fastapi run
src/upscale/benchmark/main.py` works as well — with two differences worth knowing. It binds
`0.0.0.0` rather than localhost, and on a Windows console that is not UTF-8 its startup
banner dies with a `UnicodeEncodeError` before the server ever starts:

```
set PYTHONIOENCODING=utf-8
uv run fastapi run src/upscale/benchmark/main.py
```

The script above avoids both, which is why it is the documented way in.

The server only serves `www/` and `dist/upscale/` (or `--models`). It exists because the page cannot run from `file://` — ONNX
Runtime Web fetches its own `.wasm` at runtime and that fetch is blocked from a file origin.
It sets `Cache-Control: no-store` on everything, because a benchmark that re-runs the same
model would otherwise time the browser cache, and it registers the MIME types for `.wasm`,
`.onnx` and `.js` explicitly rather than trusting the Windows registry, where `.js` has been
known to come back as `text/plain`.

**3. Run it.** Pick a model, a backend, and where the tile lives, then press *Run
benchmark*. It times one tile through the model and adds a row. **Speed only — nothing on
the page looks at the picture.**

Three things it deliberately does:

- **The tile stays in GPU memory for a WebGPU run.** Handing the runtime a JS array
  uploads the tile and reads the result back every single run, and that copy costs the same
  whatever the model does — enough to make a one-node `Resize` and a convolution stack
  report the same time. *CPU round trip* in the **Data** menu measures it the old way, and
  the gap between the two settings is the copy.
- **A sample is a batch of runs timed together and divided, and the batch is sized from a
  probe.** A 140×140 tile finishes faster than `performance.now()` resolves, so runs have
  to be batched; the size cannot be fixed, because settling a batch costs one fence and a
  fence is ~3.4 ms of latency whatever it waits for. Each batch is sized to fill ~200 ms,
  which leaves its fence under 2% of the sample.
- **The backend column reports the provider that resolved, not the one requested.** Asking
  for WebGPU with a WASM fallback lets the runtime switch silently, and a row that cannot
  say which backend produced it is not a result — so *auto* tries WebGPU alone first and
  reports what it got. The data column resolves the same way: WASM has no GPU buffers.

Everything is charged per output pixel and there is no frames-per-second column: a run is a
tile, so a rate of runs prices the geometry a graph was exported at rather than the model in
it, and two tile sizes cannot be compared by it. `ns / kept px` is a run over the output
pixels it keeps; `ns / frame px` is the same with the tiling put back, including whatever a
step computes off the edge of a 1920×1080 frame. The two are equal only for a geometry that
tiles the frame exactly. `runs / frame` divides by the batch the graph was exported at, and
`1080p frame` assumes every run costs the same and that nothing happens between them: a
floor, not a frame rate.

ONNX Runtime Web is loaded from a CDN, so the first load needs a network. Everything else is
local.

## Layout

```
desktop-streamer-ai/
├── pyproject.toml              deps and the pinned interpreter
├── datasets/                   gitignored but for the script and the mix — DATASETS_DIR moves it
│   ├── download.py             uv run datasets/download.py --list; --dest for another drive
│   ├── screens.py              uv run datasets/screens.py: desktop screens drawn in real fonts
│   ├── yt8m-mix.txt            the YouTube-8M category mix yt8m-videos fetches by default
│   ├── raw/                    the downloads: DIV2K, Flickr2K, Set5…, UVG, YouTube-8M + its videos
│   │   ├── <name>/             one folder per dataset, with its train.txt and test.txt
│   │   ├── screens/            what screens.py draws, train/ and test/, and screens.json
│   │   └── captures/           screenshots of your own
│   ├── upscale/                00_data_preprocess.ipynb's patch pairs, manifest.json, synthetic/
│   └── frame_gen_extra/        its 00_data_preprocess.ipynb's 60 fps frame runs, manifest.json, synthetic/
├── src/                        the source: code, notebooks, and what training leaves behind
│   ├── mock/                   make_mock_models.py, the three stand-in graphs the client runs
│   ├── frame_gen_intra/        summary.md only, nothing built yet
│   ├── frame_gen_extra/        the upscale layout, six notebooks in
│   │   ├── summary.md                the problem, the data, the baselines
│   │   ├── 00_data_preprocess.ipynb  still screens animated at 60 fps as a desktop in use, streamed, cut into runs
│   │   ├── 01_extrapolate_dummy.ipynb  repeat, linear and motion-shift extrapolation: the bar to beat
│   │   ├── 02_extrapolate_twoscale.ipynb  current + a correction made at 1/2 and 1/4 resolution, and its control
│   │   ├── 03_extrapolate_motion.ipynb  motion instead of pixels: a learned head, a tile search, and both
│   │   ├── 04_extrapolate_oracle.ipynb  whole frames in the client's tiles, and the most a learned picker could add
│   │   ├── 05_extrapolate_hint.ipynb  the host's motion hint for the frame it does not send - shelved: no host captures it
│   │   └── helpers/                  sources, degrade, metrics copied from upscale; scenes.py (the
│   │                                 desktop motion), runs.py, clips.py (whole frames, cut as the client
│   │                                 cuts them), models.py, training.py, speed.py, show.py (motion
│   │                                 pictures), hints.py (the host's hint), webexport.py
│   └── upscale/
│       ├── summary.md                the models, the numbers, and what they do not say
│       ├── 00_data_preprocess.ipynb  frames → LR/HR patch clips, a halo around the LR side
│       ├── 01_upscale_dummy.ipynb    static resampling filters, the bar to beat
│       ├── 02_upscale_nn.ipynb       a first PyTorch model: a plain CNN, no bicubic shortcut
│       ├── 03_upscale_web.ipynb      the same model, shaped for the browser
│       ├── 04_upscale_strategies.ipynb  three strategies, one recipe, judged against seed noise
│       ├── 05_upscale_geometry.ipynb  the shape to run one at: tile, batch, scale, ns per pixel
│       ├── 06_upscale_reparam.ipynb  the selected model, trained on the stream, published at 320x180
│       ├── 07_upscale_luma.ipynb     residual blocks that correct brightness only; colour stays bicubic
│       ├── 08_upscale_screens.ipynb  the same, trained with desktop screens, blind and told the stream
│       ├── 09_upscale_dct.ipynb      gains on the DCT of overlapping luma blocks; colour stays bicubic
│       ├── 10_upscale_lut.ipynb      look-up tables: megabytes of stored answers, a few hundred ops a pixel
│       ├── 11_upscale_fastlut.ipynb  look-up tables read at one exact address, no interpolation
│       ├── 12_upscale_dirnet.ipynb   four 1D interpolators per pixel, and a network choosing between them
│       ├── 13_upscale_hints.ipynb    what bits from the host's full-resolution frame are worth
│       ├── 14_upscale_postfilter.ipynb  the host-fitted post-filter on whole streamed clips
│       ├── 15_upscale_tradeoff.ipynb  bits, compute and quality in one currency: bandwidth
│       ├── helpers/                  modules the notebooks import, never run on their own
│       │   ├── sources.py            reads the downloaded datasets back
│       │   ├── motion.py             a still image as a moving clip: slide, zoom or hold
│       │   ├── degrade.py            the stream's degradation: downscale, 4:2:0, H.264
│       │   ├── patches.py            reads the patch set back for the model notebooks
│       │   ├── models.py             the bicubic skip and the models built on it
│       │   ├── training.py           loaders, the training loop, scoring, checkpoints
│       │   ├── metrics.py            how a frame is scored: Y'CbCr, luma weighted 6:1:1
│       │   ├── speed.py              timing and multiply-accumulate counts
│       │   ├── postfilter.py         the host-fitted post-filter: classes, fits, bits
│       │   ├── tiles.py              the tiling geometry: step, halo, kept, which tile to run
│       │   └── webexport.py          the export call, the tile step and the op budget
│       ├── data/                     gitignored — samples the notebooks write
│       ├── checkpoints/              gitignored — trained weights and intermediate graphs
│       └── benchmark/
│           ├── main.py             the FastAPI server described above
│           └── www/                the page
└── dist/                       committed — the finished models, where every export lands
    ├── upscale/                *.onnx, served to the page as models/
    └── frame_gen_extra/        *.onnx with extrapolate.onnx's two-input contract; extrapolate2x_tilesearch_r4_s2.onnx is the client's
```
