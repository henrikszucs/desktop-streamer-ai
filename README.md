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
[`datasets/download.py`](datasets/download.py), and beside it one folder per problem for what
that problem's preprocessing makes of them (`datasets/upscale/`).

Three problems, one folder each under `src/`, each with its own `summary.md` - and beside
them `src/mock/`, which is not a problem but the three stand-ins the client runs today (see
below):

| Folder | Problem | State |
| --- | --- | --- |
| [`src/upscale/`](src/upscale/summary.md) | more pixels out than in | a static baseline, three learned models and a strategy sweep |
| [`src/frame_gen_intra/`](src/frame_gen_intra/summary.md) | interpolate between two frames | not started |
| [`src/frame_gen_extra/`](src/frame_gen_extra/summary.md) | extrapolate past the newest frame | not started |

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
geometry `src/upscale/webexport.py` measured), runs every tile of a frame as one batch, and
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

Note that torch installs as a **CPU build** despite the CUDA index declared in
`pyproject.toml` — the index is there but not wired to the `torch` requirement, so every
number recorded so far is a CPU number.

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

- `src/upscale/data_preprocess.ipynb` — builds the patch dataset into `datasets/upscale/`,
  and has to run first. With no dataset downloaded and nothing in `datasets/raw/captures/`
  it trains on synthesised desktop-like frames, so it works before any real data exists. It
  is not a model and exports nothing.
- `src/upscale/upscale_dummy.ipynb` — the static baseline: the three filters ONNX can express
  as a `Resize` node, nearest, bilinear and bicubic. The bar a learned model must beat.
- `src/upscale/upscale_nn.ipynb` — the bicubic-residual model.
- `src/upscale/upscale_web.ipynb` — the same model shaped for the browser, published at three
  precisions and two tile sizes.
- `src/upscale/upscale_strategies.ipynb` — six architectures under one recipe, scored by the
  eye-weighted metric, publishing the two that win.
- `src/upscale/upscale_geometry.ipynb` — not a model: the *shape* one runs at. Sweeps tile size,
  batch and scale factor, charges everything in nanoseconds per output pixel, and publishes
  a family of geometries for the page to measure on whatever GPU is in front of it.

`webexport.py` beside them is the export call they share, and the one place the tile step
and the WebGPU operator budget are written down — `main.py` imports it rather than keeping a
second copy of either. `metrics.py` is the other shared file: **how a frame is scored**, in
Y'CbCr with luma weighted six times either chroma channel, because that is the ratio an eye
reads them at. Every notebook that measures quality should be measuring through it.

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

`[standard]` also installs the `fastapi` CLI, so `uv run fastapi run
src/upscale/benchmark/main.py` works as well — with two differences worth knowing. It binds
`0.0.0.0` rather than localhost, and on a Windows console that is not UTF-8 its startup
banner dies with a `UnicodeEncodeError` before the server ever starts:

```
set PYTHONIOENCODING=utf-8
uv run fastapi run src/upscale/benchmark/main.py
```

The script above avoids both, which is why it is the documented way in.

The server only serves `www/` and `dist/upscale/`. It exists because the page cannot run from `file://` — ONNX
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
│   ├── raw/                    the downloads: DIV2K, Flickr2K, Set5…, UVG, YouTube-8M + its videos
│   │   ├── <name>/             one folder per dataset, with its train.txt and test.txt
│   │   ├── yt8m-videos/mix.txt committed — the YouTube category mix
│   │   └── captures/           screenshots of your own
│   └── upscale/                data_preprocess.ipynb's patch pairs, manifest.json, synthetic/
├── src/                        the source: code, notebooks, and what training leaves behind
│   ├── mock/                   make_mock_models.py, the three stand-in graphs the client runs
│   ├── frame_gen_intra/        summary.md only, nothing built yet
│   ├── frame_gen_extra/        summary.md only, nothing built yet
│   └── upscale/
│       ├── summary.md              the models, the numbers, and what they do not say
│       ├── data_preprocess.ipynb   frames  → LR/HR patch pairs
│       ├── upscale_dummy.ipynb     static resampling filters, the bar to beat
│       ├── upscale_nn.ipynb        a first PyTorch model (and what not to export)
│       ├── upscale_web.ipynb       the same model, shaped for the browser
│       ├── upscale_strategies.ipynb  six architectures, one recipe, one eye-weighted metric
│       ├── upscale_geometry.ipynb  the shape to run one at: tile, batch, scale, ns per pixel
│       ├── metrics.py              how a frame is scored: Y'CbCr, luma weighted 6:1:1
│       ├── sources.py              reads the downloaded datasets back
│       ├── degrade.py              the stream's degradation: downscale, 4:2:0, H.264
│       ├── webexport.py            the export call, the tiling geometry and the op budget
│       ├── data/                   gitignored — samples the notebooks write
│       ├── checkpoints/            gitignored — trained weights and intermediate graphs
│       └── benchmark/
│           ├── main.py             the FastAPI server described above
│           └── www/                the page
└── dist/                       committed — the finished models, where every export lands
    └── upscale/                *.onnx, served to the page as models/
```
