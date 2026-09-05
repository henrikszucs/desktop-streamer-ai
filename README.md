# model/

The video work for Desktop Streamer: upscaling a received frame, and generating frames
between or after the ones that arrive. A `uv`-managed Python project, separate from the Node
application — nothing in `src/` calls it yet.

Three problems, one folder each, each with its own `summary.md`:

| Folder | Problem | State |
| --- | --- | --- |
| [`upscale/`](upscale/summary.md) | more pixels out than in | a static baseline, three learned models and a strategy sweep |
| [`frame_gen_intra/`](frame_gen_intra/summary.md) | interpolate between two frames | not started |
| [`frame_gen_extra/`](frame_gen_extra/summary.md) | extrapolate past the newest frame | not started |

## Setup

`uv` handles the interpreter and the dependencies; Python 3.14 is pinned in
`pyproject.toml`, so nothing needs to be installed first.

```
cd model
uv sync
```

Note that torch installs as a **CPU build** despite the CUDA index declared in
`pyproject.toml` — the index is there but not wired to the `torch` requirement, so every
number recorded so far is a CPU number.

## Running the benchmark server

The upscaling models are meant to run in a browser through ONNX Runtime Web, so the
measurement that decides anything is a browser measurement. `upscale/benchmark/` is a small
FastAPI app that serves the page which takes it.

**1. Publish the models.** **One notebook is one model.** Publishing one is writing its
`.onnx` into `upscale/benchmark/www/models/` and nothing else — **every model notebook ends
with its own export cell** that does exactly that. There is no staging folder and no
manifest: the server builds the dropdown out of the graphs in that folder on every request,
so dropping a file in or deleting one is the whole operation, no restart and no re-export of
anything else. Run whichever notebooks you want on the page:

- `upscale/data_preprocess.ipynb` — builds the patch dataset, and has to run first. With
  `data/raw/` empty it synthesises desktop-like frames, so it works before any real data
  exists. It is not a model and exports nothing.
- `upscale/upscale_dummy.ipynb` — the static baseline: the three filters ONNX can express
  as a `Resize` node, nearest, bilinear and bicubic. The bar a learned model must beat.
- `upscale/upscale_nn.ipynb` — the bicubic-residual model.
- `upscale/upscale_web.ipynb` — the same model shaped for the browser, published at three
  precisions and two tile sizes.
- `upscale/upscale_strategies.ipynb` — six architectures under one recipe, scored by the
  eye-weighted metric, publishing the two that win.

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

`benchmark/www/models/` is gitignored, so a fresh clone starts empty and the page will say
so.

The dependency that runs notebooks is `ipykernel`, not a frontend: open them in VS Code and
pick `model/.venv` as the kernel, or point whatever Jupyter you already have at that
interpreter. Add `jupyterlab` to the project if you want one of your own.

**2. Start the server.** It needs `fastapi[standard]` — the `[standard]` extra is what
pulls in uvicorn, without which there is nothing to serve with.

```
uv run upscale/benchmark/main.py
```

Then open <http://127.0.0.1:8000>. Options:

| Flag | Default | |
| --- | --- | --- |
| `-p`, `--port` | `8000` | |
| `--host` | `127.0.0.1` | `0.0.0.0` to measure from another machine on the network |

`[standard]` also installs the `fastapi` CLI, so `uv run fastapi run
upscale/benchmark/main.py` works as well — with two differences worth knowing. It binds
`0.0.0.0` rather than localhost, and on a Windows console that is not UTF-8 its startup
banner dies with a `UnicodeEncodeError` before the server ever starts:

```
set PYTHONIOENCODING=utf-8
uv run fastapi run upscale/benchmark/main.py
```

The script above avoids both, which is why it is the documented way in.

The server only serves `www/`. It exists because the page cannot run from `file://` — ONNX
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

The `1080p frame` column is the median multiplied by the 40 tiles a 1920×1080 output takes
at a 128 pixel step. It assumes every tile costs the same and that nothing happens between
them: a floor, not a frame rate.

ONNX Runtime Web is loaded from a CDN, so the first load needs a network. Everything else is
local.

## Layout

```
model/
├── pyproject.toml              deps and the pinned interpreter
├── frame_gen_intra/            summary.md only, nothing built yet
├── frame_gen_extra/            summary.md only, nothing built yet
└── upscale/
    ├── summary.md              the models, the numbers, and what they do not say
    ├── data_preprocess.ipynb   frames  → LR/HR patch pairs
    ├── upscale_dummy.ipynb     static resampling filters, the bar to beat
    ├── upscale_nn.ipynb        a first PyTorch model (and what not to export)
    ├── upscale_web.ipynb       the same model, shaped for the browser
    ├── upscale_strategies.ipynb  six architectures, one recipe, one eye-weighted metric
    ├── metrics.py              how a frame is scored: Y'CbCr, luma weighted 6:1:1
    ├── webexport.py            the export call, tile step and op budget, shared
    ├── data/                   gitignored — dataset and samples
    ├── checkpoints/            gitignored — trained weights
    └── benchmark/
        ├── main.py             the FastAPI server described above
        └── www/                the page, and gitignored models/ beside it — where
                                every export lands
```
