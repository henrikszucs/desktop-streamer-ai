# Why the upscaler is slow on a GTX 1660 Ti - a plan to run on that machine

Frame generation keeps up on the GTX 1660 Ti, but with the upscaler on the client shows no
more than about 25 fps "at 720p". This file is the investigation, written to be carried out
on that machine: the patches to apply, the runs to make in order, what to write down, and
what each result points at. Fill in the tables as you go; they are what comes back.

Two repositories, checked out side by side: **desktop-streamer** (the client, where the
enhancer runs) and **desktop-streamer-ai** (this one: the graph and the benchmark page).

## What a frame costs today

With only upscale on, each decoded frame goes through these steps in
`desktop-streamer/src/client/web/src/room/stream-enhance.js`, strictly one after another:

1. **`fromFrame`**: a compute pass cuts the decoded frame into one float32 batch of
   328×188 tiles (a 320×180 step plus a 4 px halo) on ONNX Runtime's GPU device.
2. **`engine.run`**: notebook 23's told 24×2 graph (`media/models/upscale.onnx`) runs over
   every tile in one batch. Its `condition` input is a 3-number CPU tensor.
3. **`toFrame`**: a render pass draws the result at 2× onto the enhancer's own canvas,
   wrapped as `new VideoFrame(canvas)`.
4. **`present`**: the drawer (`stream-draw.js`) draws that frame again, **on a second
   WebGPU device of its own**.
5. **`await engine.sync()`**: `queue.onSubmittedWorkDone()`. The next frame does not start
   until this resolves, and only one frame may wait (`PENDING_MAX = 1`).

So **frames per second = 1000 / (all five steps plus the GPU wait)**. 25 fps means about 40 ms.

The graph costs about 13,900 multiply-adds per input pixel (halo included):

| Stream → output | Tiles | GFLOP a frame | One 24-channel activation | ORT's GPU buffer cache |
|---|---|---|---|---|
| 640×360 → 1280×720 | 4 | 6.9 | 23.7 MB | reused |
| 960×540 → 1920×1080 | 9 | 15.5 | 53.3 MB | reused |
| **1280×720 → 2560×1440** | **16** | **27.5** | **94.7 MB**, rounded up to the 128 MiB size class, which keeps at most 6 spare buffers | **partly reused** |
| 1920×1080 → 3840×2160 | 36 | 61.9 | 213 MB | **never**: ORT 1.29's cache stops at 160 MB, so these are destroyed and created again every run |

What is already known, all from the RTX 3090:
- The client took **9.4 ms at 9 tiles and 70 ms at 36** (desktop-streamer's
  `.claude/CLIENT.md`, "The enhancer"). That is 4× the tiles for 7.4× the time, which fits
  the buffer-cache limit above.
- On the benchmark page, the same architecture (notebook 8's graph, fp32, graph capture)
  took **0.646 ms per tile**.
- The mock measurements in CLIENT.md found that ORT's general `Conv` has no fast path when
  the channel count isn't a multiple of 4. The trained graph's stem (3→24, 3×3) and bicubic
  skip (3→12, 5×5) both read 3 channels.
- The same notes put ORT's NCHW↔NHWC layout conversions at about 4.3 ms per graph on a
  1080p frame.

Scaling the 3090's 9-tile reading to the 1660 Ti by memory bandwidth (936 → 288 GB/s)
gives roughly **13 ms for 4 tiles and 45–55 ms for 16**. So:
- **If the stream is 1280×720** (16 tiles), 40 ms is about what the model should cost on
  this card. The fixes are in the model (step "B" under Fixes).
- **If the output is 1280×720** (a 640×360 stream, 4 tiles), 40 ms is about 3× too slow.
  The fixes are in the code around the model (step "A").

Run 1 settles which one it is.

## Setup

### Where edits take effect

The web client is served from a minified build in `desktop-streamer/tmp/web`, which the
server makes once and then reuses. **Every code edit or model swap under
`src/client/web/` needs a rebuild:**

```
cd desktop-streamer
npm run server -- --compile
```

Make the edits on the machine that runs the server. If that isn't the 1660 Ti machine,
edit there and only watch from the 1660 Ti.

### The browser

Use Chrome, not the Electron client: you need its command-line flags and DevTools. Start a
separate profile so the flags apply even with Chrome already open (Windows `cmd`):

```
"C:\Program Files\Google\Chrome\Application\chrome.exe" --user-data-dir=%TEMP%\enhance-debug --enable-webgpu-developer-features https://<server>/
```

`--enable-webgpu-developer-features` makes the GPU's timestamps precise; without it Chrome
rounds them to 100 µs. Accept the server's certificate in this profile if it is
self-signed. The stream worker's `console.log` output shows in the page's DevTools
console.

**On a laptop:** in Windows Settings → System → Display → Graphics, set `chrome.exe` to
*High performance*. Then check `chrome://gpu`: the active GPU must be the GTX 1660 Ti.

### Fix the stream

- Set a fixed stream height in the room settings, not `auto`. `auto` changes the
  resolution with the bandwidth, and every change resets the measurements.
- Put a playing video in the shared window, so the host sends a steady frame rate. The
  upscaler's cost doesn't depend on what is on screen, since it runs every tile.
- Let each run settle for 10 s, then copy about 10 lines of `[stats]` from the console.

### Debug patches (never commit these)

All are in `desktop-streamer/src/client/web/src/room/`. Revert them at the end with
`git checkout -- src/client/web/src/room/` and rebuild.

**P1. Log the stats every second: `stream-worker.js`, in `startStats()`.** Nothing in
the UI shows the enhancer's stats today. Replace the `post({"type": "stats", ...})` call
with:

```js
const message = {
    "type": "stats",
    "fps": decodedCount,
    "decoded": decodedCount,
    "dropped": droppedCount,
    "drawer": drawer?.["name"] ?? "",
    "enhance": (enhancer?.getStats() ?? null),
    "qp": (qpCount > 0 ? Math.round(qpSum / qpCount * 10) / 10 : null)
};
console.log("[stats]", lastWidth + "x" + lastHeight, JSON.stringify(message));    // DEBUG
post(message);
```

How to read it:
- `fps` counts **decoded** frames, which is what the host sends, not what is shown.
- With only upscale on, `enhance.runs` is the number of frames **shown** per second.
- `enhance.ms` is each frame's time from arrival to the GPU finishing.
- `enhance.dropped` is frames the enhancer had no time for.

**P2. Switches, per-stage timing and per-operation GPU time: `stream-enhance.js`.**

(a) In `createEnhancer`, right after `ort.env.logLevel = "warning";`:

```js
// DEBUG - bottleneck plan, never commit
const DEBUG = {"profile": false, "highPerformance": false, "nchw": false, "bufferLog": false};
if (DEBUG.highPerformance === true) {
    ort.env.webgpu.powerPreference = "high-performance";
}
if (DEBUG.bufferLog === true) {
    ort.env.debug = true;
    ort.env.logLevel = "verbose";
}
const kernelNs = new Map();
if (DEBUG.profile === true) {
    ort.env.webgpu.profiling = {"mode": "default", "ondata": function(record) {
        const key = record.kernelType + " " + (record.kernelName ?? record.kernelId);
        kernelNs.set(key, (kernelNs.get(key) ?? 0) + Number(record.endTime) - Number(record.startTime));
    }};
}
const stages = {"fromFrame": 0, "run": 0, "toFrame": 0, "present": 0, "sync": 0};
```

(b) In `sessionOptions`, replace `"executionProviders": [backend],` with:

```js
"executionProviders": [(DEBUG.nchw === true && backend === "webgpu") ? {"name": "webgpu", "preferredLayout": "NCHW"} : backend],
```

(c) In `process()`, put a timer around each stage. Declare `let t0 = 0;` at the top of the
function, then:

```js
t0 = performance.now();
current = engine.fromFrame(frame);
stages.fromFrame += performance.now() - t0;
```
```js
// the upscale step, inside `if (options["upscale"] === true)`
t0 = performance.now();
picture = await engine.run(await loadSession("upscale", source["dims"]), feedsOf("upscale", [source]), MODELS["upscale"]["scale"], condition);
stages.run += performance.now() - t0;
```
```js
t0 = performance.now();
const out = engine.toFrame(picture, Math.round(timestamp + step["at"] * interval * 1000));
stages.toFrame += performance.now() - t0;
```
```js
t0 = performance.now();
present(out, presentDelay(step["at"], interval, anchor, now));
stages.present += performance.now() - t0;
```
```js
t0 = performance.now();
await engine.sync();
stages.sync += performance.now() - t0;
```

What each timer measures:
- `run` resolves when the work is *submitted*, so it is CPU time: ORT issuing the GPU
  work.
- `sync` is whatever GPU work is still unfinished, plus the wait for the GPU to report
  back.
- `present` includes the drawer's draw on its own device.

(d) In `getStats()`, before `runCount = 0;`:

```js
const n = Math.max(1, runCount);
stats["stages"] = Object.fromEntries(Object.entries(stages).map(([key, ms]) => [key, Math.round(ms / n * 10) / 10]));
for (const key of Object.keys(stages)) {
    stages[key] = 0;
}
if (kernelNs.size > 0) {
    stats["kernels"] = [...kernelNs].sort((a, b) => b[1] - a[1]).slice(0, 15)
        .map(([key, ns]) => key + " " + (ns / n / 1e6).toFixed(2) + " ms");
    kernelNs.clear();
}
```

**P3. Which GPU each device is on.** In `stream-enhance.js` `createGPUBackend`, after the
`device` check:

```js
console.log("[enhancer adapter]", device.adapterInfo?.vendor, device.adapterInfo?.architecture,
    device.adapterInfo?.description, "shader-f16:", device.features.has("shader-f16"),
    "timestamp-query:", device.features.has("timestamp-query"));
```

and in `stream-draw.js` `createWebGPUDrawer`, after `requestDevice()`:

```js
console.log("[drawer adapter]", adapter.info?.vendor, adapter.info?.architecture, adapter.info?.description);
```

## The runs, in order

Change one thing per run. Rebuild (`--compile`) after any edit, and hard-reload the page.

### Run 0: what the browser offers (no patch needed)

In the DevTools console of the client page:

```js
for (const pref of ["default", "high-performance", "low-power"]) {
    const a = await navigator.gpu.requestAdapter(pref === "default" ? {} : {"powerPreference": pref});
    console.log(pref, a?.info.vendor, a?.info.architecture, a?.info.description, "f16:", a?.features.has("shader-f16"));
}
```

Also:
- `chrome://gpu`: the active GPU.
- `chrome://media-internals`: open the stream's entry and confirm the decoder is hardware
  (D3D11VideoDecoder), not a software one.

| | Result |
|---|---|
| default adapter | nvidia, turing, NVIDIA GeForce GTX 1660 Ti (also for `high-performance` and `low-power`: Windows already sets chrome.exe to High performance) |
| high-performance adapter | the same GTX 1660 Ti |
| shader-f16 available | yes (also `timestamp-query`, `subgroups`); ORT 1.29.0's device has all three |
| decoder | not checked yet (needs a stream) |

The machine also has an AMD Radeon Vega 10 (integrated, driving the 1920×1080 laptop panel);
the 1660 Ti drives the 1680×1050 screen.

### Run 1: the baseline

Apply P1, P2 (all switches `false`) and P3. Turn on **upscale only**, at the stream size
you saw 25 fps at.

- `[stats]` shows the stream size: 1280x720 or 640x360. That answers the question under
  "What a frame costs today".
- `[enhancer adapter]` and `[drawer adapter]` must both name the GTX 1660 Ti. If either
  names the integrated GPU, fix that first (Graphics settings, or Run 6) and start again.

### Run 2: the same code around a free model

Two comparisons that leave the code untouched:

- **2a. Interpolate only, no upscale, same stream.** Frame generation goes through the
  same `fromFrame`, `toFrame`, `present` and `sync`, but at 1× instead of 2×.
- **2b. The mock upscaler.** Same input, same 2× output, almost no model. On the server
  machine:

  ```
  SCRATCH=$HOME/upscale-debug          # any folder outside both repositories
  cd desktop-streamer-ai
  uv run src/mock/make_mock_models.py "$SCRATCH" --upscale
  cp "$SCRATCH/upscale.onnx" ../desktop-streamer/src/client/web/media/models/upscale.onnx
  cd ../desktop-streamer && npm run server -- --compile
  ```

  After the run, restore the real graph:

  ```
  git checkout -- src/client/web/media/models/upscale.onnx
  npm run server -- --compile
  ```

**The real graph's time minus the mock's is the model.** The mock's time is everything
around it at 2×.

### Run 3: does the cost grow faster than the tiles?

Upscale only, at 640×360, 960×540 and 1280×720. Divide `enhance.ms` by the tile count
(4, 9, 16).
- If ms per tile stays flat, the cost is linear and the GPU is just doing the work.
- If it jumps between 9 and 16 tiles, suspect buffer churn and do Run 5.

### Run 4: GPU time per operation

Set `DEBUG.profile = true`. `[stats]` now carries `kernels`, the 15 most expensive
operations in ms per frame. Profiling adds its own overhead, so read `stages` from Runs 1
to 3, not from this run.

What to look for:
- Do the kernels add up to about `stages.sync + stages.run`? Yes means the GPU is the
  limit. Much less means the time goes elsewhere: the output path, the GPU wait, the second
  device.
- Is the stem `Conv` or the skip `Conv` (the two that read 3 channels) out of proportion?
  The skip has 900 multiply-adds per pixel against 5,184 for one 24→24 3×3 convolution.
  If it costs anywhere near as much, the 3-channel slow path is real.
- How much do `Transpose` and the elementwise operations (`Add`, `Concat`, `DepthToSpace`)
  cost against the `Conv`s?

### Run 5: buffer churn

Set `DEBUG.bufferLog = true` (and `profile` back to `false`). Wait 10 s, clear the console,
then filter it for `GpuDataManager.create`.
- Once warmed up, nothing should be created on every frame.
- Lines repeating every frame with sizes of about 94–213 MB confirm that ORT's buffer
  cache is too small for this batch.

Confirm it with a separate Chrome start that adds
`--disable-dawn-features=lazy_clear_resource_on_first_use`. This is a diagnostic only:
it skips zero-filling new GPU buffers. If `enhance.ms` drops, new buffers being created and
cleared was part of the cost.

### Run 6: ask ORT for the high-performance GPU

Set `DEBUG.highPerformance = true`. This only matters if Run 0 showed a different adapter
for `high-performance`, or if Run 1 found the enhancer on the integrated GPU.

### Run 7: ORT without the layout conversions

Set `DEBUG.nchw = true`. Compare `enhance.ms` and, with `profile` on, the `Transpose`
lines against Run 4.

### Run 8: the graph alone, on the benchmark page

This needs uv and desktop-streamer-ai on the machine running the benchmark server. The page
takes fixed-shape graphs only, so first write fixed-shape copies of the client's graph:

```
SCRATCH=$HOME/upscale-debug          # any folder outside both repositories
cd desktop-streamer-ai
uv run python - "$SCRATCH" <<'EOF'
import sys
from pathlib import Path
import onnx

src = Path("../desktop-streamer/src/client/web/media/models/upscale.onnx")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for n in (4, 9, 16):
    m = onnx.load(src)
    for value, dims in ((m.graph.input[0], (n, 3, 188, 328)), (m.graph.output[0], (n, 3, 376, 656))):
        for dim, size in zip(value.type.tensor_type.shape.dim, dims):
            dim.dim_value = size
    meta = {p.key: p for p in m.metadata_props}
    meta["label"].value = f"client upscale.onnx, {n} tiles"
    for key, val in (("step", "320"), ("step_h", "180"), ("batch", str(n))):
        if key in meta:
            meta[key].value = val
        else:
            m.metadata_props.add(key=key, value=val)
    onnx.checker.check_model(m)
    onnx.save(m, out / f"upscale_client_b{n}_tile320x180.onnx")
EOF
uv run src/upscale/benchmark/main.py --models "$SCRATCH" --results "$SCRATCH/results.jsonl"
```

Then open both of these in the debug Chrome:
- `http://localhost:8000/?auto=all&backend=webgpu&data=gpu&mode=reuse&seconds=5`
- the same URL with `mode=capture`

Check that each posted row names the GTX 1660 Ti as its adapter.

WebGPU only works on https or localhost. If the benchmark server runs on another machine,
either forward the port (`ssh -L 8000:localhost:8000 <server>`) or start the debug Chrome
with `--unsafely-treat-insecure-origin-as-secure=http://<server-ip>:8000`.

How to read it:
- The page's time per batch is the graph alone.
- `reuse` against `capture` is what graph capture would save.
- The client's `enhance.ms` minus this is what the client adds around the model.
- The 3090 reference is 0.646 ms per tile.

## Results

Measured 2026-11-12 on this machine: the server (desktop-streamer `feature/told-upscaler` at
6e8a993, with P1-P3) ran here, and so did the host. The host was a second Chrome profile
sharing a 16:9 tab with a 60 fps canvas animation through `getDisplayMedia`, so **host and
viewer shared the GTX 1660 Ti**. The host asked for 30 fps. Stage times are ms per shown
frame; every row is the median of 6-9 one-second readings.

| Run | Stream | Tiles | enhance.ms | runs/s | dropped/s | fromFrame | run | toFrame | present | sync | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 baseline | 1278×720 | 16 | 65.5 | 15 | 14 | 0.8 | 3.4 | 0.4 | 0.7 | 59.5 | both adapters are the GTX 1660 Ti |
| 1, a 1540×720 stream | 1540×720 | 20 | 75 | 13 | 17 | 0.7 | 3.5 | 0.4 | 0.6 | 70 | a non-16:9 source costs a column of tiles |
| 2a interpolate only | 1278×720 | 16 | 17 | 29 | 0 | 0.7 | 0 | 0.6 | 0.8 | 7.7 | keeps up |
| 2b mock upscaler | 1278×720 | 16 | 6.8 | 30 | 0 | 0.4 | 0.9 | 0.3 | 0.5 | 4.8 | everything around the model at 2× |
| 3 640×360 | 640×360 | 4 | 16 | 30 | 0 | 0.5 | 2.2 | 0.3 | 0.5 | 12.6 | keeps up; 4.0 ms per tile |
| 3 852×480 (480p) | 852×480 | 9 | 37.6 | 26.5 | 3.5 | 0.7 | 2.9 | 0.4 | 0.7 | 32.8 | 4.2 ms per tile |
| 3 1278×720 | 1278×720 | 16 | 65.5 | 15 | 14 | 0.8 | 3.4 | 0.4 | 0.7 | 59.5 | 4.1 ms per tile |
| 3 1872×1054 (1080p) | 1872×1054 | 36 | 215 | 4.6 | 25 | 0.6 | 3.2 | 0.4 | 0.5 | 212 | 6.0 ms per tile |
| 5 lazy clear off, 720p | 1278×720 | 16 | 60.4 | 16.5 | 13 | 0.6 | 3.1 | 0.3 | 0.5 | 55.8 | −5 ms |
| 5 lazy clear off, 1080p | 1872×1054 | 36 | 202 | 4.8 | 25 | 0.5 | 2.8 | 0.3 | 0.5 | 198 | −13 ms |
| 6 high-performance | | | | | | | | | | | not run: every adapter is already the 1660 Ti (Run 0) |
| 7 NCHW | | | | | | | | | | | not run in the client: no gain on the graph alone (Run 8) |

The resolution menu has 480p, not 960×540, and the client sizes the stream from the shared
surface. The tab had to be exactly 16:9 for 720p to come out as 1278×720 (16 tiles); at 1282
wide it is 20 tiles.

Run 4, the `kernels` list (paste one representative second):

```
```

Run 4 in the client, 1278×720 (16 tiles). Profiling was switched on inside the stream worker
over the DevTools protocol, without a rebuild: the worker's `import()` of ORT's URL returns
the enhancer's own module, and JSEP reads `env.webgpu.profiling` on every run. Average of
126 runs, 49.0 ms per frame:

```
10.78  Conv /Conv_1        3x3 24->24
10.52  Conv /Conv_2        3x3 24->24
10.08  Conv /tail/Conv     3x3 24->4
 4.98  Conv /skip/conv     5x5 3->12
 2.81  Conv /Conv (stem)   3x3 3->24
 2.30  Add /Add_1
 1.75  Conv /point_1/Conv  1x1
 1.69  Conv /point/Conv    1x1
 1.17  Add /Add
 0.77  Transpose (output)
 0.76  Concat
 0.65  DepthToSpace
 0.58  Add /Add_2
 0.14  Transpose (input)
```

The same kernels as the graph alone, about 15% slower each. The graph alone with the host
streaming beside it takes 50.9 ms (45.1 ms on an idle GPU), so that 15% is the host sharing
the GPU, not the client.

Run 4 on the graph alone instead (benchmark page, 16 tiles, ms per run, averaged over 10 runs,
ORT 1.29.0 default NHWC layout; total 41.7 ms):

```
 9.38  Conv /Conv_1        3x3 24->24   Conv2DMatMul
 9.34  Conv /Conv_2        3x3 24->24   Conv2DMatMul
 8.78  Conv /tail/Conv     3x3 24->4    Conv2DMatMul   <- a sixth of the work, the same time
 4.04  Conv /skip/conv     5x5 3->12    Conv2DMatMul
 2.26  Conv /Conv (stem)   3x3 3->24    Conv2DMatMul
 1.53  Conv /point_1/Conv  1x1 24->24   MatMul
 1.51  Conv /point/Conv    1x1 24->24   MatMul
 1.08  Add /Add
 1.08  Add /Add_1
 0.71  Transpose (NHWC -> NCHW, output)
 0.68  Concat /Concat
 0.59  DepthToSpace
 0.54  Add /Add_2
 0.14  Transpose (NCHW -> NHWC, input)
 0.04  the three condition 1x1 Convs and their Transposes
```

Run 5, `GpuDataManager.create` lines per second after warm-up, and their sizes:

```
```

`GpuDataManager.create` is logged for every allocation, whether it is served from ORT's free
list or not, so its rate says nothing about churn. Counting the device's own `createBuffer`
from inside the worker instead (1278×720, 16 tiles):

```
67 session runs in 5 s, 67 createBuffer calls, every one 134217728 bytes (the 128 MiB class)
```

**One 128 MiB buffer is created and destroyed on every frame.** That is the size class the
94.7 MB activations fall in, and ORT's cache keeps at most 6 free buffers of it. Which
allocation overflows it is not pinned down: the benchmark page's run of the same graph shows
no sign of churn, so it may be one the client adds, its output or input. Timed on the
enhancer's own device, a fresh 128 MiB buffer costs 11.7 ms to create, clear and use once,
against 0.9 ms for one already made. With
`--disable-dawn-features=lazy_clear_resource_on_first_use` a frame is 5 ms faster at 16 tiles
and 13 ms faster at 36, so about half of that cost is the zero-fill and the rest is the
creation. At 36 tiles every intermediate is above ORT's 160 MB cap and none is cached, which
is why the client takes 215 ms where the graph alone takes 113.

nvidia-smi during 1080p upscaling: 2.5-3.3 of 6 GB used, 75% GPU busy, power-capped at
about 60-70 W of 80 W (SM clock 1.7 GHz of 2.1). Memory is not the limit.

### Where a 720p frame goes (1278×720, 16 tiles, 65.5 ms)

| Part | ms | How it was found |
|---|---|---|
| the graph's kernels on an idle GPU | 45 | Run 8 |
| the host sharing the GPU (this setup only) | +4 | 49.0 in the client's kernels; 50.9 for the graph alone beside the host |
| the 128 MiB buffer remade every frame | ~10 | 11.7 ms per fresh buffer; −5 ms with the clear off |
| everything around the model at 2× | ~6 | Run 2b's mock, 6.8 ms including its own tiny graph |

So **about 70% of the frame is the model**, 15% is ORT's buffer cache, and 10% is the
client's own path. Run 3's per-tile cost (4.0-4.2 ms from 4 to 16 tiles, 6.0 at 36) fits the
same split. The ~25 fps reported at "720p" matches the 480p stream here (852×480, 9 tiles,
26.5 fps). At 1278×720 this setup shows 15 fps, with the host on the same GPU.

What that ranks, at 16 tiles (estimates from the measurements above, not yet measured in the
client):
1. **B2, float16:** the graph goes from 41 to 29 ms of GPU time. It also halves every
   activation, to 47 MB, which moves them out of the 128 MiB class and probably ends the
   churn as well.
2. **A1, the buffer churn:** about 10 ms a frame at 16 tiles and most of the 36-tile cost.
   Chunk the batch, or use graph capture (A4), which keeps its buffers.
3. **A retrained tail:** a 1×1 tail instead of 3×3 would save about 7 ms fp32 (the 3×3 tail
   costs as much as a 24→24 convolution).
4. B1 (4 input channels) and NCHW are measured losses here. A2 (overlapping frames) cannot
   help much while the GPU is the limit.

### After B2: the float16 graph in the client

Notebook 23 now also writes `dist/upscale/client/upscale_fp16.onnx`
(`webexport.to_float16(..., keep_io_types=True)`). It is the same graph with float16 weights
and arithmetic and a `Cast` at each end, so it takes and gives float32 like `upscale.onnx`,
and it is within 0.17/255 of it. desktop-streamer ships it beside `upscale.onnx` and fetches
it on WebGPU when the adapter has `shader-f16` (`urlOf()` in `stream-enhance.js`). Same setup
as the Results above (host on the same GPU, 30 fps asked):

| Stream | Tiles | float32 ms | float16 ms | runs/s float32 → float16 | float16 stages: run / sync |
|---|---|---|---|---|---|
| 640×360 | 4 | 16 | 12.8 | 30 → 30 | 2.2 / 8.9 |
| 852×480 | 9 | 37.6 | 24 | 26.5 → **30** | 2.7 / 20.2 |
| 1278×720 | 16 | 65.5 | 41 | 15 → **24** | 4.2 / 34.8 |
| 1872×1054 | 36 | 215 | 86 | 4.6 → **12** | 2.9 / 81 |

- **It also ends the buffer churn (A1).** Counting the device's `createBuffer` calls: none
  in 121 runs at 16 tiles and none in 58 runs at 36 (the float16 activations are 47 MB at
  16 tiles and 107 MB at 36). The 36-tile case is in the same 128 MiB class that churned in
  float32, so the cause is not only the activation size. What matters here is that it
  stops.
- **The kernels at 16 tiles: 34.0 ms**, against 49.0 in float32. The 3×3 24→24
  convolutions take 7.7 and 7.5 ms, the tail 6.7, the skip 3.1 and the stem 2.0.
- **What is left at 720p:** 41 ms is 34 of kernels and about 7 of everything around the
  model (Run 2b's mock path). Graph capture (A4) would save part of the 4 ms ORT spends
  issuing the work, and that issuing already overlaps the GPU, so it is not worth its
  constraints now. Chunking (A1) still matters for a GPU without `shader-f16`, which runs
  the float32 graph and churns from 16 tiles up.
- **The target is 60 fps with 20% to spare** (`reference_60fps.md`): 13.3 ms a frame on this
  card, 9.2 of it for the graph. That was measured with the host off this GPU. The told model
  meets it only for a 640×360 stream.
- **Next, if 720p has to reach 30 fps on this card:** the model has to come down another
  ~8 ms. The 3×3 tail is the clearest candidate: in float16 it is 6.7 ms for a sixth of
  a 24→24 convolution's work. A 1×1 tail would be a `MatMul` at about 1 ms, but it needs
  retraining.

Run 8, benchmark page (ms per batch):

| Graph | reuse | capture | GPU time (profiled) |
|---|---|---|---|
| 4 tiles | 11.38 | 11.26 | 11.10 (2.84 ms per tile) |
| 9 tiles | 25.53 | 25.27 | 25.12 (2.84 ms per tile) |
| 16 tiles | 45.07 | 45.05 | 44.47 (2.82 ms per tile) |
| 36 tiles | 112.83 | 101.55 | 105.77 (3.13 ms per tile) |

Measured 2026-11-12, Chrome 157, ORT Web 1.29.0, every row on the GTX 1660 Ti. The graph is
`dist/upscale/client/upscale.onnx`, identical to desktop-streamer's
`feature/told-upscaler` copy. The 1660 Ti is 4.4× slower per tile than the 3090's 0.646 ms.
- GPU time is almost all of the wall time, and the time per tile stays flat from 4 to 16
  tiles, so the graph is GPU-bound and linear up to 16 tiles.
- Graph capture saves nothing up to 16 tiles: ORT's issuing of the work is not the cost.
- At 36 tiles, `reuse` is 11 ms slower than `capture`, which fits ORT's buffer cache
  stopping at 160 MB (A1).

Run 7 on the graph alone (16 tiles, `preferredLayout: "NCHW"`): 42.5 ms against 41.7 ms. The
24→24 convolutions get slower (10.6 ms each), the tail faster (7.2 ms), and the skip slower
(5.2 ms). NCHW is no gain.

Candidate model fixes on the graph alone, made as equivalent edits of the told graph and
checked against it with ORT's CPU backend (16 tiles, profiled GPU ms):

| Variant | GPU ms | Output against the graph as it is |
|---|---|---|
| as it is | 41.3 | |
| tail writes 12 channels, no `Concat` | 41.1 | identical |
| 4 input channels (B1) | 42.8 | identical |
| float16 inside, float32 at both ends (B2) | **29.0** | max difference 5.8e-4 |
| all three | 29.7 | max difference 5.8e-4 |

- **B1 makes it worse** here: the 4-channel stem and skip are slower (2.66 and 4.82 ms) than
  the 3-channel ones. The "3-channel slow path" from the 3090 notes does not show on this
  GPU.
- **B2 saves 30%**: every Conv is about 30% faster, and the elementwise operations halve.
- **ORT's `Conv2DMatMul` costs about the same for any output width up to 24**: 3×3 over 24
  channels is about 9 ms whether it writes 4, 12 or 24 channels. The tail's cost is its
  3×3 over 24 input channels, not its output. Making the tail 1×1 (a retrain) would turn
  it into a `MatMul` at about 1.5 ms.

## What each result points at

| Finding | Where the time goes | Fix to try |
|---|---|---|
| An adapter isn't the 1660 Ti | the wrong GPU | Windows Graphics setting; `ort.env.webgpu.powerPreference` |
| Run 2b's mock is far below the real graph, and the kernels add up to the frame time | the model itself | **B** below |
| Run 2b's mock alone takes most of the frame | the code around the model at 2× | **A2**, **A3** |
| ms per tile jumps between 9 and 16 tiles, and Run 5 shows buffers created every frame | ORT's buffer cache | **A1** |
| `run` (CPU) is several ms, and Run 8's `capture` is much faster than `reuse` | ORT issuing the work | **A4** |
| `sync` is large while the kernels are small | the GPU wait, the second device | **A2**, **A3** |
| The 3-channel `Conv`s or the `Transpose`s stand out in Run 4 | slow operator paths | **B1**, or Run 7's NCHW layout |

## Fixes, once the cause is known

**A. In the client (desktop-streamer)**

1. **Buffer churn.** Run the frame's tiles through the graph in chunks small enough for
   ORT's cache (for example 8 tiles, 47 MB activations). Or use graph capture (A4), which
   keeps its buffers.
2. **Overlap frames.** Allow 2 frames in flight instead of waiting for the GPU after each
   one. This hides the GPU wait and the CPU's share, at the cost of a frame of delay.
3. **Draw once.** Draw the tiles straight into the visible canvas, reconfigured onto ORT's
   device. This drops the enhancer's own canvas, the `VideoFrame`, the crossing to the
   drawer's device and the second draw.
4. **Graph capture.** Fix the shapes with `freeDimensionOverrides` (`N`, `H`, `W` per
   stream size), put `condition` in a GPU buffer (12 bytes, `writeBuffer`), and set
   `enableGraphCapture`. This needs one session per stream size.
5. **Wasted work.** Check that the canvas on screen is at least as large as the 2× output.
   A 2560×1440 output shown in a 1080p window gets shrunk again by the browser. Cap the
   upscale to what is displayed.

**B. In the model (desktop-streamer-ai, notebook 23's export)**

1. **4 input channels.** Pad the input to 4 channels: `toPlanes` writes a zero 4th plane,
   and the stem and skip weights get a zero 4th input channel. The output stays identical,
   and both convolutions leave the slow path.
2. **float16.** Export the told graph as float16. Turing runs fp16 at twice the fp32 rate
   and fp16 halves the memory traffic; it was 23% faster on Ampere. This needs
   `shader-f16` from Run 0.
3. **Output only the correction.** Do bicubic in the client's draw shader and have the
   graph output only the luma correction. That drops the 5×5 skip `Conv`, the `Concat` and
   the `Add`, and the output tensor becomes one third the size.
4. **A lighter model tier.** Notebook 16's leaky luma 32-16 (2.8 ms against 3.9 ms on the
   3090) for GPUs like this one.
5. **Skip tiles that didn't change.** Run only the tiles that changed since the last frame
   and reuse the rest. It's the biggest saving on a desktop stream and none on video.

When done, record the results (the tables above) in `src/upscale/summary.md` and in
desktop-streamer's `.claude/CLIENT.md` ("The enhancer"). Until now every figure there is
from the RTX 3090.
