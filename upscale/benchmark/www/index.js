"use strict";

// Speed benchmark for the ONNX upscaling models the notebooks publish into models/.
// Nothing here checks the output pixels - only how long a tile takes.

const MODELS_URL = "api/models";
const WASM_PATH = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
const WARMUP = 5;
// A sample is a batch of runs timed together and divided, never one run timed. Two
// separate things force that, and the second sets the size:
//   - performance.now() is deliberately coarse in a browser, and one 140x140 tile can
//     finish under its resolution;
//   - a batch of GPU work is only settled by a fence, and one fence costs milliseconds of
//     latency whatever it is waiting for - about 3.4 ms on an Ampere card here. Spread
//     over a fixed 20 runs that is 0.17 ms added to every run, which is twice the cost of
//     the cheapest model on the page: it is what made three different static filters all
//     report an identical 0.20 ms.
// So the batch is not a constant. It is sized from a probe, to fill TARGET_SAMPLE_MS,
// which puts the one fence at the end of it under 2% of the sample.
const TARGET_SAMPLE_MS = 200;
const PROBE_RUNS = 10;
const MIN_CHUNK = 4;
const MAX_CHUNK = 5000;
const MIN_SAMPLES = 3;
const MAX_SAMPLES = 500;

const elements = {
    model: document.getElementById("model"),
    backend: document.getElementById("backend"),
    data: document.getElementById("data"),
    mode: document.getElementById("mode"),
    seconds: document.getElementById("seconds"),
    run: document.getElementById("run"),
    clear: document.getElementById("clear"),
    detail: document.getElementById("detail"),
    status: document.getElementById("status"),
    results: document.getElementById("results"),
    rows: document.querySelector("#results tbody"),
};

let catalogue = null;

// A session and the two buffers it runs on are one thing, not three. Graph capture is
// what forces that: it records a command buffer against the exact buffers of the run it
// recorded, so a page that frees them and keeps the session replays a command buffer
// pointing at destroyed memory. That does not fail loudly - the queue raises a validation
// error, no work happens, and the timing comes back ten times too good. Allocating both
// with the session and keeping them for its lifetime is also what a client would do:
// one session, one input tile, one output, reused for every frame.
const contexts = new Map();

const setStatus = function(text, kind) {
    elements.status.textContent = text;
    elements.status.className = "status" + (kind ? " " + kind : "");
};

const findModel = function(id) {
    return catalogue["models"].find((entry) => entry["id"] === id);
};

// A session is kept per model, backend and tensor location: creating one compiles
// kernels, and paying for that inside a timed run would measure the compiler rather than
// the model. The location is part of the key because where the output goes is a session
// option - one session returns CPU tensors, the other leaves them in GPU memory.
const getContext = async function(entry, provider, location, capture) {
    const key = [entry["id"], provider, location, capture ? "capture" : ""].join("|");
    if (!contexts.has(key)) {
        const options = {
            executionProviders: [provider],
            graphOptimizationLevel: "all",
        };
        if (location === "gpu") {
            options.preferredOutputLocation = "gpu-buffer";
        }
        // Graph capture records the command buffer of one run and replays it, so every
        // run after the first skips the JavaScript that issues the dispatches. Dispatch
        // alone measured 0.046 ms per run here, which is most of what a filter-sized
        // model costs, and 40 tiles pay it 40 times a frame. The runtime accepts it only
        // with every shape static and both ends of the graph in GPU memory, which is why
        // it is bound to the GPU-resident path.
        if (capture) {
            options.enableGraphCapture = true;
        }
        contexts.set(key, (async function() {
            const session = await ort.InferenceSession.create("models/" + entry["file"],
                options);
            return {
                session: session,
                input: prepareInput(session, entry, location),
                output: prepareOutput(session, entry, location),
            };
        })());
    }
    try {
        return await contexts.get(key);
    } catch (error) {
        // Drop it: a rejected promise left in the map fails every retry with the same
        // error, including after the backend has been changed back.
        contexts.delete(key);
        throw error;
    }
};

// Resolved rather than requested. Asking for ["webgpu", "wasm"] lets the runtime fall
// back silently, and a row that cannot say which backend produced it is not a result.
const resolveProvider = async function(entry, choice) {
    if (choice !== "auto") {
        await getContext(entry, choice, "cpu", false);
        return choice;
    }
    try {
        await getContext(entry, "webgpu", "cpu", false);
        return "webgpu";
    } catch (error) {
        console.warn("webgpu unavailable, falling back to wasm:", error.message);
        await getContext(entry, "wasm", "cpu", false);
        return "wasm";
    }
};

// Where the tensors live while the model runs, and the reason this page has the option at
// all. A WebGPU run fed a CPU array uploads the tile and reads the result back every
// iteration - for a 132 tile that is 209 KB up and 836 KB down, through a queue fence
// that stalls the pipeline. The copy is the same size whatever the graph does, so it
// hides the model completely: a one-node Resize and a convolution stack come out at the
// same number, which is what "a simple resize costs what a whole network costs" actually
// means. Held on the GPU, the run measures the model.
//
// Resolved rather than requested, like the provider: WASM has no GPU buffers, and a
// runtime build without the buffer API cannot do it either.
const resolveLocation = function(provider, choice) {
    if (provider !== "webgpu" || choice === "cpu") {
        return "cpu";
    }
    if (!ort.env.webgpu.device || typeof ort.Tensor.fromGpuBuffer !== "function") {
        console.warn("this runtime exposes no GPU buffer API - timing with CPU tensors");
        return "cpu";
    }
    return "gpu";
};

// float32 -> the bits of the nearest float16, since a half-precision graph is fed halves
// and JavaScript has no half. The tile is random values in [0, 1), so the subnormal and
// infinity branches are there for correctness rather than for this caller.
const toHalf = (function() {
    const floats = new Float32Array(1);
    const bits = new Int32Array(floats.buffer);
    return function(value) {
        floats[0] = value;
        const x = bits[0];
        const sign = (x >> 16) & 0x8000;
        const exponent = (x >> 23) & 0xff;
        let mantissa = (x >> 12) & 0x07ff;
        if (exponent < 103) {
            return sign;
        }
        if (exponent > 142) {
            return sign | 0x7c00;
        }
        if (exponent < 113) {
            mantissa |= 0x0800;
            return sign | ((mantissa >> (114 - exponent)) + ((mantissa >> (113 - exponent)) & 1));
        }
        return sign | ((((exponent - 112) << 10) | (mantissa >> 1)) + (mantissa & 1));
    };
})();

// float16 bits -> the number they stand for, for reading an output back.
const fromHalf = function(bits) {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const mantissa = bits & 0x03ff;
    if (exponent === 0) {
        return sign * mantissa * 5.9604644775390625e-8;
    }
    if (exponent === 0x1f) {
        return mantissa ? NaN : sign * Infinity;
    }
    return sign * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
};

// Seeded, so every session is handed the *same* tile. Math.random() would give each one
// its own, and then two rows of the table could not be compared by their output: the
// check below would only say a run did something, never that it did the same something.
const seeded = function(seed) {
    let state = seed;
    return function() {
        state |= 0;
        state = (state + 0x6d2b79f5) | 0;
        let t = Math.imul(state ^ (state >>> 15), 1 | state);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

// One tile of pixels, in whatever type the graph asks for. The type is read out of the
// graph by the server, so a float16 export is fed halves and a uint8 one bytes without
// the page being told which model is which.
const makeTile = function(dtype, count) {
    // Re-seeded per tile, not per page: two sessions built minutes apart have to be handed
    // the identical tile, or their check columns are two numbers about two inputs.
    const random = seeded(0x9e3779b9);
    if (dtype === "float16") {
        const data = new Uint16Array(count);
        for (let i = 0; i < count; i += 1) {
            data[i] = toHalf(random());
        }
        return data;
    }
    if (dtype === "uint8") {
        const data = new Uint8Array(count);
        for (let i = 0; i < count; i += 1) {
            data[i] = Math.floor(random() * 256);
        }
        return data;
    }
    const data = new Float32Array(count);
    for (let i = 0; i < count; i += 1) {
        data[i] = random();
    }
    return data;
};

// The mean absolute value of an output, whatever type it came back as. It is not a
// picture check - the page has no reference to compare against - but it is enough to
// separate a run that computed the model from a run that computed nothing, and enough
// that two rows which disagree about it were not running the same thing.
const checksum = function(data) {
    // Decided by what came back rather than by what the graph says: a float16 output
    // arrives as raw halves when this page copies the buffer itself, and as ready-made
    // numbers when the runtime downloads it into a Float16Array. Reading the second as
    // the first is how a working model reports an output of zero.
    const halves = data instanceof Uint16Array;
    let total = 0;
    for (let i = 0; i < data.length; i += 1) {
        total += Math.abs(halves ? fromHalf(data[i]) : data[i]);
    }
    return total / data.length;
};

// Copy a buffer back to the CPU. The output of a reused run is a buffer this page
// created, and ORT will not download one of those - `fromGpuBuffer` takes no downloader,
// so `getData()` on it throws. WebGPU will, through a staging buffer it can map.
const readBuffer = async function(device, buffer, bytes) {
    const staging = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return copy;
};

const VIEWS = { float32: Float32Array, float16: Uint16Array, uint8: Uint8Array };

const BYTES = { float32: 4, float16: 2, uint8: 1 };

// STORAGE is what the runtime binds a buffer as, COPY_DST is what writeBuffer needs to
// fill it here, and the size is rounded up because the runtime reads it in 16 byte chunks.
const gpuBuffer = function(device, bytes) {
    return device.createBuffer({
        size: Math.ceil(bytes / 16) * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
};

// Resolved, like the provider and the location. Graph capture is a WebGPU option and the
// runtime rejects it unless the whole graph is static and both ends live in GPU memory,
// so asking for it on WASM or on the round-trip path is a request that cannot be honoured
// - and a row labelled with a mode that did not run is worse than no row.
const resolveMode = function(provider, location, choice) {
    // Reuse - and so capture, which needs it - only means anything on the GPU path. A
    // preallocated *CPU* output is accepted by this runtime and then not written: the run
    // hands back the very tensor it was given, contents untouched, so the page would time
    // a run whose output went nowhere and check a buffer it filled itself. Measured on
    // 1.29.0: a plain run returns 0.2291..., the same run with a fetch returns the -7 it
    // was handed.
    if (choice !== "plain" && location !== "gpu") {
        console.warn("preallocated outputs need GPU-resident tensors - timing plain runs");
        return "plain";
    }
    if (choice === "capture" && provider !== "webgpu") {
        console.warn("graph capture is a WebGPU option - reusing only");
        return "reuse";
    }
    return choice;
};

// The input, filled once. Both paths write the same random tile; the difference is
// whether it is a JS array handed over every run, or a buffer already in GPU memory.
const prepareInput = function(session, entry, location) {
    const [, channels, height, width] = entry["input"];
    const dtype = entry["dtype"] || "float32";
    const data = makeTile(dtype, channels * height * width);

    const feeds = {};
    if (location !== "gpu") {
        feeds[session.inputNames[0]] = new ort.Tensor(dtype, data, entry["input"]);
        return { feeds: feeds, dispose: function() {} };
    }

    const device = ort.env.webgpu.device;
    const buffer = gpuBuffer(device, data.byteLength);
    device.queue.writeBuffer(buffer, 0, data);

    feeds[session.inputNames[0]] = ort.Tensor.fromGpuBuffer(buffer, {
        dataType: dtype,
        dims: entry["input"],
    });
    return { feeds: feeds, dispose: function() { buffer.destroy(); } };
};

// The output, allocated once instead of per run. Left to itself the runtime returns a new
// tensor every run - a fresh GPU buffer on the GPU path, a fresh typed array otherwise -
// so an allocation and a free sit inside every timed iteration, and on the GPU path the
// buffer has to be disposed by hand or the page grows by one output per run. Handing the
// same tensor back as the fetch moves that storage into the setup, and it is also what
// lets graph capture replay a recorded command buffer: the buffers have to be the ones it
// recorded.
const prepareOutput = function(session, entry, location) {
    const dtype = entry["dtype"] || "float32";
    const dims = entry["output"];
    const count = dims.reduce((total, value) => total * value, 1);
    const fetches = {};
    const name = session.outputNames[0];

    if (location !== "gpu") {
        fetches[name] = new ort.Tensor(dtype, makeTile(dtype, count), dims);
        return { fetches: fetches, dtype: dtype, read: async function() {
            return fetches[name].data;
        } };
    }

    const device = ort.env.webgpu.device;
    const bytes = Math.ceil((count * (BYTES[dtype] || 4)) / 16) * 16;
    const buffer = gpuBuffer(device, bytes);
    fetches[name] = ort.Tensor.fromGpuBuffer(buffer, { dataType: dtype, dims: dims });
    return { fetches: fetches, dtype: dtype, read: async function() {
        return new (VIEWS[dtype] || Float32Array)(await readBuffer(device, buffer, bytes));
    } };
};

const describe = function(entry) {
    const [, , height, width] = entry["input"];
    const parts = [
        entry["precision"] || entry["dtype"],
        `${width}×${height} in, ${width * entry["scale"]}×${height * entry["scale"]} out`,
        `keeps ${entry["step"] * entry["scale"]}×${entry["step"] * entry["scale"]}`,
        `${entry["kb"]} KB`,
    ];
    if (entry["halo"] !== null) {
        parts.push(`halo ${entry["halo"]}`);
    }
    if (entry["parameters"] !== null) {
        parts.push(`${entry["parameters"].toLocaleString()} parameters`);
    }
    if (entry["cpu_ms"] !== null) {
        parts.push(`${entry["cpu_ms"]} ms on the desktop CPU runtime`);
    }
    parts.push(Object.entries(entry["ops"]).map(([op, n]) => `${op}×${n}`).join(", "));
    return parts.join(" · ");
};

const benchmark = async function(entry, provider, location, mode, seconds) {
    const capture = mode === "capture";
    const reuse = capture || mode === "reuse";
    const context = await getContext(entry, provider, location, capture);
    const session = context.session;
    const input = context.input;
    const output = context.output;
    const device = location === "gpu" ? ort.env.webgpu.device : null;

    // An output the runtime allocated has to be handed back, or a GPU-resident run leaks
    // one buffer per run and the page grows a gigabyte over a long sample. A reused
    // output is the page's own and outlives the batch, so there is nothing to release.
    const release = function(results) {
        if (location === "gpu" && !reuse) {
            Object.values(results).forEach((tensor) => tensor.dispose());
        }
        return results;
    };

    // One run, either way round: the fetch is handed in when the output is the page's.
    const once = function() {
        return reuse ? session.run(input.feeds, output.fetches) : session.run(input.feeds);
    };

    // With nothing read back, run() resolves once the work is *submitted*, not once it is
    // done - a batch that never waits would time the dispatch and report microseconds.
    // One wait settles the whole batch, which is why the batch is sized to make that one
    // wait small. Measured here: dispatch alone 0.046 ms per run, one fence 3.4 ms flat.
    const settle = async function() {
        if (device) {
            await device.queue.onSubmittedWorkDone();
        }
    };

    // One batch of `count` runs, settled once at the end, as milliseconds per run.
    const timeBatch = async function(count) {
        const started = performance.now();
        for (let i = 0; i < count; i += 1) {
            release(await once());
        }
        await settle();
        return (performance.now() - started) / count;
    };

    try {
        // The first run under graph capture is the recording, and it is slower than every
        // run after it. WARMUP covers that the same way it covers kernel compilation.
        for (let i = 0; i < WARMUP; i += 1) {
            release(await once());
        }
        await settle();

        // Before timing anything: did the run actually compute? A replayed command buffer
        // whose buffers are gone, a provider that silently did nothing - both come back as
        // a resolved promise and a very good number. Reading the output once is what makes
        // the difference between those and a real run visible.
        // Read what the run returned, not what the page allocated. A preallocated CPU
        // fetch is not always the tensor the runtime fills - it may hand back its own -
        // and reading the page's own buffer then reports the tile it was initialised
        // with: every model on the WASM path checked in at an identical 0.4993, which is
        // the mean of the random tile and not any model's output. The one case that has
        // to read the buffer is a GPU-resident fetch, because the tensor the runtime
        // returns there is the page's own and carries no downloader.
        const results = await once();
        const produced = results[session.outputNames[0]];
        let value;
        if (location === "gpu") {
            value = reuse ? checksum(await output.read()) : checksum(await produced.getData());
        } else {
            value = checksum(produced.data);
        }
        release(results);
        if (!Number.isFinite(value) || value === 0) {
            throw new Error(`the output reads back as ${value} - this run computed nothing`);
        }

        // The probe pays a whole fence over PROBE_RUNS runs, so it over-estimates a cheap
        // model - which only makes the real batch longer than it needed to be, never
        // shorter than it should be.
        const probe = await timeBatch(PROBE_RUNS);
        const chunk = Math.min(MAX_CHUNK, Math.max(MIN_CHUNK,
            Math.round(TARGET_SAMPLE_MS / Math.max(probe, 1e-4))));

        const samples = [];
        const deadline = performance.now() + seconds * 1000;
        while (samples.length < MAX_SAMPLES
                && (samples.length < MIN_SAMPLES || performance.now() < deadline)) {
            samples.push(await timeBatch(chunk));
        }

        samples.sort((a, b) => a - b);
        const median = samples[Math.floor(samples.length / 2)];
        const tilesPerFrame = Math.ceil(1080 / (entry["step"] * entry["scale"]))
            * Math.ceil(1920 / (entry["step"] * entry["scale"]));

        return {
            best: samples[0],
            median: median,
            chunk: chunk,
            runs: samples.length * chunk,
            tilesPerSecond: 1000 / median,
            frameMs: median * tilesPerFrame,
            fps: 1000 / (median * tilesPerFrame),
            provider: provider,
            location: location,
            mode: mode,
            checksum: value,
        };
    } finally {
        // The buffers belong to the session, not to this measurement: see getContext.
        await settle();
    }
};

const addRow = function(entry, result) {
    const cells = [
        entry["label"],
        result["provider"],
        result["location"] === "gpu" ? "GPU" : "CPU round trip",
        result["mode"],
        result["best"].toFixed(3),
        result["median"].toFixed(3),
        result["tilesPerSecond"].toFixed(0),
        result["frameMs"].toFixed(0) + " ms",
        result["fps"].toFixed(1),
        String(result["runs"]),
        result["checksum"].toFixed(4),
    ];

    const row = document.createElement("tr");
    cells.forEach((text, index) => {
        const cell = document.createElement("td");
        cell.textContent = text;
        if (index >= 4) {
            cell.className = "number";
        }
        row.appendChild(cell);
    });

    elements.rows.appendChild(row);
    elements.results.hidden = false;
};

const run = async function() {
    const entry = findModel(elements.model.value);
    if (!entry) {
        return;
    }

    const choice = elements.backend.value;
    const seconds = Math.max(1, Number(elements.seconds.value) || 3);

    elements.run.disabled = true;
    setStatus(`running ${entry["label"]} on ${choice}…`, "busy");

    try {
        // Yield first, so the button repaints as disabled before the thread is taken.
        await new Promise((resolve) => setTimeout(resolve, 0));
        const provider = await resolveProvider(entry, choice);
        const location = resolveLocation(provider, elements.data.value);
        const mode = resolveMode(provider, location, elements.mode.value);
        const result = await benchmark(entry, provider, location, mode, seconds);
        addRow(entry, result);
        setStatus(`${entry["label"]}: ${result["median"].toFixed(3)} ms per tile `
            + `(${result["runs"]} runs in batches of ${result["chunk"]}, `
            + `${location === "gpu" ? "GPU-resident tensors" : "CPU round trip"}, `
            + `${mode})`, "ok");
    } catch (error) {
        console.error(error);
        setStatus(`failed: ${error.message}`, "error");
    } finally {
        elements.run.disabled = false;
    }
};

const load = async function() {
    ort.env.wasm.wasmPaths = WASM_PATH;

    // The list is built by the server out of the graphs in www/models/ at request time,
    // so reloading the page is enough to pick up a model a notebook has just written.
    const response = await fetch(MODELS_URL);
    if (!response.ok) {
        throw new Error(`${MODELS_URL} answered ${response.status} - is main.py serving `
            + `this page, or is something else on the port?`);
    }
    catalogue = await response.json();

    elements.model.innerHTML = "";
    catalogue["models"].forEach((entry) => {
        const option = document.createElement("option");
        option.value = entry["id"];
        option.textContent = entry["label"];
        elements.model.appendChild(option);
    });

    if (catalogue["models"].length === 0) {
        setStatus("no models in www/models/ - run the export cell of a model notebook, "
            + "then reload", "error");
        return;
    }

    elements.detail.textContent = describe(catalogue["models"][0]);
    elements.run.disabled = false;
    setStatus(`${catalogue["models"].length} models, newest written `
        + `${catalogue["generated"]}`
        + (navigator.gpu ? " · WebGPU available" : " · no WebGPU in this browser"));
};

elements.model.addEventListener("change", () => {
    elements.detail.textContent = describe(findModel(elements.model.value));
});
elements.run.addEventListener("click", run);
elements.clear.addEventListener("click", () => {
    elements.rows.innerHTML = "";
    elements.results.hidden = true;
});

load().catch((error) => {
    console.error(error);
    setStatus(error.message, "error");
});
