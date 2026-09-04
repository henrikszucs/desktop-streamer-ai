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
    seconds: document.getElementById("seconds"),
    run: document.getElementById("run"),
    clear: document.getElementById("clear"),
    detail: document.getElementById("detail"),
    status: document.getElementById("status"),
    results: document.getElementById("results"),
    rows: document.querySelector("#results tbody"),
};

let catalogue = null;
const sessions = new Map();

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
const getSession = async function(entry, provider, location) {
    const key = entry["id"] + "|" + provider + "|" + location;
    if (!sessions.has(key)) {
        const options = {
            executionProviders: [provider],
            graphOptimizationLevel: "all",
        };
        if (location === "gpu") {
            options.preferredOutputLocation = "gpu-buffer";
        }
        sessions.set(key, ort.InferenceSession.create("models/" + entry["file"], options));
    }
    try {
        return await sessions.get(key);
    } catch (error) {
        // Drop it: a rejected promise left in the map fails every retry with the same
        // error, including after the backend has been changed back.
        sessions.delete(key);
        throw error;
    }
};

// Resolved rather than requested. Asking for ["webgpu", "wasm"] lets the runtime fall
// back silently, and a row that cannot say which backend produced it is not a result.
const resolveProvider = async function(entry, choice) {
    if (choice !== "auto") {
        await getSession(entry, choice, "cpu");
        return choice;
    }
    try {
        await getSession(entry, "webgpu", "cpu");
        return "webgpu";
    } catch (error) {
        console.warn("webgpu unavailable, falling back to wasm:", error.message);
        await getSession(entry, "wasm", "cpu");
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

// The input, filled once. Both paths write the same random tile; the difference is
// whether it is a JS array handed over every run, or a buffer already in GPU memory.
const prepareInput = function(session, entry, location) {
    const [, channels, height, width] = entry["input"];
    const data = new Float32Array(channels * height * width);
    for (let i = 0; i < data.length; i += 1) {
        data[i] = Math.random();
    }

    const feeds = {};
    if (location !== "gpu") {
        feeds[session.inputNames[0]] = new ort.Tensor("float32", data, entry["input"]);
        return { feeds: feeds, dispose: function() {} };
    }

    // STORAGE is what the runtime binds the buffer as, COPY_DST is what writeBuffer
    // needs to fill it here, and the size is rounded up because the runtime reads the
    // buffer in 16 byte chunks.
    const device = ort.env.webgpu.device;
    const buffer = device.createBuffer({
        size: Math.ceil(data.byteLength / 16) * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, data);

    feeds[session.inputNames[0]] = ort.Tensor.fromGpuBuffer(buffer, {
        dataType: "float32",
        dims: entry["input"],
    });
    return { feeds: feeds, dispose: function() { buffer.destroy(); } };
};

const describe = function(entry) {
    const [, , height, width] = entry["input"];
    const parts = [
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

const benchmark = async function(entry, provider, location, seconds) {
    const session = await getSession(entry, provider, location);
    const input = prepareInput(session, entry, location);
    const device = location === "gpu" ? ort.env.webgpu.device : null;

    // An output left in GPU memory has to be handed back, or the runtime allocates a new
    // buffer for every run and the page grows a gigabyte over a long sample.
    const release = function(results) {
        if (location === "gpu") {
            Object.values(results).forEach((tensor) => tensor.dispose());
        }
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
            release(await session.run(input.feeds));
        }
        await settle();
        return (performance.now() - started) / count;
    };

    try {
        for (let i = 0; i < WARMUP; i += 1) {
            release(await session.run(input.feeds));
        }
        await settle();

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
        };
    } finally {
        input.dispose();
    }
};

const addRow = function(entry, result) {
    const cells = [
        entry["label"],
        result["provider"],
        result["location"] === "gpu" ? "GPU" : "CPU round trip",
        result["best"].toFixed(3),
        result["median"].toFixed(3),
        result["tilesPerSecond"].toFixed(0),
        result["frameMs"].toFixed(0) + " ms",
        result["fps"].toFixed(1),
        String(result["runs"]),
    ];

    const row = document.createElement("tr");
    cells.forEach((text, index) => {
        const cell = document.createElement("td");
        cell.textContent = text;
        if (index >= 3) {
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
        const result = await benchmark(entry, provider, location, seconds);
        addRow(entry, result);
        setStatus(`${entry["label"]}: ${result["median"].toFixed(3)} ms per tile `
            + `(${result["runs"]} runs in batches of ${result["chunk"]}, `
            + `${location === "gpu" ? "GPU-resident tensors" : "CPU round trip"})`, "ok");
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
