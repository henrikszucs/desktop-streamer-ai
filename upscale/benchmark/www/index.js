"use strict";

// Speed benchmark for the ONNX upscaling models the notebooks publish into models/.
// Nothing here checks the output pixels - only how long a tile takes.

const MODELS_URL = "api/models";
// The output picture the frame columns are about.
const FRAME_WIDTH = 1920;
const FRAME_HEIGHT = 1080;
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
// Runs in the profiled pass. It only has to be enough for the per-kernel records to
// average out; it is not a sample, and it is not timed by any clock on this page.
const PROFILE_RUNS = 20;
// How long to keep listening after the profiled runs are settled. ORT reads its query
// buffer back through mapAsync, so the last records arrive some way after the work does,
// and stopping collection the moment the queue drains throws them away.
const PROFILE_DRAIN_MS = 100;

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

// Filled by ORT during a profiled pass, read by profileGpu. Module-level because the
// callback is installed once, at load, and has to outlive any one measurement.
let profileTotal = 0;
let profileRecords = 0;
let profileCollecting = false;

const onProfilingData = function(data) {
    if (!profileCollecting) {
        return;
    }
    // Nanoseconds, one record per kernel. Some builds hand the timestamps over as BigInt,
    // which is why they are put through Number() rather than subtracted as they arrive.
    const start = Number(data["startTime"]);
    const end = Number(data["endTime"]);
    if (Number.isFinite(start) && Number.isFinite(end) && end >= start) {
        profileTotal += end - start;
        profileRecords += 1;
    }
};

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

// Where the tensors live while the model runs, and the reason this page has the option at
// all. A WebGPU run fed a CPU array uploads the tile and reads the result back every
// iteration - for a 132 tile that is 209 KB up and 836 KB down, through a queue fence
// that stalls the pipeline. The copy is the same size whatever the graph does, so it
// hides the model completely: a one-node Resize and a convolution stack come out at the
// same number, which is what "a simple resize costs what a whole network costs" actually
// means. Held on the GPU, the run measures the model.
//
// Resolved rather than chosen, unlike the backend: only WebGPU has a buffer this page can
// hand a tensor to and keep. WASM has no GPU memory at all, and WebGL has textures it fills
// and reads back itself and does not expose, so both of them are a round trip whatever is
// asked for - this is a request the machine refuses on its own terms rather than a choice.
// Where it is refused the row says "CPU round trip" in its own column, so the number is
// never mistaken for a GPU-resident one.
const resolveLocation = function(provider, choice) {
    if (provider !== "webgpu" || choice === "cpu") {
        return "cpu";
    }
    // `ort.env.webgpu.device` would be the stronger check, but it only exists once a
    // WebGPU session has been created - and the session whose options this decides has
    // not been created yet. The browser's own API is the part that can be asked first;
    // whether the runtime can really use it is answered by the session creation that
    // follows, which falls back to the next backend if it cannot.
    if (!navigator.gpu || typeof ort.Tensor.fromGpuBuffer !== "function") {
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
// - and a row labelled with a mode that did not run is worse than no row. It takes no
// provider: only resolveLocation hands out "gpu", and only for WebGPU, so a GPU-resident
// location is already a WebGPU one and there is nothing left for a provider to rule out.
const resolveMode = function(location, choice) {
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
    return choice;
};

// What a backend cannot be asked for at all, refused before a session is built rather than
// after.
//
// Only the one limit that is certain. The WebGL EP has no float16 tensor type - the refusal
// comes from the tensor layer rather than from any operator, so no build of it will take a
// half-precision graph, and finding that out by downloading one and compiling shaders for
// it is a waste of several seconds. Everything else WebGL turns down is a property of one
// build of one runtime - a Resize mode it never implemented, a quantized graph it cannot
// resolve - and those are left to fail with ORT's own message, which names the operator and
// stays true when the runtime changes. A table of guesses about a backend would not.
const checkBackend = function(entry, backend) {
    if (backend === "webgl" && (entry["dtype"] !== "float32"
            || entry["out_dtype"] !== "float32")) {
        throw new Error(`the WebGL backend has only float32 tensors, and `
            + `${entry["label"]} is ${entry["dtype"]} - run it on WebGPU or WASM`);
    }
};

// The tensor location and the runtime mode, resolved against the backend the run was asked
// for - and resolved by building the session the measurement will actually run on.
//
// The backend is not among the things resolved. It is chosen, and a choice that cannot be
// created is an error. There was an "auto" here that tried WebGPU and dropped to WASM when
// the session would not build, which is a silent change of subject: two rows of the table
// would be two different runtimes, one of them measured through a fallback nobody asked
// for, and a benchmark whose backend is decided by whether something threw is not one you
// can read a comparison out of.
//
// Built rather than probed: a session compiles kernels and puts the weights on the device,
// so answering "can this backend run the model" with a session at some other location -
// which getContext then caches for the life of the page without ever running it - pays for
// both of those twice for every model.
const resolveRun = async function(entry, backend, data, runtime) {
    checkBackend(entry, backend);
    const location = resolveLocation(backend, data);
    const mode = resolveMode(location, runtime);
    await getContext(entry, backend, location, mode === "capture");
    return { provider: backend, location: location, mode: mode };
};

// What a backend has been *observed* to turn down, keyed by model and backend, holding the
// runtime's own words. Not a support table: nothing is entered here that was not either
// refused by checkBackend's one certain rule or thrown by ORT on this machine, in this
// browser, for this build. An entry that is absent means unknown, never supported - which
// is why an unprobed model stays selectable and fails with ORT's message if it cannot run.
//
// It exists because two thirds of the models cannot run on WebGL, the list is ordered
// cheapest-first, and the three cheapest are three it refuses: the page opened on a dead
// selection and read as a backend that does not work at all.
const refusals = new Map();

// The other half of the same record: pairs seen to run. A model that has run once is not
// disabled by a later failure, because that failure is something other than "this backend
// cannot run this graph" - a lost GPU context, a device gone away - and a transient that
// took a model out of the list for the life of the page would be worse than the failure.
const runnable = new Set();

// Keyed by the whole request, not by the backend alone. Data and Runtime are part of what
// was asked for - a graph capture the runtime rejects says nothing about the same model on
// a plain run - so a record made under one pair of them must not disable a model under
// another. The raw choices rather than the resolved ones, because resolving warns when it
// has to overrule a choice and this is asked once per option per repaint.
const refusalKey = function(entry, backend) {
    return [entry["id"], backend, elements.data.value, elements.mode.value].join("|");
};

// How many models this one backend has been seen to turn down. Counted per backend rather
// than off the map's size, which holds every backend asked so far.
const refusedCount = function(backend) {
    return catalogue["models"].filter(
        (entry) => refusals.has(refusalKey(entry, backend))).length;
};

// Everything already known against this pair, without running anything. checkBackend is
// consulted rather than duplicated: its float16 rule is the one refusal this page makes on
// its own, and it covers a third of the catalogue for free.
const knownRefusal = function(entry, backend) {
    const key = refusalKey(entry, backend);
    if (runnable.has(key)) {
        return null;
    }
    if (!refusals.has(key)) {
        try {
            checkBackend(entry, backend);
        } catch (error) {
            refusals.set(key, error.message);
        }
    }
    return refusals.get(key) || null;
};

// One run of the model through exactly the path a measurement would take, to find out
// whether this backend runs it. Its answer is the runtime's, and the session it built is
// the session the measurement then reuses, so a probe that succeeds is the warmup it
// would have paid anyway.
//
// Cheap in the only case it is asked for in bulk. A refusal throws on the first dispatch
// of the operator it cannot resolve, before any real work: measured on WebGL here, every
// refusal came back in 1.8-6.7 ms against 34-1123 ms for a run that succeeds. So walking a
// list of models a backend cannot run costs almost nothing, and the walk stops at the
// first one it can.
const probeBackend = async function(entry, backend) {
    const known = knownRefusal(entry, backend);
    if (known !== null) {
        return known;
    }
    try {
        const plan = await resolveRun(entry, backend, elements.data.value,
            elements.mode.value);
        const context = await getContext(entry, plan.provider, plan.location,
            plan.mode === "capture");
        const runner = makeRunner(context, plan.location, plan.mode);
        runner.release(await runner.once());
        await runner.settle();
        runnable.add(refusalKey(entry, backend));
        return null;
    } catch (error) {
        refusals.set(refusalKey(entry, backend), error.message);
        return error.message;
    }
};

// The dropdown, showing what has been learned about the backend now selected. A refused
// model is disabled and says why in its own label, so the reason is on the option rather
// than in a status line that the next click replaces.
const markModels = function(backend) {
    Array.from(elements.model.options).forEach((option) => {
        const entry = findModel(option.value);
        if (!entry) {
            return;
        }
        const refusal = knownRefusal(entry, backend);
        option.disabled = refusal !== null;
        option.textContent = refusal === null ? entry["label"]
            : `${entry["label"]} - ${backend} refused: ${refusal}`;
    });
};

// Move the selection onto a model this backend has been shown to run, so the first press
// of Run lands on something that runs. Called whenever the request changes - the backend,
// the data location or the runtime mode - because any of the three can change the answer.
const selectRunnable = async function(backend) {
    markModels(backend);

    // The model already selected is asked about first, and kept only if the backend is
    // shown to run it. "Not known to be refused" is a weaker claim than "runs", and
    // treating the two as one is what left the page opening on a model WebGL cannot run:
    // nothing is known about any model until something has asked.
    const current = findModel(elements.model.value);
    const order = current
        ? [current].concat(catalogue["models"].filter((entry) => entry !== current))
        : catalogue["models"];

    for (const entry of order) {
        setStatus(`asking ${backend} what it can run - ${entry["label"]}…`, "busy");
        const refusal = await probeBackend(entry, backend);
        markModels(backend);
        if (refusal === null) {
            elements.model.value = entry["id"];
            elements.detail.textContent = describe(entry);
            const refused = refusedCount(backend);
            setStatus(`${backend} runs ${entry["label"]}`
                + (refused > 0 ? ` · ${refused} model(s) it refused are disabled above, `
                    + `with its reason on each` : ""));
            return;
        }
    }
    setStatus(`${backend} refused every model in the list - see the reason on each`,
        "error");
};

// The input, filled once. Both paths write the same random tile; the difference is
// whether it is a JS array handed over every run, or a buffer already in GPU memory.
const prepareInput = function(session, entry, location) {
    const dims = entry["input"];
    const dtype = entry["dtype"] || "float32";
    // Every dimension, batch included: the tensor is built with the whole shape, so a
    // tile counted from three of the four is a length the constructor rejects.
    const data = makeTile(dtype, dims.reduce((total, value) => total * value, 1));

    const feeds = {};
    if (location !== "gpu") {
        feeds[session.inputNames[0]] = new ort.Tensor(dtype, data, dims);
        return { feeds: feeds };
    }

    const device = ort.env.webgpu.device;
    const buffer = gpuBuffer(device, data.byteLength);
    device.queue.writeBuffer(buffer, 0, data);

    feeds[session.inputNames[0]] = ort.Tensor.fromGpuBuffer(buffer, {
        dataType: dtype,
        dims: dims,
    });
    return { feeds: feeds };
};

// The output, allocated once instead of per run. Left to itself the runtime returns a new
// tensor every run - a fresh GPU buffer on the GPU path, a fresh typed array otherwise -
// so an allocation and a free sit inside every timed iteration, and on the GPU path the
// buffer has to be disposed by hand or the page grows by one output per run. Handing the
// same tensor back as the fetch moves that storage into the setup, and it is also what
// lets graph capture replay a recorded command buffer: the buffers have to be the ones it
// recorded.
//
// On the GPU path only. A preallocated CPU output is accepted by this runtime and then not
// written - see resolveMode, which is why no CPU run ever passes one - so off the GPU path
// there is nothing to preallocate, and an output-sized tile per session would be a tile
// nothing ever reads.
const prepareOutput = function(session, entry, location) {
    if (location !== "gpu") {
        return { fetches: null };
    }

    // The type of the *output*, which is not always the type of the input: a graph that
    // casts on the way out is fed one and hands back the other, and this is what the
    // buffer is both sized and read as. The server reads it off the graph's own output.
    const dtype = entry["out_dtype"] || entry["dtype"] || "float32";
    const dims = entry["output"];
    const count = dims.reduce((total, value) => total * value, 1);
    const device = ort.env.webgpu.device;
    const bytes = Math.ceil((count * (BYTES[dtype] || 4)) / 16) * 16;
    const buffer = gpuBuffer(device, bytes);
    const fetches = {};
    fetches[session.outputNames[0]] = ort.Tensor.fromGpuBuffer(buffer,
        { dataType: dtype, dims: dims });

    // `count` elements rather than the whole buffer: it is rounded up to 16 and the
    // padding is not part of the model's output, so a checksum over it would be diluted
    // by however much rounding this particular shape happened to need.
    return { fetches: fetches, read: async function() {
        return new (VIEWS[dtype] || Float32Array)(await readBuffer(device, buffer, bytes),
            0, count);
    } };
};

const describe = function(entry) {
    const [, , height, width] = entry["input"];
    const parts = [
        entry["precision"] || entry["dtype"],
        `${width}×${height} in, ${width * entry["scale"]}×${height * entry["scale"]} out`,
        `keeps ${entry["covers"][0]}×${entry["covers"][1]}`,
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
    if (entry["outside_budget"].length > 0) {
        parts.push(`outside the budget: ${entry["outside_budget"].join(", ")}`);
    }
    return parts.join(" · ");
};

// The model's own time on the GPU, in milliseconds per run - the number a clock around
// run() cannot produce.
//
// ORT flushes its command encoder at the end of every run(), so one queue.submit() sits
// under every wall-clock measurement whatever the graph does: tens of microseconds,
// cross-process in Chrome, and unavoidable. Sizing the batch amortises the fence at the end
// of a batch; nothing amortises a submit that happens once per run. So wall clock reports
// max(dispatch, kernels), and every model cheaper than the dispatch reports the dispatch
// instead of itself. That is how three static filters - a single Resize over 0.2 MPixel,
// which is a few microseconds of bandwidth and no arithmetic at all - came to read within a
// factor of two of a 178 MMAC convolution stack. They were never being measured.
//
// Timestamp queries are written by the GPU either side of each kernel, so they time the
// kernels and nothing around them. ORT reports one record per kernel; the kernels of a run
// are dispatched in sequence, so a run's GPU time is their total.
const profileGpu = async function(once, release, settle, runs) {
    const profiling = ort.env.webgpu ? ort.env.webgpu.profiling : null;
    if (!profiling) {
        return null;
    }
    profileTotal = 0;
    profileRecords = 0;
    profiling.mode = "default";
    profileCollecting = true;
    try {
        for (let i = 0; i < runs; i += 1) {
            release(await once());
        }
        await settle();
        await new Promise((resolve) => setTimeout(resolve, PROFILE_DRAIN_MS));
    } catch (error) {
        console.warn("gpu profiling failed - reporting wall clock only:", error.message);
        return null;
    } finally {
        profileCollecting = false;
        profiling.mode = "off";
    }
    // No records at all means this adapter has no timestamp-query, or this runtime fixed
    // its query type when the device was made and will not turn one on now. Either way
    // there is nothing to report, and a zero would read as an impossibly fast model.
    if (profileRecords === 0) {
        console.warn("no timestamp records - this adapter cannot time its own kernels");
        return null;
    }
    return { ms: profileTotal / runs / 1e6, kernels: profileRecords / runs };
};

// The three primitives a run is made of, over one context. Shared with the probe below
// rather than written twice: a probe issuing a different run from the one it clears the
// way for would be answering about some other run than the measurement's.
const makeRunner = function(context, location, mode) {
    const reuse = mode === "capture" || mode === "reuse";
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
        return reuse
            ? context.session.run(context.input.feeds, context.output.fetches)
            : context.session.run(context.input.feeds);
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

    return { release: release, once: once, settle: settle };
};

const benchmark = async function(entry, provider, location, mode, seconds) {
    const capture = mode === "capture";
    const reuse = capture || mode === "reuse";
    const context = await getContext(entry, provider, location, capture);
    const session = context.session;
    const output = context.output;
    const runner = makeRunner(context, location, mode);
    const release = runner.release;
    const once = runner.once;
    const settle = runner.settle;

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
        // `seconds` bounds the sampling, not the run: the warmup, the checksum read and
        // the probe are all spent before this clock starts, and MIN_SAMPLES can carry it
        // past the deadline. It is a budget for the timed part, not a total runtime.
        const deadline = performance.now() + seconds * 1000;
        while (samples.length < MAX_SAMPLES
                && (samples.length < MIN_SAMPLES || performance.now() < deadline)) {
            samples.push(await timeBatch(chunk));
        }

        // After the wall clock, never during it: the queries are work of their own, and a
        // sample that carried them would not be the number a client sees.
        const gpu = provider === "webgpu"
            ? await profileGpu(once, release, settle, PROFILE_RUNS)
            : null;

        samples.sort((a, b) => a - b);
        const median = samples[Math.floor(samples.length / 2)];
        // How many runs one frame takes, from what a single run covers in each dimension
        // rather than from a square tile assumed out of `step`.
        const covers = entry["covers"];
        const tilesPerFrame = Math.ceil(FRAME_WIDTH / covers[0])
            * Math.ceil(FRAME_HEIGHT / covers[1]);

        return {
            best: samples[0],
            median: median,
            chunk: chunk,
            runs: samples.length * chunk,
            tilesPerSecond: 1000 / median,
            tilesPerFrame: tilesPerFrame,
            frameMs: median * tilesPerFrame,
            fps: 1000 / (median * tilesPerFrame),
            provider: provider,
            location: location,
            mode: mode,
            gpuMs: gpu ? gpu["ms"] : null,
            gpuKernels: gpu ? gpu["kernels"] : null,
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
        // The model, and then everything the wall clock adds around it - one queue submit
        // per run, plus whatever the runtime spends in JavaScript getting there. A row
        // whose dispatch dwarfs its GPU time is a row about this page, not about the model.
        result["gpuMs"] === null ? "n/a" : result["gpuMs"].toFixed(4),
        result["gpuMs"] === null ? "n/a"
            : Math.max(0, result["median"] - result["gpuMs"]).toFixed(3),
        result["tilesPerSecond"].toFixed(0),
        String(result["tilesPerFrame"]),
        result["frameMs"].toFixed(1) + " ms",
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
        const plan = await resolveRun(entry, choice, elements.data.value,
            elements.mode.value);
        const result = await benchmark(entry, plan.provider, plan.location, plan.mode,
            seconds);
        runnable.add(refusalKey(entry, choice));
        addRow(entry, result);
        setStatus(`${entry["label"]}: ${result["median"].toFixed(3)} ms per tile `
            + `(${result["runs"]} runs in batches of ${result["chunk"]}, `
            + `${plan.location === "gpu" ? "GPU-resident tensors" : "CPU round trip"}, `
            + `${plan.mode})`
            + (result["gpuMs"] === null ? ""
                : ` \u00b7 ${result["gpuMs"].toFixed(4)} ms of that on the GPU, over `
                    + `${result["gpuKernels"].toFixed(0)} kernels`), "ok");
    } catch (error) {
        console.error(error);
        // A backend that turned this model down has just said so in its own words, which
        // is the same answer a probe would have got - so record it, and let the dropdown
        // show it rather than making the next press find out again. Not for a pair that
        // has already run: see `runnable`.
        if (!runnable.has(refusalKey(entry, choice))) {
            refusals.set(refusalKey(entry, choice), error.message);
            markModels(choice);
        }
        setStatus(`failed: ${error.message}`, "error");
    } finally {
        elements.run.disabled = false;
    }
};

const load = async function() {
    ort.env.wasm.wasmPaths = WASM_PATH;

    // Installed before any session, and so before any device: ORT asks the adapter for the
    // timestamp-query feature when it creates the device, and profiling requested after
    // that can find a device unable to answer. It is left off, because the queries cost
    // work of their own and the wall clock is taken without them; profileGpu turns it on
    // for its own pass and off again.
    if (ort.env.webgpu) {
        ort.env.webgpu.profiling = { mode: "off", ondata: onProfilingData };
    }

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

    // The list is ordered cheapest-first and the cheapest models are single resampling
    // nodes, which is exactly what a backend is most likely to lack - so the opening
    // selection is settled by asking the chosen backend, not by taking the first row.
    await backendChanged();
};

// One place the selection is settled from, so a backend change and the opening load agree.
const backendChanged = async function() {
    elements.run.disabled = true;
    try {
        await selectRunnable(elements.backend.value);
    } finally {
        elements.run.disabled = false;
    }
};

// Data and Runtime are part of what was asked for, so a change to either can change what
// runs - the selection is settled again from the same place the backend settles it.
[elements.backend, elements.data, elements.mode].forEach((element) => {
    element.addEventListener("change", () => {
        backendChanged().catch((error) => {
            console.error(error);
            setStatus(error.message, "error");
        });
    });
});

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
