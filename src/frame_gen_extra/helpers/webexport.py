"""Export an extrapolator to ONNX for the client.

Publishing a model = writing its `.onnx` file into `dist/frame_gen_extra/`. The contract is
the one `src/mock/make_mock_models.py` gives `extrapolate.onnx`, so a trained graph can
replace the mock by being copied over it: two inputs, `previous` and `current`, float32 NCHW
in [0, 1], and an `output` the size of either. The few facts a graph cannot hold by itself
(a label, the halo, the CPU time we measured) are stored in the file's own `metadata_props`,
as `../../upscale/helpers/webexport.py` does.

Usage:

    from helpers import webexport
    info = webexport.export(model, "extrapolate_name_tile320x180.onnx", webexport.TILE_INPUT,
                            label="name", halo=webexport.TILE_HALO)
    ms = webexport.time_graph(info["path"], webexport.TILE_INPUT)
    webexport.annotate(info, cpu_ms=round(ms, 2))
"""

import copy
import time
from pathlib import Path

import numpy as np
import onnx
import torch

# dist/frame_gen_extra/ at the repo root
MODELS_DIR = Path(__file__).resolve().parents[3] / "dist" / "frame_gen_extra"

# The client's tile: a 320x180 step and a halo of 4 on every side, as (height, width).
TILE_STEP = (180, 320)
TILE_HALO = 4
TILE_INPUT = (TILE_STEP[0] + 2 * TILE_HALO, TILE_STEP[1] + 2 * TILE_HALO)

# Operators the browser's WebGPU backend runs fast - the same set as upscale's. The mock's
# Mul, Add and Clip cost 1-2 ms a 1080p frame each (make_mock_models.py), which is cheap
# but not measured here, so they are reported as outside it. `Identity` copies nothing.
BUDGET = {"Conv", "Relu", "LeakyRelu", "Add", "Concat", "DepthToSpace", "Constant", "Identity"}
INPUTS = ("previous", "current")


def write_metadata(path, fields):
    """Set keys in a model file's `metadata_props` (a value of None removes the key)."""
    model = onnx.load(str(path))
    kept = [entry for entry in model.metadata_props if entry.key not in fields]
    del model.metadata_props[:]
    for entry in kept:
        model.metadata_props.add(key=entry.key, value=entry.value)
    for key, value in fields.items():
        if value is not None:
            model.metadata_props.add(key=key, value=str(value))
    onnx.save(model, str(path))


def describe(path):
    """Size and operators of a written graph, and which operators are outside BUDGET."""
    ops = {}
    for node in onnx.load(str(path)).graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    return {"path": path, "kb": round(path.stat().st_size / 1024, 1), "ops": ops,
            "outside_budget": sorted(set(ops) - BUDGET)}


def export(model, filename, size, label=None, halo=TILE_HALO, batch=1, opset=17, models_dir=None):
    """Write `model` as ONNX with two fixed inputs of `batch` x 3 x `size` (height, width).

    `size` is the model *input*: the step plus the halo on both sides.
    """
    height, width = size
    models_dir = Path(models_dir or MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    example = tuple(torch.rand(batch, 3, height, width) for _ in INPUTS)
    torch.onnx.export(
        copy.deepcopy(model).to("cpu").eval(), example, str(path),
        opset_version=opset, input_names=list(INPUTS), output_names=["output"],
        dynamo=False,  # the classic exporter: plain op sequence, no onnxscript needed
    )
    # the exporter drops an input the model never reads (Repeat's `previous`), and the client
    # feeds both by name, so it goes back in, unused, with the other one's type
    graph = onnx.load(str(path))
    present = {value.name: value for value in graph.graph.input}
    if len(present) < len(INPUTS):
        inputs = [present.get(name) or copy.deepcopy(next(iter(present.values()))) for name in INPUTS]
        for name, value in zip(INPUTS, inputs):
            value.name = name
        del graph.graph.input[:]
        graph.graph.input.extend(inputs)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
    write_metadata(path, {"label": label, "step": width - 2 * halo, "step_h": height - 2 * halo,
                          "halo": halo, "batch": batch})
    return describe(path)


def annotate(info, **fields):
    """Store measurements (like `cpu_ms`) in an exported model's metadata."""
    write_metadata(info["path"], fields)
    return info


def session(path):
    """An onnxruntime CPU session on a graph file."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run(path, previous, current):
    """Run a graph on two (N, 3, H, W) float32 numpy arrays."""
    return session(path).run(None, {"previous": previous, "current": current})[0]


def time_graph(path, size, batch=1, seconds=0.5):
    """Milliseconds per run on the desktop CPU. `size` is the input (height, width)."""
    runner = session(path)
    feed = {name: np.random.rand(batch, 3, *size).astype(np.float32) for name in INPUTS}
    for _ in range(3):
        runner.run(None, feed)
    runs, started = 0, time.perf_counter()
    while time.perf_counter() - started < seconds or runs < 3:
        runner.run(None, feed)
        runs += 1
    return (time.perf_counter() - started) / runs * 1000


def check_graph(path, model, previous, current, tolerance=2e-3):
    """The graph must compute what the torch model computes. Returns the largest difference."""
    with torch.no_grad():
        expected = copy.deepcopy(model).cpu().eval()(torch.from_numpy(previous), torch.from_numpy(current)).numpy()
    difference = float(np.abs(expected - run(path, previous, current)).max())
    assert difference < tolerance, f"{Path(path).name} computes something else: {difference:.1e}"
    return difference


def summary_line(info, ms=None):
    """One printable line about an exported graph."""
    ops = ", ".join(f"{op}x{n}" for op, n in sorted(info["ops"].items())) or "no operators"
    timing = f"{ms:8.2f} ms" if ms is not None else ""
    warning = f"   OUTSIDE BUDGET: {', '.join(info['outside_budget'])}" if info["outside_budget"] else ""
    return f"{info['path'].name:<52}{info['kb']:7.1f} KB{timing}   {ops}{warning}"
