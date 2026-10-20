"""Export a model to ONNX for the browser benchmark.

Publishing a model = writing its `.onnx` file into `dist/upscale/`. The benchmark server
lists whatever is in that folder, so there is no other list to update. The few facts a
graph cannot hold by itself (a label, the halo, the CPU time we measured) are stored in
the file's own `metadata_props`.

Usage:

    from helpers import webexport
    info = webexport.export(model, "upscale_name_tile128.onnx", size=136, label="name", halo=4)
    ms = webexport.time_graph(info["path"], (136, 136))
    webexport.annotate(info, cpu_ms=round(ms, 2))

`size` is the model *input*: the step plus the halo on both sides. `tiles.py` explains
the three sizes of a tile (step, input, kept) and picks the tile a frame is cut into.
"""

import copy
import math
import time
from pathlib import Path

import numpy as np
import onnx
import torch

from .tiles import as_size

# dist/upscale/ at the repo root - the folder the benchmark page reads
MODELS_DIR = Path(__file__).resolve().parents[3] / "dist" / "upscale"

# The step every model on the benchmark page is compared at.
TILE_STEP = 128

# Operators the browser's WebGPU backend runs fast. Anything else may fall back to the CPU.
BUDGET = {"Conv", "Relu", "LeakyRelu", "Add", "Concat", "DepthToSpace", "Constant"}

# How `count_work` charges operators other than `Conv`: one operation per output value, none
# (data only moves), or none and no memory traffic (the shape is relabelled, nothing is copied).
ELEMENTWISE = {"Relu", "LeakyRelu", "Add", "Sub", "Mul", "Div", "Neg", "Floor", "Ceil", "Clip", "Cast",
               "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And", "Or", "Not", "Where"}
MOVES = {"DepthToSpace", "Concat", "Gather", "Transpose", "Slice", "Split", "Pad"}
VIEWS = {"Reshape", "Unsqueeze", "Squeeze", "Flatten", "Identity", "Shape"}


# ---------------------------------------------------------------- writing graphs

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


def read_metadata(model):
    """A loaded graph's `metadata_props` as a dictionary."""
    return {entry.key: entry.value for entry in model.metadata_props}


def describe(path):
    """Size and operators of a written graph, and which operators are outside BUDGET."""
    ops = {}
    for node in onnx.load(str(path)).graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    return {"path": path, "kb": round(path.stat().st_size / 1024, 1), "ops": ops,
            "outside_budget": sorted(set(ops) - BUDGET)}


def count_work(path):
    """The arithmetic and the memory traffic of one run of a graph, counted from the graph.

    Operations are counted the way GPU and NPU spec sheets count them (TOPS, TFLOPS): one
    multiply-add is 2 operations.

    - `Conv`: 2 per weight for every output value, plus 1 for the bias;
    - elementwise operators (`ELEMENTWISE`: `Relu`, `Add`, `Mul`, `Floor`, a comparison, a
      `Cast`...): 1 per output value;
    - `ReduceSum`: 1 per input value;
    - `MOVES` (`DepthToSpace`, `Concat`, `Gather`, `Transpose`...) and `Constant`: 0 - they
      only move data;
    - `VIEWS` (`Reshape`, `Unsqueeze`...): 0, and no memory traffic either - they only
      relabel a tensor's shape.

    Memory traffic assumes every operator reads its inputs from memory and writes its
    output back (no fusing), plus reading every weight once. A `Gather` from a table reads
    only the entries it gathers, which its output already counts.

    Returns {"operations", "by_op" (operations per operator type), "activation_bytes",
    "weight_bytes", "input_pixels", "not_counted" (operators it does not know)}.
    """
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    graph = model.graph

    # (shape, bytes per value) of every tensor between the operators
    tensors = {}
    for value in [*graph.input, *graph.value_info, *graph.output]:
        kind = value.type.tensor_type
        itemsize = onnx.helper.tensor_dtype_to_np_dtype(kind.elem_type).itemsize
        tensors[value.name] = ([dim.dim_value for dim in kind.shape.dim], itemsize)
    weights = {init.name: list(init.dims) for init in graph.initializer}
    weight_bytes = sum(onnx.numpy_helper.to_array(init).nbytes for init in graph.initializer)
    for node in graph.node:   # the exporter shares a weight two layers hold alike through an Identity
        if node.op_type == "Identity" and node.input[0] in weights:
            weights[node.output[0]] = weights[node.input[0]]

    fixed = set(weights) | {node.output[0] for node in graph.node if node.op_type == "Constant"}

    by_op, not_counted, activation_bytes = {}, [], 0
    for node in graph.node:
        if node.op_type == "Constant" or node.op_type in VIEWS:
            continue
        out_shape = tensors[node.output[0]][0]
        if node.op_type == "Conv":
            kernel = weights[node.input[1]]   # (out channels, in channels / groups, kh, kw)
            outputs = math.prod(out_shape)
            has_bias = len(node.input) > 2 and node.input[2] != ""
            operations = 2 * kernel[1] * kernel[2] * kernel[3] * outputs + (outputs if has_bias else 0)
        elif node.op_type in ELEMENTWISE:
            operations = math.prod(out_shape)
        elif node.op_type == "ReduceSum":
            operations = math.prod(tensors[node.input[0]][0])
        elif node.op_type in MOVES:
            operations = 0
        else:
            not_counted.append(node.op_type)
            operations = 0
        by_op[node.op_type] = by_op.get(node.op_type, 0) + operations

        # read every input that is an activation (not a weight), write the output
        for name in [*node.input, node.output[0]]:
            if name in tensors and name not in fixed:
                shape, itemsize = tensors[name]
                activation_bytes += math.prod(shape) * itemsize

    input_shape = tensors[graph.input[0].name][0]
    return {"operations": sum(by_op.values()), "by_op": by_op, "activation_bytes": activation_bytes,
            "weight_bytes": weight_bytes, "input_pixels": input_shape[0] * input_shape[2] * input_shape[3],
            "not_counted": not_counted}


def export(model, filename, size, label=None, halo=None, batch=1,
           dynamic=False, opset=17, models_dir=None):
    """Write `model` as ONNX with a fixed input of `batch` x 3 x `size`.

    `size` is the model *input*: the step plus the halo on both sides, one number or
    (height, width). `halo` is the extra pixels on each side; left out, it is whatever the
    input has beyond `TILE_STEP`. `dynamic=True` lets the graph take any size instead.
    """
    height, width = as_size(size)
    models_dir = Path(models_dir or MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    torch.onnx.export(
        copy.deepcopy(model).to("cpu").eval(), (torch.randn(batch, 3, height, width),), str(path),
        opset_version=opset, input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}} if dynamic else None,
        dynamo=False,  # the classic exporter: plain op sequence, no onnxscript needed
    )
    if halo is None:
        halo = (width - TILE_STEP) // 2
    write_metadata(path, {"label": label, "step": width - 2 * halo,
                          "step_h": height - 2 * halo, "halo": halo, "batch": batch})
    return describe(path)


def annotate(info, **fields):
    """Store measurements (like `cpu_ms`) in an exported model's metadata."""
    write_metadata(info["path"], fields)
    return info


def to_float16(info, filename, models_dir=None):
    """A half precision copy of an exported graph (weights, input and output)."""
    from onnxruntime.transformers import float16

    source = onnx.load(str(info["path"]))
    target = Path(models_dir or MODELS_DIR) / filename
    # keep_io_types=False: a float16 input is half the upload, which is the point
    onnx.save(float16.convert_float_to_float16(source, keep_io_types=False), str(target))
    write_metadata(target, {**read_metadata(source), "cpu_ms": None, "precision": "float16"})
    return describe(target)


def to_int8(info, filename, calibration, models_dir=None):
    """An int8 copy of an exported graph, calibrated on `calibration` (real input tiles).

    Note: the QuantizeLinear/DequantizeLinear nodes it adds are outside BUDGET, so in the
    browser it is much slower, not faster.
    """
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, tiles):
            self.tiles = iter([{"input": tile.astype(np.float32)} for tile in tiles])

        def get_next(self):
            return next(self.tiles, None)

    source = onnx.load(str(info["path"]))
    target = Path(models_dir or MODELS_DIR) / filename
    prepared = target.with_suffix(".prepared.onnx")
    quant_pre_process(str(info["path"]), str(prepared), skip_symbolic_shape=True)
    try:
        quantize_static(str(prepared), str(target), Reader(calibration),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax)
    finally:
        prepared.unlink(missing_ok=True)
    write_metadata(target, {**read_metadata(source), "cpu_ms": None, "precision": "int8"})
    return describe(target)


def to_frame(info, filename, width=1920, height=1080, models_dir=None):
    """The same graph with a whole-frame input, so one frame is one run instead of many.

    Only the declared input shape changes - every operator here reads its size from its
    input. A `Resize` with a fixed output size cannot be reshaped, so that is refused.
    """
    source = onnx.load(str(info["path"]))
    graph = source.graph

    initializers = {init.name for init in graph.initializer}
    produced = {out for node in graph.node for out in node.output}
    for node in graph.node:
        if node.op_type == "Resize" and len(node.input) >= 4 and node.input[3]:
            if node.input[3] in initializers or node.input[3] in produced:
                raise ValueError(f"{node.name or 'Resize'} has a fixed output size, so this graph cannot be reshaped")

    metadata = read_metadata(source)
    shape_in = graph.input[0].type.tensor_type.shape
    shape_out = graph.output[0].type.tensor_type.shape
    scale = shape_out.dim[2].dim_value // shape_in.dim[2].dim_value
    step = int(metadata.get("step", TILE_STEP))
    halo = int(metadata.get("halo", (shape_in.dim[2].dim_value - step) // 2))

    shape_in.dim[2].dim_value = height // scale + 2 * halo
    shape_in.dim[3].dim_value = width // scale + 2 * halo
    shape_out.dim[2].dim_value = shape_in.dim[2].dim_value * scale
    shape_out.dim[3].dim_value = shape_in.dim[3].dim_value * scale

    # drop the tile-sized shapes inferred inside the graph and infer them again
    del graph.value_info[:]
    target = Path(models_dir or MODELS_DIR) / filename
    reshaped = onnx.shape_inference.infer_shapes(source, strict_mode=True)
    onnx.checker.check_model(reshaped)
    onnx.save(reshaped, str(target))

    label = metadata.get("label", target.stem)
    write_metadata(target, {**metadata, "cpu_ms": None, "label": f"{label} · whole frame",
                            "step": width // scale, "step_h": height // scale, "halo": halo})
    return describe(target)


# ---------------------------------------------------------------- running graphs

def session(path):
    """An onnxruntime CPU session on a graph file."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run(path_or_session, x, dtype=np.float32):
    """Run a graph on a (N, 3, H, W) numpy array and return the output as float32."""
    runner = session(path_or_session) if isinstance(path_or_session, (str, Path)) else path_or_session
    name = runner.get_inputs()[0].name
    return runner.run(None, {name: x.astype(dtype)})[0].astype(np.float32)


class OnnxModel(torch.nn.Module):
    """An ONNX graph wrapped as a torch module, so it can be scored like any model."""

    def __init__(self, path, dtype=np.float32):
        super().__init__()
        self.runner = session(path)
        self.dtype = dtype

    def forward(self, x):
        out = run(self.runner, x.detach().cpu().numpy(), self.dtype)
        return torch.from_numpy(out).to(x.device)


def time_graph(path, shape, dtype=np.float32, batch=1, seconds=0.5):
    """Milliseconds per run on the desktop CPU. `shape` is the input (height, width)."""
    runner = session(path)
    name = runner.get_inputs()[0].name
    feed = {name: np.random.rand(batch, 3, *as_size(shape)).astype(dtype)}
    for _ in range(3):
        runner.run(None, feed)
    runs, started = 0, time.perf_counter()
    while time.perf_counter() - started < seconds or runs < 3:
        runner.run(None, feed)
        runs += 1
    return (time.perf_counter() - started) / runs * 1000


def check_graph(path, model, x, dtype=np.float32, tolerance=2e-3):
    """The graph must compute what the torch model computes. Returns the largest difference.

    `x` is a float32 (N, 3, H, W) numpy array at the graph's input size.
    """
    with torch.no_grad():
        expected = copy.deepcopy(model).cpu().eval()(torch.from_numpy(x)).numpy()
    difference = float(np.abs(expected - run(path, x, dtype)).max())
    assert difference < tolerance, f"{Path(path).name} computes something else: {difference:.1e}"
    return difference


def summary_line(info, ms=None):
    """One printable line about an exported graph."""
    ops = ", ".join(f"{op}x{n}" for op, n in sorted(info["ops"].items()))
    timing = f"{ms:8.2f} ms" if ms is not None else ""
    warning = f"   OUTSIDE BUDGET: {', '.join(info['outside_budget'])}" if info["outside_budget"] else ""
    return f"{info['path'].name:<52}{info['kb']:7.1f} KB{timing}   {ops}{warning}"
