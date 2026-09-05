"""Shared ONNX export for the browser benchmark.

One notebook is one model, and every model notebook ends with an export cell that writes
its graph straight into `benchmark/www/models/` - the folder the benchmark page reads.
Nothing else is written. There is no manifest, because the server builds the model list
out of the graphs themselves on every request (`benchmark/main.py`): the shapes, the
operators, the parameter count and the size on disk are already in the file, and a second
copy of them is a copy that goes stale.

The two things a graph cannot state - the label a human chose for it, and the desktop CPU
time this notebook measured - go into the model's own `metadata_props`, so a model stays
one file that can be copied, deleted or dropped in by hand without anything else knowing.

Usage, from a notebook whose working directory is `model/upscale`:

    import webexport
    info = webexport.export(model, "upscale_name_tile128.onnx", size, label="name", halo=4)
    ...                                          # measure it through onnxruntime
    webexport.annotate(info, cpu_ms=4.2)

The same graph at a lower precision is another candidate rather than another model, and
`to_float16` and `to_int8` write one from an export that already exists. The page reads
the input's element type out of the graph, so a half-precision model needs nothing said
about it anywhere else.
"""

# internal
from pathlib import Path

# third-party
import numpy as np
import onnx
import torch

# Where the page looks. Exports land here directly, and the server lists whatever it
# finds, so publishing a model is writing the file and nothing else.
MODELS_DIR = Path("benchmark/www/models")

# The tile the page benchmarks: a model keeps this many LR pixels per step, and takes
# `TILE_STEP + 2 * halo` in, so the halo it needs is part of its own geometry.
TILE_STEP = 128

# What the WebGPU execution provider is expected to run without falling back to CPU.
# Constant carries a folded scalar rather than any work, so it is allowed through.
BUDGET = {"Conv", "Relu", "LeakyRelu", "Add", "Concat", "DepthToSpace", "Constant"}


def write_metadata(path, fields):
    """Set these keys on a model's `metadata_props`, replacing any already there."""
    model = onnx.load(str(path))
    kept = [entry for entry in model.metadata_props if entry.key not in fields]
    del model.metadata_props[:]
    for entry in kept:
        model.metadata_props.add(key=entry.key, value=entry.value)
    for key, value in fields.items():
        if value is not None:
            model.metadata_props.add(key=key, value=str(value))
    onnx.save(model, str(path))


def export(model, filename, size, label=None, halo=None, step=TILE_STEP, dynamic=False,
           opset=17, models_dir=MODELS_DIR):
    """Write `model` into the page's model folder as ONNX at a `size`x`size` input.

    `dynamo=False` selects the older TorchScript exporter deliberately: it emits the plain,
    predictable op sequence these models are designed around, and the newer path needs
    `onnxscript`, which is not a dependency here.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    torch.onnx.export(
        model.to("cpu").eval(), (torch.randn(1, 3, size, size),), str(path),
        opset_version=opset, input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}} if dynamic else None,
        dynamo=False,
    )

    # The halo is derivable from the input shape and the step, and the server derives it
    # when it is absent - but a model that states its own geometry survives being renamed.
    write_metadata(path, {"label": label, "step": step,
                          "halo": (size - step) // 2 if halo is None else halo})

    return describe(path)


def describe(path):
    """What a written graph costs and what it is made of, read back out of the file."""
    ops = {}
    for node in onnx.load(str(path)).graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    return {"path": path, "kb": round(path.stat().st_size / 1024, 1), "ops": ops,
            "outside_budget": sorted(set(ops) - BUDGET)}


def to_float16(info, filename, models_dir=MODELS_DIR):
    """Half precision, weights and I/O both.

    `keep_io_types=False` is the point: leaving the input float32 would add a Cast around
    the whole graph and keep the tile four bytes a channel on the way in, which is the
    upload this is meant to halve. The page allocates from the graph's own input type, so
    a half-precision model needs nothing told to it.
    """
    from onnxruntime.transformers import float16

    source = onnx.load(str(info["path"]))
    target = models_dir / filename
    onnx.save(float16.convert_float_to_float16(source, keep_io_types=False), str(target))

    metadata = {entry.key: entry.value for entry in source.metadata_props}
    write_metadata(target, {**metadata, "cpu_ms": None, "precision": "float16"})
    return describe(target)


def to_int8(info, filename, calibration, models_dir=MODELS_DIR):
    """Static QDQ int8, calibrated on `calibration` - real LR tiles as float arrays.

    Static rather than dynamic: dynamic quantization measures every activation's range at
    run time, which is a reduction over the tensor before each convolution can start, and
    on a tile this small that costs more than the convolution saves. Calibrating once on
    real patches puts the ranges in the graph as constants instead.

    The Q/DQ pairs it inserts are outside the op budget, which `describe` reports: this is
    a candidate to measure, not an assumption that smaller is faster.
    """
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self, tiles):
            self.tiles = [{"input": tile.astype(np.float32)} for tile in tiles]
            self.index = 0

        def get_next(self):
            if self.index >= len(self.tiles):
                return None
            self.index += 1
            return self.tiles[self.index - 1]

        def rewind(self):
            self.index = 0

    source = onnx.load(str(info["path"]))
    target = models_dir / filename
    prepared = target.with_suffix(".prepared.onnx")
    quant_pre_process(str(info["path"]), str(prepared), skip_symbolic_shape=True)
    try:
        quantize_static(str(prepared), str(target), Reader(calibration),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax)
    finally:
        prepared.unlink(missing_ok=True)

    metadata = {entry.key: entry.value for entry in source.metadata_props}
    write_metadata(target, {**metadata, "cpu_ms": None, "precision": "int8"})
    return describe(target)


def to_frame(info, filename, width=1920, height=1080, models_dir=MODELS_DIR):
    """The same graph at whole-frame size, so one frame is one run instead of forty.

    Every operator these models use reads its geometry from the tensor it is handed - Conv,
    Relu, Add, Concat, DepthToSpace, and a Resize driven by scales rather than by a baked
    output size - so the weights do not change and only the declared input shape moves. The
    tile is a property of how a model is *run*, not of the model, and the reshaped graph is
    bit-identical to the tiled one over the region a tile contributes.

    It is worth having because the difference is not small. A tiled frame pays one queue
    submit per tile, and on an Ampere card that is about 2x on the convolution stacks and
    5-7x on the resampling filters, which do so little work per tile that the dispatch is
    most of what a tiled run measures. Tiling earns its keep when only part of the picture
    changed - a remote desktop\'s dirty region - not when every frame is new.

    A model exported with `dynamic=True` needs none of this; it already takes any size.
    """
    source = onnx.load(str(info["path"]))
    graph = source.graph

    # A Resize pinned to an output size carries the tile in an initializer rather than in
    # the shape, so moving the declared input would leave the graph disagreeing with
    # itself. Nothing here exports one, but say so rather than write a broken graph.
    initializers = {init.name for init in graph.initializer}
    produced = {out for node in graph.node for out in node.output}
    for node in graph.node:
        if node.op_type == "Resize" and len(node.input) >= 4 and node.input[3]:
            if node.input[3] in initializers or node.input[3] in produced:
                raise ValueError(f"{node.name or 'Resize'} is pinned to a baked output "
                                 f"size, so this graph cannot be reshaped")

    metadata = {entry.key: entry.value for entry in source.metadata_props}
    shape_in = graph.input[0].type.tensor_type.shape
    shape_out = graph.output[0].type.tensor_type.shape
    scale = shape_out.dim[2].dim_value // shape_in.dim[2].dim_value
    step = int(metadata.get("step", TILE_STEP))
    halo = int(metadata.get("halo", (shape_in.dim[2].dim_value - step) // 2))

    shape_in.dim[2].dim_value = height // scale + 2 * halo
    shape_in.dim[3].dim_value = width // scale + 2 * halo
    shape_out.dim[2].dim_value = shape_in.dim[2].dim_value * scale
    shape_out.dim[3].dim_value = shape_in.dim[3].dim_value * scale

    # Every intermediate shape in the file was inferred for the tile. Drop them and infer
    # again, rather than leaving a graph whose middle still claims to be tile-sized.
    del graph.value_info[:]
    target = models_dir / filename
    reshaped = onnx.shape_inference.infer_shapes(source, strict_mode=True)
    onnx.checker.check_model(reshaped)
    onnx.save(reshaped, str(target))

    label = metadata.get("label", target.stem)
    write_metadata(target, {**metadata, "cpu_ms": None,
                            "label": f"{label} \u00b7 whole frame",
                            "step": width // scale, "halo": halo})
    return describe(target)


def annotate(info, **fields):
    """Add measurements to an exported model, after they have been taken.

    `cpu_ms` is the one the page shows: what this graph costs per tile on the desktop CPU
    runtime, beside the browser number the page measures itself.
    """
    write_metadata(info["path"], fields)
    return info
