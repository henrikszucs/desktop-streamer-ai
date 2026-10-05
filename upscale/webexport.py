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
import math
from pathlib import Path

# third-party
import numpy as np
import onnx
import torch

# Where the page looks. Exports land here directly, and the server lists whatever it
# finds, so publishing a model is writing the file and nothing else.
MODELS_DIR = Path("benchmark/www/models")

# The tile the page benchmarks: a model keeps this many LR pixels per step, and takes
# `TILE_STEP + 2 * halo` in, so the halo it needs is part of its own geometry. It is the
# step every model on the page is *compared* at, not the step a client should run - which
# is a different question, asked and answered in `upscale_geometry.ipynb`.
TILE_STEP = 128

# The tile area a client should aim at, in *output* pixels, which is what `plan_tiling`
# fits a frame's own divisors around. It is the tile size, and it is not the same question
# as TILE_STEP above: that one is what the page compares models at, this one is what a
# client runs.
#
# 640x360 out, from a 320x180 step and a 328x188 input, measured in
# `upscale_geometry.ipynb`: the cheapest tiling of a 1080p frame on the card it was
# measured on, and one that divides *every* 16:9 resolution exactly - 720p is 2x2 of it,
# 1080p 3x3, 1440p 4x4, 4K 6x6 - so the same session serves all of them with nothing
# computed off the edge of any.
TARGET_KEPT_PX = 640 * 360

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


def as_size(size):
    """A tile size as (height, width). A square one may be given as a single number.

    A tile does not have to be square, and the useful ones often are not: a step that
    divides the frame in both dimensions leaves nothing computed off the edge of it, and
    1920x1080 has no square step that does - see `upscale_geometry.ipynb`.
    """
    if isinstance(size, (tuple, list)):
        height, width = size
        return int(height), int(width)
    return int(size), int(size)


def geometry(step, halo, scale=2, batch=1):
    """Every size of one run, from the step it is asked to keep.

    A tile has three sizes and confusing them is the usual bug: the **step** it contributes
    to the picture, the **input** it takes (the step plus its halo at both edges), and the
    **kept** centre of its output. A graph's input shape is not a free choice - it is the
    step plus the halo the architecture needs - so this is the one place either of them is
    turned into a shape, for the exporter and for whatever ends up driving inference.

    `step` is (height, width), or one number for a square tile. The halo is the model's:
    one pixel per 3x3 convolution in its path, which `upscale_geometry.ipynb` measures
    rather than assumes.
    """
    step_h, step_w = as_size(step)
    input_h, input_w = step_h + 2 * halo, step_w + 2 * halo
    kept = batch * step_h * step_w * scale * scale
    computed = batch * input_h * input_w * scale * scale
    return {
        "step": (step_h, step_w),
        "halo": halo,
        "scale": scale,
        "batch": batch,
        "input": (batch, 3, input_h, input_w),
        "output": (batch, 3, input_h * scale, input_w * scale),
        "kept_hr": (step_h * scale, step_w * scale),
        "kept": kept,
        "computed": computed,
        # What the halo costs, as a multiplier on everything the tile does. It falls as the
        # step grows, which is half of why a bigger tile looks cheaper than it is.
        "halo_tax": computed / kept,
    }


def tiling(frame, kept_hr, batch=1):
    """How a frame is covered by a tile that keeps `kept_hr`, and what that wastes.

    Every tile runs at the same shape, edges included - a static shape is the only thing
    graph capture will record, and a second session for the edges is a second compile, a
    second set of buffers and a second thing to keep in step. So the remainder of a frame
    that the tile does not divide is computed and thrown away, and the short last batch
    runs full for the same reason. That waste is invisible to any measurement of a single
    tile, which is why it is computed here and reported beside every timing.
    """
    width, height = frame
    # Every shape here is (height, width), tensor order, and `kept_hr` is one of them - a
    # frame is the one thing written the way a person says it.
    kept_h, kept_w = kept_hr
    cols, rows = -(-width // kept_w), -(-height // kept_h)
    tiles = cols * rows
    runs = -(-tiles // batch)
    return {
        "cols": cols,
        "rows": rows,
        "tiles": tiles,
        "runs": runs,
        "tiles_run": runs * batch,
        "delivered": width * height,
        "tiling_tax": runs * batch * kept_w * kept_h / (width * height),
        "exact": cols * kept_w == width and rows * kept_h == height and tiles % batch == 0,
    }


def exact_tilings(frame, scale=2, halo=4, max_aspect=2.5):
    """Tile sizes that cover this frame with nothing computed off its edge.

    Nothing requires a tile to be square - a convolution does not care, ONNX does not care,
    WebGPU does not care - and the frames a client is asked for have no useful square tile
    that divides them. 1920x1080 would need one dividing gcd(1920, 1080) = 120, which is a
    60 pixel step, small enough that the fixed cost of a run is most of what a frame costs.
    Allow a rectangle and 1080p has plenty: 384x270, 480x360, 640x540.

    A candidate divides the frame in both dimensions and is a multiple of the scale in
    both, so its LR step is a whole number of pixels. Elongated ones are dropped: an
    8x1920 strip divides the frame perfectly and is nearly all halo.
    """
    width, height = frame
    candidates = []
    for kept_w in range(scale, width + 1, scale):
        if width % kept_w:
            continue
        for kept_h in range(scale, height + 1, scale):
            if height % kept_h or max(kept_w / kept_h, kept_h / kept_w) > max_aspect:
                continue
            shape = geometry((kept_h // scale, kept_w // scale), halo, scale)
            candidates.append({**shape, "kept_px": kept_w * kept_h,
                               "tiles": (width // kept_w) * (height // kept_h)})
    candidates.sort(key=lambda row: row["kept_px"])
    return candidates


def plan_tiling(frame, scale=2, halo=4, target_kept=TARGET_KEPT_PX, batch=1,
                max_aspect=2.5, tolerance=4.0):
    """The geometry to run this frame at: exact if the frame admits one, padded if not.

    `target_kept` is the tile area the timing argues for, in output pixels; this only finds
    the nearest geometry the frame actually admits. Nearest in the ratio rather than in the
    difference, because the cost curve is flat in the log of the tile area and steep at the
    small end of it.

    An exact tiling is only worth having if it is near the size that was asked for.
    1366x768 has exactly one candidate that is not a sliver - the whole frame as a single
    tile - and running a frame in one dispatch to avoid a 20% edge is a bad trade at any
    tile size a client would choose. So a candidate more than `tolerance` times off the
    target in area is not one, and the padded square is taken instead.
    """
    exact = [row for row in exact_tilings(frame, scale, halo, max_aspect)
             if 1 / tolerance <= row["kept_px"] / target_kept <= tolerance]
    if exact:
        # Both terms are log-multipliers on the cost of a frame: how far the tile is from
        # the area the timing wants, and what its own halo costs. The second is what keeps
        # a 240x432 from being chosen over a squarer tile of the same area - a halo is
        # charged on the perimeter, so among equal areas the squarest tile computes least.
        best = min(exact, key=lambda row: abs(math.log(row["kept_px"] / target_kept))
                   + math.log(row["halo_tax"]))
        return {**best, **tiling(frame, best["kept_hr"], batch), "padded": False}

    # A frame with no exact tiling - an odd width, a resolution nothing divides - still has
    # to be covered, so the nearest square step is taken and the edge is paid for.
    side = max(scale, int(round(math.sqrt(target_kept) / scale)) * scale)
    shape = geometry(side // scale, halo, scale, batch)
    return {**shape, "kept_px": side * side,
            **tiling(frame, shape["kept_hr"], batch), "padded": True}


def export(model, filename, size, label=None, halo=None, step=TILE_STEP, batch=1,
           dynamic=False, opset=17, models_dir=MODELS_DIR):
    """Write `model` into the page's model folder as ONNX at a `size` input.

    `size` is the *model input*: the step this tile contributes plus its halo at both
    edges, as one number or as (height, width). `batch` is how many of those tiles one run
    takes, and it is part of the shape rather than a way of running the graph - a session
    with a static batch is the only one graph capture will record, so a client that batches
    its dirty tiles runs a graph exported for that batch.

    `dynamo=False` selects the older TorchScript exporter deliberately: it emits the plain,
    predictable op sequence these models are designed around, and the newer path needs
    `onnxscript`, which is not a dependency here.
    """
    height, width = as_size(size)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    torch.onnx.export(
        model.to("cpu").eval(), (torch.randn(batch, 3, height, width),), str(path),
        opset_version=opset, input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}} if dynamic else None,
        dynamo=False,
    )

    # The halo is derivable from the input shape and the step, and the server derives it
    # when it is absent - but a model that states its own geometry survives being renamed.
    # The step is then derived back from the halo rather than taken on trust, so a graph
    # cannot claim a step its own input shape does not have.
    halo = (width - step) // 2 if halo is None else halo
    write_metadata(path, {"label": label, "step": width - 2 * halo,
                          "step_h": height - 2 * halo, "halo": halo, "batch": batch})

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
