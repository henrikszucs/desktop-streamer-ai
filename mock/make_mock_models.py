"""The three mock graphs the client's enhancement menu runs.

They are stand-ins, not models: each one is the smallest ONNX graph that has the *shape*
of the real thing - the same input, the same output, real GPU work in between - so the
client pipeline (frame in, tensor on the GPU, session run, picture out) can be built and
timed before any trained weights exist. What they compute is chosen so the picture stays
right: a client running all three with these graphs shows the stream it would show
without them, only later.

    upscale.onnx      [1, 3, H, W] -> [1, 3, 2H, 2W]   a 3x3 identity conv, then bilinear x2
    interpolate.onnx  [1, 6, H, W] -> [1, 3, H, W]     the mean of the two frames, a 1x1 conv
    extrapolate.onnx  [1, 6, H, W] -> [1, 3, H, W]     2*B - A clipped to [0, 1], a 1x1 conv

Every graph takes float32 NCHW in [0, 1] with H and W dynamic, so one file serves every
resolution the stream arrives at, and every operator in them is one the WebGPU and the WebGL
execution providers of ONNX Runtime Web both run (`Resize`, `Conv`, `Clip`). The identity
convolution in the upscaler is there for the work: a bare `Resize` measures the copy in and
out rather than the runtime. It runs *before* the resize, at the input resolution, the way
a real upscaler computes its features low and upsamples last - the same convolution after
the resize cost three times the whole frame on the WebGPU provider, for nothing a mock has
to show.

The client never hands a graph a whole frame: it cuts the picture into 328x188 tiles (a
320x180 step and a halo of 4 pixels every model is given beyond it) and merges the kept
centres back - see `src/client/web/src/room/stream-enhance.js`. A model exported for the
client has to be right on a tile of that size and to read no further than the halo.

Run from `model/` and the files land where the client reads them:

    uv run mock/make_mock_models.py
"""

# internal
from pathlib import Path

# third-party
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# where the web client reads them: media, since a graph is an asset like an image
OUT_DIR = Path(__file__).resolve().parents[2] / "src" / "client" / "web" / "media" / "models"
OPSET = 17


def save(graph, name):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    model.producer_name = "desktop-streamer mock"
    model.metadata_props.add(key="label", value=name)
    model.metadata_props.add(key="mock", value="1")
    onnx.checker.check_model(model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.onnx"
    onnx.save(model, str(path))
    print(f"{path}  {path.stat().st_size} bytes")


def make_upscale():
    # a 3x3 convolution whose kernel is the identity - centre 1 on the channel's own
    # plane, 0 everywhere else - then bilinear x2, so the output is the resize and the GPU
    # still runs a convolution over every input pixel
    weight = np.zeros((3, 3, 3, 3), dtype=np.float32)
    for channel in range(3):
        weight[channel, channel, 1, 1] = 1.0
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "weight"], ["features"],
                             kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("Resize", ["features", "", "scales"], ["output"],
                             mode="linear", coordinate_transformation_mode="half_pixel"),
        ],
        "upscale_mock",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "H", "W"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H2", "W2"])],
        initializer=[
            numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), "scales"),
            numpy_helper.from_array(weight, "weight"),
        ],
    )
    save(graph, "upscale")


def make_blend(name, weight_a, weight_b):
    # two frames stacked on the channel axis - A in 0..2, B in 3..5 - mixed per channel by
    # a 1x1 convolution, then clipped: the extrapolation can leave [0, 1], the mean cannot
    weight = np.zeros((3, 6, 1, 1), dtype=np.float32)
    for channel in range(3):
        weight[channel, channel, 0, 0] = weight_a
        weight[channel, channel + 3, 0, 0] = weight_b
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "weight"], ["mixed"], kernel_shape=[1, 1]),
            helper.make_node("Clip", ["mixed", "low", "high"], ["output"]),
        ],
        f"{name}_mock",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 6, "H", "W"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])],
        initializer=[
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), "low"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), "high"),
        ],
    )
    save(graph, name)


if __name__ == "__main__":
    make_upscale()
    make_blend("interpolate", 0.5, 0.5)
    make_blend("extrapolate", -1.0, 2.0)
