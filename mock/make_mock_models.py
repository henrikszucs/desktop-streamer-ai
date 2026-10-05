"""The three mock graphs the client's enhancement menu runs.

They are stand-ins, not models: each one is the smallest ONNX graph that has the *shape*
of the real thing - the same input, the same output, real GPU work in between - so the
client pipeline (frame in, tensor on the GPU, session run, picture out) can be built and
timed before any trained weights exist. What they compute is chosen so the picture stays
right: a client running all three with these graphs shows the stream it would show
without them, only later.

    upscale.onnx      input [N, 3, H, W]                     -> output [N, 3, 2H, 2W]
                      a depthwise 3x3 identity conv, then bilinear x2
    interpolate.onnx  previous, current [N, 3, H, W] each    -> output [N, 3, H, W]
                      the mean of the two frames
    extrapolate.onnx  previous, current [N, 3, H, W] each    -> output [N, 3, H, W]
                      2 * current - previous clipped to [0, 1]

Every tensor is float32 NCHW in [0, 1] with N, H and W dynamic, so one file serves every
resolution the stream arrives at and every batch of tiles it is cut into, and every operator
in them is one the WebGPU and the WebGL execution providers of ONNX Runtime Web both run
(`Conv`, `Resize`, `Add`, `Mul`, `Clip`). The two-frame models take the two frames as **two
inputs** rather than one six channel tensor: stacked, the client paid a copy to stack them
and the graph paid a `Slice` to take them apart again - 11 of the 17 ms an interpolated
1080p frame cost, for nothing either side wanted.

The client never hands a graph a whole frame: it cuts the picture into 328x188 tiles (a
320x180 step and a halo of 4 pixels every model is given beyond it), runs *every tile of a
frame as one batch* and merges the kept centres back - see
`src/client/web/src/room/stream-enhance.js`. A model exported for the client has to be right
on a tile of that size, read no further than the halo, and take a batch.

The shapes are chosen by what the WebGPU provider runs well, measured on a 1080p frame
(36 tiles, one run) on an Apple 8-core GPU:

    generic 3x3 conv, 3 -> 3 channels     25 ms      no vectorised path for a channel count
                                                     that is not a multiple of 4
    1x1 conv, 3 -> 16 or 16 -> 3          17 ms      the same
    depthwise 3x3 conv (group = channels)  1.4 ms    its own kernel
    bilinear Resize x2                    11 ms      the floor, and what a real upscaler pays
    NCHW <-> NHWC Transpose pair           4.3 ms    inserted by the provider around every
                                                     graph's convolutions, once per graph
    Slice of a 6 channel input in two      9 ms      which is why there are two inputs
    Mul, Add, Clip over a frame            1-2 ms    each
    float16, graph capture                 no change - the kernels are index-bound, not
                                                     bandwidth- or arithmetic-bound

So the upscaler's work is a depthwise convolution before the resize (features low, upsample
last, as a real one does), and the two blends are the elementwise arithmetic they are rather
than a 1x1 convolution that says the same thing. A trained model will pay the runtime's
prices for whatever it is built from; these are the prices.

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
    # a depthwise 3x3 convolution (one kernel per channel, group = 3) whose kernel is the
    # identity - centre 1, 0 around it - then bilinear x2, so the output is the resize and
    # the GPU still runs a convolution over every input pixel
    weight = np.zeros((3, 1, 3, 3), dtype=np.float32)
    weight[:, 0, 1, 1] = 1.0
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "weight"], ["features"],
                             kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=3),
            helper.make_node("Resize", ["features", "", "scales"], ["output"],
                             mode="linear", coordinate_transformation_mode="half_pixel"),
        ],
        "upscale_mock",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H2", "W2"])],
        initializer=[
            numpy_helper.from_array(np.array([1, 1, 2, 2], dtype=np.float32), "scales"),
            numpy_helper.from_array(weight, "weight"),
        ],
    )
    save(graph, "upscale")


def make_blend(name, weight_a, weight_b):
    # the two frames as two inputs, mixed as weight_a * previous + weight_b * current, then
    # clipped: the extrapolation can leave [0, 1], the mean cannot
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["previous", "weight_a"], ["wa"]),
            helper.make_node("Mul", ["current", "weight_b"], ["wb"]),
            helper.make_node("Add", ["wa", "wb"], ["mixed"]),
            helper.make_node("Clip", ["mixed", "low", "high"], ["output"]),
        ],
        f"{name}_mock",
        [
            helper.make_tensor_value_info("previous", TensorProto.FLOAT, ["N", 3, "H", "W"]),
            helper.make_tensor_value_info("current", TensorProto.FLOAT, ["N", 3, "H", "W"]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H", "W"])],
        initializer=[
            numpy_helper.from_array(np.array(weight_a, dtype=np.float32), "weight_a"),
            numpy_helper.from_array(np.array(weight_b, dtype=np.float32), "weight_b"),
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), "low"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), "high"),
        ],
    )
    save(graph, name)


if __name__ == "__main__":
    make_upscale()
    make_blend("interpolate", 0.5, 0.5)
    make_blend("extrapolate", -1.0, 2.0)
