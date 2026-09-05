"""Server for the upscaling speed benchmark page.

    uv run upscale/benchmark/main.py        # then open http://127.0.0.1:8000

Needs fastapi[standard], which is what brings uvicorn along. It serves ./www, and it
answers GET /api/models with the list of models the page fills its dropdown from. It
exists because the page cannot run from file:// - ONNX Runtime Web fetches its own .wasm
at runtime, and a fetch from a file:// origin is blocked.

The model list is built from the graphs in www/models/ on every request, not from anything
a notebook wrote beside them. Almost all of it is already in the file - the input and
output shapes, the operators, the parameter count, the size on disk - and a manifest
repeating that is a second copy to keep in step, which is a copy that goes stale the first
time a model is re-exported and something else is not. What genuinely cannot be read out
of a graph is the label a human chose for it and the desktop CPU time a notebook measured;
those ride in the model's own `metadata_props`, so a model is still one file. Drop an
.onnx into www/models/ by hand and it appears in the dropdown; delete one and it goes.
"""

# internal
import argparse
import mimetypes
import re
import sys
import time
from pathlib import Path

# third-party
import onnx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# first-party
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import webexport

WWW = Path(__file__).parent / "www"
MODELS = WWW / "models"

# mimetypes reads the Windows registry, where .js has been known to come back as
# text/plain and .wasm as nothing at all. Either one breaks the page, so the types that
# matter are registered rather than looked up.
TYPES = {
    ".css": "text/css",
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".mjs": "text/javascript",
    ".onnx": "application/octet-stream",
    ".wasm": "application/wasm",
}

for suffix, media_type in TYPES.items():
    mimetypes.add_type(media_type, suffix)

# What the page can allocate a tensor for, at either end. A graph whose input is some
# other type is listed by nothing here - it would load and then be fed the wrong bytes -
# and one whose output is would have a buffer of the wrong size allocated for it.
DTYPES = {
    onnx.TensorProto.FLOAT: "float32",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.UINT8: "uint8",
}

# upscale_static_bicubic_tile128.onnx -> id "static-bicubic", step 128.
NAME = re.compile(r"^(?:upscale_)?(?P<id>.+?)(?:_tile(?P<step>\d+))?$")

app = FastAPI(title="Upscale benchmark", docs_url=None, redoc_url=None)

# Reading a graph means parsing its weights, which is wasted on a request that changed
# nothing. Keyed by path, invalidated by mtime and size, so a re-exported model is picked
# up on the next request and an unchanged one is never parsed twice.
cache = {}


def dimensions(value_info):
    """The static shape of a graph input or output, or None where an axis is symbolic."""
    shape = value_info.type.tensor_type.shape
    return [dimension.dim_value if dimension.HasField("dim_value") else None
            for dimension in shape.dim]


def read_model(path):
    """One dropdown row, out of the graph itself plus whatever the exporter left in it."""
    model = onnx.load(str(path))
    graph = model.graph
    metadata = {entry.key: entry.value for entry in model.metadata_props}

    shape_in = dimensions(graph.input[0])
    shape_out = dimensions(graph.output[0])
    elem_type = graph.input[0].type.tensor_type.elem_type
    out_elem_type = graph.output[0].type.tensor_type.elem_type
    for name, value in (("input", elem_type), ("output", out_elem_type)):
        if value not in DTYPES:
            # The page fills the input and allocates the output itself, so an element type
            # it cannot allocate is not something it can benchmark - say which, rather
            # than serving a row that fails.
            raise ValueError(f"{name} element type "
                             f"{onnx.TensorProto.DataType.Name(value)} is not one the "
                             f"page can allocate")
    if None in shape_in or len(shape_in) != 4:
        # The page allocates its input tensor from this shape, so a graph that does not
        # state one is not something it can run.
        raise ValueError(f"input shape is {shape_in}, expected four static dimensions")

    ops = {}
    for node in graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    parameters = 0
    for initializer in graph.initializer:
        count = 1
        for dimension in initializer.dims:
            count *= dimension
        parameters += count

    match = NAME.match(path.stem)
    identifier = match.group("id").replace("_", "-")
    step = int(metadata.get("step") or match.group("step") or webexport.TILE_STEP)

    # The output geometry is in the graph beside the input geometry, so it is read rather
    # than multiplied out of the input: a graph that casts on the way out, or that changes
    # channel count, hands the page a tensor of the size and type it will really be given
    # instead of one derived from the other end. Only a graph that does not state its
    # output shape falls back to the scale, which is the one thing that cannot be read.
    static_out = len(shape_out) == 4 and None not in shape_out
    scale = shape_out[2] // shape_in[2] if static_out else int(metadata.get("scale", 2))

    return {
        "id": identifier,
        "label": metadata.get("label") or identifier.replace("-", " "),
        "file": path.name,
        "input": shape_in,
        "output": shape_out if static_out else [shape_in[0], shape_in[1],
                                                shape_in[2] * scale, shape_in[3] * scale],
        # Read out of the graph rather than out of the filename: a half-precision export
        # is the same model at another precision, and the page allocates from this.
        "dtype": DTYPES[elem_type],
        "out_dtype": DTYPES[out_elem_type],
        "precision": metadata.get("precision", DTYPES[elem_type]),
        "scale": scale,
        "step": step,
        # The halo is whatever the model input carries beyond the step it contributes,
        # which is a fact about the exported shape rather than a note about it.
        "halo": int(metadata["halo"]) if "halo" in metadata else (shape_in[2] - step) // 2,
        "parameters": parameters,
        "kb": round(path.stat().st_size / 1024, 1),
        "ops": ops,
        "outside_budget": sorted(set(ops) - webexport.BUDGET),
        "cpu_ms": float(metadata["cpu_ms"]) if "cpu_ms" in metadata else None,
    }


def list_models():
    """Every model in www/models/, cheapest baseline first."""
    models = []
    for path in sorted(MODELS.glob("*.onnx")):
        stat = path.stat()
        key = str(path)
        stamp = (stat.st_mtime_ns, stat.st_size)
        if key not in cache or cache[key][0] != stamp:
            try:
                cache[key] = (stamp, read_model(path))
            except Exception as error:
                print(f"warning: skipping {path.name}: {error}")
                cache[key] = (stamp, None)
        if cache[key][1] is not None:
            models.append(cache[key][1])

    for key in [key for key in cache if not Path(key).exists()]:
        del cache[key]

    # A model with no weights is a resampling filter, and the useful order is the free
    # options first, then by how much a model costs to run.
    models.sort(key=lambda row: (row["parameters"] > 0, row["parameters"], row["id"]))
    return models


@app.get("/api/models")
def models_endpoint():
    """The dropdown, built from the folder on every request.

    Deliberately not `async`: a cold cache parses every graph in the folder, weights and
    all, and on the event loop that stalls the static files the page is fetching at the
    same time. Declared like this, Starlette runs it on the threadpool.
    """
    models = list_models()
    newest = max((path.stat().st_mtime for path in MODELS.glob("*.onnx")), default=None)
    return JSONResponse({
        "generated": (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(newest))
                      if newest else None),
        "models": models,
    })


@app.middleware("http")
async def no_store(request, call_next):
    """A benchmark re-runs the same model many times, and a cached 200 would time the
    browser cache rather than the model. A stale .onnx is worse still."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"

    # Cross-origin isolation is what lets ONNX Runtime Web use threaded WASM. It is off
    # because it also blocks the CDN that serves ort.all.min.js - turn these on together
    # with a vendored copy of the runtime, or leave both off.
    # response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    # response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return response


# Mounted last: a mount at "/" matches everything, so every route above has to be
# declared before it to be reachable at all.
app.mount("/", StaticFiles(directory=WWW, html=True), name="www")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    arguments = parser.parse_args()

    found = list_models()
    if found:
        print(f"{len(found)} model(s) in {MODELS}")
        for row in found:
            print(f"  {row['id']:<28} {row['input'][2]:>3} in   halo {row['halo']}   "
                  f"{row['precision']:<8} {row['kb']:>7.1f} KB")
    else:
        print(f"warning: no models in {MODELS} - run the export cell of a model notebook "
              f"to publish one, or the page will load with an empty list")

    print(f"serving {WWW}")
    print(f"open http://{arguments.host}:{arguments.port}  (ctrl-c to stop)")
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
