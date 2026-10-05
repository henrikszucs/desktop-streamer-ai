# Frame generation, extra — summary

Extrapolation: given the frames the stream has delivered so far, generate the next one
before it arrives, so a client can keep drawing through a stall and a dropped frame does
not read as a freeze.

> **Status: nothing implemented.** No notebooks, no model, no data — this folder is empty
> apart from this file. What follows is the problem statement, not a plan that has been
> decided.

## What makes it its own problem

This is the one a live remote desktop session can actually use: it adds no latency, because
it needs nothing that has not arrived yet. It is also the harder of the two. The answer is
bounded on one side only, so the model is guessing rather than filling in, and it is wrong
whenever the user does something unpredictable — which on a desktop is most of what they
do. A click that opens a menu is unforeseeable from the frames before it.

Being wrong here is visible in a way the other two problems are not: an extrapolated frame
is shown and then contradicted by the real one. That makes the failure mode — how the guess
is corrected when the true frame lands, and whether a wrong guess is worse than a repeated
frame — a bigger question than the model architecture.

The cursor is the case worth watching. It moves constantly and predictably, and dragging it
smoothly through a stall is most of the perceived benefit, so a model that gets nothing else
right but the pointer may still be worth shipping.

## Shape it would take

The same shape as `../upscale/`: a `data_preprocess.ipynb` writing `.npy` arrays plus a
manifest, then a notebook training a minimal baseline. An example is a run of consecutive
frames where the last is the target and the ones before it the input, and the baseline to
beat is repeating the newest frame — which on a static desktop is already close to perfect,
so any metric has to be reported over the frames that actually changed or it will flatter
everything.

The split has to be by *source clip*, for the same reason `../upscale/` splits by source
frame: consecutive runs overlap, so splitting after cutting would put near-copies of the
training data into validation.

Sibling problems: `../upscale/summary.md`, `../frame_gen_intra/summary.md`.
