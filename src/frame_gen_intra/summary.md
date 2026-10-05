# Frame generation, intra — summary

Interpolation: given two frames the stream delivered, generate the frame that belongs
between them, so a low frame rate stream can be played back at a higher one.

> **Status: nothing implemented.** No notebooks, no model, no data — this folder is empty
> apart from this file. What follows is the problem statement, not a plan that has been
> decided.

## What makes it its own problem

Both endpoints are known, so the answer is bounded on either side and the model is filling
in rather than guessing. That is what makes it the easier of the two frame generation
problems, and why it is worth doing first even though the live stream cannot use it as it
stands: holding a frame back until its successor arrives costs a full frame interval of
latency, which suits recorded or buffered playback and not a remote desktop session.

Screen content is also unlike the video these models are usually built for. Motion is
mostly whole rectangles translating — a scrolling page, a dragged window — with hard edges
and text that either lands on a pixel grid or turns to mush, and long stretches of the
frame do not change at all. An optical flow that assumes smooth natural motion has the
wrong prior for this.

## Shape it would take

The same shape as `../upscale/`: a `data_preprocess.ipynb` that cuts training examples out
of source frames and writes `.npy` arrays plus a manifest, then a notebook that trains a
deliberately minimal baseline against a trivial one. Here an example is a triplet — frames
*t*, *t+1*, *t+2* — where the middle frame is the target and the outer two the input, and
the baseline to beat is showing frame *t* unchanged, or a straight blend of the two.

The split has to be by *source clip*, for the same reason `../upscale/` splits by source
frame: consecutive triplets overlap, so splitting after cutting would put near-copies of
the training data into validation.

Sibling problems: `../upscale/summary.md`, `../frame_gen_extra/summary.md`.
