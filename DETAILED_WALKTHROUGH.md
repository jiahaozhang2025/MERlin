# Detailed walkthrough of the major changes

Why each major change in [CHANGES.md](CHANGES.md) was made, compared with the
`gpu_decoding` baseline. Minor changes are self-explanatory and listed there only.

## Decode

**`distance_metric: dot_product`** (default). MERlin decodes a pixel by
normalising its intensity trace to unit length and finding the nearest codeword.
For unit vectors the Euclidean distance is `sqrt(2 - 2 cos θ)`, so the codeword
with the largest dot product is exactly the nearest one. Decoding therefore
becomes one matrix multiplication of pixels by codewords, run through torch or
BLAS, instead of a scikit-learn nearest-neighbour search. The assignments and
distances are the same as before; the computation is much faster and runs in
parallel.

**`decode_chunk_size`**. The dot product creates a pixels x codewords similarity
matrix. Processing the pixels in chunks caps that matrix's size, so memory stays
bounded on large images and many concurrent planes. The value affects speed and
memory only, never the result.

**`softmax`, `softmax_dot_product`, `softmax_temperature`**. Besides the best
codeword, these modes compute a softmax over the dot products, which gives each
pixel the probability of its top codeword. `softmax_temperature` sets how sharp
that distribution is. This gives a per-pixel confidence instead of a single
distance, for workflows that weigh calls by confidence or train models on them.

**`adaptive_crop`** (default on). After registration, each FOV has margins where
the warped image has no data, because the round was shifted relative to the
reference. The baseline excluded one fixed `crop_width` from every edge of every
FOV, which is too little where the shift is large (empty pixels get decoded) and
too much where it is small (valid pixels are thrown away). `adaptive_crop`
reads each FOV's own transformations and excludes exactly its invalid region.

## Registration

**`FiducialPolynomialWarp3D`** (replaces `FiducialCorrelationWarp3D`). In thick,
expanded tissue, the drift between rounds is not one offset: the gel swells
or shrinks between rounds, so z shifts by an amount that grows with depth, and
xy drifts by different amounts at different depths. The removed task measured
the bead stack once per round and spread the result linearly through depth.
The new task measures the z offset plane by plane by correlating the bead
stacks, fits it as a polynomial in depth, then measures xy at each corrected
depth and fits that as a polynomial too. Plain cross-correlation is used
throughout, with searches bounded around the expected answer. The rigid offset
is refined through the stack, and fits with large residuals are flagged. When a
stack has too few usable bead samples, it borrows the curve of the same round
from neighbouring FOVs instead of fitting noise.

**`CrossFOVCompositeWarp`** (from Lida Cheng). When the stage drifts between
rounds, part of a FOV's field in a late round was imaged by the neighbouring FOV
instead. Per-FOV registration can only mark that strip invalid, so its molecules
are lost even though the pixels exist. This task links each FOV to its
neighbours through their round-0 beads, measured with a self-test. It fills the
invalid strip with the neighbour's registered image and tells Decode to crop
only what still could not be filled.

## Optimize

**`finalize()`**. Each optimize iteration's scale factors, backgrounds,
barcode counts and chromatic corrections used to be computed lazily, on the
first request. That came when the next iteration's fragments started, all at
once, so every fragment missed the cache and recomputed them, the chromatic fit
included. `finalize()` computes them once when the iteration completes, and
everything downstream reads the saved result.

**`chromatic_from_fragments`** (default on). The chromatic correction used to be
fitted in one job after the iteration, which reloaded every image and barcode.
Each fragment already holds its FOV's images and barcodes, so it now saves raw
displacement samples, and `finalize()` pools them into a single fit. That gives
the same fit as one job. Averaging per-fragment fits instead would let fragments
with few barcodes distort the rotation and scale.

**`chromatic_on_preprocessed`**. Measures those samples on the preprocessed
images already in memory instead of loading the warped images again. It is
faster, but not equivalent: preprocessing clips negative values, which shifts
spot centroids slightly. It is therefore off by default.

**`chromatic_threads`**. When `chromatic_from_fragments` is off and the
correction is fitted in one job, this parallelises that job over FOV/z planes.

**`OptimizeLoop`**. A barcode optimize run is a chain of `OptimizeIteration`
tasks, each starting from the previous one's scale factors. Writing ten of them
by hand in the analysis json is long and error-prone. `OptimizeLoop` declares
the chain once and expands into the same ordinary iteration tasks, so nothing
downstream changes. `per_iteration` varies a setting between iterations.
`num_workers` runs fragments in parallel processes.

**`ImageScaleFactors`**. The barcode optimizer sets each bit's scale factor so
that the mean intensity of decoded "on" pixels matches across bits. For a
genuinely dim bit, that mean includes background that the decoder itself pulled
in. The loop answers by lowering the factor, which inflates the bit and pulls in
more background. It can then converge to a stable but wrong value, and the
codewords containing that bit get over-called. Blank-based filtering cannot
detect this, because the extra calls look like real ones. `ImageScaleFactors`
estimates each bit's signal amplitude and background directly from the
preprocessed images, with no decoding, so there is no feedback loop.
It can replace the optimize task for Decode, or seed a short barcode-based
refinement. It also records how much each bit's estimate moves across a range
of settings, so its sensitivity to them is measured on every run.

**Per-bit bias freeze**. The barcode optimizer does not settle on a fixed answer
for a dim bit: every round pushes it further in the same direction, so more
iterations are not better. Given a reference of expected gene abundances, each
round estimates every bit's bias, i.e. how far genes using that bit are over-
or under-called against the reference. If an update made a bit's bias worse, it
is rolled back for that bit only. Bits that are still improving keep updating,
and bits that have turned are held in place.

## Preprocess

**`fft_highpass_sigma`**. A high-pass filter applied in the frequency domain,
before the usual spatial high pass. It removes broad, slowly varying background
across the whole field at a cost that does not grow with the filter width. This
also lets MERlin reproduce filter chains of the form FFT high pass -> spatial
high pass -> low pass.

## Barcode filtering

**`threshold_solver_method: cumulative_bins`** (default). The adaptive filter
bins barcodes by intensity, distance and area, ranks the bins by blank
fraction, and keeps the best bins up to the target misidentification rate.
Because whole bins are kept, the rate as a function of the threshold is a step
function. The baseline searched it with Newton's method, which assumes a smooth
function, so on a step function it can stall on a flat step or jump past the
target.
`cumulative_bins` sorts the bins by blank fraction and adds them one at a time,
which finds the largest set of bins at or below the target directly and the
same way every time.

**`overshoot_toward_target`, `overshoot_tolerance`**. With whole bins the exact
target is usually unreachable: the last bin below it can leave the rate well
under target. These options allow the next bin when it lands closer to the
target, as long as it stays within the tolerance above it.

**`intensity_transform`**. The baseline always binned log10 intensity, which
spreads the bins evenly over the large intensity range. That remains the
default; `linear` is available as an alternative.

**`LogisticFilterBarcodes`, `l2_regularization`, `max_iterations`**. A
per-FOV filter that fits a logistic regression to separate blank from coding
barcodes using mean intensity, distance and area. It then keeps barcodes below
the blank-probability cut that meets the target misidentification rate. It is a
smooth model with three coefficients instead of a histogram with thousands of
bins, so it does not depend on bins being well populated. It needs no separate
threshold task. `l2_regularization` and `max_iterations` control the fit.

## Example images

Every run now leaves a few images to inspect without anyone having to ask:
preprocessed images, decoded images and filtered images. All three tasks draw
the same few FOVs and z plane from `write_images_seed`, so the same field can be
compared before decoding, after decoding and after filtering. Images used to be
off by default or written for every FOV, which was either nothing to look at or
an enormous amount of output. Decoded images are now plain 2D TIFFs per channel
that open directly in Fiji. The older zarr format remains readable.

## Grouped output layout

A dataset's analysis folder used to hold every task folder and every dataset
file side by side, which becomes hard to navigate with dozens of tasks. New
datasets group task folders by pipeline stage (Prepare, Optimize, Decode,
Segment, Export, Other) and put codebooks, data organization and other dataset
files under `Files`. Existing flat datasets keep working unchanged, and
`merlin.util.regroup` converts one when wanted.

## Segmentation

**Cellpose-SAM**. Cellpose 4 (Cellpose-SAM) segments well across tissues and
stains without training a model per tissue, and it cannot load cellpose 2/3
models. Supporting both versions meant two code paths with different APIs, and
some of the old ones were silently broken. The segmentation tasks now use
cellpose 4 only, which keeps one maintained path; older segmentations are redone
with Cellpose-SAM.

**`z_index`**. Segmenting every plane and linking the planes in z adds
errors where cells are out of focus or split between planes. Many tissue
sections are thin compared with a cell, so one well-focused plane holds most
cells. `z_index` segments that one plane and uses its outlines on every plane,
so barcodes from all planes are assigned to the 2D cells. This is a common
convention for MERFISH tissue sections.

**`FilterCells`**. Segmentation also outlines objects that are not cells: tiny
fragments, thin slivers, and large empty shapes in background areas. Size and
shape alone cannot separate all of them from small real cells. Barcode content
can, but barcodes are only assigned to cells after partitioning. `FilterCells`
therefore runs after a provisional partition and removes objects that are too
small, too thin, have too few barcodes, or have too low a barcode density. It
saves a before/after mask and a per-object status table, so every removal can
be checked. Final partitioning then uses the kept cells.

## Stitched decoding (from Lida Cheng)

Decoding FOV by FOV loses or duplicates molecules near FOV edges: a spot cut by
the edge is decoded from partial image data, and overlapping FOVs decode the
same molecule twice. The stitched workflow first registers all FOVs and rounds
into one coordinate system using bead measurements. It then decodes the sample
as one volume, giving every molecule near a boundary the surrounding image from
the neighbouring FOVs. Each molecule belongs to exactly one output tile, and
duplicates across tiles and z are removed globally.

- **Required review**: registration errors spread silently into every later
  step, so decoding does not start until a reviewer accepts the registration
  evidence with a decision file. That file names the hash of the evidence that
  was reviewed.
- **Frozen inputs and code**: a full run is long. Each run copies its
  metadata, stage scripts and MERlin code at the start and checks their hashes
  before every stage, so editing the repository mid-run cannot change a run in
  progress. Every stage writes a receipt with output hashes, which lets a rerun
  skip finished work safely.
- **Validated profile only**: the registration and sampling steps were checked
  for one acquisition geometry. Other geometries are rejected until they are
  validated, rather than run on unchecked assumptions.
