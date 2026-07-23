# Detailed changes from the forked baseline

## Scope and comparison point

This document describes the functional differences between:

- **Baseline:** `forked`, commit `15ce55f919eed7e986c7a9c79eb59ef41f39e1c8`
  from `aaronhalpern/MERlin:gpu_decoding`.
- **Documented code snapshot:** commit
  `a81de98e618acb253c5865ea952f964bc763d373` on `main`.

## 1. Decoding pipeline

### `merlin/analysis/decode.py`

The `Decode` task was expanded to support larger images, partial runs, and
additional output products.

Notable behavior and parameters include:

- `use_gpu` selects the PyTorch decoding path when available.
- `decode_chunk_size` bounds the number of pixel traces processed in one
  similarity calculation.
- `tiling_factor` divides an image into overlapping tiles to reduce peak
  memory use.
- `num_threads` controls tile-processing concurrency.
- `decode_z_index` limits a run to one z plane; the default processes all
  planes.
- `crop_in_image_space` applies the configured crop directly to decoded image
  dimensions.
- `magnitude_threshold`, `distance_metric`, `softmax_temperature`, and
  `nn_algorithm` expose additional decoder controls.
- `extract_intensity_traces` optionally stores per-barcode intensity traces.
- `write_unique_id_images` adds a unique-barcode-label output channel.
- `write_decoded_z` limits persisted decoded-image output to selected planes.
- Decoded images can be written incrementally to chunked Zarr arrays, allowing
  previously completed z planes to be reused.
- Tiled decoding performs barcode extraction per tile, removes overlap-region
  duplicates, remaps local labels to globally unique IDs, and returns
  coordinates in the full image frame.

### `merlin/util/decoding.py`

`PixelBasedDecoder` now contains explicit chunked similarity implementations:

- `_decode_pixels_by_similarity_numpy` performs chunked matrix similarity on
  the CPU.
- `_decode_pixels_by_similarity_torch` performs the same calculation through
  PyTorch and falls back to NumPy if PyTorch is unavailable.
- Optional softmax top-1 probabilities are computed from codebook
  similarities.
- Image tiling, overlap buffers, decode masks, callback-based per-tile
  extraction, and optional omission of the full normalized-pixel-trace array
  reduce memory pressure.
- Timing information is retained in `last_decode_timings`.
- Barcode extraction and refactor calculations were extended to support the
  new outputs and tiled execution path.

## 2. Barcode filtering

### `merlin/analysis/filterbarcodes.py`

Adaptive threshold selection was made more explicit and robust:

- Finite blank-fraction threshold candidates are extracted safely.
- Cumulative blank/coding curves identify achievable thresholds around a
  target misidentification rate.
- `threshold_solver_method` supports the cumulative-bin solver as well as the
  earlier numerical approach.
- `intensity_transform` supports `log10` and `linear` intensity spaces.
- `overshoot_toward_target`, `overshoot_tolerance`, and
  `report_bracketing_thresholds` control and report how discrete histogram
  thresholds approach the target.
- The same threshold logic is used by global and local adaptive filtering.

`LogisticFilterBarcodes` is a new analysis task:

- Fits a regularized logistic model using barcode intensity, decoding
  distance, and area.
- Standardizes model features and supports configurable
  `l2_regularization` and `max_iterations`.
- Chooses a probability cutoff using the requested
  `misidentification_rate`.
- Saves a per-fragment `logistic_filter_summary` with training and output
  counts, blank/coding counts, coefficients, and selected threshold.

Both adaptive and logistic filtering can optionally remove likely
z-duplicated detections using:

- `remove_z_duplicated_barcodes`
- `z_duplicate_zPlane_threshold`
- `z_duplicate_xy_pixel_threshold`

## 3. Preprocessing and restoration

### `merlin/analysis/preprocess.py`

Preprocessing tasks gained:

- Configurable global-background subtraction through
  `threshold_subtract_n` and `threshold_subtract_mode`.
- Subtraction modes based on the image mean, standard deviation, or their sum.
- Additional high-pass and low-pass helper paths.
- A reversed preprocessing path for workflows that require a different
  operation order.
- Updates to deconvolution preprocessing and z-plane selection.

### `merlin/analysis/modelrestore.py`

`ModelRestorePreprocess` is a new environment-specific task for restoring all
MERFISH bit channels jointly with a trained soft-decoding model.

The task:

- Reads warped bit images for a field of view and z plane.
- Applies chromatic correction, high-pass filtering, scale-factor correction,
  and per-FOV normalization matching the model-training pipeline.
- Runs a PyTorch checkpoint on the multi-channel stack.
- Converts predictions back to the high-pass, pre-scale-factor space expected
  by downstream MERlin optimization and decoding.
- Caches the most recently restored `(fov, z)` stack.

Important limitation: this module currently adds
`/n/home08/jiahaozhang/merfish_decode_transfer_pkg` to `sys.path` and imports
helpers from that external package. It also requires a checkpoint and
calibration assets. The task is therefore not self-contained or portable
without additional configuration.

### `merlin/util/deconvolve.py` and `merlin/util/imagefilters.py`

- Deconvolution utilities include a Gaussian-kernel helper and updated
  Lucy–Richardson/Guo processing.
- High-pass filtering is exposed as a shared image utility.

## 4. Repository-level changes

Relative to the baseline:

- Legacy `merlin/util/legacy.py` and `merlin/util/matlab.py` modules were
  removed.
- The inherited CircleCI, Codecov, and pep8speaks configuration was removed.
- Generated `merlin.egg-info` metadata was added.
- Several generated `__pycache__` files are currently tracked.
- Some files changed executable bits because the working copy moved between
  POSIX and Windows-backed filesystems; those mode changes do not describe
  functional behavior.

## Known caveats

- The model-restoration task depends on an external, user-specific package
  path and model assets.
- PyTorch is used by optional GPU/model paths but is not pinned in
  `requirements.txt`; install the build appropriate for the target CUDA
  environment.
- Packaging metadata still reports MERlin version `0.1.6`; `setup.py` also
  labels the license as restricted even though `license.md` contains the MIT
  License. The inherited package metadata does not fully describe this
  research fork.
- The inherited documentation predates several changes listed here.
- Automated CI configuration is not currently present.
- This change record is based on source and history inspection; it is not a
  claim that every task and environment-specific path has been validated.

## Reproducing the comparison

```bash
git log --oneline main
git log --oneline forked
git diff --stat forked main
git diff forked main -- merlin
```
