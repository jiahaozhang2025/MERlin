# Detailed changes from the forked baseline

## Scope and comparison point

This document describes the functional differences between:

- **Baseline:** `forked`, commit `15ce55f919eed7e986c7a9c79eb59ef41f39e1c8`
  from `aaronhalpern/MERlin:gpu_decoding`.
- **Documented code snapshot:** the current `main` branch of
  `jiahaozhang2025/MERlin`.

## Latest update (August 10, 2026)

This update focuses on making preprocessing, optimization, and decoding agree
on the same image data while reducing repeated downstream recomputation.

- Image filtering is now owned by preprocessing instead of being split between
  preprocessing and decode/optimize. New preprocessing controls include
  `fft_highpass_sigma`, `lowpass_sigma`, and `preprocess_threads`.
- Decode and optimize both gained `tile_overlap` and `adaptive_crop` so
  per-FOV invalid margins can be trimmed without paying the worst-case crop on
  every field of view.
- Decode now reuses the optimize stage's previous chromatic corrector, which
  matches the corrector used when scale factors were estimated and avoids
  triggering late chromatic re-estimation from decode jobs.
- Optimize can aggregate its per-fragment outputs in `finalize()`, cache merged
  barcode counts, optionally clean fragment intermediates, and estimate
  chromatic shifts from per-fragment samples using `chromatic_threads`,
  `chromatic_max_barcodes_per_group`, `chromatic_max_groups`,
  `chromatic_from_fragments`, `chromatic_on_preprocessed`, and
  `cleanup_fragment_results`.
- Adaptive barcode filtering can now write post-filter decoded images with
  `write_filtered_images`, `write_filtered_FOVs`, and `write_filtered_z`.
- Two-channel Cellpose segmentation now exposes `cellpose_channels` and defaults
  to `[1, 2]`, fixing a path where the second channel was previously not passed
  to the model as intended.

## 1. Decoding pipeline

### `merlin/analysis/decode.py`

The `Decode` task was expanded to support larger images, partial runs, and
additional output products.

Notable behavior and parameters include:

- `decode_chunk_size` bounds the number of pixel traces processed in one
  similarity calculation.
- `tiling_factor` divides an image into overlapping tiles to reduce peak
  memory use.
- `num_threads` controls tile-processing concurrency.
- `decode_z_index` limits a run to one z plane; the default processes all
  planes.
- `crop_in_image_space` applies the configured crop directly to decoded image
  dimensions.
- `adaptive_crop` shrinks each decoded field of view to its own valid warped
  region, rather than using only a fixed global crop.
- `tile_overlap` keeps edge-spanning objects intact during tiled decode.
- `magnitude_threshold`, `distance_metric`, `softmax_temperature`, and
  `nn_algorithm` expose additional decoder controls.
- `extract_intensity_traces` optionally stores per-barcode intensity traces.
- `write_unique_id_images` adds a unique-barcode-label output channel.
- `write_decoded_z` limits persisted decoded-image output to selected planes.
- `lowpass_sigma` is no longer a decode parameter; filtering now belongs to
  the preprocess task so optimize and decode see the same pixels.
- Decoded images can be written incrementally to chunked Zarr arrays, allowing
  previously completed z planes to be reused.
- Tiled decoding performs barcode extraction per tile, removes overlap-region
  duplicates, remaps local labels to globally unique IDs, and returns
  coordinates in the full image frame.
- Image-space decoding now computes crop bounds per field of view and restores
  global coordinates with independent x/y crop offsets, fixing skew when the
  valid warp margin is asymmetric.
- Decode uses the optimize task's previous chromatic corrector, keeping decode
  aligned with the scale/background estimates from that optimize iteration.

### `merlin/analysis/optimize.py`

`OptimizeIteration` was extended to make optimization outputs cheaper to
produce and more consistent with the eventual decode step.

- `lowpass_sigma` is no longer accepted here; decode and optimize both rely on
  preprocess-owned filtering.
- `tile_overlap`, `adaptive_crop`, and `num_threads` align optimize-time pixel
  decoding with the decode task's tiled behavior.
- `finalize()` materializes chromatic corrections, scale factors, backgrounds,
  and merged barcode counts once per completed iteration instead of letting
  multiple downstream jobs recompute them on cache miss.
- `cleanup_fragment_results` can remove fragment-level `.npy` intermediates
  after merged outputs are safely written.
- Chromatic estimation can be measured inside fragments and pooled later via
  `chromatic_from_fragments`, with optional workload caps through
  `chromatic_max_barcodes_per_group` and `chromatic_max_groups`.
- `chromatic_threads` parallelizes per-group chromatic sampling, and
  `chromatic_on_preprocessed` allows those samples to be measured on the
  already filtered image stack instead of reloading raw-warped images.

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
- Barcode extraction now accepts a two-axis `crop_offset` tuple so global
  coordinates remain correct after asymmetric image-space cropping.

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

Adaptive filtering can also optionally write decoded images containing only
the barcodes that survived filtering:

- `write_filtered_images` enables the export.
- `write_filtered_FOVs` selects which fields of view should be written.
- `write_filtered_z` optionally limits output to selected z planes.

## 3. Preprocessing and restoration

### `merlin/analysis/preprocess.py`

Preprocessing tasks gained:

- Configurable global-background subtraction through
  `threshold_subtract_n` and `threshold_subtract_mode`.
- Subtraction modes based on the image mean, standard deviation, or their sum.
- A frequency-domain high-pass stage controlled by `fft_highpass_sigma`.
- Preprocess-owned `lowpass_sigma`, now defaulting to the value that decode
  previously assumed internally.
- Additional high-pass and low-pass helper paths.
- A reversed preprocessing path for workflows that require a different
  operation order.
- Threaded bit/z preprocessing through `preprocess_threads`.
- A no-op bypass for zero-iteration deconvolution calls, avoiding unnecessary
  Lucy-Richardson buffer allocation on large images.
- Updates to deconvolution preprocessing and z-plane selection.

## 4. Segmentation and restoration

### `merlin/analysis/segment.py`

`CellPoseSegmentTwoChannel3D` now exposes the Cellpose channel selector through
`cellpose_channels`, defaulting to `[1, 2]` for the stacked
`[channel_1, channel_2]` input. This replaces an earlier hardcoded `[0, 1]`
path that effectively collapsed the stack to grayscale and prevented the
second channel from being used as the intended nuclear input.

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

## 5. Repository-level changes

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
