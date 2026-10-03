# Changes from the `gpu_decoding` fork

Compared with `aaronhalpern/MERlin:gpu_decoding` at commit
`15ce55f919eed7e986c7a9c79eb59ef41f39e1c8`. Contributed work is marked
(from ...). [DETAILED_WALKTHROUGH.md](DETAILED_WALKTHROUGH.md) explains why each
major change was made.

## Major changes

### Decode (`merlin/analysis/decode.py`, `merlin/util/decoding.py`)

- `distance_metric`: Defaults to chunked `dot_product` matrix-multiplication
  decoding, with optional `softmax` and `softmax_dot_product` modes.
- `decode_chunk_size`: Limits the number of pixel traces processed in each
  matrix multiplication to control memory use.
- `softmax_temperature`: Controls optional softmax top-1 probabilities for
  machine-learning and confidence-based workflows.
- `adaptive_crop`: Defaults to excluding each FOV's unaligned warp margins so
  invalid image regions do not contaminate barcode extraction.

### Registration (`merlin/analysis/warp.py`)

- `FiducialPolynomialWarp3D` (replaces `FiducialCorrelationWarp3D`): Measures
  each round's z and xy drift plane by plane on the fiducial bead stack and fits
  them as polynomials in depth.
- `CrossFOVCompositeWarp` (from Lida Cheng): Fills the margins that stage drift
  moved out of a FOV's frame with the same tissue imaged by the neighbouring FOV.

### Optimize (`merlin/analysis/optimize.py`)

- `finalize()`: Computes shared scale factors, backgrounds, barcode counts, and
  chromatic corrections once instead of repeating them in downstream tasks.
- `chromatic_from_fragments`: Defaults to collecting chromatic displacement
  samples during enabled chromatic optimization and pooling one final fit.
- `chromatic_on_preprocessed`: Reuses the preprocessed image stack for
  chromatic sampling instead of loading another warped stack.
- `chromatic_threads`: Parallelizes FOV/z chromatic sampling only when
  `chromatic_from_fragments` is disabled.
- `OptimizeLoop`: Declares a chain of optimize iterations once in the analysis
  json; it expands into ordinary `OptimizeIteration` tasks and runs them in order.
- `ImageScaleFactors`: New task that estimates per-bit scale factors and
  backgrounds from the preprocessed images alone, without decoding.
- Per-bit bias freeze (`bias_freeze_reference`): Rolls back a bit's scale-factor
  update when it moves the bit's gene-abundance bias further from a reference.

### Preprocess (`merlin/analysis/preprocess.py`)

- `fft_highpass_sigma`: Applies FFT-space high-pass filtering to remove broad
  background artifacts before decoding.

### Barcode filtering (`merlin/analysis/filterbarcodes.py`)

- `threshold_solver_method`: Adds cumulative-bin threshold selection for better
  control of the requested misidentification rate.
- `intensity_transform`: Selects linear or log10 intensity space for adaptive
  thresholding.
- `overshoot_toward_target`: Controls whether a discrete adaptive threshold may
  move toward the requested misidentification rate.
- `overshoot_tolerance`: Limits the permitted overshoot when selecting an
  adaptive threshold.
- `LogisticFilterBarcodes`: Adds logistic filtering based on barcode intensity,
  decoding distance, and area.
- `l2_regularization`: Controls regularization strength for the logistic filter.
- `max_iterations`: Limits logistic-model optimization iterations.

### Example images (Preprocess, Decode, adaptive filters; `merlin/util/imagesample.py`)

- Preprocess, Decode and the adaptive filters write example images by default,
  for the same few FOVs and one z plane drawn from `write_images_seed`.
- `decoded_image_format`: `tif` (default) writes one 2D file per channel and
  plane; `zarr` keeps one array per FOV. `Decode.get_decoded_image` reads both.

### Grouped output layout (`merlin/core/dataset.py`, `merlin/util/regroup.py`)

- New datasets put task folders under `Prepare`, `Optimize`, `Decode`,
  `Segment`, `Export` and `Other`, and the dataset files under `Files`.
- Older flat datasets are unchanged; `python -m merlin.util.regroup` converts one.

### Segmentation (`merlin/analysis/segment.py`)

- Cellpose-SAM replaces cellpose 2/3: the 3D cellpose tasks run cellpose >= 4
  (default model `cpsam_v2`); `CellPoseSegment` and `CellPoseSegmentSingleChannel` are removed.
- `z_index`: Segments one z plane and repeats its outlines on every plane.
- `FilterCells`: New task that removes non-cell objects after a provisional
  partition, by footprint area, width, barcode count and barcode density.

### Stitched decoding (`merlin/analysis/stitched.py`, `merlin/util/stitched/`) (from Lida Cheng)

- `StitchedInitialize`, `StitchedStage`, `StitchedFragments`: Decode a whole
  sample as one stitched volume, so molecules at FOV boundaries are decoded once.
- Requires a reviewed registration before decoding and runs each stage against
  frozen copies of its inputs and code; only the validated acquisition profile is accepted.

## Minor changes

### Decode (`merlin/analysis/decode.py`, `merlin/util/decoding.py`)

- `tiling_factor`: Divides large images into overlapping tiles to reduce peak
  decoding memory.
- `tiling_overlap`: Preserves barcodes crossing tile boundaries and supports
  overlap-duplicate removal.
- `tiling_num_threads`: Decodes the tiles of one z plane concurrently.
- `per_z_slice_num_threads`: Decodes z planes concurrently.
- `torch_num_threads`: Pins torch's thread pool while z planes run concurrently
  (default automatic), so the threads do not contend.
- `magnitude_threshold`: Filters low-magnitude pixels before barcode matching.
- `nn_algorithm`: Selects the scikit-learn nearest-neighbor algorithm.
- `decode_z_index`: Restricts decoding to selected z planes.
- `fovs`: Declares the FOVs a subset run covers, so downstream tasks treat it as
  complete.
- `extract_intensity_traces`: Saves per-barcode intensity traces.
- `write_unique_id_images`: Writes decoded images with globally unique barcode
  labels.
- `write_decoded_z`: Selects z planes for decoded-image output.
- `crop_in_image_space`: Applies edge cropping before decoding and restores the
  crop offset in output coordinates.
- `crop_offset`: Supports independent x/y offsets after asymmetric adaptive
  cropping.
- `minimum_area`: Now defaults to 2 (was 0), dropping single-pixel barcodes.
- `resumable_z_decoding: false` now empties the barcode table on rerun instead
  of appending a second copy.
- Removed `decode_3d` and `memory_map`.

### Optimize (`merlin/analysis/optimize.py`)

- `adaptive_crop`: Defaults to the same per-FOV valid warp region as Decode
  during scale-factor and chromatic estimation.
- `normalize_scale_factors`: Returns scale factors as mean-one ratios for
  consistency across preprocessing configurations.
- `cleanup_fragment_results`: Removes fragment-level intermediate arrays after
  their merged results are safely written.
- `get_previous_chromatic_corrector()`: Keeps Decode in the same chromatic image
  space used to estimate its scale factors and backgrounds.
- `chromatic_max_barcodes_per_group`, `chromatic_max_groups`: Limit the barcode
  samples and FOV/z workers for chromatic fitting when `chromatic_from_fragments` is off.
- `scale_factor_floor_ratio` (default 0, off): Raises every scale factor to at
  least this fraction of the largest.
- `initial_scale_factors` (`ones` or `image_mean`), `initial_scale_factor_planes`:
  Seeds the first iteration from the per-bit mean of a few preprocessed planes.
- `scale_factor_source` (`barcodes` or `image_mean`): `image_mean` uses the
  plane-mean factors as the final answer and skips the barcode refactor.
- `tiling_overlap`, `tiling_num_threads`: As in Decode.

### Preprocess (`merlin/analysis/preprocess.py`)

- `preprocess_threads`: Parallelizes independent bit and z-plane preprocessing.
- `lowpass_sigma`: Moved from Decode to Preprocess, so Optimize and Decode use
  the same filtered images.
- `lowpass_after_deconvolution`: Applies the low-pass after deconvolution
  instead of before it.
- `deconvolve_after_highpass`: Selects whether deconvolution runs before or
  after high-pass filtering.
- `decon_method`: `lucyrichardson` (default) or `guo` (Wiener-Butterworth
  accelerated, converges in fewer iterations).
- `threshold_subtract_n`: Sets the amount of global background subtraction.
- `threshold_subtract_mode`: Selects mean-, standard-deviation-, or combined
  background subtraction.
- `highpass_clip` (default true): Whether to zero the negative half of the
  high-passed image.
- `clip_stage`: Now defaults to `lowpass`, clipping after the low-pass instead
  of straight after the high-pass, which leaves a smaller positive background.
- `fft_highpass_clip` (default true): The same clip for the FFT high-pass.
- `preprocess_z_index`: Restricts preprocessing to selected z planes.
- `write_preprocessed_z`: Selects z planes for preprocessed-image output.
- Zero-iteration bypass: Skips Lucy-Richardson allocation when deconvolution is
  disabled.

### Barcode filtering (`merlin/analysis/filterbarcodes.py`)

- `report_bracketing_thresholds`: Reports the available adaptive thresholds
  around the requested misidentification rate for diagnostics.
- `intensity_bins`, `distance_bins`, `area_bins`: The adaptive-threshold
  histogram bin counts, previously fixed.
- `fovs`, `bin_sample_fovs`: Let the adaptive threshold run on a subset of FOVs.
- `poll_interval`: Seconds between checks while decode fragments are still
  arriving.
- `write_filtered_images`: Writes decoded images containing only retained
  barcodes.
- `write_filtered_fovs`: Selects FOVs for filtered-image output.
- `write_filtered_z`: Selects z planes for filtered-image output.

### Registration (`merlin/analysis/warp.py`)

- `median_filter` (default true): The fiducial hot-pixel median filter, always
  applied before, can be turned off.
- `register_3d`: `FiducialCorrelationWarp` option that applies one 3D bead
  correlation per round through depth (the model of the removed
  `FiducialCorrelationWarp3D`); `interpolation` sets the depth resampling.
- `clip_negative_after_highpass`: Clips negative values of the high-passed
  fiducial image.
- `write_fiducial_fovs`, `write_aligned_fovs`, `write_aligned_z`: Select FOVs
  and z planes for fiducial- and aligned-image output.
- `write_averaged_aligned_images`: Writes the average of all aligned channels
  at each z plane (`write_averaged_*` set its filtering and orientation).
- Registration metrics: Saves per-channel x/y shifts, registration error, and
  phase difference.
- `FiducialPolynomialWarp3D` options:
  - `z_polynomial_order` (default 1), `xy_polynomial_order` (default 3): Degrees
    of the depth fits.
  - `z_sample_step`, `xy_sample_step`: Plane spacing of the z and xy samples.
  - `xy_search_bound`, `minimum_overlap_pixels`, `zero_below_percentile`: Bound
    and gate each correlation.
  - `robust_trim_iterations`, `robust_trim_sigma`: Trim outlying samples from
    the fits.
  - `rigid_plane_search`, `rigid_plane_step`, `rigid_snr_ratio`: Refine the
    rigid offset by stepping into the stack while correlation improves.
  - `rigid_search_bound`, `rigid_walk_bound`: Bound the first step and each
    later step of that search.
  - `rigid_only_channels`: Channels that keep the rigid offset and skip the
    depth fit.
  - `z_min_score`, `z_reject_at_search_limit`, `z_min_good_samples`: Drop weak
    z samples and skip the z fit when too few remain.
  - `z_neighbor_substitution` (with `z_neighbor_rings_um`,
    `z_neighbor_min_donors`, `z_neighbor_min_samples`): Gives an unfitted stack
    the median curve of the same round in neighbouring FOVs.
  - `max_z_fit_residual`, `max_xy_fit_residual`, `min_xy_planes_kept`: Flag
    suspect fits in `fit_quality_flags_<fov>.csv`.
  - `z_fit_status_<fov>.csv`, `z_neighbor_substitutions.csv`, `write_qc_table`:
    QC outputs.
  - `stack_num_threads`, `per_z_slice_num_threads`, `plane_cache_size`:
    Concurrency and memory.

### Segmentation (`merlin/analysis/segment.py`, `merlin/util/spatialfeature.py`)

- Defaults: `diameter` now `null` (native resolution); `flow_threshold` and
  `cellprob_threshold` are now passed to cellpose.
- `path_to_user_model`: Takes a model path or a cellpose-4 model name and
  overrides `model_type`.
- Cell outlines are traced on each label's bounding box: identical polygons,
  much faster.
- `feature_labels_<fov>.csv`: Maps each mask label to its feature id. Mask dumps
  are zlib-compressed.
- `FilterCells` outputs: `cell_qc_<fov>.csv` (status of every label) and
  `filtered_mask<fov>.tif` (labels before and after filtering).
- `read_feature_metadata`: A FOV with no features returns an empty table.
- Cell ids read back from CSV are always strings, as pandas >= 3 parses them as
  integers.
- `contains_positions`: No longer writes into its input array (read-only under
  pandas >= 3).

### Barcode database (`merlin/util/barcodedb.py`)

- `barcode_complib`, `barcode_complevel`: Compress barcode tables (default
  `blosc:lz4`, level 5).
- Intensity columns are stored only when Decode extracts intensity traces.

### Restoration (`merlin/analysis/preprocess.py`, `merlin/analysis/modelrestore.py`)

- `CARERestorePreprocess` (replaces `CAREPreprocess`): Applies per-channel CARE
  restoration before the normal preprocessing filters.
- `care_camera_offset`: Sets the camera offset used for CARE normalization.
- `care_input_scale`: Sets the fixed input scale used for CARE normalization.
- `care_use_csbdeep_normalizer`: Enables optional csbdeep percentile
  normalization.
- `care_n_tiles`: Tiles CARE inference to limit peak memory.
- `ModelRestorePreprocess`: Adds joint restoration of all MERFISH bit channels
  with a trained soft-decoding model.

### Pipeline and compatibility

- Python 3.12 / numpy 2 / pandas 3: `setup_merlin_env.sh` builds the `merlin`
  env (cellpose 4); outputs match the previous Python 3.9 env.
- Option names are lowercase (e.g. `write_decoded_FOVs` -> `write_decoded_fovs`,
  `z_duplicate_zPlane_threshold` -> `z_duplicate_z_threshold`); old names are still read.
- `-i`: Accepts a fragment range (`0-25`) or list (`3,8,11`), run in one process.
- snakemake: Imported only when running a whole analysis with snakemake; an
  optional extra.
- Snakemake latency wait: Increases shared-filesystem latency handling from 10
  to 60 seconds.
- tifffile: `TiffWriter.save` (removed from tifffile) -> `TiffWriter.write`.
- Fiducial file parsing: Supports separate fiducial capture groups in data
  organization files.
- Dependency compatibility: Updates NumPy and pandas dtype and concatenation
  behavior.
- Deconvolution utilities: Replaces the legacy MATLAB Gaussian-kernel helper
  with a local implementation.
- Repository cleanup: Removes obsolete utility modules and inherited CI service
  configuration.
