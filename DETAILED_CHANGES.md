# Changes from the `gpu_decoding` fork

Compared with `aaronhalpern/MERlin:gpu_decoding` at commit
`15ce55f919eed7e986c7a9c79eb59ef41f39e1c8`.

## Major changes

### Decode (`merlin/analysis/decode.py`, `merlin/util/decoding.py`)

- `distance_metric`: Adds chunked matrix-multiplication decoding with
  `dot_product`, `softmax`, and `softmax_dot_product` modes as faster
  alternatives to nearest-neighbor search.
- `decode_chunk_size`: Limits the number of pixel traces processed in each
  matrix multiplication to control memory use.
- `softmax_temperature`: Controls optional softmax top-1 probabilities for
  machine-learning and confidence-based workflows.
- `adaptive_crop`: Excludes each FOV's unaligned warp margins so invalid image
  regions do not contaminate barcode extraction.

### Optimize (`merlin/analysis/optimize.py`)

- `finalize()`: Computes shared scale factors, backgrounds, barcode counts, and
  chromatic corrections once instead of repeating them in downstream tasks.
- `chromatic_from_fragments`: Collects chromatic displacement samples during
  optimization fragments and pools them for one final fit.
- `chromatic_on_preprocessed`: Reuses the preprocessed image stack for
  chromatic sampling instead of loading another warped stack.
- `chromatic_threads`: Parallelizes FOV/z chromatic sampling only when
  `chromatic_from_fragments` is disabled.
- `chromatic_max_barcodes_per_group`: Limits barcode samples in each FOV/z
  worker only when `chromatic_from_fragments` is disabled.
- `chromatic_max_groups`: Limits the FOV/z workers used for chromatic fitting
  only when `chromatic_from_fragments` is disabled.

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

## Minor changes

### Decode (`merlin/analysis/decode.py`, `merlin/util/decoding.py`)

- `tiling_factor`: Divides large images into overlapping tiles to reduce peak
  decoding memory.
- `tile_overlap`: Preserves barcodes crossing tile boundaries and supports
  overlap-duplicate removal.
- `num_threads`: Controls tile or nearest-neighbor processing concurrency.
- `magnitude_threshold`: Filters low-magnitude pixels before barcode matching.
- `nn_algorithm`: Selects the scikit-learn nearest-neighbor algorithm.
- `resumable_z_decoding`: Retains completed z planes when a decode task is
  resumed.
- `decode_z_index`: Restricts decoding to one selected z plane.
- `extract_intensity_traces`: Saves per-barcode intensity traces.
- `write_unique_id_images`: Writes decoded images with globally unique barcode
  labels.
- `write_decoded_FOVs`: Selects FOVs for decoded-image output.
- `write_decoded_z`: Selects z planes for decoded-image output.
- `crop_in_image_space`: Applies edge cropping before decoding and restores the
  crop offset in output coordinates.
- `crop_offset`: Supports independent x/y offsets after asymmetric adaptive
  cropping.

### Optimize (`merlin/analysis/optimize.py`)

- `adaptive_crop`: Uses the same per-FOV valid warp region as Decode during
  scale-factor and chromatic estimation.
- `normalize_scale_factors`: Returns scale factors as mean-one ratios for
  consistency across preprocessing configurations.
- `cleanup_fragment_results`: Removes fragment-level intermediate arrays after
  their merged results are safely written.
- `get_previous_chromatic_corrector()`: Keeps Decode in the same chromatic image
  space used to estimate its scale factors and backgrounds.

### Preprocess (`merlin/analysis/preprocess.py`)

- `preprocess_threads`: Parallelizes independent bit and z-plane preprocessing.
- `lowpass_sigma`: Moves low-pass filtering into preprocessing so Optimize and
  Decode use the same filtered images.
- `threshold_subtract_n`: Sets the amount of global background subtraction.
- `threshold_subtract_mode`: Selects mean-, standard-deviation-, or combined
  background subtraction.
- `deconvolve_after_highpass`: Selects whether deconvolution runs before or
  after high-pass filtering.
- `preprocess_z_index`: Restricts preprocessing to one selected z plane.
- Zero-iteration bypass: Skips Lucy-Richardson allocation when deconvolution is
  disabled.

### Barcode filtering (`merlin/analysis/filterbarcodes.py`)

- `report_bracketing_thresholds`: Reports the available adaptive thresholds
  around the requested misidentification rate for diagnostics.
- `remove_z_duplicated_barcodes`: Removes likely duplicate detections across
  nearby z planes.
- `z_duplicate_zPlane_threshold`: Sets the maximum z-plane separation for
  duplicate removal.
- `z_duplicate_xy_pixel_threshold`: Sets the maximum xy separation for
  duplicate removal.
- `write_filtered_images`: Writes decoded images containing only retained
  barcodes.
- `write_filtered_FOVs`: Selects FOVs for filtered-image output.
- `write_filtered_z`: Selects z planes for filtered-image output.

### Segmentation (`merlin/analysis/segment.py`)

- `cellpose_channels`: Selects the two Cellpose input channels and defaults to
  `[1, 2]`, fixing the previous two-channel configuration.

### Restoration (`merlin/analysis/preprocess.py`, `merlin/analysis/modelrestore.py`)

- `CARERestorePreprocess`: Adds per-channel CARE restoration before the normal
  preprocessing filters.
- `care_camera_offset`: Sets the camera offset used for CARE normalization.
- `care_input_scale`: Sets the fixed input scale used for CARE normalization.
- `care_use_csbdeep_normalizer`: Enables optional csbdeep percentile
  normalization.
- `care_n_tiles`: Tiles CARE inference to limit peak memory.
- `ModelRestorePreprocess`: Adds joint restoration of all MERFISH bit channels
  with a trained soft-decoding model.

### Warp (`merlin/analysis/warp.py`)

- `boundary_smooth`: Fills invalid warped edges from an edge-padded blurred
  image.
- `median_filter`: Enables or disables fiducial hot-pixel filtering.
- `sparse_bead_fix`: Enables edge removal and bright-pixel selection for sparse
  fiducials.
- `percentile_pixel_to_keep`: Selects the fiducial intensity percentile retained
  by the sparse-bead filter.
- `edge_width_to_remove`: Sets the excluded fiducial-image edge width.
- `write_aligned_FOVs`: Selects FOVs for aligned-image output.
- `write_aligned_z`: Selects z planes for aligned-image output.
- Registration metrics: Saves per-channel x/y shifts, registration error, and
  phase difference.

### Pipeline and compatibility

- Snakemake latency wait: Increases shared-filesystem latency handling from 10
  to 60 seconds.
- Fiducial file parsing: Supports separate fiducial capture groups in data
  organization files.
- Dependency compatibility: Updates NumPy and pandas dtype and concatenation
  behavior.
- Deconvolution utilities: Replaces the legacy MATLAB Gaussian-kernel helper
  with a local implementation.
- Repository cleanup: Removes obsolete utility modules and inherited CI service
  configuration.
