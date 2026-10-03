# Changes from the `gpu_decoding` fork

Compared with `aaronhalpern/MERlin:gpu_decoding` at commit
`15ce55f919eed7e986c7a9c79eb59ef41f39e1c8`.

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

### Optimize (`merlin/analysis/optimize.py`)

- `finalize()`: Computes shared scale factors, backgrounds, barcode counts, and
  chromatic corrections once instead of repeating them in downstream tasks.
- `chromatic_from_fragments`: Defaults to collecting chromatic displacement
  samples during enabled chromatic optimization and pooling one final fit.
- `chromatic_on_preprocessed`: Reuses the preprocessed image stack for
  chromatic sampling instead of loading another warped stack.
- `chromatic_threads`: Parallelizes FOV/z chromatic sampling only when
  `chromatic_from_fragments` is disabled.
- `OptimizeLoop`: Writes a chain of optimize iterations once in the analysis
  json. It expands into ordinary `OptimizeIteration` tasks
  `<name>1 .. <name>N`, linked by `previous_iteration`, and these are identical
  to a hand-written chain. Loop parameters: `iterations` (default 10),
  `random_seeds` or `random_seed_start` (default seeds 1..N), `per_iteration`
  (per-iteration values of any other parameter), `iteration_task`,
  `previous_iteration`, and `num_workers` (spawned processes for fragments;
  default 1). Every other parameter goes to every iteration. Running the loop
  runs the unfinished iterations in order and finalizes each one. It is
  complete once its last iteration is, including when the iterations were run
  by name. As an `optimize_task` it answers with its last iteration.
- `ImageScaleFactors` (new task): Estimates per-bit scale factors from the
  preprocessed images alone, with no decode in the loop. The barcode loop
  equalises the mean on-bit intensity of the pixels it decoded, and for a
  genuinely dim bit that converges to a wrong answer: on 20260609, RS0707 goes
  to 0.173, and the max/min spread reaches 9.28x against ~3.1x from the
  images. For each bit and sampled (fov, z), the background is the
  `background_percentile` (default 50) percentile and the amplitude is
  mean(selected pixels) - background. The median is taken over planes.
  `selection` chooses the pixels:
  - `sigma` (default): above background + `signal_sigma` (3.0) x MAD spread
  - `quantile`: the top `signal_quantile` (0.999)
  - `excess`: the histogram excess over the background mirrored about its
    mode (`excess_bins` 512)
  - `mean`: the whole-plane mean

  `min_pixels_above` (200): below this many pixels a plane is uninformative for
  that bit. Use `background_pixels: nonzero` after `clip_stage: lowpass`, where
  most pixels are exactly 0 and the median would be 0. `fov_per_iteration` (30)
  or `fov_index` sets the planes.

  It drops in as Decode's `optimize_task`. It supplies its own backgrounds
  (`supply_backgrounds`, default true; zeros would leave a background floor in
  every trace) and takes chromatic correction from `reference_optimize_task`.
  An `OptimizeIteration` can chain onto it as `previous_iteration`. Every run
  also saves the estimate over a fixed grid of sigma/quantile settings;
  `get_scale_factor_sensitivity()` reports how far each bit moves across it.
- Per-bit bias freeze (`bias_freeze_reference`: a gene-abundance `.npz`
  (names/total) or 2-column csv; off when unset): After each round, each bit's
  bias is its coefficient in a least-squares fit of log10(observed / reference
  gene share) against the codebook. A bit whose |bias| grew by more than
  `bias_freeze_tolerance` (0.10) has its update rolled back; the other bits
  update normally. Round k measures the factors of round k-1, so the rollback
  restores round k-2's value, and a vetoed bit hovers between two values.
  `bias_freeze_min_genes` (50): with fewer usable genes nothing is vetoed.
  `bias_freeze_min_abs` (0, off) restricts vetoes to bits with |bias| above it.
  Each round it judges writes `bias_freeze.csv`. Why: on 20260609 the total
  squared bit bias is lowest at round 2 (1.518, 0.405, 0.491, 0.549, 0.719,
  ...) and grows after it.

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

- On by default in all three: `write_preprocessed_images` now defaults to
  `True` (was `False`); Decode already wrote images by default, and the
  adaptive filters' new `write_filtered_images` is on by default too.
- `write_images_seed` (default 1): When `write_*_fovs` or `write_*_z` is not
  given, 3 fovs and one z plane are drawn at random from this seed. All three
  tasks draw the same way, so by default they write the same fovs and plane.
  The draw is stored in `task.json`. Explicit lists are kept as they are.
- `write_preprocessed_z`: New. The planes Preprocess writes; `None` means every
  preprocessed plane. Preprocess no longer runs at all for a fov that needs
  neither a histogram nor an image.
- `decoded_image_format`: `'tif'` (default) writes one 2D file per output
  channel per written plane, `images/decoded_<channel>_fov<fov>_z<z>.tif`, for
  `barcodes`, `magnitude`, `distance`, and `unique_id` when
  `write_unique_id_images` is on. `'zarr'` keeps the old single array per fov.
  `Decode.get_decoded_image` reads either format.
- The adaptive filters read decoded images through `get_decoded_image`, so they
  work on old zarr decodes too. `AdaptiveFilterBarcodesLocal` now writes
  filtered images as well. A failed filtered-image write is logged as a warning
  and no longer fails the fragment.

### Grouped output layout (`merlin/core/dataset.py`, `merlin/util/regroup.py`)

- New datasets put each task folder under a group folder and the dataset files
  under `Files`:
  - `Prepare`: GlobalAlign, warps, Preprocess
  - `Optimize`: OptimizeLoop and its iterations, ImageScaleFactors
  - `Decode`: Decode, Threshold, the barcode filters
  - `Segment`: segmentation, CleanBoundaries, CombineBoundaries, RefineCells,
    FilterCells
  - `Export`: Partition, ExportPartitioned, ExportBarcodes, CellMetadata,
    SumSignal
  - `Other`: anything else (GenerateMosaic, PlotPerformance, SlurmReport)
  - `Files`: codebooks, data organization, file map, positions, microscope
    parameters, `dataset.json`

  `snakemake/` and `logs/` stay at the top level.
- The group is the task class's `outputGroup` attribute, inherited by
  subclasses. A task that already has a folder is used wherever it is, so
  changing a group never orphans finished output.
- A dataset whose `dataset.json` is at the top level was created before this
  change and keeps the flat layout. Nothing about it changes.
- `python -m merlin.util.regroup ANALYSIS_HOME/<dataset>` moves a flat dataset
  into the grouped layout: it shows the plan, and moves only with `--apply`. It
  refuses while fragments are unfinished or relative symlinks would break, and
  writes the move list to `logs/`. Regenerate the Snakefile afterwards.
- Scripts that build task paths themselves (`$DS/$T/tasks/...`) have to look
  under the group folder. `dataSet.get_analysis_subdirectory(name)` works in
  both layouts.
- `merlin.get_analysis_datasets()` also finds grouped datasets
  (`Files/dataset.json`).

### Segmentation: Cellpose-SAM (`merlin/analysis/segment.py`)

- Cellpose-SAM replaces cellpose 2/3. `CellPoseSegmentSingleChannel3D` and
  `CellPoseSegmentTwoChannel3D` now run cellpose >= 4; `CellPoseSegment` and
  `CellPoseSegmentSingleChannel` were removed. cellpose 4 cannot load cellpose
  2/3 models, so a json that names a removed class or a `cyto2`/`cyto3` model
  fails, and such datasets are re-segmented with cpsam (or a cpsam fine-tune).
- Defaults changed: `model_type` `cyto2` -> `cpsam_v2`; `diameter` 50 ->
  `null` (native resolution; a value scales the image by 30 / diameter).
  `flow_threshold` (0.4) and `cellprob_threshold` (0.0) are now passed to
  cellpose; the 3D classes did not pass them before.
- `z_index`: Segments one z plane and repeats its outlines on every plane, so
  `PartitionBarcodes` assigns barcodes from all planes to these 2D cells. This is the
  middle-plane convention of Vizgen/Allen, e.g. Yao 2023, 10.1038/s41586-023-06812-z.
- `FilterCells` (new task): Removes non-cell objects after a provisional partition,
  because its barcode criteria need assigned barcodes:
  - `min_area_um2`, `min_width_um`: footprint area and minor axis on the label's
    largest plane
  - `min_barcodes`, `min_barcode_density`: coding barcodes from `partition_task`, and
    the same per um2 of footprint

  Every criterion is optional. It writes the kept cells as its feature database,
  `cell_qc_<fov>.csv` (every label with its measurements and status: `kept`,
  `overlap_duplicate`, or the failed criteria), and `filtered_mask<fov>.tif` (ImageJ
  [z, c, y, x]: c0 = segmented labels, c1 = kept cells). Partition into `FilterCells` to
  get tables of the kept cells. Chain used for 20260909 DS-4: Segment -> CleanBoundaries
  -> CombineBoundaries -> RefineCells -> PartitionQC -> FilterCells -> Partition*.

### Stitched decoding (`merlin/analysis/stitched.py`, `merlin/util/stitched/`) (from Lida Cheng)

- New opt-in tasks `StitchedInitialize`, `StitchedStage` and `StitchedFragments`
  decode a whole sample as one stitched volume instead of per fov. A molecule
  at a fov boundary is decoded with the neighbouring fovs' image around it, and
  each molecule belongs to exactly one output tile. Stages:
  1. bead-based within-round and cross-round XYZ registration
  2. a required review: `StitchedReview` waits for a decision file that names
     the hash of the evidence it reviewed
  3. fresh intensity optimization, decode, global adaptive filter, and
     duplicate removal across tiles and z
  4. optional: GeoPandas mask partition and an analysis report (H5AD, Leiden
     clusters, HTML)

  `python -m merlin.util.stitched.configure` writes the analysis json and a
  cluster-resource json. Guide: `docs/stitched_decoding.md`.
- Only the 0718 acquisition is accepted (profile `0718_40x_150z`: 52 fovs,
  anchor fov 13, 0.1493 um/px, 150 z planes); other geometries are rejected.
  Each run freezes copies of its inputs, its stage scripts and the current
  `merlin` package, and runs each stage in a subprocess against those copies.
- Extras: `pip install -e '.[stitched]'` (geopandas, rasterio), or
  `'.[stitched-analysis]'` for the report.
- Its optimizer calls `OptimizeIteration` helpers on a stand-in object. The
  scale-factor floor and bias-freeze helpers are bound there with both off, so
  it reproduces the validated run's scale factors on this version.

### Cross-fov composite warp (`merlin/analysis/crossfov_composite_warp.py`) (from Lida Cheng)

- `CrossFOVCompositeWarp` (new task, built on a finished `polywarp_task`):
  Fills the margins that stage drift pushed outside a fov's camera frame with
  the same tissue imaged by the neighbouring fov in that round. The two fovs
  are linked through their round-0 beads (xy offset and focus offset, gated on
  SNR and spread, with a self-test). Decode's adaptive crop then excludes only
  the margins that could not be filled. Written for 20260824, where per-fov
  cropping lost ~19% of the footprint. No test covers it yet.

## Minor changes

### Decode (`merlin/analysis/decode.py`, `merlin/util/decoding.py`)

- `tiling_factor`: Divides large images into overlapping tiles to reduce peak
  decoding memory.
- `tile_overlap`: Preserves barcodes crossing tile boundaries and supports
  overlap-duplicate removal.
- `num_threads`: Controls tile or nearest-neighbor processing concurrency.
- `magnitude_threshold`: Filters low-magnitude pixels before barcode matching.
- `nn_algorithm`: Selects the scikit-learn nearest-neighbor algorithm.
- `decode_z_index`: Restricts decoding to one selected z plane.
- `extract_intensity_traces`: Saves per-barcode intensity traces.
- `write_unique_id_images`: Writes decoded images with globally unique barcode
  labels.
- `write_decoded_z`: Selects z planes for decoded-image output.
- `crop_in_image_space`: Applies edge cropping before decoding and restores the
  crop offset in output coordinates.
- `crop_offset`: Supports independent x/y offsets after asymmetric adaptive
  cropping.
- `minimum_area`: Now defaults to **2** (was 0), so single-pixel barcodes are
  dropped at decode. Tasks created before the change keep the value in their
  `task.json`.
- `resumable_z_decoding: false` fix: The barcode table was emptied only when
  the key was absent, so an explicit `false` neither emptied nor deliberately
  kept it. Re-running such a fragment appended a second copy of every barcode
  (on 20260609, fov 0 of BFDecode and fov 20 of L2DecodeL held every row
  twice). The table is now emptied unless the key is true.

### Optimize (`merlin/analysis/optimize.py`)

- `adaptive_crop`: Defaults to the same per-FOV valid warp region as Decode
  during scale-factor and chromatic estimation.
- `normalize_scale_factors`: Returns scale factors as mean-one ratios for
  consistency across preprocessing configurations.
- `cleanup_fragment_results`: Removes fragment-level intermediate arrays after
  their merged results are safely written.
- `get_previous_chromatic_corrector()`: Keeps Decode in the same chromatic image
  space used to estimate its scale factors and backgrounds.
- `chromatic_max_barcodes_per_group`: Limits barcode samples in each FOV/z
  worker only when `chromatic_from_fragments` is disabled.
- `chromatic_max_groups`: Limits the FOV/z workers used for chromatic fitting
  only when `chromatic_from_fragments` is disabled.
- `scale_factor_floor_ratio` (default 0, off): Raises every scale factor to at
  least this fraction of the largest, on read. 0.25 reproduces the reference
  floor on 20260609: the spread is pinned at 4.00x, and Opalin falls from 32.2%
  to 3.1% of calls. The blank-based filter cannot see this over-call, because
  the spurious calls look like real ones.
- `initial_scale_factors` (`ones`, default, or `image_mean`), with
  `initial_scale_factor_planes` (4): Seeds the first iteration from the per-bit
  mean of a few sampled preprocessed planes instead of from ones.
- `scale_factor_source` (`barcodes`, default, or `image_mean`): `image_mean`
  uses the plane-mean factors as the final answer and bypasses the barcode
  refactor. Backgrounds and chromatic correction still come from the loop. A
  refactor round scales a dim bit down by a roughly constant factor wherever it
  starts, so seeding alone does not stop the drift.

### Preprocess (`merlin/analysis/preprocess.py`)

- `preprocess_threads`: Parallelizes independent bit and z-plane preprocessing.
- `lowpass_sigma`: Moved from Decode to Preprocess (default 1), so Optimize and
  Decode use the same filtered images. Decode raises an error if it is still
  set there.
- `threshold_subtract_n`: Sets the amount of global background subtraction.
- `threshold_subtract_mode`: Selects mean-, standard-deviation-, or combined
  background subtraction.
- `deconvolve_after_highpass`: Selects whether deconvolution runs before or
  after high-pass filtering.
- `preprocess_z_index`: Restricts preprocessing to one selected z plane.
- Zero-iteration bypass: Skips Lucy-Richardson allocation when deconvolution is
  disabled.
- `highpass_clip` (default true): False keeps the negative half of
  image - blur(image) (`imagefilters.highpass_filter(..., clip=False)`). With
  the clip, a pure-background pixel's nearest codeword on 20260609 is Opalin,
  rank 1 of 246. Do not combine false with `OptimizeIteration`: the rectified
  pedestal props up the on-bit mean of low-SNR bits. It is meant for an
  `ImageScaleFactors` path.

### Barcode filtering (`merlin/analysis/filterbarcodes.py`)

- `report_bracketing_thresholds`: Reports the available adaptive thresholds
  around the requested misidentification rate for diagnostics.
- `write_filtered_images`: Writes decoded images containing only retained
  barcodes.
- `write_filtered_fovs`: Selects FOVs for filtered-image output.
- `write_filtered_z`: Selects z planes for filtered-image output.

### Segmentation (`merlin/analysis/segment.py`, `merlin/util/spatialfeature.py`)

- `CellPoseSegmentSingleChannel3D` / `CellPoseSegmentTwoChannel3D` share one
  `_run_analysis`; the latter stacks `[channel_1_name, channel_2_name]`, and cpsam
  reads channels in that order. `path_to_user_model` takes a path or a
  cellpose-4 model name and overrides `model_type`.
- The module still imports under older cellpose, so its non-cellpose tasks run
  anywhere. A cellpose task raises a clear error there.
- Cell outlines are traced on each label's bounding box. The polygons are
  identical, and this takes ~1 s per fov instead of ~0.4 s per cell.
- New output `feature_labels_<fov>.csv` maps each mask label to its feature id.
  The `segmented_mask` / `segmented_images` dumps are zlib-compressed, ~0.5 MB
  instead of 67 MB per fov; a z-stack is written in one call, so it stays one
  series.
- `read_feature_metadata`: A fov with no features returns an empty table instead
  of failing, so `ExportCellMetadata` works when a filter empties a fov.
- `contains_positions`: Rounds z without writing into its input array. With
  pandas >= 3, `DataFrame.values` is read-only, and the old in-place write failed.
- Cell ids read back from CSV are strings under every pandas version. They are 128-bit
  integers, which pandas >= 3 parses as int and pandas 1.x as str. The change covers
  `CombineCleanedBoundaries.return_exported_data`, `get_partitioned_barcodes` and
  `get_sum_signals`. Before it, `RefineCellDatabases` under pandas 3 matched no
  cells and silently wrote empty databases. It also uses a set lookup now.

### Barcode database (`merlin/util/barcodedb.py`)

- `barcode_complib` (default `blosc:lz4`), `barcode_complevel` (default 5),
  set on the task that writes barcodes: Barcode tables are compressed. This
  applies to newly created tables only; a task folder may hold both kinds, and
  reads work the same.
- Intensity columns follow the data: when Decode ran with
  `extract_intensity_traces` false, the `intensity_*` columns used to be
  written as NaN, 84 bytes per row (57.5% of the row; 259 GB across four
  20260609 tasks). They are now left out.
- `write_barcodes(fov=None)` no longer appends the whole table again after
  writing it per fov. No call in this fork passes `fov=None`.

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

- `median_filter` (default true): The 3x3 fiducial hot-pixel median filter,
  which was always applied, can be turned off.
- `write_fiducial_fovs`: Selects FOVs for fiducial-image output.
- `write_aligned_fovs`: Selects FOVs for aligned-image output.
- `write_aligned_z`: Selects z planes for aligned-image output.
- Registration metrics: Saves per-channel x/y shifts, registration error, and
  phase difference.

#### `DeconvolutionPreprocess` clip stage default

- `clip_stage`: now defaults to **`lowpass`** (was `highpass`). The negative
  clip happens after the lowpass instead of straight after the highpass, which
  shrinks the rectified-noise pedestal on background about 2.6x. That pedestal
  is what makes empty background decode as one specific codeword. Tasks created
  before the change keep the `highpass` recorded in their task.json. Caution:
  the barcode optimize loop leans on the pedestal for low-SNR bits.

#### `FiducialPolynomialWarp3D` z-fit gate and neighbour substitution

- `z_polynomial_order`: Now defaults to **1**, not 3. Gel expansion between
  rounds is a bulk strain and so linear in depth; leave-one-out over the
  sampled planes on 20260609 gives 0.098 um for order 1 against 0.124 for
  order 3, and order 1 wins on 84% of stacks. `xy_polynomial_order` is
  unchanged at 3.
- `z_min_score` (default 0.5): Drops a z sample whose peak ZNCC is below it.
- `z_reject_at_search_limit` (default True): Also drops a sample whose
  |offset| sits on the +-(`z_search_range` x plane spacing) ceiling, which
  means the argmax railed. The score gate alone does not catch this, because
  ZNCC is still measured against whatever tissue lies at the edge of the
  search.
- `z_min_good_samples` (default 4): Below this many surviving samples no
  polynomial is fitted; the z coefficients are zero and the round falls back
  to its rigid xy. Without the gate, one 20260609 stack whose bead signal was
  dead on all ten sampled planes received a 5.4 um -- eleven plane -- swing
  out of a fit whose median swing is 0.85 um.
- `z_neighbor_substitution` (default True), with `z_neighbor_rings_um`,
  `z_neighbor_min_donors`, `z_neighbor_min_samples`: In `finalize()`, an
  unfitted stack takes the pointwise-median curve of the SAME round in the
  spatially adjacent fovs, re-fitted at `z_polynomial_order`. This is a
  cross-fov step and so cannot run inside a per-fov fragment; it happens once,
  from `merlin -t <task> --check-done`. It is idempotent: which stacks to
  repair is read from `z_fit_status`, which records the gate result and is
  never rewritten, and donors are by definition stacks that fitted.
- `z_fit_status_<fov>.csv`: New per-fov QC output, one row per data channel,
  recording sample counts, whether the z polynomial was fitted, and whether
  the curve came from a fit, a neighbour, or the rigid-only path.
- `z_neighbor_substitutions.csv`: New task-level record of every substitution
  and its donors.
- To restore the previous behaviour exactly: `z_min_score` 0,
  `z_reject_at_search_limit` false, `z_min_good_samples` 0,
  `z_neighbor_substitution` false, `z_polynomial_order` 3.

### Pipeline and compatibility

- Option names have no capital letters. Renamed: `write_decoded_FOVs`,
  `write_preprocessed_FOVs`, `write_filtered_FOVs`, `write_aligned_FOVs`,
  `write_fiducial_FOVs`, `write_composite_FOVs`, `dump_segmented_FOVs` ->
  `*_fovs`; `cellpose_3D_stitching` -> `cellpose_3d_stitching`;
  `z_duplicate_zPlane_threshold` -> `z_duplicate_z_threshold`; `zIndices` ->
  `z_indices`; `codebookNum` -> `codebook_num`. An analysis json or saved
  `task.json` that uses an old name is read as the new name
  (`analysistask.RENAMED_PARAMETERS`), so existing datasets load and regenerate
  unchanged. Scripts that read `task.parameters[...]` directly need the new
  names.

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
- Python 3.12 / numpy 2 / pandas 3: MERlin runs in the `merlin` conda env built by
  `envs/setup_merlin_env.sh` (Python 3.12, cellpose 4.2.1.1, numpy 2.4, pandas 3.0, torch
  2.8 cu128, tensorflow 2.20 + csbdeep; exact versions in `envs/merlin_freeze.txt`). On
  20260909 DS-4 inputs, RigidWarp, Decode, the adaptive filter, Optimize, Partition,
  FilterCells, the overlap-cleaning chain (after the cell-id fix above) and the
  metadata/partition readers give identical results to the Python 3.9 /
  numpy 1.26 / pandas 1.5 env. GenerateAdaptiveThreshold bin edges differ by <= 6.5e-7,
  so a few borderline barcodes change bin.
- tifffile: `TiffWriter.save` (removed from tifffile) -> `TiffWriter.write` everywhere.
- snakemake: Imported only when running a whole analysis json with snakemake.
  Generating tasks and running them with `-t` need no snakemake. It is an optional
  extra in `pyproject.toml` (`snakemake>=7,<8`; snakemake 8 removed
  `snakemake.snakemake()`).
- `pyproject.toml`: `numpy>=1.26` (was pinned to 1.26.4); `segmentation` extra is
  `cellpose>=4`.
