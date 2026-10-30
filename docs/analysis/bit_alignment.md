# bit_alignment.json

Optional tasks that measure each bit's residual xy offset after registration
from the decoded MERFISH molecules themselves, and decode again with those
offsets removed. They are added to a full analysis; an analysis that does not
list them is unchanged.

## Pipeline

    Optimize ─> BitCalibration ─> BitAlignment ─> BitWarp ─> BitPreprocess ─> BitDecode
                                     ^              ^                              │
    Preprocess ──────────────────────┘    Warp ─────┘        BitFilter <─ BitThreshold

`Warp`, `Preprocess`, `Optimize` and `Decode` are the base analysis's tasks.
The base chain still runs; the `Bit*` chain is a second decode of the same
images. Point downstream tasks (partition, exports) at `BitFilter`.

## Requirements

- **Base Optimize**: set `normalize_scale_factors` false on every iteration (an
  `OptimizeLoop` sets it once). The calibration keeps absolute scale factors
  and refuses a run that rescales them on read. An existing analysis cannot
  change a saved task's parameters: add a new Optimize chain with the same
  settings and fov picks instead.
- **Base warp**: the aligned sampler reads each plane at its own z with one
  transformation per channel. Accepted: `FiducialCorrelationWarp` without
  `register_3d`, and `FiducialPolynomialWarp3D` whose fit is the same on every
  plane (`xy_polynomial_order` 0, `z_polynomial_order` 0, `z_search_range` 0).
  A warp that moves with depth is refused at the first fragment.
- **Aligned tasks**: copy the base Preprocess, Decode, threshold and filter
  parameters into `BitPreprocess`, `BitDecode`, `BitThreshold` and `BitFilter`,
  so the two chains differ only in the warp.

## Tasks

- `OptimizeRawCalibration` (`BitCalibration`): freezes the bootstrap's applied
  chromatic map, scale factors and backgrounds with a checksum that the later
  tasks verify. `max_iterations` 0 freezes them as they are. Above 0 it
  continues the global fit on an explicit `calibration_fov_index` list of
  `[fov, z]` until the map moves less than `chromatic_tolerance_px` and the
  factors less than `scale_relative_tolerance` for `stable_iterations` steps;
  with `require_convergence` (default) an unconverged fit publishes nothing.
- `OptimizeBitAlignment` (`BitAlignment`): one fragment per (fov, z) over
  `fovs` x `z_indices` (default all). It decodes the plane with the frozen
  calibration, keeps confident coding molecules (area >= 4, distance <=
  `distance_threshold`, at most 160 per `tile_size` tile), localizes each in
  its ON bits of the registered image, and fits one xy shift per bit from the
  pairwise position differences (robust least squares; the mean shift is zero).
  Even tiles of a checkerboard fit, odd tiles validate; a plane keeps its
  shifts only if the held-out pair error's median falls and its 95th
  percentile rises by no more than 0.02 px as predicted and 0.03 px after
  resampling. Otherwise the plane gets zero shift. `bit_alignment_<fragment>.json` records the
  status (`accepted`, `rejected_predicted_geometry`,
  `rejected_applied_geometry`, `insufficient_or_unreliable`), the proposed and
  applied shifts and the held-out errors.
- `BitAlignedWarp` (`BitWarp`): the base warp with each codebook bit's shift
  composed into its transformation, so each image is still resampled once
  after the chromatic map. Other channels (polyT, DAPI, protein) pass through
  the base warp. Decode's adaptive crop covers both resampling passes.
  `apply_bit_alignment` false gives the same sampler with zero shifts, and
  `common_support_with_bit_alignment` makes the crop cover the shifted and
  unshifted geometry, so an aligned and an unaligned chain decode the same
  region.
- Scale factors are not refitted after alignment: `BitDecode` uses
  `BitCalibration`'s.
- `PooledRawHistogram` (optional, in place of `GenerateAdaptiveThreshold`):
  bins only after every selected fov has decoded; an empty fov contributes
  zero counts.

## Resources

`OptimizeBitAlignment` holds one plane's full image stack and decoder output:
plan about 8 GB per parallel fragment. `BitAlignedWarp` only writes a small
record per plane.
