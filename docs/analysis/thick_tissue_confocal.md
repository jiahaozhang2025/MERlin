# thick_tissue_confocal.json

A complete MERlin analysis for a thick-tissue confocal MERFISH acquisition: a
deep z stack, a constant-weight bit codebook, polyT and DAPI acquired in their
own file separately from the bit rounds, and a protein channel on a late round.

Pass it with `-a`, together with your own data organization (`-o`), codebook
(`-c`), microscope parameters (`-m`) and positions (`-p`).

## Pipeline

    GlobalAlign ─┬─────────────────────────────────────────────┐
                 │                                             │
    PolyWarp ────┼─> Preprocess ─> Optimize1 ─> Optimize2 ─> Optimize3
                 │                                             │
                 │                                             v
                 │                     Decode ─> AdaptiveThreshold ─> FilterBarcodes
                 │                                             │
                 └─> Segment ─> CleanBoundaries ─> CombineBoundaries ─> RefineCells
                        │                                      │
                        │                                      v
                        │                    Partition ─> ExportPartitioned
                        └─> SumSignal ─> ExportSumSignals
                        └─> CellMetadata          FilterBarcodes ─> ExportBarcodes

Twenty tasks. Dependencies are declared by `analysis_name`, so renaming a task
means renaming it in its dependents too. The registration (`PolyWarp`) settings
were tuned for this kind of acquisition; everything else is a working default to
adapt, in particular `decon_sigma`, the number of optimize iterations, the
misidentification rate and all of `Segment`.

## Data organization

- Give every channel a row, including a protein channel on its own round
  (`imagingRound`, `fiducialImagingRound` and `fiducial3DImagingRound` set to
  that round). A channel the data organization does not describe is not
  registered.
- Point a separately acquired channel (polyT, DAPI) at its own file's bead
  frame: `fiducialImageType` = that file's image type, `fiducialImagingRound`
  -1, `fiducialFrame` 0. Pointed at the bit rounds' round-0 bead frame instead,
  the correlation compares a frame with itself and returns exactly (0, 0) for
  every FOV.

## Registration (`FiducialPolynomialWarp3D`)

- `rigid_only_channels`: polyT and DAPI carry beads only near the coverslip, so
  they keep their rigid offset and skip the depth fit.
- `rigid_plane_search`, `rigid_search_bound`, `rigid_walk_bound`: The rigid
  offset is refined by stepping into the bead stack while the correlation
  improves. The first step may move far, to overrule a bad 2D answer; later
  steps are kept close to the running best so the search cannot drift.
- `rigid_walk_confirm`: Turn it on when the bead field is dense and
  self-similar, or when `rigid_search_bound` must be wide (hundreds of px). A
  far first step then has to be confirmed one step deeper, so a ghost peak
  cannot overrule a correct 2D answer.
- `rotation_task`: When the sample turns between rounds, give every round one
  angle. Add a `FiducialRoundRotation` task before `PolyWarp`, with the same
  filter, `rigid_*` and `z_search_range` / `z_min_score` settings, and set
  `rotation_task` to its name in `PolyWarp`:

      {"task": "FiducialRoundRotation", "module": "merlin.analysis.warp",
       "analysis_name": "RoundRotation", "parameters": {...}}

  Run `merlin -t RoundRotation --check-done` once its fragments are done and
  before `PolyWarp` (snakemake does this through the dependency).
  `round_rotation.csv` holds the round angles; `fov_rotation.csv` lists the
  FOVs whose own angle was untrusted or deviated (flagged, not corrected).
  Without `rotation_task` there is no rotation. `CrossFOVCompositeWarp` does
  not accept a `PolyWarp` with `rotation_task`.
- `z_xy_refine`: Turn it on when `z_fit_status` shows stacks with few good z
  samples although their xy fits are clean and the beads are bright, typically
  because the stack's xy moves by a few pixels with depth. z is then sampled
  at each depth's own xy; `n_good_first` keeps the first pass's count.
- `max_z_fit_residual`, `max_xy_fit_residual`, `min_xy_planes_kept`: A wrong
  rigid offset shows up as a large residual of the fit on top of it; such fits
  are listed in `transformations/fit_quality_flags_<fov>.csv`.
- `per_z_slice_num_threads` x `stack_num_threads` should stay near the number
  of allocated cores.

## Preprocess

- Camera hot pixels: add a `HotPixelMap` task and set `hot_pixel_task` to its
  name in `Preprocess`; the pixels are then replaced in every raw frame before
  the warp (the registration is unchanged):

      {"task": "HotPixelMap", "module": "merlin.analysis.hotpixel",
       "analysis_name": "HotPixels", "parameters": {}}

  With more than one camera, list each camera's colours in `camera_colors`.
  `hot_pixels.csv` lists the pixels found.

## Decode

- `crop_in_image_space` with `crop_width` 50 (the default; was 100) and
  `adaptive_crop`: only each FOV's crop region is decoded, the fixed crop
  widened on edges where the warp leaves missing data, and x, y stay in the
  full frame. At 50 px, neighbouring FOVs' decoded regions overlap by more
  than a cell at the usual ~10 % FOV overlap; keep `crop_width` at most
  (overlap - cell diameter) / 2.
- `adaptive_crop_channels` `codebook` (the default): with two libraries
  registered by one `PolyWarp`, each decode is cropped by its own bits'
  channels only.
- Give every `Optimize` task the same `crop_width`, `adaptive_crop`,
  `adaptive_crop_channels` and `adaptive_edge_margin` as `Decode`; the scale
  factors are then measured on the pixels Decode decodes
  (`scale_factor_region` `crop`, the default).
- `per_z_slice_num_threads` is the setting that scales, because z planes are
  independent. Leave `torch_num_threads` null so torch is pinned to one thread
  per plane while planes run in parallel.
- `decode_chunk_size` changes speed only; results are identical at every value.

## Segmentation

An example only: single-channel DAPI with `cpsam_v2`, segmented plane by plane
and linked across z (`cellpose_3d_stitching`). Replace it with the channels and
model suited to the tissue.

## Filtering

`AdaptiveFilterBarcodes` targets a blank-based misidentification rate. The blank
rate also falls when calls concentrate on a few abundant genes, so check the
share of the top genes before reading a low rate as good decoding.

A deep stack images most molecules on several planes, and `Decode` calls each
of them once per plane. For the target to hold per molecule, merge those
detections before the threshold:

    Decode ─> Dedup ─> MoleculeThreshold ─> FilterBarcodes

    {"task": "DeduplicateBarcodes", "module": "merlin.analysis.deduplicate",
     "analysis_name": "Dedup", "parameters": {"decode_task": "Decode"}}
    {"task": "GenerateMoleculeAdaptiveThreshold",
     "module": "merlin.analysis.deduplicate",
     "analysis_name": "MoleculeThreshold",
     "parameters": {"decode_task": "Dedup", "run_after_task": "Dedup", ...}}

`MoleculeThreshold` takes `AdaptiveThreshold`'s settings, and
`FilterBarcodes` takes `"decode_task": "Dedup"` and
`"adaptive_task": "MoleculeThreshold"`. Set `remove_z_duplicated_barcodes`
false in `Decode` (it would delete detections before they are merged).
`max_z_gap_um` (default 0.75) must be at least the plane spacing; at twice
the spacing or more, a molecule also bridges one plane it was not called on.
