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
- `affine_correction`: Turn it on when FOVs at the edge of the sample stay off
  at the frame corners after the round angle, by an amount that grows with
  depth (the gel stretches or shears there). Each FOV and stack then also gets
  a linear part that changes with depth; `affine_fit_status_<fov>.csv` says
  which stacks were fitted and how far it moves the corners. Not accepted by
  `CrossFOVCompositeWarp` either.
- `z_xy_refine`: Turn it on when `z_fit_status` shows stacks with few good z
  samples although their xy fits are clean and the beads are bright, typically
  because the stack's xy moves by a few pixels with depth. z is then sampled
  at each depth's own xy; `n_good_first` keeps the first pass's count.
- `max_z_fit_residual`, `max_xy_fit_residual`, `min_xy_planes_kept`: A wrong
  rigid offset shows up as a large residual of the fit on top of it; such fits
  are listed in `transformations/fit_quality_flags_<fov>.csv`.
- `per_z_slice_num_threads` x `stack_num_threads` should stay near the number
  of allocated cores.

## Decode

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
