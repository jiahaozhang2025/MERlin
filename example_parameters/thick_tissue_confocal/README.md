# thick_tissue_confocal

A complete MERlin `analysis.json` for a thick-tissue confocal MERFISH
acquisition that has an extra protein imaging round and a separately-acquired
nuclear stack:

    ~160 z planes over an 80 um slab at 0.5 um
    21 bit rounds, MHD4 constant-weight-4 codebook
    polyT / DAPI in their OWN file, acquired separately from the bit rounds
    a protein channel on a late imaging round

Pass it with `-a`. Supply your own dataorganization (`-o`), codebook (`-c`),
microscope parameters (`-m`) and positions (`-p`) -- those are dataset
specific. The notes below are the parts of *your* dataorganization that these
settings depend on.

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

Twenty tasks. Dependencies are declared by `analysis_name`, so if you rename a
task, rename it in its dependents too. Terminal tasks are `CellMetadata`,
`ExportPartitioned`, `ExportSumSignals`, `ExportBarcodes` and `Plots`.

**Only the PolyWarp settings were measured on the dataset this came from.**
Everything else is a working default to adapt -- in particular
`Preprocess.decon_sigma`, the number of `Optimize` iterations, the
`FilterBarcodes` misidentification rate, and all of `Segment`.

The file has been checked end to end: `merlin --generate-only` builds all 20
task definitions and resolves the dependency graph above.

## Segmentation is an EXAMPLE, and cpsam needs the right slot

The dataset this came from was segmented by a custom three-model script, not by
MERlin. What is included here is a plain single-channel DAPI segmentation with
`cpsam_v2`, as a starting point.

**On cellpose 4.x you must pass the model through `path_to_user_model`, not
`model_type`.** `CellPoseSegmentSingleChannel3D` picks its model two ways:

    path_to_user_model set  ->  CellposeModel(gpu=..., pretrained_model=...)   works
    otherwise               ->  cellpose.models.Cellpose(model_type=...)       AttributeError

`cellpose.models.Cellpose` does not exist in 4.x -- that build has only
`CellposeModel`, and `MODEL_NAMES` is `['cpsam_v2', 'cpdino', 'cpdino-vitb',
'cpsam']`. So the default `model_type: "cyto2"` cannot run there at all.
`pretrained_model` accepts a bare model name as well as a path, which is why
`"path_to_user_model": "cpsam_v2"` is the working spelling. Set `use_gpu` --
this class does forward it to the model, though note the 2D
`CellPoseSegmentSingleChannel` class does not.

`cellpose_3D_stitching: true` segments each plane in 2D and links them across z
by IoU, which on this kind of slab gives smoother cells than the `do_3D`
anisotropy path. `diameter` is largely vestigial for cpsam.

## Registering a channel MERlin does not otherwise know about

**A channel imaged in a round the dataorganization does not describe will not
be registered at all**, which is how a project ends up with a standalone offset
script beside the pipeline. Give the protein channel a row of its own -- model
it on an existing protein row, with `imagingRound`, `fiducialImagingRound` and
`fiducial3DImagingRound` all set to that round -- and PolyWarp registers it like
any other channel.

Worth doing even if a standalone script already "works": one such script had
re-implemented a subset of this task's filter and peak search and inherited
neither protection, and returned confident wrong answers on 10 of 130 fovs.
Without the median filter, single hot pixels fixed to the detector correlate
with themselves at zero lag -- 7 fovs returned (0,0) at snr 227-589, *higher*
than real beads score, so no signal-strength check could catch it. Without a
bounded peak search, a dense self-similar bead field locks onto a secondary
maximum of its own autocorrelation -- 3 fovs returned 626-1192 px against a
true ~35.

## Point a separately-acquired channel's fiducial at its own file

polyT and DAPI live in a different file from the bit rounds, so their fiducial
must be that file's own bead frame:

    fiducialImageType     <the nuclear file's image type>
    fiducialImagingRound  -1      # MERlin's marker for "no round in the filename"
    fiducialFrame         0
    fiducialColor         488

If they instead point at the bit rounds' round-0 bead frame -- easy to inherit
from a template, and common -- the task correlates channel 0's fiducial against
*itself* and returns **exactly `(0.000, 0.000)`** for every fov. That is
arithmetically correct and useless, and it is silent: nothing downstream
distinguishes "no shift" from "not measured". An offset of exactly zero is the
signature, since real cross-correlation of two acquisitions never lands on it.
It costs nothing while those channels are only used for display, and quietly
misregisters every segmentation mask the moment they are used for anything.

## `rigid_only_channels`

A channel acquired in a separate stack carries beads only near the coverslip.
Measured on this kind of data, the nuclear 488 plane correlates against the
bit-round fiducial at snr ~350 from z = 1 um to at least z = 5 um, and at 6.1
by z = 41 um, while the tissue spans 80 um. Because `_measure_z_offsets`
samples every `z_sample_step` planes across the whole slab, most of its samples
would be beadless noise and the fitted drift polynomial would be garbage rather
than merely imprecise.

Listing such a channel keeps its rigid offset and its plane walk but skips the
polynomial, writing zero coefficients. Keep the walk: on the five fovs whose 2D
nuclear frame was weak it moved the answer by up to 3.8 px and raised
confidence from snr 36 to 320.

## `rigid_plane_search`, and why there are two bounds

A 2D bead frame can give an answer that is in-band, agrees with its neighbours,
and is still wrong -- one fov's protein-round frame was wrong by 57 px while
looking entirely reasonable. The walk measures at the 2D frame, then steps into
the stack while each next plane beats the running best by `rigid_snr_ratio`,
and keeps the winner.

The two bounds do different jobs:

    rigid_search_bound  200   the FIRST step, about the ORIGIN -- wide, so a
                              bad 2D answer can be overruled
    rigid_walk_bound     15   every LATER step, about the RUNNING BEST -- tight,
                              because consecutive planes must agree

Without the second, tighter bound each step can move a full `rigid_search_bound`
and the walk drifts: one fov reached 247 px, which then wrecked its polynomial
fit.

## Check the residuals, not just the offsets

A bad rigid offset does not show up in the offset, but it does show up in the
residual of the polynomial fitted on top of it. Over 1820 fits the z residual
had median 0.038 um (p99.9 0.376) and the xy residual median 0.074 px
(p99.9 0.269), with exactly one outlier at 2.090 um and 5.947/5.456 px --
5.6x and 22x the p99.9, in a fov whose registration was independently known to
be bad. `max_z_fit_residual`, `max_xy_fit_residual` and `min_xy_planes_kept`
print that case and write it to
`transformations/fit_quality_flags_<fov>.csv`, which is written even when empty
so its absence is never read as "the guard did not run".

## Decode threading

`per_z_slice_num_threads` is the one that scales, because z planes are
independent and each is a large unit of work. Leave `torch_num_threads` null:
the task then pins torch to a single intra-op thread whenever that per-z pool
is active. Without the pin every z thread sizes its pool to the whole machine
and they contend -- 16 planes on 16 cores took 364 ms with torch at its default
against 255 ms pinned, a 1.43x difference, and numpy in the same pattern does
not scale at all (1738 ms) because a threaded OpenBLAS gemm serialises across
calling threads.

`decode_chunk_size` is a cache-behaviour knob only; results are bit-identical at
every value. The optimum moves with thread count, and 8192 sits inside both the
single-thread and the multi-thread plateau.

## A caution on the filter

`AdaptiveFilterBarcodes` targets a blank-based misidentification rate, but blank
rate falls with codeword concentration alone -- a run that collapses onto a few
abundant genes will report a *better* rate while decoding worse. Report the
top-10 gene share alongside it before reading a low rate as good decoding.
