# thick_tissue_confocal

`analysis.json` for a thick-tissue confocal MERFISH acquisition that has an
extra protein imaging round and a separately-acquired nuclear stack:

    ~160 z planes over an 80 um slab at 0.5 um
    21 bit rounds, MHD4 constant-weight-4 codebook
    polyT / DAPI in their OWN file, acquired separately from the bit rounds
    a protein (biotin) channel on a late imaging round

Pass it to MERlin with `-a`. Only the analysis definition is included --
supply your own dataorganization (`-o`), codebook (`-c`), microscope
parameters (`-m`) and positions (`-p`). The notes below are the parts of that
dataorganization that this task's settings depend on.

## Registering a channel MERlin does not otherwise know about

**A channel imaged in a round the dataorganization does not describe will not
be registered at all**, which is how a project ends up with a standalone offset
script beside the pipeline. Give the protein channel a row of its own -- model
it on an existing protein row, with `imagingRound`, `fiducialImagingRound` and
`fiducial3DImagingRound` all set to that round -- and
`FiducialPolynomialWarp3D` registers it like any other channel.

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

## Threads

`per_z_slice_num_threads: 16` with `stack_num_threads: 4` is sized for a
~22-core job; the task arbitrates them internally rather than letting them
nest and oversubscribe.
