# location_filter.json

Optional tasks that fit one adaptive-filter threshold per location (for
example nucleus, cytoplasm and outside cells) instead of one threshold over
every barcode. They are added to a full analysis; an analysis that does not
list them is unchanged.

## Pipeline

    Decode ─> LocationThreshold ─> LocationFilter ─> partition, exports
                     ^
    strata directory ┘   (made from the segmentation, outside MERlin's tasks)

`Decode` is the base analysis's task. The base threshold and filter can keep
running beside these, which gives the pooled result to compare against. Point
downstream tasks at `LocationFilter`.

## The strata directory

MERlin has no task that makes it, so build it first from your masks, with the
helpers in `merlin/util/locationstrata.py`. A directory holds one sidecar and
one label volume per fov. Each voxel holds the index of its location's name:

    import numpy as np
    from merlin.util import locationstrata

    locationstrata.write_metadata(
        directory, ['outside_cells', 'nucleus', 'cytoplasm'], outside_code=0)
    for fov in dataSet.get_fovs():
        cells, nuclei = load_masks(fov)     # your bool arrays [plane, row, col]
        codes = np.zeros(cells.shape, dtype=np.uint8)
        codes[cells] = 2
        codes[nuclei] = 1
        locationstrata.write_volume(directory, fov, codes, origin=(0, 0, 0))

Location only needs a foreground mask per compartment, not a cell assignment.

- **Every fov needs a file.** A missing file is an error, not a default. Write
  an empty `(0, 0, 0)` volume for a fov with nothing segmented.
- **The volumes must be in the barcode table's frame**: the registration the
  masks were made on, full-frame x, y, and z = plane index. `origin` is the
  table plane, row and column of `codes[0, 0, 0]`. A mask made on a crop needs
  that crop's corner there.
- A barcode outside a volume gets `outside_code`.

## Settings

- `threshold_solver_method`: must be `cumulative_bins`; the filter refuses
  others.
- `location_min_blank_molecules` (default 50000): a location with fewer fit
  blanks than this, or with no coding molecules, keeps the pooled rule. It is
  a stability floor, so lower it with care. A location that falls back is
  named in the report, and its rate is not held at the target.
- `misidentification_rate`: applied to each location separately.
- **Deduplicated molecules**: use `GenerateMoleculeAdaptiveThreshold` with the
  `DeduplicateBarcodes` task as `decode_task` on both tasks. A molecule's
  location is read at its brightest detection's plane.
- `AdaptiveFilterBarcodesLocal` does not support this option.

## Checks

- The threshold task records a digest of the sidecar, and the filter refuses
  to run if the sidecar changed after the fit. Changing the strata means a new
  threshold task.
- The pooled filter does not change, and the pooled histograms equal the sum
  of the location histograms.
- The filtered tables gain a uint8 `location_stratum` column. Each fov writes
  a `location_filter_summary`, and `location_filter_report` sums them, with
  each location's threshold, its rule (`own` or `pooled`), and the reason for
  any fallback.
- A location rule trades counts for a rate that holds in each location, so
  compare both filters per location before choosing one, and report
  misidentification alongside the top-10 gene share.
