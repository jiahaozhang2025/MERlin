# stitched_decoding.json

The stitched decoding pipeline (from Lida Cheng): the whole sample is registered
and decoded as one stitched volume instead of FOV by FOV, so a molecule at a FOV
boundary is decoded once, with the surrounding image from the neighbouring FOVs.
This file is what the configure command below writes, with placeholder paths.

Only the acquisition profile `0718_40x_150z` is accepted (52 FOVs, 2048 x 2048
frames at 0.1493 um/px, 150 fiducial planes); other geometries are rejected
until they are validated.

## Create the json

    python -m pip install -e '.[stitched]'            # through partition
    python -m pip install -e '.[stitched-analysis]'   # also the report

    python -m merlin.util.stitched.configure \
      --output stitched_decoding.json \
      --cluster-output stitched_cluster.json \
      --metadata-dir /path/to/metadata \
      --raw-root /path/to/raw \
      --codebook-index 0 \
      --review-decision /path/to/alignment_decision.json \
      --mask-archive /path/to/mask_archive \
      --analysis-report

- `--metadata-dir`: `positions.csv`, `dataorganization.csv`, `filemap.csv`,
  `microscope_parameters.json` and `codebook_<index>_<name>.csv`.
- One codebook per run; give each codebook its own `--prefix` or analysis
  directory.
- Omit `--mask-archive` and `--analysis-report` for barcodes only.
- `--reuse-registration <accepted manifest>` replaces `--review-decision` to
  reuse an already reviewed registration of the same sample.
- `--cluster-output` writes per-task CPU, memory and time requests for a
  Snakemake cluster run.

Use the json like any analysis json (`merlin -a ... --generate-only`, then run
the tasks).

## Tasks

| Stage | Tasks |
| --- | --- |
| Inputs | `StitchedInitialize` freezes metadata, stage scripts and MERlin code, and checks every raw file |
| Registration | `StitchedExtract`, `StitchedWithin`, `StitchedPool`, `StitchedCross` (bead-based, within and across rounds) |
| Review | `StitchedQAWindows`, `StitchedQA`, `StitchedReview` |
| Optimize | `StitchedCalibrationPlan`, `StitchedCalibrationImages`, `StitchedCalibrationFinalize`, `StitchedOptimize` |
| Decode | `StitchedDecodePlan`, `StitchedDecode`, `StitchedRawFinalize` |
| Filter | `StitchedAdaptiveFilter`, `StitchedDeduplicate`, `StitchedExport`, `StitchedQuality` |
| Partition | `StitchedPartitionSetup`, `StitchedPartitionGeometry`, `StitchedPartitionCoordinates`, `StitchedPartitionJoin`, `StitchedPartitionExport`, `StitchedPartitionVerify`, `StitchedCellCenters` |
| Report | `StitchedAnalysis` (H5AD, clusters, notebook, HTML) |

## Registration review

`StitchedReview` stops until a reviewer accepts the registration. It writes
`summary.json`, `evidence.json`, `all_windows.json` and
`raw_image_overlays.png` under
`StitchedInitialize/stitched_run/registration_review/`. Accept by writing the
decision file named in the json:

    {
      "decision": "accept_common_support",
      "evidence_sha256": "<hash from the current evidence>",
      "reviewed_by": "<name>",
      "acknowledged_pointwise_exceptions": [],
      "acknowledged_unsupported_sources": {},
      "limitations": ["<reviewed limitations>"]
    }

Copy the exception list and the unsupported-source map from `summary.json`,
then rerun `StitchedReview`.

## Outputs

Under `StitchedInitialize/stitched_run/`:

| Output | Path |
| --- | --- |
| Accepted registration | `registration_accepted_radial_v1.json` |
| Scale factors and convergence | `full_optimization_v1/optimization.json` |
| Filtered, deduplicated barcodes | `full_filter_v1/exports/adaptive_filtered_z_deduplicated.csv.gz` |
| Counts and blank estimate | `full_filter_v1/exports/complete.json` |
| Cell x gene matrix and metadata | `partition_sjoin_v2/exports/` |
| Report | `basic_analysis/` |

Each stage writes a receipt with file hashes; a stage whose receipt verifies is
not rerun, and a failed partial stage is not overwritten.
