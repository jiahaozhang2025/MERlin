# MERlin — Jiahao Zhang research fork

This repository is a research fork of **MERlin**, an extensible pipeline for
decoding and analyzing MERFISH data. It is maintained for the workflows and
datasets used by Jiahao Zhang.

## Provenance

The project lineage is:

1. [emanuega/MERlin](https://github.com/emanuega/MERlin) — the original MERlin
   repository developed in the Xiaowei Zhuang laboratory.
2. [aaronhalpern/MERlin](https://github.com/aaronhalpern/MERlin) — Aaron
   Halpern's fork.
3. Aaron Halpern's `gpu_decoding` branch at commit
   [`15ce55f`](https://github.com/aaronhalpern/MERlin/commit/15ce55f919eed7e986c7a9c79eb59ef41f39e1c8)
   — the baseline for this fork.
4. [jiahaozhang2025/MERlin](https://github.com/jiahaozhang2025/MERlin) — this
   repository, containing the subsequent adaptations.

The Git histories are separate. The `forked` branch preserves Aaron's
`gpu_decoding` history and is pinned to its baseline commit. The local history
on `main` begins with root commit `5dbbd02`, so Git cannot connect the earlier
adaptation work to Aaron's commits even though that fork was its source. The
`main` branch contains the current code and the subsequent commits that are
still available.

## What MERlin does

MERlin organizes a MERFISH workflow as a collection of analysis tasks. Tasks
can run locally or be split into fragments for parallel execution through
Snakemake and a cluster scheduler. The pipeline covers image registration,
preprocessing, barcode decoding and filtering, cell segmentation, and export
of spatial features and barcode data.

## Main differences in this fork

Compared with the `forked` baseline, `main` includes:

- **More scalable decoding:** chunked NumPy or PyTorch similarity decoding,
  image tiling with overlap handling, optional GPU execution, per-z decoding,
  resumable Zarr output, unique-ID images, optional intensity-trace export,
  and additional confidence and distance controls.
- **Expanded barcode filtering:** more robust blank-fraction threshold
  selection, global and local adaptive filtering updates, logistic-regression
  filtering, optional z-duplicate removal, and saved filtering summaries.
- **Additional preprocessing:** configurable global-threshold subtraction,
  reversed preprocessing paths, updated deconvolution helpers, and a
  model-restoration preprocessing task.
- **Segmentation additions:** multi-channel Cellpose segmentation with
  optional custom models, preprocessing/image dumps, and 2D-to-3D mask
  combination.
- **3D registration support:** fiducial-stack registration, piezo-drift
  correction, z-dependent transforms, and interpolation between z planes.
- **I/O and compatibility updates:** Zarr image reading, more tolerant Google
  Cloud reads, TIFF casting controls, and updates for newer NumPy, NetworkX,
  Shapely, Cellpose, and related packages.

See [DETAILED_CHANGES.md](DETAILED_CHANGES.md) for the module-by-module
tree comparison, parameters, caveats, and exact history boundary.

## Branches

| Branch | Purpose |
| --- | --- |
| `main` | Current Jiahao Zhang research version and default branch |
| `forked` | Unmodified baseline from Aaron Halpern's `gpu_decoding` branch at `15ce55f` |

## Installation

This is research software with environment-specific components. Create an
isolated environment, install system-level dependencies such as `rtree` and
PyTables as appropriate for the platform, then install this repository:

```bash
git clone https://github.com/jiahaozhang2025/MERlin.git
cd MERlin
pip install -e .
```

GPU decoding requires a PyTorch installation compatible with the local CUDA
runtime. The model-restoration task additionally depends on an external model
package, checkpoint, and calibration files that are not distributed in this
repository.

## Configuration and use

MERlin reads its main paths from `~/.merlinenv`:

```dotenv
DATA_HOME=/path/to/raw-data
ANALYSIS_HOME=/path/to/analysis-output
PARAMETERS_HOME=/path/to/merfish-parameters
```

`PARAMETERS_HOME` contains analysis JSON files, codebooks, data-organization
tables, microscope parameters, positions, and Snakemake settings. MERlin can
create the environment file interactively:

```bash
merlin --configure .
```

A typical local run is:

```bash
merlin \
  -a analysis_parameters.json \
  -m microscope.json \
  -o data_organization.csv \
  -c codebook.csv \
  -n 5 \
  dataset_name
```

The inherited documentation in [`docs/`](docs/) describes the MERlin task
model and file layout. Some pages still reflect the original MERlin release,
so verify task parameters against the current source and the detailed change
record.

## Original authors and citation

The original MERlin repository identifies:

- George Emanuel — initial work
- Stephen Eichhorn
- Leonardo Sepulveda

If MERlin is useful for your research, cite:

> Emanuel, G., Eichhorn, S. W., and Zhuang, X. (2020). *MERlin — scalable and
> extensible MERFISH analysis software*, v0.1.6.
> [doi:10.5281/zenodo.3758540](https://doi.org/10.5281/zenodo.3758540)

This fork also acknowledges Aaron Halpern's GPU-decoding work, from which the
current branch descends.

## License

MERlin is distributed under the terms in [license.md](license.md).
