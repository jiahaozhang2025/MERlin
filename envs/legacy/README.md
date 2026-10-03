# Legacy `merlin` env (retired 2026-10-03)

The env MERlin ran in until 2026-10-03: Python 3.9.0, numpy 1.26.4, pandas 1.5.3, cellpose 2.2,
snakemake 7.32.4, tensorflow 2.8.4. It was replaced by the Python 3.12 / cellpose 4 env built by
`../setup_merlin_env.sh` after the side-by-side check described in `../../DETAILED_CHANGES.md`.

To rebuild it, e.g. to reproduce a cellpose 2/3 segmentation with the code before the
Cellpose-SAM-only change:

    conda create -p <path> --file merlin_py39_cellpose2_conda_explicit.txt
    <path>/bin/python -m pip install -r merlin_py39_cellpose2_pip.txt

Then check out MERlin from before 2026-10-02. The current code needs cellpose >= 4 to segment.
