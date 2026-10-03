#!/bin/bash
# The `merlin` conda env: MERlin on Python 3.12 + Cellpose-SAM (cellpose 4). Every MERlin task,
# segmentation included, runs in it. Run on a LOGIN node (needs internet):
#   bash setup_merlin_env.sh [env path]   default: Lab/Jiahao/envs/merlin on lab storage
# The main packages are pinned to the versions MERlin was checked against; the rest resolve
# freely. Not installed: snakemake (optional; tasks are generated and run with -t).
# ~/.conda/envs/merlin is a symlink to the lab-storage env, so `conda activate merlin` and old
# scripts that name ~/.conda/envs/merlin keep working.
set -e
ENV=${1:-/n/holylfs05/LABS/zhuang_lab/Lab/Jiahao/envs/merlin}
MERLIN=$(cd "$(dirname "$0")" && pwd)
source /n/sw/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
# keep conda/pip caches and user-site packages out of the (small) home directory
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs    # local disk: mamba cache locks hang on network filesystems
export PIP_CACHE_DIR=/scratch/$USER/pip_cache
export PYTHONNOUSERSITE=1
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
mamba create -y -p "$ENV" python=3.12
conda activate "$ENV"
conda env config vars set PYTHONNOUSERSITE=1 -p "$ENV"   # ~/.local must not shadow the env
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
pip install cellpose==4.2.1.1 numpy==2.4.4 pandas==3.0.5 scikit-image==0.25.2 scipy==1.18.1 \
    tifffile==2026.3.3 shapely==2.1.2 h5py==3.16.0 tables==3.11.1 rtree==1.4.1 networkx==3.6.1 \
    opencv-python-headless==5.0.0.93 zarr==3.4.0 matplotlib==3.11.2 scikit-learn==1.9.1 \
    tensorflow==2.20.0 csbdeep==0.8.1 xmltodict boto3 pyclustering python-dotenv requests \
    seaborn Pillow
pip install --no-deps -e "$MERLIN"
python -c "from cellpose import models; models.CellposeModel(gpu=False, pretrained_model='cpsam_v2')"
python -c "import csbdeep, tensorflow as tf; print('tensorflow', tf.__version__, '| csbdeep', csbdeep.__version__)"
python -c "import merlin, cellpose, numpy, pandas, torch; from merlin.analysis import segment, decode, partition; print('merlin', merlin.version(), '| cellpose', cellpose.version, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| torch', torch.__version__)"
echo "merlin env ready at $ENV"
