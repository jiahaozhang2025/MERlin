"""The fovs and z plane that tasks write example images for by default.

Preprocess, Decode and the adaptive barcode filters each write images for a
few fovs, so every run leaves something to look at without writing the whole
dataset. All three draw them here, from the same seed and in the same order,
so by default they write the SAME fovs and the SAME plane and the
preprocessed, decoded and filtered images of one field line up.

The picks are fragment indexes, which is what these tasks compare against
(fragmentIndex in write_*_FOVs); they equal fov ids whenever fovs run 0..N-1.
They depend only on the seed and on the dataset's fov and z counts, so a task
rebuilt from its task.json reproduces them and regeneration sees no change.
"""
from typing import List, Tuple

import numpy as np

DEFAULT_SEED = 1
DEFAULT_FOV_COUNT = 3


def default_image_selection(dataSet, seed: int = DEFAULT_SEED,
                            fovCount: int = DEFAULT_FOV_COUNT
                            ) -> Tuple[List[int], List[int]]:
    """Seeded random choice of fovCount fovs and one z index.

    Returns:
        (sorted fragment indexes, [z index])
    """
    rng = np.random.default_rng(int(seed))
    fragmentCount = len(dataSet.get_fovs())
    chosen = rng.choice(fragmentCount, size=min(fovCount, fragmentCount),
                        replace=False)
    zIndex = int(rng.integers(len(dataSet.get_z_positions())))
    return sorted(int(f) for f in chosen), [zIndex]
