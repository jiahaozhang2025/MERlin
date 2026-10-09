"""FiducialPolynomialWarp3D rigid_only_channels, with the REAL measurement code.

test_polywarp_rigid_only.py stubs the measurements, so it assumes the premise
of the rigid-only table fix: that a fitted channel's y/x polynomials are the
TOTAL shift (its xy samples are absolute peaks in a window centred on the
rigid offset), so a rigid-only channel's polynomial must be the constant rigid
offset. Here nothing in the task is stubbed: rigid offsets, z search, xy
search, fits, the table and get_aligned_image run on synthetic 3D bead stacks
with known shifts. Before the fix the rigid-only rows were off by exactly the
rigid offset (about 20 px in the aligned image).

channel 0: reference stack, no shift
channel 1: fitted stack, z offset +0.3 um, xy = rigid + linear drift in depth
channel 2: rigid-only (a separate "nuclear" stack), constant (17.4, -11.2) px
channel 3: rigid-only, the same stack as channel 2

Adapted from the 2026-10-07 verification script (rigid_e2e_check.py).
Run with `python -m unittest tests/test_polywarp_rigid_only_e2e.py`.
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation

from merlin.analysis.warp import FiducialPolynomialWarp3D

SIZE = 256
Z = np.arange(48) * 0.5
STACK = {0: 'r0', 1: 'r1', 2: 'nuc', 3: 'nuc'}
RIGID = {0: (0.0, 0.0), 1: (6.3, -4.7), 2: (17.4, -11.2), 3: (17.4, -11.2)}
DZ = {0: 0.0, 1: 0.3, 2: 0.0, 3: 0.0}

_rng = np.random.default_rng(5)
BEADS = np.c_[_rng.uniform(5, SIZE - 5, (900, 2)), _rng.uniform(-2, 26, 900)]
TISSUE = _rng.uniform(5, SIZE - 5, (500, 2))


def drift(dataChannel, depth):
    """Total (dy, dx) shift of the channel at physical depth `depth`."""
    ry, rx = RIGID[dataChannel]
    if dataChannel == 1:
        return ry + 0.02 * depth, rx - 0.015 * depth
    return ry, rx


def render(rows, cols, weights=None, sigma=1.5):
    image = np.zeros((SIZE, SIZE), np.float64)
    r = 6
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    weights = np.ones(len(rows)) if weights is None else weights
    for y, x, a in zip(rows, cols, weights):
        if a < 1e-3:
            continue
        iy, ix = int(round(y)), int(round(x))
        if r <= iy < SIZE - r and r <= ix < SIZE - r:
            image[iy - r:iy + r + 1, ix - r:ix + r + 1] += 2000 * a * np.exp(
                -((yy + iy - y) ** 2 + (xx + ix - x) ** 2) / (2 * sigma ** 2))
    return image


def bead_plane(dataChannel, zMeasured):
    depth = zMeasured - DZ[dataChannel]
    sy, sx = drift(dataChannel, depth)
    weights = np.exp(-(BEADS[:, 2] - depth) ** 2 / 2.0)
    return (render(BEADS[:, 0] - sy, BEADS[:, 1] - sx, weights)
            + 100).astype(np.uint16)


STACKS = {}
for _channel, _name in STACK.items():
    if _name not in STACKS:
        STACKS[_name] = [bead_plane(_channel, z) for z in Z]


class _Org:
    def get_data_channels(self):
        return np.array([0, 1, 2, 3])

    def get_fiducial3D_filename(self, dataChannel, fov):
        return STACK[dataChannel]

    def get_fiducial3D_stack_frame_indices(self, dataChannel):
        return list(range(len(Z)))

    def get_fiducial3D_stack_frame_zPos(self, dataChannel):
        return list(Z)

    def get_data_channel_color(self, dataChannel):
        return '650'


class _DataSet:
    def __init__(self):
        self.saved = {}

    def get_data_organization(self):
        return _Org()

    def get_fovs(self):
        return [0]

    def get_z_positions(self):
        return list(Z)

    def get_image_dimensions(self):
        return (SIZE, SIZE)

    def load_image(self, name, frame):
        return STACKS[name][frame]

    def get_fiducial_image(self, dataChannel, fov):
        return STACKS[STACK[dataChannel]][0]

    def get_raw_image(self, dataChannel, fov, zPosition):
        sy, sx = drift(dataChannel, zPosition - DZ[dataChannel])
        return (render(TISSUE[:, 0] - sy, TISSUE[:, 1] - sx)
                + 50).astype(np.uint16)

    def save_dataframe_to_csv(self, dataframe, resultName, analysisTask=None,
                              resultIndex=None, subdirectory=None, **kwargs):
        self.saved[resultName, resultIndex] = dataframe.to_csv(index=False)

    def load_dataframe_from_csv(self, resultName, analysisTask=None,
                                resultIndex=None, subdirectory=None, **kwargs):
        if (resultName, resultIndex) not in self.saved:
            raise FileNotFoundError(resultName)
        return pd.read_csv(io.StringIO(self.saved[resultName, resultIndex]))

    def save_numpy_analysis_result(self, analysisResult, resultName,
                                   analysisName, resultIndex=None,
                                   subdirectory=None):
        self.saved[resultName, resultIndex] = np.array(analysisResult)


def run(rigidOnly):
    dataSet = _DataSet()
    task = FiducialPolynomialWarp3D(dataSet, {
        'rigid_only_channels': list(rigidOnly), 'highpass_sigma': 3,
        'median_filter': False, 'write_aligned_images': False,
        'write_fiducial_images': False, 'z_search_range': 4,
        'z_sample_step': 3, 'xy_sample_step': 2, 'edge_width_to_remove': 10,
        'rigid_search_bound': 60, 'rigid_plane_search': 0,
        'z_neighbor_substitution': False}, 'PolyWarp')
    with contextlib.redirect_stdout(io.StringIO()):
        task._run_analysis(0)
    return task


REFERENCE = render(TISSUE[:, 0], TISSUE[:, 1]) + 50


class RigidOnlyEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rigid = run([2, 3])
        cls.none = run([])

    def _table_error(self, task, dataChannel):
        table = task.get_transformation_table(0)
        rows = table[table.dataChannel == dataChannel]
        truth = np.array([drift(dataChannel, z) for z in rows.zPos])
        return max(np.abs(-rows.yshift.to_numpy() - truth[:, 0]).max(),
                   np.abs(-rows.xshift.to_numpy() - truth[:, 1]).max())

    def _aligned_residual(self, task, dataChannel):
        residuals = []
        for zIndex in (8, 24, 40):
            aligned = task.get_aligned_image(0, dataChannel, zIndex)
            shift, _, _ = phase_cross_correlation(
                REFERENCE, aligned.astype(float), upsample_factor=20)
            residuals.append(float(np.hypot(*shift)))
        return max(residuals)

    def test_rigid_only_rows_carry_the_rigid_offset(self):
        for dataChannel in (2, 3):
            self.assertLess(self._table_error(self.rigid, dataChannel), 0.2)
            self.assertLess(self._aligned_residual(self.rigid, dataChannel),
                            0.5)

    def test_fitted_channel_is_a_total_shift(self):
        # the premise: a fitted channel's polynomial is the total shift
        self.assertLess(self._table_error(self.rigid, 1), 0.3)
        self.assertLess(self._aligned_residual(self.rigid, 1), 0.5)
        table = self.rigid.get_transformation_table(0)
        np.testing.assert_allclose(
            table[table.dataChannel == 1].z_offset, DZ[1], atol=0.1)

    def test_rigid_only_matches_fitted_on_a_rigid_stack(self):
        # the nuclear stack has a constant shift, so depth-fitting it must
        # give the same xy as treating it as rigid-only
        for dataChannel in (2, 3):
            self.assertLess(self._table_error(self.none, dataChannel), 0.2)

    def test_fitted_rows_unchanged_by_rigid_only(self):
        rigid = self.rigid.get_transformation_table(0)
        none = self.none.get_transformation_table(0)
        for dataChannel in (0, 1):
            pd.testing.assert_frame_equal(
                rigid[rigid.dataChannel == dataChannel].reset_index(drop=True),
                none[none.dataChannel == dataChannel].reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()
