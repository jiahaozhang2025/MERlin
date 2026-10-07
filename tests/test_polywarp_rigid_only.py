"""FiducialPolynomialWarp3D with rigid_only_channels.

A rigid-only channel skips the depth fit but must still be registered by its
rigid xy offset. The transformation table (and so get_aligned_image and
get_transformation) is built from the y/x polynomials, so those polynomials
must carry the rigid offset as their constant term; all-zero polynomials left
the channel unregistered while polynomial_coefficients still reported the
rigid offset.

The measurement methods are stubbed with known offsets; everything from the
stack grouping to the saved tables is the task's own code.

Run with `python -m unittest tests/test_polywarp_rigid_only.py`.
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis.warp import FiducialPolynomialWarp3D

Z = np.arange(0.0, 30.0, 0.5)                  # 60 planes, 0.5 um apart
CHANNELS = [0, 1, 2, 3]
STACK = {0: 'round0', 1: 'round1', 2: 'round1', 3: 'nuclear'}
RIGID = {0: (0.0, 0.0), 1: (5.0, -3.0), 2: (5.0, -3.0), 3: (12.0, -7.0)}
RIGID_ONLY = 3


def drift(dataChannel, z):
    """Known depth drift of the channel's fiducial stack: (dz, dy - rigidY,
    dx - rigidX). Channels sharing a stack share its drift."""
    k = {'round0': 1, 'round1': 2, 'nuclear': 3}[STACK[dataChannel]]
    return (0.01 * k * z + 0.05 * k, 0.02 * k * z,
            -0.01 * k * z + 2e-4 * k * z ** 2)


class _Org:
    def get_data_channels(self):
        return np.array(CHANNELS)

    def get_fiducial3D_filename(self, dataChannel, fov):
        return STACK[dataChannel]

    def get_data_channel_color(self, dataChannel):
        return '650'


class _DataSet:
    """Keeps every saved table as CSV text, as the real dataset does."""

    def __init__(self, raw=None):
        self.saved = {}
        self.raw = raw

    def get_data_organization(self):
        return _Org()

    def get_fovs(self):
        return [0]

    def get_z_positions(self):
        return list(Z)

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

    def get_raw_image(self, dataChannel, fov, zPosition):
        return self.raw[dataChannel].copy()


def run_task(rigidOnly, raw=None):
    """Run one fov through _run_analysis with the measurements stubbed."""
    dataSet = _DataSet(raw)
    task = FiducialPolynomialWarp3D(
        dataSet, {'rigid_only_channels': rigidOnly}, 'PolyWarp')

    def measure_z(fov, dataChannel, rigidY, rigidX):
        z = Z[7:-7:4]
        return z, drift(dataChannel, z)[0], np.full(len(z), 0.9), 3.0

    def measure_xy(fov, dataChannel, rigidY, rigidX, zCoefficients):
        z = Z[1:-1:3]
        _, dy, dx = drift(dataChannel, z)
        return (z, rigidY + dy, rigidX + dx, np.full(len(z), 50.0),
                np.zeros(len(z), bool))

    task._measure_rigid_offsets = lambda fov: [RIGID[c] for c in CHANNELS]
    task._measure_z_offsets = measure_z
    task._measure_xy_offsets = measure_xy
    with contextlib.redirect_stdout(io.StringIO()):
        task._run_analysis(0)
    return task


class RigidOnlyChannelTests(unittest.TestCase):

    def test_table_carries_rigid_offset(self):
        task = run_task([RIGID_ONLY])
        rigidY, rigidX = RIGID[RIGID_ONLY]
        table = task.get_transformation_table(0)
        rows = table[table.dataChannel == RIGID_ONLY]
        self.assertEqual(len(rows), len(Z))
        # stored as SimilarityTransform translations: the negated shift
        np.testing.assert_array_equal(rows.yshift, -rigidY)
        np.testing.assert_array_equal(rows.xshift, -rigidX)
        np.testing.assert_array_equal(rows.z_offset, 0.0)
        np.testing.assert_array_equal(rows.zPos_new, rows.zPos)

        coeff = task.get_polynomial_coefficients(0).set_index('dataChannel')
        row = coeff.loc[RIGID_ONLY]
        self.assertEqual((row.rigid_y, row.rigid_x), (rigidY, rigidX))
        self.assertEqual((row.y_c0, row.x_c0), (rigidY, rigidX))
        for k in range(1, task.parameters['xy_polynomial_order'] + 1):
            self.assertEqual((row['y_c%d' % k], row['x_c%d' % k]), (0.0, 0.0))
        for k in range(task.parameters['z_polynomial_order'] + 1):
            self.assertEqual(row['z_c%d' % k], 0.0)

        status = task.get_z_fit_status(0).set_index('dataChannel')
        self.assertEqual(status.loc[RIGID_ONLY, 'source'], 'rigid_only')

        # every plane's transform equals the rigid transform in offsets_0
        offsets = task.dataSet.saved['offsets', 0]
        for zIndex in (0, len(Z) // 2, len(Z) - 1):
            np.testing.assert_array_equal(
                task.get_transformation(0, RIGID_ONLY, zIndex).params,
                offsets[CHANNELS.index(RIGID_ONLY)])

    def test_aligned_image_is_shifted_by_rigid_offset(self):
        rigidY, rigidX = (int(v) for v in RIGID[RIGID_ONLY])
        raw = {c: np.zeros((96, 96), np.uint16) for c in CHANNELS}
        raw[RIGID_ONLY][40, 50] = 1000
        raw[RIGID_ONLY][60:64, 20:23] = 300
        task = run_task([RIGID_ONLY], raw)
        aligned = task.get_aligned_image(0, RIGID_ONLY, len(Z) // 2)
        # (dy, dx) is the shift that brings the channel onto round 0
        expected = np.zeros_like(raw[RIGID_ONLY])
        expected[40 + rigidY, 50 + rigidX] = 1000
        expected[60 + rigidY:64 + rigidY, 20 + rigidX:23 + rigidX] = 300
        np.testing.assert_array_equal(aligned, expected)

    def test_fitted_channels_unchanged(self):
        """Declaring a channel rigid-only changes nothing for the others, and
        their tables are their fitted total drift."""
        withRigid = run_task([RIGID_ONLY])
        without = run_task([])
        tA = withRigid.get_transformation_table(0)
        tB = without.get_transformation_table(0)
        cA = withRigid.get_polynomial_coefficients(0)
        cB = without.get_polynomial_coefficients(0)
        fitted = [c for c in CHANNELS if c != RIGID_ONLY]
        pd.testing.assert_frame_equal(
            tA[tA.dataChannel.isin(fitted)].reset_index(drop=True),
            tB[tB.dataChannel.isin(fitted)].reset_index(drop=True))
        pd.testing.assert_frame_equal(
            cA[cA.dataChannel.isin(fitted)].reset_index(drop=True),
            cB[cB.dataChannel.isin(fitted)].reset_index(drop=True))
        for c in CHANNELS:
            rows = tB[tB.dataChannel == c]
            dz, dy, dx = drift(c, rows.zPos.to_numpy())
            np.testing.assert_allclose(rows.z_offset, dz, atol=1e-9)
            np.testing.assert_allclose(rows.yshift, -(RIGID[c][0] + dy),
                                       atol=1e-9)
            np.testing.assert_allclose(rows.xshift, -(RIGID[c][1] + dx),
                                       atol=1e-9)


if __name__ == '__main__':
    unittest.main()
