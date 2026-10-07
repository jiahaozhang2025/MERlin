"""FiducialPolynomialWarp3D: the centre displacement of a stack whose angle
finalize() substitutes.

A stack whose own angle fell back to 0 had its xy measured on UNROTATED
planes. A translation correlation of two frames that differ by a small turn
measures the displacement at the centroid of their overlap, not at the frame
centre: with a rigid shift s the two differ by (R(theta) - I) s / 2, about
|theta| |s| / 2. finalize() therefore corrects the shift when it gives such a
stack its neighbours' angle.

Four fovs on a 2 x 2 grid, each with a reference and a moving bead stack. The
moving round is the reference turned by THETA about the frame centre and
displaced by a large T. The per-plane angle measurement is stubbed (the donor
fovs get the true angle, the target fov none, as on an edge fov); everything
else -- rigid offset, z, xy, fits, tables, finalize() -- is the task's own
code on real correlations.

Run with `python -m unittest tests/test_polywarp_angle_shift.py`.
"""
import contextlib
import io
import unittest

import cv2
import numpy as np
import pandas as pd

from merlin.analysis.warp import FiducialPolynomialWarp3D

SIZE = 512
NPLANES = 12
Z = np.arange(NPLANES, dtype=float)
FOVS = [0, 1, 2, 3]
TARGET = 3
THETA = np.radians(0.5)
T = np.array([-150.0, 120.0])           # (x, y) displacement of the centre
C = (SIZE - 1) / 2.0


def truth():
    """output->input map: the content at reference p sits in the moving frame
    at c + R(theta) (p - c) + T"""
    c, s = np.cos(THETA), np.sin(THETA)
    return np.array([[c, -s, C - c * C + s * C + T[0]],
                     [s, c, C - s * C - c * C + T[1]], [0, 0, 1.0]])


def beads(seed, n=1400):
    rng = np.random.default_rng(seed)
    image = np.zeros((SIZE, SIZE), np.float32)
    xy = rng.integers(8, SIZE - 8, (n, 2))
    np.add.at(image, (xy[:, 1], xy[:, 0]), rng.uniform(800, 3000, n))
    return cv2.GaussianBlur(image, (0, 0), 1.2)


def moved(image):
    return cv2.warpAffine(image, truth()[:2], (SIZE, SIZE),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def noise(seed):
    # small against the 200 offset: the median pre-filter casts to uint16
    return np.random.default_rng(seed).normal(0, 30, (SIZE, SIZE)).astype(
        np.float32)


class _Org:
    def get_data_channels(self):
        return np.array([0, 1])

    def get_fiducial3D_filename(self, dataChannel, fov):
        return (dataChannel, fov)

    def get_fiducial3D_stack_frame_indices(self, dataChannel):
        return np.arange(NPLANES)

    def get_fiducial3D_stack_frame_zPos(self, dataChannel):
        return Z

    def get_data_channel_color(self, dataChannel):
        return '650'


class _DataSet:
    def __init__(self):
        self.saved = {}
        self.stacks, self.twoD = {}, {}
        for fov in FOVS:
            fixed = [beads(100 * fov + k) for k in range(NPLANES)]
            self.stacks[0, fov] = [f + 100 for f in fixed]
            self.stacks[1, fov] = [moved(f) + 100 for f in fixed]
            fixed2D = beads(9000 + fov, n=700)
            self.twoD[0, fov] = fixed2D + 200 + noise(fov)
            self.twoD[1, fov] = moved(fixed2D) + 200 + noise(50 + fov)

    def get_data_organization(self):
        return _Org()

    def get_fovs(self):
        return FOVS

    def get_fov_offset(self, fov):
        return (280.0 * (fov % 2), 280.0 * (fov // 2))

    def get_z_positions(self):
        return list(Z)

    def get_image_dimensions(self):
        return (SIZE, SIZE)

    def get_fiducial_image(self, dataChannel, fov):
        return self.twoD[dataChannel, fov]

    def load_image(self, name, frame):
        return self.stacks[name][int(frame)]

    def save_dataframe_to_csv(self, dataframe, resultName, analysisTask=None,
                              resultIndex=None, subdirectory=None, **kwargs):
        self.saved[resultName, resultIndex] = dataframe.to_csv(index=False)

    def load_dataframe_from_csv(self, resultName, analysisTask=None,
                                resultIndex=None, subdirectory=None, **kwargs):
        if (resultName, resultIndex) not in self.saved:
            raise FileNotFoundError(resultName)
        return pd.read_csv(io.StringIO(self.saved[resultName, resultIndex]),
                           **kwargs)

    def save_numpy_analysis_result(self, analysisResult, resultName,
                                   analysisName, resultIndex=None,
                                   subdirectory=None):
        self.saved[resultName, resultIndex] = np.array(analysisResult)

    def load_numpy_analysis_result(self, resultName, analysisName,
                                   resultIndex=None, subdirectory=None):
        if (resultName, resultIndex) not in self.saved:
            raise FileNotFoundError(resultName)
        return self.saved[resultName, resultIndex].copy()


def run():
    task = FiducialPolynomialWarp3D(_DataSet(), {
        'rotation_correction': True, 'rigid_search_bound': 220,
        'z_search_range': 2, 'z_sample_step': 1, 'xy_sample_step': 1,
        'z_polynomial_order': 0, 'xy_polynomial_order': 1,
        'minimum_overlap_pixels': 1000, 'edge_width_to_remove': 5,
        'z_min_score': 0.2, 'z_min_good_samples': 3,
        'z_neighbor_min_samples': 3, 'rotation_min_planes': 3,
        'write_fiducial_images': False, 'write_aligned_images': False},
        'PolyWarp')

    def measure_rotation(fov, dataChannel, rigidY, rigidX, sampleZ,
                         sampleOffset, shiftCoefficients=None):
        n = len(sampleZ)
        applied = task._measurementRotation.get((fov, dataChannel), 0.0)
        theta = np.full(n, np.nan if fov == TARGET else THETA - applied)
        return (np.asarray(sampleZ, float), theta, np.full(n, 30.0),
                np.full(n, 0.1))

    task._measure_rotation = measure_rotation
    with contextlib.redirect_stdout(io.StringIO()):
        for fov in FOVS:
            task._run_analysis(fov)
        before = task.get_transformation_table(TARGET)
        task.finalize()
    return task, before


def error_at(matrix, points):
    expected = points @ truth()[:2, :2].T + truth()[:2, 2]
    return np.linalg.norm(points @ matrix[:2, :2].T + matrix[:2, 2]
                          - expected, axis=1)


POINTS = np.array([[C, C], [60, 60], [SIZE - 60, 60], [60, SIZE - 60],
                   [SIZE - 60, SIZE - 60]])


class AngleShiftTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.task, cls.before = run()

    def test_premise(self):
        status = self.task.get_rotation_status(TARGET).set_index(
            'dataChannel')
        self.assertEqual(status.loc[1, 'source'], 'neighbor')
        self.assertEqual(status.loc[1, 'first_pass_radians'], 0.0)
        self.assertAlmostEqual(
            self.task.get_polynomial_coefficients(TARGET).set_index(
                'dataChannel').loc[1, 'rotation_radians'], THETA, places=12)
        # the unrotated measurement is the displacement at the overlap
        # centroid, c - T / 2, not at the centre
        row = self.before[(self.before.dataChannel == 1)
                          & (self.before.zIndex == 6)].iloc[0]
        centroid = np.array([C, C]) - T / 2
        expected = truth()[:2, :2] @ centroid + truth()[:2, 2] - centroid
        np.testing.assert_allclose([row.xshift, row.yshift], expected,
                                   atol=0.15)

    def test_substituted_transform_matches_truth(self):
        for zIndex in (2, 6, 9):
            matrix = self.task.get_transformation(TARGET, 1, zIndex).params
            self.assertLess(error_at(matrix, POINTS).max(), 0.15, zIndex)
            # the donors, measured on turned-back planes, for comparison
            donor = self.task.get_transformation(0, 1, zIndex).params
            self.assertLess(error_at(donor, POINTS).max(), 0.15, zIndex)

    def test_without_the_correction_the_centre_is_off(self):
        # the fragment's shift with the new angle about the centre, i.e. the
        # earlier behaviour: off by about |theta| |T| / 2 everywhere
        row = self.before[(self.before.dataChannel == 1)
                          & (self.before.zIndex == 6)].iloc[0]
        c, s = np.cos(THETA), np.sin(THETA)
        uncorrected = np.array([
            [c, -s, C - c * C + s * C + row.xshift],
            [s, c, C - s * C - c * C + row.yshift], [0, 0, 1.0]])
        expected = 2 * np.sin(THETA / 2) * np.hypot(*T) / 2
        self.assertGreater(expected, 0.8)
        np.testing.assert_allclose(error_at(uncorrected, POINTS), expected,
                                   atol=0.15)


if __name__ == '__main__':
    unittest.main()
