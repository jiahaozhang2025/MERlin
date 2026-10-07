"""FiducialPolynomialWarp3D: finalize() measures the xy of a stack whose angle
it substitutes again, on fiducial planes turned back by the new angle.

The fragment measured that stack's xy on planes turned back by its own
first-pass angle, or on unrotated planes when that fell back to 0. A
translation correlation of two frames that still differ by a small turn
measures the displacement at the bead-weighted centroid p_w of their overlap.
The overlap-centroid formula (_centroid_shift_correction) assumes p_w is the
geometric centroid p_o of the overlap, which fails when the beads fill only
part of the frame -- the usual reason a stack's angle needs substituting.

Four fovs on a 2 x 2 grid, each with a reference and a moving bead stack. The
moving round is the reference turned by THETA about the frame centre and
displaced by a large T. The target fov's beads lie only in a band across the
top of the reference frame, about 140 px from p_o, so at 0.5 deg the formula
leaves the centre about 1.2 px off; the donor fovs are evenly filled. The
per-plane angle measurement is stubbed (the donors get the true angle, the
target none, as on an edge fov); everything else -- rigid offset, z, xy,
fits, tables, finalize() -- is the task's own code on real correlations.

Run with `python -m unittest tests/test_polywarp_angle_remeasure.py`.
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
BAND = (8, 100)                         # reference rows holding the beads


def truth():
    """output->input map: the content at reference p sits in the moving frame
    at c + R(theta) (p - c) + T"""
    c, s = np.cos(THETA), np.sin(THETA)
    return np.array([[c, -s, C - c * C + s * C + T[0]],
                     [s, c, C - s * C - c * C + T[1]], [0, 0, 1.0]])


def beads(seed, rows=(8, SIZE - 8), density=1400 / (SIZE - 16) ** 2):
    rng = np.random.default_rng(seed)
    n = int(round(density * (rows[1] - rows[0]) * (SIZE - 16)))
    image = np.zeros((SIZE, SIZE), np.float32)
    x = rng.integers(8, SIZE - 8, n)
    y = rng.integers(rows[0], rows[1], n)
    np.add.at(image, (y, x), rng.uniform(800, 3000, n))
    return cv2.GaussianBlur(image, (0, 0), 1.2)


def moved(image):
    return cv2.warpAffine(image, truth()[:2], (SIZE, SIZE),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def noise(seed):
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
            rows = BAND if fov == TARGET else (8, SIZE - 8)
            fixed = [beads(100 * fov + k, rows) for k in range(NPLANES)]
            self.stacks[0, fov] = [f + 100 for f in fixed]
            self.stacks[1, fov] = [moved(f) + 100 for f in fixed]
            fixed2D = beads(9000 + fov, rows, density=700 / (SIZE - 16) ** 2)
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


PARAMETERS = {
    'rotation_correction': True, 'rigid_search_bound': 220,
    'z_search_range': 2, 'z_sample_step': 1, 'xy_sample_step': 1,
    'z_polynomial_order': 0, 'xy_polynomial_order': 1,
    'minimum_overlap_pixels': 1000, 'edge_width_to_remove': 5,
    'z_min_score': 0.2, 'z_min_good_samples': 3,
    'z_neighbor_min_samples': 3, 'rotation_min_planes': 3,
    'write_fiducial_images': False, 'write_aligned_images': False}


def make_task(dataSet, **extra):
    task = FiducialPolynomialWarp3D(dataSet, dict(PARAMETERS, **extra),
                                    'PolyWarp')

    def measure_rotation(fov, dataChannel, rigidY, rigidX, sampleZ,
                         sampleOffset, shiftCoefficients=None):
        n = len(sampleZ)
        applied = task._measurementRotation.get((fov, dataChannel), 0.0)
        theta = np.full(n, np.nan if fov == TARGET else THETA - applied)
        return (np.asarray(sampleZ, float), theta, np.full(n, 30.0),
                np.full(n, 0.1))

    task._measure_rotation = measure_rotation
    return task


def snapshot(saved):
    return {k: (v.tobytes() if isinstance(v, np.ndarray) else v)
            for k, v in saved.items()}


def error_at(matrix, points):
    expected = points @ truth()[:2, :2].T + truth()[:2, 2]
    return np.linalg.norm(points @ matrix[:2, :2].T + matrix[:2, 2]
                          - expected, axis=1)


def row_matrix(theta, xshift, yshift):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, C - c * C + s * C + xshift],
                     [s, c, C - s * C - c * C + yshift], [0, 0, 1.0]])


POINTS = np.array([[C, C], [60, 60], [SIZE - 60, 60], [60, SIZE - 60],
                   [SIZE - 60, SIZE - 60]])


class AngleRemeasureTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataSet = _DataSet()
        cls.task = make_task(dataSet)
        with contextlib.redirect_stdout(io.StringIO()):
            for fov in FOVS:
                cls.task._run_analysis(fov)
        cls.fragment = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                        for k, v in dataSet.saved.items()}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cls.task.finalize()
        cls.log = out.getvalue()
        cls.after = snapshot(dataSet.saved)

        # the same fragment outputs, finalized with the re-measurement made
        # to fail its sample count: the overlap-centroid formula fallback
        fallbackSet = _DataSet()
        fallbackSet.saved = {k: (v.copy() if isinstance(v, np.ndarray)
                                 else v) for k, v in cls.fragment.items()}
        cls.fallback = make_task(fallbackSet, min_xy_planes_kept=1.01)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cls.fallback.finalize()
        cls.fallbackLog = out.getvalue()

    def fragment_row(self, zIndex):
        table = pd.read_csv(io.StringIO(
            self.fragment['transformation_table', TARGET]))
        return table[(table.dataChannel == 1)
                     & (table.zIndex == zIndex)].iloc[0]

    def report(self, task):
        return task.dataSet.load_dataframe_from_csv(
            'rotation_neighbor_substitutions').set_index(
                ['fov', 'dataChannel'])

    def test_premise(self):
        status = self.task.get_rotation_status(TARGET).set_index(
            'dataChannel')
        self.assertEqual(status.loc[1, 'source'], 'neighbor')
        self.assertEqual(status.loc[1, 'first_pass_radians'], 0.0)
        self.assertAlmostEqual(
            self.task.get_polynomial_coefficients(TARGET).set_index(
                'dataChannel').loc[1, 'rotation_radians'], THETA, places=12)
        # the unrotated measurement is the displacement at the bead-weighted
        # centroid of the overlap (the band), not at its geometric centroid
        # c - T / 2, about 140 px further down
        row = self.fragment_row(6)
        bandCentroid = np.array([C - T[0] / 2, 0.5 * (BAND[0] + BAND[1] - 1)])
        overlapCentroid = np.array([C, C]) - T / 2
        self.assertGreater(np.hypot(*(bandCentroid - overlapCentroid)), 130)

        def displacement(p):
            return truth()[:2, :2] @ p + truth()[:2, 2] - p
        np.testing.assert_allclose([row.xshift, row.yshift],
                                   displacement(bandCentroid), atol=0.25)
        self.assertGreater(np.hypot(*(np.array([row.xshift, row.yshift])
                                      - displacement(overlapCentroid))), 1.0)

    def test_remeasured_transform_matches_truth(self):
        self.assertEqual(self.report(self.task).loc[(TARGET, 1), 'xy_method'],
                         'remeasured')
        for zIndex in (2, 6, 9):
            matrix = self.task.get_transformation(TARGET, 1, zIndex).params
            self.assertLess(error_at(matrix, POINTS).max(), 0.2, zIndex)

    def test_formula_would_be_about_a_pixel_off(self):
        row = self.fragment_row(6)
        coeff = pd.read_csv(io.StringIO(
            self.fragment['polynomial_coefficients', TARGET])).set_index(
                'dataChannel')
        dY, dX = self.task._centroid_shift_correction(
            coeff.loc[1, 'rigid_y'], coeff.loc[1, 'rigid_x'], 0.0, THETA)
        formula = row_matrix(THETA, row.xshift - dX, row.yshift - dY)
        errors = error_at(formula, POINTS)
        self.assertGreater(errors.min(), 0.9)
        self.assertLess(errors.max(), 1.6)
        # the report gives the formula's term beside the change applied
        rec = self.report(self.task).loc[(TARGET, 1)]
        self.assertAlmostEqual(rec.formula_xshift_px, -dX, places=12)
        self.assertAlmostEqual(rec.formula_yshift_px, -dY, places=12)
        self.assertGreater(np.hypot(rec.xshift_change_px - rec.formula_xshift_px,
                                    rec.yshift_change_px
                                    - rec.formula_yshift_px), 0.9)

    def test_fallback_is_flagged_and_uses_the_formula(self):
        rec = self.report(self.fallback).loc[(TARGET, 1)]
        self.assertEqual(rec.xy_method, 'formula_fallback')
        self.assertIn('kept by the fit', rec.xy_failure)
        self.assertIn('NOT re-measured', self.fallbackLog)
        self.assertIn('formula fallback for 1', self.fallbackLog)
        row = self.fragment_row(6)
        after = self.fallback.get_transformation_table(TARGET)
        new = after[(after.dataChannel == 1) & (after.zIndex == 6)].iloc[0]
        self.assertAlmostEqual(new.xshift - row.xshift, rec.formula_xshift_px,
                               places=9)
        self.assertAlmostEqual(new.yshift - row.yshift, rec.formula_yshift_px,
                               places=9)
        matrix = self.fallback.get_transformation(TARGET, 1, 6).params
        self.assertGreater(error_at(matrix, POINTS).min(), 0.9)

    def test_xy_samples_written(self):
        samples = self.task.dataSet.load_dataframe_from_csv(
            'rotation_neighbor_xy_samples')
        self.assertEqual(set(zip(samples.fov, samples.dataChannel)),
                         {(TARGET, 1)})
        self.assertEqual(len(samples), NPLANES - 2)
        self.assertTrue(samples.ok.all())
        self.assertAlmostEqual(samples.rotation_radians.iloc[0], THETA,
                               places=12)

    def test_other_fovs_untouched(self):
        fragment = snapshot(self.fragment)
        for fov in FOVS:
            if fov == TARGET:
                continue
            for name in ('transformation_table', 'polynomial_coefficients',
                         'rotation_status', 'offsets', 'z_fit_status'):
                self.assertEqual(fragment[name, fov], self.after[name, fov])

    def test_idempotent(self):
        dataSet = _DataSet()
        dataSet.saved = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                         for k, v in self.fragment.items()}
        task = make_task(dataSet)
        with contextlib.redirect_stdout(io.StringIO()):
            task.finalize()
        once = snapshot(dataSet.saved)
        self.assertEqual(once, self.after)
        calls = []
        task._measure_xy_offsets = lambda *a: calls.append(a)
        with contextlib.redirect_stdout(io.StringIO()):
            task.finalize()
        self.assertEqual(snapshot(dataSet.saved), once)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
