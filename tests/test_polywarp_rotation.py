"""FiducialPolynomialWarp3D: applying a round angle (rotation_task), and
reading a task saved with the removed per-stack angle.

Synthetic bead stacks: the moving round is the fixed round turned by a known
angle about the frame centre, translated by a depth-dependent offset and
shifted one plane in z. The round angle is given by a stand-in for a
finalized FiducialRoundRotation task (the true angle), so these tests check
how a given angle is applied: planes turned back before measuring, the
table's rotation columns, the transforms, the aligned image, Decode's crop
and the rigid transforms. test_round_rotation covers measuring the angle.
The task's own measurement, fit, table and application code run end to end;
only the dataset and the round task are stand-ins.

Run with `python -m unittest tests/test_polywarp_rotation.py`.
"""
import contextlib
import io
import unittest

import cv2
import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis.warp import FiducialPolynomialWarp3D
from merlin.analysis.decode import compute_crop_bounds
from merlin.analysis.crossfov_composite_warp import CrossFOVCompositeWarp

SHAPE = (512, 512)
CENTRE = ((SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0)
NPLANES = 12
Z = np.arange(NPLANES, dtype=float)
THETA_DEG = 0.4


def beads(seed, n=900):
    rng = np.random.default_rng(seed)
    image = np.zeros(SHAPE, np.float32)
    xy = rng.integers(8, SHAPE[0] - 8, (n, 2))
    np.add.at(image, (xy[:, 1], xy[:, 0]), rng.uniform(800, 3000, n))
    return cv2.GaussianBlur(image, (0, 0), 1.2) + 100


def truth(zPos):
    """output->input map of the moving round at depth zPos: the content at
    reference p sits in the moving frame at c + R(p - c) + t(z)."""
    theta = np.radians(THETA_DEG)
    c, s = np.cos(theta), np.sin(theta)
    tx, ty = 15.3 + 0.2 * zPos, -12.7 + 0.1 * zPos
    cx, cy = CENTRE
    return np.array([[c, -s, cx - c * cx + s * cy + tx],
                     [s, c, cy - s * cx - c * cy + ty],
                     [0, 0, 1.0]])


def moved(image, matrix):
    # forward map: moving(matrix p) = image(p)
    return cv2.warpAffine(image, matrix[:2], SHAPE[::-1],
                          flags=cv2.INTER_LINEAR, borderValue=100)


class _Org:
    def get_data_channels(self):
        return np.array([0, 1])

    def get_fiducial3D_filename(self, dataChannel, fov):
        return str(dataChannel)

    def get_fiducial3D_stack_frame_indices(self, dataChannel):
        return np.arange(NPLANES)

    def get_fiducial3D_stack_frame_zPos(self, dataChannel):
        return Z

    def get_data_channel_color(self, dataChannel):
        return '650'


class _RoundTask:
    """Stands in for a finalized FiducialRoundRotation task: channel 1's
    round has the true angle, and this fov's own angle agrees with it."""

    def get_round_angles(self):
        return {0: 0.0, 1: float(np.radians(THETA_DEG))}

    def get_fov_verdicts(self, fov):
        return {1: (float(np.radians(THETA_DEG)), False, '')}


class _DataSet:
    def __init__(self):
        self.fixed = [beads(10 + i) for i in range(NPLANES)]
        # one-plane axial offset: moving plane k+1 holds fixed plane k
        self.moving = [beads(999)] + [moved(self.fixed[k], truth(Z[k]))
                                      for k in range(NPLANES - 1)]
        self.fixed2D = beads(5)
        self.moving2D = moved(self.fixed2D, truth(0.0))
        self.saved = {}
        self.tasks = {'RoundRot': _RoundTask()}

    def get_data_organization(self):
        return _Org()

    def get_fovs(self):
        return [0]

    def get_z_positions(self):
        return list(Z)

    def get_image_dimensions(self):
        return SHAPE

    def get_fiducial_image(self, dataChannel, fov):
        return self.fixed2D if dataChannel == 0 else self.moving2D

    def load_image(self, filename, frame):
        return (self.fixed if filename == '0' else self.moving)[int(frame)]

    def get_raw_image(self, dataChannel, fov, zPosition):
        stack = self.fixed if dataChannel == 0 else self.moving
        return stack[int(round(zPosition))].astype(np.uint16)

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

    def load_analysis_task(self, name):
        return self.tasks[name]


PARAMETERS = {
    'z_search_range': 2, 'z_sample_step': 1, 'xy_sample_step': 1,
    'z_polynomial_order': 0, 'xy_polynomial_order': 1,
    'minimum_overlap_pixels': 1000, 'edge_width_to_remove': 5,
    'rigid_search_bound': 50, 'z_min_good_samples': 3, 'z_min_score': 0.2,
    'write_fiducial_images': False, 'write_aligned_images': False,
    'boundary_smooth': False}


def make(dataSet, **extra):
    task = FiducialPolynomialWarp3D(dataSet, dict(PARAMETERS, **extra),
                                    'PolyWarp')
    dataSet.tasks['PolyWarp'] = task
    return task


def run(rotation, rigidOnly=()):
    dataSet = _DataSet()
    task = make(dataSet, rigid_only_channels=list(rigidOnly),
                rotation_task='RoundRot' if rotation else None)
    with contextlib.redirect_stdout(io.StringIO()):
        task._run_analysis(0)
    return task


class RotationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.on = run(True)
        cls.off = run(False)

    def test_off_writes_no_rotation(self):
        self.assertNotIn('rotation_correction', self.off.parameters)
        self.assertIsNone(self.off.parameters['rotation_task'])
        self.assertEqual(self.off.get_dependencies(), [])
        table = self.off.get_transformation_table(0)
        self.assertFalse(any('rotation' in c for c in table.columns))
        coeff = self.off.get_polynomial_coefficients(0)
        self.assertFalse(any('rotation' in c for c in coeff.columns))
        self.assertNotIn(('rotation_status', 0), self.off.dataSet.saved)
        for row, t in zip(table.itertuples(), self.off.get_transformation(0)):
            np.testing.assert_array_equal(
                t.params, transform.SimilarityTransform(
                    translation=[row.xshift, row.yshift]).params)

    def test_round_angle_in_the_table(self):
        self.assertEqual(self.on.get_dependencies(), ['RoundRot'])
        table = self.on.get_transformation_table(0)
        moving = table[table.dataChannel == 1]
        # the given angle (to the csv round trip of this stand-in dataset):
        # nothing is measured or refined here
        np.testing.assert_allclose(moving.rotation_radians,
                                   np.radians(THETA_DEG), rtol=0, atol=1e-15)
        np.testing.assert_allclose(moving.z_offset, 1.0, atol=0.05)
        reference = table[table.dataChannel == 0]
        np.testing.assert_array_equal(reference.rotation_radians, 0.0)
        np.testing.assert_allclose(moving.rotation_center_x, CENTRE[0])
        np.testing.assert_allclose(moving.rotation_center_y, CENTRE[1])
        coeff = self.on.get_polynomial_coefficients(0).set_index(
            'dataChannel')
        self.assertAlmostEqual(coeff.loc[1, 'rotation_radians'],
                               np.radians(THETA_DEG), places=15)

    def test_rotation_status(self):
        status = self.on.get_rotation_status(0).set_index('dataChannel')
        self.assertEqual(status.loc[0, 'source'], 'reference')
        self.assertEqual(status.loc[1, 'source'], 'round')
        self.assertTrue(bool(status.loc[1, 'trusted']))
        self.assertAlmostEqual(status.loc[1, 'measured_radians'],
                               np.radians(THETA_DEG), places=15)

    def test_transform_matches_truth_at_corners(self):
        corners = np.array([[40, 40], [470, 40], [40, 470], [470, 470]])
        for zIndex in (3, 6, 9):
            fitted = self.on.get_transformation(0, 1, zIndex)
            expected = truth(Z[zIndex])
            error = np.linalg.norm(
                transform.matrix_transform(corners, fitted.params)
                - transform.matrix_transform(corners, expected), axis=1)
            self.assertLess(error.max(), 0.3, (zIndex, error))
        # translation only leaves the corners off by about theta * r
        fitted = self.off.get_transformation(0, 1, 6)
        error = np.linalg.norm(
            transform.matrix_transform(corners, fitted.params)
            - transform.matrix_transform(corners, truth(Z[6])), axis=1)
        self.assertGreater(error.max(), 2.0)

    def test_aligned_image_reproduces_reference(self):
        zIndex = 5
        aligned = self.on.get_aligned_image(0, 1, zIndex).astype(float)
        reference = self.on.dataSet.fixed[zIndex].astype(np.uint16)
        inner = (slice(60, -60), slice(60, -60))
        a, b = aligned[inner], reference[inner].astype(float)
        a, b = a - a.mean(), b - b.mean()
        self.assertGreater(float(a.ravel() @ b.ravel()
                                 / np.linalg.norm(a) / np.linalg.norm(b)), 0.97)

    def test_crop_bounds_cover_rotated_corners(self):
        top, bottom, left, right = compute_crop_bounds(
            self.on.dataSet, 'PolyWarp', 0, 0, True)
        corners = np.array([[left, top], [right - 1, top],
                            [left, bottom - 1], [right - 1, bottom - 1]])
        for t in self.on.get_transformation(0):
            mapped = t(corners)
            self.assertGreaterEqual(mapped.min(), -1e-9)
            self.assertLessEqual(mapped.max(), SHAPE[0] - 1 + 1e-9)

    def test_crop_bounds_unchanged_without_rotation(self):
        bounds = compute_crop_bounds(self.off.dataSet, 'PolyWarp', 0, 0, True)
        table = self.off.get_transformation_table(0)
        tx, ty = table.xshift.to_numpy(), table.yshift.to_numpy()
        expected = (int(np.ceil(max(0.0, (-ty).max()))),
                    SHAPE[0] - int(np.ceil(max(0.0, ty.max()))),
                    int(np.ceil(max(0.0, (-tx).max()))),
                    SHAPE[1] - int(np.ceil(max(0.0, tx.max()))))
        self.assertEqual(bounds, expected)

    def test_rigid_only_channel_gets_the_round_angle(self):
        task = run(True, rigidOnly=[1])
        table = task.get_transformation_table(0)
        rows = table[table.dataChannel == 1]
        np.testing.assert_allclose(rows.rotation_radians,
                                   np.radians(THETA_DEG), rtol=0, atol=1e-15)
        self.assertEqual(rows.xshift.nunique(), 1)
        # measured on the turned-back 2D frame, so right at the corners of
        # the 2D frame's depth (the synthetic 2D frame is the depth-0 map)
        corners = np.array([[40, 40], [470, 40], [40, 470], [470, 470]])
        error = np.linalg.norm(
            transform.matrix_transform(
                corners, task.get_transformation(0, 1, 0).params)
            - transform.matrix_transform(corners, truth(0.0)), axis=1)
        self.assertLess(error.max(), 0.5, error)
        flags = task.dataSet.load_dataframe_from_csv(
            'fit_quality_flags', resultIndex=0)
        self.assertFalse(flags.reasons.str.contains('rotation').any())
        status = task.get_rotation_status(0).set_index('dataChannel')
        self.assertEqual(status.loc[1, 'source'], 'round')

    def test_rigid_transforms_carry_angle(self):
        # offsets_<fov>.npy (and the fiducial QC images, built from the same
        # list) turn by the table's angle about the centre with rotation_task,
        # and stay a plain translation without it
        table = self.on.get_transformation_table(0)
        coeff = self.on.get_polynomial_coefficients(0).set_index('dataChannel')
        offsets = self.on.dataSet.saved['offsets', 0]
        theta = float(table[table.dataChannel == 1].rotation_radians.iloc[0])
        self.assertAlmostEqual(np.arctan2(offsets[1][1, 0], offsets[1][0, 0]),
                               theta, places=12)
        centre = np.array([[CENTRE[0], CENTRE[1]]])
        np.testing.assert_allclose(
            transform.matrix_transform(centre, offsets[1]) - centre,
            [[-coeff.loc[1, 'rigid_x'], -coeff.loc[1, 'rigid_y']]],
            atol=1e-9)
        np.testing.assert_allclose(offsets[0], np.eye(3), atol=1e-9)
        offOffsets = self.off.dataSet.saved['offsets', 0]
        offCoeff = self.off.get_polynomial_coefficients(0)\
            .set_index('dataChannel')
        for dataChannel in (0, 1):
            np.testing.assert_array_equal(
                offOffsets[dataChannel], transform.SimilarityTransform(
                    translation=[-offCoeff.loc[dataChannel, 'rigid_x'],
                                 -offCoeff.loc[dataChannel, 'rigid_y']]
                ).params)

    def test_no_angle_is_measured(self):
        m = pd.read_csv(io.StringIO(self.on.dataSet.saved['measurements', 0]))
        self.assertEqual(sorted(m.stage.unique()), ['x', 'y', 'z'])


# The per-stack angle's options, at the defaults they had: every task saved
# while they existed carries them.
OLD_DEFAULTS = {
    'rotation_correction': False, 'rotation_tiles': 6,
    'rotation_tile_min_snr': 8.0, 'rotation_min_tiles': 8,
    'rotation_min_planes': 5, 'rotation_max_degrees': 1.0,
    'rotation_max_spread_degrees': 0.02,
    'rotation_neighbor_substitution': True}


class SavedParameterTests(unittest.TestCase):
    """How an analysis json or a saved task.json written before the per-stack
    angle was removed is read."""

    def test_removed_options_are_dropped(self):
        task = make(_DataSet(), **dict(OLD_DEFAULTS, rotation_tiles=4,
                                       rotation_neighbor_substitution=False))
        for name in OLD_DEFAULTS:
            self.assertNotIn(name, task.parameters)

    def test_saved_and_regenerated_parameters_agree(self):
        # dataset.save_analysis_task compares the task loaded from task.json
        # with the one built from the analysis json; both go through __init__
        saved = make(_DataSet(), **OLD_DEFAULTS).get_parameters()
        regenerated = make(_DataSet()).get_parameters()
        saved.pop('merlin_version')
        regenerated.pop('merlin_version')
        self.assertEqual(saved, regenerated)
        roundSaved = make(_DataSet(), **dict(
            OLD_DEFAULTS, rotation_correction=True,
            rotation_task='RoundRot')).get_parameters()
        roundNew = make(_DataSet(), rotation_task='RoundRot').get_parameters()
        roundSaved.pop('merlin_version')
        roundNew.pop('merlin_version')
        self.assertEqual(roundSaved, roundNew)

    def test_rotation_correction_with_rotation_task_is_redundant(self):
        task = make(_DataSet(), rotation_correction=True,
                    rotation_task='RoundRot')
        self.assertNotIn('rotation_correction', task.parameters)
        self.assertEqual(task.get_dependencies(), ['RoundRot'])

    def test_contradiction_and_type_are_refused(self):
        with self.assertRaises(ValueError):
            make(_DataSet(), rotation_correction=False,
                 rotation_task='RoundRot')
        with self.assertRaises(ValueError):
            make(_DataSet(), rotation_correction=1)


class StackAngleTaskTests(unittest.TestCase):
    """A task saved with the removed per-stack angle (rotation_correction
    true, no rotation_task) loads, its tables are read as written, and it
    does not run."""

    @classmethod
    def setUpClass(cls):
        # tables with an angle, as such a task wrote them
        cls.written = run(True)
        cls.legacy = FiducialPolynomialWarp3D(cls.written.dataSet, dict(
            PARAMETERS, **dict(OLD_DEFAULTS, rotation_correction=True)),
            'PolyWarp')

    def test_loads(self):
        self.assertTrue(self.legacy.parameters['rotation_correction'])
        self.assertIsNone(self.legacy.parameters['rotation_task'])
        for name in FiducialPolynomialWarp3D._REMOVED_PARAMETERS:
            self.assertNotIn(name, self.legacy.parameters)

    def test_refuses_to_run(self):
        with self.assertRaisesRegex(ValueError, 'per-stack angle'):
            self.legacy._run_analysis(0)
        with self.assertRaisesRegex(ValueError, 'per-stack angle'):
            self.legacy.finalize()

    def test_tables_are_applied_as_written(self):
        for a, b in zip(self.legacy.get_transformation(0),
                        self.written.get_transformation(0)):
            np.testing.assert_array_equal(a.params, b.params)
        self.assertNotEqual(
            float(self.legacy.get_transformation(0, 1, 5).params[1, 0]), 0.0)
        aligned = self.legacy.get_aligned_image(0, 1, 5)
        np.testing.assert_array_equal(
            aligned, self.written.get_aligned_image(0, 1, 5))


class CompositeWarpRefusalTests(unittest.TestCase):
    """CrossFOVCompositeWarp reads xshift/yshift as a pure translation, so it
    refuses a PolyWarp task whose tables carry an angle."""

    def composite(self, **polyParameters):
        dataSet = _DataSet()
        make(dataSet, **polyParameters)
        return CrossFOVCompositeWarp(dataSet, {'polywarp_task': 'PolyWarp'},
                                     'Composite')

    def test_refusals(self):
        self.composite()._polywarp()
        with self.assertRaises(ValueError):
            self.composite(rotation_task='RoundRot')._polywarp()
        with self.assertRaises(ValueError):
            self.composite(rotation_correction=True)._polywarp()


if __name__ == '__main__':
    unittest.main()
