"""FiducialRoundRotation, and FiducialPolynomialWarp3D with rotation_task.

Synthetic 3 x 3 fov layout. Each moving stack is the reference turned by its
round's angle about the frame centre (the same angle in every fov), plus a
translation that differs from fov to fov and drifts with depth, and one plane
deeper. One fov's stack of one round is turned by a different angle (it must
be flagged, and must not move the round's median); one fov's stack of another
round is beadless (flagged as untrusted; in the registration its z is unfitted
and finalize() substitutes it). Both tasks' own measurement, aggregation, table
and application code run end to end; only the dataset is a stand-in.

Run with `python -m unittest tests/test_round_rotation.py`.
"""
import contextlib
import io
import unittest

import cv2
import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis.warp import FiducialPolynomialWarp3D
from merlin.analysis.warp import FiducialRoundRotation

SHAPE = (384, 384)
CENTRE = ((SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0)
NPLANES = 12
Z = np.arange(NPLANES, dtype=float)
FOVS = list(range(9))
# data channel -> fiducial stack; 1 and 2 share one, 4 is a separately
# acquired stack (its own "round")
STACK = {0: 'r0', 1: 'r1', 2: 'r1', 3: 'r2', 4: 'nuc'}
ROUND_DEG = {'r1': 0.3, 'r2': -0.2, 'nuc': 0.12}
DEVIANT = (5, 'r1', 0.15)       # fov, stack, extra degrees
BEADLESS = (4, 'r2')


def beads(seed, n=500):
    rng = np.random.default_rng(seed)
    image = np.zeros(SHAPE, np.float32)
    xy = rng.integers(8, SHAPE[0] - 8, (n, 2))
    np.add.at(image, (xy[:, 1], xy[:, 0]), rng.uniform(800, 3000, n))
    return cv2.GaussianBlur(image, (0, 0), 1.2)


def true_angle_deg(stack, fov):
    extra = DEVIANT[2] if (fov, stack) == DEVIANT[:2] else 0.0
    return ROUND_DEG[stack] + extra


def true_shift(stack, fov, z):
    """(tx, ty), the displacement of the frame centre at depth z; it
    differs per fov (stage scatter) and drifts with depth."""
    if stack == 'r1':
        return 15.3 + 0.2 * z + 0.7 * fov, -12.7 + 0.1 * z - 0.4 * fov
    if stack == 'r2':
        return -20.0 + 0.1 * z + 0.3 * fov, 30.0 - 0.6 * fov - 0.05 * z
    return 33.0 - 0.5 * fov, -21.0 + 0.2 * fov


def truth(stack, fov, z):
    """output->input map: the content at reference p sits in the moving
    frame at c + R(p - c) + t(z)."""
    theta = np.radians(true_angle_deg(stack, fov))
    t = true_shift(stack, fov, z)
    c, s = np.cos(theta), np.sin(theta)
    cx, cy = CENTRE
    return np.array([[c, -s, cx - c * cx + s * cy + t[0]],
                     [s, c, cy - s * cx - c * cy + t[1]], [0, 0, 1.0]])


def moved(image, matrix):
    # forward map: moving(matrix p) = image(p)
    return cv2.warpAffine(image, matrix[:2], SHAPE[::-1],
                          flags=cv2.INTER_LINEAR, borderValue=0)


class _Org:
    def get_data_channels(self):
        return np.array(sorted(STACK))

    def get_fiducial3D_filename(self, dataChannel, fov):
        return '%s_%d' % (STACK[int(dataChannel)], fov)

    def get_fiducial3D_stack_frame_indices(self, dataChannel):
        return np.arange(NPLANES)

    def get_fiducial3D_stack_frame_zPos(self, dataChannel):
        return Z

    def get_data_channel_color(self, dataChannel):
        return '650'

    def get_data_channel_index(self, name):
        return int(name)


class _DataSet:
    def __init__(self):
        self.saved = {}
        self.tasks = {}
        self.stacks = {}
        self.twoD = {}
        for fov in FOVS:
            fixed = [beads(1000 * fov + k) for k in range(NPLANES)]
            self.stacks['r0_%d' % fov] = [f + 100 for f in fixed]
            rng = np.random.default_rng(fov)
            fixed2D = beads(9000 + fov, n=250)
            self.twoD['r0', fov] = (fixed2D + 100 + rng.normal(
                0, 80, SHAPE).astype(np.float32))
            for stack in ('r1', 'r2', 'nuc'):
                # one plane deeper: moving plane k + 1 holds reference plane k
                planes = [beads(77 + fov) + 100] + [
                    moved(fixed[k], truth(stack, fov, Z[k])) + 100
                    for k in range(NPLANES - 1)]
                if (fov, stack) == BEADLESS:
                    noise = np.random.default_rng(5)
                    planes = [noise.normal(100, 20, SHAPE).astype(np.float32)
                              for _ in range(NPLANES)]
                self.stacks['%s_%d' % (stack, fov)] = planes
                self.twoD[stack, fov] = (
                    moved(fixed2D, truth(stack, fov, 0.0)) + 100
                    + rng.normal(0, 80, SHAPE).astype(np.float32))

    @staticmethod
    def _name(task):
        return task if isinstance(task, str) else task.get_analysis_name()

    def get_data_organization(self):
        return _Org()

    def get_fovs(self):
        return FOVS

    def get_fov_offset(self, fov):
        return (280.0 * (fov % 3), 280.0 * (fov // 3))

    def get_z_positions(self):
        return list(Z)

    def get_image_dimensions(self):
        return SHAPE

    def get_fiducial_image(self, dataChannel, fov):
        return self.twoD[STACK[int(dataChannel)], fov]

    def load_image(self, filename, frame):
        return self.stacks[filename][int(frame)]

    def get_raw_image(self, dataChannel, fov, zPosition):
        stack = self.stacks['%s_%d' % (STACK[int(dataChannel)], fov)]
        return stack[int(round(zPosition))].astype(np.uint16)

    def save_dataframe_to_csv(self, dataframe, resultName, analysisTask=None,
                              resultIndex=None, subdirectory=None, **kwargs):
        self.saved[self._name(analysisTask), resultName, resultIndex] = \
            dataframe.to_csv(index=False)

    def load_dataframe_from_csv(self, resultName, analysisTask=None,
                                resultIndex=None, subdirectory=None,
                                **kwargs):
        key = (self._name(analysisTask), resultName, resultIndex)
        if key not in self.saved:
            raise FileNotFoundError(resultName)
        return pd.read_csv(io.StringIO(self.saved[key]), **kwargs)

    def save_numpy_analysis_result(self, analysisResult, resultName,
                                   analysisName, resultIndex=None,
                                   subdirectory=None):
        self.saved[self._name(analysisName), resultName, resultIndex] = \
            np.array(analysisResult)

    def load_numpy_analysis_result(self, resultName, analysisName,
                                   resultIndex=None, subdirectory=None):
        key = (self._name(analysisName), resultName, resultIndex)
        if key not in self.saved:
            raise FileNotFoundError(resultName)
        return self.saved[key].copy()

    def load_analysis_task(self, name):
        return self.tasks[name]


SHARED = {'highpass_sigma': 3, 'median_filter': False,
          'edge_width_to_remove': 5, 'rigid_plane_search': 8,
          'rigid_plane_step': 4, 'rigid_search_bound': 80,
          'rigid_walk_bound': 15, 'rigid_walk_confirm': True,
          'z_search_range': 2, 'z_min_score': 0.3,
          'minimum_overlap_pixels': 1000, 'rotation_tiles': 4,
          'rotation_min_tiles': 8}


def round_task(dataSet, **extra):
    task = FiducialRoundRotation(dataSet, dict(SHARED, **extra), 'RoundRot')
    dataSet.tasks['RoundRot'] = task
    return task


def poly_task(dataSet, **extra):
    parameters = dict(SHARED, rotation_correction=True,
                      rotation_task='RoundRot', z_sample_step=1,
                      xy_sample_step=1, z_polynomial_order=1,
                      xy_polynomial_order=1, z_min_good_samples=3,
                      z_neighbor_min_samples=3, write_fiducial_images=False,
                      write_aligned_images=False, boundary_smooth=False)
    parameters.update(extra)
    task = FiducialPolynomialWarp3D(dataSet, parameters, 'PolyWarp')
    dataSet.tasks['PolyWarp'] = task
    return task


def run_all(task):
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for fov in FOVS:
            task._run_analysis(fov)
        task.finalize()
    return log.getvalue()


class RoundRotationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataSet = _DataSet()
        cls.rounds = round_task(cls.dataSet)
        cls.roundLog = run_all(cls.rounds)
        cls.poly = poly_task(cls.dataSet)
        cls.polyLog = run_all(cls.poly)

    # ---- the round task ----

    def test_fov_angles_recovered(self):
        for fov in FOVS:
            table = self.rounds.get_first_plane_rotation(fov)
            self.assertEqual(sorted(table.owner_channel), [1, 3, 4])
            for row in table.itertuples():
                stack = STACK[row.owner_channel]
                self.assertEqual(row.plane_index, 3)   # z_search_range + 1
                if (fov, stack) == BEADLESS:
                    self.assertFalse(row.trusted)
                    continue
                self.assertTrue(row.trusted, (fov, stack, row.reason))
                self.assertAlmostEqual(row.angle_deg,
                                       true_angle_deg(stack, fov),
                                       delta=0.01, msg=(fov, stack))
                self.assertAlmostEqual(row.depth_offset_um, 1.0, delta=0.1)
                # centre displacement at the measured (reference) depth
                tx, ty = true_shift(stack, fov, Z[3])
                self.assertAlmostEqual(row.xshift, tx, delta=0.2)
                self.assertAlmostEqual(row.yshift, ty, delta=0.2)

    def test_round_angle_is_the_median_and_flags(self):
        table = self.rounds.get_round_rotation().set_index('owner_channel')
        self.assertEqual(table.loc[0, 'source'], 'reference')
        self.assertEqual(table.loc[0, 'angle_radians'], 0.0)
        self.assertEqual(str(table.loc[1, 'members']), '1;2')
        for owner, stack in ((1, 'r1'), (3, 'r2'), (4, 'nuc')):
            self.assertAlmostEqual(table.loc[owner, 'angle_deg'],
                                   ROUND_DEG[stack], delta=0.005)
        # the deviant fov does not count against r1, the beadless one is out
        self.assertEqual(table.loc[1, 'n_fovs'], 9)
        self.assertEqual(table.loc[3, 'n_fovs'], 8)
        self.assertEqual(table.loc[3, 'n_untrusted'], 1)
        fovTable = self.rounds.get_fov_rotation()
        flagged = {(int(r.fov), int(r.owner_channel))
                   for r in fovTable.itertuples() if r.flagged}
        self.assertEqual(flagged, {(5, 1), (4, 3)})
        deviant = fovTable[(fovTable.fov == 5) & (fovTable.owner_channel == 1)]
        self.assertTrue(bool(deviant.deviating.iloc[0]))
        self.assertAlmostEqual(deviant.deviation_deg.iloc[0], 0.15,
                               delta=0.01)
        median = np.median(fovTable[fovTable.owner_channel == 1]
                           .angle_radians)
        self.assertAlmostEqual(table.loc[1, 'angle_radians'], median,
                               places=15)

    def test_getters(self):
        angles = self.rounds.get_round_angles()
        self.assertEqual(sorted(angles), [0, 1, 2, 3, 4])
        self.assertEqual(angles[0], 0.0)
        self.assertEqual(angles[1], angles[2])
        self.assertEqual(self.rounds.get_round_angle(2), angles[1])
        verdicts = self.rounds.get_fov_verdicts(5)
        self.assertTrue(verdicts[1][1] and verdicts[2][1])
        self.assertFalse(verdicts[3][1])
        self.assertTrue(np.isnan(self.rounds.get_fov_verdicts(4)[3][0]))

    def test_finalize_idempotent(self):
        before = {k: v for k, v in self.dataSet.saved.items()
                  if k[0] == 'RoundRot'}
        with contextlib.redirect_stdout(io.StringIO()):
            self.rounds.finalize()
        after = {k: v for k, v in self.dataSet.saved.items()
                 if k[0] == 'RoundRot'}
        self.assertEqual(before.keys(), after.keys())
        for key in before:
            self.assertEqual(before[key], after[key], key)

    def test_round_without_trusted_fov_has_no_angle(self):
        dataSet = _DataSet()
        task = round_task(dataSet)
        with contextlib.redirect_stdout(io.StringIO()):
            for fov in FOVS:
                task._run_analysis(fov)
        for fov in FOVS:
            table = task.get_first_plane_rotation(fov)
            table.loc[table.owner_channel == 4, 'trusted'] = False
            dataSet.save_dataframe_to_csv(table, 'first_plane_rotation',
                                          'RoundRot', resultIndex=fov)
        with contextlib.redirect_stdout(io.StringIO()):
            task.finalize()
        self.assertTrue(np.isnan(task.get_round_angles()[4]))
        with self.assertRaises(ValueError):
            task.get_round_angle(4)
        poly = poly_task(dataSet)
        with self.assertRaises(ValueError):
            poly._run_analysis(0)

    def test_measure_plane_and_single_pass(self):
        dataSet = _DataSet()
        task = round_task(dataSet, measure_plane=6, refine_passes=0)
        with contextlib.redirect_stdout(io.StringIO()):
            task._run_analysis(0)
        table = task.get_first_plane_rotation(0).set_index('owner_channel')
        self.assertTrue((table.plane_index == 6).all())
        self.assertTrue((table.plane_z_um == Z[6]).all())
        # no residual pass: the final angle is the first fit's
        np.testing.assert_array_equal(table.angle_radians,
                                      table.first_pass_radians)
        for owner, stack in ((1, 'r1'), (3, 'r2'), (4, 'nuc')):
            self.assertAlmostEqual(table.loc[owner, 'angle_deg'],
                                   ROUND_DEG[stack], delta=0.03)
        with self.assertRaises(ValueError):
            task = round_task(_DataSet(), measure_plane=10)
            with contextlib.redirect_stdout(io.StringIO()):
                task._run_analysis(0)

    def test_parameter_checks(self):
        with self.assertRaises(ValueError):
            FiducialRoundRotation(_DataSet(), {'refine_passes': -1})
        with self.assertRaises(ValueError):
            FiducialRoundRotation(_DataSet(), {'measure_plane': 2.5})
        defaults = FiducialPolynomialWarp3D(_DataSet()).parameters
        task = FiducialRoundRotation(_DataSet())
        for name in FiducialRoundRotation._MEASUREMENT_PARAMETERS:
            self.assertEqual(task.parameters[name], defaults[name], name)
        self.assertIsNone(task.parameters['measure_plane'])

    # ---- the registration with rotation_task ----

    def test_poly_parameters(self):
        self.assertEqual(self.poly.get_dependencies(), ['RoundRot'])
        self.assertEqual(FiducialPolynomialWarp3D(
            _DataSet()).get_dependencies(), [])
        with self.assertRaises(ValueError):
            FiducialPolynomialWarp3D(_DataSet(), {'rotation_task': 'X'})

    def test_poly_needs_finalized_round_task(self):
        dataSet = _DataSet()
        round_task(dataSet)
        poly = poly_task(dataSet)
        with self.assertRaises(FileNotFoundError):
            poly._run_analysis(0)

    def test_round_angle_applied_everywhere(self):
        angles = self.rounds.get_round_angles()
        for fov in FOVS:
            table = self.poly.get_transformation_table(fov)
            status = self.poly.get_rotation_status(fov).set_index(
                'dataChannel')
            for dataChannel in STACK:
                applied = table[table.dataChannel == dataChannel]\
                    .rotation_radians.to_numpy(float)
                np.testing.assert_allclose(applied, angles[dataChannel],
                                           rtol=0, atol=1e-15)
                self.assertEqual(status.loc[dataChannel, 'source'],
                                 'reference' if dataChannel == 0
                                 else 'round')
            # the rotation task's verdict is echoed, nothing more
            self.assertEqual(bool(status.loc[1, 'trusted']), fov != 5)
            self.assertEqual(bool(status.loc[3, 'trusted']), fov != 4)

    def test_z_and_corners_recovered(self):
        corners = np.array([[30, 30], [353, 30], [30, 353], [353, 353],
                            CENTRE])
        worst = 0.0
        for fov in FOVS:
            table = self.poly.get_transformation_table(fov)
            for dataChannel in (1, 2, 3, 4):
                stack = STACK[dataChannel]
                if (fov, stack) in (DEVIANT[:2], BEADLESS):
                    continue
                rows = table[table.dataChannel == dataChannel]
                np.testing.assert_allclose(rows.z_offset, 1.0, atol=0.1)
                for zIndex in (2, 5, 8):
                    fitted = self.poly.get_transformation(fov, dataChannel,
                                                          zIndex)
                    error = np.linalg.norm(
                        transform.matrix_transform(corners, fitted.params)
                        - transform.matrix_transform(
                            corners, truth(stack, fov, Z[zIndex])), axis=1)
                    worst = max(worst, error.max())
        self.assertLess(worst, 0.3)

    def test_rigid_offsets_turned_into_the_moving_frame(self):
        # the walk measures on turned-back frames (rigid_walk table); the
        # stored rigid offset is that shift turned by the round angle, and
        # the stored rigid transform is right at the frame centre
        angles = self.rounds.get_round_angles()
        for fov in (0, 8):
            walk = self.dataSet.load_dataframe_from_csv(
                'rigid_walk', 'PolyWarp', fov).set_index('dataChannel')
            coeff = self.poly.get_polynomial_coefficients(fov).set_index(
                'dataChannel')
            for dataChannel in (1, 3, 4):
                theta = angles[dataChannel]
                c, s = np.cos(theta), np.sin(theta)
                wx, wy = walk.loc[dataChannel, ['rigid_x', 'rigid_y']]
                self.assertAlmostEqual(coeff.loc[dataChannel, 'rigid_x'],
                                       c * wx - s * wy, places=9)
                self.assertAlmostEqual(coeff.loc[dataChannel, 'rigid_y'],
                                       s * wx + c * wy, places=9)
                # a gross check only: the rigid offset here comes from the
                # noisy synthetic 2D frame (the walk planes are one plane
                # apart in z and do not correlate), good to a few tenths of
                # a px; the frame conversion is checked exactly above
                rigid = self.dataSet.load_numpy_analysis_result(
                    'offsets', 'PolyWarp', fov)[dataChannel]
                expected = truth(STACK[dataChannel], fov, 0.0)
                error = np.linalg.norm(
                    transform.matrix_transform([CENTRE], rigid)
                    - transform.matrix_transform([CENTRE], expected))
                self.assertLess(error, 0.6, (fov, dataChannel))

    def test_finalize_substitutes_z_not_angles(self):
        names = {k[1] for k in self.dataSet.saved if k[0] == 'PolyWarp'}
        self.assertNotIn('rotation_neighbor_substitutions', names)
        self.assertIn('z_neighbor_substitutions', names)
        status = self.poly.get_z_fit_status(BEADLESS[0]).set_index(
            'dataChannel')
        self.assertEqual(status.loc[3, 'source'], 'neighbor')
        table = self.poly.get_transformation_table(BEADLESS[0])
        np.testing.assert_allclose(
            table[table.dataChannel == 3].z_offset, 1.0, atol=0.1)


if __name__ == '__main__':
    unittest.main()
