"""FiducialPolynomialWarp3D: finalize() replaces untrusted angles with the
median angle of the same data channel over the neighbouring fovs.

A stack's angle is untrusted when it fell back to 0 (too few planes gave an
angle) or when its per-plane angles scatter by more than
rotation_max_spread_degrees. On 20260824 that is mostly edge fovs (fov0 round
17: 4 planes, angle 0, about 2.3 px off at the corners).

Nine fovs on a 3 x 3 grid at a 280 um pitch. The measurement methods are
stubbed with known angles and drifts; the stack grouping, angle gate, tables,
rotation_status and finalize() are the task's own code.

  stack r1 (channels 1, 2):  fov 4 gives no per-plane angle   -> angle 0
  stack r2 (channel 3):      fov 0's per-plane angles scatter  -> flagged
                             fov 8's z samples all fail the gate -> z unfitted
  stack nuc (channel 4):     rigid_only (angle 0, never substituted)

Run with `python -m unittest tests/test_polywarp_angle_fallback.py`.
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis.warp import FiducialPolynomialWarp3D

SIZE = 512
Z = np.arange(0.0, 30.0, 0.5)
FOVS = list(range(9))
PITCH = 280.0
CHANNELS = [0, 1, 2, 3, 4]
STACK = {0: 'r0', 1: 'r1', 2: 'r1', 3: 'r2', 4: 'nuc'}
RIGID = {0: (0.0, 0.0), 1: (25.0, -31.0), 2: (25.0, -31.0),
         3: (60.0, -44.0), 4: (80.0, -70.0)}
NO_ANGLE = (4, 'r1')
SCATTER = (0, 'r2')
Z_UNFITTED = (8, 'r2')


def true_angle(fov, stack):
    """radians; a small fov-to-fov variation so the median is checkable"""
    base = {'r1': 0.060, 'r2': -0.080}[stack]
    return np.radians(base + 0.002 * fov)


class _Org:
    def get_data_channels(self):
        return np.array(CHANNELS)

    def get_fiducial3D_filename(self, dataChannel, fov):
        return STACK[dataChannel]

    def get_data_channel_index(self, name):
        return int(name)

    def get_data_channel_color(self, dataChannel):
        return '650'


class _DataSet:
    def __init__(self):
        self.saved = {}

    def get_data_organization(self):
        return _Org()

    def get_fovs(self):
        return FOVS

    def get_fov_offset(self, fov):
        return (PITCH * (fov % 3), PITCH * (fov // 3))

    def get_z_positions(self):
        return list(Z)

    def get_image_dimensions(self):
        return (SIZE, SIZE)

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


def stub(task):
    def rigid(fov):
        return [RIGID[c] for c in CHANNELS]

    def measure_z(fov, dataChannel, rigidY, rigidX):
        z = Z[7:-7:4]
        score = 0.1 if (fov, STACK[dataChannel]) == Z_UNFITTED else 0.9
        return z, 0.2 + 0.01 * z, np.full(len(z), score), 3.0

    def measure_xy(fov, dataChannel, rigidY, rigidX, zCoefficients):
        z = Z[1:-1:3]
        return (z, rigidY + 0.02 * z, rigidX - 0.01 * z,
                np.full(len(z), 50.0), np.zeros(len(z), bool))

    def measure_rotation(fov, dataChannel, rigidY, rigidX, sampleZ,
                         sampleOffset):
        stack = STACK[dataChannel]
        n = len(sampleZ)
        applied = task._measurementRotation.get((fov, dataChannel), 0.0)
        theta = np.full(n, true_angle(fov, stack) - applied)
        if (fov, stack) == NO_ANGLE:
            theta[:] = np.nan
        elif (fov, stack) == SCATTER:
            theta = theta + np.radians(0.05) * (-1.0) ** np.arange(n)
        return (np.asarray(sampleZ, float), theta, np.full(n, 30.0),
                np.full(n, 0.1))

    task._measure_rigid_offsets = rigid
    task._measure_z_offsets = measure_z
    task._measure_xy_offsets = measure_xy
    task._measure_rotation = measure_rotation


def run(rotation=True, substitute=None):
    dataSet = _DataSet()
    parameters = {'rotation_correction': rotation, 'rigid_only_channels': [4],
                  'write_fiducial_images': False,
                  'write_aligned_images': False}
    if substitute is not None:
        parameters['rotation_neighbor_substitution'] = substitute
    task = FiducialPolynomialWarp3D(dataSet, parameters, 'PolyWarp')
    stub(task)
    with contextlib.redirect_stdout(io.StringIO()):
        for fov in FOVS:
            task._run_analysis(fov)
    return task


def snapshot(dataSet):
    return {k: (v.tobytes() if isinstance(v, np.ndarray) else v)
            for k, v in dataSet.saved.items()}


def neighbours(fov, ring):
    x, y = _DataSet().get_fov_offset(fov)
    return [f for f in FOVS if f != fov and np.hypot(
        *np.subtract(_DataSet().get_fov_offset(f), (x, y))) <= ring]


class AngleFallbackTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.task = run()
        cls.before = snapshot(cls.task.dataSet)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cls.task.finalize()
        cls.log = out.getvalue()
        cls.after = snapshot(cls.task.dataSet)

    def angle_of(self, fov, dataChannel):
        table = self.task.get_transformation_table(fov)
        values = table[table.dataChannel == dataChannel].rotation_radians
        self.assertEqual(values.nunique(), 1)
        return float(values.iloc[0])

    def test_fragment_verdicts(self):
        status = self.task.get_rotation_status(4).set_index('dataChannel')
        self.assertEqual(list(status.loc[[1, 2], 'trusted']), [False, False])
        self.assertTrue(status.loc[1, 'reason'].startswith('only 0 planes'))
        self.assertEqual(status.loc[2, 'owner_channel'], 1)
        self.assertEqual(status.loc[0, 'source'], 'reference')
        self.assertEqual(status.loc[3, 'source'], 'fit')
        self.assertEqual(status.loc[4, 'source'], 'rigid_only')
        self.assertEqual(status.loc[1, 'measured_radians'], 0.0)
        scatter = self.task.get_rotation_status(0).set_index('dataChannel')
        self.assertIn('scatter', scatter.loc[3, 'reason'])
        self.assertAlmostEqual(scatter.loc[3, 'measured_radians'],
                               true_angle(0, 'r2'), places=12)

    def test_failed_angle_gets_neighbour_median(self):
        donors = neighbours(4, 400.0)
        self.assertEqual(len(donors), 8)
        expected = np.median([true_angle(f, 'r1') for f in donors])
        for dataChannel in (1, 2):
            self.assertAlmostEqual(self.angle_of(4, dataChannel), expected,
                                   places=12)
        coeff = self.task.get_polynomial_coefficients(4)\
            .set_index('dataChannel')
        self.assertAlmostEqual(coeff.loc[1, 'rotation_radians'], expected,
                               places=12)
        status = self.task.get_rotation_status(4).set_index('dataChannel')
        self.assertEqual(list(status.loc[[1, 2], 'source']),
                         ['neighbor', 'neighbor'])
        self.assertEqual(list(status.loc[[1, 2], 'trusted']), [False, False])
        self.assertEqual(status.loc[1, 'measured_radians'], 0.0)

    def test_scattered_angle_gets_neighbour_median(self):
        donors = neighbours(0, 400.0)
        self.assertEqual(sorted(donors), [1, 3, 4])
        expected = np.median([true_angle(f, 'r2') for f in donors])
        self.assertAlmostEqual(self.angle_of(0, 3), expected, places=12)

    def test_offsets_rebuilt_for_touched_channels_only(self):
        offsets = self.task.dataSet.saved['offsets', 4]
        coeff = self.task.get_polynomial_coefficients(4)\
            .set_index('dataChannel')
        centre = np.array([[(SIZE - 1) / 2.0, (SIZE - 1) / 2.0]])
        for dataChannel in (1, 2):
            i = CHANNELS.index(dataChannel)
            self.assertAlmostEqual(
                np.arctan2(offsets[i][1, 0], offsets[i][0, 0]),
                self.angle_of(4, dataChannel), places=12)
            np.testing.assert_allclose(
                transform.matrix_transform(centre, offsets[i]) - centre,
                [[-coeff.loc[dataChannel, 'rigid_x'],
                  -coeff.loc[dataChannel, 'rigid_y']]], atol=1e-9)
        old = np.frombuffer(self.before['offsets', 4]).reshape(offsets.shape)
        for dataChannel in (0, 3, 4):
            i = CHANNELS.index(dataChannel)
            np.testing.assert_array_equal(offsets[i], old[i])

    def test_rewritten_tables_keep_other_values_byte_identical(self):
        # only the rotation_radians of the touched channels may change: the
        # tables are read back with round-trip parsing before rewriting
        for name in ('transformation_table', 'polynomial_coefficients'):
            before = pd.read_csv(io.StringIO(self.before[name, 4]),
                                 dtype=str)
            after = pd.read_csv(io.StringIO(self.after[name, 4]), dtype=str)
            touched = after.dataChannel.isin(['1', '2'])
            pd.testing.assert_frame_equal(before[~touched], after[~touched])
            pd.testing.assert_frame_equal(
                before[touched].drop(columns='rotation_radians'),
                after[touched].drop(columns='rotation_radians'))

    def test_transform_turns_about_the_centre(self):
        # the substituted angle turns the frame about its centre and keeps
        # the fitted centre displacement
        table = self.task.get_transformation_table(4)
        row = table[(table.dataChannel == 1) & (table.zIndex == 20)].iloc[0]
        t = self.task.get_transformation(4, 1, 20)
        c = (SIZE - 1) / 2.0
        np.testing.assert_allclose(t([[c, c]]) - c,
                                   [[row.xshift, row.yshift]], atol=1e-9)
        theta = self.angle_of(4, 1)
        corner = np.array([[0.0, 0.0]])
        expected = (c + np.array([[np.cos(theta) * -c + np.sin(theta) * c,
                                   np.sin(theta) * -c - np.cos(theta) * c]])
                    + [[row.xshift, row.yshift]])
        np.testing.assert_allclose(t(corner), expected, atol=1e-9)

    def test_other_stacks_untouched(self):
        changed = {k for k in self.after if self.before.get(k) != self.after[k]}
        self.assertTrue(changed <= {
            ('polynomial_coefficients', 0), ('transformation_table', 0),
            ('rotation_status', 0), ('offsets', 0),
            ('polynomial_coefficients', 4), ('transformation_table', 4),
            ('rotation_status', 4), ('offsets', 4),
            ('polynomial_coefficients', 8), ('transformation_table', 8),
            ('z_fit_status', 8), ('z_neighbor_substitutions', None),
            ('rotation_neighbor_substitutions', None)}, changed)
        for fov in (1, 2, 3, 5, 6, 7):
            for name in ('transformation_table', 'polynomial_coefficients',
                         'rotation_status', 'offsets'):
                self.assertEqual(self.before[name, fov],
                                 self.after[name, fov])
        # the rigid-only and reference channels keep angle 0
        for fov in FOVS:
            self.assertEqual(self.angle_of(fov, 4), 0.0)
            self.assertEqual(self.angle_of(fov, 0), 0.0)

    def test_z_and_angle_substitutions_compose(self):
        # fov 8's r2 stack: z substituted, its trusted angle kept
        status = self.task.get_z_fit_status(8).set_index('dataChannel')
        self.assertEqual(status.loc[3, 'source'], 'neighbor')
        self.assertAlmostEqual(self.angle_of(8, 3), true_angle(8, 'r2'),
                               places=12)
        table = self.task.get_transformation_table(8)
        self.assertTrue((table[table.dataChannel == 3].z_offset != 0).all())
        # and fov 4's angle substitution kept its fitted z curve
        table = self.task.get_transformation_table(4)
        np.testing.assert_allclose(
            table[table.dataChannel == 1].z_offset,
            0.2 + 0.01 * table[table.dataChannel == 1].zPos, atol=1e-9)

    def test_report(self):
        report = self.task.dataSet.load_dataframe_from_csv(
            'rotation_neighbor_substitutions')
        self.assertEqual(sorted(zip(report.fov, report.dataChannel)),
                         [(0, 3), (4, 1), (4, 2)])
        self.assertEqual(set(report.n_donors), {3, 8})
        self.assertIn('substituted 3 angles in 2 fovs', self.log)

    def test_idempotent(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.task.finalize()
        self.assertEqual(snapshot(self.task.dataSet), self.after)


class AngleFallbackOffTests(unittest.TestCase):

    def test_rotation_off_writes_nothing_new(self):
        task = run(rotation=False)
        self.assertFalse(any(name == 'rotation_status'
                             for name, _ in task.dataSet.saved))
        before = snapshot(task.dataSet)
        with contextlib.redirect_stdout(io.StringIO()):
            task.finalize()
        after = snapshot(task.dataSet)
        self.assertNotIn(('rotation_neighbor_substitutions', None), after)
        # only the z substitution touched anything
        changed = {k for k in after if before.get(k) != after[k]}
        self.assertEqual(changed, {
            ('polynomial_coefficients', 8), ('transformation_table', 8),
            ('z_fit_status', 8), ('z_neighbor_substitutions', None)})

    def test_option_off_keeps_fragment_angles(self):
        task = run(substitute=False)
        before = snapshot(task.dataSet)
        with contextlib.redirect_stdout(io.StringIO()):
            task.finalize()
        after = snapshot(task.dataSet)
        self.assertNotIn(('rotation_neighbor_substitutions', None), after)
        for name in ('transformation_table', 'rotation_status', 'offsets'):
            self.assertEqual(before[name, 4], after[name, 4])
            self.assertEqual(before[name, 0], after[name, 0])

    def test_default_on(self):
        task = FiducialPolynomialWarp3D(_DataSet(), {}, 'PolyWarp')
        self.assertTrue(task.parameters['rotation_neighbor_substitution'])
        self.assertFalse(task.parameters['rotation_correction'])


if __name__ == '__main__':
    unittest.main()
