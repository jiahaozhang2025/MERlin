"""FiducialPolynomialWarp3D: finalize() replaces untrusted angles with the
median angle of the same data channel over the neighbouring fovs.

A stack's angle is untrusted when it fell back to 0 (too few planes gave an
angle) or when its per-plane angles scatter by more than
rotation_max_spread_degrees, typically on an edge fov. Donors must have a
trusted angle AND a z fitted by their own fragment. The substituted stack's xy
is measured again on planes turned back by the new angle (its xy was measured
on planes turned back by its own first-pass angle, so it is the displacement
at the bead-weighted centroid of the overlap), with the fragment's z curve and
rigid offset; only when that returns too few samples is the overlap-centroid
formula applied instead, flagged.

Nine fovs on a 3 x 3 grid at a 280 um pitch. The measurement methods are
stubbed with known angles and drifts (the xy stub returns the rigid offset it
is given plus a drift, in whatever frame it is asked about, and records the
call); the stack grouping, angle gate, tables, rotation_status and finalize()
are the task's own code.

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
CENTRE = np.array([(SIZE - 1) / 2.0, (SIZE - 1) / 2.0])
DRIFT = (-0.01, 0.02)           # (x, y) px per um, in the measurement frame
XY_SAMPLES = len(Z[1:-1:3])


def rotate(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return c * x - s * y, s * x + c * y


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


def stub(task, zUnfitted=(Z_UNFITTED,)):
    def rigid(fov):
        return [RIGID[c] for c in CHANNELS]

    def measure_z(fov, dataChannel, rigidY, rigidX):
        z = Z[7:-7:4]
        score = 0.1 if (fov, STACK[dataChannel]) in zUnfitted else 0.9
        return z, 0.2 + 0.01 * z, np.full(len(z), score), 3.0

    def measure_xy(fov, dataChannel, rigidY, rigidX, zCoefficients):
        task.xyCalls.append(dict(
            fov=fov, dataChannel=dataChannel, rigidY=rigidY, rigidX=rigidX,
            zCoefficients=np.array(zCoefficients, float),
            frame=task._measurementRotation.get((fov, dataChannel), 0.0)))
        z = Z[1:-1:3]
        return (z, rigidY + DRIFT[1] * z, rigidX + DRIFT[0] * z,
                np.full(len(z), 50.0), np.zeros(len(z), bool))

    task.xyCalls = []

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


def run(rotation=True, substitute=None, zUnfitted=(Z_UNFITTED,)):
    dataSet = _DataSet()
    parameters = {'rotation_correction': rotation, 'rigid_only_channels': [4],
                  'write_fiducial_images': False,
                  'write_aligned_images': False}
    if substitute is not None:
        parameters['rotation_neighbor_substitution'] = substitute
    task = FiducialPolynomialWarp3D(dataSet, parameters, 'PolyWarp')
    stub(task, zUnfitted)
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
        fragmentCalls = len(cls.task.xyCalls)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cls.task.finalize()
        cls.log = out.getvalue()
        cls.after = snapshot(cls.task.dataSet)
        cls.finalizeCalls = cls.task.xyCalls[fragmentCalls:]

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
        # only the angle and the xy of the touched channels may change: the
        # tables are read back with round-trip parsing before rewriting
        xy = ['%s_c%d' % (n, k) for n in 'yx' for k in range(4)]
        for name, changed in (
                ('transformation_table',
                 ['rotation_radians', 'xshift', 'yshift']),
                ('polynomial_coefficients', ['rotation_radians'] + xy)):
            before = pd.read_csv(io.StringIO(self.before[name, 4]),
                                 dtype=str)
            after = pd.read_csv(io.StringIO(self.after[name, 4]), dtype=str)
            touched = after.dataChannel.isin(['1', '2'])
            pd.testing.assert_frame_equal(before[~touched], after[~touched])
            pd.testing.assert_frame_equal(
                before[touched].drop(columns=changed),
                after[touched].drop(columns=changed))

    def fragment_table(self, fov):
        return pd.read_csv(io.StringIO(self.before['transformation_table',
                                                   fov]))

    def calls_for(self, fov):
        return [c for c in self.finalizeCalls if c['fov'] == fov]

    def assert_remeasured(self, fov, owner, members, theta):
        # one xy pass per stack, by its owner, on planes turned back by the
        # new angle, about the rigid offset re-expressed in that frame, with
        # the fragment's z curve
        calls = self.calls_for(fov)
        self.assertEqual(len(calls), 1, calls)
        call = calls[0]
        self.assertEqual(call['dataChannel'], owner)
        # (angle_of parses the table with pandas' default float parser)
        self.assertAlmostEqual(call['frame'], theta, places=15)
        sy, sx = RIGID[owner]
        frameX, frameY = rotate(sx, sy, -theta)
        self.assertAlmostEqual(call['rigidY'], frameY, places=12)
        self.assertAlmostEqual(call['rigidX'], frameX, places=12)
        np.testing.assert_allclose(call['zCoefficients'], [0.01, 0.2],
                                   atol=1e-12)
        self.assertEqual(self.task._measurementRotation, {})
        # the samples, turned back into the moving frame and refitted: the
        # rigid offset plus the drift turned by theta, for every member
        dx, dy = rotate(DRIFT[0], DRIFT[1], theta)
        table = self.task.get_transformation_table(fov)
        coeff = self.task.get_polynomial_coefficients(fov).set_index(
            'dataChannel')
        for dataChannel in members:
            rows = table[table.dataChannel == dataChannel]
            np.testing.assert_allclose(-rows.xshift, sx + dx * rows.zPos,
                                       atol=1e-9)
            np.testing.assert_allclose(-rows.yshift, sy + dy * rows.zPos,
                                       atol=1e-9)
            self.assertAlmostEqual(coeff.loc[dataChannel, 'y_c1'], dy,
                                   places=9)
            self.assertAlmostEqual(coeff.loc[dataChannel, 'x_c0'], sx,
                                   places=9)
        report = self.task.dataSet.load_dataframe_from_csv(
            'rotation_neighbor_substitutions').set_index(
                ['fov', 'dataChannel'])
        for dataChannel in members:
            rec = report.loc[(fov, dataChannel)]
            self.assertEqual(rec.xy_method, 'remeasured')
            self.assertEqual(rec.owner_channel, owner)
            self.assertEqual(rec.xy_samples, XY_SAMPLES)
            self.assertEqual(rec.xy_kept, XY_SAMPLES)

    def test_fallback_stack_remeasured_at_the_new_angle(self):
        # fov 4's r1 stack (channels 1, 2; owner 1): its angle fell back to
        # 0, so the fragment measured it on unrotated planes
        theta = self.angle_of(4, 1)
        self.assertNotEqual(theta, 0.0)
        self.assert_remeasured(4, 1, (1, 2), theta)
        before = self.fragment_table(4)
        self.assertTrue((before[before.dataChannel == 1].rotation_radians
                         == 0.0).all())

    def test_scattered_stack_remeasured_at_the_new_angle(self):
        # fov 0's r2 stack was measured on planes turned back by its own
        # first-pass angle; the new pass uses the substituted one
        status = self.task.get_rotation_status(0).set_index('dataChannel')
        self.assertAlmostEqual(status.loc[3, 'first_pass_radians'],
                               true_angle(0, 'r2'), places=12)
        self.assert_remeasured(0, 3, (3,), self.angle_of(0, 3))

    def test_report_keeps_the_formula_for_comparison(self):
        report = self.task.dataSet.load_dataframe_from_csv(
            'rotation_neighbor_substitutions').set_index(
                ['fov', 'dataChannel'])
        rec = report.loc[(4, 1)]
        dY, dX = self.task._centroid_shift_correction(
            *RIGID[1], 0.0, self.angle_of(4, 1))
        self.assertAlmostEqual(rec.formula_xshift_px, -dX, places=12)
        self.assertAlmostEqual(rec.formula_yshift_px, -dY, places=12)
        before = self.fragment_table(4)
        after = self.task.get_transformation_table(4)
        middle = len(Z) // 2
        change = (after[after.dataChannel == 1].xshift.to_numpy()
                  - before[before.dataChannel == 1].xshift.to_numpy())
        self.assertAlmostEqual(rec.xshift_change_px, change[middle],
                               places=12)
        self.assertTrue(pd.isna(rec.xy_failure) or rec.xy_failure == '')

    def test_xy_samples_written(self):
        samples = self.task.dataSet.load_dataframe_from_csv(
            'rotation_neighbor_xy_samples')
        self.assertEqual(sorted(set(zip(samples.fov, samples.dataChannel))),
                         [(0, 3), (4, 1)])
        self.assertEqual(len(samples), 2 * XY_SAMPLES)
        self.assertTrue(samples.ok.all())

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
            ('rotation_neighbor_substitutions', None),
            ('rotation_neighbor_xy_samples', None)}, changed)
        for fov in (1, 2, 3, 5, 6, 7):
            for name in ('transformation_table', 'polynomial_coefficients',
                         'rotation_status', 'offsets'):
                self.assertEqual(self.before[name, fov],
                                 self.after[name, fov])
            self.assertEqual(self.calls_for(fov), [])
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
        self.assertEqual(self.calls_for(8), [])
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
        self.assertEqual(set(report.xy_method), {'remeasured'})
        self.assertIn('substituted 3 angles in 2 fovs', self.log)
        self.assertIn('xy re-measured for 3, formula fallback for 0',
                      self.log)
        self.assertNotIn('***', self.log)

    def test_idempotent(self):
        calls = len(self.task.xyCalls)
        with contextlib.redirect_stdout(io.StringIO()):
            self.task.finalize()
        self.assertEqual(snapshot(self.task.dataSet), self.after)
        self.assertEqual(len(self.task.xyCalls), calls)


class AngleFallbackFormulaTests(unittest.TestCase):
    """When the xy pass at the new angle returns too few samples, the
    fragment's shifts are corrected by the overlap-centroid formula instead,
    and the stack is flagged."""

    @classmethod
    def setUpClass(cls):
        cls.task = run()
        cls.before = snapshot(cls.task.dataSet)

        def too_few(fov, dataChannel, rigidY, rigidX, zCoefficients):
            z = Z[1:3]
            return (z, rigidY + 0 * z, rigidX + 0 * z, np.full(2, 50.0),
                    np.zeros(2, bool))

        cls.task._measure_xy_offsets = too_few
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cls.task.finalize()
        cls.log = out.getvalue()
        cls.after = snapshot(cls.task.dataSet)

    def angle_of(self, fov, dataChannel):
        table = self.task.get_transformation_table(fov)
        return float(table[table.dataChannel == dataChannel]
                     .rotation_radians.iloc[0])

    def report(self):
        return self.task.dataSet.load_dataframe_from_csv(
            'rotation_neighbor_substitutions').set_index(
                ['fov', 'dataChannel'])

    def test_flagged(self):
        report = self.report()
        self.assertEqual(set(report.xy_method), {'formula_fallback'})
        self.assertEqual(report.loc[(4, 1), 'xy_failure'],
                         'only 2 of 2 xy samples usable')
        self.assertIn('*** fov 4 channel 1: xy NOT re-measured', self.log)
        self.assertIn('xy re-measured for 0, formula fallback for 3',
                      self.log)
        samples = self.task.dataSet.load_dataframe_from_csv(
            'rotation_neighbor_xy_samples')
        self.assertFalse(samples.ok.any())

    def test_only_constant_terms_change(self):
        for name, changed in (
                ('transformation_table',
                 ['rotation_radians', 'xshift', 'yshift']),
                ('polynomial_coefficients',
                 ['rotation_radians', 'x_c0', 'y_c0'])):
            before = pd.read_csv(io.StringIO(self.before[name, 4]),
                                 dtype=str)
            after = pd.read_csv(io.StringIO(self.after[name, 4]), dtype=str)
            touched = after.dataChannel.isin(['1', '2'])
            pd.testing.assert_frame_equal(before[~touched], after[~touched])
            pd.testing.assert_frame_equal(
                before[touched].drop(columns=changed),
                after[touched].drop(columns=changed))

    def test_fallback_shift_corrected_about_the_overlap_centroid(self):
        # fov 4's r1 stack: angle fell back to 0, so its xy was measured on
        # unrotated planes. The formula takes that as the displacement at
        # the geometric overlap centroid, c + s / 2 for a rigid shift s, and
        # the transform reproduces it there and turns about it.
        theta = self.angle_of(4, 1)
        self.assertNotEqual(theta, 0.0)
        sy, sx = RIGID[1]
        centroid = CENTRE + 0.5 * np.array([sx, sy])
        before = pd.read_csv(io.StringIO(self.before['transformation_table',
                                                     4]))
        for zIndex in (0, 17, 40):
            row = before[(before.dataChannel == 1)
                         & (before.zIndex == zIndex)].iloc[0]
            self.assertEqual(row.rotation_radians, 0.0)
            measured = centroid + [row.xshift, row.yshift]
            t = self.task.get_transformation(4, 1, zIndex)
            np.testing.assert_allclose(t([centroid])[0], measured,
                                       atol=1e-9)
            p = centroid + [100.0, -60.0]
            c, s = np.cos(theta), np.sin(theta)
            expected = measured + [c * 100.0 + s * 60.0,
                                   s * 100.0 - c * 60.0]
            np.testing.assert_allclose(t([p])[0], expected, atol=1e-9)
        # the size of the correction: |theta| x |s| / 2, a constant change
        rec = self.report().loc[(4, 1)]
        dx, dy = rec.formula_xshift_px, rec.formula_yshift_px
        self.assertAlmostEqual(np.hypot(dx, dy),
                               2 * np.sin(abs(theta) / 2)
                               * 0.5 * np.hypot(sx, sy), places=12)
        self.assertEqual(rec.xy_frame_deg, 0.0)
        self.assertAlmostEqual(rec.xshift_change_px, dx, places=12)
        self.assertAlmostEqual(rec.max_shift_change_px, np.hypot(dx, dy),
                               places=9)
        after = self.task.get_transformation_table(4)
        np.testing.assert_allclose(
            after[after.dataChannel == 1].xshift.to_numpy()
            - before[before.dataChannel == 1].xshift.to_numpy(), dx,
            atol=1e-12)

    def test_scattered_shift_corrected_by_the_angle_change_only(self):
        status = self.task.get_rotation_status(0).set_index('dataChannel')
        frame = status.loc[3, 'first_pass_radians']
        newTheta = self.angle_of(0, 3)
        expected = self.task._centroid_shift_correction(*RIGID[3], frame,
                                                        newTheta)
        rec = self.report().loc[(0, 3)]
        self.assertAlmostEqual(rec.formula_yshift_px, -expected[0],
                               places=12)
        self.assertAlmostEqual(rec.formula_xshift_px, -expected[1],
                               places=12)
        delta = newTheta - frame
        self.assertAlmostEqual(np.hypot(*expected),
                               2 * np.sin(abs(delta) / 2)
                               * 0.5 * np.hypot(*RIGID[3]), places=12)
        self.assertLess(np.hypot(*expected), 0.01)

    def test_idempotent(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.task.finalize()
        self.assertEqual(snapshot(self.task.dataSet), self.after)


class AngleRemeasureZCurveTests(unittest.TestCase):
    """A stack whose z the fragment could not fit was measured in xy with
    zero z coefficients; the xy pass at the new angle uses the same, not the
    curve finalize() substituted."""

    def test_unfitted_z_uses_the_fragments_zero_curve(self):
        task = run(zUnfitted=(Z_UNFITTED, NO_ANGLE))
        calls = len(task.xyCalls)
        with contextlib.redirect_stdout(io.StringIO()):
            task.finalize()
        status = task.get_z_fit_status(4).set_index('dataChannel')
        self.assertEqual(status.loc[1, 'source'], 'neighbor')
        table = task.get_transformation_table(4)
        self.assertTrue((table[table.dataChannel == 1].z_offset != 0).all())
        remeasured = [c for c in task.xyCalls[calls:] if c['fov'] == 4]
        self.assertEqual(len(remeasured), 1)
        np.testing.assert_array_equal(remeasured[0]['zCoefficients'],
                                      [0.0, 0.0])


class AngleDonorsNeedZFitTests(unittest.TestCase):
    """A stack whose angle is trusted but whose z did not fit is no donor."""

    def test_unfitted_z_is_no_donor(self):
        task = run(zUnfitted=(Z_UNFITTED, (1, 'r2')))
        status = task.get_rotation_status(1).set_index('dataChannel')
        self.assertEqual(status.loc[3, 'source'], 'fit')
        self.assertFalse(task.get_z_fit_status(1).set_index('dataChannel')
                         .loc[3, 'fitted'])
        with contextlib.redirect_stdout(io.StringIO()):
            task.finalize()
        report = task.dataSet.load_dataframe_from_csv(
            'rotation_neighbor_substitutions').set_index(
                ['fov', 'dataChannel'])
        # fov 0's 400 um ring now holds only fovs 3 and 4, so the 580 um
        # ring is used, without fov 1
        donors = [f for f in neighbours(0, 580.0) if f != 1]
        self.assertEqual(report.loc[(0, 3), 'donors'],
                         ','.join(str(f) for f in sorted(donors)))
        self.assertEqual(report.loc[(0, 3), 'ring_um'], 580.0)
        table = task.get_transformation_table(0)
        self.assertAlmostEqual(
            float(table[table.dataChannel == 3].rotation_radians.iloc[0]),
            np.median([true_angle(f, 'r2') for f in donors]), places=12)


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
