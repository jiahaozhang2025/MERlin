"""FiducialPolynomialWarp3D z_xy_refine: z sampled at each depth's own xy.

The z samples are measured with the moving planes shifted by one xy, the
rigid offset. When the bead stack's xy moves by a few pixels with depth, the
z samples below the move compare misaligned bead textures, their ZNCC falls
under z_min_score and they fail the gate, although the xy pass (searched
+-xy_search_bound about the rigid offset) measures the drift without
trouble. z_xy_refine measures z again with each sample's planes shifted by
the xy fit at that sample's depth.

Synthetic bead VOLUMES (beads extend over several planes in z, so a plane
correlates with its neighbours as real bead stacks do), rendered for the
reference round and for a moving round that is displaced in z by dz(z) and
in xy by a smooth ~3 px step a little below the top of the stack. The 2D
frame carries the shallow xy, so the rigid offset is right only above the
step. The task's measurement, gate, fit and table code run unchanged; only
the dataset is a stand-in.

Run with `python -m unittest tests/test_polywarp_z_xy_refine.py`.
"""
import contextlib
import io
import unittest

import cv2
import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis.warp import FiducialPolynomialWarp3D

SIZE = 256
NPLANES = 40
Z = np.arange(NPLANES) * 0.5                  # um
SIGMA_Z = 0.8                                 # um, bead extent in depth
SHALLOW = np.array([12.3, -8.7])              # measured shift (dy, dx) on top
STEP = np.array([2.2, -2.1])                  # extra shift below the step
STEP_Z, STEP_WIDTH = 2.5, 0.4                 # um


def true_dz(z):
    return 0.6 + 0.01 * np.asarray(z, float)


def true_shift(z):
    """(dy, dx) shift to apply to the moving plane at reference depth z"""
    w = 1.0 / (1.0 + np.exp(-(np.asarray(z, float) - STEP_Z) / STEP_WIDTH))
    return SHALLOW[:, None] * np.ones_like(w) + STEP[:, None] * w


def splat(ys, xs, weights, shape):
    """bilinear splat of point sources, then a lateral Gaussian blur"""
    image = np.zeros(shape, np.float64)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = ys - y0, xs - x0
    for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx),
                      (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        ok = (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
        np.add.at(image, (yy[ok], xx[ok]), (weights * w)[ok])
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.2)


def bead_volume(seed, n=4000, size=SIZE):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-40, size + 40, n), rng.uniform(-40, size + 40, n),
            rng.uniform(Z[0] - 3, Z[-1] + 3, n), rng.uniform(800, 3000, n))


def render(volume, moving, size=SIZE, shift=true_shift, dz=true_dz,
           angle=0.0):
    """planes of the reference round, or of the moving round: a bead at
    reference (x, y, z) sits in the moving round at depth z + dz(z) and at
    c + R(angle) (p - c) - shift(z), shift being what brings it back"""
    xs, ys, zs, amplitude = volume
    if moving:
        sy, sx = shift(zs)
        c = (size - 1) / 2.0
        ca, sa = np.cos(angle), np.sin(angle)
        xs, ys = (c + ca * (xs - c) - sa * (ys - c) - sx,
                  c + sa * (xs - c) + ca * (ys - c) - sy)
        zs = zs + dz(zs)
    planes = []
    for zPlane in Z:
        weight = amplitude * np.exp(-0.5 * ((zs - zPlane) / SIGMA_Z) ** 2)
        near = weight > 1.0
        planes.append(splat(ys[near], xs[near], weight[near],
                            (size, size)) + 100)
    return planes


def two_d(seed, moving, size=SIZE, angle=0.0):
    rng = np.random.default_rng(seed)
    volume = (rng.uniform(-40, size + 40, 300), rng.uniform(-40, size + 40, 300),
              np.zeros(300), rng.uniform(800, 3000, 300))
    xs, ys, _, amplitude = volume
    if moving:
        sy, sx = SHALLOW
        c = (size - 1) / 2.0
        ca, sa = np.cos(angle), np.sin(angle)
        xs, ys = (c + ca * (xs - c) - sa * (ys - c) - sx,
                  c + sa * (xs - c) + ca * (ys - c) - sy)
    # independent noise per frame: shared noise would correlate at zero shift
    noise = np.random.default_rng(seed + 1 + int(moving)).normal(
        0, 40, (size, size))
    return splat(ys, xs, amplitude, (size, size)) + 100 + noise.astype(
        np.float32)


class _Org:
    def get_data_channels(self):
        return np.array([0, 1])

    def get_fiducial3D_filename(self, dataChannel, fov):
        return dataChannel

    def get_fiducial3D_stack_frame_indices(self, dataChannel):
        return np.arange(NPLANES)

    def get_fiducial3D_stack_frame_zPos(self, dataChannel):
        return Z

    def get_data_channel_color(self, dataChannel):
        return '650'


class _DataSet:
    def __init__(self, stacks, twoD, size=SIZE, tasks=None):
        self.stacks, self.twoD, self.size = stacks, twoD, size
        self.saved = {}
        self.tasks = dict(tasks or {})

    def load_analysis_task(self, name):
        return self.tasks[name]

    def get_data_organization(self):
        return _Org()

    def get_fovs(self):
        return [0]

    def get_z_positions(self):
        return list(Z)

    def get_image_dimensions(self):
        return (self.size, self.size)

    def get_fiducial_image(self, dataChannel, fov):
        return self.twoD[dataChannel]

    def load_image(self, name, frame):
        return self.stacks[name][int(frame)]

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


VOLUME = bead_volume(3)
STACKS = {0: render(VOLUME, False), 1: render(VOLUME, True)}
TWO_D = {0: two_d(11, False), 1: two_d(11, True)}

PARAMETERS = {
    'rigid_search_bound': 40, 'rigid_plane_search': 0,
    'z_search_range': 3, 'z_sample_step': 3, 'xy_sample_step': 2,
    'z_polynomial_order': 1, 'xy_polynomial_order': 3,
    'z_min_score': 0.5, 'z_min_good_samples': 4,
    'minimum_overlap_pixels': 1000, 'edge_width_to_remove': 8,
    'z_neighbor_substitution': False,
    'write_fiducial_images': False, 'write_aligned_images': False}


def run(refine, stacks=STACKS, twoD=TWO_D, size=SIZE, tasks=None, **extra):
    parameters = dict(PARAMETERS, **extra)
    if refine is not None:
        parameters['z_xy_refine'] = refine
    task = FiducialPolynomialWarp3D(_DataSet(stacks, twoD, size, tasks),
                                    parameters, 'PolyWarp')
    with contextlib.redirect_stdout(io.StringIO()) as out:
        task._run_analysis(0)
    task.log = out.getvalue()
    return task


def table(task, name):
    return task.dataSet.load_dataframe_from_csv(name, resultIndex=0)


class ZXyRefineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.off = run(False)
        cls.on = run(True)
        cls.default = run(None)

    def status(self, task):
        return table(task, 'z_fit_status').set_index('dataChannel').loc[1]

    def z_samples(self, task, stage='z'):
        m = table(task, 'measurements')
        return m[(m.dataChannel == 1) & (m.stage == stage)]

    def test_premise_off_fails_the_gate(self):
        # every z sample sits below the step: at the rigid (shallow) xy they
        # compare beads ~3 px apart and fail the gate; z is not fitted
        status = self.status(self.off)
        self.assertLessEqual(status.n_good, 2)
        self.assertFalse(status.fitted)
        self.assertEqual(status.source, 'unfitted')
        samples = self.z_samples(self.off)
        self.assertLess(samples[samples.zPos > STEP_Z].score.max(), 0.5)
        # ... while the xy pass found the step all the same
        coeff = table(self.off, 'polynomial_coefficients')\
            .set_index('dataChannel')
        deep = Z[-5]
        self.assertLess(abs(np.polyval(
            [coeff.loc[1, 'x_c%d' % k] for k in (3, 2, 1, 0)], deep)
            - true_shift(deep)[1, 0]), 0.6)

    def test_default_is_off(self):
        self.assertFalse(self.default.parameters['z_xy_refine'])
        for name in ('z_fit_status', 'transformation_table',
                     'polynomial_coefficients', 'measurements',
                     'fit_quality_flags'):
            self.assertEqual(self.default.dataSet.saved[name, 0],
                             self.off.dataSet.saved[name, 0], name)
        self.assertEqual(list(table(self.off, 'z_fit_status').columns), [
            'fov', 'dataChannel', 'owner_channel', 'n_samples', 'n_good',
            'n_gated_out', 'fitted', 'source'])
        self.assertNotIn('z_xy_refine', self.off.log)

    def test_refine_recovers_the_z_samples(self):
        status = self.status(self.on)
        self.assertEqual(status.n_good, status.n_samples)
        self.assertTrue(status.fitted)
        self.assertTrue(status.z_refined)
        self.assertEqual(status.n_good_first, self.status(self.off).n_good)
        self.assertFalse(status.fitted_first)
        # the largest |xy(z) - rigid| the z samples were shifted by: the step
        self.assertGreater(status.refine_max_shift_px, 2.5)
        self.assertLess(status.refine_max_shift_px, 3.6)
        self.assertGreater(self.z_samples(self.on).score.min(), 0.8)

    def test_refined_z_and_xy_match_truth(self):
        t = table(self.on, 'transformation_table')
        moving = t[t.dataChannel == 1]
        inner = moving[(moving.zPos > 3.5) & (moving.zPos < Z[-1] - 2)]
        np.testing.assert_allclose(inner.z_offset, true_dz(inner.zPos),
                                   atol=0.08)
        # a cubic in z cannot follow a step this sharp within ~2 um of it
        inner = inner[inner.zPos >= STEP_Z + 2.5]
        shift = true_shift(inner.zPos.to_numpy())
        np.testing.assert_allclose(-inner.yshift, shift[0], atol=0.3)
        np.testing.assert_allclose(-inner.xshift, shift[1], atol=0.3)

    def test_first_pass_recorded_unchanged(self):
        # stage 'z_first' holds exactly the samples the option-off run used
        first = self.z_samples(self.on, 'z_first')
        off = self.z_samples(self.off)
        for column in ('zPos', 'measured', 'score', 'kept', 'fitted'):
            np.testing.assert_array_equal(first[column].to_numpy(),
                                          off[column].to_numpy(), column)
        self.assertIn('z_xy_refine: %d -> %d good z samples'
                      % (self.status(self.off).n_good,
                         self.status(self.on).n_good), self.on.log)

    def test_reference_stack_unharmed(self):
        status = table(self.on, 'z_fit_status').set_index('dataChannel').loc[0]
        self.assertEqual(status.n_good, status.n_samples)
        self.assertLess(status.refine_max_shift_px, 0.05)
        t = table(self.on, 'transformation_table')
        reference = t[t.dataChannel == 0]
        np.testing.assert_allclose(reference.z_offset, 0.0, atol=0.02)
        np.testing.assert_allclose(reference.xshift, 0.0, atol=0.05)

    def test_option_must_be_boolean(self):
        with self.assertRaises(ValueError):
            FiducialPolynomialWarp3D(_DataSet(STACKS, TWO_D),
                                     {'z_xy_refine': 1}, 'PolyWarp')


class ZXyRefineKeepsBetterSamplesTests(unittest.TestCase):
    """The refined samples replace the first pass's only if at least as many
    pass the gate; otherwise the first pass stands and nothing changes."""

    def run_stubbed(self, refinedScore):
        task = FiducialPolynomialWarp3D(_DataSet(STACKS, TWO_D), dict(
            PARAMETERS, z_xy_refine=True), 'PolyWarp')
        calls = []
        zs = Z[4:-4:3]

        def measure_z(fov, dataChannel, rigidY, rigidX,
                      shiftCoefficients=None):
            calls.append(shiftCoefficients is not None)
            score = np.full(len(zs), 0.9)
            if shiftCoefficients is not None:
                score[:] = refinedScore
            return zs, 0.5 + 0.0 * zs, score, 1.5

        task._measure_z_offsets = measure_z
        with contextlib.redirect_stdout(io.StringIO()):
            task._run_analysis(0)
        return task, calls

    def test_worse_refinement_is_discarded(self):
        task, calls = self.run_stubbed(0.1)
        self.assertEqual(calls, [False, True, False, True])
        status = table(task, 'z_fit_status').set_index('dataChannel').loc[1]
        self.assertFalse(status.z_refined)
        self.assertEqual(status.n_good, status.n_good_first)
        m = table(task, 'measurements')
        z = m[(m.dataChannel == 1) & (m.stage == 'z')]
        np.testing.assert_array_equal(z.score, 0.9)

    def test_equal_refinement_is_used(self):
        task, _ = self.run_stubbed(0.95)
        status = table(task, 'z_fit_status').set_index('dataChannel').loc[1]
        self.assertTrue(status.z_refined)
        m = table(task, 'measurements')
        z = m[(m.dataChannel == 1) & (m.stage == 'z')]
        np.testing.assert_array_equal(z.score, 0.95)


# ---- with rotation_task: the refinement runs on the turned-back planes ----

RSIZE = 384
THETA_DEG = 0.35
RVOLUME = bead_volume(21, n=9000, size=RSIZE)
RSTACKS = {0: render(RVOLUME, False, size=RSIZE),
           1: render(RVOLUME, True, size=RSIZE,
                     angle=np.radians(THETA_DEG))}
RTWO_D = {0: two_d(31, False, size=RSIZE),
          1: two_d(31, True, size=RSIZE, angle=np.radians(THETA_DEG))}


class _RoundTask:
    """Stands in for a finalized FiducialRoundRotation task: channel 1's
    round has the true angle."""

    def get_round_angles(self):
        return {0: 0.0, 1: float(np.radians(THETA_DEG))}

    def get_fov_verdicts(self, fov):
        return {1: (float(np.radians(THETA_DEG)), False, '')}


ROUND = dict(tasks={'RoundRot': _RoundTask()}, rotation_task='RoundRot')


def truth(zPos):
    """output->input map of the moving round at reference depth zPos"""
    theta = np.radians(THETA_DEG)
    c, s = np.cos(theta), np.sin(theta)
    sy, sx = true_shift(zPos)[:, 0]
    cx = cy = (RSIZE - 1) / 2.0
    return np.array([[c, -s, cx - c * cx + s * cy - sx],
                     [s, c, cy - s * cx - c * cy - sy], [0, 0, 1.0]])


class ZXyRefineRoundAngleTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.off = run(False, RSTACKS, RTWO_D, RSIZE, **ROUND)
        cls.on = run(True, RSTACKS, RTWO_D, RSIZE, **ROUND)

    def test_premise_off_unfitted(self):
        status = table(self.off, 'z_fit_status').set_index('dataChannel')
        self.assertFalse(status.loc[1, 'fitted'])

    def test_refined_z_with_the_round_angle(self):
        status = table(self.on, 'z_fit_status').set_index('dataChannel')
        self.assertEqual(status.loc[1, 'n_good'], status.loc[1, 'n_samples'])
        rotation = table(self.on, 'rotation_status').set_index('dataChannel')
        # the angle is the round's, not refined (to the csv round trip)
        self.assertAlmostEqual(rotation.loc[1, 'rotation_radians'],
                               np.radians(THETA_DEG), places=15)
        self.assertEqual(rotation.loc[1, 'unrefined_radians'],
                         rotation.loc[1, 'rotation_radians'])
        self.assertEqual(rotation.loc[1, 'source'], 'round')
        t = table(self.on, 'transformation_table')
        moving = t[t.dataChannel == 1]
        inner = moving[(moving.zPos > 3.5) & (moving.zPos < Z[-1] - 2)]
        np.testing.assert_allclose(inner.z_offset, true_dz(inner.zPos),
                                   atol=0.08)

    def test_transform_matches_truth_at_corners(self):
        corners = np.array([[40, 40], [RSIZE - 40, 40], [40, RSIZE - 40],
                            [RSIZE - 40, RSIZE - 40],
                            [(RSIZE - 1) / 2.0, (RSIZE - 1) / 2.0]])
        for zIndex in (12, 24, 32):
            fitted = self.on.get_transformation(0, 1, zIndex)
            error = np.linalg.norm(
                transform.matrix_transform(corners, fitted.params)
                - transform.matrix_transform(corners, truth(Z[zIndex])),
                axis=1)
            self.assertLess(error.max(), 0.35, (zIndex, error))


if __name__ == '__main__':
    unittest.main()
