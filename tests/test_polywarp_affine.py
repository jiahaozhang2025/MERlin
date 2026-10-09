"""FiducialPolynomialWarp3D affine_correction: a per-fov linear part (stretch,
shear and a small rotation) that changes with depth, on top of the round
angle.

Synthetic bead stacks: the moving round's plane k+1 is the fixed plane k
mapped by a known c + R(theta) L(z) (p - c) + t(z), with theta a round angle
(given by a stand-in for a finalized FiducialRoundRotation task), L(z) a
linear part that changes linearly with depth, t(z) a translation that drifts
with depth, and a one-plane z offset. The beads cover only part of the frame,
so the frame centre and the right-hand corners are extrapolated. The task's
own measurement, fit, table and application code run end to end; only the
dataset and the round task are stand-ins.

Run with `python -m unittest tests/test_polywarp_affine.py`.
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
CENTRE = np.array([(SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0])
NPLANES = 14
Z = np.arange(NPLANES, dtype=float)
THETA_DEG = 0.3
# beads only left of this column: the frame centre has none
BEAD_LIMIT_X = 230
# D = L - I at the first and the last plane, as [[xx, xy], [yx, yy]] acting
# on (x, y): stretch, shear and a local rotation, all changing with depth
D_FIRST = np.array([[1.0e-3, 2.0e-3], [-1.0e-3, -0.5e-3]])
D_LAST = np.array([[3.0e-3, -1.5e-3], [3.5e-3, -2.5e-3]])
CORNERS = np.array([[0, 0], [SHAPE[1] - 1, 0], [0, SHAPE[0] - 1],
                    [SHAPE[1] - 1, SHAPE[0] - 1]], dtype=float)


def deformation(z):
    return D_FIRST + (D_LAST - D_FIRST) * z / (NPLANES - 1)


def translation(z):
    return np.array([12.3 + 0.15 * z + 0.01 * z * z, -8.7 + 0.1 * z])


def truth(z, theta_deg=THETA_DEG, linear=True):
    """output->input map of the moving round at depth z: the content at
    reference p sits in the moving frame at c + R L(z) (p - c) + t(z)."""
    theta = np.radians(theta_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    matrix = rotation @ (np.eye(2) + (deformation(z) if linear else 0.0))
    offset = CENTRE - matrix @ CENTRE + translation(z)
    return np.array([[matrix[0, 0], matrix[0, 1], offset[0]],
                     [matrix[1, 0], matrix[1, 1], offset[1]],
                     [0.0, 0.0, 1.0]])


def beads(seed, n=700, limit=BEAD_LIMIT_X):
    rng = np.random.default_rng(seed)
    image = np.zeros(SHAPE, np.float32)
    x = rng.integers(8, limit, n)
    y = rng.integers(8, SHAPE[0] - 8, n)
    np.add.at(image, (y, x), rng.uniform(800, 3000, n))
    return cv2.GaussianBlur(image, (0, 0), 1.2) + 100


def moved(image, matrix):
    # forward map: moving(matrix p) = image(p)
    return cv2.warpAffine(image, matrix[:2], SHAPE[::-1],
                          flags=cv2.INTER_LINEAR, borderValue=100)


class _Org:
    def __init__(self, channels=(0, 1)):
        self.channels = np.array(channels)

    def get_data_channels(self):
        return self.channels

    def get_fiducial3D_filename(self, dataChannel, fov):
        return str(dataChannel)

    def get_fiducial3D_stack_frame_indices(self, dataChannel):
        return np.arange(NPLANES)

    def get_fiducial3D_stack_frame_zPos(self, dataChannel):
        return Z

    def get_data_channel_color(self, dataChannel):
        return '650'

    def get_data_channel_index(self, name):
        return int(name)


class _RoundTask:
    """Stands in for a finalized FiducialRoundRotation task."""

    def get_round_angles(self):
        return {0: 0.0, 1: float(np.radians(THETA_DEG)),
                2: float(np.radians(THETA_DEG))}

    def get_fov_verdicts(self, fov):
        return {c: (float(np.radians(THETA_DEG)), False, '') for c in (1, 2)}


class _DataSet:
    def __init__(self, theta_deg=THETA_DEG, linear=True, channels=(0, 1)):
        self.channels = channels
        self.fixed = [beads(10 + i) for i in range(NPLANES)]
        # one-plane axial offset: moving plane k+1 holds fixed plane k
        self.moving = [beads(999)] + [
            moved(self.fixed[k], truth(Z[k], theta_deg, linear))
            for k in range(NPLANES - 1)]
        self.fixed2D = beads(5)
        self.moving2D = moved(self.fixed2D, truth(0.0, theta_deg, linear))
        self.saved = {}
        self.tasks = {'RoundRot': _RoundTask()}

    def get_data_organization(self):
        return _Org(self.channels)

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
        return pd.read_csv(io.StringIO(self.saved[resultName, resultIndex]),
                           **kwargs)

    def save_numpy_analysis_result(self, analysisResult, resultName,
                                   analysisName, resultIndex=None,
                                   subdirectory=None):
        self.saved[resultName, resultIndex] = np.array(analysisResult)

    def load_analysis_task(self, name):
        return self.tasks[name]


PARAMETERS = {
    'z_search_range': 2, 'z_sample_step': 1, 'xy_sample_step': 1,
    'z_polynomial_order': 0, 'xy_polynomial_order': 2,
    'minimum_overlap_pixels': 1000, 'edge_width_to_remove': 5,
    'rigid_search_bound': 50, 'z_min_good_samples': 3, 'z_min_score': 0.2,
    'affine_min_samples': 6, 'rotation_task': 'RoundRot',
    'write_fiducial_images': False, 'write_aligned_images': False,
    'boundary_smooth': False}


def make(dataSet, **extra):
    task = FiducialPolynomialWarp3D(dataSet, dict(PARAMETERS, **extra),
                                    'PolyWarp')
    dataSet.tasks['PolyWarp'] = task
    return task


def run(dataSet=None, **extra):
    dataSet = dataSet if dataSet is not None else _DataSet()
    task = make(dataSet, **extra)
    with contextlib.redirect_stdout(io.StringIO()):
        task._run_analysis(0)
    return task


def corner_error(task, zIndex, expected, points=CORNERS):
    fitted = task.get_transformation(0, 1, zIndex)
    return np.linalg.norm(transform.matrix_transform(points, fitted.params)
                          - transform.matrix_transform(points, expected),
                          axis=1)


class AffineRecoveryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.on = run(affine_correction=True)
        cls.off = run()

    def test_corners_recovered(self):
        # every frame corner, including the two on the side without beads
        for zIndex in range(1, NPLANES - 2):
            error = corner_error(self.on, zIndex, truth(Z[zIndex]))
            self.assertLess(error.max(), 0.2, (zIndex, error))
        # the round angle alone leaves the linear part at the corners
        error = corner_error(self.off, NPLANES - 3, truth(Z[NPLANES - 3]))
        self.assertGreater(error.max(), 1.0, error)

    def test_centre_recovered_without_beads_there(self):
        # the frame centre holds no beads: its shift comes from the frame
        # shift moved from the beads' centroid with the fitted L
        centre = CENTRE[None, :]
        for zIndex in range(1, NPLANES - 2):
            error = corner_error(self.on, zIndex, truth(Z[zIndex]), centre)
            self.assertLess(error.max(), 0.1, (zIndex, error))
        samples = self.on.get_affine_samples(0)
        moving = samples[samples.dataChannel == 1]
        correction = np.hypot(moving.centre_shift_x - moving.frame_shift_x,
                              moving.centre_shift_y - moving.frame_shift_y)
        self.assertGreater(correction.max(), 0.2)
        # the tile fit's own centre shift agrees with the re-referenced one
        agree = np.hypot(moving.tile_shift_x - moving.centre_shift_x,
                         moving.tile_shift_y - moving.centre_shift_y)
        self.assertLess(np.median(agree), 0.05, agree)

    def test_linear_part_in_the_table(self):
        table = self.on.get_transformation_table(0)
        rows = table[table.dataChannel == 1].sort_values('zIndex')
        for zIndex in range(1, NPLANES - 2):
            row = rows[rows.zIndex == zIndex].iloc[0]
            linear = np.array([[row.affine_xx, row.affine_xy],
                               [row.affine_yx, row.affine_yy]])
            np.testing.assert_allclose(
                linear - np.eye(2), deformation(Z[zIndex]), atol=2.5e-4)
        np.testing.assert_allclose(rows.affine_center_x, CENTRE[0])
        np.testing.assert_allclose(rows.affine_center_y, CENTRE[1])
        reference = table[table.dataChannel == 0]
        np.testing.assert_array_equal(reference.affine_xx, 1.0)
        np.testing.assert_array_equal(reference.affine_xy, 0.0)
        coeff = self.on.get_polynomial_coefficients(0)
        for name in ('xx', 'xy', 'yx', 'yy'):
            self.assertIn('affine_d%s_c1' % name, coeff.columns)
            self.assertNotIn('affine_d%s_c2' % name, coeff.columns)

    def test_status_and_samples(self):
        status = self.on.get_affine_fit_status(0).set_index('dataChannel')
        self.assertEqual(status.loc[0, 'source'], 'reference')
        self.assertEqual(status.loc[1, 'source'], 'fit')
        self.assertTrue(bool(status.loc[1, 'fitted']))
        self.assertEqual(status.loc[1, 'n_samples'], NPLANES - 2)
        self.assertGreater(status.loc[1, 'max_corner_px'], 1.0)
        samples = self.on.get_affine_samples(0)
        self.assertEqual(sorted(samples.dataChannel.unique()), [1])
        self.assertTrue(samples.ok.all())
        flags = self.on.dataSet.load_dataframe_from_csv(
            'fit_quality_flags', resultIndex=0)
        self.assertFalse(flags.reasons.str.contains('affine').any())

    def test_off_writes_nothing_new(self):
        self.assertFalse(self.off.parameters['affine_correction'])
        table = self.off.get_transformation_table(0)
        self.assertFalse(any('affine' in c for c in table.columns))
        coeff = self.off.get_polynomial_coefficients(0)
        self.assertFalse(any('affine' in c for c in coeff.columns))
        for name in ('affine_fit_status', 'affine_samples'):
            self.assertNotIn((name, 0), self.off.dataSet.saved)

    def test_z_unchanged_by_the_option(self):
        # the linear part is measured after the z fit and does not touch it
        on = self.on.get_transformation_table(0)
        off = self.off.get_transformation_table(0)
        np.testing.assert_array_equal(on.z_offset, off.z_offset)
        np.testing.assert_allclose(on[on.dataChannel == 1].z_offset, 1.0,
                                   atol=0.05)

    def test_aligned_image_reproduces_reference(self):
        zIndex = NPLANES - 4
        aligned = self.on.get_aligned_image(0, 1, zIndex).astype(float)
        reference = self.on.dataSet.fixed[zIndex].astype(np.uint16)
        inner = (slice(30, -30), slice(30, BEAD_LIMIT_X))
        a, b = aligned[inner], reference[inner].astype(float)
        a, b = a - a.mean(), b - b.mean()
        score = float(a.ravel() @ b.ravel()
                      / np.linalg.norm(a) / np.linalg.norm(b))
        self.assertGreater(score, 0.97)

    def test_crop_bounds_cover_the_corners(self):
        top, bottom, left, right = compute_crop_bounds(
            self.on.dataSet, 'PolyWarp', 0, 0, True)
        corners = np.array([[left, top], [right - 1, top],
                            [left, bottom - 1], [right - 1, bottom - 1]])
        for t in self.on.get_transformation(0):
            mapped = t(corners)
            self.assertGreaterEqual(mapped.min(), -1e-9)
            self.assertLessEqual(mapped.max(), SHAPE[0] - 1 + 1e-9)

    def test_identity_row_gives_the_plain_transform(self):
        row = {'xshift': 3.25, 'yshift': -1.5, 'rotation_radians': 0.002,
               'rotation_center_x': CENTRE[0], 'rotation_center_y': CENTRE[1]}
        plain = FiducialPolynomialWarp3D._row_transform(row)
        identity = FiducialPolynomialWarp3D._row_transform(dict(
            row, affine_xx=1.0, affine_xy=0.0, affine_yx=0.0, affine_yy=1.0,
            affine_center_x=CENTRE[0], affine_center_y=CENTRE[1]))
        np.testing.assert_array_equal(plain.params, identity.params)
        with self.assertRaises(ValueError):
            FiducialPolynomialWarp3D._row_transform(dict(
                row, affine_xx=np.nan, affine_xy=0.0, affine_yx=0.0,
                affine_yy=1.0, affine_center_x=CENTRE[0],
                affine_center_y=CENTRE[1]))


class AffineWithoutRotationTests(unittest.TestCase):
    """affine_correction alone, with no round angle (the moving round is not
    turned)."""

    def test_corners_recovered(self):
        task = run(_DataSet(theta_deg=0.0), affine_correction=True,
                   rotation_task=None)
        table = task.get_transformation_table(0)
        self.assertFalse(any('rotation' in c for c in table.columns))
        for zIndex in range(1, NPLANES - 2):
            error = corner_error(task, zIndex, truth(Z[zIndex], 0.0))
            self.assertLess(error.max(), 0.2, (zIndex, error))


class AffineFallbackTests(unittest.TestCase):
    """A stack with too few usable samples keeps the identity and is
    registered exactly as without the option."""

    def test_identity_and_flag(self):
        dataSet = _DataSet()
        task = run(dataSet, affine_correction=True, affine_min_samples=100)
        plain = run(_DataSet())
        status = task.get_affine_fit_status(0).set_index('dataChannel')
        self.assertEqual(status.loc[1, 'source'], 'identity')
        self.assertIn('min 100', status.loc[1, 'reason'])
        flags = task.dataSet.load_dataframe_from_csv(
            'fit_quality_flags', resultIndex=0)
        self.assertTrue(flags.reasons.str.contains('affine').any())
        table = task.get_transformation_table(0)
        np.testing.assert_array_equal(table.affine_xx, 1.0)
        np.testing.assert_array_equal(table.affine_yx, 0.0)
        plainTable = plain.get_transformation_table(0)
        np.testing.assert_array_equal(
            table[plainTable.columns].to_numpy(), plainTable.to_numpy())
        for a, b in zip(task.get_transformation(0),
                        plain.get_transformation(0)):
            np.testing.assert_array_equal(a.params, b.params)
        np.testing.assert_array_equal(task.get_aligned_image(0, 1, 5),
                                      plain.get_aligned_image(0, 1, 5))

    def test_rigid_only_channel_keeps_the_identity(self):
        task = run(_DataSet(), affine_correction=True,
                   rigid_only_channels=[1])
        status = task.get_affine_fit_status(0).set_index('dataChannel')
        self.assertEqual(status.loc[1, 'source'], 'rigid_only')
        table = task.get_transformation_table(0)
        np.testing.assert_array_equal(table.affine_xx, 1.0)
        np.testing.assert_array_equal(table.affine_xy, 0.0)


class TilePositionTests(unittest.TestCase):
    """A tile that beads cover only partly measures the displacement where
    its beads are, not at its centre (see _tile_displacements)."""

    def test_weighted_position_removes_the_partial_tile_bias(self):
        # one plane pair, beads in a disc off the frame centre: the tiles on
        # its rim are partly covered, by different amounts
        rng = np.random.default_rng(3)
        image = np.zeros(SHAPE, np.float32)
        angle, radius = rng.uniform(0, 2 * np.pi, 3000), \
            200 * np.sqrt(rng.uniform(0, 1, 3000))
        x = (150 + radius * np.cos(angle)).astype(int)
        y = (300 + radius * np.sin(angle)).astype(int)
        inside = (x > 8) & (x < SHAPE[1] - 8) & (y > 8) & (y < SHAPE[0] - 8)
        np.add.at(image, (y[inside], x[inside]),
                  rng.uniform(800, 3000, inside.sum()))
        fixed = cv2.GaussianBlur(image, (0, 0), 1.2)
        D = np.array([[6e-3, 3e-3], [-2e-3, 5e-3]])
        matrix = np.eye(3)
        matrix[:2, :2] += D
        matrix[:2, 2] = -D @ CENTRE
        moving = cv2.warpAffine(fixed, matrix[:2], SHAPE[::-1],
                                flags=cv2.INTER_LINEAR)
        task = make(_DataSet(), affine_correction=True)
        errors = {}
        for position in ('centre', 'weighted'):
            task._affineTilePosition = position
            fit = task._affine_tile_fit(task._filter(fixed).astype(np.float32),
                                        task._filter(moving)
                                        .astype(np.float32))
            self.assertTrue(fit['ok'], fit['reason'])
            # the error at the frame corners, in px
            corners = CORNERS - CENTRE
            errors[position] = float(np.abs(
                (fit['D'] - D) @ corners.T).max())
        self.assertLess(errors['weighted'], 0.6 * errors['centre'], errors)
        self.assertLess(errors['weighted'], 0.3, errors)

    def test_centroid_ignores_independent_noise(self):
        # beads in the left third only, each round with its own noise: the
        # signed product's centroid stays on the beads; a product clipped at
        # zero would weight every bead-free pixel and drift to the centre
        rng = np.random.default_rng(4)
        image = np.zeros((300, 300), np.float32)
        x, y = rng.integers(10, 100, 400), rng.integers(10, 290, 400)
        np.add.at(image, (y, x), 2000.0)
        clean = cv2.GaussianBlur(image, (0, 0), 1.2).astype(np.float64)
        energy = clean * clean
        trueX = float(energy.sum(axis=0) @ np.arange(300)) / energy.sum()
        first = clean + rng.normal(0, 20, clean.shape)
        second = clean + rng.normal(0, 20, clean.shape)
        signedX, _ = FiducialPolynomialWarp3D._correlation_centroid(
            first, second)
        clipped = np.maximum(first * second, 0.0)
        clippedX = float(clipped.sum(axis=0) @ np.arange(300)) / clipped.sum()
        self.assertLess(abs(signedX - trueX), 2.0, (signedX, trueX))
        self.assertGreater(abs(clippedX - trueX), 5.0, (clippedX, trueX))
        # no positive correlation weight: no centroid
        self.assertTrue(np.isnan(FiducialPolynomialWarp3D._correlation_centroid(
            first, -second)[0]))


class AffineParameterTests(unittest.TestCase):

    def test_checks(self):
        with self.assertRaises(ValueError):
            make(_DataSet(), affine_correction=1)
        for name in ('affine_polynomial_order', 'affine_min_samples',
                     'affine_tiles', 'affine_min_tiles',
                     'affine_refine_passes'):
            with self.assertRaises(ValueError):
                make(_DataSet(), **{name: -1})
        with self.assertRaises(ValueError):
            make(_DataSet(), affine_tiles=1)

    def test_composite_warp_refuses(self):
        dataSet = _DataSet()
        make(dataSet, affine_correction=True, rotation_task=None)
        with self.assertRaises(ValueError):
            CrossFOVCompositeWarp(dataSet, {'polywarp_task': 'PolyWarp'},
                                  'Composite')._polywarp()


if __name__ == '__main__':
    unittest.main()
