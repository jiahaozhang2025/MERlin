"""Surface-first registration: FiducialRoundRotation measure_on
'fiducial_frame' starts from the 2D fiducial frame's own rigid offset, and
FiducialPolynomialWarp3D rigid_source 'frame' measures its rigid offset on the
2D frame alone (turned back by the round angle), then fits the depth drift
from it.

Synthetic, 3 fovs. Every moving round is the reference turned by its round's
angle (ANGLE_DEG) about the frame centre. Its 2D fiducial frame is moved by
frame_shift; its bead stack by frame_shift + WALK + a small drift with depth,
so the walk into the bead stack (made to overrule the 2D frame here:
rigid_snr_ratio 1, noisy 2D frames, clean bead planes) lands WALK px away from
the frame's own offset. With WALK_FAR that is beyond xy_search_bound, so a
tile fit centred on the walk offset finds no tile on the 2D frame; with
WALK_NEAR the stack's xy is still within xy_search_bound of the frame's
offset, so the depth fit can start from it.

Run with `python -m unittest tests/test_surface_rotation.py`.
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
FOVS = [0, 1, 2]
# data channel -> fiducial stack; 1 and 2 share one
STACK = {0: 'r0', 1: 'r1', 2: 'r1', 3: 'r2'}
ANGLE_DEG = {'r1': 0.25, 'r2': -0.18}
WALK_FAR = (13.0, -12.0)        # (x, y) px, beyond xy_search_bound 10
WALK_NEAR = (6.0, -5.0)         # within it


def beads(seed, n=500):
    rng = np.random.default_rng(seed)
    image = np.zeros(SHAPE, np.float32)
    xy = rng.integers(8, SHAPE[0] - 8, (n, 2))
    np.add.at(image, (xy[:, 1], xy[:, 0]), rng.uniform(800, 3000, n))
    return cv2.GaussianBlur(image, (0, 0), 1.2)


def frame_shift(stack, fov):
    """(tx, ty): displacement of the frame centre in the 2D frame."""
    if stack == 'r1':
        return 15.3 + 0.7 * fov, -12.7 - 0.4 * fov
    return -20.0 + 0.3 * fov, 30.0 - 0.6 * fov


def stack_shift(stack, fov, z, walk):
    """(tx, ty) of the bead stack at depth z: the frame's, plus walk, plus
    a drift with depth."""
    tx, ty = frame_shift(stack, fov)
    return tx + walk[0] + 0.1 * z, ty + walk[1] - 0.05 * z


def rigid_map(stack, shift):
    """output->input map: the content at reference p sits in the moving
    frame at c + R(p - c) + t."""
    theta = np.radians(ANGLE_DEG[stack])
    c, s = np.cos(theta), np.sin(theta)
    cx, cy = CENTRE
    return np.array([[c, -s, cx - c * cx + s * cy + shift[0]],
                     [s, c, cy - s * cx - c * cy + shift[1]], [0, 0, 1.0]])


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
    def __init__(self, walk):
        self.walk = walk
        self.saved = {}
        self.tasks = {}
        self.stacks = {}
        self.twoD = {}
        for fov in FOVS:
            reference = [beads(1000 * fov + k) for k in range(NPLANES)]
            self.stacks['r0_%d' % fov] = [p + 100 for p in reference]
            rng = np.random.default_rng(500 + fov)
            fixed2D = beads(7000 + fov, n=1500)
            self.twoD['r0', fov] = (fixed2D + 100 + rng.normal(
                0, 40, SHAPE).astype(np.float32))
            for stack in ('r1', 'r2'):
                self.stacks['%s_%d' % (stack, fov)] = [
                    moved(reference[k], rigid_map(
                        stack, stack_shift(stack, fov, Z[k], walk))) + 100
                    for k in range(NPLANES)]
                self.twoD[stack, fov] = (
                    moved(fixed2D, rigid_map(stack, frame_shift(stack, fov)))
                    + 100 + rng.normal(0, 40, SHAPE).astype(np.float32))

    @staticmethod
    def _name(task):
        return task if isinstance(task, str) else task.get_analysis_name()

    def get_data_organization(self):
        return _Org()

    def get_fovs(self):
        return FOVS

    def get_fov_offset(self, fov):
        return (280.0 * fov, 0.0)

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


# rigid_snr_ratio 1: the clean bead planes outscore the noisy 2D frames, so
# the walk takes over from the 2D answer (confirmed for WALK_FAR, 'agree'
# for WALK_NEAR)
SHARED = {'highpass_sigma': 3, 'median_filter': False,
          'edge_width_to_remove': 5, 'rigid_plane_search': 8,
          'rigid_plane_step': 4, 'rigid_snr_ratio': 1.0,
          'rigid_search_bound': 80, 'rigid_walk_bound': 15,
          'rigid_walk_confirm': True, 'z_search_range': 2,
          'z_min_score': 0.3, 'minimum_overlap_pixels': 1000,
          'rotation_tiles': 4, 'rotation_min_tiles': 8}


def round_task(dataSet, name='RoundRot', **extra):
    task = FiducialRoundRotation(dataSet, dict(SHARED, **extra), name)
    dataSet.tasks[name] = task
    return task


def poly_task(dataSet, name, **extra):
    parameters = dict(SHARED, rotation_task='RoundRot', z_sample_step=1,
                      xy_sample_step=1, z_polynomial_order=1,
                      xy_polynomial_order=1, z_min_good_samples=3,
                      z_neighbor_min_samples=3, z_xy_refine=True,
                      write_fiducial_images=False,
                      write_aligned_images=False, boundary_smooth=False)
    parameters.update(extra)
    task = FiducialPolynomialWarp3D(dataSet, parameters, name)
    dataSet.tasks[name] = task
    return task


def run_all(task):
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for fov in FOVS:
            task._run_analysis(fov)
        task.finalize()
    return log.getvalue()


def walk_answers(dataSet, fov, owners):
    """The default registration's rigid offsets (with the walk) and its
    rigid_walk records, on unrotated frames."""
    measurer = FiducialPolynomialWarp3D(dataSet, dict(SHARED))
    with contextlib.redirect_stdout(io.StringIO()):
        offsets = measurer._measure_rigid_offsets(fov, dataChannels=owners)
    records = {r['dataChannel']: r
               for r in measurer._rigidWalkRecords.pop(fov)}
    return dict(zip(owners, offsets)), records


def saved_of(dataSet, name):
    return {k[1:]: v for k, v in dataSet.saved.items() if k[0] == name}


class RoundRotationFromTheSurfaceTests(unittest.TestCase):
    """measure_on 'fiducial_frame': surface rigid xy, then the rotation."""

    @classmethod
    def setUpClass(cls):
        cls.dataSet = _DataSet(WALK_FAR)
        cls.task = round_task(cls.dataSet, measure_on='fiducial_frame')
        cls.log = run_all(cls.task)

    def test_the_walk_is_misleading(self):
        # precondition: the default registration's walk overrules the 2D
        # frame by more than xy_search_bound on each axis
        for fov in FOVS:
            offsets, records = walk_answers(self.dataSet, fov, [1, 3])
            for owner in (1, 3):
                stack = STACK[owner]
                self.assertEqual(records[owner]['branch'], 'confirmed')
                walkY, walkX = offsets[owner]
                tx, ty = stack_shift(stack, fov, 0.0, WALK_FAR)
                self.assertAlmostEqual(walkX, -tx, delta=0.6)
                self.assertAlmostEqual(walkY, -ty, delta=0.6)
                self.assertGreater(abs(walkX - records[owner]['x_2d']), 11)
                self.assertGreater(abs(walkY - records[owner]['y_2d']), 10)

    def test_coarse_shift_is_the_frames_own_offset(self):
        for fov in FOVS:
            _, records = walk_answers(self.dataSet, fov, [1, 3])
            walk = self.dataSet.load_dataframe_from_csv(
                'rigid_walk', 'RoundRot', fov).set_index('dataChannel')
            table = self.task.get_first_plane_rotation(fov).set_index(
                'owner_channel')
            for owner in (1, 3):
                # exactly the registration's 2D answer, no walk
                self.assertEqual(walk.loc[owner, 'branch'], 'frame')
                self.assertEqual(table.loc[owner, 'coarse_y'],
                                 records[owner]['y_2d'])
                self.assertEqual(table.loc[owner, 'coarse_x'],
                                 records[owner]['x_2d'])
                self.assertEqual(walk.loc[owner, 'rigid_y'],
                                 walk.loc[owner, 'y_2d'])
                self.assertEqual(walk.loc[owner, 'rigid_x'],
                                 walk.loc[owner, 'x_2d'])
                tx, ty = frame_shift(STACK[owner], fov)
                self.assertAlmostEqual(table.loc[owner, 'coarse_x'], -tx,
                                       delta=0.3)
                self.assertAlmostEqual(table.loc[owner, 'coarse_y'], -ty,
                                       delta=0.3)
        self.assertNotIn('rigid offset taken from plane', self.log)

    def test_rotation_recovered_from_that_shift(self):
        for fov in FOVS:
            table = self.task.get_first_plane_rotation(fov)
            for row in table.itertuples():
                stack = STACK[row.owner_channel]
                self.assertTrue(row.trusted, (fov, stack, row.reason))
                self.assertEqual(row.plane_index, -1)
                self.assertAlmostEqual(row.angle_deg, ANGLE_DEG[stack],
                                       delta=0.01, msg=(fov, stack))
                tx, ty = frame_shift(stack, fov)
                self.assertAlmostEqual(row.xshift, tx, delta=0.1)
                self.assertAlmostEqual(row.yshift, ty, delta=0.1)
        table = self.task.get_round_rotation().set_index('owner_channel')
        for owner in (1, 3):
            self.assertAlmostEqual(table.loc[owner, 'angle_deg'],
                                   ANGLE_DEG[STACK[owner]], delta=0.0075)
            self.assertEqual(table.loc[owner, 'n_fovs'], 3)

    def test_a_coarse_shift_from_the_walk_would_lose_the_fov(self):
        # why the frame's own offset: centred on the walk offset, the tiles'
        # +-xy_search_bound window does not reach the 2D frame's offset
        for fov in FOVS:
            offsets, _ = walk_answers(self.dataSet, fov, [1])
            row = self.task._measure_frame(fov, [1, 2], *offsets[1])
            self.assertFalse(row['trusted'])
            self.assertIn('tiles usable', row['reason'])

    def test_stack_plane_mode_keeps_the_walk(self):
        # the default measure_on is unchanged: its coarse shift is the
        # registration's rigid offset with the walk
        stackTask = round_task(self.dataSet, name='StackRot')
        with contextlib.redirect_stdout(io.StringIO()):
            stackTask._run_analysis(0)
        offsets, _ = walk_answers(self.dataSet, 0, [1, 3])
        table = stackTask.get_first_plane_rotation(0).set_index(
            'owner_channel')
        walk = self.dataSet.load_dataframe_from_csv(
            'rigid_walk', 'StackRot', 0).set_index('dataChannel')
        for owner in (1, 3):
            self.assertEqual(walk.loc[owner, 'branch'], 'confirmed')
            self.assertEqual((table.loc[owner, 'coarse_y'],
                              table.loc[owner, 'coarse_x']), offsets[owner])
            self.assertTrue(table.loc[owner, 'trusted'])
            self.assertAlmostEqual(table.loc[owner, 'angle_deg'],
                                   ANGLE_DEG[STACK[owner]], delta=0.01)


class RigidSourceFrameTests(unittest.TestCase):
    """FiducialPolynomialWarp3D rigid_source 'frame': rigid xy again on the
    turned-back 2D frame, then the depth fit from it."""

    @classmethod
    def setUpClass(cls):
        cls.dataSet = _DataSet(WALK_NEAR)
        cls.rounds = round_task(cls.dataSet, measure_on='fiducial_frame')
        cls.roundLog = run_all(cls.rounds)
        cls.walk = poly_task(cls.dataSet, 'PolyWalk')
        cls.walkLog = run_all(cls.walk)
        cls.frame = poly_task(cls.dataSet, 'PolyFrame',
                              rigid_source='frame')
        cls.frameLog = run_all(cls.frame)

    def test_parameter(self):
        self.assertEqual(FiducialPolynomialWarp3D(
            _DataSet(WALK_NEAR)).parameters['rigid_source'],
            'frame_and_walk')
        self.assertEqual(self.frame.parameters['rigid_source'], 'frame')
        for value in ('walk', 'stack', None):
            with self.assertRaises(ValueError):
                FiducialPolynomialWarp3D(_DataSet(WALK_NEAR),
                                         {'rigid_source': value})

    def test_explicit_frame_and_walk_is_the_default(self):
        explicit = poly_task(self.dataSet, 'PolyWalkExplicit',
                             rigid_source='frame_and_walk')
        log = run_all(explicit)
        self.assertEqual(log.replace('PolyWalkExplicit', 'PolyWalk'),
                         self.walkLog)
        mine = saved_of(self.dataSet, 'PolyWalkExplicit')
        default = saved_of(self.dataSet, 'PolyWalk')
        self.assertEqual(mine.keys(), default.keys())
        for key in mine:
            if isinstance(mine[key], np.ndarray):
                np.testing.assert_array_equal(mine[key], default[key], key)
            else:
                self.assertEqual(mine[key], default[key], key)

    def test_frame_is_the_2d_frame_alone(self):
        # rigid_source 'frame' = the walk switched off: every output equals
        # rigid_plane_search 0's, except the rigid_walk branch label
        noWalk = poly_task(self.dataSet, 'PolyNoWalk', rigid_plane_search=0)
        run_all(noWalk)
        frame = saved_of(self.dataSet, 'PolyFrame')
        off = saved_of(self.dataSet, 'PolyNoWalk')
        self.assertEqual(frame.keys(), off.keys())
        for key in frame:
            if key[0] == 'rigid_walk':
                a = pd.read_csv(io.StringIO(frame[key]))
                b = pd.read_csv(io.StringIO(off[key]))
                self.assertTrue((a.branch == 'frame').all())
                self.assertTrue((b.branch == 'off').all())
                pd.testing.assert_frame_equal(a.drop(columns='branch'),
                                              b.drop(columns='branch'))
            elif isinstance(frame[key], np.ndarray):
                np.testing.assert_array_equal(frame[key], off[key], key)
            else:
                self.assertEqual(frame[key], off[key], key)

    def test_rigid_offset_is_the_surface_frames(self):
        angles = self.rounds.get_round_angles()
        for fov in FOVS:
            walk = self.dataSet.load_dataframe_from_csv(
                'rigid_walk', 'PolyFrame', fov).set_index('dataChannel')
            coeff = self.frame.get_polynomial_coefficients(fov).set_index(
                'dataChannel')
            walkCoeff = self.walk.get_polynomial_coefficients(
                fov).set_index('dataChannel')
            for dataChannel in (1, 2, 3):
                stack = STACK[dataChannel]
                self.assertEqual(walk.loc[dataChannel, 'branch'], 'frame')
                # measured on the 2D frame turned back by the round angle,
                # stored in the moving frame
                theta = angles[dataChannel]
                c, s = np.cos(theta), np.sin(theta)
                wx, wy = walk.loc[dataChannel, ['x_2d', 'y_2d']]
                self.assertAlmostEqual(coeff.loc[dataChannel, 'rigid_x'],
                                       c * wx - s * wy, places=9)
                self.assertAlmostEqual(coeff.loc[dataChannel, 'rigid_y'],
                                       s * wx + c * wy, places=9)
                tx, ty = frame_shift(stack, fov)
                self.assertAlmostEqual(coeff.loc[dataChannel, 'rigid_x'],
                                       -tx, delta=0.3)
                self.assertAlmostEqual(coeff.loc[dataChannel, 'rigid_y'],
                                       -ty, delta=0.3)
                # the default takes the walk's, WALK_NEAR away
                tx, ty = stack_shift(stack, fov, 0.0, WALK_NEAR)
                self.assertAlmostEqual(
                    walkCoeff.loc[dataChannel, 'rigid_x'], -tx, delta=0.6)
                self.assertAlmostEqual(
                    walkCoeff.loc[dataChannel, 'rigid_y'], -ty, delta=0.6)

    def test_depth_fit_from_the_surface_offset(self):
        # the stack sits WALK_NEAR (+ drift) from the surface offset, inside
        # the xy search window: the xy polynomial finds it, and z_xy_refine
        # re-measures z at each depth's own xy
        corners = np.array([[30, 30], [353, 30], [30, 353], [353, 353],
                            CENTRE])
        for task in (self.frame, self.walk):
            worst = 0.0
            for fov in FOVS:
                status = task.get_z_fit_status(fov).set_index('dataChannel')
                table = task.get_transformation_table(fov)
                for dataChannel in (1, 2, 3):
                    stack = STACK[dataChannel]
                    self.assertTrue(bool(status.loc[dataChannel, 'fitted']))
                    self.assertEqual(status.loc[dataChannel, 'n_good'],
                                     status.loc[dataChannel, 'n_samples'])
                    np.testing.assert_allclose(
                        table[table.dataChannel == dataChannel].z_offset,
                        0.0, atol=0.1)
                    for zIndex in (2, 5, 9):
                        fitted = task.get_transformation(fov, dataChannel,
                                                         zIndex)
                        error = np.linalg.norm(
                            transform.matrix_transform(corners, fitted.params)
                            - transform.matrix_transform(
                                corners, rigid_map(stack, stack_shift(
                                    stack, fov, Z[zIndex], WALK_NEAR))),
                            axis=1)
                        worst = max(worst, error.max())
            self.assertLess(worst, 0.3, task.get_analysis_name())
        # the first z pass, at the surface offset WALK_NEAR off the stack,
        # is what z_xy_refine repairs
        status = self.frame.get_z_fit_status(0).set_index('dataChannel')
        self.assertTrue(bool(status.loc[1, 'z_refined']))
        self.assertLess(status.loc[1, 'n_good_first'],
                        status.loc[1, 'n_good'])

    def test_without_rotation_task(self):
        # no round angle: the 2D frame's offset, unrotated
        task = poly_task(self.dataSet, 'PolyPlainFrame', rotation_task=None,
                         rigid_source='frame')
        with contextlib.redirect_stdout(io.StringIO()):
            offsets = task._measure_rigid_offsets(0)
        _, records = walk_answers(self.dataSet, 0, [0, 1, 2, 3])
        for dataChannel, (y, x) in zip(range(4), offsets):
            self.assertEqual((y, x), (records[dataChannel]['y_2d'],
                                      records[dataChannel]['x_2d']))


if __name__ == '__main__':
    unittest.main()
