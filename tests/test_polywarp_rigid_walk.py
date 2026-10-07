"""FiducialPolynomialWarp3D rigid_walk_confirm: the rigid walk's ghost peak.

The rigid walk searches its FIRST plane wide (+-rigid_search_bound about
zero) so it can overrule a bad 2D frame, then every later plane only
+-rigid_walk_bound about the running best. On 20260824 fov14 plane 0 of the
bead stack has a secondary peak, hundreds of px from the true one, whose snr
is marginally higher; it beats rigid_snr_ratio x the 2D snr, so the walk
starts there and nothing corrects it. rigid_walk_confirm accepts a first-plane
answer far from the 2D answer only if a wide search on the next walk plane
agrees with it.

Synthetic stacks, the task's real measurement code (nothing stubbed):

channel 0  reference
channel 1  GHOST: plane 0 holds the true texture (weight 0.45) and the same
           texture at a ghost offset (weight 0.55); every other plane and the
           2D frame are clean at the true offset (fov14's signature)
channel 2  BAD 2D: the 2D frame is 50 px off; every plane is right, so the
           walk must still overrule the 2D answer (0609 fov52's case)
channel 3  normal
channel 4  DEEP GHOST: planes 0 and the 2D frame are right, planes 4 and 8
           agree on a ghost (20260824 fov18): agreement of two deeper planes
           is not evidence, so the check starts from plane 0

Run with `python -m unittest tests/test_polywarp_rigid_walk.py`.
"""
import contextlib
import io
import unittest

import cv2
import numpy as np
import pandas as pd

from merlin.analysis.warp import FiducialPolynomialWarp3D

SIZE = 256
NPLANES = 16
Z = np.arange(NPLANES) * 0.5
TRUE = {0: (0.0, 0.0), 1: (20.3, -14.6), 2: (-12.4, 25.8), 3: (8.6, 9.1),
        4: (-17.2, -6.3)}
GHOST = (-30.0, 40.0)             # channel 1, plane 0
DEEP_GHOST = (55.0, -40.0)        # channel 4, planes 4 and 8
BAD_2D = (40.0, -30.0)            # channel 2: added to the 2D frame's offset


def beads(seed, n=500):
    rng = np.random.default_rng(seed)
    image = np.zeros((SIZE, SIZE), np.float32)
    xy = rng.integers(6, SIZE - 6, (n, 2))
    np.add.at(image, (xy[:, 1], xy[:, 0]), rng.uniform(800, 3000, n))
    return cv2.GaussianBlur(image, (0, 0), 1.2)


def at(image, offset):
    """image moved so that the measured shift (moving -> fixed) is offset."""
    return FiducialPolynomialWarp3D._shift_image(image, -offset[0],
                                                 -offset[1])


def noisy(image, seed, sigma=100.0):
    rng = np.random.default_rng(seed)
    return image + rng.normal(0, sigma, image.shape).astype(np.float32)


FIXED = [beads(100 + k) for k in range(NPLANES)]
FIXED2D = beads(7, n=250)


def moving_plane(dataChannel, k):
    if dataChannel == 1 and k == 0:
        return 0.45 * at(FIXED[0], TRUE[1]) + 0.55 * at(FIXED[0], GHOST)
    if dataChannel == 4 and k in (4, 8):
        return at(FIXED[k], DEEP_GHOST)
    return at(FIXED[k], TRUE[dataChannel])


STACKS = {c: [(FIXED[k] if c == 0 else moving_plane(c, k)) + 100
              for k in range(NPLANES)] for c in TRUE}


def moving_2d(dataChannel):
    offset = TRUE[dataChannel]
    if dataChannel == 2:
        offset = (offset[0] + BAD_2D[0], offset[1] + BAD_2D[1])
    return noisy(at(FIXED2D, offset), 50 + dataChannel) + 100


TWO_D = {c: (noisy(FIXED2D, 49) + 100 if c == 0 else moving_2d(c))
         for c in TRUE}


class _Org:
    def get_data_channels(self):
        return np.array(sorted(TRUE))

    def get_fiducial3D_filename(self, dataChannel, fov):
        return dataChannel

    def get_fiducial3D_stack_frame_indices(self, dataChannel):
        return np.arange(NPLANES)

    def get_fiducial3D_stack_frame_zPos(self, dataChannel):
        return Z

    def get_data_channel_color(self, dataChannel):
        return '650'


class _DataSet:
    def __init__(self):
        self.saved = {}

    def get_data_organization(self):
        return _Org()

    def get_fovs(self):
        return [0]

    def get_z_positions(self):
        return list(Z)

    def get_image_dimensions(self):
        return (SIZE, SIZE)

    def get_fiducial_image(self, dataChannel, fov):
        return TWO_D[dataChannel]

    def load_image(self, name, frame):
        return STACKS[name][int(frame)]

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


PARAMETERS = {
    'highpass_sigma': 3, 'median_filter': False, 'edge_width_to_remove': 8,
    'rigid_search_bound': 100, 'rigid_walk_bound': 15,
    'rigid_plane_search': 12, 'rigid_plane_step': 4, 'rigid_snr_ratio': 1.5,
    'z_search_range': 2, 'z_sample_step': 3, 'xy_sample_step': 3,
    'z_min_good_samples': 2, 'minimum_overlap_pixels': 1000,
    'z_neighbor_substitution': False,
    'write_fiducial_images': False, 'write_aligned_images': False}


def task_for(confirm):
    parameters = dict(PARAMETERS)
    if confirm is not None:
        parameters['rigid_walk_confirm'] = confirm
    return FiducialPolynomialWarp3D(_DataSet(), parameters, 'PolyWarp')


def rigid(confirm):
    task = task_for(confirm)
    with contextlib.redirect_stdout(io.StringIO()):
        offsets = task._measure_rigid_offsets(0)
    return task, dict(zip(sorted(TRUE), offsets))


class RigidWalkConfirmTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.onTask, cls.on = rigid(True)
        _, cls.off = rigid(False)
        _, cls.default = rigid(None)
        cls.records = {r['dataChannel']: r
                       for r in cls.onTask._rigidWalkRecords[0]}

    def assertNear(self, offset, expected, tolerance=1.0):
        self.assertLess(np.hypot(offset[0] - expected[0],
                                 offset[1] - expected[1]), tolerance,
                        (offset, expected))

    def test_premise_ghost_beats_2d(self):
        # the synthetic case reproduces fov14: plane 0's wide peak is the
        # ghost, and its snr beats rigid_snr_ratio x the 2D snr
        r = self.records[1]
        self.assertNear((r['y_first'], r['x_first']), GHOST)
        self.assertGreater(r['snr_first'], 1.5 * r['snr_2d'])
        self.assertNear((r['y_2d'], r['x_2d']), TRUE[1])

    def test_default_is_off_and_takes_the_ghost(self):
        self.assertFalse(task_for(None).parameters['rigid_walk_confirm'])
        self.assertEqual(self.off, self.default)
        self.assertNear(self.off[1], GHOST)          # the old behaviour

    def test_ghost_rejected(self):
        self.assertNear(self.on[1], TRUE[1])
        r = self.records[1]
        self.assertEqual(r['branch'], 'rejected')
        self.assertEqual(r['confirm_plane'], 4)
        self.assertNear((r['y_confirm'], r['x_confirm']), TRUE[1])

    def test_bad_2d_still_overruled(self):
        r = self.records[2]
        self.assertEqual(r['branch'], 'confirmed')
        self.assertGreater(np.hypot(r['y_2d'] - TRUE[2][0],
                                    r['x_2d'] - TRUE[2][1]), 40)
        self.assertNear(self.on[2], TRUE[2])
        self.assertEqual(self.on[2], self.off[2])

    def test_deep_ghost_needs_plane_0(self):
        # planes 4 and 8 agree on a ghost; plane 0 agrees with the 2D frame,
        # so the walk is the old one and keeps the true offset
        r = self.records[4]
        self.assertEqual(r['branch'], 'agree')
        self.assertNear(self.on[4], TRUE[4])
        self.assertEqual(self.on[4], self.off[4])

    def test_unaffected_channels_identical(self):
        for dataChannel in (0, 3):
            self.assertEqual(self.on[dataChannel], self.off[dataChannel])
            self.assertNear(self.on[dataChannel], TRUE[dataChannel])
        self.assertEqual(self.records[3]['branch'], 'agree')

    def test_missing_confirm_plane_rejects(self):
        task = task_for(True)
        record = {}
        self.assertFalse(task._confirm_first_walk_plane(
            0, 2, NPLANES, *TRUE[2], record))
        self.assertEqual(record, {})

    def test_option_must_be_boolean(self):
        with self.assertRaises(ValueError):
            FiducialPolynomialWarp3D(_DataSet(), {'rigid_walk_confirm': 1},
                                     'PolyWarp')


class RigidWalkTableTests(unittest.TestCase):
    """The branch is written per channel, and far first planes are flagged,
    only with the option on."""

    @staticmethod
    def run_fov(confirm):
        task = task_for(confirm)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            task._run_analysis(0)
        return task, out.getvalue()

    def test_table_and_flags(self):
        task, log = self.run_fov(True)
        walk = task.get_rigid_walk(0).set_index('dataChannel')
        self.assertEqual(walk.loc[1, 'branch'], 'rejected')
        self.assertEqual(walk.loc[2, 'branch'], 'confirmed')
        self.assertEqual(set(walk.loc[[3, 4], 'branch']), {'agree'})
        # the reference's 2D frame correlates with itself: no walk
        self.assertEqual(walk.loc[0, 'branch'], '2d')
        coeff = task.get_polynomial_coefficients(0).set_index('dataChannel')
        np.testing.assert_allclose(walk.rigid_y, coeff.loc[walk.index,
                                                           'rigid_y'])
        self.assertLess(np.hypot(coeff.loc[1, 'rigid_y'] - TRUE[1][0],
                                 coeff.loc[1, 'rigid_x'] - TRUE[1][1]), 1.0)
        flags = task.dataSet.load_dataframe_from_csv('fit_quality_flags',
                                                     resultIndex=0)
        walkFlags = flags[flags.reasons.str.contains('rigid walk')]
        self.assertEqual(sorted(walkFlags.dataChannel), [1, 2])
        self.assertIn('NOT confirmed', walkFlags[
            walkFlags.dataChannel == 1].reasons.iloc[0])
        self.assertIn('rigid walk REJECTED', log)
        self.assertIn('rigid walk CONFIRMED', log)

    def test_nothing_written_when_off(self):
        task, log = self.run_fov(False)
        self.assertNotIn(('rigid_walk', 0), task.dataSet.saved)
        flags = task.dataSet.load_dataframe_from_csv('fit_quality_flags',
                                                     resultIndex=0)
        self.assertFalse(flags.reasons.astype(str)
                         .str.contains('rigid walk').any())
        self.assertNotIn('rigid walk', log)


if __name__ == '__main__':
    unittest.main()
