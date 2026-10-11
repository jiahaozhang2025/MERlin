"""Camera hot-pixel map (HotPixelMap), its replacement, and the correction
in the warp / preprocess image path.

- the candidate rule and its median filter (OpenCV and scipy agree, border
  replicated);
- HotPixelReplacer: an isolated hot pixel takes the median of its 8
  neighbours, a pair or a block uses only the non-hot ones (a 3 x 3 block
  falls back to 5 x 5), frame edges and corners, dtype and rounding, every
  other pixel untouched;
- HotPixelMap on a synthetic dataset whose frames carry fixed hot pixels in
  every channel, a warm pixel below the threshold, a bright feature fixed in
  ONE channel only and a defect seen only by a second camera: exactly the
  defects are hot, the one-channel feature is a candidate but not hot, the
  outputs load back, camera_colors keeps the cameras apart;
- the warps (FiducialPolynomialWarp3D with a z blend and a turned table,
  FiducialCorrelationWarp with and without register_3d): synthetic hot pixels
  in the moving frames are gone after the warp, real one-pixel structure in
  one channel is kept, and without hotPixels (or with an empty map) the image
  is bit for bit the uncorrected one;
- DeconvolutionPreprocess hot_pixel_task: dependency, the corrected image,
  no new key in the saved parameters when off, and a clear error for a warp
  task that cannot take the correction (and from DeconvolutionPreprocessDW,
  which reads its frames another way).

Run with `python -m unittest tests/test_hotpixel.py`.
"""
import contextlib
import io
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import transform

from merlin.analysis import hotpixel as hotpixeltask
from merlin.analysis import preprocess
from merlin.analysis.warp import FiducialCorrelationWarp
from merlin.analysis.warp import FiducialPolynomialWarp3D
from merlin.core import analysistask
from merlin.core import dataset
from merlin.util import hotpixel


# ------------------------------------------------------------------ rule

class CandidateRule(unittest.TestCase):

    def test_median_filter_matches_scipy_with_replicated_border(self):
        rng = np.random.default_rng(0)
        image = rng.normal(100, 20, (37, 41)).astype(np.float32)
        for size in (3, 5, 7):
            np.testing.assert_array_equal(
                hotpixel.median_filtered(image, size),
                ndimage.median_filter(image, size=size, mode='nearest'))
        with self.assertRaises(ValueError):
            hotpixel.median_filtered(image, 4)

    def test_candidate_is_strictly_above_the_threshold(self):
        frame = np.full((20, 20), 100.0, np.float32)
        frame[5, 5] = 150.0          # excess exactly 50: not a candidate
        frame[10, 10] = 150.5
        mask, excess = hotpixel.candidate_map(frame, 50.0, 5)
        self.assertEqual(excess[5, 5], 50.0)
        self.assertFalse(mask[5, 5])
        self.assertTrue(mask[10, 10])
        self.assertEqual(int(mask.sum()), 1)


# ------------------------------------------------------------- replacer

class Replacement(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.image = rng.integers(90, 140, (12, 14)).astype(np.uint16)

    def _replaced(self, hot, maxRadius=2, image=None):
        image = self.image if image is None else image
        mask = np.zeros(image.shape, bool)
        for y, x in hot:
            mask[y, x] = True
        replacer = hotpixel.HotPixelReplacer(mask, maxRadius)
        return replacer, replacer.replace(image)

    def test_isolated_pixel_takes_the_median_of_its_neighbours(self):
        image = self.image.copy()
        image[5, 6] = 4000
        replacer, out = self._replaced([(5, 6)], image=image)
        window = image[4:7, 5:8].astype(float).ravel()
        expected = np.rint(np.median(np.delete(window, 4)))
        self.assertEqual(out[5, 6], expected)
        self.assertEqual(out.dtype, np.uint16)
        changed = out != image
        changed[5, 6] = False
        self.assertFalse(changed.any())
        self.assertEqual(replacer.radii.tolist(), [1])
        self.assertEqual(replacer.neighbourCounts.tolist(), [8])

    def test_pair_uses_only_non_hot_neighbours(self):
        image = self.image.copy()
        image[3, 3] = image[3, 4] = 5000
        _, out = self._replaced([(3, 3), (3, 4)], image=image)
        for y, x in ((3, 3), (3, 4)):
            window = [(yy, xx) for yy in range(y - 1, y + 2)
                      for xx in range(x - 1, x + 2)
                      if (yy, xx) not in ((3, 3), (3, 4))]
            values = [float(image[p]) for p in window]
            self.assertEqual(len(values), 7)
            self.assertEqual(out[y, x], np.rint(np.median(values)))

    def test_edges_and_corners(self):
        image = self.image.copy()
        image[0, 0] = image[0, 7] = image[11, 13] = 6000
        replacer, out = self._replaced([(0, 0), (0, 7), (11, 13)],
                                       image=image)
        self.assertEqual(replacer.neighbourCounts.tolist(), [3, 5, 3])
        self.assertEqual(out[0, 0], np.rint(np.median(
            [image[0, 1], image[1, 0], image[1, 1]])))

    def test_block_falls_back_to_the_wider_window(self):
        hot = [(y, x) for y in range(4, 7) for x in range(5, 8)]
        image = self.image.copy()
        for p in hot:
            image[p] = 7000
        replacer, out = self._replaced(hot, image=image)
        centre = hot.index((5, 6))
        self.assertEqual(replacer.radii[centre], 2)
        ring = [image[y, x] for y in range(3, 8) for x in range(4, 9)
                if (y, x) not in hot]
        self.assertEqual(len(ring), 16)
        self.assertEqual(out[5, 6], np.rint(np.median(np.array(ring, float))))
        self.assertTrue((out[4:7, 5:8] < 1000).all())
        # with the 3 x 3 window only, the centre has no non-hot neighbour
        replacer, out = self._replaced(hot, maxRadius=1, image=image)
        self.assertEqual(replacer.unresolved_count, 1)
        self.assertEqual(out[5, 6], 7000)

    def test_rounding_clipping_and_float_images(self):
        image = np.zeros((5, 5), np.uint16)
        image[2, 2] = 9
        image[1, 1:4] = [1, 2, 2]
        image[2, 1] = image[2, 3] = 3
        image[3, 1:4] = [3, 4, 4]
        # neighbours 1 2 2 3 3 3 4 4: median 3
        _, out = self._replaced([(2, 2)], image=image)
        self.assertEqual(out[2, 2], 3)
        floats = image.astype(np.float32)
        floats[3, 3] = 5.0           # neighbours 1 2 2 3 3 3 4 5: 3.0
        floats[1, 1] = 0.5           # 0.5 2 2 3 3 3 4 5: 3.0
        floats[1, 2] = 2.25          # 0.5 2 2.25 3 3 3 4 5: 3.0
        floats[2, 1] = 2.5           # 0.5 2 2.25 2.5 3 3 4 5: 2.75
        _, out = self._replaced([(2, 2)], image=floats)
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out[2, 2]), 2.75)

    def test_no_hot_pixels_returns_the_same_image(self):
        replacer = hotpixel.HotPixelReplacer(np.zeros((12, 14), bool))
        self.assertIs(replacer.replace(self.image), self.image)

    def test_views_and_shape_check(self):
        image = self.image.copy()
        image[2, 9] = 3000
        view = image.T[::-1]           # what DataSet.load_image can return
        mask = np.zeros(view.shape, bool)
        y, x = 14 - 1 - 9, 2           # (2, 9) in the transposed, flipped view
        self.assertEqual(view[y, x], 3000)
        mask[y, x] = True
        out = hotpixel.HotPixelReplacer(mask).replace(view)
        self.assertLess(out[y, x], 200)
        np.testing.assert_array_equal(np.delete(out.ravel(), y * 12 + x),
                                      np.delete(view.ravel(), y * 12 + x))
        with self.assertRaises(ValueError):
            hotpixel.HotPixelReplacer(mask).replace(self.image)

    def test_correction_maps_channels_to_cameras(self):
        maskA = np.zeros((12, 14), bool)
        maskA[1, 1] = True
        maskB = np.zeros((12, 14), bool)
        maskB[8, 8] = True
        correction = hotpixel.HotPixelCorrection({0: maskA, 1: maskB},
                                                 {0: 0, 1: 1, 2: 1})
        image = self.image.copy()
        image[1, 1] = image[8, 8] = 5000
        a = correction.correct(image, 0)
        b = correction.correct(image, 2)
        self.assertLess(a[1, 1], 200)
        self.assertEqual(a[8, 8], 5000)
        self.assertEqual(b[1, 1], 5000)
        self.assertLess(b[8, 8], 200)
        with self.assertRaises(KeyError):
            correction.correct(image, 7)
        with self.assertRaises(ValueError):
            hotpixel.HotPixelCorrection({0: maskA}, {0: 0, 1: 1})


# --------------------------------------------------- synthetic dataset

SIZE = 64
FOVS = [0, 1, 2, 3, 4, 5]
Z = [1.0 + 0.5 * k for k in range(8)]
CHANNELS = [('Bit1', '650'), ('Bit2', '560'), ('Bit3', '650'),
            ('DAPI', '405')]
# camera defects: (y, x) -> excess over the frame, in every channel
HOT = {(10, 20): 900, (40, 41): 150, (40, 42): 140, (0, 5): 400,
       (63, 63): 3000, (25, 50): 60000}
WARM = (30, 30)                 # +30: below the 50 ADU threshold
ONE_CHANNEL = (50, 15)          # +400 in channel 2 only (fixed structure)
SECOND_CAMERA = (52, 40)        # +500 in the 560 and 405 channels only


def raw_frame(dataChannel, fov, zPosition):
    """Background, noise and tissue that change with fov and plane, plus the
    fixed features above, in the orientation get_raw_image returns."""
    seed = 1000 * fov + 10 * dataChannel + int(round(zPosition * 2))
    rng = np.random.default_rng(seed)
    image = 100.0 + rng.normal(0, 3, (SIZE, SIZE))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for y0, x0, a in zip(rng.uniform(0, SIZE, 25), rng.uniform(0, SIZE, 25),
                         rng.uniform(200, 800, 25)):
        image += a * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / 3.0)
    for (y, x), excess in HOT.items():
        image[y, x] += excess
    image[WARM] += 30
    if dataChannel == 2:
        image[ONE_CHANNEL] += 400
    if CHANNELS[dataChannel][1] in ('560', '405'):
        image[SECOND_CAMERA] += 500
    return np.clip(np.rint(image), 0, 65535).astype(np.uint16)


class _Organization:
    def get_data_channels(self):
        return np.arange(len(CHANNELS))

    def get_data_channel_name(self, dataChannel):
        return CHANNELS[dataChannel][0]

    def get_data_channel_color(self, dataChannel):
        return CHANNELS[dataChannel][1]


class _DataSet(dataset.DataSet):
    def __init__(self, root, frames=raw_frame):
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__('synthetic', dataHome=root, analysisHome=root)
        self.frames = frames
        self.reads = 0

    def get_data_organization(self):
        return _Organization()

    def get_fovs(self):
        return FOVS

    def get_z_positions(self):
        return Z

    def z_index_to_position(self, zIndex):
        return Z[zIndex]

    def get_image_dimensions(self):
        return (SIZE, SIZE)

    def get_raw_image(self, dataChannel, fov, zPosition):
        self.reads += 1
        return self.frames(dataChannel, fov, zPosition)


def run_map(ds, name='HotPix', **parameters):
    parameters.setdefault('sample_fovs', 6)
    parameters.setdefault('sample_planes', 4)
    parameters.setdefault('threads', 2)
    task = hotpixeltask.HotPixelMap(ds, parameters, name)
    task.save()
    with contextlib.redirect_stdout(io.StringIO()):
        task.run()
    return ds.load_analysis_task(name)


class HotPixelMapTask(unittest.TestCase):

    def test_finds_exactly_the_camera_defects(self):
        with tempfile.TemporaryDirectory() as root:
            ds = _DataSet(root)
            task = run_map(ds)
            self.assertEqual(ds.reads, 4 * 6 * 4)
            hot = task.get_hot_pixels()
            found = set(zip(hot['y'], hot['x']))
            self.assertEqual(found, set(HOT) | {SECOND_CAMERA})
            byPixel = hot.set_index(['y', 'x'])
            for p in HOT:
                self.assertEqual(byPixel.loc[p, 'n_votes'], 4)
            self.assertEqual(byPixel.loc[SECOND_CAMERA, 'n_votes'], 2)
            self.assertTrue((hot['n_sampled_channels'] == 4).all())
            self.assertTrue((hot['replace_radius'] == 1).all())
            self.assertGreater(byPixel.loc[(10, 20), 'excess_min'], 800)

            candidates = task.get_candidates()
            one = candidates[(candidates['y'] == ONE_CHANNEL[0])
                             & (candidates['x'] == ONE_CHANNEL[1])]
            self.assertEqual(one['data_channel'].tolist(), [2])
            self.assertFalse(one['hot'].any())
            self.assertFalse(((candidates['y'] == WARM[0])
                              & (candidates['x'] == WARM[1])).any())
            self.assertEqual(int(candidates['hot'].sum()),
                             4 * len(HOT) + 2)

            votes = task.get_vote_count()
            self.assertEqual(votes.shape, (1, SIZE, SIZE))
            self.assertEqual(votes.dtype, np.uint16)
            self.assertEqual(votes[0][ONE_CHANNEL], 1)
            self.assertEqual(int((votes[0] >= 2).sum()), len(found))

            counts = task.get_channel_counts()
            self.assertEqual(counts['data_channel'].tolist(), [0, 1, 2, 3])
            self.assertTrue(counts['sampled'].all())
            self.assertTrue((counts['n_frames'] == 24).all())
            self.assertTrue((counts['n_hot_camera'] == len(found)).all())
            self.assertEqual(
                counts.set_index('data_channel').loc[2, 'n_candidates'],
                len(found) - 1 + 1)        # misses SECOND_CAMERA, adds own
            summary = task.get_summary()
            self.assertEqual(summary['sample_fovs'], FOVS)
            self.assertEqual(summary['cameras']['0']['n_hot'], len(found))
            self.assertEqual(summary['cameras']['0']['n_unresolved'], 0)

            correction = task.get_correction()
            for dataChannel in range(4):
                frame = raw_frame(dataChannel, 0, Z[3])
                fixed = correction.correct(frame, dataChannel)
                for p in found:
                    self.assertLess(fixed[p], 1200 if p != (25, 50) else 2000,
                                    (dataChannel, p))
                if dataChannel == 2:
                    self.assertEqual(fixed[ONE_CHANNEL], frame[ONE_CHANNEL])
                keep = np.ones(frame.shape, bool)
                for p in found:
                    keep[p] = False
                np.testing.assert_array_equal(fixed[keep], frame[keep])

    def test_cameras_are_kept_apart(self):
        with tempfile.TemporaryDirectory() as root:
            ds = _DataSet(root)
            task = run_map(ds, camera_colors=[['650'], ['560', '405']])
            self.assertEqual(task.get_channel_cameras(),
                             {0: 0, 1: 1, 2: 0, 3: 1})
            hot = task.get_hot_pixels()
            for camera, expected in ((0, set(HOT)),
                                     (1, set(HOT) | {SECOND_CAMERA})):
                rows = hot[hot['camera'] == camera]
                self.assertEqual(set(zip(rows['y'], rows['x'])), expected)
                self.assertTrue((rows['n_sampled_channels'] == 2).all())
            correction = task.get_correction()
            frame = raw_frame(0, 1, Z[2])
            self.assertEqual(correction.correct(frame, 0)[SECOND_CAMERA],
                             frame[SECOND_CAMERA])
            frame = raw_frame(1, 1, Z[2])
            self.assertLess(correction.correct(frame, 1)[SECOND_CAMERA], 300)
            self.assertEqual(task.get_vote_count().shape, (2, SIZE, SIZE))

    def test_sampling_options_and_checks(self):
        self.assertEqual(hotpixeltask._evenly_spaced(list(range(10)), 4),
                         [0, 3, 6, 9])
        self.assertEqual(hotpixeltask._evenly_spaced([5, 6], 4), [5, 6])
        self.assertEqual(hotpixeltask._evenly_spaced(list(range(9)), 1), [4])
        with tempfile.TemporaryDirectory() as root:
            ds = _DataSet(root)
            # 3 fovs x 3 planes: an odd number of frames, so the median is
            # one frame's value and two-frame tissue cannot reach it
            task = run_map(ds, sample_fovs=[1, 3, 4], sample_planes=[0, 3, 7],
                           data_channels=[0, 2])
            self.assertEqual(ds.reads, 3 * 3 * 2)
            counts = task.get_channel_counts().set_index('data_channel')
            self.assertEqual(counts['sampled'].tolist(),
                             [True, False, True, False])
            # the unsampled channels are still corrected (same camera)
            self.assertTrue((counts['n_hot_camera'] == len(HOT)).all())
            self.assertEqual(task.get_summary()['sample_z_positions']['0'],
                             [Z[0], Z[3], Z[7]])
            with self.assertRaises(ValueError):
                hotpixeltask.HotPixelMap(
                    ds, {'data_channels': [0], 'min_channels': 2},
                    'Bad')._run_analysis()
            with self.assertRaises(ValueError):
                hotpixeltask.HotPixelMap(
                    ds, {'camera_colors': [['650']]},
                    'Bad2').get_channel_cameras()
            self.assertEqual(ds.reads, 18)      # refused before reading

    def test_correction_needs_a_finished_task(self):
        with tempfile.TemporaryDirectory() as root:
            ds = _DataSet(root)
            task = hotpixeltask.HotPixelMap(ds, {}, 'Unrun')
            task.save()
            with self.assertRaises(RuntimeError):
                task.get_correction()


# ------------------------------------------------------------- warps

WSIZE = 80
WZ = [1.0 + 0.5 * k for k in range(6)]
WHOT = [(20, 30), (55, 12), (60, 61), (61, 61)]
REAL_SPIKE = (35, 45)            # a real one-pixel feature, channel 1 only


def _truth(dataChannel, zPosition):
    rng = np.random.default_rng(int(round(zPosition * 2)) + 50 * dataChannel)
    field = ndimage.gaussian_filter(rng.normal(0, 1, (WSIZE, WSIZE)), 6)
    image = 400.0 + 150.0 * field / field.std()
    if dataChannel == 1:
        image[REAL_SPIKE] += 1500
    return np.rint(image).astype(np.uint16)


def _with_hot(dataChannel, zPosition):
    image = _truth(dataChannel, zPosition).astype(np.int64)
    for k, p in enumerate(WHOT):
        image[p] += 800 + 300 * k
    return np.clip(image, 0, 65535).astype(np.uint16)


class _WarpOrganization:
    def get_data_channels(self):
        return np.array([0, 1])

    def get_data_channel_color(self, dataChannel):
        return '650'

    def get_data_channel_name(self, dataChannel):
        return 'Bit%d' % (dataChannel + 1)


class _WarpDataSet:
    def __init__(self, frames):
        self.frames = frames
        self.saved = {}

    def get_data_organization(self):
        return _WarpOrganization()

    def get_fovs(self):
        return [0]

    def get_z_positions(self):
        return list(WZ)

    def z_index_to_position(self, zIndex):
        return WZ[zIndex]

    def get_image_dimensions(self):
        return (WSIZE, WSIZE)

    def get_raw_image(self, dataChannel, fov, zPosition):
        return self.frames(dataChannel, zPosition)

    def load_dataframe_from_csv(self, resultName, analysisTask=None,
                                resultIndex=None, subdirectory=None, **kw):
        return self.saved[resultName, resultIndex].copy()

    def load_numpy_analysis_result(self, resultName, analysisName,
                                   resultIndex=None, subdirectory=None,
                                   fileExtension='.npy'):
        return self.saved[resultName, resultIndex]


def _table():
    rows = []
    for dataChannel, (dx, dy, theta, dz) in {
            0: (0.0, 0.0, 0.0, 0.0), 1: (7.3, -4.6, 0.01, 0.2)}.items():
        for zIndex, z in enumerate(WZ):
            rows.append({'dataChannel': dataChannel, 'zIndex': zIndex,
                         'zPos': z, 'zPos_new': z + dz, 'xshift': dx,
                         'yshift': dy, 'rotation_radians': theta,
                         'rotation_center_x': (WSIZE - 1) / 2.0,
                         'rotation_center_y': (WSIZE - 1) / 2.0})
    return pd.DataFrame(rows)


def _offsets():
    return np.array([transform.SimilarityTransform(
        translation=t).params for t in ((0, 0), (5.6, -3.2))])


def _warps(frames):
    ds = _WarpDataSet(frames)
    ds.saved['transformation_table', 0] = _table()
    ds.saved['offsets', 0] = _offsets()
    with contextlib.redirect_stdout(io.StringIO()):
        poly = FiducialPolynomialWarp3D(ds, {}, 'PolyWarp')
        corr3d = FiducialCorrelationWarp(ds, {'register_3d': True}, 'W3')
        corr2d = FiducialCorrelationWarp(ds, {}, 'W2')
    return {'poly': poly, 'corr3d': corr3d, 'corr2d': corr2d}


def _correction(points=WHOT):
    mask = np.zeros((WSIZE, WSIZE), bool)
    for p in points:
        mask[p] = True
    return hotpixel.HotPixelCorrection({0: mask}, {0: 0, 1: 0})


class WarpCorrection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.hot = _warps(_with_hot)
        cls.truth = _warps(_truth)

    def test_hot_pixels_are_gone_after_the_warp(self):
        correction = _correction()
        for name in ('poly', 'corr3d', 'corr2d'):
            for zIndex in (0, 2, 5):
                truth = self.truth[name].get_aligned_image(0, 1, zIndex)\
                    .astype(float)
                raw = self.hot[name].get_aligned_image(0, 1, zIndex)\
                    .astype(float)
                fixed = self.hot[name].get_aligned_image(
                    0, 1, zIndex, None, hotPixels=correction).astype(float)
                self.assertGreater(np.abs(raw - truth).max(), 150,
                                   (name, zIndex))
                # what is left is the neighbour median against the true
                # value on a smooth field
                self.assertLess(np.abs(fixed - truth).max(), 25,
                                (name, zIndex))
                self.assertLess(np.count_nonzero(fixed != truth),
                                8 * 2 * len(WHOT), (name, zIndex))

    def test_real_structure_in_one_channel_is_kept(self):
        correction = _correction()
        for name in ('poly', 'corr3d', 'corr2d'):
            truth = self.truth[name].get_aligned_image(0, 1, 2)
            fixed = self.hot[name].get_aligned_image(0, 1, 2, None,
                                                     hotPixels=correction)
            peak = np.unravel_index(np.argmax(truth), truth.shape)
            self.assertGreater(int(truth[peak]), 600)
            window = (slice(peak[0] - 3, peak[0] + 4),
                      slice(peak[1] - 3, peak[1] + 4))
            np.testing.assert_array_equal(fixed[window], truth[window])

    def test_off_is_bit_identical(self):
        empty = _correction(points=[])
        for name in ('poly', 'corr3d', 'corr2d'):
            for dataChannel in (0, 1):
                for zIndex in range(len(WZ)):
                    task = self.hot[name]
                    plain = task.get_aligned_image(0, dataChannel, zIndex)
                    for other in (
                            task.get_aligned_image(0, dataChannel, zIndex,
                                                   None, hotPixels=None),
                            task.get_aligned_image(0, dataChannel, zIndex,
                                                   None, hotPixels=empty)):
                        self.assertEqual(other.dtype, plain.dtype)
                        np.testing.assert_array_equal(other, plain)

    def test_poly_off_matches_a_direct_warp(self):
        # the table row's own transform on the blend of the two raw planes
        task = self.hot['poly']
        row = _table().iloc[6 + 2]
        z = row['zPos_new']
        order = np.argsort(np.abs(np.asarray(WZ) - z))[:2]
        distances = np.abs(z - np.asarray(WZ)[order])
        weights = 1.0 - distances / distances.sum()
        blend = (_with_hot(1, WZ[order[0]]).astype(np.float32) * weights[0]
                 + _with_hot(1, WZ[order[1]]).astype(np.float32)
                 * weights[1]).astype(np.uint16)
        expected = transform.warp(blend, task._row_transform(row),
                                  preserve_range=True).astype(np.uint16)
        np.testing.assert_array_equal(task.get_aligned_image(0, 1, 2),
                                      expected)


# ---------------------------------------------------------- preprocess

class _RecordingWarp(analysistask.AnalysisTask):
    """A warp stand-in that records how get_aligned_image is called."""

    calls = []

    def get_estimated_memory(self):
        return 1

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return []

    def _run_analysis(self):
        pass

    def get_aligned_image(self, fov, dataChannel, zIndex,
                          chromaticCorrector=None, hotPixels=None, **kw):
        _RecordingWarp.calls.append(
            {'args': (fov, dataChannel, zIndex, chromaticCorrector),
             'hotPixels': hotPixels, 'kw': kw, 'passed': 'hotPixels' in kw
             or hotPixels is not None})
        image = raw_frame(dataChannel, fov, Z[zIndex])
        if hotPixels is not None:
            image = hotPixels.correct(image, dataChannel)
        return image


class _OldWarp(_RecordingWarp):
    def get_aligned_image(self, fov, dataChannel, zIndex,
                          chromaticCorrector=None):
        return raw_frame(dataChannel, fov, Z[zIndex])


PRE = {'warp_task': 'Warp', 'decon_sigma': -1, 'highpass_sigma': 2,
       'lowpass_sigma': 1, 'write_preprocessed_images': False,
       'save_pixel_histogram': False}


class PreprocessOption(unittest.TestCase):

    def test_hot_pixel_task(self):
        with tempfile.TemporaryDirectory() as root:
            ds = _DataSet(root)
            hot = run_map(ds)
            _RecordingWarp(ds, {}, 'Warp').save()
            plain = preprocess.DeconvolutionPreprocess(ds, dict(PRE), 'Pre')
            plain.save()
            fixed = preprocess.DeconvolutionPreprocess(
                ds, dict(PRE, hot_pixel_task='HotPix'), 'PreHot')
            fixed.save()
            self.assertEqual(plain.get_dependencies(), ['Warp'])
            self.assertEqual(fixed.get_dependencies(), ['Warp', 'HotPix'])
            self.assertNotIn('hot_pixel_task', plain.parameters)
            self.assertNotIn('hot_pixel_task',
                             ds.load_analysis_task('Pre').parameters)
            self.assertIsNone(plain.get_hot_pixel_correction())

            _RecordingWarp.calls.clear()
            a = plain.get_processed_image(2, 1, 3)
            self.assertEqual(len(_RecordingWarp.calls), 1)
            self.assertFalse(_RecordingWarp.calls[0]['passed'])
            np.testing.assert_array_equal(
                a, plain._preprocess_image(raw_frame(1, 2, Z[3])))

            b = fixed.get_processed_image(2, 1, 3)
            self.assertIsInstance(_RecordingWarp.calls[-1]['hotPixels'],
                                  hotpixel.HotPixelCorrection)
            corrected = hot.get_correction().correct(raw_frame(1, 2, Z[3]), 1)
            np.testing.assert_array_equal(b, fixed._preprocess_image(
                corrected))
            # the defect makes a spot in the uncorrected image only
            y, x = 10, 20
            self.assertGreater(float(a[y, x]), 10 * float(b[y, x]) + 50)

    def test_warp_without_the_argument_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            ds = _DataSet(root)
            run_map(ds)
            _OldWarp(ds, {}, 'Warp').save()
            preprocess.DeconvolutionPreprocess(ds, dict(PRE), 'Pre')
            with self.assertRaises(ValueError):
                preprocess.DeconvolutionPreprocess(
                    ds, dict(PRE, hot_pixel_task='HotPix'), 'PreHot')

    def test_deconwolf_preprocess_refuses_the_option(self):
        # it reads frames without get_aligned_image: no silent no-op
        with tempfile.TemporaryDirectory() as root:
            ds = _DataSet(root)
            with self.assertRaises(ValueError):
                preprocess.DeconvolutionPreprocessDW(
                    ds, {'warp_task': 'Warp', 'hot_pixel_task': 'HotPix'},
                    'DW')


if __name__ == '__main__':
    unittest.main()
