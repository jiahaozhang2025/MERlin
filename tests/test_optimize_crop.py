"""OptimizeIteration's crop region and the images of its chromatic samples.

The crop: crop_width (default 50, one integer or [top, bottom, left, right]),
widened by adaptive_crop over the data channels adaptive_crop_channels names
('codebook', the default, or 'all') -- the region Decode computes for the
same codebook and settings (decode.task_crop_bounds). Optimize decodes the
full frame and keeps the barcodes whose centroid lies in the region, ends
included, and records the region per fov (crop_bounds_<fov>.json).

The chromatic samples without chromatic_on_preprocessed are measured on the
warp task's raw-warped images; when the preprocess task replaces camera hot
pixels (hot_pixel_task), those images get the same correction.

Synthetic data set: the 72 x 104 frame of test_decode_imagespace_crop, 8 data
channels R0..R7; library A decodes R0-R3, library B R4-R7, six 2-of-4
codewords each. The warp leaves an invalid strip per channel (R1 9 px left,
R2 7 px bottom, R5 15 px right, R6 12 px top; fov 1 adds 20 px right on R0).
crop_width [5, 3, 4, 6]:

    A    rows 5:65,  cols 9:98      B    rows 12:69, cols 4:89
    all  rows 12:65, cols 9:89      A fov 1  rows 5:65, cols 9:84

Run with `python -m unittest tests/test_optimize_crop.py`.
"""
import contextlib
import io
import itertools
import json
import logging
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis import decode
from merlin.analysis import optimize
from merlin.core import analysistask
from merlin.core import dataset
from merlin.util import decoding

H, W = 72, 104
READOUTS = ['R%d' % c for c in range(8)]
COLORS = ['650', '750', '650', '750', '650', '750', '650', '750']
LIBRARY_BITS = {'A': READOUTS[:4], 'B': READOUTS[4:]}
CROP = [5, 3, 4, 6]
SHIFTS = {0: {1: (-9, 0), 2: (0, 7), 5: (15, 0), 6: (0, -12)}}
SHIFTS[1] = {**SHIFTS[0], 0: (20, 0)}
BOUNDS = {('A', 0): (5, 65, 9, 98), ('B', 0): (12, 69, 4, 89),
          ('all', 0): (12, 65, 9, 89), ('A', 1): (5, 65, 9, 84)}
MARGINS = {('A', 0): (0, 7, 9, 0), ('A', 1): (0, 7, 9, 20)}
# per (library, fov): fov 1's extra strip is on R0, a library A channel
EXPECTED = {('A', 0): BOUNDS[('A', 0)], ('A', 1): BOUNDS[('A', 1)],
            ('B', 0): BOUNDS[('B', 0)], ('B', 1): BOUNDS[('B', 0)]}

# (codeword, raw row, raw col): spots inside, on and outside each region
SPOTS = {
    'A': [(0, 30, 50), (1, 7, 11), (2, 63, 96), (3, 35, 53), (5, 50, 70),
          (3, 5, 40), (4, 40, 98), (1, 68, 50), (2, 40, 6), (0, 30, 101),
          (4, 20, 86)],
    'B': [(0, 40, 30), (1, 14, 6), (2, 67, 87), (5, 40, 46), (3, 12, 60),
          (4, 30, 92), (0, 8, 30), (1, 70, 20)]}


def codewords():
    return np.array([[1 if b in on else 0 for b in range(4)]
                     for on in itertools.combinations(range(4), 2)])


def library_images(library):
    images = np.zeros((4, H, W), dtype=np.float32)
    for code, row, col in SPOTS[library]:
        for bit in np.flatnonzero(codewords()[code]):
            images[bit, row - 1:row + 2, col - 1:col + 2] = 100.0
    return images


def transforms_of(fov):
    return [transform.SimilarityTransform(
        translation=SHIFTS[fov].get(c, (0, 0))) for c in range(8)]


class _Codebook:

    def __init__(self, library):
        self.library = library

    def get_barcodes(self, ignoreBlanks=False):
        return codewords()

    def get_barcode(self, index):
        return codewords()[index]

    def get_bit_names(self):
        return list(LIBRARY_BITS[self.library])

    def get_bit_count(self):
        return 4

    def get_barcode_count(self):
        return 6

    def get_blank_indexes(self):
        return [5]

    def get_coding_indexes(self):
        return [0, 1, 2, 3, 4]


class _DataOrganization:

    def get_data_channel_for_bit(self, bitName):
        return READOUTS.index(bitName)

    def get_data_channel_color(self, dataChannel):
        return COLORS[dataChannel]


class _DataSet(dataset.DataSet):

    def __init__(self, root):
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__('synthetic', dataHome=root, analysisHome=root)

    def get_fovs(self):
        return np.array([0, 1])

    def get_z_positions(self):
        return [0.0, 0.5]

    def get_image_dimensions(self):
        return (H, W)

    def get_data_organization(self):
        return _DataOrganization()


class _Task(analysistask.AnalysisTask):

    def get_estimated_memory(self):
        return 1

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return []

    def _run_analysis(self):
        pass


_UNSET = object()


class _Warp(_Task):
    """Transformations as above; get_aligned_image records how it is called
    (kw holds hotPixels only when it was passed) and returns the library's
    image of that channel, with the correction applied when one is
    passed."""

    calls = []

    def get_transformation(self, fov, dataChannel=None):
        return transforms_of(fov)

    def get_aligned_image(self, fov, dataChannel, zIndex,
                          chromaticCorrector=None, hotPixels=_UNSET):
        return self._image(fov, dataChannel, zIndex, hotPixels)

    def _image(self, fov, dataChannel, zIndex, hotPixels):
        _Warp.calls.append(dict(
            fov=fov, dataChannel=dataChannel, zIndex=zIndex,
            kw={} if hotPixels is _UNSET else {'hotPixels': hotPixels}))
        library = 'A' if dataChannel < 4 else 'B'
        image = library_images(library)[dataChannel % 4].copy()
        if hotPixels is not _UNSET and hotPixels is not None:
            image = hotPixels.correct(image, dataChannel)
        return image


class _OldWarp(_Warp):
    """A warp whose get_aligned_image cannot take hotPixels."""

    def get_aligned_image(self, fov, dataChannel, zIndex,
                          chromaticCorrector=None):
        return self._image(fov, dataChannel, zIndex, _UNSET)


class _Correction:
    """Marks pixel (0, 0) of every frame it corrects."""

    def correct(self, image, dataChannel):
        out = np.array(image, copy=True)
        out[0, 0] = -1.0
        return out


class _Preprocess(_Task):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self.parameters.setdefault('warp_task', 'Warp')
        self.parameters.setdefault('save_pixel_histogram', False)

    def get_codebook(self):
        return _Codebook(self.parameters['library'])

    def get_processed_image_set(self, fov, zIndex, chromaticCorrector=None):
        return library_images(self.parameters['library'])

    def get_hot_pixel_correction(self):
        return _Correction() if self.parameters.get('hot') else None


def make_dataset(root, warpClass=_Warp):
    ds = _DataSet(root)
    warpClass(ds, {}, 'Warp').save()
    for library in ('A', 'B'):
        _Preprocess(ds, {'library': library}, 'Pre' + library).save()
        _Preprocess(ds, {'library': library, 'hot': True},
                    'PreHot' + library).save()
    return ds


def optimize_task(ds, library='A', name='Opt', preprocess=None, **extra):
    parameters = {'preprocess_task': preprocess or 'Pre' + library,
                  'warp_task': 'Warp', 'crop_width': CROP,
                  'fov_index': [[0, 0], [1, 0], [0, 1]],
                  'write_decoded_images': False,
                  'distance_threshold': 0.5}
    parameters.update(extra)
    with contextlib.redirect_stdout(io.StringIO()):
        task = optimize.OptimizeIteration(ds, parameters, name)
        task.save()
    return task


def run_all(task):
    with contextlib.redirect_stdout(io.StringIO()):
        for f in range(task.fragment_count()):
            task.run(f)


def whole_spots(library, bounds):
    """(barcode_id, x, y) of the library's spots whose centre lies in
    bounds, ends included."""
    r0, r1, c0, c1 = bounds
    return sorted((code, float(col), float(row))
                  for code, row, col in SPOTS[library]
                  if c0 <= col <= c1 and r0 <= row <= r1)


def stored(task, fragment):
    df = task.get_barcode_database().get_barcodes(fragment)
    return sorted(zip(df['barcode_id'].astype(int), df['x'].astype(float),
                      df['y'].astype(float)))


class _LightDataSet:
    """No files: load_analysis_task from a dict."""

    def __init__(self):
        self.tasks = {'Warp': _Warp.__new__(_Warp)}
        for library in ('A', 'B'):
            pre = _Preprocess.__new__(_Preprocess)
            pre.parameters = {'warp_task': 'Warp', 'library': library}
            self.tasks['Pre' + library] = pre

    def get_image_dimensions(self):
        return (H, W)

    def get_data_organization(self):
        return _DataOrganization()

    def load_analysis_task(self, name):
        return self.tasks[name]

    def get_logger(self, task):
        return logging.getLogger('test_optimize_crop')


def light_optimize(library='A', **extra):
    parameters = {'preprocess_task': 'Pre' + library, 'warp_task': 'Warp',
                  'crop_width': CROP, 'fov_index': [[0, 0]]}
    parameters.update(extra)
    return optimize.OptimizeIteration(_LightDataSet(), parameters, 'Opt')


def light_decode(library='A', **extra):
    parameters = {'preprocess_task': 'Pre' + library, 'optimize_task': 'Opt',
                  'global_align_task': 'GA', 'crop_width': CROP,
                  'write_decoded_fovs': [], 'write_decoded_z': []}
    parameters.update(extra)
    return decode.Decode(_LightDataSet(), parameters, 'Decode')


class DefaultsTests(unittest.TestCase):

    def test_defaults(self):
        task = optimize.OptimizeIteration(_LightDataSet(), {
            'preprocess_task': 'PreA', 'warp_task': 'Warp',
            'fov_index': [[0, 0]]}, 'Opt')
        self.assertEqual(task.parameters['crop_width'], 50)
        self.assertEqual(task.parameters['adaptive_crop_channels'],
                         'codebook')
        self.assertTrue(task.parameters['adaptive_crop'])

    def test_refused_values(self):
        for bad in ([1, 2, 3], -1, 1.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                light_optimize(crop_width=bad)
        with self.assertRaises(ValueError):
            light_optimize(adaptive_crop_channels='bits')


class BoundsTests(unittest.TestCase):

    def test_same_region_as_decode(self):
        for library in ('A', 'B'):
            for extra in ({}, {'adaptive_crop_channels': 'all'},
                          {'adaptive_crop': False}, {'crop_width': 7},
                          {'crop_width': [0, 0, 0, 0]}):
                for fov in (0, 1):
                    o = light_optimize(library, **extra)
                    d = light_decode(library, **extra)
                    self.assertEqual(o._crop_bounds(fov), d._crop_bounds(fov),
                                     (library, extra, fov))
        self.assertEqual(light_optimize('A')._crop_bounds(0),
                         BOUNDS[('A', 0)])
        self.assertEqual(light_optimize('B')._crop_bounds(0),
                         BOUNDS[('B', 0)])
        self.assertEqual(light_optimize('A')._crop_bounds(1),
                         BOUNDS[('A', 1)])
        self.assertEqual(light_optimize('A', adaptive_crop_channels='all')
                         ._crop_bounds(0), BOUNDS[('all', 0)])

    def test_all_channels_is_the_previous_region(self):
        # the previous code: compute_crop_bounds over every transformation of
        # the optimize task's warp task, one crop_width
        ds = _LightDataSet()
        for cropWidth in (6, CROP):
            task = light_optimize('A', crop_width=cropWidth,
                                  adaptive_crop_channels='all')
            for fov in (0, 1):
                self.assertEqual(task._crop_bounds(fov), tuple(
                    decode.compute_crop_bounds(ds, 'Warp', fov, cropWidth,
                                               True)))


class RunTests(unittest.TestCase):

    def test_barcodes_in_the_region_and_record(self):
        for library in ('A', 'B'):
            with tempfile.TemporaryDirectory() as root:
                ds = make_dataset(root)
                task = optimize_task(ds, library)
                run_all(task)
                for fragment, (fov, _) in enumerate(
                        task.parameters['fov_index']):
                    self.assertEqual(stored(task, fragment),
                                     whole_spots(library,
                                                 EXPECTED[(library, fov)]),
                                     (library, fragment))
                for fov in (0, 1):
                    record = task.get_crop_record(fov)
                    want = EXPECTED[(library, fov)]
                    self.assertEqual(record['crop_bounds'], list(want))
                    self.assertEqual(record['barcode_filter'], list(want))
                    self.assertEqual(record['barcode_bounds'], list(want))
                    self.assertEqual(record['crop_width'], CROP)
                    self.assertEqual(record['adaptive_crop_channels'],
                                     'codebook')
                    self.assertEqual(
                        record['data_channels'],
                        [READOUTS.index(b) for b in LIBRARY_BITS[library]])
                    self.assertFalse(record['crop_in_image_space'])
                    self.assertEqual(record['scale_factors_from'],
                                     'full_frame')
                    self.assertEqual(task.get_crop_bounds(fov), want)
                self.assertEqual(task.get_crop_record(0)['fov'], 0)
                if library == 'A':
                    self.assertEqual(task.get_crop_record(1)
                                     ['adaptive_margin'],
                                     list(MARGINS[('A', 1)]))
                self.assertIsNone(task.get_crop_record(5))

    def test_codebook_region_keeps_what_all_channels_cut(self):
        with tempfile.TemporaryDirectory() as root:
            ds = make_dataset(root)
            mine = optimize_task(ds, 'A', 'OptMine')
            every = optimize_task(ds, 'A', 'OptAll',
                                  adaptive_crop_channels='all')
            run_all(mine)
            run_all(every)
            a, b = set(stored(mine, 0)), set(stored(every, 0))
            self.assertTrue(b < a)
            # library B's R5 / R6 strips cut library A's right and top
            self.assertEqual(sorted(a - b), [(1, 11.0, 7.0), (2, 96.0, 63.0),
                                             (3, 40.0, 5.0), (4, 98.0, 40.0)])
            self.assertEqual(sorted(b), whole_spots('A', BOUNDS[('all', 0)]))

    def test_without_adaptive_crop_is_the_crop_width_filter(self):
        # the previous code without adaptive_crop: extract_barcodes_with_index
        # with crop_width; now extraction without a crop and the region
        # (rows top..H - bottom, cols left..W - right, ends included)
        with tempfile.TemporaryDirectory() as root:
            ds = make_dataset(root)
            task = optimize_task(ds, 'A', adaptive_crop=False)
            run_all(task)
            decoder = decoding.PixelBasedDecoder(_Codebook('A'))
            di, pm, npt, d = decoder.decode_pixels(
                library_images('A'), np.ones(4), np.zeros(4),
                lowPassSigma=0, distanceThreshold=0.5)
            old = decoder.extract_barcodes_with_index(
                di, pm, npt, d, 0, CROP, 0, minimumArea=4)
            self.assertEqual(stored(task, 0), sorted(zip(
                old['barcode_id'].astype(int), old['x'].astype(float),
                old['y'].astype(float))))
            self.assertEqual(task.get_crop_record(0)['crop_bounds'],
                             [5, H - 3, 4, W - 6])
            self.assertIsNone(task.get_crop_record(0)['adaptive_margin'])

    def test_optimize_iteration_fov(self):
        with tempfile.TemporaryDirectory() as root:
            ds = make_dataset(root)
            with contextlib.redirect_stdout(io.StringIO()):
                task = optimize.OptimizeIterationFOV(ds, {
                    'preprocess_task': 'PreA', 'warp_task': 'Warp',
                    'crop_width': CROP, 'z_index': 0,
                    'distance_threshold': 0.5}, 'OptFOV')
                task.save()
            self.assertEqual(task.parameters['adaptive_crop_channels'],
                             'codebook')
            run_all(task)
            for fragment, fov in enumerate(task.parameters['fov_index']):
                self.assertEqual(stored(task, fragment),
                                 whole_spots('A', BOUNDS[('A', fov)]))
                self.assertEqual(task.get_crop_record(fov)['crop_bounds'],
                                 list(BOUNDS[('A', fov)]))


class ChromaticImageTests(unittest.TestCase):

    def run_chromatic(self, preprocess, warpClass=_Warp, **extra):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        ds = make_dataset(directory.name, warpClass)
        task = optimize_task(ds, 'A', preprocess=preprocess,
                             optimize_chromatic_correction=True,
                             chromatic_from_fragments=True, **extra)
        _Warp.calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            task.run(0)
        return task, list(_Warp.calls)

    def test_hot_pixels_reach_the_chromatic_images(self):
        task, calls = self.run_chromatic('PreHotA')
        self.assertEqual([c['dataChannel'] for c in calls], [0, 1, 2, 3])
        for call in calls:
            self.assertIsInstance(call['kw'].get('hotPixels'), _Correction)
        images = task._chromatic_image_set(
            0, 0, None, hotPixels=task._chromatic_hot_pixels())
        self.assertEqual(images.shape, (4, H, W))
        np.testing.assert_array_equal(images[:, 0, 0], -1.0)
        np.testing.assert_array_equal(images[:, 1:, 1:],
                                      library_images('A')[:, 1:, 1:])
        samples = task.dataSet.load_pickle_analysis_result(
            'chromatic_samples', task.analysisName, resultIndex=0)
        self.assertIn('650', samples)

    def test_without_hot_pixel_task_the_call_is_unchanged(self):
        task, calls = self.run_chromatic('PreA')
        self.assertEqual(len(calls), 4)
        for call in calls:
            self.assertEqual(call['kw'], {})
        self.assertIsNone(task._chromatic_hot_pixels())
        np.testing.assert_array_equal(
            task._chromatic_image_set(0, 0, None), library_images('A'))

    def test_on_preprocessed_reads_no_warped_images(self):
        _, calls = self.run_chromatic('PreHotA',
                                      chromatic_on_preprocessed=True)
        self.assertEqual(calls, [])

    def test_warp_without_the_argument_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'cannot take hotPixels'):
            self.run_chromatic('PreHotA', _OldWarp)
        # without a correction such a warp is called as before
        _, calls = self.run_chromatic('PreA', _OldWarp)
        self.assertEqual(len(calls), 4)

    def test_single_job_path(self):
        # chromatic_from_fragments off: the fit loads every (fov, z) group's
        # images itself, with the same correction
        with tempfile.TemporaryDirectory() as root:
            ds = make_dataset(root)
            task = optimize_task(ds, 'A', preprocess='PreHotA',
                                 optimize_chromatic_correction=True,
                                 chromatic_from_fragments=False)
            run_all(task)
            _Warp.calls.clear()
            pooled = {}

            def capture(samples, previous):
                pooled.update(samples)
                return previous
            task._fit_color_pairs = capture
            task._get_chromatic_transformations()
            groups = {(c['fov'], c['zIndex']) for c in _Warp.calls}
            # one group per (fov, z) of the barcode table it reads
            barcodes = task.get_barcode_database().get_barcodes()
            want = {(int(f), int(z)) for f, z in zip(barcodes['fov'],
                                                     barcodes['z'])}
            self.assertGreaterEqual(len(want), 2)
            self.assertEqual(groups, want)
            self.assertEqual(len(_Warp.calls), 4 * len(groups))
            for call in _Warp.calls:
                self.assertIsInstance(call['kw'].get('hotPixels'),
                                      _Correction)
            self.assertGreater(len(pooled['650']['750'][0]), 0)


class SavedParameterTests(unittest.TestCase):

    def test_saved_crop_width_kept_and_unpinned_json_drifts(self):
        with tempfile.TemporaryDirectory() as root:
            ds = make_dataset(root)
            old = {'preprocess_task': 'PreA', 'warp_task': 'Warp',
                   'crop_width': 100, 'adaptive_crop': True,
                   'fov_index': [[0, 0]], 'write_decoded_images': False}
            with contextlib.redirect_stdout(io.StringIO()):
                optimize.OptimizeIteration(ds, dict(old), 'OldOpt').save()
            # as a task.json from before adaptive_crop_channels
            path = os.path.join(ds.get_analysis_subdirectory('OldOpt',
                                                             'tasks'),
                                'task.json')
            with open(path) as fh:
                saved = json.load(fh)
            saved.pop('adaptive_crop_channels')
            with open(path, 'w') as fh:
                json.dump(saved, fh)
            loaded = ds.load_analysis_task('OldOpt')
            self.assertEqual(loaded.parameters['crop_width'], 100)
            self.assertEqual(loaded.parameters['adaptive_crop_channels'],
                             'codebook')
            # generate with the pinned value compares equal; without it the
            # new default 50 is a different task
            optimize.OptimizeIteration(ds, dict(old), 'OldOpt').save()
            unpinned = {k: v for k, v in old.items() if k != 'crop_width'}
            with self.assertRaises(
                    analysistask.AnalysisAlreadyExistsException):
                optimize.OptimizeIteration(ds, dict(unpinned),
                                           'OldOpt').save()


if __name__ == '__main__':
    unittest.main()
