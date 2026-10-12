"""OptimizeIteration's crop region and the images of its chromatic samples.

The crop: crop_width (default 50, one integer or [top, bottom, left, right]),
widened by adaptive_crop over the data channels adaptive_crop_channels names
('codebook', the default, or 'all') -- the region Decode computes for the
same codebook and settings (decode.task_crop_bounds), recorded per fov
(crop_bounds_<fov>.json). Optimize decodes the full frame; then
scale_factor_region says what the region restricts:

  'crop' (default)   the scale factors, backgrounds, barcode counts and stored
                     barcodes come from the decoded pixels inside the region
                     (exclusive ends), labelled inside it, less the components
                     adaptive_edge_margin drops -- what an image-space Decode
                     keeps (CropRegionTests);
  'full_frame'       the previous code: refactors from the whole frame, the
                     whole-frame components with centroid in the region, ends
                     included (FullFrameTests).

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

The adaptive margin sets A's bottom and left edges (and fov 1's right), B's
top and right ones; adaptive_edge_margin 2 drops centroids less than 2 px from
the outermost decoded row / column there. With strip_spots the preprocess
task adds bright spots with unequal bits outside every region (STRIP_SPOTS).

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
from skimage import measure
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
# (codeword, raw row, raw col, intensity of the codeword's first on-bit; the
# second stays 100): outside every region of the library, in the strip the
# full frame decodes; the unequal bits pull the full-frame refactors off 1
STRIP_SPOTS = {
    'A': [(0, 68, 30, 300.0), (2, 20, 6, 250.0), (5, 50, 101, 280.0)],
    'B': [(0, 6, 50, 300.0), (2, 50, 95, 250.0), (5, 8, 70, 280.0)]}


def codewords():
    return np.array([[1 if b in on else 0 for b in range(4)]
                     for on in itertools.combinations(range(4), 2)])


def library_images(library, stripSpots=False):
    images = np.zeros((4, H, W), dtype=np.float32)
    for code, row, col in SPOTS[library]:
        for bit in np.flatnonzero(codewords()[code]):
            images[bit, row - 1:row + 2, col - 1:col + 2] = 100.0
    if stripSpots:
        for code, row, col, first in STRIP_SPOTS[library]:
            for k, bit in enumerate(np.flatnonzero(codewords()[code])):
                images[bit, row - 1:row + 2, col - 1:col + 2] = \
                    first if k == 0 else 100.0
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
        return library_images(self.parameters['library'],
                              self.parameters.get('strip_spots', False))

    def get_hot_pixel_correction(self):
        return _Correction() if self.parameters.get('hot') else None


def make_dataset(root, warpClass=_Warp):
    ds = _DataSet(root)
    warpClass(ds, {}, 'Warp').save()
    for library in ('A', 'B'):
        _Preprocess(ds, {'library': library}, 'Pre' + library).save()
        _Preprocess(ds, {'library': library, 'hot': True},
                    'PreHot' + library).save()
        _Preprocess(ds, {'library': library, 'strip_spots': True},
                    'PreStrip' + library).save()
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


class _LightGlobalAlign:
    """Global = fov coordinates (Decode's image-space path asks for it)."""

    def fov_coordinate_array_to_global(self, fov, centroids):
        return np.array(centroids, dtype=np.float64, copy=True)


class _LightDataSet:
    """No files: load_analysis_task from a dict."""

    def __init__(self):
        self.tasks = {'Warp': _Warp.__new__(_Warp), 'GA': _LightGlobalAlign()}
        for library in ('A', 'B'):
            for name, extra in (('Pre', {}), ('PreStrip',
                                              {'strip_spots': True})):
                pre = _Preprocess.__new__(_Preprocess)
                pre.parameters = {'warp_task': 'Warp', 'library': library,
                                  **extra}
                self.tasks[name + library] = pre

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


# Decode with the pixel settings Optimize's decode uses (optimize_task above:
# distance 0.5, decode_pixels' default magnitude 1.0, area_threshold 4)
DECODE_AS_OPTIMIZE = {'distance_threshold': 0.5, 'magnitude_threshold': 1.0,
                      'minimum_area': 4}
COMPARED = ['barcode_id', 'x', 'y', 'z', 'area', 'mean_intensity',
            'max_intensity', 'mean_distance', 'min_distance', 'global_x',
            'global_y']


def decode_plane(library, fov, zIndex=0, preprocess=None, **extra):
    """The barcodes an image-space Decode emits for one plane of fov, with
    scale factors 1 and backgrounds 0 (an iteration without a previous
    one)."""
    parameters = dict(DECODE_AS_OPTIMIZE, crop_in_image_space=True)
    parameters.update(extra)
    task = light_decode(library, **parameters)
    task._pendingBarcodes = []      # _emit_barcodes buffers here
    with contextlib.redirect_stdout(io.StringIO()):
        task._process_independent_z_slice(
            fov, zIndex, None, np.ones(4), np.zeros(4),
            task.dataSet.load_analysis_task(preprocess or 'Pre' + library),
            decoding.PixelBasedDecoder(_Codebook(library)))
    frames = [df for df in task._pendingBarcodes if len(df)]
    return pd.concat(frames, ignore_index=True)


def ordered(df):
    return df[COMPARED].sort_values(['barcode_id', 'x', 'y']).reset_index(
        drop=True).astype(np.float64)


def fragment_arrays(task, fragment):
    """What one fragment saved for the aggregates."""
    return {name: task.dataSet.load_numpy_analysis_result(
        name, task.analysisName, resultIndex=fragment)
        for name in ('scale_refactors', 'background_refactors',
                     'barcode_counts')}


def reference_refactors(library, bounds, edges, edgeMargin=2,
                        stripSpots=False, backgrounds=True):
    """extract_refactors of the decode of the images CUT to bounds (what
    Decode decodes) after setting the components that the edge margin drops
    (by centroid) to undecoded; independent of the optimize code path."""
    r0, r1, c0, c1 = bounds
    decoder = decoding.PixelBasedDecoder(_Codebook(library))
    decoder.refactorAreaThreshold = 4
    decoder.barcodesSeenThreshold = 0
    di, pm, npt, d = decoder.decode_pixels(
        library_images(library, stripSpots)[:, r0:r1, c0:c1], np.ones(4),
        np.zeros(4), lowPassSigma=0, distanceThreshold=0.5)
    labels = measure.label(di + 1)
    for label in range(1, labels.max() + 1):
        rows, cols = np.nonzero(labels == label)
        if not decode.edge_margin_mask(
                [cols.mean() + c0], [rows.mean() + r0], bounds, edges,
                edgeMargin)[0]:
            di[labels == label] = -1
    return decoder.extract_refactors(di, pm, npt,
                                     extractBackgrounds=backgrounds)


class DefaultsTests(unittest.TestCase):

    def test_defaults(self):
        task = optimize.OptimizeIteration(_LightDataSet(), {
            'preprocess_task': 'PreA', 'warp_task': 'Warp',
            'fov_index': [[0, 0]]}, 'Opt')
        self.assertEqual(task.parameters['crop_width'], 50)
        self.assertEqual(task.parameters['crop_width'],
                         decode.DEFAULT_CROP_WIDTH)
        self.assertEqual(task.parameters['adaptive_crop_channels'],
                         'codebook')
        self.assertTrue(task.parameters['adaptive_crop'])
        self.assertEqual(task.parameters['scale_factor_region'], 'crop')
        # the same edge margin as Decode's default
        self.assertEqual(task.parameters['adaptive_edge_margin'], 2)
        bare = decode.Decode(_LightDataSet(), {
            'preprocess_task': 'PreA', 'optimize_task': 'Opt',
            'global_align_task': 'GA', 'write_decoded_fovs': [],
            'write_decoded_z': []}, 'Decode')
        self.assertEqual(bare.parameters['adaptive_edge_margin'], 2)
        self.assertEqual(bare.parameters['crop_width'],
                         decode.DEFAULT_CROP_WIDTH)

    def test_refused_values(self):
        for bad in ([1, 2, 3], -1, 1.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                light_optimize(crop_width=bad)
        with self.assertRaises(ValueError):
            light_optimize(adaptive_crop_channels='bits')
        for bad in ('Crop', 'region', None, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                light_optimize(scale_factor_region=bad)
        for bad in (-1, 1.5, True, '2'):
            with self.assertRaises(ValueError, msg=repr(bad)):
                light_optimize(adaptive_edge_margin=bad)


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
                    # and the same edges for the edge margin
                    self.assertEqual(o._edge_margin_edges(fov),
                                     d._edge_margin_edges(fov),
                                     (library, extra, fov))
        self.assertEqual(light_optimize('A')._crop_bounds(0),
                         BOUNDS[('A', 0)])
        self.assertEqual(light_optimize('B')._crop_bounds(0),
                         BOUNDS[('B', 0)])
        self.assertEqual(light_optimize('A')._crop_bounds(1),
                         BOUNDS[('A', 1)])
        self.assertEqual(light_optimize('A', adaptive_crop_channels='all')
                         ._crop_bounds(0), BOUNDS[('all', 0)])
        self.assertEqual(light_optimize('A')._edge_margin_edges(0),
                         (False, True, True, False))
        self.assertEqual(light_optimize('A')._edge_margin_edges(1),
                         (False, True, True, True))
        self.assertEqual(light_optimize('B')._edge_margin_edges(0),
                         (True, False, False, True))
        for extra in ({'adaptive_edge_margin': 0},
                      {'scale_factor_region': 'full_frame'}):
            self.assertEqual(light_optimize('A', **extra)
                             ._edge_margin_edges(0), (False,) * 4, extra)

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


class CropRegionTests(unittest.TestCase):
    """scale_factor_region 'crop' (the default)."""

    def run_task(self, library='A', **extra):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        task = optimize_task(make_dataset(directory.name), library, **extra)
        run_all(task)
        return task

    def test_stored_barcodes_are_decodes(self):
        # exactly what an image-space Decode of that fov and plane emits
        # (same pixel settings), for every fragment, both libraries, with
        # and without the edge margin and the strip spots
        for library in ('A', 'B'):
            for extra in ({}, {'adaptive_edge_margin': 0},
                          {'preprocess': 'PreStrip' + library}):
                task = self.run_task(library, **extra)
                margin = task.parameters['adaptive_edge_margin']
                for fragment, (fov, z) in enumerate(
                        task.parameters['fov_index']):
                    got = task.get_barcode_database().get_barcodes(fragment)
                    want = decode_plane(library, fov, z,
                                        preprocess=extra.get('preprocess'),
                                        adaptive_edge_margin=margin)
                    self.assertGreaterEqual(len(want), 3)
                    pd.testing.assert_frame_equal(
                        ordered(got), ordered(want),
                        obj=str((library, extra, fragment)))

    def test_refactors_from_the_region_only(self):
        for library in ('A', 'B'):
            for margin in (2, 0):
                task = self.run_task(library, optimize_background=True,
                                     adaptive_edge_margin=margin)
                for fragment, (fov, _) in enumerate(
                        task.parameters['fov_index']):
                    bounds = EXPECTED[(library, fov)]
                    edges = task._edge_margin_edges(fov)
                    want = reference_refactors(library, bounds, edges,
                                               margin)
                    got = fragment_arrays(task, fragment)
                    for name, w in zip(('scale_refactors',
                                        'background_refactors',
                                        'barcode_counts'), want):
                        np.testing.assert_array_equal(
                            got[name], w,
                            err_msg=str((library, margin, fragment, name)))

    def test_content_outside_the_region_changes_nothing(self):
        # bright, unequal-bit spots in the strip: every crop result and the
        # stored barcodes are those without them; full_frame moves
        for library in ('A', 'B'):
            for region in ('crop', 'full_frame'):
                plain = self.run_task(library, scale_factor_region=region,
                                      optimize_background=True,
                                      optimize_chromatic_correction=True)
                strip = self.run_task(library, scale_factor_region=region,
                                      optimize_background=True,
                                      optimize_chromatic_correction=True,
                                      preprocess='PreStrip' + library)
                same = all(
                    all(np.array_equal(a, b) for a, b in zip(
                        fragment_arrays(plain, f).values(),
                        fragment_arrays(strip, f).values()))
                    and stored(plain, f) == stored(strip, f)
                    for f in range(3))
                if region == 'crop':
                    self.assertTrue(same, library)
                    for f in range(3):
                        samples = [t.dataSet.load_pickle_analysis_result(
                            'chromatic_samples', t.analysisName,
                            resultIndex=f) for t in (plain, strip)]
                        for c1 in samples[0]:
                            for c2 in samples[0][c1]:
                                for a, b in zip(samples[0][c1][c2],
                                                samples[1][c1][c2]):
                                    np.testing.assert_array_equal(a, b)
                else:
                    self.assertFalse(same, library)
                    r = fragment_arrays(strip, 0)['scale_refactors']
                    self.assertGreater(np.nanmax(np.abs(r - 1)), 0.01)

    def test_edges_cut_molecules_and_edge_margin(self):
        task = self.run_task('A')
        got = task.get_barcode_database().get_barcodes(0)
        rows = sorted(zip(got['barcode_id'].astype(int),
                          got['x'].astype(float), got['y'].astype(float),
                          got['area'].astype(int)))
        # fov 0, rows 5:65, cols 9:98 (bottom and left set by the adaptive
        # margin). Kept: the inside spots, the top edge's cut spot (rows 5-6
        # of 4-6, y 5.5; the top edge is the fixed crop, no edge margin) and
        # (1, 7, 11), exactly 2 px from the left edge's col 9. Not kept:
        # (2, 63, 96), 1 px from the bottom edge's last row 64; the right
        # edge's (4, 40, 98), of which only col 97 is inside (area 3 < 4;
        # the whole-frame filter, ends included, kept it whole); every spot
        # outside the region.
        self.assertEqual(rows, [(0, 50.0, 30.0, 9), (1, 11.0, 7.0, 9),
                                (3, 40.0, 5.5, 6), (3, 53.0, 35.0, 9),
                                (4, 86.0, 20.0, 9), (5, 70.0, 50.0, 9)])
        counts = fragment_arrays(task, 0)['barcode_counts']
        np.testing.assert_array_equal(counts, [1, 1, 0, 2, 1, 1])
        # without the margin the bottom spot counts and is stored
        loose = self.run_task('A', adaptive_edge_margin=0)
        np.testing.assert_array_equal(
            fragment_arrays(loose, 0)['barcode_counts'], [1, 1, 1, 2, 1, 1])
        self.assertIn((2, 96.0, 63.0), stored(loose, 0))
        # every stored centroid inside the slice Decode cuts: the region's
        # last row / column is rowEnd - 1 / colEnd - 1 (not rowEnd, colEnd)
        for t in (task, loose):
            for f, (fov, _) in enumerate(t.parameters['fov_index']):
                r0, r1, c0, c1 = EXPECTED[('A', fov)]
                df = t.get_barcode_database().get_barcodes(f)
                self.assertTrue(df['x'].between(c0, c1 - 1).all())
                self.assertTrue(df['y'].between(r0, r1 - 1).all())

    def test_record(self):
        for library in ('A', 'B'):
            task = self.run_task(library)
            for fov in (0, 1):
                record = task.get_crop_record(fov)
                want = EXPECTED[(library, fov)]
                edges = task._edge_margin_edges(fov)
                self.assertEqual(record['crop_bounds'], list(want))
                self.assertTrue(record['crop_in_image_space'])
                self.assertIsNone(record['barcode_filter'])
                self.assertEqual(record['scale_factors_from'], 'crop')
                self.assertEqual(record['adaptive_edge_margin'], 2)
                self.assertEqual(record['edge_margin_edges'], list(edges))
                self.assertEqual(record['barcode_bounds'], list(
                    decode.edge_margin_bounds(want, edges, 2)))
                self.assertEqual(record['crop_width'], CROP)
                self.assertEqual(record['xy_frame'], 'full_frame')
                self.assertEqual(task.get_crop_bounds(fov), want)
                # the same record an image-space Decode writes, but for the
                # crop_width barcode_filter Decode keeps for barcode space
                # (null in image space, as here)
                d = light_decode(library, crop_in_image_space=True)
                self.assertEqual({k: v for k, v in record.items()
                                  if k != 'scale_factors_from'},
                                 d._crop_record(fov))

    def test_decoded_images_unchanged(self):
        # the frame is still decoded whole: the written images are the same
        # in both modes
        images = []
        for region in ('crop', 'full_frame'):
            task = self.run_task('A', scale_factor_region=region,
                                 write_decoded_images=True)
            images.append(task.dataSet.get_analysis_image_set(
                task, 'decoded', 0))
        self.assertEqual(images[0].shape, (3, H, W))
        np.testing.assert_array_equal(images[0], images[1])

    def test_optimize_iteration_fov(self):
        with tempfile.TemporaryDirectory() as root:
            ds = make_dataset(root)
            with contextlib.redirect_stdout(io.StringIO()):
                task = optimize.OptimizeIterationFOV(ds, {
                    'preprocess_task': 'PreA', 'warp_task': 'Warp',
                    'crop_width': CROP, 'z_index': 0,
                    'distance_threshold': 0.5}, 'OptFOV')
                task.save()
            self.assertEqual(task.parameters['scale_factor_region'], 'crop')
            run_all(task)
            for fragment, fov in enumerate(task.parameters['fov_index']):
                pd.testing.assert_frame_equal(
                    ordered(task.get_barcode_database().get_barcodes(
                        fragment)),
                    ordered(decode_plane('A', fov, 0)))
                self.assertEqual(task.get_crop_record(fov)['crop_bounds'],
                                 list(BOUNDS[('A', fov)]))
                self.assertEqual(
                    task.get_crop_record(fov)['scale_factors_from'], 'crop')


class FullFrameTests(unittest.TestCase):
    """scale_factor_region 'full_frame' = the code before the option."""

    def test_barcodes_in_the_region_and_record(self):
        for library in ('A', 'B'):
            with tempfile.TemporaryDirectory() as root:
                ds = make_dataset(root)
                task = optimize_task(ds, library,
                                     scale_factor_region='full_frame')
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
                    # the previous record, key for key
                    self.assertEqual(list(record), [
                        'fov', 'image_shape', 'crop_in_image_space',
                        'crop_width', 'adaptive_crop', 'adaptive_crop_channels',
                        'data_channels', 'adaptive_margin', 'crop_bounds',
                        'barcode_filter', 'barcode_bounds',
                        'scale_factors_from', 'xy_frame'])
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

    def test_refactors_are_the_whole_frame(self):
        # the previous code: extract_refactors on every decoded pixel
        for library in ('A', 'B'):
            with tempfile.TemporaryDirectory() as root:
                ds = make_dataset(root)
                task = optimize_task(ds, library,
                                     preprocess='PreStrip' + library,
                                     scale_factor_region='full_frame',
                                     optimize_background=True)
                run_all(task)
                decoder = decoding.PixelBasedDecoder(_Codebook(library))
                decoder.refactorAreaThreshold = 4
                decoder.barcodesSeenThreshold = 0
                di, pm, npt, d = decoder.decode_pixels(
                    library_images(library, True), np.ones(4), np.zeros(4),
                    lowPassSigma=0, distanceThreshold=0.5)
                want = decoder.extract_refactors(di, pm, npt,
                                                 extractBackgrounds=True)
                got = fragment_arrays(task, 0)
                for name, w in zip(('scale_refactors', 'background_refactors',
                                    'barcode_counts'), want):
                    np.testing.assert_array_equal(got[name], w)

    def test_codebook_region_keeps_what_all_channels_cut(self):
        with tempfile.TemporaryDirectory() as root:
            ds = make_dataset(root)
            mine = optimize_task(ds, 'A', 'OptMine',
                                 scale_factor_region='full_frame')
            every = optimize_task(ds, 'A', 'OptAll',
                                  adaptive_crop_channels='all',
                                  scale_factor_region='full_frame')
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
            task = optimize_task(ds, 'A', adaptive_crop=False,
                                 scale_factor_region='full_frame')
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
                    'distance_threshold': 0.5,
                    'scale_factor_region': 'full_frame'}, 'OptFOV')
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

    def test_task_json_from_before_scale_factor_region(self):
        # A saved task without the two keys loads with the new defaults
        # ('crop', 2) filled in, and an analysis json without them compares
        # equal: rerunning such a task would take the crop. Reproducing an
        # older optimize means pinning scale_factor_region 'full_frame'.
        with tempfile.TemporaryDirectory() as root:
            ds = make_dataset(root)
            old = {'preprocess_task': 'PreA', 'warp_task': 'Warp',
                   'crop_width': CROP, 'fov_index': [[0, 0]],
                   'write_decoded_images': False}
            with contextlib.redirect_stdout(io.StringIO()):
                optimize.OptimizeIteration(ds, dict(old), 'OldOpt').save()
            path = os.path.join(ds.get_analysis_subdirectory('OldOpt',
                                                             'tasks'),
                                'task.json')
            with open(path) as fh:
                saved = json.load(fh)
            saved.pop('scale_factor_region')
            saved.pop('adaptive_edge_margin')
            with open(path, 'w') as fh:
                json.dump(saved, fh)
            loaded = ds.load_analysis_task('OldOpt')
            self.assertEqual(loaded.parameters['scale_factor_region'], 'crop')
            self.assertEqual(loaded.parameters['adaptive_edge_margin'], 2)
            optimize.OptimizeIteration(ds, dict(old), 'OldOpt').save()
            with self.assertRaises(
                    analysistask.AnalysisAlreadyExistsException):
                optimize.OptimizeIteration(
                    ds, dict(old, scale_factor_region='full_frame'),
                    'OldOpt').save()


if __name__ == '__main__':
    unittest.main()
