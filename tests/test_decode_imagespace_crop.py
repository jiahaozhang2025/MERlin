"""Decode's per-codebook crop: crop_width per edge, the adaptive margin from
the codebook's own data channels, image-space decoding with full-frame x, y,
the per-fov crop record, the extractor's crop filter on non-square images,
and DeduplicateBarcodes' crop_to_valid_box on the same bounds.

Synthetic data set: a non-square 72 x 104 frame, 8 data channels R0..R7.
Library A decodes R0-R3, library B R4-R7, each with the six 2-of-4 codewords
(0-4 coding, 5 blank). The warp leaves an invalid strip per channel: R1 9 px
on the left and R2 7 px at the bottom (library A), R5 15 px on the right and
R6 12 px at the top (library B). crop_width [top, bottom, left, right] =
[5, 3, 4, 6]. Each edge is max(fixed, that edge's margin over the codebook's
channels):

    A    rows 5:65,  cols 9:98     (60 x 89, not square)
    B    rows 12:69, cols 4:89     (57 x 85)
    all  rows 12:65, cols 9:89     (adaptive_crop_channels 'all')

fov 1 adds a 20 px right strip on R0, so library A's fov 1 is cols 9:84.

Each library's images hold 3 x 3 spots of exact codewords: 'in' spots inside
its region (two just inside opposite corners, one across both tile seams of
tiling_factor 2), 'cut' spots that a region edge cuts, 'strip' spots outside
the region but inside the crop_width filter (barcode space keeps them), and
'out' spots outside both.

Run with `python -m unittest tests/test_decode_imagespace_crop.py`.
"""
import contextlib
import io
import itertools
import json
import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis import decode
from merlin.analysis import deduplicate
from merlin.analysis import warp
from merlin.core import analysistask
from merlin.core import dataset
from merlin.util import decoding

H, W = 72, 104
READOUTS = ['R%d' % c for c in range(8)]
LIBRARY_BITS = {'A': READOUTS[:4], 'B': READOUTS[4:]}
CROP = [5, 3, 4, 6]                      # top, bottom, left, right
# output (r, c) samples input (r + ty, c + tx): (tx, ty) per data channel
SHIFTS = {0: {1: (-9, 0), 2: (0, 7), 5: (15, 0), 6: (0, -12)}}
SHIFTS[1] = {**SHIFTS[0], 0: (20, 0)}
MARGINS = {('A', 0): (0, 7, 9, 0), ('B', 0): (12, 0, 0, 15),
           ('all', 0): (12, 7, 9, 15), ('A', 1): (0, 7, 9, 20)}
BOUNDS = {('A', 0): (5, 65, 9, 98), ('B', 0): (12, 69, 4, 89),
          ('all', 0): (12, 65, 9, 89), ('A', 1): (5, 65, 9, 84)}
GLOBAL_OFFSET = (5000.0, 7000.0)
Z_POSITIONS = [0.0, 0.5]

# (codeword, raw row, raw col, kind) of each spot's centre
SPOTS = {
    'A': [(0, 30, 50, 'in'),
          (1, 7, 11, 'in'),        # just inside the top-left corner
          (2, 63, 96, 'in'),       # just inside the bottom-right corner
          (3, 35, 53, 'in'),       # across both tile seams (row 35, col 53)
          (5, 50, 70, 'in'),       # blank
          (3, 5, 40, 'cut'),       # the top edge (row 5) cuts it
          (4, 40, 98, 'cut'),      # the right edge (col 98) cuts it
          (1, 68, 50, 'strip'),    # bottom strip, y 68 <= 72 - 3
          (2, 40, 6, 'strip'),     # left strip, x 6 >= 4
          (0, 30, 101, 'out')],    # x 101 > 104 - 6
    'B': [(0, 40, 30, 'in'),
          (1, 14, 6, 'in'),        # just inside the top-left corner
          (2, 67, 87, 'in'),       # just inside the bottom-right corner
          (5, 40, 46, 'in'),       # blank across both tile seams
          (3, 12, 60, 'cut'),      # the top edge (row 12) cuts it
          (4, 30, 92, 'strip'),    # right strip, x 92 <= 98
          (0, 8, 30, 'strip'),     # top strip, y 8 >= 5
          (1, 70, 20, 'out')]}     # y 70 > 72 - 3

COMPARED = ['barcode_id', 'x', 'y', 'z', 'area', 'mean_intensity',
            'max_intensity', 'mean_distance', 'min_distance', 'global_x',
            'global_y']


def codewords():
    return np.array([[1 if b in on else 0 for b in range(4)]
                     for on in itertools.combinations(range(4), 2)])


def library_images(library):
    images = np.zeros((4, H, W), dtype=np.float32)
    words = codewords()
    for code, row, col, _ in SPOTS[library]:
        for bit in np.flatnonzero(words[code]):
            images[bit, row - 1:row + 2, col - 1:col + 2] = 100.0
    return images


def transforms_of(fov):
    return [transform.SimilarityTransform(
        translation=SHIFTS[fov].get(c, (0, 0))) for c in range(8)]


# ------------------------------------------------------------- light mocks

class _Codebook:

    def __init__(self, library):
        self.library = library

    def get_barcodes(self, ignoreBlanks=False):
        return codewords()

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


class _LightWarp:

    parameters = {}

    def get_transformation(self, fov, dataChannel=None):
        return transforms_of(fov)


class _LightPreprocess:

    def __init__(self, library):
        self.library = library
        self.parameters = {'warp_task': 'Warp'}

    def get_codebook(self):
        return _Codebook(self.library)

    def get_processed_image_set(self, fov, zIndex, chromaticCorrector=None):
        return library_images(self.library)


class _LightGlobalAlign:

    def fov_coordinate_array_to_global(self, fov, centroids):
        out = np.array(centroids, dtype=np.float64, copy=True)
        out[:, 1] += GLOBAL_OFFSET[0]
        out[:, 2] += GLOBAL_OFFSET[1]
        return out


class _LightSegment:
    """A full-frame decode mask: rows 20:50, cols 20:80."""

    def _load_mask_image(self, fov, zIndex):
        mask = np.zeros((H, W), dtype=np.int32)
        mask[20:50, 20:80] = 1
        return mask


class _LightDataSet:

    def __init__(self):
        self.tasks = {'Warp': _LightWarp(), 'PreA': _LightPreprocess('A'),
                      'PreB': _LightPreprocess('B'),
                      'GA': _LightGlobalAlign(), 'Seg': _LightSegment()}

    def get_image_dimensions(self):
        return (H, W)

    def get_data_organization(self):
        return _DataOrganization()

    def load_analysis_task(self, name):
        return self.tasks[name]


def light_decode(library, imageSpace, tilingFactor=None, fov=0, **extra):
    parameters = {'preprocess_task': 'Pre' + library, 'optimize_task': 'Opt',
                  'global_align_task': 'GA', 'crop_width': CROP,
                  'crop_in_image_space': imageSpace,
                  'tiling_factor': tilingFactor, 'write_decoded_fovs': [],
                  'write_decoded_z': []}
    parameters.update(extra)
    return decode.Decode(_LightDataSet(), parameters, 'Decode' + library)


def decode_plane(library, imageSpace, tilingFactor=None, fov=0, **extra):
    """Barcodes Decode emits for one plane of fov."""
    task = light_decode(library, imageSpace, tilingFactor, fov, **extra)
    task._pendingBarcodes = []      # _emit_barcodes buffers here
    with contextlib.redirect_stdout(io.StringIO()):
        task._process_independent_z_slice(
            fov, 0, None, np.ones(4), np.zeros(4),
            task.dataSet.load_analysis_task('Pre' + library),
            decoding.PixelBasedDecoder(_Codebook(library)))
    frames = [df for df in task._pendingBarcodes if len(df)]
    return pd.concat(frames, ignore_index=True)


def ordered(df):
    return df[COMPARED].sort_values(['barcode_id', 'x', 'y']).reset_index(
        drop=True).astype(np.float64)


def spot_rows(library, kinds):
    """(barcode_id, x, y) of the whole 3 x 3 spots of these kinds."""
    return sorted((code, float(col), float(row))
                  for code, row, col, kind in SPOTS[library] if kind in kinds)


# ------------------------------------------------------------------- tests

class CropWidthTests(unittest.TestCase):

    def test_one_integer_or_four(self):
        self.assertEqual(decoding.crop_widths(100), (100, 100, 100, 100))
        self.assertEqual(decoding.crop_widths([5, 3, 4, 6]), (5, 3, 4, 6))
        self.assertEqual(decoding.crop_widths((0, 0, 0, 0)), (0, 0, 0, 0))
        self.assertEqual(decoding.crop_widths(np.int64(7)), (7, 7, 7, 7))
        self.assertEqual(decode.crop_widths(2.0), (2, 2, 2, 2))
        for bad in ([1, 2, 3], -1, 1.5, True, '5', [1, 2, 3, -4],
                    [1, 2, 3, None], float('nan')):
            with self.assertRaises(ValueError, msg=repr(bad)):
                decoding.crop_widths(bad)

    def test_decode_refuses_a_bad_crop_width(self):
        with self.assertRaises(ValueError):
            light_decode('A', True, crop_width=[1, 2, 3])
        with self.assertRaises(ValueError):
            light_decode('A', True, adaptive_crop_channels='bits')


class BoundsTests(unittest.TestCase):

    def test_per_codebook_and_per_edge(self):
        ds = _LightDataSet()
        for (library, fov), want in BOUNDS.items():
            channels = None if library == 'all' else [
                READOUTS.index(b) for b in LIBRARY_BITS[library]]
            self.assertEqual(decode.adaptive_margins(ds, 'Warp', fov,
                                                     channels),
                             MARGINS[(library, fov)])
            self.assertEqual(tuple(decode.compute_crop_bounds(
                ds, 'Warp', fov, CROP, True, channels)), want)

    def test_task_bounds_follow_adaptive_crop_channels(self):
        for library in ('A', 'B'):
            task = light_decode(library, True)
            self.assertEqual(task.parameters['adaptive_crop_channels'],
                             'codebook')
            self.assertEqual(task._crop_bounds(0), BOUNDS[(library, 0)])
            self.assertEqual(
                decode.codebook_data_channels(task.dataSet,
                                              task.get_codebook()),
                [READOUTS.index(b) for b in LIBRARY_BITS[library]])
            every = light_decode(library, True, adaptive_crop_channels='all')
            self.assertEqual(every._crop_bounds(0), BOUNDS[('all', 0)])
        self.assertEqual(light_decode('A', True)._crop_bounds(1),
                         BOUNDS[('A', 1)])

    def test_one_integer_is_every_edge_and_adaptive_off(self):
        ds = _LightDataSet()
        self.assertEqual(
            decode.compute_crop_bounds(ds, 'Warp', 0, 6, True, [0, 1, 2, 3]),
            decode.compute_crop_bounds(ds, 'Warp', 0, [6, 6, 6, 6], True,
                                       [0, 1, 2, 3]))
        self.assertEqual(
            tuple(decode.compute_crop_bounds(ds, 'Warp', 0, CROP, False)),
            (5, H - 3, 4, W - 6))
        # old positional call (Optimize): every channel, one integer
        self.assertEqual(
            tuple(decode.compute_crop_bounds(ds, 'Warp', 0, 6, True)),
            (12, H - 7, 9, W - 15))


class _PolyWarp(warp.FiducialPolynomialWarp3D):
    """FiducialPolynomialWarp3D's own get_transformation over a given
    transformation table (no data set, no run)."""

    def __init__(self, table):
        self.table = table

    def get_transformation_table(self, fov):
        return self.table


def poly_table():
    rows = []
    for channel, plane, xs, ys, theta in [
            (0, 0, 0.0, 0.0, 0.0), (0, 1, 0.0, 0.0, 0.0),
            (0, 2, 0.0, 0.0, 0.0),
            (1, 0, -3.0, 0.0, 0.0), (1, 1, -5.0, 2.0, 0.0),
            (1, 2, -4.0, 0.0, 0.01),
            (2, 0, 30.0, -8.0, 0.0), (2, 1, 30.0, -8.0, 0.0),
            (2, 2, 30.0, -8.0, 0.0)]:
        rows.append(dict(dataChannel=channel, zIndex=plane, xshift=xs,
                         yshift=ys, rotation_radians=theta,
                         rotation_center_x=W / 2, rotation_center_y=H / 2))
    return pd.DataFrame(rows)


def poly_margins(table):
    """(top, bottom, left, right), independent of decode: every corner
    moved by input = c + R(p - c) + t."""
    left = right = top = bottom = 0.0
    for _, r in table.iterrows():
        c, s = math.cos(r.rotation_radians), math.sin(r.rotation_radians)
        cx, cy = r.rotation_center_x, r.rotation_center_y
        for px, py in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
            ix = cx + c * (px - cx) - s * (py - cy) + r.xshift
            iy = cy + s * (px - cx) + c * (py - cy) + r.yshift
            left, right = max(left, px - ix), max(right, ix - px)
            top, bottom = max(top, py - iy), max(bottom, iy - py)
    return tuple(int(math.ceil(v - 1e-9)) for v in (top, bottom, left,
                                                     right))


class _PolyDataSet(_LightDataSet):

    def __init__(self):
        super().__init__()
        self.tasks['Poly'] = _PolyWarp(poly_table())


class PolyWarpChannelTests(unittest.TestCase):

    def test_rows_of_the_codebook_channels_only(self):
        ds = _PolyDataSet()
        table = poly_table()
        poly = ds.load_analysis_task('Poly')
        self.assertEqual(len(decode.warp_transformations(poly, 0)), 9)
        self.assertEqual(len(decode.warp_transformations(poly, 0, [0, 1])),
                         6)
        mine = decode.adaptive_margins(ds, 'Poly', 0, [0, 1])
        self.assertEqual(mine, poly_margins(table[table.dataChannel < 2]))
        # channel 2's +30 px right and 8 px top strips do not count
        self.assertLess(mine[3], 30)
        self.assertEqual(decode.adaptive_margins(ds, 'Poly', 0),
                         poly_margins(table))
        self.assertGreaterEqual(decode.adaptive_margins(ds, 'Poly', 0)[3],
                                30)

    def test_missing_channel_refused(self):
        ds = _PolyDataSet()
        with self.assertRaisesRegex(ValueError, 'data channels \\[5\\]'):
            decode.adaptive_margins(ds, 'Poly', 0, [0, 5])
        with self.assertRaisesRegex(ValueError, 'data channels \\[9\\]'):
            decode.adaptive_margins(_LightDataSet(), 'Warp', 0, [1, 9])


class ImageSpaceDecodeTests(unittest.TestCase):

    def test_full_frame_xy_and_global(self):
        for library in ('A', 'B'):
            for tiling in (None, 2):
                df = decode_plane(library, True, tiling)
                rowStart, rowEnd, colStart, colEnd = BOUNDS[(library, 0)]
                inside = df[df['area'] == 9]
                got = sorted(zip(inside['barcode_id'].astype(int),
                                 inside['x'].astype(float),
                                 inside['y'].astype(float)))
                self.assertEqual(got, spot_rows(library, ('in',)),
                                 (library, tiling))
                np.testing.assert_allclose(
                    df['global_x'] - df['x'], GLOBAL_OFFSET[0])
                np.testing.assert_allclose(
                    df['global_y'] - df['y'], GLOBAL_OFFSET[1])
                # nothing outside the region, i.e. no strip or out spot
                self.assertTrue(df['x'].between(colStart, colEnd - 1).all())
                self.assertTrue(df['y'].between(rowStart, rowEnd - 1).all())
                cut = [s for s in SPOTS[library] if s[3] == 'cut']
                self.assertEqual(len(df), len(spot_rows(library, ('in',)))
                                 + len(cut))

    def test_cut_spots_keep_their_inside_part(self):
        df = decode_plane('A', True)
        cut = df[df['area'] < 9].sort_values('barcode_id')
        # (3, 5, 40): rows 5-6 of rows 4-6; (4, 40, 98): col 97 of 97-99
        np.testing.assert_allclose(
            cut[['barcode_id', 'x', 'y', 'area']].to_numpy(float),
            [[3, 40, 5.5, 6], [4, 97, 40, 3]])

    def test_tiled_matches_untiled(self):
        for library in ('A', 'B'):
            for imageSpace in (True, False):
                untiled = ordered(decode_plane(library, imageSpace))
                tiled = ordered(decode_plane(library, imageSpace, 2))
                self.assertEqual(len(tiled), len(untiled))
                np.testing.assert_allclose(tiled.to_numpy(),
                                           untiled.to_numpy(), atol=1e-4)

    def test_image_space_equals_barcode_space_inside_the_region(self):
        for library in ('A', 'B'):
            image = decode_plane(library, True)
            barcode = decode_plane(library, False)
            rowStart, rowEnd, colStart, colEnd = BOUNDS[(library, 0)]
            # barcode space: every in / cut / strip spot whole, nothing else
            self.assertEqual(
                sorted(zip(barcode['barcode_id'].astype(int),
                           barcode['x'].astype(float),
                           barcode['y'].astype(float))),
                spot_rows(library, ('in', 'cut', 'strip')))
            # components that the region edge does not cut are identical
            def whole(df):
                return df[(df['x'] - 1 >= colStart)
                          & (df['x'] + 1 <= colEnd - 1)
                          & (df['y'] - 1 >= rowStart)
                          & (df['y'] + 1 <= rowEnd - 1)]
            a, b = ordered(whole(image)), ordered(whole(barcode))
            self.assertEqual(len(a), len(spot_rows(library, ('in',))))
            self.assertEqual(len(a), len(b))
            np.testing.assert_allclose(a.to_numpy(), b.to_numpy(),
                                       atol=1e-4)

    def test_barcode_space_per_edge_filter(self):
        # crop_width [top, bottom, left, right] on the full frame, ends
        # included; the image-space bounds play no part
        df = decode_plane('A', False, crop_width=[6, 3, 4, 6])
        self.assertNotIn(5.0, df['y'].tolist())         # (3, 5, 40) dropped
        df = decode_plane('A', False, crop_width=[5, 3, 7, 6])
        self.assertNotIn(6.0, df['x'].tolist())         # (2, 40, 6) dropped
        self.assertIn(5.0, df['y'].tolist())
        df = decode_plane('A', False, crop_width=[5, 4, 4, 6])
        self.assertIn(68.0, df['y'].tolist())           # y 68 <= 72 - 4
        df = decode_plane('A', False, crop_width=[5, 5, 4, 6])
        self.assertNotIn(68.0, df['y'].tolist())        # y 68 > 72 - 5

    def test_segmentation_mask_is_cut_like_the_image(self):
        for imageSpace in (True, False):
            df = decode_plane('A', imageSpace, use_segmentation_mask='Seg')
            # only spots in mask rows 20:50, cols 20:80 decode
            self.assertTrue(df['x'].between(20, 79).all())
            self.assertTrue(df['y'].between(20, 49).all())
            whole = df[df['area'] == 9]
            self.assertEqual(
                sorted(zip(whole['barcode_id'].astype(int),
                           whole['x'].astype(float),
                           whole['y'].astype(float))),
                [(0, 50.0, 30.0), (3, 53.0, 35.0)])

    def test_empty_region_refused(self):
        task = light_decode('A', True, crop_width=[40, 40, 0, 0])
        self.assertEqual(task._crop_bounds(0), (40, 32, 9, 104))


class ExtractorCropTests(unittest.TestCase):
    """extract_barcodes_with_index on a 40 x 100 (rows x cols) image."""

    @staticmethod
    def extract(cropWidth, cropOffset=(0, 0)):
        decoded = np.full((40, 100), -1, dtype=np.int32)
        for code, row, col in ((0, 20, 80), (1, 37, 50), (2, 10, 97),
                               (3, 3, 30)):
            decoded[row - 1:row + 2, col - 1:col + 2] = code
        decoder = decoding.PixelBasedDecoder(_Codebook('A'))
        df = decoder.extract_barcodes_with_index(
            decoded, np.ones(decoded.shape, np.float32), None,
            np.zeros(decoded.shape, np.float32), 0, cropWidth, 0,
            globalAligner=_LightGlobalAlign(), minimumArea=2,
            crop_offset=cropOffset)
        return df.sort_values('barcode_id')

    def test_x_against_columns_and_y_against_rows(self):
        df = self.extract(5)
        # x 80 <= 100 - 5 is kept (the old filter compared it with 40 - 5);
        # y 37 > 40 - 5, x 97 > 100 - 5 and y 3 < 5 are dropped
        self.assertEqual(df['barcode_id'].tolist(), [0])
        np.testing.assert_allclose(df[['x', 'y']].to_numpy(), [[80, 20]])

    def test_per_edge(self):
        self.assertEqual(self.extract([0, 0, 0, 0])['barcode_id'].tolist(),
                         [0, 1, 2, 3])
        self.assertEqual(self.extract([4, 0, 0, 0])['barcode_id'].tolist(),
                         [0, 1, 2])
        # ends included: y 37 <= 40 - 3 and x 97 <= 100 - 3 stay
        self.assertEqual(self.extract([0, 3, 0, 3])['barcode_id'].tolist(),
                         [0, 1, 2, 3])
        self.assertEqual(self.extract([0, 4, 0, 4])['barcode_id'].tolist(),
                         [0, 3])
        self.assertEqual(self.extract([0, 0, 81, 0])['barcode_id'].tolist(),
                         [2])

    def test_offset_after_the_filter(self):
        df = self.extract([0, 0, 0, 4], cropOffset=(7, 11))
        # x 97 is dropped on the image's own frame (97 > 100 - 4), not on
        # the shifted 104; the rest move by (7, 11), global with them
        self.assertEqual(df['barcode_id'].tolist(), [0, 1, 3])
        np.testing.assert_allclose(df[['x', 'y']].to_numpy(),
                                   [[87, 31], [57, 48], [37, 14]])
        np.testing.assert_allclose(df['global_x'] - df['x'],
                                   GLOBAL_OFFSET[0])
        np.testing.assert_allclose(df['global_y'] - df['y'],
                                   GLOBAL_OFFSET[1])


# ------------------------------------------------ full tasks on a DataSet

class _DataSet(dataset.DataSet):

    def __init__(self, root):
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__('synthetic', dataHome=root, analysisHome=root)

    def get_fovs(self):
        return [0, 1]

    def get_z_positions(self):
        return Z_POSITIONS

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


class _Warp(_Task):

    def get_transformation(self, fov, dataChannel=None):
        return transforms_of(fov)


class _Preprocess(_Task):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self.parameters.setdefault('warp_task', 'Warp')

    def get_codebook(self):
        return _Codebook(self.parameters['library'])

    def get_processed_image_set(self, fov, zIndex, chromaticCorrector=None):
        return library_images(self.parameters['library'])


class _Optimize(_Task):

    def get_scale_factors(self):
        return np.ones(4)

    def get_backgrounds(self):
        return np.zeros(4)

    def get_previous_chromatic_corrector(self):
        return None


class _GlobalAlign(_Task, _LightGlobalAlign):
    pass


def make_dataset(root):
    ds = _DataSet(root)
    _Warp(ds, {}, 'Warp').save()
    for library in ('A', 'B'):
        _Preprocess(ds, {'library': library}, 'Pre' + library).save()
    _Optimize(ds, {}, 'Opt').save()
    _GlobalAlign(ds, {}, 'GA').save()
    return ds


def full_decode(ds, name, library, imageSpace, **extra):
    parameters = {'preprocess_task': 'Pre' + library, 'optimize_task': 'Opt',
                  'global_align_task': 'GA', 'crop_width': CROP,
                  'crop_in_image_space': imageSpace,
                  'write_decoded_fovs': [0], 'write_decoded_z': [0]}
    parameters.update(extra)
    task = decode.Decode(ds, parameters, name)
    task.save()
    return task


class RecordAndDedupTests(unittest.TestCase):

    def test_record_and_cropped_decoded_image(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = make_dataset(root)
            task = full_decode(ds, 'DecodeA', 'A', True)
            self.assertIsNone(task.get_crop_record(0))
            for fov in (0, 1):
                task.run(fov)
            record = task.get_crop_record(0)
            self.assertEqual(record['crop_bounds'], list(BOUNDS[('A', 0)]))
            self.assertEqual(record['crop_width'], CROP)
            self.assertEqual(record['adaptive_margin'],
                             list(MARGINS[('A', 0)]))
            self.assertEqual(record['data_channels'], [0, 1, 2, 3])
            self.assertEqual(record['adaptive_crop_channels'], 'codebook')
            self.assertIsNone(record['barcode_filter'])
            self.assertEqual(record['xy_frame'], 'full_frame')
            self.assertEqual(record['image_shape'], [H, W])
            self.assertEqual(task.get_crop_bounds(1), BOUNDS[('A', 1)])
            table = task.get_crop_bounds_table()
            self.assertEqual(table['fov'].tolist(), [0, 1])
            self.assertEqual(
                table.loc[0, ['row_start', 'row_end', 'col_start',
                              'col_end']].tolist(), list(BOUNDS[('A', 0)]))
            self.assertEqual(table.loc[1, 'margin_right'], 20)
            self.assertEqual(table.loc[0, 'n_data_channels'], 4)
            # the decoded image is the cropped region
            rowStart, rowEnd, colStart, colEnd = BOUNDS[('A', 0)]
            image = task.get_decoded_image(0, 0)
            self.assertEqual(image.shape, (rowEnd - rowStart,
                                           colEnd - colStart))
            # the stored barcodes are full-frame, both planes
            barcodes = task.get_barcode_database().get_barcodes(0)
            self.assertEqual(sorted(barcodes['z'].unique().tolist()), [0, 1])
            whole = barcodes[barcodes['area'] == 9]
            got = sorted(zip(whole['barcode_id'].astype(int),
                             whole['x'].astype(float),
                             whole['y'].astype(float)))
            self.assertEqual(got, sorted(spot_rows('A', ('in',)) * 2))
            self.assertFalse(deduplicate.decode_xy_is_cropped(task, 0))

    def test_barcode_space_record(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = make_dataset(root)
            task = full_decode(ds, 'DecodeB', 'B', False)
            task.run(0)
            record = task.get_crop_record(0)
            self.assertEqual(record['crop_bounds'], list(BOUNDS[('B', 0)]))
            self.assertEqual(record['barcode_filter'], [5, H - 3, 4, W - 6])
            barcodes = task.get_barcode_database().get_barcodes(0)
            self.assertEqual(
                sorted(set(zip(barcodes['barcode_id'].astype(int),
                               barcodes['x'].astype(float),
                               barcodes['y'].astype(float)))),
                spot_rows('B', ('in', 'cut', 'strip')))

    def test_empty_region_raises_in_image_space(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = make_dataset(root)
            task = full_decode(ds, 'DecodeA', 'A', True,
                               crop_width=[40, 40, 0, 0])
            with self.assertRaisesRegex(ValueError, 'nothing left'):
                task.run(0)

    def test_dedup_crop_uses_the_libraries_own_bounds(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = make_dataset(root)
            outside = {'A': 2, 'B': 2}     # strip spots per plane
            for library in ('A', 'B'):
                for imageSpace in (True, False):
                    name = 'Decode%s%s' % (library, 'IS' if imageSpace
                                           else 'BS')
                    decodeTask = full_decode(ds, name, library, imageSpace)
                    dedup = deduplicate.DeduplicateBarcodes(
                        ds, {'decode_task': name, 'crop_to_valid_box': True},
                        'Dedup' + name)
                    dedup.save()
                    decodeTask.run(0)
                    dedup.run(0)
                    summary = dedup.get_summary(0)
                    self.assertEqual(summary['valid_box'],
                                     list(BOUNDS[(library, 0)]), name)
                    want = 0 if imageSpace else outside[library] * 2
                    self.assertEqual(summary['n_outside_valid_box'], want,
                                     name)
                    molecules = dedup.get_barcode_database().get_barcodes(0)
                    rowStart, rowEnd, colStart, colEnd = BOUNDS[(library, 0)]
                    self.assertTrue(
                        molecules['x'].between(colStart, colEnd).all())
                    self.assertTrue(
                        molecules['y'].between(rowStart, rowEnd).all())
                    # every in / cut spot is one molecule over the 2 planes
                    nWant = len(spot_rows(library, ('in', 'cut')))
                    self.assertEqual(len(molecules), nWant, name)

    def test_legacy_image_space_output_is_shifted(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = make_dataset(root)
            task = full_decode(ds, 'DecodeA', 'A', True)
            # no record: written before x, y were full-frame
            self.assertTrue(deduplicate.decode_xy_is_cropped(task, 0))
            barcodeTask = full_decode(ds, 'DecodeBS', 'A', False)
            self.assertFalse(deduplicate.decode_xy_is_cropped(barcodeTask, 0))
            task.run(0)
            self.assertFalse(deduplicate.decode_xy_is_cropped(task, 0))
            self.assertTrue(deduplicate.decode_xy_is_cropped(task, 1))


class SavedParameterTests(unittest.TestCase):
    """A task.json written before adaptive_crop_channels and per-edge
    crop_width loads, and generate's comparison (saved vs rebuilt from the
    analysis json) finds identical parameters."""

    OLD = {"preprocess_task": "PreA", "optimize_task": "Opt",
           "global_align_task": "GA", "minimum_area": 2, "crop_width": 100,
           "distance_threshold": 0.65, "magnitude_threshold": 20,
           "adaptive_crop": True, "crop_in_image_space": False,
           "remove_z_duplicated_barcodes": False,
           "resumable_z_decoding": True, "per_z_slice_num_threads": 16,
           "torch_num_threads": None, "decode_chunk_size": 8192,
           "write_decoded_images": True, "decoded_image_format": "tif",
           "tiling_overlap": 20, "single_fov_optimization": False,
           "use_segmentation_mask": False, "use_gpu": False,
           "write_unique_id_images": False, "tiling_factor": None,
           "distance_metric": "dot_product", "softmax_temperature": 0.15,
           "tiling_num_threads": 1, "decode_z_index": None, "fovs": None,
           "extract_intensity_traces": False, "write_images_seed": 1,
           "write_decoded_fovs": [0], "write_decoded_z": [0]}

    def _write_old(self, ds, name, parameters):
        saved = dict(parameters, merlin_version='0.1.6',
                     module='merlin.analysis.decode', **{'class': 'Decode'})
        path = os.path.join(ds.get_analysis_subdirectory(name, 'tasks'),
                            'task.json')
        with open(path, 'w') as fh:
            json.dump(saved, fh, indent=4)

    def test_old_task_json_loads_and_regenerates(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = make_dataset(root)
            self._write_old(ds, 'OldDecode', self.OLD)
            loaded = ds.load_analysis_task('OldDecode')
            self.assertEqual(loaded.parameters['adaptive_crop_channels'],
                             'codebook')
            self.assertEqual(loaded.parameters['crop_width'], 100)
            for key, value in self.OLD.items():
                self.assertEqual(loaded.parameters[key], value, key)
            # generate: rebuild from the analysis json and save; raises
            # AnalysisAlreadyExistsException if the parameters differ
            decode.Decode(ds, dict(self.OLD), 'OldDecode').save()

    def test_per_edge_crop_width_round_trips(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = make_dataset(root)
            parameters = dict(self.OLD, crop_width=[5, 3, 4, 6],
                              crop_in_image_space=True,
                              adaptive_crop_channels='all')
            decode.Decode(ds, parameters, 'NewDecode').save()
            loaded = ds.load_analysis_task('NewDecode')
            self.assertEqual(loaded.parameters['crop_width'], [5, 3, 4, 6])
            self.assertEqual(loaded._crop_bounds(0), BOUNDS[('all', 0)])
            decode.Decode(ds, dict(parameters), 'NewDecode').save()


if __name__ == '__main__':
    unittest.main()
