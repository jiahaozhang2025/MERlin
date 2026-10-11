"""Decode's tiled branch against its untiled path, both crop modes.

Decode._process_independent_z_slice runs the real PixelBasedDecoder on a
synthetic 96 x 96 frame: 3 x 3 spots with exact codewords at known raw pixel
positions, and a warp task whose shifts leave invalid margins of 10 (top),
16 (bottom), 20 (left) and 6 (right) px, wider than crop_width on three
edges. compute_crop_bounds then gives the valid region rows 10:80,
cols 20:90 for crop_width 6 and 0 alike. It is square, so tiles split it
evenly and the extractor's own edge filter cannot drop anything.

Image space (crop_in_image_space True): every spot inside the region is
kept, and x, y are full-frame raw pixels (the crop corner (colStart, rowStart)
is added back), so global x, y are raw pixel + the fov's global offset.
tiling_factor 2 has to give the same barcodes as no tiling. Barcode space
(False): the frame is not cropped, spots within crop_width of the frame edge
are dropped, and x, y are full-frame as decoded, tiled or not. The six bits
are data channels 0-5 and the codebook uses all six, so the adaptive margin
over the codebook's channels (the default) is the one over every channel.

Run with `python -m unittest tests/test_decode_tiling_crop.py`.
"""
import contextlib
import io
import itertools
import threading
import unittest

import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis import decode
from merlin.util import decoding

FRAME = 96
BITS = 6
# invalid margins (top, bottom, left, right) the warp below leaves
MARGINS = (10, 16, 20, 6)
GLOBAL_OFFSET = (5000.0, 7000.0)     # (x, y) of fov 0's raw pixel (0, 0)

# (codeword index, raw row, raw col) of each spot's centre
SPOTS = [
    (0, 12, 22),    # 2 px inside the region's top-left corner
    (1, 77, 87),    # 2 px inside its bottom-right corner
    (2, 50, 86),    # 3 px inside its right edge
    (3, 40, 50),    # interior
    (4, 45, 70),    # straddles the tile seam at local row 35
    (5, 70, 30),    # interior
    (6, 8, 50),     # in the top margin: outside the region, inside crop_width
    (7, 40, 12),    # in the left margin: outside the region, inside crop_width
    (0, 3, 3),      # within crop_width of the frame corner
]


def codewords():
    return np.array([[1 if b in on else 0 for b in range(BITS)]
                     for on in itertools.combinations(range(BITS), 2)])[:8]


class _Codebook:

    def get_barcodes(self, ignoreBlanks=False):
        return codewords()

    def get_bit_names(self):
        return ['bit%d' % b for b in range(BITS)]


class _DataOrganization:

    def get_data_channel_for_bit(self, bitName):
        return int(bitName[3:])


class _Warp:

    parameters = {}

    def get_transformation(self, fov):
        # output (r, c) samples input (r + ty, c + tx): tx -20 invalidates the
        # left 20 columns, ty -10 the top 10 rows, ty +16 the bottom 16, tx +6
        # the right 6 columns
        return [transform.SimilarityTransform(),
                transform.SimilarityTransform(translation=(-20, -10)),
                transform.SimilarityTransform(translation=(0, 16)),
                transform.SimilarityTransform(translation=(6, 0)),
                transform.SimilarityTransform(),
                transform.SimilarityTransform()]


class _Preprocess:

    parameters = {'warp_task': 'Warp'}

    def get_codebook(self):
        return _Codebook()

    def get_processed_image_set(self, fov, zIndex, chromaticCorrector=None):
        images = np.zeros((BITS, FRAME, FRAME), dtype=np.float32)
        words = codewords()
        for code, row, col in SPOTS:
            for bit in np.flatnonzero(words[code]):
                images[bit, row - 1:row + 2, col - 1:col + 2] = 100.0
        return images


class _GlobalAlign:

    def fov_coordinate_array_to_global(self, fov, centroids):
        out = np.array(centroids, dtype=np.float64, copy=True)
        out[:, 1] += GLOBAL_OFFSET[0]
        out[:, 2] += GLOBAL_OFFSET[1]
        return out


class _DataSet:

    def __init__(self):
        self.tasks = {'Warp': _Warp(), 'Pre': _Preprocess(),
                      'GA': _GlobalAlign()}

    def get_image_dimensions(self):
        return (FRAME, FRAME)

    def get_data_organization(self):
        return _DataOrganization()

    def load_analysis_task(self, name):
        return self.tasks[name]


def decode_plane(cropInImageSpace, cropWidth, tilingFactor):
    """Barcodes Decode emits for fov 0, z 0."""
    task = decode.Decode(_DataSet(), {
        'preprocess_task': 'Pre', 'optimize_task': 'Opt',
        'global_align_task': 'GA', 'crop_width': cropWidth,
        'crop_in_image_space': cropInImageSpace,
        'tiling_factor': tilingFactor, 'write_decoded_fovs': [],
        'write_decoded_z': []}, 'Decode')
    task._pendingBarcodes = []      # _emit_barcodes buffers here
    with contextlib.redirect_stdout(io.StringIO()):
        task._process_independent_z_slice(
            0, 0, None, np.ones(BITS), np.zeros(BITS), _Preprocess(),
            decoding.PixelBasedDecoder(_Codebook()))
    frames = [df for df in task._pendingBarcodes if len(df)]
    if not frames:
        return pd.DataFrame(columns=['barcode_id', 'x', 'y', 'global_x',
                                     'global_y'])
    return pd.concat(frames, ignore_index=True)


def ordered(df):
    columns = ['barcode_id', 'x', 'y', 'z', 'area', 'mean_intensity',
               'mean_distance', 'global_x', 'global_y']
    return df[columns].sort_values(['barcode_id', 'x', 'y']).reset_index(
        drop=True).astype(np.float64)


def expected(cropInImageSpace, cropWidth):
    """(barcode_id, x, y, global_x, global_y) of the spots Decode keeps."""
    top, bottom, left, right = MARGINS
    rowStart, rowEnd = max(top, cropWidth), FRAME - max(bottom, cropWidth)
    colStart, colEnd = max(left, cropWidth), FRAME - max(right, cropWidth)
    rows = []
    for code, row, col in SPOTS:
        if cropInImageSpace:
            if not (rowStart < row < rowEnd - 1
                    and colStart < col < colEnd - 1):
                continue
            x, y = col, row
        else:
            if not (cropWidth <= col <= FRAME - cropWidth
                    and cropWidth <= row <= FRAME - cropWidth):
                continue
            x, y = col, row
        rows.append((code, x, y, col + GLOBAL_OFFSET[0],
                     row + GLOBAL_OFFSET[1]))
    return pd.DataFrame(rows, columns=['barcode_id', 'x', 'y', 'global_x',
                                       'global_y']).sort_values(
        ['barcode_id', 'x', 'y']).reset_index(drop=True).astype(np.float64)


class TilingCropTests(unittest.TestCase):

    def test_region_is_wider_than_crop_width(self):
        for cropWidth in (6, 0):
            bounds = decode.compute_crop_bounds(_DataSet(), 'Warp', 0,
                                                cropWidth, True)
            self.assertEqual(tuple(bounds), (10, 80, 20, 90))

    def _check(self, cropInImageSpace, cropWidth):
        want = expected(cropInImageSpace, cropWidth)
        cols = ['barcode_id', 'x', 'y', 'global_x', 'global_y']
        untiled = ordered(decode_plane(cropInImageSpace, cropWidth, None))
        tiled = ordered(decode_plane(cropInImageSpace, cropWidth, 2))
        # the untiled path is the reference: right spots, right coordinates
        np.testing.assert_allclose(untiled[cols].to_numpy(),
                                   want.to_numpy())
        # tiling must not change which barcodes are kept or where they are
        self.assertEqual(len(tiled), len(untiled))
        np.testing.assert_allclose(tiled.to_numpy(), untiled.to_numpy())

    def test_image_space_tiled_matches_untiled(self):
        self._check(True, 6)

    def test_image_space_zero_crop_width_tiled_matches_untiled(self):
        self._check(True, 0)

    def test_barcode_space_tiled_matches_untiled(self):
        self._check(False, 6)

    def test_image_space_keeps_spots_near_the_region_edge(self):
        # the three spots within crop_width of the cropped frame's edge, at
        # their full-frame raw pixels, and their raw-pixel global coordinates
        tiled = decode_plane(True, 6, 2)
        self.assertEqual(sorted(tiled['barcode_id'].tolist()),
                         [0, 1, 2, 3, 4, 5])
        nearEdge = tiled[tiled['barcode_id'].isin([0, 1, 2])].sort_values(
            'barcode_id')
        np.testing.assert_allclose(nearEdge[['x', 'y']].to_numpy(),
                                   [[22, 12], [87, 77], [86, 50]])
        np.testing.assert_allclose(
            nearEdge[['global_x', 'global_y']].to_numpy(),
            [[5022, 7012], [5087, 7077], [5086, 7050]])

    def test_image_space_global_is_raw_pixel(self):
        # x, y = raw pixel (the cropped image's x, y plus (colStart,
        # rowStart) = (20, 10), not plus crop_width) and global = raw pixel
        # + fov offset for every spot
        raw = {code: (col, row) for code, row, col in SPOTS[:6]}
        for tilingFactor in (None, 2):
            df = decode_plane(True, 6, tilingFactor)
            for _, bc in df.iterrows():
                col, row = raw[int(bc['barcode_id'])]
                self.assertEqual((bc['x'], bc['y']), (col, row))
                self.assertEqual(
                    (bc['global_x'] - GLOBAL_OFFSET[0],
                     bc['global_y'] - GLOBAL_OFFSET[1]), (col, row))

    def test_barcode_space_keeps_the_warp_margin(self):
        # barcode space drops only crop_width: the spots in the warp margin
        # stay, the corner spot goes
        tiled = decode_plane(False, 6, 2)
        self.assertEqual(len(tiled), 8)
        margin = tiled[tiled['barcode_id'].isin([6, 7])].sort_values(
            'barcode_id')
        np.testing.assert_allclose(
            margin[['x', 'y', 'global_x', 'global_y']].to_numpy(),
            [[50, 8, 5050, 7008], [12, 40, 5012, 7040]])


if __name__ == '__main__':
    unittest.main()
