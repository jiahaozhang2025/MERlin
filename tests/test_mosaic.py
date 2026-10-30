"""GenerateMosaic: frame channels, area-averaged placement, overlap averaging and channel order, on a stand-in dataset.
Run with `python -m unittest tests/test_mosaic.py`."""
import os
import tempfile
import unittest

import numpy as np
import tifffile

from merlin.analysis.generatemosaic import GenerateMosaic, frame_source


class FakeOrganization:
    names = ['DAPI', 'GFP473', 'bit1']

    def get_image_filename(self, channel, fov):
        return 'movie_%d_%d' % (channel, fov)

    def get_data_channel_index(self, name):
        return self.names.index(name)

    def get_data_channel_name(self, channel):
        return self.names[channel]

    def get_data_channels(self):
        return np.arange(len(self.names))


class FakeDataSet:
    """load_image returns a constant image whose value encodes (file, frame); get_raw_image reads frame 10 + z index."""

    def __init__(self, size=64, mpp=1.0, fovs=(0,)):
        self.size, self.mpp, self.fovs = size, mpp, list(fovs)
        self.org = FakeOrganization()

    def get_z_positions(self):
        return [0.0, 1.5, 3.0]

    def get_data_organization(self):
        return self.org

    def load_image(self, path, frame):
        channel, fov = (int(x) for x in path.split('_')[1:])
        return np.full((self.size, self.size), 1000 * channel + 100 * fov + frame, np.uint16)

    def get_raw_image(self, channel, fov, zPosition):
        k = self.get_z_positions().index(zPosition)
        return self.load_image(self.org.get_image_filename(channel, fov), 10 + k)

    def get_microns_per_pixel(self):
        return self.mpp

    def get_fovs(self):
        return self.fovs


class FakeAlign:
    def __init__(self, offsets, mpp):
        self.offsets, self.mpp = offsets, mpp

    def fov_to_global_transform(self, fov):
        x, y = self.offsets[fov]
        return np.float32([[self.mpp, 0, x], [0, self.mpp, y], [0, 0, 1]])


def make_task(dataSet, mosaicMpp, parameters=None):
    task = object.__new__(GenerateMosaic)
    task.dataSet = dataSet
    task.parameters = dict(parameters or {})
    task.mosaicMicronsPerPixel = mosaicMpp
    return task


class FrameSourceTest(unittest.TestCase):

    def test_routes_requested_frames_and_restores(self):
        ds = FakeDataSet()
        with frame_source(ds, 1, [1, 3]):
            self.assertEqual(ds.get_raw_image(1, 0, 0.0)[0, 0], 1001)    # z index 0 -> frame 1
            self.assertEqual(ds.get_raw_image(1, 0, 1.5)[0, 0], 1003)    # z index 1 -> frame 3
            self.assertEqual(ds.get_raw_image(0, 0, 1.5)[0, 0], 11)      # other channel unchanged
        self.assertNotIn('get_raw_image', vars(ds))
        self.assertEqual(ds.get_raw_image(1, 0, 0.0)[0, 0], 1010)

    def test_more_frames_than_z_positions_is_refused(self):
        with self.assertRaises(ValueError):
            with frame_source(FakeDataSet(), 1, [0, 1, 2, 3]):
                pass


class PlacementTest(unittest.TestCase):

    def test_area_downsampling_keeps_intensity_and_excludes_zero_margin(self):
        ds = FakeDataSet(size=64, mpp=1.0)
        task = make_task(ds, mosaicMpp=4.0)
        align = FakeAlign({0: (8.0, 4.0)}, 1.0)
        image = np.full((64, 64), 200, np.float32)
        image[:, :16] = 0                                  # registration margin without data
        warped, weight = task._transform_image_to_mosaic(image, 0, align, (0, 0, 80, 80), (20, 20))
        inside = weight > 0.99
        self.assertTrue(inside.any())
        np.testing.assert_allclose(warped[inside] / weight[inside], 200, rtol=1e-4)
        # footprint: x 8..72 um -> mosaic cols 2..18; the zeroed left 16 px (4 mosaic px) carry no weight
        self.assertLess(weight[5:20, 2:5].max(), 0.01)
        self.assertGreater(weight[5:15, 7:17].min(), 0.99)

    def test_overlap_is_averaged(self):
        ds = FakeDataSet(size=40, mpp=1.0)
        task = make_task(ds, mosaicMpp=2.0)
        align = FakeAlign({0: (0.0, 0.0), 1: (20.0, 0.0)}, 1.0)
        total = np.zeros((20, 30), np.float32); weights = np.zeros_like(total)
        for fov, value in ((0, 100.0), (1, 300.0)):
            w, c = task._transform_image_to_mosaic(np.full((40, 40), value, np.float32), fov, align,
                                                   (0, 0, 60, 40), (30, 20))
            total += w; weights += c
        mosaic = total / np.maximum(weights, 1e-6)
        self.assertAlmostEqual(float(mosaic[10, 5]), 100.0, places=2)    # fov 0 only
        self.assertAlmostEqual(float(mosaic[10, 15]), 200.0, places=2)   # overlap
        self.assertAlmostEqual(float(mosaic[10, 25]), 300.0, places=2)   # fov 1 only


class ChannelOrderTest(unittest.TestCase):

    def test_frame_channels_mix_with_data_channels(self):
        task = make_task(FakeDataSet(), 1.0, {
            'data_channels': ['DAPI', 'Stain', 'GFP473'],
            'frame_channels': {'Stain': {'data_channel': 'GFP473', 'frames': [1, 3]},
                               'Extra': {'data_channel': 0, 'frames': [2]}}})
        self.assertEqual(task._mosaic_channels(), [('DAPI', 0, None), ('Stain', 1, [1, 3]),
                                                   ('GFP473', 1, None), ('Extra', 0, [2])])

    def test_default_is_every_data_channel(self):
        task = make_task(FakeDataSet(), 1.0)
        self.assertEqual([c[0] for c in task._mosaic_channels()], ['DAPI', 'GFP473', 'bit1'])


class FakeWarp:
    """get_aligned_image: constant image, value = 100 * (channel + 1) + z (so z slices and channels are told apart)."""

    def __init__(self, size):
        self.size = size

    def get_aligned_image(self, fov, channel, z, chromaticCorrector=None):
        return np.full((self.size, self.size), 100 * (channel + 1) + z, np.uint16)


class FakeWritableDataSet(FakeDataSet):
    def __init__(self, folder, **kw):
        super().__init__(**kw)
        self.folder = folder
        self.align = FakeAlign({0: (0.0, 0.0), 1: (64.0, 0.0)}, kw.get('mpp', 1.0))
        self.align.get_global_extent = lambda: (0.0, 0.0, 128.0, 64.0)
        self.warp = FakeWarp(kw.get('size', 64))

    def load_analysis_task(self, name):
        return {'GlobalAlign': self.align, 'Warp': self.warp}[name]

    def save_numpy_txt_analysis_result(self, *args, **kw):
        pass

    def get_analysis_subdirectory(self, task, subdirectory=None):
        return self.folder

    @staticmethod
    def analysis_tiff_description(sliceCount, frameCount):
        return {'ImageJ': '1.47a\n', 'images': sliceCount * frameCount, 'channels': 1, 'slices': sliceCount,
                'frames': frameCount, 'hyperstack': True, 'loop': False}

    def writer_for_analysis_images(self, task, name, imageIndex=None, imagej=False):
        return tifffile.TiffWriter(os.path.join(self.folder, name + '.tif'), imagej=imagej)


class OutputTest(unittest.TestCase):

    def run_task(self, **parameters):
        folder = tempfile.mkdtemp()
        ds = FakeWritableDataSet(folder, size=64, mpp=1.0, fovs=(0, 1))
        params = {'global_align_task': 'GlobalAlign', 'warp_task': 'Warp', 'separate_files': False,
                  'fov_crop_width': 0, 'draw_fov_labels': False, 'data_channels': ['DAPI', 'GFP473'],
                  'channel_names': {'GFP473': 'polyT'}}
        params.update(parameters)
        make_task(ds, 4.0, params)._run_analysis()
        return folder

    def test_z_slices(self):
        task = make_task(FakeDataSet(), 1.0, {'z_index': [1, 'maximum_projection']})
        self.assertEqual(task._z_slices(), [('z1', 1, False), ('zmax', 0, True)])
        self.assertEqual([s[0] for s in make_task(FakeDataSet(), 1.0)._z_slices()], ['z0', 'z1', 'z2'])

    def test_one_file_per_z_and_channel_named_and_compressed(self):
        folder = self.run_task(z_index=[0, 2], separate_files='channel', tiff_compression='zlib', write_preview=True)
        files = sorted(f for f in os.listdir(folder) if f.endswith('.tif'))
        self.assertEqual(files, ['mosaic_z0_DAPI.tif', 'mosaic_z0_polyT.tif', 'mosaic_z2_DAPI.tif', 'mosaic_z2_polyT.tif'])
        with tifffile.TiffFile(os.path.join(folder, 'mosaic_z2_polyT.tif')) as t:
            self.assertEqual(t.pages[0].compression, tifffile.COMPRESSION.ADOBE_DEFLATE)
            self.assertEqual(int(np.median(t.asarray()[t.asarray() > 0])), 202)   # GFP473 = channel 1, z 2
        self.assertIn('mosaic_preview_z2_composite.png', os.listdir(folder))

    def test_one_file_per_z_with_every_channel(self):
        folder = self.run_task(z_index=[1, 'maximum_projection'], separate_files='z')
        self.assertEqual(sorted(f for f in os.listdir(folder) if f.endswith('.tif')), ['mosaic_z1.tif', 'mosaic_zmax.tif'])
        a = tifffile.imread(os.path.join(folder, 'mosaic_zmax.tif'))
        self.assertEqual(a.shape[0], 2)
        self.assertEqual([int(np.median(c[c > 0])) for c in a], [102, 202])          # max over z 0-2

    def test_single_file_default(self):
        folder = self.run_task(z_index=1)
        a = tifffile.imread(os.path.join(folder, 'mosaic.tif'))
        self.assertEqual(a.shape[0], 2)
        self.assertEqual([int(np.median(c[c > 0])) for c in a], [101, 201])


if __name__ == '__main__':
    unittest.main()
