import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
import cv2

from merlin.core import analysistask


ExtentTuple = Tuple[float, float, float, float]

# Composite preview colours (BGR, cv2 order) for mosaic channels 1, 2, 3.
PREVIEW_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))


@contextlib.contextmanager
def frame_source(dataSet, dataChannel: int, frames):
    """Route dataSet.get_raw_image(dataChannel, fov, z) to frame
    frames[k] of that channel's image file, k being the index of the z
    position asked for (the nearest one), so the warp task registers
    arbitrary frames of a channel's movie exactly as it registers the
    channel itself. Other channels read as usual. The route is an instance
    attribute, removed again on exit; while it is in place every reader of
    the dataset sees it, so use it from one channel phase at a time."""
    zPositions = np.asarray(dataSet.get_z_positions(), dtype=float)
    if len(frames) > len(zPositions):
        raise ValueError('%d frames for data channel %d but only %d z '
                         'positions to index them by'
                         % (len(frames), dataChannel, len(zPositions)))
    original = dataSet.get_raw_image
    imagePath = dataSet.get_data_organization().get_image_filename

    def get_raw_image(channel, fov, zPosition):
        if channel != dataChannel:
            return original(channel, fov, zPosition)
        k = int(np.argmin(np.abs(zPositions - zPosition)))
        return dataSet.load_image(imagePath(dataChannel, fov), frames[k])

    dataSet.get_raw_image = get_raw_image
    try:
        yield
    finally:
        del dataSet.get_raw_image


class GenerateMosaic(analysistask.AnalysisTask):

    """
    An analysis task that generates mosaic images by compiling different
    field of views.

    Each mosaic channel is one data channel (data_channels: names or
    indexes) or a frame channel (frame_channels: {name: {"data_channel": c,
    "frames": [...]}}, the listed frames of channel c's image file, e.g. a
    fiducial or label frame that the data organization does not list as a
    data channel). data_channels may name frame channels to set the order;
    frame channels it does not name follow the data channels. Every image
    comes from the warp task (registered, optionally chromatically
    corrected); a frame channel is registered with its data channel's
    transformation and always max-projected over its frames.

    z_index: an index, 'maximum_projection', a list of those, or absent
    (every z index). Slices are named z<index> and zmax.
    separate_files: false (default) = one file, mosaic.tif, holding every
    channel x z slice; 'z' = one file per z slice with every channel,
    mosaic_z<i>.tif; 'channel' or true = one file per z slice and channel,
    mosaic_z<i>_<channel>.tif. Channel names are the data organization's
    channelName (or the frame channel's key); channel_names {name: output
    name} renames them. tiff_compression (default none, uncompressed):
    e.g. 'zlib', lossless.

    Fovs are downsampled to the mosaic pixel size by area averaging before
    they are placed, and overlapping fovs are averaged (coverage-weighted).
    flat_field (default false): divide each channel by its flat field, the
    per-pixel median over up to flat_field_fovs fovs (default 80) smoothed
    with a Gaussian of flat_field_sigma px (default 50), mean 1.
    fovs (default all), num_threads (default 1, fovs prepared in
    parallel), write_preview (default false: per z slice, 8-bit PNGs
    mosaic_preview_<z slice>_<channel> and a composite of the first three
    channels as blue, green, red, in the task's images folder).
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'microns_per_pixel' not in self.parameters:
            self.parameters['microns_per_pixel'] = 3
        if 'fov_crop_width' not in self.parameters:
            self.parameters['fov_crop_width'] = 0
        if 'separate_files' not in self.parameters:
            self.parameters['separate_files'] = False
        if 'draw_fov_labels' not in self.parameters:
            self.parameters['draw_fov_labels'] = False
        # frame_channels, channel_names, flat_field(_fovs, _sigma), fovs,
        # num_threads, write_preview and tiff_compression are read with
        # .get() so that saved tasks regenerate unchanged.

        if self.parameters['microns_per_pixel'] == 'full_resolution':
            self.mosaicMicronsPerPixel = self.dataSet.get_microns_per_pixel()
        else:
            self.mosaicMicronsPerPixel = self.parameters['microns_per_pixel']

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['global_align_task'],
                self.parameters['warp_task']]

    def get_mosaic(self) -> np.ndarray:
        """Get the mosaic generated by this analysis task.

        Returns:
            a 5-dimensional array containing the mosaic. The images are arranged
            as [channel, zIndex, 1, x, y]. The order of the channels is as
            specified in the provided parameters file or in the data
            organization if no data channels are specified.
        """
        return self.dataSet.get_analysis_image_set(self, 'mosaic')

    def _micron_to_mosaic_pixel(self, micronCoordinates,
                                micronExtents) -> np.ndarray:
        """Calculates the mosaic coordinates in pixels from the specified
        global coordinates.
        """
        return np.matmul(self._micron_to_mosaic_transform(micronExtents),
                         np.append(micronCoordinates, 1)).astype(np.int32)[:2]

    def _micron_to_mosaic_transform(self, micronExtents: ExtentTuple) \
            -> np.ndarray:
        s = 1/self.mosaicMicronsPerPixel
        return np.float32(
                [[s*1, 0, -s*micronExtents[0]],
                 [0, s*1, -s*micronExtents[1]],
                 [0, 0, 1]])

    def _transform_image_to_mosaic(
            self, inputImage: np.ndarray, fov: int, alignTask,
            micronExtents: ExtentTuple, mosaicDimensions: Tuple[int, int])\
            -> Tuple[np.ndarray, np.ndarray]:
        """The fov image and its coverage (1 where the image has data, 0
        in zeroed margins) in mosaic pixels. A fov larger than its mosaic
        footprint is first shrunk by area averaging, so the warp samples
        a smooth image instead of aliasing single pixels."""
        transform = np.matmul(self._micron_to_mosaic_transform(micronExtents),
                              alignTask.fov_to_global_transform(fov))
        image = inputImage.astype(np.float32)
        coverage = (inputImage > 0).astype(np.float32)
        scale = self.dataSet.get_microns_per_pixel() \
            / self.mosaicMicronsPerPixel
        if scale < 1:
            size = (max(1, int(round(image.shape[1] * scale))),
                    max(1, int(round(image.shape[0] * scale))))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            coverage = cv2.resize(coverage, size,
                                  interpolation=cv2.INTER_AREA)
            transform = np.matmul(transform, np.float32(
                [[inputImage.shape[1] / size[0], 0, 0],
                 [0, inputImage.shape[0] / size[1], 0],
                 [0, 0, 1]]))
        warped = cv2.warpAffine(image, transform[:2, :], mosaicDimensions)
        weight = cv2.warpAffine(coverage, transform[:2, :], mosaicDimensions)
        return warped, weight

    def _mosaic_channels(self):
        """[(name, dataChannel, frames or None)] in mosaic order."""
        dataOrganization = self.dataSet.get_data_organization()
        frameChannels = self.parameters.get('frame_channels') or {}

        def resolve(x):
            if isinstance(x, str):
                return dataOrganization.get_data_channel_index(x)
            return int(x)

        def entry(x):
            if isinstance(x, str) and x in frameChannels:
                spec = frameChannels[x]
                return x, resolve(spec['data_channel']), \
                    [int(f) for f in spec['frames']]
            c = resolve(x)
            return dataOrganization.get_data_channel_name(c), c, None

        names = self.parameters.get('channel_names') or {}

        if 'data_channels' in self.parameters:
            requested = self.parameters['data_channels']
            if isinstance(requested, (str, int)):
                requested = [requested]
        else:
            requested = list(dataOrganization.get_data_channels())
        channels = [entry(x) for x in requested]
        named = {c[0] for c in channels}
        channels += [entry(x) for x in frameChannels if x not in named]
        return [(names.get(n, n), c, f) for n, c, f in channels]

    def _fovs(self):
        fovs = self.parameters.get('fovs')
        if fovs is None:
            return list(self.dataSet.get_fovs())
        allowed = set(int(f) for f in fovs)
        return [f for f in self.dataSet.get_fovs() if int(f) in allowed]

    def _z_slices(self):
        """[(label, zIndex, maximumProjection)] from z_index: absent = every z
        index, an index, 'maximum_projection', or a list of those."""
        if 'z_index' not in self.parameters:
            requested = list(range(len(self.dataSet.get_z_positions())))
        elif isinstance(self.parameters['z_index'], (list, tuple)):
            requested = list(self.parameters['z_index'])
        else:
            requested = [self.parameters['z_index']]
        slices = []
        for z in requested:
            if z == 'maximum_projection':
                slices.append(('zmax', 0, True))
            else:
                slices.append(('z%d' % int(z), int(z), False))
        return slices

    def _write_mosaic(self, baseName, images, sliceCount, channelCount):
        compression = self.parameters.get('tiff_compression')
        description = self.dataSet.analysis_tiff_description(
            sliceCount, channelCount)
        with self.dataSet.writer_for_analysis_images(
                self, baseName, imagej=True) as outputTif:
            for image in images:
                outputTif.write(image, photometric='MINISBLACK',
                                contiguous=compression is None,
                                compression=compression, metadata=description)

    def _run_analysis(self):

        alignTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])
        micronExtents = alignTask.get_global_extent()
        self.dataSet.save_numpy_txt_analysis_result(
            self._micron_to_mosaic_transform(micronExtents),
            'micron_to_mosaic_pixel_transform', self)

        channels = self._mosaic_channels()
        slices = self._z_slices()
        print('generating mosaic for \nchannels: {}\nz slices: {}'.format(
            [c[0] for c in channels], [s[0] for s in slices]))

        def mosaic_slice(channel, zSlice):
            name, dataChannel, frames = channel
            label, zIndex, maximumProjection = zSlice
            print(f'starting mosaic channel {name}, {label}.')
            return self._prepare_mosaic_slice(
                zIndex, dataChannel, micronExtents, alignTask,
                maximumProjection, frames)

        # separate_files: false = one file, every channel x z slice
        # (mosaic.tif); 'z' = one file per z slice with every channel
        # (mosaic_z<i>.tif, mosaic_zmax.tif); 'channel' or true = one file per
        # z slice and channel (mosaic_z<i>_<channel>.tif).
        mode = self.parameters['separate_files']
        preview = self.parameters.get('write_preview', False)
        if mode == 'z':
            for zSlice in slices:
                images = [mosaic_slice(c, zSlice) for c in channels]
                self._write_mosaic('mosaic_%s' % zSlice[0], images, 1,
                                   len(channels))
                if preview:
                    self._write_previews(zSlice[0], [c[0] for c in channels],
                                         images)
        elif mode:
            for zSlice in slices:
                images = []
                for channel in channels:
                    image = mosaic_slice(channel, zSlice)
                    self._write_mosaic('mosaic_%s_%s' % (zSlice[0], channel[0]),
                                       [image], 1, 1)
                    images.append(image if preview else None)
                if preview:
                    self._write_previews(zSlice[0], [c[0] for c in channels],
                                         images)
        else:
            byChannel = [[mosaic_slice(c, s) for s in slices] for c in channels]
            self._write_mosaic('mosaic', [i for c in byChannel for i in c],
                               len(slices), len(channels))
            if preview:
                for k, zSlice in enumerate(slices):
                    self._write_previews(zSlice[0], [c[0] for c in channels],
                                         [c[k] for c in byChannel])

    def _write_previews(self, label, names, images):
        """8-bit percentile-stretched PNGs, mosaic_preview_<z slice>_<channel>,
        and a composite of the first three channels (blue, green, red),
        mosaic_preview_<z slice>_composite."""
        path = self.dataSet.get_analysis_subdirectory(self, 'images')
        stretched = []
        for name, mosaic in zip(names, images):
            values = mosaic[mosaic > 0]
            lo, hi = np.percentile(values, [1, 99.8]) if values.size \
                else (0, 1)
            image = np.clip((mosaic.astype(np.float32) - lo)
                            / max(hi - lo, 1e-9), 0, 1)
            stretched.append(image)
            cv2.imwrite(os.path.join(path, 'mosaic_preview_%s_%s.png'
                                     % (label, name)),
                        (image * 255).astype(np.uint8))
        if len(stretched) > 1:
            composite = np.zeros(stretched[0].shape + (3,), np.float32)
            for image, color in zip(stretched, PREVIEW_COLORS):
                composite += image[..., None] * np.float32(color) / 255
            cv2.imwrite(os.path.join(path, 'mosaic_preview_%s_composite.png'
                                     % label),
                        (np.clip(composite, 0, 1) * 255).astype(np.uint8))

    def _fov_image(self, fov, dataChannel, zIndex, maximumProjection,
                   frames, warpTask, chromaticCorrector):
        """The registered image of one fov and mosaic channel (float32)."""
        if frames is not None:
            zList = range(len(frames))
        elif maximumProjection:
            zList = range(len(self.dataSet.get_z_positions()))
        else:
            zList = [zIndex]
        image = None
        for z in zList:
            plane = warpTask.get_aligned_image(
                fov, dataChannel, z, chromaticCorrector).astype(np.float32)
            image = plane if image is None else np.maximum(image, plane)
        return image

    def _flat_field(self, fovs, dataChannel, zIndex, maximumProjection,
                    frames, warpTask, chromaticCorrector, threads):
        """Smoothed per-pixel median over up to flat_field_fovs fovs (4x
        binned), mean 1, at full resolution."""
        count = int(self.parameters.get('flat_field_fovs', 80))
        sample = fovs[::max(1, len(fovs) // max(count, 1))][:count]
        shape = []

        def binned(f):
            image = self._fov_image(f, dataChannel, zIndex, maximumProjection,
                                    frames, warpTask, chromaticCorrector)
            shape.append(image.shape)
            return cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4),
                              interpolation=cv2.INTER_AREA)

        with ThreadPoolExecutor(threads) as pool:
            stack = np.stack(list(pool.map(binned, sample)))
        height, width = shape[0]
        stack[stack <= 0] = np.nan
        median = np.nanmedian(stack, axis=0)
        median = np.where(np.isfinite(median), median, np.nanmedian(median))
        sigma = float(self.parameters.get('flat_field_sigma', 50)) / 4
        flat = cv2.GaussianBlur(median.astype(np.float32), (0, 0), sigma)
        flat /= flat.mean()
        return cv2.resize(flat, (width, height), interpolation=cv2.INTER_LINEAR)

    def _prepare_mosaic_slice(self, zIndex, dataChannel, micronExtents,
                              alignTask, maximumProjection, frames=None):
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

        chromaticCorrector = None
        if 'optimize_task' in self.parameters:
            chromaticCorrector = self.dataSet.load_analysis_task(
                self.parameters['optimize_task']).get_chromatic_corrector()

        cropWidth = self.parameters['fov_crop_width']
        mosaicDimensions = tuple(self._micron_to_mosaic_pixel(
                micronExtents[-2:], micronExtents))
        fovs = self._fovs()
        threads = int(self.parameters.get('num_threads', 1))

        source = frame_source(self.dataSet, dataChannel, frames) \
            if frames is not None else contextlib.nullcontext()
        with source:
            flat = None
            if self.parameters.get('flat_field', False):
                flat = self._flat_field(fovs, dataChannel, zIndex,
                                        maximumProjection, frames, warpTask,
                                        chromaticCorrector, threads)

            def placed(f):
                inputImage = self._fov_image(
                    f, dataChannel, zIndex, maximumProjection, frames,
                    warpTask, chromaticCorrector)
                if flat is not None:
                    inputImage = inputImage / flat
                if cropWidth > 0:
                    inputImage[:cropWidth, :] = 0
                    inputImage[inputImage.shape[0] - cropWidth:, :] = 0
                    inputImage[:, :cropWidth] = 0
                    inputImage[:, inputImage.shape[1] - cropWidth:] = 0
                if self.parameters['draw_fov_labels']:
                    inputImage = cv2.putText(
                        inputImage, str(f),
                        (int(0.2*inputImage.shape[1]),
                         int(0.2*inputImage.shape[0])),
                        0, 10, (65000, 65000, 65000), 20)
                return self._transform_image_to_mosaic(
                    inputImage, f, alignTask, micronExtents,
                    mosaicDimensions)

            total = np.zeros(np.flip(mosaicDimensions, axis=0), np.float32)
            weights = np.zeros_like(total)
            with ThreadPoolExecutor(threads) as pool:
                for warped, weight in pool.map(placed, fovs):
                    total += warped
                    weights += weight

        mosaic = np.where(weights > 0.5, total / np.maximum(weights, 1e-6), 0)
        return np.clip(np.round(mosaic), 0, 65535).astype(np.uint16)
