"""A dataset's camera hot-pixel map, for correcting raw frames before warping.

HotPixelMap finds the camera's hot pixels from the data itself (the rule is
in merlin.util.hotpixel):

1. For each sampled data channel, the median of its raw frames over a sample
   of fovs (sample_fovs) and planes (sample_planes). The tissue differs from
   frame to frame and is removed by the median; a pixel that is bright at the
   same camera position in every frame survives it.
2. A pixel is a CANDIDATE in the channel when that median frame exceeds its
   own median filter (median_filter_size, default 5 x 5) by more than
   excess_threshold (default 50 ADU).
3. A pixel is HOT when it is a candidate in at least min_channels (default 2)
   sampled channels of the same camera. A bright feature of one channel, such
   as real structure that persists through depth, is a candidate in that
   channel only and is not changed.

All channels sit on one camera unless camera_colors lists the colours of each
camera; every data channel of a camera is corrected with that camera's map,
sampled or not.

Correction. A preprocess task with hot_pixel_task set to this task hands its
HotPixelCorrection (get_correction) to the warp task's get_aligned_image as
hotPixels. The warp then replaces each hot pixel of every raw frame by the
median of the non-hot pixels of its 3 x 3 window (5 x 5 when all eight
neighbours are hot, up to replace_max_radius) in camera space, before the
depth blend, the chromatic correction and the warp. The warp task's saved
parameters and transformations are not changed, and without hot_pixel_task
the images are exactly the uncorrected ones.

Outputs (in this task's folder):
- hot_pixels.csv: one row per hot pixel: camera, y, x (row and column of the
  frame as DataSet.get_raw_image returns it), n_votes, n_sampled_channels,
  excess_median / excess_min / excess_max over the channels where it is a
  candidate, replace_radius (1 = 3 x 3, 0 = unresolved) and
  replace_neighbours (the non-hot pixels its median is taken over).
- candidates.csv: one row per (channel, candidate pixel): data_channel,
  channel_name, color, camera, y, x, excess, median_value, hot.
- channel_counts.csv: one row per data channel: name, color, camera,
  sampled, n_frames, frame_level (median of its median frame),
  n_candidates, n_candidates_hot, n_hot_camera (the pixels corrected in it).
- vote_count.npy: (camera, y, x) uint16, the number of sampled channels in
  which each pixel is a candidate.
- summary.json: the sampled fovs, planes and channels, counts per camera and
  run time.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import pandas

from merlin.core import analysistask
from merlin.util import hotpixel


def _evenly_spaced(values: List, count: int) -> List:
    """count items of values spread evenly from the first to the last (all
    of them when count >= len(values))."""
    values = list(values)
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    picks = np.unique(np.rint(np.linspace(0, len(values) - 1, count))
                      .astype(int))
    return [values[i] for i in picks]


class HotPixelMap(analysistask.AnalysisTask):

    """The camera hot pixels of a dataset (see the module docstring)."""

    outputGroup = 'Prepare'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # Data channels whose frames are sampled; None = every data channel.
        if 'data_channels' not in self.parameters:
            self.parameters['data_channels'] = None
        # An integer = that many fovs spread evenly over the dataset's fovs
        # (all when there are fewer); a list = those fovs.
        if 'sample_fovs' not in self.parameters:
            self.parameters['sample_fovs'] = 16
        # An integer = that many planes spread evenly over each channel's own
        # z positions; a list = those z indexes of the dataset.
        if 'sample_planes' not in self.parameters:
            self.parameters['sample_planes'] = 4
        if 'excess_threshold' not in self.parameters:
            self.parameters['excess_threshold'] = 50.0
        if 'median_filter_size' not in self.parameters:
            self.parameters['median_filter_size'] = 5
        if 'min_channels' not in self.parameters:
            self.parameters['min_channels'] = 2
        # None = one camera for every channel; otherwise a list with, for
        # each camera, the list of colours it images, e.g.
        # [["750", "650"], ["560", "488", "405"]].
        if 'camera_colors' not in self.parameters:
            self.parameters['camera_colors'] = None
        if 'replace_max_radius' not in self.parameters:
            self.parameters['replace_max_radius'] = 2
        # Threads that read frames and take the medians.
        if 'threads' not in self.parameters:
            self.parameters['threads'] = 8

        self._correction = None
        self._correctionLock = threading.Lock()

    def get_estimated_memory(self):
        return 8000

    def get_estimated_time(self):
        return 60

    def get_dependencies(self):
        return []

    # ------------------------------------------------------------ sampling
    def _data_channels(self) -> List[int]:
        return [int(c) for c in
                self.dataSet.get_data_organization().get_data_channels()]

    def _sampled_channels(self) -> List[int]:
        channels = self._data_channels()
        requested = self.parameters['data_channels']
        if requested is None:
            return channels
        requested = [int(c) for c in requested]
        unknown = sorted(set(requested) - set(channels))
        if unknown:
            raise ValueError('data_channels %s are not data channels of the '
                             'dataset' % unknown)
        return requested

    def _sample_fovs(self) -> List[int]:
        fovs = [int(f) for f in self.dataSet.get_fovs()]
        requested = self.parameters['sample_fovs']
        if isinstance(requested, (list, tuple)):
            requested = [int(f) for f in requested]
            unknown = sorted(set(requested) - set(fovs))
            if unknown:
                raise ValueError('sample_fovs %s are not fovs of the dataset'
                                 % unknown)
            return requested
        return _evenly_spaced(fovs, int(requested))

    def _channel_z_positions(self, dataChannel: int) -> List[float]:
        """The z positions the channel was imaged at."""
        organization = self.dataSet.get_data_organization()
        try:
            zPositions = organization.data.iloc[dataChannel]['zPos']
        except AttributeError:
            zPositions = self.dataSet.get_z_positions()
        return [float(z) for z in np.atleast_1d(zPositions)]

    def _sample_z_positions(self, dataChannel: int) -> List[float]:
        own = self._channel_z_positions(dataChannel)
        requested = self.parameters['sample_planes']
        if isinstance(requested, (list, tuple)):
            zPositions = self.dataSet.get_z_positions()
            wanted = [float(zPositions[int(i)]) for i in requested]
            return [z for z in wanted if z in own]
        return _evenly_spaced(own, int(requested))

    # ------------------------------------------------------------- cameras
    def get_channel_cameras(self) -> Dict[int, int]:
        """{data channel: camera index} for every data channel."""
        organization = self.dataSet.get_data_organization()
        groups = self.parameters['camera_colors']
        channels = self._data_channels()
        if groups is None:
            return {c: 0 for c in channels}
        cameraOf = {}
        for camera, colors in enumerate(groups):
            for color in colors:
                color = str(color)
                if color in cameraOf:
                    raise ValueError('colour %s is listed for two cameras in '
                                     'camera_colors' % color)
                cameraOf[color] = camera
        cameras = {}
        for c in channels:
            color = str(organization.get_data_channel_color(c))
            if color not in cameraOf:
                raise ValueError('data channel %d (colour %s) is in no '
                                 'camera of camera_colors' % (c, color))
            cameras[c] = cameraOf[color]
        return cameras

    def _check_votes(self, sampled, cameras) -> None:
        minChannels = int(self.parameters['min_channels'])
        if minChannels < 1:
            raise ValueError('min_channels must be >= 1')
        for camera in sorted(set(cameras.values())):
            n = sum(cameras[c] == camera for c in sampled)
            if n < minChannels:
                raise ValueError(
                    'camera %d has %d sampled channel(s), fewer than '
                    'min_channels %d: no pixel could be called hot'
                    % (camera, n, minChannels))

    # ---------------------------------------------------------------- run
    def _median_frame(self, dataChannel: int, fovs, zPositions, pool,
                      threads: int):
        jobs = [(f, z) for f in fovs for z in zPositions]
        if not jobs:
            raise ValueError('no frames to sample for data channel %d'
                             % dataChannel)
        stack = np.stack(list(pool.map(
            lambda job: np.asarray(self.dataSet.get_raw_image(
                dataChannel, job[0], job[1])), jobs)))
        median = np.empty(stack.shape[1:], dtype=np.float32)

        def rows(block):
            median[block] = np.median(stack[:, block], axis=0)

        bounds = np.linspace(0, stack.shape[1], threads + 1).astype(int)
        list(pool.map(rows, [slice(a, b) for a, b in zip(bounds[:-1],
                                                          bounds[1:]) if b > a]))
        return median, len(jobs)

    def _run_analysis(self):
        start = time.time()
        organization = self.dataSet.get_data_organization()
        sampled = self._sampled_channels()
        cameras = self.get_channel_cameras()
        self._check_votes(sampled, cameras)
        fovs = self._sample_fovs()
        threshold = float(self.parameters['excess_threshold'])
        size = int(self.parameters['median_filter_size'])
        minChannels = int(self.parameters['min_channels'])
        threads = max(1, int(self.parameters['threads']))
        height, width = [int(v) for v in self.dataSet.get_image_dimensions()]
        cameraIds = sorted(set(cameras.values()))
        votes = np.zeros((len(cameraIds), height, width), dtype=np.uint16)

        candidates, perChannel, masks = [], {}, {}
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for dataChannel in sampled:
                zPositions = self._sample_z_positions(dataChannel)
                median, nFrames = self._median_frame(
                    dataChannel, fovs, zPositions, pool, threads)
                if median.shape != (height, width):
                    raise ValueError('data channel %d frames are %s, the '
                                     'dataset image dimensions %s'
                                     % (dataChannel, median.shape,
                                        (height, width)))
                mask, excess = hotpixel.candidate_map(median, threshold, size)
                camera = cameras[dataChannel]
                votes[cameraIds.index(camera)] += mask
                masks[dataChannel] = mask
                ys, xs = np.nonzero(mask)
                candidates.append(pandas.DataFrame({
                    'data_channel': dataChannel,
                    'y': ys.astype(np.int32), 'x': xs.astype(np.int32),
                    'excess': excess[ys, xs].astype(np.float32),
                    'median_value': median[ys, xs].astype(np.float32)}))
                perChannel[dataChannel] = {
                    'n_frames': nFrames, 'n_planes': len(zPositions),
                    'frame_level': float(np.median(median)),
                    'n_candidates': int(mask.sum())}

        hotMasks = {camera: votes[k] >= minChannels
                    for k, camera in enumerate(cameraIds)}
        maxRadius = int(self.parameters['replace_max_radius'])
        correction = hotpixel.HotPixelCorrection(hotMasks, cameras, maxRadius)

        candidates = pandas.concat(candidates, ignore_index=True)
        candidates.insert(1, 'channel_name', [
            organization.get_data_channel_name(c)
            for c in candidates['data_channel']])
        candidates.insert(2, 'color', [
            str(organization.get_data_channel_color(c))
            for c in candidates['data_channel']])
        candidates.insert(3, 'camera', [cameras[c]
                                        for c in candidates['data_channel']])
        candidates['hot'] = [bool(hotMasks[cam][y, x]) for cam, y, x in zip(
            candidates['camera'], candidates['y'], candidates['x'])]

        hotRows = []
        for camera in cameraIds:
            replacer = correction.replacers[camera]
            nSampled = sum(cameras[c] == camera for c in sampled)
            own = candidates[(candidates['camera'] == camera)
                             & candidates['hot']]
            excessBy = own.groupby(['y', 'x'])['excess']
            stats = pandas.DataFrame({'excess_median': excessBy.median(),
                                      'excess_min': excessBy.min(),
                                      'excess_max': excessBy.max()})
            for k, (y, x) in enumerate(zip(replacer.rows, replacer.cols)):
                s = stats.loc[(y, x)]
                hotRows.append({
                    'camera': camera, 'y': int(y), 'x': int(x),
                    'n_votes': int(votes[cameraIds.index(camera)][y, x]),
                    'n_sampled_channels': nSampled,
                    'excess_median': float(s['excess_median']),
                    'excess_min': float(s['excess_min']),
                    'excess_max': float(s['excess_max']),
                    'replace_radius': int(replacer.radii[k]),
                    'replace_neighbours': int(replacer.neighbourCounts[k])})
        hotTable = pandas.DataFrame(hotRows, columns=[
            'camera', 'y', 'x', 'n_votes', 'n_sampled_channels',
            'excess_median', 'excess_min', 'excess_max', 'replace_radius',
            'replace_neighbours'])

        countRows = []
        for dataChannel in self._data_channels():
            camera = cameras[dataChannel]
            info = perChannel.get(dataChannel, {})
            mask = masks.get(dataChannel)
            countRows.append({
                'data_channel': dataChannel,
                'channel_name': organization.get_data_channel_name(
                    dataChannel),
                'color': str(organization.get_data_channel_color(
                    dataChannel)),
                'camera': camera, 'sampled': dataChannel in perChannel,
                'n_frames': info.get('n_frames', 0),
                'n_planes': info.get('n_planes', 0),
                'frame_level': info.get('frame_level', np.nan),
                'n_candidates': info.get('n_candidates', 0),
                'n_candidates_hot': int((mask & hotMasks[camera]).sum())
                if mask is not None else 0,
                'n_hot_camera': correction.replacers[camera].hot_pixel_count})
        counts = pandas.DataFrame(countRows)

        self.dataSet.save_numpy_analysis_result(votes, 'vote_count', self)
        self.dataSet.save_dataframe_to_csv(hotTable, 'hot_pixels', self,
                                           index=False)
        self.dataSet.save_dataframe_to_csv(candidates, 'candidates', self,
                                           index=False)
        self.dataSet.save_dataframe_to_csv(counts, 'channel_counts', self,
                                           index=False)
        summary = {
            'sample_fovs': fovs,
            'sampled_channels': sampled,
            'sample_z_positions': {str(c): self._sample_z_positions(c)
                                   for c in sampled},
            'cameras': {str(camera): {
                'channels': sorted(c for c in cameras
                                   if cameras[c] == camera),
                'sampled_channels': sorted(c for c in sampled
                                           if cameras[c] == camera),
                'n_hot': correction.replacers[camera].hot_pixel_count,
                'n_unresolved': correction.replacers[camera].unresolved_count}
                for camera in cameraIds},
            'n_candidates_total': int(len(candidates)),
            'seconds': round(time.time() - start, 1)}
        self.dataSet.save_json_analysis_result(summary, 'summary', self)

    # ------------------------------------------------------------- results
    def get_hot_pixels(self) -> pandas.DataFrame:
        return self.dataSet.load_dataframe_from_csv('hot_pixels', self)

    def get_candidates(self) -> pandas.DataFrame:
        return self.dataSet.load_dataframe_from_csv('candidates', self)

    def get_channel_counts(self) -> pandas.DataFrame:
        return self.dataSet.load_dataframe_from_csv('channel_counts', self)

    def get_vote_count(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('vote_count', self)

    def get_summary(self) -> Dict:
        return self.dataSet.load_json_analysis_result('summary', self)

    def get_hot_pixel_masks(self) -> Dict[int, np.ndarray]:
        """{camera: bool (y, x) mask of its hot pixels}."""
        height, width = [int(v) for v in self.dataSet.get_image_dimensions()]
        table = self.get_hot_pixels()
        cameras = sorted(set(self.get_channel_cameras().values()))
        masks = {camera: np.zeros((height, width), dtype=bool)
                 for camera in cameras}
        for camera, group in table.groupby('camera'):
            masks[int(camera)][group['y'].to_numpy(int),
                               group['x'].to_numpy(int)] = True
        return masks

    def get_correction(self) -> hotpixel.HotPixelCorrection:
        """The HotPixelCorrection for every data channel (cached); the
        object a preprocess task's hot_pixel_task hands to the warp."""
        with self._correctionLock:
            if self._correction is None:
                if not self.is_complete():
                    raise RuntimeError('hot-pixel task %s has not finished'
                                       % self.get_analysis_name())
                self._correction = hotpixel.HotPixelCorrection(
                    self.get_hot_pixel_masks(), self.get_channel_cameras(),
                    int(self.parameters['replace_max_radius']))
            return self._correction
