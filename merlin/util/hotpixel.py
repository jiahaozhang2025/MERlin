"""Camera hot pixels: how they are found and how they are replaced.

A hot pixel is a camera defect: a pixel that reads well above its neighbours
in every frame, at the same position of the camera frame, whatever the sample.
Because it sits at the same camera pixel in every imaging round, warping each
round onto the reference moves it to a different place in each round, and
where two or more rounds of one codeword overlap it decodes as a "molecule"
on every plane.

Detection (merlin.analysis.hotpixel.HotPixelMap applies it):
- per channel, the median of many raw frames of that channel (different fovs
  and planes). Tissue changes from frame to frame and is removed by the
  median; a fixed camera feature survives it.
- excess = that median frame minus its own median filter (median_filter_size,
  default 5 x 5; the border is replicated). A pixel is a CANDIDATE in the
  channel when excess > excess_threshold.
- a pixel is HOT when it is a candidate in at least min_channels channels
  imaged by the same camera. A bright feature of a single channel is not a
  camera defect and is not changed.

Replacement (HotPixelReplacer): each hot pixel takes the median of the
non-hot pixels of its 3 x 3 window in the same frame. A hot pixel whose eight
neighbours are all hot (or outside the frame) uses the non-hot pixels of the
5 x 5 window instead, and so on up to max_radius; one with no non-hot pixel
within max_radius is left unchanged and counted as unresolved. All other
pixels are returned unchanged, and integer images keep their dtype (the
median is rounded to the nearest integer and clipped to the dtype's range).

Every rule here is symmetric under transposing or flipping the frame, so it
can be applied to frames in MERlin's image orientation (DataSet.get_raw_image)
as well as in the camera's own; a map is valid for frames read the same way
it was made.
"""
from typing import Dict, Mapping, Tuple

import cv2
import numpy as np
from scipy import ndimage


def median_filtered(image: np.ndarray, size: int) -> np.ndarray:
    """size x size median filter with the border replicated, as float32."""
    image = np.ascontiguousarray(image, dtype=np.float32)
    size = int(size)
    if size < 3 or size % 2 == 0:
        raise ValueError('median_filter_size must be an odd integer >= 3')
    if size in (3, 5):
        # OpenCV filters float32 only for apertures 3 and 5; it replicates
        # the border, the same as scipy's mode 'nearest'
        return cv2.medianBlur(image, size)
    return ndimage.median_filter(image, size=size, mode='nearest')


def candidate_map(medianFrame: np.ndarray, excessThreshold: float,
                  filterSize: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """The candidate rule on one channel's median frame.

    Returns:
        (candidate mask, excess), excess = medianFrame - its median filter,
        candidate = excess > excessThreshold
    """
    frame = np.asarray(medianFrame, dtype=np.float32)
    excess = frame - median_filtered(frame, filterSize)
    return excess > float(excessThreshold), excess


class HotPixelReplacer:

    """Replaces the hot pixels of one camera's frames by the median of their
    non-hot neighbours (see the module docstring)."""

    def __init__(self, hotMask: np.ndarray, maxRadius: int = 2):
        mask = np.asarray(hotMask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError('hotMask must be 2D')
        maxRadius = int(maxRadius)
        if maxRadius < 1:
            raise ValueError('maxRadius must be >= 1')
        self.shape = mask.shape
        height, width = mask.shape
        rows, cols = np.nonzero(mask)
        neighbourRows, neighbourCols, radii = [], [], []
        for r, c in zip(rows, cols):
            for radius in range(1, maxRadius + 1):
                r0, r1 = max(0, r - radius), min(height, r + radius + 1)
                c0, c1 = max(0, c - radius), min(width, c + radius + 1)
                yy, xx = np.nonzero(~mask[r0:r1, c0:c1])
                if len(yy):
                    neighbourRows.append(yy + r0)
                    neighbourCols.append(xx + c0)
                    radii.append(radius)
                    break
            else:
                neighbourRows.append(np.zeros(0, int))
                neighbourCols.append(np.zeros(0, int))
                radii.append(0)
        self.rows = rows
        self.cols = cols
        # 1 = the 3 x 3 window, 2 = 5 x 5 ..., 0 = unresolved (left unchanged)
        self.radii = np.array(radii, dtype=int)
        self.neighbourCounts = np.array([len(n) for n in neighbourRows],
                                        dtype=int)
        resolved = self.radii > 0
        self._rows = rows[resolved]
        self._cols = cols[resolved]
        slots = max([len(n) for n in neighbourRows] + [1])
        count = int(resolved.sum())
        self._nRows = np.zeros((count, slots), dtype=np.intp)
        self._nCols = np.zeros((count, slots), dtype=np.intp)
        self._valid = np.zeros((count, slots), dtype=bool)
        k = 0
        for nr, nc, ok in zip(neighbourRows, neighbourCols, resolved):
            if not ok:
                continue
            self._nRows[k, :len(nr)] = nr
            self._nCols[k, :len(nc)] = nc
            self._valid[k, :len(nr)] = True
            k += 1

    @property
    def hot_pixel_count(self) -> int:
        return int(len(self.rows))

    @property
    def unresolved_count(self) -> int:
        return int((self.radii == 0).sum())

    def replace(self, image: np.ndarray) -> np.ndarray:
        """A copy of image with its hot pixels replaced (image itself when
        there are none to replace)."""
        if image.shape != self.shape:
            raise ValueError('image shape %s does not match the hot-pixel map '
                             '%s' % (image.shape, self.shape))
        if not len(self._rows):
            return image
        values = image[self._nRows, self._nCols].astype(np.float64)
        values[~self._valid] = np.nan
        medians = np.nanmedian(values, axis=1)
        out = np.array(image, copy=True)
        if np.issubdtype(out.dtype, np.integer):
            info = np.iinfo(out.dtype)
            medians = np.clip(np.rint(medians), info.min, info.max)
        out[self._rows, self._cols] = medians.astype(out.dtype)
        return out


class HotPixelCorrection:

    """The hot-pixel replacement for every data channel of a dataset: each
    channel is corrected with the map of the camera that imaged it.

    This is the object the warp tasks' get_aligned_image take as hotPixels:
    correct(image, dataChannel) is applied to each raw frame before the depth
    blend, the chromatic correction and the warp.
    """

    def __init__(self, cameraMasks: Mapping[int, np.ndarray],
                 channelCameras: Mapping[int, int], maxRadius: int = 2):
        self.replacers: Dict[int, HotPixelReplacer] = {
            int(camera): HotPixelReplacer(mask, maxRadius)
            for camera, mask in cameraMasks.items()}
        self.channelCameras = {int(c): int(cam)
                               for c, cam in channelCameras.items()}
        missing = set(self.channelCameras.values()) - set(self.replacers)
        if missing:
            raise ValueError('no hot-pixel map for camera(s) %s'
                             % sorted(missing))

    def replacer(self, dataChannel: int) -> HotPixelReplacer:
        try:
            camera = self.channelCameras[int(dataChannel)]
        except KeyError:
            raise KeyError('data channel %d has no camera in the hot-pixel '
                           'map' % int(dataChannel)) from None
        return self.replacers[camera]

    def correct(self, image: np.ndarray, dataChannel: int) -> np.ndarray:
        return self.replacer(dataChannel).replace(image)

    def hot_pixel_count(self, dataChannel: int) -> int:
        return self.replacer(dataChannel).hot_pixel_count
