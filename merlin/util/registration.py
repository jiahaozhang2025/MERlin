from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from skimage import transform
import numpy as np
from scipy import signal


def extract_control_points(
        referencePoints: np.ndarray, movingPoints: np.ndarray,
        gridSpacing: float=0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    If fewer than 10 points are provided for either the reference or the moving
    list, this returns no points.

    Args:
        referencePoints: a n x 2 numpy array containing the reference points.
        movingPoints: a m x 2 numpy array containing the moving points.
        gridSpacing: the spacing of the grid for the 2d histogram for
            estimating the course transformation
    Returns: two np arrays (select reference points, select moving points)
        both of which are p x 2. The i'th point in the reference list
        has been matched to the i'th point in the moving list.
    """
    if len(referencePoints) < 10 or len(movingPoints) < 10:
        return np.zeros((0, 2)), np.zeros((0, 2))

    edges = np.arange(-200, 200, gridSpacing)

    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors.fit(referencePoints)
    distances, indexes = neighbors.kneighbors(
        movingPoints, return_distance=True)
    differences = [[movingPoints[i] - referencePoints[x]
                    for x in indexes[i]]
                   for i in range(len(movingPoints))]
    counts, xedges, yedges = np.histogram2d(
        [x[0] for y in differences for x in y],
        [x[1] for y in differences for x in y],
        bins=edges)
    maxIndex = np.unravel_index(counts.argmax(), counts.shape)
    offset = (xedges[maxIndex[0]], yedges[maxIndex[1]])

    distancesShifted, indexesShifted = neighbors.kneighbors(
        movingPoints - np.tile(offset, (movingPoints.shape[0], 1)),
        return_distance=True)

    controlIndexes = [x[0] < gridSpacing for x in distancesShifted]
    referenceControls = np.array([referencePoints[x[0]]
                                  for x in indexesShifted[controlIndexes]])
    movingControls = movingPoints[controlIndexes, :]

    return referenceControls, movingControls


def estimate_transform_from_points(
        referencePoints: np.ndarray, movingPoints: np.ndarray) \
        -> transform.EuclideanTransform:
    """

    If fewer than two points are provided, this will return the identity
    transform.

    Args:
        referencePoints: a n x 2 numpy array containing the reference points
        movingPoints: a n x 2 numpy array containing the moving points, where
            the i'th point of moving points corresponds with the i'th point
            of reference points.
    Returns: a similarity transform estimated from the paired points.

    """
    tform = transform.SimilarityTransform()
    if len(referencePoints) < 2 or len(movingPoints) < 2:
        return tform
    tform.estimate(referencePoints, movingPoints)
    return tform


def lsradialcenterfit(m, b, w):
    wm2p1 = w / (m * m + 1)
    sw = np.sum(wm2p1)
    smmw = np.sum(m * m * wm2p1)
    smw = np.sum(m * wm2p1)
    smbw = np.sum(m * b * wm2p1)
    sbw = np.sum(b * wm2p1)
    det = smw * smw - smmw * sw
    xc = (smbw * sw - smw * sbw) / det
    yc = (smbw * smw - smmw * sbw) / det

    return xc, yc


def radial_center(imageIn) -> Tuple[float, float]:
    """Determine the center of the object in imageIn using radial-symmetry-based
    particle localization.

    Adapted from Raghuveer, Nature Methods, 2012
    """
    Ny, Nx = imageIn.shape
    xm_onerow = np.arange(-(Nx - 1) / 2.0 + 0.5, (Nx) / 2.0 - 0.5)
    xm = np.tile(xm_onerow, (Ny - 1, 1))
    ym_onecol = [np.arange(-(Nx - 1) / 2.0 + 0.5, (Nx) / 2.0 - 0.5)]
    ym = np.tile(ym_onecol, (Nx - 1, 1)).transpose()

    imageIn = imageIn.astype(float)

    dIdu = imageIn[0:Ny - 1, 1:Nx] - imageIn[1:Ny, 0:Nx - 1];
    dIdv = imageIn[0:Ny - 1, 0:Nx - 1] - imageIn[1:Ny, 1:Nx];

    h = np.ones((3, 3)) / 9
    fdu = signal.convolve2d(dIdu, h, 'same')
    fdv = signal.convolve2d(dIdv, h, 'same')
    dImag2 = np.multiply(fdu, fdu) + np.multiply(fdv, fdv)

    m = np.divide(-(fdv + fdu), (fdu - fdv))

    if np.any(np.isnan(m)):
        unsmoothm = np.divide(dIdv + dIdu, dIdu - dIdv)
        m[np.isnan(m)] = unsmoothm[np.isnan(m)]

    if np.any(np.isnan(m)):
        m[np.isnan(m)] = 0

    if np.any(np.isinf(m)):
        if ~np.all(np.isinf(m)):
            m[np.isinf(m)] = 10 * np.max(m[~np.isinf(m)])
        else:
            m = np.divide((dIdv + dIdu), (dIdu - dIdv))

    b = ym - np.multiply(m, xm)

    sdI2 = np.sum(dImag2)
    xcentroid = np.sum(np.sum(np.multiply(dImag2, xm))) / sdI2
    ycentroid = np.sum(np.multiply(dImag2, ym)) / sdI2
    w = np.divide(dImag2, np.sqrt(
        (xm - xcentroid) * (xm - xcentroid) + (ym - ycentroid) * (
                    ym - ycentroid)))

    xc, yc = lsradialcenterfit(m, b, w)

    xc = xc + (Nx + 1) / 2.0
    yc = yc + (Ny + 1) / 2.0

    return xc, yc


def refine_position(image, x, y, cropSize=4) -> Tuple[float, float]:
    # TODO this would be more intuitive it it retransformed the output
    # coordinates to the original image coordinates
    subImage = image[int(y + 2 - cropSize):int(y + cropSize),
                     int(x - cropSize + 2):int(x + cropSize)]
    return radial_center(subImage)


def windowed_phase_cross_correlation(
        referenceImage: np.ndarray, movingImage: np.ndarray,
        center: Tuple[float, float], window: float,
        upsampleFactor: int = 100) -> Tuple[np.ndarray, float, float]:
    """skimage.registration.phase_cross_correlation (normalization 'phase')
    with the peak search restricted to shifts within +-window of center.

    The whitened cross-correlation of sparse bead images can have a stronger
    spurious peak far from the true shift; when the expected shift is known
    (e.g. the median over all fovs of the same round), searching only near it
    recovers the true peak. The integer peak is then refined exactly as
    skimage does (upsampled DFT around it), so a window that contains the
    global maximum returns skimage's own result.

    Args:
        referenceImage, movingImage: 2D images of equal shape.
        center: expected (y, x) shift, in skimage's convention (the shift
            that registers movingImage onto referenceImage).
        window: half-width of the square search window, in pixels.
        upsampleFactor: sub-pixel upsampling factor, as in skimage.
    Returns:
        (shift, error, phasediff) as phase_cross_correlation returns them.
    """
    from skimage.registration._phase_cross_correlation import (
        _upsampled_dft, _compute_error, _compute_phasediff)

    srcFreq = np.fft.fftn(referenceImage)
    targetFreq = np.fft.fftn(movingImage)
    shape = np.array(srcFreq.shape)
    imageProduct = srcFreq * targetFreq.conj()
    eps = np.finfo(imageProduct.real.dtype).eps
    imageProduct /= np.maximum(np.abs(imageProduct), 100 * eps)
    crossCorrelation = np.abs(np.fft.ifftn(imageProduct))

    # candidate integer shifts within the window, wrapped onto array indices
    axes = [np.arange(int(np.floor(c - window)), int(np.ceil(c + window)) + 1)
            for c in center]
    sub = crossCorrelation[np.ix_(axes[0] % shape[0], axes[1] % shape[1])]
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    shift = np.array([axes[0][i], axes[1][j]], dtype=float)

    upsample = float(upsampleFactor)
    shift = np.round(shift * upsample) / upsample
    regionSize = np.ceil(upsample * 1.5)
    dftShift = np.fix(regionSize / 2.0)
    upsampled = _upsampled_dft(imageProduct.conj(), regionSize, upsample,
                               dftShift - shift * upsample).conj()
    maxima = np.unravel_index(np.argmax(np.abs(upsampled)), upsampled.shape)
    ccMax = upsampled[maxima]
    shift += (np.stack(maxima).astype(float) - dftShift) / upsample
    srcAmp = np.sum(np.real(srcFreq * srcFreq.conj()))
    targetAmp = np.sum(np.real(targetFreq * targetFreq.conj()))
    return shift, _compute_error(ccMax, srcAmp, targetAmp), \
        _compute_phasediff(ccMax)
