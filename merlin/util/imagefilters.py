import cv2
import numpy as np

"""
This module contains code for performing filtering operations on images
"""


def highpass_filter(image: np.ndarray,
                     windowSize: int,
                     sigma: float,
                     clip: bool = True) -> np.ndarray:
    """
    Args:
        image: the input image to be filtered
        windowSize: the size of the Gaussian kernel to use.
        sigma: the sigma of the Gaussian.

    Returns:
        the high pass filtered image. The returned image is the same type
        as the input image.
    """
    lowpass = cv2.GaussianBlur(image,
                               (windowSize, windowSize),
                               sigma,
                               borderType=cv2.BORDER_REPLICATE)
    gauss_highpass = image - lowpass
    if clip:
        # This single line manufactures the entire positive background the
        # decoder sees. image - blur(image) is roughly symmetric, so zeroing
        # the negative half rectifies the noise into a pedestal close to the
        # Gaussian rectification value 1/sqrt(2*pi) of the plane's own highpass
        # sigma: it tracks the NOISE, not the signal.
        #
        # It is not neutral across codewords. With the pedestal in place
        # pure-background pixels share a nearest codeword; with it removed
        # that codeword falls to the far end of the ranking. Flattening the
        # pedestal to a uniform value does not change this, so it is the
        # rectification itself, not its per-bit skew, that makes background
        # look like a codeword.
        #
        # Left on by default because turning it off would silently move every
        # existing analysis, and because OptimizeIteration depends on it: the
        # pedestal props up the measured on-bit mean of low-SNR bits, so
        # removing it drives their scale factors lower still. Only turn it off
        # on an ImageScaleFactors path.
        gauss_highpass[lowpass > image] = 0
    return gauss_highpass
    
