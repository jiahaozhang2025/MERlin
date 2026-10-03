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
        # decoder sees. image - blur(image) is roughly symmetric -- measured
        # 52.6 to 53.9 per cent of pixels negative on real 20260609 planes --
        # so zeroing the negative half rectifies the noise into a pedestal of
        # 3.08 to 4.93 ADU. That pedestal is 0.386 to 0.392 of the plane's own
        # highpass sigma across four bits, against the Gaussian rectification
        # value 1/sqrt(2*pi) = 0.3989: it tracks the NOISE, not the signal.
        #
        # It is not neutral across codewords. With the pedestal in place a
        # pure-background pixel's nearest codeword is Opalin, cosine 0.7187,
        # rank 1 of 246; with it removed Opalin falls to rank 246 at -0.6078.
        # Flattening the pedestal to a uniform value still leaves Opalin rank 1
        # at 0.6314, so it is the rectification itself, not its per-bit skew,
        # that makes background look like a codeword.
        #
        # Left on by default because turning it off would silently move every
        # existing analysis, and because OptimizeIteration depends on it: the
        # pedestal props up the measured on-bit mean of low-SNR bits, so
        # removing it drives their scale factors lower still. Only turn it off
        # on an ImageScaleFactors path.
        gauss_highpass[lowpass > image] = 0
    return gauss_highpass
    
