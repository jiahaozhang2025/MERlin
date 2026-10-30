"""windowed_phase_cross_correlation (FiducialCorrelationWarp search_window_px).

Synthetic bead images: the moving image is the reference shifted by a known amount, plus a brighter decoy pattern whose
own displacement is far away, so the unrestricted whitened phase correlation picks the decoy. A window around the true
shift must recover it to sub-pixel accuracy; a window that contains the global peak must return skimage's own result.

Run with `python -m unittest tests/test_warp_window.py`.
"""
import unittest

import numpy as np
from scipy import ndimage
from skimage import registration

from merlin.util.registration import windowed_phase_cross_correlation

N = 512
TRUE = np.array([7.3, -4.6])       # moving = reference shifted by TRUE; skimage reports -TRUE
DECOY = np.array([-180.0, 210.0])


def beads(rng, n, amp):
    img = np.zeros((N, N))
    for y, x in rng.integers(20, N - 20, (n, 2)):
        img[y - 1:y + 2, x - 1:x + 2] = rng.uniform(*amp)
    return ndimage.gaussian_filter(img, 1.2)


class TestWindowedPhaseCorrelation(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.ref = beads(rng, 60, (50, 100)) + rng.normal(0, 0.5, (N, N))
        self.mov = ndimage.shift(self.ref, TRUE, order=1) + rng.normal(0, 0.5, (N, N))
        decoy = beads(rng, 400, (400, 800))
        self.refDecoy = self.ref + decoy
        self.movDecoy = self.mov + ndimage.shift(decoy, DECOY, order=1, mode='wrap')

    def test_global_peak_is_the_decoy(self):
        shift = registration.phase_cross_correlation(
            self.refDecoy, self.movDecoy, upsample_factor=100)[0]
        self.assertGreater(np.abs(shift + TRUE).max(), 50)

    def test_window_recovers_true_shift(self):
        for center in (-TRUE, -TRUE + [10, -12], -TRUE + [-20, 20]):
            shift = windowed_phase_cross_correlation(
                self.refDecoy, self.movDecoy, center, 25, 100)[0]
            np.testing.assert_allclose(shift, -TRUE, atol=0.25)

    def test_matches_skimage_when_window_contains_global_peak(self):
        g = registration.phase_cross_correlation(self.ref, self.mov, upsample_factor=100)
        w = windowed_phase_cross_correlation(self.ref, self.mov, -TRUE + [5, 5], 25, 100)
        np.testing.assert_array_equal(w[0], g[0])
        self.assertAlmostEqual(w[1], g[1])
        self.assertAlmostEqual(w[2], g[2])


if __name__ == '__main__':
    unittest.main()
