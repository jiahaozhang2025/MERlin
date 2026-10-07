"""Synthetic test of the chromatic-aberration fit in OptimizeIteration.

Spots are rendered at known positions p in the reference colour and at T(p) in
the other colours, T a known similarity transform in MERlin's convention (maps
reference-frame (x=col, y=row) to that colour's frame; RigidChromaticCorrector
warps the colour's image with it, skimage inverse-map). OptimizeIteration's own
sample measurement and fit run on the images, and
RigidChromaticCorrector.transform_image is checked to undo T.

Layouts:
    aligned  - data organization rows in codebook bit order (bit index ==
               data-channel index)
    shifted  - one non-codebook row inserted at data channel 2, so every later
               bit's index points at the previous row. Reading the colour with
               the bit index as a data-channel index gives 13 of 16 bits the
               wrong colour and the fit measures the mirror image of the truth.
    tricolor - 560/650/750, aligned

Run with `python -m unittest tests/test_chromatic_fit.py` (about 30 s).
"""
import unittest

import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis.optimize import OptimizeIteration
from merlin.util import aberration, registration

SIZE = 768
SIGMA = 1.3
# max error of the displacement field after ONE iteration: refine_position's
# 6x6 crop over-reads offsets near 2 px by ~0.1 px, which the loop removes
TOL_ONESHOT = 0.15
TOL_PX = 0.03          # max error after iterating
LAYOUTS = ('aligned', 'shifted', 'tricolor')


# ---------------------------------------------- stubs: only what the code touches
class _Org:
    def __init__(self, readouts, colors):
        self.data = pd.DataFrame({'readoutName': readouts, 'color': colors})

    def get_data_channel_color(self, dataChannel):
        return str(self.data.at[dataChannel, 'color'])

    def get_data_channel_for_bit(self, bitName):
        return self.data[self.data['readoutName'] == bitName].index.values.item()


class _Codebook:
    def __init__(self, bits, codes):
        self.bits, self.codes = bits, codes

    def get_bit_names(self):
        return self.bits

    def get_barcode(self, index):
        return self.codes[index]


class _DataSet:
    def __init__(self, org):
        self.org = org
        self.tasks = {}

    def get_data_organization(self):
        return self.org

    def save_pickle_analysis_result(self, *args, **kwargs):
        pass

    def load_pickle_analysis_result(self, *args, **kwargs):
        raise FileNotFoundError

    def load_analysis_task(self, name):
        return self.tasks[name]


class _Task:
    """Stands in for an OptimizeIteration: any method the code under test
    calls on self is OptimizeIteration's own."""
    analysisName = 'synthetic'

    def __init__(self, org, codebook):
        self.dataSet = _DataSet(org)
        self._codebook = codebook
        self.parameters = {'optimize_chromatic_correction': True,
                           'chromatic_from_fragments': False,
                           'warp_task': 'Warp',
                           'chromatic_max_barcodes_per_group': 0,
                           'chromatic_max_groups': 0,
                           'chromatic_threads': 1}
        self.barcodes = None

    def get_codebook(self):
        return self._codebook

    def is_complete(self):
        return True

    def get_barcode_database(self):
        return type('_DB', (), {'get_barcodes': lambda _: self.barcodes})()

    def __getattr__(self, name):
        return getattr(OptimizeIteration, name).__get__(self)


class _Warp:
    """get_aligned_image by DATA CHANNEL, colour from the data organization,
    as FiducialPolynomialWarp3D does."""

    def __init__(self, org, codebook, raw):
        self.org = org
        self.rawByChannel = {org.get_data_channel_for_bit(b): raw[i]
                             for i, b in enumerate(codebook.get_bit_names())}

    def get_aligned_image(self, fov, dataChannel, zIndex, corrector=None):
        image = self.rawByChannel[dataChannel]
        if corrector is not None:
            image = corrector.transform_image(
                image, self.org.get_data_channel_color(dataChannel))
        return image


# ---------------------------------------------- synthetic data
def make_layout(kind, nBits=16, nCodes=120, seed=0):
    rng = np.random.default_rng(seed)
    colors = ['560', '650', '750'] if kind == 'tricolor' else ['635', '750']
    bits = ['b%02d' % i for i in range(nBits)]
    bitColor = [colors[i % len(colors)] for i in range(nBits)]
    codes = []
    while len(codes) < nCodes:
        on = rng.choice(nBits, 4, replace=False)
        if len({bitColor[i] for i in on}) < 2:
            continue
        c = np.zeros(nBits, bool)
        c[on] = True
        codes.append(c)
    readouts, rowColors = list(bits), list(bitColor)
    if kind == 'shifted':
        readouts.insert(2, 'empty')
        rowColors.insert(2, colors[0])
    return _Task(_Org(readouts, rowColors), _Codebook(bits, codes)), bitColor, colors[0]


def render(points):
    img = np.zeros((SIZE, SIZE), np.float64)
    r = 6
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        if not (r <= ix < SIZE - r and r <= iy < SIZE - r):
            continue
        img[iy - r:iy + r + 1, ix - r:ix + r + 1] += 1000 * np.exp(
            -((xx + ix - x) ** 2 + (yy + iy - y) ** 2) / (2 * SIGMA ** 2))
    return img


def make_scene(task, bitColor, truth, nMolecules=2500, margin=40, seed=1):
    """Raw images (bit order) with spot k at truth[color](p_k), plus the
    barcode table in the reference frame."""
    rng = np.random.default_rng(seed)
    cb = task.get_codebook()
    p = rng.uniform(margin, SIZE - margin, (nMolecules, 2))
    ids = rng.integers(0, len(cb.codes), nMolecules)
    raw = []
    for b in range(len(bitColor)):
        on = np.array([cb.codes[i][b] for i in ids])
        T = truth.get(bitColor[b], transform.SimilarityTransform())
        raw.append(render(T(p[on])))
    jitter = rng.normal(0, 0.2, p.shape)           # decoded centroids are not exact
    barcodes = pd.DataFrame({'x': p[:, 0] + jitter[:, 0],
                             'y': p[:, 1] + jitter[:, 1], 'barcode_id': ids,
                             'fov': 0, 'z': 0.0})
    barcodes.attrs['true_xy'] = p
    return np.array(raw), barcodes


def corrected(raw, bitColor, transforms, ref):
    cc = aberration.RigidChromaticCorrector(transforms, ref)
    return np.array([cc.transform_image(im, c) for im, c in zip(raw, bitColor)])


def identity(colors):
    return {u: {v: transform.SimilarityTransform() for v in colors if v >= u}
            for u in colors}


def _grid():
    return np.stack(np.meshgrid(np.linspace(0, SIZE, 9),
                                np.linspace(0, SIZE, 9)), -1).reshape(-1, 2)


def field_error(fit, true):
    g = _grid()
    return float(np.abs(fit(g) - true(g)).max())


def field_corr(fit, true):
    """Correlation of the fitted and true displacement fields (+1 = right
    direction, -1 = reversed)."""
    g = _grid()
    a, b = (fit(g) - g).ravel(), (true(g) - g).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def one_iteration(task, raw, barcodes, bitColor, ref, previous):
    """The per-fragment path: measure on corrected images, then fit."""
    images = corrected(raw, bitColor, previous, ref)
    samples = task._measure_chromatic_samples(images, barcodes)
    return task._fit_color_pairs(samples, previous)


def ground_truth_residual(task, raw, barcodes, bitColor, transforms, ref):
    """Median |spot position in the corrected image - true reference position|
    over non-reference-colour bits."""
    images = corrected(raw, bitColor, transforms, ref)
    p = barcodes.attrs['true_xy']
    ids = barcodes['barcode_id'].to_numpy()
    refImg = render(p)
    cb = task.get_codebook()
    out = []
    for b, c in enumerate(bitColor):
        if c == ref:
            continue
        for (x, y), i in zip(p, ids):
            if cb.codes[i][b] and 10 < x < SIZE - 10 and 10 < y < SIZE - 10:
                out.append(np.subtract(
                    registration.refine_position(images[b], x, y),
                    registration.refine_position(refImg, x, y)))
    out = np.array(out)
    out = out[np.isfinite(out).all(1)]
    return float(np.median(np.hypot(*out.T)))


TRUTH = {
    '2col': {'750': transform.SimilarityTransform(
        scale=0.998, rotation=5e-4, translation=(1.0, 0.6))},
    '3col': {'650': transform.SimilarityTransform(
                 scale=0.9985, rotation=2e-4, translation=(0.7, 0.5)),
             '750': transform.SimilarityTransform(
                 scale=0.9970, rotation=-3e-4, translation=(1.3, 1.0))},
}


def _case(kind):
    task, bitColor, ref = make_layout(kind)
    truth = TRUTH['3col' if kind == 'tricolor' else '2col']
    return task, bitColor, ref, truth


class ChromaticFitTests(unittest.TestCase):

    def test_bit_colors_follow_data_organization(self):
        """Each bit's colour is looked up through its data channel, not by
        using the bit index as a data-channel index."""
        for kind in LAYOUTS:
            with self.subTest(kind=kind):
                task, bitColor, _, _ = _case(kind)
                self.assertEqual(task._get_bit_colors(), bitColor)
                self.assertEqual(task._get_used_colors(),
                                 sorted(set(bitColor)))
        task, bitColor, _, _ = _case('shifted')
        org = task.dataSet.get_data_organization()
        wrong = sum(org.get_data_channel_color(i) != c
                    for i, c in enumerate(bitColor))
        self.assertEqual(wrong, 13)     # the layout does exercise the bug

    def test_refine_position_axes(self):
        """refine_position(image, x, y) returns (col, row) with the right
        sign; only differences are used."""
        p0 = np.array([[300.0, 200.0]])
        base = np.array(registration.refine_position(render(p0), 300, 200))
        for d in [(0.6, 0.0), (0.0, 0.6), (-0.5, 0.4)]:
            got = np.array(registration.refine_position(
                render(p0 + np.array(d)), 300, 200)) - base
            np.testing.assert_allclose(got, d, atol=0.05)

    def test_composition_order_convention(self):
        """`tForm + previous` applies tForm first: (A + B)(x) == B(A(x))."""
        A = transform.SimilarityTransform(scale=1.05, rotation=0.05,
                                          translation=(10, -6))
        B = transform.SimilarityTransform(scale=0.998, translation=(1.0, -0.7))
        x = np.array([[100.0, 600.0], [700.0, 50.0]])
        np.testing.assert_allclose((A + B)(x), B(A(x)))

    def test_transform_image_undoes_truth(self):
        """Warping each colour's raw image with the TRUE transform puts its
        spots on the reference positions."""
        for kind in LAYOUTS:
            with self.subTest(kind=kind):
                task, bitColor, ref, truth = _case(kind)
                tr = identity(sorted(set(bitColor)))
                for c, T in truth.items():
                    tr[ref][c] = T
                raw, barcodes = make_scene(task, bitColor, truth,
                                           nMolecules=400)
                samples = task._measure_chromatic_samples(
                    corrected(raw, bitColor, tr, ref), barcodes)
                for c in truth:
                    _, disp = samples[ref][c]
                    self.assertGreater(len(disp), 100)
                    self.assertLess(np.abs(np.median(disp, axis=0)).max(), 0.03)
                    self.assertLess(np.percentile(np.abs(disp), 90), 0.15)

    def test_fit_recovers_known_transform(self):
        """One iteration from identity recovers the known transform
        (displacements here are <= 1.5 px)."""
        for kind in LAYOUTS:
            with self.subTest(kind=kind):
                task, bitColor, ref, truth = _case(kind)
                raw, barcodes = make_scene(task, bitColor, truth)
                fit = one_iteration(task, raw, barcodes, bitColor, ref,
                                    identity(sorted(set(bitColor))))
                for c, T in truth.items():
                    corr = field_corr(fit[ref][c], T)
                    err = field_error(fit[ref][c], T)
                    self.assertGreater(corr, 0.99, '%s->%s' % (ref, c))
                    self.assertLess(err, TOL_ONESHOT, '%s->%s' % (ref, c))

    def test_single_job_path_matches_fragment_path(self):
        """chromatic_from_fragments off: the single-job path measures the
        same samples as the per-fragment path and returns the same fit."""
        for kind in ('aligned', 'shifted'):
            with self.subTest(kind=kind):
                task, bitColor, ref, truth = _case(kind)
                raw, barcodes = make_scene(task, bitColor, truth,
                                           nMolecules=800)
                task.barcodes = barcodes
                task.dataSet.tasks['Warp'] = _Warp(
                    task.dataSet.get_data_organization(),
                    task.get_codebook(), raw)
                single = task._get_chromatic_transformations()
                fragment = one_iteration(task, raw, barcodes, bitColor, ref,
                                         identity(sorted(set(bitColor))))
                for c in truth:
                    np.testing.assert_allclose(single[ref][c].params,
                                               fragment[ref][c].params,
                                               atol=1e-9)
                    self.assertGreater(field_corr(single[ref][c], truth[c]),
                                       0.99)

    def test_fitted_transform_undoes_aberration(self):
        """After 3 iterations, transform_image with the FITTED transform puts
        every colour's spots on the reference positions. Measured against
        ground truth, so it does not depend on the dataorg's colour labels."""
        for kind in LAYOUTS:
            with self.subTest(kind=kind):
                task, bitColor, ref, truth = _case(kind)
                raw, barcodes = make_scene(task, bitColor, truth)
                tr = identity(sorted(set(bitColor)))
                for _ in range(3):
                    tr = one_iteration(task, raw, barcodes, bitColor, ref, tr)
                raw2, bc2 = make_scene(task, bitColor, truth, nMolecules=600,
                                       seed=7)          # fresh spots
                resid = ground_truth_residual(task, raw2, bc2, bitColor, tr,
                                              ref)
                self.assertLess(resid, 0.05)

    def test_composition_with_large_previous(self):
        """Previous iteration's transform P is large; truth = P o R with R
        small. Two iterations starting from P must reproduce the truth, and
        the opposite composition order must be measurably worse."""
        for kind in ('aligned', 'shifted'):
            with self.subTest(kind=kind):
                task, bitColor, ref, _ = _case(kind)
                P = transform.SimilarityTransform(scale=1.05, rotation=0.05,
                                                  translation=(10, -6))
                R = transform.SimilarityTransform(scale=0.998,
                                                  translation=(1.0, -0.7))
                truth = {'750': R + P}                  # R first, then P
                raw, barcodes = make_scene(task, bitColor, truth, margin=80)
                prev = identity(sorted(set(bitColor)))
                prev[ref]['750'] = P
                fit = one_iteration(task, raw, barcodes, bitColor, ref, prev)
                # the 2nd pass removes the one-shot bias
                fit = one_iteration(task, raw, barcodes, bitColor, ref,
                                    fit)[ref]['750']
                err = field_error(fit, truth['750'])
                rHat = transform.SimilarityTransform(
                    np.linalg.inv(P.params) @ fit.params)     # fit = P o rHat
                errSwapped = field_error(P + rHat, truth['750'])
                self.assertLess(err, TOL_PX)
                self.assertGreater(errSwapped, 3 * TOL_PX)   # test has power

    def test_loop_converges(self):
        """Each iteration corrects with the previous estimate, measures and
        composes, as OptimizeLoop does."""
        for kind in LAYOUTS:
            with self.subTest(kind=kind):
                task, bitColor, ref, truth = _case(kind)
                raw, barcodes = make_scene(task, bitColor, truth)
                tr = identity(sorted(set(bitColor)))
                for _ in range(4):
                    tr = one_iteration(task, raw, barcodes, bitColor, ref, tr)
                worst = max(field_error(tr[ref][c], truth[c]) for c in truth)
                self.assertLess(worst, TOL_PX)


if __name__ == '__main__':
    unittest.main()
