"""Scientific invariants of the opt-in raw calibration path."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from skimage import transform

from merlin.util import bitregistration as br
from merlin.analysis import decode, warp
from merlin.analysis.rawcalibration import BitAlignedWarp, OptimizeRawCalibration, _check_base_warp, _state_id
from merlin.util.aberration import RigidChromaticCorrector


class BitRegistrationTests(unittest.TestCase):
    def test_recover_offsets_with_equal_molecule_weights_and_outlier(self):
        expected = np.array([[.3, -.2], [-.4, .1], [.1, .1]])
        i = np.tile([0, 0, 1], 150); j = np.tile([1, 2, 2], 150)
        rng = np.random.default_rng(24)
        d = expected[i] - expected[j] + rng.normal(0, .02, (len(i), 2))
        d[:12] += 2
        got, _ = br.robust_offsets(i, j, d, np.ones(len(i))/3, 3)
        np.testing.assert_allclose(got, expected, atol=.015)
        np.testing.assert_allclose(got.mean(0), 0, atol=1e-12)

    def test_disconnected_graph_rejected(self):
        with self.assertRaisesRegex(ValueError, 'connect'):
            br.robust_offsets([0, 0], [1, 1], [[0, 0], [0, 0]], [1, 1], 3)

    def test_pair_weight_is_one_per_molecule(self):
        table = pd.DataFrame({'barcode_id': [0, 1]})
        book = np.array([[1, 1, 0], [1, 1, 1]])
        p = np.zeros((2, 3, 2))
        _, _, _, w, n, _, _ = br.molecule_pairs(table, book, p, np.ones(2, bool))
        self.assertEqual(n, 2)
        np.testing.assert_allclose(w, [1, 1/3, 1/3, 1/3])

    def test_empty_candidate_field_can_fall_back_to_baseline(self):
        table = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in
            [('barcode_id', int), ('area', int), ('min_distance', float), ('x', float), ('y', float)]})
        selected = br.select_molecules(table, np.array([False, True]), .5176)
        self.assertEqual(len(selected), 0)
        self.assertIn('barcode_id', selected)

    def test_after_outlier_not_discarded_and_missing_is_explicit(self):
        table = pd.DataFrame({'barcode_id': [0, 0, 0]})
        book = np.ones((1, 3))
        before = np.zeros((3, 3, 2)); after = before.copy()
        after[0, 0] = [20, 0]; after[2, 0] = np.nan
        _, _, b, _, n, missing, a = br.molecule_pairs(table, book, before, np.ones(3, bool), after)
        self.assertEqual((n, missing, len(a)), (3, 1, 6))
        self.assertGreater(np.max(np.linalg.norm(a, axis=1)), 19)
        np.testing.assert_array_equal(b, np.zeros((6, 2)))

    def test_sampler_composition_and_dtype_exactly_match_two_pass_reference(self):
        rng = np.random.default_rng(25)
        image = rng.integers(0, 1000, (38, 49), dtype=np.uint16)
        c = transform.SimilarityTransform(scale=.997, translation=(.2, -.1))
        w = transform.SimilarityTransform(rotation=.01, translation=(-1.3, .7))
        corrector = RigidChromaticCorrector({'560': {'560': transform.SimilarityTransform(), '650': c}}, '560')
        obj = object.__new__(BitAlignedWarp)
        obj.dataSet = SimpleNamespace(get_raw_image=lambda *a: image,
            z_index_to_position=lambda z: 1.5*z,
            get_data_organization=lambda: SimpleNamespace(get_data_channel_color=lambda ch: '650'))
        obj._channels = [7]
        obj._calibration = SimpleNamespace(get_previous_chromatic_corrector=lambda: corrector)
        obj._base = SimpleNamespace(get_transformation=lambda *a: w)
        obj._bits = SimpleNamespace(get_offsets=lambda f, z: np.array([[.4, -.2]]))
        t = np.eye(3); t[:2, 2] = [.4, -.2]
        expected = transform.warp(corrector.transform_image(image, '650'),
                                  transform.AffineTransform(matrix=w.params @ t), preserve_range=True)
        np.testing.assert_array_equal(obj.get_aligned_image(19, 7, 3), expected)
        # Wrong multiplication order must be visible on this rotated example.
        wrong = transform.warp(corrector.transform_image(image, '650'),
                               transform.AffineTransform(matrix=t @ w.params), preserve_range=True)
        self.assertGreater(np.max(np.abs(wrong-expected)), .1)
        obj.parameters = {'apply_bit_alignment': False}
        baseline = transform.warp(corrector.transform_image(image, '650'), w, preserve_range=True)
        np.testing.assert_array_equal(obj.get_aligned_image(19, 7, 3), baseline)

    def test_crop_dispatch_uses_z_and_chromatic_support(self):
        support = [transform.AffineTransform(translation=(-7, 0)), transform.AffineTransform(translation=(5, 0))]
        task = SimpleNamespace(get_crop_transformations=Mock(return_value=support))
        self.assertEqual(decode.warp_transformations(task, 19, [7]), support)
        task.get_crop_transformations.assert_called_once_with(19, [7])

    def test_missing_z_raises_and_state_hash_includes_absolute_scale(self):
        obj = object.__new__(BitAlignedWarp)
        obj._bits = SimpleNamespace(parameters={'z_indices': [0, 1]})
        with self.assertRaisesRegex(ValueError, 'explicit zIndex'):
            obj.get_transformation(19, 7)
        self.assertNotEqual(_state_id({'scales': [1, 2]}), _state_id({'scales': [2, 4]}))

    def test_freeze_uses_applied_map_with_absolute_scales(self):
        applied = RigidChromaticCorrector({'560': {'650': transform.SimilarityTransform(translation=(2, 0))}}, '560')
        proposed = RigidChromaticCorrector({'560': {'650': transform.SimilarityTransform(translation=(5, 0))}}, '560')
        opt = SimpleNamespace(parameters={'normalize_scale_factors': False},
            get_scale_factors=lambda: np.array([40., 80.]), get_backgrounds=lambda: np.zeros(2),
            get_reference_color=lambda: '560', get_chromatic_corrector=lambda: proposed,
            get_previous_chromatic_corrector=lambda: applied)
        obj = object.__new__(OptimizeRawCalibration)
        obj.parameters = {'max_iterations': 0, 'bootstrap_task': 'NewRawBootstrap'}
        obj.dataSet = SimpleNamespace(save_json_analysis_result=Mock())
        obj.analysisName = 'NewCalibration'; obj._bootstrap = lambda: opt
        obj._run_analysis()
        saved = obj.dataSet.save_json_analysis_result.call_args.args[0]
        self.assertEqual(saved['state']['scales'], [40., 80.])
        self.assertEqual(saved['state']['chromatic']['560']['650'][0][2], 2.)

    def test_modified_frozen_global_state_is_rejected(self):
        obj = object.__new__(OptimizeRawCalibration)
        obj.is_complete = lambda: True
        record = dict(state={'scales': [40, 80]}, calibration_id=_state_id({'scales': [20, 40]}))
        obj.dataSet = SimpleNamespace(load_json_analysis_result=lambda *a: record)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            obj.get_state()

    def test_crop_bounds_both_sampling_stages_and_reference_identity(self):
        c = transform.SimilarityTransform(scale=.99, translation=(4, -2))
        corrector = RigidChromaticCorrector({'560': {
            '560': transform.SimilarityTransform(translation=(100, 100)), '650': c}}, '560')
        w = transform.SimilarityTransform(translation=(-3, 1))
        obj = object.__new__(BitAlignedWarp)
        obj._calibration = SimpleNamespace(get_previous_chromatic_corrector=lambda: corrector)
        obj._bits = SimpleNamespace(parameters={'z_indices': [0, 3]})
        obj.dataSet = SimpleNamespace(get_data_organization=lambda: SimpleNamespace(
            get_data_channels=lambda: [0, 1], get_data_channel_color=lambda ch: ['560', '650'][ch]))
        obj.get_transformation = lambda f, ch, z: w
        ts = obj.get_crop_transformations(19)
        self.assertEqual(len(ts), 6)
        np.testing.assert_array_equal(ts[0].params, w.params)
        np.testing.assert_array_equal(ts[2].params, c.params @ w.params)

    def test_baseline_and_corrected_use_identical_common_support(self):
        corrector = RigidChromaticCorrector({'560': {'650': transform.SimilarityTransform(
            scale=.99, translation=(4, -2))}}, '560')
        obj = object.__new__(BitAlignedWarp)
        obj._channels = [7]
        obj._calibration = SimpleNamespace(get_previous_chromatic_corrector=lambda: corrector)
        obj._bits = SimpleNamespace(parameters={'z_indices': [0, 3]},
            get_offsets=lambda f, z: np.array([[.4+z*.1, -.2]]))
        obj._base = SimpleNamespace(get_transformation=lambda *a: transform.SimilarityTransform(translation=(-3, 1)))
        obj.dataSet = SimpleNamespace(get_data_organization=lambda: SimpleNamespace(
            get_data_channels=lambda: [7], get_data_channel_color=lambda ch: '650'))
        matrices = []
        for applied in [False, True]:
            obj.parameters = dict(apply_bit_alignment=applied, common_support_with_bit_alignment=True)
            matrices.append(np.array([t.params for t in obj.get_crop_transformations(19)]))
        np.testing.assert_array_equal(matrices[0], matrices[1])
        self.assertEqual(len(matrices[0]), 8)

    def test_base_warp_that_moves_with_depth_is_refused(self):
        def table(**change):
            t = pd.DataFrame({'dataChannel': [7, 7, 8], 'zIndex': [0, 1, 0], 'zPos': [0., 1.5, 0.],
                              'zPos_new': [0., 1.5, 0.], 'xshift': [-2., -2., 5.], 'yshift': [1., 1., 0.]})
            for k, v in change.items():
                t[k] = v
            return t
        poly = object.__new__(warp.FiducialPolynomialWarp3D)
        poly.get_transformation_table = lambda fov: table()
        _check_base_warp(poly, 19, [7, 8])  # channels may differ, planes may not
        for change in [dict(zPos_new=[.2, 1.7, 0.]), dict(xshift=[-2., -2.5, 5.])]:
            poly.get_transformation_table = lambda fov, c=change: table(**c)
            with self.assertRaisesRegex(ValueError, 'depth-independent'):
                _check_base_warp(poly, 19, [7, 8])
        flat = object.__new__(warp.FiducialCorrelationWarp)
        flat.parameters = {'register_3d': False}
        _check_base_warp(flat, 19, [7])
        flat.parameters = {'register_3d': True}
        with self.assertRaisesRegex(ValueError, 'does not support'):
            _check_base_warp(flat, 19, [7])


if __name__ == '__main__':
    unittest.main()
