"""DeduplicateBarcodes and GenerateMoleculeAdaptiveThreshold.

Hand-built molecules check the linking rule (plane gaps, a second molecule at
the same xy deeper down, other barcode ids at the same xy, same-plane pairs,
the radius and gap at their limits, uneven plane spacing) and each
molecule_features mode. Random tables are compared with a slow reference that
shares no code with the module: a loop over all pairs, union-find, and
per-molecule sums written out by hand. The last test runs Decode ->
DeduplicateBarcodes -> GenerateMoleculeAdaptiveThreshold ->
AdaptiveFilterBarcodes on a small synthetic dataset.

Run with `python -m unittest tests/test_deduplicate.py`.
"""
import contextlib
import io
import math
import tempfile
import unittest

import numpy as np
import pandas as pd

from merlin.analysis import decode
from merlin.analysis import deduplicate
from merlin.analysis import filterbarcodes
from merlin.core import dataset

Z_UNIFORM = [1.0 + 0.5 * k for k in range(12)]


def detections(rows):
    """Decode-like table from (barcode_id, x, y, z, mean_intensity, area,
    min_distance) tuples; the other columns are derived from those."""
    frame = pd.DataFrame(rows, columns=['barcode_id', 'x', 'y', 'z',
                                        'mean_intensity', 'area',
                                        'min_distance'])
    n = len(frame)
    out = pd.DataFrame({
        'unique_id': np.arange(1, n + 1, dtype=np.int32),
        'barcode_id': frame['barcode_id'].astype(np.int32),
        'area': frame['area'].astype(np.int16),
        'mean_intensity': frame['mean_intensity'].astype(np.float32),
        'max_intensity': (frame['mean_intensity'] * 1.5).astype(np.float32),
        'mean_distance': (frame['min_distance'] + 0.05).astype(np.float32),
        'min_distance': frame['min_distance'].astype(np.float32),
        'x': frame['x'].astype(np.float32),
        'y': frame['y'].astype(np.float32),
        'z': frame['z'].astype(np.float32),
        'fov': np.zeros(n, dtype=np.int32),
        'cell_index': np.full(n, -1, dtype=np.int32)})
    out['global_z'] = out['z']
    out['global_x'] = (out['x'] * 0.14625 + 100).astype(np.float32)
    out['global_y'] = (out['y'] * 0.14625 - 50).astype(np.float32)
    out['intensity_0'] = (out['mean_intensity'] * 0.3).astype(np.float32)
    return out


def run(table, xy=2.0, gap=0.75, mode='brightest', zPositions=Z_UNIFORM):
    molecules, _ = deduplicate.deduplicate_barcodes(table, zPositions, xy,
                                                    gap, mode)
    return molecules


def groups(table, molecules):
    """Sorted molecule sizes (detections per molecule)."""
    self_check = int(molecules['n_detections'].sum())
    assert self_check == len(table), (self_check, len(table))
    return sorted(int(n) for n in molecules['n_detections'])


# ---------------------------------------------------------------- reference

def reference(table, zPositions, xy, gap, mode):
    """Slow, independent: all pairs, union-find, explicit sums."""
    table = table.reset_index(drop=True)
    n = len(table)
    rows = table.to_dict('records')
    depth = [float(np.interp(r['z'], np.arange(len(zPositions)), zPositions))
             for r in rows]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if rows[i]['barcode_id'] != rows[j]['barcode_id']:
                continue
            dxy = math.hypot(float(rows[i]['x']) - float(rows[j]['x']),
                             float(rows[i]['y']) - float(rows[j]['y']))
            dz = abs(depth[i] - depth[j])
            if dxy <= xy + 1e-6 and 1e-6 < dz <= gap + 1e-6:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    members = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    ordered = sorted(members.values(), key=min)

    result = []
    for group in ordered:
        best = group[0]
        for i in group[1:]:
            a = float(rows[i]['mean_intensity'])
            b = float(rows[best]['mean_intensity'])
            if a > b or (a == b and rows[i]['z'] < rows[best]['z']):
                best = i
        areaSum = sum(float(rows[i]['area']) for i in group)
        w = [float(rows[i]['mean_intensity']) * float(rows[i]['area'])
             for i in group]
        if not sum(w) > 0:
            w = [float(rows[i]['area']) for i in group]
        wsum = sum(w)
        record = {'unique_id': rows[best]['unique_id'],
                  'barcode_id': rows[best]['barcode_id'],
                  'z': rows[best]['z'],
                  'n_detections': len(group),
                  'n_planes': len({int(round(rows[i]['z'])) for i in group}),
                  'area_sum': areaSum,
                  'z_first_um': min(depth[i] for i in group),
                  'z_last_um': max(depth[i] for i in group),
                  'z_um': sum(wk * depth[i] for wk, i in zip(w, group)) / wsum,
                  'max_intensity': max(float(rows[i]['max_intensity'])
                                       for i in group)}
        for c in ('x', 'y', 'global_x', 'global_y', 'global_z'):
            record[c] = sum(wk * float(rows[i][c])
                            for wk, i in zip(w, group)) / wsum
        for c in ('mean_distance', 'intensity_0'):
            record[c] = sum(float(rows[i]['area']) * float(rows[i][c])
                            for i in group) / areaSum
        b = rows[best]
        if mode == 'brightest':
            record.update(mean_intensity=b['mean_intensity'],
                          min_distance=b['min_distance'], area=b['area'])
        elif mode == 'brightest_sum_area':
            record.update(mean_intensity=b['mean_intensity'],
                          min_distance=b['min_distance'], area=areaSum)
        else:
            record.update(
                mean_intensity=sum(float(rows[i]['area'])
                                   * float(rows[i]['mean_intensity'])
                                   for i in group) / areaSum,
                min_distance=sum(float(rows[i]['area'])
                                 * float(rows[i]['min_distance'])
                                 for i in group) / areaSum,
                area=areaSum)
        result.append(record)
    return pd.DataFrame(result)


def random_table(seed, n=260):
    """Overlapping molecules of 3 ids in a 24 x 24 px field over 12 planes:
    1-5 detections each, plane steps of 1-3, xy jitter up to 1.6 px, tied
    intensities and some zero-intensity detections."""
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        bid = int(rng.integers(3))
        x0, y0 = rng.uniform(0, 24, 2)
        z = int(rng.integers(12))
        for _ in range(int(rng.integers(1, 6))):
            if z >= 12:
                break
            rows.append((bid, x0 + rng.uniform(-0.8, 0.8),
                         y0 + rng.uniform(-0.8, 0.8), z,
                         float(rng.integers(0, 6)) * 10.0,
                         int(rng.integers(2, 9)),
                         round(float(rng.uniform(0, 0.65)), 3)))
            z += int(rng.integers(1, 4))
    order = rng.permutation(len(rows))
    return detections([rows[k] for k in order])


class LinkingRuleTests(unittest.TestCase):

    def test_plane_gaps(self):
        # one molecule on planes 2, 3, 5 (0.5 um steps): a 1.0 um gap links
        # all three, a 0.5 um gap splits off plane 5
        table = detections([(0, 10, 10, 2, 50, 4, 0.2),
                            (0, 10.5, 10, 3, 80, 5, 0.1),
                            (0, 10, 10.5, 5, 30, 3, 0.3)])
        self.assertEqual(groups(table, run(table, gap=1.0)), [3])
        self.assertEqual(groups(table, run(table, gap=0.5)), [1, 2])
        self.assertEqual(groups(table, run(table, gap=0)), [1, 1, 1])

    def test_spans_one_to_five_planes(self):
        rows = []
        for k, span in enumerate(range(1, 6)):
            for p in range(span):
                rows.append((0, 5 + 20 * k, 5, p, 10 + p, 3, 0.2))
        molecules = run(detections(rows), gap=0.5)
        self.assertEqual(molecules['n_planes'].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(molecules['z_first_um'].tolist(), [1.0] * 5)
        np.testing.assert_allclose(molecules['z_last_um'],
                                   [1.0, 1.5, 2.0, 2.5, 3.0])

    def test_second_molecule_deeper_at_same_xy(self):
        table = detections([(0, 10, 10, 0, 50, 4, 0.2),
                            (0, 10, 10, 1, 60, 4, 0.2),
                            (0, 10, 10, 6, 50, 4, 0.2),
                            (0, 10, 10, 7, 40, 4, 0.2)])
        molecules = run(table, gap=0.75)
        self.assertEqual(molecules['n_detections'].tolist(), [2, 2])
        self.assertEqual(molecules['z'].tolist(), [1, 6])

    def test_other_barcode_id_at_same_xy(self):
        table = detections([(0, 10, 10, 0, 50, 4, 0.2),
                            (1, 10, 10, 1, 60, 4, 0.2),
                            (2, 10, 10, 2, 50, 4, 0.2)])
        self.assertEqual(groups(table, run(table, gap=5.0)), [1, 1, 1])

    def test_same_plane_never_linked_directly(self):
        table = detections([(0, 10, 10, 3, 50, 4, 0.2),
                            (0, 11, 10, 3, 60, 4, 0.2)])
        self.assertEqual(groups(table, run(table)), [1, 1])
        # ... but a detection on the next plane bridges them
        bridged = detections([(0, 10, 10, 3, 50, 4, 0.2),
                              (0, 11, 10, 3, 60, 4, 0.2),
                              (0, 10.5, 10, 4, 40, 4, 0.2)])
        molecules = run(bridged)
        self.assertEqual(molecules['n_detections'].tolist(), [3])
        self.assertEqual(molecules['n_planes'].tolist(), [2])

    def test_radius_and_gap_limits(self):
        table = detections([(0, 0, 0, 0, 50, 4, 0.2),
                            (0, 1.2, 1.6, 1, 50, 4, 0.2),        # 2.0 px
                            (0, 40, 0, 0, 50, 4, 0.2),
                            (0, 42.001, 0, 1, 50, 4, 0.2)])     # 2.001 px
        self.assertEqual(run(table, xy=2.0, gap=0.5)['n_detections'].tolist(),
                         [2, 1, 1])
        table = detections([(0, 0, 0, 0, 50, 4, 0.2),
                            (0, 0, 0, 2, 50, 4, 0.2)])         # 1.0 um
        self.assertEqual(groups(table, run(table, gap=1.0)), [2])
        self.assertEqual(groups(table, run(table, gap=0.999)), [1, 1])

    def test_depths_come_from_the_z_positions(self):
        # planes 0, 1, 2 at 0, 0.4, 1.4 um: a 0.5 um gap links 0-1 only
        zPositions = [0.0, 0.4, 1.4]
        table = detections([(0, 5, 5, 0, 50, 4, 0.2),
                            (0, 5, 5, 1, 50, 4, 0.2),
                            (0, 5, 5, 2, 50, 4, 0.2)])
        molecules = run(table, gap=0.5, zPositions=zPositions)
        self.assertEqual(molecules['n_detections'].tolist(), [2, 1])
        np.testing.assert_allclose(molecules['z_last_um'], [0.4, 1.4])

    def test_labels_numbered_by_first_member(self):
        np.testing.assert_array_equal(
            deduplicate._labels_by_first_member(np.array([2, 0, 2, 1, 0])),
            [0, 1, 0, 2, 1])
        ordered = np.array([0, 1, 0, 2, 1, 3])
        np.testing.assert_array_equal(
            deduplicate._labels_by_first_member(ordered), ordered)

    def test_empty_table(self):
        table = detections([])
        molecules, stats = deduplicate.deduplicate_barcodes(
            table, Z_UNIFORM, 2.0, 0.75, 'all_planes')
        self.assertEqual(len(molecules), 0)
        for column in deduplicate.MOLECULE_COLUMNS:
            self.assertIn(column, molecules.columns)
        self.assertEqual(stats['n_molecules'], 0)


class FeatureModeTests(unittest.TestCase):

    TABLE = detections([(4, 10.0, 10.0, 2, 40.0, 3, 0.30),
                        (4, 10.4, 10.0, 3, 90.0, 5, 0.10),
                        (4, 10.0, 10.6, 4, 60.0, 4, 0.20),
                        (4, 30.0, 30.0, 4, 70.0, 6, 0.25)])

    def test_brightest(self):
        m = run(self.TABLE, gap=0.5, mode='brightest')
        self.assertEqual(m['mean_intensity'].tolist(), [90.0, 70.0])
        np.testing.assert_allclose(m['min_distance'], [0.10, 0.25], 1e-6)
        self.assertEqual(m['area'].tolist(), [5, 6])
        self.assertEqual(m['area_sum'].tolist(), [12, 6])

    def test_brightest_sum_area(self):
        m = run(self.TABLE, gap=0.5, mode='brightest_sum_area')
        self.assertEqual(m['mean_intensity'].tolist(), [90.0, 70.0])
        np.testing.assert_allclose(m['min_distance'], [0.10, 0.25], 1e-6)
        self.assertEqual(m['area'].tolist(), [12, 6])

    def test_all_planes(self):
        m = run(self.TABLE, gap=0.5, mode='all_planes')
        np.testing.assert_allclose(
            m['mean_intensity'], [(40 * 3 + 90 * 5 + 60 * 4) / 12, 70.0], 1e-6)
        np.testing.assert_allclose(
            m['min_distance'],
            [(0.30 * 3 + 0.10 * 5 + 0.20 * 4) / 12, 0.25], 1e-6)
        self.assertEqual(m['area'].tolist(), [12, 6])

    def test_shared_columns(self):
        m = run(self.TABLE, gap=0.5, mode='all_planes')
        w = np.array([40 * 3, 90 * 5, 60 * 4], float)
        self.assertEqual(m['z'].tolist(), [3, 4])                 # brightest
        self.assertEqual(m['unique_id'].tolist(), [2, 4])
        np.testing.assert_allclose(m['x'][0], (w @ [10, 10.4, 10]) / w.sum(),
                                   1e-6)
        np.testing.assert_allclose(m['z_um'][0],
                                   (w @ [2.0, 2.5, 3.0]) / w.sum(), 1e-6)
        np.testing.assert_allclose(m['max_intensity'], [135.0, 105.0])
        np.testing.assert_allclose(
            m['mean_distance'][0], (0.35 * 3 + 0.15 * 5 + 0.25 * 4) / 12,
            1e-6)
        self.assertEqual(m['n_planes'].tolist(), [3, 1])

    def test_single_detections_unchanged(self):
        table = random_table(3, 40)
        for mode in deduplicate.MOLECULE_FEATURES:
            molecules = run(table, gap=0, mode=mode)
            for column in table.columns:
                np.testing.assert_array_equal(
                    molecules[column].to_numpy(),
                    table[column].to_numpy().astype(
                        molecules[column].dtype), err_msg=column)

    def test_area_clipped_to_database_range(self):
        rows = [(0, 5, 5, p, 10, 30000, 0.2) for p in range(3)]
        m = run(detections(rows), gap=0.5, mode='all_planes')
        self.assertEqual(int(m['area'][0]), 65535)
        self.assertEqual(int(m['area_sum'][0]), 90000)

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            run(self.TABLE, mode='mean')


class ReferenceEqualityTests(unittest.TestCase):

    def test_matches_slow_reference(self):
        columns = ['unique_id', 'barcode_id', 'z', 'n_detections', 'n_planes',
                   'area_sum', 'z_first_um', 'z_last_um', 'z_um',
                   'max_intensity', 'x', 'y', 'global_x', 'global_y',
                   'global_z', 'mean_distance', 'intensity_0',
                   'mean_intensity', 'min_distance', 'area']
        checked = 0
        for seed in range(3):
            table = random_table(seed)
            for xy, gap in ((2.0, 0.75), (1.5, 0.5), (2.5, 1.5)):
                for mode in deduplicate.MOLECULE_FEATURES:
                    got = run(table, xy, gap, mode)
                    want = reference(table, Z_UNIFORM, xy, gap, mode)
                    self.assertEqual(len(got), len(want))
                    self.assertLess(len(got), len(table))
                    for c in columns:
                        np.testing.assert_allclose(
                            got[c].to_numpy(np.float64),
                            want[c].to_numpy(np.float64), rtol=2e-6,
                            atol=1e-5, err_msg='%s %s %s %s'
                            % (seed, (xy, gap), mode, c))
                    checked += 1
        self.assertEqual(checked, 27)


# --------------------------------------------------------- pipeline chain

class _Codebook:
    """Ids 0-7 coding, 8-9 blank, 4 bits."""

    def get_blank_indexes(self):
        return [8, 9]

    def get_coding_indexes(self):
        return list(range(8))

    def get_bit_count(self):
        return 4

    def get_barcode_count(self):
        return 10


class _DataSet(dataset.DataSet):
    def __init__(self, root):
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__('synthetic', dataHome=root, analysisHome=root)

    def get_fovs(self):
        return [0, 1]

    def get_z_positions(self):
        return Z_UNIFORM

    def get_image_dimensions(self):
        return (64, 64)


class _SyntheticDecode(decode.BarcodeSavingParallelAnalysisTask):
    """Writes a fixed random table per fov: coding molecules seen on 1-4
    planes, blank calls mostly on one, coding brighter on average."""

    outputGroup = 'Decode'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self.parameters.setdefault('distance_threshold', 0.65)
        self.parameters.setdefault('fovs', None)
        self.parameters.setdefault('decode_z_index', None)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 100

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return []

    def get_codebook(self):
        return _Codebook()

    def _run_analysis(self, fragmentIndex):
        rng = np.random.default_rng(100 + fragmentIndex)
        rows = []
        for k in range(600):
            blank = k % 6 == 0
            bid = int(rng.integers(8, 10)) if blank else int(rng.integers(8))
            x0, y0 = rng.uniform(4, 60, 2)
            z0 = int(rng.integers(10))
            span = 1 if (blank and rng.random() < 0.85) \
                else int(rng.integers(1, 5))
            for p in range(span):
                if z0 + p >= len(Z_UNIFORM):
                    break
                rows.append((bid, x0 + rng.normal(0, 0.3),
                             y0 + rng.normal(0, 0.3), z0 + p,
                             float(rng.lognormal(3.0 if blank else 3.6, 0.4)),
                             int(rng.integers(2, 7)),
                             float(rng.uniform(0.2 if blank else 0.0, 0.65))))
        table = detections(rows)
        table['fov'] = fragmentIndex
        self.get_barcode_database().write_barcodes(table, fov=fragmentIndex)


class ChainTests(unittest.TestCase):

    def test_threshold_and_filter_on_molecules(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = _DataSet(root)
            decodeTask = _SyntheticDecode(ds, {}, 'Decode')
            decodeTask.save()
            dedup = deduplicate.DeduplicateBarcodes(
                ds, {'decode_task': 'Decode',
                     'molecule_features': 'all_planes'}, 'Dedup')
            dedup.save()
            threshold = deduplicate.GenerateMoleculeAdaptiveThreshold(
                ds, {'decode_task': 'Dedup', 'run_after_task': 'Dedup',
                     'poll_interval': 0}, 'Thresh')
            threshold.save()
            filterTask = filterbarcodes.AdaptiveFilterBarcodes(
                ds, {'decode_task': 'Dedup', 'adaptive_task': 'Thresh',
                     'write_filtered_images': False}, 'Filter')
            filterTask.save()

            # the decode parameters the threshold reads are inherited
            self.assertEqual(dedup.parameters['distance_threshold'], 0.65)
            self.assertIsNone(dedup.parameters['fovs'])
            self.assertIsInstance(dedup.get_codebook(), _Codebook)

            for fov in ds.get_fovs():
                decodeTask.run(fov)
                dedup.run(fov)
            dedup.run(0)          # a rerun replaces the fov's molecules
            molecules = [dedup.get_barcode_database().get_barcodes(f)
                         for f in ds.get_fovs()]
            raw = [decodeTask.get_barcode_database().get_barcodes(f)
                   for f in ds.get_fovs()]
            for fov in ds.get_fovs():
                again, _ = deduplicate.deduplicate_barcodes(
                    raw[fov], Z_UNIFORM, 2.0, 0.75, 'all_planes')
                self.assertEqual(len(molecules[fov]), len(again))
                self.assertEqual(int(molecules[fov]['n_detections'].sum()),
                                 len(raw[fov]))
                self.assertLess(len(molecules[fov]), len(raw[fov]))
                summary = dedup.get_summary(fov)
                self.assertEqual(summary['n_molecules'], len(again))
                self.assertGreater(
                    summary['coding_detections'] / summary['coding_molecules'],
                    summary['blank_detections'] / summary['blank_molecules'])

            threshold.run()
            areaBins = threshold.get_area_bins()
            allAreas = np.concatenate([m['area'].to_numpy()
                                       for m in molecules])
            self.assertEqual(areaBins[0], allAreas.min())
            self.assertEqual(areaBins[-1], allAreas.max() + 1)
            self.assertGreater(areaBins[-1], 21)        # past the unit edges
            self.assertLessEqual(len(areaBins) - 1, 20)
            self.assertEqual(threshold.get_total_count_histogram().sum(),
                             sum(len(m) for m in molecules))

            for fov in ds.get_fovs():
                filterTask.run(fov)
            for fov in ds.get_fovs():
                kept = filterTask.get_barcode_database().get_barcodes(fov)
                self.assertGreater(len(kept), 0)
                self.assertLess(len(kept), len(molecules[fov]))
                self.assertTrue(set(kept['unique_id']).issubset(
                    set(molecules[fov]['unique_id'])))
                for column in deduplicate.MOLECULE_COLUMNS:
                    self.assertIn(column, kept.columns)

    def test_unit_area_edges(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = _DataSet(root)
            decodeTask = _SyntheticDecode(ds, {}, 'Decode')
            decodeTask.save()
            dedup = deduplicate.DeduplicateBarcodes(
                ds, {'decode_task': 'Decode'}, 'Dedup')
            dedup.save()
            threshold = deduplicate.GenerateMoleculeAdaptiveThreshold(
                ds, {'decode_task': 'Dedup', 'run_after_task': 'Dedup',
                     'poll_interval': 0, 'area_bin_edges': 'unit'}, 'Thresh')
            threshold.save()
            for fov in ds.get_fovs():
                decodeTask.run(fov)
                dedup.run(fov)
            threshold.run()
            np.testing.assert_array_equal(threshold.get_area_bins(),
                                          np.arange(1, 22))


class AreaEdgeTests(unittest.TestCase):

    def test_quantile_edges(self):
        areas = np.r_[np.full(500, 2), np.full(300, 3), np.arange(4, 204)]
        edges = deduplicate.area_quantile_edges(areas, 10)
        self.assertEqual(edges[0], 2)
        self.assertEqual(edges[-1], 204)
        self.assertTrue(np.all(np.diff(edges) > 0))
        self.assertTrue(np.all(edges == np.floor(edges)))
        self.assertLess(len(edges) - 1, 10)              # ties merged
        counts = np.histogram(areas, edges)[0]
        self.assertEqual(counts.sum(), len(areas))

    def test_empty_falls_back_to_unit_edges(self):
        np.testing.assert_array_equal(
            deduplicate.area_quantile_edges(np.array([]), 5), np.arange(1, 7))


if __name__ == '__main__':
    unittest.main()
