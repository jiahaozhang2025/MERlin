"""DeduplicateBarcodes and GenerateMoleculeAdaptiveThreshold.

Hand-built molecules check the linking rule (plane gaps, a second molecule at
the same xy deeper down, other barcode ids at the same xy, same-plane pairs,
the radius and gap at their limits, uneven plane spacing) and each
molecule_features mode ('sum', 'average', 'brightest'; the old names 'summed'
and 'all_planes', read as 'sum', and the removed 'brightest_sum_area'), with
every column of a 'sum' and an 'average' table worked out by hand. Random
tables are compared with a slow
reference that shares no code with the module: a loop over all pairs,
union-find, and per-molecule sums written out by hand. The chain tests run
Decode -> DeduplicateBarcodes -> GenerateMoleculeAdaptiveThreshold ->
AdaptiveFilterBarcodes on a small synthetic dataset. The valid-box tests give
a stand-in decode task Decode's crop parameters and a warp task with known
shifts and a rotation, and check crop_to_valid_box's region against
decode.compute_crop_bounds and Decode._crop_bounds, its edges, the
image-space and adaptive_crop-off cases, and the per-fov counts.

Run with `python -m unittest tests/test_deduplicate.py`.
"""
import contextlib
import io
import json
import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from skimage import transform

from merlin.analysis import decode
from merlin.analysis import deduplicate
from merlin.analysis import filterbarcodes
from merlin.core import analysistask
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


def run(table, xy=2.0, gap=0.75, mode='sum', zPositions=Z_UNIFORM):
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
        elif mode == 'average':
            k = len(group)
            record.update(
                mean_intensity=sum(float(rows[i]['mean_intensity'])
                                   for i in group) / k,
                min_distance=sum(float(rows[i]['min_distance'])
                                 for i in group) / k,
                area=math.floor(areaSum / k))
        else:
            assert mode == 'sum', mode
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
            table, Z_UNIFORM, 2.0, 0.75, 'sum')
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

    def test_sum(self):
        m = run(self.TABLE, gap=0.5, mode='sum')
        np.testing.assert_allclose(
            m['mean_intensity'], [(40 * 3 + 90 * 5 + 60 * 4) / 12, 70.0], 1e-6)
        np.testing.assert_allclose(
            m['min_distance'],
            [(0.30 * 3 + 0.10 * 5 + 0.20 * 4) / 12, 0.25], 1e-6)
        self.assertEqual(m['area'].tolist(), [12, 6])

    def test_average(self):
        m = run(self.TABLE, gap=0.5, mode='average')
        np.testing.assert_allclose(
            m['mean_intensity'], [(40 + 90 + 60) / 3, 70.0], 1e-6)
        np.testing.assert_allclose(
            m['min_distance'], [(0.30 + 0.10 + 0.20) / 3, 0.25], 1e-6)
        self.assertEqual(m['area'].tolist(), [4, 6])          # 12 / 3, 6 / 1
        self.assertEqual(m['area_sum'].tolist(), [12, 6])

    def test_modes(self):
        self.assertEqual(deduplicate.MOLECULE_FEATURES,
                         ('sum', 'average', 'brightest'))

    def test_old_names_are_sum(self):
        for old in ('summed', 'all_planes'):
            self.assertEqual(deduplicate.molecule_features_option(old), 'sum')
            pd.testing.assert_frame_equal(
                run(self.TABLE, gap=0.5, mode=old),
                run(self.TABLE, gap=0.5, mode='sum'), check_exact=True)

    def test_brightest_sum_area_refused(self):
        with self.assertRaisesRegex(ValueError, "'brightest_sum_area' was "
                                                "removed; use 'sum'"):
            run(self.TABLE, gap=0.5, mode='brightest_sum_area')

    def test_shared_columns(self):
        m = run(self.TABLE, gap=0.5, mode='sum')
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
        m = run(detections(rows), gap=0.5, mode='sum')
        self.assertEqual(int(m['area'][0]), 65535)
        self.assertEqual(int(m['area_sum'][0]), 90000)

    def test_bad_mode(self):
        for bad in ('mean', 'Sum', 'averaged', 'sums', '', None):
            with self.assertRaisesRegex(ValueError, "must be one of"):
                run(self.TABLE, mode=bad)
            with self.assertRaisesRegex(ValueError, "must be one of"):
                deduplicate.molecule_features_option(bad)


class HandWorkedTests(unittest.TestCase):
    """Every column of the 'sum' and 'average' molecule tables of one small
    multi-plane table, worked out by hand. 'sum' is the former 'summed'
    (the last commit with that name): same values, same dtypes, same column
    order."""

    # (barcode_id, x, y, z, mean_intensity, area, min_distance); plane depth
    # 1.0 + 0.5 z um, gap 0.5 um links neighbouring planes only.
    # A: rows 0-2, planes 1-3, areas 2, 4, 8 (rows 1 and 2 tie on intensity:
    #    the lower plane, row 1, is the brightest);
    # B: row 3, one detection;
    # C: rows 4-5, both zero magnitude (centroids fall back to area weights),
    #    areas 2, 7.
    TABLE = detections([(7, 20.0, 20.0, 1, 50.0, 2, 0.40),
                        (7, 20.5, 20.0, 2, 80.0, 4, 0.20),
                        (7, 20.0, 21.0, 3, 80.0, 8, 0.30),
                        (2, 40.0, 40.0, 5, 30.0, 3, 0.50),
                        (5, 60.0, 60.0, 4, 0.0, 2, 0.60),
                        (5, 60.0, 61.0, 5, 0.0, 7, 0.20)])

    # the former merge_molecules' output for this table
    COLUMNS = ['unique_id', 'barcode_id', 'area', 'mean_intensity',
               'max_intensity', 'mean_distance', 'min_distance', 'x', 'y',
               'z', 'fov', 'cell_index', 'global_z', 'global_x', 'global_y',
               'intensity_0', 'n_detections', 'n_planes', 'z_um',
               'z_first_um', 'z_last_um', 'area_sum']
    DTYPES = {'unique_id': np.int32, 'barcode_id': np.int32,
              'area': np.uint16, 'mean_intensity': np.float32,
              'max_intensity': np.float32, 'mean_distance': np.float32,
              'min_distance': np.float32, 'x': np.float32, 'y': np.float32,
              'z': np.float32, 'fov': np.int32, 'cell_index': np.int32,
              'global_z': np.float32, 'global_x': np.float32,
              'global_y': np.float32, 'intensity_0': np.float32,
              'n_detections': np.int32, 'n_planes': np.int16,
              'z_um': np.float32, 'z_first_um': np.float32,
              'z_last_um': np.float32, 'area_sum': np.int32}

    # Columns that do not depend on molecule_features. Magnitude weights
    # (mean_intensity x area): A 100, 320, 640 (sum 1060); C has none, so
    # its centroid weights are the areas 2, 7 (sum 9).
    SHARED = {
        'unique_id': [2, 4, 5],
        'barcode_id': [7, 2, 5],
        'max_intensity': [80 * 1.5, 30 * 1.5, 0.0],
        'mean_distance': [(0.45 * 2 + 0.25 * 4 + 0.35 * 8) / 14, 0.55,
                          (0.65 * 2 + 0.25 * 7) / 9],
        'x': [(100 * 20 + 320 * 20.5 + 640 * 20) / 1060, 40.0, 60.0],
        'y': [(100 * 20 + 320 * 20 + 640 * 21) / 1060, 40.0,
              (2 * 60 + 7 * 61) / 9],
        'z': [2, 5, 4],
        'fov': [0, 0, 0],
        'cell_index': [-1, -1, -1],
        'global_z': [(100 * 1 + 320 * 2 + 640 * 3) / 1060, 5.0,
                     (2 * 4 + 7 * 5) / 9],
        'global_x': [100 + 0.14625 * (100 * 20 + 320 * 20.5 + 640 * 20) / 1060,
                     100 + 0.14625 * 40, 100 + 0.14625 * 60],
        'global_y': [-50 + 0.14625 * (100 * 20 + 320 * 20 + 640 * 21) / 1060,
                     -50 + 0.14625 * 40,
                     -50 + 0.14625 * (2 * 60 + 7 * 61) / 9],
        'intensity_0': [(15 * 2 + 24 * 4 + 24 * 8) / 14, 9.0, 0.0],
        'n_detections': [3, 1, 2],
        'n_planes': [3, 1, 2],
        'z_um': [1 + 0.5 * (100 * 1 + 320 * 2 + 640 * 3) / 1060, 3.5,
                 1 + 0.5 * (2 * 4 + 7 * 5) / 9],
        'z_first_um': [1.5, 3.5, 3.0],
        'z_last_um': [2.5, 3.5, 3.5],
        'area_sum': [14, 3, 9]}

    # sum: area-weighted means and the summed area
    SUM = {'mean_intensity': [(50 * 2 + 80 * 4 + 80 * 8) / 14, 30.0, 0.0],
           'min_distance': [(0.40 * 2 + 0.20 * 4 + 0.30 * 8) / 14, 0.50,
                            (0.60 * 2 + 0.20 * 7) / 9],
           'area': [14, 3, 9]}

    # average: plain means over the detections; area = floor(area_sum / n):
    # A 14 / 3 = 4.67 -> 4 (not rounded to 5), C 9 / 2 = 4.5 -> 4
    AVERAGE = {'mean_intensity': [(50 + 80 + 80) / 3, 30.0, 0.0],
               'min_distance': [(0.40 + 0.20 + 0.30) / 3, 0.50,
                                (0.60 + 0.20) / 2],
               'area': [4, 3, 4]}

    def check(self, m, features):
        self.assertEqual(list(m.columns), self.COLUMNS)
        self.assertEqual({c: m[c].dtype for c in m.columns},
                         {c: np.dtype(t) for c, t in self.DTYPES.items()})
        want = dict(self.SHARED, **features)
        for column in self.COLUMNS:
            if m[column].dtype.kind in 'iu':
                self.assertEqual(m[column].tolist(), want[column], column)
            else:
                np.testing.assert_allclose(
                    m[column].to_numpy(np.float64), want[column], rtol=1e-6,
                    atol=1e-6, err_msg=column)

    def test_sum_and_old_names(self):
        m = run(self.TABLE, gap=0.5, mode='sum')
        self.check(m, self.SUM)
        for old in ('summed', 'all_planes'):
            pd.testing.assert_frame_equal(run(self.TABLE, gap=0.5, mode=old),
                                          m, check_exact=True)

    def test_average(self):
        m = run(self.TABLE, gap=0.5, mode='average')
        self.check(m, self.AVERAGE)
        # the plain and the area-weighted means differ on A and C
        s = run(self.TABLE, gap=0.5, mode='sum')
        for column in ('mean_intensity', 'min_distance'):
            self.assertNotAlmostEqual(float(m[column][0]),
                                      float(s[column][0]), places=3)
        self.assertNotAlmostEqual(float(m['min_distance'][2]),
                                  float(s['min_distance'][2]), places=3)
        # every column but the three features is the same as with 'sum'
        others = [c for c in self.COLUMNS
                  if c not in ('mean_intensity', 'min_distance', 'area')]
        pd.testing.assert_frame_equal(m[others], s[others], check_exact=True)
        # the exact mean area is in the table
        np.testing.assert_allclose(m['area_sum'] / m['n_detections'],
                                   [14 / 3, 3.0, 4.5])

    def test_average_area_stays_in_database_range(self):
        rows = [(0, 5, 5, p, 10, a, 0.2)
                for p, a in enumerate((30000, 32767, 32766))]
        m = run(detections(rows), gap=0.5, mode='average')
        self.assertEqual(m['area'].dtype, np.uint16)
        self.assertEqual(int(m['area'][0]), (30000 + 32767 + 32766) // 3)
        self.assertEqual(int(m['area_sum'][0]), 30000 + 32767 + 32766)

    def test_float_area_input(self):
        table = self.TABLE.copy()
        table['area'] = table['area'].astype(np.float32)
        m = run(table, gap=0.5, mode='average')
        self.assertEqual(m['area'].dtype, np.float32)
        self.assertEqual(m['area'].tolist(), [4.0, 3.0, 4.0])


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
                ds, {'decode_task': 'Decode'}, 'Dedup')
            dedup.save()
            threshold = deduplicate.GenerateMoleculeAdaptiveThreshold(
                ds, {'decode_task': 'Dedup', 'run_after_task': 'Dedup',
                     'poll_interval': 0, 'area_bins': 20}, 'Thresh')
            threshold.save()
            filterTask = filterbarcodes.AdaptiveFilterBarcodes(
                ds, {'decode_task': 'Dedup', 'adaptive_task': 'Thresh',
                     'write_filtered_images': False}, 'Filter')
            filterTask.save()

            # defaults
            self.assertEqual(dedup.parameters['molecule_features'], 'sum')
            self.assertIs(dedup.parameters['crop_to_valid_box'], False)
            self.assertEqual(dedup.parameters['xy_radius_px'], 2.0)
            self.assertEqual(dedup.parameters['max_z_gap_um'], 0.75)
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
                    raw[fov], Z_UNIFORM, 2.0, 0.75, 'sum')
                self.assertEqual(len(molecules[fov]), len(again))
                self.assertEqual(int(molecules[fov]['n_detections'].sum()),
                                 len(raw[fov]))
                self.assertLess(len(molecules[fov]), len(raw[fov]))
                summary = dedup.get_summary(fov)
                self.assertEqual(summary['n_molecules'], len(again))
                self.assertEqual(summary['n_decoded'], len(raw[fov]))
                self.assertEqual(summary['n_detections'], len(raw[fov]))
                self.assertEqual(summary['molecule_features'], 'sum')
                self.assertIs(summary['crop_to_valid_box'], False)
                self.assertNotIn('n_outside_valid_box', summary)
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

    def test_average_through_threshold_and_filter(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = _DataSet(root)
            decodeTask = _SyntheticDecode(ds, {}, 'Decode')
            decodeTask.save()
            dedup = deduplicate.DeduplicateBarcodes(
                ds, {'decode_task': 'Decode', 'molecule_features': 'average'},
                'Dedup')
            dedup.save()
            threshold = deduplicate.GenerateMoleculeAdaptiveThreshold(
                ds, {'decode_task': 'Dedup', 'run_after_task': 'Dedup',
                     'poll_interval': 0, 'area_bins': 20}, 'Thresh')
            threshold.save()
            filterTask = filterbarcodes.AdaptiveFilterBarcodes(
                ds, {'decode_task': 'Dedup', 'adaptive_task': 'Thresh',
                     'write_filtered_images': False}, 'Filter')
            filterTask.save()
            for fov in ds.get_fovs():
                decodeTask.run(fov)
                dedup.run(fov)
            molecules = [dedup.get_barcode_database().get_barcodes(f)
                         for f in ds.get_fovs()]
            for fov in ds.get_fovs():
                raw = decodeTask.get_barcode_database().get_barcodes(fov)
                m = molecules[fov]
                self.assertEqual(dedup.get_summary(fov)['molecule_features'],
                                 'average')
                # as written to and read back from the barcode database
                self.assertEqual(m['area'].dtype, np.uint16)
                np.testing.assert_array_equal(
                    m['area'].to_numpy(np.int64),
                    m['area_sum'].to_numpy(np.int64)
                    // m['n_detections'].to_numpy(np.int64))
                self.assertTrue(np.any(m['area_sum'] % m['n_detections']))
                same, _ = deduplicate.deduplicate_barcodes(
                    raw, Z_UNIFORM, 2.0, 0.75, 'average')
                for column in m.columns:
                    np.testing.assert_array_equal(
                        m[column].to_numpy(),
                        same[column].to_numpy().astype(m[column].dtype),
                        err_msg=column)
                # only the three features differ from 'sum'
                summed, _ = deduplicate.deduplicate_barcodes(
                    raw, Z_UNIFORM, 2.0, 0.75, 'sum')
                for column in same.columns:
                    if column in ('mean_intensity', 'min_distance', 'area'):
                        continue
                    np.testing.assert_array_equal(same[column],
                                                  summed[column], column)
            threshold.run()
            allAreas = np.concatenate([m['area'].to_numpy()
                                       for m in molecules])
            areaBins = threshold.get_area_bins()
            self.assertEqual(areaBins[0], allAreas.min())
            self.assertEqual(areaBins[-1], allAreas.max() + 1)
            self.assertEqual(threshold.get_total_count_histogram().sum(),
                             sum(len(m) for m in molecules))
            for fov in ds.get_fovs():
                filterTask.run(fov)
                kept = filterTask.get_barcode_database().get_barcodes(fov)
                self.assertGreater(len(kept), 0)
                self.assertLess(len(kept), len(molecules[fov]))
                self.assertTrue(set(kept['unique_id']).issubset(
                    set(molecules[fov]['unique_id'])))

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
                     'poll_interval': 0, 'area_bins': 12,
                     'area_bin_edges': 'unit'}, 'Thresh')
            threshold.save()
            for fov in ds.get_fovs():
                decodeTask.run(fov)
                dedup.run(fov)
            threshold.run()
            np.testing.assert_array_equal(threshold.get_area_bins(),
                                          np.arange(1, 14))


class OptionTests(unittest.TestCase):
    """molecule_features names on the task, and a task saved with an old
    name still loading and regenerating."""

    def test_alias_and_removed_names(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = _DataSet(root)
            _SyntheticDecode(ds, {}, 'Decode').save()
            for given, used in (('all_planes', 'sum'),
                                ('summed', 'sum'),
                                ('sum', 'sum'),
                                ('average', 'average'),
                                ('brightest', 'brightest')):
                task = deduplicate.DeduplicateBarcodes(
                    ds, {'decode_task': 'Decode',
                         'molecule_features': given}, 'Dedup')
                self.assertEqual(task.parameters['molecule_features'], used)
            with self.assertRaisesRegex(ValueError, "'brightest_sum_area' "
                                                    "was removed"):
                deduplicate.DeduplicateBarcodes(
                    ds, {'decode_task': 'Decode',
                         'molecule_features': 'brightest_sum_area'}, 'Dedup')
            for bad in ('mean', 'averaged', 'Sum', 'SUMMED'):
                with self.assertRaisesRegex(ValueError, 'must be one of'):
                    deduplicate.DeduplicateBarcodes(
                        ds, {'decode_task': 'Decode',
                             'molecule_features': bad}, 'Dedup')
            with self.assertRaisesRegex(ValueError, 'true or false'):
                deduplicate.DeduplicateBarcodes(
                    ds, {'decode_task': 'Decode',
                         'crop_to_valid_box': 'yes'}, 'Dedup')

    def _saved_as(self, ds, name, dropCropToValidBox=False):
        """Save Dedup, then rewrite its task.json with molecule_features
        name, as an older MERlin wrote it."""
        deduplicate.DeduplicateBarcodes(
            ds, {'decode_task': 'Decode', 'max_z_gap_um': 1.25}, 'Dedup'
        ).save()
        path = os.path.join(ds.get_task_subdirectory('Dedup'), 'task.json')
        with open(path) as f:
            saved = json.load(f)
        self.assertEqual(saved['molecule_features'], 'sum')
        saved['molecule_features'] = name
        if dropCropToValidBox:
            del saved['crop_to_valid_box']
        with open(path, 'w') as f:
            json.dump(saved, f)
        return path

    def test_task_saved_as_summed_loads_as_sum(self):
        # the task.json the former code wrote: every key, 'summed'
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = _DataSet(root)
            _SyntheticDecode(ds, {}, 'Decode').save()
            path = self._saved_as(ds, 'summed')
            loaded = ds.load_analysis_task('Dedup')
            self.assertEqual(loaded.parameters['molecule_features'], 'sum')
            self.assertIs(loaded.parameters['crop_to_valid_box'], False)
            self.assertEqual(loaded.parameters['max_z_gap_um'], 1.25)
            # regenerating from an analysis json with any name of 'sum', or
            # none (the default), is accepted ...
            for parameters in ({'molecule_features': 'summed'},
                               {'molecule_features': 'all_planes'},
                               {'molecule_features': 'sum'}, {}):
                self._saved_as(ds, 'summed')
                deduplicate.DeduplicateBarcodes(
                    ds, dict(parameters, decode_task='Decode',
                             max_z_gap_um=1.25), 'Dedup').save()
                with open(path) as f:
                    self.assertEqual(json.load(f)['molecule_features'], 'sum')
            # ... another setting is not
            for other in ('average', 'brightest'):
                self._saved_as(ds, 'summed')
                with self.assertRaises(
                        analysistask.AnalysisAlreadyExistsException):
                    deduplicate.DeduplicateBarcodes(
                        ds, {'decode_task': 'Decode', 'max_z_gap_um': 1.25,
                             'molecule_features': other}, 'Dedup').save()

    def test_task_saved_with_older_name_loads_and_regenerates(self):
        # as older code wrote it: 'all_planes', no
        # crop_to_valid_box
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = _DataSet(root)
            _SyntheticDecode(ds, {}, 'Decode').save()
            self._saved_as(ds, 'all_planes', dropCropToValidBox=True)
            loaded = ds.load_analysis_task('Dedup')
            self.assertEqual(loaded.parameters['molecule_features'], 'sum')
            self.assertIs(loaded.parameters['crop_to_valid_box'], False)
            for name in ('all_planes', 'summed', 'sum'):
                deduplicate.DeduplicateBarcodes(
                    ds, {'decode_task': 'Decode', 'max_z_gap_um': 1.25,
                         'molecule_features': name}, 'Dedup').save()
            with self.assertRaises(analysistask.AnalysisAlreadyExistsException):
                deduplicate.DeduplicateBarcodes(
                    ds, {'decode_task': 'Decode', 'max_z_gap_um': 1.25,
                         'molecule_features': 'brightest'}, 'Dedup').save()

    def test_saved_average_loads(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = _DataSet(root)
            _SyntheticDecode(ds, {}, 'Decode').save()
            deduplicate.DeduplicateBarcodes(
                ds, {'decode_task': 'Decode',
                     'molecule_features': 'average'}, 'Dedup').save()
            self.assertEqual(ds.load_analysis_task('Dedup').parameters[
                'molecule_features'], 'average')
            with self.assertRaises(analysistask.AnalysisAlreadyExistsException):
                deduplicate.DeduplicateBarcodes(
                    ds, {'decode_task': 'Decode'}, 'Dedup').save()


# ------------------------------------------------------------- valid box

FRAME = 64
CENTRE = (FRAME - 1) / 2


def translation(tx, ty):
    return [[1, 0, tx], [0, 1, ty], [0, 0, 1]]


def rotation_about_centre(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, CENTRE - c * CENTRE + s * CENTRE],
            [s, c, CENTRE - s * CENTRE - c * CENTRE],
            [0, 0, 1]]


class _Warp(analysistask.AnalysisTask):
    """get_transformation(fov): one transform per matrix in MATRICES[fov]."""

    # fov 0: shifts only. With crop_width 4: left max(2, 4), right 6, top 7,
    # bottom 9 -> rows 7..55, columns 4..58.
    # fov 1: a 0.2 rad turn about the frame centre moves the corners by up to
    # 6.886 px in x and y -> 7 on every edge -> rows and columns 7..57.
    MATRICES = {0: [translation(0, 0), translation(6, -7),
                    translation(-2, 9)],
                1: [translation(0, 0), rotation_about_centre(0.2)]}
    EXPECTED = {0: (7, 55, 4, 58), 1: (7, 57, 7, 57)}

    def get_estimated_memory(self):
        return 1

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return []

    def _run_analysis(self):
        pass

    def get_transformation(self, fov, dataChannel=None):
        return [transform.AffineTransform(matrix=np.asarray(m, dtype=float))
                for m in self.MATRICES[fov]]


class _Preprocess(analysistask.AnalysisTask):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self.parameters.setdefault('warp_task', 'Warp')

    def get_estimated_memory(self):
        return 1

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return []

    def _run_analysis(self):
        pass


class _BoxDecode(_SyntheticDecode):
    """A decode task with Decode's crop parameters, writing ROWS[fov]
    (in the frame its crop_in_image_space implies)."""

    # (barcode_id, x, y, z, mean_intensity, area, min_distance); ids 8, 9 blank
    ROWS = {
        0: [(0, 30.0, 30.0, 2, 50, 4, 0.2),     # inside, linked to the next
            (0, 30.5, 30.0, 3, 60, 4, 0.2),
            (1, 4.0, 20.0, 1, 50, 4, 0.2),      # x = colStart: kept
            (1, 3.99, 20.0, 2, 50, 4, 0.2),     # x < colStart: dropped
            (2, 58.0, 40.0, 5, 50, 4, 0.2),     # x = colEnd: kept
            (2, 58.01, 40.0, 6, 50, 4, 0.2),    # x > colEnd: dropped
            (8, 20.0, 6.5, 4, 30, 3, 0.4),      # y < rowStart: dropped
            (8, 20.0, 55.0, 4, 30, 3, 0.4),     # y = rowEnd: kept
            (3, 61.0, 61.0, 0, 50, 4, 0.2),     # dropped
            (9, 50.0, 56.0, 7, 30, 3, 0.4)],    # y > rowEnd: dropped
        1: [(0, 5.5, 30.0, 1, 50, 4, 0.2),      # inside crop_width only
            (0, 32.0, 32.0, 1, 50, 4, 0.2),
            (4, 32.0, 57.0, 2, 50, 4, 0.2),     # y = rowEnd: kept
            (4, 32.0, 58.5, 3, 50, 4, 0.2)]}    # dropped
    INSIDE = {0: [True, True, True, False, True, False, False, True, False,
                  False],
              1: [False, True, True, False]}

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self.parameters.setdefault('preprocess_task', 'Pre')
        self.parameters.setdefault('crop_width', 4)
        self.parameters.setdefault('adaptive_crop', True)
        self.parameters.setdefault('crop_in_image_space', False)
        self.cropWidth = self.parameters['crop_width']   # Decode._crop_bounds

    def _run_analysis(self, fragmentIndex):
        table = detections(self.ROWS[fragmentIndex])
        table['fov'] = fragmentIndex
        self.get_barcode_database().write_barcodes(table, fov=fragmentIndex)


class _ImageSpaceBoxDecode(_BoxDecode):
    """Rows in the cropped image's frame: fov 0's image is 54 x 48 px
    (x 0..53, y 0..47), fov 1's 50 x 50 px; corners and inner points."""

    ROWS = {0: [(0, 0.0, 0.0, 0, 50, 4, 0.2),
                (0, 53.0, 47.0, 1, 50, 4, 0.2),
                (1, 20.0, 30.0, 2, 50, 4, 0.2)],
            1: [(0, 0.0, 49.0, 0, 50, 4, 0.2),
                (0, 49.0, 0.0, 1, 50, 4, 0.2),
                (8, 25.0, 25.0, 2, 30, 3, 0.4)]}

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self.parameters['crop_in_image_space'] = True


def corner_margins(matrices):
    """Independent of compute_crop_bounds: each corner moved by each
    transform, written out; (left, right, top, bottom) invalid margins."""
    left = right = top = bottom = 0.0
    for m in matrices:
        for cx, cy in ((0, 0), (FRAME - 1, 0), (0, FRAME - 1),
                       (FRAME - 1, FRAME - 1)):
            dx = m[0][0] * cx + m[0][1] * cy + m[0][2] - cx
            dy = m[1][0] * cx + m[1][1] * cy + m[1][2] - cy
            left, right = max(left, -dx), max(right, dx)
            top, bottom = max(top, -dy), max(bottom, dy)
    return tuple(int(math.ceil(v)) for v in (left, right, top, bottom))


class ValidBoxTests(unittest.TestCase):

    def _dataset(self, root, decodeClass=None, **decodeParameters):
        ds = _DataSet(root)
        _Warp(ds, {}, 'Warp').save()
        _Preprocess(ds, {}, 'Pre').save()
        decodeTask = (decodeClass or _BoxDecode)(ds, decodeParameters,
                                                 'BoxDecode')
        decodeTask.save()
        return ds, decodeTask

    def test_region_is_decodes(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds, decodeTask = self._dataset(root)
            for fov in ds.get_fovs():
                box = deduplicate.decode_valid_box(ds, decodeTask, fov)
                self.assertEqual(tuple(box), _Warp.EXPECTED[fov])
                self.assertEqual(tuple(box), tuple(decode.compute_crop_bounds(
                    ds, 'Warp', fov, 4, True)))
                self.assertEqual(tuple(box), tuple(
                    decode.Decode._crop_bounds(decodeTask, fov)))
                left, right, top, bottom = corner_margins(
                    _Warp.MATRICES[fov])
                self.assertEqual(tuple(box), (
                    max(top, 4), FRAME - max(bottom, 4),
                    max(left, 4), FRAME - max(right, 4)))

    def test_edges_inclusive(self):
        for fov, rows in _BoxDecode.ROWS.items():
            mask = deduplicate.valid_box_mask(
                detections(rows), _Warp.EXPECTED[fov], False)
            self.assertEqual(mask.tolist(), _BoxDecode.INSIDE[fov])

    def test_image_space_coordinates_are_shifted(self):
        rowStart, rowEnd, colStart, colEnd = _Warp.EXPECTED[0]
        # the cropped image's pixels: x 0 .. colEnd - colStart - 1
        rows = [(0, 0.0, 0.0, 0, 50, 4, 0.2),
                (0, colEnd - colStart - 1, rowEnd - rowStart - 1, 1,
                 50, 4, 0.2),
                (0, colEnd - colStart + 0.5, 10.0, 2, 50, 4, 0.2),
                (0, -0.5, 10.0, 3, 50, 4, 0.2)]
        mask = deduplicate.valid_box_mask(detections(rows),
                                          _Warp.EXPECTED[0], True)
        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_adaptive_crop_off_is_the_crop_width_filter(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds, decodeTask = self._dataset(root, adaptive_crop=False)
            box = deduplicate.decode_valid_box(ds, decodeTask, 0)
            self.assertEqual(tuple(box), (4, 60, 4, 60))
            rng = np.random.default_rng(5)
            table = detections([(0, x, y, 0, 50, 4, 0.2) for x, y in
                                rng.uniform(0, FRAME, (400, 2))]
                               + [(0, 4.0, 60.0, 0, 50, 4, 0.2),
                                  (0, 60.0, 4.0, 0, 50, 4, 0.2)])
            # Decode's barcode-space crop_width filter
            decodeFilter = (table['x'].between(4, FRAME - 4)
                            & table['y'].between(4, FRAME - 4)).to_numpy()
            np.testing.assert_array_equal(
                deduplicate.valid_box_mask(table, box, False), decodeFilter)

    def test_needs_decode_crop_parameters(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = _DataSet(root)
            _SyntheticDecode(ds, {}, 'Decode').save()
            with self.assertRaisesRegex(ValueError,
                                        'Decode has no preprocess_task'):
                deduplicate.DeduplicateBarcodes(
                    ds, {'decode_task': 'Decode',
                         'crop_to_valid_box': True}, 'Dedup')

    def test_task_drops_outside_before_linking(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds, decodeTask = self._dataset(root)
            cropped = deduplicate.DeduplicateBarcodes(
                ds, {'decode_task': 'BoxDecode', 'crop_to_valid_box': True,
                     'xy_radius_px': 2.0, 'max_z_gap_um': 0.75}, 'Cropped')
            cropped.save()
            whole = deduplicate.DeduplicateBarcodes(
                ds, {'decode_task': 'BoxDecode', 'xy_radius_px': 2.0,
                     'max_z_gap_um': 0.75}, 'Whole')
            whole.save()
            for fov in ds.get_fovs():
                decodeTask.run(fov)
                cropped.run(fov)
                whole.run(fov)

            raw = decodeTask.get_barcode_database().get_barcodes(0)
            inside = np.array(_BoxDecode.INSIDE[0])
            got = cropped.get_barcode_database().get_barcodes(0)
            want, _ = deduplicate.deduplicate_barcodes(
                raw[inside], Z_UNIFORM, 2.0, 0.75, 'sum')
            self.assertEqual(len(got), len(want))
            np.testing.assert_array_equal(got['unique_id'], want['unique_id'])
            # the pair at x 4.0 / 3.99 and the pair at x 58.0 / 58.01 would
            # link; cropped first, each keeps one detection
            self.assertEqual(sorted(got['n_detections'].tolist()),
                             [1, 1, 1, 2])
            self.assertEqual(
                len(whole.get_barcode_database().get_barcodes(0)), 7)

            summary = cropped.get_summary(0)
            self.assertIs(summary['crop_to_valid_box'], True)
            self.assertEqual(summary['valid_box'], [7, 55, 4, 58])
            self.assertEqual(summary['n_decoded'], 10)
            self.assertEqual(summary['n_detections'], 5)
            self.assertEqual(summary['n_outside_valid_box'], 5)
            self.assertEqual(summary['coding_outside_valid_box'], 3)
            self.assertEqual(summary['blank_outside_valid_box'], 2)
            summary = cropped.get_summary(1)
            self.assertEqual(summary['valid_box'], [7, 57, 7, 57])
            self.assertEqual((summary['n_decoded'], summary['n_detections'],
                              summary['n_outside_valid_box']), (4, 2, 2))
            summary = whole.get_summary(0)
            self.assertEqual((summary['n_decoded'], summary['n_detections']),
                             (10, 10))
            self.assertNotIn('n_outside_valid_box', summary)

    def test_image_space_decode_drops_nothing(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds, decodeTask = self._dataset(root, _ImageSpaceBoxDecode)
            self.assertIs(decodeTask.parameters['crop_in_image_space'], True)
            cropped = deduplicate.DeduplicateBarcodes(
                ds, {'decode_task': 'BoxDecode', 'crop_to_valid_box': True},
                'Cropped')
            cropped.save()
            for fov in ds.get_fovs():
                decodeTask.run(fov)
                cropped.run(fov)
                summary = cropped.get_summary(fov)
                self.assertEqual(summary['valid_box'],
                                 list(_Warp.EXPECTED[fov]))
                self.assertEqual(
                    (summary['n_decoded'], summary['n_detections'],
                     summary['n_outside_valid_box']), (3, 3, 0))


class AreaEdgeTests(unittest.TestCase):

    def test_equal_count_edges(self):
        # 1000 molecules: area 2 x 500, 3 x 300, then 4 .. 203 once each.
        # 2 and 3 take a bin each; the 8 left split the other 200 by 25.
        areas = np.r_[np.full(500, 2), np.full(300, 3), np.arange(4, 204)]
        edges = deduplicate.area_equal_count_edges(
            np.random.default_rng(0).permutation(areas), 10)
        np.testing.assert_array_equal(
            edges, [2, 3, 4, 29, 54, 79, 104, 129, 154, 179, 204])
        counts = np.histogram(areas, edges)[0]
        self.assertEqual(counts.tolist(), [500, 300] + [25] * 8)

    def test_fewer_distinct_areas_than_bins(self):
        np.testing.assert_array_equal(
            deduplicate.area_equal_count_edges(np.array([5, 2, 2, 3, 9]), 10),
            [2, 3, 5, 9, 10])

    def test_one_bin(self):
        np.testing.assert_array_equal(
            deduplicate.area_equal_count_edges(np.array([4, 2, 7]), 1),
            [2, 8])

    def test_empty_falls_back_to_unit_edges(self):
        np.testing.assert_array_equal(
            deduplicate.area_equal_count_edges(np.array([]), 5),
            np.arange(1, 7))


if __name__ == '__main__':
    unittest.main()
