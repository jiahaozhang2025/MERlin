"""Location strata: the label-volume contract, the lookup, the per-stratum
histograms of GenerateAdaptiveThreshold (location_strata_path) and the
location-stratified AdaptiveFilterBarcodes (location_stratified).

The lookup is compared with a reference written here from the partition's
formulas (col = rint(x) - x0, row = rint(y) - y0, plane = rint(z) - z0, Python
round = half to even). The pipeline tests run a synthetic decode whose blank
rate depends on the location, a threshold task without strata and one with
them, and pooled and stratified filters, and check: the unset options change
nothing (histograms, bins, tables, task.json); pooled == sum of the strata
histograms; each stratum histogram against the reference lookup; per-stratum
thresholds against the pooled solver on that stratum; the in-sample misID per
stratum; the pooled fallback; the equivalences with the pooled filter; the
refusals; the output column, the per-fov summaries and the report.

Run with `python -m unittest tests/test_location_strata.py`.
"""
import contextlib
import importlib
import io
import json
import os
import re
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from merlin.analysis import decode
from merlin.analysis import deduplicate
from merlin.analysis import filterbarcodes
from merlin.core import analysistask
from merlin.core import dataset
from merlin.util import locationstrata

Z_UNIFORM = [1.0 + 0.5 * k for k in range(12)]
FRAME = 64
NAMES = ['not_segmented', 'outside_cells', 'nucleus', 'cytoplasm_membrane']
NEW_KEYS = ('location_strata_path', 'location_stratified',
            'location_min_blank_molecules')
BINS = {'intensity_bins': 8, 'distance_bins': 6, 'area_bins': 5}


def reference_lookup(x, y, z, codes, origin, outsideCode):
    """Row by row, from the partition's formulas."""
    out = []
    nz, ny, nx = codes.shape
    for xv, yv, zv in zip(x, y, z):
        xv, yv, zv = float(xv), float(yv), float(zv)
        if not all(np.isfinite(v) for v in (xv, yv, zv)):
            out.append(outsideCode)
            continue
        col = int(round(xv)) - int(origin[2])
        row = int(round(yv)) - int(origin[1])
        plane = int(round(zv)) - int(origin[0])
        if 0 <= plane < nz and 0 <= row < ny and 0 <= col < nx:
            out.append(int(codes[plane, row, col]))
        else:
            out.append(outsideCode)
    return np.array(out, dtype=np.uint8)


# ------------------------------------------------------------- the harness

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
        return [0, 1, 2]

    def get_z_positions(self):
        return Z_UNIFORM

    def get_image_dimensions(self):
        return (FRAME, FRAME)


# The synthetic strata of every fov: origin (z0, y0, x0) = (1, 6, 2), 10 planes
# x 50 rows x 56 columns, so planes 0 and 11, rows < 6 or >= 56 and columns < 2
# or >= 58 are outside (not_segmented). Inside: nucleus at columns x < 24 on
# planes 3..8, cytoplasm_membrane at 24 <= x < 40, the rest outside_cells.
ORIGIN = (1, 6, 2)
SHAPE = (10, 50, 56)


def synthetic_codes():
    codes = np.ones(SHAPE, dtype=np.uint8)
    planes = np.arange(SHAPE[0])[:, None, None] + ORIGIN[0]
    cols = np.arange(SHAPE[2])[None, None, :] + ORIGIN[2]
    nucleus = (cols < 24) & (planes >= 3) & (planes <= 8)
    codes[np.broadcast_to(nucleus, SHAPE)] = 2
    codes[np.broadcast_to((cols >= 24) & (cols < 40), SHAPE)] = 3
    return codes


_CODES = synthetic_codes()


def true_stratum(x, y, z):
    return int(reference_lookup([x], [y], [z], _CODES, ORIGIN, 0)[0])


def write_strata(directory, names=NAMES, codesFor=None, origin=ORIGIN,
                 outsideCode=0, fovs=(0, 1, 2), source=None):
    os.makedirs(directory, exist_ok=True)
    for fov in fovs:
        codes = synthetic_codes() if codesFor is None else codesFor(fov)
        locationstrata.write_volume(directory, fov, codes, origin,
                                    strataCount=len(names))
    locationstrata.write_metadata(directory, names, outside_code=outsideCode,
                                  source=source or {'test': True})


def table(rows):
    """Decode-like table from (barcode_id, x, y, z, mean_intensity, area,
    min_distance) tuples."""
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
    return out


class _LocDecode(decode.BarcodeSavingParallelAnalysisTask):
    """A fixed random table per fov whose blank rate depends on the location:
    nucleus 30%, cytoplasm_membrane 8%, outside_cells 12%, not_segmented 15%;
    blanks dimmer and farther on average."""

    outputGroup = 'Decode'
    BLANK_RATE = {0: 0.15, 1: 0.12, 2: 0.30, 3: 0.08}

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
        rng = np.random.default_rng(700 + fragmentIndex)
        rows = []
        for _ in range(3000):
            x, y = rng.uniform(0, FRAME - 1, 2)
            z = int(rng.integers(len(Z_UNIFORM)))
            blank = rng.random() < self.BLANK_RATE[true_stratum(x, y, z)]
            bid = int(rng.integers(8, 10)) if blank else int(rng.integers(8))
            rows.append((bid, x, y, z,
                         float(rng.lognormal(3.0 if blank else 3.5, 0.5)),
                         int(rng.integers(1, 7)),
                         float(rng.uniform(0.15 if blank else 0.0, 0.65))))
        out = table(rows)
        out['fov'] = fragmentIndex
        self.get_barcode_database().write_barcodes(out, fov=fragmentIndex)


def threshold_task(ds, name, decodeTask='Decode', strata=None, cls=None,
                   **extra):
    parameters = dict(BINS, decode_task=decodeTask, run_after_task=decodeTask,
                      poll_interval=0, bin_sample_fovs=3, **extra)
    if strata is not None:
        parameters['location_strata_path'] = strata
    task = (cls or filterbarcodes.GenerateAdaptiveThreshold)(
        ds, parameters, name)
    task.save()
    return task


def filter_task(ds, name, adaptive, decodeTask='Decode', **extra):
    parameters = dict(decode_task=decodeTask, adaptive_task=adaptive,
                      write_filtered_images=False, **extra)
    task = filterbarcodes.AdaptiveFilterBarcodes(ds, parameters, name)
    task.save()
    return task


def misid(blank, coding):
    return (blank / 2) / (coding / 8)


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ lookup

class LookupTests(unittest.TestCase):

    def setUp(self):
        self.codes = (np.arange(4 * 5 * 6).reshape(4, 5, 6) % 250 + 1
                      ).astype(np.uint8)

    def check(self, x, y, z, origin, outside=0):
        got = locationstrata.lookup(x, y, z, self.codes, origin, outside)
        want = reference_lookup(x, y, z, self.codes, origin, outside)
        self.assertEqual(got.dtype, np.uint8)
        np.testing.assert_array_equal(got, want)
        return got

    def test_round_half_to_even(self):
        codes = np.zeros((1, 1, 14), dtype=np.uint8)
        codes[0, 0, :] = np.arange(14) + 1
        got = locationstrata.lookup([10.5, 11.5, -0.5, 0.5, 1.5, 12.49],
                                    [0.0] * 6, [0.0] * 6, codes, (0, 0, 0), 99)
        # 10.5 -> 10, 11.5 -> 12, -0.5 -> -0 = column 0 (inside), 0.5 -> 0,
        # 1.5 -> 2, 12.49 -> 12
        self.assertEqual(got.tolist(), [11, 13, 1, 1, 3, 13])
        got = locationstrata.lookup([3.0], [0.0], [-0.5], codes, (0, 0, 0), 99)
        self.assertEqual(got.tolist(), [4])          # z -0.5 -> plane 0

    def test_nonzero_origin_and_faces(self):
        origin = (3, -2, 5)              # planes 3..6, rows -2..2, cols 5..10
        nz, ny, nx = self.codes.shape
        centre = [5 + 2, -2 + 2, 3 + 1]
        x, y, z = [], [], []
        for axis, (low, n) in enumerate(((5, nx), (-2, ny), (3, nz))):
            for value in (low - 0.5001, low - 0.5, low - 0.4999, low,
                          low + n - 1, low + n - 1 + 0.4999, low + n - 0.5,
                          low + n - 0.4999, low + n):
                point = list(centre)
                point[axis] = value
                x.append(point[0])
                y.append(point[1])
                z.append(point[2])
        got = self.check(x, y, z, origin, outside=0)
        # per axis 4 values are inside for sure (low - 0.4999, low,
        # low + n - 1, low + n - 1 + 0.4999); low - 0.5 and low + n - 0.5
        # round to even: x 4.5 -> 4 out, 10.5 -> 10 in; y -2.5 -> -2 in,
        # 2.5 -> 2 in; z 2.5 -> 2 out, 6.5 -> 6 in
        inside = got > 0
        self.assertEqual(int(inside.sum()), 5 + 6 + 5)
        rng = np.random.default_rng(3)
        x = np.round(rng.uniform(2, 13, 500) * 2) / 2
        y = np.round(rng.uniform(-5, 5, 500) * 2) / 2
        z = np.round(rng.uniform(1, 9, 500) * 2) / 2
        self.check(x, y, z, origin)
        self.check(rng.uniform(2, 13, 500), rng.uniform(-5, 5, 500),
                   rng.uniform(1, 9, 500), origin, outside=7)

    def test_non_finite_and_float32(self):
        x = np.array([7.0, np.nan, np.inf, -np.inf, 7.0, 7.0], np.float64)
        y = np.array([0.0, 0.0, 0.0, 0.0, np.nan, 0.0])
        z = np.array([4.0, 4.0, 4.0, 4.0, 4.0, np.inf])
        got = self.check(x, y, z, (3, -2, 5), outside=9)
        self.assertEqual(got.tolist()[1:], [9] * 5)
        x32 = np.array([6.5, 7.5, 8.49999, 10.50001], dtype=np.float32)
        self.check(x32, np.zeros(4, np.float32), np.full(4, 4, np.float32),
                   (3, -2, 5))

    def test_empty_volume_and_empty_input(self):
        empty = np.zeros((0, 0, 0), dtype=np.uint8)
        got = locationstrata.lookup([1.0, 2.0], [1.0, 2.0], [0.0, 0.0],
                                    empty, (0, 0, 0), 3)
        self.assertEqual(got.tolist(), [3, 3])
        self.assertEqual(len(locationstrata.lookup([], [], [], self.codes,
                                                   (0, 0, 0), 0)), 0)


class ContractTests(unittest.TestCase):

    def test_round_trip_and_digest(self):
        with tempfile.TemporaryDirectory() as root:
            codes = synthetic_codes()
            locationstrata.write_metadata(root, NAMES, source={'a': 1})
            locationstrata.write_volume(root, 4, codes, ORIGIN)
            got, origin = locationstrata.read_volume(root, 4, len(NAMES))
            np.testing.assert_array_equal(got, codes)
            self.assertEqual(origin.tolist(), list(ORIGIN))
            self.assertEqual(origin.dtype, np.int64)
            metadata = locationstrata.read_metadata(root)
            self.assertEqual(metadata['strata'], NAMES)
            self.assertEqual(metadata['outside_code'], 0)
            digest = locationstrata.metadata_digest(root)
            locationstrata.write_metadata(root, NAMES, source={'a': 2})
            self.assertNotEqual(digest, locationstrata.metadata_digest(root))
            self.assertEqual(sorted(os.listdir(root)),
                             ['location_strata.json', 'strata_fov004.npz'])

    def test_metadata_refusals(self):
        with tempfile.TemporaryDirectory() as root:
            for names, outside, message in (
                    (['a', 'a'], 0, 'unique'),
                    (['a', ''], 0, 'non-empty'),
                    ([], 0, 'list of 1 to 255'),
                    (['s%d' % k for k in range(256)], 0, 'list of 1 to 255'),
                    (['a', 'b'], 2, 'outside_code'),
                    (['a', 'b'], -1, 'outside_code'),
                    (['a', 'b'], True, 'outside_code'),
                    (['a', 'b'], 1.0, 'outside_code')):
                with self.assertRaisesRegex(ValueError, message):
                    locationstrata.write_metadata(root, names, outside)
            locationstrata.write_metadata(root, NAMES)
            path = os.path.join(root, locationstrata.METADATA_FILE)
            good = read_json(path)
            for key, value, message in (('format', 'other', 'format'),
                                        ('version', 2, 'version'),
                                        ('version', True, 'version')):
                bad = dict(good, **{key: value})
                with open(path, 'w') as f:
                    json.dump(bad, f)
                with self.assertRaisesRegex(ValueError, message):
                    locationstrata.read_metadata(root)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(FileNotFoundError, 'sidecar'):
                locationstrata.read_metadata(root)

    def test_volume_refusals(self):
        with tempfile.TemporaryDirectory() as root:
            codes = synthetic_codes()
            with self.assertRaisesRegex(ValueError, 'only 3 strata'):
                locationstrata.write_volume(root, 0, codes, ORIGIN,
                                            strataCount=3)
            with self.assertRaisesRegex(ValueError, 'uint8'):
                locationstrata.write_volume(root, 0, codes.astype(np.int16),
                                            ORIGIN)
            with self.assertRaisesRegex(ValueError, '3-D'):
                locationstrata.write_volume(root, 0, codes[0], ORIGIN)
            with self.assertRaisesRegex(ValueError, 'origin'):
                locationstrata.write_volume(root, 0, codes, (1.0, 2.0, 3.0))
            with self.assertRaisesRegex(ValueError, 'origin'):
                locationstrata.write_volume(root, 0, codes, (1, 2))
            # the sidecar's strata count is used when it exists
            locationstrata.write_metadata(root, ['a', 'b', 'c'])
            with self.assertRaisesRegex(ValueError, 'only 3 strata'):
                locationstrata.write_volume(root, 0, codes, ORIGIN)
            # reading checks the codes against the count it is given
            locationstrata.write_volume(root, 0, codes, ORIGIN,
                                        strataCount=4)
            with self.assertRaisesRegex(ValueError, 'only 3 strata'):
                locationstrata.read_volume(root, 0, 3)
            missing = os.path.join(root, 'strata_fov007.npz')
            with self.assertRaisesRegex(FileNotFoundError,
                                        'fov 7 not found: '
                                        + re.escape(missing)):
                locationstrata.read_volume(root, 7, 4)
            # a file under another fov's name
            os.rename(os.path.join(root, 'strata_fov000.npz'), missing)
            with self.assertRaisesRegex(ValueError, 'holds fov'):
                locationstrata.read_volume(root, 7, 4)
            # an empty volume is valid
            locationstrata.write_volume(root, 1, np.zeros((0, 0, 0), np.uint8),
                                        (0, 0, 0), strataCount=4)
            got, _ = locationstrata.read_volume(root, 1, 4)
            self.assertEqual(got.shape, (0, 0, 0))


class DeduplicatedMoleculeTests(unittest.TestCase):

    def test_brightest_plane_not_z_um(self):
        # one molecule on planes 3, 4, 5; brightest on plane 5; its
        # magnitude-weighted depth is nearest plane 4
        rows = [(1, 20.2, 30.4, 3, 50.0, 4, 0.2),
                (1, 20.3, 30.4, 4, 60.0, 4, 0.2),
                (1, 20.1, 30.5, 5, 70.0, 4, 0.2)]
        molecules, _ = deduplicate.deduplicate_barcodes(
            table(rows), Z_UNIFORM, 2.0, 0.75, 'summed')
        self.assertEqual(len(molecules), 1)
        self.assertEqual(float(molecules['z'].iloc[0]), 5.0)
        weightedPlane = (float(molecules['z_um'].iloc[0]) - 1.0) / 0.5
        self.assertEqual(int(round(weightedPlane)), 4)
        codes = np.zeros((8, 64, 64), dtype=np.uint8)
        codes[3], codes[4], codes[5] = 1, 2, 3
        got = locationstrata.lookup(molecules['x'], molecules['y'],
                                    molecules['z'], codes, (0, 0, 0), 0)
        self.assertEqual(got.tolist(), [3])


# ---------------------------------------------------------- the pipeline

class _Pipeline(unittest.TestCase):
    """One synthetic dataset, decoded once, shared by the tests below."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = cls._tmp.name
        cls.strata = os.path.join(root, 'strata')
        # a fifth stratum that no voxel has: always empty
        write_strata(cls.strata, NAMES + ['unused'])
        cls.one = os.path.join(root, 'strata_one')
        write_strata(cls.one, ['empty', 'everything'],
                     codesFor=lambda fov: np.ones((1, 1, 1), np.uint8),
                     outsideCode=1)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.ds = _DataSet(os.path.join(root, 'analysis'))
            cls.decode = _LocDecode(cls.ds, {}, 'Decode')
            cls.decode.save()
            for fov in cls.ds.get_fovs():
                cls.decode.run(fov)
            cls.raw = {f: cls.decode.get_barcode_database().get_barcodes(f)
                       for f in cls.ds.get_fovs()}
            cls.pooled = threshold_task(cls.ds, 'ThreshP')
            cls.pooled.run()
            cls.stratified = threshold_task(cls.ds, 'ThreshS',
                                            strata=cls.strata)
            cls.stratified.run()
            cls.oneThreshold = threshold_task(cls.ds, 'ThreshOne',
                                              strata=cls.one)
            cls.oneThreshold.run()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def codes_of(self, fov, frame=None):
        frame = self.raw[fov] if frame is None else frame
        return reference_lookup(frame['x'], frame['y'], frame['z'],
                                synthetic_codes(), ORIGIN, 0)

    def run_filter(self, name, adaptive, **extra):
        with contextlib.redirect_stdout(io.StringIO()):
            task = filter_task(self.ds, name, adaptive, **extra)
            for fov in self.ds.get_fovs():
                task.run(fov)
        return task, {f: task.get_barcode_database().get_barcodes(f)
                      for f in self.ds.get_fovs()}


class HistogramTests(_Pipeline):

    def test_unset_options_change_nothing(self):
        for result in ('blank_counts', 'coding_counts', 'intensity_bins',
                       'distance_bins', 'area_bins', 'complete_fragments'):
            a = self.ds.load_numpy_analysis_result(result, 'ThreshP')
            b = self.ds.load_numpy_analysis_result(result, 'ThreshS')
            np.testing.assert_array_equal(a, b, err_msg=result)
            self.assertEqual(a.dtype, b.dtype)
        saved = read_json(os.path.join(
            self.ds.get_task_subdirectory('ThreshP'), 'task.json'))
        for key in NEW_KEYS:
            self.assertNotIn(key, saved)
        self.assertFalse(self.pooled.has_location_strata())
        files = sorted(os.listdir(self.ds.get_analysis_subdirectory(
            'ThreshP')))
        self.assertNotIn('blank_counts_strata.npy', files)
        self.assertNotIn('location_strata.json', files)
        # regenerating either task from the same parameters is accepted
        with contextlib.redirect_stdout(io.StringIO()):
            threshold_task(self.ds, 'ThreshP')
            threshold_task(self.ds, 'ThreshS', strata=self.strata)
        with self.assertRaises(analysistask.AnalysisAlreadyExistsException):
            threshold_task(self.ds, 'ThreshP', strata=self.strata)

    def test_pooled_is_sum_of_strata(self):
        t = self.stratified
        blank = t.get_blank_count_histogram_strata()
        coding = t.get_coding_count_histogram_strata()
        self.assertEqual(blank.shape, (5,) + t.get_blank_count_histogram().shape)
        np.testing.assert_array_equal(blank.sum(axis=0),
                                      t.get_blank_count_histogram())
        np.testing.assert_array_equal(coding.sum(axis=0),
                                      t.get_coding_count_histogram())
        self.assertEqual(blank[4].sum() + coding[4].sum(), 0)

    def test_each_stratum_against_reference(self):
        t = self.stratified
        bins = (t.get_intensity_bins(), t.get_distance_bins(),
                t.get_area_bins())
        blank = np.zeros_like(t.get_blank_count_histogram_strata())
        coding = np.zeros_like(blank)
        fit = t.get_location_strata_fit()
        for fov, frame in self.raw.items():
            codes = self.codes_of(fov)
            isBlank = frame['barcode_id'].isin([8, 9]).to_numpy()
            for s in range(5):
                blank[s] += t._extract_counts(frame[isBlank & (codes == s)],
                                              *bins)
                coding[s] += t._extract_counts(frame[~isBlank & (codes == s)],
                                               *bins)
                self.assertEqual(fit['per_fov'][str(fov)]['blank'][s],
                                 int((isBlank & (codes == s)).sum()))
                self.assertEqual(fit['per_fov'][str(fov)]['coding'][s],
                                 int((~isBlank & (codes == s)).sum()))
        np.testing.assert_array_equal(blank,
                                      t.get_blank_count_histogram_strata())
        np.testing.assert_array_equal(coding,
                                      t.get_coding_count_histogram_strata())
        self.assertEqual(fit['strata'], NAMES + ['unused'])
        self.assertEqual(fit['fovs'], [0, 1, 2])
        self.assertEqual(fit['path'], self.strata)
        self.assertEqual(fit['metadata_digest'],
                         locationstrata.metadata_digest(self.strata))
        self.assertEqual(fit['blank_molecules'],
                         [int(v) for v in blank.reshape(5, -1).sum(1)])
        # every stratum but the empty one holds molecules, blank rates differ
        rates = [fit['blank_molecules'][s] / (fit['blank_molecules'][s]
                                              + fit['coding_molecules'][s])
                 for s in range(4)]
        self.assertGreater(rates[2], 2 * rates[3])

    def test_normalized_blank_fraction_is_the_methods(self):
        t = self.stratified
        got = filterbarcodes.normalized_blank_fraction(
            t.get_blank_count_histogram(), t.get_coding_count_histogram(), 2, 8)
        want = t.get_blank_fraction_histogram()
        self.assertTrue((t.get_coding_count_histogram() == 0).any())
        np.testing.assert_array_equal(got, want)

    def test_bin_indexes_reproduce_extract(self):
        t = self.stratified
        threshold = t.calculate_threshold_for_misidentification_rate(0.05)
        keep = (t.get_blank_fraction_histogram() < threshold).ravel()
        for fov, frame in self.raw.items():
            want = t.extract_barcodes_with_threshold(threshold, frame)
            got = frame[keep[t.barcode_bin_indexes(frame)]]
            self.assertTrue(got.equals(want))

    def test_molecule_threshold_inherits_strata(self):
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()):
            ds = _DataSet(root)
            decodeTask = _LocDecode(ds, {}, 'Decode')
            decodeTask.save()
            dedup = deduplicate.DeduplicateBarcodes(
                ds, {'decode_task': 'Decode'}, 'Dedup')
            dedup.save()
            for fov in ds.get_fovs():
                decodeTask.run(fov)
                dedup.run(fov)
            t = threshold_task(
                ds, 'MolThresh', decodeTask='Dedup', strata=self.strata,
                cls=deduplicate.GenerateMoleculeAdaptiveThreshold)
            t.run()
            np.testing.assert_array_equal(
                t.get_blank_count_histogram_strata().sum(0),
                t.get_blank_count_histogram())
            molecules = sum(len(dedup.get_barcode_database().get_barcodes(f))
                            for f in ds.get_fovs())
            self.assertEqual(t.get_total_count_histogram().sum(), molecules)
            f = filter_task(ds, 'MolFilterS', 'MolThresh', decodeTask='Dedup',
                            location_stratified=True,
                            location_min_blank_molecules=0)
            for fov in ds.get_fovs():
                f.run(fov)
                kept = f.get_barcode_database().get_barcodes(fov)
                self.assertIn('location_stratum', kept.columns)
                self.assertIn('n_planes', kept.columns)


class ThresholdTests(_Pipeline):

    def test_own_thresholds_are_the_solver_on_the_stratum(self):
        t = self.stratified
        blank = t.get_blank_count_histogram_strata()
        coding = t.get_coding_count_histogram_strata()
        for target in (0.05, 0.15):
            report, keep = t.stratum_rules(target, 0)
            pooled = t.calculate_threshold_for_misidentification_rate(target)
            self.assertEqual(report['pooled']['threshold'], pooled)
            for rule in report['strata']:
                s = rule['code']
                if s == 4:
                    self.assertEqual((rule['source'], rule['reason']),
                                     ('pooled', 'no_coding_molecules'))
                    continue
                self.assertEqual(rule['source'], 'own', rule)
                bf = filterbarcodes.normalized_blank_fraction(
                    blank[s], coding[s], 2, 8)
                want = filterbarcodes._threshold_from_cumulative_bins(
                    blank[s], coding[s], bf, target, 2, 8)
                self.assertEqual(rule['threshold'], want)
                np.testing.assert_array_equal(keep[s], (bf < want).ravel())
                keptBlank = blank[s].ravel()[keep[s]].sum()
                keptCoding = coding[s].ravel()[keep[s]].sum()
                self.assertAlmostEqual(rule['fit_misid'],
                                       misid(keptBlank, keptCoding))
                self.assertLessEqual(rule['fit_misid'], target)
            self.assertEqual(
                calculate(t, target, 0), report)

    def test_fallback_below_minimum(self):
        t = self.stratified
        fit = t.get_location_strata_fit()
        sizes = fit['blank_molecules'][:4]
        # a minimum between the smallest and the largest stratum
        minimum = sorted(sizes)[1] + 1
        report, keep = t.stratum_rules(0.05, minimum)
        pooledKeep = (t.get_blank_fraction_histogram()
                      < report['pooled']['threshold']).ravel()
        for rule in report['strata'][:4]:
            s = rule['code']
            if sizes[s] < minimum:
                self.assertEqual((rule['source'], rule['reason']),
                                 ('pooled', 'blank_molecules_below_min'))
                self.assertEqual(rule['threshold'],
                                 report['pooled']['threshold'])
                np.testing.assert_array_equal(keep[s], pooledKeep)
            else:
                self.assertEqual(rule['source'], 'own')
        self.assertEqual(sum(r['source'] == 'pooled'
                             for r in report['strata'][:4]), 2)

    def test_newton_is_refused(self):
        t = self.ds.load_analysis_task('ThreshS')
        t.parameters['threshold_solver_method'] = 'newton'
        with self.assertRaisesRegex(ValueError, 'cumulative_bins'):
            t.stratum_rules(0.05, 0)


def calculate(task, target, minimum):
    return task.calculate_stratum_thresholds(target, minimum)


class FilterTests(_Pipeline):

    def test_output_column_order_and_summaries(self):
        task, kept = self.run_filter('FilterS', 'ThreshS',
                                     location_stratified=True,
                                     location_min_blank_molecules=0)
        self.assertEqual(task.parameters['location_min_blank_molecules'], 0)
        t = self.stratified
        report, keepTable = t.stratum_rules(0.05, 0)
        totals = {s: [0, 0] for s in range(5)}
        for fov, frame in self.raw.items():
            codes = self.codes_of(fov)
            mask = keepTable[codes.astype(int), t.barcode_bin_indexes(frame)]
            out = kept[fov]
            self.assertEqual(out['location_stratum'].dtype, np.uint8)
            np.testing.assert_array_equal(out['location_stratum'],
                                          codes[mask])
            np.testing.assert_array_equal(out['location_stratum'],
                                          self.codes_of(fov, out))
            np.testing.assert_array_equal(out['unique_id'],
                                          frame['unique_id'][mask])
            self.assertTrue(out.drop(columns='location_stratum').equals(
                frame[mask]))
            summary = task.get_location_filter_summary(fov)
            self.assertEqual(summary['rows_in'], len(frame))
            self.assertEqual(summary['rows_selected'], len(out))
            self.assertEqual(summary['rows_out'], len(out))
            self.assertEqual(sum(s['kept_total'] for s in summary['strata']),
                             len(out))
            self.assertEqual(sum(s['in_total'] for s in summary['strata']),
                             len(frame))
            self.assertEqual(summary['thresholds'], report)
            isBlank = out['barcode_id'].isin([8, 9]).to_numpy()
            for s in range(5):
                inS = out['location_stratum'].to_numpy() == s
                self.assertEqual(summary['strata'][s]['kept_blank'],
                                 int((inS & isBlank).sum()))
                totals[s][0] += int((inS & isBlank).sum())
                totals[s][1] += int((inS & ~isBlank).sum())
        # filtering the fit fovs reproduces the fit: misID <= target per
        # own stratum
        for rule in report['strata'][:4]:
            s = rule['code']
            self.assertEqual(totals[s], [rule['fit_kept_blank'],
                                         rule['fit_kept_coding']])
            self.assertLessEqual(misid(*totals[s]), 0.05)
        frame = filterbarcodes.location_filter_report(self.ds, 'FilterS')
        for s in range(4):
            row = frame[frame['code'] == s].iloc[0]
            self.assertEqual((row['kept_blank'], row['kept_coding']),
                             tuple(totals[s]))
            self.assertAlmostEqual(row['misid'], misid(*totals[s]))
            self.assertAlmostEqual(row['fit_misid'],
                                   report['strata'][s]['fit_misid'])
        allRow = frame[frame['name'] == 'all'].iloc[0]
        self.assertEqual(allRow['kept_coding'],
                         sum(len(k) for k in kept.values())
                         - sum(t[0] for t in totals.values()))
        self.assertAlmostEqual(allRow['misid'], misid(
            sum(t[0] for t in totals.values()),
            sum(t[1] for t in totals.values())))
        self.assertEqual(int(frame['n_fovs'].iloc[0]), 3)
        # stratification moves the nucleus towards the target
        _, pooledKept = self.run_filter('FilterPForReport', 'ThreshS')
        nucleusPooled = [0, 0]
        for fov, out in pooledKept.items():
            codes = self.codes_of(fov, out)
            isBlank = out['barcode_id'].isin([8, 9]).to_numpy()
            nucleusPooled[0] += int(((codes == 2) & isBlank).sum())
            nucleusPooled[1] += int(((codes == 2) & ~isBlank).sum())
        self.assertGreater(misid(*nucleusPooled), 0.05)

    def test_fifteen_percent_target(self):
        task, kept = self.run_filter('FilterS15', 'ThreshS',
                                     misidentification_rate=0.15,
                                     location_stratified=True)
        self.assertEqual(task.parameters['location_min_blank_molecules'],
                         50000)
        frame = filterbarcodes.location_filter_report(self.ds, 'FilterS15')
        # 50000 is above every stratum here: all pooled = the pooled filter
        self.assertTrue((frame['source'].iloc[:5] == 'pooled').all())
        self.assertTrue((frame['reason'].iloc[:4]
                         == 'blank_molecules_below_min').all())
        _, pooled = self.run_filter('FilterP15', 'ThreshP',
                                    misidentification_rate=0.15)
        for fov in self.ds.get_fovs():
            self.assertTrue(kept[fov].drop(columns='location_stratum').equals(
                pooled[fov]))
        with contextlib.redirect_stdout(io.StringIO()):
            in_memory = filterbarcodes.AdaptiveFilterBarcodes(
                self.ds, {'decode_task': 'Decode', 'adaptive_task': 'ThreshS',
                          'misidentification_rate': 0.15,
                          'location_stratified': True,
                          'location_min_blank_molecules': 0,
                          'write_filtered_images': False}, 'NotSaved')
        counts = [0, 0]
        for fov, frame in self.raw.items():
            out, summary = in_memory.select_barcodes(fov, frame)
            for s in summary['strata'][:4]:
                self.assertEqual(s['source'], 'own')
            counts[0] += int(out['barcode_id'].isin([8, 9]).sum())
            counts[1] += int((~out['barcode_id'].isin([8, 9])).sum())
        self.assertLessEqual(misid(*counts), 0.15)
        self.assertFalse(os.path.exists(
            self.ds.get_analysis_subdirectory('NotSaved', create=False)))

    def test_unset_filter_is_the_pooled_filter(self):
        taskP, pooled = self.run_filter('FilterP', 'ThreshP')
        _, pooledOnS = self.run_filter('FilterPS', 'ThreshS')
        t = self.pooled
        threshold = t.calculate_threshold_for_misidentification_rate(0.05)
        for fov in self.ds.get_fovs():
            self.assertNotIn('location_stratum', pooled[fov].columns)
            self.assertTrue(pooled[fov].equals(pooledOnS[fov]))
            want = t.extract_barcodes_with_threshold(threshold, self.raw[fov])
            self.assertTrue(pooled[fov].equals(want))
        saved = read_json(os.path.join(
            self.ds.get_task_subdirectory('FilterP'), 'task.json'))
        for key in NEW_KEYS:
            self.assertNotIn(key, saved)
        self.assertEqual(taskP.get_estimated_memory(), 1000)
        self.assertFalse(os.path.exists(os.path.join(
            self.ds.get_analysis_subdirectory('FilterP'),
            'location_filter_summary_0.json')))
        # a task.json saved without the new keys regenerates
        with contextlib.redirect_stdout(io.StringIO()):
            filter_task(self.ds, 'FilterP', 'ThreshP')

    def test_log_lines(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            task = filter_task(self.ds, 'FilterLog', 'ThreshS',
                               location_stratified=True,
                               location_min_blank_molecules=0)
            task.run(1)
        lines = [l for l in buffer.getvalue().splitlines()
                 if l.startswith('FilterLog fov 1: ')]
        self.assertEqual(len(lines), 5)
        self.assertIn('nucleus (2) threshold', lines[2])
        self.assertIn('(own; fit ', lines[2])
        self.assertIn('unused (4) threshold', lines[4])
        self.assertIn('pooled: no_coding_molecules', lines[4])
        summary = task.get_location_filter_summary(1)
        self.assertTrue(lines[2].endswith(
            'kept %d coding + %d blank of %d + %d' % tuple(
                summary['strata'][2][k] for k in (
                    'kept_coding', 'kept_blank', 'in_coding', 'in_blank'))))
        with open(self.ds._log_path(task, 1)) as f:
            log = f.read()
        self.assertIn('WARNING', log)
        self.assertIn('unused (4)', log)
        self.assertNotIn('nucleus (2)', log)

    def test_equivalences_with_the_pooled_filter(self):
        _, pooled = self.run_filter('FilterPEq', 'ThreshP')
        # every molecule in one stratum, fit on its own
        task, one = self.run_filter('FilterOne', 'ThreshOne',
                                    location_stratified=True,
                                    location_min_blank_molecules=0)
        rules = task.get_location_filter_summary(0)['thresholds']['strata']
        self.assertEqual([(r['source'], r['reason']) for r in rules],
                         [('pooled', 'no_coding_molecules'), ('own', None)])
        # every stratum falling back to the pooled rule
        _, fallback = self.run_filter('FilterFallback', 'ThreshS',
                                      location_stratified=True,
                                      location_min_blank_molecules=10 ** 9)
        for fov in self.ds.get_fovs():
            self.assertTrue((one[fov]['location_stratum'] == 1).all())
            self.assertTrue(one[fov].drop(columns='location_stratum').equals(
                pooled[fov]))
            self.assertTrue(fallback[fov].drop(
                columns='location_stratum').equals(pooled[fov]))


class ExampleJsonTests(_Pipeline):
    """docs/analysis/location_filter.json runs as written, apart from the
    strata path, which is a placeholder there, and this fixture's settings."""

    def test_example_chain(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'docs', 'analysis', 'location_filter.json')
        threshold, chosen = read_json(path)['analysis_tasks']
        self.assertEqual(chosen['parameters']['adaptive_task'],
                         threshold['analysis_name'])
        self.assertEqual(threshold['parameters']['threshold_solver_method'],
                         'cumulative_bins')
        with contextlib.redirect_stdout(io.StringIO()):
            fit = getattr(importlib.import_module(threshold['module']),
                          threshold['task'])(
                self.ds, dict(threshold['parameters'], poll_interval=0,
                              location_strata_path=self.strata),
                threshold['analysis_name'])
            fit.save()
            fit.run()
            task = getattr(importlib.import_module(chosen['module']),
                           chosen['task'])(
                self.ds, dict(chosen['parameters'],
                              write_filtered_images=False),
                chosen['analysis_name'])
            task.save()
            for fov in self.ds.get_fovs():
                task.run(fov)
        self.assertTrue(fit.has_location_strata())
        for fov in self.ds.get_fovs():
            self.assertIn('location_stratum',
                          task.get_barcode_database().get_barcodes(fov))


class RefusalTests(_Pipeline):

    def test_option_validation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for extra, message in (
                    ({'location_min_blank_molecules': 10}, 'only read with'),
                    ({'location_stratified': False,
                      'location_min_blank_molecules': 10}, 'only read with'),
                    ({'location_stratified': 'yes'}, 'true or false'),
                    ({'location_stratified': True,
                      'location_min_blank_molecules': -1}, 'integer >= 0'),
                    ({'location_stratified': True,
                      'location_min_blank_molecules': 1.5}, 'integer >= 0'),
                    ({'location_stratified': True,
                      'location_min_blank_molecules': True}, 'integer >= 0')):
                with self.assertRaisesRegex(ValueError, message):
                    filterbarcodes.AdaptiveFilterBarcodes(
                        self.ds, dict(decode_task='Decode',
                                      adaptive_task='ThreshS', **extra), 'X')
            with self.assertRaisesRegex(ValueError, 'does not support'):
                filterbarcodes.AdaptiveFilterBarcodesLocal(
                    self.ds, {'decode_task': 'Decode', 'adaptive_task': 'L',
                              'location_stratified': True}, 'X')
            filterbarcodes.AdaptiveFilterBarcodesLocal(
                self.ds, {'decode_task': 'Decode', 'adaptive_task': 'L'}, 'X')
            for path in ('relative/strata', 5):
                with self.assertRaisesRegex(ValueError, 'absolute'):
                    filterbarcodes.GenerateAdaptiveThreshold(
                        self.ds, dict(BINS, decode_task='Decode',
                                      run_after_task='Decode',
                                      location_strata_path=path), 'X')
        f = filterbarcodes.AdaptiveFilterBarcodes(
            self.ds, {'decode_task': 'Decode', 'adaptive_task': 'ThreshS',
                      'location_stratified': True,
                      'write_filtered_images': False}, 'X')
        self.assertEqual(f.get_estimated_memory(), 4000)

    def test_threshold_without_strata(self):
        with contextlib.redirect_stdout(io.StringIO()):
            task = filter_task(self.ds, 'FilterNoStrata', 'ThreshP',
                               location_stratified=True)
        with self.assertRaisesRegex(ValueError,
                                    'set location_strata_path on ThreshP'):
            task.select_barcodes(0, self.raw[0])
        with self.assertRaisesRegex(ValueError, 'set location_strata_path'), \
                contextlib.redirect_stdout(io.StringIO()):
            task.run(0)

    def test_sidecar_changed_after_fit(self):
        copy = os.path.join(self._tmp.name, 'strata_changed')
        shutil.copytree(self.strata, copy)
        with contextlib.redirect_stdout(io.StringIO()):
            t = threshold_task(self.ds, 'ThreshChanged', strata=copy)
            t.run()
            task = filter_task(self.ds, 'FilterChanged', 'ThreshChanged',
                               location_stratified=True)
        task.select_barcodes(0, self.raw[0])
        locationstrata.write_metadata(copy, NAMES + ['unused'],
                                      source={'test': 'rewritten'})
        with self.assertRaisesRegex(RuntimeError, 'same strata'):
            task.select_barcodes(0, self.raw[0])
        # and resuming the threshold task refuses too
        with self.assertRaisesRegex(RuntimeError, 'cannot be resumed'), \
                contextlib.redirect_stdout(io.StringIO()):
            t.run()

    def test_resume_mismatch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            t = threshold_task(self.ds, 'ThreshResume', strata=self.strata)
            t.run()
            t.run()      # a rerun on consistent results is fine
        folder = self.ds.get_analysis_subdirectory('ThreshResume')
        strataFile = os.path.join(folder, 'blank_counts_strata.npy')
        original = np.load(strataFile)
        changed = original.copy()
        changed[1].flat[0] += 1
        np.save(strataFile, changed)
        # _run_analysis directly: after a failed run() the task counts as
        # running for two minutes (its run marker is fresh)
        with self.assertRaisesRegex(RuntimeError, 'sum of the strata'), \
                contextlib.redirect_stdout(io.StringIO()):
            t._run_analysis()
        os.remove(strataFile)
        with self.assertRaisesRegex(RuntimeError, 'partly saved'), \
                contextlib.redirect_stdout(io.StringIO()):
            t._run_analysis()
        np.save(strataFile, original)
        with contextlib.redirect_stdout(io.StringIO()):
            t._run_analysis()
        # pooled histograms from a run without strata
        with contextlib.redirect_stdout(io.StringIO()):
            u = threshold_task(self.ds, 'ThreshNoStrataYet')
            u.run()
            later = filterbarcodes.GenerateAdaptiveThreshold(
                self.ds, dict(u.parameters, location_strata_path=self.strata),
                'ThreshNoStrataYet')
        with self.assertRaisesRegex(RuntimeError, 'strata histograms missing'), \
                contextlib.redirect_stdout(io.StringIO()):
            later.run()

    def test_missing_volume(self):
        copy = os.path.join(self._tmp.name, 'strata_missing')
        shutil.copytree(self.strata, copy)
        os.remove(os.path.join(copy, 'strata_fov001.npz'))
        with contextlib.redirect_stdout(io.StringIO()):
            t = threshold_task(self.ds, 'ThreshMissing', strata=copy)
        with self.assertRaisesRegex(FileNotFoundError,
                                    'strata_fov001.npz'), \
                contextlib.redirect_stdout(io.StringIO()):
            t.run()


if __name__ == '__main__':
    unittest.main()
