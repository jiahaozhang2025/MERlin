"""FiducialPolynomialWarp3D xy_reject_zero_lock: xy samples that lock onto
the camera's fixed pattern at zero shift.

Every frame from one camera carries the same fixed pattern, which correlates
at exactly zero shift. On a plane with little bead texture that spike beats
the beads inside the xy window (+-xy_search_bound about the rigid offset), so
the sample reads (0, 0) although the stack is offset, and the cubic xy fit is
pulled towards zero. When the window misses zero shift by a pixel, its best
pixel is the spike's shoulder at the window edge nearest zero.

Synthetic bead VOLUMES (as tests/test_polywarp_z_xy_refine.py) rendered for
a reference round and four moving rounds, every plane and 2D frame with one
shared camera fixed pattern plus independent noise:

  channel 0  the reference stack (offset 0)
  channel 1  offset ~(6.4, -5.3) px plus a ramp in depth; no beads above
             BEAD_TOP, so the shallow samples lock at (0, 0) inside the window
  channel 2  offset ~(11.6, 0.3): the window starts at row 1, so the shallow
             samples lock on the window edge at (1, 0)
  channel 3  offset ~(0.8, -0.6), within xy_zero_lock_min_rigid_px of zero:
             its shallow samples read ~(0, 0) too, and are kept
  channel 4  no beads at any depth: every sample locks

The task's measurement, fit and table code run unchanged; only the dataset
is a stand-in. Channels 1 and 2 lock in a block at the top of the stack;
above it xy is the cubic through the kept samples (nothing is held).

A second dataset (LAYER below) has a coverglass bead layer and a second
thin layer just below it above the bulk, nothing between them and the bulk,
and a deeper stack:

  channel 5  the shallow samples are unlocked but,
             isolated above a block of locks, off the bulk and disagreeing
             with each other, trimmed by the fit; above the first kept
             sample xy is the cubic through the kept samples
  channel 6  beads in a band mid-stack only: mostly locks; beyond the band,
             at both ends, the cubic through the band's samples

Run with `python -m unittest tests/test_polywarp_zero_lock.py`.
"""
import contextlib
import io
import unittest

import cv2
import numpy as np
import pandas as pd

from merlin.analysis.warp import FiducialPolynomialWarp3D

SIZE = 256
NPLANES = 40
Z = np.arange(NPLANES) * 0.5                  # um
SIGMA_Z = 0.8                                 # um, bead extent in depth
BEAD_TOP = 7.0                                # um: no beads above this depth
OFFSETS = {0: (0.0, 0.0), 1: (6.4, -5.3), 2: (11.6, 0.3), 3: (0.8, -0.6),
           4: (-7.2, 6.1)}
RAMP = {0: (0.0, 0.0), 1: (0.08, 0.06), 2: (-0.05, 0.07), 3: (0.02, -0.03),
        4: (0.0, 0.0)}
TRUE_DZ = 0.4                                 # um


def true_shift(channel, z):
    """(dy, dx) shift to apply to the moving plane at reference depth z"""
    z = np.asarray(z, float)
    return (OFFSETS[channel][0] + RAMP[channel][0] * z,
            OFFSETS[channel][1] + RAMP[channel][1] * z)


def splat(ys, xs, weights, shape):
    """bilinear splat of point sources, then a lateral Gaussian blur"""
    image = np.zeros(shape, np.float64)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = ys - y0, xs - x0
    for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx),
                      (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        yy, xx = y0 + dy, x0 + dx
        ok = (yy >= 0) & (yy < shape[0]) & (xx >= 0) & (xx < shape[1])
        np.add.at(image, (yy[ok], xx[ok]), (weights * w)[ok])
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.2)


# the camera's fixed pattern: shared by every frame of every round
FIXED_PATTERN = np.random.default_rng(5).normal(0, 15, (SIZE, SIZE)).astype(
    np.float32)


def bead_volume(seed, n=4000, top=BEAD_TOP):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-40, SIZE + 40, n), rng.uniform(-40, SIZE + 40, n),
            rng.uniform(top, Z[-1] + 3, n), rng.uniform(800, 3000, n))


def render(volume, channel, seed):
    """planes of one round: a bead at reference (x, y, z) sits in the moving
    round at depth z + TRUE_DZ and at p - shift(z), shift being what brings it
    back; channel 4 has no beads"""
    xs, ys, zs, amplitude = volume
    if channel:
        sy, sx = true_shift(channel, zs)
        xs, ys, zs = xs - sx, ys - sy, zs + TRUE_DZ
    rng = np.random.default_rng(seed)
    planes = []
    for zPlane in Z:
        weight = amplitude * np.exp(-0.5 * ((zs - zPlane) / SIGMA_Z) ** 2)
        near = weight > 1.0
        beads = (splat(ys[near], xs[near], weight[near], (SIZE, SIZE))
                 if channel != 4 else np.zeros((SIZE, SIZE), np.float32))
        planes.append(beads + 200 + FIXED_PATTERN
                      + rng.normal(0, 10, (SIZE, SIZE)).astype(np.float32))
    return planes


def two_d(channel, seed):
    """the 2D coverglass frame: bright beads at the channel's surface offset"""
    rng = np.random.default_rng(91)
    xs, ys = rng.uniform(-40, SIZE + 40, 300), rng.uniform(-40, SIZE + 40, 300)
    amplitude = rng.uniform(8000, 30000, 300)
    sy, sx = true_shift(channel, 0.0)
    noise = np.random.default_rng(seed).normal(0, 20, (SIZE, SIZE))
    return (splat(ys - sy, xs - sx, amplitude, (SIZE, SIZE)) + 200
            + FIXED_PATTERN + noise.astype(np.float32))


CHANNELS = sorted(OFFSETS)
VOLUME = bead_volume(3)
STACKS = {c: render(VOLUME, c, 100 + c) for c in CHANNELS}
TWO_D = {c: two_d(c, 200 + c) for c in CHANNELS}


class _Org:
    def get_data_channels(self):
        return np.array(CHANNELS)

    def get_fiducial3D_filename(self, dataChannel, fov):
        return int(dataChannel)

    def get_fiducial3D_stack_frame_indices(self, dataChannel):
        return np.arange(NPLANES)

    def get_fiducial3D_stack_frame_zPos(self, dataChannel):
        return Z

    def get_data_channel_color(self, dataChannel):
        return '650'


class _DataSet:
    def __init__(self, tasks=None):
        self.saved = {}
        self.tasks = dict(tasks or {})

    def load_analysis_task(self, name):
        return self.tasks[name]

    def get_data_organization(self):
        return _Org()

    def get_fovs(self):
        return [0]

    def get_z_positions(self):
        return list(Z)

    def get_image_dimensions(self):
        return (SIZE, SIZE)

    def get_fiducial_image(self, dataChannel, fov):
        return TWO_D[int(dataChannel)]

    def load_image(self, name, frame):
        return STACKS[name][int(frame)]

    def save_dataframe_to_csv(self, dataframe, resultName, analysisTask=None,
                              resultIndex=None, subdirectory=None, **kwargs):
        self.saved[resultName, resultIndex] = dataframe.to_csv(index=False)

    def load_dataframe_from_csv(self, resultName, analysisTask=None,
                                resultIndex=None, subdirectory=None, **kwargs):
        if (resultName, resultIndex) not in self.saved:
            raise FileNotFoundError(resultName)
        return pd.read_csv(io.StringIO(self.saved[resultName, resultIndex]))

    def save_numpy_analysis_result(self, analysisResult, resultName,
                                   analysisName, resultIndex=None,
                                   subdirectory=None):
        self.saved[resultName, resultIndex] = np.array(analysisResult)


PARAMETERS = {
    'rigid_search_bound': 40, 'rigid_plane_search': 0,
    'z_search_range': 3, 'z_sample_step': 3, 'xy_sample_step': 2,
    'z_polynomial_order': 1, 'xy_polynomial_order': 3,
    'z_min_score': 0.5, 'z_min_good_samples': 4,
    'minimum_overlap_pixels': 1000, 'edge_width_to_remove': 8,
    'z_neighbor_substitution': False,
    'write_fiducial_images': False, 'write_aligned_images': False}


class _SpyWarp(FiducialPolynomialWarp3D):
    """Records every xy fit (result and arguments) and every z measurement
    with the shifts it was given (stack_num_threads 1: in order)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def _fit_xy_samples(self, *args, **kwargs):
        result = super()._fit_xy_samples(*args, **kwargs)
        self.calls.append(('fit', result, args))
        return result

    def _measure_z_offsets(self, fov, dataChannel, rigidY, rigidX,
                           shiftCoefficients=None):
        result = super()._measure_z_offsets(fov, dataChannel, rigidY, rigidX,
                                            shiftCoefficients)
        self.calls.append(('z', dataChannel, rigidY, rigidX,
                           shiftCoefficients, result[0]))
        return result


def run(guard, tasks=None, dataSet=None, taskClass=None, **extra):
    parameters = dict(PARAMETERS, **extra)
    if guard is not None:
        parameters['xy_reject_zero_lock'] = guard
    task = (taskClass or FiducialPolynomialWarp3D)(
        dataSet or _DataSet(tasks), parameters, 'PolyWarp')
    with contextlib.redirect_stdout(io.StringIO()) as out:
        task._run_analysis(0)
    task.log = out.getvalue()
    return task


def table(task, name):
    return task.dataSet.load_dataframe_from_csv(name, resultIndex=0)


def xy_samples(task, channel):
    """one row per xy sample: zPos, y, x, kept, on_edge (+ zero_lock)"""
    m = table(task, 'measurements')
    y = m[(m.dataChannel == channel) & (m.stage == 'y')].reset_index(drop=True)
    x = m[(m.dataChannel == channel) & (m.stage == 'x')].reset_index(drop=True)
    out = pd.DataFrame({'zPos': y.zPos, 'y': y.measured, 'x': x.measured,
                        'kept': y.kept, 'kept_both': y.kept & x.kept,
                        'on_edge': y.on_edge})
    if 'zero_lock' in y:
        out['zero_lock'] = y.zero_lock
    return out


def table_shift(task, channel, z):
    """(dy, dx) the transformation table applies at the plane nearest z"""
    t = table(task, 'transformation_table')
    rows = t[t.dataChannel == channel].set_index('zIndex')
    row = rows.loc[int(np.argmin(np.abs(Z - z)))]
    return -row.yshift, -row.xshift


def errors(task, channel, deep):
    """|table - truth| in px at every plane at or below BEAD_TOP (deep) or
    above it"""
    out = []
    for z in Z[(Z >= BEAD_TOP) if deep else (Z < BEAD_TOP)]:
        ty, tx = true_shift(channel, z)
        fy, fx = table_shift(task, channel, z)
        out.append(np.hypot(fy - ty, fx - tx))
    return np.array(out)


def status(task):
    return table(task, 'z_fit_status').set_index('dataChannel')


TABLES = ('z_fit_status', 'transformation_table', 'polynomial_coefficients',
          'measurements', 'fit_quality_flags')

# polynomial_coefficients, option on or off (no column for the option)
COEFFICIENT_COLUMNS = ['dataChannel', 'rigid_y', 'rigid_x', 'z_c0', 'z_c1',
                       'y_c0', 'y_c1', 'y_c2', 'y_c3',
                       'x_c0', 'x_c1', 'x_c2', 'x_c3']


def polynomial(row, axis):
    """one polynomial_coefficients row's y or x curve, np.polyval order"""
    return [row['%s_c%d' % (axis, k)] for k in (3, 2, 1, 0)]


class ZeroLockOffTests(unittest.TestCase):
    """Unset = false = the behaviour before the option, locks included."""

    @classmethod
    def setUpClass(cls):
        cls.off = run(False)
        cls.default = run(None)

    def test_default_is_off(self):
        self.assertFalse(self.default.parameters['xy_reject_zero_lock'])
        self.assertEqual(self.default.parameters['xy_zero_lock_reach_px'], 3.0)
        self.assertEqual(
            self.default.parameters['xy_zero_lock_min_rigid_px'], 2.0)
        for name in TABLES:
            self.assertEqual(self.default.dataSet.saved[name, 0],
                             self.off.dataSet.saved[name, 0], name)
        self.assertEqual(self.default.log, self.off.log)
        self.assertEqual(list(table(self.off, 'z_fit_status').columns), [
            'fov', 'dataChannel', 'owner_channel', 'n_samples', 'n_good',
            'n_gated_out', 'fitted', 'source'])
        self.assertNotIn('zero_lock', table(self.off, 'measurements').columns)
        self.assertEqual(
            list(table(self.off, 'polynomial_coefficients').columns),
            COEFFICIENT_COLUMNS)
        self.assertNotIn('zero-shift', self.off.log)

    def test_premise_locks_pull_the_fit(self):
        # the beadless shallow samples read (0, 0) and are kept by the fit
        samples = xy_samples(self.off, 1)
        shallow = samples[samples.zPos < BEAD_TOP - 1]
        self.assertGreaterEqual(len(shallow), 5)
        np.testing.assert_array_less(np.hypot(shallow.y, shallow.x), 0.1)
        self.assertGreaterEqual(int(shallow.kept.sum()), 3)
        # ... and pull the cubic off the true offset, above and below
        self.assertGreater(errors(self.off, 1, False).max(), 5.0)
        self.assertGreater(errors(self.off, 1, True).max(), 1.0)
        # the window of channel 2 starts at row 1: its locks sit on the edge
        samples = xy_samples(self.off, 2)
        shallow = samples[samples.zPos < BEAD_TOP - 1]
        self.assertTrue(shallow.on_edge.all())
        np.testing.assert_allclose(shallow.y, 1.0, atol=0.05)
        self.assertGreater(errors(self.off, 2, False).max(), 5.0)
        # a stack without beads is registered at zero shift at every depth
        t = table(self.off, 'transformation_table')
        rows = t[t.dataChannel == 4]
        np.testing.assert_array_less(np.hypot(rows.xshift, rows.yshift), 0.3)


class ZeroLockOnTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.off = run(False)
        cls.on = run(True)

    def test_in_window_locks_dropped(self):
        samples = xy_samples(self.on, 1)
        locked = samples[samples.zero_lock]
        # exactly the samples that read zero shift, all above the beads
        np.testing.assert_array_equal(
            samples.zero_lock, np.hypot(samples.y, samples.x) < 0.5)
        self.assertGreaterEqual(len(locked), 5)
        self.assertTrue((locked.zPos < BEAD_TOP).all())
        self.assertFalse(locked.kept.any())
        self.assertFalse(locked.on_edge.any())
        # the same measurements as without the option: only the fit changed
        off = xy_samples(self.off, 1)
        np.testing.assert_array_equal(samples.y, off.y)
        np.testing.assert_array_equal(samples.x, off.x)

    def test_edge_locks_dropped(self):
        samples = xy_samples(self.on, 2)
        locked = samples[samples.zero_lock]
        self.assertGreaterEqual(len(locked), 5)
        self.assertTrue(locked.on_edge.all())
        np.testing.assert_allclose(locked.y, 1.0, atol=0.05)
        self.assertTrue((locked.zPos < BEAD_TOP).all())
        self.assertFalse(locked.kept.any())
        # every sample at the window edge facing zero is one of them
        edge = samples[samples.on_edge & (np.abs(samples.y - 1.0) < 0.05)]
        self.assertTrue(edge.zero_lock.all())

    def test_fit_recovers_the_true_offset(self):
        for channel in (1, 2):
            deep = errors(self.on, channel, True)
            self.assertLess(deep.max(), 0.15, channel)
            # above the beads the cubic extrapolates from the deep samples;
            # without the option it is pulled to zero there
            self.assertLess(errors(self.on, channel, False).max(), 1.5,
                            channel)
            self.assertGreater(errors(self.off, channel, False).max(), 5.0,
                               channel)

    def test_near_zero_and_reference_stacks_keep_their_samples(self):
        st = status(self.on)
        # channel 3's rigid offset is ~1 px from zero: its beadless samples
        # also read ~(0, 0), and they are kept, as without the option
        self.assertEqual(st.loc[3, 'n_xy_zero_lock'], 0)
        self.assertEqual(st.loc[0, 'n_xy_zero_lock'], 0)
        for channel in (0, 3):
            for name in ('y', 'x'):
                np.testing.assert_array_equal(
                    xy_samples(self.on, channel)[name],
                    xy_samples(self.off, channel)[name])
        t_on = table(self.on, 'transformation_table')
        t_off = table(self.off, 'transformation_table')
        for channel in (0, 3):
            np.testing.assert_array_equal(
                t_on[t_on.dataChannel == channel].to_numpy(),
                t_off[t_off.dataChannel == channel].to_numpy())

    def test_every_sample_locked_falls_back_to_rigid(self):
        st = status(self.on)
        self.assertTrue(st.loc[4, 'xy_rigid_fallback'])
        self.assertEqual(st.loc[4, 'n_xy_zero_lock'],
                         st.loc[4, 'n_xy_samples'])
        coefficients = table(self.on, 'polynomial_coefficients')\
            .set_index('dataChannel').loc[4]
        rows = table(self.on, 'transformation_table')
        rows = rows[rows.dataChannel == 4]
        np.testing.assert_array_equal(rows.yshift, -coefficients.rigid_y)
        np.testing.assert_array_equal(rows.xshift, -coefficients.rigid_x)
        # the 2D frame's offset, the true one
        self.assertLess(np.hypot(coefficients.rigid_y - OFFSETS[4][0],
                                 coefficients.rigid_x - OFFSETS[4][1]), 0.1)
        flags = table(self.on, 'fit_quality_flags').set_index('dataChannel')
        self.assertIn('xy not fitted', flags.loc[4, 'reasons'])
        self.assertIn('XY NOT FITTED, fov 0 channel 4', self.on.log)
        for channel in (0, 1, 2, 3):
            self.assertFalse(st.loc[channel, 'xy_rigid_fallback'], channel)

    def test_status_and_qc_columns(self):
        st = table(self.on, 'z_fit_status')
        self.assertEqual(list(st.columns), [
            'fov', 'dataChannel', 'owner_channel', 'n_samples', 'n_good',
            'n_gated_out', 'fitted', 'source', 'n_xy_samples',
            'n_xy_zero_lock', 'xy_rigid_fallback'])
        st = st.set_index('dataChannel')
        for channel in CHANNELS:
            samples = xy_samples(self.on, channel)
            self.assertEqual(st.loc[channel, 'n_xy_samples'], len(samples))
            self.assertEqual(st.loc[channel, 'n_xy_zero_lock'],
                             int(samples.zero_lock.sum()))
        self.assertEqual(
            list(table(self.on, 'polynomial_coefficients').columns),
            COEFFICIENT_COLUMNS)
        m = table(self.on, 'measurements')
        self.assertFalse(m[m.stage == 'z'].zero_lock.any())
        self.assertIn('fov 0 channel 1: ', self.on.log)
        self.assertIn('%d xy zero-shift locks dropped'
                      % st.loc[1, 'n_xy_zero_lock'], self.on.log)
        # the z gate is not touched
        off = status(self.off)
        for name in ('n_samples', 'n_good', 'fitted', 'source'):
            np.testing.assert_array_equal(st[name], off[name], name)

    def test_no_hold_beyond_the_kept_samples(self):
        # above the block of locks xy is the cubic through the kept
        # samples, as below it: in the table and in the QC table's fitted
        # column (locks included); nothing is held
        coefficients = table(self.on, 'polynomial_coefficients')\
            .set_index('dataChannel')
        t = table(self.on, 'transformation_table')
        m = table(self.on, 'measurements')
        for channel in (1, 2):
            samples = xy_samples(self.on, channel)
            first = samples.zPos[samples.kept_both].min()
            # the locks are the block above the first kept sample
            self.assertGreaterEqual(
                int((samples.zero_lock & (samples.zPos < first)).sum()), 5)
            row = coefficients.loc[channel]
            rows = t[t.dataChannel == channel]
            above = rows.zPos < first
            self.assertGreater(int(above.sum()), 0)
            curves = {}
            for shift, axis in (('yshift', 'y'), ('xshift', 'x')):
                c = curves[axis] = polynomial(row, axis)
                np.testing.assert_allclose(-rows[shift],
                                           np.polyval(c, rows.zPos),
                                           rtol=0, atol=1e-12)
                qc = m[(m.dataChannel == channel) & (m.stage == axis)]
                np.testing.assert_allclose(qc.fitted, np.polyval(c, qc.zPos),
                                           rtol=0, atol=1e-12)
            # (where a hold at the first kept sample would differ)
            moved = np.hypot(
                np.polyval(curves['y'], rows.zPos[above])
                - np.polyval(curves['y'], first),
                np.polyval(curves['x'], rows.zPos[above])
                - np.polyval(curves['x'], first))
            self.assertGreater(moved.max(), 0.05, channel)
        self.assertNotIn('xy held', self.on.log)

    def test_residual_without_the_locks(self):
        flags = table(self.off, 'fit_quality_flags').set_index('dataChannel')
        flagsOn = table(self.on, 'fit_quality_flags')
        # without the option the locks wreck channel 1's xy residual
        self.assertIn('xy residual', flags.loc[1, 'reasons'])
        self.assertNotIn(1, set(flagsOn.dataChannel))


class ZeroLockRefineTests(unittest.TestCase):
    """z_xy_refine's xy fit leaves the locks out too."""

    @classmethod
    def setUpClass(cls):
        cls.off = run(False, z_xy_refine=True)
        cls.on = run(True, z_xy_refine=True, taskClass=_SpyWarp,
                     stack_num_threads=1)

    def test_refine_fit_without_locks(self):
        on, off = status(self.on), status(self.off)
        self.assertEqual(list(table(self.on, 'z_fit_status').columns)[-1],
                         'n_xy_zero_lock_first')
        self.assertGreaterEqual(on.loc[1, 'n_xy_zero_lock_first'], 5)
        # with the locks, the shallow xy fit used to shift the z samples is
        # pulled most of the way to zero; without them it is the ramp
        self.assertGreater(off.loc[1, 'refine_max_shift_px'], 5.0)
        self.assertLess(on.loc[1, 'refine_max_shift_px'], 2.5)
        self.assertGreaterEqual(on.loc[1, 'n_good'], off.loc[1, 'n_good'])
        self.assertLess(errors(self.on, 1, True).max(), 0.15)
        # an all-locked stack refines at its rigid offset
        self.assertEqual(on.loc[4, 'refine_max_shift_px'], 0.0)
        self.assertTrue(on.loc[4, 'xy_rigid_fallback'])

    def test_refine_shifts_are_the_cubic_without_the_locks(self):
        # the z samples of the refine pass are shifted by the xy fit of the
        # first pass without its locks, evaluated as it is at every depth,
        # above the kept samples too (no hold): the table's curve
        calls = self.on.calls
        st = status(self.on)
        checked = 0
        for index, call in enumerate(calls):
            if call[0] != 'z' or call[4] is None or call[1] not in (1, 2):
                continue
            _, channel, rigidY, rigidX, shift, zSamples = call
            _, ((yFit, yKept), (xFit, xKept), rigid), args = [
                c for c in calls[:index] if c[0] == 'fit'][-1]
            sampleZ, lock = np.asarray(args[0]), np.asarray(args[3], bool)
            # that fit left out the first pass's locks
            self.assertFalse(rigid, channel)
            self.assertGreaterEqual(int(lock.sum()), 5, channel)
            self.assertEqual(int(lock.sum()),
                             st.loc[channel, 'n_xy_zero_lock_first'])
            self.assertFalse((yKept | xKept)[lock].any(), channel)
            # the shift is that fit: (y, x) coefficients, nothing else
            self.assertEqual(len(shift), 2, channel)
            np.testing.assert_array_equal(shift[0], yFit)
            np.testing.assert_array_equal(shift[1], xFit)
            first = sampleZ[yKept & xKept].min()
            self.assertGreater(int((zSamples < first).sum()), 0, channel)
            moved = 0.0
            for zz in zSamples:
                sy, sx = self.on._sample_shift(rigidY, rigidX, shift, zz)
                self.assertEqual((sy, sx), (float(np.polyval(yFit, zz)),
                                            float(np.polyval(xFit, zz))))
                if zz < first:
                    moved = max(moved, np.hypot(sy - np.polyval(yFit, first),
                                                sx - np.polyval(xFit, first)))
            # (where a hold at the first kept sample would differ)
            self.assertGreater(moved, 0.05, channel)
            # refine_max_shift_px is taken over the same shifts
            expected = max(float(np.hypot(np.polyval(yFit, zz) - rigidY,
                                          np.polyval(xFit, zz) - rigidX))
                           for zz in zSamples)
            self.assertAlmostEqual(st.loc[channel, 'refine_max_shift_px'],
                                   expected, places=12)
            checked += 1
        self.assertEqual(checked, 2)


class _RoundTask:
    """Stands in for a finalized FiducialRoundRotation task: a small round
    angle on every moving stack (the beads are not turned; at this angle the
    frame corners move by 0.1 px)."""
    ANGLE = float(np.radians(0.05))

    def get_round_angles(self):
        return {c: (0.0 if c == 0 else self.ANGLE) for c in CHANNELS}

    def get_fov_verdicts(self, fov):
        return {c: (self.ANGLE, False, '') for c in CHANNELS if c}


class ZeroLockRoundAngleTests(unittest.TestCase):
    """With rotation_task the samples are measured on turned-back planes;
    zero shift is the same there, and the fallback is the rigid offset in
    the moving frame."""

    @classmethod
    def setUpClass(cls):
        cls.on = run(True, tasks={'RoundRot': _RoundTask()},
                     rotation_task='RoundRot')

    def test_locks_found_on_turned_back_planes(self):
        st = status(self.on)
        self.assertGreaterEqual(st.loc[1, 'n_xy_zero_lock'], 5)
        self.assertGreaterEqual(st.loc[2, 'n_xy_zero_lock'], 5)
        self.assertEqual(st.loc[3, 'n_xy_zero_lock'], 0)
        self.assertTrue(st.loc[4, 'xy_rigid_fallback'])
        self.assertLess(errors(self.on, 1, True).max(), 0.2)

    def test_fallback_in_the_moving_frame(self):
        coefficients = table(self.on, 'polynomial_coefficients')\
            .set_index('dataChannel').loc[4]
        self.assertAlmostEqual(coefficients.rotation_radians,
                               _RoundTask.ANGLE, places=12)
        self.assertEqual(coefficients.y_c0, coefficients.rigid_y)
        self.assertEqual(coefficients.x_c0, coefficients.rigid_x)
        for k in (1, 2, 3):
            self.assertEqual(coefficients['y_c%d' % k], 0.0)
            self.assertEqual(coefficients['x_c%d' % k], 0.0)


class ZeroLockPrimitiveTests(unittest.TestCase):

    def setUp(self):
        self.task = FiducialPolynomialWarp3D(
            _DataSet(), dict(PARAMETERS, xy_reject_zero_lock=True),
            'PolyWarp')

    def test_correlation_peak_signature(self):
        rng = np.random.default_rng(1)
        image = rng.normal(0, 1, (64, 64))
        moving = np.roll(image, (3, -4), axis=(0, 1))
        plain = self.task._bounded_correlation_peak(image, moving, 0.0, 0.0)
        self.assertEqual(len(plain), 4)
        withZero = self.task._bounded_correlation_peak(
            image, moving, 0.0, 0.0, zeroShift=True)
        self.assertEqual(withZero[:4], plain)
        zeroBest, distance = withZero[4]
        self.assertFalse(zeroBest)
        self.assertAlmostEqual(distance, 5.0)
        # an image against itself: the peak is zero shift
        same = self.task._bounded_correlation_peak(
            image, image, 6.0, -5.0, zeroShift=True)
        self.assertEqual(same[4], (True, 0.0))
        # zero shift one pixel outside the window: its edge pixel nearest
        # zero is the best, and zero shift is higher still
        edge = self.task._bounded_correlation_peak(
            cv2.GaussianBlur(image, (0, 0), 1.0),
            cv2.GaussianBlur(image, (0, 0), 1.0), 11.6, 0.0, bound=10,
            zeroShift=True)
        self.assertTrue(edge[3])
        self.assertEqual((round(edge[0]), round(edge[1])), (1, 0))
        self.assertEqual(edge[4], (True, 1.0))

    def test_lock_rule(self):
        lock = self.task._is_zero_lock
        self.assertTrue(lock((True, 0.0), 5.0, 0.0))
        self.assertTrue(lock((True, 2.9), 0.0, -2.1))
        self.assertFalse(lock((True, 3.2), 5.0, 0.0))     # beyond the reach
        self.assertFalse(lock((False, 0.0), 5.0, 0.0))    # zero not the best
        self.assertFalse(lock((True, 0.0), 1.5, 1.0))     # rigid ~1.8 px
        self.assertFalse(lock((True, 0.0), 2.0, 0.0))     # not MORE than 2
        self.assertTrue(lock((True, 0.0), 2.0, 0.1))
        self.assertTrue(lock((True, 0.4), 2.25, 0.0))     # rigid 2.25 px

    def test_fit_without_the_locks(self):
        # the locks are left out and the cubic is fitted to the rest; beyond
        # them it is the same cubic (no hold), in z_xy_refine's shift too
        z = np.arange(0.0, 100.0, 5.0)
        truth = 0.002 * (z - 20) ** 2 + 3.0
        for lock in (z > 32.0,                     # a deep block of locks
                     (z < 12.0) | (z > 82.0),      # locks at both ends
                     (z > 30.0) & (z < 60.0)):     # locks inside
            y = np.where(lock, 0.0, truth)
            x = np.where(lock, 0.0, -truth)
            result = self.task._fit_xy_samples(z, y, x, lock, 5.0, -5.0, 3)
            self.assertEqual(len(result), 3)
            (yFit, yKept), (xFit, xKept), rigid = result
            self.assertFalse(rigid)
            self.assertFalse((yKept | xKept)[lock].any())
            for fit, values in (((yFit, yKept), y), ((xFit, xKept), x)):
                expected = self.task._robust_polyfit(z, values, 3, ~lock)
                np.testing.assert_array_equal(fit[0], expected[0])
                np.testing.assert_array_equal(fit[1], expected[1])
            # the unlocked samples lie on a quadratic: the cubic follows it
            # beyond them as well
            np.testing.assert_allclose(np.polyval(yFit, z), truth, rtol=0,
                                       atol=1e-6)
            np.testing.assert_allclose(np.polyval(xFit, z), -truth, rtol=0,
                                       atol=1e-6)
            for depth in (0.0, 60.0, 95.0, 120.0):
                self.assertEqual(
                    self.task._sample_shift(5.0, -5.0, (yFit, xFit), depth),
                    (float(np.polyval(yFit, depth)),
                     float(np.polyval(xFit, depth))))

    def test_too_few_left_is_the_rigid_offset(self):
        z = np.arange(0.0, 100.0, 5.0)
        truth = 0.002 * (z - 20) ** 2 + 3.0
        # order 3 needs order + 2 = 5 samples; 4 left: the rigid offset
        few = z > 15.0
        (yFit, yKept), (xFit, xKept), rigid = self.task._fit_xy_samples(
            z, truth, -truth, few, 5.0, -5.0, 3)
        self.assertTrue(rigid)
        np.testing.assert_array_equal(yFit, [0.0, 0.0, 0.0, 5.0])
        np.testing.assert_array_equal(xFit, [0.0, 0.0, 0.0, -5.0])
        self.assertFalse(yKept.any() or xKept.any())
        # 5 left: fitted
        five = z > 20.0
        result = self.task._fit_xy_samples(z, truth, -truth, five, 5.0,
                                           -5.0, 3)
        self.assertFalse(result[2])
        self.assertTrue(result[0][1][~five].all())
        # a non-finite sample does not count
        y = truth.copy()
        y[0] = np.nan
        self.assertTrue(self.task._fit_xy_samples(z, y, -truth, five, 5.0,
                                                  -5.0, 3)[2])

    def test_option_off_fits_every_sample(self):
        # off: the two robust fits over every sample, locks included, as
        # before the option; never the rigid fallback
        z = np.arange(0.0, 100.0, 5.0)
        truth = 0.002 * (z - 20) ** 2 + 3.0
        lock = z > 32.0
        y = np.where(lock, 0.0, truth)
        x = np.where(lock, 0.0, -truth)
        off = FiducialPolynomialWarp3D(_DataSet(), dict(PARAMETERS),
                                       'PolyWarp')
        for flags in (lock, np.ones(len(z), bool)):
            result = off._fit_xy_samples(z, y, x, flags, 5.0, -5.0, 3)
            self.assertEqual(len(result), 3)
            self.assertFalse(result[2])
            for fit, values in ((result[0], y), (result[1], x)):
                expected = off._robust_polyfit(z, values, 3)
                np.testing.assert_array_equal(fit[0], expected[0])
                np.testing.assert_array_equal(fit[1], expected[1])

    def test_options_validated(self):
        for bad in ({'xy_reject_zero_lock': 1},
                    {'xy_reject_zero_lock': 'yes'},
                    {'xy_zero_lock_reach_px': -1.0},
                    {'xy_zero_lock_min_rigid_px': 'a'},
                    {'xy_zero_lock_min_rigid_px': True},
                    {'xy_zero_lock_reach_px': float('nan')}):
            with self.assertRaises(ValueError, msg=str(bad)):
                FiducialPolynomialWarp3D(_DataSet(), dict(PARAMETERS, **bad),
                                         'PolyWarp')



# ---- the second dataset: bead layers above the bulk ----
# The reference has a coverglass bead layer at z 0-0.6 um, a second thin
# layer at 1.3-1.7 um and the bulk from BEAD_TOP down, nothing between them
# and the bulk, over a stack twice as deep as the first dataset's. Per
# moving channel: the coverglass layer's offset (= the 2D frame's, the
# rigid offset), the second layer's and the bulk's steps from it, a ramp in
# depth, and which of the reference's beads it shows.
LAYER_TOP = 0.6
SECOND_LAYER = (1.3, 1.7)
LAYER_NPLANES = 80
LAYER_Z = np.arange(LAYER_NPLANES) * 0.5      # um
LAYER_OFFSET = {0: (0.0, 0.0), 5: (5.5, -6.0), 6: (-6.0, 5.5)}
LAYER_SECOND = {0: (0.0, 0.0), 5: (-4.5, 3.0), 6: (0.0, 0.0)}
LAYER_STEP = {0: (0.0, 0.0), 5: (3.0, 2.5), 6: (0.0, 0.0)}
LAYER_RAMP = {0: (0.0, 0.0), 5: (0.05, -0.04), 6: (0.03, 0.05)}
LAYER_SHOWS = {0: lambda z: np.ones(len(z), bool),
               5: lambda z: np.ones(len(z), bool),
               6: lambda z: (z >= 10.0) & (z <= 18.0)}
LAYER_CHANNELS = sorted(LAYER_OFFSET)


def layer_shift(channel, z):
    """(dy, dx) that brings the moving plane at reference depth z back"""
    z = np.asarray(z, float)
    second = (z > 1.0) & (z < 2.5)
    bulk = z >= 2.5
    return tuple(LAYER_OFFSET[channel][k]
                 + np.where(second, LAYER_SECOND[channel][k], 0.0)
                 + np.where(bulk, LAYER_STEP[channel][k], 0.0)
                 + LAYER_RAMP[channel][k] * z for k in (0, 1))


def layer_volume(seed=13):
    rng = np.random.default_rng(seed)
    layer, second, bulk = 1500, 1500, 8000
    n = layer + second + bulk
    zs = np.r_[rng.uniform(0.0, LAYER_TOP, layer),
               rng.uniform(*SECOND_LAYER, second),
               rng.uniform(BEAD_TOP, LAYER_Z[-1] + 3, bulk)]
    return (rng.uniform(-40, SIZE + 40, n), rng.uniform(-40, SIZE + 40, n),
            zs, rng.uniform(800, 3000, n))


def layer_render(volume, channel, seed):
    xs, ys, zs, amplitude = volume
    shows = LAYER_SHOWS[channel](zs)
    xs, ys, zs, amplitude = xs[shows], ys[shows], zs[shows], amplitude[shows]
    if channel:
        sy, sx = layer_shift(channel, zs)
        xs, ys, zs = xs - sx, ys - sy, zs + TRUE_DZ
    rng = np.random.default_rng(seed)
    planes = []
    for zPlane in LAYER_Z:
        weight = amplitude * np.exp(-0.5 * ((zs - zPlane) / SIGMA_Z) ** 2)
        near = weight > 1.0
        planes.append(splat(ys[near], xs[near], weight[near], (SIZE, SIZE))
                      + 200 + FIXED_PATTERN
                      + rng.normal(0, 10, (SIZE, SIZE)).astype(np.float32))
    return planes


def layer_two_d(channel, seed):
    rng = np.random.default_rng(91)
    xs, ys = rng.uniform(-40, SIZE + 40, 300), rng.uniform(-40, SIZE + 40, 300)
    amplitude = rng.uniform(8000, 30000, 300)
    sy, sx = LAYER_OFFSET[channel]
    noise = np.random.default_rng(seed).normal(0, 20, (SIZE, SIZE))
    return (splat(ys - sy, xs - sx, amplitude, (SIZE, SIZE)) + 200
            + FIXED_PATTERN + noise.astype(np.float32))


LAYER_VOLUME = layer_volume()
LAYER_STACKS = {c: layer_render(LAYER_VOLUME, c, 300 + c)
                for c in LAYER_CHANNELS}
LAYER_TWO_D = {c: layer_two_d(c, 400 + c) for c in LAYER_CHANNELS}


class _LayerOrg(_Org):
    def get_data_channels(self):
        return np.array(LAYER_CHANNELS)

    def get_fiducial3D_stack_frame_indices(self, dataChannel):
        return np.arange(LAYER_NPLANES)

    def get_fiducial3D_stack_frame_zPos(self, dataChannel):
        return LAYER_Z


class _LayerDataSet(_DataSet):
    def get_data_organization(self):
        return _LayerOrg()

    def get_z_positions(self):
        return list(LAYER_Z)

    def get_fiducial_image(self, dataChannel, fov):
        return LAYER_TWO_D[int(dataChannel)]

    def load_image(self, name, frame):
        return LAYER_STACKS[name][int(frame)]


def run_layer(guard, **extra):
    return run(guard, dataSet=_LayerDataSet(), **extra)


class ZeroLockLayerTests(unittest.TestCase):
    """Locks at an end of the stack, beyond the outermost kept sample: a
    trimmed unlocked end sample, locks at both ends,
    mostly locks. Beyond the kept samples xy is the cubic through them."""

    @classmethod
    def setUpClass(cls):
        cls.on = run_layer(True)
        cls.off = run_layer(False)

    def coefficients(self):
        return table(self.on, 'polynomial_coefficients')\
            .set_index('dataChannel')

    def test_trimmed_surface_samples(self):
        samples = xy_samples(self.on, 5)
        lock, kept = samples.zero_lock, samples.kept_both
        # the layers' samples: unlocked, and trimmed
        layer = samples.zPos < 2.0
        self.assertGreaterEqual(int(layer.sum()), 2)
        self.assertFalse(lock[layer].any())
        self.assertFalse(kept[layer].any())
        first = samples.zPos[kept].min()
        # the outermost sample is unlocked; a block of locks lies between
        # the layers and the outermost KEPT sample
        self.assertEqual(samples.zPos.iloc[0], samples.zPos[~lock].min())
        self.assertGreaterEqual(int((lock & (samples.zPos < first)).sum()), 3)
        self.assertNotIn('xy held', self.on.log)
        # the table and the QC fitted column: the polynomial at every depth,
        # above the first kept sample too
        row = self.coefficients().loc[5]
        t = table(self.on, 'transformation_table')
        rows = t[t.dataChannel == 5]
        m = table(self.on, 'measurements')
        above = rows.zPos < first
        self.assertGreater(int(above.sum()), 0)
        for shift, axis in (('yshift', 'y'), ('xshift', 'x')):
            c = polynomial(row, axis)
            np.testing.assert_allclose(-rows[shift], np.polyval(c, rows.zPos),
                                       rtol=0, atol=1e-9)
            qc = m[(m.dataChannel == 5) & (m.stage == axis)]
            np.testing.assert_allclose(qc.fitted, np.polyval(c, qc.zPos),
                                       rtol=0, atol=1e-9)
            # (where a hold at the first kept sample would differ)
            self.assertGreater(np.abs(np.polyval(c, rows.zPos[above])
                                      - np.polyval(c, first)).max(), 0.05)
        # the bulk is fitted
        for zz in LAYER_Z[(LAYER_Z >= first) & (LAYER_Z <= 35.0)]:
            ty, tx = layer_shift(5, zz)
            fy, fx = -rows.yshift[rows.zPos == zz].iloc[0], \
                -rows.xshift[rows.zPos == zz].iloc[0]
            self.assertLess(np.hypot(fy - ty, fx - tx), 0.25, zz)

    def test_cubic_beyond_both_ends(self):
        samples = xy_samples(self.on, 6)
        lock, kept = samples.zero_lock, samples.kept_both
        first, last = samples.zPos[kept].min(), samples.zPos[kept].max()
        self.assertTrue((lock & (samples.zPos < first)).any())
        self.assertTrue((lock & (samples.zPos > last)).any())
        self.assertNotIn('xy held', self.on.log)
        row = self.coefficients().loc[6]
        t = table(self.on, 'transformation_table')
        rows = t[t.dataChannel == 6]
        for shift, axis in (('yshift', 'y'), ('xshift', 'x')):
            c = polynomial(row, axis)
            np.testing.assert_allclose(-rows[shift], np.polyval(c, rows.zPos),
                                       rtol=0, atol=1e-9)
            # (where a hold at either end would differ)
            for end, beyond in ((first, rows.zPos < first),
                                (last, rows.zPos > last)):
                self.assertGreater(int(beyond.sum()), 0)
                self.assertGreater(np.abs(np.polyval(c, rows.zPos[beyond])
                                          - np.polyval(c, end)).max(), 0.05)
        # inside the band xy is within a pixel of the truth
        for zz in LAYER_Z[(LAYER_Z >= first) & (LAYER_Z <= last)]:
            ty, tx = layer_shift(6, zz)
            fy, fx = (-rows.yshift[rows.zPos == zz].iloc[0],
                      -rows.xshift[rows.zPos == zz].iloc[0])
            self.assertLess(np.hypot(fy - ty, fx - tx), 1.0, zz)

    def test_residual_without_the_locks(self):
        # channel 6 is mostly locks: left in the residual, they would be
        # its median
        samples = xy_samples(self.on, 6)
        lock = samples.zero_lock.to_numpy(bool)
        self.assertGreater(lock.mean(), 0.6)
        self.assertGreaterEqual(int((~lock).sum()), 5)
        m = table(self.on, 'measurements')
        y = m[(m.dataChannel == 6) & (m.stage == 'y')]
        x = m[(m.dataChannel == 6) & (m.stage == 'x')]
        expected = (np.median(np.abs(y.measured - y.fitted)[~lock]),
                    np.median(np.abs(x.measured - x.fitted)[~lock]))
        withLocks = (np.median(np.abs(y.measured - y.fitted)),
                     np.median(np.abs(x.measured - x.fitted)))
        self.assertGreater(max(withLocks), 1.0)
        self.assertLess(max(expected), 0.5)
        line = [l for l in self.on.log.splitlines()
                if l.startswith('fov 0 channel 6: ')][0]
        self.assertIn('xy fit residual %.3f / %.3f px' % expected, line)
        flags = table(self.on, 'fit_quality_flags')
        reasons = flags[flags.dataChannel == 6].reasons.tolist()
        self.assertFalse(any('xy residual' in r for r in reasons), reasons)

    def test_option_off_unchanged(self):
        # unset = false; no new column: the plain polynomial
        unset = run_layer(None)
        for name in TABLES:
            self.assertEqual(unset.dataSet.saved[name, 0],
                             self.off.dataSet.saved[name, 0], name)
        self.assertEqual(unset.log, self.off.log)
        coefficients = table(self.off, 'polynomial_coefficients')\
            .set_index('dataChannel')
        self.assertEqual(list(coefficients.reset_index().columns),
                         COEFFICIENT_COLUMNS)
        self.assertNotIn('zero_lock', table(self.off, 'measurements'))
        t = table(self.off, 'transformation_table')
        for channel in (5, 6):
            row = coefficients.loc[channel]
            rows = t[t.dataChannel == channel]
            for shift, axis in (('yshift', 'y'), ('xshift', 'x')):
                np.testing.assert_allclose(-rows[shift],
                                           np.polyval(polynomial(row, axis),
                                                      rows.zPos),
                                           rtol=0, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
