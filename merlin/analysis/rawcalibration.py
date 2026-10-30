"""Opt-in per-bit alignment from the MERFISH signal; existing tasks are unchanged.

The dependency graph has two preprocessors: baseline -> calibration -> bit warp
-> corrected preprocessor. See docs/analysis/bit_alignment.md for the task chain.
"""
import gc
import hashlib
import json

import numpy as np
from skimage import transform

from merlin.core import analysistask
from merlin.analysis import decode, warp, filterbarcodes
from merlin.util import aberration, bitregistration, decoding


def _state_id(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _corrector(state):
    return aberration.RigidChromaticCorrector(
        bitregistration.transformations(state['chromatic']), state['reference_color'])


def _check_base_warp(warpTask, fov, channels):
    """Bit-aligned sampling re-implements Warp.get_aligned_image: the raw frame
    at the plane's own z, the chromatic map, then the base warp's
    get_transformation(fov, channel) with the bit shift. Refuse a base warp
    whose own sampler does more (depth resampling, xy that changes with depth,
    neighbour fill) rather than drop that correction silently."""
    kind = type(warpTask)
    if kind is warp.FiducialPolynomialWarp3D:
        table = warpTask.get_transformation_table(fov)
        table = table[table['dataChannel'].isin(channels)]
        moves = bool((np.abs(table['zPos_new'] - table['zPos']) > 1e-6).any())
        for column in ('xshift', 'yshift', 'rotation_radians'):
            if column in table:
                spread = table.groupby('dataChannel')[column].agg(lambda v: v.max() - v.min())
                moves = moves or bool((spread > 1e-6).any())
        if moves:
            raise ValueError('Bit alignment needs a depth-independent FiducialPolynomialWarp3D: '
                             'fov %d has a z offset or an xy that changes with depth' % fov)
    elif not (kind.get_aligned_image is warp.Warp.get_aligned_image
              or (kind is warp.FiducialCorrelationWarp and not warpTask.parameters.get('register_3d', False))):
        raise ValueError('Bit alignment does not support %s as the base warp' % kind.__name__)


def _global_fragment(data_name, data_home, analysis_home, task_name, step, index, state):
    from merlin.core.dataset import MERFISHDataSet
    ds = MERFISHDataSet(data_name, dataHome=data_home, analysisHome=analysis_home)
    task = ds.load_analysis_task(task_name)
    opt = task._bootstrap()
    opt.parameters['scale_factor_region'] = task.parameters['continuation_scale_factor_region']
    prep = ds.load_analysis_task(task.parameters['preprocess_task'])
    fov, z = task.parameters['calibration_fov_index'][index]
    images = prep.get_processed_image_set(fov, z, _corrector(state)).astype(np.float32)
    decoder = decoding.PixelBasedDecoder(opt.get_codebook())
    decoder.refactorAreaThreshold = opt.parameters['area_threshold']
    decoder.barcodesSeenThreshold = opt.parameters['min_barcodes_for_refactoring']
    code, mag, traces, distance = decoder.decode_pixels(
        images, scaleFactors=np.asarray(state['scales']), backgrounds=np.asarray(state['backgrounds']),
        lowPassSigma=0, distanceThreshold=opt.parameters['distance_threshold'],
        distanceMetric=opt.parameters['distance_metric'], useGpu=False,
        tilingOverlap=opt.parameters['tiling_overlap'], tilingNumThreads=1,
        neighborNumJobs=int(task.parameters['neighbor_jobs']))
    ref, bg, counts, barcodes = opt._refactors_and_barcodes(
        decoder, code, mag, traces, distance, fov, z, opt.parameters['area_threshold'])
    del code, mag, traces, distance, decoder
    gc.collect()
    sampled = barcodes.copy()
    sampled['_tile'] = (sampled.y // 320).astype(int) * 10 + (sampled.x // 320).astype(int)
    sampled = (sampled.groupby('_tile', group_keys=False).sample(frac=1., random_state=1700+fov+step)
               .groupby('_tile', group_keys=False).head(100))
    samples = opt._measure_chromatic_samples(images, sampled)
    result = dict(fov=int(fov), z=int(z), step=step, barcodes=len(barcodes),
                  chromatic_barcodes=len(sampled), refactors=np.asarray(ref).tolist(),
                  background_refactors=np.asarray(bg).tolist(), barcode_counts=np.asarray(counts).tolist())
    fragment = (step - 1) * len(task.parameters['calibration_fov_index']) + index
    ds.save_json_analysis_result(result, 'fragment', task_name, resultIndex=fragment)
    ds.save_pickle_analysis_result(samples, 'chromatic_samples', task_name, resultIndex=fragment)
    return result, samples


class OptimizeRawCalibration(analysistask.AnalysisTask):
    """Freeze a bootstrap C/S pair, optionally continuing global calibration.

    bootstrap_task is an ordinary OptimizeIteration/OptimizeLoop chain run with
    normalize_scale_factors false. max_iterations=0 freezes its applied
    chromatic map C with its scale factors S. Otherwise the fit continues from
    its proposed C on calibration_fov_index; each state pairs the refitted S
    with the C actually used to estimate it.
    """
    outputGroup = 'Optimize'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        defaults = dict(max_iterations=12, calibration_fov_index=[], num_workers=1,
                        neighbor_jobs=1, chromatic_tolerance_px=.05,
                        scale_relative_tolerance=.015, stable_iterations=2,
                        require_convergence=True, continuation_scale_factor_region='crop')
        for key, value in defaults.items():
            self.parameters.setdefault(key, value)
        if int(self.parameters['max_iterations']) < 0:
            raise ValueError('max_iterations must be nonnegative')
        if self.parameters['max_iterations'] and not self.parameters['calibration_fov_index']:
            raise ValueError('Explicit calibration_fov_index is required')

    def get_dependencies(self):
        return [self.parameters['bootstrap_task'], self.parameters['preprocess_task']]

    def get_estimated_memory(self):
        return 12000 * int(self.parameters['num_workers'])

    def get_estimated_time(self):
        return 120

    def _bootstrap(self):
        task = self.dataSet.load_analysis_task(self.parameters['bootstrap_task'])
        return task.final_iteration() if hasattr(task, 'final_iteration') else task

    def get_state(self):
        if not self.is_complete():
            raise RuntimeError('Raw calibration is incomplete')
        record = self.dataSet.load_json_analysis_result('calibration', self)
        if record['calibration_id'] != _state_id(record['state']):
            raise ValueError('Frozen global calibration checksum differs')
        return record['state']

    def get_scale_factors(self):
        return np.asarray(self.get_state()['scales'])

    def get_backgrounds(self):
        return np.asarray(self.get_state()['backgrounds'])

    def get_previous_chromatic_corrector(self):
        return _corrector(self.get_state())

    get_chromatic_corrector = get_previous_chromatic_corrector

    def get_codebook(self):
        return self._bootstrap().get_codebook()

    def _run_analysis(self):
        opt = self._bootstrap()
        # Never normalize an absolute-factor run on read.
        if opt.parameters.get('normalize_scale_factors', True):
            raise ValueError('Bootstrap must set normalize_scale_factors=false')
        scales, backgrounds = opt.get_scale_factors(), opt.get_backgrounds()
        initial = dict(chromatic=bitregistration.matrix_dict(opt.get_chromatic_corrector()),
                       scales=np.asarray(scales, np.float32).ravel().tolist(),
                       backgrounds=np.asarray(backgrounds, np.float32).ravel().tolist(),
                       reference_color=opt.get_reference_color())
        if not self.parameters['max_iterations']:
            initial['chromatic'] = bitregistration.matrix_dict(opt.get_previous_chromatic_corrector())
            self._freeze(initial, 0, True)
            return
        state, stable = initial, 0
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        workers = int(self.parameters['num_workers'])
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) if workers > 1 else None
        try:
            for step in range(1, int(self.parameters['max_iterations']) + 1):
                try:
                    record = self.dataSet.load_json_analysis_result('iteration', self, resultIndex=step)
                    if record['input_state'] != state:
                        raise ValueError('Resume state differs from frozen iteration input')
                except FileNotFoundError:
                    args = [(self.dataSet.dataSetName, self.dataSet.dataHome, self.dataSet.analysisHome,
                             self.analysisName, step, i, state)
                            for i in range(len(self.parameters['calibration_fov_index']))]
                    if pool:
                        futures = [pool.submit(_global_fragment, *a) for a in args]
                        results = [f.result() for f in futures]
                    else:
                        results = [_global_fragment(*a) for a in args]
                    old = bitregistration.transformations(state['chromatic'])
                    new = {}
                    for a in old:
                        new[a] = {}
                        for b in old[a]:
                            values = [r[1][a][b] for r in results]
                            pos = np.concatenate([v[0] for v in values])
                            disp = np.concatenate([v[1] for v in values])
                            good = np.isfinite(pos).all(1) & np.isfinite(disp).all(1)
                            if good.sum() < 100:
                                raise ValueError('Too few chromatic samples for %s/%s' % (a, b))
                            update = transform.SimilarityTransform()
                            if not update.estimate(pos[good], pos[good] + disp[good]):
                                raise ValueError('Chromatic similarity fit failed')
                            new[a][b] = update + old[a][b]
                    sf, bg = np.asarray(state['scales']), np.asarray(state['backgrounds'])
                    ref = np.asarray([r[0]['refactors'] for r in results]); ref[ref == 0] = 1
                    new_sf = np.nanmedian(ref * sf, axis=0)
                    new_bg = np.nanmedian([bg + np.asarray(r[0]['background_refactors']) * sf for r in results], axis=0)
                    if not np.isfinite(new_sf).all() or not (new_sf > 0).all() or not np.isfinite(new_bg).all():
                        raise ValueError('Nonfinite/invalid global calibration')
                    change = bitregistration.chromatic_step(old, new, self.dataSet.get_image_dimensions())
                    relative = float(np.max(np.abs(new_sf / sf - 1)))
                    if change >= 1.5:
                        raise ValueError('Unreliable global chromatic step: %s' % change)
                    stable = stable + 1 if (change < self.parameters['chromatic_tolerance_px']
                                            and relative < self.parameters['scale_relative_tolerance']) else 0
                    applied = dict(state, scales=new_sf.tolist(), backgrounds=new_bg.tolist())
                    proposed = dict(applied, chromatic=bitregistration.matrix_dict(new))
                    record = dict(input_state=state, consistent_output_state=applied, proposed_next_state=proposed,
                                  max_chromatic_update_px=change, max_scale_relative_update=relative,
                                  stable_steps=stable, step=step)
                    self.dataSet.save_json_analysis_result(record, 'iteration', self, resultIndex=step)
                stable = record['stable_steps']
                print('%s step %d: chromatic %.5f px, scale %.5f, stable %d' %
                      (self.analysisName, step, record['max_chromatic_update_px'],
                       record['max_scale_relative_update'], stable), flush=True)
                converged = stable >= int(self.parameters['stable_iterations'])
                if converged or step == int(self.parameters['max_iterations']):
                    if not converged and self.parameters['require_convergence']:
                        raise RuntimeError('Global calibration did not converge; no final state published')
                    self._freeze(record['consistent_output_state'], step, converged)
                    return
                state = record['proposed_next_state']
        finally:
            if pool:
                pool.shutdown()

    def _freeze(self, state, step, converged):
        self.dataSet.save_json_analysis_result(
            dict(state=state, calibration_id=_state_id(state), step=step, converged=converged,
                 bootstrap_task=self.parameters['bootstrap_task'], absolute_scale_factors=True,
                 paired_exposure_used=False, neural_models_trained=False,
                 scale_factors_refitted_under_applied_chromatic_map=True), 'calibration', self)


class OptimizeBitAlignment(analysistask.ParallelAnalysisTask):
    """Fit and validate independent (FOV, z, bit) translations from own RNA.

    Missing/rejected fits are explicitly recorded and apply zero residual
    shift. Asking for a z plane absent from z_indices raises, rather than
    silently reusing the pilot plane's correction.
    """
    outputGroup = 'Optimize'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self.parameters.setdefault('fovs', list(map(int, dataSet.get_fovs())))
        self.parameters.setdefault('z_indices', list(range(len(dataSet.get_z_positions()))))
        self.parameters.setdefault('support_margin', 32)
        self.parameters.setdefault('distance_threshold', .5176)
        self.parameters.setdefault('magnitude_threshold', 1.)
        self.parameters.setdefault('neighbor_jobs', 1)
        self.parameters.setdefault('tile_size', 320)
        self.parameters.setdefault('intensity_factor_dtype', 'float64')
        if self.parameters['intensity_factor_dtype'] not in ('float32', 'float64'):
            raise ValueError('intensity_factor_dtype must be float32 or float64')
        if len(set(self.parameters['fovs'])) != len(self.parameters['fovs']) or len(set(self.parameters['z_indices'])) != len(self.parameters['z_indices']):
            raise ValueError('Duplicate FOV/z indices')
        if not set(self.parameters['fovs']) <= set(dataSet.get_fovs()):
            raise ValueError('Unknown FOV')
        if not set(self.parameters['z_indices']) <= set(range(len(dataSet.get_z_positions()))):
            raise ValueError('Unknown z index')

    def fragment_count(self):
        return len(self.parameters['fovs']) * len(self.parameters['z_indices'])

    def plane(self, fragment):
        f, z = divmod(fragment, len(self.parameters['z_indices']))
        return int(self.parameters['fovs'][f]), int(self.parameters['z_indices'][z])

    def fragment_for_plane(self, fov, z):
        return (self.parameters['fovs'].index(int(fov)) * len(self.parameters['z_indices'])
                + self.parameters['z_indices'].index(int(z)))

    def get_dependencies(self):
        return [self.parameters['calibration_task'], self.parameters['preprocess_task']]

    def get_estimated_memory(self):
        return 16000

    def get_estimated_time(self):
        return 10

    def get_result(self, fov, z):
        index = self.fragment_for_plane(fov, z)
        if not self.is_complete(index):
            raise RuntimeError('Bit calibration incomplete for FOV %d, z %d' % (fov, z))
        result = self.dataSet.load_json_analysis_result('bit_alignment', self, resultIndex=index)
        calibration = self.dataSet.load_analysis_task(self.parameters['calibration_task'])
        if result['calibration_id'] != _state_id(calibration.get_state()):
            raise ValueError('Bit offsets were fitted under a different global calibration')
        return result

    def get_offsets(self, fov, z):
        return np.asarray(self.get_result(fov, z)['applied_offsets_xy'])

    def _run_analysis(self, fragmentIndex):
        fov, z = self.plane(fragmentIndex)
        calibration = self.dataSet.load_analysis_task(self.parameters['calibration_task'])
        prep = self.dataSet.load_analysis_task(self.parameters['preprocess_task'])
        state, book = calibration.get_state(), prep.get_codebook()
        corrector = _corrector(state)
        channels = decode.codebook_data_channels(self.dataSet, book)
        _check_base_warp(prep.warpTask, fov, channels)
        org = self.dataSet.get_data_organization()
        colors = [str(org.get_data_channel_color(ch)) for ch in channels]
        codebook = book.get_barcodes()
        blank = np.zeros(len(codebook), bool); blank[list(book.get_blank_indexes())] = True
        # float32 normalization before candidate extraction.
        images = prep.get_processed_image_set(fov, z, corrector).astype(np.float32)
        sf = np.asarray(state['scales'], dtype=self.parameters['intensity_factor_dtype'])
        bg = np.asarray(state['backgrounds'], dtype=self.parameters['intensity_factor_dtype'])
        for b in range(len(channels)):
            images[b] = (images[b] - bg[b]) / sf[b]
        bounds = decode.compute_crop_bounds(self.dataSet, prep.parameters['warp_task'], fov, 0, True)
        r0, r1, c0, c1 = bounds; margin = int(self.parameters['support_margin'])
        mask = np.zeros(images.shape[1:], bool); mask[r0+margin:r1-margin, c0+margin:c1-margin] = True
        decoder = decoding.PixelBasedDecoder(book)
        code, mag, traces, dist = decoder.decode_pixels(
            images, scaleFactors=np.ones(len(channels)), backgrounds=np.zeros(len(channels)),
            lowPassSigma=0, distanceThreshold=self.parameters['distance_threshold'],
            magnitudeThreshold=self.parameters['magnitude_threshold'], distanceMetric=None,
            useGpu=False, neighborNumJobs=int(self.parameters['neighbor_jobs']),
            decodeMask=mask, accumulatePixelTraces=False)
        table = decoder.extract_barcodes_with_index(code, mag, None, dist, fov, cropWidth=0, zIndex=z, minimumArea=4)
        selected = bitregistration.select_molecules(table, blank, self.parameters['distance_threshold'], self.parameters['tile_size'])
        del images, decoder, code, mag, traces, dist, table
        gc.collect()
        result = dict(fov=fov, z_index=z, bit_names=list(book.get_bit_names()), colors=colors,
                      calibration_id=_state_id(state), paired_exposure_used=False,
                      neural_models_trained=False, molecules=len(selected))
        split = bitregistration.split_tiles(selected, self.parameters['tile_size'])
        if len(selected) < 600 or min(int(split.sum()), int((~split).sum())) < 300:
            result.update(accepted=False, status='insufficient_or_unreliable',
                          reason='Too few reliable molecules in disjoint spatial splits')
            self._save_result(fragmentIndex, result, selected,
                np.full((len(selected), len(channels), 2), np.nan), np.zeros((len(channels), 2)))
            return
        raw = np.asarray([prep.warpTask.get_aligned_image(fov, ch, z, corrector) for ch in channels])
        before = bitregistration.positions(raw, selected, codebook)
        del raw
        gc.collect()
        zero = np.zeros((len(channels), 2))
        offset = zero
        try:
            proposed, fit = bitregistration.fit(selected, codebook, before, self.parameters['tile_size'])
            result['fit'] = fit; result['proposed_offsets_xy'] = proposed.tolist()
            if fit['accepted']:
                # Exact existing two-pass sampler; C then W @ T, not a new
                # single-pass composition or a post-bit intensity refit.
                raw = []
                for b, ch in enumerate(channels):
                    image = self.dataSet.get_raw_image(ch, fov, self.dataSet.z_index_to_position(z))
                    image = corrector.transform_image(image, colors[b])
                    matrix = np.eye(3); matrix[:2, 2] = proposed[b]
                    pull = prep.warpTask.get_transformation(fov, ch).params @ matrix
                    raw.append(transform.warp(image, transform.AffineTransform(matrix=pull), preserve_range=True).astype(image.dtype))
                after = bitregistration.positions(np.asarray(raw), selected, codebook)
                del raw
                accepted, geometry = bitregistration.validate_applied(selected, codebook, before, after, colors, self.parameters['tile_size'])
                result.update(applied_validation=geometry, accepted=accepted,
                              validation_is_for_proposed_offsets=True,
                              status='accepted' if accepted else 'rejected_applied_geometry')
                self.dataSet.save_numpy_analysis_result(after, 'positions_after', self, resultIndex=fragmentIndex)
                if accepted:
                    offset = proposed
            else:
                result.update(accepted=False, status='rejected_predicted_geometry')
        except ValueError as e:
            result.update(accepted=False, status='insufficient_or_unreliable', reason=str(e))
        self._save_result(fragmentIndex, result, selected, before, offset)

    def _save_result(self, fragmentIndex, result, selected, before, offset):
        result['applied_offsets_xy'] = offset.tolist()
        result['alignment_id'] = _state_id(dict(global_calibration_id=result['calibration_id'],
            fov=result['fov'], z_index=result['z_index'], applied_offsets_xy=result['applied_offsets_xy']))
        self.dataSet.save_json_analysis_result(result, 'bit_alignment', self, resultIndex=fragmentIndex)
        self.dataSet.save_dataframe_to_csv(selected, 'fit_molecules', self, resultIndex=fragmentIndex, index=False)
        self.dataSet.save_numpy_analysis_result(before, 'positions_before', self, resultIndex=fragmentIndex)


class BitAlignedWarp(warp.Warp):
    """Apply a frozen global C and z-specific W @ T without changing old warps."""
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self.parameters.setdefault('apply_bit_alignment', True)
        self.parameters.setdefault('common_support_with_bit_alignment', False)
        self._base = dataSet.load_analysis_task(self.parameters['warp_task'])
        self._bits = dataSet.load_analysis_task(self.parameters['bit_alignment_task'])
        self._calibration = dataSet.load_analysis_task(self._bits.parameters['calibration_task'])
        self._channels = decode.codebook_data_channels(dataSet, self._calibration.get_codebook())

    def fragment_count(self):
        return self._bits.fragment_count()

    def get_dependencies(self):
        return [self.parameters['warp_task'], self.parameters['bit_alignment_task']]

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 1

    def _run_analysis(self, fragmentIndex):
        fov, z = self._bits.plane(fragmentIndex)
        _check_base_warp(self._base, fov, self._channels)
        result = self._bits.get_result(fov, z)
        applied = self.parameters['apply_bit_alignment']
        alignment_id = result['alignment_id'] if applied else _state_id(dict(
            global_calibration_id=result['calibration_id'], fov=fov, z_index=z,
            applied_offsets_xy=np.zeros((len(self._channels), 2)).tolist()))
        self.dataSet.save_json_analysis_result(dict(fov=fov, z_index=z,
            calibration_id=result['calibration_id'], alignment_id=alignment_id,
            bit_alignment_applied=applied, source_bit_alignment_status=result['status'],
            bit_alignment_status=result['status'] if applied else 'baseline_control',
            common_support_with_bit_alignment=self.parameters['common_support_with_bit_alignment']),
            'geometry', self, resultIndex=fragmentIndex)

    def get_transformation(self, fov, dataChannel=None, zIndex=None):
        if zIndex is None:
            zs = self._bits.parameters['z_indices']
            if len(zs) != 1:
                raise ValueError('BitAlignedWarp requires an explicit zIndex')
            zIndex = zs[0]
        if dataChannel is None:
            return [self.get_transformation(fov, int(ch), zIndex)
                    for ch in self.dataSet.get_data_organization().get_data_channels()]
        original = self._base.get_transformation(fov, dataChannel)
        if dataChannel not in self._channels or not getattr(self, 'parameters', {}).get('apply_bit_alignment', True):
            return original
        return self._with_bit_offset(fov, dataChannel, zIndex, original)

    def _with_bit_offset(self, fov, dataChannel, zIndex, original):
        if dataChannel not in self._channels:
            return original
        offset = self._bits.get_offsets(fov, zIndex)[self._channels.index(dataChannel)]
        matrix = np.eye(3); matrix[:2, 2] = offset
        return transform.AffineTransform(matrix=original.params @ matrix)

    def get_aligned_image(self, fov, dataChannel, zIndex, chromaticCorrector=None, hotPixels=None):
        # Non-codebook landmarks retain their original preprocessing contract.
        if dataChannel not in self._channels:
            return self._base.get_aligned_image(fov, dataChannel, zIndex, chromaticCorrector, hotPixels)
        frozen = self._calibration.get_previous_chromatic_corrector()
        if chromaticCorrector is not None and bitregistration.matrix_dict(chromaticCorrector) != bitregistration.matrix_dict(frozen):
            raise ValueError('Preprocessor requested a different chromatic calibration')
        image = self.dataSet.get_raw_image(dataChannel, fov, self.dataSet.z_index_to_position(zIndex))
        if hotPixels is not None:
            image = hotPixels.correct(image, dataChannel)
        color = self.dataSet.get_data_organization().get_data_channel_color(dataChannel)
        image = frozen.transform_image(image, color)
        return transform.warp(image, self.get_transformation(fov, dataChannel, zIndex),
                              preserve_range=True).astype(image.dtype)

    def get_crop_transformations(self, fov, dataChannels=None):
        """Bound both resampling passes, including the global chromatic map."""
        channels = (list(self.dataSet.get_data_organization().get_data_channels())
                    if dataChannels is None else dataChannels)
        corrector = self._calibration.get_previous_chromatic_corrector()
        output = []
        for z in self._bits.parameters['z_indices']:
            for ch in channels:
                if getattr(self, 'parameters', {}).get('common_support_with_bit_alignment', False):
                    base = self._base.get_transformation(fov, int(ch))
                    pulls = [base, self._with_bit_offset(fov, int(ch), int(z), base)]
                else:
                    pulls = [self.get_transformation(fov, int(ch), int(z))]
                color = self.dataSet.get_data_organization().get_data_channel_color(ch)
                c = (corrector.transformations[corrector.referenceColor].get(color)
                     if color != corrector.referenceColor else None)
                for w in pulls:
                    output.append(w)
                    if c is not None:
                        output.append(transform.AffineTransform(matrix=c.params @ w.params))
        return output


class PooledRawHistogram(filterbarcodes.GenerateAdaptiveThreshold):
    """Deterministic native histogram after every selected FOV has decoded.

    Empty fields contribute zero counts and cannot replace the intensity range
    with the generic NaN fallback. Existing histogram tasks are unchanged.
    """
    def get_dependencies(self):
        return [self.parameters['decode_task']]

    def get_estimated_time(self):
        return 10

    def _bins(self, rows, intensity, distance, area):
        # Keep the native numeric dtype/arithmetic, but own the buffer:
        # pandas copy-on-write may expose read-only .values arrays.
        data = rows[['mean_intensity', 'min_distance', 'area']].to_numpy(copy=True)
        data[:, 0] = filterbarcodes._transform_intensity_values(data[:, 0],
            filterbarcodes._get_intensity_transform_method(self.parameters))
        edges = [intensity, distance, area]
        return tuple(np.clip(np.digitize(data[:, k], e, right=(k != 2))-1, 0, len(e)-2)
                     for k, e in enumerate(edges))

    def _extract_counts(self, rows, intensity, distance, area):
        counts = np.zeros((len(intensity)-1, len(distance)-1, len(area)-1))
        if len(rows):
            np.add.at(counts, self._bins(rows, intensity, distance, area), 1)
        return counts

    def extract_barcodes_with_threshold(self, blankThreshold, barcodeSet):
        if barcodeSet.empty:
            return barcodeSet.copy()
        histogram = self.get_blank_fraction_histogram()
        bins = self._bins(barcodeSet, self.get_intensity_bins(), self.get_distance_bins(), self.get_area_bins())
        return barcodeSet[(histogram < blankThreshold)[bins]]

    def _run_analysis(self):
        task = self.dataSet.load_analysis_task(self.parameters['decode_task'])
        selected = self._selected_fragments()
        missing = [int(f) for f in selected if not task.is_complete(f)]
        if missing:
            raise RuntimeError('Pooled histogram needs all selected decode fields: %s' % missing)
        database, book = task.get_barcode_database(), task.get_codebook()
        maxima, empty = [], []
        for fov in selected:
            values = database.get_barcodes(fov=fov, columnList=['mean_intensity'])['mean_intensity'].to_numpy()
            if not len(values):
                empty.append(int(fov))
                continue
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError('Invalid candidate intensities in FOV %d' % fov)
            maxima.append(values.max())
        if not maxima:
            raise ValueError('No candidates in any selected FOV')
        maximum = np.max(np.asarray(maxima))
        intensity = filterbarcodes._build_intensity_bins(maximum, self.parameters['intensity_bins'],
            filterbarcodes._get_intensity_transform_method(self.parameters))
        distance = np.linspace(0, task.parameters['distance_threshold'] + .01, self.parameters['distance_bins'] + 1)
        area = np.arange(1, self.parameters['area_bins'] + 2)
        shape = (len(intensity)-1, len(distance)-1, len(area)-1)
        blank, coding = np.zeros(shape), np.zeros(shape)
        candidates = 0
        for fov in selected:
            rows = database.get_barcodes(fov=fov, columnList=['barcode_id', 'mean_intensity', 'min_distance', 'area'])
            flags = rows.barcode_id.isin(book.get_blank_indexes())
            codes = rows.barcode_id.isin(book.get_coding_indexes())
            if not (flags | codes).all():
                raise ValueError('Unknown barcode ID in pooled histogram')
            blank += self._extract_counts(rows.loc[flags], intensity, distance, area)
            coding += self._extract_counts(rows.loc[codes], intensity, distance, area)
            candidates += len(rows)
        if int(blank.sum()+coding.sum()) != candidates:
            raise RuntimeError('Pooled histogram candidate accounting differs')
        complete = np.zeros(self.fragment_count(), bool); complete[selected] = True
        for name, value in [('intensity_bins', intensity), ('distance_bins', distance), ('area_bins', area),
                            ('blank_counts', blank), ('coding_counts', coding), ('complete_fragments', complete)]:
            self.dataSet.save_numpy_analysis_result(value, name, self)
        self.dataSet.save_json_analysis_result(dict(fovs=list(map(int, selected)), empty_fovs=empty,
            max_intensity=float(maximum), candidates=candidates, blank_candidates=int(blank.sum()),
            coding_candidates=int(coding.sum()), deterministic_all_selected_fields=True), 'binning_provenance', self)
