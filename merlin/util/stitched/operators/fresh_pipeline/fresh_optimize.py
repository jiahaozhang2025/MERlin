"""Fresh intensity optimization for a registered, preprocessed virtual FOV.

Consumes mmap-friendly [bit,row,column] float32 planes plus acquired-support
masks. Calls the canonical MERlin decoder/refactor and OptimizeIteration
initialization/aggregation methods, without constructing a production dataset
or loading any earlier scales, backgrounds, histograms, or chromatic fits.
See FRESH_OPTIMIZE_INTERFACE.md for the manifest and CLI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

DEFAULT_FORK = Path(__file__).resolve().parents[2] / 'canonical'
CANONICAL_SOURCES = ('merlin/analysis/optimize.py', 'merlin/util/decoding.py',
                     'merlin/data/codebook.py')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _path(base, value):
    p = Path(value)
    return (p if p.is_absolute() else base / p).resolve()


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not data.get('bit_names') or not data.get('planes'):
        raise ValueError('Manifest needs bit_names and planes')
    if len(set(data['bit_names'])) != len(data['bit_names']):
        raise ValueError('Duplicate bit names')
    data['codebook_path'] = str(_path(path.parent, data['codebook_path']))
    if data.get('registration_manifest'):
        data['registration_manifest'] = str(_path(path.parent, data['registration_manifest']))
    seen = set()
    for plane in data['planes']:
        zi = int(plane['z_index'])
        if zi in seen:
            raise ValueError('Duplicate z_index in plane manifest')
        seen.add(zi)
        plane['z_index'] = zi
        if not np.isfinite(float(plane['physical_z_um'])):
            raise ValueError('Invalid physical Z')
        for name in ('images_path', 'mask_path'):
            plane[name] = str(_path(path.parent, plane[name]))
        if plane.get('role') not in (None, 'train', 'validation'):
            raise ValueError('Plane role must be train or validation')
    data['planes'].sort(key=lambda p: (float(p['physical_z_um']), p['z_index']))
    return data


def choose_planes(planes, max_training=50, max_validation=12):
    """Fixed, depth-spread sample; validation planes never enter calibration."""
    if max_training < 1 or max_validation < 1:
        raise ValueError('Plane sample limits must be positive')
    if any(p.get('role') for p in planes):
        if not all(p.get('role') for p in planes):
            raise ValueError('Specify roles for all planes or none')
        training = [p for p in planes if p['role'] == 'train']
        validation = [p for p in planes if p['role'] == 'validation']
        rule = 'Explicit manifest roles'
    else:
        # Hold out distributed depths, including a single middle plane for a
        # small manifest. No random resampling is hidden across iterations.
        count = min(max_validation, max(1, len(planes) // 5)) if len(planes) >= 4 else 0
        vi = set(np.linspace(1, len(planes) - 2, count, dtype=int).tolist()) if count else set()
        training = [p for i, p in enumerate(planes) if i not in vi]
        validation = [p for i, p in enumerate(planes) if i in vi]
        rule = 'Deterministic depth-spread whole-plane holdout; no holdout for fewer than four planes'
    def cap(items, count):
        if len(items) <= count:
            return items
        return [items[i] for i in np.linspace(0, len(items) - 1, count, dtype=int)]
    training, validation = cap(training, max_training), cap(validation, max_validation)
    if not training:
        raise ValueError('No training planes')
    return training, validation, dict(rule=rule,
        training_z_indices=[p['z_index'] for p in training],
        validation_z_indices=[p['z_index'] for p in validation],
        validation_independent_of_scale_background_fit=bool(validation),
        validation_is_independent_acquisition=False,
        caution='Held-out depths come from the same virtual FOV; nearby planes can contain the same molecule.')


def read_plane(plane, bit_count, mask_erosion=2):
    from scipy.ndimage import binary_erosion
    image = np.load(plane['images_path'], mmap_mode='r', allow_pickle=False)
    mask = np.load(plane['mask_path'], mmap_mode='r', allow_pickle=False)
    if image.ndim != 3 or image.shape[0] != bit_count or image.dtype != np.float32:
        raise ValueError(f'Expected float32 [B,H,W] in {plane["images_path"]}')
    if mask.dtype != bool or mask.shape != image.shape[1:]:
        raise ValueError(f'Expected boolean matching support in {plane["mask_path"]}')
    if mask_erosion < 0:
        raise ValueError('mask_erosion must be nonnegative')
    mask = np.asarray(mask, dtype=bool).copy()
    if mask_erosion:
        mask = binary_erosion(mask, structure=np.ones((3, 3), bool),
                              iterations=mask_erosion, border_value=0)
    # The canonical decoder normalizes all pixels before masking; NaN outside
    # support is therefore forbidden as well, although intensity outside the
    # mask never enters the histogram or the refactoring components.
    for bit in range(bit_count):
        if not np.isfinite(image[bit]).all() or np.any(image[bit] < 0):
            raise ValueError('Preprocessed image must be finite and nonnegative')
    return image, mask


def fresh_histogram(planes, bit_count, mask_erosion):
    # Exactly the canonical preprocessing histogram convention: uint16 cast,
    # bins np.arange(0,65535), and later nearest CDF0.9 index plus2.
    bins = np.arange(0, np.iinfo(np.uint16).max, 1)
    histogram = np.zeros((bit_count, len(bins) - 1), dtype=np.uint64)
    info = dict(valid_pixel_count=0, planes=[], values_above_uint16_max=[0] * bit_count,
                cast='uint16, matching canonical preprocessing histogram',
                bins='np.arange(0,65535,1); last edge65534',
                pixels_excluded_by_histogram_range=[0] * bit_count)
    for plane in planes:
        image, mask = read_plane(plane, bit_count, mask_erosion)
        n = int(mask.sum())
        info['valid_pixel_count'] += n
        info['planes'].append(dict(z_index=plane['z_index'], valid_pixels=n))
        for bit in range(bit_count):
            values = image[bit][mask]
            info['values_above_uint16_max'][bit] += int(np.count_nonzero(values > 65535))
            counts = np.histogram(values.astype(np.uint16), bins=bins)[0]
            histogram[bit] += counts.astype(np.uint64)
            info['pixels_excluded_by_histogram_range'][bit] += int(len(values) - counts.sum())
    if np.any(histogram.sum(axis=1) == 0):
        raise ValueError('No usable histogram pixels for one or more bits')
    return histogram, info


class MemoryResults:
    """Only the result-storage interface needed by canonical aggregation."""
    def __init__(self, initial=None):
        self.arrays = dict(initial or {})

    def load_numpy_analysis_result(self, name, task, resultIndex=None):
        key = (name, resultIndex)
        if key not in self.arrays:
            raise FileNotFoundError(key)
        return self.arrays[key]

    def save_numpy_analysis_result(self, array, name, task, resultIndex=None):
        self.arrays[name, resultIndex] = np.asarray(array).copy()


def canonical_initial_scales(OptimizeIteration, codebook, histogram):
    prep = SimpleNamespace(parameters={'save_pixel_histogram': True},
                           get_pixel_histogram=lambda: histogram)
    proxy = SimpleNamespace(parameters={'preprocess_task': 'FreshRegisteredHistogram'},
                            dataSet=SimpleNamespace(load_analysis_task=lambda _: prep),
                            get_codebook=lambda: codebook)
    return np.asarray(OptimizeIteration._calculate_initial_scale_factors(proxy), dtype=float)


def canonical_aggregate(OptimizeIteration, scales, backgrounds, scale_refactors, background_refactors):
    records = {}
    for i, (sr, br) in enumerate(zip(scale_refactors, background_refactors)):
        records['scale_refactors', i] = np.asarray(sr)
        records['background_refactors', i] = np.asarray(br)
        records['previous_scale_factors', i] = scales
        records['previous_backgrounds', i] = backgrounds
    # The scale-factor floor and the per-bit bias freeze did not exist in the
    # validated run; their off values keep get_scale_factors identical to it.
    proxy = SimpleNamespace(parameters={'fov_per_iteration': len(scale_refactors), 'normalize_scale_factors': False,
                                        'scale_factor_floor_ratio': 0, 'bias_freeze_reference': None},
                            analysisName='FreshVirtualFOV', dataSet=MemoryResults(records),
                            is_complete=lambda: True)
    # Current MERlin exposes optional scale normalization; the validated run
    # used absolute scales, so explicitly disable it and bind the real helpers
    # that get_scale_factors calls.
    from types import MethodType
    for name in ('_normalize_scale_factors', '_floor_scale_factors',
                 '_apply_bias_freeze', '_bias_reference'):
        if hasattr(OptimizeIteration, name):
            setattr(proxy, name, MethodType(getattr(OptimizeIteration, name), proxy))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        sf = OptimizeIteration.get_scale_factors(proxy)
        bg = OptimizeIteration.get_backgrounds(proxy)
    return np.asarray(sf), np.asarray(bg), [str(w.message) for w in seen]


def component_counts(decoded, barcode_count, minimum_area):
    """Count the same labeled components used by canonical extract_refactors."""
    from skimage.measure import label
    labels = label(decoded + 1)
    counts = np.bincount(labels.ravel())
    good = counts >= minimum_area
    good[0] = False
    # A label contains one barcode ID. Find its first raster occurrence.
    labels_seen, positions = np.unique(labels, return_index=True)
    take = good[labels_seen]
    bids = decoded.ravel()[positions[take]]
    return np.bincount(bids, minlength=barcode_count).astype(np.int64)


def evaluate_plane(plane, codebook, Decoder, scales, backgrounds, parameters, refactor=True):
    started = time.time()
    image, mask = read_plane(plane, codebook.get_bit_count(), parameters['mask_erosion_px'])
    read_seconds = time.time() - started
    decoder = Decoder(codebook)
    decoder.refactorAreaThreshold = parameters['area_threshold']
    decoder.barcodesSeenThreshold = parameters['min_barcodes_for_refactoring']
    di, pm, npt, distance = decoder.decode_pixels(image, scales, backgrounds,
        lowPassSigma=0, decodeMask=mask, tilingOverlap=20,
        distanceThreshold=parameters['distance_threshold'],
        magnitudeThreshold=1., distanceMetric='euclidean',
        neighborNumJobs=parameters['num_threads'], useGpu=False, tilingFactor=None)
    if np.any(di[~mask] >= 0):
        raise AssertionError('Decoded pixels outside acquired support')
    count_started = time.time()
    counts = component_counts(di, codebook.get_barcode_count(), parameters['area_threshold'])
    count_seconds = time.time() - count_started
    warnings_seen = []
    refactor_started = time.time()
    if refactor and counts.sum():
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            sf, bg, canonical_counts = decoder.extract_refactors(di, pm, npt,
                                                    extractBackgrounds=True)
        if not np.array_equal(counts, canonical_counts):
            raise AssertionError('Component-count evidence differs from canonical refactoring')
        warnings_seen = [str(w.message) for w in seen]
    else:
        sf = bg = np.full(codebook.get_bit_count(), np.nan)
    refactor_seconds = time.time() - refactor_started
    bars = np.asarray(codebook.get_barcodes(), dtype=bool)
    qualifying = counts > parameters['min_barcodes_for_refactoring']
    on_evidence = (counts[:, None] * qualifying[:, None] * bars).sum(axis=0)
    off_evidence = (counts[:, None] * qualifying[:, None] * ~bars).sum(axis=0)
    blank_ids = np.asarray(codebook.get_blank_indexes(), dtype=int)
    blank = int(counts[blank_ids].sum())
    valid = di >= 0
    metrics = dict(z_index=plane['z_index'], physical_z_um=float(plane['physical_z_um']),
                   acquired_pixels=int(mask.sum()), decoded_pixels=int(valid.sum()),
                   components=int(counts.sum()), blank_components=blank,
                   blank_fraction=blank / int(counts.sum()) if counts.sum() else None,
                   mean_decoded_pixel_distance=float(np.mean(distance[valid])) if valid.any() else None,
                   qualifying_on_components_per_bit=on_evidence.tolist(),
                   qualifying_off_components_per_bit=off_evidence.tolist(),
                   finite_scale_refactor_bits=np.isfinite(sf).tolist(),
                   finite_background_refactor_bits=np.isfinite(bg).tolist(),
                   warnings=warnings_seen, elapsed_seconds=time.time()-started,
                   read_validate_seconds=read_seconds,
                   canonical_decode_timings=dict(decoder.last_decode_timings),
                   independent_component_count_seconds=count_seconds,
                   canonical_refactor_seconds=refactor_seconds)
    return np.asarray(sf), np.asarray(bg), counts, metrics


def configure_plane_threads(num_threads):
    """Bound native libraries as well as sklearn's explicit neighbor threads."""
    if int(num_threads)<1:raise ValueError('num_threads must be positive')
    os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',
        OMP_NUM_THREADS=str(int(num_threads)),NUMEXPR_NUM_THREADS='1')
    from threadpoolctl import threadpool_limits,threadpool_info
    # Keep controllers alive for the worker lifetime. Canonical n_jobs uses
    # the requested thread count; BLAS is one to avoid nested multiplication.
    controllers=[threadpool_limits(limits=1,user_api='blas'),
                 threadpool_limits(limits=int(num_threads),user_api='openmp')]
    import cv2
    cv2.setNumThreads(1)
    return controllers,threadpool_info()


def peak_rss_mib():
    try:
        import resource
        value=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value/(1024*1024 if sys.platform=='darwin' else 1024)
    except ImportError:
        return None


_PLANE_WORKER={}


def _initialize_plane_worker(fork,codebook_path,output,num_threads,canonical_loader):
    global _PLANE_WORKER
    root=Path(output).resolve()/'workers'/str(os.getpid())
    for name in ('tmp','cache'):(root/name).mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(root/'tmp'),TEMP=str(root/'tmp'),TMP=str(root/'tmp'),
        XDG_CACHE_HOME=str(root/'cache'),MPLCONFIGDIR=str(root/'cache/matplotlib'),
        NUMBA_CACHE_DIR=str(root/'cache/numba'),PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    _install_write_guard(root)
    loader=load_canonical if canonical_loader is None else canonical_loader
    Codebook,sink,_,Decoder=loader(Path(fork),root)
    book=Codebook(sink,str(codebook_path),codebookIndex=0,codebookName='FreshWorkerCodebook')
    controllers,thread_info=configure_plane_threads(num_threads)
    _PLANE_WORKER=dict(codebook=book,decoder=Decoder,controllers=controllers,thread_info=thread_info)


def _evaluate_plane_worker(payload):
    plane,scales,backgrounds,parameters,refactor=payload
    sf,bg,counts,metric=evaluate_plane(plane,_PLANE_WORKER['codebook'],_PLANE_WORKER['decoder'],
        scales,backgrounds,parameters,refactor)
    metric.update(worker_pid=os.getpid(),peak_rss_mib=peak_rss_mib(),
                  native_thread_pools=_PLANE_WORKER['thread_info'])
    return sf,bg,counts,metric


class PlaneBatchExecutor:
    """Ordered, whole-plane map; parent alone aggregates across barriers."""
    def __init__(self,workers,fork,codebook_path,output,num_threads,codebook,Decoder,canonical_loader=None):
        if int(workers)!=workers or workers<1:raise ValueError('workers must be a positive integer')
        self.workers=int(workers);self.codebook=codebook;self.Decoder=Decoder
        self.initargs=(str(Path(fork).resolve()),str(Path(codebook_path).resolve()),
                       str(Path(output).resolve()),int(num_threads),canonical_loader)
        self.pool=None
    def __enter__(self):
        names=('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','OMP_NUM_THREADS',
               'NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')
        self.previous_environment={name:os.environ.get(name) for name in names}
        self.controllers,self.thread_info=configure_plane_threads(self.initargs[3])
        if self.workers>1:
            # Spawn avoids inheriting already initialized threaded BLAS state.
            os.environ['PYTHONDONTWRITEBYTECODE']='1'
            self.pool=ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize_plane_worker,initargs=self.initargs)
        return self
    def evaluate(self,planes,scales,backgrounds,parameters,refactor=True):
        if self.pool is None:
            def local():
                for p in planes:
                    sf,bg,counts,metric=evaluate_plane(p,self.codebook,self.Decoder,scales,backgrounds,parameters,refactor)
                    metric.update(worker_pid=os.getpid(),peak_rss_mib=peak_rss_mib(),native_thread_pools=self.thread_info)
                    yield sf,bg,counts,metric
            return local()
        # Only paths and tiny vectors cross the process boundary. map retains
        # manifest order, and the caller consumes every result before updates.
        payloads=[(p,scales.copy(),backgrounds.copy(),parameters,refactor) for p in planes]
        return self.pool.map(_evaluate_plane_worker,payloads,chunksize=1)
    def __exit__(self,exc_type,exc_value,traceback):
        try:
            if self.pool is not None:self.pool.shutdown(wait=True,cancel_futures=exc_type is not None)
        finally:
            for controller in reversed(self.controllers):controller.restore_original_limits()
            for name,value in self.previous_environment.items():
                if value is None:os.environ.pop(name,None)
                else:os.environ[name]=value


def _install_write_guard(root):
    root = root.resolve()
    import tempfile
    tempfile.tempdir = str(root / 'tmp')
    def allowed(path):
        if isinstance(path, int) or path is None:
            return True
        p = Path(os.fsdecode(path)).resolve()
        return p == Path(os.devnull).resolve() or p == root or root in p.parents
    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            write = (mode is not None and any(c in mode for c in 'wax+')) or bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if write and not allowed(path):
                raise PermissionError(f'Fresh optimizer cannot write outside output root: {path}')
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'):
            if event == 'os.mkdir' and Path(os.fsdecode(args[0])).is_dir():
                return
            if not allowed(args[0]):
                raise PermissionError(f'Fresh optimizer cannot mutate {args[0]}')
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            if any(not allowed(p) for p in args[:2]):
                raise PermissionError('Fresh optimizer cannot mutate paths outside output root')
    sys.addaudithook(audit)


def load_canonical(fork, output):
    sys.path.insert(0, str(fork))
    import merlin
    from merlin.data.codebook import Codebook
    from merlin.analysis.optimize import OptimizeIteration
    from merlin.util.decoding import PixelBasedDecoder
    if fork not in Path(merlin.__file__).resolve().parents:
        raise RuntimeError('Imported MERlin from an unexpected location')
    class CodebookSink:
        def save_codebook(self, codebook):
            codebook.get_data().to_csv(output / 'fresh_codebook.csv', index=False)
    return Codebook, CodebookSink(), OptimizeIteration, PixelBasedDecoder


def _run_optimization_body(manifest, output, codebook, OptimizeIteration, Decoder, plane_executor, *, iterations=10,
                     max_training=50, max_validation=12, num_threads=1,
                     mask_erosion_px=2, scale_tolerance=.01, background_tolerance=.005):
    """Programmatic runner; output must be an existing isolated directory."""
    output = Path(output)
    bit_names = list(codebook.get_bit_names())
    if bit_names != list(manifest['bit_names']):
        raise ValueError('Array bit order does not exactly match canonical codebook order')
    if iterations < 1:
        raise ValueError('iterations must be positive')
    train, validation, sampling = choose_planes(manifest['planes'], max_training, max_validation)
    parameters = dict(iterations=int(iterations), distance_threshold=.52, area_threshold=4,
        min_barcodes_for_refactoring=1, optimize_background=True, optimize_chromatic_correction=False,
        num_threads=int(num_threads), plane_workers=plane_executor.workers,
        process_start_method='spawn' if plane_executor.workers>1 else 'sequential',
        ordered_plane_results=True,aggregation_barrier_after_all_training_planes=True,
        mask_erosion_px=int(mask_erosion_px),
        scale_relative_change_tolerance=float(scale_tolerance),
        background_change_divided_by_previous_scale_tolerance=float(background_tolerance),
        convergence_requires_two_successive_stable_iterations=True,
        stop_rule='Fixed iteration budget; convergence is measured, never assumed')
    report = dict(status='initializing', fresh=True, reused_fitted_calibration=False,
        bit_names=bit_names, sampling=sampling, parameters=parameters, iterations=[],
        departures_from_full_dataset_canonical_task=[
            'One virtual output FOV, not 50 random physical FOV/depth samples per iteration.',
            'Fixed depth-spread training planes reused across iterations, with separate depth holdout when available.',
            'Histograms and refactors use complete-acquisition support with a small mask-edge erosion; invalid image fill excluded.',
            'No previous scales, backgrounds, histograms, chromatic calibration, or optimization task state loaded.',
            'Chromatic/XYZ geometry is supplied by the fresh registration stage; this runner fits intensity only.',
            'Convergence thresholds are pilot diagnostics added to the canonical fixed10-iteration procedure.',
            'Raw blank fraction is a diagnostic; it is not a calibrated misidentification rate or adaptive filter.'])
    histogram, histogram_info = fresh_histogram(train, len(bit_names), mask_erosion_px)
    np.save(output / 'fresh_pixel_histograms.npy', histogram)
    scales = canonical_initial_scales(OptimizeIteration, codebook, histogram)
    backgrounds = np.zeros(len(bit_names), dtype=float)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Fresh canonical histogram initialization produced invalid scales')
    report.update(histogram=histogram_info, initial_scale_factors=scales.tolist(),
                  initial_backgrounds=backgrounds.tolist(), status='optimizing')
    scale_history, background_history = [scales.copy()], [backgrounds.copy()]
    stable_streak = 0
    for iteration in range(1, iterations + 1):
        sf_parts, bg_parts, counts_parts, metrics = [], [], [], []
        evaluations=plane_executor.evaluate(train,scales,backgrounds,parameters)
        for plane,(sf,bg,counts,metric) in zip(train,evaluations):
            sf_parts.append(sf); bg_parts.append(bg); counts_parts.append(counts); metrics.append(metric)
            print(f'Fresh optimize{iteration}/{iterations} Z{plane["z_index"]}: '
                  f'{metric["components"]} components, {sum(metric["finite_scale_refactor_bits"])} usable bit refactors', flush=True)
        new_scales, new_backgrounds, notices = canonical_aggregate(OptimizeIteration, scales,
                                                        backgrounds, sf_parts, bg_parts)
        iteration_path = output / f'iteration_{iteration:02d}'
        iteration_path.mkdir(exist_ok=False)
        np.savez_compressed(iteration_path / 'refactors.npz', scale_refactors=sf_parts,
                            background_refactors=bg_parts, barcode_counts=counts_parts,
                            previous_scale_factors=scales, previous_backgrounds=backgrounds)
        valid = np.isfinite(new_scales) & (new_scales > 0) & np.isfinite(new_backgrounds)
        if not valid.all():
            report.update(status='insufficient_calibration_evidence',
                          invalid_bits=[bit_names[i] for i in np.flatnonzero(~valid)],
                          failure_iteration=iteration, failure_plane_metrics=metrics,
                          aggregation_warnings=notices)
            write_json(output / 'optimization.json', report)
            raise RuntimeError('Fresh calibration lacks finite scale/background evidence; no old calibration substituted')
        scale_change = np.abs(new_scales - scales) / scales
        bg_change = np.abs(new_backgrounds - backgrounds) / scales
        stable = bool(np.max(scale_change) <= scale_tolerance and np.max(bg_change) <= background_tolerance)
        stable_streak = stable_streak + 1 if stable else 0
        entry = dict(iteration=iteration, scales=new_scales.tolist(), backgrounds=new_backgrounds.tolist(),
            maximum_relative_scale_change=float(scale_change.max()),
            maximum_background_change_in_previous_scale_units=float(bg_change.max()),
            stable=stable, successive_stable_iterations=stable_streak,
            plane_metrics=metrics, aggregation_warnings=notices)
        scales, backgrounds = new_scales, new_backgrounds
        scale_history.append(scales.copy()); background_history.append(backgrounds.copy())
        report['iterations'].append(entry)
        np.save(iteration_path / 'scale_factors.npy', scales)
        np.save(iteration_path / 'backgrounds.npy', backgrounds)
        write_json(output / 'optimization.json', report)
    validation_metrics = []
    for _,_,_,metric in plane_executor.evaluate(validation,scales,backgrounds,parameters,refactor=False):
        validation_metrics.append(metric)
    report.update(status='converged' if stable_streak >= 2 else 'completed_not_converged',
                  convergence_passed=stable_streak >= 2, validation_metrics=validation_metrics,
                  final_scale_factors=scales.tolist(), final_backgrounds=backgrounds.tolist())
    np.save(output / 'scale_factors.npy', scales)
    np.save(output / 'backgrounds.npy', backgrounds)
    np.save(output / 'scale_factor_history.npy', scale_history)
    np.save(output / 'background_history.npy', background_history)
    write_json(output / 'optimization.json', report)
    return report


def run_optimization(manifest, output, codebook, OptimizeIteration, Decoder, *, iterations=10,
                     max_training=50,max_validation=12,num_threads=1,mask_erosion_px=2,
                     scale_tolerance=.01,background_tolerance=.005,workers=1,
                     worker_fork=DEFAULT_FORK,worker_canonical_loader=None):
    """Fresh fit with an ordered per-iteration whole-plane process option.

    worker_canonical_loader is an import-only test hook; the CLI always loads
    the actual canonical fork in each child. Numerical operators are unchanged.
    """
    if num_threads<1:raise ValueError('num_threads must be positive')
    with PlaneBatchExecutor(workers,worker_fork,manifest['codebook_path'],output,num_threads,
                            codebook,Decoder,worker_canonical_loader) as executor:
        return _run_optimization_body(manifest,output,codebook,OptimizeIteration,Decoder,executor,
            iterations=iterations,max_training=max_training,max_validation=max_validation,
            num_threads=num_threads,mask_erosion_px=mask_erosion_px,
            scale_tolerance=scale_tolerance,background_tolerance=background_tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--fork', type=Path, default=DEFAULT_FORK)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--max-training-planes', type=int, default=50)
    parser.add_argument('--max-validation-planes', type=int, default=12)
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--workers',type=int,default=1,
                        help='Independent whole-plane processes within each iteration; recommended4 with2 threads each')
    parser.add_argument('--mask-erosion-px', type=int, default=2)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    output, fork = args.output_root.resolve(), args.fork.resolve()
    inputs = [args.manifest.resolve(), Path(manifest['codebook_path'])]
    inputs += [Path(p[k]) for p in manifest['planes'] for k in ('images_path', 'mask_path')]
    if manifest.get('registration_manifest'):
        inputs.append(Path(manifest['registration_manifest']))
    inputs += [fork / name for name in CANONICAL_SOURCES]
    if any(output == p or output in p.parents for p in inputs):
        raise ValueError('Output root contains protected input files')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('tmp', 'cache'):
        (output / name).mkdir()
    os.environ.update(TMPDIR=str(output/'tmp'), TEMP=str(output/'tmp'), TMP=str(output/'tmp'),
                      XDG_CACHE_HOME=str(output/'cache'), MPLCONFIGDIR=str(output/'cache/matplotlib'),
                      NUMBA_CACHE_DIR=str(output/'cache/numba'), PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    _install_write_guard(output)
    started = time.time()
    fingerprints = {str(p): sha256(p) for p in inputs}
    write_json(output/'inputs.json', dict(manifest=manifest, input_sha256=fingerprints,
                                        runner_sha256=sha256(__file__)))
    Codebook, sink, Optimizer, Decoder = load_canonical(fork, output)
    codebook = Codebook(sink, manifest['codebook_path'], codebookIndex=0,
                        codebookName='FreshVirtualFOV')
    report = run_optimization(manifest, output, codebook, Optimizer, Decoder,
        iterations=args.iterations, max_training=args.max_training_planes,
        max_validation=args.max_validation_planes, num_threads=args.num_threads,
        mask_erosion_px=args.mask_erosion_px,workers=args.workers,worker_fork=fork)
    changed = [path for path, digest in fingerprints.items() if sha256(path) != digest]
    if changed:
        raise RuntimeError(f'Inputs changed during fresh optimization: {changed}')
    report.update(all_hashed_inputs_unchanged=True, elapsed_seconds=time.time()-started)
    write_json(output/'optimization.json', report)
    print(json.dumps(dict(status=report['status'], output=str(output),
                          elapsed_seconds=report['elapsed_seconds'])), flush=True)
    if not report['convergence_passed']:
        raise SystemExit(2)  # Prevent an afterok decode dependency treating this as converged.


if __name__ == '__main__':
    main()
