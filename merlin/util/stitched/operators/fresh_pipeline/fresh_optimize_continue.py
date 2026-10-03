"""Continue one verified fresh fit by replaying its saved evaluation prefix.

The canonical iteration body is unchanged. Previous image evaluations are not
repeated; their checked refactors are replayed from the same fresh fit. A new
histogram and identical initial state are reconstructed from unchanged images.
No production calibration is loaded. All output goes to a new directory.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import fresh_optimize as fresh


def exact(actual, expected, label):
    if not np.array_equal(np.asarray(actual), np.asarray(expected), equal_nan=True):
        raise ValueError('Checkpoint mismatch: '+label)


def registration_inputs(manifest):
    """Bind the unchanged accepted manifest to its actual models/metadata."""
    if not manifest.get('registration_manifest'): return {},{}
    path=Path(manifest['registration_manifest']).resolve()
    registration=json.loads(path.read_text())
    if (registration.get('status')!='ACCEPTED_FOR_SINGLE_FOV_PILOT'
            or registration.get('bit_names')!=manifest['bit_names']):
        raise ValueError('Registration acceptance/bit order changed')
    hashes={}
    for name,expected in registration['required_input_sha256'].items():
        source=Path(name);source=(source if source.is_absolute() else path.parent/source).resolve()
        hashes[str(source)]=expected
    for key in ('within_round_decision','cross_round_decision'):
        source=Path(registration[key]);source=(source if source.is_absolute() else path.parent/source).resolve()
        hashes[str(source)]=registration[key+'_sha256']
    for name,expected in hashes.items():
        if fresh.sha256(name)!=expected: raise ValueError('Accepted registration input changed: '+name)
    stats=registration['required_raw_file_stats']
    for name,expected in stats.items():
        current=Path(name).stat()
        if current.st_size!=expected['size'] or current.st_mtime_ns!=expected['mtime_ns']:
            raise ValueError('Accepted raw-file provenance changed: '+name)
    return hashes,stats


def verify_checkpoint(root, manifest, Optimizer, book, *, manifest_path=None,
                      canonical_source_paths=()):
    """Hash inputs and cross-check every saved iteration before any new fit."""
    root = Path(root).resolve()
    report = json.loads((root/'optimization.json').read_text())
    inputs = json.loads((root/'inputs.json').read_text())
    if (report.get('status') != 'completed_not_converged'
            or report.get('convergence_passed') is not False
            or report.get('fresh') is not True
            or report.get('reused_fitted_calibration') is not False
            or report.get('all_hashed_inputs_unchanged') is not True):
        raise ValueError('Only a finalized, unconverged fresh fit may be continued')
    if inputs['manifest'] != manifest or report['bit_names'] != manifest['bit_names']:
        raise ValueError('Checkpoint manifest or bit order differs')
    if list(book.get_bit_names()) != manifest['bit_names']:
        raise ValueError('Current codebook bit order differs')
    if inputs['runner_sha256'] != fresh.sha256(fresh.__file__):
        raise ValueError('Original numerical runner changed')
    required = [Path(manifest['codebook_path'])]
    required += [Path(p[k]) for p in manifest['planes'] for k in ('images_path','mask_path')]
    required += [Path(p) for p in canonical_source_paths]
    if manifest_path is not None: required.append(Path(manifest_path))
    if manifest.get('registration_manifest'): required.append(Path(manifest['registration_manifest']))
    hashes = inputs['input_sha256']
    if any(str(p.resolve()) not in hashes for p in required):
        raise ValueError('Checkpoint input fingerprint inventory is incomplete')
    for path, expected in hashes.items():
        if fresh.sha256(path) != expected: raise ValueError('Original input changed: '+path)
    registration_hashes,raw_stats=registration_inputs(manifest)
    entries = report['iterations']; n = len(entries); bits = len(report['bit_names'])
    if n < 1 or report['parameters']['iterations'] != n:
        raise ValueError('Checkpoint iteration budget is not complete')
    training_ids = report['sampling']['training_z_indices']
    validation_ids = report['sampling']['validation_z_indices']
    if not training_ids or not validation_ids or set(training_ids)&set(validation_ids):
        raise ValueError('Checkpoint has invalid training/heldout split')
    train, validation, sampling = fresh.choose_planes(manifest['planes'],len(training_ids),len(validation_ids))
    if sampling != report['sampling']: raise ValueError('Checkpoint depth sampling differs')
    paths = [root/name for name in ('optimization.json','inputs.json','fresh_pixel_histograms.npy',
             'scale_factors.npy','backgrounds.npy','scale_factor_history.npy','background_history.npy')]
    paths += [root/f'iteration_{i:02d}'/name for i in range(1,n+1)
              for name in ('refactors.npz','scale_factors.npy','backgrounds.npy')]
    checkpoint_hashes = {str(p):fresh.sha256(p) for p in paths}
    histogram = np.load(root/'fresh_pixel_histograms.npy',allow_pickle=False)
    if histogram.shape != (bits,65534) or histogram.dtype != np.uint64:
        raise ValueError('Unexpected fresh histogram shape/dtype')
    if not np.all(histogram.sum(axis=1)==report['histogram']['valid_pixel_count']):
        raise ValueError('Fresh histogram loses pixel counts')
    if (any(report['histogram']['values_above_uint16_max'])
            or any(report['histogram']['pixels_excluded_by_histogram_range'])):
        raise ValueError('Checkpoint histogram needs separate range review')
    initial = fresh.canonical_initial_scales(Optimizer,book,histogram)
    exact(initial,report['initial_scale_factors'],'fresh histogram initialization')
    exact(report['initial_backgrounds'],np.zeros(bits),'initial backgrounds')
    sh = np.load(root/'scale_factor_history.npy',allow_pickle=False)
    bh = np.load(root/'background_history.npy',allow_pickle=False)
    if sh.shape != (n+1,bits) or bh.shape != sh.shape: raise ValueError('History shape mismatch')
    exact(sh[0],initial,'scale history initialization'); exact(bh[0],np.zeros(bits),'background history initialization')
    saved = []; streak = 0
    for i, entry in enumerate(entries,1):
        if entry['iteration'] != i: raise ValueError('Noncontiguous checkpoint iterations')
        metrics = entry['plane_metrics']
        if [p['z_index'] for p in metrics] != training_ids: raise ValueError('Changed plane order')
        with np.load(root/f'iteration_{i:02d}/refactors.npz',allow_pickle=False) as source:
            arrays = {key:source[key].copy() for key in source.files}
        sr, br, counts = arrays['scale_refactors'],arrays['background_refactors'],arrays['barcode_counts']
        if sr.shape != (len(train),bits) or br.shape != sr.shape or counts.shape != (len(train),book.get_barcode_count()):
            raise ValueError('Refactor/count shape mismatch')
        if not np.isfinite(counts).all() or np.any(counts<0) or np.any(counts!=np.floor(counts)):
            raise ValueError('Invalid saved barcode counts')
        exact(arrays['previous_scale_factors'],sh[i-1],'previous scales')
        exact(arrays['previous_backgrounds'],bh[i-1],'previous backgrounds')
        sf,bg,notices = fresh.canonical_aggregate(Optimizer,sh[i-1],bh[i-1],sr,br)
        for actual,expected,label in ((sf,entry['scales'],'report scales'),(bg,entry['backgrounds'],'report backgrounds'),
                (sf,sh[i],'scale history'),(bg,bh[i],'background history'),
                (sf,np.load(root/f'iteration_{i:02d}/scale_factors.npy',allow_pickle=False),'iteration scales'),
                (bg,np.load(root/f'iteration_{i:02d}/backgrounds.npy',allow_pickle=False),'iteration backgrounds')):
            exact(actual,expected,label)
        if not np.isfinite(sf).all() or np.any(sf<=0) or not np.isfinite(bg).all():
            raise ValueError('Nonfinite fitted checkpoint state')
        change = float(np.max(np.abs(sf-sh[i-1])/sh[i-1]))
        bg_change = float(np.max(np.abs(bg-bh[i-1])/sh[i-1]))
        stable = change<=report['parameters']['scale_relative_change_tolerance'] and bg_change<=report['parameters']['background_change_divided_by_previous_scale_tolerance']
        streak = streak+1 if stable else 0
        if (change!=entry['maximum_relative_scale_change'] or bg_change!=entry['maximum_background_change_in_previous_scale_units']
                or stable!=entry['stable'] or streak!=entry['successive_stable_iterations']
                or notices!=entry['aggregation_warnings']):
            raise ValueError('Checkpoint convergence/aggregation diagnostics differ')
        blank_ids = list(book.get_blank_indexes())
        bars = np.asarray(book.get_barcodes(),bool)
        for row, metric in enumerate(metrics):
            if metric['physical_z_um'] != float(train[row]['physical_z_um']): raise ValueError('Changed physical depth')
            exact(np.isfinite(sr[row]),metric['finite_scale_refactor_bits'],'finite scale evidence')
            exact(np.isfinite(br[row]),metric['finite_background_refactor_bits'],'finite background evidence')
            if int(counts[row].sum())!=metric['components'] or int(counts[row,blank_ids].sum())!=metric['blank_components']:
                raise ValueError('Saved counts disagree with report')
            qualifying = counts[row]>report['parameters']['min_barcodes_for_refactoring']
            exact((counts[row,:,None]*qualifying[:,None]*bars).sum(0),metric['qualifying_on_components_per_bit'],'on evidence')
            exact((counts[row,:,None]*qualifying[:,None]*~bars).sum(0),metric['qualifying_off_components_per_bit'],'off evidence')
        saved.append((arrays,copy.deepcopy(metrics)))
    if streak>=2: raise ValueError('Checkpoint is already stable; do not continue automatically')
    exact(sh[-1],report['final_scale_factors'],'final report scales')
    exact(bh[-1],report['final_backgrounds'],'final report backgrounds')
    exact(sh[-1],np.load(root/'scale_factors.npy',allow_pickle=False),'final scale array')
    exact(bh[-1],np.load(root/'backgrounds.npy',allow_pickle=False),'final background array')
    if [m['z_index'] for m in report['validation_metrics']] != validation_ids:
        raise ValueError('Original final holdout is incomplete')
    if any(fresh.sha256(p)!=h for p,h in checkpoint_hashes.items()): raise ValueError('Checkpoint changed during verification')
    return dict(root=root,report=report,inputs=inputs,histogram=histogram,saved=saved,
                checkpoint_sha256=checkpoint_hashes,training=train,validation=validation,
                registration_source_sha256=registration_hashes,registration_raw_file_stats=raw_stats)


class PrefixReplay:
    def __init__(self, checkpoint, live, output):
        self.checkpoint,self.live,self.output = checkpoint,live,Path(output)
        self.workers = live.workers
        self.calls = 0; self.replayed_plane_evaluations = 0; self.new_plane_evaluations = 0

    def evaluate(self, planes, scales, backgrounds, parameters, refactor=True):
        report = self.checkpoint['report']; prefix = len(self.checkpoint['saved'])
        expected_ids = report['sampling']['training_z_indices' if refactor else 'validation_z_indices']
        if [p['z_index'] for p in planes] != expected_ids: raise ValueError('Replay/live plane identity differs')
        for key,value in report['parameters'].items():
            if key not in ('iterations','plane_workers','process_start_method') and parameters.get(key)!=value:
                raise ValueError('Numerical continuation parameter changed: '+key)
        if refactor:
            self.calls += 1
            if self.calls==1:
                exact(np.load(self.output/'fresh_pixel_histograms.npy',allow_pickle=False),self.checkpoint['histogram'],'recomputed fresh histogram')
                exact(scales,report['initial_scale_factors'],'recomputed initial scales')
                exact(backgrounds,report['initial_backgrounds'],'recomputed initial backgrounds')
            if self.calls<=prefix:
                arrays,metrics = self.checkpoint['saved'][self.calls-1]
                exact(scales,arrays['previous_scale_factors'],'replay incoming scales')
                exact(backgrounds,arrays['previous_backgrounds'],'replay incoming backgrounds')
                self.replayed_plane_evaluations += len(planes)
                return iter([(arrays['scale_refactors'][j].copy(),arrays['background_refactors'][j].copy(),
                              arrays['barcode_counts'][j].copy(),copy.deepcopy(metrics[j])) for j in range(len(planes))])
            if self.calls==prefix+1:
                exact(scales,report['final_scale_factors'],'continuation starting scales')
                exact(backgrounds,report['final_backgrounds'],'continuation starting backgrounds')
        elif self.calls<=prefix:
            raise ValueError('Holdout requested before new training evaluations')
        self.new_plane_evaluations += len(planes)
        return self.live.evaluate(planes,scales,backgrounds,parameters,refactor=refactor)


def run_continuation(checkpoint, manifest, output, book, Optimizer, Decoder, *,
                     additional_iterations=10, workers=None, worker_fork=fresh.DEFAULT_FORK,
                     worker_canonical_loader=None):
    if additional_iterations<1: raise ValueError('Additional fixed budget must be positive')
    old = checkpoint['report']; params = old['parameters']; prefix = len(old['iterations'])
    workers = params['plane_workers'] if workers is None else workers
    with fresh.PlaneBatchExecutor(workers,worker_fork,manifest['codebook_path'],output,
            params['num_threads'],book,Decoder,worker_canonical_loader) as live:
        replay = PrefixReplay(checkpoint,live,output)
        result = fresh._run_optimization_body(manifest,output,book,Optimizer,Decoder,replay,
            iterations=prefix+additional_iterations,max_training=len(checkpoint['training']),
            max_validation=len(checkpoint['validation']),num_threads=params['num_threads'],
            mask_erosion_px=params['mask_erosion_px'],scale_tolerance=params['scale_relative_change_tolerance'],
            background_tolerance=params['background_change_divided_by_previous_scale_tolerance'])
    for i in range(prefix):
        original={k:v for k,v in old['iterations'][i].items() if k!='evaluation_origin'}
        if result['iterations'][i] != original: raise ValueError('Replayed report prefix changed')
    if result['sampling'] != old['sampling'] or result['histogram'] != old['histogram']:
        raise ValueError('Replayed sampling/histogram changed')
    result['continuation'] = dict(same_fresh_fit=True,production_calibration_reused=False,
        source_checkpoint_root=str(checkpoint['root']),source_checkpoint_sha256=checkpoint['checkpoint_sha256'],
        prefix_iterations_replayed=prefix,additional_fixed_iteration_budget=additional_iterations,
        replayed_plane_evaluations=replay.replayed_plane_evaluations,
        new_plane_evaluations_including_final_holdout=replay.new_plane_evaluations,
        histogram_recomputed_and_identical=True,original_evaluation_timings_are_historical=True,
        original_execution_parameters=old['parameters'],
        continuation_execution_parameters=dict(plane_workers=workers,num_threads=params['num_threads'],
                                                process_start_method='spawn' if workers>1 else 'sequential'),
        inherited_validation_metrics=old['validation_metrics'],final_holdout_recomputed=True)
    result['reused_fresh_fit_prefix'] = True
    result['reused_same_fresh_fit_checkpoint'] = True
    result['reused_production_calibration'] = False
    result['prefix_replayed'] = True
    result['reused_fitted_calibration_scope'] = 'No external or production calibration; saved refactors from this same fresh fit are explicitly replayed.'
    result['departures_from_full_dataset_canonical_task'] = [
        ('Earlier evaluations of this same fresh pilot fit are verified and replayed; no production calibration is loaded.'
         if text.startswith('No previous scales') else text) for text in result['departures_from_full_dataset_canonical_task']]
    for i,entry in enumerate(result['iterations']):
        entry['evaluation_origin'] = 'verified_same_fit_checkpoint_replay' if i<prefix else 'new_image_evaluation'
    fresh.write_json(Path(output)/'optimization.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--additional-iterations',type=int,default=10)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--fork',type=Path,default=fresh.DEFAULT_FORK)
    args=parser.parse_args(); started=time.time()
    manifest=fresh.load_manifest(args.manifest); output=args.output_root.resolve(); source=args.checkpoint_root.resolve()
    protected=[source,args.manifest.resolve(),args.fork.resolve(),Path(manifest['codebook_path'])]
    # Continuation inputs retain their original chain. Protect every earlier
    # checkpoint directory as well as the directly selected one.
    lineage=json.loads((source/'inputs.json').read_text())
    while lineage:
        if lineage.get('same_fresh_fit_checkpoint'):
            protected.append(Path(lineage['same_fresh_fit_checkpoint']).resolve())
        lineage=lineage.get('original_inputs')
    protected += [Path(p[k]) for p in manifest['planes'] for k in ('images_path','mask_path')]
    if manifest.get('registration_manifest'): protected.append(Path(manifest['registration_manifest']))
    if any(output==p or output in p.parents or p in output.parents for p in protected):
        raise ValueError('New output overlaps protected input or checkpoint')
    output.mkdir(parents=True,exist_ok=False)
    for name in ('tmp','cache'):(output/name).mkdir()
    os.environ.update(TMPDIR=str(output/'tmp'),TEMP=str(output/'tmp'),TMP=str(output/'tmp'),
                      XDG_CACHE_HOME=str(output/'cache'),MPLCONFIGDIR=str(output/'cache/matplotlib'),
                      NUMBA_CACHE_DIR=str(output/'cache/numba'),PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True; fresh._install_write_guard(output)
    Codebook,sink,Optimizer,Decoder=fresh.load_canonical(args.fork.resolve(),output)
    book=Codebook(sink,manifest['codebook_path'],codebookIndex=0,codebookName='FreshVirtualFOVContinuation')
    checkpoint=verify_checkpoint(source,manifest,Optimizer,book,manifest_path=args.manifest,
        canonical_source_paths=[args.fork/name for name in fresh.CANONICAL_SOURCES])
    fingerprints=dict(checkpoint['inputs']['input_sha256'])
    fingerprints.update(checkpoint['checkpoint_sha256'])
    fingerprints.update(checkpoint['registration_source_sha256'])
    fingerprints[str(Path(__file__).resolve())]=fresh.sha256(__file__)
    fingerprints[str(Path(fresh.__file__).resolve())]=fresh.sha256(fresh.__file__)
    fresh.write_json(output/'inputs.json',dict(manifest=manifest,input_sha256=fingerprints,
        runner_sha256=fresh.sha256(fresh.__file__),continuation_runner_sha256=fresh.sha256(__file__),
        same_fresh_fit_checkpoint=str(source),original_inputs=checkpoint['inputs']))
    fresh.write_json(output/'continuation.json',dict(status='VERIFIED_PREFIX_READY',
        source_checkpoint_root=str(source),source_checkpoint_sha256=checkpoint['checkpoint_sha256'],
        additional_fixed_iteration_budget=args.additional_iterations,production_calibration_reused=False))
    result=run_continuation(checkpoint,manifest,output,book,Optimizer,Decoder,
        additional_iterations=args.additional_iterations,workers=args.workers,worker_fork=args.fork)
    if any(fresh.sha256(path)!=value for path,value in fingerprints.items()):
        raise RuntimeError('Original inputs/checkpoint/runner changed during continuation')
    registration_inputs(manifest)  # Recheck raw-file stat provenance as well.
    result.update(all_hashed_inputs_unchanged=True,elapsed_seconds=time.time()-started)
    fresh.write_json(output/'optimization.json',result)
    print(json.dumps(dict(status=result['status'],output=str(output),continuation=result['continuation'])),flush=True)
    if not result['convergence_passed']: raise SystemExit(2)


if __name__=='__main__':main()
