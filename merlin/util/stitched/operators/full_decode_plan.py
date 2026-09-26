"""Plan every output tile/depth only after reviewed registration and calibration."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from bead_cloud import numbers
from full_preprocess import load_registration,sha
from full_decode_receipts import make_tasks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--optimization-root',type=Path,required=True)
    parser.add_argument('--fork',type=Path,required=True)
    parser.add_argument('--fresh-adapters',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    optimization=args.optimization_root.resolve();fork=args.fork.resolve();adapters=args.fresh_adapters.resolve()
    if not optimization.is_relative_to(root) or not adapters.is_relative_to(root):
        raise ValueError('Optimization and adapter sources must be in this isolated run')
    sys.path.insert(0,str(adapters))
    from fresh_decode import load_accepted_optimization
    from full_preprocess import AuditedAcquisition
    calibration_path=root/'calibration_inputs_v1/optimizer_manifest.json'
    calibration=json.loads(calibration_path.read_text())
    calibration_plan_path=root/'calibration_inputs_v1/plan.json'
    if sha(calibration_plan_path)!=calibration['full_sample_plan_sha256']:
        raise RuntimeError('Calibration plan changed')
    calibration_plan=json.loads(calibration_plan_path.read_text())
    _,models,protected=load_registration(calibration['registration_manifest'])
    protected.update(calibration['verified_input_sha256'])
    sf,bg,optimization_inputs=load_accepted_optimization(optimization,calibration)
    # Require the exact full-sample calibration manifest, not merely a matching
    # bit list or a prior successful optimization in the same coordinate frame.
    if optimization_inputs['input_sha256'].get(str(calibration_path.resolve()))!=sha(calibration_path):
        raise RuntimeError('Optimization did not use this verified full-sample calibration')
    for name in ['merlin/util/decoding.py','merlin/data/codebook.py']:
        path=fork/name
        if optimization_inputs['input_sha256'].get(str(path))!=sha(path):
            raise RuntimeError('Canonical decoding source differs from calibration')
    for path,h in calibration['source_sha256'].items():
        if sha(path)!=h:raise RuntimeError('Preprocessing/canonical source differs from calibration: '+path)
    sources=[Path(__file__).with_name(n) for n in ['full_decode_plan.py','run_full_decode_chunk.py',
        'full_decode_receipts.py','full_decode_operators.py','finalize_full_decode.py','tile_layout.py']]
    sources+=list(adapters.glob('*.py'))
    protected.update({str(p.resolve()):sha(p) for p in sources})
    protected.update({str(p.resolve()):sha(p) for p in [calibration_path,calibration_plan_path,
        *[optimization/n for n in ['optimization.json','inputs.json','scale_factors.npy','backgrounds.npy']]]})
    inventory=json.loads((root/'inventory.json').read_text())
    if inventory['source_fovs']!=list(range(52)) or len(inventory['codebook_indices'])!=1:
        raise ValueError('Requested sample/codebook scope changed')
    acq=AuditedAcquisition(root,inventory)
    # zPos defines the common output reference lattice. Raw XML Z coordinates
    # still determine each channel's interpolation and acquired support.
    schedules=[numbers(acq.bit_row(bit)['zPos']) for bit in calibration['bit_names']]
    if any(not np.array_equal(schedules[0],z) for z in schedules[1:]):
        raise ValueError('Codebook bits disagree on reference output depths')
    layout=calibration_plan['layout']
    from tile_layout import build_layout
    if build_layout(models[0]['within'])!=layout:
        raise RuntimeError('Output lattice changed after calibration')
    tasks=make_tasks(layout,schedules[0])
    if any(sha(p)!=h for p,h in protected.items()):raise RuntimeError('Planning input changed')
    out=root/'full_decode_v1';out.mkdir(exist_ok=False)
    plan=dict(status='PLANNED_AFTER_REGISTRATION_AND_CALIBRATION',bit_names=calibration['bit_names'],
        codebook_path=calibration['codebook_path'],registration_manifest=calibration['registration_manifest'],
        optimization_root=str(optimization),fork=str(fork),fresh_adapters=str(adapters),layout=layout,
        z_positions_um=schedules[0].tolist(),tasks=tasks,protected_input_sha256=protected,
        requested_source_fovs=inventory['source_fovs'],unsupported_source_fovs_by_round={str(r):m['within']['unsupported_fovs'] for r,m in models.items()},
        depth_policy='z is the index in this common reference z_positions_um array; native per-channel XML depths control sampling. Calibration fragment IDs are never decoding Z indices.',
        coverage_policy='Every rectangular output tile and reference depth gets a complete receipt, including empty support. Core all-bit support masks measure usable registered coverage, not whether raw tissue was imaged.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False))
    print(json.dumps(dict(tasks=len(tasks),tiles=len(layout['tiles']),depths=len(schedules[0]),output=str(out))))
if __name__=='__main__':main()
