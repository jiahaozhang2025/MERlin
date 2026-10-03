"""Stream a frozen tile/depth chunk through the canonical decoding operators."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
from full_preprocess import load_registration,AuditedAcquisition,produce_plane,sha
from full_decode_operators import decode_with_context
from full_decode_receipts import write_plane,verify_plane


def run_task(plan_path,task_id,num_threads=1):
    plan_path=Path(plan_path).resolve();root=plan_path.parent.parent
    if plan_path!=root/'full_decode_v1/plan.json':raise ValueError('Unexpected isolated decode plan location')
    plan=json.loads(plan_path.read_text());digest=sha(plan_path)
    if plan['status']!='PLANNED_AFTER_REGISTRATION_AND_CALIBRATION':raise RuntimeError('Decode plan is not ready')
    protected={**plan['protected_input_sha256'],str(plan_path):digest}
    if any(sha(p)!=h for p,h in protected.items()):raise RuntimeError('Planned input changed')
    _,models,registration_inputs=load_registration(plan['registration_manifest'])
    if any(protected.get(p)!=h for p,h in registration_inputs.items()):raise RuntimeError('Registration differs from planned inputs')
    sys.path.insert(0,plan['fresh_adapters'])
    from fresh_decode import load_accepted_optimization,decode_arrays
    from fresh_optimize import load_canonical
    scales,backgrounds,_=load_accepted_optimization(plan['optimization_root'],plan)
    task=next(t for t in plan['tasks'] if t['task_id']==task_id)
    tile=next(t for t in plan['layout']['tiles'] if t['tile_id']==task['tile_id'])
    task_root=plan_path.parent/f'task{task_id:04d}';task_root.mkdir(exist_ok=True)
    # A unique attempt keeps failed files and logs reviewable. Completed plane
    # outputs are reused only after full receipt and byte verification below.
    runtime=task_root/f'runtime_{os.getpid()}_{time.time_ns()}';runtime.mkdir()
    for name in ['tmp','cache']:(task_root/name).mkdir(exist_ok=True)
    os.environ.update(TMPDIR=str(task_root/'tmp'),TEMP=str(task_root/'tmp'),TMP=str(task_root/'tmp'),
        XDG_CACHE_HOME=str(task_root/'cache'),MPLCONFIGDIR=str(task_root/'cache/matplotlib'),
        NUMBA_CACHE_DIR=str(task_root/'cache/numba'),PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    from fresh_optimize import _install_write_guard
    _install_write_guard(task_root)
    Codebook,sink,_,Decoder=load_canonical(Path(plan['fork']),runtime)
    book=Codebook(sink,plan['codebook_path'],codebookIndex=0,codebookName='FreshFullStitched0718')
    if list(book.get_bit_names())!=plan['bit_names']:raise RuntimeError('Codebook bit order changed')
    inventory=json.loads((root/'inventory.json').read_text());acq=AuditedAcquisition(root,inventory)
    completed=[]
    for plane in task['planes']:
        folder=task_root/f'z{plane["z_index"]:03d}'
        if folder.exists():
            receipt=verify_plane(folder,tile,plane,plan['layout'],digest,protected)
        else:
            started=time.time();core_support={}
            def produce(halo):
                image,support,owners,report=produce_plane(acq,plan['bit_names'],models,tile,plane['physical_z_um'],plan['fork'],decode_halo=halo)
                h,w=tile['shape_yx'];core_support['mask']=support[halo:halo+h,halo:halo+w].copy()
                return image,support,owners,report
            frame,rasters,report=decode_with_context(produce,tile,plan['layout'],decode_arrays,book,Decoder,
                scales,backgrounds,plane['z_index'],plane['physical_z_um'],num_threads=num_threads)
            del rasters
            report['elapsed_seconds']=time.time()-started
            receipt=write_plane(folder,frame,core_support['mask'],tile,plane,plan['layout'],report,digest,protected,acq.audit())
        completed.append(dict(plane=plane,receipt_sha256=sha(folder/'complete.json'),raw_barcodes=receipt['raw_barcodes'],supported_core_pixels=receipt['supported_core_pixels']))
        print(json.dumps(dict(task=task_id,**completed[-1])),flush=True)
    if any(sha(p)!=h for p,h in protected.items()):raise RuntimeError('Input changed before task completion')
    report=dict(status='COMPLETE',task=task,plan_sha256=digest,planes=completed,all_hashed_inputs_unchanged=True)
    receipt_path=task_root/'complete.json'
    if receipt_path.exists():
        if json.loads(receipt_path.read_text())!=report:raise RuntimeError('Existing task receipt disagrees')
    else:receipt_path.write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);p.add_argument('--num-threads',type=int,default=1);a=p.parse_args()
    run_task(Path(__file__).resolve().parent.parent/'full_decode_v1/plan.json',a.task,a.num_threads)
if __name__=='__main__':main()
