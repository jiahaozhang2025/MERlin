"""Worker for native MERlin stitched stages (no internal Slurm submissions)."""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from stitched_config import ROOT,CONFIG

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def check_receipt(p):
    rec=read(p)
    outputs=rec.get('output_sha256',{})
    if isinstance(outputs,dict):
        for name,h in outputs.items():
            if isinstance(h,str):assert sha(p.parent/name)==h,(p,name)
    elif isinstance(outputs,str):
        candidates=[p.with_suffix('.npz'),p.with_suffix('.pkl'),p.parent/'stitched_cloud_owned.npz']
        assert any(q.is_file() and sha(q)==outputs for q in candidates),p
    return rec
def call(name,*args,receipt=None):
    if receipt and Path(receipt).exists():check_receipt(Path(receipt));return
    subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/name),*map(str,args)],check=True,cwd=ROOT/'scripts')
    if receipt:check_receipt(Path(receipt))

def preflight():
    import xml.etree.ElementTree as ET
    import bead_cloud as bc
    data=ROOT/'inputs';raw=Path(CONFIG['raw_root'])
    micro=read(data/'microscope_parameters.json')
    if micro['microns_per_pixel']!=.1493 or micro['image_dimensions']!=[2048,2048]:raise ValueError('Unsupported microscope geometry for validated0718 profile')
    positions=np.loadtxt(data/'positions.csv',delimiter=',').tolist()
    with (data/'filemap.csv').open() as f:fovs=sorted({int(r['fov']) for r in csv.DictReader(f)})
    if fovs!=list(range(52)) or len(positions)!=52:raise ValueError('The current validated profile requires52 sourceFOVs, anchor13')
    tasks=[];frame_bytes=2048*2048*2
    for r in CONFIG['rounds']:
        for f in fovs:
            path,frames,zs,_,_=bc.raw_metadata(data,raw,f,r)
            xml=path.with_suffix('.xml');nodes=[e.text for e in ET.parse(xml).iter() if e.tag=='z_offsets']
            assert len(nodes)==1
            schedule=np.array([float(x) for x in nodes[0].split(',')])
            assert path.stat().st_size==len(schedule)*frame_bytes and abs(schedule[0])<1e-6
            assert len(zs)==150 and np.allclose(schedule[frames],zs,rtol=0,atol=1e-6)
            tasks.append(dict(index=len(tasks),fov=f,round=r,raw_path=str(path),raw_size=path.stat().st_size,
                raw_mtime_ns=path.stat().st_mtime_ns,xml_path=str(xml),xml_sha256=sha(xml),frame_count=len(schedule)))
    edges=[];width=2048*.1493
    for a in fovs:
        for b in fovs[a+1:]:
            dx,dy=np.abs(np.array(positions[a])-positions[b])
            if (dx<5 and 0<dy<width) or (dy<5 and 0<dx<width):edges.append([a,b])
    save(ROOT/'inventory.json',dict(status='PREFLIGHT_PASS',source_fovs=fovs,positions=positions,side_edges=edges,
        rounds=CONFIG['rounds'],reference_round=0,reference_anchor_fov=13,codebook_indices=[CONFIG['codebook_index']],
        tasks=tasks,metadata_sha256={p.name:sha(p) for p in data.iterdir()},data_dir=str(data),raw_root=str(raw),output_root=str(ROOT)))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--fragment',type=int);p.add_argument('--workers',type=int,default=52);a=p.parse_args()
    stage=a.stage;i=a.fragment
    for name,h in read(ROOT/'frozen_sources.json').items():assert sha(ROOT/name)==h,name
    done=ROOT/'stage_receipts'/f'{stage}_{i}.json'
    if done.exists():
        check_receipt(done)
        for name,h in read(done)['artifacts'].items():
            assert sha(ROOT/name)==h,name
            if name.endswith('.json'):check_receipt(ROOT/name)
        return
    artifacts=[]
    def record(p):artifacts.append(Path(p));return p
    if stage=='preflight':preflight();record(ROOT/'inventory.json')
    elif stage=='extract':
        os.environ['SLURM_ARRAY_TASK_ID']=str(i);t=read(ROOT/'inventory.json')['tasks'][i]
        call('extract_task.py',receipt=record(ROOT/'clouds'/f"fov{t['fov']}_r{t['round']}.json"))
    elif stage in ('within','pool','cross'):
        r=CONFIG['rounds'][i]
        if stage=='within':
            call('within_core_v2.py','--round',r,receipt=record(ROOT/f'registration_core_v2/round{r}/complete.json'))
            call('fit_shared_radial.py','--round',r,receipt=record(ROOT/f'registration_camera_radial_v2/round{r}/complete.json'))
        elif stage=='pool':
            for name,family in [('pool_core_v2','registration_core_v2'),('pool_radial_v2','registration_camera_radial_v2')]:
                call(name+'.py','--round',r,receipt=record(ROOT/family/f'round{r}/pool_complete.json'))
        elif r:
            for name,family in [('cross_core_v2','registration_core_v2'),('cross_radial_v2','registration_camera_radial_v2')]:
                call(name+'.py','--round',r,receipt=record(ROOT/family/f'round{r}/cross/complete.json'))
    elif stage=='qa_windows':call('make_core_qa_windows.py',receipt=record(ROOT/'qa_windows_core_v2/manifest.json'))
    elif stage=='qa':
        os.environ['SLURM_ARRAY_TASK_ID']=str(i);t=read(ROOT/'qa_windows_core_v2/manifest.json')['tasks'][i]
        call('validate_radial_full.py',receipt=record(ROOT/f"image_qa_radial_full_v2/round{t['round']}/fov{t['fov']}/complete.json"))
    elif stage=='registration_review':
        from registration_review import review
        review(ROOT,CONFIG);record(ROOT/'registration_accepted_radial_v1.json')
    elif stage=='adopt_registration':
        from registration_review import adopt
        adopt(ROOT,CONFIG);record(ROOT/'registration_accepted_radial_v1.json')
    elif stage=='calibration_plan':
        call('full_calibration_plan.py','--registration',ROOT/'registration_accepted_radial_v1.json',receipt=record(ROOT/'calibration_inputs_v1/plan.json'))
    elif stage=='calibration_images':
        call('render_calibration_plane.py','--sample',i,'--fork',CONFIG['canonical_root'],receipt=record(ROOT/f'calibration_inputs_v1/sample{i:03d}/complete.json'))
    elif stage=='calibration_finalize':call('finalize_calibration_inputs.py',receipt=record(ROOT/'calibration_inputs_v1/optimizer_manifest.json'))
    elif stage=='optimize':
        call('fresh_pipeline/fresh_optimize.py','--manifest',ROOT/'calibration_inputs_v1/optimizer_manifest.json','--output-root',ROOT/'full_optimization_v1',
             '--fork',CONFIG['canonical_root'],'--iterations',20,'--workers',CONFIG['optimizer_workers'],'--num-threads',CONFIG['num_threads'],receipt=record(ROOT/'full_optimization_v1/optimization.json'))
        assert read(ROOT/'full_optimization_v1/optimization.json')['convergence_passed'],'Optimization has not converged'
    elif stage=='decode_plan':
        call('full_decode_plan.py','--optimization-root',ROOT/'full_optimization_v1','--fork',CONFIG['canonical_root'],'--fresh-adapters',ROOT/'scripts/fresh_pipeline',receipt=record(ROOT/'full_decode_v1/plan.json'))
    elif stage=='decode':
        plan=read(ROOT/'full_decode_v1/plan.json');assert a.workers>0 and 0<=i<a.workers
        for t in plan['tasks'][i::a.workers]:
            call('run_full_decode_chunk.py','--task',t['task_id'],'--num-threads',CONFIG['num_threads'],receipt=record(ROOT/f"full_decode_v1/task{t['task_id']:04d}/complete.json"))
    elif stage=='raw_finalize':call('finalize_full_decode.py',receipt=record(ROOT/'full_decode_v1/raw_manifest.json'))
    elif stage=='adaptive_filter':call('full_filter_run.py','threshold',receipt=record(ROOT/'full_filter_v1/threshold.json'))
    elif stage=='deduplicate':
        call('full_filter_run.py','dedup','--identity',i,receipt=record(ROOT/f'full_filter_v1/dedup_by_identity/barcode{i:05d}/complete.json'))
    elif stage=='export':call('full_filter_run.py','merge',receipt=record(ROOT/'full_filter_v1/exports/complete.json'))
    elif stage=='quality':call('full_result_review.py',receipt=record(ROOT/'final_review_v1/summary.json'))
    elif stage.startswith('partition_'):
        sub=stage[len('partition_'):];out=ROOT/'partition_sjoin_v2';out.mkdir(exist_ok=True)
        if sub=='setup':
            call('validate_partition_sjoin.py',out/'validation.json',receipt=out/'validation.json')
            call('run_partition_sjoin.py','setup',receipt=record(out/'setup.json'))
        elif sub in ('geometry','join'):
            receipt=out/('geometry' if sub=='geometry' else 'joins')/f'{i:02d}.json'
            call('run_partition_sjoin.py',sub,i,receipt=record(receipt))
        else:
            receipt=out/({'coordinates':'coordinates/complete.json','export':'exports/complete.json','verify':'exports/verification.json'}[sub])
            call('run_partition_sjoin.py',sub,receipt=record(receipt))
    elif stage=='cell_centers':
        from cell_centers import run
        run(ROOT,CONFIG);record(ROOT/'partition_sjoin_v2/cell_coordinates_v3/complete.json')
    elif stage=='analysis_report':
        from analysis_report import run
        run(ROOT,CONFIG);record(ROOT/'basic_analysis/complete.json')
    else:raise ValueError('Unknown stitched stage '+stage)
    save(done,dict(status='COMPLETE',stage=stage,fragment=i,artifacts={str(p.relative_to(ROOT)):sha(p) for p in artifacts}))

if __name__=='__main__':main()
