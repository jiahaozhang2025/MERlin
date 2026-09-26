"""Evidence-bound review gate; missing or failed alignment is never accepted."""
import collections,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))

def metrics(rows):
    measured=[w for w in rows if w['status']=='IDENTIFIABLE']
    out=dict(windows=len(rows),identifiable=len(measured),status_counts=dict(collections.Counter(w['status'] for w in rows)),qualified_gates_pass=False)
    if measured:
        for k in ['xy_residual_px','z_residual_um']:
            out[k]=dict(median=float(np.median([w[k] for w in measured])),p95=float(np.percentile([w[k] for w in measured],95)))
        out['qualified_gates_pass']=bool(len(measured)>=8 and out['xy_residual_px']['median']<=1 and out['xy_residual_px']['p95']<=2 and out['z_residual_um']['median']<=.5 and out['z_residual_um']['p95']<=1)
    return out

def adopt(root,config):
    """Reuse reviewed geometry only when every required acquisition still matches."""
    root=Path(root);path=Path(config['reuse_registration']).resolve();old=read(path)
    if old['status']!='ACCEPTED_FOR_FULL_SAMPLE':raise ValueError('Only reviewed registration may be reused')
    old_inventory=read(path.parent/'inventory.json');new_inventory=read(root/'inventory.json')
    for name,digest in new_inventory['metadata_sha256'].items():
        if old_inventory['metadata_sha256'].get(name)!=digest:raise ValueError('Registration belongs to different metadata: '+name)
    old_tasks={(t['fov'],t['round']):t for t in old_inventory['tasks']}
    for t in new_inventory['tasks']:
        prior=old_tasks.get((t['fov'],t['round']))
        if not prior:raise ValueError('Reused registration lacks requested acquisition')
        for key in ['raw_path','raw_size','raw_mtime_ns','xml_sha256']:
            if t[key]!=prior[key]:raise ValueError('Reused raw acquisition differs: '+key)
    required={str(path):sha(path),str(root/'inventory.json'):sha(root/'inventory.json')}
    for name,h in old['required_input_sha256'].items():
        p=Path(name) if Path(name).is_absolute() else path.parent/name
        assert sha(p)==h,p;required[str(p.resolve())]=h
    rounds={}
    for r in config['rounds']:
        if str(r) not in old['rounds']:raise ValueError('Reused registration lacks requested round '+str(r))
        rounds[str(r)]={k:str((path.parent/v).resolve()) if v else None for k,v in old['rounds'][str(r)].items()}
    result=dict(old,rounds=rounds,required_input_sha256=required,reused_registration=str(path),
        reused_registration_sha256=sha(path),calibration_policy='Geometry reused read-only; fresh intensity optimization and thresholds required.')
    write(root/'registration_accepted_radial_v1.json',result)

def review(root,config):
    root=Path(root);out=root/'registration_review';out.mkdir(exist_ok=True)
    sources={};rows=[];rounds={};all_pass=True;protected={str(root/'inventory.json'):sha(root/'inventory.json')}
    models={};unsupported={}
    for r in config['rounds']:
        current=[]
        for f in range(52):
            p=root/f'image_qa_radial_full_v2/round{r}/fov{f}/complete.json';rec=read(p)
            assert rec['all_hashed_inputs_unchanged']
            for name,h in rec['input_sha256'].items():assert sha(name)==h,name
            for name,h in rec['output_sha256'].items():assert sha(p.parent/name)==h
            sources[str(p)]=sha(p)
            current.extend(dict(w,round=r,qa_receipt=str(p)) for w in rec['records'])
        rows.extend(current)
        within=metrics([w for w in current if w['kind']=='within']);cross=metrics([w for w in current if w['kind']=='cross'])
        depth=[]
        for lo in range(0,150,25):
            values=[w for w in current if w['kind']=='cross' and 'candidate_center_xyz_um' in w and lo<=w['candidate_center_xyz_um'][2]<lo+25]
            depth.append(dict(z_range_um=[lo,lo+25],**metrics(values)))
        all_pass &= within['qualified_gates_pass'] and (not r or cross['qualified_gates_pass'] and all(d['qualified_gates_pass'] for d in depth))
        rounds[str(r)]=dict(within=within,cross=cross,cross_by_depth=depth,
            cross_by_fov={str(f):metrics([w for w in current if w['kind']=='cross' and w['reference_source_fov']==f]) for f in range(52)})
        family=root/f'registration_camera_radial_v2/round{r}';fit=read(family/'complete.json')
        all_pass &= bool(fit['optimizer']['success'] and not fit['jacobian_failed'])
        for name,h in fit['output_sha256'].items():assert sha(family/name)==h
        path=family/'model.json';model=read(path);protected[str(path)]=sha(path)
        unsupported[str(r)]=model['unsupported_fovs'];models[str(r)]={'within':str(path.relative_to(root)),'cross':None}
        if r:
            path=family/'cross/preferred_model.json';cross_model=read(path);receipt=read(path.parent/'complete.json')
            for name,h in receipt['output_sha256'].items():assert sha(path.parent/name)==h
            all_pass &= bool(cross_model['optimization']['success'] and cross_model['assessment']['geometry']['passed'])
            protected[str(path)]=sha(path);models[str(r)]['cross']=str(path.relative_to(root))
    exceptions=[f"r{w['round']}:{w['id']}" for w in rows if w['status']=='IDENTIFIABLE' and not w['pointwise_accuracy_pass']]
    evidence=dict(qa_sources=sources,model_sources=protected)
    write(out/'evidence.json',evidence);evidence_hash=sha(out/'evidence.json')
    write(out/'summary.json',dict(status='REQUIRES_REVIEW',aggregate_and_depth_gates_pass=bool(all_pass),rounds=rounds,
        unsupported_sources_by_round=unsupported,pointwise_exception_ids=exceptions,evidence_sha256=evidence_hash))
    write(out/'all_windows.json',rows)
    # Native raw-window overlays selected by residual rank, never used for fitting.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    measured=sorted([w for w in rows if w['status']=='IDENTIFIABLE'],key=lambda w:w['xy_residual_px'])
    selected=[measured[k] for k in sorted(set([0,len(measured)//2,*range(max(0,len(measured)-4),len(measured))]))] if measured else []
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for ax in axes.ravel():ax.axis('off')
    for ax,w in zip(axes.ravel(),selected):
        with np.load(Path(w['qa_receipt']).parent/'window_previews.npz') as z:
            images=z['images'][np.flatnonzero(z['record_index']==w['index'])[0]]
        planes=[np.clip(x/max(float(np.percentile(x,99.5)),1),0,1) for x in images]
        ax.imshow(np.stack([planes[1],planes[0],planes[1]],-1));ax.set_title(f"R{w['round']} {w['id']}\nXY {w['xy_residual_px']:.2f}px Z {w['z_residual_um']:.2f}µm",fontsize=8)
    fig.savefig(out/'raw_image_overlays.png',dpi=150);plt.close(fig)
    if not all_pass:raise RuntimeError('Stitched alignment failed aggregate/depth/geometry gates; see registration_review/summary.json')
    decision_path=config.get('review_decision')
    if not decision_path or not Path(decision_path).exists():
        raise RuntimeError('Alignment review required: inspect summary and raw_image_overlays.png, then provide review_decision with matching evidence_sha256')
    decision=read(decision_path)
    if decision.get('decision')!='accept_common_support' or decision.get('evidence_sha256')!=evidence_hash or not decision.get('reviewed_by'):
        raise ValueError('Review decision is missing, stale or belongs to different evidence')
    if sorted(decision.get('acknowledged_pointwise_exceptions',[]))!=sorted(exceptions):raise ValueError('Every pointwise exception must be acknowledged explicitly')
    if decision.get('acknowledged_unsupported_sources')!=unsupported:raise ValueError('Unsupported source coverage must be acknowledged explicitly')
    write(out/'decision.json',decision);protected[str(out/'decision.json')]=sha(out/'decision.json')
    protected[str(out/'evidence.json')]=evidence_hash
    write(root/'registration_accepted_radial_v1.json',dict(status='ACCEPTED_FOR_FULL_SAMPLE',acceptance_scope='REGISTERED_ACQUIRED_COMMON_SUPPORT_ONLY',
        rounds=models,required_input_sha256=protected,alignment_evidence_sha256=sources,
        pointwise_exceptions=[w for w in rows if f"r{w['round']}:{w['id']}" in exceptions],unsupported_sources_by_round=unsupported,
        limitations=decision.get('limitations',[])+['Aggregate criteria do not certify every point; ambiguous, sparse and unacquired windows remain unvalidated.']))
