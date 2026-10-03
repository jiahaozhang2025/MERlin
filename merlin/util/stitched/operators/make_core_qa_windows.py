"""Freeze raw-image validation locations without using post-fit image residuals."""
import hashlib,json
from pathlib import Path
import numpy as np
import global_quadratic as gq

root=Path(__file__).resolve().parent.parent
inventory=json.loads((root/'inventory.json').read_text())
out=root/'qa_windows_core_v2';out.mkdir(exist_ok=False)
reference=json.loads((root/'registration_core_v2/round0/model.json').read_text())
with np.load(root/'registration_core_v2/round0/stitched_cloud_owned.npz',allow_pickle=False) as d:
    cloud={k:d[k] for k in ['xyz_um','native_xyz_um','source_fov','source_index','within_round_heldout']}
jobs={(r,f):[] for r in inventory['rounds'] for f in inventory['source_fovs']}
sources={}
def bind(path):sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
bind(root/'inventory.json');bind(root/'registration_core_v2/round0/stitched_cloud_owned.npz')
for r in inventory['rounds']:
    folder=root/'registration_core_v2'/f'round{r}'
    model=json.loads((folder/'model.json').read_text());bind(folder/'model.json');bind(folder/'matches.npz')
    with np.load(folder/'matches.npz',allow_pickle=False) as archive:
        for edge in model['assessment']['edges']:
            a,b=edge['k'],edge['n'];prefix=f'{a}_{b}_'
            native=archive[prefix+'heldout_native_a_xyz_um']
            pairs=archive[prefix+'heldout_pairs']
            split=edge['spatial_split'];axis='xy'.index(split['axis'])
            lo,hi=np.array(split['nominal_overlap_low_um']),np.array(split['nominal_overlap_high_um'])
            ends=split['boundaries_um']
            for zbin in range(6):
                for side in ['lower','upper']:
                    bounds=np.stack([lo.copy(),hi.copy()]);bounds[:,2]=[25*zbin,25*(zbin+1)]
                    if side=='lower':bounds[1,axis]=ends[0]
                    else:bounds[0,axis]=ends[1]
                    target=bounds.mean(0);target[2]=10+25*zbin
                    eligible=np.flatnonzero(np.all((native>=bounds[0])&(native<bounds[1]),axis=1)
                        &(native[:,2]>=5)&(native[:,2]<=145))
                    rec=dict(kind='within',id=f'within_r{r}_{a}_{b}_z{zbin}_{side}',round=r,
                        source_fovs=[a,b],z_bin=zbin,side=side,nominal_native_xyz_um=target.tolist(),
                        native_region_fov=a,native_region_bounds_xyz_um=bounds.tolist())
                    if len(eligible):
                        distance=np.linalg.norm((native[eligible]-target)/[50,50,12.5],axis=1)
                        chosen=int(eligible[np.argmin(distance)])
                        rec.update(selection_status='SELECTED',native_center_xyz_um=native[chosen].tolist(),
                            center_xyz_um=gq.apply_global(native[chosen],a,model).tolist(),
                            frozen_heldout_pair_indices=pairs[chosen].tolist())
                    else:rec.update(selection_status='NO_HELD_OUT_FEATURE')
                    jobs[(r,a)].append(rec)
for f in inventory['source_fovs']:
    for zbin in range(6):
        used=set()
        for quadrant,fraction in enumerate((1/3,2/3)):
            target=np.array([2047*.1493*fraction]*2+[10+25*zbin])
            eligible=np.flatnonzero((cloud['source_fov']==f)&cloud['within_round_heldout']
                &(cloud['native_xyz_um'][:,2]>=25*zbin)&(cloud['native_xyz_um'][:,2]<25*(zbin+1))
                &(cloud['native_xyz_um'][:,2]>=5)&(cloud['native_xyz_um'][:,2]<=145))
            eligible=np.array([i for i in eligible if int(i) not in used],int)
            rec=dict(kind='cross',id=f'cross_fov{f}_z{zbin}_q{quadrant}',reference_source_fov=f,
                z_bin=zbin,spatial_target=quadrant,nominal_native_xyz_um=target.tolist())
            if len(eligible):
                distance=np.linalg.norm((cloud['native_xyz_um'][eligible]-target)/[100,100,12.5],axis=1)
                selected=int(eligible[np.argmin(distance)]);used.add(selected)
                rec.update(selection_status='SELECTED',center_xyz_um=cloud['xyz_um'][selected].tolist(),
                    reference_source_index=int(cloud['source_index'][selected]),inherited_within_heldout=True,
                    native_center_xyz_um=cloud['native_xyz_um'][selected].tolist())
            else:rec.update(selection_status='NO_HELD_OUT_FEATURE')
            for r in inventory['rounds']:
                if r:jobs[(r,f)].append(dict(rec,round=r))
manifest=[]
for (r,f),windows in jobs.items():
    path=out/f'round{r}_fov{f}.json'
    payload=dict(round=r,fov=f,windows=windows,
        window_shape_zyx=[17,129,129],spacing_zyx_um=[.5,.1493,.1493],
        selection='Heldout central features; fixed six25um depth bins and two XY targets/sides. Nearest eligible feature to predetermined target; no image residual or intensity ranking.',
        caveats='Feature-centered raw validation preferentially samples detectable beads; missing/sparse cells are reported. Cross-round central reference features inherit within-round holdout exclusion.')
    path.write_text(json.dumps(payload,indent=2)+'\n')
    manifest.append(dict(task_index=len(manifest),round=r,fov=f,definition=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),windows=len(windows)))
report=dict(status='FROZEN',scope='Measured component QA only; missing graph edges and unsupported FOVs remain explicitly unresolved in core model/edge_diagnostics files',tasks=manifest,input_sha256=sources,
    selected_windows=sum(x['selection_status']=='SELECTED' for rows in jobs.values() for x in rows),
    missing_windows=sum(x['selection_status']!='SELECTED' for rows in jobs.values() for x in rows))
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['tasks','input_sha256']}))
