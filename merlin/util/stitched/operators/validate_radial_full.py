"""Measure raw bead-image registration at frozen full-field validation windows."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation
import camera_geometry as gq
from full_sampling_radial import NativeAcquisition,stitch,MPP


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correlation(a,b):
    aa=np.asarray(a,float);bb=np.asarray(b,float);aa=aa-aa.mean();bb=bb-bb.mean()
    den=np.linalg.norm(aa)*np.linalg.norm(bb)
    return float(np.sum(aa*bb)/den) if den>0 else 0.


def measure_pair(images,taper,core):
    hp=[im-ndimage.gaussian_filter(im,(0,3,3)) for im in images]
    before=correlation(hp[0][core],hp[1][core])
    if any(np.linalg.norm(im*taper)==0 for im in hp):
        return dict(status='AMBIGUOUS_IMAGE_MATCH',reason='NO_IMAGE_CONTRAST',ncc_before=before,
            ncc_after=0.,xy_residual_px=None,z_residual_um=None,pointwise_accuracy_pass=False),hp
    shift,_,_=phase_cross_correlation(hp[0]*taper,hp[1]*taper,upsample_factor=20,normalization=None)
    if not np.isfinite(shift).all():
        return dict(status='AMBIGUOUS_IMAGE_MATCH',reason='NONFINITE_IMAGE_SHIFT',ncc_before=before,
            ncc_after=None,xy_residual_px=None,z_residual_um=None,pointwise_accuracy_pass=False),hp
    shifted=ndimage.shift(hp[1],shift,order=1,mode='constant',cval=0,prefilter=False)
    after=correlation(hp[0][core],shifted[core]);xy=float(np.linalg.norm(shift[1:]));dz=float(abs(shift[0]*.5))
    identifiable=bool(after>=.8 and dz<3 and xy<16)
    return dict(status='IDENTIFIABLE' if identifiable else 'AMBIGUOUS_IMAGE_MATCH',
        xy_residual_px=xy,z_residual_um=dz,ncc_before=before,ncc_after=after,
        shift_to_apply_moving_xyz_um=[float(shift[2]*MPP),float(shift[1]*MPP),float(shift[0]*.5)],
        pointwise_accuracy_pass=bool(identifiable and xy<=2 and dz<=1)),hp


def main():
    root=Path(__file__).resolve().parent.parent;started=time.time()
    manifest_path=root/'qa_windows_core_v2/manifest.json';manifest=json.loads(manifest_path.read_text())
    task=manifest['tasks'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    definition_path=Path(task['definition']);assert sha(definition_path)==task['sha256']
    definitions=json.loads(definition_path.read_text());r=task['round'];fov=task['fov']
    inventory=json.loads((root/'inventory.json').read_text())
    inputs={str(manifest_path):sha(manifest_path),str(definition_path):sha(definition_path)}
    within={}
    for ir in sorted({0,r}):
        path=root/'registration_camera_radial_v2'/f'round{ir}'/'model.json'
        assert sha(path)==json.loads(path.with_name('complete.json').read_text())['output_sha256']['model.json']
        within[ir]=json.loads(path.read_text());inputs[str(path)]=sha(path)
    cross=None
    if r:
        path=root/'registration_camera_radial_v2'/f'round{r}'/'cross/preferred_model.json'
        record=json.loads(path.with_name('complete.json').read_text())
        assert sha(path)==record['output_sha256'][path.name] and record['all_hashed_inputs_unchanged']
        cross=json.loads(path.read_text());inputs[str(path)]=sha(path)
    output=root/'image_qa_radial_full_v2'/f'round{r}'/f'fov{fov}'
    output.mkdir(parents=True,exist_ok=False)
    acq=NativeAcquisition(inventory['data_dir'],inventory['raw_root'])
    rows={}
    for ir in sorted({0,r}):
        row=next(x.copy() for x in acq.organization if int(x['imagingRound'])==ir)
        row.update(frame=row['fiducial3DStackFrames'],zPos=row['fiducial3DzPos']);rows[ir]=row
    zz,yy,xx=np.meshgrid(np.arange(-4,4.01,.5),np.arange(-64,65)*MPP,np.arange(-64,65)*MPP,indexing='ij')
    offsets=np.stack([xx,yy,zz],axis=-1)
    taper=np.hanning(17)[:,None,None]*np.hanning(129)[None,:,None]*np.hanning(129)[None,None,:]
    core=np.zeros(zz.shape,bool);core[4:-4,16:-16,16:-16]=True
    records=[];previews=[];preview_indexes=[]
    for index,window in enumerate(definitions['windows']):
        record=dict(window,index=index)
        if window['selection_status']!='SELECTED':
            record['status']='NO_HELD_OUT_FEATURE';records.append(record);continue
        ir=r if window['kind']=='within' else 0
        source=window['source_fovs'][0] if window['kind']=='within' else window['reference_source_fov']
        center=gq.apply_global(np.asarray(window['native_center_xyz_um']),source,within[ir])
        record['candidate_center_xyz_um']=center.tolist()
        grid=offsets+center
        images=[];masks=[]
        if window['kind']=='within':
            for source in window['source_fovs']:
                image,mask=acq.sample(rows[r],source,gq.inverse_global(grid,source,within[r]))
                images.append(image);masks.append(mask)
        else:
            for ir,cm in [(0,None),(r,cross)]:
                image,mask,_=stitch(acq,rows[ir],grid,within[ir],cm)
                images.append(image);masks.append(mask)
        available=masks[0]&masks[1];record['available_fraction']=float(available.mean())
        if not available.all():
            record['status']='UNACQUIRED_SUPPORT';records.append(record);continue
        measurement,hp=measure_pair(images,taper,core)
        record.update(measurement)
        records.append(record);previews.append(np.array([np.max(im,axis=0) for im in hp],np.float32));preview_indexes.append(index)
        if index%10==0:print(json.dumps(dict(round=r,fov=fov,window=index,seconds=time.time()-started)),flush=True)
    for path,stats in acq.raw_stats.items():
        assert (path.stat().st_size,path.stat().st_mtime_ns)==stats
        expected=next(t for t in inventory['tasks'] if t['raw_path']==str(path))
        assert stats==(expected['raw_size'],expected['raw_mtime_ns'])
    if any(sha(path)!=digest for path,digest in inputs.items()):
        raise RuntimeError('Validation inputs changed')
    np.savez_compressed(output/'window_previews.npz',images=np.array(previews),record_index=np.array(preview_indexes))
    report=dict(status='MEASURED_REQUIRES_DECISION',round=r,fov=fov,records=records,
        gates=dict(ncc_after_min=.8,pointwise_xy_max_px=2,pointwise_z_max_um=1,aggregate_xy_median_px=1,
            aggregate_xy_p95_px=2,aggregate_z_median_um=.5,aggregate_z_p95_um=1),
        input_sha256=inputs,all_hashed_inputs_unchanged=True,raw_source_stats={str(k):list(v) for k,v in acq.raw_stats.items()},
        frame_schedule_provenance=acq.schedule_provenance,elapsed_seconds=time.time()-started,
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('full_sampling_radial.py')]},
        output_sha256={'window_previews.npz':sha(output/'window_previews.npz')})
    (output/'complete.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(round=r,fov=fov,windows=len(records),identifiable=sum(x['status']=='IDENTIFIABLE' for x in records),
        pointwise_failed=sum(x.get('pointwise_accuracy_pass') is False for x in records),seconds=time.time()-started)),flush=True)


if __name__=='__main__':main()
