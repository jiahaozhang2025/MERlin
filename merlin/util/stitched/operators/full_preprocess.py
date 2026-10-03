"""Stream registered all-bit images with independent preprocessing/decode context."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy import ndimage
import cross_round
import local_crosswarp_v2
import full_sampling_radial as sampling
import stitched_sampler as pilot


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class AuditedAcquisition(sampling.NativeAcquisition):
    def __init__(self,root,inventory):
        super().__init__(Path(root)/'inputs',inventory['raw_root'])
        self.expected={t['raw_path']:t for t in inventory['tasks']};self.checked={}
    def frame_grid(self,row,fov):
        path=self.movie_path(row,fov)
        if str(path) not in self.checked:
            expected=self.expected[str(path)]
            assert (path.stat().st_size,path.stat().st_mtime_ns)==(expected['raw_size'],expected['raw_mtime_ns'])
            assert sha(expected['xml_path'])==expected['xml_sha256']
            self.checked[str(path)]=expected
        return super().frame_grid(row,fov)
    def audit(self):
        for path,expected in self.checked.items():
            p=Path(path)
            assert (p.stat().st_size,p.stat().st_mtime_ns)==(expected['raw_size'],expected['raw_mtime_ns'])
            assert sha(expected['xml_path'])==expected['xml_sha256']
        return dict(checked_raw_sources=list(self.checked),
            raw_size_mtime_unchanged=True,xml_schedule_hashes_unchanged=True,
            raw_DAX_content_hashed=False)


def load_registration(path):
    path=Path(path);root=path.parent;registration=json.loads(path.read_text())
    if registration['status']!='ACCEPTED_FOR_FULL_SAMPLE':
        raise RuntimeError('Full-sample registration requires an explicit reviewed acceptance')
    protected={str(path):sha(path)}
    for name,digest in registration['required_input_sha256'].items():
        source=Path(name) if Path(name).is_absolute() else root/name
        if sha(source)!=digest:raise RuntimeError('Accepted registration source changed: '+str(source))
        protected[str(source.resolve())]=digest
    inventory_path=(root/'inventory.json').resolve()
    if str(inventory_path) not in protected:
        raise RuntimeError('Accepted registration must protect the frozen inventory')
    inventory=json.loads(inventory_path.read_text())
    if set(map(int,registration['rounds']))!=set(inventory['rounds']):
        raise RuntimeError('Accepted registration does not cover the requested rounds')
    models={}
    for r,entry in registration['rounds'].items():
        if set(entry)!={'within','cross'} or not entry['within'] or (int(r)!=0 and not entry['cross']):
            raise RuntimeError('Incomplete registration mapping for round '+r)
        if int(r)==0 and entry['cross'] is not None:raise RuntimeError('Reference round cannot have a cross transform')
        loaded={}
        for key,value in entry.items():
            source=(root/value).resolve() if value else None
            if source and str(source) not in protected:
                raise RuntimeError('Registration model has no protected input hash: '+str(source))
            loaded[key]=json.loads(source.read_text()) if source else None
        within=loaded['within'];measured=set(map(int,within['models']))
        unsupported=set(within['unsupported_fovs'])
        if measured&unsupported or measured|unsupported!=set(inventory['source_fovs']):
            raise RuntimeError('Every requested source FOV must be accounted for explicitly')
        models[int(r)]=loaded
    return registration,models,protected


def restrict_model(model,moving):
    flat=np.asarray(moving).reshape(-1,3)
    flat=flat[np.isfinite(flat).all(1)]
    if not len(flat):return dict(model,models={})
    lo=flat.min(0);hi=flat.max(0)
    subset={}
    for f,entry in model['models'].items():
        bounds=sampling.source_bounds(model,int(f),(0.,151.))
        if np.all(bounds[1]>=lo) and np.all(bounds[0]<=hi):subset[f]=entry
    return dict(model,models=subset)


def supported_local_inverse(points,model,tolerance_um=1e-6,max_iterations=40,chunk_size=65536):
    """Use the existing fixed-point inverse only inside its audited spline domain.

    Padding outside that domain is unknown, not extrapolated. The initial
    reference point and every iterate must stay inside. This deliberately does
    not recover the thin edge strip whose inverse might exist despite starting
    outside; such support needs a separately validated boundary treatment.
    Nonconvergence of an in-domain query remains an error, not missing data.
    """
    p=np.asarray(points,float);flat=p.reshape(-1,3);result=np.full_like(flat,np.nan)
    grid=model['grid'];origin=np.asarray(grid['origin_xyz_um']);spacing=np.asarray(grid['spacing_xyz_um'])
    low=origin+spacing;high=origin+(np.asarray(grid['shape_xyz'])-3)*spacing
    def inside(x):return np.isfinite(x).all(1)&np.all((x>=low)&(x<=high),axis=1)
    for start in range(0,len(flat),chunk_size):
        reference=flat[start:start+chunk_size];aligned=reference.copy();valid=inside(aligned)
        for _ in range(max_iterations):
            indices=np.flatnonzero(valid)
            if not len(indices):break
            old=aligned[indices]
            updated=reference[indices]-local_crosswarp_v2.residual_displacement(old,model)
            retained=inside(updated);valid[indices[~retained]]=False
            indices=indices[retained];updated=updated[retained];old=old[retained]
            if not len(indices):break
            aligned[indices]=updated
            if np.max(np.abs(updated-old))<tolerance_um:
                result[start+indices]=cross_round.inverse_cross(updated,model['base_model'])
                break
        else:raise RuntimeError('In-domain local cross-warp inverse did not converge')
    return result.reshape(p.shape)


def reference_to_moving(points,cross):
    if cross is None:return np.asarray(points)
    kind=cross.get('model_type','')
    if kind=='global_polynomial_plus_cubic_bspline_residual':
        return supported_local_inverse(points,cross)
    if kind in ('fresh_cross_round_affine','fresh_cross_round_quadratic'):
        return cross_round.inverse_cross(points,cross)
    raise ValueError('Unsupported accepted cross-round transform: '+kind)


def produce_plane(acquisition,bits,models,tile,z_um,fork,decode_halo=0,preprocess_halo=64):
    core=np.asarray(tile['shape_yx'],int)
    shape=core+2*decode_halo
    origin=np.asarray(tile['origin_xy_um'])-decode_halo*sampling.MPP
    grid=pilot.reference_grid(origin,z_um,shape,preprocess_halo)
    pre=pilot.canonical_preprocessor(fork)
    image=np.zeros((len(bits),*shape),np.float32)
    owners=np.zeros((len(bits),*shape),np.uint16)
    complete=np.ones(grid.shape[:2],bool)
    order=sorted(enumerate(bits),key=lambda x:int(acquisition.bit_row(x[1])['imagingRound']))
    current=None;started=time.time();steps=[]
    for bi,bit in order:
        row=acquisition.bit_row(bit);r=int(row['imagingRound'])
        if r!=current:
            entry=models[r]
            moving=reference_to_moving(grid,entry['cross'])
            within=restrict_model(entry['within'],moving);current=r
        raw,valid,owner=sampling.stitch(acquisition,row,moving,within,None)
        complete&=valid
        steps.append(dict(bit=bit,round=r,raw_supported_pixels=int(valid.sum()),
            cross_transform_supported_pixels=int(np.isfinite(moving).all(-1).sum()),
            candidate_source_fovs=sorted(map(int,within['models']))))
        if not complete.any():
            # No location can satisfy every bit after an empty intersection.
            # Arrays remain invalid everywhere; omitted later-bit ownership is
            # explicitly recorded and never interpreted as measured intensity.
            image.fill(0);owners.fill(0)
            return image,np.zeros(tuple(shape),bool),owners,dict(elapsed_seconds=time.time()-started,
                origin_xy_um=origin.tolist(),decode_halo=decode_halo,preprocess_halo=preprocess_halo,
                bit_steps=steps,early_empty_all_bit_support=True)
        processed=pre._preprocess_image(raw)
        image[bi]=processed[preprocess_halo:preprocess_halo+shape[0],preprocess_halo:preprocess_halo+shape[1]]
        owners[bi]=owner[preprocess_halo:preprocess_halo+shape[0],preprocess_halo:preprocess_halo+shape[1]]
    safe=ndimage.minimum_filter(complete,size=2*preprocess_halo+1,mode='constant',cval=0)
    support=safe[preprocess_halo:preprocess_halo+shape[0],preprocess_halo:preprocess_halo+shape[1]].copy()
    image[:,~support]=0
    if not np.isfinite(image).all() or np.any(image<0):raise ValueError('Invalid preprocessed signal')
    if np.any(owners[:,support]==0):raise ValueError('Support contains an unacquired bit')
    return image,support,owners,dict(elapsed_seconds=time.time()-started,origin_xy_um=origin.tolist(),
        decode_halo=decode_halo,preprocess_halo=preprocess_halo,bit_steps=steps,early_empty_all_bit_support=False)
