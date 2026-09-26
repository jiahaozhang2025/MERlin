"""Seven-term displacement plus shared radial XY camera correction."""
import numpy as np
import global_quadratic as legacy
import fit_graph as fg

basis=legacy.basis
FEATURE_NAMES=legacy.FEATURE_NAMES

def radial(points,model):
    p=np.asarray(points,float);q=(p-np.asarray(model['normalization']['center_xyz_um']))/model['normalization']['scale_xyz_um']
    out=np.zeros(p.shape);r2=q[...,0]**2+q[...,1]**2
    out[...,:2]=q[...,:2]*r2[...,None]*model.get('radial_xy_coefficient_um',0.)
    return out

def apply_global(points,fov,model):return legacy.apply_global(points,fov,model)+radial(points,model)

def inverse_global(points,fov,model,tolerance_um=1e-7,max_iterations=40):
    points=np.asarray(points,float);origin=np.asarray(model['models'][str(fov)]['stage_origin_xyz_um']);target=points-origin;native=target.copy()
    if not points.size:return native
    for _ in range(max_iterations):
        correction=apply_global(native,fov,model)-native-origin
        updated=target-correction;error=np.max(np.abs(updated-native));native=updated
        if error<tolerance_um:return native
    raise RuntimeError('Camera-map inverse did not converge')

def jacobian(points,fov,model):
    p=np.asarray(points,float);j=legacy.jacobian(p,fov,model)
    scale=np.asarray(model['normalization']['scale_xyz_um']);q=(p-model['normalization']['center_xyz_um'])/scale;k=model.get('radial_xy_coefficient_um',0.)
    j[...,0,0]+=k*(3*q[...,0]**2+q[...,1]**2)/scale[0]
    j[...,1,0]+=k*2*q[...,0]*q[...,1]/scale[0]
    j[...,0,1]+=k*2*q[...,0]*q[...,1]/scale[1]
    j[...,1,1]+=k*(q[...,0]**2+3*q[...,1]**2)/scale[1]
    return j

def assess(clouds,edges,model,archive,cfg):
    reasons=[];evidence=[]
    for e in edges:
        a,b=e['k'],e['n'];prefix=f'{a}_{b}_';record={k:v for k,v in e.items() if k not in ['training_pairs','heldout_pairs']}
        for split in ['training','heldout']:
            pair=e[split+'_pairs'];aa=clouds[a][pair[:,0]];bb=clouds[b][pair[:,1]]
            residual=apply_global(aa,a,model)-apply_global(bb,b,model);metric=fg.metrics(residual,min(e['spatial_split'][split+'_counts']),cfg)
            record['quadratic_'+split]=metric
            archive[prefix+'quadratic_'+split+'_residual_xyz_um']=residual
            archive[prefix+split+'_native_a_xyz_um']=aa;archive[prefix+split+'_native_b_xyz_um']=bb
            minimum=cfg['min_fit_pairs'] if split=='training' else cfg['min_holdout_pairs']
            reasons.extend(f'{a}-{b} '+r for r in fg.gate_metrics(metric,split,minimum,cfg,split=='heldout'))
        evidence.append(record)
    width=cfg['image_width_pixels']*cfg['microns_per_pixel']
    grid=np.stack(np.meshgrid(np.linspace(0,width,17),np.linspace(0,width,17),np.linspace(0,150,5),indexing='ij'),axis=-1).reshape(-1,3)
    geometries={}
    for f in map(int,model['models']):
        j=jacobian(grid,f,model);det=np.linalg.det(j);sv=np.linalg.svd(j,compute_uv=False);contraction=np.linalg.svd(j-np.eye(3),compute_uv=False)[:,0]
        error=float(np.max(np.linalg.norm(inverse_global(apply_global(grid,f,model),f,model)-grid,axis=1)))
        passed=det.min()>.95 and det.max()<1.05 and sv.min()>.97 and sv.max()<1.03 and contraction.max()<.05 and error<1e-5
        geometries[str(f)]=dict(determinant_min=float(det.min()),determinant_max=float(det.max()),singular_value_min=float(sv.min()),singular_value_max=float(sv.max()),
            displacement_jacobian_norm_max=float(contraction.max()),inverse_error_max_um=error,passed=bool(passed),sample_grid='17x17XY,5Z over full camera0..150um')
        if not passed:reasons.append(f'FOV{f} camera-wide Jacobian or inverse gate failed')
    if not model['optimization']['success']:reasons.append('Optimizer did not converge')
    return dict(status='EXPLORATORY_ALIGNMENT_PASS' if not reasons else 'FAIL',reasons=reasons,edges=evidence,camera_grid_jacobians=geometries,
        frozen_jacobian_gates=dict(determinant=[.95,1.05],singular_values=[.97,1.03],displacement_jacobian_norm_max=.05,inverse_error_max_um=1e-5))
