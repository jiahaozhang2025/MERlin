"""Smooth cubic B-spline residual after a fresh global cross-round transform.

Training-only regularized robust fitting. Evaluation/inversion preserve a
continuous global map; this module never grants alignment acceptance.
"""
import itertools
import numpy as np
from scipy import ndimage,sparse
from scipy.sparse.linalg import lsmr
import cross_round as cr


def weights(t):
    return np.stack([(1-t)**3,3*t**3-6*t*t+4,-3*t**3+3*t*t+3*t+1,t**3],axis=-1)/6


def make_grid(points,spacing=(100.,100.,50.)):
    spacing=np.asarray(spacing,float)
    low=(np.floor(np.min(points,axis=0)/spacing)-3)*spacing
    high=(np.ceil(np.max(points,axis=0)/spacing)+3)*spacing
    shape=np.rint((high-low)/spacing).astype(int)+1
    return dict(origin_xyz_um=low.tolist(),spacing_xyz_um=spacing.tolist(),shape_xyz=shape.tolist())


def design(points,grid):
    p=np.asarray(points,float);q=(p-np.asarray(grid['origin_xyz_um']))/grid['spacing_xyz_um']
    base=np.floor(q).astype(int);t=q-base;ww=[weights(t[:,axis]) for axis in range(3)]
    shape=np.asarray(grid['shape_xyz'],int)
    if np.any(base<1) or np.any(base+2>=shape):raise ValueError('Training points outside spline interior')
    rows=[];cols=[];values=[]
    for i,j,k in itertools.product(range(4),repeat=3):
        idx=base+np.array([i,j,k])-1
        rows.append(np.arange(len(p)));cols.append(np.ravel_multi_index(idx.T,shape))
        values.append(ww[0][:,i]*ww[1][:,j]*ww[2][:,k])
    return sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(p),int(np.prod(shape)))).tocsr()


def curvature_matrix(grid):
    shape=tuple(grid['shape_xyz']);indexes=np.arange(np.prod(shape)).reshape(shape)
    rows=[];cols=[];vals=[];offset=0
    for axis in range(3):
        slices=[slice(None)]*3;slices[axis]=slice(1,-1);middle=indexes[tuple(slices)].ravel()
        stride=int(np.prod(shape[axis+1:]))
        n=len(middle)
        for delta,value in [(-stride,1.),(0,-2.),(stride,1.)]:
            rows.append(np.arange(n)+offset);cols.append(middle+delta);vals.append(np.full(n,value))
        offset+=n
    return sparse.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(offset,int(np.prod(shape)))).tocsr()


def fit_residual(reference,moving,base_model,grid=None,spacing=(100.,100.,50.),initial_model=None,max_iterations=80):
    aligned=cr.apply_cross(np.asarray(moving,float),base_model)
    target=np.asarray(reference,float)-aligned
    grid=make_grid(aligned,spacing) if grid is None else grid
    a=design(aligned,grid);curvature=curvature_matrix(grid)
    n=a.shape[1];coef=np.zeros((n,3));history=[]
    if max_iterations<4:raise ValueError('At least four IRLS iterations are required')
    if initial_model is not None:
        if initial_model['grid']!=grid or initial_model['base_model']!=base_model:raise ValueError('Warm start must use the same grid and base transform')
        coef=np.asarray(initial_model['coefficients_xyz_um'],float).reshape(n,3).copy()
        if not np.isfinite(coef).all():raise ValueError('Nonfinite warm start')
    residual_scale=np.array([cr.MPP,cr.MPP,.5])
    # Fixed priors selected before validation: curvature0.2um/control-cell,
    # and weak3um displacement amplitude. They stabilize sparse outer nodes.
    prior=sparse.vstack([curvature/.2,sparse.eye(n,format='csr')/3],format='csr')
    for iteration in range(max_iterations):
        previous=coef.copy();scaled=(a@coef-target)/residual_scale
        for d in range(3):
            weight=(1+scaled[:,d]**2)**(-.25)/residual_scale[d]
            system=sparse.vstack([a.multiply(weight[:,None]),prior],format='csr')
            rhs=np.r_[target[:,d]*weight,np.zeros(prior.shape[0])]
            result=lsmr(system,rhs,atol=1e-8,btol=1e-8,maxiter=1500,x0=coef[:,d])
            coef[:,d]=result[0]
            if result[1] not in (0,1,2,4,5):raise RuntimeError('Spline least-squares did not converge: '+str(result[1]))
        delta=float(np.max(np.abs(coef-previous)));residual=a@coef-target
        history.append(dict(iteration=iteration,maximum_coefficient_change_um=delta,training=cr.stats(residual)))
        if iteration>=3 and delta<1e-4:break
    model=dict(model_type='global_polynomial_plus_cubic_bspline_residual',base_model=base_model,grid=grid,
        coefficients_xyz_um=coef.reshape((*grid['shape_xyz'],3)).tolist(),
        residual_definition='reference = base(moving) + cubic_Bspline(base(moving)); spatially continuous XYZ map',
        fit=dict(robust_loss='soft_l1 via IRLS',iterations=len(history),history=history,
            curvature_prior_std_um=.2,displacement_prior_std_um=3.,residual_scale_xyz_um=residual_scale.tolist(),
            maximum_iterations=max_iterations,warm_started=initial_model is not None,converged=delta<1e-4),
        scientific_status='EXPLORATORY_REQUIRES_HELDOUT_RAW_IMAGE_QA')
    return model


def residual_displacement(aligned,model):
    p=np.asarray(aligned,float);shape=p.shape
    q=((p.reshape(-1,3)-model['grid']['origin_xyz_um'])/model['grid']['spacing_xyz_um']).T
    coefficients=np.asarray(model['coefficients_xyz_um'],float)
    if np.any(q<1) or np.any(q>np.asarray(model['grid']['shape_xyz'])[:,None]-3):
        raise ValueError('Query outside audited local-warp spline interior')
    result=np.stack([ndimage.map_coordinates(coefficients[...,d],q,order=3,prefilter=False,mode='constant',cval=np.nan) for d in range(3)],axis=-1)
    if not np.isfinite(result).all():raise RuntimeError('Nonfinite spline displacement')
    return result.reshape(shape)


def apply(points,model):
    aligned=cr.apply_cross(points,model['base_model'])
    return aligned+residual_displacement(aligned,model)


def inverse(points,model,tolerance_um=1e-6,max_iterations=40):
    p=np.asarray(points,float);aligned=p.copy()
    if not p.size:return p.copy()
    for _ in range(max_iterations):
        update=p-residual_displacement(aligned,model)
        change=float(np.max(np.abs(update-aligned)));aligned=update
        if change<tolerance_um:return cr.inverse_cross(aligned,model['base_model'])
    raise RuntimeError('Local cross-warp inverse did not converge')


def geometry(points,model):
    p=np.asarray(points,float)
    jj=np.stack([(apply(p+np.eye(3)[d]*.01,model)-apply(p-np.eye(3)[d]*.01,model))/.02 for d in range(3)],axis=-1)
    det=np.linalg.det(jj);sv=np.linalg.svd(jj,compute_uv=False);change=np.linalg.svd(jj-np.eye(3),compute_uv=False)[...,0]
    error=float(np.max(np.linalg.norm(inverse(apply(p,model),model)-p,axis=-1)))
    # Same cross-round plausibility limits as the reviewed global model.
    passed=det.min()>.65 and det.max()<1.5 and sv.min()>.7 and sv.max()<1.35 and change.max()<.4 and jj[...,2,2].min()>.7 and error<1e-5
    return dict(passed=bool(passed),determinant_range=[float(det.min()),float(det.max())],singular_value_range=[float(sv.min()),float(sv.max())],
        maximum_displacement_jacobian_norm=float(change.max()),axial_derivative_range=[float(jj[...,2,2].min()),float(jj[...,2,2].max())],inverse_error_max_um=error)
