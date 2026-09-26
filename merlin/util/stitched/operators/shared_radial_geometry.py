"""Alternative within-round model: common camera curvature plus small local affine terms.

The anchor center at Z0 defines the origin; the entire anchor camera is not
forced to be distortion-free. This is a training-only model hypothesis.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
import camera_geometry as gq
import fit_graph as fg

def fit_model(clouds,edges,cfg):
    fovs=sorted(clouds);unknown=[f for f in fovs if f!=13];nf=len(fovs)
    width=cfg['image_width_pixels']*cfg['microns_per_pixel'];center=np.array([width/2,width/2,75.]);scale=np.array([100.,100.,50.])
    nt=len(unknown)*3;na=nf*9;ncols=nt+na+10
    def column(f,feature,d):
        if feature==0:return None if f==13 else unknown.index(f)*3+d
        if feature<=3:return nt+fovs.index(f)*9+(feature-1)*3+d
        return nt+na+(feature-4)*3+d
    rs=np.array([cfg['microns_per_pixel']]*2+[1.]);rows=[];cols=[];values=[];base=[];offset=0
    for e in edges:
        a,b=e['k'],e['n'];pair=e['training_pairs'];aa=clouds[a][pair[:,0]];bb=clouds[b][pair[:,1]]
        base.extend(((aa+fg.seed_origin(a)-bb-fg.seed_origin(b))/rs).ravel())
        for f,pp,sign in [(a,aa,1.),(b,bb,-1.)]:
            feature=gq.basis(pp,center,scale)
            for j in range(7):
                for d in range(3):
                    ci=column(f,j,d)
                    if ci is None:continue
                    rows.extend(offset+3*np.arange(len(pp))+d);cols.extend([ci]*len(pp));values.extend(sign*feature[:,j]/rs[d])
        for f,pp,sign in [(a,aa,1.),(b,bb,-1.)]:
            q=(pp-center)/scale;r2=q[:,0]**2+q[:,1]**2
            for d in range(2):
                rows.extend(offset+3*np.arange(len(pp))+d);cols.extend([ncols-1]*len(pp));values.extend(sign*q[:,d]*r2/rs[d])
        offset+=len(pair)*3
    # Physical-camera hypothesis: field-specific linear deformation is small;
    # common quadratic curvature is available in every camera, including13.
    linear_prior=.003
    for f in fovs:
        for feature in range(1,4):
            for d in range(3):
                rows.append(offset);cols.append(column(f,feature,d));values.append(1/(scale[feature-1]*linear_prior));base.append(0.);offset+=1
    for feature in range(4,7):
        for d in range(3):
            rows.append(offset);cols.append(column(13,feature,d));values.append(1.);base.append(0.);offset+=1
    rows.append(offset);cols.append(ncols-1);values.append(2.);base.append(0.);offset+=1
    a=sparse.coo_matrix((values,(rows,cols)),shape=(offset,ncols)).tocsr();base=np.asarray(base)
    fit=least_squares(lambda x:a@x+base,np.zeros(ncols),jac=lambda x:a.copy(),loss='soft_l1',f_scale=1.,
        ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=250,tr_solver='lsmr',tr_options={'atol':1e-11,'btol':1e-11,'maxiter':max(1000,4*ncols)})
    model={'model_type':'shared_camera_radial_plus_quadratic_and_fov_affine','radial_xy_coefficient_um':float(fit.x[-1]),'anchor_fov':13,'coordinate_units':'micrometers','coordinate_order':'xyz',
        'normalization':{'center_xyz_um':center.tolist(),'scale_xyz_um':scale.tolist()},'features':gq.FEATURE_NAMES,'models':{},
        'optimization':{'success':bool(fit.success),'message':str(fit.message),'cost':float(fit.cost),'evaluations':int(fit.nfev),'coefficients':ncols,
            'loss':'soft_l1','linear_prior_fraction':linear_prior,'shared_quadratic_prior_um':1.,'radial_prior_um':.5,'residual_scale_xyz_um':rs.tolist(),
            'hypothesis':'Common radial camera correction and quadratic curvature; small field-specific affine deformation; stage-relative translations. No full-camera anchor identity constraint.'}}
    for f in fovs:
        beta=np.zeros((7,3))
        for j in range(7):
            for d in range(3):
                ci=column(f,j,d)
                if ci is not None:beta[j,d]=fit.x[ci]
        model['models'][str(f)]={'stage_origin_xyz_um':fg.seed_origin(f).tolist(),'coefficients_7x3_um':beta.tolist()}
    anchor=np.array([width/2,width/2,0.]);delta=gq.apply_global(anchor,13,model)-anchor
    for e in model['models'].values():
        beta=np.asarray(e['coefficients_7x3_um']);beta[0]-=delta;e['coefficients_7x3_um']=beta.tolist()
    model['gauge']='Anchor13 camera center at nativeZ0 fixed; camera shape can be corrected like other FOVs.'
    return model
