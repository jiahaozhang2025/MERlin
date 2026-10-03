"""Preserve raw feature identities when changing stitched coordinates/ownership."""
import numpy as np
from scipy.spatial import cKDTree

def inherit_prior_cross_holdouts(cloud,prior,side,remapped_prior_heldout_xyz=None):
    indexes=prior[side+'_heldout_indices']
    keys=set(zip(prior[side+'_source_fov'][indexes].tolist(),prior[side+'_source_index'][indexes].tolist()))
    inherited=np.array([(int(f),int(i)) in keys for f,i in zip(cloud['source_fov'],cloud['source_index'])])
    if remapped_prior_heldout_xyz is not None and len(indexes):
        points=np.asarray(remapped_prior_heldout_xyz,float)
        if points.shape!=(len(indexes),3):raise ValueError('Prior heldout positions must retain archive order')
        tree=cKDTree(points/np.array([.25,.25,.5]))
        neighbors=tree.query_ball_point(cloud['xyz_um']/np.array([.25,.25,.5]),1.)
        old_fovs=prior[side+'_source_fov'][indexes]
        for i,nn in enumerate(neighbors):
            if nn and np.any(old_fovs[nn]!=cloud['source_fov'][i]):inherited[i]=True
    output={k:np.array(v,copy=True) for k,v in cloud.items()}
    output['prior_cross_round_heldout']=inherited
    output['within_round_heldout']|=inherited
    return output
