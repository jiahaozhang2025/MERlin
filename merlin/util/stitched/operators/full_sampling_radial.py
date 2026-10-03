"""One-pass native interpolation with exact source ownership across any FOV graph."""
import numpy as np
from scipy import ndimage
import camera_geometry as gq
import stitched_sampler as pilot

MPP = pilot.MPP


def source_bounds(model, fov, z_limits=(1.,150.)):
    """Conservative polynomial interval enclosure; used only for candidate pruning."""
    center=np.asarray(model['normalization']['center_xyz_um'],float)
    scale=np.asarray(model['normalization']['scale_xyz_um'],float)
    native=np.array([[0.,0.,z_limits[0]],[2047*MPP,2047*MPP,z_limits[1]]])
    q=(native-center)/scale
    def squared(lo,hi):
        return (0. if lo<=0<=hi else min(lo*lo,hi*hi),max(lo*lo,hi*hi))
    xx=squared(q[0,0],q[1,0]);yy=squared(q[0,1],q[1,1])
    xy=np.array([q[i,0]*q[j,1] for i in (0,1) for j in (0,1)])
    feature=np.array([[1.,*q[0],xx[0],yy[0],xy.min()],
                      [1.,*q[1],xx[1],yy[1],xy.max()]])
    entry=model['models'][str(fov)];beta=np.asarray(entry['coefficients_7x3_um'],float)
    products=feature[:,:,None]*beta[None,:,:]
    lo=native[0]+np.asarray(entry['stage_origin_xyz_um'])+products.min(axis=0).sum(axis=0)
    hi=native[1]+np.asarray(entry['stage_origin_xyz_um'])+products.max(axis=0).sum(axis=0)
    radial=np.zeros(3)
    qmax=np.max(np.abs(q),axis=0)
    radial[:2]=abs(model.get('radial_xy_coefficient_um',0.))*qmax[:2]*(qmax[0]**2+qmax[1]**2)
    return np.stack([lo-radial-1e-7,hi+radial+1e-7])


def choose_sources(moving_xyz, model, z_limits=None):
    """Pick maximum native camera-edge distance, breaking ties by sorted FOV id."""
    shape=np.asarray(moving_xyz).shape[:-1]
    points=np.asarray(moving_xyz).reshape(-1,3)
    owner=np.zeros(len(points),np.uint16)
    score=np.full(len(points),-np.inf)
    selected_native=np.full(points.shape,np.nan)
    for fov in sorted(map(int,model['models'])):
        limits=(1.,150.) if z_limits is None else z_limits[fov]
        bounds=source_bounds(model,fov,limits)
        possible=np.flatnonzero(np.all((points>=bounds[0])&(points<=bounds[1]),axis=1))
        if not len(possible):
            continue
        native=gq.inverse_global(points[possible],fov,model)
        xy=pilot.pixel_xy(native)
        valid=np.isfinite(native).all(1)&np.all((xy>=0)&(xy<=2047),axis=1)
        valid&=(native[:,2]>=limits[0])&(native[:,2]<=limits[1])
        edge=np.min(np.c_[xy,2047-xy],axis=1)
        take=valid&(edge>score[possible])
        indexes=possible[take]
        score[indexes]=edge[take];owner[indexes]=fov+1;selected_native[indexes]=native[take]
    return owner.reshape(shape),selected_native.reshape((*shape,3))


class NativeAcquisition(pilot.NativeAcquisition):
    def sample(self,row,fov,native_xyz):
        """Exactly the pilot's linear XYZ sampling, loading only the needed XY patch."""
        shape=native_xyz.shape[:-1];points=np.asarray(native_xyz).reshape(-1,3)
        frames,zs=self.frame_grid(row,fov);xy=pilot.pixel_xy(points)
        valid=np.isfinite(points).all(1)&np.all((xy>=0)&(xy<=2047),axis=1)
        valid&=(points[:,2]>=zs[0])&(points[:,2]<=zs[-1])
        output=np.zeros(len(points),np.float32)
        if np.any(valid):
            indexes=np.flatnonzero(valid);pp=points[valid];coords=xy[valid]
            upper=np.clip(np.searchsorted(zs,pp[:,2],side='right'),1,len(zs)-1);lower=upper-1
            fraction=(pp[:,2]-zs[lower])/(zs[upper]-zs[lower]);values=np.zeros(len(pp),np.float32)
            for levels,weights in [(lower,1-fraction),(upper,fraction)]:
                for zi in np.unique(levels[weights>0]):
                    take=(levels==zi)&(weights>0);sample_xy=coords[take]
                    lo=np.floor(sample_xy.min(0)).astype(int)
                    hi=np.minimum(np.floor(sample_xy.max(0)).astype(int)+2,2048)
                    patch=self.plane(row,fov,frames[zi])[lo[1]:hi[1],lo[0]:hi[0]].astype(np.float32)
                    sampled=ndimage.map_coordinates(patch,[sample_xy[:,1]-lo[1],sample_xy[:,0]-lo[0]],
                        order=1,mode='constant',cval=0,prefilter=False)
                    values[take]+=sampled*weights[take]
            output[indexes]=values
        return output.reshape(shape),valid.reshape(shape)


def stitch(acquisition,row,reference_xyz,within,cross=None):
    if cross is None:
        moving=np.asarray(reference_xyz)
    else:
        import cross_round
        moving=cross_round.inverse_cross(reference_xyz,cross)
    limits={f:(zs[0],zs[-1]) for f in map(int,within['models'])
            for _,zs in [acquisition.frame_grid(row,f)]}
    owner,native=choose_sources(moving,within,limits)
    image=np.zeros(owner.shape,np.float32)
    for code in np.unique(owner):
        if code==0:
            continue
        take=owner==code
        values,valid=acquisition.sample(row,int(code)-1,native[take])
        assert valid.all()
        image[take]=values
    return image,owner>0,owner
