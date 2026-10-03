"""Disjoint regular output grid enclosing the complete reference acquisition."""
import numpy as np
from full_sampling_radial import source_bounds,MPP

CORE=2048


def build_layout(reference_model):
    bounds={int(f):source_bounds(reference_model,int(f),(0.,151.)) for f in reference_model['models']}
    low=np.min([b[0,:2] for b in bounds.values()],axis=0)
    high=np.max([b[1,:2] for b in bounds.values()],axis=0)
    origin=np.floor(low/MPP).astype(np.int64)
    shape_xy=np.ceil((high/MPP-origin)/CORE).astype(int)
    tiles=[]
    for row in range(int(shape_xy[1])):
        for col in range(int(shape_xy[0])):
            start=origin+np.array([col,row])*CORE
            lo=start*MPP;hi=(start+CORE)*MPP
            sources=[f for f,b in bounds.items() if np.all(b[1,:2]>=lo)&np.all(b[0,:2]<hi)]
            tiles.append(dict(tile_id=len(tiles),grid_row=row,grid_column=col,
                origin_pixel_xy=start.tolist(),origin_xy_um=lo.tolist(),shape_yx=[CORE,CORE],
                nominal_source_fovs=sorted(sources)))
    return dict(origin_pixel_xy=origin.tolist(),shape_tiles_yx=shape_xy[::-1].tolist(),
        tile_shape_yx=[CORE,CORE],microns_per_pixel=MPP,tiles=tiles,
        coordinate_policy='One FOV13/round0 reference pixel lattice, absolute native-mosaic micrometers; not production stage global coordinates.',
        ownership='Half-open regular core rectangles. Decode with surrounding context; retain component centroid in its unique core. Source image ownership is independently determined by native camera-edge distance.',
        conservative_reference_bounds_xy_um=[low.tolist(),high.tolist()])


def owner_for_global_xy(points,layout):
    points=np.asarray(points,float)
    pixel=points/MPP-np.asarray(layout['origin_pixel_xy'])
    # Multiplying an exact tile boundary into micrometers and back can land
    # a few floating-point ulps to its left. Resolve only numerical boundary
    # ambiguity (1e-8 pixel), not meaningful spatial offsets.
    boundary=np.rint(pixel/CORE)*CORE
    pixel=np.where(np.abs(pixel-boundary)<1e-8,boundary,pixel)
    grid=np.floor(pixel/CORE).astype(int)
    rows,cols=layout['shape_tiles_yx']
    valid=(grid[...,0]>=0)&(grid[...,0]<cols)&(grid[...,1]>=0)&(grid[...,1]<rows)
    return np.where(valid,grid[...,1]*cols+grid[...,0],-1)
