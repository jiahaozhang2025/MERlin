"""Deterministic spatial/depth sampling independent of image intensity/barcodes."""
import numpy as np

def select_planes(records,training_count=50,validation_count=12,minimum_support=.1):
    candidates=sorted([dict(r) for r in records if r['coarse_all_bit_fraction']>=minimum_support],key=lambda r:(r['tile_id'],r['z_index']))
    if len({(r['tile_id'],r['z_index']) for r in candidates})!=len(candidates):raise ValueError('Duplicate candidate tile/depth')
    required=training_count+validation_count
    if len(candidates)<required:raise ValueError('Insufficient supported planes for fixed calibration design')
    coords=np.array([[*r['center_xy_um'],r['physical_z_um']] for r in candidates],float)
    coords=(coords-coords.min(0))/np.maximum(np.ptp(coords,axis=0),1.)
    # First near the sample center, then maximize distance to existing picks.
    first=int(np.argmin(np.linalg.norm(coords-.5,axis=1)));chosen=[];distance=np.full(len(coords),np.inf)
    for _ in range(required):
        index=first if not chosen else int(np.argmax(distance))
        chosen.append(index);distance=np.minimum(distance,np.linalg.norm(coords-coords[index],axis=1));distance[chosen]=-np.inf
    result=[]
    for n,index in enumerate(chosen):
        r=candidates[index];r.update(sample_id=n,role='validation' if n<validation_count else 'train');result.append(r)
    return result
