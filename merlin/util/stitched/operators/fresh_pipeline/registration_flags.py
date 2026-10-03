"""Annotate whole barcode components against retained registration uncertainty."""
import json
from pathlib import Path
import numpy as np
from cross_round import inverse_cross

def annotate_registration(frame,labels,z_um,origin_xy_um,registration,cross_models,mpp=.1493):
    frame=frame.copy();ids=frame.unique_id.to_numpy(dtype=int)
    hits={int(i):[] for i in ids};round_hits={int(i):set() for i in ids}
    kinds={int(i):set() for i in ids};counts={}
    regions=registration.get('uncertainty_regions',[])
    if len(ids) and regions:
        yy,xx=np.nonzero((labels>0)&np.isin(labels,ids));label_ids=labels[yy,xx]
        points=np.column_stack([origin_xy_um[0]+xx*mpp,origin_xy_um[1]+yy*mpp,np.full(len(xx),z_um)])
        for r in sorted({int(x['round_id']) for x in regions}):
            moving=points if r==0 else inverse_cross(points,cross_models[r])
            for region in [x for x in regions if int(x['round_id'])==r]:
                lo,hi=np.asarray(region['global_bounds_min_max_xyz_um'],float)
                if np.any(hi<lo):raise ValueError('Uncertainty region has reversed bounds')
                touched=np.unique(label_ids[np.all((moving>=lo)&(moving<=hi),axis=1)])
                counts[region['region_id']]=len(touched)
                for label in touched:
                    hits[int(label)].append(region['region_id']);round_hits[int(label)].add(r);kinds[int(label)].add(region['kind'])
    frame['registration_uncertainty']=[bool(hits[int(i)]) for i in ids]
    frame['registration_uncertainty_regions']=['|'.join(hits[int(i)]) for i in ids]
    frame['registration_uncertainty_rounds']=['|'.join(str(r) for r in sorted(round_hits[int(i)])) for i in ids]
    frame['registration_centroid_stratum_flag']=['heldout_centroid_stratum_failed' in kinds[int(i)] for i in ids]
    frame['registration_ambiguous_image_flag']=[any('ambiguous' in k for k in kinds[int(i)]) for i in ids]
    return frame,dict(registration_uncertain_barcodes=int(frame.registration_uncertainty.sum()),
        uncertainty_region_component_counts=counts,
        registration_flag_rule='Any retained component pixel, mapped into the relevant round stitchedXYZ, intersects a retained conservative uncertainty box; flags do not reject barcodes.')

def load_registration(path,sha256):
    path=Path(path);registration=json.loads(path.read_text());base=path.parent
    if registration['status']!='ACCEPTED_FOR_SINGLE_FOV_PILOT':raise RuntimeError('Registration is not accepted for this pilot')
    protected={str(path):sha256(path)};cross={}
    for key,expected in registration['required_input_sha256'].items():
        source=Path(key);source=source if source.is_absolute() else base/source
        if sha256(source)!=expected:raise RuntimeError(f'Accepted registration source changed: {source}')
        protected[str(source)]=expected
    for r,entry in registration['rounds'].items():
        if entry.get('cross') is not None:
            source=Path(entry['cross']);source=source if source.is_absolute() else base/source
            cross[int(r)]=json.loads(source.read_text())
    needed={int(x['round_id']) for x in registration.get('uncertainty_regions',[]) if x['round_id']!=0}
    if not needed.issubset(cross):raise RuntimeError('Missing cross-round map for uncertainty annotation')
    return registration,cross,protected
