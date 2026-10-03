"""Map archived mask centers through any supported same-catalog reference."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

def map_fragments(cells,model,setup,apply_global):
    from partition_masks import candidate_catalog
    reverse={f:[] for f in range(52)}
    for ref,entries in candidate_catalog(setup['offsets']).items():
        if str(ref) in model['models']:
            for seg,tx,ty,how in entries:reverse[seg].append((ref,tx,ty,how))
    frames=[];width=2047*.1493
    for seg,part in cells.groupby('stack'):
        part=part.copy();n=len(part);xy=part[['cx_ds','cy_ds']].to_numpy()*2*.1493
        z=part.cz_ds.to_numpy()+1-setup['offsets'].get('dz_global',0);candidates=[]
        for ref,tx,ty,how in reverse[seg]:
            native=np.c_[xy+(np.array(setup['crops'][str(ref)])-[tx,ty])*.1493,z]
            valid=np.isfinite(native).all(1)&(native[:,2]>=1)&(native[:,2]<=149)&np.all((native[:,:2]>=0)&(native[:,:2]<=width),axis=1)
            margin=np.minimum.reduce([native[:,0],width-native[:,0],native[:,1],width-native[:,1]])
            mapped=np.full((n,3),np.nan);mapped[valid]=apply_global(native[valid],ref,model)
            score=np.where(valid,margin+(1e4 if ref==seg else 0),-np.inf)
            candidates.append((ref,how,native,valid,mapped,score))
        coords=np.full((n,3),np.nan);refs=np.full(n,-1,int);nv=np.zeros(n,int);disagreement=np.full(n,np.nan)
        if candidates:
            scores=np.stack([c[5] for c in candidates]);best=np.argmax(scores,axis=0);ok=np.isfinite(scores.max(0))
            nv=np.stack([c[3] for c in candidates]).sum(0)
            for i,c in enumerate(candidates):
                use=ok&(best==i);coords[use]=c[4][use];refs[use]=c[0]
            distances=np.stack([np.linalg.norm(c[4][:,:2]-coords[:,:2],axis=1) for c in candidates])
            disagreement[ok]=np.nanmax(distances[:,ok],axis=0)
        part[['global_x','global_y','global_z']]=coords;part['reference_fov']=refs
        part['candidate_references']=nv;part['max_candidate_xy_disagreement_um']=disagreement
        part['used_neighbor']=(refs>=0)&(refs!=seg);frames.append(part)
    return pd.concat(frames,ignore_index=True)

def run(root,config):
    import camera_geometry
    root=Path(root);partition=root/'partition_sjoin_v2';out=partition/'cell_coordinates_v3'
    if (out/'complete.json').exists():return
    out.mkdir(exist_ok=False)
    setup=json.loads((partition/'setup.json').read_text());model=json.loads(Path(setup['model']).read_text())
    archive=Path(config['mask_archive']);fr=pd.read_csv(partition/'exports/fragments_cell_ids.csv')
    sources=[];pieces=[]
    for f in range(52):
        p=archive/f'segmentation/cells/cells_{f:02d}.csv';sources.append(p)
        cell=pd.read_csv(p);cell['stack']=f;pieces.append(cell)
    cells=pd.concat(pieces,ignore_index=True).merge(fr,on=['stack','label'],validate='one_to_one')
    mapped=map_fragments(cells,model,setup,camera_geometry.apply_global)
    mapped.to_csv(out/'fragment_center_mapping.csv',index=False)
    old=pd.read_csv(partition/'exports/feature_metadata.csv',index_col='feature_id');meta=old.copy()
    good=mapped.reference_fov>=0;valid=mapped.loc[good].copy()
    assert np.isfinite(valid[['global_x','global_y','global_z']]).all().all()
    weights=valid.groupby('cell_uid').nvox.sum();columns=['stitched_center_x_um','stitched_center_y_um','stitched_center_z_um']
    for src,dest in zip(['global_x','global_y','global_z'],columns):
        sums=(valid[src]*valid.nvox).groupby(valid.cell_uid).sum();meta[dest]=(sums/weights).reindex(meta.index)
    meta['stitched_center_supported']=meta[columns].notna().all(axis=1)
    grouped=valid.groupby('cell_uid')
    meta['center_neighbor_fragment_count']=grouped.used_neighbor.sum().reindex(meta.index,fill_value=0).astype(int)
    meta['center_supported_fragment_count']=grouped.size().reindex(meta.index,fill_value=0).astype(int)
    meta['center_max_candidate_xy_disagreement_um']=grouped.max_candidate_xy_disagreement_um.max().reindex(meta.index)
    meta['center_mapping_review_needed']=meta.center_max_candidate_xy_disagreement_um>5
    meta['center_coordinate_version']='overlapping_reference_v3'
    keep=[c for c in old if c not in columns+['stitched_center_supported']]
    assert meta[keep].equals(old[keep]);assert not (old.stitched_center_supported&~meta.stitched_center_supported).any()
    meta.to_csv(out/'feature_metadata.csv')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    (out/'complete.json').write_text(json.dumps(dict(status='COMPLETE',cells=len(meta),supported=int(meta.stitched_center_supported.sum()),
        flagged=int(meta.center_mapping_review_needed.sum()),ids_counts_and_legacy_fields_unchanged=True,output_sha256=hashes,
        limitations='Reuses existing mask offsets; >5um candidate disagreement is flagged, not repaired. Unsupported coordinates remain missing.'),indent=2))
