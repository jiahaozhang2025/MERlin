from stitched_config import ROOT, CONFIG
"""Isolated, restart-explicit 0718 GeoPandas partition pipeline.

Stages: setup, geometry[FOV], coordinates, join[reference FOV], export, verify.
"""
import gzip, hashlib, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from partition_masks import candidate_catalog
from partition_sjoin_core import mask_polygons, reference_polygons, spatial_pairs, first_hosts, resolve

ROOT = ROOT
OLD = Path(CONFIG['mask_archive'])
OUT = ROOT/'partition_sjoin_v2'
BARCODES = ROOT/'full_filter_v1/exports/adaptive_filtered_z_deduplicated.csv.gz'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def save(p, data):
    p = Path(p); temp = p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)); os.replace(temp, p)


def read(p): return json.loads(Path(p).read_text())


def log(**kw): print(json.dumps(kw), flush=True)


def context():
    s = read(OUT/'setup.json')
    for p, h in s['protected'].items():
        # Big mask and barcode files have separate stage-end hash verification.
        assert sha(p) == h, p
    return s


def setup():
    import geopandas, shapely, rasterio
    assert not (OUT/'setup.json').exists()
    assert read(OUT/'validation.json')['passed'] and len(read(OUT/'validation.json')['synthetic']) == 15
    for name in ['geometry','coordinates','joins','exports']:
        (OUT/name).mkdir(exist_ok=False)
    protected = {}
    def protect(p):
        protected[str(p)] = sha(p); return p
    for n in ['run_partition_sjoin.py','partition_sjoin_core.py','partition_masks.py','camera_geometry.py',
              'full_sampling_radial.py','global_quadratic.py','stitched_sampler.py','fit_graph.py',
              'validate_partition_sjoin.py']:
        protect(ROOT/'scripts'/n)
    for n in ['partition.py','../util/spatialfeature.py']:
        protect(Path(CONFIG['canonical_root'])/'merlin/analysis'/n)
    reg = read(protect(ROOT/'registration_accepted_radial_v1.json'))
    assert reg['status'] == 'ACCEPTED_FOR_FULL_SAMPLE'
    model = str(protect(ROOT/reg['rounds']['0']['within']))
    export = read(protect(ROOT/'full_filter_v1/exports/complete.json'))
    offsets = read(protect(OLD/'segmentation/seg_offsets_final.json'))
    aff = read(protect(OLD/'segmentation/aff_constants.json'))
    pos = np.loadtxt(protect(OLD/'data/positions.csv'), delimiter=',')
    book = pd.read_csv(protect(ROOT/'full_filter_v1/exports/codebook.csv'))
    fragments = pd.read_csv(protect(OLD/'segmentation/fragments_true_v71.csv'))
    cells = pd.read_csv(protect(OLD/'segmentation/cells_final_true_v71.csv'))
    assert list(cells.cell_uid) == list(range(1, len(cells)+1))
    assert not fragments.duplicated(['stack','label']).any()
    assert set(fragments.cell_uid) == set(cells.cell_uid) and len(book) > 0
    crops = {}
    for f in range(52):
        c = (np.array(aff[str(f)][:2])-pos[f])/.1493
        assert np.max(np.abs(c-np.rint(c))) < .001
        crops[str(f)] = np.rint(c).tolist()
        protect(OLD/f'segmentation/cells/cells_{f:02d}.csv')
    s = dict(status='SETUP', protected=protected, model=model, crops=crops,
             versions=dict(python=sys.version,geopandas=geopandas.__version__,shapely=shapely.__version__,
                           rasterio=rasterio.__version__,numpy=np.__version__,pandas=pd.__version__),
             ncell=len(cells), ncodewords=len(book), rows=export['deduplicated_barcodes'], offsets=offsets,
             input_barcode_sha256=export['output_sha256'][BARCODES.name], buffer_um=.5,
             seed=7182026, source_post_dedup_blank_estimate=export['estimated_rate_after_dedup'],
             method='Per-native-depth GeoPandas within joins; union same cell ID across neighboring mask fragments; MERlin 0.5um buffer decisions with sample-wide interior evidence; one output row per input barcode.',
             limitations=['Existing segmentation offsets are reused, including tissue/analytic/fallback methods; not newly registered.',
              'Masks are converted to exact pixel-footprint polygons, not regenerated MERlin segmentation contours.',
              'Buffer distance is 0.5um in native physical XY; no Z dilation or endpoint clipping.',
              'Shared archived cell IDs are unioned per reference view; sample-wide interior evidence replaces redundant per-FOV evidence.',
              'Unsupported round0 mappings or native depths outside 1..149um remain unassigned.'])
    save(OUT/'setup.json', s); log(stage='setup', rows=s['rows'], cells=s['ncell'])


def geometry(f):
    s = context(); path = OUT/'geometry'/f'{f:02d}.pkl'; assert not path.exists()
    src = OLD/f'segmentation/masks/masks_{f:02d}.npz'; digest = sha(src)
    fr = pd.read_csv(OLD/'segmentation/fragments_true_v71.csv'); fr = fr[fr['stack'] == f]
    with np.load(src) as data: masks = data['masks']
    assert masks.shape == (149,1024,1024) and masks.dtype == np.uint32
    mapping = np.zeros(int(masks.max())+1, np.int64)
    mapping[fr.label.to_numpy(int)] = fr.cell_uid.to_numpy(int)
    assert np.all(mapping[np.unique(masks)[1:]] > 0)
    planes=[]; start=time.time()
    for z in range(149):
        polys=mask_polygons(masks[z],mapping); planes.append(polys)
        area=sum(p.area for p in polys.values())
        assert np.isclose(area, np.count_nonzero(masks[z])*.2986**2, atol=1e-6, rtol=1e-10)
        if z % 25 == 0: log(stage='geometry',fov=f,z=z,seconds=time.time()-start)
    with path.open('wb') as h: pickle.dump(planes,h,protocol=4)
    assert sha(src)==digest
    save(path.with_suffix('.json'),dict(status='COMPLETE',fov=f,planes=149,mask_sha256=digest,
         source=str(src),output_sha256=sha(path),area_conservation=True,seconds=time.time()-start))


def coordinates():
    import camera_geometry
    from full_sampling_radial import choose_sources
    s=context(); model=read(s['model']); start=time.time(); total=0; counts=np.zeros(52,int)
    assert not list((OUT/'coordinates').iterdir())
    owner_all=np.lib.format.open_memmap(OUT/'coordinates/owner.npy',mode='w+',dtype=np.int16,shape=(s['rows'],))
    depth_all=np.lib.format.open_memmap(OUT/'coordinates/depth.npy',mode='w+',dtype=np.int16,shape=(s['rows'],))
    native_all=np.lib.format.open_memmap(OUT/'coordinates/native.npy',mode='w+',dtype=np.float64,shape=(s['rows'],3))
    for i,frame in enumerate(pd.read_csv(BARCODES,chunksize=200000,usecols=['global_x','global_y','physical_z_um','barcode_id'],float_precision='round_trip')):
        points=frame[['global_x','global_y','physical_z_um']].to_numpy(float)
        owner,native=choose_sources(points,model); zz=np.full(len(frame),-1,np.int16)
        valid=(owner>0)&(native[:,2]>=1)&(native[:,2]<=149)
        zz[valid]=np.rint(native[valid,2]-1+s['offsets'].get('dz_global',0)).astype(np.int16)
        ids=np.arange(total,total+len(frame)); owner_all[ids]=owner.astype(int)-1;depth_all[ids]=zz;native_all[ids]=native
        for code in np.unique(owner[valid]):
            rows=np.flatnonzero(valid&(owner==code)); f=int(code)-1
            chk=rows[::100]; assert np.max(np.abs(camera_geometry.apply_global(native[chk],f,model)-points[chk]))<1e-5
            np.savez_compressed(OUT/'coordinates'/f'f{f:02d}_c{i:04d}.npz',rows=ids[rows],xyz=native[rows],z=zz[rows],types=frame.barcode_id.to_numpy(int)[rows])
            counts[f]+=len(rows)
        total+=len(frame);log(stage='coordinates',rows=total,seconds=time.time()-start)
    assert total==s['rows'] and sha(BARCODES)==s['input_barcode_sha256']
    owner_all.flush();depth_all.flush();native_all.flush()
    save(OUT/'coordinates/complete.json',dict(status='COMPLETE',rows=total,eligible=int(counts.sum()),counts=counts.tolist(),
        array_sha256={n:sha(OUT/'coordinates'/n) for n in ['owner.npy','depth.npy','native.npy']},input_sha256=s['input_barcode_sha256']))


def join(f):
    s=context(); c=read(OUT/'coordinates/complete.json'); path=OUT/'joins'/f'{f:02d}.npz'; assert not path.exists()
    shards=[]
    for p in sorted((OUT/'coordinates').glob(f'f{f:02d}_c*.npz')):
        with np.load(p) as d: shards.append({k:d[k] for k in d.files})
    keys=['rows','xyz','z','types']; data={k:np.concatenate([d[k] for d in shards]) for k in keys} if shards else dict(rows=np.empty(0,np.int64),xyz=np.empty((0,3)),z=np.empty(0,int),types=np.empty(0,int))
    n=len(data['rows']); assert n==c['counts'][f]
    normal=np.zeros(n,np.int64); shrunk=np.zeros(n,np.int64); expanded=[]; interior=[]; start=time.time()
    candidates=candidate_catalog(s['offsets'])[f]; planes={}
    if n:
        for j,tx,ty,how in candidates:
            p=OUT/'geometry'/f'{j:02d}.pkl';receipt=read(p.with_suffix('.json'))
            assert sha(p)==receipt['output_sha256']
            with p.open('rb') as h: planes[j]=pickle.load(h)
    for z in np.unique(data['z']):
        ids=np.flatnonzero(data['z']==z)
        polys=reference_polygons({j:v[int(z)] for j,v in planes.items()},candidates,s['crops'][str(f)])
        for distance,dest in [(0,normal),(-.5,shrunk),(.5,None)]:
            pairs=spatial_pairs(data['xyz'][ids,:2],polys,distance)
            if len(pairs):
                pairs[:,0]=ids[pairs[:,0]]
                if dest is not None:
                    rr,at=np.unique(pairs[:,0],return_index=True);dest[rr]=pairs[at,1]
                if distance<0:interior.append(np.c_[pairs[:,1],data['types'][pairs[:,0]]])
                if distance>0:expanded.append(pairs)
        if int(z)%25==0:log(stage='join',fov=f,z=int(z),rows=n,seconds=time.time()-start)
        if f in [0,13,25,38] and int(z)==74:
            preview(f,polys,data['xyz'][ids,:2])
    expanded=np.concatenate(expanded) if expanded else np.empty((0,2),np.int64)
    iv=np.concatenate(interior) if interior else np.empty((0,2),np.int64)
    ic,nn=np.unique(iv,axis=0,return_counts=True)
    np.savez_compressed(path,rows=data['rows'],types=data['types'],normal=normal,shrunk=shrunk,expanded=expanded,interior=ic,interior_counts=nn)
    save(path.with_suffix('.json'),dict(status='COMPLETE',fov=f,rows=n,expanded_pairs=len(expanded),interior_pairs=int(nn.sum()),output_sha256=sha(path),seconds=time.time()-start))


def preview(f,polys,xy):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,ax=plt.subplots(figsize=(9,9)); lines=[]
    for p in polys.values():
        for part in (p.geoms if hasattr(p,'geoms') else [p]):
            if part.geom_type=='Polygon':lines.append(np.asarray(part.exterior.coords))
    ax.add_collection(LineCollection(lines,colors='green',linewidths=.4));ax.scatter(xy[:,0],xy[:,1],s=.3,c='magenta',alpha=.45)
    ax.set(xlim=(0,2047*.1493),ylim=(2047*.1493,0),xlabel='Native X (um)',ylabel='Native Y (um)',title=f'FOV {f}, native mask Z75um: existing masks and barcodes')
    ax.set_aspect('equal');fig.tight_layout();fig.savefig(OUT/'joins'/f'mask_review_{f:02d}.png',dpi=160);plt.close(fig)


def export():
    import camera_geometry
    s=context(); c=read(OUT/'coordinates/complete.json'); dest=OUT/'exports'; assert not list(dest.iterdir())
    interior=np.zeros((s['ncell']+1,s['ncodewords']),np.int64); receipts={}; seen=np.zeros(s['rows'],bool)
    for f in range(52):
        p=OUT/'joins'/f'{f:02d}.npz';r=read(p.with_suffix('.json'));assert sha(p)==r['output_sha256'];receipts[str(f)]=r
        with np.load(p) as d:
            rows=d['rows']; assert not seen[rows].any() and len(np.unique(rows))==len(rows);seen[rows]=True
            iv=d['interior'];np.add.at(interior,(iv[:,0],iv[:,1]),d['interior_counts'])
    assert int(seen.sum())==c['eligible']
    assignments=np.zeros(s['rows'],np.int64);counts=np.zeros_like(interior);rng=np.random.RandomState(s['seed'])
    for f in range(52):
        with np.load(OUT/'joins'/f'{f:02d}.npz') as d:
            uid=resolve(d['normal'],d['shrunk'],d['expanded'],d['types'],interior,rng)
            assignments[d['rows']]=uid;keep=uid>0;np.add.at(counts,(uid[keep],d['types'][keep]),1)
    np.save(dest/'cell_index.npy',assignments)
    owner=np.load(OUT/'coordinates/owner.npy',mmap_mode='r');depth=np.load(OUT/'coordinates/depth.npy',mmap_mode='r')
    total=blank_assigned=0;byf=np.zeros(52,np.int64);allpath=dest/'barcodes_with_cell_index.csv.gz'
    texttypes={'barcode_uid':'string','registration_uncertainty_rounds':'string','registration_uncertainty_regions':'string'}
    with gzip.open(allpath,'wt',compresslevel=1,newline='') as out,gzip.open(dest/'barcode_cell_assignments.csv.gz','wt',compresslevel=1,newline='') as links:
        for i,frame in enumerate(pd.read_csv(BARCODES,chunksize=200000,dtype=texttypes,float_precision='round_trip')):
            sl=slice(total,total+len(frame));uid=assignments[sl];keep=uid>0
            frame['cell_index']=uid;frame['partition_reference_fov']=owner[sl];frame['mask_plane_index']=depth[sl]
            frame['partition_method']='geopandas_merlin_boundary_0.5um'
            frame.to_csv(out,index=False,header=i==0)
            frame[['barcode_uid','barcode_id','global_x','global_y','physical_z_um','is_blank','cell_index','partition_reference_fov','mask_plane_index']].to_csv(links,index=False,header=i==0)
            blank_assigned+=int(frame.loc[keep,'is_blank'].sum());byf+=np.bincount(owner[sl][keep],minlength=52)
            total+=len(frame);log(stage='export',rows=total)
    assert total==s['rows'] and int(counts.sum())==int((assignments>0).sum())
    book=pd.read_csv(ROOT/'full_filter_v1/exports/codebook.csv')
    matrix=pd.DataFrame(counts[1:],index=np.arange(1,s['ncell']+1),columns=book.name);matrix.index.name='feature_id';matrix.to_csv(dest/'barcodes_per_feature.csv')
    cells=pd.read_csv(OLD/'segmentation/cells_final_true_v71.csv')
    meta=cells.rename(columns={k:'legacy_'+k for k in cells.columns if k!='cell_uid'}).set_index('cell_uid');meta.index.name='feature_id';meta['n_transcripts']=counts[1:].sum(1)
    fr=pd.read_csv(OLD/'segmentation/fragments_true_v71.csv');model=read(s['model']);sums=np.zeros((s['ncell']+1,3));weights=np.zeros(s['ncell']+1)
    for f in sorted(map(int,model['models'])):
        frag=pd.read_csv(OLD/f'segmentation/cells/cells_{f:02d}.csv');off=s['offsets']['own'][str(f)]['t'];crop=s['crops'][str(f)]
        native=np.c_[(frag.cx_ds.to_numpy()*2-off[0]+crop[0])*.1493,(frag.cy_ds.to_numpy()*2-off[1]+crop[1])*.1493,frag.cz_ds.to_numpy()+1-s['offsets'].get('dz_global',0)]
        valid=np.isfinite(native).all(1)&np.all((native[:,:2]>=0)&(native[:,:2]<=2047*.1493),axis=1)&(native[:,2]>=1)&(native[:,2]<=149)
        mapping=fr[fr['stack']==f].set_index('label').cell_uid
        u=mapping.loc[frag.label.to_numpy()[valid]].to_numpy(int);w=frag.nvox.to_numpy(float)[valid];mapped=camera_geometry.apply_global(native[valid],f,model)
        np.add.at(weights,u,w);np.add.at(sums,u,mapped*w[:,None])
    centers=np.full((s['ncell'],3),np.nan);good=weights[1:]>0;centers[good]=sums[1:][good]/weights[1:][good,None]
    meta[['stitched_center_x_um','stitched_center_y_um','stitched_center_z_um']]=centers;meta['stitched_center_supported']=good;meta.to_csv(dest/'feature_metadata.csv')
    fr.to_csv(dest/'fragments_cell_ids.csv',index=False);book.to_csv(dest/'codebook.csv',index=False)
    assert sha(BARCODES)==s['input_barcode_sha256'];context()
    for f in range(52):
        r=read(OUT/'geometry'/f'{f:02d}.json');assert sha(r['source'])==r['mask_sha256']
    assigned=int(counts.sum());report=dict(status='COMPLETE_REQUIRES_VISUAL_REVIEW',input_barcodes=total,assigned_barcodes=assigned,
        unassigned_barcodes=total-assigned,assigned_blank_barcodes=blank_assigned,cells=s['ncell'],cells_with_barcodes=int((counts[1:].sum(1)>0).sum()),
        assigned_normalized_blank_estimate=(blank_assigned/int(book.name.str.contains('blank',case=False).sum()))/((assigned-blank_assigned)/int((~book.name.str.contains('blank',case=False)).sum())) if assigned>blank_assigned else None,
        original_post_dedup_blank_estimate=s['source_post_dedup_blank_estimate'],all_inputs_unchanged=True,count_conservation_passed=True,
        assignment_by_reference_fov=byf.tolist(),boundary_buffer_um=.5,method=s['method'],limitations=s['limitations'],join_receipts=receipts,
        output_sha256={p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    save(dest/'complete.json',report);log(stage='complete',assigned=assigned,blank_assigned=blank_assigned)


def verify():
    s=context();dest=OUT/'exports';r=read(dest/'complete.json')
    for n,h in r['output_sha256'].items():assert sha(dest/n)==h,n
    counts=np.zeros((s['ncell']+1,s['ncodewords']),np.int64);rows=assigned=blank=0
    for frame in pd.read_csv(dest/'barcode_cell_assignments.csv.gz',chunksize=200000):
        uid=frame.cell_index.to_numpy(int);bid=frame.barcode_id.to_numpy(int);keep=uid>0
        assert np.all((uid>=0)&(uid<=s['ncell'])) and np.all((bid>=0)&(bid<s['ncodewords']))
        np.add.at(counts,(uid[keep],bid[keep]),1);rows+=len(frame);assigned+=int(keep.sum());blank+=int(frame.loc[keep,'is_blank'].sum())
    matrix=pd.read_csv(dest/'barcodes_per_feature.csv',index_col=0);meta=pd.read_csv(dest/'feature_metadata.csv',index_col=0)
    assert rows==s['rows'] and assigned==r['assigned_barcodes'] and blank==r['assigned_blank_barcodes']
    assert np.array_equal(counts[1:],matrix.to_numpy()) and np.array_equal(counts[1:].sum(1),meta.n_transcripts.to_numpy())
    assert list(matrix.index)==list(meta.index)==list(range(1,s['ncell']+1))
    # Independently verify the full export's row identities, original coordinates,
    # and cell assignments, not only its smaller linkage table.
    import itertools
    columns=['barcode_uid','barcode_id','global_x','global_y','physical_z_um']
    options=dict(chunksize=200000,dtype={'barcode_uid':'string'},float_precision='round_trip')
    original=pd.read_csv(BARCODES,usecols=columns,**options)
    full=pd.read_csv(dest/'barcodes_with_cell_index.csv.gz',usecols=columns+['cell_index'],**options)
    assigned_array=np.load(dest/'cell_index.npy',mmap_mode='r');checked=0
    for a,b in itertools.zip_longest(original,full):
        assert a is not None and b is not None
        assert a[columns].equals(b[columns])
        assert np.array_equal(b.cell_index.to_numpy(),assigned_array[checked:checked+len(b)])
        checked+=len(b)
    assert checked==s['rows']
    save(dest/'verification.json',dict(status='VERIFIED_EXPORT_REQUIRES_VISUAL_REVIEW',rows=rows,assigned=assigned,assigned_blanks=blank,
        every_cell_codeword_count_matches=True,all_output_hashes_passed=True,full_export_original_rows_and_assignments_match=True,
        source_receipt_sha256=sha(dest/'complete.json')))


if __name__=='__main__':
    stage=sys.argv[1]
    if stage in ['geometry','join']:globals()[stage](int(sys.argv[2]))
    else:globals()[stage]()
