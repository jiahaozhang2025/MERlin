from stitched_config import ROOT, CONFIG
"""Compare spatial join output directly against the unmodified legacy method."""
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Dict
import ast, json, os, sys, time, uuid
import numpy as np
import pandas as pd
from shapely import geometry
from partition_sjoin_core import partition_counts

SOURCE = Path(CONFIG['canonical_root'])
namespace = dict(np=np, pandas=pd, geometry=geometry, List=List, Tuple=Tuple, Dict=Dict, uuid=uuid)
tree = ast.parse((SOURCE/'merlin/util/spatialfeature.py').read_text())
feature_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpatialFeature')
exec(compile(ast.Module(body=[feature_node], type_ignores=[]), 'original_spatialfeature.py', 'exec'), namespace)
SpatialFeature = namespace['SpatialFeature']
tree = ast.parse((SOURCE/'merlin/analysis/partition.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PartitionBarcodes')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_run_analysis_boundary_correction')
exec(compile(ast.Module(body=[method], type_ignores=[]), 'original_partition.py', 'exec'), namespace)
original = namespace['_run_analysis_boundary_correction']


def legacy(cells, xyz, types, ntypes, buffer):
    frame = pd.DataFrame(xyz, columns=['global_x', 'global_y', 'z'])
    frame['barcode_id'] = types
    codebook = SimpleNamespace(get_barcode_count=lambda: ntypes, get_name_for_barcode_index=lambda i: str(i))
    tasks = {
        'filter': SimpleNamespace(get_codebook=lambda: codebook,
                                 get_barcode_database=lambda: SimpleNamespace(get_barcodes=lambda f: frame)),
        'assignment': SimpleNamespace(get_feature_database=lambda: SimpleNamespace(read_features=lambda f: cells)),
        'alignment': SimpleNamespace(get_fov_boxes=lambda: [geometry.box(-1e6, -1e6, 1e6, 1e6)])}
    output = []
    ds = SimpleNamespace(load_analysis_task=lambda name: tasks[name],
                         save_dataframe_to_csv=lambda frame, *args: output.append(frame))
    task = SimpleNamespace(dataSet=ds, get_analysis_name=lambda: 'test',
                           parameters=dict(filter_task='filter', assignment_task='assignment',
                                           alignment_task='alignment', boundary_correction_buffer_size=buffer))
    original(task, 0)
    return output[0].to_numpy()


def check(cells, xyz, types, ntypes, buffer, seed):
    np.random.seed(seed)
    start = time.perf_counter()
    expected = legacy(cells, xyz, types, ntypes, buffer)
    legacy_s = time.perf_counter() - start
    state_expected = np.random.get_state()
    np.random.seed(seed)
    start = time.perf_counter()
    actual = partition_counts(cells, xyz, types, ntypes, buffer)
    sjoin_s = time.perf_counter() - start
    state_actual = np.random.get_state()
    np.testing.assert_array_equal(actual, expected)
    assert all(np.array_equal(a, b) for a, b in zip(state_actual, state_expected)), 'Random generator state differs'
    return dict(cells=len(cells), barcodes=len(types), buffer=buffer, seed=seed,
                exact_counts=True, exact_rng_state=True, assigned=int(actual.sum()),
                legacy_seconds=legacy_s, sjoin_seconds=sjoin_s)


def synthetic():
    square = geometry.box(0, 0, 4, 4)
    hole = geometry.Polygon([(0,0),(6,0),(6,6),(0,6)], holes=[[(2,2),(4,2),(4,4),(2,4)]])
    cells = [SpatialFeature([[square, square], [], [hole]], 0, uniqueID=100),
             SpatialFeature([[geometry.box(3,0,7,4)], [square], [square]], 0, uniqueID=101),
             SpatialFeature([[], [], []], 0, uniqueID=102),
             SpatialFeature([[geometry.box(0,0,.1,.1)], [], []], 0, uniqueID=103)]
    rng = np.random.RandomState(123)
    xyz = np.column_stack([rng.uniform(-1,8,2000), rng.uniform(-1,8,2000), rng.choice([-.5,0,.49,.5,1,1.5,2,2.5,3], 2000)])
    special = [[0,0,0],[4,2,0],[3.5,2,0],[.5,.5,0],[2,2,2],[3,3,2],
               [.05,.05,0],[np.nan,2,0],[np.inf,1,0],[1,1,np.nan],[1,1,-1]]
    xyz = np.vstack([xyz, special]); types = rng.randint(0, 7, len(xyz))
    results = [check(cells,xyz,types,7,buffer,seed) for buffer in [.2,.5,1.] for seed in [7,99,2026]]
    results.append(check([],xyz,types,7,.5,1))
    results.append(check(cells,np.empty((0,3)),np.array([],dtype=int),7,.5,1))
    # Disconnected and overlapping polygons, a narrow neck split by erosion,
    # overlapping cells, noninteger Z and strict boundary points are covered.
    for trial in range(4):
        random_cells=[]
        for c in range(12):
            boundaries=[]
            for z in range(7):
                x,y=rng.uniform(0,15,2)
                shape=geometry.box(x,y,x+4,y+3).union(geometry.box(x+3,y+1.3,x+6,y+1.7)).union(geometry.box(x+5,y,x+9,y+3))
                boundaries.append([shape] if rng.rand()>.2 else [])
            random_cells.append(SpatialFeature(boundaries,0,uniqueID=c))
        xyz=np.column_stack([rng.uniform(0,23,4000),rng.uniform(0,20,4000),rng.uniform(-1,8,4000)])
        types=rng.randint(0,9,len(xyz))
        results.append(check(random_cells,xyz,types,9,.5,trial))
    return results


if __name__ == '__main__':
    result={'synthetic':synthetic()}
    from partition_sjoin_core import mask_polygons, reference_polygons, spatial_pairs
    from partition_masks import candidate_catalog
    from full_sampling_radial import choose_sources
    root=ROOT
    old=Path(CONFIG['mask_archive'])
    # Exact pixel footprints, including holes and disconnected pieces.
    mask=np.array([[1,1,1,0],[1,0,1,2],[1,1,1,2]],np.uint32)
    polys=mask_polygons(mask,np.array([0,11,12]))
    yy,xx=np.indices(mask.shape);xy=np.c_[xx.ravel()*2+.5,yy.ravel()*2+.5]*.1493
    pairs=spatial_pairs(xy,polys,0);actual=np.zeros(mask.size,int)
    actual[pairs[:,0]]=pairs[:,1]
    np.testing.assert_array_equal(actual,np.array([0,11,12])[mask.ravel()])
    assert np.isclose(sum(p.area for p in polys.values()),np.count_nonzero(mask)*.2986**2)
    result['pixel_footprints_exact']=True
    # Actual decoded coordinates against actual reused 0718 mask outlines.
    reg=json.loads((root/'registration_accepted_radial_v1.json').read_text())
    model=json.loads((root/reg['rounds']['0']['within']).read_text())
    frame=pd.read_csv(root/'full_filter_v1/exports/adaptive_filtered_z_deduplicated.csv.gz',nrows=200000,
                      usecols=['global_x','global_y','physical_z_um','barcode_id'],float_precision='round_trip')
    owner,native=choose_sources(frame[['global_x','global_y','physical_z_um']].to_numpy(),model)
    valid=(owner>0)&(native[:,2]>=1)&(native[:,2]<=149)
    ids=np.flatnonzero(valid); zz=np.rint(native[ids,2]-1).astype(int)
    bins=(owner[ids].astype(int)-1)*149+zz
    chosen=int(np.bincount(bins).argmax());f=chosen//149;z=chosen%149;rows=ids[bins==chosen]
    offsets=json.loads((old/'segmentation/seg_offsets_final.json').read_text())
    aff=json.loads((old/'segmentation/aff_constants.json').read_text());pos=np.loadtxt(old/'data/positions.csv',delimiter=',')
    crop=np.rint((np.array(aff[str(f)][:2])-pos[f])/.1493)
    fragments=pd.read_csv(old/'segmentation/fragments_true_v71.csv');planes={}
    candidates=candidate_catalog(offsets)[f]
    for j,tx,ty,how in candidates:
        with np.load(old/f'segmentation/masks/masks_{j:02d}.npz') as d: m=d['masks'][z]
        fr=fragments[fragments['stack']==j];mapping=np.zeros(max(int(m.max()),int(fr.label.max()))+1,np.int64)
        mapping[fr.label.to_numpy(int)]=fr.cell_uid.to_numpy(int)
        planes[j]=mask_polygons(m,mapping)
    polys=reference_polygons(planes,candidates,crop)
    cells=[SpatialFeature([list(p.geoms) if p.geom_type=='MultiPolygon' else [p]],f,uniqueID=u) for u,p in sorted(polys.items())]
    xyz=np.c_[native[rows,:2],np.zeros(len(rows))]
    result['real_0718']=check(cells,xyz,frame.barcode_id.to_numpy()[rows],len(pd.read_csv(root/'full_filter_v1/exports/codebook.csv')),.5,7182026)
    result['real_0718'].update(reference_fov=f,mask_plane=z)
    result['passed']=True
    path=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('synthetic_results.json')
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
