"""Fit the measured 3D connected component; preserve unsupported edges/FOVs.

This is an exploratory intermediate, never an accepted full-sample model.
No inferred XYZ transform is manufactured for disconnected sources.
"""
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import tifffile
import full_geometry
import global_quadratic as gq
import fit_graph as fg
import fit_z0 as z0


def connected(fovs,edges,anchor=13):
    seen={anchor}
    while True:
        updated=seen|{f for a,b in edges if a in seen or b in seen for f in (a,b)}
        if updated==seen:return sorted(seen)
        seen=updated


def configure_round(inventory,round_id):
    full_geometry.configure(inventory)
    # Z0 phase-image comparison r0 vs r1/r10 finds ~43X/80Y first-field
    # discontinuity, shared by FOV0/1, absent from FOV2/9/10. This only
    # re-centers the search. Refined volume ties estimate the final transform.
    correction={0:np.array([-43.,-80.,0.]),1:np.array([-43.,-80.,0.])} if round_id==0 else {}
    origins={f:fg.seed_origin(f)+correction.get(f,np.zeros(3)) for f in inventory['source_fovs']}
    # Express initialization in the existing stage-origin interface.
    anchor=np.array(inventory['positions'][13])
    fg.STAGES={f:tuple(anchor-origins[f][:2]) for f in origins}
    width=2048*.1493
    edges=[]
    for a,b in itertools.combinations(inventory['source_fovs'],2):
        overlap=width-np.abs(origins[a][:2]-origins[b][:2])
        # Include meaningful side overlaps and changed first-field overlaps;
        # exclude tiny nominal corner contacts from the fitting graph.
        if np.min(overlap)>0 and np.prod(overlap)>=8000:edges.append((a,b))
    fg.CFG=dict(fg.CFG,search_xy_um=16.)
    z0.CFG=dict(z0.CFG,search_um=16.)
    gq.SIDE_EDGES=edges;fg.SIDES=set(edges)
    return edges,{str(f):v.tolist() for f,v in correction.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parent.parent;started=time.time()
    inventory_path=root/'inventory.json';inventory=json.loads(inventory_path.read_text())
    proposed,correction=configure_round(inventory,args.round)
    out=root/'registration_core_v2'/f'round{args.round}';out.mkdir(parents=True,exist_ok=False)
    clouds={};surface={};sources={};input_hash={str(inventory_path):hashlib.sha256(inventory_path.read_bytes()).hexdigest()}
    for f in inventory['source_fovs']:
        stem=root/'clouds'/f'fov{f}_r{args.round}'
        rec=json.loads(stem.with_suffix('.json').read_text())
        for path in [stem.with_suffix('.npz'),Path(str(stem)+'_z0.tif')]:
            h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==rec['output_sha256'][path.name];input_hash[str(path)]=h
        with np.load(stem.with_suffix('.npz')) as d:
            keep=np.isfinite(d['xyz_um']).all(1)&np.isfinite(d['amplitude'])&(d['amplitude']>=500)
            clouds[f]=d['xyz_um'][keep]
        surface[f],_,det=z0.detect(tifffile.imread(str(stem)+'_z0.tif'))
        sources[str(f)]={'volume_bright_candidates':len(clouds[f]),'surface_detection':det}
    splits={};ssplits={};held={f:set() for f in clouds};sheld={f:set() for f in clouds}
    for a,b in proposed:
        seed=fg.seed_origin(b)-fg.seed_origin(a)
        if min(len(clouds[a]),len(clouds[b])):
            splits[a,b]=gq.new_spatial_split(clouds[a],clouds[b],seed,fg.CFG)
            held[a].update(splits[a,b][2]);held[b].update(splits[a,b][3])
        ssplits[a,b]=z0.split_edge(surface[a],surface[b],seed[:2])
        sheld[a].update(ssplits[a,b][2]);sheld[b].update(ssplits[a,b][3])
    archive={};edges=[];diagnostics=[]
    for a,b in proposed:
        key=f'{a}_{b}_';seed=fg.seed_origin(b)-fg.seed_origin(a);entry={'edge':[a,b],'stage_seed':seed.tolist()};surface_shift=None
        ta,tb,ha,hb,split=ssplits[a,b]
        ta=np.array([i for i in ta if i not in sheld[a]],int);tb=np.array([i for i in tb if i not in sheld[b]],int)
        try:
            shift,pairs,res,diag=z0.fit_translation(surface[a][ta],surface[b][tb],seed[:2])
            metrics=z0.stats(res,min(len(ta),len(tb)))
            coherent=metrics['median_pixels']<=2.5 and metrics['matched_fraction_of_smaller_feature_set']>=.25
            entry['surface']={'status':'COHERENT_SEED' if coherent else 'AMBIGUOUS_SEED','shift':shift.tolist(),'training':metrics,'diagnostics':diag}
            if coherent:surface_shift=shift
            hp,hr=z0.reciprocal(surface[a][ha],surface[b][hb],shift,z0.CFG['heldout_match_gate_um'])
            for label,value in [('surface_training_native_a_xy_um',surface[a][ta[pairs[:,0]]]),('surface_training_native_b_xy_um',surface[b][tb[pairs[:,1]]]),
                ('surface_heldout_native_a_xy_um',surface[a][ha[hp[:,0]]]),('surface_heldout_native_b_xy_um',surface[b][hb[hp[:,1]]])]:archive[key+label]=value
        except (ValueError,RuntimeError) as e:entry['surface']={'status':'FAILED','reason':str(e)}
        if (a,b) not in splits:
            entry['volume']={'status':'FAILED','reason':'Empty bright source cloud'};diagnostics.append(entry);continue
        ta,tb,ha,hb,split=splits[a,b]
        ta=np.array([i for i in ta if i not in held[a]],int);tb=np.array([i for i in tb if i not in held[b]],int)
        split['training_counts']=[len(ta),len(tb)]
        for name,value in [('training_a_indices',ta),('training_b_indices',tb),('heldout_a_indices',ha),('heldout_b_indices',hb)]:archive[key+name]=value
        try:
            start=np.r_[surface_shift,seed[2]] if surface_shift is not None else seed
            translation,pairs,res,diag=fg.fit_translation(clouds[a][ta],clouds[b][tb],start,fg.CFG)
            train_metric=fg.metrics(res,min(len(ta),len(tb)),fg.CFG)
            if train_metric['xy_median_pixels']>1.5 or train_metric['z_median_um']>.75:
                raise ValueError('Incoherent training translation; median exceeds1.5pxXY/.75umZ')
            hp,hr=fg.reciprocal_matches(clouds[a][ha],clouds[b][hb],translation,fg.CFG['validation_match_xy_um'],fg.CFG['validation_match_z_um'])
            tp=np.c_[ta[pairs[:,0]],tb[pairs[:,1]]];hp=np.c_[ha[hp[:,0]],hb[hp[:,1]]]
            edge=dict(k=a,n=b,training_pairs=tp,heldout_pairs=hp,spatial_split=split,initial_translation_xyz_um=translation.tolist(),
                training_translation_fit=diag,translation_training=train_metric,translation_heldout=fg.metrics(hr,min(len(ha),len(hb)),fg.CFG))
            edges.append(edge);entry['volume']={'status':'FITTED','translation':translation.tolist(),'training':train_metric,'heldout':edge['translation_heldout']}
            for name,value in [('training_pairs',tp),('heldout_pairs',hp),('translation_training_residual_xyz_um',res),('translation_heldout_residual_xyz_um',hr)]:archive[key+name]=value
        except (ValueError,RuntimeError) as e:entry['volume']={'status':'FAILED','reason':str(e)}
        diagnostics.append(entry);print(json.dumps(entry),flush=True)
    component=connected(inventory['source_fovs'],[(e['k'],e['n']) for e in edges])
    if len(component)<4:raise ValueError('Insufficient measured connected component')
    core_edges=[e for e in edges if e['k'] in component and e['n'] in component]
    (out/'edge_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    gq.FOVS=fg.FOVS=tuple(component)
    model=full_geometry.fit_model({f:clouds[f] for f in component},core_edges,fg.CFG)
    model['assessment']=gq.assess(clouds,core_edges,model,archive,fg.CFG)
    model.update(round_id=args.round,sources=sources,measured_fovs=component,unsupported_fovs=sorted(set(inventory['source_fovs'])-set(component)),
        initial_origin_corrections_xyz_um=correction,proposed_edges=[list(e) for e in proposed],
        candidate_policy='Original >=500ADU plus original localized PSF detector',
        validation_design='Graph-wide exclusion of all heldout feature IDs on every proposed overlap before fitting. Initial round0first-field coarse shift uses independentZ0 cross-round image evidence; volumetric holdouts never initialize or fit the model.',
        scope='Measured3Dcomponent only; unsupported fields have NO inferred transform; not accepted for full decoding')
    np.savez_compressed(out/'matches.npz',**archive)
    (out/'model.json').write_text(json.dumps(model,indent=2))
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h for path,h in input_hash.items()):raise RuntimeError('Input changed')
    receipt={'status':'FITTED_CORE_REQUIRES_REVIEW','round':args.round,'measured_fovs':component,'unsupported_fovs':model['unsupported_fovs'],
        'proposed_edges':len(proposed),'fitted_edges':len(core_edges),'optimization':model['optimization'],'aggregate':model['assessment']['status'],
        'input_sha256':input_hash,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(full_geometry.__file__),Path(gq.__file__),Path(fg.__file__),Path(z0.__file__)]},
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},'elapsed_seconds':time.time()-started}
    (out/'complete.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if 'sha256' not in k}),flush=True)

if __name__=='__main__':main()
