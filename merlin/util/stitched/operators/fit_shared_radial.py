"""Test common-camera curvature on the same frozen within-round training ties."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import within_core_v2 as core
import shared_radial_geometry as shared
import camera_geometry as gq
import fit_graph as fg

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parent.parent;source=root/'registration_core_v2'/f'round{a.round}'
 out=root/'registration_camera_radial_v2'/f'round{a.round}';out.mkdir(parents=True,exist_ok=False);started=time.time()
 receipt=json.loads((source/'complete.json').read_text())
 for n,h in receipt['output_sha256'].items():assert sha(source/n)==h
 old=json.loads((source/'model.json').read_text());inventory=json.loads((root/'inventory.json').read_text());core.configure_round(inventory,a.round)
 gq.FOVS=fg.FOVS=tuple(old['measured_fovs']);clouds={};inputs={str(source/n):sha(source/n) for n in ['model.json','matches.npz','complete.json']}
 for f in old['measured_fovs']:
  path=root/'clouds'/f'fov{f}_r{a.round}.npz';inputs[str(path)]=sha(path);assert inputs[str(path)]==receipt['input_sha256'][str(path)]
  with np.load(path) as d:clouds[f]=d['xyz_um'][np.isfinite(d['xyz_um']).all(1)&np.isfinite(d['amplitude'])&(d['amplitude']>=500)]
 with np.load(source/'matches.npz') as d:archive={k:d[k].copy() for k in d.files}
 edges=[]
 for e in old['assessment']['edges']:
  prefix=f"{e['k']}_{e['n']}_"
  edges.append(dict(e,training_pairs=archive[prefix+'training_pairs'],heldout_pairs=archive[prefix+'heldout_pairs']))
 model=shared.fit_model(clouds,edges,fg.CFG);model['assessment']=gq.assess(clouds,edges,model,archive,fg.CFG)
 model.update(round_id=a.round,measured_fovs=old['measured_fovs'],unsupported_fovs=old['unsupported_fovs'],proposed_edges=old['proposed_edges'],
     validation_design='Exactly the previous core-v2 training and heldout bead pairs; no rematching or heldout fitting.',
     scientific_status='EXPLORATORY_CAMERA_HYPOTHESIS_REQUIRES_RAW_IMAGE_QA')
 (out/'model.json').write_text(json.dumps(model,indent=2));np.savez_compressed(out/'matches.npz',**archive)
 if any(sha(p)!=h for p,h in inputs.items()):raise RuntimeError('Input changed')
 report={'status':'FITTED_REQUIRES_RAW_IMAGE_QA','round':a.round,'all_hashed_inputs_unchanged':True,'input_sha256':inputs,
     'optimizer':model['optimization'],'aggregate':model['assessment']['status'],
     'jacobian_failed':[f for f,v in model['assessment']['camera_grid_jacobians'].items() if not v['passed']],
     'reasons':model['assessment']['reasons'],'elapsed_seconds':time.time()-started,
     'source_sha256':{Path(p).name:sha(p) for p in [__file__,shared.__file__,gq.__file__]},
     'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
 (out/'complete.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if 'sha256' not in k}),flush=True)

if __name__=='__main__':main()
