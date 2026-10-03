"""Pool only measured component clouds; retain global heldout exclusions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import camera_geometry as gq
from full_sampling_radial import choose_sources

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parent.parent;folder=root/'registration_camera_radial_v2'/f'round{a.round}'
 rec=json.loads((folder/'complete.json').read_text())
 for name,h in rec['output_sha256'].items():assert sha(folder/name)==h
 model=json.loads((folder/'model.json').read_text())
 parts={k:[] for k in ['xyz_um','native_xyz_um','source_fov','source_index','amplitude','within_round_heldout','within_round_training']}
 with np.load(folder/'matches.npz') as ties:
  for f in map(int,model['models']):
   path=root/'clouds'/f'fov{f}_r{a.round}.npz';assert sha(path)==rec['input_sha256'][str(path)]
   with np.load(path) as d:
    keep=np.isfinite(d['xyz_um']).all(1)&np.isfinite(d['amplitude'])&(d['amplitude']>=500)
    native=d['xyz_um'][keep];amp=d['amplitude'][keep];indices=np.flatnonzero(keep)
   held=np.zeros(len(native),bool);train=held.copy()
   for left,right in model['proposed_edges']:
    if f not in (left,right):continue
    side='a' if f==left else 'b'
    for label,mask in [('training',train),('heldout',held)]:
     name=f'{left}_{right}_{label}_{side}_indices'
     if name in ties:mask[ties[name]]=True
   assert not np.any(held&train)
   for name,value in dict(xyz_um=gq.apply_global(native,f,model),native_xyz_um=native,source_fov=np.full(len(native),f),source_index=indices,
       amplitude=amp,within_round_heldout=held,within_round_training=train).items():parts[name].append(value)
 payload={k:np.concatenate(v) for k,v in parts.items()};owners,_=choose_sources(payload['xyz_um'],model)
 keep=owners==payload['source_fov']+1;out=folder/'stitched_cloud_owned.npz'
 if out.exists():raise FileExistsError(out)
 np.savez_compressed(out,**{k:v[keep] for k,v in payload.items()})
 report={'status':'COMPLETE','round':a.round,'source_points':len(keep),'owned_points':int(keep.sum()),'heldout_points':int(payload['within_round_heldout'][keep].sum()),
   'source_fovs':model['measured_fovs'],'unsupported_fovs':model['unsupported_fovs'],'output_sha256':sha(out),'within_receipt_sha256':sha(folder/'complete.json'),
   'scope':'Diagnostic measured component; unsupported sources excluded explicitly, not treated as registered'}
 (folder/'pool_complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
