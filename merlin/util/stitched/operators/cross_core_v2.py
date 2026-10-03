"""Diagnostic cross-round XYZ on measured connected components; pending raw QA."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import cross_round as cr
import global_quadratic as gq


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seam_regions(model,edges):
    centers={f:gq.apply_global(np.array([153.,153.,75.]),f,model) for f in map(int,model['models'])}
    regions=[]
    for a,b in edges:
        center=(centers[a]+centers[b])/2
        axis=int(np.argmax(np.abs(centers[a][:2]-centers[b][:2])))
        half=np.full(2,153.);half[axis]=20.
        regions.append(dict(name=f'{a}_{b}',source_fovs=[a,b],
            bounds_xy_um=[(center[:2]-half).tolist(),(center[:2]+half).tolist()],
            definition='40um-wide seam neighborhood centered between reference FOV centers atZ75; diagnostic region, not exact per-Z ownership boundary'))
    return regions


def main():
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    inventory=json.loads((root/'inventory.json').read_text())
    assert a.round in inventory['rounds'] and a.round!=0
    inputs={str(root/'inventory.json'):sha(root/'inventory.json')}
    clouds={}
    for r in (0,a.round):
        folder=root/'registration_core_v2'/f'round{r}'
        record=json.loads((folder/'pool_complete.json').read_text())
        path=folder/'stitched_cloud_owned.npz'
        assert record['status']=='COMPLETE' and sha(path)==record['output_sha256']
        assert sha(folder/'complete.json')==record['within_receipt_sha256']
        for source in [path,folder/'pool_complete.json',folder/'model.json',folder/'complete.json']:
            inputs[str(source)]=sha(source)
        clouds[r]=cr.load_cloud(path)
        if r==0:
            reference_model=json.loads((folder/'model.json').read_text())
    cr.SEAM_REGIONS=seam_regions(reference_model,[e for e in reference_model['proposed_edges'] if all(str(f) in reference_model['models'] for f in e)])
    out=root/'registration_core_v2'/f'round{a.round}'/'cross'
    report=cr.fit_cross(clouds[0],clouds[a.round],out,reference_round=0,moving_round=a.round)
    if any(sha(p)!=digest for p,digest in inputs.items()):
        raise RuntimeError('Cross-round fitting inputs changed')
    receipt=dict(status='FITTED_REQUIRES_RAW_IMAGE_QA',round=a.round,input_sha256=inputs,
        all_hashed_inputs_unchanged=True,source_sha256={Path(p).name:sha(p) for p in [__file__,cr.__file__]},
        preferred=report['preferred_candidate_for_raw_image_review'],
        output_sha256={x.name:sha(x) for x in out.iterdir() if x.is_file()})
    (out/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(round=a.round,preferred=receipt['preferred'],status=receipt['status'],
        candidates={k:v['overall'] for k,v in report['candidates'].items()})),flush=True)


if __name__=='__main__':main()
