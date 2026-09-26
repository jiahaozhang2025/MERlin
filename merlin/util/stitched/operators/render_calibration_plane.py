"""Render one planned full-field calibration plane; never load prior scales."""
import argparse,json
from pathlib import Path
import numpy as np
from full_preprocess import load_registration,AuditedAcquisition,produce_plane,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--sample',type=int,required=True);p.add_argument('--fork',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent.parent;plan_path=root/'calibration_inputs_v1/plan.json';plan=json.loads(plan_path.read_text())
    inputs={str(plan_path):sha(plan_path),**plan['protected_input_sha256']}
    sources=list((Path(a.fork)/'merlin').rglob('*.py'))
    if not sources:raise ValueError('Canonical MERlin sources not found')
    sources += [Path(__file__).with_name(n) for n in ['render_calibration_plane.py','full_preprocess.py','full_sampling_radial.py','camera_geometry.py','global_quadratic.py','fit_graph.py','local_crosswarp_v2.py','cross_round.py','stitched_sampler.py','bead_cloud.py']]
    source_hashes={str(p):sha(p) for p in sources};inputs.update(source_hashes)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise RuntimeError('Planned input changed: '+path)
    _,models,protected=load_registration(plan['registration_path']);inputs.update(protected)
    plane=next(x for x in plan['planes'] if x['sample_id']==a.sample)
    tile=next(x for x in plan['layout']['tiles'] if x['tile_id']==plane['tile_id'])
    inventory=json.loads((root/'inventory.json').read_text());acq=AuditedAcquisition(root,inventory)
    image,support,owners,report=produce_plane(acq,plan['bit_names'],models,tile,plane['physical_z_um'],a.fork)
    if support.sum()<1000:raise RuntimeError('Selected calibration plane lacks usable exact support; do not silently replace it')
    if any(sha(path)!=digest for path,digest in inputs.items()):raise RuntimeError('Render inputs changed')
    out=root/'calibration_inputs_v1'/f'sample{a.sample:03d}';out.mkdir(exist_ok=False)
    np.save(out/'images.npy',image,allow_pickle=False);np.save(out/'support.npy',support,allow_pickle=False)
    receipt=dict(status='COMPLETE',plane=plane,preprocessing=report,raw_audit=acq.audit(),input_sha256=inputs,source_sha256=source_hashes,
        supported_pixels=int(support.sum()),shape=list(image.shape),output_sha256={name:sha(out/name) for name in ['images.npy','support.npy']})
    (out/'complete.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(sample=a.sample,supported_pixels=int(support.sum()))))
if __name__=='__main__':main()
