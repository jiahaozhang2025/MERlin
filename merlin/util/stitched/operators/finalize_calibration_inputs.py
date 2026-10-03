"""Publish optimizer input manifest only after every planned image is verified."""
import json
from pathlib import Path
import numpy as np
from full_preprocess import sha

def assemble(plan_path):
    plan_path=Path(plan_path);root=plan_path.parent;plan=json.loads(plan_path.read_text());digest=sha(plan_path)
    inputs={str(plan_path):digest};planes=[];sources=None
    ids=[p['sample_id'] for p in plan['planes']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate calibration sample IDs')
    if sum(p['role']=='train' for p in plan['planes'])!=50 or sum(p['role']=='validation' for p in plan['planes'])!=12:
        raise ValueError('Incomplete fixed calibration design')
    for p in plan['planes']:
        folder=root/f'sample{p["sample_id"]:03d}';receipt_path=folder/'complete.json';rec=json.loads(receipt_path.read_text())
        if rec['status']!='COMPLETE' or rec['plane']!=p:raise ValueError('Incomplete or mismatched calibration plane')
        if rec['input_sha256'].get(str(plan_path))!=digest:raise ValueError('Plane belongs to another plan')
        if sources is None:sources=rec['source_sha256']
        if sources!=rec['source_sha256']:raise ValueError('Calibration planes used inconsistent scientific sources')
        for path,h in rec['input_sha256'].items():
            if path in inputs and inputs[path]!=h:raise ValueError('Inconsistent protected input')
            inputs[path]=h
        inputs[str(receipt_path)]=sha(receipt_path)
        for name in ['images.npy','support.npy']:
            if sha(folder/name)!=rec['output_sha256'][name]:raise ValueError('Changed calibration array')
        image=np.load(folder/'images.npy',mmap_mode='r');mask=np.load(folder/'support.npy',mmap_mode='r')
        if image.dtype!=np.float32 or image.shape[0]!=len(plan['bit_names']) or mask.dtype!=bool or mask.shape!=image.shape[1:]:raise ValueError('Calibration array schema mismatch')
        planes.append(dict(z_index=p['sample_id'],sample_id=p['sample_id'],output_tile=p['tile_id'],acquisition_z_index=p['z_index'],
            physical_z_um=p['physical_z_um'],role=p['role'],images_path=str(folder/'images.npy'),mask_path=str(folder/'support.npy')))
    if any(sha(path)!=h for path,h in inputs.items()):raise RuntimeError('Calibration inputs changed during assembly')
    return dict(bit_names=plan['bit_names'],codebook_path=plan['codebook_path'],registration_manifest=plan['registration_path'],planes=planes,
        full_sample_plan_sha256=digest,verified_input_sha256=inputs,source_sha256=sources,
        index_policy='Optimizer z_index is a unique fragment identifier across tile/depth combinations; acquisition_z_index and physical_z_um retain actual depth. This manifest is exclusively for intensity calibration, never cross-Z barcode deduplication.')

def main():
    root=Path(__file__).resolve().parent.parent/'calibration_inputs_v1';path=root/'optimizer_manifest.json'
    if path.exists():raise FileExistsError(path)
    manifest=assemble((root/'plan.json').resolve());path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(planes=len(manifest['planes']),output=str(path))))
if __name__=='__main__':main()
