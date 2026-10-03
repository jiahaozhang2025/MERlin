"""Read-only native fiducials for the replacement stitched-volume pilot."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import tifffile
import bead_cloud as bc

from stitched_config import ROOT, CONFIG
DATA=ROOT/'inputs'
RAW=Path(CONFIG['raw_root'])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--fov',type=int,required=True)
    p.add_argument('--round',type=int,required=True); p.add_argument('--output-root',required=True)
    a=p.parse_args(); root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True)
    stem=root/f'fov{a.fov}_r{a.round}'
    if stem.with_suffix('.json').exists(): raise FileExistsError(stem)
    start=time.time(); path,frames,zs,micro,meta=bc.raw_metadata(DATA,RAW,a.fov,a.round)
    raw=bc.read_native(path,frames,micro)
    arrays,stats=bc.detect(raw,zs,micro['microns_per_pixel'])
    np.savez_compressed(stem.with_suffix('.npz'),**arrays)
    z0=bc.read_native(path,[0],micro)[0]
    tifffile.imwrite(str(stem)+'_z0.tif',z0)
    # Fixed 4x block means preserve registration evidence without full raw duplication.
    coarse=raw.astype(np.float32).reshape(len(zs),512,4,512,4).mean((2,4))
    np.save(str(stem)+'_coarse.npy',coarse)
    meta.update(detection=stats,elapsed_seconds=time.time()-start,job_id=os.environ.get('SLURM_JOB_ID'),
        raw_policy='read only',coarse_axis_order='zyx',coarse_downsample_xy=4,
        z0_raw_index=0,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bead_detector_sha256=hashlib.sha256(Path(bc.__file__).read_bytes()).hexdigest())
    meta['output_sha256']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in
        [stem.with_suffix('.npz'),Path(str(stem)+'_z0.tif'),Path(str(stem)+'_coarse.npy')]}
    stem.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'fov':a.fov,'round':a.round,'beads':stats['retained_beads'],'seconds':meta['elapsed_seconds']}),flush=True)
if __name__=='__main__': main()
