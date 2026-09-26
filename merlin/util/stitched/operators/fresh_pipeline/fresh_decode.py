"""Decode one preprocessed plane of the fresh stitched virtual FOV.

Requires accepted fresh intensity optimization and exact codebook bit order.
This module never reads a production dataset or any old fitted calibration.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage

from fresh_optimize import (DEFAULT_FORK, load_manifest, load_canonical,
                            read_plane, sha256, write_json, _install_write_guard)
import registration_flags

MPP=.1493
SOURCE_FOVS=[13,14,25,26]
DECODE_PARAMETERS=dict(distance_threshold=.65,minimum_area=4,magnitude_threshold=1.,lowpass_sigma=0)
SOURCE_OWNER_MAP={'0':None,**{str(i+1):f for i,f in enumerate(SOURCE_FOVS)}}


def manifest_origin(manifest):
    origin=np.asarray(manifest.get('origin_xy_um',manifest.get('output_origin_xy_um',[])),dtype=float)
    if origin.shape!=(2,) or not np.isfinite(origin).all():
        raise ValueError('Manifest requires explicit global origin_xy_um')
    return origin


def owners_path(plane):
    path=plane.get('source_owners_path')
    return Path(path) if path else Path(plane['images_path']).parent/'source_owners.npz'


def resolve_decode_manifest(path):
    manifest=load_manifest(path)
    base=Path(path).resolve().parent
    for plane in manifest['planes']:
        if plane.get('source_owners_path'):
            p=Path(plane['source_owners_path'])
            plane['source_owners_path']=str((p if p.is_absolute() else base/p).resolve())
    manifest_origin(manifest)
    return manifest


def load_accepted_optimization(root,manifest):
    root=Path(root).resolve()
    report=json.loads((root/'optimization.json').read_text())
    if (report.get('status')!='converged' or not report.get('convergence_passed')
            or not report.get('fresh') or report.get('reused_fitted_calibration')
            or not report.get('all_hashed_inputs_unchanged')):
        raise RuntimeError('Fresh optimization is not accepted and converged')
    if report.get('bit_names')!=manifest['bit_names']:
        raise RuntimeError('Optimization/codebook bit order mismatch')
    inputs=json.loads((root/'inputs.json').read_text())
    original=inputs['manifest']
    if original['bit_names']!=manifest['bit_names']:
        raise RuntimeError('Optimization input bit order mismatch')
    expected=inputs['input_sha256'].get(str(Path(original['codebook_path']).resolve()))
    if expected is None or sha256(manifest['codebook_path'])!=expected:
        raise RuntimeError('Codebook differs from fresh optimization input')
    old_registration=original.get('registration_manifest')
    new_registration=manifest.get('registration_manifest')
    if bool(old_registration)!=bool(new_registration):
        raise RuntimeError('Registration provenance differs from optimization')
    if old_registration:
        expected=inputs['input_sha256'].get(str(Path(old_registration).resolve()))
        if expected is None or sha256(new_registration)!=expected:
            raise RuntimeError('Registration changed after fresh optimization')
    sf=np.load(root/'scale_factors.npy',allow_pickle=False)
    bg=np.load(root/'backgrounds.npy',allow_pickle=False)
    if sf.shape!=(len(manifest['bit_names']),) or bg.shape!=sf.shape or not np.isfinite(sf).all() or not np.isfinite(bg).all() or np.any(sf<=0):
        raise RuntimeError('Invalid fresh calibration arrays')
    if not np.array_equal(sf,np.asarray(report['final_scale_factors'])) or not np.array_equal(bg,np.asarray(report['final_backgrounds'])):
        raise RuntimeError('Fresh calibration arrays disagree with accepted report')
    return sf,bg,inputs


def load_owners(path,shape,mask):
    with np.load(path,allow_pickle=False) as source:
        owners=source['source_owners']
        fovs=source['source_fovs'].tolist()
    if fovs!=SOURCE_FOVS or owners.shape!=shape or owners.dtype!=np.uint8 or np.any(owners>4):
        raise ValueError('Unexpected source-owner coding, shape, or dtype')
    if np.any(owners[:,mask]==0):
        raise ValueError('Complete support includes a missing required source bit')
    return owners


def decode_arrays(image,support,owners,codebook,Decoder,scales,backgrounds,z_index,
                  physical_z_um,origin_xy_um,num_threads=1):
    """Canonical decode/extraction; annotations cover each whole component."""
    mask=np.asarray(support,bool).copy()
    mask[[0,-1],:]=False;mask[:,[0,-1]]=False
    decoder=Decoder(codebook)
    di,pm,traces,distances=decoder.decode_pixels(image,scales,backgrounds,
        distanceThreshold=.65,magnitudeThreshold=1.,lowPassSigma=0,
        distanceMetric='euclidean',decodeMask=mask,useGpu=False,tilingFactor=None,
        neighborNumJobs=int(num_threads),accumulatePixelTraces=True,tilingOverlap=20)
    if np.any(di[~mask]>=0):
        raise AssertionError('Canonical decoder assigned pixels outside complete support')
    frame,labels=decoder.extract_barcodes_with_index(di,pm,traces,distances,
        fov=0,cropWidth=0,zIndex=int(z_index),globalAligner=None,
        minimumArea=4,outputLabels=True,extractIntensityTraces=True)
    if np.any(labels[~mask]!=0):
        raise AssertionError('Barcode labels extend outside complete support')
    origin=np.asarray(origin_xy_um,dtype=float)
    frame['global_x']=origin[0]+frame.x.astype(float)*MPP
    frame['global_y']=origin[1]+frame.y.astype(float)*MPP
    frame['global_z']=float(physical_z_um)
    frame['physical_z_um']=float(physical_z_um)
    frame['is_blank']=frame.barcode_id.isin(set(codebook.get_blank_indexes()))
    frame['pilot_barcode_id']=[f'tile0_z{int(z_index):03d}_label{int(v)}' for v in frame.unique_id]
    mixed=(owners.min(axis=0)!=owners.max(axis=0))&mask
    seam=np.zeros(mask.shape,bool)
    for owner in owners:
        seam|=ndimage.maximum_filter(owner,size=5,mode='constant',cval=0)!=ndimage.minimum_filter(owner,size=5,mode='constant',cval=0)
    boundary=~ndimage.minimum_filter(mask,size=5,mode='constant',cval=0)
    flags=dict(mixed_source_across_bits=mixed,source_seam_within_2px=seam&mask,
               coverage_boundary_within_2px=boundary&mask)
    ids=frame.unique_id.to_numpy(dtype=int)
    for name,raster in flags.items():
        touched=np.bincount(labels.ravel(),weights=raster.ravel(),minlength=int(labels.max())+1)>0
        frame[name]=touched[ids]
    yi=np.clip(np.rint(frame.y).astype(int),0,mask.shape[0]-1)
    xi=np.clip(np.rint(frame.x).astype(int),0,mask.shape[1]-1)
    for bit in range(len(owners)):
        frame[f'source_owner_bit_{bit}']=owners[bit,yi,xi]
    if len(frame) and (not np.all(frame.z.to_numpy()==z_index) or not frame.pilot_barcode_id.is_unique):
        raise AssertionError('Unstable canonical Z index or barcode IDs')
    report=dict(z_index=int(z_index),physical_z_um=float(physical_z_um),
        acquired_pixels=int(support.sum()),decode_pixels=int(mask.sum()),
        outside_support_assigned_pixels=int(np.count_nonzero(di[~mask]>=0)),
        raw_barcodes=len(frame),raw_blank_barcodes=int(frame.is_blank.sum()),
        mixed_source_barcodes=int(frame.mixed_source_across_bits.sum()),
        seam_barcodes=int(frame.source_seam_within_2px.sum()),
        coverage_boundary_barcodes=int(frame.coverage_boundary_within_2px.sum()))
    return frame,dict(barcode_id=di,magnitude=pm,distance=distances,labels=labels,decode_mask=mask),report


def _runtime(output):
    for name in ('tmp','cache'):(output/name).mkdir()
    os.environ.update(TMPDIR=str(output/'tmp'),TEMP=str(output/'tmp'),TMP=str(output/'tmp'),
        XDG_CACHE_HOME=str(output/'cache'),MPLCONFIGDIR=str(output/'cache/matplotlib'),
        NUMBA_CACHE_DIR=str(output/'cache/numba'),PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    _install_write_guard(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--optimization-root',type=Path,required=True)
    parser.add_argument('--z-index',type=int,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--fork',type=Path,default=DEFAULT_FORK)
    parser.add_argument('--num-threads',type=int,default=1)
    parser.add_argument('--save-images',action='store_true')
    args=parser.parse_args();started=time.time()
    manifest=resolve_decode_manifest(args.manifest)
    selected=[p for p in manifest['planes'] if p['z_index']==args.z_index]
    if len(selected)!=1:raise ValueError('Requested Z index absent or duplicated')
    plane=selected[0]
    sf,bg,opt_inputs=load_accepted_optimization(args.optimization_root,manifest)
    # When this plane contributed to fitting, its exact bytes must match.
    for name in ('images_path','mask_path'):
        expected=opt_inputs['input_sha256'].get(str(Path(plane[name]).resolve()))
        if expected is not None and sha256(plane[name])!=expected:
            raise RuntimeError('Optimization input image/support changed')
    source_owner=owners_path(plane)
    output=args.output_root.resolve()/f'z{args.z_index:03d}'
    protected=[args.manifest.resolve(),Path(manifest['codebook_path']),Path(plane['images_path']),Path(plane['mask_path']),source_owner,
               *[args.optimization_root.resolve()/name for name in ('optimization.json','inputs.json','scale_factors.npy','backgrounds.npy')]]
    protected += [args.fork.resolve()/name for name in ('merlin/util/decoding.py','merlin/data/codebook.py')]
    for source in protected[-2:]:
        expected=opt_inputs['input_sha256'].get(str(source))
        if expected is None or sha256(source)!=expected:
            raise RuntimeError('Canonical decode/codebook source differs from fresh optimization')
    if any(output==p or output in p.parents for p in protected):raise ValueError('Output contains protected input')
    output.mkdir(parents=True,exist_ok=False);_runtime(output)
    fingerprints={str(p):sha256(p) for p in protected}
    registration,cross_models,reg_fingerprints=registration_flags.load_registration(manifest['registration_manifest'],sha256)
    fingerprints.update(reg_fingerprints)
    fingerprints[str(Path(registration_flags.__file__).resolve())]=sha256(registration_flags.__file__)
    Codebook,sink,_,Decoder=load_canonical(args.fork.resolve(),output)
    codebook=Codebook(sink,manifest['codebook_path'],codebookIndex=0,codebookName='FreshVirtualFOV')
    if list(codebook.get_bit_names())!=manifest['bit_names']:raise ValueError('Codebook bit order mismatch')
    image,mask=read_plane(plane,codebook.get_bit_count(),mask_erosion=0)
    owners=load_owners(source_owner,image.shape,mask)
    frame,images,report=decode_arrays(image,mask,owners,codebook,Decoder,sf,bg,args.z_index,
        plane['physical_z_um'],manifest_origin(manifest),args.num_threads)
    frame,registration_report=registration_flags.annotate_registration(frame,images['labels'],plane['physical_z_um'],
        manifest_origin(manifest),registration,cross_models)
    report.update(registration_report)
    frame.to_csv(output/'raw.csv.gz',index=False)
    outputs=['raw.csv.gz','fresh_codebook.csv']
    if args.save_images or args.z_index in (24,74,124):
        np.savez_compressed(output/'decode_images.npz',**images);outputs.append('decode_images.npz')
    if any(sha256(path)!=digest for path,digest in fingerprints.items()):raise RuntimeError('Input changed during decoding')
    report.update(status='COMPLETE',bit_names=manifest['bit_names'],origin_xy_um=manifest_origin(manifest).tolist(),
        microns_per_pixel=MPP,image_shape_yx=list(mask.shape),source_owner_code_to_fov=SOURCE_OWNER_MAP,
        decode_parameters=DECODE_PARAMETERS,
        calibration_status='Fresh converged intensity optimization; no prior fitted calibration',
        optimization_root=str(args.optimization_root.resolve()),
        input_sha256=fingerprints,output_sha256={name:sha256(output/name) for name in outputs},
        runner_sha256=sha256(__file__),all_hashed_inputs_unchanged=True,elapsed_seconds=time.time()-started,
        seam_flag_rule='Any pixel of the retained connected component lies within2px of any per-bit owner boundary',
        z_convention='z is reference dataset index; global_z and physical_z_um are reference physicalmicrons')
    write_json(output/'complete.json',report)
    print(json.dumps({k:report[k] for k in ('z_index','raw_barcodes','raw_blank_barcodes','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
