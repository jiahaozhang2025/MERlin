"""Independent native-coordinate 3D fiducial detection; never reads warp tables."""
import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def numbers(value):
    return np.array([float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', value)])


def raw_metadata(data_dir, raw_root, fov, imaging_round):
    data_dir, raw_root = Path(data_dir), Path(raw_root)
    with (data_dir/'dataorganization.csv').open() as f:
        organization = list(csv.DictReader(f))
    row = next(r for r in organization if int(r['fiducial3DImagingRound']) == imaging_round)
    with (data_dir/'filemap.csv').open() as f:
        file_row = next(r for r in csv.DictReader(f) if int(r['fov']) == fov
                        and int(r['imagingRound']) == imaging_round
                        and r['imageType'] == row['fiducial3DImageType'])
    microscope = json.loads((data_dir/'microscope_parameters.json').read_text())
    frames, zs = numbers(row['fiducial3DStackFrames']).astype(int), numbers(row['fiducial3DzPos'])
    if len(frames) != len(zs) or np.any(np.diff(zs) <= 0):
        raise ValueError('Fiducial frame/z organization is inconsistent')
    path = raw_root/Path(file_row['imagePath']).name
    return path, frames, zs, microscope, {
        'raw_path': str(path), 'resolved_raw_path': str(path.resolve()),
        'file_size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
        'fov': fov, 'round': imaging_round, 'signal': '488_fiducial',
        'frames_zero_based': frames.tolist(), 'z_positions_um': zs.tolist(),
        'microscope': microscope, 'coordinate_order': 'xyz_physical_um',
        'native_orientation': 'metadata transpose and flips; no saved warping',
        'metadata_sha256': {n: hashlib.sha256((data_dir/n).read_bytes()).hexdigest()
                            for n in ['dataorganization.csv', 'filemap.csv', 'microscope_parameters.json']}}


def read_native(path, frames, microscope):
    h, w = microscope['image_dimensions']
    shape = (w, h) if microscope.get('transpose') else (h, w)
    raw = np.empty((len(frames), *shape), np.uint16)
    with Path(path).open('rb') as f:
        for j, index in enumerate(frames):
            f.seek(int(index)*h*w*2)
            plane = np.fromfile(f, '<u2', h*w)
            if plane.size != h*w:
                raise ValueError(f'Incomplete frame {index}')
            plane = plane.reshape(h, w)
            if microscope.get('transpose'): plane = plane.T
            if microscope.get('flip_horizontal'): plane = plane[:, ::-1]
            if microscope.get('flip_vertical'): plane = plane[::-1, :]
            raw[j] = plane
    return raw


def localize(raw, peak_zyx, zs, mpp):
    z, y, x = map(int, peak_zyx)
    if z < 3 or z >= len(zs)-3 or min(y, x) < 7 or y >= raw.shape[1]-7 or x >= raw.shape[2]-7:
        return None, 'boundary'
    patch = raw[z-3:z+4, y-6:y+7, x-6:x+7].astype(float)
    if patch.max() >= 65530:
        return None, 'dtype_saturation'
    pedestal = np.percentile(patch, 20, axis=(1, 2))
    signal = np.maximum(patch-pedestal[:, None, None], 0)
    # Marginal axial peak must be internal and above both neighboring tails.
    profile = signal[:, 4:9, 4:9].sum(axis=(1, 2))
    k = int(np.argmax(profile))
    if k in [0, 6] or profile[k] < 1.20*max(profile[0], profile[-1]):
        return None, 'no_resolved_axial_peak'
    core = signal.copy()
    core[:, :2] = core[:, -2:] = 0
    core[:, :, :2] = core[:, :, -2:] = 0
    # Local support follows the connected central PSF, not a global projection.
    mask = core > core.max()*0.25
    labels, _ = ndimage.label(mask, np.ones((3, 3, 3)))
    apex = np.unravel_index(np.argmax(core), core.shape)
    weights = np.where(labels == labels[apex], core, 0)
    total = weights.sum()
    if total <= 0:
        return None, 'empty'
    zz, yy, xx = np.meshgrid(zs[z-3:z+4], np.arange(y-6, y+7)*mpp,
                              np.arange(x-6, x+7)*mpp, indexing='ij')
    center = np.array([(weights*xx).sum(), (weights*yy).sum(), (weights*zz).sum()])/total
    widths = np.sqrt(np.array([(weights*(xx-center[0])**2).sum(),
                               (weights*(yy-center[1])**2).sum(),
                               (weights*(zz-center[2])**2).sum()])/total)
    if max(widths[:2]) > .8 or widths[2] > 2.2 or min(widths[:2]) < .035:
        return None, 'implausible_core_width'
    if np.linalg.norm(center[:2]/mpp-[x, y]) > 4.5:
        return None, 'peak_displaced_outside_refinement_window'
    return {'xyz_um': center, 'width_xyz_um': widths, 'amplitude': float(core.max()),
            'integrated_core': float(total), 'axial_profile': profile, 'cutout': patch}, None


def detect(raw, zs, mpp, threshold_mad=7., max_candidates=8000):
    if len(zs) < 9 or not np.allclose(np.diff(zs), np.median(np.diff(zs))):
        raise ValueError('Detector requires a uniform z grid with at least nine planes')
    if raw.shape[1] % 2 or raw.shape[2] % 2:
        raise ValueError('Detector currently requires even XY dimensions')
    coarse = np.empty((len(zs), raw.shape[1]//2, raw.shape[2]//2), np.float32)
    plane_profiles = []
    for i, plane in enumerate(raw):
        coarse[i] = plane.astype(np.float32).reshape(plane.shape[0]//2, 2, plane.shape[1]//2, 2).mean((1, 3))
        plane_profiles.append(np.percentile(plane[::4, ::4], [50, 99, 99.9, 99.99]).tolist())
    response = ndimage.gaussian_filter(coarse, (.65, .65, .65))
    response -= ndimage.gaussian_filter(coarse, (1.5, 2.2, 2.2))
    del coarse
    sample = response[:, ::4, ::4]
    med = np.median(sample, axis=(1, 2))
    noise = 1.4826*np.median(np.abs(sample-med[:, None, None]), axis=(1, 2))
    cutoff = med + threshold_mad*np.maximum(noise, .1)
    peaks = (response == ndimage.maximum_filter(response, (5, 5, 5))) & (response > cutoff[:, None, None])
    coords = np.column_stack(np.nonzero(peaks))
    heights = response[tuple(coords.T)]
    selected = np.argsort(-heights, kind='stable')[:max_candidates]
    coords, heights = coords[selected], heights[selected]
    del peaks, response
    records, reasons = [], {}
    for (z, cy, cx), height in zip(coords, heights):
        record, reason = localize(raw, (z, 2*cy+1, 2*cx+1), zs, mpp)
        if reason:
            reasons[reason] = reasons.get(reason, 0)+1
        else:
            record['response'] = float(height)
            records.append(record)
    # Merge multiple maxima from one physical bead before any train/test split.
    keep = []
    if records:
        positions = np.array([r['xyz_um'] for r in records])
        tree = cKDTree(positions/np.array([.6, .6, 1.5]))
        removed = set()
        for i in np.argsort([-r['amplitude'] for r in records]):
            if int(i) in removed: continue
            keep.append(int(i))
            removed.update(tree.query_ball_point(positions[i]/[.6, .6, 1.5], 1.))
    records = [records[i] for i in keep]
    xyz = np.array([r['xyz_um'] for r in records]).reshape(-1, 3)
    stats = {'raw_3d_maxima': int(len(heights)), 'candidate_cap': max_candidates,
             'retained_beads': len(records), 'rejections': reasons, 'threshold_mad': threshold_mad,
             'detector_downsample_xy': 2, 'refinement': 'native 7z x13x13 connected core centroid',
             'plane_raw_percentiles_50_99_999_9999': plane_profiles,
             'z_histogram_edges_um': np.arange(0, 161, 10).tolist(),
             'z_histogram_counts': np.histogram(xyz[:, 2], np.arange(0, 161, 10))[0].tolist(),
             'z_percentiles_10_50_90_um': np.percentile(xyz[:, 2], [10, 50, 90]).tolist() if len(xyz) else None,
             'limitations': ['Detected pointlike 488 features are bead candidates, not independently identified beads.',
                            '3D core moments are not calibrated localization uncertainties or PSF fits.',
                            'Near-boundary and axially unresolved features are excluded; inspect raw profiles if cloud is sparse.',
                            'Detector and thresholds are frozen for this pilot, not calibrated to molecule decoding tolerance.']}
    arrays = {'xyz_um': xyz, 'width_xyz_um': np.array([r['width_xyz_um'] for r in records]).reshape(-1, 3),
              'amplitude': np.array([r['amplitude'] for r in records]),
              'response': np.array([r['response'] for r in records]),
              'sample_cutouts': np.array([r['cutout'] for r in records[:24]]),
              'sample_axial_profiles': np.array([r['axial_profile'] for r in records[:24]])}
    return arrays, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--fov', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    t0 = time.time()
    path, frames, zs, microscope, meta = raw_metadata(args.data_dir, args.raw_root, args.fov, args.round)
    raw = read_native(path, frames, microscope)
    arrays, stats = detect(raw, zs, microscope['microns_per_pixel'])
    meta.update(detection=stats, elapsed_seconds=time.time()-t0,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2, allow_nan=False))
    print(json.dumps({'output': str(output), 'beads': stats['retained_beads'],
                      'z_percentiles': stats['z_percentiles_10_50_90_um'], 'seconds': meta['elapsed_seconds']}), flush=True)


if __name__ == '__main__': main()
