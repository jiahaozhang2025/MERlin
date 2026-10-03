"""Independent, alignment-only XY fit from the four native raw Z=0 images.

No volumetric fit is loaded. The separate reference image cannot measure Z.
Native inputs must already have raw.T[:, ::-1] orientation applied.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import tifffile

FOVS = (13, 14, 25, 26)
EDGES = ((13, 14), (13, 26), (14, 25), (25, 26))
ORIGINS = {13: np.array([0., 0.]), 14: np.array([270., 0.]),
           25: np.array([270., -270.]), 26: np.array([0., -270.])}
CFG = dict(microns_per_pixel=.1493, camera_width_px=2048,
           detection_highpass_threshold_minimum=300., detection_noise_multiple=8.,
           detection_minimum_separation_px=7, detection_border_px=5,
           search_um=8., vote_bin_um=.30, train_match_gate_um=.90,
           heldout_match_gate_um=1.50, minimum_training_pairs=12,
           minimum_heldout_pairs=5, holdout_end_fraction=.20,
           xy_median_gate_px=1., xy_p95_gate_px=2.)


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detect(image):
    """Subpixel positive-signal centroids around isolated high-pass maxima."""
    image = np.asarray(image, np.float32)
    high = ndimage.gaussian_filter(image, .7)-ndimage.gaussian_filter(image, 4.5)
    noise_image = image-ndimage.gaussian_filter(image, 1.)
    noise = float(1.4826*np.median(np.abs(noise_image-np.median(noise_image))))
    threshold = max(CFG['detection_highpass_threshold_minimum'],
                    CFG['detection_noise_multiple']*noise)
    maxima = (high == ndimage.maximum_filter(high, CFG['detection_minimum_separation_px'])) & (high > threshold)
    border = CFG['detection_border_px']
    maxima[:border] = False; maxima[-border:] = False
    maxima[:, :border] = False; maxima[:, -border:] = False
    yy, xx = np.nonzero(maxima)
    points, amplitudes, rejected = [], [], 0
    gy, gx = np.mgrid[-3:4, -3:4]
    for y, x in zip(yy, xx):
        patch = high[y-3:y+4, x-3:x+4]
        if np.max(image[y-2:y+3, x-2:x+3]) >= 65530:
            rejected += 1
            continue
        weights = np.maximum(patch-.15*high[y, x], 0)
        total = weights.sum()
        if total <= 0:
            rejected += 1
            continue
        dx, dy = float((weights*gx).sum()/total), float((weights*gy).sum()/total)
        if abs(dx) > 1.5 or abs(dy) > 1.5:
            rejected += 1
            continue
        points.append([x+dx, y+dy]); amplitudes.append(float(high[y, x]))
    return np.asarray(points, float).reshape(-1, 2)*CFG['microns_per_pixel'], np.asarray(amplitudes), {
        'noise_robust_adu': noise, 'threshold_highpass_adu': threshold,
        'local_maxima': int(len(yy)), 'retained_features': len(points),
        'saturated_or_unstable_rejected': rejected,
        'detector_note': 'Features are bright pointlike reference-image peaks; identity as fiducial beads requires image inspection.'}


def split_edge(a, b, seed):
    width = CFG['camera_width_px']*CFG['microns_per_pixel']
    low = np.maximum(np.zeros(2), seed)
    high = np.minimum(np.full(2, width), seed+width)
    axis = int(np.argmax(high-low))
    left, right = low[axis]+.2*(high[axis]-low[axis]), low[axis]+.8*(high[axis]-low[axis])
    guard = CFG['search_um']+CFG['heldout_match_gate_um']
    def choose(p):
        inside = np.all((p >= low-CFG['search_um']) & (p <= high+CFG['search_um']), axis=1)
        held = inside & ((p[:, axis] <= left) | (p[:, axis] >= right))
        training = inside & (p[:, axis] > left+guard) & (p[:, axis] < right-guard)
        return np.flatnonzero(training), np.flatnonzero(held)
    ta, ha = choose(a); tb, hb = choose(b+seed)
    return ta, tb, ha, hb, dict(long_axis='xy'[axis], overlap_low_um=low.tolist(),
        overlap_high_um=high.tolist(), heldout_end_boundaries_um=[left, right], guard_um=guard,
        rule='Outer 20% at both ends held out; central 60% trains after guard bands. All graph-wide heldout feature IDs excluded from training.')


def reciprocal(a, b, shift, gate):
    if min(len(a), len(b)) == 0:
        return np.empty((0, 2), int), np.empty((0, 2))
    aa = cKDTree(a); bb = cKDTree(b+shift)
    dist, ai = aa.query(b+shift)
    _, bi = bb.query(a)
    take = (bi[ai] == np.arange(len(b))) & (dist <= gate)
    pairs = np.column_stack([ai[take], np.flatnonzero(take)])
    return pairs, a[pairs[:, 0]]-b[pairs[:, 1]]-shift


def fit_translation(a, b, seed):
    radius = CFG['search_um']
    bins = np.linspace(-radius, radius, int(np.ceil(2*radius/CFG['vote_bin_um']))+1)
    hist = np.zeros((len(bins)-1, len(bins)-1))
    tree = cKDTree(a)
    candidates = 0
    for p in b+seed:
        neighbors = tree.query_ball_point(p, radius*np.sqrt(2))
        d = a[neighbors]-p
        d = d[np.all(np.abs(d) <= radius, axis=1)]
        if len(d):
            hist += np.histogramdd(d, bins=[bins, bins])[0]
            candidates += len(d)
    if not candidates:
        raise ValueError('No point pairs in fixed stage search')
    smooth = ndimage.gaussian_filter(hist, .7)
    peak = np.unravel_index(np.argmax(smooth), smooth.shape)
    if any(i in (0, len(bins)-2) for i in peak):
        raise ValueError('Translation vote at search boundary')
    shift = seed+np.array([(bins[i]+bins[i+1])/2 for i in peak])
    initial = shift.copy()
    for iteration in range(20):
        gate = 1.5 if iteration < 2 else CFG['train_match_gate_um']
        pairs, _ = reciprocal(a, b, shift, gate)
        if len(pairs) < CFG['minimum_training_pairs']:
            raise ValueError('Too few reciprocal training pairs: %d' % len(pairs))
        differences = a[pairs[:, 0]]-b[pairs[:, 1]]
        result = least_squares(lambda t: ((t-differences)/CFG['microns_per_pixel']).ravel(),
                               np.median(differences, axis=0), loss='soft_l1', f_scale=1.)
        change = np.linalg.norm(result.x-shift)
        shift = result.x
        if iteration >= 2 and change < 1e-7:
            break
    if np.any(np.abs(shift-seed) >= radius):
        raise ValueError('Refined translation exceeds stage search')
    pairs, residual = reciprocal(a, b, shift, CFG['train_match_gate_um'])
    return shift, pairs, residual, dict(candidate_pairs=candidates, initial_shift_um=initial.tolist(),
        vote_peak=float(smooth[peak]), iterations=iteration+1)


def stats(residual, denominator=None):
    radius = np.linalg.norm(residual, axis=-1)/CFG['microns_per_pixel']
    out = dict(pairs=len(radius), median_pixels=float(np.median(radius)) if len(radius) else None,
               p95_pixels=float(np.percentile(radius, 95)) if len(radius) else None,
               fraction_above_one_pixel=float(np.mean(radius > 1)) if len(radius) else None)
    if denominator is not None:
        out['matched_fraction_of_smaller_feature_set'] = len(radius)/max(1, denominator)
    return out


def basis(points):
    q = (np.asarray(points)-CFG['camera_width_px']*CFG['microns_per_pixel']/2)/100.
    return np.stack([np.ones(q.shape[:-1]), q[..., 0], q[..., 1], q[..., 0]**2,
                     q[..., 1]**2, q[..., 0]*q[..., 1]], axis=-1)


def apply_global(points, fov, model):
    p = np.asarray(points, float)
    return p+np.asarray(model['models'][str(fov)]['stage_origin_xy_um'])+basis(p)@np.asarray(model['models'][str(fov)]['coefficients_6x2_um'])


def inverse_global(points, fov, model):
    points = np.asarray(points, float)
    target = points-np.asarray(model['models'][str(fov)]['stage_origin_xy_um'])
    native = target.copy()
    beta = np.asarray(model['models'][str(fov)]['coefficients_6x2_um'])
    for _ in range(40):
        updated = target-basis(native)@beta
        if np.max(np.abs(updated-native), initial=0) < 1e-7:
            return updated
        native = updated
    raise RuntimeError('XY inverse did not converge')


# Public renderer API: (...,2) native/global XY in micrometers.
apply_xy = apply_global
inverse_xy = inverse_global


def joint_model(points, edges, quadratic):
    unknown = (14, 25, 26)
    nfeature = 6 if quadratic else 1
    ncols = len(unknown)*nfeature*2
    count = sum(len(e['training_pairs']) for e in edges)*2
    design, base = np.zeros((count, ncols)), np.empty(count)
    offset = 0
    for edge in edges:
        k, n, pairs = edge['k'], edge['n'], edge['training_pairs']
        aa, bb = points[k][pairs[:, 0]], points[n][pairs[:, 1]]
        base[offset:offset+2*len(pairs)] = ((aa+ORIGINS[k]-bb-ORIGINS[n])/CFG['microns_per_pixel']).ravel()
        for fov, pp, sign in [(k, aa, 1), (n, bb, -1)]:
            if fov == 13:
                continue
            feature = basis(pp)[:, :nfeature]
            fi = unknown.index(fov)
            for dim in range(2):
                rows = offset+2*np.arange(len(pp))+dim
                cols = fi*nfeature*2+2*np.arange(nfeature)+dim
                design[np.ix_(rows, cols)] = sign*feature/CFG['microns_per_pixel']
        offset += 2*len(pairs)
    if quadratic:
        prior = np.zeros((30, ncols)); row = 0
        for fi in range(3):
            for feature in range(1, 6):
                for dim in range(2):
                    prior[row, fi*12+feature*2+dim] = 1.
                    row += 1
        design = np.concatenate([design, prior]); base = np.concatenate([base, np.zeros(len(prior))])
    result = least_squares(lambda p: design@p+base, np.zeros(ncols), jac=lambda p: design.copy(),
                           loss='soft_l1', f_scale=1., max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    model = dict(model_type='z0_independent_xy_quadratic' if quadratic else 'z0_independent_xy_translation',
        coordinate_units='micrometers', coordinate_order='xy', anchor_fov=13,
        normalization=dict(center_xy_um=[CFG['camera_width_px']*CFG['microns_per_pixel']/2]*2, scale_xy_um=[100., 100.]),
        features=['1', 'qx', 'qy', 'qx^2', 'qy^2', 'qx*qy'],
        equation='global_xy = native_xy + stage_origin_xy + basis(native_xy) @ coefficients', models={},
        fit=dict(success=bool(result.success), evaluations=int(result.nfev), cost=float(result.cost),
                 regularization='1 um normalized nonconstant coefficient scale; 100 um coordinate normalization' if quadratic else None))
    for fov in FOVS:
        coefficients = np.zeros((6, 2))
        if fov != 13:
            i = unknown.index(fov)*nfeature*2
            coefficients[:nfeature] = result.x[i:i+nfeature*2].reshape(nfeature, 2)
        model['models'][str(fov)] = dict(stage_origin_xy_um=ORIGINS[fov].tolist(), coefficients_6x2_um=coefficients.tolist())
    return model


def assess(points, edges, model, archive):
    records, reasons = [], []
    for edge in edges:
        k, n = edge['k'], edge['n']
        record = dict(k=k, n=n)
        for split in ['training', 'heldout']:
            pair = edge[split+'_pairs']
            residual = apply_global(points[k][pair[:, 0]], k, model)-apply_global(points[n][pair[:, 1]], n, model)
            metric = stats(residual)
            record[split] = metric
            archive[f'{k}_{n}_{model["model_type"]}_{split}_residual_xy_um'] = residual
            minimum = CFG['minimum_training_pairs'] if split == 'training' else CFG['minimum_heldout_pairs']
            if len(pair) < minimum:
                reasons.append(f'{k}-{n} {split} has {len(pair)} pairs, requires {minimum}')
            if split == 'heldout' and len(pair):
                if metric['median_pixels'] > CFG['xy_median_gate_px'] or metric['p95_pixels'] > CFG['xy_p95_gate_px']:
                    reasons.append(f'{k}-{n} heldout residual exceeds frozen 1 px median / 2 px p95 gates')
        records.append(record)
    width = CFG['camera_width_px']*CFG['microns_per_pixel']
    xx, yy = np.meshgrid(np.linspace(0, width, 21), np.linspace(0, width, 21))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    jacobian = {}
    for fov in FOVS:
        q = (grid-width/2)/100.
        beta = np.asarray(model['models'][str(fov)]['coefficients_6x2_um'])
        jj = np.broadcast_to(np.eye(2), (len(grid), 2, 2)).copy()
        jj[:, :, 0] += (beta[1]+2*q[:, 0, None]*beta[3]+q[:, 1, None]*beta[5])/100.
        jj[:, :, 1] += (beta[2]+2*q[:, 1, None]*beta[4]+q[:, 0, None]*beta[5])/100.
        sv = np.linalg.svd(jj, compute_uv=False)
        det = np.linalg.det(jj)
        inverse_error = np.max(np.linalg.norm(inverse_global(apply_global(grid, fov, model), fov, model)-grid, axis=1))
        jacobian[str(fov)] = dict(determinant_range=[float(det.min()), float(det.max())],
            singular_value_range=[float(sv.min()), float(sv.max())], inverse_max_error_um=float(inverse_error))
        if det.min() < .95 or det.max() > 1.05 or sv.min() < .97 or sv.max() > 1.03 or inverse_error > 1e-5:
            reasons.append(f'FOV {fov} fails mild/invertible deformation gate')
    model['assessment'] = dict(passes_frozen_gates=not reasons, reasons=reasons, edges=records, jacobian=jacobian)


def plot(points, edges, report, output, archive):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    colors = {13:'#a642af', 14:'#308852', 25:'#277bba', 26:'#d78828'}
    for fov in FOVS:
        p = points[fov]+ORIGINS[fov]
        axes[0, 0].scatter(p[:, 0], p[:, 1], s=3, color=colors[fov], alpha=.5, label=f'FOV {fov}: {len(p)} peaks')
    axes[0, 0].set(title='Raw Z=0 feature detections, stage placement', xlabel='X (um)', ylabel='Y (um)')
    axes[0, 0].set_aspect('equal'); axes[0, 0].invert_yaxis(); axes[0, 0].legend(fontsize=8)
    for ei, edge in enumerate(edges):
        k, n = edge['k'], edge['n']; pair = edge['heldout_pairs']
        p = points[k][pair[:, 0]]+ORIGINS[k]
        axes[0, 1].scatter(p[:, 0], p[:, 1], s=10, label=f'{k}-{n}: {len(pair)} held-out pairs')
        for mi, model in enumerate(report['candidate_models']):
            residual = archive[f'{k}_{n}_{model["model_type"]}_heldout_residual_xy_um']
            rr = np.sort(np.linalg.norm(residual, axis=1)/CFG['microns_per_pixel'])
            if len(rr):
                axes[1, mi].plot(rr, np.arange(1, len(rr)+1)/len(rr), label=f'{k}-{n}')
    axes[0, 1].set(title='Validation points excluded from every training edge', xlabel='X (um)', ylabel='Y (um)')
    axes[0, 1].set_aspect('equal'); axes[0, 1].invert_yaxis(); axes[0, 1].legend(fontsize=8)
    for ax, name in zip(axes[1], ['Joint translation', 'Joint quadratic']):
        ax.set(title=f'{name}: same frozen validation pairs', xlabel='XY disagreement (pixels)', ylabel='Cumulative fraction', xlim=(0, 10), ylim=(0, 1.02))
        ax.axvline(1, color='grey', linestyle=':'); ax.axvline(2, color='grey', linestyle=':'); ax.legend()
    fig.suptitle('Independent Z=0 reference-image alignment; no Z correction measured', fontsize=14)
    fig.tight_layout(); fig.savefig(output/'z0_matches_and_errors.png', dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), parameters=CFG,
        scope='Independent XY alignment of native raw frame 0 at recorded acquisition Z=0. No decoding, no 3D fit reuse.',
        features={}, edges=[], candidate_models=[], status='incomplete', limitations=[
            'A single plane provides XY registration only; it does not determine relative physical Z or volumetric distortion.',
            'Validation residuals are conditional on reciprocal candidates within 1.5 um of training translation; unmatched features remain unvalidated.',
            'Overlap features constrain overlaps; model behavior in FOV interiors is not independently measured.'])
    points, archive, preliminary = {}, {}, []
    for fov in FOVS:
        path = args.input/f'fov{fov}_frame0.tif'
        image = tifffile.imread(path)
        if image.shape != (2048, 2048):
            raise ValueError(f'Unexpected image dimensions for {path}: {image.shape}')
        points[fov], amplitude, detection = detect(image)
        report['features'][str(fov)] = dict(path=str(path.resolve()), sha256=hash_file(path), **detection)
        archive[f'fov{fov}_native_xy_um'] = points[fov]
        archive[f'fov{fov}_amplitude'] = amplitude
    held_global = {fov:set() for fov in FOVS}
    for k, n in EDGES:
        seed = ORIGINS[n]-ORIGINS[k]
        ta, tb, ha, hb, info = split_edge(points[k], points[n], seed)
        held_global[k].update(ha.tolist()); held_global[n].update(hb.tolist())
        preliminary.append((k, n, seed, ta, tb, ha, hb, info))
    edges = []
    for k, n, seed, ta, tb, ha, hb, info in preliminary:
        ta = np.array([i for i in ta if i not in held_global[k]], int)
        tb = np.array([i for i in tb if i not in held_global[n]], int)
        record = dict(k=k, n=n, stage_seed_xy_um=seed.tolist(), split=info,
            training_feature_counts=[len(ta), len(tb)], heldout_feature_counts=[len(ha), len(hb)])
        try:
            shift, pairs, residual, diagnostics = fit_translation(points[k][ta], points[n][tb], seed)
            reverse, _, _, _ = fit_translation(points[n][tb], points[k][ta], -seed)
            hp, hr = reciprocal(points[k][ha], points[n][hb], shift, CFG['heldout_match_gate_um'])
            training_pairs = np.column_stack([ta[pairs[:, 0]], tb[pairs[:, 1]]])
            heldout_pairs = np.column_stack([ha[hp[:, 0]], hb[hp[:, 1]]])
            record.update(shift_xy_um=shift.tolist(), reverse_closure_px=float(np.linalg.norm(shift+reverse)/CFG['microns_per_pixel']),
                diagnostics=diagnostics, training=stats(residual), heldout=stats(hr, min(len(ha), len(hb))), success=True)
            edge = dict(k=k, n=n, training_pairs=training_pairs, heldout_pairs=heldout_pairs, shift=shift)
            edges.append(edge)
            for name, value in [('training_pairs', training_pairs), ('heldout_pairs', heldout_pairs),
                                ('independent_translation_training_residual_xy_um', residual), ('independent_translation_heldout_residual_xy_um', hr)]:
                archive[f'{k}_{n}_{name}'] = value
        except ValueError as error:
            record.update(success=False, failure=str(error))
        report['edges'].append(record)
    if len(edges) == len(EDGES):
        translations = {(e['k'],e['n']): e['shift'] for e in edges}
        cycle = translations[13,14]+translations[14,25]+translations[25,26]-translations[13,26]
        report['independent_translation_cycle'] = dict(xy_um=cycle.tolist(), length_pixels=float(np.linalg.norm(cycle)/CFG['microns_per_pixel']))
        for quadratic in (False, True):
            model = joint_model(points, edges, quadratic)
            assess(points, edges, model, archive)
            report['candidate_models'].append(model)
            (args.output/(model['model_type']+'.json')).write_text(json.dumps(model, indent=2))
        passing = [m for m in report['candidate_models'] if m['assessment']['passes_frozen_gates']]
        report['preferred_candidate'] = passing[0]['model_type'] if passing else None
        report['status'] = 'alignment_candidate_for_review' if passing else 'alignment_candidates_fail_some_frozen_gates'
        plot(points, edges, report, args.output, archive)
    else:
        report['status'] = 'insufficient_coherent_reference_image_matches'
    np.savez_compressed(args.output/'z0_feature_matches.npz', **archive)
    (args.output/'z0_alignment.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in ['status','features','edges']}, indent=2))


if __name__ == '__main__':
    main()
