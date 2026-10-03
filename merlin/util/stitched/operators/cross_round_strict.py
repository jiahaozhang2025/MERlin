"""Fresh registration of two stitched fiducial point clouds.

Moving-global XYZ -> reference-global XYZ. No saved per-FOV cross-round warp
or intensity calibration is used. A broad density correlation initializes
translation, then fresh reciprocal feature ties fit an affine and one mildly
regularized quadratic candidate. Raw-image validation is required separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage, signal
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


MPP = .1493
SEAM_REGIONS = []  # Explicit full-field reference seam boxes supplied by runner.
AFFINE = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
QUADRATIC = AFFINE + [[2,0,0], [0,2,0], [0,0,2], [1,1,0], [1,0,1], [0,1,1]]
CFG = dict(xy_search_um=150., density_bin_um=1., z_search_um=20.,
           block_size_xyz_um=[60.,60.,20.], split_guard_xyz_um=[3.,3.,1.],
           dedup_xy_um=.25, dedup_z_um=.5, minimum_training_pairs=100,
           heldout_pair_gate_xyz_um=[2.,2.,5.],
           xy_median_gate_px=1., xy_p95_gate_px=2.,
           z_median_gate_um=.5, z_p95_gate_um=1.)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_cloud(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k:data[k] for k in data.files if data[k].ndim >= 1 and data[k].shape[0] == len(data['xyz_um'])}
    xyz = np.asarray(result['xyz_um'], float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Invalid stitched XYZ cloud')
    if len(xyz) < 200:
        raise ValueError('Too few stitched features for a cross-round fit')
    result['xyz_um'] = xyz
    result.setdefault('source_fov', np.zeros(len(xyz), dtype=int))
    result.setdefault('source_index', np.arange(len(xyz)))
    result.setdefault('amplitude', np.ones(len(xyz)))
    result.setdefault('within_round_heldout', np.zeros(len(xyz), bool))
    return result


def deduplicate(cloud):
    """Remove residual near-identical cross-FOV observations before splitting."""
    points = cloud['xyz_um']
    scale = np.array([CFG['dedup_xy_um'], CFG['dedup_xy_um'], CFG['dedup_z_um']])
    tree = cKDTree(points / scale)
    removed = np.zeros(len(points), bool)
    keep, duplicate_counts = [], []
    for i in np.argsort(-np.asarray(cloud['amplitude']), kind='stable'):
        if removed[i]:
            continue
        neighbors = np.asarray(tree.query_ball_point(points[i] / scale, 1.), dtype=int)
        # Within-FOV detections were already de-duplicated by extraction; do
        # not merge nearby distinct detections from that same source image.
        neighbors = neighbors[(cloud['source_fov'][neighbors] != cloud['source_fov'][i]) & ~removed[neighbors]]
        inherited_heldout = bool(cloud['within_round_heldout'][i] or np.any(cloud['within_round_heldout'][neighbors]))
        keep.append(int(i)); duplicate_counts.append((len(neighbors) + 1, inherited_heldout))
        removed[neighbors] = True; removed[i] = True
    keep = np.asarray(keep, dtype=int)
    order = np.argsort(keep)
    keep = keep[order]
    result = {k:np.asarray(v)[keep] for k,v in cloud.items()}
    result['pooled_input_index'] = keep
    result['near_duplicate_count'] = np.asarray(duplicate_counts, dtype=int)[order,0]
    result['within_round_heldout'] = np.asarray(duplicate_counts, dtype=int)[order,1].astype(bool)
    return result


def _features(points, model):
    q = (np.asarray(points, dtype=float) - model['normalization']['center_xyz_um']) / model['normalization']['scale_xyz_um']
    if model['powers'] == AFFINE or model['powers'] == QUADRATIC:
        # Millions of pixel coordinates use this path during raw sampling;
        # avoid generic 3D powers/products for each of the10 basis terms.
        x,y,z=q[...,0],q[...,1],q[...,2]
        terms=[np.ones(q.shape[:-1]),x,y,z]
        if model['powers'] == QUADRATIC:
            terms += [x*x,y*y,z*z,x*y,x*z,y*z]
        return np.stack(terms,axis=-1)
    return np.stack([np.prod(q ** np.asarray(power), axis=-1) for power in model['powers']], axis=-1)


def apply_cross(points, model):
    points = np.asarray(points, dtype=float)
    if points.shape[-1:] != (3,):
        raise ValueError('Expected [...,3] XYZ micrometers')
    return points + _features(points, model) @ np.asarray(model['coefficients_um'])


def inverse_cross(points, model, tolerance_um=1e-7, max_iterations=80):
    points = np.asarray(points, dtype=float)
    native = points - np.asarray(model['coefficients_um'])[0]
    for _ in range(max_iterations):
        updated = points - _features(native, model) @ np.asarray(model['coefficients_um'])
        if np.max(np.abs(updated-native), initial=0.) < tolerance_um:
            return updated
        native = updated
    raise RuntimeError('Cross-round inverse did not converge')


def jacobian(points, model):
    points = np.asarray(points, dtype=float)
    scale = np.asarray(model['normalization']['scale_xyz_um'])
    q = (points - model['normalization']['center_xyz_um']) / scale
    coeff = np.asarray(model['coefficients_um'])
    result = np.broadcast_to(np.eye(3), (*points.shape[:-1],3,3)).copy()
    for term,power in enumerate(model['powers']):
        for axis in range(3):
            if power[axis]:
                derivative_power = np.array(power); derivative_power[axis] -= 1
                value = power[axis] * np.prod(q ** derivative_power, axis=-1) / scale[axis]
                result[..., :, axis] += value[...,None] * coeff[term]
    return result


def spatial_split(points, inherited_holdout=None):
    """Fixed spatial/depth blocks; does not use fit residuals or intensities."""
    size = np.asarray(CFG['block_size_xyz_um'])
    guard = np.asarray(CFG['split_guard_xyz_um'])
    cell = np.floor(points / size).astype(np.int64)
    fold = np.mod(cell[:,0] + 2*cell[:,1] + 3*cell[:,2], 5)
    inside = np.mod(points, size)
    away = np.all((inside >= guard) & (inside <= size-guard), axis=1)
    inherited = np.zeros(len(points), bool) if inherited_holdout is None else np.asarray(inherited_holdout, bool)
    held = ((fold == 0) & away) | inherited
    train = (fold != 0) & away & ~inherited
    return train, held, cell


def initial_translation(reference, moving, search_xy_um=150., search_z_um=20.):
    """Image-style broad XY point-density FFT, followed by a fresh Z vote."""
    bin_um = CFG['density_bin_um']
    low = np.floor(np.minimum(reference[:,:2].min(axis=0), moving[:,:2].min(axis=0)) / bin_um) * bin_um - 2*bin_um
    high = np.ceil(np.maximum(reference[:,:2].max(axis=0), moving[:,:2].max(axis=0)) / bin_um) * bin_um + 2*bin_um
    edges = [np.arange(low[d], high[d] + bin_um, bin_um) for d in range(2)]
    images = []
    for points in (reference, moving):
        hist = np.histogram2d(points[:,0], points[:,1], bins=edges)[0].astype(np.float32)
        hist = ndimage.gaussian_filter(hist, .7)
        hist -= ndimage.gaussian_filter(hist, 10.)
        images.append(hist)
    corr = signal.fftconvolve(images[0], images[1][::-1,::-1], mode='full')
    lagx = (np.arange(corr.shape[0]) - images[1].shape[0] + 1) * bin_um
    lagy = (np.arange(corr.shape[1]) - images[1].shape[1] + 1) * bin_um
    allowed = (np.abs(lagx[:,None]) <= search_xy_um) & (np.abs(lagy[None,:]) <= search_xy_um)
    corr[~allowed] = -np.inf
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    xy = np.array([lagx[peak[0]], lagy[peak[1]]])
    other = corr.copy()
    other[max(0,peak[0]-8):peak[0]+9,max(0,peak[1]-8):peak[1]+9] = -np.inf
    runner_up = float(other.max())
    tree = cKDTree(reference[:,:2])
    z_differences = []
    for p in moving:
        neighbors = tree.query_ball_point(p[:2] + xy, 3.)
        if neighbors:
            delta = reference[neighbors,2] - p[2]
            z_differences.extend(delta[np.abs(delta) <= search_z_um].tolist())
    if len(z_differences) < 30:
        raise RuntimeError('Broad density XY peak has too little axial-vote evidence')
    z_edges = np.arange(-search_z_um, search_z_um + .25001, .25)
    z_hist = np.histogram(z_differences, bins=z_edges)[0]
    z_smooth = ndimage.gaussian_filter1d(z_hist.astype(float), 2.)
    zp = int(np.argmax(z_smooth))
    shift = np.r_[xy, .5*(z_edges[zp]+z_edges[zp+1])]
    return shift, dict(method='Fresh full stitched-cloud density FFT XY followed by local-XY Z-difference vote',
        density_bin_um=bin_um, shift_xyz_um=shift.tolist(),
        xy_peak_value=float(corr[peak]), xy_runner_up_outside_8um=float(runner_up),
        xy_peak_to_runner_up_ratio=float(corr[peak]/runner_up) if runner_up > 0 else None,
        z_vote_pairs=len(z_differences), z_histogram_edges_um=z_edges.tolist(),
        z_histogram_counts=z_hist.tolist(), xy_search_um=search_xy_um, z_search_um=search_z_um,
        search_boundary_hit=bool(np.any(np.abs(xy) >= search_xy_um-bin_um) or zp in (0,len(z_hist)-1)))


def reciprocal(reference, transformed_moving, gate):
    """Mutual nearest feature ties in anisotropically scaled XYZ space."""
    if not len(reference) or not len(transformed_moving):
        return np.empty((0,2), dtype=int)
    scale = np.asarray(gate, dtype=float)
    rt, mt = cKDTree(reference/scale), cKDTree(transformed_moving/scale)
    distance, ref_index = rt.query(transformed_moving/scale)
    _, mov_index = mt.query(reference/scale)
    good = (distance <= 1.) & (mov_index[ref_index] == np.arange(len(transformed_moving)))
    return np.column_stack([ref_index[good], np.flatnonzero(good)])


def empty_model(moving, kind, shift=None):
    low, high = np.percentile(moving, [1,99], axis=0)
    center = .5*(low+high)
    scale = np.maximum(.5*(high-low), [100.,100.,30.])
    powers = AFFINE if kind == 'affine' else QUADRATIC
    coeff = np.zeros((len(powers),3))
    if shift is not None:
        coeff[0] = shift
    return dict(model_type='fresh_cross_round_'+kind, coordinate_order='xyz',
        coordinate_units='micrometers', direction='moving_stitched_global_to_reference_stitched_global',
        equation='reference = moving + monomials((moving-center)/scale) @ coefficients_um',
        normalization=dict(center_xyz_um=center.tolist(), scale_xyz_um=scale.tolist()),
        powers=powers, coefficients_um=coeff.tolist())


def fit_ties(reference_ties, moving_ties, initial, kind):
    model = empty_model(moving_ties, kind)
    features = _features(moving_ties, model)
    nterm = features.shape[1]
    residual_scale = np.array([MPP, MPP, .5])
    # Fixed priors are weak relative to hundreds of ties: unpenalized rigid
    # translation, 10% normalized linear deformation, 5um quadratic terms.
    prior_width = np.array([np.inf] + [20.,20.,10.] + ([5.]*6 if kind != 'affine' else []))
    start = np.linalg.lstsq(features, apply_cross(moving_ties, initial)-moving_ties, rcond=None)[0]
    def residual(beta):
        b = beta.reshape(nterm,3)
        data = (moving_ties + features@b - reference_ties) / residual_scale
        prior = b[1:] / prior_width[1:,None]
        return np.r_[data.ravel(), prior.ravel()]
    fit = least_squares(residual, start.ravel(), loss='soft_l1', f_scale=1.,
                        max_nfev=150, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    model['coefficients_um'] = fit.x.reshape(nterm,3).tolist()
    model['optimization'] = dict(success=bool(fit.success), message=str(fit.message),
        cost=float(fit.cost), evaluations=int(fit.nfev), pairs=len(moving_ties),
        loss='soft_l1', residual_scale_xyz_um=residual_scale.tolist(),
        prior_width_um=['unpenalized']+prior_width[1:].tolist())
    return model


def stats(residual, minimum=12):
    if len(residual) == 0:
        return dict(status='INSUFFICIENT_EVIDENCE', pairs=0, minimum_pairs=minimum)
    xy = np.linalg.norm(residual[:,:2], axis=1)/MPP
    zz = np.abs(residual[:,2])
    out = dict(pairs=len(residual), minimum_pairs=minimum,
        xy_median_pixels=float(np.median(xy)), xy_p95_pixels=float(np.percentile(xy,95)),
        z_median_um=float(np.median(zz)), z_p95_um=float(np.percentile(zz,95)),
        signed_median_xyz_um=np.median(residual,axis=0).tolist())
    passes = (out['xy_median_pixels']<=CFG['xy_median_gate_px'] and
              out['xy_p95_pixels']<=CFG['xy_p95_gate_px'] and
              out['z_median_um']<=CFG['z_median_gate_um'] and out['z_p95_um']<=CFG['z_p95_gate_um'])
    out['status'] = 'INSUFFICIENT_EVIDENCE' if len(residual)<minimum else ('PASS' if passes else 'FAIL')
    return out


def assess(reference, moving, pairs, model, reference_fovs, *, minimum=12):
    rr, mm = reference[pairs[:,0]], moving[pairs[:,1]]
    residual = apply_cross(mm,model)-rr
    out = dict(overall=stats(residual,50), source_fovs={}, depth_bins=[], spatial_blocks=[], seams={})
    for fov in sorted(np.unique(reference_fovs)):
        chosen = reference_fovs[pairs[:,0]] == fov
        out['source_fovs'][str(fov)] = stats(residual[chosen],minimum)
    for low in range(0,150,25):
        selected = (rr[:,2]>=low)&(rr[:,2]<low+25)
        out['depth_bins'].append(dict(z_range_um=[low,low+25], **stats(residual[selected],minimum)))
    center = .5*(np.min(reference[:,:2],axis=0)+np.max(reference[:,:2],axis=0))
    for xside in (0,1):
        for yside in (0,1):
            sel = ((rr[:,0]>=center[0])==bool(xside))&((rr[:,1]>=center[1])==bool(yside))
            out['spatial_blocks'].append(dict(quadrant_xy=[xside,yside],center_xy_um=center.tolist(),
                                               **stats(residual[sel],minimum)))
    # Full-field seam boxes are frozen from the round0 reference geometry.
    if not SEAM_REGIONS:
        raise ValueError('Full-field seam definitions must be configured')
    for region in SEAM_REGIONS:
        bounds=np.asarray(region['bounds_xy_um'])
        sel=np.all((rr[:,:2]>=bounds[0])&(rr[:,:2]<=bounds[1]),axis=1)
        out['seams'][region['name']]=dict(**region, **stats(residual[sel],minimum))
    _,_,cells = spatial_split(rr)
    out['spatial_depth_cells'] = []
    for cell in np.unique(cells,axis=0):
        selected = np.all(cells==cell,axis=1)
        out['spatial_depth_cells'].append(dict(block_xyz=cell.tolist(),
            bounds_min_max_xyz_um=[(cell*np.array(CFG['block_size_xyz_um'])).tolist(),
                                  ((cell+1)*np.array(CFG['block_size_xyz_um'])).tolist()],
            **stats(residual[selected],minimum)))
    return out, residual


def geometry_check(moving, model):
    low, high = np.percentile(moving,[0,100],axis=0)
    grid = np.stack(np.meshgrid(*[np.linspace(a,b,n) for a,b,n in zip(low,high,[13,13,9])],indexing='ij'),axis=-1)
    jj = jacobian(grid,model)
    det, sv = np.linalg.det(jj), np.linalg.svd(jj,compute_uv=False)
    contraction = np.linalg.svd(jj-np.eye(3),compute_uv=False)[...,0]
    try:
        error = float(np.max(np.linalg.norm(inverse_cross(apply_cross(grid,model),model)-grid,axis=-1)))
    except RuntimeError:
        error = None
    passed = (det.min()>.65 and det.max()<1.5 and sv.min()>.7 and sv.max()<1.35
              and contraction.max()<.4 and jj[...,2,2].min()>.7 and error is not None and error<1e-5)
    return dict(passed=bool(passed), determinant_range=[float(det.min()),float(det.max())],
        singular_value_range=[float(sv.min()),float(sv.max())],
        axial_derivative_range=[float(jj[...,2,2].min()),float(jj[...,2,2].max())],
        maximum_displacement_jacobian_norm=float(contraction.max()), inverse_error_max_um=error,
        bounds_min_max_xyz_um=[low.tolist(),high.tolist()],
        interpretation='Invertibility/deformation plausibility only; not image registration accuracy')


def fit_cross(reference_cloud, moving_cloud, output_dir, *, reference_round=0, moving_round=10):
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=False)
    started = time.time()
    reference_cloud, moving_cloud = deduplicate(reference_cloud), deduplicate(moving_cloud)
    ref, mov = reference_cloud['xyz_um'], moving_cloud['xyz_um']
    mt, mh, _ = spatial_split(mov,moving_cloud['within_round_heldout'])
    # Broad initialization never uses the held-out moving features. Both
    # reference and moving train/holdout identities are frozen before the seed;
    # inherited prior holdouts remain excluded even when ownership changes.
    rt, rh, _ = spatial_split(ref,reference_cloud['within_round_heldout'])
    shift, init = initial_translation(ref[rt],mov[mt],CFG['xy_search_um'],CFG['z_search_um'])
    rtrain,mtrain = np.flatnonzero(rt),np.flatnonzero(mt)
    rheld,mheld = np.flatnonzero(rh),np.flatnonzero(mh)
    current = empty_model(mov,'affine',shift)
    history=[]
    for step,gate in enumerate(([3.,3.,10.],[2.,2.,7.],[1.5,1.5,5.],[1.2,1.2,3.])):
        local = reciprocal(ref[rtrain],apply_cross(mov[mtrain],current),gate)
        pairs = np.column_stack([rtrain[local[:,0]],mtrain[local[:,1]]])
        if len(pairs)<CFG['minimum_training_pairs']:
            _write_json(output/'failure.json',dict(status='INSUFFICIENT_FRESH_TIES',step=step,pairs=len(pairs),initializer=init))
            raise RuntimeError('Too few fresh reciprocal training ties after broad initialization')
        current=fit_ties(ref[pairs[:,0]],mov[pairs[:,1]],current,'affine')
        history.append(dict(step=step,gate_xyz_um=list(gate),pairs=len(pairs),
                            training=stats(apply_cross(mov[pairs[:,1]],current)-ref[pairs[:,0]])))
    affine=current
    # Freeze correspondence sets once under the training-only affine. The
    # quadratic sees precisely the same training pairs and cannot choose a
    # different convenient set of held-out matches for its evaluation.
    local=reciprocal(ref[rheld],apply_cross(mov[mheld],affine),CFG['heldout_pair_gate_xyz_um'])
    held_pairs=np.column_stack([rheld[local[:,0]],mheld[local[:,1]]])
    assert not np.intersect1d(pairs[:,0],held_pairs[:,0]).size
    assert not np.intersect1d(pairs[:,1],held_pairs[:,1]).size
    quadratic=fit_ties(ref[pairs[:,0]],mov[pairs[:,1]],affine,'quadratic')
    archive=dict(reference_xyz_um=ref,moving_xyz_um=mov,training_pairs=pairs,heldout_pairs=held_pairs,
        reference_source_fov=reference_cloud['source_fov'],moving_source_fov=moving_cloud['source_fov'],
        reference_source_index=reference_cloud['source_index'],moving_source_index=moving_cloud['source_index'],
        reference_pooled_input_index=reference_cloud['pooled_input_index'],moving_pooled_input_index=moving_cloud['pooled_input_index'],
        reference_train_indices=rtrain,moving_train_indices=mtrain,
        reference_heldout_indices=rheld,moving_heldout_indices=mheld,
        reference_inherited_within_round_heldout=reference_cloud['within_round_heldout'],
        moving_inherited_within_round_heldout=moving_cloud['within_round_heldout'])
    report=dict(reference_round=reference_round,moving_round=moving_round,parameters=CFG,
        initializer=init,affine_refinement_history=history,
        points_after_dedup=dict(reference=len(ref),moving=len(mov)),
        split_counts=dict(reference_training=len(rtrain),moving_training=len(mtrain),
                          reference_heldout=len(rheld),moving_heldout=len(mheld)),
        split_rule='Spatial/depth block hash modulo5;3umXY/1umZ boundary guards; inherited within-round heldouts excluded from training; exact feature IDs disjoint',
        validation_scope='Held-out features test this cross-round fit. Model comparison is exploratory; independent frozen raw-image-grid validation must decide acceptance.',
        heldout_pair_selection='Reciprocal matches with generous2umXY/5umZ gates under training-only affine, then frozen for both models; unmatched features are reported',
        heldout_matched_fraction_of_smaller_cloud=len(held_pairs)/max(1,min(len(rheld),len(mheld))),
        candidates={},status='RAW_IMAGE_VALIDATION_REQUIRED')
    for name,model in [('affine',affine),('quadratic',quadratic)]:
        assessment,residual=assess(ref,mov,held_pairs,model,reference_cloud['source_fov'])
        assessment['geometry']=geometry_check(mov,model)
        assessment['training']=stats(apply_cross(mov[pairs[:,1]],model)-ref[pairs[:,0]])
        model.update(reference_round=reference_round,moving_round=moving_round,assessment=assessment,
                     scientific_status='Provisional fresh cross-round candidate; raw-image validation required')
        report['candidates'][name]=assessment
        archive[name+'_heldout_residual_xyz_um']=residual
        _write_json(output/(name+'.json'),model)
    # Conservative candidate for independent image review. This is not an
    # acceptance decision or a claim of unbiased model-selection validation.
    preferred='affine'
    aa,qq=report['candidates']['affine'],report['candidates']['quadratic']
    if qq['geometry']['passed'] and len(held_pairs)>=50:
        ao,qo=aa['overall'],qq['overall']
        no_worse=(qo['xy_p95_pixels']<=1.05*ao['xy_p95_pixels'] and qo['z_p95_um']<=1.05*ao['z_p95_um'])
        improved=(qo['xy_median_pixels']<.9*ao['xy_median_pixels'] or qo['z_median_um']<.9*ao['z_median_um'])
        if no_worse and improved:
            preferred='quadratic'
    if not report['candidates'][preferred]['geometry']['passed']:
        preferred=None
    report['preferred_candidate_for_raw_image_review']=preferred
    report['elapsed_seconds']=time.time()-started
    np.savez_compressed(output/'matches.npz',**archive)
    _write_json(output/'comparison.json',report)
    if preferred:
        chosen=affine if preferred=='affine' else quadratic
        _write_json(output/'preferred_model.json',chosen)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-cloud',type=Path,required=True)
    parser.add_argument('--moving-cloud',type=Path,required=True)
    parser.add_argument('--reference-within-model',type=Path,required=True)
    parser.add_argument('--moving-within-model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-round',type=int,default=0)
    parser.add_argument('--moving-round',type=int,default=10)
    args=parser.parse_args()
    paths=[args.reference_cloud,args.moving_cloud,args.reference_within_model,args.moving_within_model]
    hashes={str(p.resolve()):_sha(p) for p in paths}
    report=fit_cross(load_cloud(args.reference_cloud),load_cloud(args.moving_cloud),args.output,
                     reference_round=args.reference_round,moving_round=args.moving_round)
    _write_json(args.output/'inputs.json',dict(source_sha256=hashes,script_sha256=_sha(__file__),
                  saved_production_cross_round_transforms_read=False))
    if any(_sha(p)!=digest for p,digest in hashes.items()):
        raise RuntimeError('Input changed during fresh cross-round fit')
    print(json.dumps(dict(status=report['status'],preferred=report['preferred_candidate_for_raw_image_review'],
                          initializer=report['initializer']['shift_xyz_um'],
                          heldout_pairs=report['candidates']['affine']['overall']['pairs'],
                          output=str(args.output))),flush=True)


if __name__=='__main__':
    main()
