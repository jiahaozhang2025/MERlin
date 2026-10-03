"""Alignment-only 0718 global quadratic candidate, with a new spatial holdout.

Public mapping functions accept any (...,3) XYZ-micron array:
    apply_global(native_points, fov, model_json_dict)
    inverse_global(global_points, fov, model_json_dict)
Native means the camera orientation already applied by bead_cloud.py. A source
image must be sampled in that orientation, using the returned per-pixel XYZ.
This script never launches decoding or modifies an existing analysis.
"""
import argparse
import datetime
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

import fit_graph as fg


FOVS = fg.FOVS
SIDE_EDGES = [(13, 14), (13, 26), (14, 25), (25, 26)]
FEATURE_NAMES = ['1', 'qx', 'qy', 'qz', 'qx2', 'qy2', 'qx_qy']


def basis(points, center, scale):
    q = (points-center)/scale
    return np.stack([np.ones(q.shape[:-1]), q[..., 0], q[..., 1], q[..., 2],
                     q[..., 0]**2, q[..., 1]**2, q[..., 0]*q[..., 1]], axis=-1)


def apply_global(points, fov, model):
    """Native camera XYZ microns -> anchor-13 global XYZ microns."""
    points = np.asarray(points, float)
    entry = model['models'][str(fov)]
    if points.shape[-1] != 3:
        raise ValueError('Expected (...,3) XYZ-micron coordinates')
    features = basis(points, np.asarray(model['normalization']['center_xyz_um']),
                     np.asarray(model['normalization']['scale_xyz_um']))
    return points + np.asarray(entry['stage_origin_xyz_um']) + features@np.asarray(entry['coefficients_7x3_um'])


def inverse_global(points, fov, model, tolerance_um=1e-7, max_iterations=40):
    """Global -> native XYZ by fixed point; fails if a map cannot be inverted.

    The gate on the displacement Jacobian ensures contraction for an accepted
    model. No camera-edge or Z-range clipping is performed: the renderer must
    mark a sample invalid when its native coordinates lie outside real data.
    """
    points = np.asarray(points, float)
    entry = model['models'][str(fov)]
    target = points - np.asarray(entry['stage_origin_xyz_um'])
    native = target.copy()
    center = np.asarray(model['normalization']['center_xyz_um'])
    scale = np.asarray(model['normalization']['scale_xyz_um'])
    coefficients = np.asarray(entry['coefficients_7x3_um'])
    if not points.size:
        return native
    for _ in range(max_iterations):
        updated = target-basis(native, center, scale)@coefficients
        error = np.max(np.abs(updated-native))
        native = updated
        if error < tolerance_um:
            return native
    raise RuntimeError('Inverse map did not converge for FOV%s; max update %.6g um' % (fov, error))


def jacobian(points, fov, model):
    points = np.asarray(points, float)
    scale = np.asarray(model['normalization']['scale_xyz_um'])
    center = np.asarray(model['normalization']['center_xyz_um'])
    q = (points-center)/scale
    beta = np.asarray(model['models'][str(fov)]['coefficients_7x3_um'])
    result = np.broadcast_to(np.eye(3), (*points.shape[:-1], 3, 3)).copy()
    result[..., :, 0] += (beta[1]+2*q[..., 0, None]*beta[4]+q[..., 1, None]*beta[6])/scale[0]
    result[..., :, 1] += (beta[2]+2*q[..., 1, None]*beta[5]+q[..., 0, None]*beta[6])/scale[1]
    result[..., :, 2] += beta[3]/scale[2]
    return result


def new_spatial_split(a, b, seed, cfg):
    """Frozen before the quadratic fit: outer 20% slabs held out, center trains."""
    width = cfg['image_width_pixels']*cfg['microns_per_pixel']
    lo = np.maximum([0., 0., a[:, 2].min()], np.array([0., 0., b[:, 2].min()])+seed)
    hi = np.minimum([width, width, a[:, 2].max()], np.array([width, width, b[:, 2].max()])+seed)
    # Four side neighbors only: choose the longer XY overlap dimension.
    axis = int(np.argmax((hi-lo)[:2]))
    left, right = lo[axis]+.20*(hi[axis]-lo[axis]), lo[axis]+.80*(hi[axis]-lo[axis])
    guard = cfg['search_xy_um']+cfg['validation_match_xy_um']
    expansion = np.array([cfg['search_xy_um']]*2+[cfg['search_z_um']])

    def select(p):
        inside = np.all((p >= lo-expansion) & (p <= hi+expansion), axis=1)
        train = inside & (p[:, axis] > left+guard) & (p[:, axis] < right-guard)
        holdout = inside & ((p[:, axis] <= left) | (p[:, axis] >= right))
        return np.flatnonzero(train), np.flatnonzero(holdout)
    ta, ha = select(a); tb, hb = select(b+seed)
    assert not np.intersect1d(ta, ha).size and not np.intersect1d(tb, hb).size
    return ta, tb, ha, hb, dict(axis='xy'[axis], boundaries_um=[float(left), float(right)],
        guard_um=guard, training_counts=[len(ta), len(tb)], heldout_counts=[len(ha), len(hb)],
        nominal_overlap_low_um=lo.tolist(), nominal_overlap_high_um=hi.tolist(),
        rule='Held out: outer20% at either end of longXY overlap axis. Train:center60% minus guardbands.')


def prepare_edges(clouds, cfg):
    edges, archive = [], {}
    for k, n in SIDE_EDGES:
        a, b = clouds[k], clouds[n]
        seed = fg.seed_origin(n)-fg.seed_origin(k)
        ta, tb, ha, hb, split = new_spatial_split(a, b, seed, cfg)
        key = '%s_%s_' % (k, n)
        for name, array in [('training_a_indices', ta), ('training_b_indices', tb),
                            ('heldout_a_indices', ha), ('heldout_b_indices', hb)]:
            archive[key+name] = array
        translation, pairs, residual, diagnostics = fg.fit_translation(a[ta], b[tb], seed, cfg)
        held_pairs, held_residual = fg.reciprocal_matches(a[ha], b[hb], translation,
            cfg['validation_match_xy_um'], cfg['validation_match_z_um'])
        global_pairs = np.column_stack([ta[pairs[:, 0]], tb[pairs[:, 1]]])
        global_held_pairs = np.column_stack([ha[held_pairs[:, 0]], hb[held_pairs[:, 1]]])
        archive[key+'training_pairs'] = global_pairs
        archive[key+'heldout_pairs'] = global_held_pairs
        archive[key+'translation_training_residual_xyz_um'] = residual
        archive[key+'translation_heldout_residual_xyz_um'] = held_residual
        assert not np.intersect1d(global_pairs[:, 0], global_held_pairs[:, 0]).size
        assert not np.intersect1d(global_pairs[:, 1], global_held_pairs[:, 1]).size
        edges.append(dict(k=k, n=n, spatial_split=split, training_pairs=global_pairs,
            heldout_pairs=global_held_pairs, initial_translation_xyz_um=translation.tolist(),
            training_translation_fit=diagnostics,
            translation_training=fg.metrics(residual, min(len(ta), len(tb)), cfg),
            translation_heldout=fg.metrics(held_residual, min(len(ha), len(hb)), cfg)))
    return edges, archive


def fit_model(clouds, edges, cfg):
    """Joint robust solve using training ties only, anchor13 identity."""
    unknown = [f for f in FOVS if f != 13]
    width = cfg['image_width_pixels']*cfg['microns_per_pixel']
    center, scale = np.array([width/2, width/2, 75.]), np.array([100., 100., 50.])
    residual_scale = np.array([cfg['microns_per_pixel']]*2+[1.])
    ncols = 3*7*3
    nrows = sum(len(e['training_pairs']) for e in edges)*3
    design = np.zeros((nrows, ncols))
    base = np.empty(nrows)
    offset = 0
    for e in edges:
        k, n, pair = e['k'], e['n'], e['training_pairs']
        aa, bb = clouds[k][pair[:, 0]], clouds[n][pair[:, 1]]
        base[offset:offset+3*len(pair)] = ((aa+fg.seed_origin(k)-bb-fg.seed_origin(n))/residual_scale).ravel()
        for fov, pp, sign in [(k, aa, 1.), (n, bb, -1.)]:
            if fov == 13:
                continue
            fi = unknown.index(fov)
            features = basis(pp, center, scale)
            for d in range(3):
                ri = offset+3*np.arange(len(pp))+d
                ci = fi*21+3*np.arange(7)+d
                design[np.ix_(ri, ci)] = sign*features/residual_scale[d]
        offset += 3*len(pair)
    # Fixed mild regularization: 1% linear deformation, 1Âµm normalized quadratic
    # coefficient. This does not use heldout errors to choose a strength.
    prior = np.zeros((54, ncols)); row = 0
    for fi in range(3):
        for feature in range(1, 7):
            for d in range(3):
                prior[row, fi*21+feature*3+d] = 1./(scale[feature-1]*.01) if feature <= 3 else 1.
                row += 1
    design_all, base_all = np.concatenate([design, prior]), np.concatenate([base, np.zeros(len(prior))])
    fit = least_squares(lambda p: design_all@p+base_all, np.zeros(ncols),
        jac=lambda p: design_all.copy(), loss='soft_l1', f_scale=1., max_nfev=1000,
        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    model = dict(model_type='native_xyz_to_anchor13_global_quadratic_displacement',
        anchor_fov=13, coordinate_units='micrometers', coordinate_order='xyz',
        equation='global = native + stage_origin + [1,qx,qy,qz,qx^2,qy^2,qx*qy] @ beta; q=(native-center)/scale',
        normalization=dict(center_xyz_um=center.tolist(), scale_xyz_um=scale.tolist()),
        features=FEATURE_NAMES, models={}, optimization=dict(loss='soft_l1', f_scale=1.,
            residual_scale_xyz_um=residual_scale.tolist(), linear_prior_fraction=.01,
            quadratic_prior_um=1., success=bool(fit.success), message=str(fit.message),
            cost=float(fit.cost), evaluations=int(fit.nfev),
            objective_uses='Only new-central-slab training ties, plus fixed zero-deformation prior'))
    for fov in FOVS:
        beta = np.zeros((7, 3)) if fov == 13 else fit.x[unknown.index(fov)*21:unknown.index(fov)*21+21].reshape(7, 3)
        model['models'][str(fov)] = dict(stage_origin_xyz_um=fg.seed_origin(fov).tolist(),
                                        coefficients_7x3_um=beta.tolist())
    return model


def assess(clouds, edges, model, archive, cfg):
    reasons, evidence = [], []
    for e in edges:
        k, n = e['k'], e['n']; key = '%s_%s_' % (k, n)
        record = {name: val for name, val in e.items() if name not in ['training_pairs', 'heldout_pairs']}
        for split in ['training', 'heldout']:
            pair = e[split+'_pairs']
            residual = apply_global(clouds[k][pair[:, 0]], k, model)-apply_global(clouds[n][pair[:, 1]], n, model)
            metrics = fg.metrics(residual, min(e['spatial_split'][split+'_counts']), cfg)
            record['quadratic_'+split] = metrics
            archive[key+'quadratic_'+split+'_residual_xyz_um'] = residual
            archive[key+split+'_native_a_xyz_um'] = clouds[k][pair[:, 0]]
            archive[key+split+'_native_b_xyz_um'] = clouds[n][pair[:, 1]]
            minimum = cfg['min_fit_pairs'] if split == 'training' else cfg['min_holdout_pairs']
            reasons += ['%s-%s %s' % (k, n, reason) for reason in fg.gate_metrics(metrics, split, minimum, cfg, split == 'heldout')]
        evidence.append(record)
    width = cfg['image_width_pixels']*cfg['microns_per_pixel']
    xx, yy, zz = np.meshgrid(np.linspace(0, width, 17), np.linspace(0, width, 17),
                             np.linspace(0, 150., 5), indexing='ij')
    grid = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    jacobians = {}
    for fov in FOVS:
        jj = jacobian(grid, fov, model)
        det, sv = np.linalg.det(jj), np.linalg.svd(jj, compute_uv=False)
        contraction = np.linalg.svd(jj-np.eye(3), compute_uv=False)[:, 0]
        back = inverse_global(apply_global(grid, fov, model), fov, model)
        inverse_error = float(np.max(np.linalg.norm(grid-back, axis=1)))
        passed = det.min() > .95 and det.max() < 1.05 and sv.min() > .97 and sv.max() < 1.03
        passed &= contraction.max() < .05 and inverse_error < 1e-5
        jacobians[str(fov)] = dict(determinant_min=float(det.min()), determinant_max=float(det.max()),
            singular_value_min=float(sv.min()), singular_value_max=float(sv.max()),
            displacement_jacobian_norm_max=float(contraction.max()), inverse_error_max_um=inverse_error,
            passed=bool(passed), sample_grid='17x17XY across complete camera;5Z across0..150Âµm')
        if not passed:
            reasons.append('FOV%s camera-wide Jacobian or inverse gate failed' % fov)
    if not model['optimization']['success']:
        reasons.append('Optimizer did not converge')
    return dict(status='EXPLORATORY_ALIGNMENT_PASS' if not reasons else 'FAIL', reasons=reasons,
        edges=evidence, camera_grid_jacobians=jacobians,
        frozen_jacobian_gates=dict(determinant=[.95, 1.05], singular_values=[.97, 1.03],
            displacement_jacobian_norm_max=.05, inverse_error_max_um=1e-5),
        interpretation='Passing supports review of this alignment candidate only; no decoding is authorized by this script.')


def write_plot(model, archive, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, edge in zip(axes.ravel(), model['assessment']['edges']):
        k, n = edge['k'], edge['n']; key = '%s_%s_' % (k, n)
        old = archive[key+'translation_heldout_residual_xyz_um']
        new = archive[key+'quadratic_heldout_residual_xyz_um']
        for res, name, color in [(old, 'Translation', '#a3a3a3'), (new, 'Joint quadratic', '#166534')]:
            xy = np.sort(np.linalg.norm(res[:, :2], axis=1)/fg.CFG['microns_per_pixel'])
            ax.plot(xy, np.arange(1, len(xy)+1)/max(1, len(xy)), label=name, color=color)
        ax.axvline(1., color='orange', ls=':', lw=1)
        ax.axvline(2., color='red', ls=':', lw=1)
        ax.axhline(.5, color='gray', ls=':', lw=.7)
        ax.axhline(.95, color='gray', ls=':', lw=.7)
        ax.set_title('FOV%sâ€“%s: %s held-out bead pairs' % (k, n, len(new)))
        ax.set_xlabel('Held-out radial XY disagreement (pixels)')
        ax.set_ylabel('Fraction of held-out matches')
        ax.set_xlim(0, max(3., np.percentile(np.linalg.norm(new[:, :2], axis=1)/fg.CFG['microns_per_pixel'], 99)))
        ax.legend(loc='lower right')
    fig.suptitle('New outer-slab validation: '+model['assessment']['status']+'\nAlignment review only; no decoding')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(cloud_dir, output_dir):
    cloud_dir, output_dir = Path(cloud_dir), Path(output_dir)
    if (output_dir/'global_model.json').exists():
        raise FileExistsError('Do not overwrite alignment evidence')
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time(); clouds, sources = {}, []
    bright_index = {}
    for fov in FOVS:
        path = cloud_dir/('fov%s_r0.npz' % fov)
        with np.load(path, allow_pickle=False) as f:
            xyz, amp = f['xyz_um'], f['amplitude']
        keep = np.isfinite(xyz).all(axis=1)&np.isfinite(amp)&(amp >= fg.CFG['amplitude_minimum'])
        clouds[fov] = xyz[keep]
        bright_index['fov%s_bright_source_indices' % fov] = np.flatnonzero(keep)
        sources.append(dict(fov=fov, path=str(path.resolve()), sha256=fg.sha256(path),
                            bright_beads=int(keep.sum()), metadata_sha256=fg.sha256(path.with_suffix('.json'))))
    edges, archive = prepare_edges(clouds, fg.CFG)
    archive.update(bright_index)
    model = fit_model(clouds, edges, fg.CFG)
    model['assessment'] = assess(clouds, edges, model, archive, fg.CFG)
    model.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_code_sha256=fg.sha256(__file__), registration_code_sha256=fg.sha256(fg.__file__),
        sources=sources, parameters=fg.CFG, validation_status='Exploratory same-acquisition validation; not independent acquisition',
        validation_design='Outer20% slabs on each side overlap; disjoint from old central30% validation. Recomputed correspondence seeds and joint fit use new central training only. This model class was selected using a different earlier validation region.',
        scope='Alignment only. User will inspect images before any downstream work.',
        elapsed_seconds=time.time()-start)
    np.savez_compressed(output_dir/'global_matches.npz', **archive)
    model['matches_sha256'] = fg.sha256(output_dir/'global_matches.npz')
    (output_dir/'global_model.json').write_text(json.dumps(model, indent=2, allow_nan=False))
    write_plot(model, archive, output_dir/'heldout_comparison.png')
    print(json.dumps(dict(status=model['assessment']['status'], reasons=model['assessment']['reasons'],
        edges=[dict(k=e['k'], n=e['n'], heldout=e['quadratic_heldout']) for e in model['assessment']['edges']],
        elapsed_seconds=time.time()-start, output=str(output_dir))), flush=True)
    return 0 if model['assessment']['status'] == 'EXPLORATORY_ALIGNMENT_PASS' else 2


def self_test():
    rng = np.random.default_rng(719)
    center, scale = [153., 153., 75.], [100., 100., 50.]
    model = dict(normalization=dict(center_xyz_um=center, scale_xyz_um=scale), models={})
    for fov in FOVS:
        beta = rng.normal(0, .035, (7, 3)); beta[0] = rng.normal(0, 1., 3)
        if fov == 13: beta[:] = 0.
        model['models'][str(fov)] = dict(stage_origin_xyz_um=fg.seed_origin(fov).tolist(), coefficients_7x3_um=beta.tolist())
    points = rng.uniform([0, 0, 0], [306, 306, 150], (1000, 3))
    errors = []
    for fov in FOVS:
        moved = apply_global(points, fov, model)
        returned = inverse_global(moved, fov, model)
        errors.append(float(np.max(np.abs(points-returned))))
        assert errors[-1] < 1e-6
        numerical = np.stack([(apply_global(points+np.eye(3)[d]*1e-4, fov, model)-
            apply_global(points-np.eye(3)[d]*1e-4, fov, model))/(2e-4) for d in range(3)], axis=-1)
        assert np.max(np.abs(jacobian(points, fov, model)-numerical)) < 1e-7
    print(json.dumps(dict(status='PASS', tests=['native-global inverse', 'analytic Jacobian', 'anchor identity'],
                          max_inverse_errors_um=errors)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if not args.cloud_dir or not args.output_dir:
        parser.error('--cloud-dir and --output-dir are required')
    return run(args.cloud_dir, args.output_dir)


if __name__ == '__main__':
    raise SystemExit(main())
