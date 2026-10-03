"""Isolated 0718 round-0 native-bead translation graph with spatial holdout.

Input: fov{13,14,25,26}_r0.npz, keys xyz_um (native XYZ) and amplitude.
No existing image warp or decoded barcode is used in fitting. All coordinates
are microns. Convention: native XYZ + origin XYZ = global XYZ; thus a global
depth Z samples the native depth Z-origin_Z. Thresholds below are fixed before
looking at this pilot's results. Failure is reported, never silently relaxed.
"""
import argparse
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


FOVS = (13, 14, 25, 26)
STAGES = {13: (2335., -2288.), 14: (2065., -2288.),
          25: (2065., -2018.), 26: (2335., -2018.)}
SIDES = {(13, 14), (14, 25), (25, 26), (13, 26)}
CFG = dict(microns_per_pixel=.1493, image_width_pixels=2048,
           stage_x_sign=-1., stage_y_sign=-1.,
           amplitude_minimum=500., search_xy_um=8., search_z_um=4.,
           vote_xy_bin_um=.30, vote_z_bin_um=.5,
           fit_match_xy_um=.60, fit_match_z_um=1.5,
           validation_match_xy_um=.90, validation_match_z_um=2.,
           min_fit_pairs=30, min_holdout_pairs=20,
           min_holdout_match_fraction=.25,
           xy_median_max_pixels=1., xy_p95_max_pixels=2.,
           z_median_max_um=.5, z_p95_max_um=1.,
           reverse_xy_max_pixels=1., reverse_z_max_um=.5,
           cycle_xy_max_pixels=2., cycle_z_max_um=1.)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_origin(fov):
    x, y = STAGES[fov]
    xa, ya = STAGES[13]
    # Established from the freshly extracted native bead clouds, not inferred
    # from historical image-layout transforms. Initial +stage-X guess produced
    # no coherent horizontal matches; -stage-X gives hundreds with reverse
    # closure below numerical precision. The v1 evidence remains preserved.
    return np.array([CFG['stage_x_sign']*(x-xa), CFG['stage_y_sign']*(y-ya), 0.])


def reciprocal_matches(a, b, translation, xy_gate, z_gate):
    """One-to-one reciprocal nearest neighbors; no fitting happens here."""
    if len(a) == 0 or len(b) == 0:
        return np.empty((0, 2), int), np.empty((0, 3), float)
    scale = np.array([xy_gate, xy_gate, z_gate])
    aa, bb = a/scale, (b+translation)/scale
    ta, tb = cKDTree(aa), cKDTree(bb)
    _, a_of_b = ta.query(bb, k=1)
    _, b_of_a = tb.query(aa, k=1)
    bi = np.arange(len(b))
    reciprocal = b_of_a[a_of_b] == bi
    residual = a[a_of_b] - (b + translation)
    accepted = reciprocal & (np.linalg.norm(residual[:, :2], axis=1) <= xy_gate)
    accepted &= np.abs(residual[:, 2]) <= z_gate
    pairs = np.column_stack([a_of_b[accepted], bi[accepted]])
    return pairs, residual[accepted]


def coarse_vote(a, b, seed, cfg):
    """Translation histogram from training points only; robust to unmatched beads."""
    limits = np.array([cfg['search_xy_um']]*2+[cfg['search_z_um']])
    binsize = np.array([cfg['vote_xy_bin_um']]*2+[cfg['vote_z_bin_um']])
    edges = [np.linspace(-r, r, int(np.ceil(2*r/s))+1)
             for r, s in zip(limits, binsize)]
    hist = np.zeros(tuple(len(e)-1 for e in edges), np.float64)
    tree = cKDTree(a/limits)
    candidate_count = 0
    # Chunking bounds peak memory for point clouds with many unrelated features.
    for start in range(0, len(b), 500):
        bb = b[start:start+500] + seed
        neighbors = tree.query_ball_point(bb/limits, np.sqrt(3.))
        differences = []
        for p, nn in zip(bb, neighbors):
            if not nn:
                continue
            d = a[np.asarray(nn)] - p
            d = d[np.all(np.abs(d) <= limits, axis=1)]
            if len(d):
                differences.append(d)
        if differences:
            differences = np.concatenate(differences)
            hist += np.histogramdd(differences, bins=edges)[0]
            candidate_count += len(differences)
    if candidate_count == 0:
        raise ValueError('No training candidate pairs inside frozen search range')
    smooth = ndimage.gaussian_filter(hist, sigma=.7)
    index = np.unravel_index(np.argmax(smooth), smooth.shape)
    correction = np.array([.5*(e[i]+e[i+1]) for e, i in zip(edges, index)])
    on_edge = any(i in (0, n-1) for i, n in zip(index, hist.shape))
    return seed+correction, dict(candidate_pairs=int(candidate_count),
        peak_smoothed_votes=float(smooth[index]), peak_raw_votes=float(hist[index]),
        correction_um=correction.tolist(), search_boundary_peak=bool(on_edge))


def fit_translation(a, b, seed, cfg):
    if min(len(a), len(b)) < cfg['min_fit_pairs']:
        raise ValueError('Insufficient training beads')
    translation, vote = coarse_vote(a, b, seed, cfg)
    if vote['search_boundary_peak']:
        raise ValueError('Training translation vote reaches frozen search boundary')
    scale = np.array([cfg['microns_per_pixel']]*2+[.5])
    # Broad initial basin, then the same fixed matching radius at every fine step.
    for iteration in range(15):
        xy = max(1.5, cfg['fit_match_xy_um']) if iteration < 2 else cfg['fit_match_xy_um']
        pairs, _ = reciprocal_matches(a, b, translation, xy, cfg['fit_match_z_um'])
        if len(pairs) < cfg['min_fit_pairs']:
            raise ValueError('Insufficient reciprocal training matches: %d' % len(pairs))
        differences = a[pairs[:, 0]] - b[pairs[:, 1]]
        fit = least_squares(lambda t: ((t-differences)/scale).ravel(),
                            np.median(differences, axis=0), loss='soft_l1', f_scale=1.,
                            max_nfev=100, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        change = np.linalg.norm(fit.x-translation)
        translation = fit.x
        if iteration >= 2 and change < 1e-7:
            break
    pairs, residual = reciprocal_matches(a, b, translation,
        cfg['fit_match_xy_um'], cfg['fit_match_z_um'])
    if np.any(np.abs(translation[:2]-seed[:2]) >= cfg['search_xy_um']) or \
            abs(translation[2]-seed[2]) >= cfg['search_z_um']:
        raise ValueError('Refined translation exceeds frozen search range')
    return translation, pairs, residual, dict(vote=vote, iterations=iteration+1)


def spatial_split(a, b, seed, cfg):
    """A central spatial slab is held out; guard bands prevent bead identity leakage.

    The slab axis is the longest side of the nominal overlap box. For side
    neighbors it is normally XY; for a small corner overlap it may be Z.
    Neither held-out points nor their amplitudes enter coarse/fine registration.
    """
    width = cfg['image_width_pixels'] * cfg['microns_per_pixel']
    lo_a = np.array([0., 0., np.min(a[:, 2])])
    hi_a = np.array([width, width, np.max(a[:, 2])])
    lo_b = np.array([0., 0., np.min(b[:, 2])])+seed
    hi_b = np.array([width, width, np.max(b[:, 2])])+seed
    lo, hi = np.maximum(lo_a, lo_b), np.minimum(hi_a, hi_b)
    if np.any(hi <= lo):
        raise ValueError('No nominal 3D overlap')
    axis = int(np.argmax(hi-lo))
    extent = hi[axis]-lo[axis]
    held_lo, held_hi = lo[axis]+.35*extent, lo[axis]+.65*extent
    # Each camera assigns membership in its nominal global coordinates. A gap
    # wider than the entire seed search prevents one physical point crossing
    # between the training and held-out sets in the two cameras.
    guard = cfg['search_z_um']+cfg['validation_match_z_um'] if axis == 2 else \
            cfg['search_xy_um']+cfg['validation_match_xy_um']
    expansion = np.array([cfg['search_xy_um']]*2+[cfg['search_z_um']])

    def split(p):
        inside = np.all((p >= lo-expansion) & (p <= hi+expansion), axis=1)
        held = inside & (p[:, axis] >= held_lo) & (p[:, axis] <= held_hi)
        train = inside & ((p[:, axis] < held_lo-guard) | (p[:, axis] > held_hi+guard))
        return np.flatnonzero(train), np.flatnonzero(held)
    ta, ha = split(a)
    tb, hb = split(b+seed)
    assert not np.intersect1d(ta, ha).size and not np.intersect1d(tb, hb).size
    return ta, tb, ha, hb, dict(axis='xyz'[axis], nominal_overlap_low_um=lo.tolist(),
        nominal_overlap_high_um=hi.tolist(), heldout_slab_um=[held_lo, held_hi],
        guard_um=float(guard), training_counts=[len(ta), len(tb)],
        heldout_counts=[len(ha), len(hb)],
        rule='central 30% slab held out; training outside slab plus guard; no fit/holdout sharing')


def metrics(residual, denominator, cfg):
    if not len(residual):
        return dict(pairs=0, match_fraction=0., xy_median_pixels=None,
                    xy_p95_pixels=None, z_median_um=None, z_p95_um=None)
    xy = np.linalg.norm(residual[:, :2], axis=1)/cfg['microns_per_pixel']
    zz = np.abs(residual[:, 2])
    return dict(pairs=int(len(residual)), match_fraction=float(len(residual)/max(1, denominator)),
                xy_median_pixels=float(np.median(xy)), xy_p95_pixels=float(np.percentile(xy, 95)),
                z_median_um=float(np.median(zz)), z_p95_um=float(np.percentile(zz, 95)),
                signed_median_xyz_um=np.median(residual, axis=0).tolist())


def gate_metrics(m, prefix, minimum, cfg, include_fraction=False):
    reasons = []
    if m['pairs'] < minimum:
        reasons.append('%s pairs %d < %d' % (prefix, m['pairs'], minimum))
    for name, threshold in [('xy_median_pixels', cfg['xy_median_max_pixels']),
                            ('xy_p95_pixels', cfg['xy_p95_max_pixels']),
                            ('z_median_um', cfg['z_median_max_um']),
                            ('z_p95_um', cfg['z_p95_max_um'])]:
        if m[name] is None or m[name] >= threshold:
            reasons.append('%s %s %s >= %s' % (prefix, name, m[name], threshold))
    if include_fraction and m['match_fraction'] < cfg['min_holdout_match_fraction']:
        reasons.append('%s match fraction %.3f < %.3f' %
                       (prefix, m['match_fraction'], cfg['min_holdout_match_fraction']))
    return reasons


def register_edge(k, n, a, b, seed, cfg):
    edge = dict(k=int(k), n=int(n), required_side=(k, n) in SIDES,
                seed_translation_xyz_um=seed.tolist(), accepted=False, reasons=[])
    arrays = {}
    try:
        ta, tb, ha, hb, split = spatial_split(a, b, seed, cfg)
        edge['spatial_split'] = split
        for name, idx in [('train_a', ta), ('train_b', tb), ('holdout_a', ha), ('holdout_b', hb)]:
            arrays[name+'_indices'] = idx
        translation, train_pairs, train_residual, diagnostics = fit_translation(a[ta], b[tb], seed, cfg)
        reverse, reverse_pairs, reverse_residual, reverse_diag = fit_translation(b[tb], a[ta], -seed, cfg)
        hold_pairs, hold_residual = reciprocal_matches(a[ha], b[hb], translation,
            cfg['validation_match_xy_um'], cfg['validation_match_z_um'])
        train = metrics(train_residual, min(len(ta), len(tb)), cfg)
        held = metrics(hold_residual, min(len(ha), len(hb)), cfg)
        rev_error = translation+reverse
        reverse_xy = float(np.linalg.norm(rev_error[:2])/cfg['microns_per_pixel'])
        reverse_z = float(abs(rev_error[2]))
        reasons = gate_metrics(train, 'training', cfg['min_fit_pairs'], cfg)
        reasons += gate_metrics(held, 'heldout', cfg['min_holdout_pairs'], cfg, True)
        if reverse_xy >= cfg['reverse_xy_max_pixels'] or reverse_z >= cfg['reverse_z_max_um']:
            reasons.append('independent reverse closure %.4f pixels XY, %.4f um Z' % (reverse_xy, reverse_z))
        edge.update(translation_xyz_um=translation.tolist(), training=train, heldout=held,
            forward_fit=diagnostics, reverse_fit=reverse_diag,
            reverse_translation_xyz_um=reverse.tolist(),
            reverse_closure_xyz_um=rev_error.tolist(), reverse_xy_pixels=reverse_xy,
            reverse_z_um=reverse_z, accepted=not reasons, reasons=reasons)
        arrays.update(training_pairs=np.column_stack([ta[train_pairs[:, 0]], tb[train_pairs[:, 1]]]),
            training_residual_xyz_um=train_residual,
            heldout_pairs=np.column_stack([ha[hold_pairs[:, 0]], hb[hold_pairs[:, 1]]]),
            heldout_residual_xyz_um=hold_residual,
            reverse_training_pairs=np.column_stack([tb[reverse_pairs[:, 0]], ta[reverse_pairs[:, 1]]]),
            reverse_training_residual_xyz_um=reverse_residual)
        # Retain all test residuals and locations, including observations that
        # fail quality gates; no outlier trimming is applied to test metrics.
        arrays['heldout_reference_xyz_um'] = a[ha[hold_pairs[:, 0]]]
        arrays['heldout_neighbor_xyz_um'] = b[hb[hold_pairs[:, 1]]]
        assert not np.intersect1d(arrays['training_pairs'][:, 0], arrays['heldout_pairs'][:, 0]).size
        assert not np.intersect1d(arrays['training_pairs'][:, 1], arrays['heldout_pairs'][:, 1]).size
    except (ValueError, RuntimeError) as error:
        edge['reasons'].append(str(error))
    return edge, arrays


def solve_graph(edges, cfg):
    accepted = [e for e in edges if e['accepted']]
    seen = {13}
    for _ in FOVS:
        for e in accepted:
            if e['k'] in seen or e['n'] in seen:
                seen.update([e['k'], e['n']])
    if set(FOVS) != seen:
        return dict(status='FAIL', reasons=['Accepted graph is disconnected'],
                    connected_fovs=sorted(seen), origins_xyz_um=None)
    unknown = [f for f in FOVS if f != 13]
    scale = np.array([cfg['microns_per_pixel']]*2+[.5])

    def unpack(x):
        return {13: np.zeros(3), **{f: x[3*i:3*i+3] for i, f in enumerate(unknown)}}

    def residuals(x):
        origin = unpack(x)
        return np.concatenate([(origin[e['n']]-origin[e['k']]-e['translation_xyz_um'])/scale
                               for e in accepted])
    fit = least_squares(residuals, np.concatenate([seed_origin(f) for f in unknown]),
                        loss='soft_l1', f_scale=1., max_nfev=1000,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    origin = unpack(fit.x)
    reasons = []
    graph_residuals = []
    for e in accepted:
        rr = origin[e['n']]-origin[e['k']]-e['translation_xyz_um']
        xy, zz = float(np.linalg.norm(rr[:2])/cfg['microns_per_pixel']), float(abs(rr[2]))
        graph_residuals.append(dict(k=e['k'], n=e['n'], residual_xyz_um=rr.tolist(),
                                    xy_pixels=xy, z_um=zz))
        if xy >= cfg['cycle_xy_max_pixels'] or zz >= cfg['cycle_z_max_um']:
            reasons.append('graph edge %s-%s is inconsistent' % (e['k'], e['n']))
    e_map = {(e['k'], e['n']): np.asarray(e['translation_xyz_um']) for e in accepted}

    def link(k, n):
        if (k, n) in e_map:
            return e_map[k, n]
        if (n, k) in e_map:
            return -e_map[n, k]
        return None
    # All triangles with observed diagonal links plus the four-side square.
    cycles = [list(t)+[t[0]] for t in itertools.combinations(FOVS, 3)]
    cycles += [[13, 14, 25, 26, 13]]
    closure = []
    for cycle in cycles:
        links = [link(k, n) for k, n in zip(cycle[:-1], cycle[1:])]
        if any(t is None for t in links):
            continue
        rr = np.sum(links, axis=0)
        xy, zz = float(np.linalg.norm(rr[:2])/cfg['microns_per_pixel']), float(abs(rr[2]))
        passed = xy < cfg['cycle_xy_max_pixels'] and zz < cfg['cycle_z_max_um']
        closure.append(dict(cycle=cycle, residual_xyz_um=rr.tolist(), xy_pixels=xy, z_um=zz, passed=passed))
        if not passed:
            reasons.append('loop closure failed for %s' % cycle)
    missing_sides = sorted(SIDES-set(e_map))
    if missing_sides:
        reasons.append('Required side edges rejected: %s' % missing_sides)
    if not fit.success:
        reasons.append('Graph optimizer failed: %s' % fit.message)
    return dict(status='PASS' if not reasons else 'FAIL', reasons=reasons,
        anchor_fov=13, origins_xyz_um={str(f): v.tolist() for f, v in origin.items()},
        graph_edge_residuals=graph_residuals, cycle_closure=closure,
        optimizer=dict(loss='soft_l1', f_scale=1., axis_scales_um=scale.tolist(),
                       edge_weighting='equal per accepted edge; heldout metrics never weight the fit',
                       success=bool(fit.success), cost=float(fit.cost), evaluations=int(fit.nfev)))


def plot_report(report, arrays, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, e in zip(axes.ravel(), report['edges']):
        key = '%d_%d_heldout_residual_xyz_um' % (e['k'], e['n'])
        residual = arrays.get(key, np.empty((0, 3)))
        if len(residual):
            xy = np.linalg.norm(residual[:, :2], axis=1)/CFG['microns_per_pixel']
            ax.scatter(xy, np.abs(residual[:, 2]), s=7, alpha=.45)
            ax.axvline(CFG['xy_p95_max_pixels'], color='red', ls='--', lw=1)
            ax.axhline(CFG['z_p95_max_um'], color='red', ls='--', lw=1)
        else:
            ax.text(.5, .5, '\n'.join(e['reasons']), ha='center', va='center', transform=ax.transAxes, wrap=True)
        ax.set_title('%sâ€“%s: %s (%s)' % (e['k'], e['n'], 'accepted' if e['accepted'] else 'rejected',
                                        'required' if e['required_side'] else 'optional'))
        ax.set_xlabel('Held-out radial XY residual (pixels)')
        ax.set_ylabel('Held-out |Z residual| (Âµm)')
    fig.suptitle('0718 round-0 native-bead graph: '+report['graph']['status'])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(cloud_dir, output_dir):
    start = time.time()
    cloud_dir, output_dir = Path(cloud_dir), Path(output_dir)
    if (output_dir/'graph.json').exists() or (output_dir/'matches.npz').exists():
        raise FileExistsError('Use a fresh output directory; graph evidence is not overwritten')
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, clouds, all_arrays = [], {}, {}
    for fov in FOVS:
        path = cloud_dir/('fov%d_r0.npz' % fov)
        with np.load(path, allow_pickle=False) as f:
            xyz, amplitude = np.asarray(f['xyz_um'], float), np.asarray(f['amplitude'], float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(amplitude):
            raise ValueError('Malformed cloud: %s' % path)
        keep = np.isfinite(xyz).all(axis=1) & np.isfinite(amplitude) & (amplitude >= CFG['amplitude_minimum'])
        clouds[fov] = xyz[keep]
        all_arrays['fov%d_bright_source_indices' % fov] = np.flatnonzero(keep)
        source = dict(fov=fov, cloud_path=str(path.resolve()), cloud_sha256=sha256(path),
                      total_beads=int(len(xyz)), bright_beads=int(keep.sum()),
                      stage_xy_um=list(STAGES[fov]), stage_seed_origin_xyz_um=seed_origin(fov).tolist())
        meta_path = path.with_suffix('.json')
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            source.update(metadata_path=str(meta_path.resolve()), metadata_sha256=sha256(meta_path),
                          extraction_provenance=metadata)
            microscope = metadata.get('microscope', {})
            if microscope and (abs(float(microscope['microns_per_pixel'])-CFG['microns_per_pixel']) > 1e-9
                  or microscope.get('transpose') is not True
                  or microscope.get('flip_horizontal') is not True
                  or microscope.get('flip_vertical', False) is not False):
                raise ValueError('Cloud microscope metadata differs from frozen pilot convention')
        sources.append(source)
    edges = []
    for k, n in itertools.combinations(FOVS, 2):
        edge, arrays = register_edge(k, n, clouds[k], clouds[n], seed_origin(n)-seed_origin(k), CFG)
        edges.append(edge)
        all_arrays.update({'%d_%d_%s' % (k, n, name): value for name, value in arrays.items()})
        print(json.dumps({'edge': [k, n], 'accepted': edge['accepted'],
                          'training': edge.get('training'), 'heldout': edge.get('heldout'),
                          'reasons': edge['reasons']}), flush=True)
    graph = solve_graph(edges, CFG)
    np.savez_compressed(output_dir/'matches.npz', **all_arrays)
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_code_sha256=sha256(__file__), parameters=CFG, sources=sources,
        coordinate_convention='native XYZ + FOV origin XYZ = global XYZ; global Z samples native Z-origin_Z',
        stage_orientation_evidence='Fresh native round-0 clouds establish stage-X and stage-Y signs both negative. Initial +stage-X seed in registration_v1 is obsolete; extraction itself is unaffected. Historical mosaic transforms are not native image origins.',
        matching_convention='edge k,n translation added to n native XYZ yields k native XYZ',
        split_audit='Saved *_indices index bright cloud rows; fov*_bright_source_indices map them to original NPZ rows. No held-out bead participates in coarse votes, fine registration, or graph residuals. Heldout results gate acceptance only.',
        limitations=['Translation-only round-0 model: rotation, scale, nonlinear distortion and depth trends require separate validation.',
                     'Reciprocal matching uses fixed broad gates; unmatched fraction is reported alongside accepted-pair residuals.',
                     'Bright pointlike fiducial candidates are not independently chemically identified beads.',
                     'This graph does not validate between-round registration or molecular decoding.'],
        edges=edges, graph=graph, matches_path=str((output_dir/'matches.npz').resolve()),
        matches_sha256=sha256(output_dir/'matches.npz'), elapsed_seconds=time.time()-start)
    (output_dir/'graph.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    try:
        plot_report(report, all_arrays, output_dir/'heldout_residuals.png')
    except ImportError as error:
        print('Optional plot unavailable: '+str(error), file=sys.stderr)
    print(json.dumps({'status': graph['status'], 'reasons': graph['reasons'],
                      'graph_json': str(output_dir/'graph.json'), 'elapsed_seconds': time.time()-start}), flush=True)
    return 0 if graph['status'] == 'PASS' else 2


def self_test():
    rng = np.random.default_rng(718)
    cfg = dict(CFG)
    # Known translation plus unmatched points tests both sign and robust matching.
    a0 = rng.uniform([4., 4., 4.], [300., 300., 96.], (4500, 3))
    truth = np.array([-2.4, 1.6, -.35])
    a = np.concatenate([a0, rng.uniform([4., 4., 4.], [300., 300., 96.], (1000, 3))])
    b = np.concatenate([a0-truth+rng.normal(0, [.025, .025, .08], a0.shape),
                        rng.uniform([4., 4., 4.], [300., 300., 96.], (1800, 3))])
    edge, arrays = register_edge(13, 14, a, b, np.array([-2., 1., 0.]), cfg)
    assert edge['accepted'], edge
    fitted = np.asarray(edge['translation_xyz_um'])
    assert np.linalg.norm(fitted[:2]-truth[:2]) < .01 and abs(fitted[2]-truth[2]) < .02
    assert not np.intersect1d(arrays['training_pairs'][:, 0], arrays['heldout_pairs'][:, 0]).size
    assert not np.intersect1d(arrays['training_pairs'][:, 1], arrays['heldout_pairs'][:, 1]).size
    # Perturb heldout beads while calling the fitter on the recorded training
    # rows: the training result must be exactly unchanged, including coarse vote.
    ta, tb = arrays['train_a_indices'], arrays['train_b_indices']
    bb = b.copy()
    bb[arrays['holdout_b_indices']] += [4., -3., 2.]
    repeat, _, _, _ = fit_translation(a[ta], bb[tb], np.array([-2., 1., 0.]), cfg)
    assert np.array_equal(repeat, fitted), (repeat, fitted)
    # Exact synthetic graph checks raw+origin signs and independent cycle closure.
    origins = {f: seed_origin(f)+np.array([.1*(f-13), -.03*(f-13), .04*(f-13)]) for f in FOVS}
    exact = [dict(k=k, n=n, accepted=True, translation_xyz_um=(origins[n]-origins[k]).tolist())
             for k, n in itertools.combinations(FOVS, 2)]
    solved = solve_graph(exact, cfg)
    assert solved['status'] == 'PASS', solved
    for f in FOVS:
        assert np.allclose(solved['origins_xyz_um'][str(f)], origins[f], atol=1e-7)
    corrupted = [dict(e) for e in exact]
    corrupted[0]['translation_xyz_um'] = (np.array(corrupted[0]['translation_xyz_um'])+[1., 0., 0.]).tolist()
    rejected = solve_graph(corrupted, cfg)
    assert rejected['status'] == 'FAIL'
    assert any('loop closure failed' in r for r in rejected['reasons'])
    # Deliberately bad heldout observations fail validation without changing fit.
    bad_edge, _ = register_edge(13, 14, a, bb, np.array([-2., 1., 0.]), cfg)
    assert not bad_edge['accepted']
    print(json.dumps(dict(status='PASS', tests=['translation sign', 'unmatched-point robustness',
        'spatial holdout index separation', 'heldout excluded from fitting',
        'global-origin solve', 'cycle-outlier rejection', 'bad-holdout rejection'],
        recovered_translation_xyz_um=fitted.tolist(), truth_xyz_um=truth.tolist(),
        training_pairs=edge['training']['pairs'], heldout_pairs=edge['heldout']['pairs'])), flush=True)


def diagnose_models(registration_dir, output_dir):
    """Exploratory model comparison on fixed saved train/validation identities.

    This deliberately does not change graph.json or authorize decoding. The
    spatial validation set has now informed model selection; an eventual
    selected model requires a new independent validation region.
    """
    registration_dir, output_dir = Path(registration_dir), Path(output_dir)
    if (output_dir/'model_comparison.json').exists():
        raise FileExistsError('Exploratory evidence is not overwritten')
    output_dir.mkdir(parents=True, exist_ok=True)
    original = json.loads((registration_dir/'graph.json').read_text())
    saved = np.load(registration_dir/'matches.npz', allow_pickle=False)
    cfg = original['parameters']
    clouds = {}
    for source in original['sources']:
        fov = source['fov']
        with np.load(source['cloud_path'], allow_pickle=False) as archive:
            xyz = archive['xyz_um']
        clouds[fov] = xyz[saved['fov%d_bright_source_indices' % fov]]
    comparisons, archive_arrays = [], {}
    regression_scale = np.array([cfg['microns_per_pixel']]*2+[.5])
    qualified_edges = [e for e in original['edges'] if 'translation_xyz_um' in e]
    for e in qualified_edges:
        k, n = e['k'], e['n']
        key = '%d_%d_' % (k, n)
        train, held = saved[key+'training_pairs'], saved[key+'heldout_pairs']
        aa, bb = clouds[k][train[:, 0]], clouds[n][train[:, 1]]
        ah, bh = clouds[k][held[:, 0]], clouds[n][held[:, 1]]
        center, scale = np.mean(bb, axis=0), np.array([100., 100., 50.])
        result = dict(k=k, n=n, training_pairs=len(train), heldout_pairs=len(held),
                      translation_heldout=e['heldout'], models={})
        for model in ['affine', 'quadratic_xy']:
            def design(x):
                p = (x-center)/scale
                features = [np.ones(len(x)), p[:, 0], p[:, 1], p[:, 2]]
                if model == 'quadratic_xy':
                    features += [p[:, 0]**2, p[:, 1]**2, p[:, 0]*p[:, 1]]
                return np.column_stack(features)
            xx, xh = design(bb), design(bh)
            target = aa-bb
            initial = np.linalg.lstsq(xx, target, rcond=None)[0]
            fit = least_squares(lambda v: ((xx@v.reshape(-1, 3)-target)/regression_scale).ravel(),
                initial.ravel(), loss='soft_l1', f_scale=1., max_nfev=1000,
                ftol=1e-12, xtol=1e-12, gtol=1e-12)
            beta = fit.x.reshape(-1, 3)
            train_residual, held_residual = aa-(bb+xx@beta), ah-(bh+xh@beta)
            train_metrics = metrics(train_residual, min(e['spatial_split']['training_counts']), cfg)
            held_metrics = metrics(held_residual, min(e['spatial_split']['heldout_counts']), cfg)
            reasons = gate_metrics(train_metrics, 'training', cfg['min_fit_pairs'], cfg)
            reasons += gate_metrics(held_metrics, 'heldout', cfg['min_holdout_pairs'], cfg, True)
            result['models'][model] = dict(training=train_metrics, heldout=held_metrics,
                passes_residual_gates=not reasons, gate_reasons=reasons,
                center_xyz_um=center.tolist(), scale_xyz_um=scale.tolist(),
                coefficients_feature_by_xyz_um=beta.tolist(),
                feature_names=['1', 'qx', 'qy', 'qz']+([] if model == 'affine' else ['qx2', 'qy2', 'qx_qy']),
                mapping='p_reference = p_neighbor + features((p_neighbor-center)/scale) @ coefficients')
            archive_arrays[key+model+'_training_residual_xyz_um'] = train_residual
            archive_arrays[key+model+'_heldout_residual_xyz_um'] = held_residual
        archive_arrays[key+'training_neighbor_xyz_um'] = bb
        archive_arrays[key+'heldout_neighbor_xyz_um'] = bh
        comparisons.append(result)

    # Joint anchored affine trial: all four training edges only, no held-out
    # points in this objective. Coefficients parametrize displacement in native
    # coordinates about the center of the camera; prior is 1% on matrix terms.
    unknown = [f for f in FOVS if f != 13]
    center = np.array([.5*cfg['image_width_pixels']*cfg['microns_per_pixel']]*2+[75.])
    scale = np.array([100., 100., 50.])
    joint_scale = np.array([cfg['microns_per_pixel']]*2+[1.])
    terms = []
    for e in qualified_edges:
        k, n = e['k'], e['n']; key = '%d_%d_' % (k, n)
        pair = saved[key+'training_pairs']
        aa, bb = clouds[k][pair[:, 0]], clouds[n][pair[:, 1]]
        terms.append((k, n, aa, bb))
    rows = sum(len(t[2]) for t in terms)*3
    design_matrix = np.zeros((rows, 36))
    base = np.empty(rows)
    offset = 0
    for k, n, aa, bb in terms:
        bas = aa+seed_origin(k)-bb-seed_origin(n)
        base[offset:offset+3*len(aa)] = (bas/joint_scale).ravel()
        for fov, coords, sign in [(k, aa, 1.), (n, bb, -1.)]:
            if fov == 13:
                continue
            fi = unknown.index(fov)
            xx = np.column_stack([np.ones(len(coords)), (coords-center)/scale])
            for d in range(3):
                row_idx = offset+3*np.arange(len(coords))+d
                col_idx = fi*12+3*np.arange(4)+d
                design_matrix[np.ix_(row_idx, col_idx)] = sign*xx/joint_scale[d]
        offset += 3*len(aa)
    prior = np.zeros((27, 36))
    ri = 0
    for fi in range(3):
        for feature in range(1, 4):
            for d in range(3):
                prior[ri, fi*12+feature*3+d] = 1./(scale[feature-1]*.01)
                ri += 1
    all_design = np.concatenate([design_matrix, prior])
    all_base = np.concatenate([base, np.zeros(27)])
    joint_fit = least_squares(lambda v: all_design@v+all_base, np.zeros(36),
        jac=lambda v: all_design.copy(), loss='soft_l1', f_scale=1., max_nfev=1000,
        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    matrices = {13: np.eye(4)}
    for fi, fov in enumerate(unknown):
        beta = joint_fit.x[fi*12:fi*12+12].reshape(4, 3)
        delta = beta[1:].T/scale
        mat = np.eye(4)
        mat[:3, :3] += delta
        mat[:3, 3] = seed_origin(fov)+beta[0]-delta@center
        matrices[fov] = mat

    def apply(fov, points):
        return points@matrices[fov][:3, :3].T+matrices[fov][:3, 3]
    joint_edges, joint_reasons = [], []
    for e in qualified_edges:
        k, n = e['k'], e['n']; key = '%d_%d_' % (k, n)
        em = dict(k=k, n=n)
        for split in ['training', 'heldout']:
            pairs = saved[key+split+'_pairs']
            residual = apply(k, clouds[k][pairs[:, 0]])-apply(n, clouds[n][pairs[:, 1]])
            em[split] = metrics(residual, min(e['spatial_split'][split+'_counts']), cfg)
            archive_arrays[key+'joint_affine_'+split+'_residual_xyz_um'] = residual
            minimum = cfg['min_fit_pairs'] if split == 'training' else cfg['min_holdout_pairs']
            reasons = gate_metrics(em[split], split, minimum, cfg, split == 'heldout')
            joint_reasons.extend(['%s-%s %s' % (k, n, r) for r in reasons])
        joint_edges.append(em)
    present_sides = {(e['k'], e['n']) for e in qualified_edges}
    if not SIDES <= present_sides:
        joint_reasons.append('Not all four side edges have supported correspondences')
    joint = dict(status='EXPLORATORY_PASS' if not joint_reasons else 'FAIL',
        reasons=joint_reasons, edges=joint_edges, native_to_anchor_matrices_4x4={str(f): m.tolist() for f, m in matrices.items()},
        anchor=13, model='joint affine; 13 identity; soft_l1 robust residuals; linear part 1% prior',
        axis_residual_scales_um=joint_scale.tolist(), objective_uses='training matches only',
        jacobians={str(f): dict(determinant=float(np.linalg.det(m[:3, :3])),
                     singular_values=np.linalg.svd(m[:3, :3], compute_uv=False).tolist()) for f, m in matrices.items()})
    e_map = {(e['k'], e['n']): np.asarray(e['translation_xyz_um']) for e in qualified_edges}
    cycle = None
    if SIDES <= set(e_map):
        rr = e_map[13, 14]+e_map[14, 25]+e_map[25, 26]-e_map[13, 26]
        cycle = dict(xyz_um=rr.tolist(), xy_pixels=float(np.linalg.norm(rr[:2])/cfg['microns_per_pixel']), z_um=float(abs(rr[2])))
    report = dict(status='EXPLORATORY_ONLY_NOT_APPROVED_FOR_DECODING',
        source_code_sha256=sha256(__file__), source_registration=str(registration_dir.resolve()),
        source_graph_sha256=sha256(registration_dir/'graph.json'),
        source_matches_sha256=sha256(registration_dir/'matches.npz'),
        parameters=cfg, pair_models=comparisons, joint_affine=joint,
        unaccepted_translation_square_closure=cycle,
        validation_warning='These held-out beads were excluded from every fit but have now informed model selection. Validate any chosen extension in a new independent spatial region before production.',
        broad_z_search_note='Training-only Â±30Âµm Z search with obsolete +stage-X signs produced only4â€“6 reciprocal matches; corrected stage-X signs produce hundreds within originalÂ±4Âµm Z range. This is orientation, not large focus drift.')
    np.savez_compressed(output_dir/'model_residuals.npz', **archive_arrays)
    (output_dir/'model_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(status=report['status'], joint_affine_status=joint['status'],
                         joint_affine_reasons=joint['reasons'], output=str(output_dir))), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--diagnose-models', metavar='REGISTRATION_DIR',
                        help='Exploratory affine/quadratic models on saved train/holdout matches')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.diagnose_models:
        if not args.output_dir:
            parser.error('--output-dir is required with --diagnose-models')
        return diagnose_models(args.diagnose_models, args.output_dir)
    if not args.cloud_dir or not args.output_dir:
        parser.error('--cloud-dir and --output-dir are required unless --self-test')
    return run(args.cloud_dir, args.output_dir)


if __name__ == '__main__':
    raise SystemExit(main())
