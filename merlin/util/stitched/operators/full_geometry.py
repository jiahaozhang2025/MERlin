"""Generalize the pilot's within-round graph size; preserve its objective."""
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

import fit_graph as fg
import global_quadratic as gq


def configure(inventory):
    fovs = tuple(inventory['source_fovs'])
    if 13 not in fovs or len(set(fovs)) != len(fovs):
        raise ValueError('Unique source FOVs including anchor13 required')
    edges = [tuple(e) for e in inventory['side_edges']]
    reached = {13}
    while True:
        new = reached | {f for e in edges if set(e) & reached for f in e}
        if new == reached:
            break
        reached = new
    if reached != set(fovs):
        raise ValueError('Disconnected source graph')
    fg.FOVS = gq.FOVS = fovs
    fg.STAGES = {f: tuple(inventory['positions'][f]) for f in fovs}
    fg.SIDES = set(edges)
    gq.SIDE_EDGES = edges
    gq.fit_model = fit_model


def fit_model(clouds, edges, cfg):
    """Same robust joint polynomial residuals/priors as the pilot, sparse Jacobian."""
    fovs = tuple(gq.FOVS)
    if set(clouds) != set(fovs):
        raise ValueError('Source clouds do not match configured graph')
    unknown = [f for f in fovs if f != 13]
    width = cfg['image_width_pixels'] * cfg['microns_per_pixel']
    center, scale = np.array([width/2, width/2, 75.]), np.array([100., 100., 50.])
    residual_scale = np.array([cfg['microns_per_pixel']]*2 + [1.])
    ncols = len(unknown) * 21
    ndata = sum(len(e['training_pairs']) for e in edges) * 3
    nprior = len(unknown) * 18
    base = np.zeros(ndata + nprior)
    rows, cols, values = [], [], []
    offset = 0
    for e in edges:
        k, n, pair = e['k'], e['n'], e['training_pairs']
        aa, bb = clouds[k][pair[:, 0]], clouds[n][pair[:, 1]]
        base[offset:offset+3*len(pair)] = ((aa + fg.seed_origin(k) - bb - fg.seed_origin(n))/residual_scale).ravel()
        for fov, pp, sign in [(k, aa, 1.), (n, bb, -1.)]:
            if fov == 13:
                continue
            fi = unknown.index(fov)
            features = gq.basis(pp, center, scale)
            for d in range(3):
                rr = offset + 3*np.arange(len(pp)) + d
                cc = fi*21 + 3*np.arange(7) + d
                rows.append(np.repeat(rr, 7))
                cols.append(np.tile(cc, len(rr)))
                values.append((sign*features/residual_scale[d]).ravel())
        offset += len(pair)*3
    index = ndata
    for fi in range(len(unknown)):
        for feature in range(1, 7):
            for d in range(3):
                rows.append(np.array([index])); cols.append(np.array([fi*21+feature*3+d]))
                values.append(np.array([1./(scale[feature-1]*.01) if feature <= 3 else 1.]))
                index += 1
    assert index == ndata+nprior and offset == ndata
    design = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(ndata+nprior, ncols)).tocsr()
    fit = least_squares(lambda p: design @ p + base, np.zeros(ncols),
        jac=lambda p: design.copy(), loss='soft_l1', f_scale=1., max_nfev=1000,
        ftol=1e-12, xtol=1e-12, gtol=1e-12, tr_solver='lsmr',
        tr_options=dict(atol=1e-12, btol=1e-12, maxiter=max(1000, 4*ncols)))
    model = dict(model_type='native_xyz_to_anchor13_global_quadratic_displacement',
        anchor_fov=13, coordinate_units='micrometers', coordinate_order='xyz',
        equation='global = native + stage_origin + [1,qx,qy,qz,qx^2,qy^2,qx*qy] @ beta; q=(native-center)/scale',
        normalization=dict(center_xyz_um=center.tolist(), scale_xyz_um=scale.tolist()),
        features=gq.FEATURE_NAMES, models={}, optimization=dict(loss='soft_l1', f_scale=1.,
            residual_scale_xyz_um=residual_scale.tolist(), linear_prior_fraction=.01,
            quadratic_prior_um=1., success=bool(fit.success), message=str(fit.message),
            cost=float(fit.cost), evaluations=int(fit.nfev), jacobian_storage='sparse CSR',
            sources=len(fovs), coefficients=ncols, training_residual_rows=ndata,
            objective_uses='Only central-slab training ties plus identical fixed zero-deformation prior'))
    for fov in fovs:
        beta = np.zeros((7,3)) if fov == 13 else fit.x[unknown.index(fov)*21:unknown.index(fov)*21+21].reshape(7,3)
        model['models'][str(fov)] = dict(stage_origin_xyz_um=fg.seed_origin(fov).tolist(), coefficients_7x3_um=beta.tolist())
    return model
