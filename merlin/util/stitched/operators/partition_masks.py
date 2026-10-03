"""Reuse the archived 0718 true-mask assignment in a fresh stitched frame.

Existing cell identities, offsets, candidate order and four-rank fallback are
preserved. The native-to-production crop conversion is explicit. Out-of-depth
points are left unassigned rather than clipped into an endpoint mask plane.
"""
import numpy as np


def candidate_catalog(offsets):
    catalog = {k: [] for k in range(52)}
    for k, v in offsets['own'].items():
        if v and v.get('t'):
            catalog[int(k)].append((int(k), *v['t'], v.get('how', 'unknown')))
    for key, v in offsets['pairs'].items():
        k, j = map(int, key.split('_'))
        catalog[k].append((j, *v['t'], v.get('how', 'unknown')))
    return catalog


def assign_native(owner, native, crop_xy, catalog, masks, uid_maps, dz=0):
    n = len(owner)
    uid = np.zeros(n, np.int64)
    stack = np.full(n, -1, np.int16)
    label = np.zeros(n, np.uint32)
    rank_used = np.full(n, -1, np.int8)
    confidence = np.full(n, 'unassigned', dtype='<U32')
    z = np.full(n, -999, np.int32)
    finite = np.isfinite(native).all(1) & (owner > 0)
    z[finite] = np.rint(native[finite, 2] - 1.0 + dz).astype(np.int32)
    depth_ok = finite & (native[:, 2] >= 1) & (native[:, 2] <= 149) & (z >= 0) & (z < 149)
    for f in np.unique(owner[depth_ok] - 1):
        rows = np.flatnonzero(depth_ok & (owner == f + 1))
        candidates = catalog[int(f)]
        if not candidates:
            continue
        xy = native[rows, :2] / .1493 - crop_xy[int(f)]
        margins = np.array([np.minimum.reduce([xy[:, 0]+tx, 2047-xy[:, 0]-tx,
                           xy[:, 1]+ty, 2047-xy[:, 1]-ty]) for j, tx, ty, how in candidates])
        order = np.argsort(-margins, axis=0)
        unresolved = np.ones(len(rows), bool)
        for rank in range(min(4, len(candidates))):
            choices = order[rank]
            eligible = unresolved & (margins[choices, np.arange(len(rows))] > 0)
            for ci in np.unique(choices[eligible]):
                j, tx, ty, how = candidates[ci]
                local = np.flatnonzero(eligible & (choices == ci))
                rr = rows[local]
                ix = np.rint(xy[local, 0]+tx).astype(int)//2
                iy = np.rint(xy[local, 1]+ty).astype(int)//2
                labs = masks(j)[z[rr], iy, ix]
                hit = labs > 0
                if not hit.any():
                    continue
                rr = rr[hit]; labs = labs[hit]
                mapping = uid_maps[j]
                if int(labs.max()) >= len(mapping) or np.any(mapping[labs] == 0):
                    raise ValueError('Mask label lacks archived cell identity')
                uid[rr] = mapping[labs]; stack[rr] = j; label[rr] = labs
                rank_used[rr] = rank; confidence[rr] = how
                unresolved[local[hit]] = False
    return dict(cell_index=uid, mask_stack=stack, mask_label=label,
                mask_candidate_rank=rank_used, mask_registration_method=confidence,
                mask_plane_index=z, mask_depth_supported=depth_ok)
