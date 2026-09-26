"""Per-depth spatial joins and MERlin boundary-correction decisions.

Cell IDs are global archived IDs. Repeated views of the same cell are unioned
before buffering; candidates and interior evidence are deduplicated by cell ID.
"""
import numpy as np
import geopandas as gpd
from shapely import affinity
from shapely.geometry import shape
from shapely.ops import unary_union


def mask_polygons(mask, uid_map):
    from rasterio.features import shapes
    from affine import Affine
    # Existing lookup round(full-resolution pixel)//2: exact pixel footprint.
    transform = Affine(.2986, 0, -.5*.1493, 0, .2986, -.5*.1493)
    result = {}
    for geo, label in shapes(mask.astype(np.int32), mask=mask > 0, transform=transform):
        uid = int(uid_map[int(label)])
        assert uid > 0
        p = shape(geo)
        assert p.is_valid
        result.setdefault(uid, []).append(p)
    return {uid: unary_union(parts) for uid, parts in result.items()}


def reference_polygons(plane_maps, candidates, crop):
    result = {}
    for j, tx, ty, how in candidates:
        for uid, poly in plane_maps[j].items():
            moved = affinity.translate(poly, (.1493*(crop[0]-tx)), (.1493*(crop[1]-ty)))
            result.setdefault(uid, []).append(moved)
    return {uid: unary_union(parts) for uid, parts in result.items()}


def spatial_pairs(xy, polygons, buffer):
    if not len(xy) or not polygons:
        return np.empty((0, 2), np.int64)
    ids, geos = [], []
    for uid in sorted(polygons):
        p = polygons[uid]
        p = p if buffer == 0 else p.buffer(buffer)
        if not p.is_empty:
            ids.append(uid); geos.append(p)
    if not geos:
        return np.empty((0, 2), np.int64)
    cells = gpd.GeoDataFrame({'cell': ids}, geometry=geos)
    points = gpd.GeoDataFrame({'row': np.arange(len(xy))}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]))
    joined = gpd.sjoin(points, cells, how='inner', predicate='within')
    return np.unique(joined[['row', 'cell']].to_numpy(np.int64), axis=0)


def first_hosts(pairs, n):
    first = np.zeros(n, np.int64)
    if len(pairs):
        rows, at = np.unique(pairs[:, 0], return_index=True)
        first[rows] = pairs[at, 1]
    return first


def resolve(normal, shrunk, expanded, types, interior, rng):
    """Same decision order as MERlin_jiahao; zero is unassigned.

    Unlike per-FOV MERlin output, each deduplicated barcode is owned once.
    Interior type evidence is pooled across all reference FOVs and depths.
    """
    result = np.zeros(len(types), np.int64)
    present = np.zeros(len(types), bool)
    if len(expanded):
        present[expanded[:, 0]] = True
    inside = present & (shrunk > 0)
    result[inside] = shrunk[inside]
    remaining = present & ~inside
    ordinary = remaining & (normal > 0)
    result[ordinary] = normal[ordinary]
    supported_normal = np.zeros(len(types), bool)
    rows = np.flatnonzero(ordinary)
    supported_normal[rows] = interior[normal[rows], types[rows]] > 0
    if len(expanded):
        rows, owners = expanded.T
        candidates = expanded[remaining[rows] & ~supported_normal[rows] & (interior[owners, types[rows]] > 0)]
        if len(candidates):
            candidates = candidates[np.lexsort((candidates[:, 1], candidates[:, 0]))]
            ids, starts = np.unique(candidates[:, 0], return_index=True)
            ends = np.r_[starts[1:], len(candidates)]
            for row, a, b in zip(ids, starts, ends):
                result[row] = rng.choice(candidates[a:b, 1])
    return result


def partition_counts(cells, xyz, barcode_types, barcode_count, buffer=.5, rng=None):
    """Validation adapter for comparison with the actual MERlin method."""
    rng = np.random if rng is None else rng
    xyz = np.asarray(xyz); types = np.asarray(barcode_types, dtype=int)
    n = len(types); normal = np.zeros(n, int); shrunk = np.zeros(n, int)
    expanded = []; interior = np.zeros((len(cells)+1, barcode_count), np.int64)
    finite = np.isfinite(xyz).all(1)
    for z in np.unique(np.rint(xyz[finite, 2])):
        rows = np.flatnonzero(finite & (np.rint(xyz[:, 2]) == z))
        polys = {}
        for i, cell in enumerate(cells):
            bounds = cell.get_boundaries()
            if 0 <= z < len(bounds) and bounds[int(z)]:
                # Preserve individual polygons for exact canonical buffering.
                polys[i+1] = bounds[int(z)]
        for b, dest in [(0, normal), (-buffer, shrunk), (buffer, None)]:
            shapes = {u: unary_union([p if b == 0 else p.buffer(b) for p in ps]) for u, ps in polys.items()}
            pairs = spatial_pairs(xyz[rows, :2], shapes, 0)
            if len(pairs):
                pairs[:, 0] = rows[pairs[:, 0]]
                if dest is not None:
                    rr, at = np.unique(pairs[:, 0], return_index=True); dest[rr] = pairs[at, 1]
                if b < 0:
                    np.add.at(interior, (pairs[:, 1], types[pairs[:, 0]]), 1)
                if b > 0:
                    expanded.append(pairs)
    pairs = np.concatenate(expanded) if expanded else np.empty((0, 2), int)
    assigned = resolve(normal, shrunk, pairs, types, interior, rng)
    counts = np.zeros_like(interior); keep = assigned > 0
    np.add.at(counts, (assigned[keep], types[keep]), 1)
    return counts[1:]
