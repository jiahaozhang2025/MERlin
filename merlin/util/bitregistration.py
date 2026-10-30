"""Barcode-weighted residual bit alignment from decoded molecules.

All coordinates are (x, y), and offsets are output-to-input pull translations.
"""
import itertools

import numpy as np


def robust_offsets(i, j, delta, weights, bit_count, max_offset=2.):
    i, j = np.asarray(i, int), np.asarray(j, int)
    delta, base = np.asarray(delta, float), np.asarray(weights, float)
    design = np.zeros((len(i), bit_count))
    design[np.arange(len(i)), i] = 1
    design[np.arange(len(i)), j] = -1
    if np.linalg.matrix_rank(design) != bit_count - 1:
        raise ValueError('Molecule-pair graph does not connect all bits')
    w, offset = base.copy(), np.zeros((bit_count, 2))
    for iteration in range(12):
        lhs = design.T @ (w[:, None] * design) + np.ones((bit_count, bit_count)) / bit_count
        new = np.linalg.solve(lhs, design.T @ (w[:, None] * delta))
        error = np.linalg.norm(design @ new - delta, axis=1)
        scale = max(.15, float(1.4826 * np.median(np.abs(error - np.median(error)))))
        w = base * np.minimum(1., 1.5 * scale / np.maximum(error, 1e-9))
        change, offset = float(np.max(np.abs(new - offset))), new
        if change < 1e-5:
            break
    if not np.isfinite(offset).all() or np.max(np.abs(offset)) >= max_offset:
        raise ValueError('Unreliable bit offset magnitude')
    return offset, dict(iterations=iteration + 1,
                        training_residual_median_px=float(np.median(error)),
                        training_residual_p95_px=float(np.quantile(error, .95)),
                        gauge='mean bit offset zero',
                        max_abs_offset_px=float(np.max(np.abs(offset))))


def positions(images, table, book):
    from merlin.util.registration import refine_position
    result = np.full((len(table), book.shape[1], 2), np.nan)
    xy, ids = table[['x', 'y']].to_numpy(float), table.barcode_id.to_numpy(int)
    for bit in range(book.shape[1]):
        for row in np.flatnonzero(book[ids, bit] > 0):
            result[row, bit] = refine_position(images[bit], *xy[row])
    return result


def molecule_pairs(table, book, pos, subset, after=None):
    """Fixed before/after pairs; never reject a large AFTER residual.

    Each molecule contributes unit total weight, regardless of its bit count.
    """
    ii, jj, dd, ww, aa = [], [], [], [], []
    eligible = missing = 0
    ids = table.barcode_id.to_numpy(int)
    for row in np.flatnonzero(subset):
        on = np.flatnonzero(book[ids[row]])
        p = pos[row, on]
        if not np.isfinite(p).all() or np.max(np.abs(p - np.median(p, axis=0))) > 2.5:
            continue
        eligible += 1
        if after is not None and not np.isfinite(after[row, on]).all():
            missing += 1
            continue
        combos = list(itertools.combinations(range(len(on)), 2))
        for a, b in combos:
            ii.append(on[a]); jj.append(on[b]); dd.append(p[a] - p[b])
            ww.append(1. / len(combos))
            if after is not None:
                aa.append(after[row, on[a]] - after[row, on[b]])
    return (np.asarray(ii, int), np.asarray(jj, int),
            np.asarray(dd, float).reshape(-1, 2), np.asarray(ww),
            eligible, missing, np.asarray(aa, float).reshape(-1, 2))


def errors(delta):
    norms = np.linalg.norm(delta, axis=1)
    if not len(norms):
        raise ValueError('No eligible molecule pairs')
    return dict(median_px=float(np.median(norms)), p95_px=float(np.quantile(norms, .95)),
                pairs=len(norms))


def select_molecules(table, blank, max_distance, tile=320, cap=160):
    take = ((table.area >= 4) & (table.min_distance <= max_distance)
            & ~blank[table.barcode_id.to_numpy(int)])
    selected = table.loc[take].copy()
    mod = np.mod(selected[['x', 'y']].to_numpy(float), tile)
    selected = selected.loc[((mod >= 8) & (mod < tile - 8)).all(axis=1)].copy()
    selected['_tile'] = ((selected.y // tile).astype(int) * 10
                         + (selected.x // tile).astype(int))
    if selected.empty:
        return selected.reset_index(drop=True)
    return (selected.groupby('_tile', group_keys=False).sample(frac=1., random_state=1717)
            .groupby('_tile', group_keys=False).head(cap).reset_index(drop=True))


def split_tiles(table, tile=320):
    return ((np.floor(table.x / tile) + np.floor(table.y / tile)).to_numpy(int) % 2) == 0


def fit(table, book, before, tile=320):
    train = split_tiles(table, tile)
    if len(table) < 600 or min(train.sum(), (~train).sum()) < 300:
        raise ValueError('Too few reliable molecules in disjoint spatial splits')
    i, j, d, w, n, _, _ = molecule_pairs(table, book, before, train)
    if n < 200:
        raise ValueError('Too few localized training molecules')
    offset, info = robust_offsets(i, j, d, w, book.shape[1])
    vi, vj, vd, vw, n, _, _ = molecule_pairs(table, book, before, ~train)
    if n < 200:
        raise ValueError('Too few localized validation molecules')
    old, new = errors(vd), errors(vd - (offset[vi] - offset[vj]))
    accepted = new['median_px'] < old['median_px'] and new['p95_px'] <= old['p95_px'] + .02
    replicate_offsets = disagreement = None
    try:
        replicate, _ = robust_offsets(vi, vj, vd, vw, book.shape[1])
        replicate_offsets = replicate.tolist()
        disagreement = np.linalg.norm(offset-replicate, axis=1).tolist()
    except ValueError:
        pass  # A diagnostic replicate does not change the frozen fit policy.
    return offset, dict(accepted=bool(accepted), training=info, before=old, predicted_after=new,
                        molecules=len(table), training_molecules=int(train.sum()),
                        validation_molecules=int((~train).sum()),
                        fit_and_validation_tiles_disjoint=True,
                        validation_replicate_offsets_xy=replicate_offsets,
                        split_disagreement_px=disagreement,
                        training_pairs_per_bit=np.bincount(np.r_[i, j], minlength=book.shape[1]).tolist(),
                        validation_pairs_per_bit=np.bincount(np.r_[vi, vj], minlength=book.shape[1]).tolist())


def validate_applied(table, book, before, after, colors, tile=320):
    i, j, b, _, n, missing, a = molecule_pairs(table, book, before, ~split_tiles(table, tile), after)
    if n < 200 or missing / max(n, 1) > .01:
        raise ValueError('Insufficient identical held-out pairs or >1% missing after correction')
    rows = {}
    colors = np.asarray(colors)
    for color in ['all'] + sorted(set(colors.tolist())):
        use = np.ones(len(i), bool) if color == 'all' else (colors[i] == color) & (colors[j] == color)
        if use.any():
            rows[color] = dict(before=errors(b[use]), after=errors(a[use]),
                               eligible_molecules=n, missing_after_molecules=missing)
    old, new = rows['all']['before'], rows['all']['after']
    return bool(new['median_px'] < old['median_px'] and new['p95_px'] <= old['p95_px'] + .03), rows


def matrix_dict(value):
    value = getattr(value, 'transformations', value)
    return {a: {b: t.params.tolist() for b, t in row.items()} for a, row in value.items()}


def transformations(value):
    from skimage.transform import SimilarityTransform
    return {a: {b: SimilarityTransform(matrix=np.asarray(m)) for b, m in row.items()}
            for a, row in value.items()}


def chromatic_step(old, new, shape):
    h, w = shape
    grid = np.array([[.09375*w, .09375*h, 1], [.09375*w, .90625*h, 1],
                     [.90625*w, .09375*h, 1], [.90625*w, .90625*h, 1], [.5*w, .5*h, 1]])
    reference = sorted(old)[0]
    return max(float(np.linalg.norm(((new[reference][c].params - old[reference][c].params)
                                    @ grid.T).T[:, :2], axis=1).max())
               for c in old[reference] if c != reference)
