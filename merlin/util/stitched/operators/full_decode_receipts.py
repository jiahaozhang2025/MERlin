"""Frozen tile/depth scheduling and auditable, resumable decode outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from full_preprocess import sha


def make_tasks(layout, z_positions, depths_per_task=10):
    z = np.asarray(z_positions, float)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all() or not np.all(np.diff(z) > 0):
        raise ValueError('Reference depths must be finite and strictly increasing')
    if depths_per_task < 1:
        raise ValueError('Invalid depth chunk size')
    tiles = layout['tiles']
    if len({t['tile_id'] for t in tiles}) != len(tiles):
        raise ValueError('Duplicate output tile')
    # Include even geometrically empty tiles. An explicit empty support receipt
    # is distinguishable from a failed, skipped, or never scheduled output.
    tasks = []
    for tile in tiles:
        for start in range(0, len(z), depths_per_task):
            tasks.append(dict(task_id=len(tasks), tile_id=tile['tile_id'],
                planes=[dict(z_index=i, physical_z_um=float(z[i]))
                        for i in range(start, min(start + depths_per_task, len(z)))]))
    return tasks


def validate_rows(frame, tile, plane, layout):
    from tile_layout import owner_for_global_xy
    required = ['barcode_uid', 'output_tile', 'z', 'physical_z_um', 'global_x', 'global_y', 'is_blank']
    if any(n not in frame for n in required):
        raise ValueError('Missing decode output columns')
    if not frame.barcode_uid.is_unique:
        raise ValueError('Repeated barcode identity')
    if len(frame):
        if not np.isfinite(frame[['global_x', 'global_y', 'z', 'physical_z_um']].to_numpy()).all():
            raise ValueError('Nonfinite barcode coordinates')
        if not (frame.output_tile == tile['tile_id']).all() or not (frame.z == plane['z_index']).all():
            raise ValueError('Barcode belongs to another tile/depth')
        if not (frame.physical_z_um == plane['physical_z_um']).all():
            raise ValueError('Wrong physical depth')
        if not (owner_for_global_xy(frame[['global_x', 'global_y']].to_numpy(), layout) == tile['tile_id']).all():
            raise ValueError('Barcode violates unique output ownership')
        if not frame.is_blank.isin([True, False]).all():
            raise ValueError('Invalid blank annotation')


def write_plane(folder, frame, support, tile, plane, layout, report, plan_sha, input_hashes, raw_audit):
    folder = Path(folder)
    validate_rows(frame, tile, plane, layout)
    if support.dtype != bool or list(support.shape) != tile['shape_yx']:
        raise ValueError('Core support raster has incorrect schema')
    if not raw_audit.get('raw_size_mtime_unchanged') or not raw_audit.get('xml_schedule_hashes_unchanged'):
        raise ValueError('Raw audit did not pass')
    if any(sha(p) != h for p, h in input_hashes.items()):
        raise RuntimeError('Decode input changed before commit')
    folder.mkdir(parents=True, exist_ok=False)
    frame.to_csv(folder/'raw.csv.gz', index=False)
    np.savez_compressed(folder/'core_support.npz', support=support)
    receipt = dict(status='COMPLETE', plan_sha256=plan_sha, tile_id=tile['tile_id'], plane=plane,
        raw_barcodes=len(frame), raw_blank_barcodes=int(frame.is_blank.sum()),
        supported_core_pixels=int(support.sum()), core_pixels=int(support.size),
        decode_report=report, input_sha256=input_hashes, raw_audit=raw_audit,
        all_hashed_inputs_unchanged=True,
        output_sha256={n:sha(folder/n) for n in ['raw.csv.gz', 'core_support.npz']})
    if any(sha(p) != h for p, h in input_hashes.items()):
        raise RuntimeError('Decode input changed during commit; receipt withheld')
    (folder/'complete.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    return receipt


def verify_plane(folder, tile, plane, layout, plan_sha, input_hashes, check_inputs=True):
    folder = Path(folder)
    rec = json.loads((folder/'complete.json').read_text())
    if rec['status'] != 'COMPLETE' or rec['plan_sha256'] != plan_sha or rec['tile_id'] != tile['tile_id'] or rec['plane'] != plane:
        raise ValueError('Receipt belongs to a different decode plan/tile/depth')
    if not rec['all_hashed_inputs_unchanged'] or rec['input_sha256'] != input_hashes:
        raise ValueError('Receipt has different protected inputs')
    if check_inputs and any(sha(p) != h for p,h in input_hashes.items()):
        raise RuntimeError('Protected decode input changed')
    for name in ['raw.csv.gz', 'core_support.npz']:
        if sha(folder/name) != rec['output_sha256'][name]:
            raise ValueError('Decode output changed: '+name)
    frame = pd.read_csv(folder/'raw.csv.gz')
    validate_rows(frame, tile, plane, layout)
    with np.load(folder/'core_support.npz', allow_pickle=False) as arrays:
        support = arrays['support']
    if support.dtype != bool or list(support.shape) != tile['shape_yx'] or int(support.sum()) != rec['supported_core_pixels'] or support.size != rec['core_pixels']:
        raise ValueError('Support receipt disagrees with measured mask')
    if len(frame) != rec['raw_barcodes'] or int(frame.is_blank.sum()) != rec['raw_blank_barcodes']:
        raise ValueError('Barcode count receipt disagrees with output')
    if not rec['raw_audit'].get('raw_size_mtime_unchanged') or not rec['raw_audit'].get('xml_schedule_hashes_unchanged'):
        raise ValueError('Completed decode has no passing raw audit')
    return rec
