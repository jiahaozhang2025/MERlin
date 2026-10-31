"""Location strata: a per-fov label volume that gives every barcode a location.

The adaptive filter fits one blank-fraction histogram over every barcode, so
a location where blank calls are more common than average passes at a
threshold calibrated mostly on the others. With location
strata the threshold task also builds one histogram per location and the
filter can apply one threshold per location
(GenerateAdaptiveThreshold location_strata_path, AdaptiveFilterBarcodes
location_stratified).

A strata directory holds:
- location_strata.json (the sidecar): format 'merlin_location_strata',
  version 1; strata, the location names (a barcode's code is the index of its
  name, 1 to 255 unique non-empty names); outside_code, the code of a barcode
  outside the volume; frame and lookup, free text for humans; source,
  provenance that is not read here.
- strata_fov<fov:03d>.npz, one per fov (numpy.savez_compressed): codes, a
  uint8 3-D array [plane, row, column] with every value below the number of
  strata; origin, int64 (z0, y0, x0), the table plane, row and column of
  codes[0, 0, 0]; fov, the fov. A fov with nothing segmented is an empty
  (0, 0, 0) volume; a missing file is an error, never a silent default.

Lookup: col = rint(x) - x0, row = rint(y) - y0, plane = rint(z) - z0, with
numpy.rint (round half to even) on float64 copies of the barcode table's x, y
and z columns (x = column and y = row of the full frame, z = the plane index).
A barcode inside the volume with finite x, y and z gets codes[plane, row, col];
any other gets outside_code. For a deduplicated molecule z is its brightest
detection's plane and x, y its magnitude-weighted centroid, so the location is
read at the voxel a partition by (rint(x), rint(y), plane) assigns it to.

The volume must be in the barcode table's frame: the registration the
segmentation was made on, and the same full-frame x, y as the table.
"""
import hashlib
import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

FORMAT = 'merlin_location_strata'
VERSION = 1
METADATA_FILE = 'location_strata.json'
MAX_STRATA = 255


def fov_file_name(fov: int) -> str:
    """File name of a fov's volume in a strata directory."""
    return 'strata_fov%03d.npz' % int(fov)


def metadata_path(directory: str) -> str:
    return os.path.join(directory, METADATA_FILE)


def volume_path(directory: str, fov: int) -> str:
    return os.path.join(directory, fov_file_name(fov))


def validate_metadata(metadata: Dict, where: str = 'location strata') -> Dict:
    """Raise ValueError unless metadata follows the sidecar contract."""
    if not isinstance(metadata, dict):
        raise ValueError('%s: the sidecar must be a json object' % where)
    if metadata.get('format') != FORMAT:
        raise ValueError('%s: format must be %r, not %r'
                         % (where, FORMAT, metadata.get('format')))
    version = metadata.get('version')
    if isinstance(version, bool) or version != VERSION:
        raise ValueError('%s: version must be %d, not %r'
                         % (where, VERSION, version))
    strata = metadata.get('strata')
    if not isinstance(strata, list) or not 1 <= len(strata) <= MAX_STRATA:
        raise ValueError('%s: strata must be a list of 1 to %d names'
                         % (where, MAX_STRATA))
    if not all(isinstance(s, str) and s for s in strata):
        raise ValueError('%s: every stratum name must be a non-empty string'
                         % where)
    if len(set(strata)) != len(strata):
        raise ValueError('%s: stratum names must be unique, got %s'
                         % (where, strata))
    outside = metadata.get('outside_code')
    if isinstance(outside, bool) or not isinstance(outside, int) \
            or not 0 <= outside < len(strata):
        raise ValueError('%s: outside_code must be an integer in [0, %d), '
                         'not %r' % (where, len(strata), outside))
    return metadata


def write_metadata(directory: str, strata: Sequence[str],
                   outside_code: int = 0, frame: Optional[str] = None,
                   lookup: Optional[str] = None,
                   source: Optional[Dict] = None) -> str:
    """Write (replace) the directory's sidecar; returns its path."""
    if isinstance(outside_code, (bool, np.bool_)) \
            or not isinstance(outside_code, (int, np.integer)):
        raise ValueError('%s: outside_code must be an integer, not %r'
                         % (directory, outside_code))
    metadata = {
        'format': FORMAT,
        'version': VERSION,
        'strata': [str(s) for s in strata],
        'outside_code': int(outside_code),
        'frame': frame if frame is not None else (
            'x = barcode-table column x (full-frame px, image column), '
            'y = table column y (image row), z = table column z (plane '
            'index into the dataset z positions)'),
        'lookup': lookup if lookup is not None else (
            'numpy.rint (round half to even) of float64(x, y, z), minus '
            'origin; outside the volume or non-finite -> outside_code'),
        'source': source if source is not None else {}}
    validate_metadata(metadata, directory)
    path = metadata_path(directory)
    temporary = path + '.tmp'
    with open(temporary, 'w') as f:
        json.dump(metadata, f, indent=1)
        f.write('\n')
    os.replace(temporary, path)
    return path


def read_metadata(directory: str) -> Dict:
    """The directory's validated sidecar (FileNotFoundError if absent)."""
    path = metadata_path(directory)
    if not os.path.exists(path):
        raise FileNotFoundError('location strata sidecar not found: %s' % path)
    with open(path) as f:
        metadata = json.load(f)
    return validate_metadata(metadata, path)


def metadata_digest(directory: str) -> str:
    """sha256 of the sidecar's bytes: fit and filter must see the same one."""
    path = metadata_path(directory)
    if not os.path.exists(path):
        raise FileNotFoundError('location strata sidecar not found: %s' % path)
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _validate_volume(codes, origin, strataCount: Optional[int],
                     where: str) -> Tuple[np.ndarray, np.ndarray]:
    codes = np.asarray(codes)
    if codes.dtype != np.uint8:
        raise ValueError('%s: codes must be uint8, not %s' % (where, codes.dtype))
    if codes.ndim != 3:
        raise ValueError('%s: codes must be 3-D [plane, row, column], not %d-D'
                         % (where, codes.ndim))
    origin = np.asarray(origin)
    if origin.shape != (3,) or origin.dtype.kind not in 'iu':
        raise ValueError('%s: origin must be 3 integers (z0, y0, x0), not %r'
                         % (where, origin))
    if strataCount is not None and codes.size \
            and int(codes.max()) >= int(strataCount):
        raise ValueError('%s: codes up to %d, but only %d strata'
                         % (where, int(codes.max()), strataCount))
    return codes, origin.astype(np.int64)


def write_volume(directory: str, fov: int, codes: np.ndarray,
                 origin: Sequence[int],
                 strataCount: Optional[int] = None) -> str:
    """Write (replace) a fov's volume; returns its path. strataCount: the
    number of strata the codes must stay below (default: the directory's
    sidecar if it exists, else only the uint8 range)."""
    if strataCount is None and os.path.exists(metadata_path(directory)):
        strataCount = len(read_metadata(directory)['strata'])
    path = volume_path(directory, fov)
    codes, origin = _validate_volume(codes, origin, strataCount, path)
    temporary = path + '.tmp'
    with open(temporary, 'wb') as f:
        np.savez_compressed(f, codes=np.ascontiguousarray(codes),
                            origin=origin, fov=np.int64(fov))
    os.replace(temporary, path)
    return path


def read_volume(directory: str, fov: int, strataCount: int
                ) -> Tuple[np.ndarray, np.ndarray]:
    """(codes, origin) of a fov, validated; FileNotFoundError naming the
    path if the fov has no volume."""
    path = volume_path(directory, fov)
    if not os.path.exists(path):
        raise FileNotFoundError(
            'location strata volume for fov %d not found: %s' % (fov, path))
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in ('codes', 'origin', 'fov') if k not in f.files]
        if missing:
            raise ValueError('%s: missing %s' % (path, ', '.join(missing)))
        codes, origin, stored = f['codes'], f['origin'], f['fov']
    if stored.shape != () or int(stored) != int(fov):
        raise ValueError('%s: holds fov %r, not %d' % (path, stored, fov))
    codes, origin = _validate_volume(codes, origin, strataCount, path)
    return codes, origin


def lookup(x, y, z, codes: np.ndarray, origin: Sequence[int],
           outsideCode: int) -> np.ndarray:
    """uint8 code of each (x, y, z) (see the module docstring)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    if not len(x) == len(y) == len(z):
        raise ValueError('x, y and z must have the same length')
    out = np.full(len(x), outsideCode, dtype=np.uint8)
    nz, ny, nx = codes.shape
    if len(x) == 0 or codes.size == 0:
        return out
    z0, y0, x0 = (float(v) for v in origin)
    with np.errstate(invalid='ignore'):
        plane = np.rint(z) - z0
        row = np.rint(y) - y0
        col = np.rint(x) - x0
        # NaN fails every comparison; +-inf fails one bound
        inside = ((plane >= 0) & (plane < nz) & (row >= 0) & (row < ny)
                  & (col >= 0) & (col < nx))
    out[inside] = codes[plane[inside].astype(np.int64),
                        row[inside].astype(np.int64),
                        col[inside].astype(np.int64)]
    return out


def strata_of_barcodes(barcodes, directory: str, fov: int,
                       metadata: Optional[Dict] = None) -> np.ndarray:
    """uint8 code of each row of a barcode table (columns x, y, z), read
    from the fov's volume in directory."""
    if metadata is None:
        metadata = read_metadata(directory)
    codes, origin = read_volume(directory, fov, len(metadata['strata']))
    return lookup(barcodes['x'].to_numpy(), barcodes['y'].to_numpy(),
                  barcodes['z'].to_numpy(), codes, origin,
                  int(metadata['outside_code']))
