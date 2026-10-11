"""Merge the detections of one molecule across z planes, before filtering.

Decode works plane by plane, so a molecule that is in focus over several
planes is decoded once per plane. If the adaptive threshold is fit on those
per-plane detections, its misidentification target holds per detection, not
per molecule: real molecules repeat across planes far more often than blank
calls do, so after collapsing duplicates the per-molecule misidentification is
higher than the target. To make the target hold per molecule, the duplicates
are merged on the raw decoded barcodes, before the threshold is fit:

    Decode -> DeduplicateBarcodes -> GenerateMoleculeAdaptiveThreshold
           -> AdaptiveFilterBarcodes (decode_task = the DeduplicateBarcodes)

DeduplicateBarcodes writes one row per molecule and can stand in for a decode
task: it has a barcode database, a codebook, and the decode parameters the
threshold reads (distance_threshold, fovs, decode_z_index).

Linking rule. Two detections belong to the same molecule when they have the
same barcode_id, their x/y centroids are at most xy_radius_px apart, and they
are on different planes at most max_z_gap_um apart (plane depths from the
dataset's z positions). Molecules are the connected components of these links,
so one molecule can span more than max_z_gap_um in total. Two detections on
the same plane are never linked directly: they are separate components of the
decoded image.

Molecule row. The decode table's columns plus n_detections, n_planes, z_um,
z_first_um, z_last_um and area_sum, with:
- mean_intensity, min_distance, area: set by molecule_features (below);
- barcode_id, fov, cell_index, unique_id, z: those of the brightest detection
  (highest mean_intensity; ties go to the lower plane, then the earlier row).
  z stays an integer plane index so that per-plane consumers (the filtered
  image, partition by plane) see a plane on which the molecule was detected,
  with the unique_id of its component on that plane;
- x, y, global_x, global_y, global_z, z_um: centroid of the detections weighted
  by their summed magnitude (mean_intensity x area);
- max_intensity: the maximum over the detections; mean_distance and any
  intensity_<bit> columns: the pixel-weighted (area-weighted) mean;
- area_sum: the summed pixel count over the molecule's detections; z_first_um
  and z_last_um: the shallowest and deepest detection depths.

molecule_features:
- 'summed' (default): over all the molecule's detections, area = area_sum,
  mean_intensity = area-weighted mean of the detections' mean_intensity (the
  mean magnitude over all the molecule's pixels) and min_distance =
  area-weighted mean of the detections' min_distance;
- 'brightest': all three from the brightest detection (the rule
  merlin.util.barcodefilters applies after filtering).
The former name 'all_planes' is read as 'summed'. 'brightest_sum_area'
(brightest intensity and distance with the summed area) was removed and is
refused.

The decode table's area column is 16-bit unsigned; an area_sum above 65535 is
clipped to 65535 in 'area' (area_sum itself is 32-bit).

Valid box (crop_to_valid_box, off by default). Decode trims each fov to its
valid region, decode.compute_crop_bounds: crop_width on each edge, widened
(with adaptive_crop) to the widest strip that the warp task's transforms
leave without data, over the decode codebook's own data channels
(adaptive_crop_channels 'codebook', the default; 'all' = every channel) and
every plane, and at the turned corners of a rotation. Decode applies that
region only when it crops in image space (crop_in_image_space); in barcode
space it removes just the barcodes within crop_width of the frame edge, so
detections in the strip where some bit has no warped data are kept. With
crop_to_valid_box the detections whose x, y centroid lies outside the region
are dropped before linking: x in [colStart, colEnd] and y in [rowStart,
rowEnd], inclusive like Decode's crop_width filter. The region is the one
the decode recorded for the fov (Decode.get_crop_bounds), so it is per
codebook, the same bounds an image-space decode was cut to. An image-space
decode has nothing outside it to drop: its x, y are full-frame and inside the
region (an image-space decode from before x, y were written full-frame has
x, y relative to (colStart, rowStart), which are shifted back first); with
adaptive_crop off the region is the crop_width box and nothing more is
dropped either.

Summed areas run far beyond the 1..area_bins+1 area edges the adaptive
threshold uses by default (every molecule above that would share the top bin).
GenerateMoleculeAdaptiveThreshold therefore sets the area edges from the
molecules themselves (area_bin_edges 'equal_count', see
area_equal_count_edges)
before the base threshold runs; 'unit' keeps the base edges.
"""
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from merlin.analysis import decode
from merlin.analysis import filterbarcodes

MOLECULE_FEATURES = ('summed', 'brightest')
# Former molecule_features names, read as the new one so that an analysis
# json or a saved task.json that uses them still loads.
MOLECULE_FEATURE_ALIASES = {'all_planes': 'summed'}
REMOVED_MOLECULE_FEATURES = {
    'brightest_sum_area': "molecule_features 'brightest_sum_area' was removed;"
                          " use 'summed' (summed area, area-weighted mean "
                          "intensity and distance) or 'brightest'"}

# A link at exactly the radius or the gap is kept; these absorb float rounding.
XY_TOLERANCE_PX = 1e-6
Z_TOLERANCE_UM = 1e-6

# Columns added to the decode table's columns, in this order.
MOLECULE_COLUMNS = ('n_detections', 'n_planes', 'z_um', 'z_first_um',
                    'z_last_um', 'area_sum')

# Centroid columns, weighted by mean_intensity x area.
_CENTROID_COLUMNS = ('x', 'y', 'global_x', 'global_y', 'global_z')

_AREA_MAX = np.iinfo(np.uint16).max


def molecule_features_option(value: str) -> str:
    """The molecule_features name to use for value: an alias is replaced by
    its new name; a removed or unknown value raises ValueError."""
    if value in REMOVED_MOLECULE_FEATURES:
        raise ValueError(REMOVED_MOLECULE_FEATURES[value])
    value = MOLECULE_FEATURE_ALIASES.get(value, value)
    if value not in MOLECULE_FEATURES:
        raise ValueError('molecule_features must be one of %s, not %r'
                         % (MOLECULE_FEATURES, value))
    return value


def plane_depths_um(zIndex: np.ndarray, zPositions) -> np.ndarray:
    """Depth in um of each plane index (interpolated for a fractional one)."""
    zPositions = np.asarray(zPositions, dtype=np.float64)
    return np.interp(np.asarray(zIndex, dtype=np.float64),
                     np.arange(len(zPositions)), zPositions)


def link_detections(barcodeIds: np.ndarray, x: np.ndarray, y: np.ndarray,
                    zUm: np.ndarray, xyRadiusPx: float, maxZGapUm: float
                    ) -> Tuple[np.ndarray, int]:
    """Molecule label (0..n_molecules-1) of every detection.

    Detections i and j are linked when barcodeIds[i] == barcodeIds[j],
    hypot(x[i] - x[j], y[i] - y[j]) <= xyRadiusPx and
    0 < |zUm[i] - zUm[j]| <= maxZGapUm; molecules are the connected
    components. A label's value is the order of its lowest detection index,
    so the labels do not depend on how the links were found.

    Candidate pairs come from one k-d tree query in a Chebyshev box: barcode
    ids are spread apart along x by more than the x range, and z is scaled so
    that the box's z half-width is the gap; the exact radius, gap and
    different-plane conditions are applied to the candidates.

    Returns:
        (labels, number of links)
    """
    n = len(barcodeIds)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    if xyRadiusPx < 0 or maxZGapUm < 0:
        raise ValueError('xy_radius_px and max_z_gap_um must be >= 0')
    if maxZGapUm == 0:
        return np.arange(n, dtype=np.int64), 0

    ids = np.asarray(barcodeIds, dtype=np.int64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(zUm, dtype=np.float64)

    boxXY = xyRadiusPx + XY_TOLERANCE_PX
    boxZ = maxZGapUm + Z_TOLERANCE_UM
    span = float(x.max() - x.min()) + 4 * boxXY + 1.0
    points = np.column_stack(((x - x.min()) + (ids - ids.min()) * span,
                              y, z * (boxXY / boxZ)))
    tree = cKDTree(points, leafsize=32, balanced_tree=False,
                   compact_nodes=False)
    pairs = tree.query_pairs(boxXY * (1 + 1e-9), p=np.inf,
                             output_type='ndarray')
    del tree, points
    if len(pairs) == 0:
        return np.arange(n, dtype=np.int64), 0

    i, j = pairs[:, 0], pairs[:, 1]
    dz = np.abs(z[i] - z[j])
    keep = ((ids[i] == ids[j])
            & ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 <= boxXY ** 2)
            & (dz > Z_TOLERANCE_UM) & (dz <= boxZ))
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return np.arange(n, dtype=np.int64), 0
    graph = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)),
                       shape=(n, n)).tocsr()
    _, labels = connected_components(graph, directed=False)
    return _labels_by_first_member(labels), int(len(i))


def _labels_by_first_member(labels: np.ndarray) -> np.ndarray:
    """Renumber labels in the order of each component's lowest index."""
    labels = np.asarray(labels, dtype=np.int64)
    # Already so (scipy labels an undirected graph node by node) when the
    # running maximum starts at 0 and never steps by more than 1.
    runningMax = np.maximum.accumulate(labels)
    if labels[0] == 0 and np.all(np.diff(runningMax) <= 1):
        return labels
    _, first, inverse = np.unique(labels, return_index=True,
                                  return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind='stable')] = np.arange(len(first))
    return rank[inverse.ravel()]


def merge_molecules(barcodes: pandas.DataFrame, labels: np.ndarray,
                    zUm: np.ndarray, moleculeFeatures: str
                    ) -> pandas.DataFrame:
    """One row per label; see the module docstring for every column."""
    moleculeFeatures = molecule_features_option(moleculeFeatures)
    barcodes = barcodes.reset_index(drop=True)
    n = len(barcodes)
    if n == 0:
        out = barcodes.copy()
        for column in MOLECULE_COLUMNS:
            out[column] = np.array([], dtype=_molecule_dtype(column))
        return out

    labels = np.asarray(labels, dtype=np.int64)
    count = int(labels.max()) + 1
    zUm = np.asarray(zUm, dtype=np.float64)
    meanIntensity = barcodes['mean_intensity'].to_numpy(np.float64)
    minDistance = barcodes['min_distance'].to_numpy(np.float64)
    area = barcodes['area'].to_numpy(np.float64)
    planes = np.rint(barcodes['z'].to_numpy(np.float64)).astype(np.int64)

    # brightest detection of each molecule: by label, then mean_intensity
    # descending, then plane, then row (lexsort is stable)
    order = np.lexsort((planes, -meanIntensity, labels))
    sortedLabels = labels[order]
    starts = np.flatnonzero(np.r_[True, sortedLabels[1:] != sortedLabels[:-1]])
    if len(starts) != count:
        raise ValueError('labels must be 0..k-1 with every value used')
    brightest = order[starts]

    out = barcodes.iloc[brightest].reset_index(drop=True)

    def per_molecule_sum(values):
        return np.bincount(labels, weights=values, minlength=count)

    areaSum = per_molecule_sum(area)
    weight = meanIntensity * area
    weightSum = per_molecule_sum(weight)
    # a molecule whose detections all have zero magnitude: pixel weights
    useArea = ~(weightSum > 0)
    if np.any(useArea):
        weight = np.where(useArea[labels], area, weight)
        weightSum = per_molecule_sum(weight)

    def weighted(values, w, total):
        return per_molecule_sum(np.asarray(values, dtype=np.float64) * w) \
            / total

    for column in _CENTROID_COLUMNS:
        if column in out.columns:
            out[column] = weighted(barcodes[column].to_numpy(np.float64),
                                   weight, weightSum).astype(
                barcodes[column].dtype)
    if 'max_intensity' in out.columns:
        out['max_intensity'] = np.maximum.reduceat(
            barcodes['max_intensity'].to_numpy()[order], starts)
    pixelMeanColumns = ['mean_distance'] + [
        c for c in barcodes.columns if c.startswith('intensity_')]
    for column in pixelMeanColumns:
        if column in out.columns:
            out[column] = weighted(barcodes[column].to_numpy(np.float64),
                                   area, areaSum).astype(
                barcodes[column].dtype)

    if moleculeFeatures == 'summed':
        out['mean_intensity'] = weighted(meanIntensity, area, areaSum).astype(
            barcodes['mean_intensity'].dtype)
        out['min_distance'] = weighted(minDistance, area, areaSum).astype(
            barcodes['min_distance'].dtype)
        out['area'] = np.minimum(areaSum, _AREA_MAX).astype(
            _summed_area_dtype(barcodes['area'].dtype))

    zSorted = zUm[order]
    # distinct (molecule, plane) pairs; a sort, as np.unique is several times
    # slower on millions of int64 keys
    planeBase = int(planes.max() - planes.min()) + 1
    planeKeys = np.sort(labels * planeBase + (planes - planes.min()))
    planeKeys = planeKeys[np.r_[True, planeKeys[1:] != planeKeys[:-1]]]
    out['n_detections'] = np.bincount(labels, minlength=count).astype(
        _molecule_dtype('n_detections'))
    out['n_planes'] = np.bincount(planeKeys // planeBase,
                                  minlength=count).astype(
        _molecule_dtype('n_planes'))
    out['z_um'] = weighted(zUm, weight, weightSum).astype(np.float32)
    out['z_first_um'] = np.minimum.reduceat(zSorted, starts).astype(np.float32)
    out['z_last_um'] = np.maximum.reduceat(zSorted, starts).astype(np.float32)
    out['area_sum'] = areaSum.astype(_molecule_dtype('area_sum'))
    return out


def _summed_area_dtype(dtype):
    """The input's area dtype if it can hold 65535, else uint16 (the barcode
    database's area type)."""
    dtype = np.dtype(dtype)
    if dtype.kind == 'f' or (dtype.kind in 'iu'
                             and np.iinfo(dtype).max >= _AREA_MAX):
        return dtype
    return np.dtype(np.uint16)


def _molecule_dtype(column: str):
    return {'n_detections': np.int32, 'n_planes': np.int16,
            'z_um': np.float32, 'z_first_um': np.float32,
            'z_last_um': np.float32, 'area_sum': np.int32}[column]


def deduplicate_barcodes(barcodes: pandas.DataFrame, zPositions,
                         xyRadiusPx: float, maxZGapUm: float,
                         moleculeFeatures: str
                         ) -> Tuple[pandas.DataFrame, Dict]:
    """Molecule table of one fov's decoded barcodes, and link statistics."""
    barcodes = barcodes.reset_index(drop=True)
    if len(barcodes) == 0:
        return merge_molecules(barcodes, np.zeros(0, np.int64),
                               np.zeros(0), moleculeFeatures), {
            'n_detections': 0, 'n_molecules': 0, 'n_links': 0}
    zUm = plane_depths_um(barcodes['z'].to_numpy(), zPositions)
    labels, links = link_detections(
        barcodes['barcode_id'].to_numpy(), barcodes['x'].to_numpy(),
        barcodes['y'].to_numpy(), zUm, xyRadiusPx, maxZGapUm)
    molecules = merge_molecules(barcodes, labels, zUm, moleculeFeatures)
    return molecules, {'n_detections': int(len(barcodes)),
                       'n_molecules': int(len(molecules)),
                       'n_links': links}


# The decode task parameters crop_to_valid_box reads.
_VALID_BOX_PARAMETERS = ('preprocess_task', 'crop_width', 'adaptive_crop',
                         'crop_in_image_space')


def _check_valid_box_parameters(decodeTask) -> Dict:
    """The decode task's parameters, if it has those crop_to_valid_box
    reads."""
    parameters = decodeTask.get_parameters()
    missing = [k for k in _VALID_BOX_PARAMETERS if k not in parameters]
    if missing:
        raise ValueError(
            'crop_to_valid_box needs a decode task with %s; %s has no %s'
            % (', '.join(_VALID_BOX_PARAMETERS),
               decodeTask.get_analysis_name(), ', '.join(missing)))
    return parameters


def decode_valid_box(dataSet, decodeTask, fov: int,
                     legacyImageSpace: bool = False
                     ) -> Tuple[int, int, int, int]:
    """(rowStart, rowEnd, colStart, colEnd) of the decode task's valid region
    of fov, as Decode itself computes it: a Decode's recorded bounds of the
    fov (Decode.get_crop_bounds; the current settings if the fov has no
    record), otherwise decode.task_crop_bounds, i.e. compute_crop_bounds
    with the warp task of the decode task's preprocess_task, its crop_width,
    adaptive_crop and adaptive_crop_channels (all channels without it).

    legacyImageSpace (decode_xy_is_cropped): the region an image-space
    decode from before the per-codebook crop was cut to, over every channel
    of the warp task, whose corner its x, y are relative to."""
    parameters = _check_valid_box_parameters(decodeTask)
    if legacyImageSpace:
        warpTaskName = dataSet.load_analysis_task(
            parameters['preprocess_task']).get_parameters()['warp_task']
        return tuple(int(b) for b in decode.compute_crop_bounds(
            dataSet, warpTaskName, fov, parameters['crop_width'],
            parameters['adaptive_crop']))
    getBounds = getattr(decodeTask, 'get_crop_bounds', None)
    if getBounds is not None:
        return tuple(int(b) for b in getBounds(fov))
    return tuple(int(b) for b in decode.task_crop_bounds(decodeTask, fov))


def decode_xy_is_cropped(decodeTask, fov: int) -> bool:
    """True when the decode task's x, y of fov are relative to its crop
    corner (colStart, rowStart) rather than full-frame: an image-space
    decode that wrote no full-frame crop record, i.e. ran before Decode
    wrote x, y in the full frame."""
    if not decodeTask.get_parameters().get('crop_in_image_space', False):
        return False
    getRecord = getattr(decodeTask, 'get_crop_record', None)
    record = getRecord(fov) if getRecord is not None else None
    return record is None or record.get('xy_frame') != 'full_frame'


def valid_box_mask(barcodes: pandas.DataFrame,
                   bounds: Tuple[int, int, int, int],
                   cropInImageSpace: bool) -> np.ndarray:
    """True for the detections whose x, y centroid lies in bounds
    (rowStart, rowEnd, colStart, colEnd): colStart <= x <= colEnd and
    rowStart <= y <= rowEnd. With cropInImageSpace, x and y are relative to
    (colStart, rowStart), the corner of the image the decode cropped (an
    image-space decode from before x, y were written full-frame, see
    decode_xy_is_cropped), and are shifted back to the full frame first."""
    rowStart, rowEnd, colStart, colEnd = (float(b) for b in bounds)
    x = barcodes['x'].to_numpy(np.float64)
    y = barcodes['y'].to_numpy(np.float64)
    if cropInImageSpace:
        x = x + colStart
        y = y + rowStart
    return (x >= colStart) & (x <= colEnd) & (y >= rowStart) & (y <= rowEnd)


def area_equal_count_edges(areas: np.ndarray, binCount: int) -> np.ndarray:
    """Integer area bin edges with about equal numbers of molecules per bin.

    Areas are integers and one area value is never split between bins.
    Walking up from the smallest area, a bin is closed as soon as it holds
    its share of the molecules not yet binned (their count over the bins
    left), so an area value that alone holds more than its share gets a bin of
    its own and the remaining bins are shared out over the rest. This gives
    binCount bins unless there are fewer distinct areas (then one bin per
    area). A bin [e_k, e_k+1) holds the areas e_k .. e_k+1 - 1; the first
    edge is the smallest area and the last the largest area + 1. Areas outside
    that range (in fovs not sampled) go to the end bins, as with the base
    threshold.
    """
    areas = np.asarray(areas, dtype=np.float64)
    areas = np.floor(areas[np.isfinite(areas)]).astype(np.int64)
    if areas.size == 0:
        return np.arange(1, binCount + 2, dtype=np.float64)
    low = int(areas.min())
    counts = np.bincount(areas - low)
    values = np.flatnonzero(counts)
    counts = counts[values]
    edges = [int(values[0])]
    binsLeft, moleculesLeft, inBin = int(binCount), int(counts.sum()), 0
    for k in range(len(values) - 1):
        inBin += int(counts[k])
        if binsLeft > 1 and inBin * binsLeft >= moleculesLeft:
            edges.append(int(values[k + 1]))
            moleculesLeft -= inBin
            binsLeft -= 1
            inBin = 0
    edges.append(int(values[-1]) + 1)
    return np.asarray(edges, dtype=np.float64) + low


class DeduplicateBarcodes(decode.BarcodeSavingParallelAnalysisTask):

    """Merge each molecule's per-plane detections into one barcode row.

    Reads a decode task's barcodes per fov and writes one row per molecule;
    see the module docstring for the linking rule and the columns. Use it as
    the decode_task of GenerateMoleculeAdaptiveThreshold and
    AdaptiveFilterBarcodes so the misidentification target holds per
    molecule.

    Parameters:
        decode_task: the task whose barcodes are merged (required).
        xy_radius_px (2.0): largest x/y distance between linked detections.
        max_z_gap_um (0.75): largest depth difference between linked
            detections; 0 links nothing (every detection is its own
            molecule, with the molecule columns added).
        molecule_features ('summed'): 'summed' or 'brightest', see the
            module docstring ('all_planes' is read as 'summed').
        crop_to_valid_box (False): drop the detections outside the decode
            task's valid region before linking, see the module docstring.
            Needs a decode task with preprocess_task, crop_width,
            adaptive_crop and crop_in_image_space (Decode has them).
        distance_threshold, fovs, decode_z_index: the decode task's values
            unless given; GenerateAdaptiveThreshold reads them from its
            decode task.
    A per-fov 'deduplicate_summary' json (get_summary) holds the counts:
    n_decoded (rows read), n_detections (rows linked, after the valid-box
    crop), n_molecules, n_links, coding and blank detections per molecule
    and the n_planes histograms; with crop_to_valid_box also valid_box
    (rowStart, rowEnd, colStart, colEnd) and the detections dropped outside
    it (n_outside_valid_box, coding_ and blank_outside_valid_box); and the
    read / crop / link + merge / write times.
    """

    outputGroup = 'Decode'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'decode_task' not in self.parameters:
            raise ValueError('DeduplicateBarcodes needs decode_task')
        if 'xy_radius_px' not in self.parameters:
            self.parameters['xy_radius_px'] = 2.0
        if 'max_z_gap_um' not in self.parameters:
            self.parameters['max_z_gap_um'] = 0.75
        if 'molecule_features' not in self.parameters:
            self.parameters['molecule_features'] = 'summed'
        self.parameters['molecule_features'] = molecule_features_option(
            self.parameters['molecule_features'])
        if 'crop_to_valid_box' not in self.parameters:
            self.parameters['crop_to_valid_box'] = False
        if self.parameters['crop_to_valid_box'] not in (True, False):
            raise ValueError('crop_to_valid_box must be true or false')
        if 'resumable_z_decoding' not in self.parameters:
            # never resumable: a rerun empties the fov's table first
            self.parameters['resumable_z_decoding'] = False

        # What the adaptive threshold reads from its decode task: the distance
        # bins' range, and which fovs / planes the decode covers. They are the
        # decode task's values unless set here; reading them needs the decode
        # task to be saved first (earlier in the analysis json).
        inherited = ('distance_threshold', 'fovs', 'decode_z_index')
        if any(key not in self.parameters for key in inherited):
            decodeParameters = self._decode_task().get_parameters()
            for key in inherited:
                if key not in self.parameters:
                    self.parameters[key] = decodeParameters.get(key)
        if self.parameters['crop_to_valid_box']:
            _check_valid_box_parameters(self._decode_task())

    def _decode_task(self):
        if getattr(self, '_decodeTask', None) is None:
            self._decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
        return self._decodeTask

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 8000

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['decode_task']]

    def get_codebook(self):
        return self._decode_task().get_codebook()

    def get_decoded_image(self, fov: int, zIndex: int,
                          channel: str = 'barcodes'):
        """The decode task's decoded image (for the filtered-image output)."""
        return self._decode_task().get_decoded_image(fov, zIndex, channel)

    def get_decoded_image_z(self, fov: int):
        return self._decode_task().get_decoded_image_z(fov)

    def get_summary(self, fov: int) -> Dict:
        """Counts and timings saved by the fov's run."""
        return self.dataSet.load_json_analysis_result(
            'deduplicate_summary', self, resultIndex=fov)

    def _run_analysis(self, fragmentIndex):
        decodeTask = self._decode_task()
        start = time.time()
        barcodes = decodeTask.get_barcode_database().get_barcodes(
            fragmentIndex)
        readSeconds = time.time() - start
        decodedCount = len(barcodes)

        boxSummary = {}
        if self.parameters['crop_to_valid_box']:
            start = time.time()
            legacy = decode_xy_is_cropped(decodeTask, fragmentIndex)
            bounds = decode_valid_box(self.dataSet, decodeTask, fragmentIndex,
                                      legacy)
            inside = valid_box_mask(barcodes, bounds, legacy)
            boxSummary = _outside_box_counts(barcodes, inside,
                                             decodeTask.get_codebook())
            boxSummary['valid_box'] = [int(b) for b in bounds]
            if not inside.all():
                barcodes = barcodes.loc[inside]
            boxSummary['seconds_crop'] = round(time.time() - start, 3)

        start = time.time()
        molecules, stats = deduplicate_barcodes(
            barcodes, self.dataSet.get_z_positions(),
            float(self.parameters['xy_radius_px']),
            float(self.parameters['max_z_gap_um']),
            self.parameters['molecule_features'])
        mergeSeconds = time.time() - start

        start = time.time()
        self.get_barcode_database().write_barcodes(molecules,
                                                   fov=fragmentIndex)
        writeSeconds = time.time() - start

        summary = dict(stats, fov=int(fragmentIndex),
                       n_decoded=int(decodedCount),
                       xy_radius_px=float(self.parameters['xy_radius_px']),
                       max_z_gap_um=float(self.parameters['max_z_gap_um']),
                       molecule_features=self.parameters['molecule_features'],
                       crop_to_valid_box=bool(
                           self.parameters['crop_to_valid_box']),
                       seconds_read=round(readSeconds, 3),
                       seconds_link_merge=round(mergeSeconds, 3),
                       seconds_write=round(writeSeconds, 3))
        summary.update(boxSummary)
        summary.update(_planes_per_molecule(molecules,
                                            decodeTask.get_codebook()))
        self.dataSet.save_json_analysis_result(
            summary, 'deduplicate_summary', self, resultIndex=fragmentIndex)
        dropped = ''
        if boxSummary:
            dropped = ' (%d of %d outside the valid box %s dropped)' % (
                boxSummary['n_outside_valid_box'], decodedCount,
                boxSummary['valid_box'])
        print('%s fov %d: %d detections%s -> %d molecules (%d links); read '
              '%.1f s, link + merge %.1f s, write %.1f s'
              % (self.get_analysis_name(), fragmentIndex,
                 stats['n_detections'], dropped, stats['n_molecules'],
                 stats['n_links'], readSeconds, mergeSeconds, writeSeconds),
              flush=True)


def _outside_box_counts(barcodes: pandas.DataFrame, inside: np.ndarray,
                        codebook) -> Dict:
    """Detections outside the valid box: all, coding and blank."""
    outside = ~np.asarray(inside, dtype=bool)
    blank = barcodes['barcode_id'].isin(
        codebook.get_blank_indexes()).to_numpy()
    return {'n_outside_valid_box': int(outside.sum()),
            'coding_outside_valid_box': int((outside & ~blank).sum()),
            'blank_outside_valid_box': int((outside & blank).sum())}


def _planes_per_molecule(molecules: pandas.DataFrame, codebook) -> Dict:
    """Mean detections per molecule and the n_planes histogram, coding and
    blank separately (blank calls repeat across planes far less often)."""
    result = {}
    if len(molecules) == 0:
        return result
    blank = molecules['barcode_id'].isin(codebook.get_blank_indexes())
    for name, mask in (('coding', ~blank), ('blank', blank)):
        part = molecules.loc[mask]
        result['%s_molecules' % name] = int(len(part))
        result['%s_detections' % name] = int(part['n_detections'].sum())
        counts = np.bincount(part['n_planes'].to_numpy(np.int64))
        result['%s_n_planes_histogram' % name] = [int(c) for c in counts]
    return result


class GenerateMoleculeAdaptiveThreshold(
        filterbarcodes.GenerateAdaptiveThreshold):

    """GenerateAdaptiveThreshold with area edges set from the molecules.

    area_bin_edges:
    - 'equal_count' (default): before the histogram is built, the edges are
      area_equal_count_edges of the areas of all molecules (coding and blank)
      in bin_sample_fovs fovs (default 20) spread evenly over the selected
      fovs: area_bins bins of about equal molecule counts, a heavily tied
      small area getting a bin of its own. Saved as the task's area_bins
      result, which the base task then uses (and reuses on a rerun).
    - 'unit': the base edges 1, 2, ..., area_bins + 1 (areas above pooled in
      the top bin).
    Everything else is the base task's.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        if 'area_bin_edges' not in self.parameters:
            self.parameters['area_bin_edges'] = 'equal_count'
        if self.parameters['area_bin_edges'] not in ('equal_count', 'unit'):
            raise ValueError("area_bin_edges must be 'equal_count' or 'unit'")

    def _area_sample_fragments(self) -> List[int]:
        selected = self._selected_fragments()
        requested = self.parameters['bin_sample_fovs']
        target = max(1, min(20 if requested is None else int(requested),
                            len(selected)))
        positions = np.unique(np.rint(
            np.linspace(0, len(selected) - 1, target)).astype(int))
        return [selected[p] for p in positions]

    def _run_analysis(self):
        if self.parameters['area_bin_edges'] == 'equal_count' and \
                self.dataSet.load_numpy_analysis_result_if_available(
                    'area_bins', self, None) is None:
            decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
            sample = self._area_sample_fragments()
            while not all(decodeTask.is_complete(f) for f in sample):
                time.sleep(self.parameters['poll_interval'])
            barcodeDB = decodeTask.get_barcode_database()
            areas = np.concatenate([
                barcodeDB.get_barcodes(f, columnList=['area'])[
                    'area'].to_numpy(np.float64) for f in sample])
            edges = area_equal_count_edges(areas,
                                           self.parameters['area_bins'])
            self.dataSet.save_numpy_analysis_result(edges, 'area_bins', self)
            print('%s: %d area bins from %d molecules in fovs %s: edges %s'
                  % (self.get_analysis_name(), len(edges) - 1, len(areas),
                     sample, edges.astype(int).tolist()), flush=True)
        super()._run_analysis()
