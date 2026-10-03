import os
import cv2
import numpy as np
from skimage import measure
from skimage import segmentation
from skimage import exposure
from skimage import transform
import rtree
from shapely import geometry
from typing import List, Dict, Tuple
from scipy.spatial import cKDTree
import scipy.ndimage
import cellpose.models
import importlib.metadata
import tifffile

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed
import pandas
import networkx as nx

# Segmentation is Cellpose-SAM only (cellpose >= 4, e.g. cpsam_v2). The
# cellpose 2/3 classes and code paths (models.Cellpose, cyto2/nuclei, CP3 user
# models) were removed on 2026-10-02; they are in the git history before then.
# The module still imports under older cellpose so its other tasks can run.
CELLPOSE_MAJOR = int(importlib.metadata.version('cellpose').split('.')[0])


def _label_footprints(masks: np.ndarray, micronsPerPixel: float) -> pandas.DataFrame:
    """Area (um2) and width (minor axis, um) of each label of a [z, y, x]
    label image, measured on the plane where the label is largest."""
    rows = []
    for region in measure.regionprops(masks):
        z = np.bincount(region.coords[:, 0]).argmax()
        _, r0, c0, _, r1, c1 = region.bbox
        plane = measure.regionprops(
            (masks[z, r0:r1, c0:c1] == region.label).astype(np.uint8))[0]
        rows.append((region.label, plane.area * micronsPerPixel ** 2,
                     plane.minor_axis_length * micronsPerPixel))
    return pandas.DataFrame(rows, columns=['label', 'area_um2', 'width_um'])


class FeatureSavingAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves features into a spatial feature
    database.
    """

    outputGroup = 'Segment'

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)
        self.get_feature_database().empty_database(fragmentIndex)

    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)


class WatershedSegment(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a watershed algorithm.
    
    Since each field of view is analyzed individually, the segmentation results
    should be cleaned in order to merge cells that cross the field of
    view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'seed_channel_name' not in self.parameters:
            self.parameters['seed_channel_name'] = 'DAPI'
        if 'watershed_channel_name' not in self.parameters:
            self.parameters['watershed_channel_name'] = 'polyT'

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        seedIndex = self.dataSet.get_data_organization().get_data_channel_index(
            self.parameters['seed_channel_name'])
        seedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                       seedIndex, 5)

        watershedIndex = self.dataSet.get_data_organization() \
            .get_data_channel_index(self.parameters['watershed_channel_name'])
        watershedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                            watershedIndex, 5)
        seeds = watershed.separate_merged_seeds(
            watershed.extract_seeds(seedImages))
        normalizedWatershed, watershedMask = watershed.prepare_watershed_images(
            watershedImages)

        seeds[np.invert(watershedMask)] = 0
        watershedOutput = segmentation.watershed(
            normalizedWatershed, measure.label(seeds), mask=watershedMask,
            connectivity=np.ones((3, 3, 3)), watershed_line=True)

        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (watershedOutput == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
            for i in np.unique(watershedOutput) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _read_and_filter_image_stack(self, fov: int, channelIndex: int,
                                     filterSigma: float) -> np.ndarray:
        filterSize = int(2*np.ceil(2*filterSigma)+1)
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        return np.array([cv2.GaussianBlur(
            warpTask.get_aligned_image(fov, channelIndex, z),
            (filterSize, filterSize), filterSigma)
            for z in range(len(self.dataSet.get_z_positions()))])


class CellPoseSegmentSingleChannel3D(FeatureSavingAnalysisTask):

    """
    Segment the cells of each field of view with Cellpose-SAM (cellpose >= 4,
    https://github.com/MouseLand/cellpose), from one channel.

    The model is `path_to_user_model` if set (a path, or a cellpose-4 model
    name such as a fine-tuned cpsam), otherwise `model_type` ('cpsam_v2').
    By default every z plane is segmented in 2D and stitched into 3D
    (`stitch_threshold`); `cellpose_3D_stitching: false` runs cellpose's 3D
    mode instead. `z_index` segments that one plane only and repeats its
    outlines on every plane, so barcodes from all planes are partitioned
    into these 2D cells. `diameter` null runs at native resolution (cpsam
    needs no diameter); a value scales the image by 30 / diameter.

    Writes the cell outlines (feature database), segmented_mask<fov>.tif
    (labels, zlib) and feature_labels_<fov>.csv (mask label -> feature id).
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'diameter' not in self.parameters:
            self.parameters['diameter'] = None
        if 'channel_name' not in self.parameters:
            self.parameters['channel_name'] = 'DAPI'
        if 'model_type' not in self.parameters:
            self.parameters['model_type'] = 'cpsam_v2'
        if 'path_to_user_model' not in self.parameters:
            self.parameters['path_to_user_model'] = False

        if 'dump_segmented_masks' not in self.parameters:
            self.parameters['dump_segmented_masks'] = True
        if 'dump_segmented_images' not in self.parameters:
            self.parameters['dump_segmented_images'] = True
        if 'dump_segmented_FOVs' not in self.parameters:
            self.parameters['dump_segmented_FOVs'] = list(range(self.fragment_count()))

        if 'use_gpu' not in self.parameters:
            self.parameters['use_gpu'] = False

        # true: segment each plane in 2D and stitch planes whose masks overlap
        # by stitch_threshold (IoU); false: cellpose's 3D mode, with z scaled
        # by anisotropy (z step / pixel size)
        if 'cellpose_3D_stitching' not in self.parameters:
            self.parameters['cellpose_3D_stitching'] = True
        if 'stitch_threshold' not in self.parameters:
            self.parameters['stitch_threshold'] = 0.4
        if 'anisotropy' not in self.parameters:
            self.parameters['anisotropy'] = 1

        # shrink the images by this factor before cellpose to save memory;
        # the masks are scaled back up
        if 'downsample_factor' not in self.parameters:
            self.parameters['downsample_factor'] = None

        # segment only this z index and put its outlines on every z plane
        if 'z_index' not in self.parameters:
            self.parameters['z_index'] = None

        if 'flow_threshold' not in self.parameters:
            self.parameters['flow_threshold'] = 0.4
        if 'cellprob_threshold' not in self.parameters:
            self.parameters['cellprob_threshold'] = 0.0

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _load_cellpose_model(self):
        if CELLPOSE_MAJOR < 4:
            raise RuntimeError(
                '%s needs cellpose >= 4 (Cellpose-SAM); this environment has '
                'cellpose %s' % (type(self).__name__,
                                 importlib.metadata.version('cellpose')))
        return cellpose.models.CellposeModel(
            gpu=self.parameters['use_gpu'],
            pretrained_model=(self.parameters['path_to_user_model']
                              or self.parameters['model_type']))

    def _eval_cellpose(self, model, images, diameter, channelAxis):
        """Masks [z, y, x] for a z-first stack, [z, y, x] or [z, y, x, c];
        channels are used in the order given."""
        kwargs = dict(diameter=diameter, z_axis=0, channel_axis=channelAxis,
                      flow_threshold=self.parameters['flow_threshold'],
                      cellprob_threshold=self.parameters['cellprob_threshold'])
        if self.parameters['cellpose_3D_stitching']:
            kwargs.update(do_3D=False,
                          stitch_threshold=self.parameters['stitch_threshold'])
        else:
            kwargs.update(do_3D=True, anisotropy=self.parameters['anisotropy'])
        # copy: cellpose normalises a float32 input in place
        masks = model.eval(np.array(images, copy=True), **kwargs)[0]
        # eval squeezes a single plane to 2D
        return masks.reshape(images.shape[:3])

    def _read_image_stack(self, fov: int, channelIndex: int) -> np.ndarray:
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        zPositions = self.dataSet.get_z_positions()
        zIndices = [self.dataSet.position_to_z_index(z) for z in zPositions]
        if self.parameters['z_index'] is not None:
            zIndices = [self.parameters['z_index']]
        return np.array([warpTask.get_aligned_image(fov, channelIndex, z)
                         for z in zIndices])

    def _segmentation_images(self, fov):
        """The images cellpose segments and their channel axis (None: one
        channel)."""
        channel = self.dataSet.get_data_organization().get_data_channel_index(
            self.parameters['channel_name'])
        return self._read_image_stack(fov, channel), None

    def _extend_to_all_planes(self, masks):
        """The one segmented plane (z_index) repeated on every z plane."""
        return np.repeat(masks.reshape((1,) + masks.shape[-2:]),
                         len(self.dataSet.get_z_positions()), axis=0)

    def _features_from_masks(self, masks, fov, transformationMatrix, zPos):
        """One feature per label of a [z, y, x] label image. Same result as
        feature_from_label_matrix(masks == label) on the whole image (~0.4 s
        per label at 2048 x 2048 x 8), computed on the label's bounding box
        plus a 1 px margin, which is all its contours depend on, and shifted
        back through the transform. Also returns the labels, in feature order."""
        rows, cols = masks.shape[1:]
        features, labels = [], []
        for region in measure.regionprops(masks):
            _, r0, c0, _, r1, c1 = region.bbox
            r0, c0 = max(r0 - 1, 0), max(c0 - 1, 0)
            r1, c1 = min(r1 + 1, rows), min(c1 + 1, cols)
            # contour points are (x, y) = (column, row)
            shift = np.array([[1, 0, c0], [0, 1, r0], [0, 0, 1]], dtype=float)
            features.append(
                spatialfeature.SpatialFeature.feature_from_label_matrix(
                    masks[:, r0:r1, c0:c1] == region.label, fov,
                    transformationMatrix @ shift, zPos))
            labels.append(region.label)
        return features, labels

    def _save_feature_labels(self, fov, features, labels):
        """feature_labels_<fov>.csv: mask label -> feature id, so later tasks
        (FilterCells) can map cells back onto segmented_mask<fov>.tif."""
        self.dataSet.save_dataframe_to_csv(
            pandas.DataFrame({'label': labels, 'feature_id': [
                str(f.get_feature_id()) for f in features]}),
            'feature_labels', self, fov, index=False)

    def _save_tiff_images(self, fov, filename_prefix, image_stack):
        '''Save a stack of images as a tiff file.'''
        with self.dataSet.writer_for_analysis_images(self, filename_prefix, fov) as outputTif:
            if len(image_stack.shape) == 3:
                # one call -> one [z, y, x] series of z pages (zlib); written
                # frame by frame, compressed pages would each be a series
                outputTif.write(image_stack,photometric='MINISBLACK',compression='zlib')
            elif len(image_stack.shape)>2:
                for frame in image_stack:
                    outputTif.write(frame,photometric='MINISBLACK',contiguous=True)
            else:
                outputTif.write(image_stack,photometric='MINISBLACK',compression='zlib')

    # reader for the segmented masks (used by Decode/Optimize with
    # use_segmentation_mask); takes a z index, not a frame or z position
    def _load_mask_image(self, fov, zIndex, filename_prefix = 'segmented_mask'):
        imagePath = self.dataSet._analysis_image_name(self, filename_prefix, fov)
        return self.dataSet.load_image(imagePath, zIndex, transform = False)

    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])
        images, channelAxis = self._segmentation_images(fragmentIndex)
        model = self._load_cellpose_model()

        diameter = self.parameters['diameter']
        factor = self.parameters['downsample_factor']
        if factor is not None:
            fullShape = images.shape
            images = transform.resize(
                images, (fullShape[0], int(fullShape[1] / factor),
                         int(fullShape[2] / factor)) + fullShape[3:],
                preserve_range=True).astype(images.dtype)
            if diameter:
                diameter = diameter / factor

        masks = self._eval_cellpose(model, images, diameter, channelAxis)

        if factor is not None:
            masks = transform.resize(masks, fullShape[:3], order=0,
                                     preserve_range=True).astype(masks.dtype)
        if self.parameters['z_index'] is not None:
            masks = self._extend_to_all_planes(masks)

        if self.parameters['dump_segmented_masks'] and fragmentIndex in self.parameters['dump_segmented_FOVs']:
            self._save_tiff_images(fragmentIndex, 'segmented_mask', masks)
        if self.parameters['dump_segmented_images'] and fragmentIndex in self.parameters['dump_segmented_FOVs']:
            self._save_tiff_images(fragmentIndex, 'segmented_images', images)

        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList, labels = self._features_from_masks(
            masks, fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
        self._save_feature_labels(fragmentIndex, featureList, labels)
        self.get_feature_database().write_features(featureList, fragmentIndex)


class CellPoseSegmentTwoChannel3D(CellPoseSegmentSingleChannel3D):

    """
    CellPoseSegmentSingleChannel3D on two channels, stacked
    [channel_1_name, channel_2_name] (e.g. polyT, DAPI) in that order.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'channel_1_name' not in self.parameters:
            self.parameters['channel_1_name'] = 'polyT'
        if 'channel_2_name' not in self.parameters:
            self.parameters['channel_2_name'] = 'DAPI'

    def _segmentation_images(self, fov):
        dataOrganization = self.dataSet.get_data_organization()
        stacks = [self._read_image_stack(
            fov, dataOrganization.get_data_channel_index(self.parameters[name]))
            for name in ('channel_1_name', 'channel_2_name')]
        return np.stack(stacks, axis=3), 3


class CleanCellBoundaries(analysistask.ParallelAnalysisTask):
    '''
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.
    '''

    outputGroup = 'Segment'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def return_exported_data(self, fragmentIndex) -> nx.Graph:
        return self.dataSet.load_graph_from_pickle(
            'cleaned_cells', self, fragmentIndex)

    def _run_analysis(self, fragmentIndex) -> None:
        allFOVs = np.array(self.dataSet.get_fovs())
        fovBoxes = self.alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])
        intersectingFOVs = list(allFOVs[np.array(fovIntersections)])

        spatialTree = rtree.index.Index()
        count = 0
        idToNum = dict()
        for currentFOV in intersectingFOVs:
            cells = self.segmentTask.get_feature_database()\
                .read_features(currentFOV)
            cells = spatialfeature.simple_clean_cells(cells)

            spatialTree, count, idToNum = spatialfeature.construct_tree(
                cells, spatialTree, count, idToNum)

        graph = nx.Graph()
        cells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        cells = spatialfeature.simple_clean_cells(cells)
        graph = spatialfeature.construct_graph(graph, cells,
                                               spatialTree, fragmentIndex,
                                               allFOVs, fovBoxes)

        self.dataSet.save_graph_as_pickle(
            graph, 'cleaned_cells', self, fragmentIndex)


class CombineCleanedBoundaries(analysistask.AnalysisTask):
    """
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.

    """

    outputGroup = 'Segment'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['cleaning_task'])

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['cleaning_task']]

    def return_exported_data(self):
        # cell ids are 128-bit integers: pandas >= 3 would parse them as int,
        # pandas 1.x as str; read them as str so they match str(feature id)
        kwargs = {'index_col': 0, 'dtype': {'cell_id': str}}
        return self.dataSet.load_dataframe_from_csv(
            'all_cleaned_cells', analysisTask=self.analysisName, **kwargs)

    def _run_analysis(self):
        allFOVs = self.dataSet.get_fovs()
        graph = nx.Graph()
        for currentFOV in allFOVs:
            subGraph = self.cleaningTask.return_exported_data(currentFOV)
            graph = nx.compose(graph, subGraph)

        cleanedCells = spatialfeature.remove_overlapping_cells(graph)

        self.dataSet.save_dataframe_to_csv(cleanedCells, 'all_cleaned_cells',
                                           analysisTask=self)


class RefineCellDatabases(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['combine_cleaning_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['combine_cleaning_task']]

    def _run_analysis(self, fragmentIndex):

        cleanedCells = self.cleaningTask.return_exported_data()
        originalCells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        featureDB = self.get_feature_database()
        cleanedC = cleanedCells[cleanedCells['originalFOV'] == fragmentIndex]
        cleanedGroups = cleanedC.groupby('assignedFOV')
        for k, g in cleanedGroups:
            cellsToConsider = set(g['cell_id'].astype(str))
            featureList = [x for x in originalCells if
                           str(x.get_feature_id()) in cellsToConsider]
            featureDB.write_features(featureList, fragmentIndex)


class FilterCells(FeatureSavingAnalysisTask):

    """Remove non-cell objects from a cell database, using the barcodes a
    partition assigned to each cell and its footprint in the segmentation
    masks. It runs after a (provisional) partition because the barcode
    criteria need assigned barcodes; partition into this task to get count
    tables of the kept cells only.

    Per fov it writes the kept cells as its feature database, cell_qc_<fov>.csv
    with one row per segmented label (measurements and status: 'kept',
    'overlap_duplicate' = dropped by assignment_task's overlap cleaning, or
    the failed criteria joined by ';'), and filtered_mask<fov>.tif, an ImageJ
    [z, c, y, x] hyperstack with c0 = segment_task's labels (before) and c1 =
    the kept cells (after), same label ids.

    Needs segment_task's dumped masks and feature_labels. Criteria, each
    optional (null = not applied):
        min_area_um2, min_width_um   footprint area and minor axis, on the
                                     label's largest plane
        min_barcodes                 coding barcodes assigned by partition_task
        min_barcode_density          those barcodes per um2 of footprint
    """

    CRITERIA = (('min_area_um2', 'area_um2', 'small'),
                ('min_width_um', 'width_um', 'thin'),
                ('min_barcodes', 'barcodes', 'few_barcodes'),
                ('min_barcode_density', 'barcode_density', 'low_density'))

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        for key, _, _ in self.CRITERIA:
            if key not in self.parameters:
                self.parameters[key] = None
        if 'write_masks' not in self.parameters:
            self.parameters['write_masks'] = True

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 4096

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['assignment_task'],
                self.parameters['partition_task']]

    def get_cell_qc(self, fov: int) -> pandas.DataFrame:
        return self.dataSet.load_dataframe_from_csv(
            'cell_qc', self, fov, dtype={'feature_id': str})

    def _run_analysis(self, fragmentIndex):
        segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])
        partitionTask = self.dataSet.load_analysis_task(
            self.parameters['partition_task'])

        masks = tifffile.imread(self.dataSet._analysis_image_name(
            segmentTask, 'segmented_mask', fragmentIndex))
        masks = masks.reshape((-1,) + masks.shape[-2:])
        qc = _label_footprints(masks, self.dataSet.get_microns_per_pixel())
        qc = qc.merge(self.dataSet.load_dataframe_from_csv(
            'feature_labels', segmentTask, fragmentIndex,
            dtype={'feature_id': str}), on='label', how='left')

        cells = {str(c.get_feature_id()): c for c in
                 assignmentTask.get_feature_database().read_features(
                     fragmentIndex)}
        counts = partitionTask.get_partitioned_barcodes(fragmentIndex)
        counts.index = counts.index.astype(str)
        codebook = self.dataSet.load_analysis_task(
            partitionTask.parameters['filter_task']).get_codebook()
        blanks = {codebook.get_name_for_barcode_index(i)
                  for i in codebook.get_blank_indexes()}
        coding = [g for g in counts.columns if g not in blanks]
        qc['barcodes'] = counts[coding].sum(axis=1).reindex(
            qc.feature_id).values
        qc['barcode_density'] = qc.barcodes / qc.area_um2

        status = []
        for row in qc.itertuples():
            if row.feature_id not in cells:
                status.append('overlap_duplicate')
                continue
            failed = [reason for key, column, reason in self.CRITERIA
                      if self.parameters[key] is not None
                      and getattr(row, column) < self.parameters[key]]
            status.append(';'.join(failed) if failed else 'kept')
        qc['status'] = status
        kept = qc.status == 'kept'

        self.get_feature_database().write_features(
            [cells[i] for i in qc.feature_id[kept]], fragmentIndex)
        self.dataSet.save_dataframe_to_csv(
            qc, 'cell_qc', self, fragmentIndex, index=False)

        if self.parameters['write_masks']:
            keep = np.zeros(masks.max() + 1, bool)
            keep[qc.label[kept].values] = True
            tifffile.imwrite(
                self.dataSet._analysis_image_name(
                    self, 'filtered_mask', fragmentIndex),
                np.stack([masks, np.where(keep[masks], masks, 0)],
                         axis=1).astype(np.uint16),
                imagej=True, compression='zlib',
                metadata={'axes': 'ZCYX', 'Labels': [
                    '%s z%d' % (c, z) for z in range(masks.shape[0])
                    for c in ('before', 'after')]})


class ExportCellMetadata(analysistask.AnalysisTask):
    """
    An analysis task exports cell metadata.
    """

    outputGroup = 'Export'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task']]

    def _run_analysis(self):
        df = self.segmentTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, 'feature_metadata',
                                           self.analysisName)
