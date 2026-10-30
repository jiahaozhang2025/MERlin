import pandas
import numpy as np

from merlin.core import analysistask
from merlin.util import decoding
from merlin.util import spatialfeature

class PartitionBarcodes(analysistask.ParallelAnalysisTask):

    """
    An analysis task that assigns RNAs and sequential signals to cells
    based on the boundaries determined during the segment task.
    """

    outputGroup = 'Export'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        if 'boundary_correction_buffer_size' not in self.parameters:
            self.parameters['boundary_correction_buffer_size'] = 0.5

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['filter_task'],
                self.parameters['assignment_task'],
                self.parameters['alignment_task']]

    def get_partitioned_barcodes(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the cell by barcode matrixes calculated from this
        analysis task.

        Args:
            fov: the fov to get the barcode table for. If not specified, the
                combined table for all fovs are returned.

        Returns:
            A pandas data frame containing the parsed barcode information.
        """
        if fov is None:
            return pandas.concat(
                [self.get_partitioned_barcodes(fov)
                 for fov in self.dataSet.get_fovs()]
            )

        counts = self.dataSet.load_dataframe_from_csv(
            'counts_per_cell', self.get_analysis_name(), fov, index_col=0)
        # cell ids (128-bit integers) as str under every pandas version
        counts.index = counts.index.astype(str)
        return counts

    def _decoded_region(self, fov, filterTask, alignTask):
        """Global (xmin, ymin, xmax, ymax) of the region of fov whose barcodes
        its decode kept. Image space: the crop bounds (only those pixels are
        decoded). Barcode space: the barcode filter, crop_width only -- the
        whole frame is decoded and the adaptive margin removes no barcodes,
        so the crop bounds would undercount the region wherever a shift
        exceeds crop_width. Taken from the decode's crop record when it has
        one, else from its parameters."""
        decodeTask = self.dataSet.load_analysis_task(
            filterTask.parameters['decode_task'])
        record = decodeTask.get_crop_record(fov) \
            if hasattr(decodeTask, 'get_crop_record') else None
        if record is not None:
            if record.get('barcode_filter') is not None:
                rowStart, rowEnd, colStart, colEnd = record['barcode_filter']
            else:
                rowStart, rowEnd, colStart, colEnd = record['crop_bounds']
        elif not decodeTask.parameters.get('crop_in_image_space', True):
            top, bottom, left, right = decoding.crop_widths(
                decodeTask.parameters['crop_width'])
            rows, cols = self.dataSet.get_image_dimensions()
            rowStart, rowEnd, colStart, colEnd = \
                top, rows - bottom, left, cols - right
        else:
            rowStart, rowEnd, colStart, colEnd = decodeTask._crop_bounds(fov)
        x0, y0 = alignTask.fov_coordinates_to_global(fov, (colStart, rowStart))
        x1, y1 = alignTask.fov_coordinates_to_global(fov, (colEnd, rowEnd))
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def _pooled_barcodes(self, fragmentIndex, fovIntersections, filterTask,
                         alignTask) -> pandas.DataFrame:
        """Barcodes of this fov plus those of overlapping fovs, each location
        taken from exactly one fov.

        Overlapping fovs decode the same molecules in their shared strip;
        pooling every overlapping fov's barcodes (as before) counted those
        molecules once per fov in cells there. This fov's own barcodes are all
        kept; a neighbour's are kept only outside the decoded regions of this
        fov and of the neighbours already taken (ascending fov order), so its
        barcodes only fill in what was not already covered. `fovs` (optional)
        restricts the neighbours to that list.
        """
        bcDB = filterTask.get_barcode_database()
        fovs = self.parameters.get('fovs')
        allowed = None if fovs is None else set(int(f) for f in fovs)
        order = [fragmentIndex] + [f for f in fovIntersections
                                   if f != fragmentIndex
                                   and (allowed is None or f in allowed)]
        covered, parts = [], []
        for fi in order:
            partialBC = bcDB.get_barcodes(fi)
            if len(partialBC) and covered:
                gx = partialBC['global_x'].to_numpy(float)
                gy = partialBC['global_y'].to_numpy(float)
                outside = np.ones(len(partialBC), bool)
                for xmin, ymin, xmax, ymax in covered:
                    outside &= ~((gx >= xmin) & (gx <= xmax)
                                 & (gy >= ymin) & (gy <= ymax))
                partialBC = partialBC[outside]
            parts.append(partialBC)
            covered.append(self._decoded_region(fi, filterTask, alignTask))
        return pandas.concat(parts, axis=0).reset_index().copy(deep=True)

    def _run_analysis(self, fragmentIndex):
        fovs = self.parameters.get('fovs')
        if fovs is not None and fragmentIndex not in set(int(f) for f in fovs):
            # an excluded fov: no cells, nothing counted
            filterTask = self.dataSet.load_analysis_task(
                self.parameters['filter_task'])
            codebook = filterTask.get_codebook()
            empty = pandas.DataFrame(
                columns=[codebook.get_name_for_barcode_index(x)
                         for x in range(codebook.get_barcode_count())])
            self.dataSet.save_dataframe_to_csv(
                empty, 'counts_per_cell', self.get_analysis_name(),
                fragmentIndex)
            return
        if self.parameters['boundary_correction_buffer_size'] == 0:
            self._run_analysis_traditional(fragmentIndex)
        else:
            self._run_analysis_boundary_correction(fragmentIndex)

    def _run_analysis_boundary_correction(self, fragmentIndex):
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_task'])
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])
        alignTask = self.dataSet.load_analysis_task(
            self.parameters['alignment_task'])

        fovBoxes = alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])

        codebook = filterTask.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        currentFOVBarcodes = self._pooled_barcodes(
            fragmentIndex, fovIntersections, filterTask, alignTask)

        sDB = assignmentTask.get_feature_database()
        currentCells = sDB.read_features(fragmentIndex)

        print(f'FOV {fragmentIndex} has {len(currentCells)} cells.')

        countsDF = pandas.DataFrame(
            data=np.zeros((len(currentCells), barcodeCount)),
            columns=range(barcodeCount),
            index=[x.get_feature_id() for x in currentCells])

        countsDF_shrinked = pandas.DataFrame(
            data=np.zeros((len(currentCells), barcodeCount)),
            columns=range(barcodeCount),
            index=[x.get_feature_id() for x in currentCells])

        # Make a copy of the currentCells with the changed boundaries
        buffer_size = self.parameters['boundary_correction_buffer_size']
        currentCells_expanded = [c.make_a_buffered_copy(buffer_size) for c in currentCells]
        currentCells_shrinked = [c.make_a_buffered_copy(-buffer_size) for c in currentCells]

        # Store the host cells of each barcode
        N_barcodes = currentFOVBarcodes.shape[0]
        host_cells_normal = [[] for i in range(N_barcodes)]
        host_cells_shrinked = [[] for i in range(N_barcodes)]
        host_cells_expanded = [[] for i in range(N_barcodes)]
    
        # Get the host cells for each barcode
        for i in range(len(currentCells)):
            cell_normal = currentCells[i]
            cell_shrinked = currentCells_shrinked[i]
            cell_expanded = currentCells_expanded[i]

            contained_normal = cell_normal.contains_positions(currentFOVBarcodes.loc[:,
                                                ['global_x', 'global_y',
                                                 'z']].values)
            
            contained_shrinked = cell_shrinked.contains_positions(currentFOVBarcodes.loc[:,
                                                ['global_x', 'global_y',
                                                 'z']].values)
            
            contained_expanded = cell_expanded.contains_positions(currentFOVBarcodes.loc[:,
                                                ['global_x', 'global_y',
                                                 'z']].values)
           
            ids_normal = np.where(contained_normal)[0]
            ids_shrinked = np.where(contained_shrinked)[0]
            ids_expanded = np.where(contained_expanded)[0]

            for j in ids_normal:
                host_cells_normal[j].append(i)
            for j in ids_shrinked:
                host_cells_shrinked[j].append(i)
            for j in ids_expanded:
                host_cells_expanded[j].append(i)

            # Generate a counting matrix for the shrinked cells
            count_shrinked = currentFOVBarcodes[contained_shrinked].groupby('barcode_id').size()
            count_shrinked = count_shrinked.reindex(range(barcodeCount), fill_value=0)
            countsDF_shrinked.loc[cell_shrinked.get_feature_id(), :] = count_shrinked.values.tolist()

        # Assign each barcode to a cell
        barcode_types = list(currentFOVBarcodes['barcode_id'])

        for i in range(N_barcodes):
            # Continue if this barcode is not within any cell
            if len(host_cells_expanded[i]) == 0:
                continue

            barcode_id = barcode_types[i]
            
            # If the barcode is within the shrinked region of a cell, add it to the cell
            if len(host_cells_shrinked[i]) > 0:
                countsDF.iloc[host_cells_shrinked[i][0], barcode_id] += 1
                continue

            # If a cell contains a barcode in its expanded regions but
            # doesn't have it in the interior filter it out
            hce_filtered = [c for c in host_cells_expanded[i] 
                              if countsDF_shrinked.iloc[c, barcode_id] > 0]

            # If the barcode is within the normal region of a cell
            if len(host_cells_normal[i]) > 0:
                cell_normal_id = host_cells_normal[i][0]

                # Assign the barcode to the normal cell if the cell already 
                # have a same type barcode in its interior
                if countsDF_shrinked.iloc[cell_normal_id, barcode_id] > 0:
                    countsDF.iloc[cell_normal_id, barcode_id] += 1
                
                # If any other cell that contains the barcode in the expanded region
                # and have the same barcode type in its interior, assign the barcode 
                # to that cell
                elif len(hce_filtered) > 0:
                    hce = np.random.choice(hce_filtered)
                    countsDF.iloc[hce, barcode_id] += 1

                # If no other filtered cells contains the barcode, asign it to the normal cell
                else:
                    countsDF.iloc[cell_normal_id, barcode_id] += 1
                
                continue

            # If a cell only exist in the expanded region, assign it
            if len(hce_filtered) > 0:
                hce = np.random.choice(hce_filtered)
                countsDF.iloc[hce, barcode_id] += 1

        barcodeNames = [codebook.get_name_for_barcode_index(x)
                        for x in countsDF.columns.values.tolist()]
        countsDF.columns = barcodeNames

        self.dataSet.save_dataframe_to_csv(
                countsDF, 'counts_per_cell', self.get_analysis_name(),
                fragmentIndex)
    
    def _run_analysis_traditional(self, fragmentIndex):
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_task'])
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])
        alignTask = self.dataSet.load_analysis_task(
            self.parameters['alignment_task'])

        fovBoxes = alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])

        codebook = filterTask.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        currentFOVBarcodes = self._pooled_barcodes(
            fragmentIndex, fovIntersections, filterTask, alignTask)

        sDB = assignmentTask.get_feature_database()
        currentCells = sDB.read_features(fragmentIndex)

        countsDF = pandas.DataFrame(
            data=np.zeros((len(currentCells), barcodeCount)),
            columns=range(barcodeCount),
            index=[x.get_feature_id() for x in currentCells])

        for cell in currentCells:
            contained = cell.contains_positions(currentFOVBarcodes.loc[:,
                                                ['global_x', 'global_y',
                                                 'z']].values)
            count = currentFOVBarcodes[contained].groupby('barcode_id').size()
            count = count.reindex(range(barcodeCount), fill_value=0)
            countsDF.loc[cell.get_feature_id(), :] = count.values.tolist()

        barcodeNames = [codebook.get_name_for_barcode_index(x)
                        for x in countsDF.columns.values.tolist()]
        countsDF.columns = barcodeNames

        self.dataSet.save_dataframe_to_csv(
                countsDF, 'counts_per_cell', self.get_analysis_name(),
                fragmentIndex)


class ExportPartitionedBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that combines counts per cells data from each
    field of view into a single output file.
    """

    outputGroup = 'Export'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['partition_task']]

    def _run_analysis(self):
        pTask = self.dataSet.load_analysis_task(
                    self.parameters['partition_task'])
        parsedBarcodes = pTask.get_partitioned_barcodes()

        self.dataSet.save_dataframe_to_csv(
                    parsedBarcodes, 'barcodes_per_feature',
                    self.get_analysis_name())
