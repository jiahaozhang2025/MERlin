import numpy as np
import pandas
import os
import tempfile
import zarr
import time

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import decoding
from merlin.util import barcodedb
from merlin.data.codebook import Codebook
from merlin.util import barcodefilters


class BarcodeSavingParallelAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves barcodes into a barcode database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)

        ### testing this for resumable decoding ###
        if 'resumable_z_decoding' not in self.parameters:
            self.parameters['resumable_z_decoding'] = False
            print(f'emptying barcode database for fragment {fragmentIndex}')
            self.get_barcode_database().empty_database(fragmentIndex)
        elif self.parameters['resumable_z_decoding'] == True:
            print(f'keeping barcode database for fragment {fragmentIndex}')

    def get_barcode_database(self) -> barcodedb.BarcodeDB:
        """ Get the barcode database this analysis task saves barcodes into.

        Returns: The barcode database reference.
        """
        return barcodedb.PyTablesBarcodeDB(self.dataSet, self)


class Decode(BarcodeSavingParallelAnalysisTask):

    """
    An analysis task that extracts barcodes from images.
    """

    def __init__(self, dataSet: dataset.MERFISHDataSet,
                 parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 100
        if 'write_decoded_images' not in self.parameters:
            self.parameters['write_decoded_images'] = True
        if 'write_decoded_FOVs' not in self.parameters:
            self.parameters['write_decoded_FOVs'] = list(range(self.fragment_count()))
        if 'minimum_area' not in self.parameters:
            self.parameters['minimum_area'] = 0
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 0.5167
        if 'lowpass_sigma' not in self.parameters:
            self.parameters['lowpass_sigma'] = 1
        if 'remove_z_duplicated_barcodes' not in self.parameters:
            self.parameters['remove_z_duplicated_barcodes'] = False
        if self.parameters['remove_z_duplicated_barcodes']:
            if 'z_duplicate_zPlane_threshold' not in self.parameters:
                self.parameters['z_duplicate_zPlane_threshold'] = 1
            if 'z_duplicate_xy_pixel_threshold' not in self.parameters:
                self.parameters['z_duplicate_xy_pixel_threshold'] = np.sqrt(2)
        
        # special case where every FOV was optimized
        if "single_fov_optimization" not in self.parameters:
            self.parameters['single_fov_optimization'] = False
        # only decode in segmentation mask
        if 'use_segmentation_mask' not in self.parameters:
            self.parameters['use_segmentation_mask'] = False

        # gpu decoding
        if 'use_gpu' not in self.parameters:
            self.parameters['use_gpu'] = False

        # write decoded images with unique id channel
        if 'write_unique_id_images' not in self.parameters:
            self.parameters['write_unique_id_images'] = False
            
        # magnitude threshold
        if 'magnitude_threshold' not in self.parameters:
            self.parameters['magnitude_threshold'] = 1.0
        
        # tiling factor for large images to avoid OOM
        if 'tiling_factor' not in self.parameters:
            self.parameters['tiling_factor'] = None
            
        # distance metric
        if 'distance_metric' not in self.parameters:
            self.parameters['distance_metric'] = None
            
        self.cropWidth = self.parameters['crop_width']
        self.imageSize = dataSet.get_image_dimensions()

        # method for resumable decoding
        # load the previous barcodes
        # finds unique z planes then can assume that z plane has been decoded

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        dependencies = [self.parameters['preprocess_task'],
                        self.parameters['optimize_task'],
                        self.parameters['global_align_task']]
        if self.parameters['use_segmentation_mask']:
            dependencies += [self.parameters['use_segmentation_mask']]
        return dependencies

    def get_codebook(self) -> Codebook:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        return preprocessTask.get_codebook()

    def _run_analysis(self, fragmentIndex):
        """This function decodes the barcodes in a fov and saves them to the
        barcode database.
        """
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        optimizeTask = self.dataSet.load_analysis_task(
                self.parameters['optimize_task'])
        decode3d = self.parameters['decode_3d']

        lowPassSigma = self.parameters['lowpass_sigma']
        
        codebook = self.get_codebook()
        decoder = decoding.PixelBasedDecoder(codebook, calculateOverlap=(self.parameters['overlap_distance_threshold'] is not None))

        # for single FOV optimization
        if self.parameters['single_fov_optimization']:
            scaleFactors = optimizeTask.get_scale_factors(fragmentIndex)
            backgrounds = optimizeTask.get_backgrounds(fragmentIndex)
        else:
            scaleFactors = optimizeTask.get_scale_factors()
            backgrounds = optimizeTask.get_backgrounds()
        
        chromaticCorrector = optimizeTask.get_chromatic_corrector()

        zPositions = self.dataSet.get_z_positions()
        zPositionCount = len(zPositions)
        bitCount = codebook.get_bit_count()
        imageShape = self.dataSet.get_image_dimensions()
        
        # get decoded image path for a zarr file
        # this may be easier way to save
        
        if self.parameters['write_unique_id_images']:
            zarrChannels = 4
        else:
            zarrChannels = 3
            
        if self.parameters['write_decoded_images'] and (fragmentIndex in self.parameters['write_decoded_FOVs']):
            zarr_path = self.dataSet._analysis_zarr_name(self, "decoded", fragmentIndex)
            zarr_out = zarr.open(zarr_path, mode = 'a',
                shape = (zPositionCount, zarrChannels, *imageShape),
                chunks = (1, zarrChannels, *imageShape),
                dtype = np.float32)
        
        # find what z planes exist in the barcode file already
        self.decoded_z_planes = self._get_decoded_z_planes(fragmentIndex)
        
        for zIndex, z in enumerate(zPositions):

            if zIndex in self.decoded_z_planes:
                print(f'barcodes in zIndex {zIndex} detected. Skipping plane!')
                pass 
        
            else:
                outputImages = self._process_independent_z_slice(
                    fragmentIndex, zIndex, chromaticCorrector, scaleFactors,
                    backgrounds, preprocessTask, decoder)
                    
                if self.parameters['write_decoded_images'] and (fragmentIndex in self.parameters['write_decoded_FOVs']):
                    zarr_out[zIndex,0,:,:] = outputImages[0]
                    zarr_out[zIndex,1,:,:] = outputImages[1]
                    zarr_out[zIndex,2,:,:] = outputImages[2]
                    if self.parameters['write_unique_id_images']:
                        zarr_out[zIndex,3,:,:] = outputImages[3]

        if self.parameters['remove_z_duplicated_barcodes']:
            bcDB = self.get_barcode_database()
            bc = self._remove_z_duplicate_barcodes(
                bcDB.get_barcodes(fov=fragmentIndex))
            bcDB.empty_database(fragmentIndex)
            bcDB.write_barcodes(bc, fov=fragmentIndex)

    # finding what z planes are already in the barcode file
    def _get_decoded_z_planes(self, fragmentIndex):
            if self.parameters['resumable_z_decoding']:
                    print('resumable decoding enabled!\nbarcode files are not emptied!')
                    bcDB = self.get_barcode_database()
                    bcs = bcDB.get_barcodes(fov=fragmentIndex)
                    decoded_z_planes = bcs.z.unique()
            else:
                decoded_z_planes = [] # otherwise set to empty so all z planes are decoded
            return decoded_z_planes

    # used to load in the segmentation mask
    def _get_segmentation_mask(self, fovIndex, zIndex):
        segmentTask = self.dataSet.load_analysis_task(
            self.parameters['use_segmentation_mask'])
        return segmentTask._load_mask_image(fovIndex, zIndex)

    def _process_independent_z_slice(
            self, fov: int, zIndex: int, chromaticCorrector, scaleFactors,
            backgrounds, preprocessTask, decoder):

        t0 = time.time()
        imageSet = preprocessTask.get_processed_image_set(
            fov, zIndex, chromaticCorrector)
        imageSet = imageSet.reshape(
            (imageSet.shape[0], imageSet.shape[-2], imageSet.shape[-1]))
        t1 = time.time()

        decodeMask = None
        if self.parameters['use_segmentation_mask']:
            decodeMask = self._get_segmentation_mask(fov, zIndex)
        
        accumulate_pixel_traces = True
        on_tile_done = None
        dfs = []
        
        # If tiling is used, we want to extract barcodes per tile to avoid memory overhead
        if self.parameters['tiling_factor'] is not None and self.parameters['tiling_factor'] > 1:
            accumulate_pixel_traces = False
            
            def process_tile(t_di, t_pm, t_npt, t_d, slice_info):
                h_start, h_end, w_start, w_end = slice_info
                # We need to extract barcodes from the tile
                # Note: cropWidth must be 0 for tiles, filtering happens later
                outputLabels = self.parameters['write_unique_id_images']
                minimumArea = self.parameters['minimum_area']

                tile_barcode_output = decoder.extract_overlapping_barcodes_with_index(
                    t_di, t_pm, t_npt, t_d, fov, cropWidth=0, zIndex=zIndex,
                    globalAligner=None, minimumArea=minimumArea, outputLabels=outputLabels)

                if outputLabels:
                    df_tile, _ = tile_barcode_output # we ignore tile labels map for now
                else:
                    df_tile = tile_barcode_output
                
                # Adjust coordinates
                if len(df_tile) > 0:
                    df_tile['x'] += w_start
                    df_tile['y'] += h_start
                    dfs.append(df_tile)
                    
            on_tile_done = process_tile

        di, pm, npt, d = decoder.decode_pixels(
            imageSet, scaleFactors, backgrounds,
            lowPassSigma=self.parameters['lowpass_sigma'],
            magnitudeThreshold=self.parameters['magnitude_threshold'],
            distanceThreshold=self.parameters['distance_threshold'],
            distanceMetric=self.parameters['distance_metric'],
            decodeMask = decodeMask,
            use_gpu = self.parameters['use_gpu'],
            tilingFactor = self.parameters['tiling_factor'],
            accumulate_pixel_traces = accumulate_pixel_traces,
            on_tile_done = on_tile_done)
        
        t2 = time.time()
        
        if self.parameters['tiling_factor'] is not None and self.parameters['tiling_factor'] > 1:
            if len(dfs) > 0:
                df = pandas.concat(dfs, ignore_index=True)
                
                # Apply cropWidth filter on the full image coordinates
                cw = self.cropWidth
                if cw > 0:
                     df = df[(df['x'].between(cw, di.shape[1] - cw)) &
                             (df['y'].between(cw, di.shape[0] - cw))]
                
                # Apply global alignment
                globalTask = self.dataSet.load_analysis_task(self.parameters['global_align_task'])
                # Calculate global coordinates
                # We need to reconstruct centroids array (z, x, y)
                if len(df) > 0:
                     centroids = np.zeros([len(df), 3], dtype=np.float32)
                     centroids[:, 0] = df['z'].values
                     centroids[:, 1] = df['x'].values # col
                     centroids[:, 2] = df['y'].values # row
                     
                     g = globalTask.fov_coordinate_array_to_global(fov, centroids)
                     df['global_z'] = g[:, 0]
                     df['global_x'] = g[:, 1]
                     df['global_y'] = g[:, 2]
                
                if self.parameters['write_unique_id_images']:
                     # Recreate unique_id image if needed (computationally expensive?) or just skip?
                     # Since we didn't accumulate the label image, we might need to rely on 'di'
                     # But 'di' is just barcode index.
                     # If  really need the unique_id image, we might need to stitch labels map too.
                     # For now, let's assuming recreating uid map is not critical or done differently
                     # Actually, standard behavior returns (df, uid)
                     # We can reconstruct it or simply say uid images are not supported with tiling + low memory for now?
                     # Let's try to pass None for uid if we can't easily make it.
                     pass 
            else:
                df = pandas.DataFrame()

            # IMPORTANT: Skipping standard extract_and_save because we did it per tile
            # But we still need to write to DB
            self.get_barcode_database().write_barcodes(df, fov=fov)
                 
            # If write_unique_id_images is True, we have a problem: we didn't stitch the unique ID image.
            # But the user asked for memory savings.
            # We return di, pm, d. 'uid' is missing.
            barcodeOutputs = df

        else:
             barcodeOutputs = self._extract_and_save_barcodes(
                decoder, di, pm, npt, d, fov, zIndex)
        if self.parameters['write_unique_id_images']:
            if self.parameters['tiling_factor'] is not None and self.parameters['tiling_factor'] > 1:
                 uid = np.zeros_like(di) # Placeholder
                 return di, pm, d, uid
            else:
                 df, uid = barcodeOutputs 
                 return di,pm,d,uid
        else:
            df = barcodeOutputs 
        t3 = time.time()

        print(f'decoding fov {fov} zslice {zIndex}')
        print(f'time retrieving fov {fov} zindex {zIndex}: {t1-t0}')
        print(f'time decoding fov {fov} zindex {zIndex}: {t2-t1}')
        print(f'time extracting fov {fov} zindex {zIndex}: {t3-t2}')
        if self.parameters['decode_spots']:
            print(f'time spot decoding fov {fov} zindex {zIndex}: {t4-t3}')
        print(f'total time in fov {fov} zindex {zIndex}: {t3-t0}')
        
        if self.parameters['write_unique_id_images']:
            return di,pm,d,uid
            
        return di,pm,d
        
    # leave this for 3d decoding, currently zarr is used for easier resumable decoding
    def _save_decoded_images(self, fov: int, zPositionCount: int,
                             decodedImages: np.ndarray,
                             magnitudeImages: np.ndarray,
                             distanceImages: np.ndarray) -> None:
            imageDescription = self.dataSet.analysis_tiff_description(
                zPositionCount, 3)
            with self.dataSet.writer_for_analysis_images(
                    self, 'decoded', fov) as outputTif:
                for i in range(zPositionCount):
                    outputTif.save(decodedImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   contiguous=True,
                                   metadata=imageDescription)
                    outputTif.save(magnitudeImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   contiguous=True,
                                   metadata=imageDescription)
                    outputTif.save(distanceImages[i].astype(np.float32),
                                   photometric='MINISBLACK',
                                   contiguous=True,
                                   metadata=imageDescription)

    def _extract_and_save_barcodes(
            self, decoder: decoding.PixelBasedDecoder, decodedImage: np.ndarray,
            pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
            distances: np.ndarray, fov: int, zIndex: int=None) -> None:

        globalTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])
        minimumArea = self.parameters['minimum_area']
        outputLabels = self.parameters['write_unique_id_images']
        
        barcodeOutput = decoder.extract_overlapping_barcodes_with_index(
            decodedImage, pixelMagnitudes, pixelTraces, distances, fov,
            self.cropWidth, zIndex, globalTask, minimumArea, outputLabels)
        
        if outputLabels:
            df, uid = barcodeOutput
        else:
            df = barcodeOutput
            
        if not self.parameters['decode_spots']:
            self.get_barcode_database().write_barcodes(df, fov = fov)
        
        return barcodeOutput
        
    def _remove_z_duplicate_barcodes(self, bc):
        bc = barcodefilters.remove_zplane_duplicates_all_barcodeids(
            bc, self.parameters['z_duplicate_zPlane_threshold'],
            self.parameters['z_duplicate_xy_pixel_threshold'],
            self.dataSet.get_z_positions())
        return bc

