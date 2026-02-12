"""
PolarisDecode: DeepCell Polaris Decoding for MERlin

Author: Zhiyun Lei
Date: 2025-04-23
Description: Custom decoding pipeline for MERFISH using Polaris.
"""

import os
import numpy as np
import itertools
from skimage import transform
from typing import Dict
from typing import List
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import merlin
from merlin.analysis import decode
from merlin.util import decoding
from merlin.util import registration
from merlin.util import aberration
from merlin.data.codebook import Codebook
from merlin.core import analysistask

import tensorflow as tf

from deepcell.utils.plot_utils import create_rgb_image
from deepcell.datasets import SpotNetExampleData
from deepcell_spots.applications import Polaris
from deepcell_spots.utils.results_utils import mask_spots
from deepcell_spots.dotnet_losses import DotNetLosses



class PolarisDecode(decode.BarcodeSavingParallelAnalysisTask):

    """
    An analysis task for decoding MERFISH data using DeepCell Polaris.

    This task integrates DeepCell Polaris into the MERlin pipeline for
    high-accuracy decoding of RNA barcodes in multiplexed smFISH images.
    It replaces traditional MERlin decoding methods with a deep learning-based
    spot detection and probabilistic barcode assignment.

    Key steps:
    - Loads chromatically-corrected, globally-aligned image stacks for each bit.
    - Applies XY cropping to remove noisy image borders.
    - Reformats the image for compatibility with Polaris ([z, x, y, channel]).
    - Runs Polaris to detect spots and decode barcodes.
    - Filters decoded results by probability and masking confidence.
    - Converts local pixel coordinates to global spatial coordinates.
    - Saves decoded barcodes in MERlin’s barcode database format.

    This task allows drop-in replacement of MERlin's decoding pipeline
    with Polaris, enabling improved decoding accuracy in challenging datasets.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 20
        if 'probability_threshold' not in self.parameters:
            self.parameters['probability_threshold'] = 0.95
        if 'spot_threshold' not in self.parameters:
            self.parameters['spot_threshold'] = 0.85

        os.environ["DEEPCELL_ACCESS_TOKEN"] = self.parameters['API_key']
        

    def get_estimated_memory(self):
        return 30000

    def get_estimated_time(self):
        return 600

    def get_dependencies(self):
        if 'Optimize_iteration' not in self.parameters:
            dependencies = [self.parameters['warp_task'], self.parameters['preprocess_task']]
        else:
            dependencies = [self.parameters['Optimize_iteration'],
            self.parameters['warp_task'], self.parameters['preprocess_task']]
        return dependencies

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_codebook(self) -> Codebook:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        return preprocessTask.get_codebook()

    def get_reference_color(self):
        return min(self._get_used_colors())

    def crop_xy_edges(self, image_stack: np.ndarray) -> np.ndarray:
	    """
	    Crop `crop_width` pixels from each edge of X and Y in a 4D image stack: [channel, z, x, y]

	    Args:
	        image_stack (np.ndarray): Input image stack of shape [C, Z, X, Y]

	    Returns:
	        np.ndarray: Cropped image stack of shape [C, Z, X - 2*crop_width, Y - 2*crop_width]
	    """
	    crop_width = self.parameters['crop_width']

	    if image_stack.ndim != 4:
	        raise ValueError("Expected 4D image stack with shape [channel, z, x, y]")

	    c, z, x, y = image_stack.shape
	    if crop_width * 2 >= x or crop_width * 2 >= y:
	        raise ValueError("Crop width is too large for the image dimensions")

	    return image_stack[:, :, crop_width:-crop_width, crop_width:-crop_width]

    def reformat_to_zxyc(self, image_stack: np.ndarray) -> np.ndarray:
	    """
	    Reformat image stack from [channel, z, x, y] to [z, x, y, channel]
	    """
	    if image_stack.ndim != 4:
	        raise ValueError("Expected 4D array of shape [channel, z, x, y]")
	    return np.transpose(image_stack, (1, 2, 3, 0))  # [z, x, y, channel]

    def _get_previous_chromatic_transformations(self) -> Dict[str, Dict[str, transform.SimilarityTransform]]:
        if 'Optimize_iteration' not in self.parameters:
            usedColors = self._get_used_colors()
            return {u: {v: transform.SimilarityTransform()
                        for v in usedColors if v >= u} for u in usedColors}
        else:
            previousIteration = self.dataSet.load_analysis_task(
                self.parameters['Optimize_iteration'])
            return previousIteration._get_chromatic_transformations()

    def _get_used_colors(self) -> List[str]:
        dataOrganization = self.dataSet.get_data_organization()
        codebook = self.get_codebook()
        return sorted({dataOrganization.get_data_channel_color(
            dataOrganization.get_data_channel_for_bit(x))
            for x in codebook.get_bit_names()})



    def _run_analysis(self, fragmentIndex):
    	
	    globalAligner = self.dataSet.load_analysis_task(
	    	self.parameters['global_align_task'])
	    warpTask = self.dataSet.load_analysis_task(
	    	self.parameters['warp_task'])
	    zPositionCount = len(self.dataSet.get_z_positions())
	    codebook = self.get_codebook()
	    dataOrganization = self.dataSet.get_data_organization()

	    pixel_size = self.dataSet.get_microns_per_pixel()

	    chromaticTransformations = self._get_previous_chromatic_transformations()


	    ## Reformat codebook for Polaris
	    df_barcodes = codebook.get_data().copy()
	    df_barcodes['Gene'] = df_barcodes['name']
	    df_barcodes.drop(['name', 'id'], axis=1, inplace=True)
	    cols = df_barcodes.columns.tolist()
	    cols.insert(0, cols.pop(cols.index('Gene')))
	    df_barcodes = df_barcodes[cols]


	    ## initialize Polaris class
	    # load model to avoid race condition
	    if 'model_path' in self.parameters:
	        sp_model = tf.keras.models.load_model(
	            self.parameters['model_path'],
	            custom_objects={
	            'regression_loss': DotNetLosses.regression_loss,
	            'classification_loss': DotNetLosses.classification_loss
	            }
	            )
	        multiplex_app = Polaris(
	            image_type='multiplex',
	            segmentation_type='no segmentation',
	            spots_model=sp_model,
	            decoding_kwargs={
	            'rounds': codebook.get_bit_count(),
	            'channels': 1,
	            'df_barcodes': df_barcodes
	            }
	            )
	    else:
	        multiplex_app = Polaris(
	            image_type='multiplex',
	            segmentation_type='no segmentation',
	            decoding_kwargs={
	            'rounds': codebook.get_bit_count(),
	            'channels': 1,
	            'df_barcodes': df_barcodes
	            }
	            )

	    ## Get chromatic corrected image
	    chromaticCorrector = aberration.RigidChromaticCorrector(
	        chromaticTransformations, self.get_reference_color())

	    bit_names = codebook.get_bit_names()
	    warpedImages = []

	    for bit in bit_names:
	        channel_id = dataOrganization.get_data_channel_for_bit(bit)
	        channelImg = []
	        for zIndex in range(zPositionCount):
	            aligned = warpTask.get_aligned_image(fragmentIndex, channel_id, zIndex, chromaticCorrector)
	            channelImg.append(aligned)
	        channelImg = np.array(channelImg)  # [z, x, y]
	        warpedImages.append(channelImg)

	    warpedImages = np.array(warpedImages)  # [channel, z, x, y]
	    warpedImages = self.crop_xy_edges(warpedImages)
	    warpedImages = self.reformat_to_zxyc(warpedImages)  # [z, x, y, channel]
	    # self.dataSet.save_numpy_analysis_result(
	    #     warpedImages, 'all_images',
	    #     self.analysisName, resultIndex=fragmentIndex)


	    ## Polaris decoding
	    background_image = np.max(warpedImages, axis=-1, keepdims=True)
	    mask = mask_spots(background_image, 0.99)
        
	    multiplex_pred = multiplex_app.predict(
	        spots_image=warpedImages,
	        mask=mask,
	        image_mpp=pixel_size,
	        clip=True,
	        threshold=self.parameters['spot_threshold']
	    )
	    df_decode_results = multiplex_pred[0].copy()

	    # filter barcodes
	    df_decode_results = df_decode_results[
	    (df_decode_results['probability'] >= self.parameters['probability_threshold']) &
	    (df_decode_results['masked'] == 0)
	    ]


	    ## Format output for MERlin
	    df_decode_results['cell_index'] = -1
	    df_decode_results['fov'] = fragmentIndex
	    df_decode_results.rename(columns={
	        "batch_id": "z",
	        "predicted_id": "barcode_id",
	        "predicted_name": "gene"},
	        inplace=True)
	    # Polaris codebook is 1-indexed; MERlin expects 0-indexed
	    df_decode_results["barcode_id"] = df_decode_results["barcode_id"] - 1
	    df_decode_results["barcode_id"] = df_decode_results["barcode_id"].astype(int)

	    # Convert to global coordinates
	    df_decode_results['x'] += self.parameters['crop_width']
	    df_decode_results['y'] += self.parameters['crop_width']
	    df_decode_results[['x', 'y']] = df_decode_results[['y', 'x']]
	    xyz_pixels = df_decode_results[['z', 'x', 'y']].values
	    global_xyz = np.array([
	        globalAligner.fov_coordinates_to_global(fragmentIndex, coord)
	        for coord in xyz_pixels
	    ])
	    df_decode_results[['global_z', 'global_x', 'global_y']] = global_xyz

	    columns_to_keep = [
	    'barcode_id', 'gene', 'fov',
	    'probability',
	    'x', 'y', 'z',
	    'global_x', 'global_y', 'global_z',
	    'cell_index']
	    df_decode_results = df_decode_results[columns_to_keep]

	    columnAdditional = ['mean_intensity', 'max_intensity',
	    'area', 'mean_distance', 'min_distance', 
	    'mean_probability', 'max_probability', 'loglikehood']
	    for col in columnAdditional:
	        df_decode_results[col] = 0

	    # self.dataSet.save_dataframe_to_csv(
	    #     multiplex_pred[0], 'decoded_raw',
	    #     self.analysisName, resultIndex=fragmentIndex)
	    
	    n_rounds = codebook.get_bit_count()
	    for i in range(n_rounds):
	        col = f'intensity_{i}'
	        if col not in df_decode_results.columns:
	            df_decode_results[col] = 0.0  # or np.nan if you prefer
	        
	    print(df_decode_results.columns)
	    self.get_barcode_database().write_barcodes(df_decode_results, fov=fragmentIndex)

