from typing import List
from typing import Union
import numpy as np
import pandas as pd
import time
import os
import pickle
from skimage import registration
from skimage import transform
from skimage import registration
from skimage import morphology
import cv2

from merlin.core import analysistask
from merlin.util import aberration


class Warp(analysistask.ParallelAnalysisTask):

    """
    An abstract class for warping a set of images so that the corresponding
    pixels align between images taken in different imaging rounds.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'write_fiducial_images' not in self.parameters:
            self.parameters['write_fiducial_images'] = False
        if 'write_aligned_images' not in self.parameters:
            self.parameters['write_aligned_images'] = False
        if 'write_aligned_FOVs' not in self.parameters:
            self.parameters['write_aligned_FOVs'] = [-1]
        if 'write_aligned_z' not in self.parameters:
            # None = save all z; otherwise list of zIndexes to write
            self.parameters['write_aligned_z'] = None
        if 'write_averaged_aligned_images' not in self.parameters:
            self.parameters['write_averaged_aligned_images'] = False
        if 'write_averaged_lowpass_sigma' not in self.parameters:
            self.parameters['write_averaged_lowpass_sigma'] = None  
        if 'write_averaged_post_lowpass_sigma' not in self.parameters:
            self.parameters['write_averaged_post_lowpass_sigma'] = None  
        if 'write_averaged_reverse_transform' not in self.parameters:
            self.parameters['write_averaged_reverse_transform'] = True 
        if 'write_fiducial_FOVs' not in self.parameters:
            self.parameters['write_fiducial_FOVs'] = [-1]

        # this is an attempt to fix boundary issues for excessive warping
        # may be useful for long codebooks
        if 'boundary_smooth' not in self.parameters:
            self.parameters['boundary_smooth'] = False

    def get_aligned_image_set(
            self, fov: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        """Get the set of transformed images for the specified fov.

        Args:
            fov: index of the field of view
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 4-dimensional numpy array containing the aligned images. The
                images are arranged as [channel, zIndex, x, y]
        """
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zIndexes = range(len(self.dataSet.get_z_positions()))
        return np.array([[self.get_aligned_image(fov, d, z, chromaticCorrector)
            for z in zIndexes] for d in dataChannels])

    def get_aligned_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        """Get the specified transformed image

        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 2-dimensional numpy array containing the specified image
        """
        inputImage = self.dataSet.get_raw_image(
            dataChannel, fov, self.dataSet.z_index_to_position(zIndex))
        transformation = self.get_transformation(fov, dataChannel)

        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization()\
                .get_data_channel_color(dataChannel)
            inputImage = chromaticCorrector.transform_image(
                inputImage, imageColor)

        # this is the warped image with no padding
        warped_image = transform.warp(inputImage, transformation,
            preserve_range=True)
        
        # an overly complicated attempt to smooth boundary at poorly warped images
        if self.parameters['boundary_smooth']:
            warped_image_blur = transform.warp(inputImage, transformation,
                preserve_range=True, mode = 'edge')
            warped_image_blur = cv2.GaussianBlur(warped_image_blur,
                ksize = (23, 23), 
                sigmaX = 11, 
                borderType=cv2.BORDER_REPLICATE)
            
            mask = (warped_image == 0)
            mask = morphology.binary_dilation(mask) # dilate by one pixel
            warped_image[mask] = warped_image_blur[mask]
            
        return warped_image.astype(inputImage.dtype)

    def _process_transformations(self, transformationList, fov) -> None:
        """
        Process the transformations determined for a given fov. 

        The list of transformation is used to write registered images and 
        the transformation list is archived.

        Args:
            transformationList: A list of transformations that contains a
                transformation for each data channel. 
            fov: The fov that is being transformed.
        """

        dataChannels = self.dataSet.get_data_organization().get_data_channels()

        _alignedFOVs = self.parameters['write_aligned_FOVs']
        _alignedZ = self.parameters['write_aligned_z']
        if self.parameters['write_aligned_images'] \
                and (_alignedFOVs == [-1] or fov in _alignedFOVs):
            zPositions = self.dataSet.get_z_positions()
            imageDescription = self.dataSet.analysis_tiff_description(
                len(zPositions), len(dataChannels))

            with self.dataSet.writer_for_analysis_images(
                self, 'aligned_images', fov) as outputTif:
                for t, x in zip(transformationList, dataChannels):
                    for zi, z in enumerate(zPositions):
                        if _alignedZ is not None and zi not in _alignedZ:
                            continue
                        inputImage = self.dataSet.get_raw_image(x, fov, z)
                        transformedImage = transform.warp(
                            inputImage, t, preserve_range=True).astype(inputImage.dtype)
                        outputTif.save(
                            transformedImage,
                            photometric='MINISBLACK',
                            contiguous=True,
                            metadata=imageDescription)
        
        if self.parameters['write_averaged_aligned_images']:
            zPositions = self.dataSet.get_z_positions()
            imageDescription = self.dataSet.analysis_tiff_description(
                len(zPositions), 1)

            with self.dataSet.writer_for_analysis_images(
                self, 'averaged_aligned_images', fov) as outputTif:
                for z in zPositions:
                    sumImage = None
                    for t, x in zip(transformationList, dataChannels):
                        inputImage = self.dataSet.get_raw_image(x, fov, z)
                        transformedImage = transform.warp(
                            inputImage, t, preserve_range=True).astype('float64')
                        transformedImage = transformedImage / np.mean(transformedImage) * 1000 # for normalization
                        if self.parameters['write_averaged_lowpass_sigma'] is not None:
                            lowPassSigma = self.parameters['write_averaged_lowpass_sigma']
                            filterSize = int(2 * np.ceil(2 * lowPassSigma) + 1)
                            transformedImage = cv2.GaussianBlur(transformedImage, (filterSize, filterSize), lowPassSigma, borderType=cv2.BORDER_REPLICATE)
                        if self.parameters['write_averaged_reverse_transform']:
                            if self.dataSet.flipVertical:
                                transformedImage = np.flip(transformedImage, axis=0)
                            if self.dataSet.flipHorizontal:
                                transformedImage = np.flip(transformedImage, axis=1)
                            if self.dataSet.transpose:
                                transformedImage = np.transpose(transformedImage)

                        if sumImage is None:
                            sumImage = transformedImage
                        else:
                            sumImage += transformedImage
                            
                    avgImage = sumImage / float(len(dataChannels))
                    if self.parameters['write_averaged_post_lowpass_sigma'] is not None:
                        lowPassSigma = self.parameters['write_averaged_post_lowpass_sigma']
                        filterSize = int(2 * np.ceil(2 * lowPassSigma) + 1)
                        avgImage = cv2.GaussianBlur(avgImage, (filterSize, filterSize), lowPassSigma, borderType=cv2.BORDER_REPLICATE)
                            
                    outputTif.save(
                        avgImage.astype(inputImage.dtype),
                        photometric='MINISBLACK',
                        contiguous=True,
                        metadata=imageDescription)
        
        if self.parameters['write_fiducial_images']:
            fiducialFOVs = self.parameters['write_fiducial_FOVs']
            if isinstance(fiducialFOVs, np.ndarray):
                fiducialFOVs = fiducialFOVs.tolist()
            elif not isinstance(fiducialFOVs, list):
                fiducialFOVs = list(fiducialFOVs)

            if fiducialFOVs == [-1] or (fov in fiducialFOVs):
                fiducialImageDescription = self.dataSet.analysis_tiff_description(
                    1, len(dataChannels))
                    
                with self.dataSet.writer_for_analysis_images(
                    self, 'fiducial_images', fov) as outputTif:
                    for t, x in zip(transformationList, dataChannels):
                        inputImage = self.dataSet.get_fiducial_image(x, fov)
                        transformedImage = transform.warp(
                            inputImage, t, preserve_range=True).astype(inputImage.dtype)
                        outputTif.save(
                            transformedImage, 
                            photometric='MINISBLACK',
                            contiguous=True,
                            metadata=fiducialImageDescription)

        self._save_transformations(transformationList, fov)

    def _save_transformations(self, transformationList: List, fov: int) -> None:
    
        # fix for futurewarning np.array object
        # save the matrix directly from the transform.SimilarityTransform object
        transformationList = np.array([t.params for t in transformationList])
        self.dataSet.save_numpy_analysis_result(
            np.array(transformationList), 'offsets',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations')

    def get_transformation(self, fov: int, dataChannel: int=None
                            ) -> Union[transform.EuclideanTransform,
                                 List[transform.EuclideanTransform]]:
        """Get the transformations for aligning images for the specified field
        of view.

        Args:
            fov: the fov to get the transformations for.
            dataChannel: the index of the data channel to get the transformation
                for. If None, then all data channels are returned.
        Returns:
            a EuclideanTransform if dataChannel is specified or a list of
                EuclideanTransforms for all dataChannels if dataChannel is
                not specified.
        """
        transformationMatrices = self.dataSet.load_numpy_analysis_result(
            'offsets', self, resultIndex=fov, subdirectory='transformations')
        
        # fix for futurewarning np.array object 
        # convert the matrix back to transform.SimilarityTransform object
        transformationMatrices = [transform.SimilarityTransform(mat) for mat in transformationMatrices]
        
        if dataChannel is not None:
            return transformationMatrices[dataChannel]
        else:
            return transformationMatrices


class FiducialCorrelationWarp(Warp):

    """
    An analysis task that warps a set of images taken in different imaging
    rounds based on the crosscorrelation between fiducial images.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 3
        if 'clip_negative_after_highpass' not in self.parameters:
            self.parameters['clip_negative_after_highpass'] = False
        # 3x3 median pre-filter on the fiducial image to suppress hot pixels.
        # Default True (prior behaviour); set False to match runs that skip it.
        if 'median_filter' not in self.parameters:
            self.parameters['median_filter'] = True

        # add this parameter to control the next two parameters
        # if beads are not sparse its probably not necessary...
            #percentile_pixel_to_keep
            #edge_width_to_remove
        if 'sparse_bead_fix' not in self.parameters:
            self.parameters['sparse_bead_fix'] = False

        # xingjie parameters to add
        if 'percentile_pixel_to_keep' not in self.parameters:
            self.parameters['percentile_pixel_to_keep'] = 99
        if 'edge_width_to_remove' not in self.parameters: # not entirely sure the point of this one...
            self.parameters['edge_width_to_remove'] = 10

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []

    def _filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassSigma = self.parameters['highpass_sigma']
        if highPassSigma is None:
            high_passed_img = inputImage.astype(float)
        else:
            highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)

            # median filter to deal with hot pixels (optional)
            if self.parameters['median_filter']:
                inputImage = cv2.medianBlur(inputImage.astype(np.uint16), ksize = 3)
    
            high_passed_img = inputImage.astype(float) - cv2.GaussianBlur(
                inputImage, (highPassFilterSize, highPassFilterSize),
                highPassSigma, borderType=cv2.BORDER_REPLICATE)

            if self.parameters['clip_negative_after_highpass']:
                high_passed_img[high_passed_img < 0] = 0
    
        # add some features from Xingjie https://github.com/xingjiepan/MERlin/blob/xingjie/merlin/analysis/warp.py
        
        if self.parameters['sparse_bead_fix']:

            # Remove the boundaries
            edge_width_to_remove = self.parameters['edge_width_to_remove']
            high_passed_img[:edge_width_to_remove] = 0
            high_passed_img[high_passed_img.shape[0] - edge_width_to_remove:] = 0
            high_passed_img[:, :edge_width_to_remove] = 0
            high_passed_img[:, high_passed_img.shape[1] - edge_width_to_remove:] = 0

            # Only keep the most bright pixels
            # this is useful for sparse beads
            percentile_pixel_to_keep = self.parameters['percentile_pixel_to_keep']
            high_passed_img[high_passed_img <
                    np.percentile(high_passed_img, percentile_pixel_to_keep)] = 0

        return high_passed_img        
    
    def _run_analysis(self, fragmentIndex: int):
        # map fragment index -> real FOV id
        fov = list(self.dataSet.get_fovs())[fragmentIndex]

        if self.parameters['write_fiducial_images']:
            if self.parameters['write_fiducial_FOVs'] == [-1]:
                self.parameters['write_fiducial_FOVs'] = self.dataSet.get_fovs()

        fixedImage = self._filter(
            self.dataSet.get_fiducial_image(0, fov))
        
        dataChannels = list(self.dataSet.get_data_organization().get_data_channels())

        results = [registration.phase_cross_correlation(
            fixedImage, self._filter(self.dataSet.get_fiducial_image(ch, fov)),
            upsample_factor=100
            ) for ch in dataChannels]
        offsets = [r[0] for r in results]      # (y, x)
        errors  = [r[1] for r in results]      # registration error
        phases  = [r[2] for r in results]      # phase diff

        transformations = [
            transform.SimilarityTransform(translation=[-x[1], -x[0]])
            for x in offsets
        ]

        self._process_transformations(transformations, fov)
        
        '''metrics = np.column_stack([
            np.array(dataChannels, dtype=float),
            np.array([s[0] for s in offsets], dtype=float),
            np.array([s[1] for s in offsets], dtype=float),
            np.array(errors, dtype=float),
            np.array(phases, dtype=float),
        ])
        
        self.dataSet.save_numpy_txt_analysis_result(
            metrics, "metrics",
            self.get_analysis_name(), resultIndex=fov,
            subdirectory="transformations")'''
            
        df_metrics = pd.DataFrame({
            "channel": dataChannels,
            "shift_y": [float(s[0]) for s in offsets],
            "shift_x": [float(s[1]) for s in offsets],
            "error":   [float(e) for e in errors],
            "phasediff": [float(p) for p in phases],
        })

        self.dataSet.save_dataframe_to_csv(
            df_metrics,
            "metrics",
            self.get_analysis_name(),
            resultIndex=fov,
            subdirectory="transformations",
            index=False,
        )


class FiducialCorrelationWarp3D(FiducialCorrelationWarp):

    """
    An analysis task that warps a set of images taken in different imaging
    rounds based on the crosscorrelation between fiducial images.
    
    
    General plan - there are three corrections applied
    The first correction is that every stack is corrected for the piezo induced drift
        This uses a calibration of the z-stage position from the .off file
    The second correction is the XY registration from the fiducial bead frame
    The third correction is using an XYZ registration of the fiducial3D bead stacks
    
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        
        if 'piezo_correction_filepath' not in self.parameters:
            self.parameters['piezo_correction_filepath'] = None

        self.load_piezo_parameters(self.parameters['piezo_correction_filepath'])
        

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 4096

    def get_estimated_time(self):
        return 5
        
    def load_piezo_parameters(self, path):
        
        if os.path.exists(path):
            with open(path, 'rb') as inputFile:
                self.piezoParameters = pickle.load(inputFile)
            # get the interpolation functions
            # these are the pickled interpolation functions in 2D
            # inputs are 
            # in HAL terms:     z_offset,   piezo stage-z
            # in Merlin terms:  z position, true piezo position from .off file 
            self.piezo_yshift_function = self.piezoParameters.get(
                'yshift', None)
            self.piezo_xshift_function = self.piezoParameters.get(
                'xshift', None)
        else:
            # clunky...
            print('no piezo pickle file found, no correction to be applied')
            def piezo_yshift_function(self,a,b):
                return 0
            def piezo_xshift_function(self,a,b):
                return 0
        
    def get_piezo_corrected_frame(self,
                                fov: int,
                                dataChannel: int,
                                zIndex: int,
                                chromaticCorrector: aberration.ChromaticCorrector=None
                                ) -> np.ndarray:
        """Get the specified image corrected for piezo drift in XY
        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
        Returns:
            a 2-dimensional numpy array containing the specified image
        """
        
        inputImage = self.dataSet.get_raw_image(dataChannel, fov, self.dataSet.z_index_to_position(zIndex))
        inputImage_zstage_position = self.dataSet.get_raw_image_zstage_positions(dataChannel, fov)[zIndex]

        #### make sure this is correct
        # these are the interpolation functions - note the negative sign
        # inputs are merlin z position and true zposition of piezo
        x_correction = -self.piezo_xshift_function(self.dataSet.z_index_to_position(zIndex),
                                                           inputImage_zstage_position)
        y_correction = -self.piezo_yshift_function(self.dataSet.z_index_to_position(zIndex),
                                                           inputImage_zstage_position)

        transformation = transform.SimilarityTransform(translation=[x_correction, y_correction])
        
        return transform.warp(inputImage, transformation, 
            preserve_range=True).astype(inputImage.dtype)                         

    def get_piezo_corrected_fiducial3D_stack(self, dataChannel, fragmentIndex):
        # this will return a piezo corrected stack of the fiducial3D image 
        # the assumption here is that the first frame index is the beads on the coverglass surface
        
        stack = self.dataSet.get_fiducial3D_stack(dataChannel, fragmentIndex)


        stack_zpositions = self.dataSet.get_data_organization().get_fiducial3D_stack_frame_zPos(
                                                            dataChannel)
        stack_zstage_positions = self.dataSet.get_fiducial_image_zstage_positions(dataChannel, 
                                                            fragmentIndex)

        x_correction = -self.piezo_xshift_function(stack_zpositions,
                                                           stack_zstage_positions)
        y_correction = -self.piezo_yshift_function(stack_zpositions,
                                                           stack_zstage_positions)

        transforms = [transform.SimilarityTransform(translation=[x, y]) for x,y in zip(x_correction,y_correction)]
        
        for i in range(len(stack)):
            stack[i] = transform.warp(stack[i], transforms[i], preserve_range=True).astype(stack.dtype)
        
        return stack
    
    # this is standard MERFISH registration
    # should not need to correct this for piezo since we correct to the fiducial frame postion anyways
    def _find_2D_offsets(self, fragmentIndex: int):
        
        fixedImage = self._filter(
                self.dataSet.get_fiducial_image(0, fragmentIndex))
        
        # phase cross cor returns Y X shifts
        offsets = [registration.phase_cross_correlation(
            fixedImage,
            self._filter(self.dataSet.get_fiducial_image(x, fragmentIndex)),
            upsample_factor = 100)[0] for x in
                   self.dataSet.get_data_organization().get_data_channels()]
        
        # should be Y X order
        offsets2D = [[x[0], x[1]] for x in offsets]
        return offsets2D
    
    def _find_offsets_from_3D_stacks(self, fragmentIndex: int):
        # this is for registration of a bead stack at the top of a 3D tissue
        # first register the zero plane
        fixedImage = self.get_piezo_corrected_fiducial3D_stack(0, fragmentIndex)
        
        offsets3D_base = []
        offsets3D = []
        
        for dataChannel in self.dataSet.get_data_organization().get_data_channels():
            movingImage = self.get_piezo_corrected_fiducial3D_stack(dataChannel, fragmentIndex)
            
            # 2D base offset this should be Y X
            offsets3D_base.append(registration.phase_cross_correlation(fixedImage[0], movingImage[0], upsample_factor = 100)[0])
            # 3D offsets.. this should be Z Y X
            offsets3D.append(registration.phase_cross_correlation(fixedImage[1:], movingImage[1:], upsample_factor = 100)[0])
            
        return offsets3D_base, offsets3D

    def _save_transformation_dataFrame(self, offsets2D, offsets3D_base, offsets3D, fragmentIndex: int):
        df = pd.DataFrame(columns = ['dataChannel','zPos','zPos_new','xshift','yshift'])
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zPos_orig = self.dataSet.get_data_organization().get_z_positions()
        # assume all fiducial zpos are the same across channels
        fiducial3D_zpos = self.dataSet.get_data_organization().get_fiducial3D_stack_frame_zPos(0) 
        fiducial3D_zpos_center = np.mean(fiducial3D_zpos[1:]) # first frame is assumed to be zero

        # zip datachannel, 2d offsets, 3d base offsets, 3d stack offsets
        for dc, (off2D_y, off2D_x), (off3Db_y,off3Db_x), (off3D_z, off3D_y, off3D_x) in zip(dataChannels, offsets2D, offsets3D_base, offsets3D):
            df_temp = pd.DataFrame(columns = ['dataChannel','zPos','zPos_new','xshift','yshift'])
            # these are the 2d shift and a z-dependant shift from the 3d registration
            # make negative since we are shifting the fixed image

            #                   2d          3d - 3d base    * some percentage of the height
            df_temp['yshift'] = -off2D_y - (off3D_y-off3Db_y)*np.array(zPos_orig)/fiducial3D_zpos_center
            df_temp['xshift'] = -off2D_x - (off3D_x-off3Db_x)*np.array(zPos_orig)/fiducial3D_zpos_center
            df_temp['zPos'] = zPos_orig # original z pos just so its in the df
            # new zpos. shift is the direction to move the moving image
            # if zshift is negative, the moving image has expanded 
            # so our new zpos should be larger hence the negative below to make the ratio larger
            df_temp['zPos_new'] = np.array(zPos_orig) * (fiducial3D_zpos_center - off3D_z)/fiducial3D_zpos_center
            # include the datachannel
            df_temp['dataChannel'] = dc
            # include raw z shift for troubleshooting purposes
            df_temp['zshift'] = off3D_z

            df = pd.concat([df, df_temp], ignore_index=True)
        
        self.dataSet.save_dataframe_to_csv(df,
                                           'transformation_table',
                                           self.get_analysis_name(),
                                           resultIndex=fragmentIndex,
                                           subdirectory='transformations')                   
    
    def get_transformation_table(self, fov: int) -> pd.DataFrame:
        """Get the transformations for aligning images for the specified field
        of view.

        Args:
            fov: the fov to get the transformations for.
            dataChannel: the index of the data channel to get the transformation
                for. If None, then all data channels are returned.
        Returns:
            a EuclideanTransform if dataChannel is specified or a list of
                EuclideanTransforms for all dataChannels if dataChannel is
                not specified.
        """
        transformation_table = self.dataSet.load_dataframe_from_csv(
            'transformation_table', self, resultIndex=fov, subdirectory='transformations')
            
        return transformation_table

    def get_aligned_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        """Get the specified transformed image
        corrected using 3d beads

        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 2-dimensional numpy array containing the specified image
        """
        df = self.get_transformation_table(fov)
        zPos = self.dataSet.z_index_to_position(zIndex)
        df = df[(df['dataChannel'] == dataChannel) & 
                     (df['zPos'] == zPos)]
        
        xshift = df['xshift'].values[0]
        yshift = df['yshift'].values[0]
        zPos_new = df['zPos_new'].values[0]

        zPos_all = np.array(self.dataSet.get_z_positions())
        
        if zPos_new > np.amax(zPos_all):
            zPos_new = np.amax(zPos_all)
        
        # interpolate the two nearest frames
        zPos_nearest = zPos_all[np.abs(zPos_all - zPos_new).argsort()[0:2]] # take two nearest zpos
        zPos_nearest_distances = np.abs(zPos_new - zPos_nearest) # find distance
        weights = 1 - zPos_nearest_distances/np.sum(zPos_nearest_distances) # get weighting factor
        
        images_nearest = [self.dataSet.get_raw_image(dataChannel, fov, z) for z in zPos_nearest]
        
        # interpolated image
        inputImage = (images_nearest[0] * weights[0] +
                      images_nearest[1] * weights[1]).astype(
                      images_nearest[0].dtype) 
                      
        transformation = transform.SimilarityTransform(
            translation=[xshift, yshift])

        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization()\
                            .get_data_channel_color(dataChannel)
            return transform.warp(chromaticCorrector.transform_image(
                inputImage, imageColor), transformation, preserve_range=True
                ).astype(inputImage.dtype)
        else:
            return transform.warp(inputImage, transformation,
                                  preserve_range=True).astype(inputImage.dtype)
                                  

    def _process_transformations(self, fov) -> None:
        """
        Process the transformations determined for a given fov. 

        The list of transformation is used to write registered images and 
        the transformation list is archived.

        Args:
            transformationList: A list of transformations that contains a
                transformation for each data channel. 
            fov: The fov that is being transformed.
        """

        dataChannels = self.dataSet.get_data_organization().get_data_channels()

        if self.parameters['write_aligned_images']:
            zPositions = self.dataSet.get_z_positions()

            imageDescription = self.dataSet.analysis_tiff_description(
                    len(zPositions), len(dataChannels))

            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_images', fov) as outputTif:
                for x in dataChannels:
                    for z in zPositions:
                        #print('aligning channel {} zpos {}'.format(x, z))
                        transformedImage = self.get_aligned_image(fov, x, 
                                                self.dataSet.position_to_z_index(z))
                        outputTif.save(
                                transformedImage,
                                photometric='MINISBLACK',
                                metadata=imageDescription)

        # this should be unchanged from normal merlin
        if self.writeAlignedFiducialImages:

            transformationList = self.get_transformation(fov)

            fiducialImageDescription = self.dataSet.analysis_tiff_description(
                    1, len(dataChannels))

            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_fiducial_images', fov) as outputTif:
                for t, x in zip(transformationList, dataChannels):
                    inputImage = self.dataSet.get_fiducial_image(x, fov)
                    transformedImage = transform.warp(
                            inputImage, t, preserve_range=True) \
                        .astype(inputImage.dtype)
                    outputTif.save(
                            transformedImage, 
                            photometric='MINISBLACK',
                            metadata=fiducialImageDescription)
                    

    def _run_analysis(self, fragmentIndex: int):
        print('running 2d registration')
        offsets2D = self._find_2D_offsets(fragmentIndex)

        # just save the 2D transformation like normal merlin
        transformations2D = [transform.SimilarityTransform(
            translation=[-x[1], -x[0]]) for x in offsets2D]
        
        self._save_transformations(transformations2D, fragmentIndex)
        
        # now do 3D registration
        print('running 3d registration')
        offsets3D_base, offsets3D = self._find_offsets_from_3D_stacks(fragmentIndex)
        
        self._save_transformation_dataFrame(offsets2D, offsets3D_base, offsets3D, fragmentIndex)

        self._process_transformations(fragmentIndex)
