from typing import List
from typing import Union
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
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

    outputGroup = 'Prepare'

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
                        outputTif.write(
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
                            
                    outputTif.write(
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
                        outputTif.write(
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

    With register_3d off (the default) this is the standard MERFISH
    registration: one xy offset per imaging round, measured by phase
    correlation between the 2D coverglass fiducial frames.

    With register_3d on, two further corrections are applied for thick samples:

      * piezo-induced xy drift, from a calibration of the z stage position
        against the .off file, if piezo_correction_filepath is given
      * a single 3D phase correlation of the whole fiducial bead stack per
        round, which becomes an xy shift growing linearly with depth and a
        linear rescaling of z

    The 3D model is deliberately proportional: one measurement per round,
    extrapolated through depth. FiducialPolynomialWarp3D measures every plane
    and fits a polynomial instead, and reduces to this one at order 1 with the
    intercept forced through zero.
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

        # Correct depth-dependent drift as well as the per-round xy offset.
        # Requires fiducial3D columns in the data organization.
        if 'register_3d' not in self.parameters:
            self.parameters['register_3d'] = False
        # Pickle of interpolation functions {'yshift':f, 'xshift':f} mapping
        # (merlin z position, true piezo position from the .off file) to a
        # shift. Optional: without it the piezo term is zero and only the bead
        # stack registration is applied.
        if 'piezo_correction_filepath' not in self.parameters:
            self.parameters['piezo_correction_filepath'] = None
        # 'weighted' interpolates between the two nearest planes when
        # resampling to the corrected depth; 'nearest' snaps to one plane.
        if 'interpolation' not in self.parameters:
            self.parameters['interpolation'] = 'weighted'

        self._piezoYShift = None
        self._piezoXShift = None
        if self.parameters['register_3d']:
            self._load_piezo_parameters(
                self.parameters['piezo_correction_filepath'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 4096 if self.parameters['register_3d'] else 2048

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

    def _find_2D_offsets(self, fov: int):
        """The per-round xy offset from the 2D coverglass fiducial frame.

        Returns (y, x) shifts, i.e. what has to be added to the moving image to
        bring it onto the reference.
        """
        fixedImage = self._filter(self.dataSet.get_fiducial_image(0, fov))
        dataChannels = list(
            self.dataSet.get_data_organization().get_data_channels())
        results = [registration.phase_cross_correlation(
            fixedImage, self._filter(self.dataSet.get_fiducial_image(ch, fov)),
            upsample_factor=100) for ch in dataChannels]
        return dataChannels, results

    # ==================================================================
    # 3D registration, only used when register_3d is set
    # ==================================================================

    def _load_piezo_parameters(self, path) -> None:
        """Load the piezo drift calibration, if there is one.

        A missing or unset path is not an error: the piezo term simply drops
        out and the bead stack registration carries the whole correction.
        """
        if not path or not os.path.exists(path):
            print('no piezo calibration found, no piezo correction applied')
            return
        with open(path, 'rb') as inputFile:
            piezoParameters = pickle.load(inputFile)
        # Pickled 2D interpolation functions. Inputs are, in HAL terms,
        # (z_offset, piezo stage z); in MERlin terms, (z position, true piezo
        # position from the .off file).
        self._piezoYShift = piezoParameters.get('yshift', None)
        self._piezoXShift = piezoParameters.get('xshift', None)

    def _piezo_correction(self, zPositions, zStagePositions):
        """Negated piezo shift at the given positions, as (x, y)."""
        zPositions = np.asarray(zPositions)
        if self._piezoXShift is None or self._piezoYShift is None:
            zeros = np.zeros(zPositions.shape)
            return zeros, zeros
        return (-self._piezoXShift(zPositions, zStagePositions),
                -self._piezoYShift(zPositions, zStagePositions))

    def get_piezo_corrected_frame(
            self, fov: int, dataChannel: int, zIndex: int) -> np.ndarray:
        """One raw image plane with the piezo xy drift taken out."""
        zPosition = self.dataSet.z_index_to_position(zIndex)
        inputImage = self.dataSet.get_raw_image(dataChannel, fov, zPosition)
        if self._piezoXShift is None or self._piezoYShift is None:
            return inputImage
        stagePosition = self.dataSet.get_raw_image_zstage_positions(
            dataChannel, fov)[zIndex]
        xCorrection, yCorrection = self._piezo_correction(
            zPosition, stagePosition)
        return transform.warp(
            inputImage,
            transform.SimilarityTransform(
                translation=[float(xCorrection), float(yCorrection)]),
            preserve_range=True).astype(inputImage.dtype)

    def get_piezo_corrected_fiducial3D_stack(self, dataChannel: int,
                                             fov: int) -> np.ndarray:
        """The fiducial bead stack with the piezo xy drift taken out.

        The first frame is assumed to be the beads on the coverglass surface.
        """
        stack = self.dataSet.get_fiducial3D_stack(dataChannel, fov)
        if self._piezoXShift is None or self._piezoYShift is None:
            # Nothing to correct, and reading the stage positions would open
            # the .off file for no reason -- it is not always present.
            return stack

        stackZPositions = self.dataSet.get_data_organization()\
            .get_fiducial3D_stack_frame_zPos(dataChannel)
        stackStagePositions = self.dataSet.get_fiducial_image_zstage_positions(
            dataChannel, fov)
        xCorrection, yCorrection = self._piezo_correction(
            stackZPositions, stackStagePositions)

        for i in range(len(stack)):
            stack[i] = transform.warp(
                stack[i],
                transform.SimilarityTransform(
                    translation=[float(xCorrection[i]), float(yCorrection[i])]),
                preserve_range=True).astype(stack.dtype)
        return stack

    def _find_offsets_from_3D_stacks(self, fov: int):
        """One (z, y, x) offset per round from the whole bead stack, plus the
        (y, x) offset of its surface plane."""
        fixedStack = self.get_piezo_corrected_fiducial3D_stack(0, fov)

        baseOffsets, stackOffsets = [], []
        for dataChannel in self.dataSet.get_data_organization()\
                .get_data_channels():
            movingStack = self.get_piezo_corrected_fiducial3D_stack(
                dataChannel, fov)
            baseOffsets.append(registration.phase_cross_correlation(
                fixedStack[0], movingStack[0], upsample_factor=100)[0])
            stackOffsets.append(registration.phase_cross_correlation(
                fixedStack[1:], movingStack[1:], upsample_factor=100)[0])
        return baseOffsets, stackOffsets

    def _save_transformation_dataFrame(self, offsets2D, baseOffsets,
                                       stackOffsets, fov: int) -> None:
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zPositions = np.array(
            self.dataSet.get_data_organization().get_z_positions())
        # assume all fiducial z positions are the same across channels
        fiducialZPositions = self.dataSet.get_data_organization()\
            .get_fiducial3D_stack_frame_zPos(0)
        # the first frame is the coverglass surface, so it is excluded
        centre = np.mean(fiducialZPositions[1:])

        tables = []
        for (dataChannel, (offset2DY, offset2DX), (baseY, baseX),
             (stackZ, stackY, stackX)) in zip(
                dataChannels, offsets2D, baseOffsets, stackOffsets):
            table = pd.DataFrame({'zPos': zPositions})
            # the 2D shift plus a z-dependent shift from the 3D registration,
            # negated because it is the fixed image that moves
            table['yshift'] = -offset2DY - (stackY - baseY) * zPositions / centre
            table['xshift'] = -offset2DX - (stackX - baseX) * zPositions / centre
            # a negative z shift means the moving image expanded, so its new z
            # position has to be larger -- hence the minus in the numerator
            table['zPos_new'] = zPositions * (centre - stackZ) / centre
            table['dataChannel'] = dataChannel
            table['zshift'] = stackZ
            tables.append(table)

        self.dataSet.save_dataframe_to_csv(
            pd.concat(tables, ignore_index=True), 'transformation_table',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations')

    def get_transformation_table(self, fov: int) -> pd.DataFrame:
        """The per-plane transformations, only written when register_3d is on."""
        return self.dataSet.load_dataframe_from_csv(
            'transformation_table', self, resultIndex=fov,
            subdirectory='transformations')

    def _resample_to_depth(self, fov: int, dataChannel: int,
                           newZPosition: float) -> np.ndarray:
        zPositions = np.asarray(self.dataSet.get_z_positions(), dtype=float)
        newZPosition = float(np.clip(newZPosition,
                                     zPositions.min(), zPositions.max()))
        if self.parameters['interpolation'] == 'nearest':
            nearest = zPositions[np.abs(zPositions - newZPosition).argmin()]
            return self.dataSet.get_raw_image(dataChannel, fov, nearest)
        if self.parameters['interpolation'] != 'weighted':
            raise ValueError('Unknown interpolation: %s'
                             % self.parameters['interpolation'])
        order = np.abs(zPositions - newZPosition).argsort()
        nearZ = zPositions[order[:2]]
        distances = np.abs(newZPosition - nearZ)
        if distances.sum() < 1e-10:
            return self.dataSet.get_raw_image(dataChannel, fov, nearZ[0])
        weights = 1.0 - distances / distances.sum()
        first = self.dataSet.get_raw_image(dataChannel, fov, nearZ[0])
        second = self.dataSet.get_raw_image(dataChannel, fov, nearZ[1])
        return (first.astype(np.float32) * weights[0]
                + second.astype(np.float32) * weights[1]).astype(first.dtype)

    def get_aligned_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        if not self.parameters['register_3d']:
            return super().get_aligned_image(
                fov, dataChannel, zIndex, chromaticCorrector)

        table = self.get_transformation_table(fov)
        zPosition = self.dataSet.z_index_to_position(zIndex)
        row = table[(table['dataChannel'] == dataChannel)
                    & (table['zPos'] == zPosition)].iloc[0]

        inputImage = self._resample_to_depth(fov, dataChannel, row['zPos_new'])
        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization()\
                            .get_data_channel_color(dataChannel)
            inputImage = chromaticCorrector.transform_image(
                inputImage, imageColor).astype(inputImage.dtype)
        return transform.warp(
            inputImage,
            transform.SimilarityTransform(
                translation=[row['xshift'], row['yshift']]),
            preserve_range=True).astype(inputImage.dtype)

    def _run_analysis(self, fragmentIndex: int):
        # map fragment index -> real FOV id
        fov = list(self.dataSet.get_fovs())[fragmentIndex]

        if self.parameters['write_fiducial_images']:
            if self.parameters['write_fiducial_FOVs'] == [-1]:
                self.parameters['write_fiducial_FOVs'] = self.dataSet.get_fovs()

        dataChannels, results = self._find_2D_offsets(fov)
        offsets = [r[0] for r in results]      # (y, x)
        errors  = [r[1] for r in results]      # registration error
        phases  = [r[2] for r in results]      # phase diff

        transformations = [
            transform.SimilarityTransform(translation=[-x[1], -x[0]])
            for x in offsets
        ]

        self._process_transformations(transformations, fov)

        if self.parameters['register_3d']:
            baseOffsets, stackOffsets = self._find_offsets_from_3D_stacks(fov)
            self._save_transformation_dataFrame(
                offsets, baseOffsets, stackOffsets, fov)

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


class FiducialPolynomialWarp3D(FiducialCorrelationWarp):

    """
    An analysis task that registers each imaging round to the first one in
    three dimensions, modelling the drift as a smooth polynomial function of
    depth.

    Gel-embedded thick samples drift in both z and xy as a function of depth,
    and the drift differs from round to round. A single rigid offset per round
    therefore leaves a depth-dependent residual, while measuring an independent
    offset for every plane is noisy and occasionally catastrophic. This task
    measures per-plane offsets from the 3D fiducial bead stack and then reduces
    them to a low-order polynomial in z, which suppresses the per-plane noise
    and removes isolated failures altogether.

    The pipeline, per (fov, imaging round):

      1. Rigid xy, from the 2D coverglass fiducial frame. This is exactly what
         FiducialCorrelationWarp does and it is accurate to about a pixel, but
         only at the depth the beads sit at.
      2. Per-plane z offset, measured at the rigid xy by maximising the
         normalised cross-correlation over integer plane offsets and refining
         the peak with a parabola. Fit robustly with a polynomial of order
         z_polynomial_order to give dz(z).
      3. Per-plane xy offset, measured AFTER resampling the moving fiducial
         stack to z + dz(z) so the xy estimate is not contaminated by a z
         mismatch. The correlation peak is searched only within
         xy_search_bound pixels of the rigid offset: an unbounded search on
         this kind of dense, self-similar texture regularly locks onto a
         secondary maximum of the image autocorrelation hundreds of pixels
         away. Fit robustly with a polynomial of order xy_polynomial_order to
         give dy(z), dx(z).
      4. Apply: resample the raw image at z + dz(z) and translate it by
         (dx(z), dy(z)).

    Setting z_polynomial_order and xy_polynomial_order to 1 recovers a linear
    ramp in z; setting them to 0 recovers a rigid offset per round with a
    constant z shift.

    THE Z FIT IS GATED, AND A STACK THAT CANNOT BE FITTED BORROWS ITS CURVE
    FROM ITS NEIGHBOURS. Step 2 above assumes every sampled plane produced a
    usable correlation peak. Deep in a thick gel that is not true: the bead
    signal can be gone over a contiguous run of planes, and the argmax of a
    flat landscape is arbitrary. On 20260609 that put a 5.4 um swing on one
    (fov, round) -- eleven planes -- out of a fit whose median swing is 0.85.
    So:

      a) a z sample is DROPPED if its peak ZNCC is below z_min_score, or if
         |offset| sits on the +-(z_search_range * plane spacing) ceiling,
         which means the argmax railed rather than found a peak;
      b) if fewer than z_min_good_samples survive, NO POLYNOMIAL IS FITTED --
         the z coefficients are zero and the round falls back to its rigid xy,
         which is honest about having no depth information;
      c) finalize() then replaces those zeroed curves with the curve of the
         same round in the spatially adjacent fovs, if z_neighbor_substitution
         is on. This is a cross-fov step, so it cannot happen inside a
         per-fov fragment; it runs once, from `merlin -t <task> --check-done`.

    The substitution is justified by leave-one-out over the 1,700 fully
    sampled 20260609 stacks: a curve borrowed from the 8-neighbourhood
    reproduces a stack's own fit to 0.084 um RMS (p90 0.226) against curves
    that span a median 0.847 um. Measured end to end on the decode, at a
    matched misidentification rate, the worst fov gained 13.1% (misID 0.15)
    and 19.8% (0.05) coding barcodes over the ungated cubic, while falling all
    the way back to a rigid offset at the first plane LOST 79.6% and 93.2%.
    The depth-dependent xy polynomial is doing most of the work even when the
    z fit is broken, so dropping to rigid is the worst option, not the safe
    one.

    To restore the pre-gate behaviour exactly: z_min_score 0,
    z_reject_at_search_limit false, z_min_good_samples 0,
    z_neighbor_substitution false, z_polynomial_order 3.

    All correlations use plain cross-correlation rather than phase
    correlation. Whitening the spectrum amplifies the high-frequency noise
    that dominates the deep, dim planes and was the direct cause of the large
    outliers this task exists to avoid.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # ---- drift model ----
        # Order of the polynomial in z used for the z offset and for the xy
        # residual. 1 gives a linear ramp, 0 a constant offset, 3 a cubic.
        #
        # z defaults to 1, xy stays at 3. Gel expansion between rounds is a
        # bulk strain, which is linear in depth; the extra cubic freedom buys
        # nothing and costs stability where the samples are sparse. Measured
        # on 20260609 by leave-one-out over the sampled planes -- hold one
        # sample out, fit the rest, predict it -- order 1 gives 0.098 um
        # against order 3's 0.124, and wins on 84% of stacks. The xy residual
        # is not a bulk strain (it is stage and optical) and is left cubic.
        if 'z_polynomial_order' not in self.parameters:
            self.parameters['z_polynomial_order'] = 1
        if 'xy_polynomial_order' not in self.parameters:
            self.parameters['xy_polynomial_order'] = 3

        # ---- measurement ----
        # Half-width, in fiducial planes, of the integer z search.
        if 'z_search_range' not in self.parameters:
            self.parameters['z_search_range'] = 6

        # ---- z sample gate (see the class docstring) ----
        # Minimum peak ZNCC for a z sample to enter the fit. 0.5 on 20260609
        # removes 221 of 16,900 samples; it is well below the 0.9-1.0 a real
        # bead lock scores and well above the 0.2-0.4 of a dead plane. 0
        # disables the gate.
        if 'z_min_score' not in self.parameters:
            self.parameters['z_min_score'] = 0.5
        # Also drop a sample whose |offset| is at the search ceiling. A railed
        # argmax is not a measurement, and its score can still be high because
        # ZNCC is measured against whatever tissue is there. This caught 7
        # samples that the score gate alone let through on 20260609.
        if 'z_reject_at_search_limit' not in self.parameters:
            self.parameters['z_reject_at_search_limit'] = True
        # Fewer surviving samples than this and no polynomial is fitted at all.
        # 4 leaves order 1 two degrees of freedom of slack. 0 disables.
        if 'z_min_good_samples' not in self.parameters:
            self.parameters['z_min_good_samples'] = 4

        # ---- neighbour substitution, applied in finalize() ----
        if 'z_neighbor_substitution' not in self.parameters:
            self.parameters['z_neighbor_substitution'] = True
        # Stage-coordinate radii, in microns, tried in order until one holds
        # at least z_neighbor_min_donors usable donors. The defaults are the
        # 8-neighbourhood, the two-step ring and one wider, for a 280 um
        # pitch; scale them with the tile pitch on other layouts.
        if 'z_neighbor_rings_um' not in self.parameters:
            self.parameters['z_neighbor_rings_um'] = [400.0, 580.0, 800.0]
        if 'z_neighbor_min_donors' not in self.parameters:
            self.parameters['z_neighbor_min_donors'] = 3
        # A donor must itself be well measured, not merely fitted.
        if 'z_neighbor_min_samples' not in self.parameters:
            self.parameters['z_neighbor_min_samples'] = 8
        # Sample every n'th fiducial plane when measuring z / xy. The fits need
        # far fewer points than there are planes, and the z search is the
        # expensive half of the task.
        if 'z_sample_step' not in self.parameters:
            self.parameters['z_sample_step'] = 16
        if 'xy_sample_step' not in self.parameters:
            self.parameters['xy_sample_step'] = 6
        # Half-width, in pixels, of the xy peak search around the rigid offset.
        # The result is insensitive to this between about 3 and 20 px; it only
        # has to exclude the distant secondary maxima.
        if 'xy_search_bound' not in self.parameters:
            self.parameters['xy_search_bound'] = 10
        # Restrict the correlation to the first n rows of the frame. None uses
        # the whole frame. Only useful for held-out validation, where the
        # remaining rows score a shift they did not contribute to.
        # Minimum number of jointly non-zero pixels for a usable correlation.
        if 'minimum_overlap_pixels' not in self.parameters:
            self.parameters['minimum_overlap_pixels'] = 5000
        # Filtered fiducial planes to hold in memory. Each is a float32 frame.
        if 'plane_cache_size' not in self.parameters:
            self.parameters['plane_cache_size'] = 150

        # ---- robust fitting ----
        # Iteratively drop points more than robust_trim_sigma robust standard
        # deviations from the fit. Without this a single bad plane drags the
        # cubic by tens of pixels; with it the fit is unaffected.
        if 'robust_trim_iterations' not in self.parameters:
            self.parameters['robust_trim_iterations'] = 3
        if 'robust_trim_sigma' not in self.parameters:
            self.parameters['robust_trim_sigma'] = 3.0

        # ---- fiducial filtering ----
        # Overrides the inherited defaults. Zeroing the dimmest pixels and the
        # frame edges is what makes the bead texture correlate cleanly; the
        # inherited sparse_bead_fix path keeps only the top 1%, which is right
        # for sparse beads on a coverglass and wrong for a bead stack through
        # tissue.
        if 'edge_width_to_remove' not in self.parameters:
            self.parameters['edge_width_to_remove'] = 20
        # Zero every pixel below this percentile of the high-passed frame.
        # 5 keeps the brightest 95%, matching DenseZWarp's
        # percentile_pixel_to_keep=95. Note this is the opposite sense to the
        # inherited percentile_pixel_to_keep, which is a keep-above threshold.
        if 'zero_below_percentile' not in self.parameters:
            self.parameters['zero_below_percentile'] = 5

        # ---- application ----
        # 'weighted' interpolates between the two nearest planes; 'nearest'
        # snaps to the closest plane and discards the sub-step remainder.
        if 'interpolation' not in self.parameters:
            self.parameters['interpolation'] = 'weighted'

        if 'write_qc_table' not in self.parameters:
            self.parameters['write_qc_table'] = True

        # Threads over sampled z planes when measuring. Every plane's z search
        # and xy correlation is independent of every other, and the heavy
        # steps (OpenCV filtering, the FFTs, the ZNCC reductions) release the
        # GIL, so this scales the same way Decode's per-z pool does.
        # ---- rigid offset: which plane to measure it on ----
        # The 2D coverglass frame (fiducialFrame) is the default and is a poor
        # choice on some datasets. On 20260609, 10 of 130 fovs give either no
        # correlation peak at all or a lock onto a secondary maximum hundreds
        # of pixels away when measured there -- the frame is dominated by
        # tissue autofluorescence rather than beads. The SAME fovs correlate at
        # snr 1000-1800 a few microns into the sample, where puncta counts are
        # a healthy 6800-8000. Beads are present at every depth; it is frame 0
        # specifically that is bad.
        #
        # So: measure at the 2D frame, then walk INTO the stack while each next
        # plane is rigid_snr_ratio times better than the best so far, and keep
        # the best plane's offset. 1.0 would chase noise; the default demands a
        # real improvement. rigid_plane_search=0 keeps the old behaviour.
        if 'rigid_plane_search' not in self.parameters:
            self.parameters['rigid_plane_search'] = 0
        if 'rigid_plane_step' not in self.parameters:
            self.parameters['rigid_plane_step'] = 4
        if 'rigid_snr_ratio' not in self.parameters:
            self.parameters['rigid_snr_ratio'] = 1.5
        # The rigid search must be wide: the offset itself is ~35 px on
        # 20260609 and ~163 px on 20260103, both far beyond xy_search_bound,
        # which bounds the per-plane RESIDUAL after the rigid shift.
        if 'rigid_search_bound' not in self.parameters:
            self.parameters['rigid_search_bound'] = 200
        # How far a LATER plane may move the estimate from the previous plane's.
        # Consecutive planes see the same rigid offset, so this is small; it is
        # what stops the walk from drifting cumulatively.
        if 'rigid_walk_bound' not in self.parameters:
            self.parameters['rigid_walk_bound'] = 15
        # Channels that get a rigid xy offset and NO 3D warp. Empty by
        # default, so nothing changes for a dataset that does not set it.
        if 'rigid_only_channels' not in self.parameters:
            self.parameters['rigid_only_channels'] = []
        # Fit-quality guard. A bad rigid offset does not announce itself in the
        # offset -- fov52's biotin 2D frame was in-band, agreed with its
        # neighbours to 8 px and was wrong by 57 -- but it DOES show up in the
        # residual of the polynomial fitted on top of it. Measured over 1820
        # fits (130 fovs x 14 stacks) on 20260609: z residual median 0.038 um
        # with p99.9 0.376, xy median 0.074 px with p99.9 0.269, and exactly
        # one outlier, fov52 channel 16, at 2.090 um and 5.947/5.456 px. These
        # thresholds sit far above the p99.9 and far below that outlier, so
        # they flag a real failure without crying about ordinary variation.
        if 'max_z_fit_residual' not in self.parameters:
            self.parameters['max_z_fit_residual'] = 0.5       # microns
        if 'max_xy_fit_residual' not in self.parameters:
            self.parameters['max_xy_fit_residual'] = 0.5       # pixels
        if 'min_xy_planes_kept' not in self.parameters:
            self.parameters['min_xy_planes_kept'] = 0.5        # fraction
        if 'stack_num_threads' not in self.parameters:
            # Threads over DISTINCT FIDUCIAL STACKS. per_z_slice_num_threads
            # only fans out sampled planes, and the z pass has few of those --
            # 10 on a 159-plane dataset at z_sample_step 16 -- so 16 per-z
            # threads sit at 62% occupancy while the stacks queue up behind
            # each other. 1 keeps the previous serial-over-stacks behaviour.
            # Total concurrency is stack_num_threads x per_z_slice_num_threads,
            # so keep the product near the cores actually allocated.
            self.parameters['stack_num_threads'] = 1
        if 'per_z_slice_num_threads' not in self.parameters:
            self.parameters['per_z_slice_num_threads'] = 1

        self._planeCache = {}
        self._planeCacheLock = threading.Lock()
        # rfft2 of the FIXED plane, which is channel 0 and therefore identical
        # for every data channel at a given plane index. Without this it is
        # recomputed once per channel per sampled plane -- 13 stacks x 27
        # sampled planes of redundant 2048^2 transforms per fov.
        self._fftCache = {}
        self._fftCacheLock = threading.Lock()

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 8192

    def get_estimated_time(self):
        return 60

    def get_dependencies(self):
        return []

    # ==================================================================
    # filtering and correlation primitives
    # ==================================================================

    def _filter(self, inputImage: np.ndarray) -> np.ndarray:
        """High-pass the fiducial frame and suppress everything that does not
        carry alignable texture."""
        highPassSigma = self.parameters['highpass_sigma']
        image = np.asarray(inputImage)

        if self.parameters['median_filter']:
            image = cv2.medianBlur(image.astype(np.uint16), ksize=3)

        if highPassSigma is None:
            filtered = image.astype(np.float64)
        else:
            filterSize = int(2 * np.ceil(2 * highPassSigma) + 1)
            filtered = image.astype(np.float64) - cv2.GaussianBlur(
                image, (filterSize, filterSize), highPassSigma,
                borderType=cv2.BORDER_REPLICATE)

        edgeWidth = self.parameters['edge_width_to_remove']
        if edgeWidth > 0:
            filtered[:edgeWidth] = 0
            filtered[-edgeWidth:] = 0
            filtered[:, :edgeWidth] = 0
            filtered[:, -edgeWidth:] = 0

        percentile = self.parameters['zero_below_percentile']
        if percentile is not None and percentile > 0:
            filtered[filtered < np.percentile(filtered, percentile)] = 0

        return filtered

    def _fiducial_plane(self, dataChannel: int, fov: int,
                        planeIndex: int) -> np.ndarray:
        """Filtered plane planeIndex of the 3D fiducial stack, cached.

        The stack is read a frame at a time rather than through
        get_fiducial3D_stack: a 399 plane stack is 3.3 GB as uint16 and twice
        that once filtered, and only a handful of planes are live at once.
        """
        key = (dataChannel, fov, planeIndex)
        cached = self._planeCache.get(key)
        if cached is not None:
            return cached

        dataOrganization = self.dataSet.get_data_organization()
        frameIndices = dataOrganization.get_fiducial3D_stack_frame_indices(
            dataChannel)
        planeIndex = int(np.clip(planeIndex, 0, len(frameIndices) - 1))
        key = (dataChannel, fov, planeIndex)
        cached = self._planeCache.get(key)
        if cached is not None:
            return cached

        rawPlane = self.dataSet.load_image(
            dataOrganization.get_fiducial3D_filename(dataChannel, fov),
            frameIndices[planeIndex])
        filtered = self._filter(rawPlane).astype(np.float32)
        # Filtering happens outside the lock, so two threads may occasionally
        # compute the same plane; that is wasted work, not a wrong answer. The
        # lock only covers the eviction check, which is the part that is not
        # atomic. plane_cache_size may need raising when threading: many more
        # planes are in flight at once.
        with self._planeCacheLock:
            if len(self._planeCache) >= self.parameters['plane_cache_size']:
                self._planeCache.clear()
            self._planeCache[key] = filtered
        return filtered

    def _fiducial_plane_at_z(self, dataChannel: int, fov: int,
                             zPosition: float) -> np.ndarray:
        """Filtered fiducial plane at an arbitrary z, linearly interpolated
        between the two bracketing planes."""
        zPositions = np.asarray(
            self.dataSet.get_data_organization()
            .get_fiducial3D_stack_frame_zPos(dataChannel), dtype=float)
        planeCount = len(zPositions)
        fractionalIndex = np.interp(zPosition, zPositions,
                                    np.arange(planeCount, dtype=float))
        lowerIndex = int(np.clip(np.floor(fractionalIndex), 0, planeCount - 2))
        weight = float(fractionalIndex - lowerIndex)
        if weight <= 1e-9:
            return self._fiducial_plane(dataChannel, fov, lowerIndex)
        return ((1.0 - weight) * self._fiducial_plane(
                    dataChannel, fov, lowerIndex)
                + weight * self._fiducial_plane(
                    dataChannel, fov, lowerIndex + 1))

    @staticmethod
    def _shift_image(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
        """Translate an image by (dy, dx), filling with zeros."""
        return cv2.warpAffine(
            np.ascontiguousarray(image, dtype=np.float32),
            np.float32([[1, 0, dx], [0, 1, dy]]),
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR, borderValue=0)

    def _zncc(self, first: np.ndarray, second: np.ndarray) -> float:
        """Normalised cross-correlation over the jointly non-zero pixels."""
        overlap = (first != 0) & (second != 0)
        if overlap.sum() < self.parameters['minimum_overlap_pixels']:
            return np.nan
        a = first[overlap] - first[overlap].mean()
        b = second[overlap] - second[overlap].mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    @staticmethod
    def _parabolic_vertex(left: float, centre: float, right: float) -> float:
        """Sub-sample offset of the vertex of the parabola through three
        equally spaced samples, clipped to the central interval."""
        denominator = left - 2 * centre + right
        if abs(denominator) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -1, 1))

    def _fixed_fft(self, fixedImage: np.ndarray, fixedKey):
        """rfft2 of the fixed plane, cached on fixedKey.

        Exact: the same transform, computed once instead of once per channel.
        fixedKey=None disables the cache.
        """
        if fixedKey is None:
            return np.fft.rfft2(fixedImage)
        cached = self._fftCache.get(fixedKey)
        if cached is not None:
            return cached
        transformed = np.fft.rfft2(fixedImage)
        with self._fftCacheLock:
            if len(self._fftCache) >= self.parameters['plane_cache_size']:
                self._fftCache.clear()
            self._fftCache[fixedKey] = transformed
        return transformed

    def _bounded_correlation_peak(self, fixedImage: np.ndarray,
                                  movingImage: np.ndarray,
                                  centreY: float, centreX: float,
                                  fixedKey=None, bound=None):
        """Peak of the cross-correlation of two frames, searched only within
        xy_search_bound pixels of (centreY, centreX).

        Returns (dy, dx, snr) where (dy, dx) is the shift to apply to
        movingImage to bring it onto fixedImage.
        """
        correlation = np.fft.irfft2(
            self._fixed_fft(fixedImage, fixedKey)
            * np.conj(np.fft.rfft2(movingImage)),
            s=fixedImage.shape)
        height, width = correlation.shape

        if bound is None:
            bound = self.parameters['xy_search_bound']
        rows = np.arange(int(np.floor(centreY)) - bound,
                         int(np.ceil(centreY)) + bound + 1)
        columns = np.arange(int(np.floor(centreX)) - bound,
                            int(np.ceil(centreX)) + bound + 1)
        window = correlation[np.mod(rows, height)[:, None],
                             np.mod(columns, width)[None, :]]

        i, j = np.unravel_index(np.argmax(window), window.shape)
        peakY, peakX = rows[i], columns[j]

        # Refine against the full surface so the parabola is not distorted by
        # the window edge, but only when the peak is interior to the window.
        subY = subX = 0.0
        if 0 < i < window.shape[0] - 1:
            subY = self._parabolic_vertex(
                correlation[(peakY - 1) % height, peakX % width],
                correlation[peakY % height, peakX % width],
                correlation[(peakY + 1) % height, peakX % width])
        if 0 < j < window.shape[1] - 1:
            subX = self._parabolic_vertex(
                correlation[peakY % height, (peakX - 1) % width],
                correlation[peakY % height, peakX % width],
                correlation[peakY % height, (peakX + 1) % width])

        snr = float((window.max() - correlation.mean())
                    / (correlation.std() + 1e-12))
        onEdge = bool(i in (0, window.shape[0] - 1)
                      or j in (0, window.shape[1] - 1))
        return peakY + subY, peakX + subX, snr, onEdge

    def _robust_polyfit(self, z: np.ndarray, values: np.ndarray,
                        order: int) -> np.ndarray:
        """Polynomial fit with iterative sigma trimming.

        Returns the coefficients in np.polyval order. Points that survive the
        trimming are the ones the fit is based on; the trimmed ones are
        recorded in the QC table by the caller.
        """
        z = np.asarray(z, dtype=float)
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(z) & np.isfinite(values)
        if finite.sum() < order + 2:
            return np.zeros(order + 1), finite

        kept = finite.copy()
        for _ in range(self.parameters['robust_trim_iterations']):
            if kept.sum() < order + 2:
                break
            coefficients = np.polyfit(z[kept], values[kept], order)
            residuals = values - np.polyval(coefficients, z)
            scale = 1.4826 * np.median(
                np.abs(residuals[kept] - np.median(residuals[kept]))) + 1e-9
            candidate = finite & (np.abs(residuals)
                                  < self.parameters['robust_trim_sigma'] * scale)
            if candidate.sum() < order + 2:
                break
            kept = candidate

        if kept.sum() < order + 2:
            kept = finite
        return np.polyfit(z[kept], values[kept], order), kept

    # ==================================================================
    # measurement
    # ==================================================================

    def _map_over_planes(self, function, planeIndices):
        """Apply function to each sampled plane, in order, possibly threaded."""
        threads = max(1, int(self.parameters['per_z_slice_num_threads']))
        if threads > 1 and len(planeIndices) > 1:
            with ThreadPoolExecutor(
                    max_workers=min(threads, len(planeIndices))) as pool:
                return list(pool.map(function, planeIndices))
        return [function(p) for p in planeIndices]

    def _rigid_only_channel_set(self):
        """Channels to register with a rigid xy offset and NO drift polynomial.

        A channel acquired in a SEPARATE stack from the bit rounds -- polyT and
        DAPI on 20260609 live in their own nuclear dax -- has fiducial beads
        only near the coverslip. Measured on 20260609: the nuclear 488 plane
        correlates against the bit-round fiducial at snr ~350 from z = 1 um to
        at least z = 5 um, and at 6.1 by z = 41 um, while the tissue spans
        80 um. _measure_z_offsets samples every 16th plane across the whole
        slab, so the great majority of its samples would be beadless noise and
        the fitted polynomial would be garbage rather than merely imprecise.

        The rigid offset and the plane WALK still run, and both are worth
        having: on the five fovs whose 2D nuclear frame is weak the walk moves
        the answer by up to 3.8 px and raises the confidence from snr 36 to
        320. Only the polynomial is skipped; its coefficients are written as
        zero.

        Accepts channel names or indices. Empty by default.
        """
        requested = self.parameters['rigid_only_channels']
        if not requested:
            return set()
        dataOrganization = self.dataSet.get_data_organization()
        resolved = set()
        for entry in requested:
            if isinstance(entry, str):
                resolved.add(int(
                    dataOrganization.get_data_channel_index(entry)))
            else:
                resolved.add(int(entry))
        return resolved

    def _measure_rigid_offsets(self, fov: int):
        """Rigid xy per data channel, from the 2D fiducial frame or, if
        rigid_plane_search is on, from whichever early plane correlates best.

        The search is BOUNDED at rigid_search_bound about zero. An unbounded
        peak search on a dense self-similar bead field regularly locks onto a
        secondary maximum of the autocorrelation -- measured on 20260609, two
        fovs returned 626 and 953 px against a true 34 px. Bounding alone
        recovers one of them and turns the other into an honest zero.
        """
        bound = self.parameters['rigid_search_bound']
        search = int(self.parameters['rigid_plane_search'])
        step = max(1, int(self.parameters['rigid_plane_step']))
        ratio = float(self.parameters['rigid_snr_ratio'])
        fixed2D = self._filter(self.dataSet.get_fiducial_image(0, fov))
        offsets = []
        for dataChannel in self.dataSet.get_data_organization()\
                .get_data_channels():
            moving2D = self._filter(
                self.dataSet.get_fiducial_image(dataChannel, fov))
            dy, dx, snr, _ = self._bounded_correlation_peak(
                fixed2D, moving2D, 0.0, 0.0, bound=bound)
            best = (dy, dx, snr, -1)
            plane = 0
            first = True
            while search and plane < search:
                try:
                    f = self._fiducial_plane(0, fov, plane)
                    m = self._fiducial_plane(dataChannel, fov, plane)
                except Exception:
                    break
                # The FIRST plane searches wide about the ORIGIN, not about the
                # 2D answer, because the whole point is to be able to overrule a
                # bad 2D answer -- fov52's is wrong by 57 px. Every later plane
                # searches TIGHTLY about the running best, because consecutive
                # planes must agree: without that cap each step could move a
                # full `bound` and the walk drifts. fov123 channel 27 reached
                # 247 px that way, which then wrecked its polynomial fit
                # (z residual 1.03 um against a normal 0.03).
                if first:
                    cy, cx, csnr, _ = self._bounded_correlation_peak(
                        f, m, 0.0, 0.0, bound=bound)
                else:
                    cy, cx, csnr, _ = self._bounded_correlation_peak(
                        f, m, best[0], best[1],
                        bound=self.parameters['rigid_walk_bound'])
                if csnr <= ratio * best[2]:
                    break            # no longer clearly better: stop walking
                best = (cy, cx, csnr, plane)
                first = False
                plane += step
            if best[3] >= 0:
                print('fov %d channel %d: rigid offset taken from plane %d '
                      '(snr %.0f) rather than the 2D frame (snr %.0f)'
                      % (fov, dataChannel, best[3], best[2], snr), flush=True)
            offsets.append((float(best[0]), float(best[1])))
        return offsets

    def _measure_z_offsets(self, fov: int, dataChannel: int,
                           rigidY: float, rigidX: float):
        """Per-plane z offset in microns, measured at the rigid xy."""
        dataOrganization = self.dataSet.get_data_organization()
        zPositions = np.asarray(
            dataOrganization.get_fiducial3D_stack_frame_zPos(dataChannel),
            dtype=float)
        planeCount = len(zPositions)
        planeSpacing = float(np.median(np.diff(zPositions))) if planeCount > 1 \
            else 1.0

        searchRange = self.parameters['z_search_range']
        planeIndices = list(range(searchRange + 1,
                                  planeCount - searchRange - 1,
                                  self.parameters['z_sample_step']))

        def measure_one(planeIndex):
            fixedImage = self._fiducial_plane(0, fov, planeIndex)
            scores = []
            for delta in range(-searchRange, searchRange + 1):
                movingImage = self._fiducial_plane(
                    dataChannel, fov, planeIndex + delta)
                scores.append(self._zncc(
                    fixedImage,
                    self._shift_image(movingImage, rigidY, rigidX)))
            scores = np.where(np.isfinite(scores), scores, -1.0)

            best = int(np.argmax(scores))
            sub = 0.0
            if 0 < best < len(scores) - 1:
                sub = self._parabolic_vertex(
                    scores[best - 1], scores[best], scores[best + 1])
            return (zPositions[planeIndex],
                    (best - searchRange + sub) * planeSpacing,
                    float(scores[best]))

        results = self._map_over_planes(measure_one, planeIndices)
        measuredZ, measuredOffset, measuredScore = (
            zip(*results) if results else ((), (), ()))
        # The ceiling is returned rather than recomputed by the caller so the
        # gate can never disagree with the search that produced the numbers.
        return (np.array(measuredZ), np.array(measuredOffset),
                np.array(measuredScore), searchRange * planeSpacing)

    def _z_sample_gate(self, offsets: np.ndarray, scores: np.ndarray,
                       ceiling: float) -> np.ndarray:
        """Which z samples may enter the fit. See the class docstring.

        Two independent failure modes, and the score alone does not catch the
        second: a railed argmax is still scored against whatever tissue sits
        at the edge of the search, which can correlate perfectly well.
        """
        good = np.isfinite(offsets) & np.isfinite(scores)
        minScore = float(self.parameters['z_min_score'] or 0.0)
        if minScore > 0:
            good &= scores >= minScore
        if self.parameters['z_reject_at_search_limit'] and ceiling > 0:
            good &= np.abs(offsets) < ceiling - 1e-6
        return good

    def _measure_xy_offsets(self, fov: int, dataChannel: int,
                            rigidY: float, rigidX: float,
                            zCoefficients: np.ndarray):
        """Per-plane xy offset, measured after the z correction is applied."""
        dataOrganization = self.dataSet.get_data_organization()
        zPositions = np.asarray(
            dataOrganization.get_fiducial3D_stack_frame_zPos(dataChannel),
            dtype=float)
        planeCount = len(zPositions)
        planeIndices = list(range(1, planeCount - 1,
                                  self.parameters['xy_sample_step']))

        def measure_one(planeIndex):
            zPosition = zPositions[planeIndex]
            zOffset = float(np.polyval(zCoefficients, zPosition))
            fixedImage = self._fiducial_plane(0, fov, planeIndex)
            movingImage = self._fiducial_plane_at_z(
                dataChannel, fov, zPosition + zOffset)
            dy, dx, snr, onEdge = self._bounded_correlation_peak(
                fixedImage, movingImage, rigidY, rigidX, fixedKey=planeIndex)
            return zPosition, dy, dx, snr, onEdge

        results = self._map_over_planes(measure_one, planeIndices)
        measuredZ, measuredY, measuredX, measuredSnr, measuredEdge = (
            zip(*results) if results else ((), (), (), (), ()))
        return (np.array(measuredZ), np.array(measuredY), np.array(measuredX),
                np.array(measuredSnr), np.array(measuredEdge, dtype=bool))

    # ==================================================================
    # storage
    # ==================================================================

    def _save_results(self, fov: int, coefficients, measurements) -> None:
        dataOrganization = self.dataSet.get_data_organization()
        dataChannels = list(dataOrganization.get_data_channels())
        zPositions = np.asarray(self.dataSet.get_z_positions(), dtype=float)

        # Rigid transforms, in the same format the other warp tasks use, so
        # anything that only wants a per-channel offset still works.
        rigidTransformations = [
            transform.SimilarityTransform(
                translation=[-coefficients[c]['rigid_x'],
                             -coefficients[c]['rigid_y']])
            for c in dataChannels]
        self._save_transformations(rigidTransformations, fov)

        coefficientRows = []
        for dataChannel in dataChannels:
            entry = coefficients[dataChannel]
            row = {'dataChannel': dataChannel,
                   'rigid_y': entry['rigid_y'],
                   'rigid_x': entry['rigid_x']}
            for name in ('z', 'y', 'x'):
                for order, value in enumerate(entry[name][::-1]):
                    row['%s_c%d' % (name, order)] = value
            coefficientRows.append(row)
        self.dataSet.save_dataframe_to_csv(
            pd.DataFrame(coefficientRows), 'polynomial_coefficients',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations', index=False)

        tableRows = []
        for dataChannel in dataChannels:
            entry = coefficients[dataChannel]
            zOffset = np.polyval(entry['z'], zPositions)
            shiftY = np.polyval(entry['y'], zPositions)
            shiftX = np.polyval(entry['x'], zPositions)
            tableRows.append(pd.DataFrame({
                'dataChannel': dataChannel,
                'zIndex': np.arange(len(zPositions)),
                'zPos': zPositions,
                'zPos_new': zPositions + zOffset,
                'z_offset': zOffset,
                # stored as transform.SimilarityTransform translation
                # parameters, i.e. the negated measured shift
                'xshift': -shiftX,
                'yshift': -shiftY}))
        self.dataSet.save_dataframe_to_csv(
            pd.concat(tableRows, ignore_index=True), 'transformation_table',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations', index=False)

        if self.parameters['write_qc_table'] and len(measurements):
            self.dataSet.save_dataframe_to_csv(
                pd.concat(measurements, ignore_index=True), 'measurements',
                self.get_analysis_name(), resultIndex=fov,
                subdirectory='transformations', index=False)

    def get_transformation_table(self, fov: int) -> pd.DataFrame:
        return self.dataSet.load_dataframe_from_csv(
            'transformation_table', self, resultIndex=fov,
            subdirectory='transformations')

    def get_polynomial_coefficients(self, fov: int) -> pd.DataFrame:
        return self.dataSet.load_dataframe_from_csv(
            'polynomial_coefficients', self, resultIndex=fov,
            subdirectory='transformations')

    def get_z_fit_status(self, fov: int) -> pd.DataFrame:
        return self.dataSet.load_dataframe_from_csv(
            'z_fit_status', self, resultIndex=fov,
            subdirectory='transformations')

    # ==================================================================
    # cross-fov repair
    # ==================================================================

    def finalize(self) -> None:
        """Give every unfitted z stack the curve of its neighbouring fovs.

        Runs once, from `merlin -t <task> --check-done`, which is the only
        point in the DAG where every fragment is finished and nothing
        downstream has started. It has to be here rather than in
        _run_analysis: a fragment knows one fov, and the donors are other
        fovs that may not have been solved yet when it ran.

        Idempotent, as finalize() must be: the decision of WHICH stacks to
        repair comes from z_fit_status, which records the gate result and is
        never rewritten, and the donors are by definition stacks that fitted,
        whose coefficients this never touches. Running it twice writes the
        same numbers.
        """
        if not self.parameters['z_neighbor_substitution']:
            return

        fovs = list(self.dataSet.get_fovs())
        try:
            status = pd.concat([self.get_z_fit_status(f) for f in fovs],
                               ignore_index=True)
        except FileNotFoundError:
            print('%s: no z_fit_status tables, nothing to substitute'
                  % self.get_analysis_name(), flush=True)
            return
        needed = status[status.source == 'unfitted']
        if not len(needed):
            print('%s: every z stack fitted, no neighbour substitution needed'
                  % self.get_analysis_name(), flush=True)
            return

        zPositions = np.asarray(self.dataSet.get_z_positions(), dtype=float)
        zOrder = self.parameters['z_polynomial_order']
        minSamples = int(self.parameters['z_neighbor_min_samples'])
        minDonors = int(self.parameters['z_neighbor_min_donors'])
        rings = [float(r) for r in self.parameters['z_neighbor_rings_um']]
        xy = {f: self.dataSet.get_fov_offset(f) for f in fovs}
        coeffCache = {}

        def coefficients_of(fov):
            if fov not in coeffCache:
                coeffCache[fov] = self.get_polynomial_coefficients(
                    fov).set_index('dataChannel')
            return coeffCache[fov]

        def z_curve(fov, dataChannel):
            row = coefficients_of(fov).loc[dataChannel]
            poly = [row['z_c%d' % k]
                    for k in range(zOrder, -1, -1) if 'z_c%d' % k in row]
            return np.polyval(poly, zPositions)

        fitted = {(int(r.fov), int(r.dataChannel)): int(r.n_good)
                  for r in status.itertuples()
                  if r.source == 'fit' and int(r.n_good) >= minSamples}

        records = []
        for target in needed.itertuples():
            fov, dataChannel = int(target.fov), int(target.dataChannel)
            tx, ty = xy[fov]
            donors, usedRing = [], rings[-1]
            for ring in rings:
                donors = [f for f in fovs
                          if f != fov and (f, dataChannel) in fitted
                          and np.hypot(xy[f][0] - tx, xy[f][1] - ty) <= ring]
                if len(donors) >= minDonors:
                    usedRing = ring
                    break
            if not donors:
                print('  fov %d channel %d: NO usable donor within %.0f um, '
                      'left unfitted' % (fov, dataChannel, rings[-1]),
                      flush=True)
                continue
            # Pointwise median of the donors' curves, then re-fit at the
            # task's own order. Order-agnostic, and for the order-1 default it
            # reproduces the median-slope / median-intercept estimator that
            # the leave-one-out validation was run on.
            curve = np.median(np.vstack([z_curve(f, dataChannel)
                                         for f in donors]), axis=0)
            newPoly = np.polyfit(zPositions, curve, zOrder)
            records.append(dict(
                fov=fov, dataChannel=dataChannel, n_donors=len(donors),
                ring_um=usedRing, donors=','.join(str(f) for f in sorted(donors)),
                span_um=float(np.ptp(np.polyval(newPoly, zPositions))),
                coefficients=newPoly))

        if not records:
            print('%s: %d unfitted stacks but no donors for any of them'
                  % (self.get_analysis_name(), len(needed)), flush=True)
            return

        byFov = OrderedDict()
        for rec in records:
            byFov.setdefault(rec['fov'], []).append(rec)
        for fov, recs in byFov.items():
            coeff = self.get_polynomial_coefficients(fov).set_index('dataChannel')
            for rec in recs:
                for k, value in enumerate(rec['coefficients'][::-1]):
                    coeff.loc[rec['dataChannel'], 'z_c%d' % k] = value
            coeff = coeff.reset_index()
            self.dataSet.save_dataframe_to_csv(
                coeff, 'polynomial_coefficients', self.get_analysis_name(),
                resultIndex=fov, subdirectory='transformations', index=False)
            self._rewrite_transformation_table(fov, coeff, zPositions, zOrder)
            st = self.get_z_fit_status(fov)
            touched = {r['dataChannel'] for r in recs}
            st.loc[st.dataChannel.isin(touched), 'source'] = 'neighbor'
            self.dataSet.save_dataframe_to_csv(
                st, 'z_fit_status', self.get_analysis_name(), resultIndex=fov,
                subdirectory='transformations', index=False)
            coeffCache.pop(fov, None)

        report = pd.DataFrame([{k: v for k, v in rec.items()
                                if k != 'coefficients'} for rec in records])
        self.dataSet.save_dataframe_to_csv(
            report, 'z_neighbor_substitutions', self.get_analysis_name(),
            subdirectory='transformations', index=False)
        print('%s: substituted %d z stacks in %d fovs from their neighbours'
              % (self.get_analysis_name(), len(records), len(byFov)),
              flush=True)
        print(report.to_string(index=False), flush=True)

    def _rewrite_transformation_table(self, fov, coeff, zPositions, zOrder):
        """Recompute only the z columns of an existing transformation table.

        xy is deliberately left alone. It was measured at the plane the
        UNFITTED (zero) z curve pointed at, and re-measuring it at the
        substituted depth would mean going back to the images from finalize().
        It is not worth it: the fitted xy drift spans about 2.6 px over this
        dataset's 80 um, so the largest substitution seen (1.1 um, two planes)
        moves the xy by ~0.03 px -- two orders of magnitude below the
        measurement noise. This is also exactly the construction that the
        end-to-end decode comparison was run on.
        """
        table = self.get_transformation_table(fov)
        byChannel = coeff.set_index('dataChannel')
        for dataChannel in table['dataChannel'].unique():
            row = byChannel.loc[dataChannel]
            poly = [row['z_c%d' % k]
                    for k in range(zOrder, -1, -1) if 'z_c%d' % k in row]
            select = table['dataChannel'] == dataChannel
            zPos = table.loc[select, 'zPos'].to_numpy(float)
            offset = np.polyval(poly, zPos)
            table.loc[select, 'z_offset'] = offset
            table.loc[select, 'zPos_new'] = zPos + offset
        self.dataSet.save_dataframe_to_csv(
            table, 'transformation_table', self.get_analysis_name(),
            resultIndex=fov, subdirectory='transformations', index=False)

    def get_transformation(self, fov: int, dataChannel: int = None,
                           zIndex: int = None):
        """Get the xy transformation for a field of view.

        With a dataChannel and a zIndex, the transformation for that plane.
        With a dataChannel only, the transformation at the middle plane, which
        is the closest thing to a single representative offset for the round.
        With neither, the transformations for every (channel, plane) pair.
        Decode's adaptive crop consumes that last form and takes the extreme
        translation over the whole list, which is exactly the union of the
        invalid margins over depth.
        """
        table = self.get_transformation_table(fov)

        def _transform(row):
            return transform.SimilarityTransform(
                translation=[row['xshift'], row['yshift']])

        if dataChannel is not None:
            channelTable = table[table['dataChannel'] == dataChannel]
            if zIndex is None:
                zIndex = len(channelTable) // 2
            row = channelTable[channelTable['zIndex'] == zIndex]
            if len(row) == 0:
                row = channelTable.iloc[[min(zIndex, len(channelTable) - 1)]]
            return _transform(row.iloc[0])

        if zIndex is not None:
            planeTable = table[table['zIndex'] == zIndex]
            return [_transform(r) for _, r in planeTable.iterrows()]

        return [_transform(r) for _, r in table.iterrows()]

    # ==================================================================
    # application
    # ==================================================================

    def get_aligned_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        """Get the specified image, resampled to the corrected depth and
        translated by the fitted xy offset."""
        table = self.get_transformation_table(fov)
        row = table[(table['dataChannel'] == dataChannel)
                    & (table['zIndex'] == zIndex)]
        if len(row) == 0:
            raise Exception(
                'No transformation for fov %d channel %d zIndex %d'
                % (fov, dataChannel, zIndex))
        row = row.iloc[0]

        zPositions = np.asarray(self.dataSet.get_z_positions(), dtype=float)
        newZPosition = float(np.clip(row['zPos_new'],
                                     zPositions.min(), zPositions.max()))

        if self.parameters['interpolation'] == 'nearest':
            nearest = zPositions[np.abs(zPositions - newZPosition).argmin()]
            inputImage = self.dataSet.get_raw_image(dataChannel, fov, nearest)
        elif self.parameters['interpolation'] == 'weighted':
            order = np.abs(zPositions - newZPosition).argsort()
            nearZ = zPositions[order[:2]]
            distances = np.abs(newZPosition - nearZ)
            if distances.sum() < 1e-10:
                inputImage = self.dataSet.get_raw_image(
                    dataChannel, fov, nearZ[0])
            else:
                weights = 1.0 - distances / distances.sum()
                first = self.dataSet.get_raw_image(
                    dataChannel, fov, nearZ[0])
                second = self.dataSet.get_raw_image(
                    dataChannel, fov, nearZ[1])
                inputImage = (first.astype(np.float32) * weights[0]
                              + second.astype(np.float32) * weights[1]
                              ).astype(first.dtype)
        else:
            raise ValueError('Unknown interpolation: %s'
                             % self.parameters['interpolation'])

        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization()\
                .get_data_channel_color(dataChannel)
            inputImage = chromaticCorrector.transform_image(
                inputImage, imageColor).astype(inputImage.dtype)

        transformation = transform.SimilarityTransform(
            translation=[row['xshift'], row['yshift']])
        warpedImage = transform.warp(inputImage, transformation,
                                     preserve_range=True)

        if self.parameters['boundary_smooth']:
            blurred = cv2.GaussianBlur(
                transform.warp(inputImage, transformation,
                               preserve_range=True, mode='edge'),
                ksize=(23, 23), sigmaX=11, borderType=cv2.BORDER_REPLICATE)
            mask = morphology.binary_dilation(warpedImage == 0)
            warpedImage[mask] = blurred[mask]

        return warpedImage.astype(inputImage.dtype)

    # ==================================================================
    # main analysis
    # ==================================================================

    def _process_transformations(self, transformationList, fov) -> None:
        """Write the requested fiducial and aligned image stacks.

        The base implementation warps with one transformation per channel,
        which cannot express a per-plane offset, so aligned images go through
        get_aligned_image instead.
        """
        dataChannels = self.dataSet.get_data_organization().get_data_channels()

        alignedFOVs = self.parameters['write_aligned_FOVs']
        alignedZ = self.parameters['write_aligned_z']
        if self.parameters['write_aligned_images'] \
                and (alignedFOVs == [-1] or fov in alignedFOVs):
            zPositions = self.dataSet.get_z_positions()
            imageDescription = self.dataSet.analysis_tiff_description(
                len(zPositions), len(dataChannels))
            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_images', fov) as outputTif:
                for dataChannel in dataChannels:
                    for zIndex in range(len(zPositions)):
                        if alignedZ is not None and zIndex not in alignedZ:
                            continue
                        outputTif.write(
                            self.get_aligned_image(fov, dataChannel, zIndex),
                            photometric='MINISBLACK', contiguous=True,
                            metadata=imageDescription)

        if self.parameters['write_fiducial_images']:
            fiducialFOVs = self.parameters['write_fiducial_FOVs']
            if not isinstance(fiducialFOVs, list):
                fiducialFOVs = list(fiducialFOVs)
            if fiducialFOVs == [-1] or fov in fiducialFOVs:
                fiducialImageDescription = \
                    self.dataSet.analysis_tiff_description(
                        1, len(dataChannels))
                with self.dataSet.writer_for_analysis_images(
                        self, 'fiducial_images', fov) as outputTif:
                    for t, dataChannel in zip(transformationList,
                                              dataChannels):
                        inputImage = self.dataSet.get_fiducial_image(
                            dataChannel, fov)
                        outputTif.write(
                            transform.warp(inputImage, t, preserve_range=True)
                            .astype(inputImage.dtype),
                            photometric='MINISBLACK', contiguous=True,
                            metadata=fiducialImageDescription)

    def _run_analysis(self, fragmentIndex: int):
        fov = list(self.dataSet.get_fovs())[fragmentIndex]
        dataOrganization = self.dataSet.get_data_organization()
        dataChannels = list(dataOrganization.get_data_channels())

        zOrder = self.parameters['z_polynomial_order']
        xyOrder = self.parameters['xy_polynomial_order']

        rigidOffsets = self._measure_rigid_offsets(fov)

        # Data channels imaged in the same round share a fiducial stack, so
        # the drift only has to be measured once per distinct stack.
        # Group channels by fiducial stack FIRST. Channels sharing a stack
        # share its drift, so only one measurement per distinct stack is made
        # -- 11 for 21 bits on 0103, 13 for 27 bits on 0609. Grouping up front
        # (rather than skipping duplicates inside the loop) is what lets the
        # stacks be solved concurrently.
        rigidOnly = self._rigid_only_channel_set()
        stackMembers = OrderedDict()
        rigidOnlyMembers = []
        for dataChannel, (rigidY, rigidX) in zip(dataChannels, rigidOffsets):
            if dataChannel in rigidOnly:
                rigidOnlyMembers.append((dataChannel, rigidY, rigidX))
                continue
            key = dataOrganization.get_fiducial3D_filename(dataChannel, fov)
            stackMembers.setdefault(key, []).append(
                (dataChannel, rigidY, rigidX))

        def solve_stack(item):
            key, members = item
            dataChannel, rigidY, rigidX = members[0]
            zSampleZ, zSampleOffset, zSampleScore, zCeiling = \
                self._measure_z_offsets(fov, dataChannel, rigidY, rigidX)
            zGate = self._z_sample_gate(zSampleOffset, zSampleScore, zCeiling)
            nGood = int(zGate.sum())
            zFitted = nGood >= int(self.parameters['z_min_good_samples'] or 0)
            if zFitted:
                zCoefficients, gateKept = self._robust_polyfit(
                    zSampleZ[zGate], zSampleOffset[zGate], zOrder)
                # gateKept indexes the gated subset; lift it back so the QC
                # table keeps one row per sample and a gated-out sample reads
                # as not kept rather than vanishing.
                zKept = np.zeros(len(zGate), bool)
                zKept[np.flatnonzero(zGate)] = gateKept
            else:
                # No depth information worth having. Zero coefficients mean
                # this round falls back to its rigid xy, and finalize() may
                # replace the curve from the neighbouring fovs.
                zCoefficients = np.zeros(zOrder + 1)
                zKept = np.zeros(len(zGate), bool)

            xySampleZ, xySampleY, xySampleX, xySnr, xyEdge = \
                self._measure_xy_offsets(fov, dataChannel, rigidY, rigidX,
                                         zCoefficients)
            yCoefficients, yKept = self._robust_polyfit(
                xySampleZ, xySampleY, xyOrder)
            xCoefficients, xKept = self._robust_polyfit(
                xySampleZ, xySampleX, xyOrder)

            entry = {'z': zCoefficients, 'y': yCoefficients,
                     'x': xCoefficients}
            status = {'fov': fov, 'dataChannel': dataChannel,
                      'n_samples': int(len(zGate)), 'n_good': nGood,
                      'n_gated_out': int(len(zGate) - nGood),
                      'fitted': bool(zFitted),
                      'source': 'fit' if zFitted else 'unfitted'}

            stackMeasurements = []
            if self.parameters['write_qc_table']:
                measurements = stackMeasurements
                measurements.append(pd.DataFrame({
                    'dataChannel': dataChannel, 'stage': 'z',
                    'zPos': zSampleZ, 'measured': zSampleOffset,
                    'fitted': np.polyval(zCoefficients, zSampleZ),
                    'score': zSampleScore, 'kept': zKept,
                    'on_edge': False}))
                measurements.append(pd.DataFrame({
                    'dataChannel': dataChannel, 'stage': 'y',
                    'zPos': xySampleZ, 'measured': xySampleY,
                    'fitted': np.polyval(yCoefficients, xySampleZ),
                    'score': xySnr, 'kept': yKept, 'on_edge': xyEdge}))
                measurements.append(pd.DataFrame({
                    'dataChannel': dataChannel, 'stage': 'x',
                    'zPos': xySampleZ, 'measured': xySampleX,
                    'fitted': np.polyval(xCoefficients, xySampleZ),
                    'score': xySnr, 'kept': xKept, 'on_edge': xyEdge}))

            residualY = xySampleY - np.polyval(yCoefficients, xySampleZ)
            residualX = xySampleX - np.polyval(xCoefficients, xySampleZ)
            zResidual = float(np.median(np.abs(
                zSampleOffset - np.polyval(zCoefficients, zSampleZ))))
            yResidual = float(np.median(np.abs(residualY)))
            xResidual = float(np.median(np.abs(residualX)))
            keptFraction = (float(yKept.sum()) / len(xySampleZ)
                            if len(xySampleZ) else 0.0)
            line = ('fov %d channel %d: z fit residual %.3f um, '
                    'xy fit residual %.3f / %.3f px, %d/%d xy planes kept'
                    % (fov, dataChannel, zResidual, yResidual, xResidual,
                       int(yKept.sum()), len(xySampleZ)))

            reasons = []
            if zResidual > self.parameters['max_z_fit_residual']:
                reasons.append('z residual %.3f um > %.3f'
                               % (zResidual, self.parameters['max_z_fit_residual']))
            if max(yResidual, xResidual) > self.parameters['max_xy_fit_residual']:
                reasons.append('xy residual %.3f/%.3f px > %.3f'
                               % (yResidual, xResidual,
                                  self.parameters['max_xy_fit_residual']))
            if keptFraction < self.parameters['min_xy_planes_kept']:
                reasons.append('only %.0f%% of xy planes kept' % (100 * keptFraction))
            flag = None
            if reasons:
                line += '\n  *** REGISTRATION SUSPECT, fov %d channel %d: %s. ' \
                        'The rigid offset this was fitted on top of is ' \
                        'probably wrong -- do not trust this channel here ' \
                        'without looking. ***' % (fov, dataChannel, '; '.join(reasons))
                flag = {'fov': fov, 'dataChannel': dataChannel,
                        'z_residual_um': zResidual, 'y_residual_px': yResidual,
                        'x_residual_px': xResidual, 'fraction_kept': keptFraction,
                        'reasons': '; '.join(reasons)}
            if not zFitted:
                line += ('\n  *** Z NOT FITTED, fov %d channel %d: only %d of '
                         '%d samples passed the gate (min %s). z_offset is 0 '
                         'here; finalize() will substitute the neighbouring '
                         'fovs curve if z_neighbor_substitution is on. ***'
                         % (fov, dataChannel, nGood, len(zGate),
                            self.parameters['z_min_good_samples']))
            return key, entry, stackMeasurements, line, flag, status

        stackThreads = max(1, int(self.parameters['stack_num_threads']))
        items = list(stackMembers.items())
        if stackThreads > 1 and len(items) > 1:
            with ThreadPoolExecutor(
                    max_workers=min(stackThreads, len(items))) as pool:
                solvedStacks = list(pool.map(solve_stack, items))
        else:
            solvedStacks = [solve_stack(i) for i in items]

        # Reassemble in the ORIGINAL channel order, so the output is identical
        # whatever order the threads happened to finish in.
        coefficients = {}
        measurements = []
        entryOf = {key: entry for key, entry, _, _, _, _ in solvedStacks}
        qcFlags = []
        zStatus = []
        for key, _, stackMeasurements, line, flag, status in solvedStacks:
            zStatus.append(status)
            measurements.extend(stackMeasurements)
            print(line, flush=True)
            if flag is not None:
                qcFlags.append(flag)
        # Written whether or not anything tripped, so its ABSENCE never has to
        # be read as "the guard did not run".
        self.dataSet.save_dataframe_to_csv(
            pd.DataFrame(qcFlags, columns=[
                'fov', 'dataChannel', 'z_residual_um', 'y_residual_px',
                'x_residual_px', 'fraction_kept', 'reasons']),
            'fit_quality_flags', self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations', index=False)
        statusOf = {row['dataChannel']: row for row in zStatus}
        statusRows = []
        for key, members in stackMembers.items():
            owner = members[0][0]
            for dataChannel, rigidY, rigidX in members:
                coefficients[dataChannel] = dict(
                    entryOf[key], rigid_y=rigidY, rigid_x=rigidX)
                # Channels sharing a fiducial stack share its fit, so they
                # share its status; `owner` names the one actually measured.
                statusRows.append(dict(statusOf[owner], dataChannel=dataChannel,
                                       owner_channel=owner))
        for dataChannel, rigidY, rigidX in rigidOnlyMembers:
            coefficients[dataChannel] = {
                'z': np.zeros(zOrder + 1), 'y': np.zeros(xyOrder + 1),
                'x': np.zeros(xyOrder + 1),
                'rigid_y': rigidY, 'rigid_x': rigidX}
            print('fov %d channel %d: rigid only (%.2f, %.2f), no 3D warp '
                  'fitted' % (fov, dataChannel, rigidY, rigidX), flush=True)
            statusRows.append(dict(
                fov=fov, dataChannel=dataChannel, owner_channel=dataChannel,
                n_samples=0, n_good=0, n_gated_out=0, fitted=False,
                source='rigid_only'))

        # One row per data channel, always written, so finalize() can find the
        # stacks needing substitution without re-reading any measurement, and
        # so the ABSENCE of a row never has to be read as "the gate did not
        # run". source is 'fit', 'unfitted' or 'rigid_only'; finalize()
        # rewrites 'unfitted' to 'neighbor' when it substitutes one.
        self.dataSet.save_dataframe_to_csv(
            pd.DataFrame(statusRows, columns=[
                'fov', 'dataChannel', 'owner_channel', 'n_samples', 'n_good',
                'n_gated_out', 'fitted', 'source']).sort_values('dataChannel'),
            'z_fit_status', self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations', index=False)

        self._save_results(fov, coefficients, measurements)
        self._planeCache.clear()
        self._fftCache.clear()

        self._process_transformations(
            [transform.SimilarityTransform(
                translation=[-coefficients[c]['rigid_x'],
                             -coefficients[c]['rigid_y']])
             for c in dataChannels], fov)
