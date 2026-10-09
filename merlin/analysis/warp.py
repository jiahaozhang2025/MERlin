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
        if 'write_aligned_fovs' not in self.parameters:
            self.parameters['write_aligned_fovs'] = [-1]
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
        if 'write_fiducial_fovs' not in self.parameters:
            self.parameters['write_fiducial_fovs'] = [-1]

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

        _alignedFOVs = self.parameters['write_aligned_fovs']
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
            fiducialFOVs = self.parameters['write_fiducial_fovs']
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
            if self.parameters['write_fiducial_fovs'] == [-1]:
                self.parameters['write_fiducial_fovs'] = self.dataSet.get_fovs()

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

    IN-PLANE ROTATION, ONE ANGLE PER ROUND (rotation_task, default unset).
    The sample can turn slightly about the optical axis between rounds, and a
    translation cannot absorb that: the frame centre is aligned and the
    corners stay off by theta * r. The design takes the turn to come from
    the stage: the whole sample turns as one rigid body, so every fov of a
    round turns by the same angle, and the angle is not measured here.
    rotation_task names a FiducialRoundRotation task, which measures each
    fov's angle on one shallow bead plane and takes the median over fovs as
    the round's angle (see its docstring). What a fov's bead field does
    beyond that with depth (stretch, shear) is not treated as a rotation;
    affine_correction (below) models it as a linear part per fov. Per fov,
    every moving stack of a round is then handled as follows:

      t1. its fiducial planes and its 2D fiducial frame are turned back by
          the round angle about the frame centre, from the start;
      t2. the rigid offset (2D frame, walk, rigid_walk_confirm) is measured
          on those turned-back frames;
      t3. steps 2-3 (and z_xy_refine, if on) run on the turned-back planes,
          exactly as without rotation;
      t4. the rigid offset and the xy shifts are turned back into the
          moving frame before they are stored, so rigid_y/rigid_x and
          xshift/yshift keep their meaning (xshift/yshift = displacement of
          the frame centre).

    Applied as one resampling, input = c + R(theta) (p - c) + t(z) with c
    the frame centre. The table keeps xshift/yshift as the displacement OF
    THE CENTRE, t(z), and gains rotation_radians (the round angle),
    rotation_center_x and rotation_center_y, so a consumer that ignores
    those columns is still right at the centre. rotation_status has source
    'round', and its measured_radians, trusted and reason echo the rotation
    task's verdict on this fov's own first-plane angle (trusted false = that
    task flagged this fov), for information only: the round angle is
    applied regardless. rigid_only_channels get the round angle too, with
    their rigid offset measured on turned-back frames. The reference stack
    gets angle 0. The rigid_walk table holds the walk as measured, i.e. on
    the turned-back frames.

    Without rotation_task there is no rotation at all: no frame is turned
    and the tables have no rotation columns. The round angle is the only
    way an angle is estimated. An earlier per-stack angle (rotation_correction
    without rotation_task: an angle measured on every stack through its
    depth, with a neighbour substitution in finalize()) was removed. A task
    saved with it still loads, so the tables it wrote can be read and
    applied (their angle is in rotation_radians), but it does not run (see
    __init__).

    PER-FOV LINEAR PART THAT CHANGES WITH DEPTH (affine_correction, default
    off). The round angle is the rigid turn of the whole sample. On top of
    it a fov's bead field can stretch, shear and turn a little further, by
    amounts that change with depth: the gel deforms, unevenly and not as a
    rigid body. A translation cannot absorb that either. With the option on,
    every non-reference stack also gets a 2x2 linear part L(z) about the
    frame centre, per fov and changing with depth, measured at the depths
    and fitted the way the xy shift is. The option requires rotation_task
    (__init__ refuses it otherwise): without the round angle, L would also
    take up the turn of the whole sample.

      a1. at every depth the xy pass sampled (every xy_sample_step'th
          plane), on the planes it used -- the reference plane, and the
          moving plane at the fitted z, turned back by the round angle --
          the moving plane is shifted by that depth's xy sample (its
          whole-frame correlation peak);
      a2. both planes are cut into affine_tiles x affine_tiles tiles and
          each tile pair is correlated as FiducialRoundRotation does (peak
          within xy_search_bound px of zero; empty tiles, peaks on the
          window edge and snr below affine_tile_min_snr dropped). The tile
          displacements are fitted, weighted by tile snr and robustly
          trimmed as FiducialRoundRotation does, with a translation and a
          linear part D = L - I: d(v) = t + D v, v the tile position
          relative to the frame centre. A tile's position is where its
          beads are (the centroid of the signed product of the two tiles,
          i.e. of each pixel's contribution to the tile's correlation), not
          its centre: a tile that beads cover only partly measures the
          displacement there. A sample is usable if at least
          affine_min_tiles tiles survive and they spread over the frame in
          both directions (_AFFINE_MIN_SPREAD_TILES);
      a3. affine_refine_passes residual passes (default 1): the moving plane
          is resampled with the fit so far and a2 is repeated on it; the
          residual is composed in. As for the round angle, a single fit of
          sub-pixel tile displacements reads a few percent short;
      a4. each element of D is fitted as a polynomial in depth of order
          affine_polynomial_order, with the xy fit's sigma trimming. With
          fewer than affine_min_samples usable samples nothing is fitted: L
          is the identity, the stack is registered exactly as without the
          option, and fit_quality_flags says so. The same holds for a stack
          whose z is unfitted (fewer than z_min_good_samples good z
          samples), which is not measured at all: its planes would be
          compared at the zero z curve. finalize() leaves its L at the
          identity when it substitutes the z curve;
      a5. the xy samples are moved to the frame centre. A whole-frame
          correlation peak measures the shift where the frame's beads are,
          at the centroid b of the signed product of the two planes; with a
          linear part the shift at the frame centre differs from it by
          D(z) (b - c), which is added before the y/x polynomials are
          fitted. The tile fit's own centre shift is recorded beside it, as
          a check.

    Applied as one resampling: input = c + R(theta) L(z) (p - c) + t(z). The
    table gains affine_xx, affine_xy, affine_yx and affine_yy (L, acting on
    (x, y), in the frame turned back by the round angle) and
    affine_center_x/y; xshift/yshift stay the displacement of the frame
    centre. polynomial_coefficients gains the polynomials of D's elements
    (affine_dxx_c0, ...), and two QC tables are written, affine_fit_status
    per channel and affine_samples per sample (see their getters); with the
    option on, the y/x rows of the measurements table hold the samples moved
    to the frame centre. z is not touched: the linear part is measured after
    the z fit (and z_xy_refine, which still shifts the planes it samples z
    on by a translation only). The reference stack and rigid_only_channels
    keep the identity; offsets_<fov>.npy and the fiducial QC images stay
    rigid. L is extrapolated beyond the depths of the usable samples, as the
    xy polynomial is, and beyond the part of the frame that holds beads.

    The tiles are compared at one depth per sample, the frame's. Where the
    depth match changes across the frame (a z tilt between rounds), tiles
    on opposite sides are compared at different relative defocus, and a
    tile's apparent displacement can depend on its defocus; this adds
    scatter to the samples, which the depth fit averages over but does not
    remove. affine_samples (scatter of the samples about the fit) and
    affine_fit_status (residual_ppm) show how large it is.

    RIGID WALK CONFIRMATION (rigid_walk_confirm, default off). The walk's
    first plane is searched wide, so that it can overrule a bad 2D frame. On
    a self-similar bead field that wide search can also lock onto a ghost
    peak that no later plane corrects. With the option on, a first-plane
    answer far from the 2D answer must be confirmed by a wide search on the
    next walk plane, which must agree with it, pass the same snr test and
    not be a zero-shift (camera fixed-pattern) lock; otherwise the 2D answer
    is kept. The branch taken per channel is written to the rigid_walk
    table.

    Z SAMPLED AT EACH DEPTH'S OWN XY (z_xy_refine, default off). Step 2
    measures z at one xy, the rigid offset, which describes the bead stack
    only near the depth it was measured at. When the stack's xy moves by a
    few pixels with depth, the deeper z samples compare misaligned bead
    textures, their ZNCC collapses and they fail the gate, so the z fit is
    lost or rests on a few shallow samples, although step 3 recovers the
    xy drift without trouble (it searches +-xy_search_bound about the rigid
    offset). Neither a different rigid offset nor a different walk plane
    fixes this: no single xy is right at every depth. With the option on,
    after steps 2-3:

      r1. the xy samples are fitted in the frame they were measured in, and
          z is measured again with each sample's moving planes shifted by
          that fit at the sample's depth. Same gate, robust fit and
          z_min_good_samples. The refined samples replace the first ones
          unless fewer of them pass the gate.
      r2. xy is measured again on the new z curve, as in step 3.

    If the first z fit was unfitted, the first xy pass uses zero z
    coefficients, as without the option, and the refinement still runs.
    z_fit_status gains n_good_first, fitted_first, z_refined and
    refine_max_shift_px (the largest |xy(z) - rigid| used); the QC table
    gains the first pass's z samples (stage 'z_first'). With rotation_task
    the refinement runs on the turned-back planes and the angle is not
    refined; rotation_status gains unrefined_radians and unrefined_trusted,
    which repeat rotation_radians and trusted.

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

    # Tuning options of the removed per-stack angle (see __init__ for how an
    # old analysis json or task.json that sets them is read).
    # FiducialRoundRotation has options of the same names for its own fit.
    _REMOVED_PARAMETERS = (
        'rotation_tiles', 'rotation_tile_min_snr', 'rotation_min_tiles',
        'rotation_min_planes', 'rotation_max_degrees',
        'rotation_max_spread_degrees', 'rotation_neighbor_substitution')

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
        # 5 keeps the brightest 95%. Note this is the opposite sense to the
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
        # Confirm a far first walk plane before it may overrule the 2D answer.
        # The first plane is searched wide (about zero), which is what lets the
        # walk overrule a bad 2D frame. On a self-similar bead field it can
        # also lock onto a ghost peak, a secondary maximum hundreds of px from
        # the true one whose snr is marginally higher and so beats
        # rigid_snr_ratio x the 2D snr. Every later plane is searched only
        # +-rigid_walk_bound about it, so nothing corrects it. With this on, a
        # first-plane answer more than rigid_walk_bound from the 2D answer is
        # accepted only if a wide search on the next walk plane
        # (rigid_plane_step deeper, and still inside rigid_plane_search) lands
        # within rigid_walk_bound of it AND passes the same snr test
        # (rigid_snr_ratio x the 2D snr). A confirming peak within a pixel of
        # zero shift does not count when the 2D answer is more than
        # rigid_walk_bound from zero: two frames from the same camera share
        # its fixed pattern, which correlates at exactly zero shift, so two
        # planes "agreeing" there is no evidence. Otherwise the 2D answer is
        # kept and the walk stops. Restarting the walk from the 2D answer
        # instead (the first plane searched tightly about it) is worse: that
        # plane's local peak can sit a few px from the 2D answer, and z
        # samples measured at the offset it gives fail the gate. Agreement of
        # two DEEPER planes is not used either: they can agree on a ghost
        # while the first plane and the 2D frame are right, so the check
        # starts from the first plane and its snr test. False keeps the
        # earlier behaviour exactly.
        if 'rigid_walk_confirm' not in self.parameters:
            self.parameters['rigid_walk_confirm'] = False
        if not isinstance(self.parameters['rigid_walk_confirm'],
                          (bool, np.bool_)):
            raise ValueError('rigid_walk_confirm must be true or false')
        # ---- z sampled at each depth's own xy (see the class docstring) ----
        # The z samples are measured with the moving plane shifted by ONE xy,
        # the rigid offset, which is right only at the depth the rigid offset
        # was measured at. Where the stack's xy moves by a few px with depth,
        # the deep z samples compare misaligned bead textures, their ZNCC
        # collapses and they fail the gate, although the xy pass (searched
        # +-xy_search_bound about the rigid offset) finds the drift without
        # trouble. With this on, after the first z and xy fits the z samples
        # are measured again with each moving plane shifted by the xy
        # polynomial at that sample's depth, z is refitted (same gate, robust
        # fit and z_min_good_samples), and xy is measured again on the new z
        # curve. False keeps the earlier behaviour exactly.
        if 'z_xy_refine' not in self.parameters:
            self.parameters['z_xy_refine'] = False
        if not isinstance(self.parameters['z_xy_refine'], (bool, np.bool_)):
            raise ValueError('z_xy_refine must be true or false')
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

        # ---- in-plane rotation: one angle per round (see the class docstring) ----
        # Name of a FiducialRoundRotation task. Every stack of a round gets
        # that task's round angle. None (the default): no rotation at all,
        # and nothing is computed or written differently.
        if 'rotation_task' not in self.parameters:
            self.parameters['rotation_task'] = None

        # ---- per-fov linear part that changes with depth (see the class
        # docstring, affine_correction) ----
        # False (the default): no linear part; nothing is measured, computed
        # or written differently.
        if 'affine_correction' not in self.parameters:
            self.parameters['affine_correction'] = False
        if not isinstance(self.parameters['affine_correction'],
                          (bool, np.bool_)):
            raise ValueError('affine_correction must be true or false')
        # Order of the polynomial in depth fitted to each element of the
        # linear part; 1 is a straight line in depth. On real thick-gel
        # stacks, leaving one sample out at a time, orders 1, 2 and 3
        # predicted it equally well; predicting the deepest quarter of the
        # samples from the rest, order 1 was better than 2 in about 70% of
        # cases and than 3 in about 80% (the higher orders swing at the
        # ends, which a fitted L reaches wherever the deep tile fits fail).
        # Order 0 was worse than 1 where L changes with depth.
        if 'affine_polynomial_order' not in self.parameters:
            self.parameters['affine_polynomial_order'] = 1
        # Fewer usable samples than this and no linear part is fitted: the
        # stack keeps the identity and is registered exactly as without the
        # option, and a fit-quality flag is written.
        if 'affine_min_samples' not in self.parameters:
            self.parameters['affine_min_samples'] = 8
        # The tile measurement, as FiducialRoundRotation's: tiles per side,
        # minimum tile snr, minimum usable tiles per sample.
        if 'affine_tiles' not in self.parameters:
            self.parameters['affine_tiles'] = 6
        if 'affine_tile_min_snr' not in self.parameters:
            self.parameters['affine_tile_min_snr'] = 8.0
        if 'affine_min_tiles' not in self.parameters:
            self.parameters['affine_min_tiles'] = 8
        # Residual passes of the tile fit on the plane resampled by the fit so
        # far (as FiducialRoundRotation's refine_passes); 0 = one fit only.
        if 'affine_refine_passes' not in self.parameters:
            self.parameters['affine_refine_passes'] = 1
        for name in ('affine_polynomial_order', 'affine_min_samples',
                     'affine_tiles', 'affine_min_tiles',
                     'affine_refine_passes'):
            value = self.parameters[name]
            if not isinstance(value, (int, np.integer)) \
                    or isinstance(value, bool) or value < 0:
                raise ValueError('%s must be an integer >= 0' % name)
        if self.parameters['affine_tiles'] < 2:
            raise ValueError('affine_tiles must be at least 2')
        # The linear part is what a fov does on top of the round angle.
        # Without the round angle it would also take up the turn of the whole
        # sample, so the option requires rotation_task.
        if self.parameters['affine_correction'] \
                and not self.parameters['rotation_task']:
            raise ValueError(
                'affine_correction requires rotation_task: the linear part is '
                'measured on top of the round angle, and without it would '
                'take up the rotation of the whole sample. Set rotation_task '
                'to a FiducialRoundRotation task.')

        # Options of the removed per-stack angle, and how an old analysis json
        # or saved task.json that still sets them is read. Dropping them where
        # they carry nothing keeps the saved and the regenerated parameters of
        # an existing dataset equal.
        #   - the tuning options in _REMOVED_PARAMETERS: dropped, whatever
        #     their value (they had defaults, so every task saved while they
        #     existed carries them);
        #   - rotation_correction false: dropped (the old default: no
        #     rotation);
        #   - rotation_correction true with rotation_task: dropped (earlier
        #     versions required it for the round angle; it adds nothing);
        #   - rotation_correction true without rotation_task: a task that
        #     measured an angle per stack. It is kept, so the task still loads
        #     and the tables it wrote (the angle is in their rotation_radians
        #     column) can be read and applied, but _run_analysis and
        #     finalize() refuse to run it.
        for name in self._REMOVED_PARAMETERS:
            self.parameters.pop(name, None)
        legacy = self.parameters.get('rotation_correction')
        if legacy is not None:
            if not isinstance(legacy, (bool, np.bool_)):
                raise ValueError('rotation_correction must be true or false')
            if not legacy and self.parameters['rotation_task']:
                raise ValueError(
                    'rotation_correction false contradicts rotation_task; '
                    'drop rotation_correction (rotation_task alone turns the '
                    'round angle on)')
            if not legacy or self.parameters['rotation_task']:
                del self.parameters['rotation_correction']
        # (fov, dataChannel) -> angle the moving planes are turned back by
        # while they are measured; empty unless rotation_task is set.
        self._measurementRotation = {}
        # z_xy_refine: the number of refinement passes. Deliberately not an
        # option; it is an attribute only so the alternative can be
        # re-measured. Measured on real stacks whose xy moves by a few px with
        # depth: a second pass left the good-sample counts unchanged (already
        # all samples), moved the residuals both ways and cost a fifth to a
        # third more time.
        self._zXyRefinePasses = 1
        # fov -> per-channel record of the rigid walk, written as the
        # rigid_walk table when rigid_walk_confirm is on.
        self._rigidWalkRecords = {}

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
        if self.parameters['rotation_task']:
            return [self.parameters['rotation_task']]
        return []

    def _refuse_stack_rotation(self, step: str) -> None:
        """A task saved with the removed per-stack angle (rotation_correction
        true without rotation_task) loads, so its tables can still be read,
        but it does not run: rerunning it without the angle would silently
        change what the saved parameters say was done."""
        if self.parameters.get('rotation_correction', False):
            raise ValueError(
                '%s: cannot run %s. rotation_correction without rotation_task '
                'is the per-stack angle, which was removed. Set rotation_task '
                'to a FiducialRoundRotation task (one angle per round) or drop '
                'rotation_correction (no rotation). The tables this task has '
                'already written can still be read.'
                % (self.get_analysis_name(), step))

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
        theta = self._measurementRotation.get((fov, dataChannel), 0.0)
        key = (dataChannel, fov, planeIndex, theta)
        cached = self._planeCache.get(key)
        if cached is not None:
            return cached

        dataOrganization = self.dataSet.get_data_organization()
        frameIndices = dataOrganization.get_fiducial3D_stack_frame_indices(
            dataChannel)
        planeIndex = int(np.clip(planeIndex, 0, len(frameIndices) - 1))
        key = (dataChannel, fov, planeIndex, theta)
        cached = self._planeCache.get(key)
        if cached is not None:
            return cached

        rawPlane = self.dataSet.load_image(
            dataOrganization.get_fiducial3D_filename(dataChannel, fov),
            frameIndices[planeIndex])
        filtered = self._filter(rawPlane).astype(np.float32)
        if theta != 0.0:
            filtered = self._rotate_about_centre(filtered, theta)
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

    @staticmethod
    def _rotate_about_centre(image: np.ndarray, theta: float) -> np.ndarray:
        """out(q) = image(c + R(theta) (q - c)), c the frame centre, zero
        fill: a moving plane put back into the reference orientation."""
        height, width = image.shape
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        c, s = np.cos(theta), np.sin(theta)
        matrix = np.float32([[c, -s, cx - c * cx + s * cy],
                             [s, c, cy - s * cx - c * cy]])
        return cv2.warpAffine(
            np.ascontiguousarray(image, dtype=np.float32), matrix,
            (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderValue=0)

    @staticmethod
    def _rotate_vector(vector, theta: float):
        """R(theta) applied to an (x, y) pair (or to coefficient arrays)."""
        x, y = vector
        c, s = np.cos(theta), np.sin(theta)
        return c * x - s * y, s * x + c * y

    @staticmethod
    def _sample_shift(rigidY: float, rigidX: float, shiftCoefficients,
                      zPosition: float):
        """The (dy, dx) a moving plane is shifted by when sampled at
        zPosition: the rigid offset, or, given (y, x) polynomial
        coefficients (z_xy_refine), the xy fit at that depth."""
        if shiftCoefficients is None:
            return rigidY, rigidX
        yCoefficients, xCoefficients = shiftCoefficients
        return (float(np.polyval(yCoefficients, zPosition)),
                float(np.polyval(xCoefficients, zPosition)))

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
    # tile displacements: FiducialRoundRotation's angle and
    # affine_correction's linear part
    # ==================================================================

    def _tile_displacements(self, fixedImage: np.ndarray,
                            movingImage: np.ndarray, tiles: int,
                            minSnr: float, weightedPosition: bool = False):
        """The tile measurement shared by FiducialRoundRotation and
        affine_correction.

        Both planes are cut into tiles x tiles tiles, and each tile pair is
        correlated with its peak searched within xy_search_bound px of zero
        (_bounded_correlation_peak). A tile is dropped if either side is
        empty, if its peak lies on the edge of the search window or if its
        snr is below minSnr.

        Returns (vx, vy, dx, dy, weight, measured), one entry per usable tile:
        (vx, vy) its position relative to the frame centre, (dx, dy) its
        displacement (the content at reference position p sits in movingImage
        at p + d, i.e. minus the measured shift), weight its snr; measured is
        the number of tile pairs with both sides non-empty. The position is
        the tile centre, or with weightedPosition the centroid of each pixel's
        contribution to the tile's correlation peak (_correlation_centroid).
        A tile that is only partly covered by beads measures the displacement
        where its beads are, not at its centre; the two differ wherever the
        displacement varies across the frame.
        """
        height, width = fixedImage.shape
        cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
        rows = np.linspace(0, height, tiles + 1).astype(int)
        cols = np.linspace(0, width, tiles + 1).astype(int)
        vx, vy, dx, dy, weight = [], [], [], [], []
        measured = 0
        for i in range(tiles):
            for j in range(tiles):
                f = fixedImage[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
                m = movingImage[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
                if not f.any() or not m.any():
                    continue
                measured += 1
                sy, sx, snr, onEdge = self._bounded_correlation_peak(
                    f, m, 0.0, 0.0)
                if onEdge or not np.isfinite(snr) or snr < minSnr:
                    continue
                px = 0.5 * (cols[j] + cols[j + 1] - 1)
                py = 0.5 * (rows[i] + rows[i + 1] - 1)
                if weightedPosition:
                    wx, wy = self._correlation_centroid(f, m)
                    if np.isfinite(wx):
                        px, py = cols[j] + wx, rows[i] + wy
                vx.append(px - cx)
                vy.append(py - cy)
                dx.append(-sx)          # displacement = minus the shift
                dy.append(-sy)
                weight.append(snr)
        return (np.asarray(vx), np.asarray(vy), np.asarray(dx),
                np.asarray(dy), np.asarray(weight), measured)

    @staticmethod
    def _correlation_centroid(fixedImage: np.ndarray,
                              movingImage: np.ndarray):
        """(x, y), in pixels of the image, of the centroid of each pixel's
        contribution to the correlation of two aligned images: their
        product. This is where a correlation measures its shift when the
        shift varies across the image. The product is signed: in bead-free
        pixels the two rounds' noise is independent, so its product averages
        to zero there (clipping it at zero would give those pixels a positive
        weight and pull the centroid towards the image centre). (NaN, NaN)
        if the products do not sum to a positive value; the centroid is
        clipped to the image."""
        product = (np.asarray(fixedImage, dtype=np.float64)
                   * np.asarray(movingImage, dtype=np.float64))
        total = float(product.sum())
        if not total > 0:
            return np.nan, np.nan
        height, width = product.shape
        x = float(product.sum(axis=0) @ np.arange(width)) / total
        y = float(product.sum(axis=1) @ np.arange(height)) / total
        return (float(np.clip(x, 0, width - 1)),
                float(np.clip(y, 0, height - 1)))

    @staticmethod
    def _robust_tile_lstsq(design: np.ndarray, target: np.ndarray,
                           weight: np.ndarray, minTiles: int):
        """Weighted least squares of tile displacements with robust trimming,
        shared by FiducialRoundRotation and affine_correction.

        design has 2k rows, the k tiles' x rows then their y rows; target is
        (dx..., dy...); weight is the tiles' snr. Tiles whose residual (the
        length of the tile's residual vector) is more than 3 robust sd
        (1.4826 x the median residual) are dropped and the fit repeated,
        twice at most, never keeping fewer than minTiles tiles. Returns
        (solution, keep, rms, solve): rms is the snr-weighted rms residual of
        the kept tiles, and solve(matrix, keep) fits another design to the
        same tiles."""
        k = len(weight)
        rootWeight = np.sqrt(np.r_[weight, weight])

        def solve(matrix, keep):
            rowsKept = np.r_[keep, keep]
            solution, *_ = np.linalg.lstsq(
                matrix[rowsKept] * rootWeight[rowsKept, None],
                target[rowsKept] * rootWeight[rowsKept], rcond=None)
            residual = target - matrix @ solution
            return solution, np.hypot(residual[:k], residual[k:])

        keep = np.ones(k, bool)
        for _ in range(2):
            _, tileResidual = solve(design, keep)
            scale = 1.4826 * np.median(tileResidual) + 1e-6
            good = tileResidual < 3.0 * scale
            if good.sum() < minTiles:
                break
            keep = good
        solution, tileResidual = solve(design, keep)
        rms = float(np.sqrt(np.sum(weight[keep] * tileResidual[keep] ** 2)
                            / np.sum(weight[keep])))
        return solution, keep, rms, solve

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
        320. Only the depth fit is skipped: the z coefficients are zero and
        the y/x polynomials are the constant rigid offset, so the
        transformation table carries the rigid xy at every plane.

        With rotation_task a rigid-only channel gets its round's angle, and
        its rigid offset is measured on frames turned back by it.

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

    def _measure_rigid_offsets(self, fov: int, dataChannels=None):
        """Rigid xy per data channel, from the 2D fiducial frame or, if
        rigid_plane_search is on, from whichever early plane correlates best.

        dataChannels: the channels to measure, in order (default: every data
        channel). A channel with a measurement rotation set (rotation_task)
        is measured on frames turned back by it: its walk planes through
        _fiducial_plane, its 2D frame here.

        The search is BOUNDED at rigid_search_bound about zero. An unbounded
        peak search on a dense self-similar bead field regularly locks onto a
        secondary maximum of the autocorrelation -- measured on 20260609, two
        fovs returned 626 and 953 px against a true 34 px. Bounding alone
        recovers one of them and turns the other into an honest zero.

        With rigid_walk_confirm on, a first-plane answer more than
        rigid_walk_bound from the 2D answer must be confirmed by a wide search
        on the next walk plane (see __init__); the branch taken for every
        channel is kept in self._rigidWalkRecords[fov] and written by
        _run_analysis as the rigid_walk table.
        """
        bound = self.parameters['rigid_search_bound']
        search = int(self.parameters['rigid_plane_search'])
        step = max(1, int(self.parameters['rigid_plane_step']))
        ratio = float(self.parameters['rigid_snr_ratio'])
        walkBound = self.parameters['rigid_walk_bound']
        confirm = bool(self.parameters['rigid_walk_confirm'])
        fixed2D = self._filter(self.dataSet.get_fiducial_image(0, fov))
        offsets = []
        records = []
        if dataChannels is None:
            dataChannels = self.dataSet.get_data_organization()\
                .get_data_channels()
        for dataChannel in dataChannels:
            moving2D = self._filter(
                self.dataSet.get_fiducial_image(dataChannel, fov))
            turn = self._measurementRotation.get((fov, dataChannel), 0.0)
            if turn != 0.0:
                moving2D = self._rotate_about_centre(moving2D, turn)
            dy, dx, snr, _ = self._bounded_correlation_peak(
                fixed2D, moving2D, 0.0, 0.0, bound=bound)
            best = (dy, dx, snr, -1)
            record = {'fov': fov, 'dataChannel': int(dataChannel),
                      'branch': '2d' if search else 'off',
                      'y_2d': dy, 'x_2d': dx, 'snr_2d': snr,
                      'y_first': np.nan, 'x_first': np.nan,
                      'snr_first': np.nan, 'confirm_plane': -1,
                      'y_confirm': np.nan, 'x_confirm': np.nan,
                      'snr_confirm': np.nan}
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
                    if confirm:
                        record.update(y_first=cy, x_first=cx, snr_first=csnr)
                        if csnr > ratio * best[2]:
                            if np.hypot(cy - dy, cx - dx) <= walkBound:
                                record['branch'] = 'agree'
                            elif self._confirm_first_walk_plane(
                                    fov, dataChannel, plane + step, cy, cx,
                                    record, (dy, dx, snr)):
                                record['branch'] = 'confirmed'
                            else:
                                # a ghost: keep the 2D answer, stop walking
                                record['branch'] = 'rejected'
                                break
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
            if confirm:
                record.update(plane=best[3], rigid_y=float(best[0]),
                              rigid_x=float(best[1]), snr=best[2])
                if record['branch'] in ('confirmed', 'rejected'):
                    print('fov %d channel %d: rigid walk %s -- first plane '
                          '(%.1f, %.1f) snr %.0f is %.1f px from the 2D answer '
                          '(%.1f, %.1f) snr %.0f; plane %d wide (%.1f, %.1f) '
                          'snr %.0f; rigid offset (%.1f, %.1f) from %s'
                          % (fov, dataChannel, record['branch'].upper(),
                             record['y_first'], record['x_first'],
                             record['snr_first'],
                             np.hypot(record['y_first'] - dy,
                                      record['x_first'] - dx),
                             dy, dx, snr, record['confirm_plane'],
                             record['y_confirm'], record['x_confirm'],
                             record['snr_confirm'], best[0], best[1],
                             'plane %d' % best[3] if best[3] >= 0
                             else 'the 2D frame'), flush=True)
                records.append(record)
            offsets.append((float(best[0]), float(best[1])))
        if confirm:
            self._rigidWalkRecords[fov] = records
        return offsets

    # A peak this close to zero shift, in px, can be the camera's fixed
    # pattern, which two frames from the same camera share (rigid_walk_confirm).
    _ZERO_LOCK_PX = 1.0

    def _confirm_first_walk_plane(self, fov: int, dataChannel: int,
                                  plane: int, y: float, x: float,
                                  record: dict, answer2D) -> bool:
        """rigid_walk_confirm: does a WIDE search (+-rigid_search_bound about
        zero) on fiducial plane `plane` confirm the first walk plane's answer
        (y, x)? answer2D = (dy, dx, snr) of the 2D frame. It confirms only if
        its peak

          - lands within rigid_walk_bound of (y, x),
          - has snr above rigid_snr_ratio x the 2D snr, the test the first
            plane itself had to pass, and
          - is not a zero-shift lock: within _ZERO_LOCK_PX of (0, 0) while
            the 2D answer is more than rigid_walk_bound from zero.

        False without a search when `plane` is outside the walk
        (>= rigid_plane_search) or the stack has no such plane. The
        confirming peak, when searched, is recorded in `record`."""
        if plane >= int(self.parameters['rigid_plane_search']):
            return False
        dataOrganization = self.dataSet.get_data_organization()
        planeCount = min(
            len(dataOrganization.get_fiducial3D_stack_frame_indices(0)),
            len(dataOrganization.get_fiducial3D_stack_frame_indices(
                dataChannel)))
        if plane >= planeCount:
            return False
        try:
            f = self._fiducial_plane(0, fov, plane)
            m = self._fiducial_plane(dataChannel, fov, plane)
        except Exception:
            return False
        py, px, psnr, _ = self._bounded_correlation_peak(
            f, m, 0.0, 0.0, bound=self.parameters['rigid_search_bound'])
        record.update(confirm_plane=int(plane), y_confirm=py, x_confirm=px,
                      snr_confirm=psnr)
        walkBound = self.parameters['rigid_walk_bound']
        dy, dx, snr2D = answer2D
        zeroLock = bool(np.hypot(py, px) <= self._ZERO_LOCK_PX
                        and np.hypot(dy, dx) > walkBound)
        return bool(np.hypot(py - y, px - x) <= walkBound
                    and psnr > float(self.parameters['rigid_snr_ratio'])
                    * snr2D
                    and not zeroLock)

    def _measure_z_offsets(self, fov: int, dataChannel: int,
                           rigidY: float, rigidX: float,
                           shiftCoefficients=None):
        """Per-plane z offset in microns, measured at the rigid xy or, given
        shiftCoefficients (z_xy_refine), with each sample's moving planes
        shifted by the xy fit at that sample's depth."""
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
            shiftY, shiftX = self._sample_shift(
                rigidY, rigidX, shiftCoefficients, zPositions[planeIndex])
            scores = []
            for delta in range(-searchRange, searchRange + 1):
                movingImage = self._fiducial_plane(
                    dataChannel, fov, planeIndex + delta)
                scores.append(self._zncc(
                    fixedImage,
                    self._shift_image(movingImage, shiftY, shiftX)))
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

    @staticmethod
    def _row_transform(row):
        """The output->input map of one transformation-table row.

        input = c + R(theta) (p - c) + t, t = (xshift, yshift); a table
        without rotation columns, or with theta 0, gives exactly the plain
        translation of a table without rotation.

        With affine_correction the row also holds the linear part L
        (affine_xx, affine_xy, affine_yx, affine_yy, acting on (x, y)) and
        its centre, and input = c + R(theta) L (p - c) + t: L is applied in
        the measurement frame (planes turned back by theta), then the round
        angle, as the measurement composed them. A row whose L is exactly
        the identity gives exactly the transform of a row without it.
        """
        theta = float(row['rotation_radians']) \
            if 'rotation_radians' in row else 0.0
        if 'affine_xx' in row:
            linear = np.array([
                [float(row['affine_xx']), float(row['affine_xy'])],
                [float(row['affine_yx']), float(row['affine_yy'])]])
            if not np.all(np.isfinite(linear)):
                raise ValueError('transformation row with a non-finite '
                                 'linear part: %s' % linear.tolist())
            if not np.array_equal(linear, np.eye(2)):
                if not np.isfinite(theta):
                    theta = 0.0
                c, s = np.cos(theta), np.sin(theta)
                matrix = np.array([[c, -s], [s, c]]) @ linear
                centre = np.array([float(row['affine_center_x']),
                                   float(row['affine_center_y'])])
                offset = centre - matrix @ centre + np.array(
                    [float(row['xshift']), float(row['yshift'])])
                return transform.AffineTransform(matrix=np.array([
                    [matrix[0, 0], matrix[0, 1], offset[0]],
                    [matrix[1, 0], matrix[1, 1], offset[1]],
                    [0.0, 0.0, 1.0]]))
        if not np.isfinite(theta) or theta == 0.0:
            return transform.SimilarityTransform(
                translation=[row['xshift'], row['yshift']])
        cx = float(row['rotation_center_x'])
        cy = float(row['rotation_center_y'])
        c, s = np.cos(theta), np.sin(theta)
        return transform.AffineTransform(matrix=np.array([
            [c, -s, cx - c * cx + s * cy + float(row['xshift'])],
            [s, c, cy - s * cx - c * cy + float(row['yshift'])],
            [0.0, 0.0, 1.0]]))

    def _rigid_transformations(self, coefficients, dataChannels):
        """One depth-independent transform per channel, for offsets_<fov>.npy
        and the fiducial QC images: the rigid offset, as a
        SimilarityTransform translation (the negated measured shift), and with
        rotation_task also the round angle about the frame centre, through
        the same _row_transform the table uses. With rotation_task the rigid
        offset was measured on frames turned back by the round angle and
        turned into the moving frame, so it is the displacement of the frame
        centre. Channels without an angle (the reference stack, or no
        rotation_task) get the plain translation."""
        rotation = bool(self.parameters['rotation_task'])
        if rotation:
            height, width = self.dataSet.get_image_dimensions()
        transformations = []
        for dataChannel in dataChannels:
            entry = coefficients[dataChannel]
            row = {'xshift': -entry['rigid_x'], 'yshift': -entry['rigid_y']}
            if rotation:
                row.update(rotation_radians=entry.get('theta', 0.0),
                           rotation_center_x=(width - 1) / 2.0,
                           rotation_center_y=(height - 1) / 2.0)
            transformations.append(self._row_transform(row))
        return transformations

    # ==================================================================
    # rotation_task: one angle per round from a FiducialRoundRotation task
    # ==================================================================

    def _round_rotation_inputs(self, fov: int):
        """The rotation task's round angle for every data channel (0 for the
        channels on the reference stack), and its verdict on this fov's own
        first-plane angle per data channel: {channel: (own angle rad or NaN,
        flagged, reason)}. Raises if the task has not been finalized or a
        round has no angle."""
        name = self.parameters['rotation_task']
        rotationTask = self.dataSet.load_analysis_task(name)
        dataOrganization = self.dataSet.get_data_organization()
        referenceStack = dataOrganization.get_fiducial3D_filename(0, fov)
        allAngles = rotationTask.get_round_angles()
        fovVerdicts = rotationTask.get_fov_verdicts(fov)
        angles, verdicts = {}, {}
        for dataChannel in dataOrganization.get_data_channels():
            dataChannel = int(dataChannel)
            if dataOrganization.get_fiducial3D_filename(
                    dataChannel, fov) == referenceStack:
                angles[dataChannel] = 0.0
                continue
            angle = allAngles.get(dataChannel, np.nan)
            if not np.isfinite(angle):
                raise ValueError(
                    '%s: rotation task %s has no round angle for data channel '
                    '%d (no trusted fov in its round; see its round_rotation '
                    'table)' % (self.get_analysis_name(), name, dataChannel))
            angles[dataChannel] = float(angle)
            verdicts[dataChannel] = fovVerdicts.get(
                dataChannel, (np.nan, True,
                              'no first-plane measurement for this fov'))
        return angles, verdicts

    def _round_status_row(self, fov: int, dataChannel: int, angle: float,
                          onReference: bool, verdicts) -> dict:
        """One rotation_status row under rotation_task (see
        get_rotation_status): the round angle, and for a channel off the
        reference stack the rotation task's verdict on this fov's own
        first-plane angle."""
        row = {'fov': fov, 'dataChannel': dataChannel,
               'rotation_radians': angle, 'measured_radians': angle,
               'first_pass_radians': angle, 'spread_radians': np.nan,
               'n_planes': 0, 'n_sampled': 0, 'trusted': True, 'reason': '',
               'source': 'reference'}
        if not onReference:
            own, flagged, reason = verdicts[dataChannel]
            row.update(measured_radians=own, n_planes=int(np.isfinite(own)),
                       n_sampled=1, trusted=not flagged, reason=reason,
                       source='round')
        if self.parameters['z_xy_refine']:
            row.update(unrefined_radians=angle,
                       unrefined_trusted=row['trusted'])
        return row

    def _to_moving_frame(self, shiftY: float, shiftX: float, angle: float):
        """A shift measured on planes turned back by angle, re-expressed in
        the moving frame: s = R(angle) s'. Unchanged when angle is 0."""
        if angle == 0.0:
            return shiftY, shiftX
        x, y = self._rotate_vector((shiftX, shiftY), angle)
        return float(y), float(x)

    # ==================================================================
    # affine_correction: a per-fov linear part that changes with depth
    # ==================================================================

    # A sample counts only if its kept tiles spread over the frame in both
    # directions: the snr-weighted standard deviation of their positions,
    # along the direction where it is smallest, must be at least this many
    # tile sizes. Below it (every tile in one row, say) the linear part is
    # not determined by the tiles.
    _AFFINE_MIN_SPREAD_TILES = 0.5
    # Where a tile's displacement is taken to be: 'weighted', where its beads
    # are (see _tile_displacements), or 'centre', the tile centre.
    # Deliberately not an option; an attribute only so the two can be
    # compared.
    _affineTilePosition = 'weighted'
    # Elements of the linear part, L = [[xx, xy], [yx, yy]], acting on (x, y).
    _AFFINE_ELEMENTS = ('xx', 'xy', 'yx', 'yy')

    @staticmethod
    def _resample_affine(image: np.ndarray, linear: np.ndarray,
                         shiftX: float, shiftY: float) -> np.ndarray:
        """out(q) = image(c + linear (q - c) - s), c the frame centre,
        s = (shiftX, shiftY), zero fill: a moving plane put onto the
        reference by a linear part about the centre and a centre shift (s in
        the sense of _bounded_correlation_peak: the shift applied to the
        moving plane)."""
        height, width = image.shape
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        (a, b), (c, d) = linear
        matrix = np.array([[a, b, cx - a * cx - b * cy - shiftX],
                           [c, d, cy - c * cx - d * cy - shiftY]],
                          dtype=np.float64)
        return cv2.warpAffine(
            np.ascontiguousarray(image, dtype=np.float32), matrix,
            (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderValue=0)

    def _affine_tile_fit(self, fixedImage: np.ndarray,
                         movingImage: np.ndarray) -> dict:
        """Translation and linear part fitted to the tile displacements of
        movingImage against fixedImage: d(v) = t + D v, v the tile position
        relative to the frame centre, so the content at reference p sits in
        movingImage at c + (I + D)(p - c) + t. Tiles, tile rejection and the
        robust weighted fit are FiducialRoundRotation's (_tile_displacements,
        _robust_tile_lstsq), with this task's affine_tiles,
        affine_tile_min_snr and affine_min_tiles; tile positions follow
        _affineTilePosition.

        Returns a dict: ok, reason, n_measured, n_usable, n_used, spread_px
        (see _AFFINE_MIN_SPREAD_TILES), rms_px (snr-weighted rms tile
        residual), D (2x2, acting on (x, y)) and tx, ty."""
        tiles = int(self.parameters['affine_tiles'])
        minTiles = int(self.parameters['affine_min_tiles'])
        vx, vy, dx, dy, weight, measured = self._tile_displacements(
            fixedImage, movingImage, tiles,
            float(self.parameters['affine_tile_min_snr']),
            weightedPosition=self._affineTilePosition == 'weighted')
        result = {'ok': False, 'reason': '', 'n_measured': measured,
                  'n_usable': len(vx), 'n_used': 0, 'spread_px': np.nan,
                  'rms_px': np.nan, 'D': None, 'tx': np.nan, 'ty': np.nan}
        # six unknowns: fewer tiles than that cannot be fitted at all
        if len(vx) < max(minTiles, 3):
            result['reason'] = ('only %d of %d tiles usable (min %d)'
                                % (len(vx), tiles * tiles, minTiles))
            return result
        k = len(vx)
        # dx = tx + Dxx vx + Dxy vy; dy = ty + Dyx vx + Dyy vy
        design = np.zeros((2 * k, 6))
        design[:k, 0] = 1.0
        design[k:, 1] = 1.0
        design[:k, 2] = vx
        design[:k, 3] = vy
        design[k:, 4] = vx
        design[k:, 5] = vy
        solution, keep, rms, _ = self._robust_tile_lstsq(
            design, np.r_[dx, dy], weight, max(minTiles, 3))
        w = weight[keep] / np.sum(weight[keep])
        px, py = vx[keep], vy[keep]
        px, py = px - np.sum(w * px), py - np.sum(w * py)
        covariance = np.array([[np.sum(w * px * px), np.sum(w * px * py)],
                               [np.sum(w * px * py), np.sum(w * py * py)]])
        spread = float(np.sqrt(max(np.linalg.eigvalsh(covariance)[0], 0.0)))
        height, width = fixedImage.shape
        minSpread = self._AFFINE_MIN_SPREAD_TILES * min(height, width) / tiles
        result.update(n_used=int(keep.sum()), spread_px=spread, rms_px=rms)
        if spread < minSpread:
            result['reason'] = ('kept tiles spread %.0f px, below %.0f px: '
                                'linear part not determined'
                                % (spread, minSpread))
            return result
        result.update(ok=True, tx=float(solution[0]), ty=float(solution[1]),
                      D=np.array([[solution[2], solution[3]],
                                  [solution[4], solution[5]]]))
        return result

    def _measure_affine_samples(self, fov: int, dataChannel: int,
                                zCoefficients, sampleY, sampleX):
        """affine_correction, steps a1-a3 of the class docstring, at every
        depth the xy pass sampled, on the same planes it used: the reference
        plane, and the moving plane at the fitted z (turned back by the round
        angle). sampleY, sampleX are the xy pass's frame shifts in the
        measurement frame, in its order. Returns one dict per sample."""
        zPositions = np.asarray(
            self.dataSet.get_data_organization()
            .get_fiducial3D_stack_frame_zPos(dataChannel), dtype=float)
        planeIndices = list(range(1, len(zPositions) - 1,
                                  self.parameters['xy_sample_step']))
        if len(planeIndices) != len(sampleY):
            raise ValueError('affine_correction: %d xy samples for %d sample '
                             'planes' % (len(sampleY), len(planeIndices)))
        frameShift = dict(zip(planeIndices, zip(sampleY, sampleX)))
        passes = int(self.parameters['affine_refine_passes'])

        def measure_one(planeIndex):
            frameY, frameX = (float(v) for v in frameShift[planeIndex])
            zPosition = float(zPositions[planeIndex])
            fixedImage = self._fiducial_plane(0, fov, planeIndex)
            movingImage = self._fiducial_plane_at_z(
                dataChannel, fov,
                zPosition + float(np.polyval(zCoefficients, zPosition)))
            # a1: the moving plane shifted by the xy pass's own frame shift
            shifted = self._shift_image(movingImage, frameY, frameX)
            record = dict(zPos=zPosition, plane_index=int(planeIndex),
                          ok=False, reason='', frame_shift_y=frameY,
                          frame_shift_x=frameX)
            # where the frame's correlation peak is measured: the centroid of
            # each pixel's contribution to it
            record['centroid_x'], record['centroid_y'] = \
                self._correlation_centroid(fixedImage, shifted)
            # a2: tile fit on the shifted plane
            fit = self._affine_tile_fit(fixedImage, shifted)
            prefix = 'first pass: '
            first = fit
            if fit['ok']:
                # content at reference p sits in the moving plane at
                # c + (I + D)(p - c) + t - s_w: centre shift s = s_w - t
                linear = np.eye(2) + fit['D']
                shift = np.array([frameX - fit['tx'], frameY - fit['ty']])
                # a3: residual passes on the plane resampled by the fit so
                # far; the residual (D_r, t_r) composes as
                # L <- L (I + D_r), s <- s - L t_r
                for _ in range(passes):
                    prefix = 'residual pass: '
                    residual = self._affine_tile_fit(
                        fixedImage, self._resample_affine(
                            movingImage, linear, shift[0], shift[1]))
                    fit = residual
                    if not residual['ok']:
                        break
                    shift = shift - linear @ np.array(
                        [residual['tx'], residual['ty']])
                    linear = linear @ (np.eye(2) + residual['D'])
            record.update(n_measured=fit['n_measured'],
                          n_usable=fit['n_usable'], n_used=fit['n_used'],
                          spread_px=fit['spread_px'], rms_px=fit['rms_px'])
            for name in self._AFFINE_ELEMENTS:
                record['d' + name] = np.nan
                record['first_d' + name] = np.nan
            record.update(tile_shift_y=np.nan, tile_shift_x=np.nan)
            if first['ok']:
                for (i, j), name in zip(((0, 0), (0, 1), (1, 0), (1, 1)),
                                        self._AFFINE_ELEMENTS):
                    record['first_d' + name] = float(first['D'][i, j])
            if not fit['ok']:
                record['reason'] = prefix + fit['reason']
                return record
            deformation = linear - np.eye(2)
            for (i, j), name in zip(((0, 0), (0, 1), (1, 0), (1, 1)),
                                    self._AFFINE_ELEMENTS):
                record['d' + name] = float(deformation[i, j])
            record.update(ok=True, tile_shift_x=float(shift[0]),
                          tile_shift_y=float(shift[1]))
            return record

        return self._map_over_planes(measure_one, planeIndices)

    def _affine_matrices(self, coefficients, zPositions) -> np.ndarray:
        """D(z) = L(z) - I at each depth, from the per-element polynomials:
        an array (n, 2, 2) acting on (x, y)."""
        zPositions = np.asarray(zPositions, dtype=float)
        matrices = np.zeros((len(zPositions), 2, 2))
        for (i, j), name in zip(((0, 0), (0, 1), (1, 0), (1, 1)),
                                self._AFFINE_ELEMENTS):
            matrices[:, i, j] = np.polyval(coefficients[name], zPositions)
        return matrices

    def _solve_affine(self, fov: int, dataChannel: int, onReference: bool,
                      zCoefficients, sampleZ, sampleY, sampleX,
                      zUnfitted: str = None) -> dict:
        """affine_correction for one stack: measure (a1-a3), fit in depth
        (a4) and re-reference the xy samples to the frame centre (a5); see
        the class docstring. sampleZ/Y/X are the xy pass's samples in the
        measurement frame. zUnfitted: None if the stack's z was fitted, else
        why not; such a stack is not measured.

        Returns a dict: source ('fit', 'identity', or 'reference' for the
        reference stack, which is not measured), fitted, n_samples, n_good,
        reason, coefficients ({element: polynomial of D's element in z}),
        samples (the per-sample records, with the fit and the kept flags)
        and, if fitted, centre_y / centre_x (the re-referenced xy samples)."""
        order = int(self.parameters['affine_polynomial_order'])
        zeros = {name: np.zeros(order + 1) for name in self._AFFINE_ELEMENTS}
        info = {'source': 'reference', 'fitted': False,
                'n_samples': 0, 'n_good': 0, 'reason': '',
                'coefficients': zeros, 'samples': []}
        if onReference:
            return info
        if zUnfitted is not None:
            # its planes would be compared at the zero z curve, not at the
            # depth finalize() may substitute: the identity, not measured
            info.update(source='identity',
                        reason='z unfitted (%s), not measured' % zUnfitted)
            return info
        samples = self._measure_affine_samples(
            fov, dataChannel, zCoefficients, sampleY, sampleX)
        z = np.array([s['zPos'] for s in samples], dtype=float)
        good = np.array([bool(s['ok']) for s in samples], dtype=bool)
        nGood = int(good.sum())
        # the robust fit needs order + 2 points to trim anything
        minSamples = max(int(self.parameters['affine_min_samples']),
                         order + 2)
        fitted = nGood >= minSamples
        info.update(source='fit' if fitted else 'identity', fitted=fitted,
                    n_samples=len(samples), n_good=nGood, samples=samples)
        if not fitted:
            info['reason'] = ('only %d of %d samples usable (min %d)'
                              % (nGood, len(samples), minSamples))
        coefficients = {}
        for name in self._AFFINE_ELEMENTS:
            values = np.array([s['d' + name] for s in samples], dtype=float)
            kept = np.zeros(len(samples), bool)
            if fitted:
                coefficients[name], goodKept = self._robust_polyfit(
                    z[good], values[good], order)
                kept[np.flatnonzero(good)] = goodKept
            else:
                coefficients[name] = np.zeros(order + 1)
            fitValues = np.polyval(coefficients[name], z)
            for record, value, isKept in zip(samples, fitValues, kept):
                record['fit_d' + name] = float(value) if fitted else np.nan
                record['kept_d' + name] = bool(isKept)
        info['coefficients'] = coefficients
        if fitted:
            # a5: the frame shift is the displacement where the frame's beads
            # are (their centroid b); at the frame centre it is
            # s_w + D(z) (b - c)
            height, width = self.dataSet.get_image_dimensions()
            centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
            matrices = self._affine_matrices(coefficients, z)
            centreY, centreX = [], []
            for record, matrix in zip(samples, matrices):
                b = np.array([record['centroid_x'], record['centroid_y']])
                correction = matrix @ (b - centre) if np.all(np.isfinite(b)) \
                    else np.zeros(2)
                record['centre_shift_x'] = record['frame_shift_x'] \
                    + float(correction[0])
                record['centre_shift_y'] = record['frame_shift_y'] \
                    + float(correction[1])
                centreX.append(record['centre_shift_x'])
                centreY.append(record['centre_shift_y'])
            info['centre_y'] = np.array(centreY)
            info['centre_x'] = np.array(centreX)
        else:
            for record in samples:
                record['centre_shift_x'] = record['frame_shift_x']
                record['centre_shift_y'] = record['frame_shift_y']
        return info

    def _affine_status_row(self, fov: int, dataChannel: int, info: dict,
                           zPositions) -> dict:
        """One affine_fit_status row (see get_affine_fit_status)."""
        row = {'fov': fov, 'dataChannel': dataChannel,
               'n_samples': info['n_samples'], 'n_good': info['n_good'],
               'fitted': bool(info['fitted']), 'source': info['source'],
               'reason': info['reason'],
               'order': int(self.parameters['affine_polynomial_order']),
               'z_min_good_um': np.nan, 'z_max_good_um': np.nan,
               'kept_fraction': np.nan, 'residual_ppm': np.nan,
               'max_corner_px': 0.0}
        samples = info['samples']
        good = [s for s in samples if s['ok']]
        if good:
            row.update(z_min_good_um=min(s['zPos'] for s in good),
                       z_max_good_um=max(s['zPos'] for s in good))
        if info['fitted']:
            keptFraction, residual = [], []
            for name in self._AFFINE_ELEMENTS:
                kept = [s for s in good if s['kept_d' + name]]
                keptFraction.append(len(kept) / float(len(good)))
                residual.append(np.median(np.abs(
                    [s['d' + name] - s['fit_d' + name] for s in kept])))
            height, width = self.dataSet.get_image_dimensions()
            corners = np.array([[0.0, 0.0], [width - 1.0, 0.0],
                                [0.0, height - 1.0],
                                [width - 1.0, height - 1.0]]) \
                - np.array([(width - 1) / 2.0, (height - 1) / 2.0])
            matrices = self._affine_matrices(info['coefficients'], zPositions)
            row.update(kept_fraction=float(min(keptFraction)),
                       residual_ppm=float(1e6 * max(residual)),
                       max_corner_px=float(np.max(np.hypot(
                           *np.einsum('nij,kj->ink', matrices, corners)))))
        return row

    # ==================================================================
    # storage
    # ==================================================================

    def _save_results(self, fov: int, coefficients, measurements) -> None:
        dataOrganization = self.dataSet.get_data_organization()
        dataChannels = list(dataOrganization.get_data_channels())
        zPositions = np.asarray(self.dataSet.get_z_positions(), dtype=float)

        # Rigid transforms, in the same format the other warp tasks use, so
        # anything that only wants a per-channel offset still works.
        self._save_transformations(
            self._rigid_transformations(coefficients, dataChannels), fov)

        coefficientRows = []
        for dataChannel in dataChannels:
            entry = coefficients[dataChannel]
            row = {'dataChannel': dataChannel,
                   'rigid_y': entry['rigid_y'],
                   'rigid_x': entry['rigid_x']}
            if self.parameters['rotation_task']:
                row['rotation_radians'] = entry.get('theta', 0.0)
            for name in ('z', 'y', 'x'):
                for order, value in enumerate(entry[name][::-1]):
                    row['%s_c%d' % (name, order)] = value
            if self.parameters['affine_correction']:
                # polynomials in z of the elements of D = L - I
                for name in self._AFFINE_ELEMENTS:
                    for order, value in enumerate(
                            entry['affine'][name][::-1]):
                        row['affine_d%s_c%d' % (name, order)] = value
            coefficientRows.append(row)
        self.dataSet.save_dataframe_to_csv(
            pd.DataFrame(coefficientRows), 'polynomial_coefficients',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations', index=False)

        tableRows = []
        if self.parameters['rotation_task']:
            height, width = self.dataSet.get_image_dimensions()
        if self.parameters['affine_correction']:
            affineHeight, affineWidth = self.dataSet.get_image_dimensions()
        for dataChannel in dataChannels:
            entry = coefficients[dataChannel]
            zOffset = np.polyval(entry['z'], zPositions)
            shiftY = np.polyval(entry['y'], zPositions)
            shiftX = np.polyval(entry['x'], zPositions)
            planeTable = pd.DataFrame({
                'dataChannel': dataChannel,
                'zIndex': np.arange(len(zPositions)),
                'zPos': zPositions,
                'zPos_new': zPositions + zOffset,
                'z_offset': zOffset,
                # stored as transform.SimilarityTransform translation
                # parameters, i.e. the negated measured shift; with
                # rotation_task, the displacement of the frame centre
                'xshift': -shiftX,
                'yshift': -shiftY})
            if self.parameters['rotation_task']:
                planeTable['rotation_radians'] = entry.get('theta', 0.0)
                planeTable['rotation_center_x'] = (width - 1) / 2.0
                planeTable['rotation_center_y'] = (height - 1) / 2.0
            if self.parameters['affine_correction']:
                # L(z) = I + D(z), in the measurement frame (see
                # _row_transform); exactly the identity where nothing was
                # fitted
                matrices = self._affine_matrices(entry['affine'], zPositions)
                planeTable['affine_xx'] = 1.0 + matrices[:, 0, 0]
                planeTable['affine_xy'] = matrices[:, 0, 1]
                planeTable['affine_yx'] = matrices[:, 1, 0]
                planeTable['affine_yy'] = 1.0 + matrices[:, 1, 1]
                planeTable['affine_center_x'] = (affineWidth - 1) / 2.0
                planeTable['affine_center_y'] = (affineHeight - 1) / 2.0
            tableRows.append(planeTable)
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

    def get_rotation_status(self, fov: int) -> pd.DataFrame:
        """Per data channel, written only with rotation_task: the angle in
        the tables (rotation_radians, the round angle), its source ('round',
        or 'reference' on the reference stack), and the rotation task's
        verdict on this fov's own first-plane angle (measured_radians, NaN
        if untrusted; trusted, false where that task flagged the fov;
        reason). first_pass_radians repeats rotation_radians, and
        spread_radians is NaN; with z_xy_refine, unrefined_radians and
        unrefined_trusted repeat rotation_radians and trusted. These columns
        keep the layout of the removed per-stack angle, whose tables (sources
        'fit', 'untrusted', 'neighbor', 'rigid_only') read the same way."""
        return self.dataSet.load_dataframe_from_csv(
            'rotation_status', self, resultIndex=fov,
            subdirectory='transformations')

    def get_rigid_walk(self, fov: int) -> pd.DataFrame:
        """Per data channel: the rigid walk's branch ('off', '2d', 'agree',
        'confirmed', 'rejected') and the peaks behind it. Written only with
        rigid_walk_confirm on."""
        return self.dataSet.load_dataframe_from_csv(
            'rigid_walk', self, resultIndex=fov,
            subdirectory='transformations')

    def get_affine_fit_status(self, fov: int) -> pd.DataFrame:
        """Per data channel, written only with affine_correction: whether
        the linear part was fitted (source 'fit'), left at the identity
        because too few samples were usable or the stack's z is unfitted
        ('identity', with the reason; a z-unfitted stack is not measured,
        n_samples 0), or not measured ('reference': the reference stack;
        'rigid_only').
        n_samples / n_good: the xy sample depths and how many gave a usable
        tile fit; z_min_good_um / z_max_good_um: the depth range of those (a
        fitted L is extrapolated beyond it); kept_fraction: the smallest
        fraction of usable samples the robust depth fit kept, over the four
        elements; residual_ppm: the largest, over the elements, median
        |sample - fit| of the kept samples, in ppm; max_corner_px: the
        largest displacement L(z) - I gives a frame corner at any table
        depth, i.e. how much the linear part changes the registration."""
        return self.dataSet.load_dataframe_from_csv(
            'affine_fit_status', self, resultIndex=fov,
            subdirectory='transformations')

    def get_affine_samples(self, fov: int) -> pd.DataFrame:
        """Per measured stack and xy sample depth, written only with
        affine_correction: the tile fit (ok, reason, n_measured, n_usable,
        n_used, spread_px, rms_px), the measured elements of D = L - I
        (dxx, dxy, dyx, dyy; first_d.. = before the residual pass), the depth
        fit at that depth (fit_d..) and whether the fit kept the sample
        (kept_d..), and the shifts, in px in the measurement frame (planes
        turned back by the round angle) and in the sense of the xy samples:
        frame_shift (the xy pass's whole-frame peak), tile_shift
        (the tile fit's centre shift), centre_shift (the frame shift moved
        from the beads' centroid (centroid_x, centroid_y) to the frame
        centre with the fitted L; what the y/x polynomials are fitted on)."""
        return self.dataSet.load_dataframe_from_csv(
            'affine_samples', self, resultIndex=fov,
            subdirectory='transformations')

    # ==================================================================
    # cross-fov repair
    # ==================================================================

    def finalize(self) -> None:
        """Repair stacks from their neighbouring fovs: unfitted z curves
        (z_neighbor_substitution).

        Runs once, from `merlin -t <task> --check-done`, which is the only
        point in the DAG where every fragment is finished and nothing
        downstream has started. It has to be here rather than in
        _run_analysis: a fragment knows one fov, and the donors are other
        fovs that may not have been solved yet when it ran.

        Idempotent, as finalize() must be: the decision of WHICH stacks to
        repair comes from z_fit_status, which records the fragment's verdict
        and is never rewritten except for the source label, and the donors
        are by definition stacks that fitted, whose values this never
        touches. A repaired stack's source becomes 'neighbor', so a second
        run finds nothing left to repair and writes nothing.

        No angle is repaired: with rotation_task every stack has its round's
        angle, and without it there is none. No linear part is repaired
        either (affine_correction): a stack whose z was unfitted keeps the
        identity once its z curve is substituted.
        """
        self._refuse_stack_rotation('finalize()')
        if self.parameters['z_neighbor_substitution']:
            self._substitute_z_curves()

    def _neighbor_donors(self, fov, dataChannel, usable, fovXY):
        """Donor fovs for (fov, dataChannel): the fovs within the first ring
        of z_neighbor_rings_um that holds z_neighbor_min_donors usable
        stacks, else whatever the widest ring holds. Returns (donors, ring)."""
        minDonors = int(self.parameters['z_neighbor_min_donors'])
        rings = [float(r) for r in self.parameters['z_neighbor_rings_um']]
        tx, ty = fovXY[fov]
        donors, usedRing = [], rings[-1]
        for ring in rings:
            donors = [f for f in fovXY
                      if f != fov and (f, dataChannel) in usable
                      and np.hypot(fovXY[f][0] - tx, fovXY[f][1] - ty) <= ring]
            if len(donors) >= minDonors:
                usedRing = ring
                break
        return donors, usedRing

    def _substitute_z_curves(self) -> None:
        """Give every unfitted z stack the curve of its neighbouring fovs
        (z_neighbor_substitution; see finalize())."""
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
            donors, usedRing = self._neighbor_donors(fov, dataChannel,
                                                     fitted, xy)
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
        corner displacement over the whole list, which is exactly the union
        of the invalid margins over depth (and over the turned corners when
        rotation_task is set).
        """
        table = self.get_transformation_table(fov)
        _transform = self._row_transform

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
        moved by the fitted xy offset (and angle, with rotation_task)."""
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

        transformation = self._row_transform(row)
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
        get_aligned_image instead (depth, per-plane xy and, with
        rotation_task, the angle). The fiducial images use
        transformationList, the per-channel rigid transforms of
        _rigid_transformations: rigid offset plus, with rotation_task,
        the round angle.
        """
        dataChannels = self.dataSet.get_data_organization().get_data_channels()

        alignedFOVs = self.parameters['write_aligned_fovs']
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
            fiducialFOVs = self.parameters['write_fiducial_fovs']
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
        self._refuse_stack_rotation('a fragment')
        fov = list(self.dataSet.get_fovs())[fragmentIndex]
        dataOrganization = self.dataSet.get_data_organization()
        dataChannels = list(dataOrganization.get_data_channels())

        zOrder = self.parameters['z_polynomial_order']
        xyOrder = self.parameters['xy_polynomial_order']

        # rotation_task: every moving channel's frames are turned back by its
        # round's angle BEFORE anything is measured (steps t1-t2 of the class
        # docstring); None without the option, and then nothing is turned
        roundAngles, roundVerdicts = None, None
        if self.parameters['rotation_task']:
            roundAngles, roundVerdicts = self._round_rotation_inputs(fov)
            for dataChannel, angle in roundAngles.items():
                if angle != 0.0:
                    self._measurementRotation[(fov, dataChannel)] = angle

        rigidOffsets = self._measure_rigid_offsets(fov)
        walkRecords = self._rigidWalkRecords.pop(fov, None)

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

        referenceStack = (dataOrganization.get_fiducial3D_filename(0, fov)
                          if roundAngles is not None else None)
        # affine_correction: the reference stack (channel 0's) is not
        # measured; its linear part is the identity
        affineReference = (dataOrganization.get_fiducial3D_filename(0, fov)
                           if self.parameters['affine_correction'] else None)

        def solve_stack(item):
            key, members = item
            dataChannel, rigidY, rigidX = members[0]
            # the angle this stack's planes are turned back by while they are
            # measured: its round's angle with rotation_task (0 on the
            # reference stack), else 0
            theta = roundAngles[dataChannel] if roundAngles is not None \
                else 0.0
            zSampleZ, zSampleOffset, zSampleScore, zCeiling = \
                self._measure_z_offsets(fov, dataChannel, rigidY, rigidX)
            zGate = self._z_sample_gate(zSampleOffset, zSampleScore, zCeiling)

            def fit_z(gate, sampleZ, sampleOffset):
                good = int(gate.sum())
                fitted = good >= int(self.parameters['z_min_good_samples']
                                     or 0)
                if fitted:
                    coefficients, gateKept = self._robust_polyfit(
                        sampleZ[gate], sampleOffset[gate], zOrder)
                    # gateKept indexes the gated subset; lift it back so the
                    # QC table keeps one row per sample and a gated-out
                    # sample reads as not kept rather than vanishing.
                    kept = np.zeros(len(gate), bool)
                    kept[np.flatnonzero(gate)] = gateKept
                else:
                    # No depth information worth having. Zero coefficients
                    # mean this round falls back to its rigid xy, and
                    # finalize() may replace the curve from the neighbouring
                    # fovs.
                    coefficients = np.zeros(zOrder + 1)
                    kept = np.zeros(len(gate), bool)
                return good, fitted, coefficients, kept

            nGood, zFitted, zCoefficients, zKept = fit_z(
                zGate, zSampleZ, zSampleOffset)
            xySampleZ, xySampleY, xySampleX, xySnr, xyEdge = \
                self._measure_xy_offsets(fov, dataChannel, rigidY, rigidX,
                                         zCoefficients)

            # z_xy_refine: z measured again with each moving plane shifted by
            # the xy fit at its own depth, then z refitted and xy measured
            # again on the new z curve. Everything stays in the measurement
            # frame (planes turned back by theta, if any) until the shifts
            # are turned back below.
            refineStatus, firstZ = None, None
            if self.parameters['z_xy_refine']:
                refineStatus = {'n_good_first': nGood,
                                'fitted_first': bool(zFitted),
                                'z_refined': False,
                                'refine_max_shift_px': np.nan}
                firstZ = (zSampleZ, zSampleOffset, zSampleScore, zKept,
                          zCoefficients)
                for _ in range(max(1, int(self._zXyRefinePasses))):
                    shiftCoefficients = (
                        self._robust_polyfit(xySampleZ, xySampleY, xyOrder)[0],
                        self._robust_polyfit(xySampleZ, xySampleX, xyOrder)[0])
                    rZ, rOffset, rScore, rCeiling = self._measure_z_offsets(
                        fov, dataChannel, rigidY, rigidX, shiftCoefficients)
                    rGate = self._z_sample_gate(rOffset, rScore, rCeiling)
                    if len(rZ):
                        refineStatus['refine_max_shift_px'] = float(max(
                            np.hypot(sy - rigidY, sx - rigidX)
                            for sy, sx in (self._sample_shift(
                                rigidY, rigidX, shiftCoefficients, z)
                                for z in rZ)))
                    if int(rGate.sum()) < nGood:
                        # fewer refined samples pass the gate: the xy fit is
                        # not to be trusted for this, keep what there is
                        break
                    refineStatus['z_refined'] = True
                    zSampleZ, zSampleOffset, zSampleScore, zGate = \
                        rZ, rOffset, rScore, rGate
                    nGood, zFitted, zCoefficients, zKept = fit_z(
                        zGate, zSampleZ, zSampleOffset)
                    xySampleZ, xySampleY, xySampleX, xySnr, xyEdge = \
                        self._measure_xy_offsets(fov, dataChannel, rigidY,
                                                 rigidX, zCoefficients)

            # affine_correction (steps a1-a5 of the class docstring): the
            # linear part, measured at the final xy sample depths on the same
            # (turned-back) planes, and the xy samples moved from where the
            # frame's beads are to the frame centre with it; nothing changes
            # for a stack that is not fitted, and a stack whose z is unfitted
            # is not measured
            affineInfo = None
            if self.parameters['affine_correction']:
                affineInfo = self._solve_affine(
                    fov, dataChannel, key == affineReference, zCoefficients,
                    xySampleZ, xySampleY, xySampleX,
                    zUnfitted=None if zFitted else (
                        '%d of %d z samples passed the gate, min %s'
                        % (nGood, len(zGate),
                           self.parameters['z_min_good_samples'])))
                if affineInfo['fitted']:
                    xySampleY, xySampleX = \
                        affineInfo['centre_y'], affineInfo['centre_x']

            if theta != 0.0:
                # shifts measured on the turned-back planes -> shift of the
                # frame centre in the moving frame: s = R(theta) s'
                xySampleX, xySampleY = self._rotate_vector(
                    (xySampleX, xySampleY), theta)
            self._measurementRotation.pop((fov, dataChannel), None)
            yCoefficients, yKept = self._robust_polyfit(
                xySampleZ, xySampleY, xyOrder)
            xCoefficients, xKept = self._robust_polyfit(
                xySampleZ, xySampleX, xyOrder)

            entry = {'z': zCoefficients, 'y': yCoefficients,
                     'x': xCoefficients}
            if roundAngles is not None:
                entry['theta'] = theta
            affineStatus = None
            if affineInfo is not None:
                entry['affine'] = affineInfo['coefficients']
                affineStatus = self._affine_status_row(
                    fov, dataChannel, affineInfo, self.dataSet.get_z_positions())
            status = {'fov': fov, 'dataChannel': dataChannel,
                      'n_samples': int(len(zGate)), 'n_good': nGood,
                      'n_gated_out': int(len(zGate) - nGood),
                      'fitted': bool(zFitted),
                      'source': 'fit' if zFitted else 'unfitted'}
            if refineStatus is not None:
                # n_good above is the refined count when z_refined
                status.update(refineStatus)

            stackMeasurements = []
            if self.parameters['write_qc_table']:
                measurements = stackMeasurements
                measurements.append(pd.DataFrame({
                    'dataChannel': dataChannel, 'stage': 'z',
                    'zPos': zSampleZ, 'measured': zSampleOffset,
                    'fitted': np.polyval(zCoefficients, zSampleZ),
                    'score': zSampleScore, 'kept': zKept,
                    'on_edge': False}))
                if firstZ is not None:
                    # z_xy_refine: the first pass's z samples (at the rigid
                    # offset) and the fit they gave, for comparison
                    fZ, fOffset, fScore, fKept, fCoefficients = firstZ
                    measurements.append(pd.DataFrame({
                        'dataChannel': dataChannel, 'stage': 'z_first',
                        'zPos': fZ, 'measured': fOffset,
                        'fitted': np.polyval(fCoefficients, fZ),
                        'score': fScore, 'kept': fKept, 'on_edge': False}))
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
            if roundAngles is not None and key != referenceStack:
                line += ', round angle %.4f deg' % np.degrees(theta)
            if refineStatus is not None:
                line += (', z_xy_refine: %d -> %d good z samples, max xy '
                         'shift %.2f px%s' % (
                             refineStatus['n_good_first'], nGood,
                             refineStatus['refine_max_shift_px'],
                             '' if refineStatus['z_refined']
                             else ' (fewer passed: first pass kept)'))
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
            if affineStatus is not None and affineStatus['source'] == 'fit':
                line += (', affine: %d/%d samples, up to %.2f px at the '
                         'corners' % (affineStatus['n_good'],
                                      affineStatus['n_samples'],
                                      affineStatus['max_corner_px']))
            elif affineStatus is not None \
                    and affineStatus['source'] == 'identity':
                line += ('\n  *** AFFINE NOT FITTED, fov %d channel %d: %s; '
                         'linear part left at the identity (registered as '
                         'without affine_correction). ***'
                         % (fov, dataChannel, affineStatus['reason']))
            # rotation_task: the round angle and the rotation task's verdict
            # on this fov's own first-plane angle, written as rotation_status
            rotationStatus = None
            if roundAngles is not None:
                rotationStatus = self._round_status_row(
                    fov, dataChannel, theta, key == referenceStack,
                    roundVerdicts)
            return (key, entry, stackMeasurements, line, flag, status,
                    rotationStatus, affineInfo, affineStatus)

        stackThreads = max(1, int(self.parameters['stack_num_threads']))
        items = list(stackMembers.items())
        if stackThreads > 1 and len(items) > 1:
            with ThreadPoolExecutor(
                    max_workers=min(stackThreads, len(items))) as pool:
                solvedStacks = list(pool.map(solve_stack, items))
        else:
            solvedStacks = [solve_stack(i) for i in items]
        if roundAngles is not None:
            # every plane this fov needs has been measured
            for dataChannel in roundAngles:
                self._measurementRotation.pop((fov, dataChannel), None)

        # Reassemble in the ORIGINAL channel order, so the output is identical
        # whatever order the threads happened to finish in.
        coefficients = {}
        measurements = []
        entryOf = {key: entry for key, entry, *_ in solvedStacks}
        qcFlags = []
        zStatus = []
        rotationOf = {}
        affineOf = {}
        affineSamples = []
        for key, _, stackMeasurements, line, flag, status, rotationStatus, \
                affineInfo, affineStatus in solvedStacks:
            zStatus.append(status)
            rotationOf[status['dataChannel']] = rotationStatus
            measurements.extend(stackMeasurements)
            print(line, flush=True)
            if flag is not None:
                qcFlags.append(flag)
            if affineStatus is not None:
                affineOf[status['dataChannel']] = affineStatus
                affineSamples.extend(
                    dict(record, fov=fov, dataChannel=status['dataChannel'])
                    for record in affineInfo['samples'])
                if affineStatus['source'] == 'identity':
                    qcFlags.append({
                        'fov': fov, 'dataChannel': status['dataChannel'],
                        'z_residual_um': np.nan, 'y_residual_px': np.nan,
                        'x_residual_px': np.nan, 'fraction_kept': np.nan,
                        'reasons': 'affine: %s; linear part left at the '
                                   'identity' % affineStatus['reason']})
        if self.parameters['rigid_walk_confirm'] and walkRecords is not None:
            # a far first walk plane either overruled the 2D answer (confirmed)
            # or was thrown away as a ghost (rejected); both are worth a look
            for record in walkRecords:
                if record['branch'] not in ('confirmed', 'rejected'):
                    continue
                distance = np.hypot(record['y_first'] - record['y_2d'],
                                    record['x_first'] - record['x_2d'])
                if record['branch'] == 'confirmed':
                    reason = ('rigid walk: 2D answer (%.1f, %.1f) overruled by '
                              'plane 0 (%.1f, %.1f), %.1f px away, confirmed by '
                              'plane %d (%.1f, %.1f)')
                else:
                    reason = ('rigid walk: plane 0 peak (%.1f, %.1f) %.1f px '
                              'from the 2D answer (%.1f, %.1f) NOT confirmed '
                              'by plane %d (%.1f, %.1f); 2D answer kept')
                values = ((record['y_2d'], record['x_2d'], record['y_first'],
                           record['x_first'], distance)
                          if record['branch'] == 'confirmed' else
                          (record['y_first'], record['x_first'], distance,
                           record['y_2d'], record['x_2d']))
                qcFlags.append({
                    'fov': fov, 'dataChannel': record['dataChannel'],
                    'z_residual_um': np.nan, 'y_residual_px': np.nan,
                    'x_residual_px': np.nan, 'fraction_kept': np.nan,
                    'reasons': reason % (values + (
                        record['confirm_plane'], record['y_confirm'],
                        record['x_confirm']))})
            # one row per data channel: which branch the walk took
            self.dataSet.save_dataframe_to_csv(
                pd.DataFrame(walkRecords, columns=[
                    'fov', 'dataChannel', 'branch', 'rigid_y', 'rigid_x',
                    'snr', 'plane', 'y_2d', 'x_2d', 'snr_2d', 'y_first',
                    'x_first', 'snr_first', 'confirm_plane', 'y_confirm',
                    'x_confirm', 'snr_confirm']),
                'rigid_walk', self.get_analysis_name(), resultIndex=fov,
                subdirectory='transformations', index=False)
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
                if roundAngles is not None:
                    # measured on turned-back frames -> moving frame (t4)
                    rigidY, rigidX = self._to_moving_frame(
                        rigidY, rigidX, roundAngles[dataChannel])
                coefficients[dataChannel] = dict(
                    entryOf[key], rigid_y=rigidY, rigid_x=rigidX)
                # Channels sharing a fiducial stack share its fit, so they
                # share its status; `owner` names the one actually measured.
                statusRows.append(dict(statusOf[owner], dataChannel=dataChannel,
                                       owner_channel=owner))
        for dataChannel, rigidY, rigidX in rigidOnlyMembers:
            # The y/x polynomials are the TOTAL shift (a fitted channel's xy
            # samples are measured about its rigid offset, not relative to
            # it), and they are what the transformation table and therefore
            # get_aligned_image apply. A rigid-only channel's polynomial is
            # therefore the constant rigid offset, not zero: all-zero
            # coefficients left these channels unregistered in the table.
            if roundAngles is not None:
                rigidY, rigidX = self._to_moving_frame(
                    rigidY, rigidX, roundAngles[dataChannel])
            coefficients[dataChannel] = {
                'z': np.zeros(zOrder + 1),
                'y': np.r_[np.zeros(xyOrder), rigidY],
                'x': np.r_[np.zeros(xyOrder), rigidX],
                'rigid_y': rigidY, 'rigid_x': rigidX}
            if roundAngles is not None:
                coefficients[dataChannel]['theta'] = roundAngles[dataChannel]
            if self.parameters['affine_correction']:
                # no depth information: the identity, as for the 3D warp
                rigidOnlyAffine = {
                    'source': 'rigid_only', 'fitted': False, 'n_samples': 0,
                    'n_good': 0, 'reason': '', 'samples': [],
                    'coefficients': {
                        name: np.zeros(int(self.parameters[
                            'affine_polynomial_order']) + 1)
                        for name in self._AFFINE_ELEMENTS}}
                coefficients[dataChannel]['affine'] = \
                    rigidOnlyAffine['coefficients']
                affineOf[dataChannel] = self._affine_status_row(
                    fov, dataChannel, rigidOnlyAffine,
                    self.dataSet.get_z_positions())
            print('fov %d channel %d: rigid only (%.2f, %.2f), no 3D warp '
                  'fitted' % (fov, dataChannel, rigidY, rigidX), flush=True)
            statusRows.append(dict(
                fov=fov, dataChannel=dataChannel, owner_channel=dataChannel,
                n_samples=0, n_good=0, n_gated_out=0, fitted=False,
                source='rigid_only'))
            if self.parameters['z_xy_refine']:
                statusRows[-1].update(n_good_first=0, fitted_first=False,
                                      z_refined=False,
                                      refine_max_shift_px=np.nan)

        # One row per data channel, always written, so finalize() can find the
        # stacks needing substitution without re-reading any measurement, and
        # so the ABSENCE of a row never has to be read as "the gate did not
        # run". source is 'fit', 'unfitted' or 'rigid_only'; finalize()
        # rewrites 'unfitted' to 'neighbor' when it substitutes one. With
        # z_xy_refine, n_good / fitted are the final values and the first
        # pass's are kept beside them.
        statusColumns = ['fov', 'dataChannel', 'owner_channel', 'n_samples',
                         'n_good', 'n_gated_out', 'fitted', 'source']
        if self.parameters['z_xy_refine']:
            statusColumns += ['n_good_first', 'fitted_first', 'z_refined',
                              'refine_max_shift_px']
        self.dataSet.save_dataframe_to_csv(
            pd.DataFrame(statusRows, columns=statusColumns)
            .sort_values('dataChannel'),
            'z_fit_status', self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations', index=False)

        if roundAngles is not None:
            # One row per data channel, like z_fit_status (see
            # get_rotation_status).
            rotationRows = []
            for key, members in stackMembers.items():
                owner = members[0][0]
                for dataChannel, _, _ in members:
                    rotationRows.append(dict(rotationOf[owner],
                                             dataChannel=dataChannel,
                                             owner_channel=owner))
            for dataChannel, _, _ in rigidOnlyMembers:
                # the round angle, as for every stack
                row = self._round_status_row(
                    fov, dataChannel, roundAngles[dataChannel],
                    dataOrganization.get_fiducial3D_filename(
                        dataChannel, fov) == referenceStack, roundVerdicts)
                row.update(owner_channel=dataChannel)
                rotationRows.append(row)
            rotationColumns = [
                'fov', 'dataChannel', 'owner_channel', 'rotation_radians',
                'measured_radians', 'first_pass_radians', 'spread_radians',
                'n_planes', 'n_sampled', 'trusted', 'reason', 'source']
            if self.parameters['z_xy_refine']:
                rotationColumns += ['unrefined_radians', 'unrefined_trusted']
            self.dataSet.save_dataframe_to_csv(
                pd.DataFrame(rotationRows, columns=rotationColumns)
                .sort_values('dataChannel'),
                'rotation_status', self.get_analysis_name(), resultIndex=fov,
                subdirectory='transformations', index=False)

        if self.parameters['affine_correction']:
            # One row per data channel, like z_fit_status (see
            # get_affine_fit_status), and one row per measured stack and
            # sample depth (see get_affine_samples).
            affineRows = []
            for key, members in stackMembers.items():
                owner = members[0][0]
                for dataChannel, _, _ in members:
                    affineRows.append(dict(affineOf[owner],
                                           dataChannel=dataChannel,
                                           owner_channel=owner))
            for dataChannel, _, _ in rigidOnlyMembers:
                affineRows.append(dict(affineOf[dataChannel],
                                       owner_channel=dataChannel))
            self.dataSet.save_dataframe_to_csv(
                pd.DataFrame(affineRows, columns=[
                    'fov', 'dataChannel', 'owner_channel', 'n_samples',
                    'n_good', 'fitted', 'source', 'reason', 'order',
                    'z_min_good_um', 'z_max_good_um', 'kept_fraction',
                    'residual_ppm', 'max_corner_px'])
                .sort_values('dataChannel'),
                'affine_fit_status', self.get_analysis_name(),
                resultIndex=fov, subdirectory='transformations', index=False)
            sampleColumns = (
                ['fov', 'dataChannel', 'zPos', 'plane_index', 'ok', 'reason',
                 'n_measured', 'n_usable', 'n_used', 'spread_px', 'rms_px']
                + ['d' + n for n in self._AFFINE_ELEMENTS]
                + ['fit_d' + n for n in self._AFFINE_ELEMENTS]
                + ['kept_d' + n for n in self._AFFINE_ELEMENTS]
                + ['first_d' + n for n in self._AFFINE_ELEMENTS]
                + ['frame_shift_y', 'frame_shift_x', 'tile_shift_y',
                   'tile_shift_x', 'centroid_y', 'centroid_x',
                   'centre_shift_y', 'centre_shift_x'])
            self.dataSet.save_dataframe_to_csv(
                pd.DataFrame(affineSamples, columns=sampleColumns),
                'affine_samples', self.get_analysis_name(), resultIndex=fov,
                subdirectory='transformations', index=False)

        self._save_results(fov, coefficients, measurements)
        self._planeCache.clear()
        self._fftCache.clear()

        self._process_transformations(
            self._rigid_transformations(coefficients, dataChannels), fov)


class FiducialRoundRotation(analysistask.ParallelAnalysisTask):

    """
    One in-plane rotation angle per imaging round, shared by every fov, for
    FiducialPolynomialWarp3D's rotation_task.

    Between rounds the whole sample can turn slightly about the optical
    axis. That is one rigid rotation of the whole slab, so every fov of a
    round turns by the same angle, while the translation differs from fov to
    fov (the stage does not return to exactly the same position). This task
    therefore measures the angle in every fov, on one shallow plane of the
    3D bead stack, and takes the median over the fovs as the round's angle.
    FiducialPolynomialWarp3D then applies that one angle to every fov and
    measures only translation and depth.

    Channels are grouped by fiducial bead stack, as FiducialPolynomialWarp3D
    groups them: the channels of one imaging round share a stack, and a
    separately acquired stack (a nuclear stack with its own beads, say) is a
    group of its own. A group is named by its first data channel
    (owner_channel). Channel 0's stack is the reference and has angle 0.

    PER FOV (one fragment), for every non-reference stack:

      1. coarse shift: FiducialPolynomialWarp3D's rigid offset on unrotated
         frames -- the 2D fiducial frame, refined by the walk into the bead
         stack (rigid_plane_search, rigid_plane_step, rigid_snr_ratio,
         rigid_search_bound, rigid_walk_bound), with rigid_walk_confirm's
         false-peak guard if set.
      2. depth match: plane measure_plane of the reference bead stack is
         compared with the moving stack's planes within +-z_search_range of
         the same index, each shifted by the coarse shift. The highest
         normalised correlation, refined with a parabola, gives the moving
         depth. The angle is not known yet, so this correlation is weakened
         towards the frame edges by the turn itself; it chooses the depth
         and is recorded (depth_score), but is not what the gate tests.
      3. tile fit: the moving plane at that depth (interpolated between the
         two nearest planes) is shifted by the coarse shift; it and the
         reference plane are cut into rotation_tiles x rotation_tiles tiles,
         and each tile pair is correlated with its peak searched within
         xy_search_bound px of zero. A tile is dropped if either side is
         empty, if its peak lies on the edge of the search window or if its
         snr is below rotation_tile_min_snr. The remaining tile
         displacements are fitted by weighted least squares, weight = tile
         snr, with a translation plus a rotation about the frame centre
         (rigid: no scale). Tiles more than 3 robust sd from the fit are
         dropped and the fit repeated (twice at most).
      4. residual pass (refine_passes, default 1): the moving plane is
         resampled with the fitted rotation and translation, step 3 is
         repeated on it, and the residual angle and translation are added.
         Parabolic refinement of a correlation peak pulls a sub-pixel
         displacement towards the nearest whole pixel, so the first fit
         comes out a few percent short of the true angle; on the resampled
         plane the displacements are close to zero, where that bias
         vanishes. 0 skips this pass.
      5. diagnostic only: on the first pass's tiles a similarity fit
         (translation, rotation and one isotropic scale) is made as well;
         its scale (scale_ppm) is recorded and never applied. (On
         synthetic frames it reads about 5% low; the residual pass's tiles,
         where the rotation is gone, read lower still.)
      6. alignment check: the moving plane at the matched depth, resampled
         with the final rotation and translation, is correlated with the
         reference plane (aligned_score).

    A fov's angle is trusted when FiducialPolynomialWarp3D's z sample gate
    passes -- aligned_score at least z_min_score, and the depth not on the
    +-z_search_range limit (z_reject_at_search_limit) -- every pass keeps at
    least rotation_min_tiles tiles, and |angle| is within
    rotation_max_degrees; otherwise `reason` says why.

    ACROSS FOVS, in finalize() (`merlin -t <task> --check-done`, once every
    fragment is done), per stack group:

      - round angle = median of the trusted fov angles;
      - robust sd = 1.4826 x the median absolute deviation from it;
        standard error of the median = 1.2533 x robust sd / sqrt(n);
      - a fov is FLAGGED when its angle is untrusted, or deviates from the
        round angle by more than max(deviation_min_degrees,
        deviation_robust_sd x robust sd). A flag is a report only: the fov
        still gets the round angle downstream.

    A round with no trusted fov has no angle (NaN), and
    FiducialPolynomialWarp3D refuses to run on it rather than assume 0.
    finalize() aggregates whatever fragments have written their table and
    reports how many fovs were missing.

    OUTPUTS, under transformations/:

      first_plane_rotation_<fov>.csv  per fragment, one row per stack group
      rigid_walk_<fov>.csv            per fragment, with rigid_walk_confirm
      round_rotation.csv              finalize(): one row per stack group
      fov_rotation.csv                finalize(): one row per fov and group,
                                      with the deviation and the flag

    Conventions are FiducialPolynomialWarp3D's: the content at reference
    position p sits in the moving frame at c + R(theta) (p - c) + t, c the
    frame centre, theta in radians; xshift/yshift is t, the displacement of
    the frame centre, as in its transformation table. scale_ppm > 0 means
    the moving frame's content lies further from the centre than the
    reference's.

    The measurement code (filtering, cached planes, bounded correlation,
    rigid walk, z gate) is FiducialPolynomialWarp3D's own, run by an
    internal instance of it built from this task's parameters. The shared
    parameters have the same names, meanings and defaults; give them the
    same values as the registration task that uses this one.
    """

    outputGroup = 'Prepare'

    # Passed to the internal FiducialPolynomialWarp3D; their defaults are
    # whatever that class defaults to. The tile-fit options (rotation_*) are
    # this task's own.
    _MEASUREMENT_PARAMETERS = (
        'highpass_sigma', 'median_filter', 'edge_width_to_remove',
        'zero_below_percentile', 'rigid_plane_search', 'rigid_plane_step',
        'rigid_snr_ratio', 'rigid_search_bound', 'rigid_walk_bound',
        'rigid_walk_confirm', 'z_search_range', 'z_min_score',
        'z_reject_at_search_limit', 'xy_search_bound',
        'minimum_overlap_pixels', 'plane_cache_size',
        'per_z_slice_num_threads')

    _FRAGMENT_COLUMNS = [
        'fov', 'owner_channel', 'members', 'stack_label', 'plane_index',
        'plane_z_um', 'coarse_y', 'coarse_x', 'depth_offset_um',
        'depth_score', 'aligned_score', 'depth_ok', 'first_pass_radians',
        'first_pass_tiles_used', 'first_pass_residual_px', 'n_tiles',
        'n_tiles_usable', 'n_tiles_used', 'residual_px', 'angle_radians',
        'angle_deg', 'xshift', 'yshift', 'scale_ppm', 'trusted', 'reason']

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        defaults = FiducialPolynomialWarp3D(dataSet).parameters
        for name in self._MEASUREMENT_PARAMETERS:
            if name not in self.parameters:
                self.parameters[name] = defaults[name]
        # ---- the tile fit (step 3) ----
        # Tiles per side. 6 (341 px tiles on a 2048 frame) recovers a known
        # 0.05-0.2 deg rotation of synthetic 2048 px bead frames to within
        # 3% in one pass; 4 to within 5%.
        if 'rotation_tiles' not in self.parameters:
            self.parameters['rotation_tiles'] = 6
        # A tile counts only if its peak is inside the bounded window and its
        # snr (as FiducialPolynomialWarp3D's bounded correlation defines it)
        # reaches this.
        if 'rotation_tile_min_snr' not in self.parameters:
            self.parameters['rotation_tile_min_snr'] = 8.0
        # A fit needs at least this many usable tiles.
        if 'rotation_min_tiles' not in self.parameters:
            self.parameters['rotation_min_tiles'] = 8
        # Index of the reference bead-stack plane the angle is measured on.
        # None = z_search_range + 1, the shallowest plane whose depth search
        # stays inside the stack and the first plane FiducialPolynomialWarp3D
        # samples z on.
        if 'measure_plane' not in self.parameters:
            self.parameters['measure_plane'] = None
        # Residual passes of the tile fit (step 4); 0 = the first fit only.
        if 'refine_passes' not in self.parameters:
            self.parameters['refine_passes'] = 1
        # A fov's angle beyond this is not trusted.
        if 'rotation_max_degrees' not in self.parameters:
            self.parameters['rotation_max_degrees'] = 1.0
        # finalize(): flag a fov whose angle is more than
        # max(deviation_min_degrees, deviation_robust_sd x robust sd) from
        # its round's angle.
        if 'deviation_min_degrees' not in self.parameters:
            self.parameters['deviation_min_degrees'] = 0.01
        if 'deviation_robust_sd' not in self.parameters:
            self.parameters['deviation_robust_sd'] = 4.0
        # Threads over the stacks of a fov.
        if 'stack_num_threads' not in self.parameters:
            self.parameters['stack_num_threads'] = 1

        plane = self.parameters['measure_plane']
        if plane is not None and (not isinstance(plane, (int, np.integer))
                                  or isinstance(plane, bool) or plane < 0):
            raise ValueError('measure_plane must be null or a plane index')
        passes = self.parameters['refine_passes']
        if not isinstance(passes, (int, np.integer)) \
                or isinstance(passes, bool) or passes < 0:
            raise ValueError('refine_passes must be an integer >= 0')

        self._measurer = FiducialPolynomialWarp3D(
            dataSet, {name: self.parameters[name]
                      for name in self._MEASUREMENT_PARAMETERS},
            self.analysisName)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 8192

    def get_estimated_time(self):
        return 15

    def get_dependencies(self):
        return []

    # ==================================================================
    # per fov
    # ==================================================================

    def _measure_plane(self) -> int:
        plane = self.parameters['measure_plane']
        if plane is None:
            return int(self.parameters['z_search_range']) + 1
        return int(plane)

    def _stack_groups(self, fov: int):
        """{fiducial stack: [data channels]} in data channel order, as
        FiducialPolynomialWarp3D groups them."""
        dataOrganization = self.dataSet.get_data_organization()
        groups = OrderedDict()
        for dataChannel in dataOrganization.get_data_channels():
            dataChannel = int(dataChannel)
            groups.setdefault(dataOrganization.get_fiducial3D_filename(
                dataChannel, fov), []).append(dataChannel)
        return groups

    def _stack_label(self, dataChannel: int) -> str:
        """Image type and imaging round of a channel's bead stack, for the
        tables; empty if the data organization does not say."""
        data = getattr(self.dataSet.get_data_organization(), 'data', None)
        try:
            return '%s round %s' % (
                data.loc[dataChannel, 'fiducial3DImageType'],
                data.loc[dataChannel, 'fiducial3DImagingRound'])
        except (AttributeError, KeyError, TypeError):
            return ''

    @staticmethod
    def _resample_rigid(image: np.ndarray, theta: float, tx: float,
                        ty: float) -> np.ndarray:
        """out(p) = image(c + R(theta) (p - c) + t), c the frame centre,
        zero fill: a moving plane put into the reference frame by a rigid
        transform."""
        height, width = image.shape
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        c, s = np.cos(theta), np.sin(theta)
        matrix = np.float32([[c, -s, cx - c * cx + s * cy + tx],
                             [s, c, cy - s * cx - c * cy + ty]])
        return cv2.warpAffine(
            np.ascontiguousarray(image, dtype=np.float32), matrix,
            (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderValue=0)

    def _tile_fit(self, fixedImage: np.ndarray, movingImage: np.ndarray):
        """Rigid fit (translation + small rotation about the frame centre)
        to the tile displacements of movingImage against fixedImage, weighted
        by tile snr, with robust trimming; plus a similarity fit (one more
        parameter, an isotropic scale) on the kept tiles, as a diagnostic.

        Returns a dict: ok, reason, n_tiles, n_measured (both sides of the
        tile non-empty), n_usable (peak inside the window, snr high enough),
        n_used (kept by the trimming), theta, tx, ty (displacement sense:
        the content at reference p sits in movingImage at
        c + R(theta) (p - c) + t), rms_px (snr-weighted rms tile residual of
        the rigid fit) and scale_ppm (of the similarity fit)."""
        measurer = self._measurer
        n = int(self.parameters['rotation_tiles'])
        minSnr = float(self.parameters['rotation_tile_min_snr'])
        minTiles = int(self.parameters['rotation_min_tiles'])
        # the tile measurement and the robust fit are FiducialPolynomialWarp3D's
        # (shared with its affine_correction); tile positions are tile centres
        vx, vy, dx, dy, weight, measured = measurer._tile_displacements(
            fixedImage, movingImage, n, minSnr)
        result = {'ok': False, 'reason': '', 'n_tiles': n * n,
                  'n_measured': measured, 'n_usable': len(vx), 'n_used': 0,
                  'theta': np.nan, 'tx': np.nan, 'ty': np.nan,
                  'rms_px': np.nan, 'scale_ppm': np.nan}
        if len(vx) < minTiles:
            result['reason'] = ('only %d of %d tiles usable (min %d)'
                                % (len(vx), n * n, minTiles))
            return result
        k = len(vx)
        # d = t + theta * (-v_y, v_x): rows 0..k-1 are x, k..2k-1 are y
        design = np.zeros((2 * k, 3))
        design[:k, 0] = 1.0
        design[k:, 1] = 1.0
        design[:k, 2] = -vy
        design[k:, 2] = vx
        solution, keep, rms, solve = measurer._robust_tile_lstsq(
            design, np.r_[dx, dy], weight, minTiles)
        # diagnostic: d = t + theta * (-v_y, v_x) + kappa * (v_x, v_y),
        # scale = 1 + kappa
        similarity = np.zeros((2 * k, 4))
        similarity[:, :3] = design
        similarity[:k, 3] = vx
        similarity[k:, 3] = vy
        similaritySolution, _ = solve(similarity, keep)
        result.update(ok=True, theta=float(solution[2]),
                      tx=float(solution[0]), ty=float(solution[1]),
                      n_used=int(keep.sum()), rms_px=rms,
                      scale_ppm=float(similaritySolution[3] * 1e6))
        return result

    def _measure_stack(self, fov: int, members, coarseY: float,
                       coarseX: float) -> dict:
        """Steps 2-5 of the class docstring for one stack; returns its row
        of first_plane_rotation."""
        measurer = self._measurer
        owner = members[0]
        zPositions = np.asarray(
            self.dataSet.get_data_organization()
            .get_fiducial3D_stack_frame_zPos(owner), dtype=float)
        planeCount = len(zPositions)
        searchRange = int(self.parameters['z_search_range'])
        plane = self._measure_plane()
        if plane - searchRange < 0 or plane + searchRange > planeCount - 1:
            raise ValueError(
                'measure_plane %d with z_search_range %d needs planes %d-%d; '
                'the bead stack of channel %d has %d'
                % (plane, searchRange, plane - searchRange,
                   plane + searchRange, owner, planeCount))
        spacing = float(np.median(np.diff(zPositions))) if planeCount > 1 \
            else 1.0
        row = dict.fromkeys(self._FRAGMENT_COLUMNS, np.nan)
        row.update(fov=fov, owner_channel=owner,
                   members=';'.join(str(c) for c in members),
                   stack_label=self._stack_label(owner), plane_index=plane,
                   plane_z_um=float(zPositions[plane]),
                   coarse_y=float(coarseY), coarse_x=float(coarseX),
                   trusted=False, reason='')
        reasons = []

        # 2. depth match
        fixedImage = measurer._fiducial_plane(0, fov, plane)

        def score(delta):
            return measurer._zncc(fixedImage, measurer._shift_image(
                measurer._fiducial_plane(owner, fov, plane + delta),
                coarseY, coarseX))

        scores = np.asarray(measurer._map_over_planes(
            score, list(range(-searchRange, searchRange + 1))), dtype=float)
        scores = np.where(np.isfinite(scores), scores, -1.0)
        best = int(np.argmax(scores))
        sub = 0.0
        if 0 < best < len(scores) - 1:
            sub = measurer._parabolic_vertex(
                scores[best - 1], scores[best], scores[best + 1])
        depthOffset = (best - searchRange + sub) * spacing
        row.update(depth_offset_um=float(depthOffset),
                   depth_score=float(scores[best]))

        # 3. tile fit on the plane at the matched depth, coarse-shifted
        movingImage = measurer._fiducial_plane_at_z(
            owner, fov, zPositions[plane] + depthOffset)
        fit = self._tile_fit(fixedImage, measurer._shift_image(
            movingImage, coarseY, coarseX))
        row.update(first_pass_radians=fit['theta'],
                   first_pass_tiles_used=fit['n_used'],
                   first_pass_residual_px=fit['rms_px'],
                   scale_ppm=fit['scale_ppm'])
        final, alignedScore = fit, np.nan
        if not fit['ok']:
            reasons.append('first pass: ' + fit['reason'])
        else:
            # the shifted plane holds moving(q - coarse): t - coarse is the
            # displacement of the centre in the moving plane itself
            theta = fit['theta']
            tx, ty = fit['tx'] - coarseX, fit['ty'] - coarseY
            # 4. residual passes on the plane resampled by the fit so far
            for _ in range(int(self.parameters['refine_passes'])):
                residualFit = self._tile_fit(
                    fixedImage, self._resample_rigid(movingImage, theta,
                                                     tx, ty))
                if not residualFit['ok']:
                    reasons.append('residual pass: '
                                   + residualFit['reason'])
                    final = residualFit
                    break
                # content at p sits at c + R(theta) R(r) (p - c)
                #                       + R(theta) t_r + t
                rx, ry = measurer._rotate_vector(
                    (residualFit['tx'], residualFit['ty']), theta)
                final = residualFit
                theta += residualFit['theta']
                tx, ty = tx + rx, ty + ry
            row.update(angle_radians=theta, angle_deg=float(np.degrees(theta)),
                       xshift=float(tx), yshift=float(ty))
            # 6. how well the final rigid fit aligns the two planes
            alignedScore = measurer._zncc(
                fixedImage, self._resample_rigid(movingImage, theta, tx, ty))
            if abs(np.degrees(theta)) > float(
                    self.parameters['rotation_max_degrees']):
                reasons.append('angle %.3f deg beyond rotation_max_degrees'
                               % np.degrees(theta))
        row.update(n_tiles=final['n_tiles'], n_tiles_usable=final['n_usable'],
                   n_tiles_used=final['n_used'], residual_px=final['rms_px'])
        # FiducialPolynomialWarp3D's z sample gate, on the aligned planes
        depthOk = bool(measurer._z_sample_gate(
            np.array([depthOffset]), np.array([alignedScore]),
            searchRange * spacing)[0])
        row.update(aligned_score=float(alignedScore), depth_ok=depthOk)
        if not depthOk and fit['ok']:
            reasons.append('z gate failed (aligned score %.2f, depth offset '
                           '%+.2f um)' % (alignedScore, depthOffset))
        row.update(trusted=not reasons, reason='; '.join(reasons))
        return row

    def _run_analysis(self, fragmentIndex: int):
        fov = list(self.dataSet.get_fovs())[fragmentIndex]
        measurer = self._measurer
        referenceStack = self.dataSet.get_data_organization()\
            .get_fiducial3D_filename(0, fov)
        items = [(key, members) for key, members
                 in self._stack_groups(fov).items() if key != referenceStack]
        owners = [members[0] for _, members in items]
        try:
            # 1. coarse shift of every non-reference stack (unrotated frames)
            coarse = dict(zip(owners, measurer._measure_rigid_offsets(
                fov, dataChannels=owners)))
            walkRecords = measurer._rigidWalkRecords.pop(fov, None)

            def solve(item):
                _, members = item
                return self._measure_stack(fov, members, *coarse[members[0]])

            threads = max(1, int(self.parameters['stack_num_threads']))
            if threads > 1 and len(items) > 1:
                with ThreadPoolExecutor(
                        max_workers=min(threads, len(items))) as pool:
                    rows = list(pool.map(solve, items))
            else:
                rows = [solve(item) for item in items]
        finally:
            measurer._planeCache.clear()
            measurer._fftCache.clear()

        for row in rows:
            print('fov %d stack %d (%s): angle %.4f deg (first pass %.4f), '
                  'depth %+.2f um (score %.2f, aligned %.2f), %d/%d tiles, '
                  'residual %.3f px, scale %+.0f ppm%s'
                  % (fov, row['owner_channel'], row['stack_label'],
                     np.degrees(row['angle_radians']),
                     np.degrees(row['first_pass_radians']),
                     row['depth_offset_um'], row['depth_score'],
                     row['aligned_score'],
                     row['n_tiles_used'], row['n_tiles'], row['residual_px'],
                     row['scale_ppm'],
                     '' if row['trusted']
                     else '  *** NOT TRUSTED: %s ***' % row['reason']),
                  flush=True)
        self.dataSet.save_dataframe_to_csv(
            pd.DataFrame(rows, columns=self._FRAGMENT_COLUMNS),
            'first_plane_rotation', self.get_analysis_name(),
            resultIndex=fov, subdirectory='transformations', index=False)
        if walkRecords is not None:
            self.dataSet.save_dataframe_to_csv(
                pd.DataFrame(walkRecords, columns=[
                    'fov', 'dataChannel', 'branch', 'rigid_y', 'rigid_x',
                    'snr', 'plane', 'y_2d', 'x_2d', 'snr_2d', 'y_first',
                    'x_first', 'snr_first', 'confirm_plane', 'y_confirm',
                    'x_confirm', 'snr_confirm']),
                'rigid_walk', self.get_analysis_name(), resultIndex=fov,
                subdirectory='transformations', index=False)

    # ==================================================================
    # across fovs
    # ==================================================================

    @staticmethod
    def _text(value) -> str:
        """A text cell read back from csv: NaN (an empty string) -> ''."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ''
        return str(value)

    @staticmethod
    def _is_true(value) -> bool:
        return str(value) == 'True'

    def finalize(self) -> None:
        """The round angle of every stack group and the per-fov flags (see
        the class docstring). Idempotent: it only reads the fragments'
        tables and rewrites round_rotation and fov_rotation from them."""
        fovs = list(self.dataSet.get_fovs())
        tables, missing = [], []
        for fov in fovs:
            try:
                tables.append(self.get_first_plane_rotation(fov))
            except FileNotFoundError:
                missing.append(fov)
        if not tables:
            print('%s: no first_plane_rotation tables, nothing to aggregate'
                  % self.get_analysis_name(), flush=True)
            return
        if missing:
            print('%s: WARNING %d of %d fovs have no first_plane_rotation '
                  'table and are left out: %s'
                  % (self.get_analysis_name(), len(missing), len(fovs),
                     ' '.join(str(f) for f in missing)), flush=True)
        measured = pd.concat(tables, ignore_index=True)
        minDeviation = np.radians(float(
            self.parameters['deviation_min_degrees']))
        sdFactor = float(self.parameters['deviation_robust_sd'])

        firstFov = int(measured.fov.iloc[0])
        referenceStack = self.dataSet.get_data_organization()\
            .get_fiducial3D_filename(0, firstFov)
        referenceMembers = self._stack_groups(firstFov)[referenceStack]
        roundRows = [dict(
            owner_channel=referenceMembers[0],
            members=';'.join(str(c) for c in referenceMembers),
            stack_label=self._stack_label(referenceMembers[0]),
            source='reference', angle_radians=0.0, angle_deg=0.0, n_fovs=0,
            n_measured=0, n_untrusted=0, n_flagged=0, robust_sd_deg=np.nan,
            se_deg=np.nan, deviation_threshold_deg=np.nan,
            plane_index=np.nan, plane_z_um=np.nan, median_residual_px=np.nan,
            median_scale_ppm=np.nan, n_fovs_missing=len(missing))]
        fovRows = []
        for owner, group in measured.groupby('owner_channel', sort=True):
            group = group.sort_values('fov')
            trusted = np.array([self._is_true(v) for v in group.trusted])
            angles = group.angle_radians.to_numpy(float)[trusted]
            count = len(angles)
            if count:
                roundAngle = float(np.median(angles))
                robustSd = float(1.4826 * np.median(
                    np.abs(angles - roundAngle)))
                standardError = 1.2533 * robustSd / np.sqrt(count)
                threshold = max(minDeviation, sdFactor * robustSd)
            else:
                roundAngle = robustSd = standardError = np.nan
                threshold = minDeviation
            memberSets = sorted(set(group.members.astype(str)))
            if len(memberSets) > 1:
                print('%s: WARNING stack %d has different members in '
                      'different fovs: %s' % (self.get_analysis_name(), owner,
                                              memberSets), flush=True)
            flagged = 0
            for row, isTrusted in zip(group.itertuples(), trusted):
                deviation = float(row.angle_radians) - roundAngle
                deviating = bool(isTrusted and np.isfinite(deviation)
                                 and abs(deviation) > threshold)
                if not isTrusted:
                    reason = 'untrusted: %s' % self._text(row.reason)
                elif deviating:
                    reason = ('deviates %+.4f deg from the round angle '
                              '(threshold %.4f deg)'
                              % (np.degrees(deviation),
                                 np.degrees(threshold)))
                else:
                    reason = ''
                flagged += int(bool(reason))
                fovRows.append(dict(
                    fov=int(row.fov), owner_channel=int(owner),
                    members=str(row.members),
                    stack_label=self._text(row.stack_label),
                    angle_radians=float(row.angle_radians),
                    angle_deg=float(np.degrees(row.angle_radians)),
                    trusted=bool(isTrusted),
                    round_angle_radians=roundAngle,
                    deviation_deg=float(np.degrees(deviation)),
                    deviation_threshold_deg=float(np.degrees(threshold)),
                    deviating=deviating, flagged=bool(reason),
                    flag_reason=reason,
                    residual_px=float(row.residual_px),
                    n_tiles_used=row.n_tiles_used,
                    scale_ppm=float(row.scale_ppm),
                    depth_offset_um=float(row.depth_offset_um)))
            roundRows.append(dict(
                owner_channel=int(owner), members=memberSets[0],
                stack_label=self._text(group['stack_label'].iloc[0]),
                source='measured',
                angle_radians=roundAngle,
                angle_deg=float(np.degrees(roundAngle)), n_fovs=count,
                n_measured=len(group), n_untrusted=int((~trusted).sum()),
                n_flagged=flagged, robust_sd_deg=float(np.degrees(robustSd)),
                se_deg=float(np.degrees(standardError)),
                deviation_threshold_deg=float(np.degrees(threshold)),
                plane_index=int(group.plane_index.iloc[0]),
                plane_z_um=float(group.plane_z_um.iloc[0]),
                median_residual_px=float(np.median(
                    group.residual_px.to_numpy(float)[trusted]))
                if count else np.nan,
                median_scale_ppm=float(np.median(
                    group.scale_ppm.to_numpy(float)[trusted]))
                if count else np.nan,
                n_fovs_missing=len(missing)))

        roundTable = pd.DataFrame(roundRows)
        fovTable = pd.DataFrame(fovRows)
        self.dataSet.save_dataframe_to_csv(
            roundTable, 'round_rotation', self.get_analysis_name(),
            subdirectory='transformations', index=False)
        self.dataSet.save_dataframe_to_csv(
            fovTable, 'fov_rotation', self.get_analysis_name(),
            subdirectory='transformations', index=False)
        print('%s: round angles from %d fovs' % (
            self.get_analysis_name(), len(tables)), flush=True)
        print(roundTable[['owner_channel', 'stack_label', 'source', 'angle_deg',
                          'n_fovs', 'n_measured', 'robust_sd_deg', 'se_deg',
                          'n_flagged']].to_string(index=False), flush=True)
        flaggedRows = fovTable[fovTable.flagged]
        if len(flaggedRows):
            print('flagged (fov, stack: reason):', flush=True)
            for row in flaggedRows.itertuples():
                print('  fov %d stack %d: %s' % (row.fov, row.owner_channel,
                                                 row.flag_reason), flush=True)

    # ==================================================================
    # getters
    # ==================================================================

    def _load(self, name: str, fov: int = None) -> pd.DataFrame:
        try:
            return self.dataSet.load_dataframe_from_csv(
                name, self, resultIndex=fov, subdirectory='transformations',
                float_precision='round_trip')
        except FileNotFoundError:
            if fov is not None:
                raise
            raise FileNotFoundError(
                '%s: no %s table yet; run `merlin -t %s --check-done` once '
                'every fragment is done' % (self.get_analysis_name(), name,
                                            self.get_analysis_name()))

    def get_first_plane_rotation(self, fov: int) -> pd.DataFrame:
        """One fragment's measurements, one row per non-reference stack."""
        return self._load('first_plane_rotation', fov)

    def get_round_rotation(self) -> pd.DataFrame:
        """finalize()'s round table, one row per stack group."""
        return self._load('round_rotation')

    def get_fov_rotation(self) -> pd.DataFrame:
        """finalize()'s per-fov table, with the deviation flags."""
        return self._load('fov_rotation')

    def get_round_angles(self) -> dict:
        """{data channel: its round's angle in radians}, for every channel
        of every group (0 on the reference stack, NaN for a round without a
        trusted fov)."""
        angles = {}
        for row in self.get_round_rotation().itertuples():
            for member in str(row.members).split(';'):
                angles[int(member)] = float(row.angle_radians)
        return angles

    def get_round_angle(self, dataChannel: int) -> float:
        """The angle, in radians, of dataChannel's round. Raises if its
        round has no angle."""
        angles = self.get_round_angles()
        if int(dataChannel) not in angles:
            raise KeyError('%s: data channel %d is in no stack group'
                           % (self.get_analysis_name(), dataChannel))
        angle = angles[int(dataChannel)]
        if not np.isfinite(angle):
            raise ValueError('%s: the round of data channel %d has no '
                             'trusted fov, so no angle'
                             % (self.get_analysis_name(), dataChannel))
        return angle

    def get_fov_verdicts(self, fov: int) -> dict:
        """{data channel: (this fov's own angle in radians, NaN if
        untrusted; flagged; flag reason)} for the non-reference stacks of
        one fov."""
        table = self.get_fov_rotation()
        verdicts = {}
        for row in table[table.fov == fov].itertuples():
            own = float(row.angle_radians) if self._is_true(row.trusted) \
                else np.nan
            for member in str(row.members).split(';'):
                verdicts[int(member)] = (own, self._is_true(row.flagged),
                                         self._text(row.flag_reason))
        return verdicts
