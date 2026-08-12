import os
import subprocess
import cv2
import numpy as np
import scipy as sp
import scipy.fft as sp_fft
from concurrent.futures import ThreadPoolExecutor

from merlin.core import analysistask
from merlin.util import deconvolve
from merlin.util import aberration
from merlin.util import imagefilters
from merlin.data import codebook

from skimage import transform
from skimage import io

class Preprocess(analysistask.ParallelAnalysisTask):

    """
    An abstract class for preparing data for barcode calling.
    """

    def _image_name(self, fov):
        destPath = self.dataSet.get_analysis_subdirectory(
                self.analysisName, subdirectory='preprocessed_images')
        return os.sep.join([destPath, 'fov_' + str(fov) + '.tif'])

    def get_pixel_histogram(self, fov=None):
        if fov is not None:
            return self.dataSet.load_numpy_analysis_result(
                'pixel_histogram', self.analysisName, fov, 'histograms')

        pixelHistogram = np.zeros(self.get_pixel_histogram(
                self.dataSet.get_fovs()[0]).shape)
        for f in self.dataSet.get_fovs():
            pixelHistogram += self.get_pixel_histogram(f)

        return pixelHistogram

    def _save_pixel_histogram(self, histogram, fov):
        self.dataSet.save_numpy_analysis_result(
            histogram, 'pixel_histogram', self.analysisName, fov, 'histograms')

class DeconvolutionPreprocess(Preprocess):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 3
        # Frequency-domain high pass applied BEFORE the spatial high pass. The
        # transfer function is 1 - exp(-|k|^2 / (2 sigma^2)) with |k| measured in
        # FFT bins from the centred spectrum, matching the fft_hp3 filter that
        # won the M1 preprocessing comparison. Off by default; 0 disables it.
        if 'fft_highpass_sigma' not in self.parameters:
            self.parameters['fft_highpass_sigma'] = 0
        # All image filtering lives here now, so this carries the default that
        # Decode used to hold. 0 disables it.
        if 'lowpass_sigma' not in self.parameters:
            self.parameters['lowpass_sigma'] = 1
        if 'decon_sigma' not in self.parameters:
            self.parameters['decon_sigma'] = 2
        if 'decon_filter_size' not in self.parameters:
            self.parameters['decon_filter_size'] = \
                int(2 * np.ceil(2 * self.parameters['decon_sigma']) + 1)
        if 'decon_iterations' not in self.parameters:
            self.parameters['decon_iterations'] = 20
        if 'codebook_index' not in self.parameters:
            self.parameters['codebook_index'] = 0
        
        # add some options to save preprocessed images
        if 'write_preprocessed_images' not in self.parameters:
            self.parameters['write_preprocessed_images'] = False                
        if 'write_preprocessed_FOVs' not in self.parameters:
            self.parameters['write_preprocessed_FOVs'] = list(range(self.fragment_count()))
        if 'save_pixel_histogram' not in self.parameters:
            self.parameters['save_pixel_histogram'] = True
        if 'deconvolve_after_highpass' not in self.parameters:
            self.parameters['deconvolve_after_highpass'] = True
        if 'preprocess_z_index' not in self.parameters:
            self.parameters['preprocess_z_index'] = None
        if 'threshold_subtract_n' not in self.parameters:
            self.parameters['threshold_subtract_n'] = 0.0
        if 'threshold_subtract_mode' not in self.parameters:
            self.parameters['threshold_subtract_mode'] = 'none'
        # Bits of one FOV are filtered independently, so they can be fanned out
        # over threads. Must be matched by the cpus requested for this task and
        # for any task that calls get_processed_image_set (Optimize, Decode).
        if 'preprocess_threads' not in self.parameters:
            self.parameters['preprocess_threads'] = 1

        self._highPassSigma = self.parameters['highpass_sigma']
        self._fftHighPassSigma = self.parameters['fft_highpass_sigma']
        self._fftTransfer = None            # cached per image shape
        self._lowPassSigma = self.parameters['lowpass_sigma']
        self._deconSigma = self.parameters['decon_sigma']
        self._deconIterations = self.parameters['decon_iterations']
        self._thresholdSubtractN = float(self.parameters['threshold_subtract_n'])
        self._thresholdSubtractMode = str(
            self.parameters['threshold_subtract_mode']).lower()
        validSubtractModes = {'none', 'mean', 'std', 'both'}
        if self._thresholdSubtractMode not in validSubtractModes:
            raise ValueError(
                'threshold_subtract_mode must be one of '
                f'{sorted(validSubtractModes)}')

        self.warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task']]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters['codebook_index'])

    def get_processed_image_set(
            self, fov, zIndex: int = None,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        """Read, warp and filter every bit of one FOV.

        This is the hot loop of the whole pipeline: Optimize and Decode both go
        through it, and preprocessing dominates their runtime. The bits are
        independent, so preprocess_threads > 1 fans them out over a thread pool.
        Threads (not processes) because the heavy steps -- the FFT, the OpenCV
        blurs and the warp -- all release the GIL, and the results stay in
        shared memory instead of being pickled back.
        """
        org = self.dataSet.get_data_organization()
        channels = [org.get_data_channel_for_bit(b)
                    for b in self.get_codebook().get_bit_names()]
        zIndexes = ([zIndex] if zIndex is not None
                    else list(range(len(self.dataSet.get_z_positions()))))

        # Build the cached transfer function once, before any fan-out, so the
        # workers never race to populate it.
        if self._fftHighPassSigma:
            self._fft_transfer(tuple(self.dataSet.get_image_dimensions()))

        jobs = [(c, z) for c in channels for z in zIndexes]
        threads = max(1, int(self.parameters['preprocess_threads']))
        if threads > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(threads, len(jobs))) as pool:
                flat = list(pool.map(
                    lambda job: self.get_processed_image(
                        fov, job[0], job[1], chromaticCorrector), jobs))
        else:
            flat = [self.get_processed_image(fov, c, z, chromaticCorrector)
                    for c, z in jobs]

        stack = np.array(flat)
        if zIndex is None:
            return stack.reshape(len(channels), len(zIndexes),
                                 *stack.shape[-2:])
        return stack

    def get_processed_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        inputImage = self.warpTask.get_aligned_image(fov, dataChannel, zIndex,
                                                     chromaticCorrector)
        return self._preprocess_image(inputImage)

    def _highpass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        hpImage = inputImage
        if self._highPassSigma is not None and self._highPassSigma != 0:
            highPassFilterSize = int(2 * np.ceil(2 * self._highPassSigma) + 1)
            hpImage = imagefilters.highpass_filter(inputImage.astype(np.float32),
                                                    highPassFilterSize,
                                                    self._highPassSigma)
        return hpImage.astype(np.float32)

    def _fft_transfer(self, shape) -> np.ndarray:
        if self._fftTransfer is None or self._fftTransfer.shape != shape:
            height, width = shape
            u = np.arange(height) - height // 2
            v = np.arange(width) - width // 2
            vv, uu = np.meshgrid(v, u)
            sigma = float(self._fftHighPassSigma)
            lowpass = np.exp(-(uu ** 2 + vv ** 2) / (2.0 * sigma ** 2))
            # float32 to match the complex64 spectrum -- a float64 transfer
            # would silently promote the product back to complex128
            self._fftTransfer = (1.0 - lowpass).astype(np.float32)
        return self._fftTransfer

    def _fft_highpass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        """Frequency-domain high pass, negatives clipped to 0.

        Matches run_low_data_filter_decode_compare.fft_highpass so that a MERlin
        run with fft_highpass_sigma=3, highpass_sigma=3, lowpass_sigma=0.5
        reproduces the standalone fft_hp3_hp3_lp05 pipeline.

        scipy.fft is used rather than numpy.fft because numpy always promotes to
        complex128, which costs 2.7x the time and twice the memory for a
        transform whose input is float32 to begin with. scipy keeps float32 ->
        complex64, agreeing with the float64 result to ~2e-7 relative.
        """
        if self._fftHighPassSigma is None or self._fftHighPassSigma == 0:
            return inputImage.astype(np.float32, copy=False)
        image = np.asarray(inputImage, dtype=np.float32)
        spectrum = sp_fft.fftshift(sp_fft.fft2(image))
        spectrum *= self._fft_transfer(image.shape)
        output = np.real(sp_fft.ifft2(sp_fft.ifftshift(spectrum))).astype(np.float32)
        np.maximum(output, 0.0, out=output)
        return output

    def _deconvolve(self, inputImage: np.ndarray) -> np.ndarray:
        # deconvolve_lucyrichardson allocates ~10 full-size buffers before its
        # loop, so a 0-iteration call is an expensive no-op on 3200^2 frames.
        if not self._deconIterations or not self._deconSigma:
            return inputImage.astype(np.float32, copy=False)
        return deconvolve.deconvolve_lucyrichardson(
            inputImage, self.parameters['decon_filter_size'],
            self._deconSigma, self._deconIterations)

    def _lowpass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        lpImage = inputImage.astype(np.float32, copy=False)
        if self._lowPassSigma is None or self._lowPassSigma == 0:
            return lpImage
        lowPassFilterSize = int(2 * np.ceil(2 * self._lowPassSigma) + 1)
        return cv2.GaussianBlur(
            lpImage,
            (lowPassFilterSize, lowPassFilterSize),
            self._lowPassSigma,
            borderType=cv2.BORDER_REPLICATE).astype(np.float32)

    def _run_analysis(self, fragmentIndex):
            
        if self.parameters['save_pixel_histogram'] or (fragmentIndex in self.parameters['write_preprocessed_FOVs']):
    
            warpTask = self.dataSet.load_analysis_task(
                    self.parameters['warp_task'])

            histogramBins = np.arange(0, np.iinfo(np.uint16).max, 1)
            pixelHistogram = np.zeros(
                    (self.get_codebook().get_bit_count(), len(histogramBins)-1))

                # this currently only is to calculate the pixel histograms in order
                # to estimate the initial scale factors. This is likely unnecessary?
            
            outputTif = None
            zIndexes = self._get_z_indexes_to_preprocess()
            for bi, b in enumerate(self.get_codebook().get_bit_names()):
                dataChannel = self.dataSet.get_data_organization()\
                        .get_data_channel_for_bit(b)
                
                for i in zIndexes:
                    inputImage = warpTask.get_aligned_image(
                            fragmentIndex, dataChannel, i)
                    if self.parameters['deconvolve_after_highpass']:
                        deconvolvedImage = self._preprocess_image(inputImage)
                    else:
                        deconvolvedImage = self._preprocess_image_reversed(inputImage)
                    
                    pixelHistogram[bi, :] += np.histogram(
                        deconvolvedImage.astype(np.uint16), bins=histogramBins)[0]
                        
                    if self.parameters['write_preprocessed_images'] and fragmentIndex in self.parameters['write_preprocessed_FOVs']:
                        if outputTif is None:
                            outputTif = self.dataSet.writer_for_analysis_images(
                                self.analysisName, 'preprocessed_images', fragmentIndex).__enter__() 
                        outputTif.save(deconvolvedImage, photometric='MINISBLACK')
                            
            if outputTif is not None:
                outputTif.__exit__(None, None, None)
            
            self._save_pixel_histogram(pixelHistogram, fragmentIndex)

    def _get_z_indexes_to_preprocess(self) -> list[int]:
        zPositionCount = len(self.dataSet.get_z_positions())
        zIndex = self.parameters.get('preprocess_z_index')
        if zIndex is None:
            return list(range(zPositionCount))
        zIndex = int(zIndex)
        if zIndex < 0 or zIndex >= zPositionCount:
            raise ValueError(
                f'preprocess_z_index {zIndex} out of range for '
                f'{zPositionCount} z-positions')
        return [zIndex]
    
    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        filteredImage = self._fft_highpass_filter(inputImage)
        filteredImage = self._highpass_filter(filteredImage)
        filteredImage = self._subtract_global_threshold(filteredImage)
        filteredImage = self._lowpass_filter(filteredImage)
        return self._deconvolve(filteredImage)

    def _preprocess_image_reversed(self, inputImage: np.ndarray) -> np.ndarray:
        deconvolvedImage = self._deconvolve(inputImage.astype(np.float32))
        filteredImage = self._fft_highpass_filter(deconvolvedImage)
        filteredImage = self._highpass_filter(filteredImage)
        filteredImage = self._subtract_global_threshold(filteredImage)
        filteredImage = self._lowpass_filter(filteredImage)
        return filteredImage

    def _subtract_global_threshold(self, inputImage: np.ndarray) -> np.ndarray:
        mode = self._thresholdSubtractMode
        nFactor = self._thresholdSubtractN
        imageFloat = inputImage.astype(np.float32, copy=False)

        if mode == 'none' or nFactor == 0:
            return imageFloat

        imageMean = float(np.mean(imageFloat))
        imageStd = float(np.std(imageFloat))

        if mode == 'mean':
            threshold = nFactor * imageMean
        elif mode == 'std':
            threshold = nFactor * imageStd
        else:
            threshold = nFactor * (imageMean + imageStd)

        return np.maximum(imageFloat - threshold, 0.0).astype(np.float32)


class CARERestorePreprocess(DeconvolutionPreprocess):
    """DeconvolutionPreprocess with a CARE restoration applied to each warped image
    before the normal filter chain.

    It subclasses DeconvolutionPreprocess so the restored images go through exactly the
    same filter chain as a plain decode (e.g. M2 uses fft_highpass_sigma 3 ->
    highpass_sigma 3 -> lowpass_sigma 0.5); the only difference between a restored and a
    plain decode is then the restoration itself. An earlier CAREPreprocess class applied
    only a single spatial high pass, which would have filtered restored images
    differently from the decodes they are compared against; it has been removed in favour
    of this one.

    Normalization is explicit and must match how the model was trained:

        model_input = (warped_image - care_camera_offset) / care_input_scale
        restored    = model.predict(...) * care_input_scale + care_camera_offset

    care_use_csbdeep_normalizer defaults to FALSE, i.e. predict(normalizer=None). This
    matters: csbdeep's default is PercentileNormalizer(2, 99.8, do_after=True), which
    re-normalizes every image to its own percentile range before the network and maps
    the result back. For a model trained on a fixed global affine that is a train/apply
    mismatch which measurably halves the restored brightness. Set it True only for models
    that were themselves trained under csbdeep's percentile normalization.

    Parameters
      care_model_directory        path to the CARE model dir (parent dir + model name)
      care_camera_offset          default 0.0
      care_input_scale            default 65535.0 (uint16 full scale)
      care_use_csbdeep_normalizer default False -- see above
      care_n_tiles                optional [ny, nx] to bound peak memory
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'care_model_directory' not in self.parameters:
            raise ValueError(
                'CARERestorePreprocess requires care_model_directory')
        self.parameters.setdefault('care_camera_offset', 0.0)
        self.parameters.setdefault('care_input_scale', 65535.0)
        self.parameters.setdefault('care_use_csbdeep_normalizer', False)
        self.parameters.setdefault('care_n_tiles', None)
        # The histogram is only used to seed initial scale factors, and is expensive
        # here because every image must go through the network first. Off by default.
        self.parameters.setdefault('save_pixel_histogram', False)

        self._careModel = None      # loaded lazily; TF import is slow

    def _get_care_model(self):
        if self._careModel is None:
            try:
                from csbdeep.models import CARE
            except ImportError:
                raise ImportError(
                    '***CARE package (csbdeep.models.CARE) not found***')
            basedir, name = os.path.split(self.parameters['care_model_directory'])
            self._careModel = CARE(config=None, name=name, basedir=basedir)
        return self._careModel

    def _restore_image(self, inputImage: np.ndarray) -> np.ndarray:
        offset = float(self.parameters['care_camera_offset'])
        scale = float(self.parameters['care_input_scale'])
        kwargs = {}
        if not self.parameters['care_use_csbdeep_normalizer']:
            kwargs['normalizer'] = None
        if self.parameters['care_n_tiles'] is not None:
            kwargs['n_tiles'] = tuple(self.parameters['care_n_tiles'])
        x = (inputImage.astype(np.float32) - offset) / scale
        restored = self._get_care_model().predict(x, 'YX', **kwargs)
        return (restored * scale + offset).astype(np.float32)

    # both get_processed_image and _run_analysis funnel through these two, so
    # overriding them restores every path without touching the filter chain
    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        return super()._preprocess_image(self._restore_image(inputImage))

    def _preprocess_image_reversed(self, inputImage: np.ndarray) -> np.ndarray:
        return super()._preprocess_image_reversed(self._restore_image(inputImage))


class DeconvolutionPreprocessDW(Preprocess):
    
    def __init__(self, dataSet, parameters=None, analysisName=None):
            super().__init__(dataSet, parameters, analysisName)

            if 'codebook_index' not in self.parameters:
                self.parameters['codebook_index'] = 0
            if 'highpass_sigma' not in self.parameters:
                self.parameters['highpass_sigma'] = 3
            # turn off save pixel histogram?
            # this will assume initial scale factors are = 1 in Optimization
            if 'save_pixel_histogram' not in self.parameters:
                self.parameters['save_pixel_histogram'] = True
            if 'histogram_bin_max' not in self.parameters:
                self.parameters['histogram_bin_max'] = 10000000
                # due to the way scale factors are calculated
                # we need to bin at integer amounts
                # for float 32 we would need ridiculous number of bins...
                # not sure the best way to overcome this for now...

            self._highPassSigma = self.parameters['highpass_sigma']

            self.warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])
            
            # here are params for deconwolf

            if 'dw_path' not in self.parameters: 
                self.parameters['dw_path'] = 'dw' # assumes dw is in path
                
            if 'iterations' not in self.parameters:
                self.parameters['iterations'] = 15

            if 'use_gpu' not in self.parameters:
                self.parameters['use_gpu'] = True

            if 'overwrite' not in self.parameters:
                # will enable resumable decon
                self.parameters['overwrite'] = False 
            
            """
            # TURNOFF TILING it seems incompatible with --float...
            if 'tilesize' not in self.parameters:
                self.parameters['tilesize'] = 1024

            if 'tilepad' not in self.parameters:
                self.parameters['tilepad'] = 128
            """

            # find all the wavelengths and channels
            # but only for bits in codebook

            self.bits = self.get_codebook().get_bit_names()
            self.channels = [self.dataSet.get_data_organization().get_data_channel_for_bit(b) 
                             for b in self.bits]
            self.wavelengths = [self.dataSet.get_data_organization().get_data_channel_color(channel) 
                            for channel in self.channels]
            self.wavelengths = set(self.wavelengths)

            # make a dictionary of the PSF paths
            if 'psf_directory' in self.parameters:
                
                self.PSF_paths = {}
                base_path = self.parameters['psf_directory']
                for wavelength in self.wavelengths:
                    fpath = os.path.join(base_path, f'PSF_{wavelength}.tif')
                    if os.path.isfile(fpath):
                        self.PSF_paths[wavelength] = fpath
                    else:
                        raise ValueError(f'could not find PSF_{wavelength}.tif')
            else:
                    raise ValueError(f'no PSFs found')
            
            if 'remove_conventional_image' not in self.parameters:
                self.parameters['remove_conventional_image'] = True # saves space
            
    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 16384

    def get_estimated_time(self):
        return 60

    def get_dependencies(self):
        return [self.parameters['warp_task']]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters['codebook_index'])

    def get_raw_image_name(self, dataChannel: int) -> str:
        return f"channel_{dataChannel}_fov_"

    def get_raw_image_path(self, dataChannel: int, fov: int) -> str:
        imageBaseName = self.get_raw_image_name(dataChannel)
        return self.dataSet._analysis_image_name(
                self.analysisName, imageBaseName, fov)
    
    def get_dw_image_path(self, dataChannel: int, fov: int) -> str:
        imageBaseName = "dw_" + self.get_raw_image_name(dataChannel)
        return self.dataSet._analysis_image_name(
                self.analysisName, imageBaseName, fov)

    def get_processed_image_set(
            self, fov, zIndex: int = None,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        
        if zIndex is None:
            return np.array([[self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                for zIndex in range(len(self.dataSet.get_z_positions()))]
                for b in self.get_codebook().get_bit_names()])
        else:
            return np.array([self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                    for b in self.get_codebook().get_bit_names()])

    def get_processed_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None) -> np.ndarray:

        imagePath = self.get_dw_image_path(dataChannel, fov)
        inputImage = self.dataSet.load_image(imagePath, zIndex, transform = False) # images are already transformed

        transformation = self.warpTask.get_transformation(fov, dataChannel)

        # this is from the warp class
        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization()\
                            .get_data_channel_color(dataChannel)
            outputImage =  transform.warp(chromaticCorrector.transform_image(
                inputImage, imageColor), transformation, preserve_range=True
                ).astype(inputImage.dtype)
        else:
            outputImage = transform.warp(inputImage, transformation,
                                  preserve_range=True).astype(inputImage.dtype)

        # here is where the high pass happens
        outputImage = self._highpass_filter(outputImage)

        return outputImage
        
    def _highpass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        hpImage = inputImage
        if self._highPassSigma is None:
            highPassFilterSize = int(2 * np.ceil(2 * self._highPassSigma) + 1)
            hpImage = imagefilters.highpass_filter(inputImage.astype(np.float32),
                                                    highPassFilterSize,
                                                    self._highPassSigma)
        return hpImage.astype(np.float32)
    
    def _run_analysis(self, fragmentIndex):

        for bi, b in enumerate(self.bits): # this will only do bits in the codebook
            dataChannel = self.dataSet.get_data_organization().get_data_channel_for_bit(b)
            wavelength = self.dataSet.get_data_organization().get_data_channel_color(dataChannel)
            
            #  check if the channel is already deconvolved
            if self.parameters['overwrite'] == False:
                dw_image_path = self.get_dw_image_path(dataChannel, fragmentIndex)
                dw_image_name = os.path.split(dw_image_path)[-1]
                if os.path.exists(dw_image_path):
                    print(f'found {dw_image_name}, skipping dw on channel {dataChannel}')
                    continue # skip the loop
            
            # write the raw image zstacks to disk
            with self.dataSet.writer_for_analysis_images(
                     self.analysisName,
                     self.get_raw_image_name(dataChannel), 
                     fragmentIndex) as outputTif:
                
                for zPosition in self.dataSet.get_z_positions():
                        frame = self.dataSet.get_raw_image(dataChannel, fragmentIndex, zPosition)
                        outputTif.save(frame, photometric='MINISBLACK')

            # this is the path of the image that was just saved
            inputImagePath = self.get_raw_image_path(dataChannel, fragmentIndex)

            # compose the dw command
            dw_command = []
            dw_command.append(self.parameters['dw_path'])
            dw_command.append('--iter')
            dw_command.append(str(self.parameters['iterations']))
            if self.parameters['use_gpu']:
                dw_command.append('--gpu')
            if self.parameters['overwrite']:
                dw_command.append('--overwrite')
            dw_command.append('--float') # this may be important so the image is not scaled funny...

            # turning off tiling, it seems incompatible with the --float option
            # also scale does not seem to work...
            #dw_command.append('--tilesize')
            #dw_command.append(str(self.parameters['tilesize']))
            #dw_command.append('--tilepad')
            #dw_command.append(str(self.parameters['tilepad']))
            #dw_command.append('--out') # don't use
            #dw_command.append(outputImagePath) # don't use
            
            dw_command.append(inputImagePath)
            dw_command.append(self.PSF_paths[wavelength])

            if True: # for troubleshooting
                print('running dw command: ' + ' '.join(dw_command))

            # run dw
            try:
                ret = subprocess.run(dw_command, check = True)
            except subprocess.CalledProcessError as e:
                raise Exception(f'dw error on channel {dataChannel} fov {fragmentIndex}')
                # I believe this should get caught by the analysistask

            #if ret.returncode != 0:
            #    raise Exception(f'dw error on channel {dataChannel} fov {fragmentIndex}')

            # remove the conventional image?
            if self.parameters['remove_conventional_image']:
                os.remove(inputImagePath)
        
        # calculate pixel histogram?
        if self.parameters['save_pixel_histogram']:

            # see note in params about histogram bin max
            # annoying to calculate this for float32 images
            # but may be necessary since thats what dw should output

            histogramBins = np.arange(0, self.parameters['histogram_bin_max'], 1)
            pixelHistogram = np.zeros(
                            (len(self.bits),
                             len(histogramBins)-1), np.int32)

            for bi, b in enumerate(self.bits): # only do bits in the codebook
                dataChannel = self.dataSet.get_data_organization().get_data_channel_for_bit(b)

                imagePath = self.get_dw_image_path(dataChannel, fragmentIndex)
                dw_image = io.imread(imagePath)
                preprocessedImage = np.array([self._highpass_filter(im) for im in dw_image])
                # since this is a lot of data and a lot of bins to histogram
                # do a max projection
                preprocessedImage = np.amax(preprocessedImage, axis = 0)

                # finally do histogram
                h, _ = np.histogram(preprocessedImage, bins=histogramBins)

                # write that to the histogram file
                pixelHistogram[bi, :] = h

            self._save_pixel_histogram(pixelHistogram, fragmentIndex)
    
    def _save_pixel_histogram(self, histogram, fov):
        # get a save path
        savePath = self.dataSet._analysis_result_save_path(
                'pixel_histogram', 
                self.analysisName, 
                fov, 
                'histograms',
                '.npz')
        # convert to spares matrix
        sparse_matrix = sp.sparse.csr_matrix(histogram)
        sp.sparse.save_npz(savePath, sparse_matrix)

class DeconvolutionPreprocessGuo(DeconvolutionPreprocess):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # Check for 'decon_iterations' in parameters instead of
        # self.parameters as 'decon_iterations' is added to
        # self.parameters by the super-class with a default value
        # of 20, but we want the default value to be 2.
        if 'decon_iterations' not in parameters:
            self.parameters['decon_iterations'] = 2
        
        self._deconIterations = self.parameters['decon_iterations']
        
    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        deconFilterSize = self.parameters['decon_filter_size']
        filteredImage = self._highpass_filter(inputImage.astype(np.float32))
        filteredImage = self._subtract_global_threshold(filteredImage)
        filteredImage = self._lowpass_filter(filteredImage)
        deconvolvedImage = deconvolve.deconvolve_lucyrichardson_guo(
            filteredImage, deconFilterSize, self._deconSigma,
            self._deconIterations)
        deconvolvedImage = deconvolvedImage.astype(np.uint16)
        return deconvolvedImage
