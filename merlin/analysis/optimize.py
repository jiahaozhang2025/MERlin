import numpy as np
import scipy as sp
import itertools
from skimage import transform
from typing import Dict
from typing import List
import pandas
import random
import pickle
import os
import time

from concurrent.futures import ThreadPoolExecutor

from merlin.analysis import decode
from merlin.core import analysistask
from merlin.util import decoding
from merlin.util import registration
from merlin.util import aberration
from merlin.data.codebook import Codebook


class OptimizeIteration(decode.BarcodeSavingParallelAnalysisTask):

    """
    An analysis task for performing a single iteration of scale factor
    optimization.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'distance_metric' not in self.parameters:
            self.parameters['distance_metric'] = 'euclidean'
        if 'fov_per_iteration' not in self.parameters:
            self.parameters['fov_per_iteration'] = 50
        if 'area_threshold' not in self.parameters:
            self.parameters['area_threshold'] = 4
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 0.5176
        if 'optimize_background' not in self.parameters:
            self.parameters['optimize_background'] = False
        if 'optimize_chromatic_correction' not in self.parameters:
            self.parameters['optimize_chromatic_correction'] = False
        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 100
        # See the matching note in decode.Decode: all image filtering is done by
        # the preprocess task, so Optimize decodes exactly the pixels Decode
        # will decode.
        if 'lowpass_sigma' in self.parameters:
            raise ValueError(
                'lowpass_sigma is no longer an OptimizeIteration parameter -- '
                'set it on the preprocess task instead.')
        if 'tile_overlap' not in self.parameters:
            self.parameters['tile_overlap'] = 20
        # threads for the nearest-neighbour decode (sklearn n_jobs); must be
        # matched by the cpus requested for this task
        if 'num_threads' not in self.parameters:
            self.parameters['num_threads'] = 1
        # threads for estimating this iteration's chromatic corrections, which
        # fan out over independent (fov, z) groups. Set the cpus on the
        # ChromaticCorrection task that drives it, not on this one.
        if 'chromatic_threads' not in self.parameters:
            self.parameters['chromatic_threads'] = 1
        # Caps on how much data the chromatic fit consumes. 0 = no cap, which
        # reproduces the previous behaviour exactly. See the sampling note in
        # _get_chromatic_transformations for why capping costs no precision.
        if 'chromatic_max_barcodes_per_group' not in self.parameters:
            self.parameters['chromatic_max_barcodes_per_group'] = 0
        if 'chromatic_max_groups' not in self.parameters:
            self.parameters['chromatic_max_groups'] = 0
        # remove per-fragment results once finalize() has aggregated them
        if 'cleanup_fragment_results' not in self.parameters:
            self.parameters['cleanup_fragment_results'] = False
        # Measure the chromatic displacement samples inside each fragment,
        # which already holds that (fov, z)'s images and barcodes, instead of
        # re-loading everything in one job afterwards. Fragments save raw
        # SAMPLES, not fitted transforms -- finalize() pools them, which
        # reproduces the single-job fit exactly. Averaging per-fragment fits
        # would not: a fragment with few barcodes gives a badly conditioned
        # rotation/scale estimate that contaminates the mean.
        if 'chromatic_from_fragments' not in self.parameters:
            self.parameters['chromatic_from_fragments'] = False
        # Which images the fragment measures on. The preprocessed set is already
        # in memory (free); the raw-warped set costs an extra load but is what
        # the single-job path uses. They are NOT interchangeable -- both
        # high-pass steps clip negatives, which is asymmetric and shifts
        # centroids.
        if 'chromatic_on_preprocessed' not in self.parameters:
            self.parameters['chromatic_on_preprocessed'] = False
        # Optimize decodes the FULL frame and discards barcodes within
        # crop_width of the edge (barcode space, not image space). With
        # adaptive_crop the discarded margin is this FOV's own invalid region
        # instead of a fixed worst-case border, so the scale factors and the
        # chromatic samples are fit on valid pixels only.
        if 'adaptive_crop' not in self.parameters:
            self.parameters['adaptive_crop'] = False
        if 'random_seed' in self.parameters:
            # set the random seed
            # make sure to set a different one for each optimize
            np.random.seed(self.parameters['random_seed'])

            # save the optimized images
        if 'write_decoded_images' not in self.parameters:
            self.parameters['write_decoded_images'] = True

        if 'fov_index' in self.parameters:
            logger = self.dataSet.get_logger(self)
            logger.info('Setting fov_per_iteration to length of fov_index')

            self.parameters['fov_per_iteration'] = \
                len(self.parameters['fov_index'])
        
        # specify fovs and zIndices separately
        elif ('fovs' in self.parameters) and ('zIndices' in self.parameters):
        
            self.parameters['fov_index'] = []
            fovIndex = np.random.choice(list(self.parameters['fovs']), 
                size = self.parameters['fov_per_iteration'])
                    
            zIndex = np.random.choice(list(self.parameters['zIndices']),
                size = self.parameters['fov_per_iteration'])
                
            self.parameters['fov_index'] = [[int(fov),int(ind)] for fov, ind in zip(fovIndex, zIndex)]
            
        # this should fix the issue of optimize choosing different FOVs on rerun...
        else:
            
            self.parameters['fov_index'] = []
            fovIndex = np.random.choice(list(self.dataSet.get_fovs()), 
                size = self.parameters['fov_per_iteration'])
                    
            zIndex = np.random.choice(list(range(len(self.dataSet.get_z_positions()))),
                size = self.parameters['fov_per_iteration'])
                
            self.parameters['fov_index'] = [[int(fov),int(ind)] for fov, ind in zip(fovIndex, zIndex)]
        # add parameter to only optimize inside the segmentation mask:
        # probably only do this with cellpose 3D class
        # specify the segment task to use
        if 'use_segmentation_mask' not in self.parameters:
            self.parameters['use_segmentation_mask'] = False

        # gpu decoding
        if 'use_gpu' not in self.parameters:
            self.parameters['use_gpu'] = False

        if 'min_barcodes_for_refactoring' not in self.parameters:
            self.parameters['min_barcodes_for_refactoring'] = 0

        # tiling factor for large images to avoid OOM
        if 'tiling_factor' not in self.parameters:
            self.parameters['tiling_factor'] = None


    def get_estimated_memory(self):
        return 4000

    def get_estimated_time(self):
        return 60

    def get_dependencies(self):
        dependencies = [self.parameters['preprocess_task'],
                        self.parameters['warp_task']]
        if 'previous_iteration' in self.parameters:
            dependencies += [self.parameters['previous_iteration']]
        if self.parameters['use_segmentation_mask']:
            dependencies += [self.parameters['use_segmentation_mask']]
        return dependencies

    def fragment_count(self):
        return self.parameters['fov_per_iteration']

    def _measure_chromatic_samples(self, images, barcodes):
        """Per-colour-pair (position, displacement) samples for one (fov, z).

        Shared by the per-fragment path and the single-job path so both measure
        identically; only where it runs and which images it sees differ.
        """
        codebook = self.get_codebook()
        org = self.dataSet.get_data_organization()
        usedColors = self._get_used_colors()
        out = {u: {v: ([], []) for v in usedColors if v >= u} for u in usedColors}
        X = barcodes['x'].to_numpy(float)
        Y = barcodes['y'].to_numpy(float)
        B = barcodes['barcode_id'].to_numpy(int)
        hLimit = images.shape[1] - 10
        wLimit = images.shape[2] - 10
        for bx, by, bid in zip(X, Y, B):
            if not (bx > 10 and by > 10 and hLimit > bx and wLimit > by):
                continue
            onBits = np.where(codebook.get_barcode(bid))[0]
            refined = np.array([registration.refine_position(images[i], bx, by)
                                for i in onBits])
            for p in itertools.combinations(enumerate(onBits), 2):
                c1 = org.get_data_channel_color(p[0][1])
                c2 = org.get_data_channel_color(p[1][1])
                if c1 < c2:
                    out[c1][c2][0].append((bx, by))
                    out[c1][c2][1].append(refined[p[1][0]] - refined[p[0][0]])
                else:
                    out[c2][c1][0].append((bx, by))
                    out[c2][c1][1].append(refined[p[0][0]] - refined[p[1][0]])
        # Two compact (N, 2) arrays per pair. Storing a list of 2-element numpy
        # arrays instead meant pickling ~200k tiny objects per fragment (179 MB
        # an iteration), which cost more than the measurement it was saving.
        return {c1: {c2: (np.asarray(v[0], dtype=np.float64).reshape(-1, 2),
                          np.asarray(v[1], dtype=np.float64).reshape(-1, 2))
                     for c2, v in inner.items()}
                for c1, inner in out.items()}

    def finalize(self) -> None:
        """Compute this iteration's aggregates once, in the Done rule.

        All three of these are lazy-on-cache-miss, and nothing triggers them
        until the next iteration's fragments ask -- at which point all of them
        ask at once, all miss, and all recompute. The chromatic estimate is the
        expensive one by orders of magnitude; the other two are a median over
        small per-fragment .npy files. Results are written under this task's own
        directory, so Optimize{N}/chromatic_corrections.pkl is where they live.
        """
        self._get_chromatic_transformations()
        self.get_scale_factors()
        self.get_backgrounds()
        self._merge_barcode_counts()
        if self.parameters['cleanup_fragment_results']:
            self._cleanup_fragment_results()

    def _merge_barcode_counts(self) -> None:
        """Collapse the per-fragment barcode counts into a single array.

        These are the one per-fragment result a LATER iteration still reads:
        get_barcode_count_history() walks back through every previous iteration.
        Merging them here is what makes cleanup possible at all.
        """
        try:
            self.dataSet.load_numpy_analysis_result(
                'barcode_counts_merged', self.analysisName)
            return
        except (FileNotFoundError, OSError, ValueError):
            pass
        countsMean = np.mean([self.dataSet.load_numpy_analysis_result(
            'barcode_counts', self.analysisName, resultIndex=i)
            for i in range(self.parameters['fov_per_iteration'])], axis=0)
        self.dataSet.save_numpy_analysis_result(
            countsMean, 'barcode_counts_merged', self.analysisName)

    def _cleanup_fragment_results(self) -> None:
        """Delete per-fragment results whose aggregates are now on disk.

        Only files that provably have no remaining reader are removed:
        scale_refactors/previous_scale_factors are folded into scale_factors,
        background_refactors/previous_backgrounds into backgrounds, and
        barcode_counts into barcode_counts_merged. select_frame is kept -- it
        records which (fov, z) each fragment used, which is the provenance you
        want when a fit looks wrong, and it is a two-element array.
        """
        merged = {'scale_factors', 'backgrounds', 'barcode_counts_merged'}
        for name in merged:                       # refuse to delete without them
            self.dataSet.load_numpy_analysis_result(name, self.analysisName)
        stale = ['scale_refactors', 'previous_scale_factors',
                 'background_refactors', 'previous_backgrounds',
                 'barcode_counts']
        removed = 0
        for i in range(self.parameters['fov_per_iteration']):
            for name in stale:
                path = self.dataSet._analysis_result_save_path(
                    name, self.analysisName, resultIndex=i,
                    fileExtension='.npy')
                if os.path.exists(path):
                    os.remove(path)
                    removed += 1
        self.dataSet.get_logger(self).info(
            'Removed %i per-fragment result files after aggregation', removed)

    def get_codebook(self) -> Codebook:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        return preprocessTask.get_codebook()
    
    # used to load in the segmentation mask
    def get_segmentation_mask(self, fovIndex, zIndex):
        segmentTask = self.dataSet.load_analysis_task(
            self.parameters['use_segmentation_mask'])
        #downsample_factor = segmentTask.parameters['downsample_factor'] # not necessary
        return segmentTask._load_mask_image(fovIndex, zIndex)

    def _run_analysis(self, fragmentIndex):
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        codebook = self.get_codebook()

        fovIndex, zIndex = self.parameters['fov_index'][fragmentIndex]

        scaleFactors = self._get_previous_scale_factors()
        backgrounds = self._get_previous_backgrounds()
        chromaticTransformations = \
            self._get_previous_chromatic_transformations()

        self.dataSet.save_numpy_analysis_result(
            scaleFactors, 'previous_scale_factors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            backgrounds, 'previous_backgrounds', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_pickle_analysis_result(
            chromaticTransformations, 'previous_chromatic_corrections',
            self.analysisName, resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            np.array([fovIndex, zIndex]), 'select_frame', self.analysisName,
            resultIndex=fragmentIndex)

        t0 = time.time()

        chromaticCorrector = aberration.RigidChromaticCorrector(
            chromaticTransformations, self.get_reference_color())
        warpedImages = preprocessTask.get_processed_image_set(
            fovIndex, zIndex=zIndex, chromaticCorrector=chromaticCorrector)

        t1 = time.time()

        decoder = decoding.PixelBasedDecoder(codebook)
        areaThreshold = self.parameters['area_threshold']
        distance_threshold = self.parameters['distance_threshold']
        decoder.refactorAreaThreshold = areaThreshold

        # this defaults to zero and will cause no change
        decoder.barcodesSeenThreshold = self.parameters['min_barcodes_for_refactoring']

        decodeMask = None
        if self.parameters['use_segmentation_mask']: # masked decode
            decodeMask = self.get_segmentation_mask(fovIndex, zIndex)
            
        di, pm, npt, d = decoder.decode_pixels(warpedImages,
                                            scaleFactors,
                                            backgrounds,
                                            decodeMask = decodeMask,
                                            lowPassSigma = 0,
                                            overlap = self.parameters['tile_overlap'],
                                            numThreads = self.parameters['num_threads'],
                                            distanceThreshold = distance_threshold,
                                            distanceMetric = self.parameters['distance_metric'],
                                            useGpu = self.parameters['use_gpu'],
                                            tilingFactor = self.parameters['tiling_factor'])
        
        t2 = time.time()

        refactors, backgrounds, barcodesSeen = \
            decoder.extract_refactors(
                di, pm, npt, extractBackgrounds=self.parameters[
                    'optimize_background'])

        t3 = time.time()

        # TODO this saves the barcodes under fragment instead of fov
        # the barcodedb should be made more general
        cropWidth = self.parameters['crop_width']

        extracted = decoder.extract_barcodes_with_index(
            di, pm, npt, d, fovIndex,
            0 if self.parameters['adaptive_crop'] else cropWidth,
            zIndex, minimumArea=areaThreshold)
        if self.parameters['adaptive_crop']:
            r0, r1, c0, c1 = decode.compute_crop_bounds(
                self.dataSet, self.parameters['warp_task'], fovIndex,
                cropWidth, True)
            # x is a column, y is a row
            extracted = extracted[extracted['x'].between(c0, c1)
                                  & extracted['y'].between(r0, r1)]
        self.get_barcode_database().write_barcodes(extracted,
                                                   fov=fragmentIndex)
        
        t4 = time.time()

        self.dataSet.save_numpy_analysis_result(
            refactors, 'scale_refactors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            backgrounds, 'background_refactors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            barcodesSeen, 'barcode_counts', self.analysisName,
            resultIndex=fragmentIndex)

        if self.parameters['chromatic_from_fragments']:
            # This fragment already holds its (fov, z) images and barcodes, so
            # measuring here avoids the single-job path re-loading and
            # re-warping all 24 images per group afterwards.
            ownBarcodes = self.get_barcode_database().get_barcodes(
                fov=fragmentIndex)
            if self.parameters['chromatic_on_preprocessed']:
                chromaticImages = warpedImages      # already in memory, free
            else:
                warpTask = self.dataSet.load_analysis_task(
                    self.parameters['warp_task'])
                chromaticImages = np.array([warpTask.get_aligned_image(
                    fovIndex,
                    self.dataSet.get_data_organization()
                        .get_data_channel_for_bit(b),
                    int(zIndex), chromaticCorrector)
                    for b in codebook.get_bit_names()])
            self.dataSet.save_pickle_analysis_result(
                self._measure_chromatic_samples(chromaticImages, ownBarcodes),
                'chromatic_samples', self.analysisName,
                resultIndex=fragmentIndex)

        # save the decoded image from optimize
        if self.parameters['write_decoded_images']:
            imageDescription = self.dataSet.analysis_tiff_description(1, 3)
            with self.dataSet.writer_for_analysis_images(
                    self, 'decoded', fragmentIndex) as outputTif:
                for im in [di, pm, d]:
                    outputTif.save(im.astype(np.float32),
                                   photometric='MINISBLACK',
                                   contiguous=True,
                                   metadata=imageDescription)

        print(f'optimize fragment {fragmentIndex} fov {fovIndex} zIndex {zIndex}')
        print(f'time fetching images: {t1-t0}')
        print(f'time decoding images: {t2-t1}')
        print(f'time extracting refactors: {t3-t2}')
        print(f'time extracing barcodes: {t4-t3}')
        print(f'total time in optimize{fragmentIndex}: {t4-t0}')

    def _get_used_colors(self) -> List[str]:
        dataOrganization = self.dataSet.get_data_organization()
        codebook = self.get_codebook()
        return sorted({dataOrganization.get_data_channel_color(
            dataOrganization.get_data_channel_for_bit(x))
            for x in codebook.get_bit_names()})

    def _calculate_initial_scale_factors(self) -> np.ndarray:
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        bitCount = self.get_codebook().get_bit_count()

        # from Rongxin - setting initial scale factors = 1 if no pixel histograms
        initialScaleFactors = np.ones(bitCount, dtype = np.float32)

        if preprocessTask.parameters['save_pixel_histogram']:
            pixelHistograms = preprocessTask.get_pixel_histogram()
            for i in range(bitCount):
                h = pixelHistograms[i]
                if isinstance(h, sp.sparse.spmatrix): # allow for sparse matrix
                    h = h.toarray()
                cumulativeHistogram = np.cumsum(h)
                cumulativeHistogram = cumulativeHistogram/cumulativeHistogram[-1]
                # Add two to match matlab code.
                # TODO: Does +2 make sense? Used to be consistent with Matlab code
                initialScaleFactors[i] = \
                    np.argmin(np.abs(cumulativeHistogram-0.9)) + 2
            
        return initialScaleFactors

    def _get_previous_scale_factors(self) -> np.ndarray:
        if 'previous_iteration' not in self.parameters:
            scaleFactors = self._calculate_initial_scale_factors()
        else:
            previousIteration = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration'])
            scaleFactors = previousIteration.get_scale_factors()

        return scaleFactors

    def _get_previous_backgrounds(self) -> np.ndarray:
        if 'previous_iteration' not in self.parameters:
            backgrounds = np.zeros(self.get_codebook().get_bit_count())
        else:
            previousIteration = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration'])
            backgrounds = previousIteration.get_backgrounds()

        return backgrounds

    def _get_previous_chromatic_transformations(self)\
            -> Dict[str, Dict[str, transform.SimilarityTransform]]:
        
        # try to load in a pre-corrected chromatic transformation first
        # and save it as the chromatic_corrections.pkl file
        # this should avoid doing the majority of the work in _get_chromatic_transformations()
        # however make sure to have parameters['optimize_chromatic_correction'] = true
        if 'chromatic_correction_file' in self.parameters:
            with open(self.parameters['chromatic_correction_file'], 'rb') as f:
                chromaticTransformations = pickle.load(f)
            # is it necessary to save?
            savePath = self.dataSet._analysis_result_save_path(
                'chromatic_corrections', self.analysisName)
            if not os.path.exists(savePath):
                self.dataSet.save_pickle_analysis_result(
                    chromaticTransformations, 'chromatic_corrections', self.analysisName)
                    
            return chromaticTransformations
        
        # I believe this should only apply for the first optimization round where it is not specified
        if 'previous_iteration' not in self.parameters:
            usedColors = self._get_used_colors()
            return {u: {v: transform.SimilarityTransform()
                        for v in usedColors if v >= u} for u in usedColors}
        
        # this is a time consuming step... see above
        else:
            previousIteration = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration'])
            return previousIteration._get_chromatic_transformations()

    # TODO the next two functions could be in a utility class. Make a
    #  chromatic aberration utility class

    def get_reference_color(self):
        return min(self._get_used_colors())

    def get_previous_chromatic_corrector(self) -> aberration.ChromaticCorrector:
        """The corrector this iteration's own fragments decoded under.

        This iteration's scale factors were fit on images corrected with these
        transformations, not with the ones estimated afterwards from this
        iteration's barcodes. Downstream tasks that consume the scale factors
        should use this to stay self-consistent.
        """
        return aberration.RigidChromaticCorrector(
            self._get_previous_chromatic_transformations(),
            self.get_reference_color())

    def get_chromatic_corrector(self) -> aberration.ChromaticCorrector:
        """Get the chromatic corrector estimated from this optimization
        iteration

        Returns:
            The chromatic corrector.
        """
        return aberration.RigidChromaticCorrector(
            self._get_chromatic_transformations(), self.get_reference_color())

    def _get_chromatic_transformations(self) \
            -> Dict[str, Dict[str, transform.SimilarityTransform]]:
        """Get the estimated chromatic corrections from this optimization
        iteration.

        Returns:
            a dictionary of dictionary of transformations for transforming
            the farther red colors to the most blue color. The transformation
            for transforming the farther red color, e.g. '750', to the
            farther blue color, e.g. '560', is found at result['560']['750']
        """
        if not self.is_complete():
            raise Exception('Analysis is still running. Unable to get scale '
                            + 'factors.')

        if not self.parameters['optimize_chromatic_correction']:
            usedColors = self._get_used_colors()
            return {u: {v: transform.SimilarityTransform()
                        for v in usedColors if v >= u} for u in usedColors}

        try:
            return self.dataSet.load_pickle_analysis_result(
                'chromatic_corrections', self.analysisName)
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError):
            # TODO - this is messy. It can be broken into smaller subunits and
            # most parts could be included in a chromatic aberration class
            previousTransformations = \
                self._get_previous_chromatic_transformations()

            if self.parameters['chromatic_from_fragments']:
                # Pool the samples the fragments already measured. Pooling (not
                # averaging their fits) makes this identical to the single-job
                # result for the same images.
                usedColors = self._get_used_colors()
                acc = {u: {v: ([], []) for v in usedColors if v >= u}
                       for u in usedColors}
                for i in range(self.parameters['fov_per_iteration']):
                    part = self.dataSet.load_pickle_analysis_result(
                        'chromatic_samples', self.analysisName, resultIndex=i)
                    for c1 in part:
                        for c2 in part[c1]:
                            pos, disp = part[c1][c2]
                            acc[c1][c2][0].append(pos)
                            acc[c1][c2][1].append(disp)
                pooled = {c1: {c2: (np.concatenate(v[0]) if v[0] else
                                    np.zeros((0, 2), np.float64),
                                    np.concatenate(v[1]) if v[1] else
                                    np.zeros((0, 2), np.float64))
                               for c2, v in inner.items()}
                          for c1, inner in acc.items()}
                return self._fit_color_pairs(pooled, previousTransformations)

            previousCorrector = aberration.RigidChromaticCorrector(
                previousTransformations, self.get_reference_color())
            codebook = self.get_codebook()
            dataOrganization = self.dataSet.get_data_organization()

            barcodes = self.get_barcode_database().get_barcodes()
            uniqueFOVs = np.unique(barcodes['fov'])
            warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])

            usedColors = self._get_used_colors()
            colorPairDisplacements = {u: {v: [] for v in usedColors if v >= u}
                                      for u in usedColors}

            # Each (fov, z) group is independent: it loads its own 24 warped
            # images and contributes displacement samples that are pooled into
            # one least-squares fit per colour pair at the end. Order of the
            # samples does not affect the fit, but results are merged in group
            # order anyway so the output is reproducible regardless of thread
            # scheduling.
            groups = [(int(fov), z)
                      for fov in uniqueFOVs
                      for z in np.unique(barcodes[barcodes['fov'] == fov]['z'])]

            def measure_group(group):
                fov, z = group
                local = {u: {v: [] for v in usedColors if v >= u}
                         for u in usedColors}
                currentBarcodes = barcodes[(barcodes['fov'] == fov)
                                           & (barcodes['z'] == z)]
                # The fit is a 4-DOF similarity transform per colour pair, so
                # its precision saturates after a few thousand samples: the
                # standard error goes as sigma/sqrt(N), which at sigma ~0.5 px
                # is already 0.006 px by N=8000, against chromatic offsets of
                # order 0.1-1 px. Everything past that is 300 us per barcode
                # (four refine_position calls) bought for nothing.
                maxBC = self.parameters['chromatic_max_barcodes_per_group']
                if maxBC and len(currentBarcodes) > maxBC:
                    currentBarcodes = currentBarcodes.sample(
                        n=maxBC, random_state=int(self.parameters
                                                  .get('random_seed', 0)))
                warpedImages = np.array([warpTask.get_aligned_image(
                    fov, dataOrganization.get_data_channel_for_bit(b),
                    int(z),  previousCorrector)
                    for b in codebook.get_bit_names()])

                # pandas iterrows costs 14.5 us per row against 0.1 us for a
                # plain numpy column read, which is real money next to the
                # ~300 us of refinement it wraps
                bcX = currentBarcodes['x'].to_numpy(float)
                bcY = currentBarcodes['y'].to_numpy(float)
                bcId = currentBarcodes['barcode_id'].to_numpy(int)
                hLimit = warpedImages.shape[1] - 10
                wLimit = warpedImages.shape[2] - 10
                for bx, by, bid in zip(bcX, bcY, bcId):
                    onBits = np.where(codebook.get_barcode(bid))[0]

                    # TODO this can be done by crop width when decoding
                    if bx > 10 and by > 10 and hLimit > bx and wLimit > by:

                        refinedPositions = np.array(
                            [registration.refine_position(
                                warpedImages[i, :, :], bx, by)
                                for i in onBits])
                        for p in itertools.combinations(
                                enumerate(onBits), 2):
                            c1 = dataOrganization.get_data_channel_color(
                                p[0][1])
                            c2 = dataOrganization.get_data_channel_color(
                                p[1][1])

                            if c1 < c2:
                                local[c1][c2].append(
                                    [np.array([bx, by]),
                                     refinedPositions[p[1][0]]
                                     - refinedPositions[p[0][0]]])
                            else:
                                local[c2][c1].append(
                                    [np.array([bx, by]),
                                     refinedPositions[p[0][0]]
                                     - refinedPositions[p[1][0]]])
                return local

            maxGroups = self.parameters['chromatic_max_groups']
            if maxGroups and len(groups) > maxGroups:
                # evenly spaced rather than the first N, so the sample spans the
                # whole set of fovs the iteration touched
                pick = np.linspace(0, len(groups) - 1, maxGroups).astype(int)
                groups = [groups[i] for i in pick]

            threads = max(1, int(self.parameters['chromatic_threads']))
            if threads > 1 and len(groups) > 1:
                with ThreadPoolExecutor(
                        max_workers=min(threads, len(groups))) as pool:
                    perGroup = list(pool.map(measure_group, groups))
            else:
                perGroup = [measure_group(g) for g in groups]

            acc = {u: {v: ([], []) for v in usedColors if v >= u}
                   for u in usedColors}
            for local in perGroup:
                for c1, inner in local.items():
                    for c2, (pos, disp) in inner.items():
                        acc[c1][c2][0].append(pos)
                        acc[c1][c2][1].append(disp)
            colorPairDisplacements = {
                c1: {c2: (np.concatenate(v[0]) if v[0] else
                          np.zeros((0, 2), np.float64),
                          np.concatenate(v[1]) if v[1] else
                          np.zeros((0, 2), np.float64))
                     for c2, v in inner.items()}
                for c1, inner in acc.items()}

            return self._fit_color_pairs(colorPairDisplacements,
                                         previousTransformations)

    def _fit_color_pairs(self, colorPairDisplacements, previousTransformations):
        """Fit one similarity transform per colour pair and compose it onto the
        previous iteration's, then cache. Shared by the single-job and
        per-fragment paths so the two differ only in where the samples came
        from, never in how they are fitted."""
        tForms = {}
        for k, v in colorPairDisplacements.items():
            tForms[k] = {}
            for k2, (pos, disp) in v.items():
                tForm = transform.SimilarityTransform()
                good = np.isfinite(disp).all(axis=1)
                tForm.estimate(pos[good], pos[good] + disp[good])
                tForms[k][k2] = tForm + previousTransformations[k][k2]

        self.dataSet.save_pickle_analysis_result(
            tForms, 'chromatic_corrections', self.analysisName)

        return tForms

    def get_scale_factors(self) -> np.ndarray:
        """Get the final, optimized scale factors.

        Returns:
            a one-dimensional numpy array where the i'th entry is the
            scale factor corresponding to the i'th bit.
        """
        if not self.is_complete():
            raise Exception('Analysis is still running. Unable to get scale '
                            + 'factors.')

        try:
            return self.dataSet.load_numpy_analysis_result(
                'scale_factors', self.analysisName)
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError):
            refactors = np.array([self.dataSet.load_numpy_analysis_result(
                    'scale_refactors', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])])

            # Don't rescale bits that were never seen
            refactors[refactors == 0] = 1

            previousFactors = np.array([self.dataSet.load_numpy_analysis_result(
                'previous_scale_factors', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])])

            scaleFactors = np.nanmedian(
                    np.multiply(refactors, previousFactors), axis=0)

            self.dataSet.save_numpy_analysis_result(
                scaleFactors, 'scale_factors', self.analysisName)

            return scaleFactors

    def get_backgrounds(self) -> np.ndarray:
        if not self.is_complete():
            raise Exception('Analysis is still running. Unable to get ' +
                            'backgrounds.')

        try:
            return self.dataSet.load_numpy_analysis_result(
                'backgrounds', self.analysisName)
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError):
            refactors = np.array([self.dataSet.load_numpy_analysis_result(
                    'background_refactors', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])])

            previousBackgrounds = np.array(
                [self.dataSet.load_numpy_analysis_result(
                    'previous_backgrounds', self.analysisName, resultIndex=i)
                    for i in range(self.parameters['fov_per_iteration'])])

            previousFactors = np.array([self.dataSet.load_numpy_analysis_result(
                'previous_scale_factors', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])])

            backgrounds = np.nanmedian(np.add(
                previousBackgrounds, np.multiply(refactors, previousFactors)),
                axis=0)

            self.dataSet.save_numpy_analysis_result(
                backgrounds, 'backgrounds', self.analysisName)

            return backgrounds

    def get_scale_factor_history(self) -> np.ndarray:
        """Get the scale factors cached for each iteration of the optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            scale factor corresponding to the i'th bit in the j'th
            iteration.
        """
        if 'previous_iteration' not in self.parameters:
            return np.array([self.get_scale_factors()])
        else:
            previousHistory = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration']
            ).get_scale_factor_history()
            return np.append(
                previousHistory, [self.get_scale_factors()], axis=0)

    def get_barcode_count_history(self) -> np.ndarray:
        """Get the set of barcode counts for each iteration of the
        optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            barcode count corresponding to the i'th barcode in the j'th
            iteration.
        """
        # finalize() merges the per-fragment counts into one array so that
        # this -- the only consumer, via PlotPerformance's optimization plot --
        # does not have to re-read fragment_count() files from every earlier
        # iteration, and so those files can be cleaned up afterwards.
        try:
            countsMean = self.dataSet.load_numpy_analysis_result(
                'barcode_counts_merged', self.analysisName)
        except (FileNotFoundError, OSError, ValueError):
            countsMean = np.mean([self.dataSet.load_numpy_analysis_result(
                'barcode_counts', self.analysisName, resultIndex=i)
                for i in range(self.parameters['fov_per_iteration'])], axis=0)

        if 'previous_iteration' not in self.parameters:
            return np.array([countsMean])
        else:
            previousHistory = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration']
            ).get_barcode_count_history()
            return np.append(previousHistory, [countsMean], axis=0)


class OptimizeIterationFOV(OptimizeIteration):

    """
    An analysis task for performing a single iteration of scale factor
    optimization.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'area_threshold' not in self.parameters:
            self.parameters['area_threshold'] = 4
        if 'optimize_background' not in self.parameters:
            self.parameters['optimize_background'] = False
        if 'optimize_chromatic_correction' not in self.parameters:
            self.parameters['optimize_chromatic_correction'] = False
        if 'crop_width' not in self.parameters:
            self.parameters['crop_width'] = 50
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 0.5176 # this is the default of decoder
            # maybe should make it bigger?
        if 'z_index' not in self.parameters:
            zpos = self.dataSet.get_data_organization().get_z_positions()
            self.parameters['z_index'] = int(len(zpos)/2)

        # for now just do all FOVs
        self.parameters['fov_index'] = self.dataSet.get_fovs().tolist() # for serializing json converts numpy type to python type...
        self.parameters['fov_per_iteration'] = len(self.parameters['fov_index'])

    def _run_analysis(self, fragmentIndex):
        preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        codebook = self.get_codebook()

        # this is where the FOV and zindex are decided
        fovIndex = self.parameters['fov_index'][fragmentIndex]
        zIndex = self.parameters['z_index']

        scaleFactors = self._get_previous_scale_factors(fragmentIndex)
        backgrounds = self._get_previous_backgrounds(fragmentIndex)
        chromaticTransformations = \
            self._get_previous_chromatic_transformations()

        self.dataSet.save_numpy_analysis_result(
            scaleFactors, 'previous_scale_factors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            backgrounds, 'previous_backgrounds', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_pickle_analysis_result(
            chromaticTransformations, 'previous_chromatic_corrections',
            self.analysisName, resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            np.array([fovIndex, zIndex]), 'select_frame', self.analysisName,
            resultIndex=fragmentIndex)

        chromaticCorrector = aberration.RigidChromaticCorrector(
            chromaticTransformations, self.get_reference_color())
        warpedImages = preprocessTask.get_processed_image_set(
            fovIndex, zIndex=zIndex, chromaticCorrector=chromaticCorrector)

        decoder = decoding.PixelBasedDecoder(codebook)
        areaThreshold = self.parameters['area_threshold']
        decoder.refactorAreaThreshold = areaThreshold
        # is it smart to put distance threshold in optimize?
        di, pm, npt, d = decoder.decode_pixels(
            warpedImages,
            scaleFactors,
            backgrounds,
            lowPassSigma=0,
            overlap=self.parameters['tile_overlap'],
            distanceThreshold=self.parameters['distance_threshold'],
            distanceMetric=self.parameters['distance_metric'])

        refactors, backgrounds, barcodesSeen = \
            decoder.extract_refactors(
                di, pm, npt, extractBackgrounds=self.parameters[
                    'optimize_background'])

        # TODO this saves the barcodes under fragment instead of fov
        # the barcodedb should be made more general
        cropWidth = self.parameters['crop_width']
        self.get_barcode_database().write_barcodes(
            decoder.extract_barcodes_with_index(
                di, pm, npt, d, fovIndex, cropWidth,
                zIndex, minimumArea=areaThreshold),
            fov=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            refactors, 'scale_refactors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            backgrounds, 'background_refactors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            barcodesSeen, 'barcode_counts', self.analysisName,
            resultIndex=fragmentIndex)

    def _get_previous_scale_factors(self, fragmentIndex) -> np.ndarray:
        if 'previous_iteration' not in self.parameters:
            scaleFactors = self._calculate_initial_scale_factors()
        else:
            previousIteration = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration'])
            scaleFactors = previousIteration.get_scale_factors(fragmentIndex)

        return scaleFactors

    def _get_previous_backgrounds(self, fragmentIndex) -> np.ndarray:
        if 'previous_iteration' not in self.parameters:
            backgrounds = np.zeros(self.get_codebook().get_bit_count())
        else:
            previousIteration = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration'])
            backgrounds = previousIteration.get_backgrounds(fragmentIndex)
        return backgrounds

    def get_scale_factors(self, fragmentIndex) -> np.ndarray:
        """Get the final, optimized scale factors.

        Returns:
            a one-dimensional numpy array where the i'th entry is the
            scale factor corresponding to the i'th bit.
        """
        if not self.is_complete():
            raise Exception('Analysis is still running. Unable to get scale '
                            + 'factors.')
        try:
            return self.dataSet.load_numpy_analysis_result(
                'scale_factors', self.analysisName, resultIndex=fragmentIndex)
        
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError):
            refactors = self.dataSet.load_numpy_analysis_result(
                    'scale_refactors', self.analysisName, resultIndex=fragmentIndex)

            # Don't rescale bits that were never seen
            refactors[refactors == 0] = 1

            previousFactors = self.dataSet.load_numpy_analysis_result(
                'previous_scale_factors', self.analysisName, resultIndex=fragmentIndex)

            scaleFactors = refactors * previousFactors

            # in case there are nans?
            scaleFactors[scaleFactors == np.nan] = previousFactors[scaleFactors == np.nan]

            self.dataSet.save_numpy_analysis_result(
                scaleFactors, 'scale_factors', self.analysisName,
                resultIndex=fragmentIndex)

            return scaleFactors

    def get_backgrounds(self, fragmentIndex) -> np.ndarray:
        if not self.is_complete():
            raise Exception('Analysis is still running. Unable to get ' +
                            'backgrounds.')

        try:
            return self.dataSet.load_numpy_analysis_result(
                'backgrounds', self.analysisName, resultIndex=fragmentIndex)
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError):
            refactors = self.dataSet.load_numpy_analysis_result(
                    'background_refactors', self.analysisName, resultIndex=fragmentIndex)

            previousBackgrounds = self.dataSet.load_numpy_analysis_result(
                    'previous_backgrounds', self.analysisName, resultIndex=fragmentIndex)

            previousFactors = self.dataSet.load_numpy_analysis_result(
                'previous_scale_factors', self.analysisName, resultIndex=fragmentIndex)

            backgrounds = np.add(previousBackgrounds, np.multiply(refactors, previousFactors))

            # in case there are nans?
            backgrounds[backgrounds == np.nan] = previousFactors[backgrounds == np.nan]

            self.dataSet.save_numpy_analysis_result(
                backgrounds, 'backgrounds', self.analysisName,
                resultIndex=fragmentIndex)

            return backgrounds

    def get_scale_factor_history(self, fragmentIndex) -> np.ndarray:
        """Get the scale factors cached for each iteration of the optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            scale factor corresponding to the i'th bit in the j'th
            iteration.
        """
        if 'previous_iteration' not in self.parameters:
            return np.array([self.get_scale_factors(fragmentIndex)])
        else:
            previousHistory = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration']
            ).get_scale_factor_history(fragmentIndex)
            return np.append(
                previousHistory, [self.get_scale_factors(fragmentIndex)], axis=0)

    def get_barcode_count_history(self, fragmentIndex) -> np.ndarray:
        """Get the set of barcode counts for each iteration of the
        optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            barcode count corresponding to the i'th barcode in the j'th
            iteration.
        """
        countsMean = self.dataSet.load_numpy_analysis_result(
            'barcode_counts', self.analysisName, resultIndex=fragmentIndex)

        if 'previous_iteration' not in self.parameters:
            return np.array([countsMean])
        else:
            previousHistory = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration']
            ).get_barcode_count_history(fragmentIndex)
            return np.append(previousHistory, [countsMean], axis=0)
