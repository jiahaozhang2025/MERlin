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

    outputGroup = 'Optimize'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        # Which names the caller actually passed, as opposed to the
        # defaults filled in below. The rename migration needs this so an
        # explicit new-style setting always beats a stale old-style one.
        explicitParameters = set(parameters or {})

        if 'distance_metric' not in self.parameters:
            self.parameters['distance_metric'] = 'dot_product'
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
        if 'tiling_overlap' not in self.parameters:
            self.parameters['tiling_overlap'] = 20
        # threads for the nearest-neighbour decode (sklearn n_jobs); must be
        # matched by the cpus requested for this task
        if 'tiling_num_threads' not in self.parameters:
            self.parameters['tiling_num_threads'] = 1
        # These were renamed. task.json records whatever names were current
        # when it was written, so refusing outright would make every existing
        # analysis directory unloadable. Migrate instead, and say so.
        for oldName, newName in (('tile_overlap', 'tiling_overlap'),
                 ('num_threads', 'tiling_num_threads')):
            if oldName in self.parameters:
                value = self.parameters.pop(oldName)
                if newName not in explicitParameters:
                    self.parameters[newName] = value
                print('%s was renamed to %s; using %s=%s'
                      % (oldName, newName, newName, self.parameters[newName]))
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
            self.parameters['chromatic_from_fragments'] = True
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
            self.parameters['adaptive_crop'] = True
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
        
        # specify fovs and z_indices separately
        elif ('fovs' in self.parameters) and ('z_indices' in self.parameters):
        
            self.parameters['fov_index'] = []
            fovIndex = np.random.choice(list(self.parameters['fovs']), 
                size = self.parameters['fov_per_iteration'])
                    
            zIndex = np.random.choice(list(self.parameters['z_indices']),
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

        # Return scale factors as ratios with mean 1 rather than absolute intensities.
        # Without this the convention depends on the preprocess task: with
        # save_pixel_histogram=True the initial factors are the pixel histogram's 90th
        # percentile (absolute, e.g. 177 / 50 on the two M2 arms), with it False they
        # start at np.ones and stay near 1 -- so the same pipeline produced M1 at ~1 and
        # M2 at ~177, which is a trap when comparing arms or feeding factors between
        # datasets. Normalizing is decode-safe: decoding.py:394 computes
        # (image - background)/scaleFactors and unit-normalizes the pixel trace at :397,
        # so dividing the factors by a constant is a pure global scale that cancels.
        # NOTE backgrounds are deliberately NOT rescaled -- they are subtracted before
        # the division, so scaling them too would change the result rather than scale it.
        if 'normalize_scale_factors' not in self.parameters:
            self.parameters['normalize_scale_factors'] = True

        # Clamp every scale factor to at least this fraction of the largest one.
        # 0 (the default) disables it, so existing analyses are unaffected.
        #
        # Why this exists. The refactor loop is a feedback system: a bit whose factor is
        # estimated too low has its normalized intensity inflated, which pulls more
        # background pixels into codewords containing that bit, and those pixels are dim,
        # which drags the next iteration's estimate lower still. On 20260609, where
        # readouts RS0584 and RS0707 are genuinely dim (raw signal 114 and 85 against
        # 241-526 for the other 560 nm bits), the loop diverges instead of converging:
        # max/min runs 3.48x -> 9.28x over ten iterations while RS0707 halves from 0.3484
        # to 0.1731. The codewords carrying those bits become noise sinks -- Opalin
        # reaches 24.2% of the library against a reference expectation of a few percent,
        # and the nine worst genes sum to ~50%.
        #
        # The blank-based adaptive filter cannot see this: on matched FOVs our Filter15
        # and a floored reference decode both realize ~1.47% blanks while their Opalin
        # shares differ eightfold, because the spurious calls have the same area,
        # intensity and distance statistics as real ones.
        #
        # 0.25 reproduces the reference implementation for 20260609 (Miao Xu's
        # OptimizeIterationFloor / scale_factor_floor_ratio), which pins the spread at
        # exactly 4.00x on every iteration and takes Opalin from 32.2% to 3.1%.
        #
        # Applied on read, after normalization and like it idempotent, so it reaches both
        # Decode and the next iteration's _get_previous_scale_factors without any
        # recompute. It deliberately does not re-impose the mean-1 convention: decoding.py
        # computes (image - background)/scaleFactors and then unit-normalizes the pixel
        # trace, so a global rescale cancels and only the ratios between bits matter.
        if 'scale_factor_floor_ratio' not in self.parameters:
            self.parameters['scale_factor_floor_ratio'] = 0

        # 'ones' (stock) or 'image_mean' to seed the first iteration from the
        # per-bit plane mean. See _calculate_initial_scale_factors.
        # Where the FINAL scale factors come from.
        #   'barcodes'   -- stock: the refactor loop's own estimate.
        #   'image_mean' -- the per-bit plane mean, used directly, with the
        #                   barcode refactoring of scale factors bypassed.
        # Measured reason for the second option: a refactor round multiplies a
        # dim bit's relative weight down by a roughly constant factor no matter
        # where it starts -- RS0707 went 0.2691 (image) -> 0.2076 after one
        # round -> 0.1692 after two, heading for the loop's 0.1077. Seeding the
        # loop does not avoid that, it only moves where the drag begins, which
        # is why 'image_mean' here bypasses rather than seeds. Chromatic
        # correction and backgrounds still come from the loop as normal.
        if 'scale_factor_source' not in self.parameters:
            self.parameters['scale_factor_source'] = 'barcodes'
        if 'initial_scale_factors' not in self.parameters:
            self.parameters['initial_scale_factors'] = 'ones'
        if 'initial_scale_factor_planes' not in self.parameters:
            self.parameters['initial_scale_factor_planes'] = 4

        # ---- per-bit bias freeze ----
        # A path to an external abundance reference switches this on; without
        # one it is inert and the loop behaves exactly as before.
        #
        # WHAT IT DOES. After every round, each bit gets a bias -- the
        # coefficient on that bit in a least-squares fit of
        # log10(observed gene share / reference gene share) against the
        # codebook. A round that makes a bit's |bias| worse by more than
        # bias_freeze_tolerance has its update to THAT BIT rolled back, while
        # every other bit updates normally. The veto is re-decided from
        # scratch each round, so a bit rolled back once can move again later.
        #
        # WHY, and it is not a convergence fix. The refactor loop has no fixed
        # point for a dim bit: each round multiplies its weight down by a
        # roughly constant factor, so ten rounds is not "more converged" than
        # two, it is further away. Measured on 20260609, total squared bit bias
        # by round: 1.518, 0.405, 0.491, 0.549, 0.719, ... -- best at round 2
        # and monotonically worse after. A per-bit veto keeps the bits that are
        # still improving moving while pinning the ones that have turned.
        #
        # THE ONE-ROUND LAG, which is unavoidable. Round k decodes with the
        # factors round k-1 produced, so round k's counts measure the bias OF
        # s_{k-1}; nothing can measure s_k's bias without decoding with it. So
        # the comparison available at round k is |bias(s_{k-1})| against
        # |bias(s_{k-2})|, and a rollback restores s_{k-2} -- the last value
        # actually measured to be better -- rather than s_{k-1}, which is the
        # value just shown to be worse. Expect a vetoed bit to hover between
        # two neighbouring values rather than settle on one; that hover
        # brackets the zero crossing, which is the point.
        if 'bias_freeze_reference' not in self.parameters:
            self.parameters['bias_freeze_reference'] = None
        # Fractional increase in |bias| that counts as "worse". 0.10 = a bit
        # must get more than 10% worse before its update is rolled back, which
        # keeps sampling noise from vetoing everything.
        if 'bias_freeze_tolerance' not in self.parameters:
            self.parameters['bias_freeze_tolerance'] = 0.10
        # Below this many usable genes the fit is not trustworthy and no bit is
        # vetoed.
        if 'bias_freeze_min_genes' not in self.parameters:
            self.parameters['bias_freeze_min_genes'] = 50
        # Optional absolute guard, OFF by default (0) so the rule is purely the
        # relative one. A percentage test treats a bit sitting at |bias| 0.01
        # exactly like one at 0.50, and on a 30-fov subsample a 10% move on a
        # near-zero bias is mostly sampling noise: dry-run on the shipped
        # 20260609 trajectory vetoes 37% of all bit-rounds at tolerance 0.10,
        # and most of those are bits whose bias is already negligible. Setting
        # this to e.g. 0.05 restricts vetoes to bits whose |bias| actually
        # exceeds it, which targets the rule at the bits that matter without
        # changing its logic.
        if 'bias_freeze_min_abs' not in self.parameters:
            self.parameters['bias_freeze_min_abs'] = 0.0

    def _normalize_scale_factors(self, scaleFactors: np.ndarray) -> np.ndarray:
        """Rescale to mean 1 if requested. Idempotent, so it is safe to apply on every
        read including to values cached from an earlier (un-normalized) run."""
        if not self.parameters.get('normalize_scale_factors', True):
            return scaleFactors
        sf = np.asarray(scaleFactors, dtype=float)
        m = np.nanmean(sf)
        if not np.isfinite(m) or m == 0:
            return self._floor_scale_factors(scaleFactors)
        return self._floor_scale_factors(sf / m)

    def _floor_scale_factors(self, scaleFactors: np.ndarray) -> np.ndarray:
        """Clamp each factor to >= scale_factor_floor_ratio * max, or pass through
        unchanged when the ratio is 0. Idempotent: flooring leaves the max untouched,
        so re-applying it is a no-op."""
        ratio = self.parameters.get('scale_factor_floor_ratio', 0)
        if not ratio:
            return scaleFactors
        sf = np.asarray(scaleFactors, dtype=float)
        top = np.nanmax(sf)
        if not np.isfinite(top) or top <= 0:
            return scaleFactors
        return np.maximum(sf, ratio * top)

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
                                            tilingOverlap = self.parameters['tiling_overlap'],
                                            tilingNumThreads = self.parameters['tiling_num_threads'],
                                            neighborNumJobs = 1 if self.parameters['tiling_num_threads'] > 1 else -1,
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

        if (self.parameters['optimize_chromatic_correction']
                and self.parameters['chromatic_from_fragments']):
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
                    outputTif.write(im.astype(np.float32),
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

        # Seed from the images instead of from ones. This is the whole fix for
        # the dim-bit runaway, in the place it belongs: the refactor loop was
        # never wrong, it was started at np.ones and given ten iterations to
        # walk away, which on 20260609 took RS0707 from 0.348 to 0.173 and the
        # spread from 3.48x to 9.28x. Seeded from the per-bit plane mean it
        # starts at 3.1x, where every image-based estimate lands.
        #
        # The MEAN specifically, rather than anything cleverer: measured against
        # a thresholded estimator on the same planes it differs by a median of
        # 0.0101 in a bit's share of the maximum, against 0.0950 between two
        # reasonable threshold families and 0.0231 between 3 and 30 planes of
        # the same estimator. Every elaboration sits below the sampling noise of
        # the thing it elaborates, so the simplest statistic wins.
        #
        # Known weakness: the plane mean is background + density * amplitude, so
        # a bit carrying unusually abundant genes reads high regardless of its
        # per-spot brightness. Tolerable as a SEED for a loop that then refines
        # against the barcodes; less tolerable as a final answer.
        if str(self.parameters.get('initial_scale_factors', 'ones')).lower() \
                == 'image_mean':
            return self._image_mean_scale_factors(preprocessTask, bitCount)

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

    def _image_mean_scale_factors(self, preprocessTask=None,
                                  bitCount=None) -> np.ndarray:
        """Per-bit mean of a few sampled preprocessed planes, rescaled to mean 1.

        Memoised on the instance: _get_previous_scale_factors runs once per
        fragment and one job runs many fragments in a single process, so the
        planes are paid for once per job rather than once per fragment.
        """
        if getattr(self, '_imageMeanSeed', None) is not None:
            return self._imageMeanSeed
        if preprocessTask is None:
            preprocessTask = self.dataSet.load_analysis_task(
                self.parameters['preprocess_task'])
        if bitCount is None:
            bitCount = self.get_codebook().get_bit_count()

        planes = int(self.parameters.get('initial_scale_factor_planes', 4))
        sampled = self.parameters['fov_index'][:max(1, planes)]
        corrector = None
        if 'previous_iteration' in self.parameters:
            corrector = self.dataSet.load_analysis_task(
                self.parameters['previous_iteration']
            ).get_previous_chromatic_corrector()

        means = []
        for fov, zIndex in sampled:
            imageSet = preprocessTask.get_processed_image_set(
                fov, zIndex, corrector)
            means.append([float(np.mean(imageSet[b])) for b in range(bitCount)])

        merged = np.nanmedian(np.array(means, dtype=float), axis=0)
        usable = np.isfinite(merged) & (merged > 0)
        if not usable.any():
            raise ValueError(
                'every bit of the image-mean seed is unusable in %s'
                % self.analysisName)
        merged = np.where(usable, merged, np.nanmedian(merged[usable]))
        self._imageMeanSeed = (merged / np.nanmean(merged)).astype(np.float32)
        return self._imageMeanSeed

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

        if str(self.parameters.get('scale_factor_source', 'barcodes')).lower() \
                == 'image_mean':
            # Deliberately ignores scale_factors.npy. The fragments still ran,
            # so the chromatic corrections and backgrounds this task provides
            # are the ordinary barcode-derived ones; only the scale factors are
            # replaced.
            return self._normalize_scale_factors(
                self._image_mean_scale_factors())

        try:
            return self._normalize_scale_factors(
                self.dataSet.load_numpy_analysis_result(
                    'scale_factors', self.analysisName))
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

            # Roll back the update for any bit this round made worse. Inert
            # unless bias_freeze_reference is set. Applied BEFORE caching, so
            # the cached scale_factors.npy is what downstream actually uses and
            # the veto is computed exactly once.
            scaleFactors = self._apply_bias_freeze(scaleFactors)

            # cache the raw values; normalization is applied on read and is idempotent
            self.dataSet.save_numpy_analysis_result(
                scaleFactors, 'scale_factors', self.analysisName)

            return self._normalize_scale_factors(scaleFactors)

    # ==================================================================
    # per-bit bias freeze
    # ==================================================================

    def _bias_reference(self):
        """gene name -> abundance, from .npz (names/total) or a 2-column csv."""
        path = self.parameters['bias_freeze_reference']
        if not path:
            return None
        if path.endswith('.npz'):
            z = np.load(path, allow_pickle=True)
            key = 'total' if 'total' in z else z.files[-1]
            return dict(zip([str(x) for x in z['names']], np.asarray(z[key], float)))
        table = pandas.read_csv(path)
        return dict(zip(table.iloc[:, 0].astype(str),
                        table.iloc[:, 1].astype(float)))

    def _iteration_counts(self, task) -> np.ndarray:
        """Per-codeword counts from one iteration's own sampled decode."""
        try:
            return self.dataSet.load_numpy_analysis_result(
                'barcode_counts_merged', task.analysisName)
        except (FileNotFoundError, OSError, ValueError):
            return np.mean([self.dataSet.load_numpy_analysis_result(
                'barcode_counts', task.analysisName, resultIndex=i)
                for i in range(task.parameters['fov_per_iteration'])], axis=0)

    def _bit_bias(self, counts: np.ndarray, reference: dict):
        """Per-bit bias: the coefficient on each bit in a least-squares fit of
        log10(observed share / reference share) against the codebook.

        A REGRESSION, not a per-bit median over "genes containing bit j".
        Every gene carries several on-bits, so a median is contaminated by the
        other ones; on 20260609 the regression isolates the two dim bits
        cleanly while a median smears them across their codeword neighbours.
        Returns None when too few genes are usable to trust the fit.
        """
        codebook = self.get_codebook()
        names = [str(n) for n in codebook.get_data()['name']]
        coding = set(codebook.get_coding_indexes())
        bitCount = codebook.get_bit_count()
        rows, obs, exp = [], [], []
        for i, name in enumerate(names):
            if i not in coding or i >= len(counts):
                continue
            value = reference.get(name.strip())
            if value is None or value <= 0 or counts[i] <= 0:
                continue
            rows.append(codebook.get_barcode(i))
            obs.append(counts[i])
            exp.append(value)
        if len(rows) < self.parameters['bias_freeze_min_genes']:
            return None
        obs, exp = np.asarray(obs, float), np.asarray(exp, float)
        y = np.log10(obs / obs.sum()) - np.log10(exp / exp.sum())
        design = np.hstack([np.asarray(rows, float),
                            np.ones((len(rows), 1))])
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        return beta[:bitCount]

    def _apply_bias_freeze(self, candidate: np.ndarray) -> np.ndarray:
        """Roll back this round's update for any bit whose bias got worse.

        See the parameter block for the one-round lag: the rollback target is
        s_{k-2}, the last value measured to be better, not s_{k-1}, which is
        the value just shown to be worse.
        """
        reference = self._bias_reference()
        if reference is None or 'previous_iteration' not in self.parameters:
            return candidate
        previous = self.dataSet.load_analysis_task(
            self.parameters['previous_iteration'])
        if 'previous_iteration' not in previous.parameters:
            # Round 2: only one measured bias exists, nothing to compare to.
            return candidate
        grandparent = self.dataSet.load_analysis_task(
            previous.parameters['previous_iteration'])
        try:
            biasNow = self._bit_bias(self._iteration_counts(self), reference)
            biasBefore = self._bit_bias(self._iteration_counts(previous),
                                        reference)
            fallback = np.asarray(grandparent.get_scale_factors(), float)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print('%s: bias freeze skipped (%s)' % (self.analysisName, exc),
                  flush=True)
            return candidate
        if biasNow is None or biasBefore is None:
            print('%s: bias freeze skipped, too few usable genes'
                  % self.analysisName, flush=True)
            return candidate

        tolerance = float(self.parameters['bias_freeze_tolerance'])
        floor = float(self.parameters['bias_freeze_min_abs'])
        worse = np.abs(biasNow) > np.abs(biasBefore) * (1.0 + tolerance)
        if floor > 0:
            worse &= np.abs(biasNow) > floor
        out = np.asarray(candidate, float).copy()
        # fallback is normalised on read and candidate is not, so compare like
        # with like: rescale the fallback onto the candidate's mean.
        scale = np.nanmean(out) / max(np.nanmean(fallback), 1e-12)
        out[worse] = fallback[worse] * scale
        rolled = [(int(j), float(biasBefore[j]), float(biasNow[j]),
                   float(candidate[j]), float(out[j]))
                  for j in np.flatnonzero(worse)]
        print('%s: bias freeze -- %d of %d bits rolled back (tolerance %.0f%%)'
              % (self.analysisName, len(rolled), len(out), 100 * tolerance),
              flush=True)
        for j, b0, b1, was, now in rolled:
            print('    bit %2d  |bias| %.3f -> %.3f   scale factor %.4f -> %.4f'
                  % (j, abs(b0), abs(b1), was, now), flush=True)
        self.dataSet.save_dataframe_to_csv(
            pandas.DataFrame(dict(
                bit=np.arange(len(out)), bias_previous=biasBefore,
                bias_current=biasNow, rolled_back=worse,
                candidate=np.asarray(candidate, float), applied=out)),
            'bias_freeze', self.analysisName, index=False)
        return out

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
            tilingOverlap=self.parameters['tiling_overlap'],
            distanceThreshold=self.parameters['distance_threshold'],
            distanceMetric=self.parameters['distance_metric'])

        refactors, backgrounds, barcodesSeen = \
            decoder.extract_refactors(
                di, pm, npt, extractBackgrounds=self.parameters[
                    'optimize_background'])

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
            extracted = extracted[extracted['x'].between(c0, c1)
                                  & extracted['y'].between(r0, r1)]
        self.get_barcode_database().write_barcodes(
            extracted, fov=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            refactors, 'scale_refactors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            backgrounds, 'background_refactors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            barcodesSeen, 'barcode_counts', self.analysisName,
            resultIndex=fragmentIndex)

        if (self.parameters['optimize_chromatic_correction']
                and self.parameters['chromatic_from_fragments']):
            if self.parameters['chromatic_on_preprocessed']:
                chromaticImages = warpedImages
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
                self._measure_chromatic_samples(chromaticImages, extracted),
                'chromatic_samples', self.analysisName,
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
            return self._normalize_scale_factors(
                self.dataSet.load_numpy_analysis_result(
                    'scale_factors', self.analysisName,
                    resultIndex=fragmentIndex))

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

            # cache the raw values; normalization is applied on read and is idempotent
            self.dataSet.save_numpy_analysis_result(
                scaleFactors, 'scale_factors', self.analysisName,
                resultIndex=fragmentIndex)

            return self._normalize_scale_factors(scaleFactors)

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


class ImageScaleFactors(analysistask.ParallelAnalysisTask):
    """Per-bit scale factors estimated from the preprocessed images alone.

    Why this exists. OptimizeIteration's estimand is "make the mean ON-BIT
    INTENSITY equal across bits", measured on the pixels the current decode
    selected. For a genuinely dim bit the only way to equalise a
    signal-plus-background mean is to divide by a very small number, and that
    inflates the bit's share of the L2-normalised trace until its own background
    passes as signal. On 20260609 the loop does not diverge -- it converges, and
    it converges to sf(RS0707) = 0.173 against ~1.0 typical. With a floor holding
    sf at 0.414 the estimator still demands a further 0.52x cut on nine
    consecutive iterations: a converged, reproducible, wrong answer. A floor caps
    the symptom; it does not fix the estimand.

    This task estimates the SIGNAL amplitude of each bit with the decode taken
    out of the loop entirely. Per bit and per sampled (fov, z): take the
    preprocessed image, call background the background_percentile'th percentile,
    build a robust spread from the MAD, threshold at background +
    signal_sigma * spread, and report mean(pixels above threshold) - background.
    Nothing here depends on any barcode assignment, so there is no fixed point
    to be wrong about -- it is a single pass, and a dim bit gets a small scale
    factor only to the extent its SPOTS are genuinely dim rather than its
    background being amplified.

    Drop-in replacement for Decode's optimize_task: get_backgrounds and
    get_previous_chromatic_corrector delegate to reference_optimize_task, so the
    images are corrected exactly as that task's own fragments were and the two
    runs stay comparable.
    """

    outputGroup = 'Optimize'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'background_percentile' not in self.parameters:
            self.parameters['background_percentile'] = 50
        # Threshold at background + signal_sigma * MAD-based spread. 3 keeps
        # only pixels that are implausible as background for THAT bit, which is
        # the point: the cut is per-bit and in absolute image units, so it does
        # not inherit the scale it is trying to measure.
        if 'signal_sigma' not in self.parameters:
            self.parameters['signal_sigma'] = 3.0
        # A bit with almost nothing above threshold on a plane is uninformative
        # rather than dim; report NaN and let the median over fragments ignore it.
        if 'min_pixels_above' not in self.parameters:
            self.parameters['min_pixels_above'] = 200
        # How the contributing pixels are chosen.
        #   'sigma'    -- everything above background + signal_sigma * spread.
        #                 Simple, but it admits a DIFFERENT NUMBER of pixels per
        #                 bit depending on that bit's SNR, and a selection that
        #                 varies with the quantity being measured is exactly the
        #                 confound this task exists to avoid. For a low-SNR bit
        #                 the 3-sigma tail is mostly noise, so the mean above it
        #                 is dragged toward background -- the same direction of
        #                 bias as the optimize loop, just far weaker.
        #   'quantile' -- the top signal_quantile fraction of each bit's own
        #                 pixels. Every bit contributes the SAME pixel count, so
        #                 the selection cannot vary with SNR. Closer in spirit to
        #                 MERlin's save_pixel_histogram initial factors, which
        #                 take each bit's 90th percentile.
        if 'background_pixels' not in self.parameters:
            # Which pixels the background percentile is taken over.
            #   'all'     -- every pixel. Correct when the plane has no atom of
            #                probability mass at a single value.
            #   'nonzero' -- only pixels > 0. Required when the preprocessing
            #                rectifies AFTER the lowpass (clip_stage 'lowpass'),
            #                because then 56.6 per cent of pixels are EXACTLY
            #                zero and the 50th percentile lands on that atom and
            #                returns 0.0 for every bit. Measured: ScaleImgLC's
            #                backgrounds came back 0.0000 on all 21 bits, so that
            #                arm decoded with no background subtraction at all.
            self.parameters['background_pixels'] = 'all'
        if 'selection' not in self.parameters:
            self.parameters['selection'] = 'sigma'
        if 'signal_quantile' not in self.parameters:
            self.parameters['signal_quantile'] = 0.999
        # Supply this task's OWN per-bit backgrounds to Decode instead of the
        # reference task's. This matters and is not optional bookkeeping: the
        # estimator reports a background-SUBTRACTED amplitude S = mean(above) - bg,
        # while decoding.py:397 computes (I - bg)/sf. If Decode is handed
        # backgrounds of zero -- which is what a reference OptimizeIteration run
        # with optimize_background false returns -- then every bit's normalised
        # trace carries an additive floor bg/S. Measured on 20260609 that floor
        # runs 0.0991 to 0.2397 across the 21 bits, a 2.42x asymmetry, and it acts
        # as a preferred direction in codeword space: cosine-matching the floor
        # vector against the 246 codewords puts Il11 and Opalin first and second,
        # which are also the two most-called genes, and puts Blank-6 third, which
        # is called at 4.44x the mean blank share. A blank codeword cannot come
        # from mRNA, so that is the unambiguous signature of a manufactured call.
        # Default true because zeros are the wrong answer, not the safe one. A task
        # whose fragments predate this and so have no saved backgrounds raises a
        # message naming the re-run rather than silently reintroducing the floor.
        if 'supply_backgrounds' not in self.parameters:
            self.parameters['supply_backgrounds'] = True

        if 'excess_bins' not in self.parameters:
            self.parameters['excess_bins'] = 512
        if str(self.parameters['selection']).lower() not in (
                'sigma', 'quantile', 'excess', 'mean'):
            raise ValueError(
                "selection must be 'sigma', 'quantile', 'excess' or 'mean'")
        if 'fov_per_iteration' not in self.parameters:
            self.parameters['fov_per_iteration'] = 30
        if 'normalize_scale_factors' not in self.parameters:
            self.parameters['normalize_scale_factors'] = True
        if 'reference_optimize_task' not in self.parameters:
            self.parameters['reference_optimize_task'] = None

        if 'random_seed' in self.parameters:
            np.random.seed(self.parameters['random_seed'])
        if 'fov_index' in self.parameters:
            self.parameters['fov_per_iteration'] = \
                len(self.parameters['fov_index'])
        else:
            fovIndex = np.random.choice(
                list(self.dataSet.get_fovs()),
                size=self.parameters['fov_per_iteration'])
            zIndex = np.random.choice(
                list(range(len(self.dataSet.get_z_positions()))),
                size=self.parameters['fov_per_iteration'])
            self.parameters['fov_index'] = [
                [int(f), int(z)] for f, z in zip(fovIndex, zIndex)]

    # (selection, value) pairs spanning the plausible range of each knob.
    SENSITIVITY_GRID = [('sigma', 2.0), ('sigma', 3.0), ('sigma', 4.0),
                        ('sigma', 6.0), ('quantile', 0.99), ('quantile', 0.998),
                        ('quantile', 0.999), ('quantile', 0.9995)]

    def get_scale_factor_sensitivity(self):
        """How much the normalised scale-factor vector moves across the grid.

        Returns a dict with the per-setting normalised vectors and, for each
        bit, the spread of its share of the maximum. A bit whose share is stable
        across every setting is not being decided by the parameter; one that
        swings is, and should be reported as such rather than quietly shipped.
        """
        grids = np.array([
            self.dataSet.load_numpy_analysis_result(
                'image_scale_factor_grid', self.analysisName, resultIndex=i)
            for i in range(self.fragment_count())], dtype=float)
        merged = np.nanmedian(grids, axis=0)
        normalised = merged / np.nanmax(merged, axis=1, keepdims=True)
        spread = np.nanmax(normalised, axis=0) - np.nanmin(normalised, axis=0)
        return dict(settings=list(self.SENSITIVITY_GRID),
                    normalised=normalised, spread=spread,
                    worst_bit=int(np.nanargmax(spread)),
                    worst_spread=float(np.nanmax(spread)))

    def fragment_count(self):
        return len(self.parameters['fov_index'])

    def get_estimated_memory(self):
        return 16000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        deps = [self.parameters['preprocess_task']]
        if self.parameters['reference_optimize_task']:
            deps.append(self.parameters['reference_optimize_task'])
        return deps

    def get_codebook(self) -> Codebook:
        return self.dataSet.load_analysis_task(
            self.parameters['preprocess_task']).get_codebook()

    def _reference_task(self):
        name = self.parameters['reference_optimize_task']
        return self.dataSet.load_analysis_task(name) if name else None

    def _run_analysis(self, fragmentIndex):
        preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        fov, zIndex = self.parameters['fov_index'][fragmentIndex]

        reference = self._reference_task()
        corrector = (reference.get_previous_chromatic_corrector()
                     if reference is not None else None)

        imageSet = preprocessTask.get_processed_image_set(
            fov, zIndex, corrector)

        percentile = float(self.parameters['background_percentile'])
        sigma = float(self.parameters['signal_sigma'])
        minPixels = int(self.parameters['min_pixels_above'])
        selection = str(self.parameters['selection']).lower()
        quantile = float(self.parameters['signal_quantile'])
        backgroundPixels = str(self.parameters['background_pixels']).lower()
        if backgroundPixels not in ('all', 'nonzero'):
            raise ValueError("background_pixels must be 'all' or 'nonzero'")
        excessBins = int(self.parameters['excess_bins'])
        if selection not in ('sigma', 'quantile', 'excess', 'mean'):
            raise ValueError(
                "selection must be 'sigma', 'quantile', 'excess' or 'mean'")

        factors = np.full(imageSet.shape[0], np.nan)
        backgrounds = np.full(imageSet.shape[0], np.nan)
        for bit in range(imageSet.shape[0]):
            image = np.asarray(imageSet[bit], dtype=np.float32).ravel()
            if backgroundPixels == 'nonzero':
                positive = image[image > 0]
                if positive.size < minPixels:
                    continue
                background = np.percentile(positive, percentile)
            else:
                background = np.percentile(image, percentile)
            if selection == 'quantile':
                cut = np.quantile(image, quantile)
            else:
                # 1.4826 = 1 / Phi^-1(0.75), the factor that turns the median
                # absolute deviation into a consistent Gaussian sigma estimate.
                # Do NOT read signal_sigma as "sigmas of the noise": these
                # highpass planes are non-negative (measured frac_neg = 0.0000),
                # so the cut lands at the 87th-91st percentile and admits
                # 8.6-12.7 per cent of the plane. It is a top-decile brightness
                # statistic wearing a noise-sigma label. Measured sensitivity is
                # low either way -- sigma 2 to 6 moves the dim bit's share of the
                # maximum by 5.0 per cent, against 21.6 per cent for
                # signal_quantile between 0.998 and 0.999 -- which is why sigma
                # stays the default despite the quantile rule being better
                # motivated in principle.
                spread = 1.4826 * np.median(np.abs(image - np.median(image)))
                if not np.isfinite(spread) or spread <= 0:
                    continue
                cut = background + sigma * spread
            if selection == 'mean':
                # The whole-plane mean, with no threshold, no quantile and no
                # constants. Measured against the sigma estimator on the same
                # three planes it differs by a median of 0.0101 in a bit's share
                # of the maximum, against 0.0950 between the sigma and quantile
                # families and 0.0231 between 3 and 30 planes of the SAME
                # estimator -- i.e. inside the sampling noise of the method it
                # is being compared with. Nearly all of the gain over the
                # optimize loop (9.28x spread against 3.1x) comes from
                # estimating off the images at all, not from how.
                # The background is still taken the normal way: the mean gives
                # no background, and the background is separately worth 2.5x on
                # the worst gene.
                factors[bit] = float(image.mean())
                backgrounds[bit] = float(background)
                continue
            if selection == 'excess':
                value = self._excess_mass(image, excessBins, minPixels)
                if value is not None:
                    factors[bit], backgrounds[bit] = value
                continue
            above = image[image > cut]
            if above.size >= minPixels:
                factors[bit] = float(above.mean() - background)
                backgrounds[bit] = float(background)

        # Emit the answer under a GRID of the free parameters as well, so the
        # arbitrariness of the chosen values is measured on every run instead of
        # argued about. The images are already in memory, so this is nearly free
        # compared with loading and filtering them. Rows are the settings in
        # SENSITIVITY_GRID, columns the bits. get_scale_factor_sensitivity()
        # reduces it to the only thing that matters: how much the NORMALISED
        # vector moves, since decode sees ratios between bits and nothing else.
        grid = np.full((len(self.SENSITIVITY_GRID), imageSet.shape[0]), np.nan)
        for row, (mode, value) in enumerate(self.SENSITIVITY_GRID):
            for bit in range(imageSet.shape[0]):
                image = np.asarray(imageSet[bit], dtype=np.float32).ravel()
                if backgroundPixels == 'nonzero':
                    positive = image[image > 0]
                    if positive.size < minPixels:
                        continue
                    bg = np.percentile(positive, percentile)
                else:
                    bg = np.percentile(image, percentile)
                if mode == 'quantile':
                    cut = np.quantile(image, value)
                else:
                    spread = 1.4826 * np.median(np.abs(image - np.median(image)))
                    if not np.isfinite(spread) or spread <= 0:
                        continue
                    cut = bg + value * spread
                above = image[image > cut]
                if above.size >= minPixels:
                    grid[row, bit] = float(above.mean() - bg)
        self.dataSet.save_numpy_analysis_result(
            grid, 'image_scale_factor_grid', self.analysisName,
            resultIndex=fragmentIndex)

        self.dataSet.save_numpy_analysis_result(
            factors, 'image_scale_factors', self.analysisName,
            resultIndex=fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            backgrounds, 'image_backgrounds', self.analysisName,
            resultIndex=fragmentIndex)

    @staticmethod
    def _excess_mass(image, bins, minPixels):
        """Background and signal amplitude with no threshold to choose.

        The problem with every quantile rule is that a fixed rank samples a
        different part of each bit's signal distribution, because bits differ in
        how many spots they carry. Measured on this dataset, signal_quantile
        0.998 versus 0.999 moves the dim bit's share of the maximum by 21.6 per
        cent -- a hand-picked constant that changes the answer.

        This instead assumes only that the BACKGROUND is symmetric about its
        mode, which signal cannot make untrue: spots add mass to the right and
        never to the left. So mirror the left side of the histogram about the
        mode to get the background's own right side, and whatever the observed
        right side has in excess of that mirror is signal. The reported
        amplitude is the count-weighted mean of that excess above the mode.

        The only knob left is the bin count, and unlike a quantile that is
        checkable for stability rather than chosen for taste.

        Returns (amplitude, background) or None if the plane is unusable.
        """
        finite = image[np.isfinite(image)]
        if finite.size < minPixels:
            return None
        # A robust range: the extreme signal tail must not set the bin width, or
        # the background occupies one bin and the mirror has no resolution.
        low, high = np.percentile(finite, [0.1, 99.9])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return None
        counts, edges = np.histogram(finite, bins=bins, range=(low, high))
        centres = 0.5 * (edges[:-1] + edges[1:])
        peak = int(np.argmax(counts))
        background = float(centres[peak])
        # Mirror the left side onto the right, bin for bin.
        nRight = counts.size - peak - 1
        nUsable = min(peak, nRight)
        if nUsable < 2:
            return None
        right = counts[peak + 1:peak + 1 + nUsable].astype(float)
        mirrored = counts[peak - nUsable:peak][::-1].astype(float)
        excess = np.clip(right - mirrored, 0, None)
        if excess.sum() < minPixels:
            return None
        offsets = centres[peak + 1:peak + 1 + nUsable] - background
        amplitude = float((excess * offsets).sum() / excess.sum())
        if not np.isfinite(amplitude) or amplitude <= 0:
            return None
        return amplitude, background

    def _merge_fragments(self, name: str,
                         requirePositive: bool = True) -> np.ndarray:
        """Median across fragments, with unmeasurable bits filled from the rest.

        Median rather than mean because a sampled plane can miss the tissue
        entirely.

        requirePositive applies to scale factors, which are divisors and must be
        positive, and NOT to backgrounds. A background of zero or below is a
        perfectly good measurement: with highpass_clip false the preprocessed
        plane is roughly zero-mean, so its median background is approximately
        zero and frequently negative. Treating that as unmeasurable is what made
        the first clip-off run die.
        """
        values = np.array([
            self.dataSet.load_numpy_analysis_result(
                name, self.analysisName, resultIndex=i)
            for i in range(self.fragment_count())], dtype=float)
        merged = np.nanmedian(values, axis=0)
        usable = np.isfinite(merged)
        if requirePositive:
            usable &= merged > 0
        if not usable.any():
            raise ValueError(
                'no bit of %s is usable in %s; refusing to return a NaN vector, '
                'which would decode to an empty barcode table with only a '
                'RuntimeWarning rather than failing'
                % (name, self.analysisName))
        return np.where(usable, merged, np.nanmedian(merged[usable]))

    def get_scale_factors(self, fragmentIndex: int = None) -> np.ndarray:
        """Median of the per-fragment signal amplitudes, rescaled to mean 1.

        The mean-1 rescale is cosmetic: decoding.py computes
        (image - background)/scaleFactors and then unit-normalises the pixel
        trace, so a global factor cancels and only the ratios matter. That holds
        for any background vector, because the background is subtracted in the
        numerator -- see get_backgrounds, which must therefore NOT be rescaled.

        fragmentIndex is accepted and ignored: this estimate is global by
        construction. Decode only passes it when single_fov_optimization is set,
        which would otherwise raise TypeError on every fragment.
        """
        if not self.is_complete():
            raise Exception('Analysis is still running. Unable to get scale '
                            + 'factors.')
        merged = self._merge_fragments('image_scale_factors')
        if not self.parameters.get('normalize_scale_factors', True):
            return merged
        mean = np.nanmean(merged)
        return merged / mean if np.isfinite(mean) and mean != 0 else merged

    def get_backgrounds(self, fragmentIndex: int = None) -> np.ndarray:
        """This task's own per-bit backgrounds, in RAW image units.

        Deliberately NOT rescaled by the mean used to normalise the scale
        factors. decoding.py:397 computes (I - bg)/sf, so returning sf/m with bg
        raw gives m*(I - bg)/S, a uniform factor that cancels exactly in the L2
        normalisation of the pixel trace. Dividing bg by m as well would give
        (m*I - bg)/S, which is not a rescale of anything and would genuinely
        change the assignment. Same convention OptimizeIteration states for its
        own backgrounds.
        """
        if not self.parameters.get('supply_backgrounds', True):
            reference = self._reference_task()
            if reference is None:
                return np.zeros(len(self.get_codebook().get_bit_names()))
            return reference.get_backgrounds()
        if not self.is_complete():
            raise Exception('Analysis is still running. Unable to get '
                            + 'backgrounds.')
        try:
            return self._merge_fragments('image_backgrounds',
                                         requirePositive=False)
        except (FileNotFoundError, OSError) as error:
            raise FileNotFoundError(
                '%s has no image_backgrounds results, so it predates background '
                'estimation. Re-run it, or set supply_backgrounds false to fall '
                'back to %s -- which returns zeros and leaves an uncorrected '
                'background floor in the normalised trace.'
                % (self.analysisName,
                   self.parameters['reference_optimize_task'])) from error

    def get_previous_chromatic_corrector(self) -> aberration.ChromaticCorrector:
        reference = self._reference_task()
        if reference is None:
            raise ValueError(
                'reference_optimize_task is required to supply a chromatic '
                'corrector to downstream tasks')
        return reference.get_previous_chromatic_corrector()

    def _get_chromatic_transformations(self):
        """Delegate, so an OptimizeIteration can name this task as its
        previous_iteration.

        OptimizeIteration asks its predecessor for exactly three things --
        get_scale_factors, get_backgrounds and _get_chromatic_transformations --
        so implementing this makes a barcode-based refinement round chainable
        straight onto the image-based estimate. That is worth having: the
        closed loop's failure was never the refactor arithmetic, it was starting
        from np.ones and iterating ten times, which lets a dim bit walk down to
        0.173. Started from an estimate that is already close, a SINGLE
        refinement can correct whatever the image statistic gets wrong about the
        pixels the decoder actually uses, without the room to run away.
        """
        reference = self._reference_task()
        if reference is None:
            raise ValueError(
                'reference_optimize_task is required to supply chromatic '
                'transformations to a chained OptimizeIteration')
        return reference._get_chromatic_transformations()


# Worker side of OptimizeLoop's process pool. Spawned rather than forked: the
# parent has already started OpenMP and torch thread pools, which a forked
# child can deadlock on. Each worker reopens the dataset once and keeps it.
_loopWorkerDataSet = None


def _init_loop_worker(dataSetName, dataHome, analysisHome):
    global _loopWorkerDataSet
    from merlin.core import dataset
    _loopWorkerDataSet = dataset.MERFISHDataSet(
        dataSetName, dataHome=dataHome, analysisHome=analysisHome)


def _run_loop_fragment(taskName, fragmentIndex):
    _loopWorkerDataSet.load_analysis_task(taskName).run(fragmentIndex)
    return fragmentIndex


class OptimizeLoop(analysistask.AnalysisTask):

    """
    A chain of optimize iterations written once in the analysis json.

    Expands into iteration tasks named <analysis_name>1 .. <analysis_name>N --
    an OptimizeLoop called Optimize gives Optimize1 .. Optimize10 -- each an
    ordinary OptimizeIteration saved in the dataset and linked through
    previous_iteration, exactly like a chain written out by hand. Each
    iteration still draws its own (fov, z) sample from its random_seed.

    Running this task runs, in order, every iteration that is not complete yet,
    finalizing each one before the next starts. The iterations can also be run
    one by one by name (e.g. -t Optimize3 -i 0-29), and this task counts as
    complete once its last iteration is.

    Downstream tasks can give this task as their optimize_task: whatever it
    does not define itself (get_scale_factors, get_backgrounds,
    get_previous_chromatic_corrector, the histories, ...) is answered by the
    last iteration.

    Loop parameters:
      iterations         number of iterations (default 10)
      random_seeds       one seed per iteration; by default
                         random_seed_start, random_seed_start + 1, ...
      random_seed_start  default 1, so the default seeds are 1 .. N
      per_iteration      {parameter: [one value per iteration]}, for any
                         other parameter that should change between iterations
      iteration_task     class of each iteration in this module (default
                         'OptimizeIteration')
      previous_iteration optional: an existing optimize task for the first
                         iteration to continue from
      num_workers        processes that run an iteration's fragments when this
                         task runs them itself (default 1: one after another,
                         in this process). Workers are spawned, so a script
                         that runs this task directly, rather than through
                         python -m merlin, needs the usual
                         if __name__ == '__main__' guard.
    Every other parameter is given unchanged to every iteration.
    """

    outputGroup = 'Optimize'

    _LOOP_PARAMETERS = ('iterations', 'random_seeds', 'random_seed_start',
                        'per_iteration', 'iteration_task', 'num_workers',
                        'previous_iteration')
    _BOOKKEEPING = ('merlin_version', 'module', 'class')

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'random_seed' in self.parameters:
            raise ValueError(
                'OptimizeLoop takes random_seeds (one per iteration) or '
                'random_seed_start, not random_seed')
        if 'iterations' not in self.parameters:
            self.parameters['iterations'] = 10
        if 'random_seed_start' not in self.parameters:
            self.parameters['random_seed_start'] = 1
        if 'random_seeds' not in self.parameters:
            self.parameters['random_seeds'] = None
        if 'per_iteration' not in self.parameters:
            self.parameters['per_iteration'] = {}
        if 'iteration_task' not in self.parameters:
            self.parameters['iteration_task'] = 'OptimizeIteration'
        if 'num_workers' not in self.parameters:
            self.parameters['num_workers'] = 1

        count = int(self.parameters['iterations'])
        if count < 1:
            raise ValueError('iterations must be at least 1')
        seeds = self.parameters['random_seeds']
        if seeds is None:
            seeds = [int(self.parameters['random_seed_start']) + i
                     for i in range(count)]
        if len(seeds) != count:
            raise ValueError('random_seeds has %d entries for %d iterations'
                             % (len(seeds), count))
        perIteration = self.parameters['per_iteration']
        for name, values in perIteration.items():
            if name in self._LOOP_PARAMETERS + self._BOOKKEEPING \
                    or name == 'random_seed':
                raise ValueError('%s cannot be set per iteration' % name)
            if len(values) != count:
                raise ValueError('per_iteration %s has %d entries for %d '
                                 'iterations' % (name, len(values), count))

        iterationClass = globals().get(self.parameters['iteration_task'])
        if not (isinstance(iterationClass, type)
                and issubclass(iterationClass, OptimizeIteration)):
            raise ValueError('iteration_task %s is not an OptimizeIteration '
                             'in merlin.analysis.optimize'
                             % self.parameters['iteration_task'])
        self._iterationClass = iterationClass

        shared = {k: v for k, v in self.parameters.items()
                  if k not in self._LOOP_PARAMETERS + self._BOOKKEEPING}
        self._iterationNames = ['%s%d' % (self.analysisName, i + 1)
                                for i in range(count)]
        self._iterationParameters = []
        for i in range(count):
            iterationParameters = dict(shared)
            iterationParameters['random_seed'] = int(seeds[i])
            for name, values in perIteration.items():
                iterationParameters[name] = values[i]
            previous = self._iterationNames[i - 1] if i > 0 \
                else self.parameters.get('previous_iteration')
            if previous is not None:
                iterationParameters['previous_iteration'] = previous
            self._iterationParameters.append(iterationParameters)
        self._finalIteration = None

    def get_iteration_names(self) -> List[str]:
        return list(self._iterationNames)

    def _new_iteration(self, i):
        return self._iterationClass(self.dataSet, self._iterationParameters[i],
                                    self._iterationNames[i])

    def save(self, overwrite=False) -> None:
        # Iterations first, so a clash with an existing task of the same name
        # stops before the loop itself is recorded.
        for i in range(len(self._iterationNames)):
            self._new_iteration(i).save(overwrite)
        super().save(overwrite)

    def get_dependencies(self):
        # what the first iteration needs from outside the loop
        return self._new_iteration(0).get_dependencies()

    def get_estimated_memory(self):
        return 4000

    def get_estimated_time(self):
        return 60 * len(self._iterationNames)

    def final_iteration(self) -> OptimizeIteration:
        if self._finalIteration is None:
            self._finalIteration = self.dataSet.load_analysis_task(
                self._iterationNames[-1])
        return self._finalIteration

    def __getattr__(self, name):
        # Only reached for attributes this task does not have itself.
        if name.startswith('__') or '_iterationNames' not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.final_iteration(), name)

    def is_complete(self):
        if super().is_complete():
            return True
        try:
            final = self.final_iteration()
        except FileNotFoundError:
            return False
        if final.is_complete():
            # the iterations were run by name rather than through this task
            self.dataSet.record_analysis_complete(self)
            return True
        return False

    def finalize(self) -> None:
        # idempotent; covers iterations that were run by name
        for name in self._iterationNames:
            self.dataSet.load_analysis_task(name).finalize()

    def _run_analysis(self):
        workers = int(self.parameters['num_workers'])
        pool = None
        try:
            for i, name in enumerate(self._iterationNames):
                task = self.dataSet.load_analysis_task(name)
                if not task.is_complete():
                    todo = [f for f in range(task.fragment_count())
                            if not task.is_complete(f)]
                    print('%s: iteration %d of %d (%s), %d fragments on %d '
                          'worker(s)' % (self.analysisName, i + 1,
                                         len(self._iterationNames), name,
                                         len(todo), workers), flush=True)
                    if workers <= 1:
                        for f in todo:
                            task.run(f)
                    else:
                        if pool is None:
                            pool = self._start_pool(workers)
                        futures = [pool.submit(_run_loop_fragment, name, f)
                                   for f in todo]
                        for future in futures:
                            future.result()
                    if not task.is_complete():
                        missing = [f for f in range(task.fragment_count())
                                   if not task.is_complete(f)]
                        raise RuntimeError('%s is still incomplete after '
                                           'running; fragments %s'
                                           % (name, missing))
                task.finalize()
        finally:
            if pool is not None:
                pool.shutdown()

    def _start_pool(self, workers):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        return ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_loop_worker,
            initargs=(self.dataSet.dataSetName, self.dataSet.dataHome,
                      self.dataSet.analysisHome))
