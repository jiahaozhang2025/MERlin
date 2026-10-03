import os
import time
import numpy as np
import pandas
import zarr
from scipy import optimize, special

from merlin.core import analysistask
from merlin.analysis import decode
from merlin.util import barcodefilters
from merlin.util import imagesample


def _extract_finite_threshold_candidates(blank_fraction_hist: np.ndarray,
                                         max_points: int = 2048
                                         ) -> np.ndarray:
    """Return sorted finite threshold candidates from a blank-fraction histogram."""
    finite_values = blank_fraction_hist[np.isfinite(blank_fraction_hist)]
    finite_values = finite_values[finite_values >= 0]
    if finite_values.size == 0:
        return np.array([], dtype=float)
    finite_values = np.unique(np.sort(finite_values.astype(float)))
    if finite_values.size > max_points:
        idx = np.linspace(0, finite_values.size - 1, max_points, dtype=int)
        finite_values = finite_values[idx]
    return finite_values


def _cumulative_misid_curve(blank_hist: np.ndarray,
                            coding_hist: np.ndarray,
                            blank_fraction_hist: np.ndarray,
                            blank_barcode_count: int,
                            coding_barcode_count: int):
    """Sort bins by normalized blank fraction and return
    (ratios_sorted, cumulative_misid, finite_indices), or None if empty.

    cumulative_misid[k] is the misidentification rate achieved by keeping every
    bin whose blank fraction is <= ratios_sorted[k] (a whole-bin selection). The
    curve is monotonically non-decreasing in k."""
    valid = (
        np.isfinite(blank_fraction_hist)
        & (blank_fraction_hist >= 0)
        & (coding_hist > 0)
    )
    if not np.any(valid):
        return None
    ratios = blank_fraction_hist[valid].astype(float)
    blank_vals = blank_hist[valid].astype(float)
    coding_vals = coding_hist[valid].astype(float)
    order = np.argsort(ratios)
    ratios = ratios[order]
    blank_vals = blank_vals[order]
    coding_vals = coding_vals[order]
    cum_blank = np.cumsum(blank_vals)
    cum_coding = np.cumsum(coding_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        cumulative_misid = ((cum_blank / blank_barcode_count)
                            / (cum_coding / coding_barcode_count))
    finite = np.isfinite(cumulative_misid)
    if not np.any(finite):
        return None
    return ratios, cumulative_misid, np.where(finite)[0]


def cumulative_bins_bracketing(blank_hist: np.ndarray,
                               coding_hist: np.ndarray,
                               blank_fraction_hist: np.ndarray,
                               target_misidentification_rate: float,
                               blank_barcode_count: int,
                               coding_barcode_count: int) -> dict:
    """Report the two whole-bin blank-fraction thresholds that bracket the target
    misidentification rate: the largest achievable misid that is <= target
    (``below``) and the smallest achievable misid that is > target (``above``),
    along with the misid each achieves. A side is None when it does not exist."""
    report = {
        'target_misidentification_rate': float(target_misidentification_rate),
        'below_threshold': None, 'below_misid': None,
        'above_threshold': None, 'above_misid': None}
    curve = _cumulative_misid_curve(
        blank_hist, coding_hist, blank_fraction_hist,
        blank_barcode_count, coding_barcode_count)
    if curve is None:
        return report
    ratios, cumulative_misid, finite_idx = curve
    below = finite_idx[cumulative_misid[finite_idx]
                       <= target_misidentification_rate]
    above = finite_idx[cumulative_misid[finite_idx]
                       > target_misidentification_rate]
    if below.size > 0:
        i = int(below[-1])
        report['below_threshold'] = float(np.nextafter(ratios[i], np.inf))
        report['below_misid'] = float(cumulative_misid[i])
    if above.size > 0:
        i = int(above[0])
        report['above_threshold'] = float(np.nextafter(ratios[i], np.inf))
        report['above_misid'] = float(cumulative_misid[i])
    return report


def _threshold_from_cumulative_bins(blank_hist: np.ndarray,
                                    coding_hist: np.ndarray,
                                    blank_fraction_hist: np.ndarray,
                                    target_misidentification_rate: float,
                                    blank_barcode_count: int,
                                    coding_barcode_count: int,
                                    overshoot_toward_target: bool = False,
                                    overshoot_tolerance: float = 0.20) -> float:
    """
    Select threshold by sorting bins by normalized blank fraction, then
    cumulatively including whole bins. Both this method and the newton solver
    only ever keep whole bins (a barcode is kept iff its bin's blank fraction is
    below the returned threshold); the threshold is just the blank-fraction
    cutoff, not a sub-bin boundary.

    By default the largest whole-bin set with misid <= target is returned, which
    undershoots the target (the next whole bin would push misid over target).
    If ``overshoot_toward_target`` is True, and adding that next bin lands closer
    to the target than stopping short (|misid_over - target| < |target -
    misid_under|) AND stays within ``overshoot_tolerance`` (misid_over <=
    target*(1+overshoot_tolerance)), the next bin is included instead (a slight
    overshoot that is nearer the target).
    """
    curve = _cumulative_misid_curve(
        blank_hist, coding_hist, blank_fraction_hist,
        blank_barcode_count, coding_barcode_count)
    if curve is None:
        return np.nan
    ratios, cumulative_misid, finite_idx = curve
    below = finite_idx[cumulative_misid[finite_idx]
                       <= target_misidentification_rate]
    above = finite_idx[cumulative_misid[finite_idx]
                       > target_misidentification_rate]

    if below.size == 0:
        # nothing reaches at/under the target; fall back to the lowest-misid bin
        return float(np.nextafter(ratios[int(finite_idx[0])], np.inf))

    chosen_idx = int(below[-1])
    if overshoot_toward_target and above.size > 0:
        over_idx = int(above[0])
        under_mag = target_misidentification_rate - cumulative_misid[chosen_idx]
        over_mag = cumulative_misid[over_idx] - target_misidentification_rate
        within_tolerance = (
            cumulative_misid[over_idx]
            <= target_misidentification_rate * (1.0 + overshoot_tolerance))
        if (over_mag < under_mag) and within_tolerance:
            chosen_idx = over_idx
    return float(np.nextafter(ratios[chosen_idx], np.inf))


def _threshold_from_newton(error_fn, tolerance: float) -> float:
    """Original Newton/secant style threshold solver."""
    return float(optimize.newton(
        error_fn, 0.2, tol=tolerance, x1=0.3, disp=False))


def _get_intensity_transform_method(parameters: dict) -> str:
    """Return configured intensity transform method."""
    transformMethod = str(parameters.get('intensity_transform', 'log10')).lower()
    validMethods = {'log10', 'linear'}
    if transformMethod not in validMethods:
        raise ValueError(
            'intensity_transform must be one of '
            + str(sorted(validMethods)))
    return transformMethod


def _transform_intensity_values(values: np.ndarray, transformMethod: str) -> np.ndarray:
    if transformMethod == 'linear':
        return values
    return np.log10(values)


def _build_intensity_bins(maxIntensity: float,
                          intensityBinCount: int,
                          transformMethod: str) -> np.ndarray:
    if transformMethod == 'linear':
        upper = 2 * maxIntensity
    else:
        upper = 2 * np.log10(maxIntensity)

    if not np.isfinite(upper) or upper <= 0:
        upper = 1.0
    return np.linspace(0, upper, intensityBinCount + 1)


def _standardize_logistic_features(features: np.ndarray
                                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.nanmean(features, axis=0).astype(float)
    scales = np.nanstd(features, axis=0).astype(float)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    return (features - centers) / scales, centers, scales


def _fit_logistic_regression(features: np.ndarray,
                             labels: np.ndarray,
                             l2_regularization: float,
                             max_iterations: int
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    standardized, centers, scales = _standardize_logistic_features(features)
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    labels = labels.astype(float)
    l2 = float(max(l2_regularization, 0.0))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ beta
        loss_terms = np.logaddexp(0.0, logits) - labels * logits
        residual = special.expit(logits) - labels
        grad = design.T @ residual
        if l2 > 0:
            loss_terms = loss_terms.sum() + 0.5 * l2 * np.sum(beta[1:] ** 2)
            grad[1:] += l2 * beta[1:]
        else:
            loss_terms = loss_terms.sum()
        return float(loss_terms), grad

    initial = np.zeros(design.shape[1], dtype=float)
    blank_fraction = np.clip(labels.mean(), 1e-6, 1 - 1e-6)
    initial[0] = np.log(blank_fraction / (1 - blank_fraction))
    result = optimize.minimize(
        lambda b: objective(b)[0],
        initial,
        jac=lambda b: objective(b)[1],
        method='L-BFGS-B',
        options={'maxiter': int(max_iterations)})
    return result.x.astype(float), centers, scales, bool(result.success)


def _score_logistic_regression(features: np.ndarray,
                               coefficients: np.ndarray,
                               centers: np.ndarray,
                               scales: np.ndarray) -> np.ndarray:
    standardized = (features - centers) / scales
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    return special.expit(design @ coefficients)


def _select_logistic_probability_threshold(scores: np.ndarray,
                                           is_blank: np.ndarray,
                                           blank_barcode_count: int,
                                           coding_barcode_count: int,
                                           target_misidentification_rate: float
                                           ) -> tuple[float, dict]:
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    sorted_blank = is_blank[order].astype(int)
    cumulative_blank = np.cumsum(sorted_blank)
    cumulative_coding = np.cumsum(1 - sorted_blank)

    with np.errstate(divide='ignore', invalid='ignore'):
        misidentification = (
            (cumulative_blank / blank_barcode_count)
            / (cumulative_coding / coding_barcode_count))

    finite = np.isfinite(misidentification) & (cumulative_coding > 0)
    valid = finite & (misidentification <= target_misidentification_rate)
    if np.any(valid):
        selected = int(np.flatnonzero(valid)[-1])
    elif np.any(finite):
        selected = int(np.flatnonzero(finite)[0])
    else:
        return -np.inf, {
            'selected_count': 0,
            'selected_blank_count': 0,
            'selected_coding_count': 0,
            'estimated_misidentification_rate': np.nan}

    threshold = float(np.nextafter(sorted_scores[selected], np.inf))
    selected_blank = int(cumulative_blank[selected])
    selected_coding = int(cumulative_coding[selected])
    if selected_coding == 0:
        estimated_misid = np.nan
    else:
        estimated_misid = float(
            (selected_blank / blank_barcode_count)
            / (selected_coding / coding_barcode_count))
    return threshold, {
        'selected_count': int(selected + 1),
        'selected_blank_count': selected_blank,
        'selected_coding_count': selected_coding,
        'estimated_misidentification_rate': estimated_misid}


class AbstractFilterBarcodes(decode.BarcodeSavingParallelAnalysisTask):
    """
    An abstract class for filtering barcodes identified by pixel-based decoding.
    """

    outputGroup = 'Decode'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_codebook(self):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        return decodeTask.get_codebook()


class FilterBarcodes(AbstractFilterBarcodes):

    """
    An analysis task that filters barcodes based on area and mean
    intensity.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'area_threshold' not in self.parameters:
            self.parameters['area_threshold'] = 3
        if 'intensity_threshold' not in self.parameters:
            self.parameters['intensity_threshold'] = 200
        if 'distance_threshold' not in self.parameters:
            self.parameters['distance_threshold'] = 1e6

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['decode_task']]

    def _run_analysis(self, fragmentIndex):
        decodeTask = self.dataSet.load_analysis_task(
                self.parameters['decode_task'])
        areaThreshold = self.parameters['area_threshold']
        intensityThreshold = self.parameters['intensity_threshold']
        distanceThreshold = self.parameters['distance_threshold']
        barcodeDB = self.get_barcode_database()
        barcodeDB.write_barcodes(
            decodeTask.get_barcode_database().get_filtered_barcodes(
                areaThreshold, intensityThreshold,
                distanceThreshold=distanceThreshold, fov=fragmentIndex),
            fov=fragmentIndex)


class GenerateAdaptiveThreshold(analysistask.AnalysisTask):

    """
    An analysis task that generates a three-dimension mean intenisty,
    area, minimum distance histogram for barcodes as they are decoded.
    """

    outputGroup = 'Decode'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'tolerance' not in self.parameters:
            self.parameters['tolerance'] = 0.001
        # ensure decode_task is specified
        decodeTask = self.parameters['decode_task']
        if 'intensity_bins' not in self.parameters:
            self.parameters['intensity_bins'] = 199
        if 'distance_bins' not in self.parameters:
            self.parameters['distance_bins'] = 66
        if 'area_bins' not in self.parameters:
            self.parameters['area_bins'] = 33
        if 'threshold_solver_method' not in self.parameters:
            self.parameters['threshold_solver_method'] = 'cumulative_bins'
        if 'intensity_transform' not in self.parameters:
            self.parameters['intensity_transform'] = 'log10'
        if 'overshoot_toward_target' not in self.parameters:
            self.parameters['overshoot_toward_target'] = False
        if 'overshoot_tolerance' not in self.parameters:
            self.parameters['overshoot_tolerance'] = 0.20
        if 'report_bracketing_thresholds' not in self.parameters:
            self.parameters['report_bracketing_thresholds'] = False
        # Which fragments this histogram is built from and waited on. None
        # means every fov in the dataset, which is right for a full run and
        # wrong for a subset: the wait below can only ever be satisfied by
        # fovs the decode task actually produced, so a subset run with the
        # default spins forever.
        if 'fovs' not in self.parameters:
            self.parameters['fovs'] = None
        # How many completed fovs to sample when choosing the intensity bin
        # edges, and equivalently the minimum number needed before binning can
        # start. The bins only need the intensity range, so a sample is enough
        # -- but the sample size also gates the whole task, so on a subset run
        # it has to come down to the number of fovs being decoded.
        if 'bin_sample_fovs' not in self.parameters:
            self.parameters['bin_sample_fovs'] = None
        # Seconds to wait between polls while decode fragments are still
        # arriving. The original loop had no sleep at all and burned a core.
        if 'poll_interval' not in self.parameters:
            self.parameters['poll_interval'] = 30

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def _selected_fragments(self):
        """Fragment indices this task builds its histogram from.

        An explicit `fovs` on this task wins. Otherwise the decode task is
        asked what it covers, so a subset run does not have to declare the
        same thing twice. Failing both, every fov in the dataset.
        """
        fovs = self.parameters['fovs']
        if fovs is None:
            fovs = self._decode_fov_subset()
        if fovs is None:
            return list(range(self.fragment_count()))
        return sorted({int(f) for f in fovs})

    def _decode_fov_subset(self):
        """The fov subset the decode task declares, or None for all of them.

        A z-plane subset (decode_z_index) needs no special handling: the
        barcodes simply only exist for those planes, and the histogram is
        built from whatever barcodes are there.
        """
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        declared = decodeTask.get_parameters().get('fovs')
        if declared is None:
            return None
        return [int(f) for f in declared]

    def _describe_selection(self) -> str:
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        zIndex = decodeTask.get_parameters().get('decode_z_index')
        zDescription = 'all z' if zIndex is None else (
            '%d z planes' % len(zIndex)
            if isinstance(zIndex, (list, tuple)) else 'z index %s' % zIndex)
        return '%d of %d fovs, %s' % (len(self._selected_fragments()),
                                      self.fragment_count(), zDescription)

    def get_estimated_memory(self):
        return 5000

    def get_estimated_time(self):
        return 1800

    def get_dependencies(self):
        return [self.parameters['run_after_task']]

    def get_blank_count_histogram(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('blank_counts', self)

    def get_coding_count_histogram(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('coding_counts', self)

    def get_total_count_histogram(self) -> np.ndarray:
        return self.get_blank_count_histogram() \
               + self.get_coding_count_histogram()

    def get_area_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('area_bins', self)

    def get_distance_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result(
            'distance_bins', self)

    def get_intensity_bins(self) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result(
            'intensity_bins', self, None)

    def get_blank_fraction_histogram(self) -> np.ndarray:
        """ Get the normalized blank fraction histogram indicating the
        normalized blank fraction for each intensity, distance, and area
        bin.

        Returns: The normalized blank fraction histogram. The histogram
            has three dimensions: mean intensity, minimum distance, and area.
            The bins in each dimension are defined by the bins returned by
            get_area_bins, get_distance_bins, and get_area_bins, respectively.
            Each entry indicates the number of blank barcodes divided by the
            number of coding barcodes within the corresponding bin
            normalized by the fraction of blank barcodes in the codebook.
            With this normalization, when all (both blank and coding) barcodes
            are selected with equal probability, the blank fraction is
            expected to be 1.
        """
        blankHistogram = self.get_blank_count_histogram()
        totalHistogram = self.get_coding_count_histogram()
        blankFraction = blankHistogram / totalHistogram
        blankFraction[totalHistogram == 0] = np.finfo(blankFraction.dtype).max
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankFraction /= blankBarcodeCount/(
                blankBarcodeCount + codingBarcodeCount)
        return blankFraction

    def calculate_misidentification_rate_for_threshold(
            self, threshold: float) -> float:
        """ Calculate the misidentification rate for a specified blank
        fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The estimated misidentification rate, estimated as the
            number of blank barcodes per blank barcode divided
            by the number of coding barcodes per coding barcode.
        """
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankHistogram = self.get_blank_count_histogram()
        codingHistogram = self.get_coding_count_histogram()
        blankFraction = self.get_blank_fraction_histogram()

        selectBins = blankFraction < threshold
        codingCounts = np.sum(codingHistogram[selectBins])
        blankCounts = np.sum(blankHistogram[selectBins])

        return ((blankCounts/blankBarcodeCount) /
                (codingCounts/codingBarcodeCount))

    def calculate_threshold_for_misidentification_rate(
            self, targetMisidentificationRate: float) -> float:
        """ Calculate the normalized blank fraction threshold that achieves
        a specified misidentification rate.

        Args:
            targetMisidentificationRate: the target misidentification rate
        Returns: the normalized blank fraction threshold that achieves
            targetMisidentificationRate
        """
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        solverMethod = self.parameters.get(
            'threshold_solver_method', 'newton')
        if self.parameters.get('report_bracketing_thresholds', False):
            self.dataSet.save_json_analysis_result(
                cumulative_bins_bracketing(
                    self.get_blank_count_histogram(),
                    self.get_coding_count_histogram(),
                    self.get_blank_fraction_histogram(),
                    targetMisidentificationRate,
                    blankBarcodeCount, codingBarcodeCount),
                'threshold_bracketing', self)
        if solverMethod == 'cumulative_bins':
            return _threshold_from_cumulative_bins(
                self.get_blank_count_histogram(),
                self.get_coding_count_histogram(),
                self.get_blank_fraction_histogram(),
                targetMisidentificationRate,
                blankBarcodeCount,
                codingBarcodeCount,
                overshoot_toward_target=self.parameters.get(
                    'overshoot_toward_target', False),
                overshoot_tolerance=self.parameters.get(
                    'overshoot_tolerance', 0.20))
        if solverMethod == 'newton':
            tolerance = self.parameters['tolerance']

            def misidentification_rate_error_for_threshold(x):
                return self.calculate_misidentification_rate_for_threshold(x) \
                    - targetMisidentificationRate

            return _threshold_from_newton(
                misidentification_rate_error_for_threshold, tolerance)
        raise ValueError('Unrecognized threshold_solver_method: '
                         + str(solverMethod))

    def calculate_barcode_count_for_threshold(self, threshold: float) -> float:
        """ Calculate the number of barcodes remaining after applying
        the specified normalized blank fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The number of barcodes passing the threshold.
        """
        blankHistogram = self.get_blank_count_histogram()
        codingHistogram = self.get_coding_count_histogram()
        blankFraction = self.get_blank_fraction_histogram()
        return np.sum(blankHistogram[blankFraction < threshold]) \
            + np.sum(codingHistogram[blankFraction < threshold])

    def extract_barcodes_with_threshold(self, blankThreshold: float,
                                        barcodeSet: pandas.DataFrame
                                        ) -> pandas.DataFrame:
        selectData = barcodeSet[
            ['mean_intensity', 'min_distance', 'area']].values
        intensityTransform = _get_intensity_transform_method(self.parameters)
        selectData[:, 0] = _transform_intensity_values(
            selectData[:, 0], intensityTransform)
        blankFractionHistogram = self.get_blank_fraction_histogram()

        barcodeBins = np.array(
            (np.digitize(selectData[:, 0], self.get_intensity_bins(),
                         right=True),
             np.digitize(selectData[:, 1], self.get_distance_bins(),
                         right=True),
             np.digitize(selectData[:, 2], self.get_area_bins()))) - 1
        barcodeBins[0, :] = np.clip(
            barcodeBins[0, :], 0, blankFractionHistogram.shape[0]-1)
        barcodeBins[1, :] = np.clip(
            barcodeBins[1, :], 0, blankFractionHistogram.shape[1]-1)
        barcodeBins[2, :] = np.clip(
            barcodeBins[2, :], 0, blankFractionHistogram.shape[2]-1)
        raveledIndexes = np.ravel_multi_index(
            barcodeBins[:, :], blankFractionHistogram.shape)

        thresholdedBlankFraction = blankFractionHistogram < blankThreshold
        return barcodeSet[np.take(thresholdedBlankFraction, raveledIndexes)]

    def _extract_counts(self, barcodes, intensityBins, distanceBins, areaBins):
        barcodeData = barcodes[
            ['mean_intensity', 'min_distance', 'area']].values
        intensityTransform = _get_intensity_transform_method(self.parameters)
        barcodeData[:, 0] = _transform_intensity_values(
            barcodeData[:, 0], intensityTransform)
        shape = (len(intensityBins) - 1, len(distanceBins) - 1,
                 len(areaBins) - 1)
        counts = np.zeros(shape, dtype=float)
        if barcodeData.shape[0] == 0:
            return counts
        # Bin identically to extract_barcodes_with_threshold: digitize + clip so
        # out-of-range barcodes are counted in the edge bins instead of being
        # dropped (as np.histogramdd would). This keeps the histogram used to
        # pick the threshold consistent with how the threshold is later applied,
        # so the achieved misidentification rate matches the target estimate.
        iBin = np.clip(np.digitize(barcodeData[:, 0], intensityBins,
                                   right=True) - 1, 0, shape[0] - 1)
        dBin = np.clip(np.digitize(barcodeData[:, 1], distanceBins,
                                   right=True) - 1, 0, shape[1] - 1)
        aBin = np.clip(np.digitize(barcodeData[:, 2], areaBins) - 1,
                       0, shape[2] - 1)
        np.add.at(counts, (iBin, dBin, aBin), 1)
        return counts

    def _run_analysis(self):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        barcodeDB = decodeTask.get_barcode_database()
        completeFragments = \
            self.dataSet.load_numpy_analysis_result_if_available(
                'complete_fragments', self, [False]*self.fragment_count())
        selectedFragments = self._selected_fragments()
        requestedSample = self.parameters['bin_sample_fovs']
        if requestedSample is None:
            requestedSample = 20
        sampleTarget = max(1, min(requestedSample, len(selectedFragments)))
        print('%s: building thresholds from %s (sampling %d for the intensity '
              'range)' % (self.get_analysis_name(), self._describe_selection(),
                          sampleTarget), flush=True)
        pendingFragments = [
            decodeTask.is_complete(i) and not completeFragments[i]
            for i in range(self.fragment_count())]

        areaBins = self.dataSet.load_numpy_analysis_result_if_available(
            'area_bins', self, np.arange(1, self.parameters['area_bins'] + 2))
        defaultDistanceBins = np.linspace(
            0, decodeTask.parameters['distance_threshold'] + 0.01,
            self.parameters['distance_bins'] + 1)
        distanceBins = self.dataSet.load_numpy_analysis_result_if_available(
            'distance_bins', self, defaultDistanceBins)
        intensityBins = self.dataSet.load_numpy_analysis_result_if_available(
            'intensity_bins', self, None)

        blankCounts = self.dataSet.load_numpy_analysis_result_if_available(
            'blank_counts', self, None)
        codingCounts = self.dataSet.load_numpy_analysis_result_if_available(
            'coding_counts', self, None)

        self.dataSet.save_numpy_analysis_result(
            areaBins, 'area_bins', self)
        if distanceBins is not None:
            self.dataSet.save_numpy_analysis_result(
                distanceBins, 'distance_bins', self)

        updated = False
        while not all(completeFragments[i] for i in selectedFragments):
            if (intensityBins is None or distanceBins is None or
                    blankCounts is None or codingCounts is None):
                for i in selectedFragments:
                    if not pendingFragments[i] and decodeTask.is_complete(i):
                        pendingFragments[i] = decodeTask.is_complete(i)

                readyFragments = [i for i in selectedFragments
                                  if pendingFragments[i]]
                if len(readyFragments) < sampleTarget:
                    # Nothing to do until more decode fragments land. Without
                    # a sleep this is a busy-wait on a full core.
                    time.sleep(self.parameters['poll_interval'])
                    continue

                if True:
                    def extreme_values(inputData: pandas.Series):
                        return inputData.min(), inputData.max()
                    sampleSize = min(sampleTarget, len(readyFragments))
                    sampledFragments = np.random.choice(
                            readyFragments, size=sampleSize, replace=False)
                    intensityExtremes = [
                        extreme_values(barcodeDB.get_barcodes(
                            i, columnList=['mean_intensity'])['mean_intensity'])
                        for i in sampledFragments]
                    maxIntensity = np.max([x[1] for x in intensityExtremes])
                    intensityBins = _build_intensity_bins(
                        maxIntensity,
                        self.parameters['intensity_bins'],
                        _get_intensity_transform_method(self.parameters))
                    self.dataSet.save_numpy_analysis_result(
                        intensityBins, 'intensity_bins', self)

                    blankCounts = np.zeros((len(intensityBins)-1,
                                            len(distanceBins)-1,
                                            len(areaBins)-1))
                    codingCounts = np.zeros((len(intensityBins)-1,
                                            len(distanceBins)-1,
                                            len(areaBins)-1))

            else:
                progressed = False
                for i in selectedFragments:
                    if not completeFragments[i] and decodeTask.is_complete(i):
                        progressed = True
                        barcodes = barcodeDB.get_barcodes(
                            i, columnList=['barcode_id', 'mean_intensity',
                                           'min_distance', 'area'])
                        blankCounts += self._extract_counts(
                            barcodes[barcodes['barcode_id'].isin(
                                codebook.get_blank_indexes())],
                            intensityBins, distanceBins, areaBins)
                        codingCounts += self._extract_counts(
                            barcodes[barcodes['barcode_id'].isin(
                                codebook.get_coding_indexes())],
                            intensityBins, distanceBins, areaBins)
                        updated = True
                        completeFragments[i] = True

                if updated:
                    self.dataSet.save_numpy_analysis_result(
                        completeFragments, 'complete_fragments', self)
                    self.dataSet.save_numpy_analysis_result(
                        blankCounts, 'blank_counts', self)
                    self.dataSet.save_numpy_analysis_result(
                        codingCounts, 'coding_counts', self)
                if not progressed:
                    time.sleep(self.parameters['poll_interval'])


class AdaptiveFilterBarcodes(AbstractFilterBarcodes):

    """
    An analysis task that filters barcodes based on a mean intensity threshold
    for each area based on the abundance of blank barcodes. The threshold
    is selected to achieve a specified misidentification rate.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'misidentification_rate' not in self.parameters:
            self.parameters['misidentification_rate'] = 0.05

        if 'remove_z_duplicated_barcodes' not in self.parameters:
            self.parameters['remove_z_duplicated_barcodes'] = False
        if self.parameters['remove_z_duplicated_barcodes']:
            if 'z_duplicate_z_threshold' not in self.parameters:
                self.parameters['z_duplicate_z_threshold'] = 1
            if 'z_duplicate_xy_pixel_threshold' not in self.parameters:
                self.parameters['z_duplicate_xy_pixel_threshold'] = np.sqrt(2)

        # Post-filter images. The decode task writes its image BEFORE any
        # filtering exists, so it shows every pixel within the distance
        # threshold -- on this data 936k components per plane of which 1.1%
        # survive, median component area 1 px. Judging decode quality from that
        # image judges the unfiltered decode. This writes the same field with
        # only the surviving barcodes, which is what should be inspected.
        # On by default for a few fovs and one plane: the point is that every
        # run leaves behind images to eyeball. The fovs and plane come from
        # merlin.util.imagesample with write_images_seed, the same draw
        # Preprocess and Decode make, so the filtered image is of a field
        # whose decoded and preprocessed images exist too.
        if 'write_images_seed' not in self.parameters:
            self.parameters['write_images_seed'] = imagesample.DEFAULT_SEED
        if 'write_filtered_images' not in self.parameters:
            self.parameters['write_filtered_images'] = True
        if 'write_filtered_fovs' not in self.parameters \
                or 'write_filtered_z' not in self.parameters:
            fovs, zIndexes = imagesample.default_image_selection(
                self.dataSet, self.parameters['write_images_seed'])
            self.parameters.setdefault('write_filtered_fovs', fovs)
            self.parameters.setdefault('write_filtered_z', zIndexes)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def _write_filtered_images(self, fragmentIndex, barcodes) -> None:
        """Write the decoded image masked to the barcodes that survived.

        A barcode's unique_id is the connected-component label that
        extract_barcodes_with_index assigned via measure.label(decoded + 1), so
        relabelling the saved decoded image reproduces those ids exactly
        (verified: 100% area agreement over 9,894 barcodes). No re-decode.
        """
        from skimage import measure
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])

        # Only planes that were decoded carry anything, and only planes Decode
        # wrote an image for can be drawn. Fall back to such a plane if the
        # requested ones are not among them, for the same reason Decode does.
        decodedZ = sorted(barcodes['z'].astype(int).unique().tolist())
        imageZ = decodeTask.get_decoded_image_z(fragmentIndex)   # None = zarr
        candidates = decodedZ if imageZ is None \
            else [z for z in decodedZ if z in imageZ]
        zSel = self.parameters['write_filtered_z']
        if zSel is None:
            zIndexes = candidates
        else:
            zIndexes = [z for z in zSel if z in candidates]
            if not zIndexes and candidates:
                zIndexes = [candidates[0]]
        planes = [(z, decodeTask.get_decoded_image(fragmentIndex, z))
                  for z in zIndexes]
        planes = [(z, im) for z, im in planes if im is not None]
        if not planes:
            self.dataSet.get_logger(self).info(
                'no decoded image for fov %s, skipping filtered image',
                fragmentIndex)
            return

        with self.dataSet.writer_for_analysis_images(
                self, 'filtered', fragmentIndex) as outputTif:
            for z, decodedImage in planes:
                decoded = np.asarray(decodedImage).astype(np.int32)
                labels = measure.label(decoded + 1)
                keep = barcodes.loc[barcodes['z'] == z,
                                    'unique_id'].to_numpy(np.int64)
                out = np.where(np.isin(labels, keep), decoded, -1)
                outputTif.write(out.astype(np.float32),
                               photometric='MINISBLACK', contiguous=True)

    def _write_filtered_images_if_requested(self, fragmentIndex,
                                            barcodes) -> None:
        """The example image is for looking at, so it must never fail the
        fragment whose barcodes are already written."""
        if not (self.parameters['write_filtered_images']
                and fragmentIndex in self.parameters['write_filtered_fovs']):
            return
        try:
            self._write_filtered_images(fragmentIndex, barcodes)
        except Exception as e:
            self.dataSet.get_logger(self, fragmentIndex).warning(
                'filtered image for fov %s not written: %r', fragmentIndex, e)

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['adaptive_task'],
                self.parameters['decode_task']]

    def get_adaptive_thresholds(self):
        """ Get the adaptive thresholds used for filtering barcodes.

        Returns: The GenerateaAdaptiveThershold task using for this
            adaptive filter.
        """
        return self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])

    def _run_analysis(self, fragmentIndex):
        adaptiveTask = self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])

        threshold = adaptiveTask.calculate_threshold_for_misidentification_rate(
            self.parameters['misidentification_rate'])
        if not np.isfinite(threshold):
            raise RuntimeError('Adaptive threshold is non-finite. '
                               'Check threshold histograms and codebook '
                               'blank/coding assignments.')

        bcDatabase = self.get_barcode_database()
        currentBarcodes = decodeTask.get_barcode_database()\
            .get_barcodes(fragmentIndex)
        print(len(currentBarcodes))
        print(currentBarcodes.columns)
        print(threshold)
        currentBarcodes = adaptiveTask.extract_barcodes_with_threshold(
            threshold, currentBarcodes)
        print(len(currentBarcodes))
        
        # do z duplicates after adaptive threshold
        if self.parameters['remove_z_duplicated_barcodes']:
            currentBarcodes = self._remove_z_duplicate_barcodes(currentBarcodes)

        bcDatabase.write_barcodes(currentBarcodes, fov=fragmentIndex)
        self._write_filtered_images_if_requested(fragmentIndex, currentBarcodes)



    # lets expose this filtering here, it feels more natural than in decode
    # I don't want to waste time filtering during the decode step with gpu nodes
    # same function from decode.py
    def _remove_z_duplicate_barcodes(self, bc):
        bc = barcodefilters.remove_zplane_duplicates_all_barcodeids(
            bc, self.parameters['z_duplicate_z_threshold'],
            self.parameters['z_duplicate_xy_pixel_threshold'],
            self.dataSet.get_z_positions())
        return bc


class LogisticFilterBarcodes(AbstractFilterBarcodes):

    """
    A per-FOV barcode filter that fits a logistic model to distinguish blank
    from coding barcodes using mean intensity, minimum distance, and area.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'misidentification_rate' not in self.parameters:
            self.parameters['misidentification_rate'] = 0.05
        if 'l2_regularization' not in self.parameters:
            self.parameters['l2_regularization'] = 1.0
        if 'max_iterations' not in self.parameters:
            self.parameters['max_iterations'] = 200
        if 'remove_z_duplicated_barcodes' not in self.parameters:
            self.parameters['remove_z_duplicated_barcodes'] = False
        if self.parameters['remove_z_duplicated_barcodes']:
            if 'z_duplicate_z_threshold' not in self.parameters:
                self.parameters['z_duplicate_z_threshold'] = 1
            if 'z_duplicate_xy_pixel_threshold' not in self.parameters:
                self.parameters['z_duplicate_xy_pixel_threshold'] = np.sqrt(2)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 60

    def get_dependencies(self):
        return [self.parameters['decode_task']]

    def _run_analysis(self, fragmentIndex):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankIndexes = set(codebook.get_blank_indexes())
        codingIndexes = set(codebook.get_coding_indexes())
        blankBarcodeCount = len(blankIndexes)
        codingBarcodeCount = len(codingIndexes)

        currentBarcodes = decodeTask.get_barcode_database()\
            .get_barcodes(fragmentIndex)
        featureColumns = ['mean_intensity', 'min_distance', 'area']
        validBarcodeMask = currentBarcodes['barcode_id'].isin(
            list(blankIndexes | codingIndexes)).values
        finiteFeatureMask = np.isfinite(
            currentBarcodes.loc[:, featureColumns].values.astype(float)
        ).all(axis=1)
        fitMask = validBarcodeMask & finiteFeatureMask
        fitBarcodes = currentBarcodes.loc[fitMask, :]

        bcDatabase = self.get_barcode_database()
        if fitBarcodes.empty:
            bcDatabase.write_barcodes(currentBarcodes.iloc[0:0],
                                      fov=fragmentIndex)
            self._save_filter_summary(
                fragmentIndex, currentBarcodes, currentBarcodes.iloc[0:0],
                None, np.nan, {
                    'selected_count': 0,
                    'selected_blank_count': 0,
                    'selected_coding_count': 0,
                    'estimated_misidentification_rate': np.nan},
                False, 'no finite barcodes to fit')
            return

        features = fitBarcodes.loc[:, featureColumns].values.astype(float)
        isBlank = fitBarcodes['barcode_id'].isin(blankIndexes).values
        if np.unique(isBlank).size < 2:
            if np.any(isBlank):
                selectedMask = np.zeros(len(currentBarcodes), dtype=bool)
                reason = 'only blank barcodes available'
            else:
                selectedMask = fitMask
                reason = 'only coding barcodes available'
            selectedBarcodes = currentBarcodes.loc[selectedMask, :]
            bcDatabase.write_barcodes(selectedBarcodes, fov=fragmentIndex)
            self._save_filter_summary(
                fragmentIndex, currentBarcodes, selectedBarcodes, None, np.nan,
                {
                    'selected_count': int(len(selectedBarcodes)),
                    'selected_blank_count': 0,
                    'selected_coding_count': int(len(selectedBarcodes)),
                    'estimated_misidentification_rate': 0.0},
                False, reason)
            return

        coefficients, centers, scales, fitSuccess = _fit_logistic_regression(
            features, isBlank.astype(int),
            self.parameters['l2_regularization'],
            self.parameters['max_iterations'])
        blankProbability = _score_logistic_regression(
            features, coefficients, centers, scales)
        threshold, selectionSummary = _select_logistic_probability_threshold(
            blankProbability, isBlank, blankBarcodeCount, codingBarcodeCount,
            self.parameters['misidentification_rate'])

        keepFitRows = blankProbability <= threshold
        selectedBarcodes = fitBarcodes.loc[keepFitRows, :].copy()

        if self.parameters['remove_z_duplicated_barcodes']:
            selectedBarcodes = self._remove_z_duplicate_barcodes(
                selectedBarcodes)

        bcDatabase.write_barcodes(selectedBarcodes, fov=fragmentIndex)
        model = {
            'feature_columns': featureColumns,
            'coefficients': coefficients.tolist(),
            'feature_centers': centers.tolist(),
            'feature_scales': scales.tolist()}
        self._save_filter_summary(
            fragmentIndex, currentBarcodes, selectedBarcodes, model, threshold,
            selectionSummary, fitSuccess, 'ok')

    def _save_filter_summary(self, fragmentIndex, inputBarcodes,
                             selectedBarcodes, model, threshold,
                             selectionSummary, fitSuccess, reason):
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankIndexes = set(codebook.get_blank_indexes())
        inputBlank = int(inputBarcodes['barcode_id'].isin(blankIndexes).sum())
        outputBlank = int(selectedBarcodes['barcode_id'].isin(blankIndexes).sum())
        summary = {
            'fov': int(fragmentIndex),
            'decode_task': self.parameters['decode_task'],
            'misidentification_rate': self.parameters['misidentification_rate'],
            'l2_regularization': self.parameters['l2_regularization'],
            'fit_success': bool(fitSuccess),
            'reason': reason,
            'probability_threshold': threshold,
            'input_count': int(len(inputBarcodes)),
            'input_blank_count': inputBlank,
            'input_coding_count': int(len(inputBarcodes) - inputBlank),
            'output_count': int(len(selectedBarcodes)),
            'output_blank_count': outputBlank,
            'output_coding_count': int(len(selectedBarcodes) - outputBlank),
            'selection': selectionSummary,
            'model': model}
        self.dataSet.save_json_analysis_result(
            summary, 'logistic_filter_summary', self, fragmentIndex)

    def _remove_z_duplicate_barcodes(self, bc):
        bc = barcodefilters.remove_zplane_duplicates_all_barcodeids(
            bc, self.parameters['z_duplicate_z_threshold'],
            self.parameters['z_duplicate_xy_pixel_threshold'],
            self.dataSet.get_z_positions())
        return bc


class GenerateAdaptiveThresholdLocal(analysistask.ParallelAnalysisTask):

    """
    An analysis task that generates a three-dimension mean intensity,
    area, minimum distance histogram for barcodes as they are decoded.

    This version uses a local area to generate the threshold.
    Specify the # of nearest neighboring FOV to take. 
    Only the barcodes in those FOV are used for the adaptive filter
    3x3 FOV area set neighbors = 8
    4x4 FOV area set neighbors = 15
    NxN FOV area set neighbors = N^2 - 1
    """

    outputGroup = 'Decode'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'tolerance' not in self.parameters:
            self.parameters['tolerance'] = 0.001
        # ensure decode_task is specified
        decodeTask = self.parameters['decode_task']

        if 'neighbors' not in self.parameters:
            self.parameters['neighbors'] = 15
        if 'intensity_bins' not in self.parameters:
            self.parameters['intensity_bins'] = 199
        if 'area_bins' not in self.parameters:
            self.parameters['area_bins'] = 33
        if 'distance_bins' not in self.parameters:
            self.parameters['distance_bins'] = 66
        if 'threshold_solver_method' not in self.parameters:
            self.parameters['threshold_solver_method'] = 'cumulative_bins'
        if 'intensity_transform' not in self.parameters:
            self.parameters['intensity_transform'] = 'log10'
        if 'overshoot_toward_target' not in self.parameters:
            self.parameters['overshoot_toward_target'] = False
        if 'overshoot_tolerance' not in self.parameters:
            self.parameters['overshoot_tolerance'] = 0.20
        if 'report_bracketing_thresholds' not in self.parameters:
            self.parameters['report_bracketing_thresholds'] = False

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 8000

    def get_estimated_time(self):
        return 1800

    def get_dependencies(self):
        return [self.parameters['run_after_task']]

    def get_neighboring_fovs(self, fragmentIndex):

        position_df  = self.dataSet.positions
        position_df['distance'] = np.sqrt((position_df.loc[fragmentIndex].X - position_df.X)**2 +
                                (position_df.loc[fragmentIndex].Y - position_df.Y)**2)
        # take the number of neighbors (and include the FOV itself )
        neighboring_fovs = np.argsort(position_df['distance'].values)[0:self.parameters['neighbors']+1]
        return neighboring_fovs

    def get_blank_count_histogram(self, fragmentIndex) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('blank_counts', self, fragmentIndex)

    def get_coding_count_histogram(self, fragmentIndex) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('coding_counts', self, fragmentIndex)

    def get_total_count_histogram(self, fragmentIndex) -> np.ndarray:
        return self.get_blank_count_histogram(fragmentIndex) \
               + self.get_coding_count_histogram(fragmentIndex)

    def get_area_bins(self, fragmentIndex) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result('area_bins', self, fragmentIndex)

    def get_distance_bins(self, fragmentIndex) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result(
            'distance_bins', self, fragmentIndex)

    def get_intensity_bins(self, fragmentIndex) -> np.ndarray:
        return self.dataSet.load_numpy_analysis_result(
            'intensity_bins', self, fragmentIndex)

    def get_blank_fraction_histogram(self, fragmentIndex) -> np.ndarray:
        """ Get the normalized blank fraction histogram indicating the
        normalized blank fraction for each intensity, distance, and area
        bin.

        Returns: The normalized blank fraction histogram. The histogram
            has three dimensions: mean intensity, minimum distance, and area.
            The bins in each dimension are defined by the bins returned by
            get_area_bins, get_distance_bins, and get_area_bins, respectively.
            Each entry indicates the number of blank barcodes divided by the
            number of coding barcodes within the corresponding bin
            normalized by the fraction of blank barcodes in the codebook.
            With this normalization, when all (both blank and coding) barcodes
            are selected with equal probability, the blank fraction is
            expected to be 1.
        """
        blankHistogram = self.get_blank_count_histogram(fragmentIndex)
        totalHistogram = self.get_coding_count_histogram(fragmentIndex)
        blankFraction = blankHistogram / totalHistogram
        blankFraction[totalHistogram == 0] = np.finfo(blankFraction.dtype).max
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankFraction /= blankBarcodeCount/(
                blankBarcodeCount + codingBarcodeCount)
        return blankFraction

    def calculate_misidentification_rate_for_threshold(
            self, threshold: float, fragmentIndex) -> float:
        """ Calculate the misidentification rate for a specified blank
        fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The estimated misidentification rate, estimated as the
            number of blank barcodes per blank barcode divided
            by the number of coding barcodes per coding barcode.
        """
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        blankHistogram = self.get_blank_count_histogram(fragmentIndex)
        codingHistogram = self.get_coding_count_histogram(fragmentIndex)
        blankFraction = self.get_blank_fraction_histogram(fragmentIndex)

        selectBins = blankFraction < threshold
        codingCounts = np.sum(codingHistogram[selectBins])
        blankCounts = np.sum(blankHistogram[selectBins])

        return ((blankCounts/blankBarcodeCount) /
                (codingCounts/codingBarcodeCount))

    def calculate_threshold_for_misidentification_rate(self,
            targetMisidentificationRate: float,
            fragmentIndex
            ) -> float:
        """ Calculate the normalized blank fraction threshold that achieves
        a specified misidentification rate.

        Args:
            targetMisidentificationRate: the target misidentification rate
        Returns: the normalized blank fraction threshold that achieves
            targetMisidentificationRate
        """
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        blankBarcodeCount = len(codebook.get_blank_indexes())
        codingBarcodeCount = len(codebook.get_coding_indexes())
        solverMethod = self.parameters.get(
            'threshold_solver_method', 'newton')
        if self.parameters.get('report_bracketing_thresholds', False):
            self.dataSet.save_json_analysis_result(
                cumulative_bins_bracketing(
                    self.get_blank_count_histogram(fragmentIndex),
                    self.get_coding_count_histogram(fragmentIndex),
                    self.get_blank_fraction_histogram(fragmentIndex),
                    targetMisidentificationRate,
                    blankBarcodeCount, codingBarcodeCount),
                'threshold_bracketing', self, fragmentIndex)
        if solverMethod == 'cumulative_bins':
            return _threshold_from_cumulative_bins(
                self.get_blank_count_histogram(fragmentIndex),
                self.get_coding_count_histogram(fragmentIndex),
                self.get_blank_fraction_histogram(fragmentIndex),
                targetMisidentificationRate,
                blankBarcodeCount,
                codingBarcodeCount,
                overshoot_toward_target=self.parameters.get(
                    'overshoot_toward_target', False),
                overshoot_tolerance=self.parameters.get(
                    'overshoot_tolerance', 0.20))
        if solverMethod == 'newton':
            tolerance = self.parameters['tolerance']

            def misidentification_rate_error_for_threshold(x):
                return self.calculate_misidentification_rate_for_threshold(
                    x, fragmentIndex) - targetMisidentificationRate

            return _threshold_from_newton(
                misidentification_rate_error_for_threshold, tolerance)
        raise ValueError('Unrecognized threshold_solver_method: '
                         + str(solverMethod))

    def calculate_barcode_count_for_threshold(self, threshold: float, fragmentIndex) -> float:
        """ Calculate the number of barcodes remaining after applying
        the specified normalized blank fraction threshold.

        Args:
            threshold: the normalized blank fraction threshold
        Returns: The number of barcodes passing the threshold.
        """
        blankHistogram = self.get_blank_count_histogram(fragmentIndex)
        codingHistogram = self.get_coding_count_histogram(fragmentIndex)
        blankFraction = self.get_blank_fraction_histogram(fragmentIndex)
        return np.sum(blankHistogram[blankFraction < threshold]) \
            + np.sum(codingHistogram[blankFraction < threshold])

    def extract_barcodes_with_threshold(self, 
                                        blankThreshold: float,
                                        barcodeSet: pandas.DataFrame,
                                        fragmentIndex
                                        ) -> pandas.DataFrame:
        selectData = barcodeSet[
            ['mean_intensity', 'min_distance', 'area']].values
        intensityTransform = _get_intensity_transform_method(self.parameters)
        selectData[:, 0] = _transform_intensity_values(
            selectData[:, 0], intensityTransform)
        blankFractionHistogram = self.get_blank_fraction_histogram(fragmentIndex)

        barcodeBins = np.array(
            (np.digitize(selectData[:, 0], self.get_intensity_bins(fragmentIndex),
                         right=True),
             np.digitize(selectData[:, 1], self.get_distance_bins(fragmentIndex),
                         right=True),
             np.digitize(selectData[:, 2], self.get_area_bins(fragmentIndex)))) - 1
        barcodeBins[0, :] = np.clip(
            barcodeBins[0, :], 0, blankFractionHistogram.shape[0]-1)
        barcodeBins[1, :] = np.clip(
            barcodeBins[1, :], 0, blankFractionHistogram.shape[1]-1)
        barcodeBins[2, :] = np.clip(
            barcodeBins[2, :], 0, blankFractionHistogram.shape[2]-1)
        raveledIndexes = np.ravel_multi_index(
            barcodeBins[:, :], blankFractionHistogram.shape)

        thresholdedBlankFraction = blankFractionHistogram < blankThreshold
        return barcodeSet[np.take(thresholdedBlankFraction, raveledIndexes)]

    def _extract_counts(self, barcodes, intensityBins, distanceBins, areaBins):
        barcodeData = barcodes[
            ['mean_intensity', 'min_distance', 'area']].values
        intensityTransform = _get_intensity_transform_method(self.parameters)
        barcodeData[:, 0] = _transform_intensity_values(
            barcodeData[:, 0], intensityTransform)
        shape = (len(intensityBins) - 1, len(distanceBins) - 1,
                 len(areaBins) - 1)
        counts = np.zeros(shape, dtype=float)
        if barcodeData.shape[0] == 0:
            return counts
        # Bin identically to extract_barcodes_with_threshold: digitize + clip so
        # out-of-range barcodes are counted in the edge bins instead of being
        # dropped (as np.histogramdd would). This keeps the histogram used to
        # pick the threshold consistent with how the threshold is later applied,
        # so the achieved misidentification rate matches the target estimate.
        iBin = np.clip(np.digitize(barcodeData[:, 0], intensityBins,
                                   right=True) - 1, 0, shape[0] - 1)
        dBin = np.clip(np.digitize(barcodeData[:, 1], distanceBins,
                                   right=True) - 1, 0, shape[1] - 1)
        aBin = np.clip(np.digitize(barcodeData[:, 2], areaBins) - 1,
                       0, shape[2] - 1)
        np.add.at(counts, (iBin, dBin, aBin), 1)
        return counts

    def _run_analysis(self, fragmentIndex):

        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])
        codebook = decodeTask.get_codebook()
        barcodeDB = decodeTask.get_barcode_database()
        areaBins = self.dataSet.load_numpy_analysis_result_if_available(
            'area_bins', self,
            np.arange(1, self.parameters['area_bins'] + 2),
            fragmentIndex)
        
        defaultDistanceBins = np.arange(
            0, decodeTask.parameters['distance_threshold'] + 0.02, 0.01)
        distanceBins = self.dataSet.load_numpy_analysis_result_if_available(
            'distance_bins', self, defaultDistanceBins, fragmentIndex)
        
        intensityBins = self.dataSet.load_numpy_analysis_result_if_available(
            'intensity_bins', self, None,
            fragmentIndex)

        blankCounts = self.dataSet.load_numpy_analysis_result_if_available(
            'blank_counts', self, None, fragmentIndex)
        codingCounts = self.dataSet.load_numpy_analysis_result_if_available(
            'coding_counts', self, None, fragmentIndex)

        self.dataSet.save_numpy_analysis_result(
            areaBins, 'area_bins', self, fragmentIndex)
        if distanceBins is not None:
            self.dataSet.save_numpy_analysis_result(
                distanceBins, 'distance_bins', self, fragmentIndex)

        def extreme_values(inputData: pandas.Series):
            return inputData.min(), inputData.max()
        
        sampledFragments = self.get_neighboring_fovs(fragmentIndex)
        self.dataSet.save_numpy_txt_analysis_result(
            sampledFragments, 'fov_neighbors', self, fragmentIndex)

        intensityExtremes = [
            extreme_values(barcodeDB.get_barcodes(
                i, columnList=['mean_intensity'])['mean_intensity'])
            for i in sampledFragments]
        
        maxIntensity = np.max([x[1] for x in intensityExtremes])
        intensityBins = _build_intensity_bins(
            maxIntensity,
            self.parameters['intensity_bins'],
            _get_intensity_transform_method(self.parameters))
        
        self.dataSet.save_numpy_analysis_result(
            intensityBins, 'intensity_bins', self, fragmentIndex)

        blankCounts = np.zeros((len(intensityBins)-1,
                                len(distanceBins)-1,
                                len(areaBins)-1))
        codingCounts = np.zeros((len(intensityBins)-1,
                                len(distanceBins)-1,
                                len(areaBins)-1))
        
        for i in sampledFragments:
            barcodes = barcodeDB.get_barcodes(
                i, columnList=['barcode_id', 'mean_intensity',
                                'min_distance', 'area'])
            blankCounts += self._extract_counts(
                barcodes[barcodes['barcode_id'].isin(
                    codebook.get_blank_indexes())],
                intensityBins, distanceBins, areaBins)
            codingCounts += self._extract_counts(
                barcodes[barcodes['barcode_id'].isin(
                    codebook.get_coding_indexes())],
                intensityBins, distanceBins, areaBins)

        self.dataSet.save_numpy_analysis_result(
            blankCounts, 'blank_counts', self, fragmentIndex)
        self.dataSet.save_numpy_analysis_result(
            codingCounts, 'coding_counts', self, fragmentIndex)
                    
class AdaptiveFilterBarcodesLocal(AdaptiveFilterBarcodes):

    """
    An analysis task that filters barcodes based on a mean intensity threshold
    for each area based on the abundance of blank barcodes. The threshold
    is selected to achieve a specified misidentification rate.

    Here we use the local adaptive filter generated from GenerateAdaptiveThresholdLocal
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'misidentification_rate' not in self.parameters:
            self.parameters['misidentification_rate'] = 0.05

    def get_adaptive_thresholds(self):
        """ Get the adaptive thresholds used for filtering barcodes.

        Returns: The GenerateaAdaptiveThershold task using for this
            adaptive filter.
        """
        return self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])

    def _run_analysis(self, fragmentIndex):
        adaptiveTask = self.dataSet.load_analysis_task(
            self.parameters['adaptive_task'])
        decodeTask = self.dataSet.load_analysis_task(
            self.parameters['decode_task'])

        threshold = adaptiveTask.calculate_threshold_for_misidentification_rate(
            self.parameters['misidentification_rate'],
            fragmentIndex)
        if not np.isfinite(threshold):
            raise RuntimeError('Adaptive threshold is non-finite. '
                               'Check threshold histograms and codebook '
                               'blank/coding assignments.')

        bcDatabase = self.get_barcode_database()
        currentBarcodes = decodeTask.get_barcode_database()\
            .get_barcodes(fragmentIndex)

        currentBarcodes = adaptiveTask.extract_barcodes_with_threshold(
            threshold, currentBarcodes, fragmentIndex)
        bcDatabase.write_barcodes(currentBarcodes, fov=fragmentIndex)
        self._write_filtered_images_if_requested(fragmentIndex, currentBarcodes)
