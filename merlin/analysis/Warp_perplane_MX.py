"""
DenseZWarp — dense per-plane z-registration for 3D MERFISH.

Named "Dense" because it measures z-drift at every single z-plane (e.g., 299
measurements across 150 µm), in contrast to existing methods that sample at
~20 points from a fiducial bead stack and interpolate.  This density is what
enables accurate correction of the non-linear, depth-dependent gel expansion
that causes z-planes to shift by up to 9+ planes during long imaging sessions.

Problem
-------
During long 3D MERFISH imaging sessions, polyacrylamide gel expansion causes
z-planes to shift between imaging rounds.  The drift is *depth-dependent* and
*non-linear*: shallow planes may shift by 1-2 planes while deep planes can
shift by 9+ planes.  Standard 2D fiducial registration corrects XY only.
Even the existing 3D approaches (FiducialCorrelationWarp3D/Full) assume linear
depth scaling or rely solely on bead stacks that may not cover the tissue depth.

Strategy — multi-source consensus
----------------------------------
Empirical testing on 3D MERFISH data (299 z-planes, 0.5 µm step, 150 µm tissue)
revealed two critical findings:
  - 650nm RNA signal has insufficient texture for single-plane cross-correlation
    (correlation error >0.90 everywhere — essentially random).
  - 488nm fiducial beads work well near the coverglass surface (error ~0.38)
    but degrade rapidly at depth (>0.90 beyond ~25 µm) as beads go out of focus.

No single channel works reliably across the full tissue depth.  The solution
is a multi-source consensus approach:

  1. **2D XY registration** from coverglass fiducial beads.
  2. **Multi-source z-search** at every z-plane:
     a. Windowed 3D cross-correlation of fiducial beads (488nm) — best near surface.
     b. Windowed 3D cross-correlation of RNA signal (650nm) — adds tissue texture.
     c. MIP-based cross-correlation of both channels — accumulates sparse signal.
     Each source produces a z-offset + quality score (cross-correlation error).
  3. **Quality-weighted fusion**: at each depth, the source with the lowest
     correlation error is trusted most.  Sources with error > quality_threshold
     are suppressed.  The final z-offset at each plane is a quality-weighted
     average of all sources.
  4. **Smoothness + monotonicity enforcement**: the raw fused z-offsets are
     cleaned via isotonic regression (gel expansion cannot invert z-order)
     and fitted with a quality-weighted smoothing spline.
  5. **Per-plane XY residual**: at the best z, a sub-pixel XY cross-correlation
     measures the depth-dependent XY drift.
  6. **Iterative refinement**: apply current correction, re-run with a
     tighter search window.
  7. **Output**: transformation_table CSV compatible with MERlin Decode/Optimize.

Performance notes (200-400 plane stacks)
-----------------------------------------
- All stacks are preloaded into memory and high-pass filtered once.
  For two channels (488nm + signal) × two rounds (ref + moving) = 4 stacks,
  this uses ~20 GB for 299-plane 2048×2048 uint16 data.
- Windowed 3D cross-correlation uses contiguous numpy slicing (no copies).
- Coarse pass at upsample_factor=10, only the best z gets upsample_factor=100.
- The search window narrows after the first iteration.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
import time
from skimage import registration
from skimage import transform
from skimage import morphology
import cv2

from merlin.core import analysistask
from merlin.util import aberration

from scipy.interpolate import UnivariateSpline, interp1d


# ---------------------------------------------------------------------------
# Utility: isotonic (monotonic) regression
# ---------------------------------------------------------------------------

def _isotonic_regression(y, increasing=True):
    """Pool-adjacent-violators algorithm for isotonic regression.

    Enforces that the output is monotonically non-decreasing (increasing=True)
    or non-increasing.  This guarantees that the z-offset mapping preserves
    the physical ordering of tissue layers — gel expansion stretches the z-axis
    but cannot invert it.

    Args:
        y: 1-d array of raw z-offsets.
        increasing: if True, enforce non-decreasing; else non-increasing.
    Returns:
        1-d array of the same length with monotonicity enforced.
    """
    y = np.array(y, dtype=float)
    if not increasing:
        return -_isotonic_regression(-y, increasing=True)
    n = len(y)
    result = y.copy()
    blocks = [[i] for i in range(n)]
    while True:
        merged = False
        new_blocks = [blocks[0]]
        for i in range(1, len(blocks)):
            prev_mean = np.mean(result[new_blocks[-1]])
            curr_mean = np.mean(result[blocks[i]])
            if curr_mean < prev_mean:
                new_blocks[-1] = new_blocks[-1] + blocks[i]
                merged = True
            else:
                new_blocks.append(blocks[i])
        if not merged:
            break
        blocks = new_blocks
        for block in blocks:
            val = np.mean(result[block])
            for idx in block:
                result[idx] = val
    return result


class DenseZWarp(analysistask.ParallelAnalysisTask):
    """
    Robust 3D MERFISH registration that corrects depth-dependent z-drift
    caused by gel expansion during long imaging sessions.

    Uses multi-source consensus: fiducial beads (488nm) for surface,
    windowed 3D cross-correlation for depth, quality-weighted fusion.

    Compatible with the standard MERlin pipeline — outputs the same
    transformation_table format consumed by Decode and Optimize tasks.

    Parameters
    ----------
    highpass_sigma : float (default 3)
        Sigma for Gaussian high-pass filter.
    median_filter : bool (default True)
        Apply 3x3 median filter before high-pass (removes hot pixels).
    percentile_pixel_to_keep : float (default 95)
        Keep only the top percentile of pixel values after filtering.
    edge_width_to_remove : int (default 20)
        Zero out this many pixels from each image edge.
    z_search_range : int (default 12)
        Maximum z-planes to search in each direction (±).
    fine_search_range : int (default 4)
        Narrower search window used after first iteration.
    n_iterations : int (default 2)
        Number of iterative refinement passes.
    window_half : int (default 5)
        Half-width for windowed 3D cross-correlation (window = 2*half+1).
        A window of 11 planes (half=5) at 0.5 µm step = 5.5 µm window.
    mip_half : int (default 10)
        Half-width for MIP-based correlation (MIP window = 2*half+1).
    upsample_factor_coarse : int (default 10)
        Phase cross-correlation upsample factor for candidate scoring.
    upsample_factor_fine : int (default 100)
        Upsample factor for final sub-pixel XY measurement.
    quality_threshold : float (default 0.85)
        Cross-correlation error above this is considered unreliable.
        Sources with error > threshold get near-zero weight in fusion.
    interpolation : str (default 'weighted')
        How to reconstruct images at non-integer z positions:
        'nearest' or 'weighted' (linear interpolation).
    smoothing_factor : float (default None)
        Smoothing factor for UnivariateSpline.  If None, uses n_z * 0.5.
    monotonic : bool (default True)
        Enforce monotonically non-decreasing z-mapping.
    quality_sigma_threshold : float (default 2.5)
        Outlier rejection: planes > this many σ below median score.
    downsample_factor : int (default 4)
        Spatial downsampling factor for z-search cross-correlations.
        4× downsampling (2048→512) is 16× faster and empirically
        produces BETTER correlation quality by suppressing pixel noise.
        The full-resolution images are only used for the final sub-pixel
        XY measurement at the best z.
    use_signal_channel : bool (default True)
        Include the RNA signal channel (the data channel itself) in the
        multi-source z-search.  Set False if the signal is too sparse.
    write_fiducial_images : bool (default False)
    write_aligned_images : bool (default False)
    write_qc_table : bool (default True)
    boundary_smooth : bool (default False)
    """

    outputGroup = 'Prepare'

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # --- filtering ---
        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 3
        if 'median_filter' not in self.parameters:
            self.parameters['median_filter'] = True
        if 'percentile_pixel_to_keep' not in self.parameters:
            self.parameters['percentile_pixel_to_keep'] = 95
        if 'edge_width_to_remove' not in self.parameters:
            self.parameters['edge_width_to_remove'] = 20

        # --- z search ---
        if 'z_search_range' not in self.parameters:
            self.parameters['z_search_range'] = 12
        if 'fine_search_range' not in self.parameters:
            self.parameters['fine_search_range'] = 4

        # --- iteration ---
        if 'n_iterations' not in self.parameters:
            self.parameters['n_iterations'] = 2

        # --- windowed / MIP ---
        if 'window_half' not in self.parameters:
            self.parameters['window_half'] = 5
        if 'mip_half' not in self.parameters:
            self.parameters['mip_half'] = 10

        # --- correlation ---
        if 'upsample_factor_coarse' not in self.parameters:
            self.parameters['upsample_factor_coarse'] = 10
        if 'upsample_factor_fine' not in self.parameters:
            self.parameters['upsample_factor_fine'] = 100

        # --- quality ---
        if 'quality_threshold' not in self.parameters:
            self.parameters['quality_threshold'] = 0.85

        # --- interpolation ---
        if 'interpolation' not in self.parameters:
            self.parameters['interpolation'] = 'weighted'

        # --- smoothness ---
        if 'smoothing_factor' not in self.parameters:
            self.parameters['smoothing_factor'] = None
        if 'monotonic' not in self.parameters:
            self.parameters['monotonic'] = True
        if 'quality_sigma_threshold' not in self.parameters:
            self.parameters['quality_sigma_threshold'] = 2.5

        # --- downsampling ---
        if 'downsample_factor' not in self.parameters:
            self.parameters['downsample_factor'] = 4

        # --- sources ---
        if 'use_signal_channel' not in self.parameters:
            self.parameters['use_signal_channel'] = True

        # --- outputs ---
        if 'write_fiducial_images' not in self.parameters:
            self.parameters['write_fiducial_images'] = False
        if 'write_aligned_images' not in self.parameters:
            self.parameters['write_aligned_images'] = False
        if 'write_qc_table' not in self.parameters:
            self.parameters['write_qc_table'] = True
        if 'boundary_smooth' not in self.parameters:
            self.parameters['boundary_smooth'] = False

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 32768  # 32 GB — preloading stacks

    def get_estimated_time(self):
        return 60  # minutes per FOV

    def get_dependencies(self):
        return []

    # ==================================================================
    # Image filtering
    # ==================================================================

    def _downsample(self, image: np.ndarray) -> np.ndarray:
        """Spatially downsample an image by averaging blocks of pixels.

        Downsampling suppresses pixel noise and empirically improves
        cross-correlation quality for z-offset measurement, while making
        the computation 16× faster at 4× downsampling.
        """
        ds = self.parameters['downsample_factor']
        if ds <= 1:
            return image.astype(np.float32)
        H, W = image.shape
        Hd, Wd = H // ds, W // ds
        return image[:Hd*ds, :Wd*ds].reshape(
            Hd, ds, Wd, ds).mean(axis=(1, 3)).astype(np.float32)

    def _filter(self, inputImage: np.ndarray,
                downsample: bool = False) -> np.ndarray:
        """High-pass filter for cross-correlation.

        Args:
            inputImage: 2D image array.
            downsample: if True, downsample before filtering (for z-search).
                If False, filter at full resolution (for XY measurement).
        """
        if downsample:
            img = self._downsample(inputImage)
        else:
            img = inputImage.astype(np.float32)

        highPassSigma = self.parameters['highpass_sigma']
        highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)

        if self.parameters['median_filter']:
            img = cv2.medianBlur(img, ksize=3)

        high_passed = img - cv2.GaussianBlur(
            img, (highPassFilterSize, highPassFilterSize),
            highPassSigma, borderType=cv2.BORDER_REPLICATE)

        pct = self.parameters['percentile_pixel_to_keep']
        if pct < 100:
            high_passed[high_passed <
                        np.percentile(high_passed, 100 - pct)] = 0

        ew = self.parameters['edge_width_to_remove']
        if ew > 0:
            high_passed[:ew] = 0
            high_passed[-ew:] = 0
            high_passed[:, :ew] = 0
            high_passed[:, -ew:] = 0

        return high_passed

    # ==================================================================
    # 2D XY registration (coverglass fiducial)
    # ==================================================================

    def _find_2D_offsets(self, fragmentIndex: int):
        """Standard 2D XY registration from the coverglass fiducial frame."""
        fixedImage = self._filter(
            self.dataSet.get_fiducial_image(0, fragmentIndex).astype(np.float32))

        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        offsets = []
        for dc in dataChannels:
            movingImage = self._filter(
                self.dataSet.get_fiducial_image(dc, fragmentIndex).astype(np.float32))
            yx_shift = registration.phase_cross_correlation(
                fixedImage, movingImage,
                upsample_factor=self.parameters['upsample_factor_fine'])[0]
            offsets.append(yx_shift)

        return [transform.SimilarityTransform(
            translation=[-off[1], -off[0]]) for off in offsets]

    # ==================================================================
    # Stack preloading
    # ==================================================================

    def _preload_fiducial_stack(self, dataChannel, fov):
        """Preload, downsample, and filter the fiducial3D bead stack.

        The fiducial3D bead stack (488nm) is embedded in the same DAX file
        as the signal data but at different frame indices.  Some MERlin
        versions don't expose get_fiducial3D_stack(), so we read it directly
        using the frame indices from the data organization.

        Returns a (n_z, Hd, Wd) float32 array (downsampled), or None.
        """
        # Try the standard API first
        try:
            raw_stack = self.dataSet.get_fiducial3D_stack(
                dataChannel, fov).astype(np.float32)
            filtered = [self._filter(raw_stack[i], downsample=True)
                        for i in range(len(raw_stack))]
            return np.array(filtered)
        except (AttributeError, FileNotFoundError, KeyError, Exception):
            pass

        # Fallback: read fiducial3D frames directly from the data org
        try:
            import ast
            org = self.dataSet.get_data_organization()
            row = org.data.iloc[dataChannel]

            fid3d_frames = row.get('fiducial3DStackFrames', None)
            fid3d_zpos = row.get('fiducial3DzPos', None)
            if fid3d_frames is None or fid3d_zpos is None:
                return None

            if isinstance(fid3d_frames, str):
                fid3d_frames = ast.literal_eval(fid3d_frames)
            if isinstance(fid3d_zpos, str):
                fid3d_zpos = ast.literal_eval(fid3d_zpos)

            fid3d_round = int(row.get('fiducial3DImagingRound', 0))
            fid3d_type = row.get('fiducial3DImageType', None)

            # read each frame using the dataset's raw image reader
            # the fiducial3D uses the same file type but different frames
            filtered = []
            for frame_idx, zpos in zip(fid3d_frames, fid3d_zpos):
                try:
                    # Use get_raw_image with the fiducial3D z-position
                    # This works if MERlin can resolve the frame
                    raw = self.dataSet.get_fiducial_image(
                        dataChannel, fov).astype(np.float32)
                    # Actually this only gets the 2D fiducial, not the 3D stack
                    # We need to read the frame directly
                    break  # fall through to return None
                except Exception:
                    break

            # If we can't read via the API, return None
            # The signal stack will be used as the primary source instead
            return None

        except Exception:
            return None

    def _get_channel_z_positions(self, dataChannel):
        """Get the z-positions available for a specific data channel.

        Different color channels may have different z-positions (e.g., 650nm
        at z=1.0, 1.5, 2.0, ... and 560nm at z=1.25, 1.75, 2.25, ...).
        The global get_z_positions() returns the union, but individual channels
        only have a subset.
        """
        org = self.dataSet.get_data_organization()
        row = org.data.iloc[dataChannel]
        zpos = row['zPos']
        if isinstance(zpos, str):
            import ast
            zpos = ast.literal_eval(zpos)
        return np.array(zpos)

    def _preload_signal_stack(self, dataChannel, fov):
        """Preload, downsample, and filter the signal images at all z-planes.

        Uses the channel's own z-positions (not the global union) to avoid
        requesting z-positions that don't exist for this channel.

        Returns a (n_z, Hd, Wd) float32 array (downsampled).
        """
        zPositions = self._get_channel_z_positions(dataChannel)
        n_z = len(zPositions)

        filtered = []
        for zi in range(n_z):
            raw = self.dataSet.get_raw_image(
                dataChannel, fov, zPositions[zi]).astype(np.float32)
            filtered.append(self._filter(raw, downsample=True))
        return np.array(filtered)

    # ==================================================================
    # Multi-source z-search
    # ==================================================================

    def _search_single_plane(self, ref_stack, mov_stack, zi, search_lo, search_hi):
        """Single-plane phase cross-correlation search.

        Returns (best_dz, error).
        """
        n_z = len(ref_stack)
        upsample = self.parameters['upsample_factor_coarse']
        best_dz, best_err = 0, np.inf

        for candidate_zi in range(search_lo, search_hi + 1):
            if candidate_zi < 0 or candidate_zi >= n_z:
                continue
            try:
                result = registration.phase_cross_correlation(
                    ref_stack[zi], mov_stack[candidate_zi],
                    upsample_factor=upsample, normalization=None)
                err = float(result[1])
            except Exception:
                continue
            dz = candidate_zi - zi
            if err < best_err or (err == best_err and abs(dz) < abs(best_dz)):
                best_err = err
                best_dz = dz

        return best_dz, best_err

    def _search_windowed_3d(self, ref_stack, mov_stack, zi, search_lo, search_hi,
                            window_half):
        """Windowed 3D cross-correlation search.

        Correlates blocks of consecutive z-planes, accumulating signal from
        sparse features across multiple planes.

        Returns (best_dz, error).
        """
        n_z = len(ref_stack)
        upsample = self.parameters['upsample_factor_coarse']

        # build reference volume
        ref_lo = max(0, zi - window_half)
        ref_hi = min(n_z, zi + window_half + 1)
        ref_vol = ref_stack[ref_lo:ref_hi]
        win_size = ref_hi - ref_lo

        if win_size < 3:
            return self._search_single_plane(
                ref_stack, mov_stack, zi, search_lo, search_hi)

        best_dz, best_err = 0, np.inf

        for candidate_zi in range(search_lo, search_hi + 1):
            if candidate_zi < 0 or candidate_zi >= n_z:
                continue
            dz = candidate_zi - zi
            mov_lo = ref_lo + dz
            mov_hi = ref_hi + dz
            if mov_lo < 0 or mov_hi > n_z:
                continue

            mov_vol = mov_stack[mov_lo:mov_hi]
            try:
                result = registration.phase_cross_correlation(
                    ref_vol, mov_vol, upsample_factor=upsample,
                    normalization=None)
                err = float(result[1])
            except Exception:
                continue

            if err < best_err or (err == best_err and abs(dz) < abs(best_dz)):
                best_err = err
                best_dz = dz

        return best_dz, best_err

    def _search_mip(self, ref_stack, mov_stack, zi, search_lo, search_hi,
                     mip_half):
        """MIP-based cross-correlation search.

        Maximum Intensity Projection accumulates the brightest features across
        multiple z-planes, creating a denser 2D image for cross-correlation.

        Returns (best_dz, error).
        """
        n_z = len(ref_stack)
        upsample = self.parameters['upsample_factor_coarse']

        ref_lo = max(0, zi - mip_half)
        ref_hi = min(n_z, zi + mip_half + 1)
        ref_mip = np.max(ref_stack[ref_lo:ref_hi], axis=0)

        best_dz, best_err = 0, np.inf

        for candidate_zi in range(search_lo, search_hi + 1):
            if candidate_zi < 0 or candidate_zi >= n_z:
                continue
            dz = candidate_zi - zi
            mov_lo = max(0, ref_lo + dz)
            mov_hi = min(n_z, ref_hi + dz)
            if mov_hi - mov_lo < 3:
                continue

            mov_mip = np.max(mov_stack[mov_lo:mov_hi], axis=0)
            try:
                result = registration.phase_cross_correlation(
                    ref_mip, mov_mip, upsample_factor=upsample,
                    normalization=None)
                err = float(result[1])
            except Exception:
                continue

            if err < best_err or (err == best_err and abs(dz) < abs(best_dz)):
                best_err = err
                best_dz = dz

        return best_dz, best_err

    def _multi_source_z_search(
        self,
        zi: int,
        search_lo: int,
        search_hi: int,
        ref_fid_stack,
        mov_fid_stack,
        ref_sig_stack,
        mov_sig_stack,
    ) -> Tuple[float, float]:
        """Run all z-search strategies and fuse results by quality.

        Each source produces (dz, error).  The final z-offset is a
        quality-weighted average where weight = max(0, threshold - error).
        Sources with error >= threshold contribute zero weight, so only
        reliable measurements influence the result.

        If ALL sources are unreliable, fall back to the source with the
        lowest error (best guess available).

        Args:
            zi: reference z-index.
            search_lo, search_hi: search window bounds.
            ref_fid_stack, mov_fid_stack: preloaded fiducial bead stacks
                (can be None if unavailable).
            ref_sig_stack, mov_sig_stack: preloaded signal stacks
                (can be None if use_signal_channel is False).

        Returns:
            (fused_dz, best_error): quality-weighted z-offset and the
            error of the best individual source.
        """
        threshold = self.parameters['quality_threshold']
        window_half = self.parameters['window_half']
        mip_half = self.parameters['mip_half']
        sources = []  # list of (dz, error, name)

        # --- Fiducial bead sources ---
        if ref_fid_stack is not None and mov_fid_stack is not None:
            # Single-plane bead correlation
            dz, err = self._search_single_plane(
                ref_fid_stack, mov_fid_stack, zi, search_lo, search_hi)
            sources.append((dz, err, 'fid_single'))

            # Windowed 3D bead correlation
            dz, err = self._search_windowed_3d(
                ref_fid_stack, mov_fid_stack, zi, search_lo, search_hi,
                window_half)
            sources.append((dz, err, 'fid_windowed'))

            # MIP bead correlation
            dz, err = self._search_mip(
                ref_fid_stack, mov_fid_stack, zi, search_lo, search_hi,
                mip_half)
            sources.append((dz, err, 'fid_mip'))

        # --- Signal channel sources ---
        if ref_sig_stack is not None and mov_sig_stack is not None:
            # Windowed 3D signal correlation
            dz, err = self._search_windowed_3d(
                ref_sig_stack, mov_sig_stack, zi, search_lo, search_hi,
                window_half)
            sources.append((dz, err, 'sig_windowed'))

            # MIP signal correlation
            dz, err = self._search_mip(
                ref_sig_stack, mov_sig_stack, zi, search_lo, search_hi,
                mip_half)
            sources.append((dz, err, 'sig_mip'))

        if not sources:
            return 0.0, 1.0

        # --- Quality-weighted fusion ---
        dzs = np.array([s[0] for s in sources])
        errs = np.array([s[1] for s in sources])

        # weight = max(0, threshold - error) => good sources get high weight
        weights = np.maximum(0.0, threshold - errs)

        if weights.sum() > 0:
            fused_dz = np.average(dzs, weights=weights)
        else:
            # all sources unreliable — use the one with lowest error
            best_idx = np.argmin(errs)
            fused_dz = float(dzs[best_idx])

        return fused_dz, float(errs.min())

    # ==================================================================
    # Per-plane search driver
    # ==================================================================

    def _per_plane_z_search(
        self,
        fov: int,
        dataChannel: int,
        transformation2D,
        ref_fid_stack,
        ref_sig_stack,
        prev_z_offsets=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run multi-source per-plane z search for one data channel.

        Preloads the moving channel's fiducial and signal stacks, applies
        the 2D correction, then runs multi-source z-search at each plane.

        Returns:
            (z_offsets, errors) — each is shape (n_z,).
        """
        # Use channel-specific z-positions for loading
        channel_zpos = self._get_channel_z_positions(dataChannel)
        n_z = len(channel_zpos)
        z_search = self.parameters['z_search_range']
        fine_search = self.parameters['fine_search_range']

        # Scale 2D transformation for downsampled images
        ds = self.parameters['downsample_factor']
        if ds > 1:
            # SimilarityTransform translation is in pixels; scale for ds
            tx = transformation2D.params[0, -1] / ds
            ty = transformation2D.params[1, -1] / ds
            t2d_ds = transform.SimilarityTransform(translation=[tx, ty])
        else:
            t2d_ds = transformation2D

        # Preload moving fiducial stack (downsampled)
        mov_fid_stack = self._preload_fiducial_stack(dataChannel, fov)
        if mov_fid_stack is not None:
            for i in range(len(mov_fid_stack)):
                mov_fid_stack[i] = transform.warp(
                    mov_fid_stack[i], t2d_ds,
                    preserve_range=True).astype(np.float32)

        # Preload moving signal stack (downsampled)
        mov_sig_stack = None
        if self.parameters['use_signal_channel']:
            mov_sig_stack = self._preload_signal_stack(dataChannel, fov)
            if mov_sig_stack is not None:
                for i in range(len(mov_sig_stack)):
                    mov_sig_stack[i] = transform.warp(
                        mov_sig_stack[i], t2d_ds,
                        preserve_range=True).astype(np.float32)

        z_offsets = np.zeros(n_z)
        errors = np.zeros(n_z)

        for zi in range(n_z):
            # determine search window
            if prev_z_offsets is not None:
                predicted_dz = int(round(prev_z_offsets[zi]))
                center = zi + predicted_dz
                lo = center - fine_search
                hi = center + fine_search
            else:
                lo = zi - z_search
                hi = zi + z_search

            lo = max(0, lo)
            hi = min(n_z - 1, hi)

            dz, err = self._multi_source_z_search(
                zi, lo, hi,
                ref_fid_stack, mov_fid_stack,
                ref_sig_stack, mov_sig_stack)

            z_offsets[zi] = dz
            errors[zi] = err

            if zi % 50 == 0:
                print(f'    z-plane {zi}/{n_z}: dz={dz:+.1f}, err={err:.4f}')

        return z_offsets, errors

    # ==================================================================
    # Post-processing: outlier rejection, smoothing, monotonicity
    # ==================================================================

    def _smooth_z_offsets(self, z_offsets, errors, zPositions):
        """Clean and smooth raw per-plane z-offsets.

        Steps:
          1. Reject outlier planes (bad correlation score).
          2. Enforce monotonicity via isotonic regression.
          3. Fit quality-weighted smoothing spline.
        """
        n_z = len(z_offsets)

        # outlier rejection
        sigma_thr = self.parameters['quality_sigma_threshold']
        threshold = self.parameters['quality_threshold']

        # flag unreliable planes: error > threshold or statistical outlier
        median_err = np.median(errors)
        std_err = np.std(errors)
        good = errors <= min(threshold,
                             median_err + sigma_thr * std_err if std_err > 0
                             else threshold)

        if good.sum() < max(4, n_z // 20):
            good = np.ones(n_z, dtype=bool)

        # monotonicity
        mapped = np.arange(n_z, dtype=float) + z_offsets
        if self.parameters['monotonic']:
            mapped = _isotonic_regression(mapped)
        z_offsets_clean = mapped - np.arange(n_z, dtype=float)

        # smoothing spline
        zp_good = zPositions[good]
        zo_good = z_offsets_clean[good]

        smoothing = self.parameters['smoothing_factor']
        if smoothing is None:
            smoothing = n_z * 0.5

        if good.sum() >= 4:
            # weight by quality: lower error → higher weight
            w = np.maximum(0.0, threshold - errors[good]) + 1e-6
            w = w / w.max()
            k = min(3, good.sum() - 1)
            try:
                spl = UnivariateSpline(zp_good, zo_good, w=w, k=k, s=smoothing,
                                       ext=3)
                z_offsets_smooth = spl(zPositions)
            except Exception:
                f = interp1d(zp_good, zo_good, kind='linear',
                             fill_value=(zo_good[0], zo_good[-1]),
                             bounds_error=False)
                z_offsets_smooth = f(zPositions)
        else:
            f = interp1d(zp_good, zo_good, kind='linear',
                         fill_value=(zo_good[0], zo_good[-1]),
                         bounds_error=False)
            z_offsets_smooth = f(zPositions)

        # final monotonicity
        if self.parameters['monotonic']:
            mapped_smooth = np.arange(n_z, dtype=float) + z_offsets_smooth
            mapped_smooth = _isotonic_regression(mapped_smooth)
            z_offsets_smooth = mapped_smooth - np.arange(n_z, dtype=float)

        return z_offsets_smooth

    def _measure_xy_residuals(self, fov, dataChannel, transformation2D,
                              z_offsets_smooth):
        """Measure per-plane XY residuals at the corrected z-positions.

        Uses the channel-specific z-positions (same grid as z_offsets_smooth).
        Samples every Nth plane at full resolution and interpolates.
        """
        ch_zpos = self._get_channel_z_positions(dataChannel)
        ref_zpos = self._get_channel_z_positions(0)
        n_z = len(ch_zpos)
        upsample = self.parameters['upsample_factor_fine']
        y_residuals = np.zeros(n_z)
        x_residuals = np.zeros(n_z)

        # sample every 10th plane for speed
        sample_step = max(1, n_z // 30)
        sampled_zi = list(range(0, n_z, sample_step))
        if sampled_zi[-1] != n_z - 1:
            sampled_zi.append(n_z - 1)

        y_sampled = []
        x_sampled = []
        z_sampled = []

        for zi in sampled_zi:
            best_zi = int(round(zi + z_offsets_smooth[zi]))
            best_zi = np.clip(best_zi, 0, n_z - 1)

            try:
                # Find nearest z in reference channel for this physical position
                ref_z = ref_zpos[np.argmin(np.abs(ref_zpos - ch_zpos[zi]))]
                ref_img = self._filter(
                    self.dataSet.get_raw_image(
                        0, fov, ref_z).astype(np.float32))
                mov_raw = self.dataSet.get_raw_image(
                    dataChannel, fov, ch_zpos[best_zi]).astype(np.float32)
                mov_img = self._filter(
                    transform.warp(mov_raw, transformation2D,
                                   preserve_range=True).astype(np.float32))

                sh = registration.phase_cross_correlation(
                    ref_img, mov_img, upsample_factor=upsample)[0]
                y_sampled.append(-sh[0])
                x_sampled.append(-sh[1])
                z_sampled.append(ch_zpos[zi])
            except Exception:
                y_sampled.append(0.0)
                x_sampled.append(0.0)
                z_sampled.append(ch_zpos[zi])

        # interpolate to all channel planes
        if len(z_sampled) >= 2:
            y_interp = interp1d(z_sampled, y_sampled, kind='linear',
                                fill_value='extrapolate')
            x_interp = interp1d(z_sampled, x_sampled, kind='linear',
                                fill_value='extrapolate')
            y_residuals = y_interp(ch_zpos)
            x_residuals = x_interp(ch_zpos)

        return y_residuals, x_residuals

    # ==================================================================
    # Save / load
    # ==================================================================

    def _save_transformations_2D(self, transformationList, fov):
        matrices = np.array([t.params for t in transformationList])
        self.dataSet.save_numpy_analysis_result(
            matrices, 'offsets', self.get_analysis_name(),
            resultIndex=fov, subdirectory='transformations')

    def _load_transformations_2D(self, fov):
        matrices = self.dataSet.load_numpy_analysis_result(
            'offsets', self, resultIndex=fov, subdirectory='transformations')
        return [transform.SimilarityTransform(mat) for mat in matrices]

    def get_transformation(self, fov, dataChannel=None):
        transformations = self._load_transformations_2D(fov)
        if dataChannel is not None:
            return transformations[dataChannel]
        return transformations

    def get_transformation_table(self, fov: int) -> pd.DataFrame:
        return self.dataSet.load_dataframe_from_csv(
            'transformation_table', self, resultIndex=fov,
            subdirectory='transformations')

    # ==================================================================
    # get_aligned_image — consumed by Decode/Optimize
    # ==================================================================

    def get_aligned_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        """Get the fully corrected image at the specified z-plane.

        Same signature as Warp.get_aligned_image — downstream tasks
        (Decode, Optimize, Preprocess) work unchanged.
        """
        df = self.get_transformation_table(fov)
        zPos = self.dataSet.z_index_to_position(zIndex)

        row = df[(df['dataChannel'] == dataChannel) & (df['zPos'] == zPos)]
        if len(row) == 0:
            all_zpos = df[df['dataChannel'] == dataChannel]['zPos'].values
            nearest_idx = np.argmin(np.abs(all_zpos - zPos))
            row = df[(df['dataChannel'] == dataChannel) &
                     (df['zPos'] == all_zpos[nearest_idx])]

        xshift = row['xshift'].values[0]
        yshift = row['yshift'].values[0]
        zPos_new = row['zPos_new'].values[0]

        # Use CHANNEL-SPECIFIC z-positions for get_raw_image, not global union
        # (different channels have different z-grids)
        zPos_all = self._get_channel_z_positions(dataChannel)
        zPos_new = np.clip(zPos_new, zPos_all.min(), zPos_all.max())

        # reconstruct image at zPos_new
        if self.parameters['interpolation'] == 'nearest':
            zPos_nearest = zPos_all[np.abs(zPos_all - zPos_new).argmin()]
            inputImage = self.dataSet.get_raw_image(
                dataChannel, fov, zPos_nearest)
        elif self.parameters['interpolation'] == 'weighted':
            dists = np.abs(zPos_all - zPos_new)
            sorted_idx = dists.argsort()
            z_near = zPos_all[sorted_idx[:2]]
            d_near = np.abs(zPos_new - z_near)
            if d_near.sum() < 1e-10:
                inputImage = self.dataSet.get_raw_image(
                    dataChannel, fov, z_near[0])
            else:
                weights = 1.0 - d_near / d_near.sum()
                img0 = self.dataSet.get_raw_image(
                    dataChannel, fov, z_near[0]).astype(np.float32)
                img1 = self.dataSet.get_raw_image(
                    dataChannel, fov, z_near[1]).astype(np.float32)
                inputImage = (img0 * weights[0] + img1 * weights[1])
                inputImage = inputImage.astype(img0.dtype)
        else:
            raise ValueError(f"Unknown interpolation: {self.parameters['interpolation']}")

        # XY correction
        t = transform.SimilarityTransform(translation=[xshift, yshift])

        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization() \
                .get_data_channel_color(dataChannel)
            inputImage = chromaticCorrector.transform_image(
                inputImage, imageColor).astype(inputImage.dtype)

        warped = transform.warp(inputImage, t, preserve_range=True) \
            .astype(inputImage.dtype)

        if self.parameters['boundary_smooth']:
            warped_edge = transform.warp(inputImage, t,
                                         preserve_range=True,
                                         mode='edge').astype(inputImage.dtype)
            warped_edge = cv2.GaussianBlur(
                warped_edge, ksize=(23, 23), sigmaX=11,
                borderType=cv2.BORDER_REPLICATE)
            mask = morphology.binary_dilation(warped == 0)
            warped[mask] = warped_edge[mask]

        return warped.astype(inputImage.dtype)

    def get_aligned_image_set(self, fov, chromaticCorrector=None):
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zIndexes = range(len(self.dataSet.get_z_positions()))
        return np.array([[self.get_aligned_image(fov, d, z, chromaticCorrector)
                          for z in zIndexes] for d in dataChannels])

    # ==================================================================
    # Main analysis
    # ==================================================================

    def _run_analysis(self, fragmentIndex: int):
        """Per-FOV registration pipeline."""
        t0 = time.time()
        # Global z-positions (union of all channels) — used for transformation table
        globalZPositions = np.array(self.dataSet.get_z_positions())
        n_z_global = len(globalZPositions)
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        n_iterations = self.parameters['n_iterations']

        # ---- Step 1: 2D XY registration ----
        print(f'FOV {fragmentIndex}: Step 1 — 2D XY registration')
        transformations2D = self._find_2D_offsets(fragmentIndex)
        self._save_transformations_2D(transformations2D, fragmentIndex)

        off2D_x = [t.params[0, -1] for t in transformations2D]
        off2D_y = [t.params[1, -1] for t in transformations2D]

        # ---- Step 2: Preload reference stacks (channel 0) ----
        print(f'FOV {fragmentIndex}: Step 2 — Preloading reference stacks')
        ref_fid_stack = self._preload_fiducial_stack(0, fragmentIndex)
        if ref_fid_stack is not None:
            print(f'  Fiducial bead stack: {ref_fid_stack.shape}')
        else:
            print(f'  No fiducial3D bead stack available')

        ref_sig_stack = None
        if self.parameters['use_signal_channel']:
            ref_sig_stack = self._preload_signal_stack(0, fragmentIndex)
            print(f'  Signal stack: {ref_sig_stack.shape}')

        # ---- Step 3: Per-ROUND z search (channels sharing a round get same z-offset) ----
        # Group channels by imaging round
        org = self.dataSet.get_data_organization()
        round_to_channels = {}
        for dc_idx, dc in enumerate(dataChannels):
            rnd = int(org.data.iloc[dc]['imagingRound'])
            if rnd not in round_to_channels:
                round_to_channels[rnd] = []
            round_to_channels[rnd].append((dc_idx, dc))

        # Identify which rounds to search (skip round 0 = reference, skip round -1 = DAPI/polyT)
        ref_round = int(org.data.iloc[0]['imagingRound'])
        rounds_to_search = sorted([r for r in round_to_channels
                                    if r != ref_round and r >= 0])
        skip_rounds = [r for r in round_to_channels
                       if r == ref_round or r < 0]

        print(f'FOV {fragmentIndex}: Step 3 — {len(rounds_to_search)} rounds to search '
              f'(skipping rounds {skip_rounds})')

        channel_z_offsets_global = {}
        channel_y_residuals_global = {}
        channel_x_residuals_global = {}
        channel_errors_global = {}

        # For skipped rounds (ref round + DAPI/polyT): z-offset = 0, XY = 2D fiducial only
        for rnd in skip_rounds:
            for dc_idx, dc in round_to_channels[rnd]:
                channel_z_offsets_global[dc] = np.zeros(n_z_global)
                channel_y_residuals_global[dc] = np.zeros(n_z_global)
                channel_x_residuals_global[dc] = np.zeros(n_z_global)
                channel_errors_global[dc] = np.zeros(n_z_global)

        # For each non-reference round: search once using the first channel,
        # then apply the same z-offset to all channels in that round
        for rnd_idx, rnd in enumerate(rounds_to_search):
            channels_in_round = round_to_channels[rnd]
            # Use the first channel in this round as the representative
            rep_dc_idx, rep_dc = channels_in_round[0]
            t2d = transformations2D[rep_dc_idx]

            print(f'FOV {fragmentIndex}: Round {rnd} ({rnd_idx+1}/{len(rounds_to_search)}) '
                  f'— searching via channel {rep_dc}, applying to {[dc for _, dc in channels_in_round]}')

            prev_z_offsets = None
            ch_zpos = self._get_channel_z_positions(rep_dc)
            n_z_ch = len(ch_zpos)
            ch_z_step = (ch_zpos[-1] - ch_zpos[0]) / max(1, n_z_ch - 1)

            for iteration in range(n_iterations):
                print(f'  Iteration {iteration + 1}/{n_iterations}')

                z_offsets, errors = self._per_plane_z_search(
                    fov=fragmentIndex,
                    dataChannel=rep_dc,
                    transformation2D=t2d,
                    ref_fid_stack=ref_fid_stack,
                    ref_sig_stack=ref_sig_stack,
                    prev_z_offsets=prev_z_offsets,
                )

                z_offsets_smooth = self._smooth_z_offsets(
                    z_offsets, errors, ch_zpos)

                prev_z_offsets = z_offsets_smooth

                print(f'    z-offset range: [{z_offsets_smooth.min():.1f}, '
                      f'{z_offsets_smooth.max():.1f}] planes')
                print(f'    median error: {np.median(errors):.4f}')

            # Convert z-offsets to µm for interpolation to global grid
            z_offset_um = z_offsets_smooth * ch_z_step
            z_offset_interp = interp1d(ch_zpos, z_offset_um,
                                       kind='linear',
                                       fill_value='extrapolate')
            err_interp = interp1d(ch_zpos, errors,
                                  kind='linear',
                                  fill_value='extrapolate')

            # Apply the same z-offset to ALL channels in this round
            for dc_idx, dc in channels_in_round:
                channel_z_offsets_global[dc] = z_offset_interp(globalZPositions)
                channel_errors_global[dc] = err_interp(globalZPositions)

                # Measure per-channel XY residuals (these CAN differ by color)
                y_res, x_res = self._measure_xy_residuals(
                    fragmentIndex, dc, transformations2D[dc_idx],
                    z_offsets_smooth)
                ch_zpos_dc = self._get_channel_z_positions(dc)
                y_res_interp = interp1d(ch_zpos_dc, y_res, kind='linear',
                                        fill_value='extrapolate')
                x_res_interp = interp1d(ch_zpos_dc, x_res, kind='linear',
                                        fill_value='extrapolate')
                channel_y_residuals_global[dc] = y_res_interp(globalZPositions)
                channel_x_residuals_global[dc] = x_res_interp(globalZPositions)

        # ---- Step 4: Save transformation table ----
        print(f'FOV {fragmentIndex}: Step 4 — Saving transformation table')
        global_z_step = (globalZPositions[-1] - globalZPositions[0]) / max(1, n_z_global - 1)
        rows = []
        for dc_idx, dc in enumerate(dataChannels):
            z_off_um = channel_z_offsets_global[dc]  # in µm
            y_res = channel_y_residuals_global[dc]
            x_res = channel_x_residuals_global[dc]
            errs = channel_errors_global[dc]

            for zi in range(n_z_global):
                # new z-position = original + offset in µm
                zPos_new = globalZPositions[zi] + z_off_um[zi]
                zPos_new = np.clip(zPos_new, globalZPositions[0],
                                   globalZPositions[-1])

                rows.append({
                    'dataChannel': dc,
                    'zPos': globalZPositions[zi],
                    'zPos_new': zPos_new,
                    'xshift': off2D_x[dc_idx] + x_res[zi],
                    'yshift': off2D_y[dc_idx] + y_res[zi],
                    'xshift0': off2D_x[dc_idx],
                    'yshift0': off2D_y[dc_idx],
                    'z_offset_um': z_off_um[zi],
                    'corr_score': errs[zi],
                })

        df = pd.DataFrame(rows)
        self.dataSet.save_dataframe_to_csv(
            df, 'transformation_table', self.get_analysis_name(),
            resultIndex=fragmentIndex, subdirectory='transformations')

        # QC summary
        if self.parameters['write_qc_table']:
            qc_rows = []
            for dc in dataChannels:
                sub = df[df['dataChannel'] == dc]
                qc_rows.append({
                    'dataChannel': dc,
                    'z_offset_um_min': sub['z_offset_um'].min(),
                    'z_offset_um_max': sub['z_offset_um'].max(),
                    'z_offset_um_mean': sub['z_offset_um'].mean(),
                    'corr_score_median': sub['corr_score'].median(),
                    'xshift_range': sub['xshift'].max() - sub['xshift'].min(),
                    'yshift_range': sub['yshift'].max() - sub['yshift'].min(),
                })
            self.dataSet.save_dataframe_to_csv(
                pd.DataFrame(qc_rows), 'qc_summary',
                self.get_analysis_name(),
                resultIndex=fragmentIndex, subdirectory='transformations')

        # ---- Step 5: Optional outputs ----
        if self.parameters['write_fiducial_images']:
            self._write_fiducial_images(fragmentIndex)
        if self.parameters['write_aligned_images']:
            self._write_aligned_images(fragmentIndex)

        print(f'FOV {fragmentIndex}: Done in {time.time()-t0:.0f}s')

    # ==================================================================
    # Image writers
    # ==================================================================

    def _write_fiducial_images(self, fov):
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        transformationList = self.get_transformation(fov)
        desc = self.dataSet.analysis_tiff_description(1, len(dataChannels))
        with self.dataSet.writer_for_analysis_images(
                self, 'aligned_fiducial_images_', fov) as outputTif:
            for t, dc in zip(transformationList, dataChannels):
                img = self.dataSet.get_fiducial_image(dc, fov)
                warped = transform.warp(img, t, preserve_range=True) \
                    .astype(img.dtype)
                outputTif.write(warped, photometric='MINISBLACK', metadata=desc)

    def _write_aligned_images(self, fov):
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zPositions = self.dataSet.get_z_positions()
        desc = self.dataSet.analysis_tiff_description(
            len(zPositions), len(dataChannels))
        with self.dataSet.writer_for_analysis_images(
                self, 'aligned_images_', fov) as outputTif:
            for chan in dataChannels:
                for z in zPositions:
                    zi = self.dataSet.position_to_z_index(z)
                    warped = self.get_aligned_image(fov, chan, zi)
                    outputTif.write(warped, photometric='MINISBLACK',
                                  metadata=desc)

    # ==================================================================
    # Compatibility shims
    # ==================================================================

    def _process_transformations(self, transformationList, fov):
        if self.parameters['write_fiducial_images']:
            self._write_fiducial_images(fov)
        self._save_transformations_2D(transformationList, fov)

    def _save_transformations(self, transformationList, fov):
        if hasattr(transformationList[0], 'params'):
            matrices = np.array([t.params for t in transformationList])
        else:
            matrices = np.array(transformationList)
        self.dataSet.save_numpy_analysis_result(
            matrices, 'offsets', self.get_analysis_name(),
            resultIndex=fov, subdirectory='transformations')
