"""MERlin preprocess task that restores warped MERFISH bit images with a trained
soft-decode restoration model (per-FOV, all bits jointly), so downstream
Optimize/Decode run on the model-restored images.

Unlike CAREPreprocess (a per-channel 2D denoiser), this model is multi-channel:
it consumes all codebook-bit channels of a FOV together. We therefore restore the
whole bit-stack once per (fov, z) and return the requested channel(s).

Output space matches DeconvolutionPreprocess (high-pass-filtered, pre-scale-factor),
so MERlin's OptimizeIteration/Decode handle scale-factor + chromatic + decoding
natively. The model input pipeline (chromatic -> highpass -> /scale_factors ->
per-FOV channel_max normalize) reproduces training; the model output (in
highpass/scale-factor space) is multiplied back by the scale-factor ratios to
return to high-pass space.

Parameters (analysis JSON):
  warp_task:           name of the FiducialCorrelationWarp task (Kang's offsets)
  checkpoint:          path to the trained model best.pt
  scale_factors:       path to scale_factors.npy
  chromatic_pkl:       path to chromatic_corrections.pkl
  data_organization_csv: path to the bit dataorganization (for chromatic bit colors)
  highpass_sigma:      high-pass sigma used in training prep (default 3)
  crop_width:          border (px) to exclude from the model on each side; the
                       border is restored by identity (passes the high-pass input
                       through). 0 = run the (fully-convolutional) model on the
                       full frame. (default 0)
  codebook_index:      codebook index (default 0)
  device:              'cpu' or 'cuda' (default 'cpu')
"""
import os
import sys

import numpy as np

from merlin.core import analysistask
from merlin.util import aberration
from merlin.util import imagefilters
from merlin.data import codebook

# the trained-model inference helpers live in the restoration package
_PKG = "/n/home08/jiahaozhang/merfish_decode_transfer_pkg"
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)


class ModelRestorePreprocess(analysistask.ParallelAnalysisTask):
    """Restore warped bit images with the trained per-FOV soft-decode model."""

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        p = self.parameters
        p.setdefault("highpass_sigma", 3)
        p.setdefault("crop_width", 0)
        # The training/eval data used extract_low_first560 (first of each duplicated 560
        # frame). The full 'merged' dataset SUMS both 560 frames -> 560 bits are ~2x too
        # bright. Divide the 560-color bits by this to match the model's training input.
        p.setdefault("divide_560_by", 1.0)
        p.setdefault("codebook_index", 0)
        p.setdefault("device", "cpu")
        p.setdefault("scale_factors",
                     os.path.join(_PKG, "data2", "scale_factors.npy"))
        p.setdefault("chromatic_pkl",
                     os.path.join(_PKG, "data2", "chromatic_corrections.pkl"))
        for required in ("warp_task", "checkpoint", "data_organization_csv"):
            if required not in p:
                raise ValueError(f"ModelRestorePreprocess requires '{required}'")

        self._highPassSigma = p["highpass_sigma"]
        self._cropWidth = int(p["crop_width"])
        self.warpTask = self.dataSet.load_analysis_task(p["warp_task"])

        import torch
        from infer_merfish_softdecode_checkpoint import load_checkpoint_bundle
        from merfish_softdecode_trainlib import compute_scale_factor_ratios
        from merlin_decode_helpers import load_chromatic_corrector

        self._torch = torch
        self._device = torch.device(p["device"] if (p["device"] != "cuda" or torch.cuda.is_available()) else "cpu")
        (self._model, self._cfg, self._input_lo, self._input_hi,
         self._pred_lo, self._pred_hi) = load_checkpoint_bundle(p["checkpoint"], self._device)
        self._sf_ratios = compute_scale_factor_ratios(
            np.load(p["scale_factors"]).astype(np.float32))
        self._bit_colors, self._chromatic_corr, _ = load_chromatic_corrector(
            p["chromatic_pkl"], p["data_organization_csv"], expected_bit_count=22)
        # cache the most recent restored (fov, z) bit-stack
        self._cache_key = None
        self._cache_stack = None

    # ---- MERlin task plumbing -------------------------------------------
    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 8192

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters["warp_task"]]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters["codebook_index"])

    # ---- image accessors ------------------------------------------------
    def _bit_data_channels(self):
        org = self.dataSet.get_data_organization()
        return [org.get_data_channel_for_bit(b)
                for b in self.get_codebook().get_bit_names()]

    def _restore_fov_z(self, fov: int, zIndex: int,
                       chromaticCorrector: aberration.ChromaticCorrector = None) -> np.ndarray:
        """Restore all bit channels of one (fov, z). Returns (n_bits, Y, X) in
        high-pass (pre-scale-factor) space, matching DeconvolutionPreprocess."""
        from train_merfish_softdecode_pipeline import apply_model_to_stack, prepare_model_input_np
        from merfish_softdecode_trainlib import robust_normalize_with_stats
        from merlin_decode_helpers import apply_chromatic_correction

        channels = self._bit_data_channels()
        # warped raw bit-stack (no MERlin chromatic; we apply training chromatic below)
        raw = np.stack([self.warpTask.get_aligned_image(fov, ch, zIndex)
                        for ch in channels], axis=0).astype(np.float32)
        # undo the merged-dataset 560-frame summation to match training intensity
        div560 = float(self.parameters["divide_560_by"])
        if div560 != 1.0:
            for bi, color in enumerate(self._bit_colors):
                if str(color) == "560":
                    raw[bi] /= div560
        raw = apply_chromatic_correction(raw, self._bit_colors, self._chromatic_corr)

        cw = self._cropWidth
        sub = raw[:, cw:raw.shape[1] - cw, cw:raw.shape[2] - cw] if cw > 0 else raw

        prep = prepare_model_input_np(sub, input_space="raw",
                                      scale_factors_np=self._sf_ratios,
                                      highpass_sigma=self._highPassSigma, lowpass_sigma=0.0,
                                      input_highpass_before_model=True)
        prenorm, _lo, hi = robust_normalize_with_stats(prep, mode="channel_max")  # this FOV's own channel_max
        prenorm = np.clip(prenorm, 0, 1).astype(np.float32)
        corrected = apply_model_to_stack(
            self._model, sub, self._device,
            np.zeros(raw.shape[0], np.float32), np.ones(raw.shape[0], np.float32),
            input_space="raw", prediction_space="scaled",
            prediction_lo=np.zeros(raw.shape[0], np.float32), prediction_hi=hi.astype(np.float32),
            scale_factors_np=self._sf_ratios, highpass_sigma=self._highPassSigma, lowpass_sigma=0.0,
            prepared_input_np=prenorm, input_highpass_before_model=True)
        # model output is in highpass/scale-factor space -> back to highpass space
        restored_sub = corrected * self._sf_ratios[:, None, None]

        if cw > 0:
            # border: pass the high-pass-filtered raw through (identity restore)
            out = np.empty_like(raw)
            for bi in range(raw.shape[0]):
                out[bi] = imagefilters.highpass_filter(
                    raw[bi], int(2 * np.ceil(2 * self._highPassSigma) + 1), self._highPassSigma)
            out[:, cw:raw.shape[1] - cw, cw:raw.shape[2] - cw] = restored_sub
        else:
            out = restored_sub
        return out.astype(np.float32)

    def get_processed_image_set(
            self, fov, zIndex: int = None,
            chromaticCorrector: aberration.ChromaticCorrector = None) -> np.ndarray:
        if zIndex is None:
            return np.array([[self.get_processed_image(
                fov, self.dataSet.get_data_organization().get_data_channel_for_bit(b),
                z, chromaticCorrector)
                for z in range(len(self.dataSet.get_z_positions()))]
                for b in self.get_codebook().get_bit_names()])
        return np.array([self.get_processed_image(
            fov, self.dataSet.get_data_organization().get_data_channel_for_bit(b),
            zIndex, chromaticCorrector)
            for b in self.get_codebook().get_bit_names()])

    def _load_restored(self, fov: int) -> np.ndarray:
        """Load the persisted restored bit-stack for a FOV, shape (n_bits, n_z, Y, X).
        Cached in memory so Decode's per-(channel,z) calls don't reload."""
        if self._cache_key == fov and self._cache_stack is not None:
            return self._cache_stack
        stack = self.dataSet.load_numpy_analysis_result(
            "restored", self.analysisName, fov, "restored_images")
        self._cache_key, self._cache_stack = fov, stack
        return stack

    def get_processed_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None) -> np.ndarray:
        bit_index = self._bit_data_channels().index(dataChannel)
        return self._load_restored(fov)[bit_index, zIndex]

    def _run_analysis(self, fragmentIndex):
        """GPU step: restore every z-plane of this FOV (one z at a time) and
        persist the (n_bits, n_z, Y, X) stack for Decode to read."""
        import gc
        n_z = len(self.dataSet.get_z_positions())
        per_z = [self._restore_fov_z(fragmentIndex, z) for z in range(n_z)]  # each (n_bits, Y, X)
        stack = np.stack(per_z, axis=1).astype(np.float32)                   # (n_bits, n_z, Y, X)
        self.dataSet.save_numpy_analysis_result(
            stack, "restored", self.analysisName, fragmentIndex, "restored_images")
        # free per-FOV memory so it doesn't accumulate across the job's FOVs
        self._cache_key, self._cache_stack = None, None
        del per_z, stack
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
            mem = self._torch.cuda.memory_allocated() / 1e9
            print(f"[modelrestore] FOV {fragmentIndex} done; GPU mem_allocated={mem:.2f}GB", flush=True)
