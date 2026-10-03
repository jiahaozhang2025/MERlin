"""Memory-bounded adapters around canonical adaptive and cross-Z operators.

Input completeness, hashes and output receipts belong to the eventual runner.
These functions do not authorize a run or load any previous fitted threshold.
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd

PARAMETERS=dict(decode_task='FreshDecode',run_after_task='FreshDecode',tolerance=.001,
 intensity_bins=199,distance_bins=66,area_bins=33,threshold_solver_method='cumulative_bins',
 intensity_transform='log10',overshoot_toward_target=False,overshoot_tolerance=.20,
 report_bracketing_thresholds=True)

class HistogramStore:
 def __init__(self,codebook):
  self.arrays={};self.json={};self.decode=SimpleNamespace(get_codebook=lambda:codebook)
 def load_analysis_task(self,name):
  if name!='FreshDecode':raise ValueError(name)
  return self.decode
 def load_numpy_analysis_result(self,name,task,resultIndex=None):return self.arrays[name]
 def save_json_analysis_result(self,value,name,task):self.json[name]=value

def validate_chunk(frame,codebook):
 for name in ('mean_intensity','min_distance','area','barcode_id'):
  if name not in frame or not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):raise ValueError('Non-numeric '+name)
  if not np.isfinite(frame[name]).all():raise ValueError('Nonfinite '+name)
 if (frame.mean_intensity<=0).any() or (frame.area<4).any():raise ValueError('Intensity/area violates canonical decode contract')
 for name in ('area','barcode_id'):
  if not np.equal(frame[name],np.floor(frame[name])).all():raise ValueError('Noninteger '+name)
 if not frame.barcode_id.isin(range(codebook.get_barcode_count())).all():raise ValueError('Unknown codeword')

def fit_streaming_threshold(chunks,codebook,canonical):
 """Two passes; exactly the pilot's one-global-FOV histogram and solver.

 `chunks` must return a new iterator over the same frozen rows on each call.
 The runner must hash its inputs; row/count checks here are additional checks.
 """
 if not len(codebook.get_blank_indexes()) or not len(codebook.get_coding_indexes()):raise ValueError('Both blank and coding codewords are required')
 count=0;maximum=0.;identity_counts=np.zeros(codebook.get_barcode_count(),np.int64)
 for frame in chunks():
  validate_chunk(frame,codebook);count+=len(frame)
  if len(frame):maximum=max(maximum,float(frame.mean_intensity.max()))
  identity_counts+=np.bincount(frame.barcode_id.to_numpy(dtype=int),minlength=len(identity_counts))
 if not count:raise ValueError('No barcodes for a fresh threshold')
 store=HistogramStore(codebook);task=object.__new__(canonical.GenerateAdaptiveThreshold)
 task.dataSet=store;task.analysisName='FreshAdaptiveThreshold';task.parameters=PARAMETERS.copy()
 bins=[canonical._build_intensity_bins(maximum,199,'log10'),np.linspace(0,.66,67),np.arange(1,35)]
 for name,value in zip(['intensity_bins','distance_bins','area_bins'],bins):store.arrays[name]=value
 blank=np.zeros((199,66,33));coding=np.zeros_like(blank);second=np.zeros_like(identity_counts)
 for frame in chunks():
  validate_chunk(frame,codebook)
  second+=np.bincount(frame.barcode_id.to_numpy(dtype=int),minlength=len(second))
  blank+=task._extract_counts(frame.loc[frame.barcode_id.isin(codebook.get_blank_indexes())],*bins)
  coding+=task._extract_counts(frame.loc[frame.barcode_id.isin(codebook.get_coding_indexes())],*bins)
 if not np.array_equal(second,identity_counts) or int(blank.sum()+coding.sum())!=count:raise RuntimeError('Streaming histogram row conservation failed')
 store.arrays.update(blank_counts=blank,coding_counts=coding)
 threshold=float(task.calculate_threshold_for_misidentification_rate(.05))
 if not np.isfinite(threshold):raise RuntimeError('Insufficient evidence for finite adaptive threshold')
 predicted=int(task.calculate_barcode_count_for_threshold(threshold))
 rate=float(task.calculate_misidentification_rate_for_threshold(threshold))
 return task,dict(raw_barcodes=count,raw_blank_barcodes=int(blank.sum()),raw_coding_barcodes=int(coding.sum()),
  blank_fraction_threshold=threshold,predicted_selected_barcodes=predicted,estimated_rate_before_dedup=rate,
  target_met=bool(np.isfinite(rate) and rate<=.05+1e-12),histogram_count_conservation_passed=True,
  histogram_scope='All completed output tiles and reference depths treated as one stitched field; global maximum intensity, unchanged canonical bins/counts/solver.')

def deduplicate_identity(frame,z_positions,canonical_function,mpp=.1493):
 """Call the unmodified canonical operator across ALL tiles for one identity.

 Translate global microns into one common pixel lattice for distance checks;
 return original row coordinates and annotations unchanged. Never split an
 identity by output tile or depth: canonical chains can cross both boundaries.
 """
 if frame.empty:return frame.copy(),frame.copy()
 if frame.barcode_id.nunique()!=1:raise ValueError('Exactly one complete barcode identity is required')
 if not frame.barcode_uid.is_unique:raise ValueError('Duplicate stable row IDs')
 for name in ('global_x','global_y','z','mean_intensity'):
  if not np.isfinite(frame[name]).all():raise ValueError('Nonfinite '+name)
 if (frame.z<0).any() or (frame.z>=len(z_positions)).any() or not np.equal(frame.z,np.floor(frame.z)).all():raise ValueError('Invalid Z index')
 if not np.all(np.diff(z_positions)>0):raise ValueError('Reference depths must increase')
 native=frame.copy(deep=True)
 native['x']=native.global_x.to_numpy(dtype=float)/mpp
 native['y']=native.global_y.to_numpy(dtype=float)/mpp
 kept=canonical_function(native,2,1.4,z_positions)
 selected=frame.barcode_uid.isin(kept.barcode_uid)
 if selected.sum()!=len(kept):raise RuntimeError('Canonical dedup returned unknown or repeated row IDs')
 return frame.loc[selected].copy(),frame.loc[~selected].copy()

def blank_estimate(coding,blank,codebook):
 if coding==0:return None
 return (blank/len(codebook.get_blank_indexes()))/(coding/len(codebook.get_coding_indexes()))
