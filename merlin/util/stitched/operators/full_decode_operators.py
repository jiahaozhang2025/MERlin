"""Canonical component extraction with disjoint global output ownership."""
import numpy as np
from tile_layout import owner_for_global_xy,MPP

class DecodeHaloTooSmall(RuntimeError):pass

def own_components(frame,labels,tile,layout,halo,z_index):
 """Keep whole components by centroid, after rejecting context truncation.

 A truncated component that intersects the core is rejected even if its
 current centroid is outside: completing it can change its eventual owner.
 """
 h,w=map(int,tile['shape_yx'])
 if halo<2 or labels.shape!=(h+2*halo,w+2*halo):raise ValueError('Invalid component canvas/halo')
 if not frame.unique_id.is_unique:raise ValueError('Repeated component labels')
 if len(frame) and not np.isin(frame.unique_id.to_numpy(dtype=int),labels).all():raise ValueError('Missing component label')
 edge=np.unique(np.r_[labels[1,:],labels[-2,:],labels[:,1],labels[:,-2]])
 core=np.unique(labels[halo:halo+h,halo:halo+w])
 truncated=np.intersect1d(edge[edge>0],core[core>0])
 if len(truncated):raise DecodeHaloTooSmall('Core-intersecting components reach the context edge: '+str(truncated.tolist()))
 owners=owner_for_global_xy(frame[['global_x','global_y']].to_numpy(),layout)
 keep=frame.loc[owners==tile['tile_id']].copy()
 keep['output_tile']=int(tile['tile_id']);keep['fov']=int(tile['tile_id'])
 keep['decode_canvas_x']=keep.x.copy();keep['decode_canvas_y']=keep.y.copy()
 # Export native-to-output-core coordinates; source acquisition IDs are the
 # separate per-bit owner fields. Global coordinates retain float64 precision.
 keep['x']=keep.x.astype(float)-halo;keep['y']=keep.y.astype(float)-halo
 keep['barcode_uid']=[f'tile{tile["tile_id"]:04d}_z{z_index:03d}_label{int(i)}' for i in keep.unique_id]
 keep=keep.drop(columns=['pilot_barcode_id'],errors='ignore')
 if not keep.barcode_uid.is_unique:raise ValueError('Repeated stable barcode IDs')
 return keep,dict(context_components=len(frame),owned_components=len(keep),context_components_discarded=len(frame)-len(keep),
  truncated_core_components=0,decode_halo_pixels=int(halo),ownership='Centroid in unique half-open global output core; entire canonical component retained.')

def decode_with_context(produce,tile,layout,canonical_decode_arrays,codebook,Decoder,scales,backgrounds,z_index,z_um,num_threads=1,halos=(64,128,256,512,1024,2048)):
 """Produce and decode anew with larger context only when truncation requires it.

 `produce(halo)` returns image, support, source-owner rasters and provenance
 from the accepted full-sample preprocessing path. No results are committed
 until complete-component ownership succeeds.
 """
 if tile['shape_yx'][0]!=tile['shape_yx'][1]:
  raise ValueError('Canonical extraction requires the planned square output tiles')
 attempts=[]
 for halo in halos:
  image,support,owners,provenance=produce(halo)
  if owners.shape!=image.shape or owners.dtype!=np.uint16 or np.any(owners>52):raise ValueError('Invalid full-sample source ownership')
  if np.any(owners[:,support]==0):raise ValueError('Required source absent inside decode support')
  origin=np.asarray(tile['origin_xy_um'])-halo*MPP
  frame,rasters,report=canonical_decode_arrays(image,support,owners,codebook,Decoder,scales,backgrounds,z_index,z_um,origin,num_threads)
  try:owned,ownership=own_components(frame,rasters['labels'],tile,layout,halo,z_index)
  except DecodeHaloTooSmall as exc:
   attempts.append(dict(halo=int(halo),reason=str(exc)));continue
  return owned,rasters,dict(canonical_canvas_report=report,ownership=ownership,preprocessing=provenance,
   halo_retries=attempts,raw_barcodes=len(owned),raw_blank_barcodes=int(owned.is_blank.sum()),
   output_tile=int(tile['tile_id']),z_index=int(z_index),physical_z_um=float(z_um),
   source_owner_code_to_fov={'0':None,**{str(f+1):f for f in range(52)}})
 raise DecodeHaloTooSmall('No output written; still truncated at maximum context: '+str(attempts))
