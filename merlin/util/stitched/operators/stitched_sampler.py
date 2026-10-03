"""Single-sampling native XYZ -> within-round mosaic -> reference mosaic.

Acquired support is tracked independently of intensity. The output-to-input
composition evaluates the new cross-round map inverse and new mosaic inverse,
then samples original native raw planes once. No production fitted task is read.
"""
import csv,hashlib,json,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import global_quadratic as gq
from bead_cloud import numbers

FOVS=(13,14,25,26)
MPP=.1493

def pixel_xy(native_xyz):
 xy=np.asarray(native_xyz)[...,:2]/MPP
 # Remove floating point roundoff at exact pixel centers only; this does not
 # admit any measurable out-of-camera support or pad missing acquisitions.
 xy=np.where(np.abs(xy)<1e-9,0.,xy)
 return np.where(np.abs(xy-2047)<1e-9,2047.,xy)

class NativeAcquisition:
 def __init__(self,data_dir,raw_root):
  self.data_dir=Path(data_dir);self.raw_root=Path(raw_root)
  with (self.data_dir/'dataorganization.csv').open() as f:self.organization=list(csv.DictReader(f))
  with (self.data_dir/'filemap.csv').open() as f:self.filemap=list(csv.DictReader(f))
  self.microscope=json.loads((self.data_dir/'microscope_parameters.json').read_text())
  if self.microscope['microns_per_pixel']!=MPP:raise ValueError('Unexpected pixel calibration')
  self.movies={};self.schedules={};self.schedule_provenance={};self.raw_stats={}
 def bit_row(self,bit):
  return next(r for r in self.organization if r['readoutName']==bit)
 def movie_path(self,row,fov):
  entry=next(r for r in self.filemap if int(r['fov'])==fov and int(r['imagingRound'])==int(row['imagingRound']) and r['imageType']==row['imageType'])
  return self.raw_root/Path(entry['imagePath']).name
 def frame_grid(self,row,fov):
  path=self.movie_path(row,fov);frames=numbers(row['frame']).astype(int)
  if path not in self.schedules:
   xml=path.with_suffix('.xml');doc=ET.parse(xml)
   values=[e.text for e in doc.iter() if e.tag=='z_offsets']
   if len(values)!=1:raise ValueError(f'Expected one acquisition Z schedule: {xml}')
   zs=np.array([float(x) for x in values[0].split(',')])
   h,w=self.microscope['image_dimensions']
   if len(zs)!=path.stat().st_size//(h*w*2):raise ValueError('XML andDAXframecounts differ')
   self.schedules[path]=zs
   self.schedule_provenance[str(xml)]={'sha256':hashlib.sha256(xml.read_bytes()).hexdigest(),
    'frame_count':len(zs),'interpretation':'Recorded targetZschedule; not independent encoder measurement'}
  zs=self.schedules[path][frames]
  if len(zs)<2 or np.any(np.diff(zs)<=0):raise ValueError('Signal physicalZschedule must be increasing')
  return frames,zs
 def movie(self,row,fov):
  path=self.movie_path(row,fov)
  if path not in self.movies:
   h,w=self.microscope['image_dimensions'];framebytes=h*w*2
   if path.stat().st_size%framebytes:raise ValueError('Incomplete raw movie')
   self.raw_stats[path]=(path.stat().st_size,path.stat().st_mtime_ns)
   self.movies[path]=np.memmap(path,dtype='<u2',mode='r',shape=(path.stat().st_size//framebytes,h,w))
  return self.movies[path]
 def plane(self,row,fov,frame):
  p=self.movie(row,fov)[int(frame)]
  if self.microscope.get('transpose'):p=p.T
  if self.microscope.get('flip_horizontal'):p=p[:,::-1]
  if self.microscope.get('flip_vertical'):p=p[::-1,:]
  return p
 def sample(self,row,fov,native_xyz):
  """Linear XY and physical-Z sampling; never declares clipped depths valid."""
  shape=native_xyz.shape[:-1];p=np.asarray(native_xyz).reshape(-1,3)
  frames,zs=self.frame_grid(row,fov)
  if len(frames)!=len(zs) or np.any(np.diff(zs)<=0):raise ValueError('Bad signal frame/Z ordering')
  xy=pixel_xy(p)
  valid=np.isfinite(p).all(1)&(xy[:,0]>=0)&(xy[:,0]<=2047)&(xy[:,1]>=0)&(xy[:,1]<=2047)&(p[:,2]>=zs[0])&(p[:,2]<=zs[-1])
  output=np.zeros(len(p),np.float32)
  if np.any(valid):
   inds=np.flatnonzero(valid); pp=p[valid]; coords=xy[valid]
   upper=np.searchsorted(zs,pp[:,2],side='right');upper=np.clip(upper,1,len(zs)-1);lower=upper-1
   fraction=(pp[:,2]-zs[lower])/(zs[upper]-zs[lower])
   vals=np.zeros(len(pp),np.float32)
   for indexes,weights in [(lower,1-fraction),(upper,fraction)]:
    for zi in np.unique(indexes[weights>0]):
     keep=(indexes==zi)&(weights>0)
     plane=self.plane(row,fov,frames[zi])
     values=ndimage.map_coordinates(plane.astype(np.float32),[coords[keep,1],coords[keep,0]],order=1,mode='constant',cval=0,prefilter=False)
     vals[keep]+=values*weights[keep]
   output[inds]=vals
  return output.reshape(shape),valid.reshape(shape)

def reference_grid(origin_xy_um,z_um,shape,halo=0):
 yy,xx=np.mgrid[-halo:shape[0]+halo,-halo:shape[1]+halo]
 return np.stack([origin_xy_um[0]+xx*MPP,origin_xy_um[1]+yy*MPP,np.full(xx.shape,z_um)],axis=-1)

def native_maps(global_xyz,within_model,cross_model=None):
 if cross_model is None:moving=np.asarray(global_xyz)
 else:
  import cross_round
  moving=cross_round.inverse_cross(global_xyz,cross_model)
 return {f:gq.inverse_global(moving,f,within_model) for f in FOVS}

def stitch_bit(acquisition,row,maps):
 """Select one acquired source by distance to the native XY camera edge.

Same geometry gives same source for both colors of a round; masks remain
bit-specific when physical Z grids differ. No intensity-weighted selection.
"""
 shape=next(iter(maps.values())).shape[:-1]
 image=np.zeros(shape,np.float32);score=np.full(shape,-np.inf,np.float32);owner=np.zeros(shape,np.uint8)
 for i,f in enumerate(FOVS,1):
  _,zs=acquisition.frame_grid(row,f)
  p=maps[f];xy=pixel_xy(p);x,y=xy[...,0],xy[...,1]
  valid=np.isfinite(p).all(-1)&(x>=0)&(x<=2047)&(y>=0)&(y<=2047)&(p[...,2]>=zs[0])&(p[...,2]<=zs[-1])
  edge=np.minimum.reduce([x,y,2047-x,2047-y]);take=valid&(edge>score)
  score[take]=edge[take];owner[take]=i
 for i,f in enumerate(FOVS,1):
  take=owner==i
  # Decide final ownership before sampling; avoid reading overwritten sources.
  if np.any(take):
   values,ok=acquisition.sample(row,f,maps[f][take]);assert ok.all()
   image[take]=values
 return image,owner>0,owner

def canonical_preprocessor(fork):
 sys.path.insert(0,str(fork))
 from merlin.analysis.preprocess import DeconvolutionPreprocess
 p=object.__new__(DeconvolutionPreprocess)
 p.parameters=dict(highpass_sigma=3,fft_highpass_sigma=0,lowpass_sigma=1,decon_sigma=2,decon_iterations=5,
  decon_filter_size=9,deconvolve_after_highpass=True,threshold_subtract_n=0.,threshold_subtract_mode='none',lowpass_after_deconvolution=False,decon_method='lucyrichardson',fft_highpass_clip=True)
 p._highPassSigma=3;p._fftHighPassSigma=0;p._fftTransfer=None;p._lowPassSigma=1
 p._deconSigma=2;p._deconIterations=5;p._fftHighPassClip=True;p._thresholdSubtractN=0.;p._thresholdSubtractMode='none'
 return p

def preprocess_plane(acquisition,bit_names,round_models,z_um,origin_xy_um,shape,fork,halo=64,progress=False):
 """Stitch registered raw images first, then canonical preprocessing."""
 grid=reference_grid(origin_xy_um,z_um,shape,halo);pre=canonical_preprocessor(fork);started=time.time()
 result=np.empty((len(bit_names),*shape),np.float32);complete=np.ones(grid.shape[:2],bool)
 owners=[];round_current=None;maps=None
 order=sorted(enumerate(bit_names),key=lambda item:int(acquisition.bit_row(item[1])['imagingRound']))
 for bi,bit in order:
  row=acquisition.bit_row(bit);r=int(row['imagingRound'])
  if r!=round_current:
   entry=round_models[r];maps=native_maps(grid,entry['within'],entry.get('cross'));round_current=r
  raw,valid,owner=stitch_bit(acquisition,row,maps)
  processed=pre._preprocess_image(raw)
  result[bi]=processed[halo:halo+shape[0],halo:halo+shape[1]]
  complete &= valid
  owners.append((bi,owner[halo:halo+shape[0],halo:halo+shape[1]].copy()))
  if progress:print(json.dumps(dict(bit=bit,round=r,z_um=z_um,raw_acquired_fraction=float(valid.mean()),elapsed_seconds=time.time()-started)),flush=True)
 # Include the complete canonical filtering footprint and a safety margin.
 # Padding invalidates array edges, but output starts64px inside this context.
 # Square kernels compose to a square footprint: Euclidean distance would
 # wrongly admit a diagonally adjacent missing pixel inside that footprint.
 safe=ndimage.minimum_filter(complete,size=2*halo+1,mode='constant',cval=0)
 supported=safe[halo:halo+shape[0],halo:halo+shape[1]]
 result[:,~supported]=0
 owner_stack=np.stack([x[1] for x in sorted(owners)])
 return result,supported,owner_stack
