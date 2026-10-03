"""Plan fixed calibration fragments from accepted full-sample geometry only."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from full_preprocess import load_registration,AuditedAcquisition,reference_to_moving,restrict_model,sha
import full_sampling_radial as sampling
from tile_layout import build_layout,MPP
from calibration_selection import select_planes

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent;registration_path=Path(args.registration).resolve()
    if not registration_path.is_relative_to(root):raise ValueError('Registration must belong to this isolated run')
    accepted,models,protected=load_registration(registration_path)
    inventory=json.loads((root/'inventory.json').read_text())
    if len(inventory['codebook_indices'])!=1:raise ValueError('One explicit codebook per stitched run required')
    for name,digest in inventory['metadata_sha256'].items():
        path=root/'inputs'/name
        if sha(path)!=digest:raise RuntimeError('Frozen metadata changed: '+name)
        protected[str(path)]=digest
    codebook=next((root/'inputs').glob(f"codebook_{inventory['codebook_indices'][0]}_*.csv"))
    with codebook.open() as f:header=next(csv.reader(f))
    if header[:2]!=['name','id']:raise ValueError('Unexpected codebook schema')
    bits=header[2:];acq=AuditedAcquisition(root,inventory);layout=build_layout(models[0]['within'])
    limits={};rows=[acq.bit_row(b) for b in bits]
    for bi,row in enumerate(rows):
        r=int(row['imagingRound']);limits[bi]={}
        for f in map(int,models[r]['within']['models']):
            _,zs=acq.frame_grid(row,f);limits[bi][f]=(zs[0],zs[-1])
    records=[]
    for tile in layout['tiles']:
        if not tile['nominal_source_fovs']:continue
        h,w=tile['shape_yx'];x=np.linspace(64,w-65,12);y=np.linspace(64,h-65,12)
        yy,xx=np.meshgrid(y,x,indexing='ij')
        for z in [10.,35.,60.,85.,110.,135.]:
            grid=np.stack([xx*MPP+tile['origin_xy_um'][0],yy*MPP+tile['origin_xy_um'][1],np.full_like(xx,z)],axis=-1)
            available=np.ones(xx.shape,bool);current=None
            for bi,row in enumerate(rows):
                r=int(row['imagingRound'])
                if r!=current:
                    moving=reference_to_moving(grid,models[r]['cross']);within=restrict_model(models[r]['within'],moving);current=r
                owner,_=sampling.choose_sources(moving,within,limits[bi]);available&=owner>0
                if not available.any():break
            records.append(dict(tile_id=tile['tile_id'],z_index=int(z)-1,physical_z_um=z,
                center_xy_um=(np.asarray(tile['origin_xy_um'])+np.array([w,h])*MPP/2).tolist(),coarse_all_bit_fraction=float(available.mean())))
    selected=select_planes(records)
    if any(sha(p)!=h for p,h in protected.items()):raise RuntimeError('Planning input changed')
    audit=acq.audit();out=root/'calibration_inputs_v1';out.mkdir(exist_ok=False)
    report=dict(status='PLANNED_FROM_ACCEPTED_REGISTRATION',registration_path=str(registration_path),bit_names=bits,codebook_path=str(codebook),
        layout=layout,planes=selected,candidates=records,protected_input_sha256=protected,raw_audit=audit,
        selection='12x12 geometric acquired-support probe at six fixed depths; fraction>=0.1; deterministic normalized XYZ farthest-point sample,12validation+50training. No image intensity or decoded counts used.',
        limitations='Coarse support only guides calibration sampling; exact masks come from preprocessing. Validation shares the same acquisition and can share molecules across adjacent output tiles. These are calibration fragments, not decoding coverage estimates.')
    (out/'plan.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(training=50,validation=12,candidates=len(records),output=str(out))))
if __name__=='__main__':main()
