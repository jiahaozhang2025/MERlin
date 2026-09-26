"""Verify all frozen tile/depth outputs before any global adaptive filtering."""
import json
from pathlib import Path
from full_preprocess import sha
from full_decode_receipts import make_tasks,verify_plane


def assemble(plan_path):
    path=Path(plan_path).resolve();root=path.parent;plan=json.loads(path.read_text());digest=sha(path)
    inputs={**plan['protected_input_sha256'],str(path):digest}
    if any(sha(p)!=h for p,h in inputs.items()):raise RuntimeError('Planned input changed')
    # Verify exhaustive coverage independently of successful task receipts.
    if plan['tasks']!=make_tasks(plan['layout'],plan['z_positions_um']):
        raise ValueError('Plan does not contain exactly all output tiles and depths')
    tiles={t['tile_id']:t for t in plan['layout']['tiles']};records=[];receipt_hashes={}
    for task in plan['tasks']:
        folder=root/f'task{task["task_id"]:04d}';receipt_path=folder/'complete.json'
        receipt=json.loads(receipt_path.read_text());receipt_hashes[str(receipt_path)]=sha(receipt_path)
        if receipt['status']!='COMPLETE' or receipt['task']!=task or receipt['plan_sha256']!=digest or not receipt['all_hashed_inputs_unchanged']:
            raise ValueError('Incomplete or mismatched decode task')
        actual=[]
        for plane in task['planes']:
            output=folder/f'z{plane["z_index"]:03d}';tile=tiles[task['tile_id']]
            rec=verify_plane(output,tile,plane,plan['layout'],digest,inputs,check_inputs=False)
            phash=sha(output/'complete.json');receipt_hashes[str(output/'complete.json')]=phash
            actual.append(dict(plane=plane,receipt_sha256=phash,raw_barcodes=rec['raw_barcodes'],supported_core_pixels=rec['supported_core_pixels']))
            records.append(dict(tile_id=task['tile_id'],**plane,raw_path=str(output/'raw.csv.gz'),
                raw_sha256=rec['output_sha256']['raw.csv.gz'],receipt_path=str(output/'complete.json'),receipt_sha256=phash,
                core_support_path=str(output/'core_support.npz'),core_support_sha256=rec['output_sha256']['core_support.npz'],
                raw_barcodes=rec['raw_barcodes'],raw_blank_barcodes=rec['raw_blank_barcodes'],
                supported_core_pixels=rec['supported_core_pixels'],core_pixels=rec['core_pixels']))
        if actual!=receipt['planes']:raise ValueError('Task and plane receipts disagree')
    if any(sha(p)!=h for p,h in {**inputs,**receipt_hashes}.items()):raise RuntimeError('Input/receipt changed during full audit')
    return dict(status='RAW_DECODE_COMPLETE_REQUIRES_FRESH_ADAPTIVE_FILTER',bit_names=plan['bit_names'],
        codebook_path=plan['codebook_path'],z_positions_um=plan['z_positions_um'],layout=plan['layout'],planes=records,
        input_sha256=inputs,receipt_sha256=receipt_hashes,all_hashed_inputs_unchanged=True,
        raw_barcodes=sum(r['raw_barcodes'] for r in records),raw_blank_barcodes=sum(r['raw_blank_barcodes'] for r in records),
        supported_core_pixel_planes=sum(r['supported_core_pixels'] for r in records),
        rectangular_core_pixel_planes=sum(r['core_pixels'] for r in records),
        empty_support_planes=sum(r['supported_core_pixels']==0 for r in records),
        limitations='Counts are before adaptive filtering and cross-Z deduplication. Rectangular coverage includes empty space outside tissue and fields; it is not the fraction of imaged tissue successfully decoded.')


def main():
    root=Path(__file__).resolve().parent.parent/'full_decode_v1';output=root/'raw_manifest.json'
    if output.exists():raise FileExistsError(output)
    result=assemble(root/'plan.json');output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ['status','raw_barcodes','raw_blank_barcodes','empty_support_planes']}))
if __name__=='__main__':main()
