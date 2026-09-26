"""Compute-node CLI for full-sample threshold, identity dedup and merged export."""
import argparse,json,os,sys,tempfile,time
from pathlib import Path
from full_preprocess import sha
from full_filter_pipeline import check_inputs,threshold_and_partition,dedup_identity,merge_results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['threshold','dedup','merge'])
    parser.add_argument('--identity',type=int);args=parser.parse_args()
    if (args.stage=='dedup')!=(args.identity is not None):raise ValueError('Identity is required only for dedup')
    root=Path(__file__).resolve().parent.parent;decode=root/'full_decode_v1'
    plan=json.loads((decode/'plan.json').read_text());manifest_path=decode/'raw_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='RAW_DECODE_COMPLETE_REQUIRES_FRESH_ADAPTIVE_FILTER' or not manifest['all_hashed_inputs_unchanged']:
        raise RuntimeError('All full-sample decode receipts are required')
    inputs={**manifest['input_sha256'],str(manifest_path):sha(manifest_path)}
    sources=[Path(__file__).with_name(n) for n in ['full_filter_run.py','full_filter_pipeline.py','full_filter_operators.py']]
    inputs.update({str(p):sha(p) for p in sources});check_inputs(inputs)
    output=root/'full_filter_v1'
    if args.stage=='threshold':
        check_inputs(manifest['receipt_sha256']);output.mkdir(exist_ok=False)
    else:
        threshold=json.loads((output/'threshold.json').read_text())
        if threshold['input_sha256']!=inputs:raise RuntimeError('Filtering inputs or sources changed since threshold')
    runtime=output/f'runtime_{args.stage}_{args.identity}_{os.getpid()}_{time.time_ns()}';runtime.mkdir()
    for name in ['tmp','cache']:(runtime/name).mkdir()
    os.environ.update(TMPDIR=str(runtime/'tmp'),TEMP=str(runtime/'tmp'),TMP=str(runtime/'tmp'),
        XDG_CACHE_HOME=str(runtime/'cache'),MPLCONFIGDIR=str(runtime/'cache/matplotlib'),
        NUMBA_CACHE_DIR=str(runtime/'cache/numba'),PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True;sys.path.insert(0,plan['fresh_adapters'])
    from fresh_optimize import load_canonical,_install_write_guard
    _install_write_guard(output)
    # The shared guard resets tempfile's cached directory to output/tmp.
    # Keep temporary library files in this worker's existing runtime directory.
    tempfile.tempdir=str(runtime/'tmp')
    Codebook,sink,_,_=load_canonical(Path(plan['fork']),runtime)
    from merlin.analysis import filterbarcodes
    from merlin.util import barcodefilters
    for module in [filterbarcodes,barcodefilters]:
        path=Path(module.__file__).resolve()
        if inputs.get(str(path))!=sha(path):raise RuntimeError('Canonical filtering source differs from frozen full run')
    book=Codebook(sink,manifest['codebook_path'],codebookIndex=0,codebookName='FreshFullStitched0718')
    if list(book.get_bit_names())!=manifest['bit_names']:raise RuntimeError('Filtering bit order mismatch')
    if args.stage=='threshold':
        report=threshold_and_partition(manifest,book,filterbarcodes,output,inputs)
        print(json.dumps(dict(status=report['status'],selected_barcodes=report['selected_barcodes'],rate=report['estimated_rate_before_dedup'])))
    elif args.stage=='dedup':
        report=dedup_identity(output/'threshold.json',args.identity,barcodefilters.remove_zplane_duplicates_single_barcodeid,output/'dedup_by_identity')
        print(json.dumps(report))
    else:
        report=merge_results(output/'threshold.json',output/'dedup_by_identity',book,output/'exports')
        print(json.dumps({k:report[k] for k in ['status','deduplicated_barcodes','estimated_rate_after_dedup']}))
if __name__=='__main__':main()
