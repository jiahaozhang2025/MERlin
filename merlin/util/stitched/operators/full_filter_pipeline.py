"""Disk-backed global adaptive filtering, complete-identity dedup and exports."""
import gzip,hashlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd
from full_preprocess import sha
from full_filter_operators import fit_streaming_threshold,deduplicate_identity,blank_estimate,validate_chunk

COMPRESSION=dict(method='gzip',compresslevel=1)
TEXT_DTYPES={'barcode_uid':'string','registration_uncertainty_rounds':'string','registration_uncertainty_regions':'string'}


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


class HashReader(io.RawIOBase):
    """Hash compressed source bytes during the actual CSV read, not another pass."""
    def __init__(self,path):
        self.source=Path(path).open('rb',buffering=0);self.digest=hashlib.sha256()
    def readable(self):return True
    def readinto(self,buffer):
        n=self.source.readinto(buffer)
        if n:self.digest.update(memoryview(buffer)[:n])
        return n
    def close(self):
        self.source.close();super().close()


def verified_chunks(records,chunksize=100000,columns=None):
    for record in records:
        count=0
        with HashReader(record['raw_path']) as raw:
            with io.BufferedReader(raw) as buffered,gzip.GzipFile(fileobj=buffered,mode='rb') as stream:
                with pd.read_csv(stream,chunksize=chunksize,usecols=columns,dtype=TEXT_DTYPES,float_precision='round_trip') as reader:
                    for frame in reader:
                        count+=len(frame)
                        if len(frame):yield frame
                # Read to EOF even if a parser returned immediately on a header-only table.
                while stream.read(1024*1024):pass
            digest=raw.digest.hexdigest()
        if digest!=record['raw_sha256']:raise RuntimeError('Raw barcode source changed: '+record['raw_path'])
        if count!=record['raw_barcodes']:raise RuntimeError('Raw barcode count differs from receipt')


def check_inputs(inputs):
    if any(sha(p)!=h for p,h in inputs.items()):raise RuntimeError('Protected filtering input changed')


def threshold_and_partition(manifest,codebook,canonical,output,inputs,chunksize=100000):
    output=Path(output);check_inputs(inputs)
    selected_root=output/'selected_by_identity';selected_root.mkdir(exist_ok=False)
    columns=['barcode_id','mean_intensity','min_distance','area']
    adaptive,report=fit_streaming_threshold(lambda:verified_chunks(manifest['planes'],chunksize,columns),codebook,canonical)
    if report['raw_barcodes']!=manifest['raw_barcodes'] or report['raw_blank_barcodes']!=manifest['raw_blank_barcodes']:
        raise RuntimeError('Fresh histograms disagree with complete raw manifest')
    hist=output/'histograms';hist.mkdir()
    for name,value in adaptive.dataSet.arrays.items():np.save(hist/(name+'.npy'),value,allow_pickle=False)
    selected_counts=np.zeros(codebook.get_barcode_count(),np.int64);schema=None
    for frame in verified_chunks(manifest['planes'],chunksize):
        validate_chunk(frame,codebook)
        if not frame.is_blank.eq(frame.barcode_id.isin(codebook.get_blank_indexes())).all():
            raise ValueError('Raw blank annotations disagree with codebook')
        if schema is None:schema=list(frame.columns)
        if schema!=list(frame.columns):raise ValueError('Inconsistent raw barcode schema')
        selected=adaptive.extract_barcodes_with_threshold(report['blank_fraction_threshold'],frame).copy()
        for identity,part in selected.groupby('barcode_id',sort=True):
            identity=int(identity);path=selected_root/f'barcode{identity:05d}.csv.gz'
            part.to_csv(path,index=False,mode='a',header=not path.exists(),compression=COMPRESSION)
            selected_counts[identity]+=len(part)
    if int(selected_counts.sum())!=report['predicted_selected_barcodes']:
        raise RuntimeError('Histogram threshold and actual row selection disagree')
    blanks=int(selected_counts[codebook.get_blank_indexes()].sum());coding=int(selected_counts.sum())-blanks
    achieved=blank_estimate(coding,blanks,codebook)
    if achieved is None or not np.isclose(achieved,report['estimated_rate_before_dedup'],rtol=1e-12,atol=1e-12):
        raise RuntimeError('Selected row blank estimate differs from histogram')
    partitions=[]
    for identity,count in enumerate(selected_counts):
        path=selected_root/f'barcode{identity:05d}.csv.gz'
        partitions.append(dict(barcode_id=identity,rows=int(count),is_blank=identity in set(codebook.get_blank_indexes()),
            path=str(path) if count else None,sha256=sha(path) if count else None))
    check_inputs(inputs)
    result=dict(status='THRESHOLDED_REQUIRES_GLOBAL_DEDUP',report=report,
        selected_barcodes=int(selected_counts.sum()),selected_blank_barcodes=blanks,
        estimated_rate_before_dedup=achieved,schema=schema,partitions=partitions,
        z_positions_um=manifest['z_positions_um'],input_sha256=inputs,all_hashed_inputs_unchanged=True,
        histogram_sha256={str(p):sha(p) for p in hist.glob('*.npy')},
        partition_policy='Each codeword includes all output tiles and all reference depths, preserving canonical cross-boundary duplicate chains.')
    write_json(output/'threshold.json',result)
    return result


def dedup_identity(threshold_path,identity,canonical_function,output):
    threshold_path=Path(threshold_path);threshold=json.loads(threshold_path.read_text());digest=sha(threshold_path)
    check_inputs(threshold['input_sha256'])
    if threshold['status']!='THRESHOLDED_REQUIRES_GLOBAL_DEDUP':raise ValueError('Threshold stage incomplete')
    part=next(p for p in threshold['partitions'] if p['barcode_id']==identity)
    if part['rows']:
        if sha(part['path'])!=part['sha256']:raise RuntimeError('Selected barcode partition changed')
        frame=pd.read_csv(part['path'],dtype=TEXT_DTYPES,float_precision='round_trip')
        if len(frame)!=part['rows'] or not (frame.barcode_id==identity).all():raise ValueError('Incomplete codeword partition')
        kept,removed=deduplicate_identity(frame,threshold['z_positions_um'],canonical_function)
        if sha(part['path'])!=part['sha256']:raise RuntimeError('Partition changed during dedup')
    else:
        kept=pd.DataFrame(columns=threshold['schema']);removed=kept.copy()
    if len(kept)+len(removed)!=part['rows']:raise RuntimeError('Dedup row conservation failed')
    folder=Path(output)/f'barcode{identity:05d}';folder.mkdir(parents=True,exist_ok=False)
    kept.to_csv(folder/'kept.csv.gz',index=False,compression=COMPRESSION)
    removed.to_csv(folder/'removed.csv.gz',index=False,compression=COMPRESSION)
    check_inputs(threshold['input_sha256'])
    if sha(threshold_path)!=digest:raise RuntimeError('Threshold changed during dedup')
    receipt=dict(status='COMPLETE',barcode_id=identity,is_blank=part['is_blank'],threshold_sha256=digest,
        selected_rows=part['rows'],kept_rows=len(kept),removed_rows=len(removed),
        input_partition_sha256=part['sha256'],output_sha256={n:sha(folder/n) for n in ['kept.csv.gz','removed.csv.gz']},
        z_dedup_parameters=dict(z_plane_index_threshold=2,xy_pixel_threshold=1.4),all_hashed_inputs_unchanged=True)
    write_json(folder/'complete.json',receipt)
    return receipt


def merge_results(threshold_path,dedup_root,codebook,output,chunksize=100000):
    threshold_path=Path(threshold_path);threshold=json.loads(threshold_path.read_text());digest=sha(threshold_path)
    check_inputs(threshold['input_sha256'])
    partitions=threshold['partitions']
    if [p['barcode_id'] for p in partitions]!=list(range(codebook.get_barcode_count())):
        raise RuntimeError('Threshold does not account for every codeword')
    receipts=[];receipt_hashes={}
    # Check every identity receipt before creating any merged export.
    for part in partitions:
        folder=Path(dedup_root)/f'barcode{part["barcode_id"]:05d}';path=folder/'complete.json'
        receipt=json.loads(path.read_text());receipt_hashes[str(path)]=sha(path)
        if receipt['status']!='COMPLETE' or not receipt['all_hashed_inputs_unchanged'] or receipt['threshold_sha256']!=digest or receipt['barcode_id']!=part['barcode_id'] or receipt['selected_rows']!=part['rows'] or receipt['input_partition_sha256']!=part['sha256'] or receipt['is_blank']!=part['is_blank']:
            raise ValueError('Dedup receipt does not match complete codeword partition')
        if receipt['kept_rows']+receipt['removed_rows']!=part['rows']:raise ValueError('Dedup count conservation failed')
        receipts.append((folder,receipt))
    output=Path(output);output.mkdir(exist_ok=False)
    totals={'kept':0,'removed':0};blank_kept=0;summary=[];spatial={}
    for kind in ['kept','removed']:
        name='adaptive_filtered_z_deduplicated.csv.gz' if kind=='kept' else 'removed_z_duplicates.csv.gz'
        with gzip.open(output/name,'wt',compresslevel=1,newline='') as target:
            pd.DataFrame(columns=threshold['schema']).to_csv(target,index=False)
            for folder,receipt in receipts:
                source=folder/(kind+'.csv.gz');count=0
                record=dict(raw_path=str(source),raw_sha256=receipt['output_sha256'][kind+'.csv.gz'],raw_barcodes=receipt[kind+'_rows'])
                for frame in verified_chunks([record],chunksize):
                    if list(frame.columns)!=threshold['schema'] or not (frame.barcode_id==receipt['barcode_id']).all():raise ValueError('Merged output schema/identity mismatch')
                    frame.to_csv(target,index=False,header=False);count+=len(frame)
                    if kind=='kept':
                        for (tile,z),group in frame.groupby(['output_tile','z'],sort=False):
                            key=(int(tile),int(z))
                            entry=spatial.setdefault(key,dict(output_tile=key[0],z_index=key[1],
                                physical_z_um=threshold['z_positions_um'][key[1]],barcodes=0,blank_barcodes=0,
                                mixed_source_barcodes=0,source_seam_barcodes=0,coverage_boundary_barcodes=0))
                            entry['barcodes']+=len(group);entry['blank_barcodes']+=len(group) if receipt['is_blank'] else 0
                            for column,total in [('mixed_source_across_bits','mixed_source_barcodes'),
                                ('source_seam_within_2px','source_seam_barcodes'),('coverage_boundary_within_2px','coverage_boundary_barcodes')]:
                                if column in group:entry[total]+=int(group[column].sum())
                totals[kind]+=count
                if kind=='kept':
                    blank_kept+=count if receipt['is_blank'] else 0
                    summary.append(dict(barcode_id=receipt['barcode_id'],name=codebook.get_name_for_barcode_index(receipt['barcode_id']),is_blank=receipt['is_blank'],
                        selected=receipt['selected_rows'],kept=count,removed=receipt['removed_rows']))
    if totals['kept']+totals['removed']!=threshold['selected_barcodes']:raise RuntimeError('Merged counts disagree with adaptive selection')
    check_inputs({**threshold['input_sha256'],**receipt_hashes,str(threshold_path):digest})
    pd.DataFrame(summary).to_csv(output/'counts_by_codeword.csv',index=False)
    spatial_columns=['output_tile','z_index','physical_z_um','barcodes','blank_barcodes',
        'mixed_source_barcodes','source_seam_barcodes','coverage_boundary_barcodes']
    spatial_frame=pd.DataFrame([spatial[k] for k in sorted(spatial)],columns=spatial_columns)
    if spatial_frame.barcodes.sum()!=totals['kept'] or spatial_frame.blank_barcodes.sum()!=blank_kept:
        raise RuntimeError('Spatial summary does not conserve final barcode counts')
    spatial_frame.to_csv(output/'counts_by_tile_and_depth.csv',index=False)
    codebook.get_data().to_csv(output/'codebook.csv',index=False)
    rate=blank_estimate(totals['kept']-blank_kept,blank_kept,codebook)
    post_pass=rate is not None and rate<=.05+1e-12
    result=dict(status='COMPLETE' if threshold['report']['target_met'] and post_pass else 'COMPLETE_WITH_BLANK_ESTIMATE_LIMITATION',
        adaptive_target=.05,estimated_rate_before_dedup=threshold['estimated_rate_before_dedup'],
        estimated_rate_after_dedup=rate,adaptive_target_met=threshold['report']['target_met'],post_dedup_target_met=post_pass,
        raw_barcodes=threshold['report']['raw_barcodes'],thresholded_barcodes=threshold['selected_barcodes'],
        deduplicated_barcodes=totals['kept'],deduplicated_blank_barcodes=blank_kept,removed_z_duplicates=totals['removed'],
        threshold_sha256=digest,identity_receipt_sha256=receipt_hashes,all_hashed_inputs_unchanged=True,
        output_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        limitations='Blank estimates are codebook-normalized blank/coding count ratios, not independently measured molecular error rates. Threshold is fitted before deduplication; the post-dedup rate is reported without retuning. Unsupported registration and acquired-support coverage require the accompanying decode/registration reports.')
    write_json(output/'complete.json',result)
    return result
