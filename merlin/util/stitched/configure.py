"""Write an opt-in native MERlin analysis JSON, without submitting anything.

python -m merlin.util.stitched.configure --output analysis.json --codebook-index 0
"""
import argparse,json
from pathlib import Path

def make_pipeline(config,partition=False,report=False,prefix='Stitched'):
    if report and not partition:raise ValueError('Cell analysis requires partitioned counts')
    tasks=[];init=prefix+'Initialize'
    tasks.append(dict(task='StitchedInitialize',module='merlin.analysis.stitched',analysis_name=init,parameters=dict(configuration=config,memory_mb=8192,minutes=30)))
    def add(label,stage,dependencies,parallel=False,memory=16384,minutes=120,**extra):
        name=prefix+label
        tasks.append(dict(task='StitchedFragments' if parallel else 'StitchedStage',module='merlin.analysis.stitched',analysis_name=name,
            parameters=dict(initialize_task=init,stage=stage,dependencies=[prefix+x for x in dependencies],memory_mb=memory,minutes=minutes,**extra)))
    if config.get('reuse_registration'):
        add('Review','adopt_registration',['Initialize'],False,8192,120)
    else:
        add('Extract','extract',['Initialize'],True,16384,60)
        add('Within','within',['Extract'],True,32768,480)
        add('Pool','pool',['Within'],True,16384,240)
        add('Cross','cross',['Pool'],True,32768,240)
        add('QAWindows','qa_windows',['Cross'],False,8192,30)
        add('QA','qa',['QAWindows'],True,8192,60)
        add('Review','registration_review',['QA'],False,8192,60)
    add('CalibrationPlan','calibration_plan',['Review'],False,16384,120)
    add('CalibrationImages','calibration_images',['CalibrationPlan'],True,16384,120)
    add('CalibrationFinalize','calibration_finalize',['CalibrationImages'],False,8192,120)
    add('Optimize','optimize',['CalibrationFinalize'],False,65536,1440)
    add('DecodePlan','decode_plan',['Optimize'],False,16384,120)
    add('Decode','decode',['DecodePlan'],True,24576,2880,workers=52)
    add('RawFinalize','raw_finalize',['Decode'],False,16384,360)
    add('AdaptiveFilter','adaptive_filter',['RawFinalize'],False,24576,1440)
    add('Deduplicate','deduplicate',['AdaptiveFilter'],True,16384,360)
    add('Export','export',['Deduplicate'],False,16384,720)
    add('Quality','quality',['Export'],False,8192,120)
    if partition:
        add('PartitionSetup','partition_setup',['Export'],False,16384,120)
        add('PartitionGeometry','partition_geometry',['PartitionSetup'],True,16384,240)
        add('PartitionCoordinates','partition_coordinates',['PartitionSetup'],False,16384,240)
        add('PartitionJoin','partition_join',['PartitionGeometry','PartitionCoordinates'],True,16384,240)
        add('PartitionExport','partition_export',['PartitionJoin'],False,32768,360)
        add('PartitionVerify','partition_verify',['PartitionExport'],False,32768,240)
        add('CellCenters','cell_centers',['PartitionVerify'],False,8192,120)
    if report:add('Analysis','analysis_report',['CellCenters'],False,32768,240)
    return dict(analysis_tasks=tasks)

def cluster_resources(pipeline):
    """Per-rule Slurm requests for MERlin's existing Snakemake integration."""
    result={'__default__':dict(cpus=1,mem_mb=4096,minutes=30)}
    config=pipeline['analysis_tasks'][0]['parameters']['configuration']
    for task in pipeline['analysis_tasks']:
        p=task['parameters'];stage=p.get('stage','initialize')
        cpus=int(config.get('num_threads',2)) if stage=='decode' else 1
        if stage=='optimize':cpus=int(config.get('num_threads',2))*int(config.get('optimizer_workers',4))
        if stage in ('within','cross','calibration_images'):cpus=max(cpus,2)
        result[task['analysis_name']]=dict(cpus=cpus,mem_mb=p['memory_mb'],minutes=p['minutes'])
        if task['task']=='StitchedFragments':result[task['analysis_name']+'Done']=dict(cpus=1,mem_mb=4096,minutes=30)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--codebook-index',type=int,default=0)
    p.add_argument('--metadata-dir');p.add_argument('--raw-root');p.add_argument('--review-decision');p.add_argument('--reuse-registration')
    p.add_argument('--mask-archive');p.add_argument('--analysis-report',action='store_true');p.add_argument('--prefix',default='Stitched')
    p.add_argument('--cluster-output',help='Also write per-rule Slurm resources for Snakemake')
    a=p.parse_args()
    if not a.review_decision and not a.reuse_registration:p.error('Specify --review-decision for new registration, or --reuse-registration for an already reviewed matching sample')
    config=dict(profile='0718_40x_150z',codebook_index=a.codebook_index)
    for name in ['metadata_dir','raw_root','mask_archive','review_decision','reuse_registration']:
        if getattr(a,name):config[name]=str(Path(getattr(a,name)).resolve())
    content=make_pipeline(config,partition=bool(a.mask_archive),report=a.analysis_report,prefix=a.prefix)
    for path in [a.output,a.cluster_output]:
        if path and Path(path).exists():p.error('Refusing to replace existing configuration: '+path)
    with Path(a.output).open('x') as f:json.dump(content,f,indent=2)
    if a.cluster_output:
        with Path(a.cluster_output).open('x') as f:json.dump(cluster_resources(content),f,indent=2)

if __name__=='__main__':main()
