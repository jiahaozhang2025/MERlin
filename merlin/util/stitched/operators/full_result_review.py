"""Reviewable final coverage/counts/blank report; never changes decoded results."""
import csv,hashlib,json,math
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def is_blank(value):
    if value is True or value=='True':return True
    if value is False or value=='False':return False
    raise ValueError('Invalid blank identity flag')


def summarize(raw,export,counts,spatial):
    """Reconcile independent summaries before plotting or publishing."""
    if raw['status']!='RAW_DECODE_COMPLETE_REQUIRES_FRESH_ADAPTIVE_FILTER' or not raw['all_hashed_inputs_unchanged']:
        raise ValueError('Raw decode audit incomplete')
    if export['status'] not in ('COMPLETE','COMPLETE_WITH_BLANK_ESTIMATE_LIMITATION') or not export['all_hashed_inputs_unchanged']:
        raise ValueError('Merged result incomplete')
    ids=[int(x['barcode_id']) for x in counts]
    if ids!=list(range(len(ids))):raise ValueError('Codeword inventory is incomplete')
    kept=sum(int(x['kept']) for x in counts);removed=sum(int(x['removed']) for x in counts)
    selected=sum(int(x['selected']) for x in counts)
    blanks=sum(int(x['kept']) for x in counts if is_blank(x['is_blank']))
    if any(int(x['kept'])+int(x['removed'])!=int(x['selected']) for x in counts):raise ValueError('Codeword count conservation failed')
    if (kept!=export['deduplicated_barcodes'] or removed!=export['removed_z_duplicates']
            or selected!=export['thresholded_barcodes'] or blanks!=export['deduplicated_blank_barcodes']
            or export['raw_barcodes']!=raw['raw_barcodes'] or selected>raw['raw_barcodes']):
        raise ValueError('Merged and codeword summaries disagree')
    blank_words=sum(is_blank(x['is_blank']) for x in counts);coding_words=len(counts)-blank_words
    rate=(blanks/blank_words)/((kept-blanks)/coding_words) if blank_words and coding_words and kept>blanks else None
    expected=export['estimated_rate_after_dedup']
    if (rate is None)!=(expected is None) or (rate is not None and not math.isclose(rate,expected,rel_tol=1e-12,abs_tol=1e-12)):
        raise ValueError('Actual post-dedup blank estimate disagrees')
    tiles={int(x['tile_id']):dict(tile_id=int(x['tile_id']),grid_row=x['grid_row'],grid_column=x['grid_column'],
        origin_xy_um=x['origin_xy_um'],supported_pixel_planes=0,rectangular_pixel_planes=0,raw_barcodes=0,
        final_barcodes=0,final_blank_barcodes=0) for x in raw['layout']['tiles']}
    depths=[dict(z_index=i,physical_z_um=float(z),supported_pixels=0,rectangular_pixels=0,
        raw_barcodes=0,final_barcodes=0,final_blank_barcodes=0) for i,z in enumerate(raw['z_positions_um'])]
    expected_pairs={(t,z) for t in tiles for z in range(len(depths))};pairs=set()
    for p in raw['planes']:
        key=(int(p['tile_id']),int(p['z_index']))
        if key not in expected_pairs or key in pairs:raise ValueError('Duplicate or unknown raw tile/depth')
        pairs.add(key);t=tiles[key[0]];z=depths[key[1]]
        if not math.isclose(float(p['physical_z_um']),z['physical_z_um'],abs_tol=1e-9):raise ValueError('Raw depth mismatch')
        support=int(p['supported_core_pixels']);core=int(p['core_pixels'])
        if not 0<=support<=core:raise ValueError('Invalid acquired-support count')
        t['supported_pixel_planes']+=support;t['rectangular_pixel_planes']+=core;t['raw_barcodes']+=int(p['raw_barcodes'])
        z['supported_pixels']+=support;z['rectangular_pixels']+=core;z['raw_barcodes']+=int(p['raw_barcodes'])
    if pairs!=expected_pairs:raise ValueError('Raw summary omits output tile/depth slots')
    seen=set()
    for p in spatial:
        key=(int(p['output_tile']),int(p['z_index']))
        if key not in pairs or key in seen:raise ValueError('Duplicate or unknown final tile/depth')
        seen.add(key);t=tiles[key[0]];z=depths[key[1]]
        if not math.isclose(float(p['physical_z_um']),z['physical_z_um'],abs_tol=1e-9):raise ValueError('Final depth mismatch')
        n=int(p['barcodes']);b=int(p['blank_barcodes'])
        if not 0<=b<=n:raise ValueError('Invalid spatial blank count')
        t['final_barcodes']+=n;t['final_blank_barcodes']+=b;z['final_barcodes']+=n;z['final_blank_barcodes']+=b
    if sum(t['final_barcodes'] for t in tiles.values())!=kept or sum(t['final_blank_barcodes'] for t in tiles.values())!=blanks:
        raise ValueError('Spatial summary does not conserve final counts')
    total_support=sum(t['supported_pixel_planes'] for t in tiles.values())
    total_core=sum(t['rectangular_pixel_planes'] for t in tiles.values())
    if (sum(z['raw_barcodes'] for z in depths)!=raw['raw_barcodes'] or total_support!=raw['supported_core_pixel_planes']
            or total_core!=raw['rectangular_core_pixel_planes']):raise ValueError('Raw summary count conservation failed')
    for t in tiles.values():t['mean_supported_fraction']=t['supported_pixel_planes']/max(1,t['rectangular_pixel_planes'])
    for z in depths:z['supported_fraction']=z['supported_pixels']/max(1,z['rectangular_pixels'])
    return dict(raw_barcodes=raw['raw_barcodes'],adaptive_filtered_barcodes=selected,
        final_barcodes=kept,final_coding_barcodes=kept-blanks,final_blank_barcodes=blanks,removed_duplicates=removed,
        actual_post_dedup_blank_estimate=rate,adaptive_target=export['adaptive_target'],
        post_dedup_target_met=export['post_dedup_target_met'],supported_pixel_planes=total_support,
        rectangular_pixel_planes=total_core,rectangular_support_fraction=total_support/max(1,total_core),
        tiles=list(tiles.values()),depths=depths,
        support_interpretation='All-bit acquired support after registration and preprocessing margins, divided by the rectangular output lattice. Includes empty space outside fields/tissue; NOT fraction of tissue recovered.',
        blank_interpretation='Codebook-normalized blank/coding count ratio after complete cross-Z and output-boundary duplicate removal; not an independently measured molecular error rate.')


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def main():
    root=Path(__file__).resolve().parent.parent;exports=root/'full_filter_v1/exports'
    paths=[root/'full_decode_v1/raw_manifest.json',exports/'complete.json',root/'full_decode_v1/plan.json',
           root/'full_optimization_v1/optimization.json']
    raw,completion,plan,opt=[json.loads(p.read_text()) for p in paths]
    protected={str(p):sha(p) for p in paths}
    for n,h in completion['output_sha256'].items():
        p=exports/n
        if sha(p)!=h:raise RuntimeError('Final export changed: '+str(p))
        protected[str(p)]=h
    if opt['status']!='converged' or not opt['convergence_passed']:raise RuntimeError('Optimization not converged')
    registration_path=Path(plan['registration_manifest']);registration=json.loads(registration_path.read_text())
    protected[str(registration_path)]=sha(registration_path)
    if raw['input_sha256'].get(str(registration_path))!=protected[str(registration_path)]:raise RuntimeError('Registration differs from decoded inputs')
    result=summarize(raw,completion,read_csv(exports/'counts_by_codeword.csv'),read_csv(exports/'counts_by_tile_and_depth.csv'))
    result.update(status='FINAL_RESULTS_REVIEWABLE',export_status=completion['status'],
        input_sha256=protected,registration_scope=registration['acceptance_scope'],
        unsupported_sources_by_round=registration['unsupported_sources_by_round'],
        registration_pointwise_exceptions=registration['pointwise_exceptions'],
        registration_limitations=registration['limitations'],codebook_path=raw['codebook_path'],
        source_sha256=sha(__file__))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=root/'final_review_v1';out.mkdir(exist_ok=False)
    shape=raw['layout']['shape_tiles_yx'];coverage=np.full(shape,np.nan);counts=np.full(shape,np.nan)
    for t in result['tiles']:
        coverage[t['grid_row'],t['grid_column']]=t['mean_supported_fraction']
        counts[t['grid_row'],t['grid_column']]=t['final_barcodes']
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    im=axes[0,0].imshow(coverage,vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=axes[0,0],label='Mean supported fraction across Z')
    axes[0,0].set_title('Registered all-bit support per output tile')
    im=axes[0,1].imshow(np.log10(1+counts),cmap='magma');fig.colorbar(im,ax=axes[0,1],label='log10(1 + final barcode count)')
    axes[0,1].set_title('After adaptive filtering and duplicate removal')
    for ax in axes[0]:ax.set(xlabel='Output tile column',ylabel='Output tile row')
    z=[d['physical_z_um'] for d in result['depths']]
    axes[1,0].plot(z,[d['raw_barcodes'] for d in result['depths']],label='Raw')
    axes[1,0].plot(z,[d['final_barcodes'] for d in result['depths']],label='Final');axes[1,0].legend()
    axes[1,0].set(xlabel='Reference depth (um)',ylabel='Barcode count',title='Counts by depth')
    axes[1,1].plot(z,[d['supported_fraction'] for d in result['depths']]);axes[1,1].set_ylim(0,1)
    axes[1,1].set(xlabel='Reference depth (um)',ylabel='Fraction of rectangular output area',title='Usable acquired support by depth')
    fig.suptitle('0718 stitched decode â€” source FOVs 0â€“51, codebook 0\nCoverage includes empty space outside tissue; unsupported sources remain excluded')
    fig.tight_layout();fig.savefig(out/'coverage_and_counts.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4));steps=opt['iterations']
    for ax,key,label,limit in [(axes[0],'maximum_relative_scale_change','Maximum relative scale change',.01),
                               (axes[1],'maximum_background_change_in_previous_scale_units','Maximum background change / previous scale',.005)]:
        ax.plot([x['iteration'] for x in steps],[x[key] for x in steps],marker='o',markersize=3)
        ax.axhline(limit,color='red',linestyle='--');ax.set(xlabel='Fresh optimization iteration',ylabel=label)
    fig.tight_layout();fig.savefig(out/'optimization_convergence.png',dpi=160);plt.close(fig)
    rate=result['actual_post_dedup_blank_estimate'];rate_text='unavailable' if rate is None else f'{rate:.3%}'
    lines=['# 0718 stitched decoding results','',
        f"Final barcodes: **{result['final_barcodes']:,}** ({result['final_coding_barcodes']:,} coding; {result['final_blank_barcodes']:,} blank).",
        f"Actual codebook-normalized blank estimate after duplicate removal: **{rate_text}**; target **5%**.",
        f"Raw: {result['raw_barcodes']:,}; adaptive filtered: {result['adaptive_filtered_barcodes']:,}; duplicates removed: {result['removed_duplicates']:,}.",'',
        '![Coverage and counts](coverage_and_counts.png)','',result['support_interpretation'],'',result['blank_interpretation'],'',
        '## Registration limitations','']
    lines+=['- '+x for x in result['registration_limitations']]
    lines+=['','## Files','',f'- Final barcodes: `{exports / "adaptive_filtered_z_deduplicated.csv.gz"}`',
        f'- Removed duplicates: `{exports / "removed_z_duplicates.csv.gz"}`',f'- Codeword counts: `{exports / "counts_by_codeword.csv"}`',
        f'- Tile/depth counts: `{exports / "counts_by_tile_and_depth.csv"}`','',
        '![Optimization convergence](optimization_convergence.png)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    for p,h in protected.items():
        if sha(p)!=h:raise RuntimeError('Inputs changed during review: '+p)
    result['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(out),final_barcodes=result['final_barcodes'],post_dedup_blank_estimate=rate)))


if __name__=='__main__':main()
