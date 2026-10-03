"""Optional count QC, clustering, H5AD and interactive tissue/UMAP reports."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

def run(root,config):
    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    root=Path(root);out=root/'basic_analysis'
    if (out/'complete.json').exists():return
    out.mkdir(exist_ok=False)
    exports=root/'partition_sjoin_v2/exports';centers=root/'partition_sjoin_v2/cell_coordinates_v3/feature_metadata.csv'
    matrix=pd.read_csv(exports/'barcodes_per_feature.csv',index_col=0);meta=pd.read_csv(centers,index_col=0)
    assert matrix.index.equals(meta.index) and np.array_equal(matrix.sum(axis=1),meta.n_transcripts)
    blank=matrix.columns.str.contains('blank',case=False);coding=matrix.loc[:,~blank]
    a=ad.AnnData(sparse.csr_matrix(coding.to_numpy(np.float32)),obs=meta.copy(),var=pd.DataFrame(index=coding.columns))
    a.obs_names=a.obs_names.astype(str);a.obs['blank_counts']=matrix.loc[:,blank].sum(1).to_numpy()
    a.obs['total_counts']=np.asarray(a.X.sum(1)).ravel();a.obs['n_genes_by_counts']=np.asarray((a.X>0).sum(1)).ravel()
    a.obs['qc_pass']=(a.obs.total_counts>10)&(a.obs.n_genes_by_counts>5)&(a.obs.legacy_volume_um3>100)
    a.obs.to_csv(out/'all_cell_qc.csv');a=a[a.obs.qc_pass].copy()
    if a.n_obs<3 or a.n_vars<3:raise ValueError('Too few cells or genes for clustering')
    a.layers['counts']=a.X.copy();sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a);a.raw=a.copy()
    c=a.copy();sc.pp.scale(c,max_value=10);pcs=min(30,c.n_vars-1,c.n_obs-1)
    sc.tl.pca(c,n_comps=pcs,svd_solver='arpack',random_state=718)
    sc.pp.neighbors(c,n_neighbors=min(20,c.n_obs-1),n_pcs=pcs,random_state=718)
    sc.tl.leiden(c,resolution=1,key_added='leiden',random_state=718,flavor='igraph',n_iterations=2,directed=False)
    sc.tl.umap(c,min_dist=.1,random_state=718)
    a.obs['leiden']=c.obs.leiden.copy()
    for k in ['X_pca','X_umap']:a.obsm[k]=c.obsm[k].copy()
    for k in ['connectivities','distances']:a.obsp[k]=c.obsp[k].copy()
    for k in ['pca','neighbors','umap']:a.uns[k]=c.uns[k].copy()
    a.varm['PCs']=c.varm['PCs'].copy()
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sc.pl.umap(a,color='leiden',show=False);plt.savefig(out/'umap_clusters.png',dpi=180,bbox_inches='tight');plt.close('all')
    sc.tl.rank_genes_groups(a,'leiden',method='wilcoxon',use_raw=True)
    sc.get.rank_genes_groups_df(a,group=None).to_csv(out/'cluster_markers.csv',index=False)
    u=a.copy();sc.tl.umap(u,n_components=3,min_dist=.1,random_state=718);a.obsm['X_umap3d']=u.obsm['X_umap'].copy()
    palette=a.uns['leiden_colors'];labels=a.obs.leiden.astype(str).to_numpy();traces=[];spatial=[]
    columns=['stitched_center_x_um','stitched_center_y_um','stitched_center_z_um'];xyz=a.obs[columns].to_numpy()
    valid=a.obs.stitched_center_supported.to_numpy()&np.isfinite(xyz).all(1)
    for label,color in zip(a.obs.leiden.cat.categories,palette):
        ids=np.flatnonzero(labels==label);points=a.obsm['X_umap3d'][ids]
        traces.append(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',name='Cluster '+label,
            text=a.obs_names[ids],marker=dict(color=color,size=2,opacity=.75),hovertemplate='Cell %{text}<extra>%{fullData.name}</extra>'))
        ids=np.flatnonzero((labels==label)&valid)
        spatial.append(dict(label=label,color=color,x=xyz[ids,0].tolist(),y=xyz[ids,1].tolist(),z=xyz[ids,2].tolist(),
            cell_ids=a.obs_names[ids].tolist(),counts=a.obs.total_counts.iloc[ids].astype(int).tolist(),review=a.obs.center_mapping_review_needed.iloc[ids].astype(int).tolist()))
    fig=go.Figure(traces);fig.update_layout(title='Stitched sample: 3D expression UMAP',scene=dict(xaxis_title='UMAP1',yaxis_title='UMAP2',zaxis_title='UMAP3'))
    fig.write_html(out/'interactive_umap3d.html',include_plotlyjs=True)
    if valid.any():
        payload=dict(traces=spatial,zmin=float(np.floor(xyz[valid,2].min())),zmax=float(np.ceil(xyz[valid,2].max())),
            spans=np.maximum(np.ptp(xyz[valid],axis=0),1e-8).tolist(),cells=int(valid.sum()),flagged=int(a.obs.center_mapping_review_needed.to_numpy()[valid].sum()))
        template=(root/'scripts/tissue_template.html').read_text()
        template=template.replace('0718','Stitched sample').replace('78,159',f"{payload['cells']:,}").replace('4,382',f"{payload['flagged']:,}").replace('18 Leiden',f'{len(spatial)} Leiden')
        template=template.replace('__PLOTLY__',get_plotlyjs()).replace('__PAYLOAD__',json.dumps(payload,separators=(',',':')))
        (out/'interactive_tissue3d.html').write_text(template,encoding='utf-8')
    a.write_h5ad(out/'clustered_cells.h5ad');a.obs.to_csv(out/'cell_metadata_and_clusters.csv')
    # Explore frozen outputs without recomputing or overwriting clusters.
    cells=[]
    def markdown(text):cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(True)))
    def code(text):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(True)))
    markdown('# Stitched sample basic analysis\nOpen from this result directory. Leiden clusters are exploratory. Counts and corrected spatial coordinates are preserved. This notebook reads the completed analysis; it does not rerun decoding, filtering or clustering.')
    code("from pathlib import Path\nimport anndata as ad\nimport scanpy as sc\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, FileLink\nroot=Path.cwd()\nadata=ad.read_h5ad(root/'clustered_cells.h5ad')\nqc=pd.read_csv(root/'all_cell_qc.csv',index_col=0)\nprint(f'{len(qc):,} cells before QC; {adata.n_obs:,} retained; {adata.n_vars} coding genes')\ndisplay(qc[['total_counts','n_genes_by_counts','blank_counts','legacy_volume_um3']].describe())")
    markdown('## QC and barcode counts\nBlank counts are excluded from the expression matrix. A cell-assignment blank estimate is distinct from the global post-deduplication estimate.')
    code("fig,axes=plt.subplots(1,3,figsize=(12,3))\nfor ax,key in zip(axes,['total_counts','n_genes_by_counts','blank_counts']):\n    ax.hist(np.log1p(qc[key]),bins=60);ax.set(xlabel='log1p '+key,ylabel='Cells')\nplt.tight_layout()")
    markdown('## Expression clusters and markers\nThe same cluster colors are used in the UMAP and spatial map. These are not cell-type annotations.')
    code("sc.pl.umap(adata,color=['leiden','total_counts','n_genes_by_counts'])\ndisplay(adata.obs.groupby('leiden',observed=True).size().rename('cells'))\nmarkers=pd.read_csv(root/'cluster_markers.csv')\ndisplay(markers.groupby('group',sort=False).head(5))")
    markdown('## Tissue coordinates\nCoordinates are in micrometres, not UMAP units. Unsupported centers remain absent; mapping ambiguity flags are retained.')
    code("fig,ax=plt.subplots(figsize=(8,7))\nvalid=adata.obs.stitched_center_supported.astype(bool)\nfor cluster,color in zip(adata.obs.leiden.cat.categories,adata.uns['leiden_colors']):\n    rows=adata.obs.loc[valid & (adata.obs.leiden==cluster)]\n    ax.scatter(rows.stitched_center_x_um,rows.stitched_center_y_um,s=1,c=color,label=cluster,rasterized=True)\nax.set(xlabel='Stitched X (um)',ylabel='Stitched Y (um)',aspect='equal')\nax.legend(markerscale=5,bbox_to_anchor=(1,1),title='Leiden')\nplt.show()\nprint('Flagged centers:',int(adata.obs.center_mapping_review_needed.sum()))\ndisplay(FileLink('interactive_tissue3d.html'),FileLink('interactive_umap3d.html'))")
    markdown('## Gene exploration\nChoose a measured coding gene. UMAP expression uses log-normalized values; raw counts remain in `layers["counts"]`.')
    code("gene=adata.var_names[0]  # replace with a measured gene\nsc.pl.umap(adata,color=gene,use_raw=True)\ncounts=np.asarray(adata.layers['counts'].sum(axis=0)).ravel()\ndisplay(pd.Series(counts,index=adata.var_names,name='raw_counts').sort_values(ascending=False).head(20))")
    markdown('External TS2 comparisons require a supplied reference and matching gene names. They are not inferred from these cells. See the parent workflow quality reports for acquired support, registration limitations and the actual post-deduplication blank estimate.')
    for i,cell in enumerate(cells):cell['id']='stitched-%02d'%i
    nb=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},cells=cells)
    (out/'basic_analysis.ipynb').write_text(json.dumps(nb,indent=2))
    (out/'complete.json').write_text(json.dumps(dict(status='COMPLETE',all_cells=len(matrix),qc_cells=a.n_obs,genes=a.n_vars,
        clusters=len(a.obs.leiden.cat.categories),spatial_cells=int(valid.sum()),seed=718,
        limitations=['Exploratory clusters, not biological cell types. Existing segmentation offsets and coordinate-review flags remain.'],
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2))
