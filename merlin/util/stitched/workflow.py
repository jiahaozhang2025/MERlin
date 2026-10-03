"""Configuration, input freezing and subprocess dispatch for stitched tasks."""
import csv,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):digest.update(b)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def configuration(dataset,parameters):
    """Resolve without analysis outputs: fragment counts work during DAG creation."""
    import merlin
    c=dict(parameters.get('configuration',{}))
    if c.get('profile')!='0718_40x_150z':
        raise ValueError('Explicit supported acquisition profile 0718_40x_150z required; other geometries need validation.')
    if 'metadata_dir' not in c:
        c['metadata_dir']=str(Path(dataset.analysisPath)/'data')
        if not (Path(c['metadata_dir'])/'dataorganization.csv').exists():
            c['metadata_dir']=str(Path(dataset.analysisPath))
    if not (Path(c['metadata_dir'])/'dataorganization.csv').is_file():
        raise ValueError('Metadata directory has no dataorganization.csv: '+c['metadata_dir'])
    c.setdefault('raw_root',dataset.rawDataPath)
    c.setdefault('canonical_root',str(Path(merlin.__file__).resolve().parent.parent))
    c.setdefault('codebook_index',0);c.setdefault('optimizer_workers',4)
    c.setdefault('num_threads',2);c.setdefault('review_decision',None)
    directory=Path(c['metadata_dir']).resolve()
    books=list(directory.glob('codebook_%d_*.csv'%c['codebook_index']))
    if len(books)!=1:raise ValueError('Exactly one metadata codebook for requested index is required')
    with books[0].open() as f:
        reader=csv.DictReader(f);words=list(reader);bits=reader.fieldnames[2:]
    if reader.fieldnames[:2]!=['name','id'] or not words:raise ValueError('Unexpected codebook schema')
    with (directory/'dataorganization.csv').open() as f:org=list(csv.DictReader(f))
    rows=[]
    for bit in bits:
        found=[r for r in org if r['readoutName']==bit]
        if len(found)!=1:raise ValueError('Missing or ambiguous readout '+bit)
        rows.append(found[0])
    c.update(metadata_dir=str(directory),raw_root=str(Path(c['raw_root']).resolve()),
             codebook_file=books[0].name,bit_names=bits,codeword_count=len(words),
             rounds=sorted({0,*[int(r['imagingRound']) for r in rows]}))
    return c


def initialize(root,config):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    complete=root/'initialization.json'
    if complete.exists():
        expected=dict(config,canonical_root=str((root/'canonical').resolve()))
        if read(root/'config.json')!=expected:raise ValueError('Existing stitched configuration differs')
        verify_sources(root);return
    # An incomplete attempt is deliberately not overwritten on MERlin retry.
    if (root/'config.json').exists():raise RuntimeError('Incomplete initialization: inspect and preserve this attempt before retrying')
    (root/'config.json').write_text(json.dumps(config,indent=2))
    package=Path(__file__).parent
    shutil.copytree(package/'operators',root/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['runner.py','registration_review.py','cell_centers.py','analysis_report.py','tissue_template.html']:
        shutil.copyfile(package/name,root/'scripts'/name)
    (root/'scripts/stitched_config.py').write_text("from pathlib import Path\nimport json\nROOT=Path(__file__).resolve().parent.parent\nCONFIG=json.loads((ROOT/'config.json').read_text())\n")
    (root/'inputs').mkdir();(root/'logs').mkdir();(root/'stage_receipts').mkdir()
    names=['positions.csv','dataorganization.csv','filemap.csv','microscope_parameters.json',config['codebook_file']]
    for name in names:shutil.copyfile(Path(config['metadata_dir'])/name,root/'inputs'/name)
    # Freeze the actual current canonical code. Running jobs never import later
    # checkout edits. Codebook, decoder, optimizer and filter hashes are recorded.
    canonical=root/'canonical';canonical.mkdir()
    shutil.copytree(Path(config['canonical_root'])/'merlin',canonical/'merlin',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    config['canonical_root']=str(canonical.resolve())
    (root/'config.json').write_text(json.dumps(config,indent=2))
    protected={str(p.relative_to(root)):sha(p) for d in ['scripts','canonical','inputs'] for p in (root/d).rglob('*') if p.is_file()}
    protected['config.json']=sha(root/'config.json')
    (root/'frozen_sources.json').write_text(json.dumps(protected,indent=2))
    execute(root,'preflight')
    complete.write_text(json.dumps(dict(status='COMPLETE',config_sha256=sha(root/'config.json'),inventory_sha256=sha(root/'inventory.json')),indent=2))


def verify_sources(root):
    for name,h in read(Path(root)/'frozen_sources.json').items():
        if sha(Path(root)/name)!=h:raise RuntimeError('Frozen stitched input changed: '+name)


def execute(root,stage,fragment=None,workers=52):
    root=Path(root);verify_sources(root)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    config=read(root/'config.json');env['OMP_NUM_THREADS']=str(config['num_threads'])
    env['PYTHONPATH']=str(root/'canonical')+os.pathsep+env.get('PYTHONPATH','')
    command=[sys.executable,'-B','-u',str(root/'scripts/runner.py'),stage,'--workers',str(workers)]
    if fragment is not None:command+=['--fragment',str(fragment)]
    subprocess.run(command,check=True,cwd=root/'scripts',env=env)
    verify_sources(root)
