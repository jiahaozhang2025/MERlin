"""Run one audited full-sample fiducial extraction in its isolated namespace."""
import hashlib
import json
import os
from pathlib import Path
import sys

import extract_registration

root = Path(__file__).resolve().parent.parent
inventory = json.loads((root / 'inventory.json').read_text())
task = inventory['tasks'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
data = Path(inventory['data_dir'])
for name, expected in inventory['metadata_sha256'].items():
    assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
raw = Path(task['raw_path'])
assert (raw.stat().st_size, raw.stat().st_mtime_ns) == (task['raw_size'], task['raw_mtime_ns'])
assert hashlib.sha256(Path(task['xml_path']).read_bytes()).hexdigest() == task['xml_sha256']
sys.argv = ['extract_registration.py', '--fov', str(task['fov']), '--round', str(task['round']),
            '--output-root', str(root / 'clouds')]
extract_registration.main()
assert (raw.stat().st_size, raw.stat().st_mtime_ns) == (task['raw_size'], task['raw_mtime_ns'])
