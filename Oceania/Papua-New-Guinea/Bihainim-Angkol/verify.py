import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import lagoon

ROOT = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as scratch:
    folder = Path(scratch)
    (folder / 'main.py').write_bytes((ROOT / 'main.py').read_bytes())
    (folder / 'test-input.csv').write_bytes((ROOT / 'train-input.csv').read_bytes())
    started = time.monotonic()
    subprocess.run([sys.executable, 'main.py'], cwd=folder, check=True, timeout=60)
    elapsed = time.monotonic() - started
    policy_path = folder / 'test-output.csv'
    problems = lagoon.check_policy_file(str(policy_path))
    assert not problems, problems
    policy = lagoon.read_policy(str(policy_path))
    expected = lagoon.read_policy(str(ROOT / 'results/policy.csv'))
    assert policy == expected
    with (folder / 'test-input.csv').open(newline='') as stream:
        inputs = list(csv.DictReader(stream))
    with policy_path.open(newline='') as stream:
        decisions = list(csv.DictReader(stream))
    assert len(inputs) == len(decisions)
    for source, decision in zip(inputs, decisions):
        assert all(source[key] == decision[key] for key in ['row', 'col', 'hold', 'steps_left'])
    reference = json.loads((ROOT / 'train-output.json').read_text())
    mean = sum(lagoon.play_day(policy, seed)[0] for seed in reference['days']) / len(reference['days'])
    ratio = mean / reference['angkol_mean']
    assert ratio >= reference['target_ratio']
    report = {'standalone': True, 'rows': len(decisions), 'runtime_seconds': elapsed,
              'csv_valid': True, 'export_matches_trained_policy': True,
              'practice_mean': mean, 'practice_ratio': ratio, 'local_pass': True}
(ROOT / 'results/standalone-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
