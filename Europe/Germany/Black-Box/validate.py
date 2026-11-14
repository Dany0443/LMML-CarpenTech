import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import evaluation_script as grader
from main import NAMES, solve

ROOT = Path(__file__).resolve().parent


class SyntheticOracle:
    def __init__(self, function):
        self.function = function
        self.cache = {}
        self.spent = 0

    @property
    def remaining(self):
        return 150 - self.spent

    def ask_many(self, points, phase=None):
        answers = []
        for point in points:
            key = tuple(round(float(coordinate), 9) for coordinate in point)
            if key not in self.cache:
                if self.remaining == 0:
                    raise RuntimeError('Synthetic query budget exceeded')
                self.cache[key] = round(float(self.function(np.array(point))), 6)
                self.spent += 1
            answers.append(self.cache[key])
        return answers


rng = np.random.default_rng(20261114)
checks = []
for case in range(100):
    pair = sorted(rng.choice(5, size=2, replace=False).tolist())
    gate = sorted(rng.choice(5, size=2, replace=False).tolist())
    weights = rng.uniform(0.5, 2.0, size=2)
    signs = rng.choice([-1, 1], size=2)
    normal = np.zeros(5)
    normal[gate] = weights * signs
    offset = float(np.sum(weights[signs < 0]))
    invisible_axis_limit = weights.sum() / 2 + weights.max() / 2
    threshold = float(rng.uniform(invisible_axis_limit + 0.02 * weights.min(), weights.sum() - 0.04 * weights.min()) - offset)
    linear = rng.uniform(-2, 2, size=5)
    linear[rng.random(5) < 0.5] = 0
    intercept = float(rng.uniform(-2, 3))
    interaction = float(rng.choice([-1, 1]) * rng.uniform(1, 6))
    jump = float(rng.uniform(0.5, 3))

    def function(point):
        return intercept + point @ linear + interaction * (point[pair[0]] - 0.5) * (point[pair[1]] - 0.5) + jump * (point @ normal >= threshold)

    challenge = rng.uniform(0, 1, size=(300, 5)).tolist()
    relevant = set(pair) | set(gate) | set(np.flatnonzero(linear))
    truth = {
        'relevant': [NAMES[index] for index in sorted(relevant)],
        'interaction': {'variables': [NAMES[index] for index in pair], 'coefficient': interaction},
        'discontinuity': {'gate_variables': [NAMES[index] for index in gate], 'boundary': dict(zip(NAMES, normal.tolist())), 'threshold': threshold, 'jump': jump},
    }
    grader.HF = SimpleNamespace(VAR_NAMES=NAMES, TRUTH=truth, evaluate=lambda point: round(float(function(np.array(point))), 6), gate_active=lambda point: float(np.array(point) @ normal) >= threshold)
    oracle = SyntheticOracle(function)
    started = time.perf_counter()
    submission = solve(oracle, challenge)
    elapsed = time.perf_counter() - started
    report = grader.evaluate({'submission': submission, 'queries': [None] * oracle.spent, 'error': None}, 'synthetic', challenge)
    checks.append({'case': case, 'score': report['score'], 'queries': oracle.spent, 'seconds': elapsed, 'rmse': report['metrics']['rmse_overall']})
reports = ROOT / 'results'
reports.mkdir(exist_ok=True)
with (reports / 'synthetic_checks.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(checks[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(checks)
summary = {'cases': len(checks), 'passed': sum(check['score'] >= 70 for check in checks), 'minimum_score': min(check['score'] for check in checks), 'maximum_queries': max(check['queries'] for check in checks), 'maximum_seconds': max(check['seconds'] for check in checks)}
(reports / 'synthetic_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
