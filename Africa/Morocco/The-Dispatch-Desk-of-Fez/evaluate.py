import csv
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import statistics
import tempfile
from collections import defaultdict


ROOT = Path(__file__).resolve().parent
with (ROOT / 'data/train-input.csv').open(encoding='utf-8', newline='') as stream:
    requests = list(csv.DictReader(stream))
with (ROOT / 'data/train-answers.csv').open(encoding='utf-8', newline='') as stream:
    answers = {row['request_id']: row for row in csv.DictReader(stream)}
with tempfile.TemporaryDirectory() as scratch:
    shutil.copyfile(ROOT / 'data/train-input.csv', Path(scratch) / 'test-input.csv')
    previous = Path.cwd()
    os.chdir(scratch)
    namespace = runpy.run_path(str(ROOT / 'main.py'))
    os.chdir(previous)
    shutil.copyfile(Path(scratch) / 'routing_log.csv', ROOT / 'results/train-routing.csv')
category = namespace['category']
route = namespace['route']
prompts = [request['prompt'] for request in requests]
correctness = []
groups = defaultdict(list)
for index, request in enumerate(requests):
    answer = answers[request['request_id']]
    reference = answer['reference_answer'].strip().casefold()
    pair = tuple(answer[model + '_answer'].strip().casefold() == reference for model in ['cheap', 'expensive'])
    correctness.append(pair)
    groups[category(request['prompt'])].append(index)


def score(indices, choices):
    correct = sum(correctness[index][model == 'expensive'] for index, model in zip(indices, choices))
    cost = sum(25 if model == 'expensive' else 1 for model in choices)
    quality = correct / len(indices)
    return {'quality': quality, 'cost': cost, 'requests': len(indices), 'pass': quality >= 0.8 and cost <= 20 * len(indices)}


indices = list(range(len(requests)))
summary = {
    'cheap_only': score(indices, ['cheap'] * len(indices)),
    'expensive_only': score(indices, ['expensive'] * len(indices)),
    'router_training': score(indices, route(prompts)),
}
with (ROOT / 'results/categories.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['category', 'requests', 'cheap_correct', 'expensive_correct', 'gain'])
    for name, members in sorted(groups.items()):
        cheap = sum(correctness[index][0] for index in members)
        expensive = sum(correctness[index][1] for index in members)
        writer.writerow([name, len(members), cheap, expensive, (expensive - cheap) / len(members)])
checks = []
for seed in range(20):
    shuffled = indices.copy()
    random.Random(seed).shuffle(shuffled)
    for fold in range(5):
        held = shuffled[fold::5]
        held_set = set(held)
        gains = {}
        for name, members in groups.items():
            fitting = [index for index in members if index not in held_set]
            gains[name] = sum(int(correctness[index][1]) - int(correctness[index][0]) for index in fitting) / len(fitting) if fitting else 0.6
        metrics = score(held, route([prompts[index] for index in held], gains))
        checks.append({'seed': seed, 'fold': fold, **metrics})
with (ROOT / 'results/split-checks.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
    writer.writeheader()
    writer.writerows(checks)
summary['exploratory_splits'] = {
    'folds': len(checks),
    'mean_quality': statistics.mean(check['quality'] for check in checks),
    'min_quality': min(check['quality'] for check in checks),
    'passing_folds': sum(check['pass'] for check in checks),
    'max_cost_per_request': max(check['cost'] / check['requests'] for check in checks),
    'caveat': 'Gain rates are refitted without held-fold answers; category definitions and routing threshold were selected after inspecting the full training set. These are exploratory checks, not independent validation.',
}
(ROOT / 'results/metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
