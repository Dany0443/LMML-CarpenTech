import csv
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path
from time import perf_counter

from detector import Detector

FOLDER = Path(__file__).resolve().parent


def metrics(scores, labels):
    flags = [score >= .5 for score in scores]
    actual = [int(label['is_anomaly']) for label in labels]
    tp = sum(flag and truth for flag, truth in zip(flags, actual))
    fp = sum(flag and not truth for flag, truth in zip(flags, actual))
    fn = sum(not flag and truth for flag, truth in zip(flags, actual))
    f1 = 2. * tp / max(1, 2 * tp + fp + fn)
    recalls = {}
    for kind in ('point', 'drift', 'collective'):
        total = sum(label['anomaly_type'] == kind for label in labels)
        hits = sum(flag and label['anomaly_type'] == kind for flag, label in zip(flags, labels))
        recalls[kind] = hits / total if total else 0.
    grade = 100. * (.8 * f1 + .05 * recalls['point']
                    + .075 * recalls['drift'] + .075 * recalls['collective'])
    return {
        'score': max(1., min(100., grade)),
        'f1': f1,
        'precision': tp / max(1, tp + fp),
        'recall': tp / max(1, tp + fn),
        'type_recall': recalls,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'readings': len(labels),
    }


with (FOLDER / 'data/train-input.csv').open(encoding='utf-8', newline='') as stream:
    readings = list(csv.DictReader(stream))
with (FOLDER / 'data/train-output.csv').open(encoding='utf-8', newline='') as stream:
    labels = list(csv.DictReader(stream))
if len(readings) != len(labels):
    raise ValueError('Reading and label counts differ')
if any(reading['timestamp'] != label['timestamp'] for reading, label in zip(readings, labels)):
    raise ValueError('Reading and label timestamps differ')

detector = Detector()
scores = []
started = perf_counter()
for reading in readings:
    score = detector.update(reading)
    if not math.isfinite(score) or not 0. <= score <= 1.:
        raise ValueError('Detector returned an invalid score')
    scores.append(score)
elapsed = perf_counter() - started
split = len(readings) // 2
summary = {
    'full_stream': metrics(scores, labels),
    'development_first_half': metrics(scores[:split], labels[:split]),
    'chronological_later_half': metrics(scores[split:], labels[split:]),
    'zone_scores': {},
    'update_seconds': elapsed,
    'python_version': sys.version.split()[0],
    'serialized_state_bytes': len(pickle.dumps(detector)),
    'detector_sha256': hashlib.sha256((FOLDER / 'detector.py').read_bytes()).hexdigest(),
    'input_sha256': {
        filename: hashlib.sha256((FOLDER / 'data' / filename).read_bytes()).hexdigest()
        for filename in ('train-input.csv', 'train-output.csv')
    },
}
for zone in sorted({reading['bunker_zone'] for reading in readings}):
    indices = [index for index, reading in enumerate(readings) if reading['bunker_zone'] == zone]
    summary['zone_scores'][zone] = metrics([scores[index] for index in indices],
                                          [labels[index] for index in indices])

(FOLDER / 'results').mkdir(exist_ok=True)
with (FOLDER / 'results/predictions.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['timestamp', 'bunker_zone', 'score', 'prediction', 'is_anomaly', 'anomaly_type'])
    for reading, score, label in zip(readings, scores, labels):
        writer.writerow([reading['timestamp'], reading['bunker_zone'], f'{score:.9f}',
                         int(score >= .5), label['is_anomaly'], label['anomaly_type']])
(FOLDER / 'results/metrics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
