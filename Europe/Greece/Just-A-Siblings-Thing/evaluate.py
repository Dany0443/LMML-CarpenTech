import csv
import hashlib
import json
from pathlib import Path

import torch

from model import FEATURE_COLUMNS, MythicThreatClassifier

FOLDER = Path(__file__).resolve().parent


def binary_metrics(labels, predictions):
    if len(labels) != len(predictions) or not labels:
        raise ValueError('Expected matching nonempty labels and predictions')
    pairs = list(zip(labels, predictions))
    tp = sum(truth == 1 and predicted == 1 for truth, predicted in pairs)
    fp = sum(truth == 0 and predicted == 1 for truth, predicted in pairs)
    tn = sum(truth == 0 and predicted == 0 for truth, predicted in pairs)
    fn = sum(truth == 1 and predicted == 0 for truth, predicted in pairs)
    recall = tp / (tp + fn) if tp + fn else 0.
    specificity = tn / (tn + fp) if tn + fp else 0.
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': (tp + tn) / len(labels),
        'precision': tp / (tp + fp) if tp + fp else 0.,
        'recall': recall,
        'specificity': specificity,
        'balanced_accuracy': (recall + specificity) / 2.,
        'f1': 2. * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        'f2': 5. * tp / (5 * tp + 4 * fn + fp) if 5 * tp + 4 * fn + fp else 0.,
    }


def load_test(folder):
    with (folder / 'data/test_dataset.csv').open(encoding='utf-8', newline='') as stream:
        encounters = list(csv.DictReader(stream))
    labels_by_id = {}
    with (folder / 'data/test_labels.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            encounter_id = row['encounter_id']
            if encounter_id in labels_by_id:
                raise ValueError('Duplicate label ID')
            label = int(row['is_catastrophic_threat'])
            if label not in (0, 1):
                raise ValueError('Expected binary labels')
            labels_by_id[encounter_id] = label
    ids = [encounter['encounter_id'] for encounter in encounters]
    if len(set(ids)) != len(ids) or set(ids) != set(labels_by_id):
        raise ValueError('Encounter and label IDs must match one to one')
    matrix = []
    labels = []
    for encounter in encounters:
        matrix.append([float(encounter[column]) for column in FEATURE_COLUMNS])
        labels.append(labels_by_id[encounter['encounter_id']])
    features = torch.tensor(matrix, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError('Encounter features must be finite')
    return encounters, features, labels


def predict(checkpoint, features):
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    oracle = MythicThreatClassifier()
    oracle.load_state_dict(weights, strict=True)
    oracle.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(oracle(features)).tolist()
    return probabilities, [int(probability >= .5) for probability in probabilities]


def evaluate_models():
    encounters, features, labels = load_test(FOLDER)
    comparisons = []
    predictions_by_id = []
    for encounter, label in zip(encounters, labels):
        predictions_by_id.append({'encounter_id': encounter['encounter_id'], 'actual': label})
    for name in ('ares', 'athena', 'apollo'):
        probabilities, predictions = predict(FOLDER / f'models/model_{name}.pth', features)
        comparisons.append(dict(model=name.upper(), **binary_metrics(labels, predictions)))
        for row, probability, prediction in zip(predictions_by_id, probabilities, predictions):
            row[f'{name}_probability'] = probability
            row[f'{name}_prediction'] = prediction
    selected = min(comparisons, key=lambda oracle: (oracle['fn'], oracle['fp']))
    flag = 'SIGMOID_' + selected['model']
    artifacts = FOLDER / 'results'
    artifacts.mkdir(exist_ok=True)
    for filename, rows in [('metrics.csv', comparisons), ('predictions.csv', predictions_by_id)]:
        with (artifacts / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    originals = [FOLDER / 'data/test_dataset.csv', FOLDER / 'data/test_labels.csv']
    originals.extend(sorted((FOLDER / 'models').glob('*.pth')))
    summary = {
        'flag': flag,
        'selected_model': selected['model'],
        'threshold': .5,
        'encounters': len(labels),
        'catastrophic_threats': sum(labels),
        'feature_order': FEATURE_COLUMNS,
        'torch_version': torch.__version__,
        'comparisons': comparisons,
        'sha256': {str(path.relative_to(FOLDER)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in originals},
    }
    (artifacts / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (artifacts / 'flag.txt').write_text(flag + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
