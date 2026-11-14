import json
from functools import partial
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
PAIRS = [None] + list(combinations(range(10), 2))


def classifier():
    return make_pipeline(
        FunctionTransformer(partial(np.multiply, 1 / 16)),
        SVC(C=10, gamma=0.2, probability=True, random_state=42),
    )


def repair(probabilities, given, penalty=14):
    candidates = np.tile(given[:, None], (1, len(PAIRS)))
    for state, (left, right) in enumerate(PAIRS[1:], 1):
        candidates[given == left, state] = right
        candidates[given == right, state] = left
    likelihood = probabilities[np.arange(len(given))[:, None], candidates]
    gains = np.log(np.maximum(likelihood, 1e-9))
    gains -= gains[:, [0]]
    scores = np.zeros(len(PAIRS))
    scores[1:] = -penalty
    backtracks = []
    for gain in gains:
        best = scores.argmax()
        switched = scores[best] - penalty
        backtracks.append(np.where(scores >= switched, np.arange(len(PAIRS)), best))
        scores = np.maximum(scores, switched) + gain
    state = scores.argmax()
    states = []
    for backtrack in reversed(backtracks):
        states.append(state)
        state = backtrack[state]
    states = np.array(states[::-1])
    return candidates[np.arange(len(given)), states], states


def audit_table(scans, probabilities, corrected, states):
    audit = scans[['sample_id', 'given_label']].copy()
    audit['image_prediction'] = probabilities.argmax(axis=1)
    audit['corrected_label'] = corrected
    audit['changed'] = corrected != scans.given_label.to_numpy()
    audit['swap'] = ['none' if PAIRS[state] is None else f'{PAIRS[state][0]}<->{PAIRS[state][1]}' for state in states]
    return audit


train = pd.read_csv(ROOT / 'data/train-input.csv').sort_values('sample_id').reset_index(drop=True)
truth = pd.read_csv(ROOT / 'data/train-output.csv')
train = train.merge(truth, on='sample_id', validate='one_to_one')
test = pd.read_csv(ROOT / 'data/test-input.csv')
ordered = test.sort_values('sample_id').reset_index(drop=True)
pixels = [f'pixel_{column}' for column in range(64)]
train_pixels = train[pixels].to_numpy()
labels = train.corrected_label.to_numpy()
folds = StratifiedKFold(5, shuffle=True, random_state=42)
practice_probs = cross_val_predict(classifier(), train_pixels, labels, cv=folds, method='predict_proba')
practice_labels, practice_states = repair(practice_probs, train.given_label.to_numpy())
needed = train.given_label.to_numpy() != labels
changed = practice_labels != train.given_label.to_numpy()
recovered = int(np.sum(needed & changed & (practice_labels == labels)))
false_repairs = int(changed.sum()) - recovered
missed = int(needed.sum()) - recovered
precision = recovered / max(1, recovered + false_repairs)
recall = recovered / max(1, recovered + missed)
summary = {
    'practice_rows': len(train),
    'practice_image_cv_accuracy': float(np.mean(practice_probs.argmax(axis=1) == labels)),
    'practice_true_repairs': recovered,
    'practice_false_repairs': false_repairs,
    'practice_missed_repairs': missed,
    'practice_repair_f1': 2 * precision * recall / max(1e-12, precision + recall),
    'validation_limit': 'five-fold image predictions; sequence rules and parameters explored on this practice set',
}
recognizer = classifier().fit(train_pixels, labels)
test_probs = recognizer.predict_proba(ordered[pixels].to_numpy())
corrected, states = repair(test_probs, ordered.given_label.to_numpy())
corrections = pd.DataFrame({'sample_id': ordered.sample_id, 'corrected_label': corrected})
submission = test[['sample_id']].merge(corrections, on='sample_id', validate='one_to_one')
submission.to_csv(ROOT / 'test-output.csv', index=False)
reports = ROOT / 'results'
reports.mkdir(exist_ok=True)
audit_table(train, practice_probs, practice_labels, practice_states).to_csv(reports / 'practice_audit.csv', index=False)
audit_table(ordered, test_probs, corrected, states).to_csv(reports / 'test_audit.csv', index=False)
stability = []
for penalty in (6, 10, 14, 20, 30):
    alternate, _ = repair(test_probs, ordered.given_label.to_numpy(), penalty)
    stability.append({'penalty': penalty, 'repairs': int(np.sum(alternate != ordered.given_label)), 'matches_selected': bool(np.array_equal(alternate, corrected))})
pd.DataFrame(stability).to_csv(reports / 'stability.csv', index=False)
summary['test_rows'] = len(test)
summary['test_proposed_repairs'] = int(np.sum(corrected != ordered.given_label.to_numpy()))
summary['model_training_rows'] = len(train) + len(test)
summary['model_training_labels'] = 'provided practice truth plus repaired test labels'
final_pixels = np.concatenate([train_pixels, ordered[pixels].to_numpy()])
final_labels = np.concatenate([labels, corrected])
joblib.dump(classifier().fit(final_pixels, final_labels), ROOT / 'model.joblib', compress=3)
(reports / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
