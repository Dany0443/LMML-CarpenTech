import csv
import tempfile
import unittest
from pathlib import Path

import torch

from evaluate import binary_metrics, load_test, predict
from model import FEATURE_COLUMNS, MythicThreatClassifier


class EvaluationTests(unittest.TestCase):
    def test_confusion_counts_and_metrics(self):
        metrics = binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        for name in ('tp', 'fp', 'tn', 'fn'):
            self.assertEqual(metrics[name], 1)
        for name in ('accuracy', 'precision', 'recall', 'specificity', 'balanced_accuracy', 'f1', 'f2'):
            self.assertEqual(metrics[name], .5)

    def test_no_positive_predictions_is_safe(self):
        metrics = binary_metrics([0, 1], [0, 0])
        self.assertEqual(metrics['fn'], 1)
        self.assertEqual(metrics['precision'], 0.)
        self.assertEqual(metrics['recall'], 0.)
        self.assertEqual(metrics['f2'], 0.)

    def test_labels_are_joined_by_encounter_id(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'data').mkdir()
            with (folder / 'data/test_dataset.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['encounter_id', *FEATURE_COLUMNS])
                writer.writeheader()
                for encounter_id in ('first', 'second'):
                    writer.writerow(dict(encounter_id=encounter_id, **dict.fromkeys(FEATURE_COLUMNS, 1.)))
            (folder / 'data/test_labels.csv').write_text(
                'encounter_id,is_catastrophic_threat\nsecond,1\nfirst,0\n'
            )
            encounters, features, labels = load_test(folder)
            self.assertEqual(labels, [0, 1])
            self.assertEqual(features.shape, (2, 9))
            self.assertEqual(encounters[0]['encounter_id'], 'first')

    def test_sigmoid_threshold_includes_zero_logit(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'toy.pth'
            oracle = MythicThreatClassifier()
            with torch.no_grad():
                oracle.linear.weight.zero_()
                oracle.linear.bias.zero_()
                oracle.linear.weight[0, 7] = 1.
            torch.save(oracle.state_dict(), checkpoint)
            features = torch.zeros((3, 9), dtype=torch.float32)
            features[:, 7] = torch.tensor([-1., 0., 1.])
            probabilities, predictions = predict(checkpoint, features)
            self.assertEqual(predictions, [0, 1, 1])
            self.assertEqual(probabilities[1], .5)
            self.assertLess(probabilities[0], .5)
            self.assertGreater(probabilities[2], .5)
